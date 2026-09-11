using Test
import Random: Xoshiro
import Statistics: mean, var
import LMC: ComplexLMCState, LMCParams, LMCState, ProblemWithEG, RealLMCState, sample!

# H(ψ) = 1/2 ∑ᵢ |ψᵢ|²
function gaussian_energy(ψ)
    E = 0.0

    @inbounds @simd for i in eachindex(ψ)
        E += abs2(ψ[i])
    end

    return E / 2
end

function gaussian_gradient!(ψ::Vector{T}, grad) where {T<:Real}
    @inbounds @simd for i in eachindex(ψ, grad)
        grad[i] = ψ[i]
    end

    return nothing
end

function gaussian_gradient!(ψ::Vector{Complex{T}}, grad) where {T<:Real}
    @inbounds @simd for i in eachindex(ψ, grad)
        grad[i] = ψ[i] / 2
    end

    return nothing
end

function gaussian_energy_and_gradient!(ψ, grad)
    E = 0.0

    gaussian_gradient!(ψ, grad)

    @inbounds @simd for i in eachindex(ψ, grad)
        E += abs2(ψ[i])
    end

    return E / 2
end

function sample_gaussian(ψ; seed)
    problem = ProblemWithEG(gaussian_gradient!, gaussian_energy, gaussian_energy_and_gradient!)

    beta = 1.0
    epsilon = 0.8
    sigma = sqrt(2epsilon)

    params = LMCParams(problem, beta, epsilon, sigma)
    state = LMCState(ψ)
    samples, energies, stats = sample!(params, state, 200_000; rng=Xoshiro(seed), save_every=10)

    return state, samples, energies, stats
end

@testset "Gaussian sampling" begin
    @testset "real state" begin
        state, samples, energies, stats = sample_gaussian(zeros(Float64, 4); seed=1234)
        x = vec(samples[1, :])

        @test state isa RealLMCState{Float64}
        @test length(state.noise) == length(state.ψ)
        @test eltype(samples) == Float64
        @test 0.5 < stats.accept_rate < 0.9
        @test abs(mean(x)) < 0.05
        @test abs(var(x) - 1.0) < 0.08
        @test all(isfinite, energies)
    end

    @testset "complex state" begin
        state, samples, energies, stats = sample_gaussian(zeros(ComplexF64, 4); seed=5678)
        x = vec(real.(samples[1, :]))
        y = vec(imag.(samples[1, :]))

        @test state isa ComplexLMCState{Float64}
        @test length(state.noise) == 2length(state.ψ)
        @test eltype(samples) == ComplexF64
        @test 0.5 < stats.accept_rate < 0.9
        @test abs(mean(x)) < 0.05
        @test abs(mean(y)) < 0.05
        @test abs(var(x) - 1.0) < 0.08
        @test abs(var(y) - 1.0) < 0.08
        @test all(isfinite, energies)
    end
end

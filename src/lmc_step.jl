
using LinearAlgebra, Random

export Problem, LMCParams, LMCState, RealLMCState, ComplexLMCState, step!, ProblemWithEG

# ----------------------------
# Problem / Params / State
# ----------------------------
abstract type AbstractProblem end

struct Problem{TG,TH} <: AbstractProblem
    grad!::TG
    H::TH
end

struct ProblemWithEG{TG,TH,TEG} <: AbstractProblem
    grad!::TG
    H::TH
    energy_and_grad!::TEG
end

@inline function energy_and_grad!(problem::Problem, ψ, G)
    problem.grad!(ψ, G)
    return problem.H(ψ)
end

@inline function energy_and_grad!(problem::ProblemWithEG, ψ, G)
    return problem.energy_and_grad!(ψ, G)
end

mutable struct LMCParams{TP <:AbstractProblem}
    problem::TP
    β::Float64
    ϵ::Float64
    σ::Float64
end

mutable struct LMCState{T<:Real,S<:Union{T,Complex{T}}}
    ψ::Vector{S}       # current accepted state
    ψtmp::Vector{S}    # proposal buffer

    G::Vector{S}       # grad at current ψ
    Gtmp::Vector{S}    # grad at proposal ψtmp


    E::T                        # H(current ψ)
    Etmp::T                     # H(proposal ψtmp)

    noise::Vector{T}
    initialized::Bool
end

const RealLMCState{T} = LMCState{T, T}
const ComplexLMCState{T} = LMCState{T, Complex{T}}

function LMCState(ψ::Vector{T}) where {T<:Real}
    ψ = copy(ψ)
    ψtmp = similar(ψ)
    G = similar(ψ)
    Gtmp = similar(ψ)
    H = zero(T)
    Htmp = zero(T)
    noise = Array{T}(undef, length(ψ))
    initialized = false
    return RealLMCState{T}(ψ, ψtmp, G, Gtmp, H, Htmp, noise, initialized)
end

function LMCState(ψ::Vector{Complex{T}}) where {T<:Real}
    ψ = copy(ψ)
    ψtmp = similar(ψ)
    G = similar(ψ)
    Gtmp = similar(ψ)
    H = zero(T)
    Htmp = zero(T)
    noise = Array{T}(undef, 2*length(ψ))
    initialized = false
    return ComplexLMCState{T}(ψ, ψtmp, G, Gtmp, H, Htmp, noise, initialized)
end


# ----------------------------
# Internal kernels
# ----------------------------
function propose_from_grad!(
    ψ::Vector{T},
    ψ0::Vector{T},
    G0::Vector{T},
    noise::Vector{T},
    ϵ::Real,
    σ::Real;
    rng=Random.GLOBAL_RNG,
) where {T<:Real}

    ϵT = T(ϵ) / T(2)
    σT = T(σ) / sqrt(T(2))

    randn!(rng, noise)

    @inbounds @simd for i in eachindex(ψ, ψ0, G0, noise)
        ψ[i] = ψ0[i] - ϵT * G0[i] + σT * noise[i]
    end

    return nothing
end

function propose_from_grad!(
    ψ::Vector{Complex{T}},
    ψ0::Vector{Complex{T}},
    G0::Vector{Complex{T}},
    noise::Vector{T},
    ϵ::Real,
    σ::Real;
    rng=Random.GLOBAL_RNG,
) where {T<:Real}

    ϵT = T(ϵ)
    σT = T(σ) / sqrt(T(2))

    randn!(rng, noise)

    @inbounds @simd for i in eachindex(ψ, ψ0, G0)
        j = 2i - 1
        ξ = Complex{T}(noise[j], noise[j + 1])
        ψ[i] = ψ0[i] - ϵT * G0[i] + σT * ξ
    end

    return nothing
end



gradient_scale(::Type{T}) where {T<:Real} = T(1) / T(2)
gradient_scale(::Type{Complex{T}}) where {T<:Real} = one(T)

function logq_langevin(
    ψ_to::Vector{S},
    ψ_from::Vector{S},
    G_from::Vector{S},
    ϵ::Real,
    σ::Real,
) where {S<:Number}

    T = typeof(real(zero(S)))
    ϵT = T(ϵ) * gradient_scale(S)
    σT = T(σ)

    s = zero(T)

    @inbounds @simd for i in eachindex(ψ_to, ψ_from, G_from)
        r = ψ_to[i] - ψ_from[i] + ϵT * G_from[i]
        s += abs2(r)
    end

    return -s / σT^2
end

function lmc_step!(p::LMCParams, state::LMCState{T,S}; rng=Random.GLOBAL_RNG) where {T<:Real,S<:Number}
    problem = p.problem
    β, ϵ, σ = p.β, p.ϵ, p.σ

    ψ    = state.ψ
    ψtmp = state.ψtmp
    G    = state.G
    Gtmp = state.Gtmp
    noise = state.noise
    # initialize cache if needed
    if !state.initialized
        state.E = energy_and_grad!(problem, ψ, G)
        state.initialized = true
    end

    Eψ = state.E

    # propose ψtmp from current ψ and cached G
    propose_from_grad!(ψtmp, ψ, G, noise, ϵ, σ; rng=rng)

    # compute proposal energy and gradient together if available
    Etmp = energy_and_grad!(problem, ψtmp, Gtmp)

    logq_forward  = logq_langevin(ψtmp, ψ, G, ϵ, σ)
    logq_backward = logq_langevin(ψ, ψtmp, Gtmp, ϵ, σ)

    logα = β * (Eψ - Etmp) + logq_backward - logq_forward
    if logα >= 0 || (isfinite(logα) && log(rand(rng)) < logα)
        # proposal becomes current state
        state.ψ, state.ψtmp = state.ψtmp, state.ψ
        state.G, state.Gtmp = state.Gtmp, state.G

        state.E = Etmp
        #state.Htmp = Hψ

        return true
    else
        # current ψ, G, H remain valid
        #state.Htmp = T(Htmp)
        return false
    end
end

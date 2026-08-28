module LMCPT

using Random
using Polyester

import ParallelTemperingSamplers
import ParallelTemperingSamplers:
    AbstractReplicas,
    step!,
    step_slot!,
    getenergy,
    getbeta,
    getwalkerid,
    getstate,
    swapwalkers!

import ..LMC
import ..LMC: LMCParams, LMCState, LMCAdaptParams

export LMCReplicas, initialize_lmc_replicas!

export burnin!, adapt_replicas!

mutable struct SlotRNG{TR}
    rng::TR
end

mutable struct LMCReplicas{TP, TS, TB, TW, TE, TR} <: AbstractReplicas
    params::Vector{TP}     # params[slot], fixed beta-slot transition parameters
    states::Vector{TS}     # states[walker]
    betas::TB              # betas[slot], fixed sorted beta ladder
    walkerids::TW          # walkerids[slot] = walker id
    energies::TE           # energies[walker]
    accepted::Vector{Bool} # accepted[slot]
    rngs::TR               # rngs[slot], fixed slot-local RNGs
end

function LMCReplicas(params::Vector{TP}, states::Vector{TS}, betas::AbstractVector{<:Real}; walkerids = collect(1:length(betas)), rng = Random.GLOBAL_RNG) where {TP <: LMCParams, TS <: LMCState}
    K = length(betas)

    length(params) == K || error("params must have length K.")
    length(states) == K || error("states must have length K.")
    length(walkerids) == K || error("walkerids must have length K.")
    sort(walkerids) == collect(1:K) || error("walkerids must be a permutation of 1:K.")

    betas_vec = collect(Float64, betas)
    walkerids_vec = collect(Int, walkerids)
    energies = zeros(Float64, K)
    accepted = fill(false, K)
    rngs = [SlotRNG(Random.Xoshiro(rand(rng, UInt))) for _ in 1:K]

    TB = typeof(betas_vec)
    TW = typeof(walkerids_vec)
    TE = typeof(energies)
    TR = typeof(rngs)

    reps = LMCReplicas{TP,TS,TB,TW,TE,TR}(params, states, betas_vec, walkerids_vec, energies, accepted, rngs)

    initialize_lmc_replicas!(reps)

    return reps
end

@inline function set_lmc_beta!(p::LMCParams, beta::Real)
    p.β = beta
    return nothing
end

function initialize_lmc_replicas!(reps::LMCReplicas)
    K = length(reps.betas)

    @inbounds for slot in 1:K
        walker = reps.walkerids[slot]

        p = reps.params[slot]
        state = reps.states[walker]

        set_lmc_beta!(p, reps.betas[slot])

        if !state.initialized
            state.E = LMC.energy_and_grad!(p.problem, state.ψ, state.G)
            state.initialized = true
        end

        reps.energies[walker] = state.E
    end

    return nothing
end

function burnin!(reps::LMCReplicas, n_burnin::Int = 1000; rngs = [Xoshiro(rand(UInt)) for _ in 1:length(reps)])
    @inbounds for k in 1:length(reps)
        for _ in 1:n_burnin
            LMC.lmc_step!(reps.params[k], reps.states[k]; rng=rngs[k])
        end
    end
    nothing
end

function adapt_replicas!(reps::LMCReplicas, adaptparams::LMCAdaptParams; rngs=[Xoshiro(rand(UInt)) for _ in 1:length(reps)])
    K = length(reps)

    adapt_success = Vector{Bool}(undef, K)
    adapt_epsilon = Vector{Float64}(undef, K)
    adapt_rate    = Vector{Float64}(undef, K)

    for k in 1:K
        stats = LMC.adapt!(reps.params[k], reps.states[k], adaptparams; rng=rngs[k])
        adapt_success[k] = stats.success
        adapt_epsilon[k] = stats.ϵ
        adapt_rate[k]    = stats.rate
    end
    initialize_lmc_replicas!(reps)

    return (success=adapt_success, epsilon=adapt_epsilon, rate=adapt_rate)
end


function ParallelTemperingSamplers.step_slot!(reps::LMCReplicas, slot::Int)
    @inbounds begin
        walker = reps.walkerids[slot]

        p = reps.params[slot]
        state = reps.states[walker]

        accepted = LMC.lmc_step!(p, state; rng=reps.rngs[slot].rng)

        reps.accepted[slot] = accepted
        reps.energies[walker] = state.E

        return accepted
    end
end

function ParallelTemperingSamplers.steps!(reps::LMCReplicas, n_steps::Int)
    n_steps > 0 || error("n_steps must be positive.")

    n_accepts = zeros(Int, length(reps.betas))

    Polyester.@batch per=thread for slot in eachindex(reps.betas)
        @inbounds begin
            walker = reps.walkerids[slot]

            p = reps.params[slot]
            state = reps.states[walker]
            rng = reps.rngs[slot].rng

            accepted = false
            count = 0

            for _ in 1:n_steps
                accepted = LMC.lmc_step!(p, state; rng=rng)
                count += accepted
            end

            n_accepts[slot] = count
            reps.accepted[slot] = accepted
            reps.energies[walker] = state.E
        end
    end

    return n_accepts
end

@inline function ParallelTemperingSamplers.step!(reps::LMCReplicas)
    Polyester.@batch per=thread for slot in eachindex(reps.betas)
        step_slot!(reps, slot)
    end

    return reps.accepted
end


Base.length(reps::LMCReplicas) = length(reps.betas)

ParallelTemperingSamplers.getwalkerid(reps::LMCReplicas, slot::Int) = reps.walkerids[slot]
ParallelTemperingSamplers.getwalkerids(reps::LMCReplicas) = reps.walkerids
ParallelTemperingSamplers.getbeta(reps::LMCReplicas, slot::Int) = reps.betas[slot]

function ParallelTemperingSamplers.getenergy(reps::LMCReplicas, slot::Int)
    walker = reps.walkerids[slot]
    return reps.energies[walker]
end

function ParallelTemperingSamplers.getstate(reps::LMCReplicas, slot::Int)
    walker = reps.walkerids[slot]
    return reps.states[walker].ψ
end

function ParallelTemperingSamplers.swapwalkers!(reps::LMCReplicas, slot_i::Int, slot_j::Int)
    reps.walkerids[slot_i], reps.walkerids[slot_j] = reps.walkerids[slot_j], reps.walkerids[slot_i]

    return nothing
end

end

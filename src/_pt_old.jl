module LMCPT

using Random

import ReplicaExchange
import ReplicaExchange:
    AbstractReplicas,
    step!,
    getenergies,
    getbetas,
    getwalkerids,
    getstate

import ..LMC
import ..LMC: LMCParams, LMCState, LMCAdaptParams

export LMCReplicas, initialize_lmc_replicas!
export burnin!, adapt_replicas!

mutable struct LMCReplicas{TP, TS, TB, TW, TE} <: AbstractReplicas
    params::Vector{TP}     # params[slot], fixed beta-slot transition parameters
    states::Vector{TS}     # states[walker]
    betas::TB              # betas[slot], fixed sorted beta ladder
    walkerids::TW          # walkerids[slot] = walker id
    energies::TE           # energies[walker]
end

function LMCReplicas(params::Vector{TP}, states::Vector{TS}, betas::AbstractVector{<:Real}; walkerids = collect(1:length(betas))) where {TP <: LMCParams, TS <: LMCState}
    K = length(betas)

    length(params) == K || error("params must have length K.")
    length(states) == K || error("states must have length K.")
    length(walkerids) == K || error("walkerids must have length K.")
    sort(walkerids) == collect(1:K) || error("walkerids must be a permutation of 1:K.")

    betas_vec = collect(Float64, betas)
    walkerids_vec = collect(Int, walkerids)
    energies = zeros(Float64, K)

    TB = typeof(betas_vec)
    TW = typeof(walkerids_vec)
    TE = typeof(energies)

    reps = LMCReplicas{TP,TS,TB,TW,TE}(params, states, betas_vec, walkerids_vec, energies)

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

function ReplicaExchange.step!(reps::LMCReplicas; rng = Random.GLOBAL_RNG)
    K = length(reps.betas)

    Threads.@threads for slot in 1:K
        walker = reps.walkerids[slot]

        p = reps.params[slot]
        state = reps.states[walker]

        set_lmc_beta!(p, reps.betas[slot])

        LMC.lmc_step!(p, state; rng = rng)

        reps.energies[walker] = state.E
    end

    return nothing
end

ReplicaExchange.getenergies(reps::LMCReplicas) = reps.energies
ReplicaExchange.getbetas(reps::LMCReplicas) = reps.betas
ReplicaExchange.getwalkerids(reps::LMCReplicas) = reps.walkerids
ReplicaExchange.getstate(reps::LMCReplicas, walker::Int) = reps.states[walker].ψ


# adaptation utilities

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



end

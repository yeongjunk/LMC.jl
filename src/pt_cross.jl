module LMCPT

using Random
import ReplicaExchange
import ReplicaExchange: AbstractReplica

import ..LMC
import ..LMC: LMCParams, LMCState

export LMCReplica

mutable struct LMCReplica{TP, T} <: AbstractReplica
    β::T
    lmcparams::LMCParams{TP}
    state::LMCState{T}
end

ReplicaExchange.energy(r::LMCReplica) = r.state.E
ReplicaExchange.beta(r::LMCReplica) = r.β
ReplicaExchange.cross_energy(ri::LMCReplica, rj::LMCReplica) =
    ri.lmcparams.problem.H(rj.state.ψ)

function ReplicaExchange.replica_exchange!(ri::LMCReplica, rj::LMCReplica, Eij, Eji)
    ri.state, rj.state = rj.state, ri.state
    ri.state.E = Eij
    rj.state.E = Eji
    ri.state.initialized = false
    rj.state.initialized = false
    return nothing
end

function ReplicaExchange.step!(r::LMCReplica; rng=Random.GLOBAL_RNG)
    LMC.lmc_step!(r.lmcparams, r.state; rng=rng)
    return nothing
end

end

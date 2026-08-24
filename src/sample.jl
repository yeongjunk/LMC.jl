
export sample!

function sample!(p::LMCParams, state::LMCState,n_steps::Int; rng = Random.GLOBAL_RNG, save_every::Int = 10) 
    n_saved = n_steps ÷ save_every
    dim = length(state.ψ)

    samples = Matrix{eltype(state.ψ)}(undef, dim, n_saved)
    energies = Vector{Float64}(undef, n_saved)
    accept = 0 
    save_idx = 0 

    for i in 1:n_steps
        accept += step!(p, state; rng=rng)

        if i % save_every == 0
            save_idx += 1
            copyto!(view(samples, :, save_idx), state.ψ)
            energies[save_idx] = state.H
        end 
    end 

    stats = (accept_rate = accept / n_steps, n_steps = n_steps, n_saved = save_idx, save_every = save_every)

    return samples, energies, stats
end

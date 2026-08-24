using Parameters

export LMCAdaptParams, adapt!

@with_kw struct LMCAdaptParams{F}
    stop_every::Int = 100
    max_windows::Int = 100
    rate_lower::F = 0.5
    rate_upper::F = 0.9
    grow::F = 1.5
end


"""
    adapt!(p::LMCParams, state::LMCState; stop_every=100, max_windows=30, rng=Random.GLOBAL_RNG)

Find a reasonable Langevin step size.
NOTE:
- Mutates `p` and `state`.
- Uses short LMC windows and adjusts `p.ϵ` based on the acceptance rate.
- After this function, `p.ϵ` and `p.σ` are fixed and should be used for sampling.
"""
function adapt!(p::LMCParams, state::LMCState; stop_every::Int = 100, max_windows::Int = 100, rate_lower::Float64 = 0.5, rate_upper::Float64 = 0.9, grow::Float64 = 1.5, rng = Random.GLOBAL_RNG) 
    ϵ_low = nothing
    ϵ_high = nothing

    # optional initial guess
    # p.ϵ = 0.001
    # p.σ = sqrt(2 * p.ϵ / p.β)

    rate = NaN
    for window in 1:max_windows
        N_acc = 0

        for _ in 1:stop_every
            N_acc += lmc_step!(p, state; rng=rng)
        end

        rate = N_acc / stop_every

        if rate > rate_upper
            # proposal is too small / too conservative
            ϵ_low = p.ϵ

            if isnothing(ϵ_high)
                p.ϵ *= grow
            else
                p.ϵ = sqrt(ϵ_low * ϵ_high)
            end

            p.σ = sqrt(2 * p.ϵ / p.β)

        elseif rate < rate_lower
            # proposal is too large / too aggressive
            ϵ_high = p.ϵ

            if isnothing(ϵ_low)
                p.ϵ /= grow
            else
                p.ϵ = sqrt(ϵ_low * ϵ_high)
            end

            p.σ = sqrt(2 * p.ϵ / p.β)

        else
            return (success=true, ϵ=p.ϵ, rate=rate)
        end
    end
    return (success=false, ϵ=p.ϵ, rate=rate)
end

function adapt!(p::LMCParams, state::LMCState, ap::LMCAdaptParams; rng = Random.GLOBAL_RNG)
    return adapt!(p, state; stop_every = ap.stop_every, max_windows = ap.max_windows, rate_lower = ap.rate_lower, rate_upper = ap.rate_upper, grow = ap.grow, rng = rng)
end

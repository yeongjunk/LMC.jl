module BetaGrid

using CSV, DataFrames



function geometric_ladder(β_min::F, β_max::F, K::Int) where {F}
    K >= 2 || error("K must be at least 2.")
    β_min > 0 || error("β_min must be positive.")
    β_max >= β_min || error("β_max must be >= β_min.")
    return [β_min * (β_max / β_min)^((k - 1) / (K - 1)) for k in 1:K]
end

function linear_ladder(β_min::F, β_max::F, K::Int) where {F}
    K >= 2 || error("K must be at least 2.")
    β_min > 0 || error("β_min must be positive.")
    β_max >= β_min || error("β_max must be >= β_min.")
    return [β_min + (β_max - β_min) * (k - 1) / (K - 1) for k in 1:K]
end

function make_replicas(problems::Vector, ψ_init::Vector{T}, βs::Vector{F}, ϵ0) where {T,F}
    K = length(βs)
    length(problems) == K || error("problems and βs must have the same length.")

    replicas = Vector{Replica}(undef, K)

    for k in 1:K
        β = βs[k]
        σ = sqrt(2 * ϵ0 / β)
        p = LMCParams(problems[k], β, ϵ0, σ)
        state = LMCState(copy(ψ_init))
        replicas[k] = Replica(k, p, state)
    end

    return replicas
end

function nearest_neighbor_edge_groups(K::Int)
    odd_edges  = [(i, i+1) for i in 1:2:K-1]
    even_edges = [(i, i+1) for i in 2:2:K-1]
    return odd_edges, even_edges
end

"""
    make_beta_param_grid_graph(βs_layers, As_layers, κs_layers)

Construct a 2D PT graph with coordinates `(i, j)`.

- `i = 1:nβ` is the beta index.
- `j = 1:nlayer` is the auxiliary ladder/layer index.
- Each vertex `(i,j)` has parameters `(β_ij, A_ij, κ_ij)`.

Inputs are matrices of size `(nβ, nlayer)`:
- `βs_layers[i,j]`
- `As_layers[i,j]`
- `κs_layers[i,j]`

Returns:
- `βs::Vector`
- `As::Vector`
- `κs::Vector`
- `edge_groups::Vector{Vector{Tuple{Int,Int}}}`
- `vid::Function`
"""
function make_beta_param_grid_graph(
    βs_layers::AbstractMatrix{F},
    As_layers::AbstractMatrix{F},
    κs_layers::AbstractMatrix{F},
) where {F}

    size(βs_layers) == size(As_layers) == size(κs_layers) ||
        error("βs_layers, As_layers, and κs_layers must have the same size.")

    nβ, nlayer = size(βs_layers)
    K = nβ * nlayer

    # Coordinate -> vertex index.
    # Layer-major blocks:
    # j = 1: vertices 1:nβ
    # j = 2: vertices nβ+1:2nβ
    vid(i, j) = begin
        1 <= i <= nβ || error("beta index out of range.")
        1 <= j <= nlayer || error("layer index out of range.")
        i + (j - 1) * nβ
    end

    βs = Vector{F}(undef, K)
    As = Vector{F}(undef, K)
    κs = Vector{F}(undef, K)

    for j in 1:nlayer
        for i in 1:nβ
            v = vid(i, j)
            βs[v] = βs_layers[i, j]
            As[v] = As_layers[i, j]
            κs[v] = κs_layers[i, j]
        end
    end

    # Vertical beta-direction swaps:
    # (i,j) <-> (i+1,j)
    beta_odd_edges  = Tuple{Int,Int}[]
    beta_even_edges = Tuple{Int,Int}[]

    for j in 1:nlayer
        for i in 1:2:nβ-1
            push!(beta_odd_edges, (vid(i, j), vid(i+1, j)))
        end
        for i in 2:2:nβ-1
            push!(beta_even_edges, (vid(i, j), vid(i+1, j)))
        end
    end

    # Horizontal layer-direction swaps:
    # (i,j) <-> (i,j+1)
    layer_odd_edges  = Tuple{Int,Int}[]
    layer_even_edges = Tuple{Int,Int}[]

    for i in 1:nβ
        for j in 1:2:nlayer-1
            push!(layer_odd_edges, (vid(i, j), vid(i, j+1)))
        end
        for j in 2:2:nlayer-1
            push!(layer_even_edges, (vid(i, j), vid(i, j+1)))
        end
    end

    edge_groups = [
        beta_odd_edges,
        layer_odd_edges,
        beta_even_edges,
        layer_even_edges,
    ]

    return βs, As, κs, edge_groups, vid
end

end

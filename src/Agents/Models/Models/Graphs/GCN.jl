mutable struct GCN <: GNN
    layers::Int
    input_len::Int
    feature_len::Int
    hidden_widths::Union{Int, Vector{Int}}
    weights::Vector{Matrix{Float32}}
    readout::Union{Nothing, Chain}
    # put two fields here, one for weights and one for the initial filter matrix Θ.
    function GCN(layers::Int, input_len::Int, feature_len::Int, hidden_widths::Union{Int, Vector{Int}})
        if typeof(hidden_widths)==Vector{Int}
            ns = [input_len, hidden_widths..., feature_len]
        else
            ns = [input_len, [hidden_widths for _ in 1:layers-1]..., feature_len]
        end
        weight_list = Vector{Matrix{Float32}}()
        for i in 1:layers
            wm = rand(Normal(0f0, sqrt(2f0/ns[i])), (ns[i], ns[i+1]))
            push!(weight_list, wm)
        end
        return new(layers, input_len, feature_len, hidden_widths, weight_list, nothing)
    end
    function GCN(layers::Int, input_len::Int, feature_len::Int, hidden_widths::Union{Int, Vector{Int}}, readout::Chain)
        if typeof(hidden_widths)==Vector{Int}
            ns = [input_len, hidden_widths..., feature_len]
        else
            ns = [input_len, [hidden_widths for _ in 1:layers-1]..., feature_len]
        end
        weight_list = Vector{Matrix{Float32}}()
        for i in 1:layers
            wm = rand(Normal(0f0, sqrt(2f0/ns[i])), (ns[i], ns[i+1]))
            push!(weight_list, wm)
        end
        return new(layers, input_len, feature_len, hidden_widths, weight_list, readout)
    end
end
# model function to calculate node embeddings
function (m::GCN)(node_features::Matrix{Float32}, graph_matrix::Union{Matrix{Float32}, FlowMat}, node_id::Int)
    A_tild = Float32.(graph_matrix) + I
    D = dropdims(sum(A_tild, dims = 1), dims=1)
    D_tild = Diagonal(1 ./ sqrt.(D))
    h = node_features
    for l_idx in 1:m.layers
        h_new = relu.(D_tild * A_tild * D_tild * h * m.weights[l_idx])
        h=h_new
    end
    if m.readout !== nothing
        output = m.readout(permutedims(h[node_id, :]))
        return dropdims(output, dims=1)
    else
        return h[node_id, :]
    end
end
function (m::GCN)(node_features::Matrix{Float32}, graph_matrix::Union{Matrix{Float32}, FlowMat})
    A_tild = Float32.(graph_matrix) + I
    D = dropdims(sum(A_tild, dims = 1), dims=1)
    D_tild = Diagonal(1 ./ sqrt.(D))
    h = node_features
    for l_idx in 1:m.layers
        h_new = relu.(D_tild * A_tild * D_tild * h * m.weights[l_idx])
        h=h_new
    end
    if m.readout !== nothing
        output = m.readout(permutedims(h[node_id, :]))
        return dropdims(output, dims=1)
    else
        return h
    end
end
function get_embeddings(m::GCN, state::Tuple{Matrix{Float32}, FlowMat, Int})
    node_features, graph_matrix, node_idx = state
    A_tild = Float32.(graph_matrix + I)
    D = dropdims(sum(A_tild, dims = 1), dims=1)
    D_tild = Diagonal(1 ./ sqrt.(D))
    h = node_features
    for l_idx in 1:m.layers
        h_new = tanh.(D_tild * A_tild * D_tild * h * m.weights[l_idx])
        h=h_new
    end
    return h
end

# function (m::GCN)(state::Tuple{Matrix{Float32}, FlowMat, Int})
#     h = get_embeddings(m, state)
#     output = m.readout(h)
#     return output
# end

function (m::GCN)(state::Vector{Tuple{Matrix{Float32}, FlowMat, Int}})
    hs = get_embeddings.(fill(m, length(state)), state)
    if typeof(m.readout) == Nothing
        return hcat([hs[i][last(state[i]),:] for i in eachindex(state)]...)
    else
        selected_hs = [hs[i][last(state[i]),:] for i in eachindex(state)]
        output = m.readout.(selected_hs)
        return hcat(output...)
    end
end

function (m::GCN)(state::Tuple{Matrix{Float32}, FlowMat, Int})
    features, graph, node_idx = state
    h = get_embeddings(m, state)
    if m.readout !== nothing
        flattened_h = h[node_idx, :]
        output = m.readout(flattened_h)
        return output
    else
        return h[node_idx, :]
    end
end

Flux.@functor GCN
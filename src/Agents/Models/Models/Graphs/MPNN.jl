

# define a model: This struct will containg a Chain function for each of the the constituent FUNCTIONS
# in the GNN:
#   * Encoder: Takes nodal attribute functions and returns and input to the Message passing phase
#               of the correct length.
#   * Message: Generates messages from pairs of node hidden states, node attribute vectors and
#               the attributes (and possibly hidden states) for the link joining them.
struct MPNN <: GNN
    initialize_function
    reshape_function
    encoder_function::Chain
    message_function::Chain
    aggregation_function::Chain
    update_function::Chain
    readout_function::Chain
    depth::Int
end


###### Message Passing Funcs
# take the vector of concatenated hidden states for eac node as a batch and compute the propagation function jointly on that.
# we can then parallelise this computation for all nodes on the network
function pass_messages(h::Matrix{Float32}, x::Matrix{Float32}, message_function, aggregation_function, reshape_function, ratemat::Union{FlowMat, Matrix{Float32}})
    # reshape the current-this is a vector where each element is itself a vector containing
    # the concatenated vector (h_u, h_i, c_{u,i}) for each i in N(u)
    inputs =  reshape_function(h,x, ratemat) 
    single_messgages = message_function.(inputs) # calculate each single message between neighbout and node
    global_messages =  aggregation_function(single_messgages) # sum over single mesages - can we pass all messages into an rnn? No (variable length)
    return reduce(hcat, global_messages) # can we weight this sum using the capacity??
end
function pass_messages(h::Matrix{Float32}, x::Matrix{Float32}, message_function, aggregation_function, reshape_function, net::N) where {N <: Network}
    return pass_messages(h, x, message_function, aggregation_function, reshape_function, net.rate_matrix)
end

# define a function that takes the update func and reshapes the inputs, then evaluates them
function update_hidden(x::Matrix{Float32}, h::Matrix{Float32}, M::Matrix{Float32},  update_function, ratemat::Union{FlowMat, Matrix{Float32}})
    N = size(ratemat)[1]
    inputs = cat(x,h, M, dims=1)
    new_h = update_function(inputs)
    return new_h
end
function update_hidden(x::Matrix{Float32}, h::Matrix{Float32}, M::Matrix{Float32},  update_function, net::N) where {N <: Network}
    return update_hidden(x, h, M, update_function, net.rate_matrix)
end
# define full message passing phase. This will result in embeddings for each node

function (m::MPNN)(node_features::Matrix{Float32}, graph_matrix::Union{Matrix{Float32}, FlowMat}, node_idx::Int)
    h = m.encoder_function(node_features)
    for l in 1:m.depth
        # 1. pass_messages
        messages = pass_messages(h, node_features, m.message_function, m.aggregation_function, m.reshape_function, graph_matrix)
        # 2. Update the hidden states based on these messages and h
        h_new = update_hidden(node_features, h, messages, m.update_function, graph_matrix)
        h=h_new
    end
    return h[node_idx, :]
end
function (m::MPNN)(node_features::Matrix{Float32}, net::N, node_idx::Int) where {N <: Network}
    return m(node_features, net.rate_matrix, node_idx)
end
Flux.@functor MPNN


#### HELPER FUNCTIONS
# The initialisation function is completely case dependent, however we will want the initialization function
# to always take in some representation of the network, possibly the net itself, or rate_matrix and coords. And output a matrix of 
# shape (num_attributes X num_nodes). where each row of the matrix is a nodal attribute vector. The encoder will then convert this to
# a matrix of length (hidden_length X num_nodes)

# The propagation reshape is actually less case dependent, but for now we will typically be interested in always passing the attribute vectors
# and the hidden vectors as well as the joining link weight. There ought to be a more eff... But we leave this here for not way
# it takes in the network, attribute vectors and hidden vectors and for each node v_i builds a Matrix of size
# (2*hidden_len + attribute_len +1, |ne(v_i)|) where the columns are the input vectors for the messag function
# Infact do we need the message to include self node info if we are including that in the update func?
# If yes we can build a very complex message model and replace the update with a weighted sum of the messages,
# if no we remove the broadcasted x nd h.
function full_reshape(h::Matrix{Float32}, x::Matrix{Float32}, ratemat::Union{FlowMat, Matrix{Float32}})
    N = size(ratemat)[1]
    neigh_idx = findall.(!iszero, [ratemat[:,i] for i in 1:N]) # Find the neighbour indices for all nodes # Turn the above from matrix to vector of vectors
    broadcasted_h = [h[:,i] .* ones(Float32, size(h[:, neigh_idx[i]])) for i in 1:N]
    broadcasted_x = [x[:,i] .* ones(Float32, size(x[:, neigh_idx[i]])) for i in 1:N]
    broadcasted_rates = [reshape(collect(ratemat[i, neigh_idx[i]]), (1, length(ratemat[i, neigh_idx[i]]))) for i in 1:N]
    neighbour_hs = [h[:, neigh_idx[i]] for i in 1:N]
    neighbour_xs = [x[:, neigh_idx[i]] for i in 1:N]
    x = cat.(broadcasted_h, broadcasted_x, broadcasted_rates, neighbour_hs, neighbour_xs,  dims=1)
    return x
end
function full_reshape(h::Matrix{Float32}, x::Matrix{Float32}, net::N) where {N <: Network}
    return propagation_reshape(h, x, net.rate_matrix)
end

function partial_reshape(h::Matrix{Float32}, x::Matrix{Float32}, ratemat::Union{FlowMat, Matrix{Float32}})
    N = size(ratemat)[1]
    neigh_idx = findall.(!iszero, [ratemat[:,i] for i in 1:N]) # Find the neighbour indices for all nodes # Turn the above from matrix to vector of vectors
    broadcasted_rates = [reshape(collect(ratemat[i, neigh_idx[i]]), (1, length(ratemat[i, neigh_idx[i]]))) for i in 1:N]
    neighbour_hs = [h[:, neigh_idx[i]] for i in 1:N]
    neighbour_xs = [x[:, neigh_idx[i]] for i in 1:N]
    x = cat.(broadcasted_rates, neighbour_hs, neighbour_xs,  dims=1)
    return x
end
function partial_reshape(h::Matrix{Float32}, x::Matrix{Float32}, net::N) where {N <: Network}
    return propagation_reshape(h, x, net.rate_matrix)
end

function binary_embedding(ratemat::Union{FlowMat, Matrix{Float32}}; buffer_length=0)
    N = size(ratemat)[2]
    bin_length = length(digits(N, base=2,pad=0) |> reverse)
    hidden_state = zeros(Float32, (bin_length+buffer_length, N))
    parse_func(x) = digits(x, base=2,pad=bin_length+buffer_length) |> reverse
    hidden_state[:, :] = hcat(parse_func.(collect(1:N))...) 
    return permutedims(hidden_state)
end
function binary_embedding(net::N; buffer_length=0) where {N <: SpatialNetwork}
    return binary_embedding(net.rate_matrix; buffer_length=buffer_length)
end
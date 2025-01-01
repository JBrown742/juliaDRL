struct CNN <: AbstractModel
    model::Chain
end

function (m::CNN)(state::Matrix{Float32})
    return dropdims(m.model(reshape(state, (size(state)..., 1, 1))), dims=2)
end
function (m::CNN)(states::Vector{Matrix{Float32}})
    state_mat = cat(state..., dims=ndims(states[1])+1)
    return m.model(state_mat)
end

function (m::CNN)(state::Array{Float32, 3})
    return dropdims(m.model(reshape(state, (size(state)..., 1))), dims=2)
end
function (m::CNN)(states::Vector{Array{Float32, 3}})
    state_mat = cat(states..., dims=ndims(states[1])+1)
    return m.model(state_mat)
end
function (m::CNN)(state::Array{Float32, 4})
    return dropdims(state)
end
Functors.@functor CNN

function save_model(CNN_model::CNN, save_dir::String; model_info::String="")
    num_saved_models = length(readdir(save_dir))
    model = CNN_model.model
    if model_info==""
        @save save_dir * "/model_$(num_saved_models+1).bson" model
    else
        @save save_dir * "/model_" * model_info * ".bson" model
    end
end

function load_model(v::Type{CNN}, load_path::String)
    @load load_path model
    CNN_model = CNN(model)
    return CNN_model
end


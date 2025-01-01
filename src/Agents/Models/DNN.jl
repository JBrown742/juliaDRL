struct DNN <: AbstractModel
    model::Chain
end
function (m::DNN)(state::VectorObs)
    return dropdims(m.model(reshape(state, (length(state), 1))), dims=2)
end
function (m::DNN)(state::Vector{O}) where {O <: VectorObs}
    reshaped_state = reduce(hcat, state)
    return m.model(reshaped_state)
end
function (m::DNN)(state::Matrix{Float32})
    return m.model(state)
end
# function (m::DNN)(state::CuArray{Float32, 2, CUDA.DeviceMemory})
#     return m.model(state)
# end
# function (m::DNN)(state::CuArray{Float32, 1, CUDA.DeviceMemory})
#     return m.model(state)
# end
Functors.@functor DNN

function save_model(DNN_model::DNN, save_dir::String; model_info::String="")
    if !isdir(save_dir)
        mkdir(save_dir)
    end
    num_saved_models = length(readdir(save_dir))
    model = DNN_model.model
    if model_info==""
        @save save_dir * "/model_$(num_saved_models+1).bson" model
    else
        @save save_dir * "/model_" * model_info * ".bson" model
    end
end

function load_model(load_path::String)
    @load load_path model
    DNN_model = DNN(model)
    return DNN_model
end


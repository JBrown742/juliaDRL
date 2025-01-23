struct DNN <: AbstractModel
    model::Chain
end
function (m::DNN)(state::VectorObs)
    outputs = m.model(reshape(state, (length(state), 1)))
    if length(outputs) > 1
        return dropdims.(outputs, dims=2)
    else
        return dropdims(outputs, dims=2)
    end
end
function (m::DNN)(state::Vector{O}) where {O <: AbstractObservation}
    reshaped_state = reduce(hcat, state)
    return m.model(reshaped_state)
end
function (m::DNN)(state::Union{Matrix{Float32}, Matrix{Float64}})
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

function save_model(DNN_model::Vector{M}, save_dir::String; model_info::String="") where {M <: AbstractModel}
    if !isdir(save_dir)
        mkdir(save_dir)
    end
    num_saved_models = length(readdir(save_dir))
    for (i, mod) in enumerate(DNN_model)
        if model_info==""
            @save save_dir * "/model_$(num_saved_models+1)_$(i).bson" mod
        else
            @save save_dir * "/model_" * model_info * "_$(i).bson" mod
        end
    end
end

function load_model(load_path::String)

    @load load_path model
    DNN_model = DNN(model)
    return DNN_model
end


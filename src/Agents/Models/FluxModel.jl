struct FluxModel{C} <: AbstractModel
    # An abstract wrapper struct for any Flux compatible model
    model::C # The Flux Model object itself
    optimizer::Union{AbstractRule, Nothing} # The optimizer to use for training
    _optimizer_state::Union{NamedTuple, Nothing} # private attribute for storing optimizer state
    
    function FluxModel(model::C, optimizer::O) where {C, O <: AbstractRule} # Constructor dispatch for when optimizer defined.
        return new{C}(model, optimizer, Flux.setup(optimizer, model))
    end
    
    function FluxModel(model::C) where {C} # constructor dispatch for when no optimizer defined. Used for inference in certain cases.
        return new{C}(model, nothing, nothing)
    end
end

function (m::FluxModel)(x::AbstractArray)
    # If the input is a vector, it's a single observation so we need to add a batch dimension.
    # If it's a matrix or higher, it's already a batch or high-D observation so we can pass through.
    res = if ndims(x) == 1
        m.model(reshape(x, :, 1))
    else
        m.model(x)
    end
    
    # Safety fix: Clamp outputs to prevent NaN/Inf propagation to simulations
    # This is critical for physical simulations like Box2D (BipedalWalker). 
    # This is a hack, however the model should be receiving normalised observations
    # so this should simply act as a fall-back safety mechanism in the case that the 
    # environment has no static normalisation and running normalisation is switched off.
    if res isa AbstractArray
        return clamp.(res, -100.0f0, 100.0f0)
    elseif res isa Tuple
        return map(r -> clamp.(r, -100.0f0, 100.0f0), res)
    else
        return res
    end
end

# This dispatch handles a Vector of observations by using idiomatic Flux batching
function (m::FluxModel)(x::Vector{<:AbstractArray})
    # Flux.batch handles N-D tensors automatically.
    return m.model(Flux.batch(x))
end

# Fallback for custom observation types (like GraphObs or wrapped types)
function (m::FluxModel)(x::Vector{T}) where {T <: AbstractObservation}
    # If the observation is a custom struct, we attempt to batch it.
    # Users can define Flux.batch(::Vector{MyObs}) to support custom logic.
    return m.model(Flux.batch(x))
end

Functors.@functor FluxModel # makes the model directly callable.

function save_model(m::FluxModel, save_dir::String; model_info::String="")
    if !isdir(save_dir)
        mkdir(save_dir)
    end
    num_saved_models = length(readdir(save_dir))
    model = m.model
    # Moving away from BSON to standard Serialization for better reliability in modern Julia
    filename = model_info == "" ? "model_$(num_saved_models+1).jls" : "$(model_info)_model.jls"
    serialize(joinpath(save_dir, filename), model)
end

function load_model(::Type{FluxModel}, load_path::String)
    model = deserialize(load_path)
    return FluxModel(model)
end

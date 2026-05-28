struct FluxModel{C} <: AbstractModel
    model::C
    optimizer::Union{AbstractRule, Nothing}
    _optimizer_state::Union{NamedTuple, Nothing}
    
    function FluxModel(model::C, optimizer::O) where {C, O <: AbstractRule}
        return new{C}(model, optimizer, Flux.setup(optimizer, model))
    end
    
    function FluxModel(model::C) where {C}
        return new{C}(model, nothing, nothing)
    end
end

# The "Seamless" Forward Pass
function (m::FluxModel)(x::AbstractArray)
    # If the input is a vector, it's a single observation; add batch dimension.
    # If it's a matrix or higher, it's already a batch or high-D observation; pass through.
    res = if ndims(x) == 1
        m.model(reshape(x, :, 1))
    else
        m.model(x)
    end
    
    # SAFETY FIX: Clamp outputs to prevent NaN/Inf propagation to simulations
    # This is critical for physical simulations like Box2D (BipedalWalker)
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
    # Flux.batch is the industry standard. It handles N-D tensors automatically.
    return m.model(Flux.batch(x))
end

# Fallback for custom observation types (like GraphObs or wrapped types)
function (m::FluxModel)(x::Vector{T}) where {T <: AbstractObservation}
    # If the observation is a custom struct, we attempt to batch it.
    # Users can define Flux.batch(::Vector{MyObs}) to support custom logic.
    return m.model(Flux.batch(x))
end

Functors.@functor FluxModel

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

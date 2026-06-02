# Define abstract types 
abstract type AbstractModel end
abstract type AbstractAgent end
abstract type AbstractObservation end

# Type-safe observation hierarchies are critical for Julia performance.
# Define 4 common observation types. Vector for standard vector inputs
# MatrixObs as a standard generalisation to this. ArrayObs for things 
# like image data or other Matrix based data with several channels.
# GraphObs is self explanatory.
struct VectorObs <: AbstractObservation
    data::Vector{Float32}
end
struct MatrixObs <: AbstractObservation
    data::Matrix{Float32}
end
struct ArrayObs <: AbstractObservation
    data::Array{Float32, 3}
end
mutable struct GraphObs <: AbstractObservation
    features::Matrix{Float32}
    adjacency::Matrix{Float32}
end

# Type aliases for each of the four possible action spaces
const DiscreteAct = Int
const ContinuousAct = Float32
const MultiDiscreteAct = Vector{DiscreteAct}
const MultiContinuousAct = Vector{ContinuousAct}

const AbstractAction = Union{DiscreteAct, ContinuousAct, MultiDiscreteAct, MultiContinuousAct}

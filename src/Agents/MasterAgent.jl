abstract type AbstractModel end

mutable struct AbstractAgent
    model::Vector{AbstractModel}
end

const VectorObs = Union{Vector{Float64}, Vector{Float32}}
const MatrixObs = Union{Matrix{Float64}, Matrix{Float32}}
const ArrayObs = Union{Array{Float64}, Array{Float32}}
mutable struct GraphObs 
    features::Union{Matrix{Float64}, Matrix{Float32}}
    adjacency::Union{Matrix{Float64}, Matrix{Float32}}
end
const AbstractObservation = Union{VectorObs, MatrixObs, ArrayObs, GraphObs}

const DiscreteAct = Int
const ContinuousAct = Float32
const MultiDiscreteAct = Vector{Int}
const MultiContinuousAct = Union{Vector{Float32}, Vector{Float64}}

const AbstractAction = Union{DiscreteAct, ContinuousAct, MultiDiscreteAct, MultiContinuousAct}


struct Split{T}
    paths::T
end
  
Split(paths...) = Split(paths)
  
Flux.@layer Split
  
(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)
abstract type AbstractModel end
abstract type AbstractPolicy end

mutable struct AbstractAgent
    model::AbstractModel
    policy::AbstractPolicy
end




const VectorObs = Union{Vector{Float64}, Vector{Float32}}

const MatrixObs = Union{Matrix{Float64}, Matrix{Float32}}

const ArrayObs = Union{Array{Float64}, Matrix{Float32}}

mutable struct GraphObs 
    features::Matrix{Float64}
    adjacency::Matrix{Float64}
end

const AbstractObservation = Union{VectorObs, MatrixObs, ArrayObs, GraphObs}

struct Split{T}
    paths::T
end
  
Split(paths...) = Split(paths)
  
Flux.@layer Split
  
(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)
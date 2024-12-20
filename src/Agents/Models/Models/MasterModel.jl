abstract type LearningModel end
abstract type GNN <: LearningModel end

struct Split{T}
    paths::T
end
  
Split(paths...) = Split(paths)
  
Flux.@layer Split
  
(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)
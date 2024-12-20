module Algorithms

using Base: rand
using StatsBase
using Distributions
using Distributed
using NNlib
using LinearAlgebra
using Flux
using Shuffle
using CUDA

# using ..juliaDRL: # import functions from other module above


export

    LearningAlgorithm,
    OptimizationAlgorithm



include("./Algorithms/MasterAlgorithm.jl")
include("./Algorithms/DQN/DQN.jl")

end # module Models

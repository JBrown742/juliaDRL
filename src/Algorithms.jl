module Algorithms

using Base: rand, SizeUnknown
using StatsBase
using Distributions
using SpecialFunctions
using Functors
using Distributed
using NNlib
using LinearAlgebra
using Flux
using Shuffle
using Plots # used for API function
using Random
using Distributions
using Serialization


using ..ProximalPolicy.Environments
using ..ProximalPolicy.Agents


export
    AbstractAlgorithm,
    PPO,
    learn,
    get_action,
    validation_episode!,
    visualise_learning,
    visualise_best


include("./Algorithms/MasterAlgorithm.jl")
include("./Algorithms/RL/PPO/PPO.jl")
include("./Algorithms/RL/PPO/Utilities.jl")
include("./Algorithms/RL/PPO/API.jl")

end # module Algorithms

module Algorithms

using Base: rand, SizeUnknown
using StatsBase
using Distributions
using Distributed
using NNlib
using LinearAlgebra
using Flux
using Shuffle
using Plots # used for API function
using JSON # used for API function
using Random
using Distributions
using Serialization


using ..juliaDRL.Environments
using ..juliaDRL.Agents


export
    AbstractAlgorithm,
    PPO,
    learn,
    get_action,
    validation_episode!,
    visualise_learning


include("./Algorithms/MasterAlgorithm.jl")
include("./Algorithms/RL/PPO/PPO.jl")
include("./Algorithms/RL/PPO/Utilities.jl")
include("./Algorithms/RL/PPO/API.jl")

end # module Algorithms

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
    AbstractExperience, 
    AbstractBuffer,
    AbstractAlgorithm,

    # Import RL related Algorithm functionality and types
    ### Alg functions
    train!,
    validation_episode!,

    ### API
    learn,
    visualise_learning,

    ## PPO 
    PPO,
    get_action


include("./Algorithms/MasterAlgorithm.jl")
include("./Algorithms/RL/PPO/PPO.jl")
include("./Algorithms/RL/PPO/Utilities.jl")
include("./Algorithms/RL/PPO/API.jl")

end # module Algorithms

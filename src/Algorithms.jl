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

using ..juliaDRL: AbstractAgent, AbstractModel, AbstractEnv,
step!, render!, reset!, close!, get_action, AbstractObservation, VectorObs

export
    AbstractExperience, 
    AbstractBuffer,

    AbstractAlgorithm,
    Experience,
    Buffer,

    # Import RL related Algorithm functionality and types
    ## DQN
    DQNexperience,
    ExperienceBuffer,
    buffer_add!,
    buffer_pop!,
    buffer_shuffle!,
    buffer_update!,
    sample,
    buffer_size_check,
    states,
    rewards, 
    next_states,
    terminals,

    ### Alg types
    DQN,

    ### Alg functions
    train!,
    learning_episode!,
    validation_episode!,
    update_target!,
    soft_update_target!




include("./Algorithms/MasterAlgorithm.jl")
include("./Algorithms/RL/Buffer.jl")
include("./Algorithms/RL/DQN/DQN.jl")

end # module Models


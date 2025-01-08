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
using Plots # used for API function
using JSON # used for API function

using ..juliaDRL: AbstractAgent, AbstractModel, AbstractEnv,
step!, render!, reset!, close!, Cartpole, save_model, load_model,
AbstractObservation, AbstractAction, DiscreteAct

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

    ## Alg types
    DQN,

    ### Alg functions
    train!,
    learning_episode!,
    validation_episode!,
    update_target!,
    soft_update_target!,

    ### API
    learn,
    visualise_learning,

    ## PPO 
    PPO,
    full_training_procedure!,
    get_action



include("./Algorithms/MasterAlgorithm.jl")
include("./Algorithms/RL/Buffer.jl")
include("./Algorithms/RL/DQN/DQN.jl")
include("./Algorithms/RL/DQN/API.jl")
include("./Algorithms/RL/PPO/PPO.jl")

end # module Models


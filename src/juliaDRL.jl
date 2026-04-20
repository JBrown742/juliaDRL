module juliaDRL


#This is the main format in order to import a module to the overall package
include("Agents.jl")
using .Agents

export
    # Define abstract types to supertype our observation 
    # and model types
    AbstractObservation,
    AbstractModel,
    AbstractAgent,
    AbstractAction,

    StandardActorCritic,
    CombinedActorCritic,

    # Specific Action Types
    DiscreteAct,
    ContinuousAct,
    MultiDiscreteAct,
    MultiContinuousAct,
    
    # Model types
    FluxModel,

    load_agent, 
    save_model,
    load_model,
    Split


include("Environments.jl")
using .Environments

export
    AbstractEnv, 

    Cartpole,
    Pendulum,
    CarRacing,
    BipedalWalker,
    ParticleChase,

    step!,
    render!,
    reset!, 
    close!,
    renderize!,
    clone


include("Algorithms.jl")
using .Algorithms

export
    AbstractExperience, 
    AbstractBuffer,
    AbstractAlgorithm,

    ### Alg functions
    train!,
    validation_episode!,

    ## API
    learn,
    visualise_learning,

    PPO,
    get_action


    
end # module juliaDRL

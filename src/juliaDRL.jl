module juliaDRL


#This is the main format in order to import a module to the overall package
include("Agents.jl")
using .Agents

export
    # Define abstract types to supertype our observation 
    # and model types
    AbstractAgent, 
    SingleModelAgent, 
    DualModelAgent, 
    AbstractModel,
    AbstractObservation,
    AbstractAction,
    
    # Define the possible observation types. 
    # Currently we assume models can only work 
    # with vector inputs (ANN), matrix or array inputs (CNN)
    # or graph input (GNN)
    VectorObs,
    MatrixObs,
    ArrayObs,
    GraphObs,

    DiscreteAct,
    ContinuousAct,
    MultiDiscreteAct,
    MultiContinuousAct,
    
    # Model exports
    LearningModel,

    CNN,
    DNN,
    Recurrent, 

    #Policies
    EpsilonGreedy,
    get_action,

    # Helper function
    save_model,
    load_model,
    Split


include("Environments.jl")
using .Environments


export
    Cartpole,
    Pendulum,
    CarRacing,

    step!,
    render!,
    reset!, 
    close!,

    normalise    
       
include("Algorithms.jl")
using .Algorithms

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
    soft_update_target!,

    ## APIl
    learn,
    visualise_learning,

    PPO,
    full_training_procedure!,
    get_action,

    ContinuousPPO



    
end # module telaQML


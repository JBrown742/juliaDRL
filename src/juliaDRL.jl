module juliaDRL

#This is the main format in order to import a module to the overall package
include("Agents.jl")
using .Agents

export
    # Define abstract types to supertype our observation 
    # and model types
    AbstractAgent, 
    LearningModel,
    AbstractObservation,

    # Define the possible observation types. 
    # Currently we assume models can only work 
    # with vector inputs (ANN), matrix or array inputs (CNN)
    # or graph input (GNN)
    VectorObs,
    MatrixObs,
    ArrayObs,
    GraphObs,

    # Model exports
    DNN, 
    
    #utils 
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

    normalise    
       
include("Algorithms.jl")
using .Algorithms

export
    LearningAlgorithm,

    # DQN
    DQN,
    train!,
    learning_episode!,
    validation_episode!,
    update_target!,
    soft_update_target!
end # module telaQML


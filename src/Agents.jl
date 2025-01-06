module Agents

# model imports go here
using Flux
using CUDA
using Flux: params, Statistics
using Distributions: Normal
using LinearAlgebra
using BSON: @save, @load
using Functors

export
    # Define abstract types to supertype our observation 
    # and model types
    AbstractAgent, 
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

include("./Agents/MasterAgent.jl")
include("./Agents/Models/DNN.jl")
include("./Agents/Models/CNN.jl")
include("./Agents/Models/Recurrent.jl")

end # module Models
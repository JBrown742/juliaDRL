module Agents

# model imports go here
using Flux
using Flux: params, Statistics
using Distributions: Normal
using LinearAlgebra
using Functors
using Serialization

export
    # Define abstract types to supertype our observation 
    # and model types
    AbstractAgent, 
    StandardActorCritic,
    CombinedActorCritic,
    StandardPolicy,
    StandardValue,
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
    FluxModel, 

    #Policies
    EpsilonGreedy,
    get_action,


    # Helper function
    
    save_agent,
    load_agent, 
    save_model,
    load_model,
    Split

include("./Agents/Types.jl")
include("./Agents/Models/FluxModel.jl")
include("./Agents/MasterAgent.jl")

end # module Agents
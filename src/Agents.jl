module Agents

# model imports go here
using Flux
using CUDA
using Flux: params, Statistics
using Distributions: Normal
using LinearAlgebra
using BSON: @save, @load

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
    GraphObs



# include statements go here
include("./Agents/MasterAgent.jl")


end # module Models
module Environments

using Plots
using ComputedFieldTypes
using Combinatorics
using Distributed
using PyCall 
using LinearAlgebra
using StatsBase
using Random
using Flux
using NNlib



# using ..juliaDRL: 

export
    Cartpole,
    Pendulum,
    CarRacing,

    step!,
    render!,
    reset!, 
    
    normalise


include("./Environments/MasterEnv.jl")

end # module Models

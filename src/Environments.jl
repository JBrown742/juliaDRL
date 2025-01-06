module Environments

using Plots
using Combinatorics
using Distributed
using PyCall 
using LinearAlgebra
using StatsBase
using Random
using Flux
using NNlib

const gym = PyNULL()
function __init__()
    copy!(gym,  pyimport("gymnasium"))
end

using ..juliaDRL: AbstractObservation, AbstractAction
export
    AbstractEnv, 

    Cartpole,
    Pendulum,
    CarRacing,

    step!,
    render!,
    reset!, 
    close!,
    
    normalise


include("./Environments/MasterEnv.jl")
include("./Environments/Cartpole.jl")
include("./Environments/Pendulum.jl")
include("./Environments/CarRacing.jl")

end # module Models

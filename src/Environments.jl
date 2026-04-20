module Environments

using Plots
using Combinatorics
using Distributed
using Distributions
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
    BipedalWalker,
    ParticleChase,

    step!,
    render!,
    reset!, 
    close!,
    renderize!,
    clone,
    distribute_worker_envs


include("./Environments/MasterEnv.jl")
include("./Environments/Cartpole.jl")
include("./Environments/Pendulum.jl")
include("./Environments/CarRacing.jl")
include("./Environments/BipedalWalker.jl")
include("./Environments/ParticleChase.jl")

end # module Models

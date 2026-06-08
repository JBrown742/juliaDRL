module Environments

using Plots
using Combinatorics
using Distributed
using Distributions
using PythonCall 
import PythonCall: pynew, pycopy!
using LinearAlgebra
using StatsBase
using Random
using Flux
using NNlib

const gym = pynew()
function __init__()
    try
        pycopy!(gym, pyimport("gymnasium"))
    catch e
        @warn "Could not import 'gymnasium'. Some environments will not be available. Error: $e"
    end
end

using ..ProximalPolicy: AbstractObservation, AbstractAction
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

end # module Environments

mutable struct Pendulum{S, A} <: AbstractEnv
    episode_length::Int64
    pyenv::PyObject
    state::Vector{Float32}
    terminal::Bool
    action_type::Type
    action_extent::Tuple{Float32, Float32}
    function Pendulum(len::Int64, pyenv::PyObject)
        obs, info = pyenv.reset()
        return new{Vector{Float32}, Float32}(len, pyenv, obs, false, Float32, (-2f0, 2f0))
    end
end

state_type(::Type{C}) where C <: Pendulum = Vector{Float32}
action_type(::Type{C}) where C <: Pendulum = Float32

function step!(env::Pendulum, action::Float32)
    observation, reward, terminated, truncated, info = env.pyenv.step([action])
    env.state = observation
    env.terminal = terminated
    return observation, reward, terminated || truncated 
end

function render!(env::Pendulum)
    env.pyenv.render()
    return
end

function reset!(env::Pendulum) 
    (observation, info) = env.pyenv.reset()
    env.state = observation
    env.terminal = false
    return observation
end

# =======================... Utility functions...================================== #

function normalize(env::Pendulum, x::Vector{Float32}) 
    if length(x) == 3
        return x ./ Float32.([1.,1.,8.])
    else
        return x ./ Float32.([1.,1.,8., 1.])
    end
end
function normalize(env::Pendulum, x::Matrix{Float32}) 
    if size(x)[1] == 3
        return x ./ Float32.([1., 1., 8.] .* ones(size(x)))
    else
        return x ./ Float32.([1., 1., 8., 1.] .* ones(size(x)))
    end
end

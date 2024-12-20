mutable struct CarRacing{S, A} <: AbstractEnv
    episode_length::Int
    pyenv::PyObject
    state::Array{Float32, 3}
    terminal::Bool
    actions::Vector{Int}
    function CarRacing(len::Int, pyenv::PyObject)
        obs, info = pyenv.reset()
        num_actions = pyenv.action_space.n
        return new{Array{Float32, 3}, Int}(len, pyenv, Float32.(obs) ./ 255, false, collect(0:num_actions-1))
    end
end

# state_type(::Type{C}) where C <: CarRacing = Vector{Float64}
# action_type(::Type{C}) where C <: CarRacing = Int

function step!(env::CarRacing, action::Int)
    act = env.actions[action]
    observation, reward, terminated, truncated, info = env.pyenv.step(act)
    env.state = observation
    env.terminal = terminated
    return Float32.(observation) ./ 255, reward, terminated || truncated
end



function render!(env::CarRacing)
    env.pyenv.render()
    return
end

function reset!(env::CarRacing) 
    (observation, info) = env.pyenv.reset()
    env.state = observation
    env.terminal = false
    return Float32.(observation) ./ 255
end


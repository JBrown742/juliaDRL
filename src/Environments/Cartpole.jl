mutable struct Cartpole{S, A} <: AbstractEnv
    episode_length::Int64
    pyenv::PyObject
    state::Vector{Float32}
    terminal::Bool
    actions::Vector{Int}
    possible_actions::Vector{Int}
    function Cartpole(len::Int64, pyenv::PyObject)
        obs, info = pyenv.reset()
        return new{Vector{Float32}, Int}(len, pyenv, obs, false, [0,1], [1,2])
    end
end

state_type(::Type{C}) where C <: Cartpole = Vector{Float64}
action_type(::Type{C}) where C <: Cartpole = Int

function action_mask(env::Cartpole)
    return ones(Float32, 2)
end

function get_action_probs(env::Cartpole, actions::Vector{Int}, model_outputs::Vector{Float32})
    return model_outputs
end

function step!(env::Cartpole, action::Int)
    act = env.actions[action]
    observation, reward, terminated, truncated, info = env.pyenv.step(act)
    env.state = observation
    env.terminal = terminated
    return observation, reward, terminated || truncated
end



function render!(env::Cartpole)
    env.pyenv.render()
    return
end

function reset!(env::Cartpole) 
    (observation, info) = env.pyenv.reset()
    env.state = observation
    env.terminal = false
    return observation
end

# =======================... Utility functions...================================== #

function normalize(x::Vector{Float32}) 
    return x ./ Float32.([4.8, 3, 0.418, 3])
end
function normalize(x::Matrix{Float32}) 
    return x ./ Float32.([4.8, 3, 0.418, 3] .* ones(size(x)))
end

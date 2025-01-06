mutable struct Cartpole <: AbstractEnv
    episode_length::Int64
    pyenv::PyObject
    state::Vector{Float64}
    terminal::Bool
    actions::Vector{Int}
    action_mask::Vector{Float32}
    function Cartpole(len::Int64; render=false)
        if render
            pe = gym.make("CartPole-v1", render_mode="human");
        else            
            pe = gym.make("CartPole-v1");
        end
        obs, info = pe.reset()
        return new(len, pe, obs, false, [0, 1], ones(Float32, 2))
    end
end

function step!(env::Cartpole, action::Int)
    act = env.actions[action]
    observation, reward, terminated, truncated, info = env.pyenv.step(act)
    env.state = normalize(observation)
    env.terminal = terminated
    return normalize(observation), reward, terminated || truncated
end



function render!(env::Cartpole)
    env.pyenv.render()
    return
end

function reset!(env::Cartpole) 
    observation, info = env.pyenv.reset()
    env.state = normalize(observation)
    env.terminal = false
    return normalize(observation)
end

function close!(env::Cartpole)
    env.pyenv.close()
end

# =======================... Utility functions...================================== #

function normalize(x::Vector{Float32}) 
    return x ./ Float32.([4.8, 3, 0.418, 3])
end
function normalize(x::Matrix{Float32}) 
    return x ./ Float32.([4.8, 3, 0.418, 3] .* ones(size(x)))
end
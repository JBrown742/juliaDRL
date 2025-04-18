mutable struct Cartpole <: AbstractEnv
    episode_length::Int64
    pyenv::PyObject
    state::Vector{Float64}
    terminal::Bool
    actions::Vector{Int}
    action_mask::Vector{Float32}
    renderize::Bool
    function Cartpole(len::Int64; render=false)
        if render
            pe = gym.make("CartPole-v1", render_mode="human");
        else            
            pe = gym.make("CartPole-v1");
        end
        obs, info = pe.reset()
        return new(len, pe, obs, false, [0, 1], ones(Float32, 2), render)
    end
end

function step!(env::Cartpole, action::Int)
    act = env.actions[action]
    observation, reward, terminated, truncated, info = env.pyenv.step(act)
    env.state = normalize(observation)
    env.terminal = terminated
    return env.state, reward, terminated || truncated
end



function render!(env::Cartpole)
    env.pyenv.render()
    return
end

function reset!(env::Cartpole) 
    observation, info = env.pyenv.reset()
    env.state = normalize(observation)
    env.terminal = false
    return env.state
end

function close!(env::Cartpole)
    env.pyenv.close()
end

function renderize!(env::Cartpole)
    env.pyenv = gym.make("CartPole-v1", render_mode="human");
    env.renderize = true
end


# =======================... Utility functions...================================== #

function normalize(x::Vector{Float32}) 
    return x ./ Float32.([4.8, 3, 0.418, 3])
end
function normalize(x::Matrix{Float32}) 
    return x ./ Float32.([4.8, 3, 0.418, 3] .* ones(size(x)))
end
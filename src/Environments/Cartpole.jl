mutable struct Cartpole <: AbstractEnv
    episode_length::Int64
    pyenv::Py
    state::Vector{Float32}
    terminal::Bool
    truncated::Bool
    actions::Vector{Int}
    action_mask::Vector{Float32}
    renderize::Bool
    seed::Union{Int, Nothing}
    function Cartpole(len::Int64; render=false, seed=nothing)
        if render
            pe = gym.make("CartPole-v1", render_mode="human");
        else            
            pe = gym.make("CartPole-v1");
        end
        res = pe.reset(seed=seed)
        obs = pyconvert(Vector{Float32}, res[0])
        return new(len, pe, obs, false, false, [0, 1], ones(Float32, 2), render, seed)
    end
end

function clone(s::Cartpole)
    return Cartpole(
        s.episode_length;
        render=s.renderize,
        seed=s.seed
    )
end

function step!(env::Cartpole, action::Int)
    act = env.actions[action]
    res = env.pyenv.step(act)
    observation = pyconvert(Vector{Float32}, res[0])
    reward = pyconvert(Float32, res[1])
    terminated = pyconvert(Bool, res[2])
    truncated = pyconvert(Bool, res[3])
    
    env.state = normalize(observation)
    env.terminal = terminated || truncated
    env.truncated = truncated
    return env.state, reward, env.terminal
end



function render!(env::Cartpole)
    env.pyenv.render()
    return
end

function reset!(env::Cartpole; seed=nothing) 
    res = env.pyenv.reset(seed=seed)
    observation = pyconvert(Vector{Float32}, res[0])
    env.state = normalize(observation)
    env.terminal = false
    env.truncated = false
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
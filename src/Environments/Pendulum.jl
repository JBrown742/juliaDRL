mutable struct Pendulum <: AbstractEnv
    episode_length::Int64
    pyenv::PyObject
    state::Vector{Float32}
    terminal::Bool
    action_type::Type
    action_extent::Tuple{Float32, Float32}
    function Pendulum(len::Int64; render=false)
        if render
            pe = gym.make("Pendulum-v1", render_mode="human");
        else            
            pe = gym.make("Pendulum-v1");
        end
        obs, info = pe.reset()
        return new(len, pe, obs, false, Float32, (-2f0, 2f0))
    end
end

function step!(env::Pendulum, action::Union{Float32, Float64})
    scaled_action = clamp(action, env.action_extent[1], env.action_extent[2])
    observation, reward, terminated, truncated, info = env.pyenv.step([scaled_action])
    env.state = normalize(env, observation)
    env.terminal = terminated
    return  normalize(env, observation), reward, terminated || truncated 
end

function render!(env::Pendulum)
    env.pyenv.render()
    return
end

function close!(env::Pendulum)
    env.pyenv.close()
end

function reset!(env::Pendulum) 
    (observation, info) = env.pyenv.reset()
    env.state = normalize(env, observation)
    env.terminal = false
    return normalize(env, observation)
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

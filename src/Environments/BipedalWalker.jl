mutable struct BipedalWalker <: AbstractEnv
    episode_length::Int64
    episode_step::Int64
    pyenv::PyObject
    state::Vector{Float32}
    terminal::Bool
    action_type::Vector{Type}
    observation_lows::Vector{Float32}
    observation_highs::Vector{Float32}
    renderize::Bool
    function BipedalWalker(len::Int; render::Bool=false)
        if render
            pe = gym.make("BipedalWalker-v3", hardcore=true, render_mode="human");
        else            
            pe = gym.make("BipedalWalker-v3", hardcore=true);
        end
        state, info = pe.reset()
        highs = Float32.(pe.unwrapped.observation_space.high)
        lows = Float32.(pe.unwrapped.observation_space.low)
        n_actions = pe.unwrapped.action_space.shape[1]
        obs = 2 .* ((state .- lows) ./ (highs .- lows)) .- 1
        return new(len, 0, pe, Float32.(obs), false, fill(Float32, n_actions), lows, highs, render)
    end
end

function clone(s::BipedalWalker)
    return BipedalWalker(
        s.episode_length;
        render=s.renderize
    )
end

function step!(env::BipedalWalker, action::Vector{Float32})
    scaled_actions = 2f0 .* action .- 1f0
    clipped_action = clamp.(scaled_actions, -1f0, 1f0)
    s, reward, terminated, truncated, info = env.pyenv.step(clipped_action)
    observation = process_state(env, s)
    env.state = observation
    env.terminal = terminated
    env.episode_step += 1
    if env.episode_step == env.episode_length
        term = true
    else
        term = terminated || truncated
    end
    return Float32.(observation), Float32.(reward), term
end

function render!(env::BipedalWalker)
    env.pyenv.render()
    return
end

function reset!(env::BipedalWalker) 
    (s, info) = env.pyenv.reset()
    observation = process_state(env, s)
    env.state = observation
    env.terminal = false
    env.episode_step = 0
    return Float32.(env.state)
end

function close!(env::BipedalWalker)
    env.pyenv.close()
end

function renderize!(env::BipedalWalker)
    env.pyenv = gym.make("BipedalWalker-v3", hardcore=true, render_mode="human");
    env.renderize = true
end

function process_state(env::BipedalWalker, state::Vector{Float32})
    return 2 .* ((state .- env.observation_lows) ./ (env.observation_highs .- env.observation_lows)) .- 1
end
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
    hardcore::Bool
    stuck_counter::Int
    stuck_threshold::Int
    function BipedalWalker(len::Int; render::Bool=false, hardcore=false, stuck_threshold=2000)
        if render
            pe = gym.make("BipedalWalker-v3", render_mode="human", hardcore=hardcore);
        else            
            pe = gym.make("BipedalWalker-v3", hardcore=hardcore);
        end
        state, info = pe.reset()
        highs = Float32.(pe.unwrapped.observation_space.high)
        lows = Float32.(pe.unwrapped.observation_space.low)
        n_actions = pe.unwrapped.action_space.shape[1]
        obs = 2 .* ((state .- lows) ./ (highs .- lows)) .- 1
        return new(len, 0, pe, Float32.(obs), false, fill(Float32, n_actions), lows, highs, render, hardcore, 0, stuck_threshold)
    end
end

function clone(s::BipedalWalker)
    return BipedalWalker(
            s.episode_length;
            render=s.renderize,
            hardcore=s.hardcore,
            stuck_threshold=s.stuck_threshold
        )
end

function step!(env::BipedalWalker, action::Vector{Float32})
    # println("state:: ", env.state)
    s, reward, terminated, truncated, info = env.pyenv.step(action)
    # println("next state:: ", s)
    # println("reward:: ", reward)
    observation = process_state(env, s)
    if isapprox(observation[1:end-10], env.state[1:end-10], rtol=1e-4)
        env.stuck_counter += 1
    elseif env.stuck_counter > 0
        env.stuck_counter = 0
    end
    if env.stuck_counter >= env.stuck_threshold
        terminated = true
        reward -= 100f0
    end
    env.state = observation
    env.terminal = terminated
    env.episode_step += 1
    if env.episode_step == env.episode_length
        # env.episode_length += 1
        term = true
        # reward += 300
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
    if env.renderize
        pe = gym.make("BipedalWalker-v3", render_mode="human", hardcore=env.hardcore);
    else            
        pe = gym.make("BipedalWalker-v3", hardcore=env.hardcore);
    end
    (s, info) = env.pyenv.reset()
    observation = process_state(env, s)
    env.state = observation
    env.stuck_counter = 0
    env.terminal = false
    env.episode_step = 0
    return Float32.(env.state)
end

function close!(env::BipedalWalker)
    env.pyenv.close()
end

function renderize!(env::BipedalWalker)
    env.pyenv = gym.make("BipedalWalker-v3", render_mode="human", hardcore=env.hardcore);
    env.renderize = true
end

function process_state(env::BipedalWalker, state::Vector{Float32})
    return 2 .* ((state .- env.observation_lows) ./ (env.observation_highs .- env.observation_lows)) .- 1
end
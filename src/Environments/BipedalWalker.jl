mutable struct BipedalWalker <: AbstractEnv
    episode_length::Int64
    episode_step::Int64
    pyenv::PyObject
    state::Vector{Float32}
    terminal::Bool
    truncated::Bool
    action_type::Vector{Type}
    observation_lows::Vector{Float32}
    observation_highs::Vector{Float32}
    renderize::Bool
    hardcore::Bool
    stuck_counter::Int
    stuck_threshold::Int
    function BipedalWalker(len::Int; render::Bool=false, hardcore=false, stuck_threshold=200)
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
        return new(len, 0, pe, Float32.(obs), false, false, fill(Float32, n_actions), lows, highs, render, hardcore, 0, stuck_threshold)
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

function step!(env::BipedalWalker, action::AbstractVector{<:AbstractFloat})
    # Ensure Float32 for PyCall and internal consistency
    action_f32 = Float32.(action)
    
    # If the policy has collapsed and produced NaNs, we intercept them here
    # to prevent the physics engine (Box2D) from crashing.
    if any(isnan, action_f32)
        action_f32 = zeros(Float32, length(action_f32))
    end

    # Hard clamp inside the environment wrapper
    # This ensures that even if the policy produces an extreme value, 
    # the simulation (Box2D) receives a valid physical torque.
    safe_action = clamp.(action_f32, -1.0f0, 1.0f0)
    s, reward, terminated, truncated, info = env.pyenv.step(safe_action)
    
    observation = process_state(env, s)

    env.state = observation
    env.terminal = terminated || truncated
    env.truncated = truncated
    env.episode_step += 1
    
    if env.episode_step >= env.episode_length
        env.terminal = true
        env.truncated = true
    end
    
    return Float32.(observation), Float32.(reward), env.terminal
end

function render!(env::BipedalWalker)
    env.pyenv.render()
    return
end

function reset!(env::BipedalWalker) 
    (s, info) = env.pyenv.reset()
    observation = process_state(env, s)
    env.state = observation
    env.stuck_counter = 0
    env.terminal = false
    env.truncated = false
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
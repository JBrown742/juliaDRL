mutable struct ParticleChase <: AbstractEnv
    episode_length::Int64
    episode_step::Int64
    dims::Int
    current_position::Vector{Float32}
    target_position::Vector{Float32}
    state::Vector{Float32}
    terminal::Bool
    max_speed::Float32
    hard::Bool
    renderize::Bool
    function ParticleChase(len::Int, dims::Int; render::Bool=false, max_speed::Float32=1f0, hard::Bool=false)
        current_position = rand(Float32, dims) .* 100
        target_position = rand(Float32, dims) .* 100
        state = target_position .- current_position
        return new(len, 0, dims, current_position, target_position, state, false, max_speed, hard, render)
    end
end

function clone(s::ParticleChase)
    return ParticleChase(
        s.episode_length, s.dims;
        render=s.renderize,
        max_speed=s.max_speed,
        hard=s.hard
    )
end

function step!(env::ParticleChase, action::Vector{I}) where {I <: Integer}
    # Convert MultiDiscrete integers to Float32 actions
    # Assuming action[1] is Move-X (-1, 0, 1) and action[2] is Move-Y (-1, 0, 1)
    # However, MultiDiscrete indices are usually 1-based in Julia sampling
    # We map 1 -> -1, 2 -> 0, 3 -> 1
    float_action = Float32.(action .- 2)
    return step!(env, float_action)
end

function step!(env::ParticleChase, action::Vector{Float32})
    clipped_action = clamp.(action, -1f0, 1f0)
    env.current_position .+= clipped_action .* env.max_speed
    env.state = env.target_position .- env.current_position
    env.episode_step += 1
    if env.episode_step == env.episode_length
        env.terminal = true
    end
    distance = sqrt(sum((env.current_position .- env.target_position) .^ 2)) ./ 142f0
    if env.hard==true
        reward = distance < 1e-2 ? 10f0 : 0f0
    else
        reward = -1f0 * distance
    end
    if any(env.current_position .<= 0) || any(env.current_position .>= 100)
        env.terminal = true
        reward -= 10
    end
    if distance < 1f-2
        reward += 10
        env.target_position = rand(Float32, env.dims) .* 100
    end
    return env.state, reward, env.terminal
end

function render!(env::ParticleChase)
    plt = plot([], [], xlim=(0, 100), ylim=(0, 100), framestyle=:box, legend=false)
    scatter!(map(x -> [x], env.target_position)..., marker=:star, color="green", markersize=10)
    scatter!(map(x -> [x], env.current_position)..., marker=:circle, color="red", markersize=10)
    display(plt)
    sleep(0.05)
    return
end

function reset!(env::ParticleChase) 
    env.current_position = rand(Float32, env.dims) .* 100
    env.target_position = rand(Float32, env.dims) .* 100
    env.state = env.target_position .- env.current_position
    env.terminal = false
    env.episode_step = 0
    return Float32.(env.state)
end

function close!(env::ParticleChase)
end

function renderize!(env::ParticleChase)
    env.renderize = true
end



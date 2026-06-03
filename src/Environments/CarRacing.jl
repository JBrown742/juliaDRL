mutable struct CarRacing <: AbstractEnv
    episode_length::Int64
    episode_step::Int64
    pyenv::PyObject
    state::Array{Float32}
    state_hist::Vector{Matrix{Float32}}
    terminal::Bool
    action_type::Vector{Type}
    action_extent::Vector{Tuple{Float32, Float32}}
    renderize::Bool
    seed::Union{Int, Nothing}
    function CarRacing(len::Int; render::Bool=false, seed=nothing)
        if render
            pe = gym.make("CarRacing-v3", domain_randomize=false, render_mode="human");
        else            
            pe = gym.make("CarRacing-v3", domain_randomize=false);
        end
        obs, info = pe.reset(seed=seed)
        obs = process_state(obs)
        state_hist = fill(obs, 4)
        return new(len, 0, pe, Float32.(cat(state_hist..., dims =3)) ./ 255, state_hist, false, [Float32, Float32, Float32], [(-1,1), (0,1), (0,1)], render, seed)
    end
end

function clone(s::CarRacing)
    return CarRacing(
        s.episode_length;
        render=s.renderize,
        seed=s.seed
    )
end

function step!(env::CarRacing, action::Vector{Float32})
    scaled_actions = (action .* (last.(env.action_extent) .- first.(env.action_extent))) .- first.(env.action_extent)
    clipped_actions = clamp.(scaled_actions, first.(env.action_extent), last.(env.action_extent))
    observation, reward, terminated, truncated, info = env.pyenv.step(clipped_actions)
    popfirst!(env.state_hist)
    push!(env.state_hist, process_state(observation))
    env.state = cat(env.state_hist..., dims =3)
    env.terminal = terminated
    env.episode_step += 1
    if env.episode_step >= env.episode_length
        term = true
    else
        term = terminated || truncated
    end
    return Float32.(env.state) ./ 255, reward, term
end



function render!(env::CarRacing)
    env.pyenv.render()
    return
end

function reset!(env::CarRacing; seed=nothing) 
    (observation, info) = env.pyenv.reset(seed=seed)
    env.state_hist = fill(process_state(observation), 4)
    env.state = cat(env.state_hist..., dims =3)
    env.terminal = false
    env.episode_step = 0
    return Float32.(env.state) ./ 255
end

function close!(env::CarRacing)
    env.pyenv.close()
end

function renderize!(env::CarRacing)
    env.pyenv = gym.make("CarRacing-v3", domain_randomize=false, render_mode="human");
    env.renderize = true
end

function process_state(state::Array{UInt8, 3})
    return 0.299f0 * state[21:80, 21:60, 1] + 0.587f0 * state[21:80, 21:60, 2] + 0.114f0 * state[21:80, 21:60, 3]
end
mutable struct Pendulum <: AbstractEnv
    episode_length::Int64
    pyenv::Py
    state::Vector{Float32}
    terminal::Bool
    truncated::Bool
    action_type::Type
    action_extent::Tuple{Float32, Float32}
    renderize::Bool
    seed::Union{Int, Nothing}
    function Pendulum(len::Int64; render=false, seed=nothing)
        if render
            pe = gym.make("Pendulum-v1", render_mode="human");
        else            
            pe = gym.make("Pendulum-v1");
        end
        res = pe.reset(seed=seed)
        obs = pyconvert(Vector{Float32}, res[0])
        return new(len, pe, obs, false, false, Float32, (-2f0, 2f0), render, seed)
    end
end
function clone(s::Pendulum) return Pendulum(s.episode_length; render=s.renderize, seed=s.seed) end
function step!(env::Pendulum, action::Union{Float32, Float64, Vector{Float32}})
    u = action isa Vector ? action[1] : action
    scaled_action = clamp(2.0f0 * u, env.action_extent[1], env.action_extent[2])
    res = env.pyenv.step([scaled_action])
    observation = pyconvert(Vector{Float32}, res[0])
    reward = pyconvert(Float32, res[1])
    terminated = pyconvert(Bool, res[2])
    truncated = pyconvert(Bool, res[3])
    
    env.state = observation
    env.terminal = terminated || truncated
    env.truncated = truncated
    return env.state, reward, env.terminal 
end
function render!(env::Pendulum) env.pyenv.render() end
function close!(env::Pendulum) env.pyenv.close() end
function reset!(env::Pendulum; seed=nothing) 
    res = env.pyenv.reset(seed=seed)
    observation = pyconvert(Vector{Float32}, res[0])
    env.state = observation
    env.terminal = false
    env.truncated = false
    return env.state
end
function renderize!(env::Pendulum) env.pyenv = gym.make("Pendulum-v1", render_mode="human"); env.renderize=true end

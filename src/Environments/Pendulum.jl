mutable struct Pendulum <: AbstractEnv
    episode_length::Int64
    pyenv::PyObject
    state::Vector{Float32}
    terminal::Bool
    action_type::Type
    action_extent::Tuple{Float32, Float32}
    renderize::Bool
    function Pendulum(len::Int64; render=false)
        if render
            pe = gym.make("Pendulum-v1", render_mode="human");
        else            
            pe = gym.make("Pendulum-v1");
        end
        obs, info = pe.reset()
        return new(len, pe, obs, false, Float32, (-2f0, 2f0), render)
    end
end
function clone(s::Pendulum) return Pendulum(s.episode_length; render=s.renderize) end
function step!(env::Pendulum, action::Union{Float32, Float64, Vector{Float32}})
    u = action isa Vector ? action[1] : action
    scaled_action = clamp(2.0f0 * u, env.action_extent[1], env.action_extent[2])
    observation, reward, terminated, truncated, info = env.pyenv.step([scaled_action])
    env.state = Float32.(observation)
    env.terminal = terminated
    return env.state, Float32(reward), terminated || truncated 
end
function render!(env::Pendulum) env.pyenv.render() end
function close!(env::Pendulum) env.pyenv.close() end
function reset!(env::Pendulum) 
    (observation, info) = env.pyenv.reset()
    env.state = Float32.(observation)
    env.terminal = false
    return env.state
end
function renderize!(env::Pendulum) env.pyenv = gym.make("Pendulum-v1", render_mode="human"); env.renderize=true end

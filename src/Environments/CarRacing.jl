mutable struct CarRacing <: AbstractEnv
    episode_length::Int64
    pyenv::PyObject
    state::Array{Float32}
    terminal::Bool
    action_type::Vector{Type}
    action_extent::Vector{Tuple{Float32, Float32}}
    renderize::Bool
    function CarRacing(len::Int; render::Bool=false)
        if render
            pe = gym.make("CarRacing-v3", domain_randomize=true, render_mode="human");
        else            
            pe = gym.make("CarRacing-v3", domain_randomize=true);
        end
        obs, info = pe.reset()
        return new(len, pe, Float32.(obs) ./ 255, false, [Float32, Float32, Float32], [(-1,1), (0,1), (0,1)], render)
    end
end

function clone(s::CarRacing)
    return CarRacing(
        s.episode_length;
        render=s.renderize
    )
end

function step!(env::CarRacing, action::Vector{Float32})
    scaled_actions = clamp(2 * action, first.(env.action_extent), last.(env.action_extent))

    observation, reward, terminated, truncated, info = env.pyenv.step(scaled_actions)
    env.state = observation
    env.terminal = terminated
    return Float32.(observation) ./ 255, reward, terminated || truncated
end



function render!(env::CarRacing)
    env.pyenv.render()
    return
end

function reset!(env::CarRacing) 
    (observation, info) = env.pyenv.reset()

    env.state = observation
    env.terminal = false
    return Float32.(env.state) ./ 255
end

function close!(env::CarRacing)
    env.pyenv.close()
end

function renderize!(env::CarRacing)
    env.pyenv = gym.make("CarRacing-v3", render_mode="human");
    env.renderize = true
end

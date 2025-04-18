mutable struct CarRacing <: AbstractEnv
    episode_length::Int64
    pyenv::PyObject
    state::Array{Float64}
    terminal::Bool
    action_type::Vector{Type}
    action_extent::Vector{Tuple{Float32, Float32}}
    renderize::Bool
    function CarRacing(len::Int, pyenv::PyObject)
        obs, info = pyenv.reset()
        if render
            pe = gym.make("CarRacing-v3", domain_randomize=true, render_mode="human");
        else            
            pe = gym.make("CarRacing-v3", domain_randomize=true);
        end
        return new(len, pyenv, Float32.(obs) ./ 255, false, [Float32, Float32, Float32], [(-1,1), (0,1), (0,1)], render)
    end
end

function step!(env::CarRacing, action::Vector{Int})
    act = env.actions[action]
    observation, reward, terminated, truncated, info = env.pyenv.step(act)
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
    return Float32.(observation) ./ 255
end


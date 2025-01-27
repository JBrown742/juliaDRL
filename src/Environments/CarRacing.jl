mutable struct CarRacing{S, A} <: AbstractEnv
    episode_length::Int
    pyenv::PyObject
    state::Array{Float32, 3}
    terminal::Bool
    actions::Vector{Float32}
    function CarRacing(len::Int, pyenv::PyObject)
        obs, info = pyenv.reset()
        if render
            pe = gym.make("CarRacing-v3", domain_randomize=true, render_mode="human");
        else            
            pe = gym.make("CarRacing-v3", domain_randomize=true);
        end
        num_actions = pyenv.action_space.n
        return new{Array{Float32, 3}, Int}(len, pyenv, Float32.(obs) ./ 255, false, collect(0:num_actions-1))
    end
end

function step!(env::CarRacing, action::Int)
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


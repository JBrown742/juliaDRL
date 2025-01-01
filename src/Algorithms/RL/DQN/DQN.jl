 mutable struct DQN{S, A} <: AbstractAlgorithm
    buffer::Buffer{S,A} # CH: Does buffer need to be part of DQN? It ties the Env to the learner, which might not be necessary 
    agent::AbstractAgent
    target_model::AbstractModel
    batch_size::Int
    γ::Float64
    τ::Float64
    optimizer::AbstractRule
    optimizer_s::NamedTuple
    n::Int
    """env.actions

        DQN(model::AbstractModel, lr::Float64, reward_func::Function, batch_size::Int, γ::Float64, ϵ::Float64, optimizer::Flux.Optimise.Optimiser, max_buffer_len::Int)

    Struct for dealing with DQN problems.
    """
    function DQN(::Type{S},::Type{A}, agent::AbstractAgent, 
        batch_size::Int, γ::Float64, τ::Float64, optimizer::O) where {S,A,O <: AbstractRule}
        replay = Buffer{S,A}(50000)
        su = Flux.setup(optimizer, agent.model)
        return new{S,A}(replay, agent, deepcopy(agent.model), batch_size, γ, τ, optimizer, su, 1)
    end
    function DQN(::Type{S},::Type{A}, agent::AbstractAgent, 
        batch_size::Int, γ::Float64, τ::Float64, optimizer::O, n::Int) where {S,A,O <: AbstractRule}
        replay = Buffer{S,A}(n)
        su = Flux.setup(optimizer, agent.model)
        return new{S,A}(replay, agent, deepcopy(agent.model), batch_size, γ, τ, optimizer, su, n)
    end
end

## The environment is still baked into this algo...
function train!(alg::DQN{S,A}) where {S,A}
    # initialise vectors to store batched data
    experience_batch, sample_idxs, IS_weights = sample(alg.buffer, alg.batch_size) 
    states = Vector{S}()
    actions = Vector{A}()
    next_states = Vector{S}()
    rewards = Vector{Float32}()
    terminals = Vector{Bool}()
    for experience in experience_batch # iteratively fill data vectors
        push!(states, experience.state)
        push!(actions, experience.action)
        push!(next_states, experience.next_state)
        push!(rewards, experience.reward)
        push!(terminals, experience.done)
    end
    # need to calculate the DQN target
    # get return 
    next_state_outputs = alg.agent.model(next_states)
    # println(length(next_state_outputs))
    next_best_actions = dropdims(first.(Tuple.(argmax(next_state_outputs, dims=1))), dims=1)
    next_best_act_mask = indicatormat(next_best_actions, size(next_state_outputs, 1))
    targ_model_outputs = alg.target_model(next_states) # call the model with the next state
    argmax_act = dropdims(sum(next_best_act_mask .* targ_model_outputs, dims=1) , dims=1)
    targs = rewards .+ (1 .- Int.(terminals)) .* alg.γ .* argmax_act   
    mask = indicatormat(actions, size(targ_model_outputs, 1))
    model_outputs_untracked = alg.agent.model(states) 
    value_of_actual_actions_untracked = dropdims(sum( mask .* model_outputs_untracked, dims=1), dims=1)
    # println("value_of_actual_actions_untracked: $(value_of_actual_actions_untracked), targs: $(targs)")
    δ = targs .- value_of_actual_actions_untracked
    ∇ = Flux.gradient(alg.agent.model) do m # track gradients
        model_outputs = m(states) 
        value_of_actual_actions = dropdims(sum( mask .* model_outputs, dims=1), dims=1)
        Flux.Losses.mse(targs, value_of_actual_actions)
    end 
    Flux.update!(alg.optimizer_s, alg.agent.model, ∇[1])
    alg.buffer.bellman_errors[sample_idxs] .= abs.(δ)
end


function learning_episode!(env::E, alg::DQN{S,A}) where {E <: AbstractEnv, S,A}
    state = reset!(env)
    episode_playback = Vector{Experience{S,A}}() # initialize vector to store episode experiences
    priorities = Vector{Float64}() # initialize vector to store initial priorities
    step = 0
    while env.terminal == false # bool flag to denote whether routing has finished
        # Generalises below...
        action = get_action(alg.agent, state)

        #step!(env, agent, action, reward) # take a step of the hopper env
        new_state, reward, terminal = step!(env, action) # take a step of the hopper env
        #new_state = (node_features, env.current_graph, agent.position) # collect call the tuple of current position and graph the state
        experience = Experience{S,A}(state, action, new_state, reward, terminal)
        push!(episode_playback, experience)
        push!(priorities, 1e3)
        step+=1
        if step==env.episode_length
            env.terminal = true
        end
        # println("are the states the same?? : $(all(state[1] .== new_state[1])), $(all(state[2] .== new_state[2]))")
    end
    buffer_update!(alg.buffer, episode_playback; bellman_errors=priorities)
end



function validation_episode!(env::E, alg::DQN{S,A}; render::Bool=false) where {E<:AbstractEnv,S,A}
    # initialise vectors to store the history of states actions and rewards for the entire episode
    state = reset!(env)
    term = false
    episode_reward = Vector{Float64}()
    step=0
    while env.terminal == false # bool flag to denote whether routing has finished
        # calculate the mode outputs based on the current graph

        action = get_action(alg.agent, state; det=true)
        if render==true
            sleep(0.01)
            render!(env)
        end
        state, reward, term = step!(env, action)
        push!(episode_reward, reward)
        step+=1
        if step==env.episode_length
            env.terminal = true
        end
    end
    return sum(episode_reward)
end

function update_target!(alg::DQN{S,A}) where {S, A}
    for (idx,p) in enumerate(Flux.params(alg.agent.model))
        Flux.params(alg.target_model)[idx] .= copy(p)
    end
end

function soft_update_target!(alg::DQN{S,A}) where {S, A}
    for (idx,p) in enumerate(Flux.params(alg.model))
        mix = (1 - τ) .* Flux.params(alg.target_model)[idx] .+  alg.τ .* copy(p)
        Flux.params(alg.target_model)[idx] .= mix
    end
end



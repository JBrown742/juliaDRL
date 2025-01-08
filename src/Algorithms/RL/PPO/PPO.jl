#
# Nowhere should the algorithm be dependent on the state type
# as long as the state is of one of the 3 defined types, then it only depends on the action type
# and in that case an algorithm which supports two different action types may differ enough
# to warrant two sparate structs
mutable struct PPO <: AbstractAlgorithm
    N::Int # Number of concurrent workers for gathering experience segments.
    T::Int # Trajectory learning segment length 
    K::Int # Epoch, number of updates to carry out with a given trajectory.
    central_agent::AbstractAgent # this holds the single central agent
    # actor learners will be defined by takingn the model from the central agent, and the policies from the list of policies
    actor_learners::Vector{AbstractAgent} 
    #parameters
    batch_size::Int
    γ::Float32
    λ::Float32
    ϵ::Float32
    c1::Float32
    c2::Float32
    sync_frequency::Int
    advantage_coefficients::Vector{Float32}
    optimizer::AbstractRule
    optimizer_s::NamedTuple
    function PPO(N::Int, T::Int, K::Int, agent::A, batch_size::Int, γ::Float64, λ::Float64, 
        ϵ::Float64, c1::Float64, c2::Float64, sync_frequency::Int, optimizer::O) where {A <: AbstractAgent, O <: AbstractRule}
        m = agent.model
        # |> gpu -- removing GPU functionality for now
        als = [AbstractAgent(deepcopy(m)) for i in 1:N]
        exponents = collect(0:T*100)
        advantage_coefficients = (λ * γ) .^ exponents
        su = Flux.setup(optimizer, agent.model)
        return new(N, T, K, agent, als, batch_size, γ, λ, ϵ, c1, c2, sync_frequency, advantage_coefficients, optimizer, su)
    end
    function PPO(N::Int, T::Int, K::Int,  agent::A,
        optimizer::O) where {A <: AbstractAgent, O <: AbstractRule}
        m = agent.model
        # |> gpu -- removing GPU functionality for now
        als = [AbstractAgent(deepcopy(m)) for i in 1:N]
        exponents = collect(0:T*100)
        advantage_coefficients = (0.95 * 0.99) .^ exponents
        su = Flux.setup(optimizer, agent.model)
        return new(N, T, K, agent, als, 64, 0.99, 0.95, 0.2, 0.5, 0.001, 2, advantage_coefficients, optimizer, su)
    end

end

## The environment is still baked into this algo...
function train!(alg::PPO, transitions::Vector{Experience}, probabilities::Vector{Vector{Float32}},
    advantages::Vector{Float32}, bellman_targets::Vector{Float32}, bellman_errors::Vector{Float32})
    # initialise vectors to store batched data
    available_batch_indices = collect(1:length(transitions))
    for update_step in 1:alg.K
        if length(available_batch_indices) == 0
            available_batch_indices = collect(1:length(transitions))
        end

        if length(available_batch_indices) > alg.batch_size
            batch_indices = sample(available_batch_indices, alg.batch_size; replace=false)
        else
            batch_indices = copy(available_batch_indices)
        end
        filter!(x -> x ∉ batch_indices, available_batch_indices)
        states = Vector{AbstractObservation}()
        actions = Vector{DiscreteAct}() # currently PPO will only support discrete actions
        next_states = Vector{AbstractObservation}()
        rewards = Vector{Float32}()
        terminals = Vector{Bool}()
        for experience in transitions[batch_indices] # iteratively fill data vectors
            push!(states, experience.state)
            push!(actions, experience.action)
            push!(next_states, experience.next_state)
            push!(rewards, experience.reward)
            push!(terminals, experience.done)
        end
        batch_probabilities = probabilities[batch_indices]
        batch_advantages = (advantages[batch_indices] .- mean(advantages[batch_indices])) ./ std(advantages[batch_indices])
        batch_bellman_targets = reshape(bellman_targets[batch_indices], (1, length(batch_indices)))
        actual_action_mask = indicatormat(actions, first(alg.central_agent.model.model.layers[end].paths[1].bias |> size))
        probability_mask = hcat(infer_mask.(batch_probabilities)...) # vector of action masks
        ∇ = Flux.gradient(alg.central_agent.model) do m # track gradients
            action_probs, state_values = m(reduce(hcat, states))
            masked_probs = masked_probabilities(probability_mask, action_probs)
            L_q_learning = Flux.Losses.mse(batch_bellman_targets, state_values)
            new_probs = dropdims(sum(masked_probs .* actual_action_mask, dims=1), dims=1)
            old_probs = dropdims(sum(hcat(batch_probabilities...) .* actual_action_mask, dims=1), dims=1)
            r = new_probs ./ old_probs
            clamped_r = clamp.(r, 1 - alg.ϵ, 1 + alg.ϵ)
            vals = minimum.(batch_advantages .* clamped_r)
            L_CLIP = 1 * mean(vals)
            entropy = -1 * mean(sum(exp.(action_probs) .* log2.(exp.(action_probs)), dims=1))
            L_q_learning - alg.c1 * L_CLIP - alg.c2 * entropy
        end
        Flux.update!(alg.optimizer_s, alg.central_agent.model, ∇[1])
    end
end

function collect_trajectory_segment!(env::E, agent::A, info::Dict{Symbol, Any}) where {E <: AbstractEnv, A <: AbstractAgent}
    T::Int = info[:T]
    γ::Float32 = info[:γ]
    advantage_coefficients::Vector{Float32} = info[:advantage_coefficients]
    # initialize vectors to store all the transitions encountered
    local_segment_count = 0
    total_trajectory = Vector{Experience}()
    all_targets = Vector{Float32}()
    all_errors = Vector{Float32}()
    all_advantages = Vector{Float32}()
    trajectory_probabilities = Vector{Vector{Float32}}()
    # When starting a new segment collection we need to decide whether to continue with a current episode or start a-new

    while local_segment_count < T # while we still haven't fully collected a segment
        # initialise inner vectors to collect each subsegment. Important for if T > ep_len
        current_ep_trajectory = Vector{Experience}() 
        local_targets = Vector{Float32}()
        local_errors = Vector{Float32}()
        if env.terminal
            state = reset!(env)
        else
            state = env.state
        end
        while env.terminal == false && local_segment_count < T # bool flag to denote whether episode has finished
            # Generalises below...
            action, state_value, probs = get_action(PPO, agent, state; mask=env.action_mask) # get the action, the value and the probability
            push!(trajectory_probabilities, probs)
            new_state, reward, terminal = step!(env, action) # take a step of the hopper envs
            experience = Experience(state, action, new_state, reward, terminal)
            push!(current_ep_trajectory, experience)
            _, next_state_value, _ = get_action(PPO, agent, state; mask=env.action_mask) # get the value of the next state to calculate bellman error & advantage
            target = reward .+ (1 .- Int.(terminal)) .* γ .* next_state_value
            push!(local_targets, target[1])
            push!(local_errors, target[1] - state_value[1])
            state = new_state
            local_segment_count += 1
            env.terminal = terminal
            if env.terminal || local_segment_count == T
                len_current_seg = length(current_ep_trajectory)
                advantages = zeros(Float32, len_current_seg)
                for t in 1:len_current_seg
                    advantages[t] = sum(local_errors[t:end] .* advantage_coefficients[1:(len_current_seg-t+1)])
                end
                push!(total_trajectory, current_ep_trajectory...)
                push!(all_advantages, advantages...)
                push!(all_errors, local_errors...)
                push!(all_targets, local_targets...)
                current_ep_trajectory = Vector{Experience}()
                local_targets = Vector{Float32}()
                local_errors = Vector{Float32}()
            end
        end
    end
    return total_trajectory, trajectory_probabilities, all_advantages, all_targets, all_errors
end

function full_training_procedure!(alg::PPO, envs::Vector{E}) where {E <: AbstractEnv}
    info = Dict{Symbol, Any}(:T => alg.T, :γ => alg.γ, :advantage_coefficients => alg.advantage_coefficients)
    agents = alg.actor_learners
    results = pmap(collect_trajectory_segment!, WorkerPool(workers()), envs, agents, fill(info, alg.N))
    all_transitions, all_probabilities, all_advantages, all_targets, all_errors = unzip(results)
    train!(alg, all_transitions, all_probabilities, all_advantages, all_targets, all_errors)
end



function validation_episode!(env::E, alg::PPO; render::Bool=false) where {E<:AbstractEnv}
    # initialise vectors to store the history of states actions and rewards for the entire episode
    state = reset!(env)
    term = false
    episode_reward = Vector{Float64}()
    step=0
    while env.terminal == false # bool flag to denote whether routing has finished
        # calculate the mode outputs based on the current graph
        action, value, probs = get_action(PPO, alg.central_agent, state; det=true)
        if render==true
            sleep(0.005)
            render!(env)
        end
        state, reward, term = step!(env, action)
        push!(episode_reward, reward)
        step+=1
        if step==10000 || term
            env.terminal = true
        end
    end
    return sum(episode_reward)
end

function update_actor_learners!(alg::PPO)
    for (idx,p) in enumerate(Flux.params(alg.central_agent.model))
        for agent_idx in 1:alg.N
            Flux.params(alg.actor_learners.model[agent_idx])[idx] .= copy(p |> cpu)
        end
    end
end

function unzip(a; dims = 1)
    return map(x -> cat(getfield.(a, x)..., dims=dims), fieldnames(eltype(a)))
end

function masked_probabilities(mask::Array{Float32}, outputs::Array{Float32})
    weights = softmax(mask .+ outputs; dims = 1)
    return weights
end

function infer_mask(probabilities::Vector{Float32})
    idxs = findall(iszero, probabilities)
    N = length(probabilities)
    mask = zeros(Float32, N) # build a mask vector to zero out all nodes ∉ NH
    mask[idxs] .= mask[idxs] .- Inf32
    return mask
end

# ------------ get_actions functions ------- #
function get_action(::Type{PPO}, agent::A, obs::O; det=false, mask=nothing) where {A <: AbstractAgent, O <: AbstractObservation}
    outputs, value = agent.model(obs)
    if isnothing(mask)
        mask = ones(length(outputs))
    end
    masked_outputs = outputs .+ mask
    ws = softmax(masked_outputs; dims = 1)
    indices = collect(1:length(ws))
    if det == true
        action = argmax(ws)
    else
        action = sample(indices, Weights(ws))
    end
    return action, value, ws
end

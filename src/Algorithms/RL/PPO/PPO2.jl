mutable struct PPO2{S, A} <: LearningAlgorithm
    N::Int # Number of concurrent workers for gathering experience segments.
    T::Int # Trajectory learning segment length 
    K::Int # Epoch, number of updates to carry out with a given trajectory.
    central_model::LearningModel
    actor_learners::Vector{LearningModel}
    policies::Vector{Function}
    policy_ϵ::Vector{Float32} # need to make this an active element
    #parameters
    batch_size::Int
    γ::Float32
    λ::Float32
    ϵ::Float32
    c1::Float32
    c2::Float32
    sync_frequency::Int
    advantage_coefficients::Vector{Float32}
    optimizer::Flux.Optimise.AbstractOptimiser
    function PPO2(::Type{S},::Type{A}, N::Int, T::Int, K::Int, model::LearningModel, 
        policies::Vector{F}, policy_ϵ, batch_size::Int, γ::Float64, λ::Float64, 
        ϵ::Float32, c1::Float32, c2::Float32, sync_frequency::Int, optimizer::O) where {S,A,O <: Flux.Optimise.AbstractOptimiser, F <: Function}
        cm = model |> gpu
        als = [deepcopy(model) for i in 1:N]
        exponents = collect(0:T*100)
        advantage_coefficients = (λ * γ) .^ exponents

        return new{S,A}(N, T, K, cm, als,  policies, policy_ϵ, batch_size, γ, λ, ϵ, c1, c2, sync_frequency, advantage_coefficients, optimizer)
    end
    function PPO2(::Type{S},::Type{A}, N::Int, T::Int, K::Int,  model::LearningModel, 
        policies::Vector{F}, 
        optimizer::O) where {S,A,O <: Flux.Optimise.AbstractOptimiser, F <: Function}
        cm = model |> gpu
        als = [deepcopy(model) for i in 1:N]
        exponents = collect(0:T)
        advantage_coefficients = (0.95 * 0.99) .^ exponents
        policy_ϵ = collect(0:1/(N-1):1)
        return new{S,A}(N, T, K, cm, als, policies, policy_ϵ, 64, 0.99, 0.95, 0.2, 0.5, 0.001, 2, advantage_coefficients, optimizer)
    end
    function PPO2(::Type{S},::Type{A}, N::Int, T::Int, K::Int,  model::LearningModel, 
        policy::F, 
        optimizer::O) where {S,A,O <: Flux.Optimise.AbstractOptimiser, F <: Function}
        cm = model |> gpu
        als = [deepcopy(model) for i in 1:N]
        exponents = collect(0:T)
        advantage_coefficients = (0.95 * 0.99) .^ exponents
        policy_ϵ = collect(0:1/(N-1):1)
        return new{S,A}(N, T, K, cm, als, fill(policy, N), policy_ϵ, 64, 0.99, 0.95, 0.2, 0.5, 0.001, 2, advantage_coefficients, optimizer)
    end
end

## The environment is still baked into this algo...
function train!(alg::PPO2{S,A}, transitions::Vector{Experience}, probabilities::Vector{Vector{Float32}},
    advantages::Vector{Float32}, bellman_targets::Vector{Float32}, bellman_errors::Vector{Float32}) where {S,A}
    θ = Flux.params(alg.central_model) # get the model parameters. # Model needs to be in a consistent format whereby θ encapsulates all params
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
        states = Vector{Vector{Float32}}()
        actions = Vector{Int}()
        next_states = Vector{Vector{Float32}}()
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
        actual_action_mask = indicatormat(actions, size.(θ)[end-2][1])
        probability_mask = hcat(infer_mask.(batch_probabilities)...) # vector of action masks
        ∇ = Flux.gradient(θ) do # track gradients
            action_probs, state_values = alg.central_model(reduce(hcat, states) |> gpu) .|> cpu
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
        Flux.update!(alg.optimizer, θ, ∇)
        if any(any.([isnan.(p) for p in θ]))
            # println("NaN detected in network weights:: Printing most recent values...")
            # println("Last states:: $(states)")
            # println("next states:: $(next_states)")
            # println("Last rewards:: $(rewards)")
            # println("Last batch probs:: $(batch_probabilities)")
            # println("Last batch advantages:: $(batch_advantages)")
            # println("Last batch targets:: $(batch_bellman_targets)")
            # println("Last max, min, meaan ∇:: $((maximum.([g for g in ∇]), minimum.([g for g in ∇]), mean.([g for g in ∇])))")
            throw(ErrorException("NaN detected"))
        end
    end
end

function collect_trajectory_segment!(envs::Vector{E}, alg::PPO2{S,A}, agent::Int) where {E <: AbstractEnv, S,A}
    # initialize vectors to store all the transitions encountered
    local_segment_count = 0
    total_trajectory = Vector{Experience{S,A}}()
    all_targets = Vector{Float32}()
    all_errors = Vector{Float32}()
    all_advantages = Vector{Float32}()
    trajectory_probabilities = Vector{Vector{Float32}}()
    # When starting a new segment collection we need to decide whether to continue with a current episode or start a-new

    while local_segment_count < alg.T # while we still haven't fully collected a segment
        # initialise inner vectors to collect each subsegment. Important for if T > ep_len
        current_ep_trajectory = Vector{Experience{S, A}}() 
        local_targets = Vector{Float32}()
        local_errors = Vector{Float32}()
        if envs[agent].terminal
            state = reset!(envs[agent])
        else
            state = envs[agent].state
        end
        while envs[agent].terminal == false && local_segment_count < alg.T # bool flag to denote whether episode has finished
            # Generalises below...
            action, state_value, probs = alg.policies[agent](envs[agent], alg, agent;ϵ=alg.policy_ϵ[agent]) # get the action, the value and the probability
            push!(trajectory_probabilities, probs) # record all probabilities instead of only that for the action
            new_state, reward, terminal, info = step!(envs[agent], action) # take a step of the hopper envs
            experience = Experience{S,A}(state, action, new_state, reward, terminal)
            push!(current_ep_trajectory, experience)
            _, next_state_value, _ = alg.policies[agent](envs[agent], alg, agent) # get the value of the next state to calculate bellman error & advantage
            target = reward .+ (1 .- Int.(terminal)) .* alg.γ .* next_state_value
            push!(local_targets, target[1])
            push!(local_errors, target[1] - state_value[1])
            state = new_state
            local_segment_count += 1
            envs[agent].terminal = terminal
            if envs[agent].terminal || local_segment_count == alg.T
                len_current_seg = length(current_ep_trajectory)
                advantages = zeros(Float32, len_current_seg)
                for t in 1:len_current_seg
                    advantages[t] = sum(local_errors[t:end] .* alg.advantage_coefficients[1:(len_current_seg-t+1)])
                end
                push!(total_trajectory, current_ep_trajectory...)
                push!(all_advantages, advantages...)
                push!(all_errors, local_errors...)
                push!(all_targets, local_targets...)
                current_ep_trajectory = Vector{Experience{S, A}}()
                local_targets = Vector{Float32}()
                local_errors = Vector{Float32}()
            end
        end
    end
    return total_trajectory, trajectory_probabilities, all_advantages, all_targets, all_errors
end
function collect_trajectory_segment_alt!(env::ParallelRoutingEnv, model::LearningModel, policy::Function, info::Dict{Symbol, Any})
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
    ep_count = 1
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
            state = env.state
            # Generalises below...
            action, state_value, probs = policy(env, model, state) # get the action, the value and the probability
            push!(trajectory_probabilities, Vector{Float32}.(eachcol(probs))...)
            new_state, reward, terminal = step!(env, action) # take a step of the hopper env
            push!(current_ep_trajectory, Experience.(state, action, new_state, reward, terminal)...)
            _, next_state_value, _ = policy(env, model, new_state) # get the value of the next state to calculate bellman error & advantage
            target = reward .+ (1 .- Int.(terminal)) .* γ .* dropdims(next_state_value, dims = 1)
            push!(local_targets, target...)
            push!(local_errors, (target - dropdims(state_value, dims = 1))...)
            state = new_state
            local_segment_count += 1
            env.terminal = terminal[1]
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


function collect_trajectory_segment_alt!(env::E, model::LearningModel, policy::Function, info::Dict{Symbol, Any}) where {E <: AbstractEnv}
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
            action, state_value, probs = policy(env, model) # get the action, the value and the probability
            push!(trajectory_probabilities, probs)
            new_state, reward, terminal = step!(env, action) # take a step of the hopper envs
            experience = Experience(state, action, new_state, reward, terminal)
            push!(current_ep_trajectory, experience)
            _, next_state_value, _ = policy(env, model) # get the value of the next state to calculate bellman error & advantage
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

function full_training_procedure_alt!(alg::PPO2{S, A}, envs::Vector{E}) where {S, A, E <: AbstractEnv}
    info = Dict{Symbol, Any}(:T => alg.T, :γ => alg.γ, :advantage_coefficients => alg.advantage_coefficients)
    models = alg.actor_learners
    policies = alg.policies
    results = pmap(collect_trajectory_segment_alt!, WorkerPool(workers()), envs, models, policies, fill(info, alg.N))
    all_transitions, all_probabilities, all_advantages, all_targets, all_errors = unzip(results)
    train!(alg, all_transitions, all_probabilities, all_advantages, all_targets, all_errors)
end

function full_training_procedure!(alg::PPO2{S, A}, envs::Vector{E}) where {S, A, E <: AbstractEnv}
    all_transitions = Vector{Experience{S, A}}()
    all_probabilities = Vector{Vector{Float32}}()
    all_advantages = Vector{Float32}()
    all_targets = Vector{Float32}()
    all_errors= Vector{Float32}()
    for al_idx in 1:alg.N
        transitions, probabilities, advantages, targets, errors = collect_trajectory_segment!(envs, alg, al_idx)
        push!(all_transitions, transitions...)
        push!(all_probabilities, probabilities...)
        push!(all_advantages, advantages...)
        push!(all_targets, targets...)
        push!(all_errors, errors...)
    end
    train!(alg, all_transitions, all_probabilities, all_advantages, all_targets, all_errors)
end




function validation_episode!(env::E, alg::PPO2{S,A}; render::Bool=false) where {E<:AbstractEnv,S,A}
    # initialise vectors to store the history of states actions and rewards for the entire episode
    state = reset!(env)
    term = false
    episode_reward = Vector{Float64}()
    step=0
    while env.terminal == false # bool flag to denote whether routing has finished
        # calculate the mode outputs based on the current graph
        action, value, probs = alg.policies[1](env, alg.central_model; det=true)
        if render==true
            sleep(0.1)
            render!(env, action)
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


function validation_episode!(env::ParallelRoutingEnv, alg::PPO2{S,A}; render::Bool=false) where {S,A}
    # initialise vectors to store the history of states actions and rewards for the entire episode
    state = reset!(env)
    term = false
    episode_reward = Vector{Float64}()
    step=0
    while env.terminal == false # bool flag to denote whether routing has finished
        # calculate the mode outputs based on the current graph
        action, value, probs = alg.policies[1](env, alg.central_model; det=true)
        state, reward, term = step!(env, dropdims(action, dims=1))
        push!(episode_reward, reward...)
        step+=1
        if step==env.episode_length || all(term)
            env.terminal = true
        end
    end
    return sum(episode_reward)
end

function update_actor_learners!(alg::PPO2{S,A}) where {S, A}
    for (idx,p) in enumerate(Flux.params(alg.central_model))
        for agent_idx in 1:alg.N
            Flux.params(alg.actor_learners[agent_idx])[idx] .= copy(p |> cpu)
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





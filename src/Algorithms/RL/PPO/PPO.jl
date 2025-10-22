# using G as the abtract typeholder for action from the synonym 'gesture' since A is used for agent

"""
    calculate_advantage_coefficients(T::Int, γ::Float32, λ::Float32)

Calculates the coefficients for Generalized Advantage Estimation (GAE).
"""
function calculate_advantage_coefficients(T::Int, γ::Float32, λ::Float32)
    return (λ * γ) .^ (0:T)
end


mutable struct PPO{G <: AbstractAction} <: AbstractAlgorithm
    N::Int # Number of concurrent workers for gathering experience segments.
    T::Int # Trajectory learning segment length 
    K::Int # Epoch, number of updates to carry out with a given trajectory.
    central_agent::Union{StandardActorCritic, CombinedActorCritic} # this holds the single central agent
    # actor learners will be defined by takingn the model from the central agent, and the policies from the list of policies
    worker_agents::Union{Vector{StandardActorCritic}, Vector{CombinedActorCritic}}
    #parameters
    batch_size::Int # number of samples to use per learning update
    γ::Float32 # Discount factor
    λ::Float32 # GAE lambda
    ϵ::Float32 # the epsilon used in the clipped policy objective to ensure policy remains in trust region
    c1::Float32 # Used in CombinedActorCritic architectures to weight the value loss relative to CLIP surrogate objective
    c2::Float32 # to weight the entropy objective relative to the clipped objective
    sync_frequency::Int # the number of learning interations to carry out between each synchronisation from central model to workers
    advantage_coefficients::Vector{Float32} # Holds advantage coefficients for the algorithm.
    # debug variables
    average_policy_loss::Vector{Float32}
    average_value_loss::Vector{Float32}
    average_entropy_loss::Vector{Float32}
    clipfrac::Vector{Float32}
    approxkl::Vector{Float32}
    average_gradient_norm::Vector{Float32}

    function PPO(::Type{G}, N::Int, T::Int, K::Int, agent::A;
                 batch_size::Int=64, γ::Float64=0.99, λ::Float64=0.95,
                 ϵ::Float64=0.2, c1::Float64=0.5, c2::Float64=0.001,
                 sync_frequency::Int=2) where {G <: AbstractAction, A <: AbstractAgent}

        wrkrs = [deepcopy(agent) for _ in 1:N]

        γ_f32, λ_f32, ϵ_f32, c1_f32, c2_f32 = Float32.((γ, λ, ϵ, c1, c2))
        
        advantage_coefficients = calculate_advantage_coefficients(T, γ_f32, λ_f32)
        
        new{G}(N, T, K, agent, wrkrs, batch_size, γ_f32, λ_f32, ϵ_f32, c1_f32, c2_f32, sync_frequency, advantage_coefficients,
              Vector{Float32}(), Vector{Float32}(), Vector{Float32}(), Vector{Float32}(), Vector{Float32}(), Vector{Float32}())
    end
end


function train!(alg::PPO{G}, states::Vector{S}, actions::Vector{G}, probabilities::Union{Vector{Vector{Float32}}, Vector{Float32}},
    advantages::Vector{Float32}, bellman_targets::Vector{Float32}, bellman_errors::Vector{Float32}) where {G <: AbstractAction, S <: AbstractObservation}
    
    # Normalize the advantages over the current batch
    mean_advantage_for_batch = mean(advantages)
    std_advantage_for_batch = std(advantages)
    normalized_advantages = (advantages .- mean_advantage_for_batch) ./ std_advantage_for_batch + 1e-6

    # 1. Calculate the magnitude of the Bellman Errors
    # The magnitude (absolute value) is what matters for prioritization.
    abs_errors = abs.(bellman_errors)

    # 2. Normalize to create a probability distribution
    # Add a small epsilon for numerical stability and to ensure all samples have a chance of being selected.
    # A small constant 'ϵ' (e.g., 1e-6) prevents zero probability.
    ϵ = 1f-6
    sampling_weights = abs_errors .+ ϵ 
    # The 'sample' function from StatsBase handles normalization internally, 
    # but explicitly normalizing makes the intent clearer and can be useful for debugging:
    sampling_probs = sampling_weights ./ sum(sampling_weights)
    
    # initialise vectors to store batched data
    available_batch_indices = 1:length(states)
    for _ in 1:alg.K
        if length(available_batch_indices) > alg.batch_size
            batch_indices = sample(
            available_batch_indices, 
            sampling_probs,       # The probability distribution/weights
            alg.batch_size, 
            replace=true          # PER-style sampling is usually with replacement
        )
        else
            batch_indices = collect(available_batch_indices)
        end

        batch_states = states[batch_indices]
        batch_actions = actions[batch_indices]
        batch_probabilities = probabilities[batch_indices]
        batch_advantages = normalized_advantages[batch_indices]
        batch_bellman_targets = bellman_targets[batch_indices]
        gradient_calculation_and_update!(alg, alg.central_agent, batch_states, batch_actions, batch_advantages, batch_probabilities, batch_bellman_targets)
    end
end

# ------------------- Dispatches for the gradient calculation depending on whether ------------------- #
# --------------------------------- parameters are shared and ---------------------------------------- #
# ----------------------- whether we are using discrete or continuous actions ------------------------ #

## OK
function gradient_calculation_and_update!(alg::PPO{DiscreteAct}, agent::StandardActorCritic, states::Vector{AbstractObservation}, 
    actions::Vector{DiscreteAct}, batch_advantages::Vector{Float32}, batch_probabilities::Vector{Vector{Float32}}, 
    batch_bellman_targets::Vector{Float32})
    # Get the action mask matrix for all actions taken in this batch
    actual_action_mask = indicatormat(actions, first(agent.actor_model.model.layers[end].bias |> size))
    probability_mask = hcat(infer_mask.(batch_probabilities)...) # vector of action masks

    act_m = agent.actor_model.model
    crit_m = agent.critic_model.model
    # define explicity policy loss function 
    function policy_loss_calculation(m)
        action_probs = m(reduce(hcat, states))
        masked_probs = masked_probabilities(probability_mask, action_probs)
        new_probs = dropdims(sum(masked_probs .* actual_action_mask, dims=1), dims=1)
        old_probs = dropdims(sum(hcat(batch_probabilities...) .* actual_action_mask, dims=1), dims=1)
        r = new_probs ./ old_probs
        clamped_r = clamp.(r, 1 - alg.ϵ, 1 + alg.ϵ)
        L_CLIP = mean(min.(r .* batch_advantages, clamped_r .* batch_advantages))
        entropy = -1 * mean(sum(action_probs .* log.(action_probs), dims=1))
        return -1 * L_CLIP - alg.c2 * entropy
    end
    # define value loss function
    function value_loss(m)
        state_values = dropdims(m(reduce(hcat, states)), dims=1)
        return  Flux.Losses.mse(batch_bellman_targets, state_values)
    end
    # get actor and critic gradients
    ∇_actor = gradient(act_m -> policy_loss_calculation(act_m), act_m) 
    ∇_critic = gradient(crit_m -> value_loss(crit_m), crit_m)  # Updated to use Flux.trainable
    # update actor nd critic models 
    Flux.update!(agent.actor_model._optimizer_state, agent.actor_model.model, ∇_actor[1])
    Flux.update!(agent.critic_model._optimizer_state, agent.critic_model.model, ∇_critic[1])
end

# OK
function gradient_calculation_and_update!(alg::PPO{DiscreteAct}, agent::CombinedActorCritic, states::Vector{AbstractObservation}, 
    actions::Vector{DiscreteAct}, batch_advantages::Vector{Float32}, batch_probabilities::Vector{Vector{Float32}}, 
    batch_bellman_targets::Vector{Float32})
    actual_action_mask = indicatormat(actions, first(agent.combined_model.model.layers[end].paths[1].bias |> size))
    probability_mask = hcat(infer_mask.(batch_probabilities)...) # vector of action masks

    combined_m = agent.combined_model.model
    # define explicity policy loss function 
    function loss_calculation(m)
        action_probs, state_values = m(reduce(hcat, states))
        masked_probs = masked_probabilities(probability_mask, action_probs)
        L_q_learning = Flux.Losses.mse(batch_bellman_targets, dropdims(state_values, dims=1))
        new_probs = dropdims(sum(masked_probs .* actual_action_mask, dims=1), dims=1)
        old_probs = dropdims(sum(hcat(batch_probabilities...) .* actual_action_mask, dims=1), dims=1)
        r = new_probs ./ old_probs
        clamped_r = clamp.(r, 1 - alg.ϵ, 1 + alg.ϵ)
        L_CLIP = mean(min.(r .* batch_advantages, clamped_r .* batch_advantages))
        entropy = -1 * mean(sum(action_probs .* log.(exp.(action_probs)), dims=1))
        return alg.c1 * L_q_learning - L_CLIP - alg.c2 * entropy
    end
    # get combined and critic gradients
    ∇_combined = gradient(combined_m -> loss_calculation(combined_m), combined_m) 
    # update combined nd critic models 
    Flux.update!(agent.combined_model._optimizer_state, agent.combined_model.model, ∇_combined[1])
end

# OK
function gradient_calculation_and_update!(alg::PPO{ContinuousAct}, agent::StandardActorCritic, states::Vector{AbstractObservation}, 
    actions::Vector{ContinuousAct}, batch_advantages::Vector{Float32}, batch_probabilities::Vector{Float32}, 
    batch_bellman_targets::Vector{Float32})
    # get_actor and critic models
    act_m = agent.actor_model.model
    crit_m = agent.critic_model.model
    # define explicity policy loss function 
    function policy_loss_calculation(m)
        μ, log_σ = dropdims.(m(reduce(hcat, states)), dims=1)
        σ = exp.(log_σ)
        new_log_probs = log_gauss_pdf(actions, μ, σ)
        old_log_probs = batch_probabilities
        r = exp.(new_log_probs .- old_log_probs)
        clamped_r = clamp.(r, 1 - alg.ϵ, 1 + alg.ϵ)
        L_CLIP = mean(min.(r .* batch_advantages, clamped_r .* batch_advantages))     
        entropy = mean(0.5 .* log.(2 * π .* σ .^ 2 .+ 0.5)) # there is a closed form for the entropy of a gaussian
        return -1 * L_CLIP - alg.c2 * entropy
    end
    # define value loss function
    function value_loss(m)
        state_values = dropdims(m(reduce(hcat, states)), dims=1)
        return Flux.Losses.mse(batch_bellman_targets, state_values)
    end
    # get actor and critic gradients
    ∇_actor = gradient(act_m -> policy_loss_calculation(act_m), act_m) 
    ∇_critic = gradient(crit_m -> value_loss(crit_m), crit_m)  # Updated to use Flux.trainable
    # update actor nd critic models 
    Flux.update!(agent.actor_model._optimizer_state, agent.actor_model.model, ∇_actor[1])
    Flux.update!(agent.critic_model._optimizer_state, agent.critic_model.model, ∇_critic[1])
    
end
# OK
function gradient_calculation_and_update!(alg::PPO{ContinuousAct}, agent::CombinedActorCritic, states::Vector{AbstractObservation}, 
    actions::Vector{ContinuousAct}, batch_advantages::Vector{Float32}, batch_probabilities::Vector{Float32}, 
    batch_bellman_targets::Vector{Float32})

    combined_m = agent.combined_model.model
    # define explicity policy loss function 
    function loss_calculation(m)
        μ, log_σ, state_values = dropdims.(m(reduce(hcat, states)), dims=1)
        σ = exp.(log_σ)
        new_log_probs = log_gauss_pdf(actions, μ, σ)
        old_log_probs = batch_probabilities
        r = exp.(new_log_probs .- old_log_probs)
        clamped_r = clamp.(r, 1 - alg.ϵ, 1 + alg.ϵ)
        L_CLIP = mean(min.(r .* batch_advantages, clamped_r .* batch_advantages))
        L_value_loss = Flux.Losses.mse(batch_bellman_targets, state_values)
        entropy = mean(0.5 .* log.(2 * π .* σ .^ 2 .+ 0.5)) # there is a closed form for the entropy of a gaussian
        return alg.c1 * L_value_loss - L_CLIP - alg.c2 * entropy
    end
    # get combined and critic gradients
    ∇_combined = gradient(combined_m -> loss_calculation(combined_m), combined_m) 
    # update combined nd critic models 
    Flux.update!(agent.combined_model._optimizer_state, agent.combined_model.model, ∇_combined[1])
end
# OK
function gradient_calculation_and_update!(alg::PPO{MultiContinuousAct}, agent::StandardActorCritic, states::Vector{AbstractObservation}, 
    actions::Vector{MultiContinuousAct}, batch_advantages::Vector{Float32}, batch_probabilities::Vector{Float32}, 
    batch_bellman_targets::Vector{Float32})
    state_dim = ndims(states[1])
    state_mat = cat(states..., dims=state_dim+1)
    # get actor and critic model
    act_m = agent.actor_model.model
    crit_m = agent.critic_model.model
    # define explicity policy loss function 
    function policy_loss_calculation(m)
        μ, log_σ = m(state_mat)
        σ = exp.(log_σ)

        new_log_probs = log_gauss_pdf_multi(actions, μ, σ)
        old_log_probs = batch_probabilities

        r = exp.(new_log_probs .- old_log_probs)
        clamped_r = clamp.(r, 1 - alg.ϵ, 1 + alg.ϵ)

        # PPO's clipped objective
        L_CLIP = mean(min.(r .* batch_advantages, clamped_r .* batch_advantages))
        
        # Entropy bonus
        entropy = mean(sum(0.5f0 .* (log.(2f0 * π .* σ .^ 2) .+ 0.5f0), dims=1))
        
        return -L_CLIP - alg.c2 * entropy
    end
    # define value loss function
    function value_loss(m)
        state_values = dropdims(m(state_mat), dims=1)
        return Flux.Losses.mse(batch_bellman_targets, state_values)
    end
    # get actor and critic gradients
    ∇_actor = gradient(act_m -> policy_loss_calculation(act_m), act_m) 
    ∇_critic = gradient(crit_m -> value_loss(crit_m), crit_m)  # Updated to use Flux.trainable
    # update actor nd critic models 
    Flux.update!(agent.actor_model._optimizer_state, agent.actor_model.model, ∇_actor[1])
    Flux.update!(agent.critic_model._optimizer_state, agent.critic_model.model, ∇_critic[1])
end

# OK
function gradient_calculation_and_update!(alg::PPO{MultiContinuousAct}, agent::CombinedActorCritic, states::Vector{AbstractObservation}, 
    actions::Vector{MultiContinuousAct}, batch_advantages::Vector{Float32}, batch_probabilities::Vector{Float32}, 
    batch_bellman_targets::Vector{Float32})
    state_dim = ndims(states[1])
    state_mat = cat(states..., dims=state_dim+1)
    combined_m = agent.combined_model.model
    # define explicity policy loss function 
    function loss_calculation(m)
        μ, log_σ, state_values = m(state_mat)
        σ = exp.(log_σ)
        new_log_probs = log_gauss_pdf_multi(actions, μ, σ)
        old_log_probs = batch_probabilities
        r = exp.(new_log_probs .- old_log_probs)
        clamped_r = clamp.(r, 1 - alg.ϵ, 1 + alg.ϵ)

        # PPO's clipped objective
        L_CLIP = mean(min.(r .* batch_advantages, clamped_r .* batch_advantages))
        
        L_value_loss = Flux.Losses.mse(batch_bellman_targets, dropdims(state_values, dims=1))
        
        # Entropy bonus
        entropy = mean(sum(0.5f0 .* (log.(2f0 * π .* σ .^ 2) .+ 0.5f0), dims=1)) # Closed-form entropy for Gaussian
        return alg.c1 * L_value_loss - L_CLIP  - alg.c2 *entropy
    end
    # get combined and critic gradients
    ∇_combined = gradient(combined_m -> loss_calculation(combined_m), combined_m) 
    # update combined nd critic models 
    Flux.update!(agent.combined_model._optimizer_state, agent.combined_model.model, ∇_combined[1])
end

function collect_trajectory_segment!(::Type{E}, agent::A, info::Dict{Symbol, Any}; random_policy::Bool=false) where {E <: AbstractEnv, A <: AbstractAgent}
    # Using a dictionary here is a hacky means of using distributed processing and passing alg level parameters to each worker_agents
    # without needing to explicity transfer the alg struct which is much more light weight
    T::Int = info[:T] # Trajectory length
    γ::Float32 = info[:γ] # Discount factor
    advantage_coefficients::Vector{Float32} = info[:advantage_coefficients] # Advantage coefficients for GAE
    action_type = info[:action_type] # Action type
    algtype = info[:algtype] # Algorithm type

    # initialize vectors to store all the transitions encountered
    local_segment_count = 0 # This keeps an internal count on the worker as to which step we are at
    trajectory_states = Vector{AbstractObservation}() # the states encountered within the trajectory
    trajectory_actions = Vector{action_type}() # actions taken in the trajectory
    all_targets = Vector{Float32}() # the bellman targets of each action -- r_t + γV(s_{t+1})
    all_errors = Vector{Float32}() # Bellman target - V(s_t)
    all_advantages = Vector{Float32}() # weighted sum of the bellman errors for the remainder of the trajectory, with later errors downweighted.
    # This advantage estimate provides a proxy for the "goodness" of the action. An action which has a high reward and and a high estimate for the next state value will haven
    # a high positive advantage. We strongly associate this to the current action. We also partially credit the current action for the following rewards downweighted by λ.
    # This bellman error can also be used as a measure of suprise.
    if action_type <: DiscreteAct
        # if we have discrete actions the trajectory probability will have to be vectors. We need to retain the probability of every action under the policy used during 
        # collection for use at training time.
        trajectory_probabilities = Vector{Vector{Float32}}()
    else
        # If we have a continuous action we need only the action itself given that we can infer the probability of any other action if we have the mean and standard deviation
        trajectory_probabilities = Vector{Float32}()
    end

    # When starting a new segment collection we need to decide whether to continue with a current episode or start a-new

    while local_segment_count < T # while we still haven't fully collected a segment
        # initialise inner vectors to collect each subsegment. Important for if T > ep_len
        current_ep_states = Vector{AbstractObservation}()
        current_ep_actions = Vector{action_type}()
        local_targets = Vector{Float32}()
        local_errors = Vector{Float32}()
        # The pattern below is used because we have to initialize the environment on each worker, but we cant rely simply on assigning it to the env variable.
        # We could have multiple different types of env on each worker, but crucially only one of each type. So we initilize a dict and hold each
        # env keyed by its envtype E. Then when collecting a trajectory explicitly using this via environments[E] which will be a globally defined dict 
        # on the worker.
        if environments[E].terminal
            state = reset!(environments[E])
        else
            state = environments[E].state
        end
        while environments[E].terminal == false && local_segment_count < T # bool flag to denote whether episode has finished
            # Generalises below...
            action, state_value, probs = get_action(algtype, agent, environments[E]; random_policy=random_policy) # get the action, the value and the probability
            push!(trajectory_probabilities, probs)

            new_state, reward, terminal = step!(environments[E], action) # take a step of the hopper environments[E]s
            push!(current_ep_states, state) # We need the state for V(s_t) during learning
            push!(current_ep_actions, action) # We obviously need the action for π(a_t | s_t)

            _, next_state_value, _ = get_action(algtype, agent, environments[E]) # get the value of the next state to calculate bellman error & advantage
            target = reward + (1 - Int(terminal)) * γ * next_state_value[1] # the target looks good
            push!(local_targets, target)
            push!(local_errors, target - state_value[1])
            state = new_state # uodate state
            local_segment_count += 1 # increment trajectory segment count
            environments[E].terminal = terminal # update env terminal flag
            if environments[E].terminal || local_segment_count == T # if terminal
                len_current_seg = length(current_ep_states) # obtain number of experiences collected in current subsegment
                advantages = zeros(Float32, len_current_seg) # initial advantages to be zero with length of subsegment
                for t in 1:len_current_seg
                    advantages[t] = sum(local_errors[t:end] .* advantage_coefficients[1:(len_current_seg-t+1)])
                end
                push!(trajectory_states, current_ep_states...)
                push!(trajectory_actions, current_ep_actions...)
                push!(all_advantages, advantages...)
                push!(all_errors, local_errors...)
                push!(all_targets, local_targets...)
                current_ep_states = Vector{AbstractObservation}()
                current_ep_actions = Vector{action_type}()
                local_targets = Vector{Float32}()
                local_errors = Vector{Float32}()
            end
        end
    end
    return trajectory_states, trajectory_actions, trajectory_probabilities, all_advantages, all_targets, all_errors
end

function get_trajectories!(alg::PPO{G}, env::E; random_policy::Bool=false) where {E <: AbstractEnv, G <: AbstractAction}
    info = Dict{Symbol, Any}(:T => alg.T, 
            :γ => alg.γ, :advantage_coefficients => alg.advantage_coefficients, 
            :action_type => G, :algtype => typeof(alg))

    agents = alg.worker_agents # get agents from algorithm

    futures = [@spawnat p collect_trajectory_segment!(typeof(env), agents[i], info, random_policy=random_policy) for (i, p) in enumerate(workers())]
    results = fetch.(futures)
    final_results = vcat(results...)  
    all_states, all_actions, all_probabilities, all_advantages, all_targets, all_errors = unzip(final_results)
    return all_states, all_actions, all_probabilities, all_advantages, all_targets, all_errors
end

function validation_episode!(alg::PPO{G}, env::E, agent::A; render::Bool=false) where {E<:AbstractEnv, A <: AbstractAgent, G <: AbstractAction}
    # initialise vectors to store the history of states actions and rewards for the entire episode
    state = reset!(env)
    term = false
    episode_reward = Vector{Float64}()
    step=0
    while env.terminal == false # bool flag to denote whether routing has finished
        # calculate the mode outputs based on the current graph
        action, _, _ = get_action(typeof(alg), agent, state; det=true)
        if render==true
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

function validation_episode!(env::E, alg::PPO{G}; render::Bool=false) where {E<:AbstractEnv, G <: AbstractAction}
    # initialise vectors to store the history of states actions and rewards for the entire episode
    return validation_episode!(alg, env, alg.central_agent; render=render)
end
     

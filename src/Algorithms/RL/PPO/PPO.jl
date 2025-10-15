# using G as the abtract typeholder for action from the synonym 'gesture' since A is used for agent
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
    function PPO(::Type{G}, N::Int, T::Int, K::Int, agent::A, batch_size::Int, γ::Float64, λ::Float64, 
        ϵ::Float64, c1::Float64, c2::Float64, sync_frequency::Int) where {G <: AbstractAction, A <: AbstractAgent}
        wrkrs = [deepcopy(agent) for _ in 1:N]
        exponents = collect(0:T)
        advantage_coefficients = (λ * γ) .^ exponents
        return new{G}(N, T, K, agent, wrkrs, batch_size, γ, λ, ϵ, c1, c2, sync_frequency, advantage_coefficients,
        Vector{Float32}(), Vector{Float32}(), Vector{Float32}(), Vector{Float32}(), Vector{Float32}(), Vector{Float32}())
    end
    function PPO(::Type{G}, N::Int, T::Int, K::Int,  agent::A) where {G <: AbstractAction, A <: AbstractAgent}
        wrkrs = [deepcopy(agent) for _ in 1:N]
        exponents = collect(0:T)
        advantage_coefficients = (0.95 * 0.99) .^ exponents
        return new{G}(N, T, K, agent, wrkrs, 64, 0.99, 0.95, 0.2, 0.5, 0.001, 2, advantage_coefficients,
        Vector{Float32}(), Vector{Float32}(), Vector{Float32}(), Vector{Float32}(), Vector{Float32}(), Vector{Float32}())
    end

end


function train!(alg::PPO{G}, states::Vector{S}, actions::Vector{G}, probabilities::Union{Vector{Vector{Float32}}, Vector{Float32}},
    advantages::Vector{Float32}, bellman_targets::Vector{Float32}, bellman_errors::Vector{Float32}) where {G <: AbstractAction, S <: AbstractObservation}
    # initialise vectors to store batched data
    # available_batch_indices = collect(1:length(states))
    available_batch_indices = 1:length(states)
    for _ in 1:alg.K
        # if length(available_batch_indices) == 0
        #     available_batch_indices = collect(1:length(transitions))
        # end

        if length(available_batch_indices) > alg.batch_size
            batch_indices = sample(available_batch_indices, alg.batch_size; replace=false)
        else
            batch_indices = collect(available_batch_indices)
        end

        batch_states = states[batch_indices]
        batch_actions = actions[batch_indices]
        batch_probabilities = probabilities[batch_indices]
        # Normalize the advantages over the current batch?
        batch_advantages = (advantages[batch_indices] .- mean(advantages[batch_indices])) ./ std(advantages[batch_indices])
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
        vals = minimum.(batch_advantages .* clamped_r)
        L_CLIP = 1 * mean(vals)
        entropy = -1 * mean(sum(exp.(action_probs) .* log2.(exp.(action_probs)), dims=1))
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
        vals = minimum.(batch_advantages .* clamped_r)
        L_CLIP = 1 * mean(vals)
        entropy = -1 * mean(sum(exp.(action_probs) .* log2.(exp.(action_probs)), dims=1))
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
        vals = minimum.(batch_advantages .* clamped_r)
        L_CLIP = 1 * mean(vals)
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
        vals = minimum.(batch_advantages .* clamped_r)
        L_CLIP = 1 * mean(vals)
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
        
        vals = minimum.(batch_advantages .* clamped_r)
        L_CLIP = mean(vals)
        
        entropy = mean(0.5 .* log.(2 * π .* σ .^ 2 .+ 0.5)) # Closed-form entropy for Gaussian
        
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
        vals = minimum.(batch_advantages .* clamped_r)
        L_CLIP = 1 * mean(vals)
        L_value_loss = Flux.Losses.mse(batch_bellman_targets, dropdims(state_values, dims=1))
        entropy = mean(0.5 .* log.(2 * π .* σ .^ 2 .+ 0.5)) # there is a closed form for the entropy of a gaussian
        return alg.c1 * L_value_loss - L_CLIP  - alg.c2 *entropy
    end
    # get combined and critic gradients
    ∇_combined = gradient(combined_m -> loss_calculation(combined_m), combined_m) 
    # update combined nd critic models 
    Flux.update!(agent.combined_model._optimizer_state, agent.combined_model.model, ∇_combined[1])
end

function collect_trajectory_segment!(::Type{E}, agent::A, info::Dict{Symbol, Any}) where {E <: AbstractEnv, A <: AbstractAgent}
    T::Int = info[:T]
    γ::Float32 = info[:γ]
    advantage_coefficients::Vector{Float32} = info[:advantage_coefficients]
    action_type = info[:action_type]
    algtype = info[:algtype]

    # initialize vectors to store all the transitions encountered
    local_segment_count = 0
    # total_trajectory = Vector{Experience}()
    trajectory_states = Vector{AbstractObservation}()
    trajectory_actions = Vector{action_type}()
    all_targets = Vector{Float32}()
    all_errors = Vector{Float32}()
    all_advantages = Vector{Float32}()
    if action_type <: DiscreteAct
        trajectory_probabilities = Vector{Vector{Float32}}()
    else
        trajectory_probabilities = Vector{Float32}()
    end

    # When starting a new segment collection we need to decide whether to continue with a current episode or start a-new

    while local_segment_count < T # while we still haven't fully collected a segment
            # initialise inner vectors to collect each subsegment. Important for if T > ep_len
        # current_ep_trajectory = Vector{Experience}() 
        current_ep_states = Vector{AbstractObservation}()
        current_ep_actions = Vector{action_type}()
        local_targets = Vector{Float32}()
        local_errors = Vector{Float32}()
        if environments[E].terminal
            state = reset!(environments[E])
        else
            state = environments[E].state
        end
        while environments[E].terminal == false && local_segment_count < T # bool flag to denote whether episode has finished
            # Generalises below...
            action, state_value, probs = get_action(algtype, agent, environments[E]) # get the action, the value and the probability
            push!(trajectory_probabilities, probs)
            new_state, reward, terminal = step!(environments[E], action) # take a step of the hopper environments[E]s
            # experience = Experience(state, action, new_state, reward, terminal)
            # push!(current_ep_trajectory, experience)
            push!(current_ep_states, state)
            push!(current_ep_actions, action)
            _, next_state_value, _ = get_action(algtype, agent, environments[E]) # get the value of the next state to calculate bellman error & advantage
            target = reward .+ (1 .- Int.(terminal)) .* γ .* next_state_value
            push!(local_targets, target[1])
            push!(local_errors, target[1] - state_value[1])
            state = new_state
            local_segment_count += 1
            environments[E].terminal = terminal
            if environments[E].terminal || local_segment_count == T
                len_current_seg = length(current_ep_states)
                advantages = zeros(Float32, len_current_seg)
                for t in 1:len_current_seg
                    advantages[t] = sum(local_errors[t:end] .* advantage_coefficients[1:(len_current_seg-t+1)])
                end
                push!(trajectory_states, current_ep_states...)
                push!(trajectory_actions, current_ep_actions...)
                push!(all_advantages, advantages...)
                push!(all_errors, local_errors...)
                push!(all_targets, local_targets...)
                # current_ep_trajectory = Vector{Experience}()
                current_ep_states = Vector{AbstractObservation}()
                current_ep_actions = Vector{action_type}()
                local_targets = Vector{Float32}()
                local_errors = Vector{Float32}()
            end
        end
    end
    return trajectory_states, trajectory_actions, trajectory_probabilities, all_advantages, all_targets, all_errors
end

function get_trajectories!(alg::PPO{G}, env::E) where {E <: AbstractEnv, G <: AbstractAction}
    info = Dict{Symbol, Any}(:T => alg.T, :γ => alg.γ, :advantage_coefficients => alg.advantage_coefficients, :action_type => G, :algtype => typeof(alg))
    agents = alg.worker_agents

    futures = [@spawnat p collect_trajectory_segment!(typeof(env), agents[i], info) for (i, p) in enumerate(workers())]
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
     

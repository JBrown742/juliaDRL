# using G as the abtract typeholder for action from the synonym 'gesture' since A is used for agent

"""
    RunningStat

Keeps a running estimate of mean and variance for observation normalization.
"""
mutable struct RunningStat
    mean::Vector{Float32}
    var::Vector{Float32}
    count::Float32
end

function RunningStat()
    return RunningStat(Float32[], Float32[], 1f-4) # small count to avoid div by zero
end

function update!(s::RunningStat, x::AbstractVector)
    if isempty(s.mean)
        s.mean = copy(x)
        s.var = ones(Float32, length(x))
        s.count = 1.0f0
        return
    end
    s.count += 1.0f0
    old_mean = copy(s.mean)
    s.mean .+= (x .- s.mean) ./ s.count
    s.var .+= (x .- old_mean) .* (x .- s.mean)
end

function normalize(s::RunningStat, x::AbstractVector)
    if isempty(s.mean) return x end
    std = sqrt.(s.var ./ s.count) .+ 1f-8
    return (x .- s.mean) ./ std
end

function normalize(s::RunningStat, x::AbstractMatrix)
    if isempty(s.mean) return x end
    std = sqrt.(s.var ./ s.count) .+ 1f-8
    return (x .- s.mean) ./ std
end

"""
    calculate_advantage_coefficients(T::Int, γ::Float32, λ::Float32)

Calculates the coefficients for Generalized Advantage Estimation (GAE).
"""
function calculate_advantage_coefficients(T::Int, γ::Float32, λ::Float32)
    return Float32.((λ * γ) .^ (0:T))
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
    obs_normalizer::RunningStat # Running statistics for observation normalization
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
        
        new{G}(N, T, K, agent, wrkrs, batch_size, γ_f32, λ_f32, ϵ_f32, c1_f32, c2_f32, sync_frequency, advantage_coefficients, RunningStat(),
              Vector{Float32}(), Vector{Float32}(), Vector{Float32}(), Vector{Float32}(), Vector{Float32}(), Vector{Float32}())
    end
end


# STABILITY FIX: Global Gradient Norm Clipping
# Preserves gradient direction while controlling magnitude
function robust_update!(model_obj, grads; max_norm=0.5f0)
    has_nan(x::AbstractArray) = any(isnan, x)
    has_nan(x::Union{NamedTuple, Tuple}) = any(has_nan, x)
    has_nan(x) = false

    if has_nan(grads)
        return
    end

    # Calculate global norm across all parameter arrays
    gnorm = 0.0f0
    fmap(grads) do x
        if x isa AbstractArray
            gnorm += sum(abs2, x)
        end
    end
    gnorm = sqrt(gnorm)

    # Scale if norm exceeds threshold
    safe_grads = if gnorm > max_norm
        scale = max_norm / (gnorm + 1f-6)
        fmap(x -> x isa AbstractArray ? x .* scale : x, grads)
    else
        grads
    end
    
    Flux.update!(model_obj._optimizer_state, model_obj.model, safe_grads)
end

function train!(alg::PPO{G}, states::Vector, actions::Vector, probabilities::Vector,
    advantages::Vector, bellman_targets::Vector, bellman_errors::Vector) where {G <: AbstractAction}
    
    # 1. HPC Optimization: Stack and normalize ONCE per update (not once per epoch)
    # This reduces allocations from O(K * Batch) to O(1)
    state_tensor = stack(states)
    action_tensor = stack(actions)
    prob_tensor = stack(probabilities)
    
    # Normalize advantages for the whole batch
    mean_adv = mean(advantages)
    std_adv = std(advantages) + 1f-8
    norm_advantages = (advantages .- mean_adv) ./ std_adv
    
    num_samples = length(states)

    for _ in 1:alg.K
        # Shuffle indices for each epoch to improve generalization
        idxs = Random.shuffle(1:num_samples)
        chunks = Iterators.partition(idxs, alg.batch_size)
        
        for batch_indices in chunks
            # High-speed slicing (no copies where possible)
            batch_states = selectdim(state_tensor, ndims(state_tensor), batch_indices)
            batch_actions = selectdim(action_tensor, ndims(action_tensor), batch_indices)
            batch_probs = selectdim(prob_tensor, ndims(prob_tensor), batch_indices)
            batch_advantages = norm_advantages[batch_indices]
            batch_targets = bellman_targets[batch_indices]
            
            gradient_calculation_and_update!(alg, alg.central_agent, batch_states, batch_actions, batch_advantages, batch_probs, batch_targets)
        end
    end
end

# ------------------- Dispatches for the gradient calculation depending on whether ------------------- #
# --------------------------------- parameters are shared and ---------------------------------------- #
# ----------------------- whether we are using discrete or continuous actions ------------------------ #

## OK
function gradient_calculation_and_update!(alg::PPO{DiscreteAct}, agent::StandardActorCritic, states::AbstractArray, 
    actions::AbstractArray, batch_advantages::AbstractArray, batch_probabilities::AbstractArray, 
    batch_bellman_targets::AbstractArray)
    # The below masking functionality is explicitly for environments where each state may have a different available subset of
    # the total action set
    # Get the action mask matrix for all actions taken in this batch
    actual_action_mask = indicatormat(actions, first(agent.actor_model.model.layers[end].bias |> size))
    # HPC Fix: Handle batch_probabilities as a Matrix/tensor if it was stacked
    if batch_probabilities isa AbstractMatrix
        probability_mask = hcat(infer_mask.(eachcol(batch_probabilities))...)
        batch_prob_mat = batch_probabilities
    elseif batch_probabilities isa AbstractVector && !isempty(batch_probabilities) && batch_probabilities[1] isa AbstractVector
        probability_mask = hcat(infer_mask.(batch_probabilities)...) # vector of action masks
        batch_prob_mat = hcat(batch_probabilities...)
    else
        # Fallback for unexpected shapes
        batch_prob_mat = batch_probabilities
        probability_mask = zeros(Float32, size(batch_prob_mat))
    end

    act_m = agent.actor_model.model
    crit_m = agent.critic_model.model
    # define explicity policy loss function 
    function policy_loss_calculation(m)
        # What does the current model say the probability for all of our actions should be?
        action_logits = m(states) # get the output logits for each action on each of the states
        # We need to use the additive probability mask to calculate these new probs accurately
        action_probs = masked_probabilities(probability_mask, action_logits) # use probability_mask to mask and softmax logits
        # We then use the multiplicative actual action mask to address only the actions actually taken! 
        new_probs = dropdims(sum(action_probs .* actual_action_mask, dims=1), dims=1)
        old_probs = dropdims(sum(batch_prob_mat .* actual_action_mask, dims=1), dims=1)
        r = new_probs ./ old_probs
        clamped_r = clamp.(r, 1 - alg.ϵ, 1 + alg.ϵ)
        L_CLIP = mean(min.(r .* batch_advantages, clamped_r .* batch_advantages))
        entropy = -1 * mean(sum(action_probs .* log.(action_probs .+ 1f-10), dims=1))
        return -1 * L_CLIP - alg.c2 * entropy
    end
    # define value loss function
    function value_loss(m)
        state_values = dropdims(m(states), dims=1)
        return  Flux.Losses.mse(batch_bellman_targets, state_values)
    end
    # get actor and critic gradients
    ∇_actor = gradient(act_m -> policy_loss_calculation(act_m), act_m) 
    ∇_critic = gradient(crit_m -> value_loss(crit_m), crit_m)
    
    robust_update!(agent.actor_model, ∇_actor[1])
    robust_update!(agent.critic_model, ∇_critic[1])
end

# OK
function gradient_calculation_and_update!(alg::PPO{DiscreteAct}, agent::CombinedActorCritic, states::AbstractArray, 
    actions::AbstractArray, batch_advantages::AbstractArray, batch_probabilities::AbstractArray, 
    batch_bellman_targets::AbstractArray)
    actual_action_mask = indicatormat(actions, first(agent.combined_model.model.layers[end].paths[1].bias |> size))
    # HPC Fix: Handle batch_probabilities as a Matrix/tensor if it was stacked
    if batch_probabilities isa AbstractMatrix
        probability_mask = hcat(infer_mask.(eachcol(batch_probabilities))...)
        batch_prob_mat = batch_probabilities
    elseif batch_probabilities isa AbstractVector && !isempty(batch_probabilities) && batch_probabilities[1] isa AbstractVector
        probability_mask = hcat(infer_mask.(batch_probabilities)...) # vector of action masks
        batch_prob_mat = hcat(batch_probabilities...)
    else
        # Fallback for unexpected shapes
        batch_prob_mat = batch_probabilities
        probability_mask = zeros(Float32, size(batch_prob_mat))
    end

    combined_m = agent.combined_model.model
    # define explicity policy loss function 
    function loss_calculation(m)
        action_logits, state_values = m(states)
        action_probs = masked_probabilities(probability_mask, action_logits)
        L_q_learning = Flux.Losses.mse(batch_bellman_targets, dropdims(state_values, dims=1))
        new_probs = dropdims(sum(action_probs .* actual_action_mask, dims=1), dims=1)
        old_probs = dropdims(sum(batch_prob_mat .* actual_action_mask, dims=1), dims=1)
        r = new_probs ./ old_probs
        clamped_r = clamp.(r, 1 - alg.ϵ, 1 + alg.ϵ)
        L_CLIP = mean(min.(r .* batch_advantages, clamped_r .* batch_advantages))
        entropy = -1 * mean(sum(action_probs .* log.(action_probs .+ 1f-10), dims=1))
        return alg.c1 * L_q_learning - L_CLIP - alg.c2 * entropy
    end
    # get combined and critic gradients
    ∇_combined = gradient(combined_m -> loss_calculation(combined_m), combined_m) 
    
    robust_update!(agent.combined_model, ∇_combined[1])
end

# OK
function gradient_calculation_and_update!(alg::PPO{ContinuousAct}, agent::StandardActorCritic, states::AbstractArray, 
    actions::AbstractArray, batch_advantages::AbstractArray, batch_probabilities::AbstractArray, 
    batch_bellman_targets::AbstractArray)
    # get_actor and critic models
    act_m = agent.actor_model.model
    crit_m = agent.critic_model.model
    # define explicity policy loss function 
    function policy_loss_calculation(m)
        α_raw, β_raw = dropdims.(m(states), dims=1)
        α = Flux.softplus.(α_raw) .+ 1.0f0
        β = Flux.softplus.(β_raw) .+ 1.0f0
        
        # Convert actions back to [0, 1] samples
        u = (actions .+ 1.0f0) ./ 2.0f0
        
        # HPC Optimization: Use shared beta_logpdf utility
        new_log_probs = beta_logpdf(α, β, u)
        old_log_probs = batch_probabilities
        r = exp.(new_log_probs .- old_log_probs)
        clamped_r = clamp.(r, 1 - alg.ϵ, 1 + alg.ϵ)
        L_CLIP = mean(min.(r .* batch_advantages, clamped_r .* batch_advantages))     
        
        entropy = mean(beta_entropy.(α, β))
        return -1 * L_CLIP - alg.c2 * entropy
    end
    # define value loss function
    function value_loss(m)
        state_values = dropdims(m(states), dims=1)
        return Flux.Losses.mse(batch_bellman_targets, state_values)
    end
    # get actor and critic gradients
    ∇_actor = gradient(act_m -> policy_loss_calculation(act_m), act_m) 
    ∇_critic = gradient(crit_m -> value_loss(crit_m), crit_m)
    
    robust_update!(agent.actor_model, ∇_actor[1])
    robust_update!(agent.critic_model, ∇_critic[1])
end
# OK
function gradient_calculation_and_update!(alg::PPO{ContinuousAct}, agent::CombinedActorCritic, states::AbstractArray, 
    actions::AbstractArray, batch_advantages::AbstractArray, batch_probabilities::AbstractArray, 
    batch_bellman_targets::AbstractArray)

    combined_m = agent.combined_model.model
    # define explicity policy loss function 
    function loss_calculation(m)
        α_raw, β_raw, state_values = dropdims.(m(states), dims=1)
        α = Flux.softplus.(α_raw) .+ 1.0f0
        β = Flux.softplus.(β_raw) .+ 1.0f0
        
        u = (actions .+ 1.0f0) ./ 2.0f0
        
        # HPC Optimization: Use shared beta_logpdf utility
        new_log_probs = beta_logpdf(α, β, u)
        old_log_probs = batch_probabilities
        r = exp.(new_log_probs .- old_log_probs)
        clamped_r = clamp.(r, 1 - alg.ϵ, 1 + alg.ϵ)
        L_CLIP = mean(min.(r .* batch_advantages, clamped_r .* batch_advantages))
        L_value_loss = Flux.Losses.mse(batch_bellman_targets, state_values)
        entropy = mean(beta_entropy.(α, β))
        return alg.c1 * L_value_loss - L_CLIP - alg.c2 * entropy
    end
    # get combined and critic gradients
    ∇_combined = gradient(combined_m -> loss_calculation(combined_m), combined_m) 
    
    robust_update!(agent.combined_model, ∇_combined[1])
end
# OK
function gradient_calculation_and_update!(alg::PPO{MultiContinuousAct}, agent::StandardActorCritic, states::AbstractArray, 
    actions::AbstractArray, batch_advantages::AbstractArray, batch_probabilities::AbstractArray, 
    batch_bellman_targets::AbstractArray)
    # get actor and critic model
    act_m = agent.actor_model.model
    crit_m = agent.critic_model.model
    # define explicity policy loss function 
    function policy_loss_calculation(m)
        α_raw, β_raw = m(states)
        α = Flux.softplus.(α_raw) .+ 1.0f0
        β = Flux.softplus.(β_raw) .+ 1.0f0

        # Beta distribution is on [0, 1]. Actions are on [-1, 1].
        u = (actions .+ 1.0f0) ./ 2.0f0

        # HPC Optimization: Use shared beta_logpdf utility and sum over heads
        new_log_probs = dropdims(sum(beta_logpdf(α, β, u), dims=1), dims=1)

        old_log_probs = batch_probabilities

        r = exp.(new_log_probs .- old_log_probs)
        clamped_r = clamp.(r, 1 - alg.ϵ, 1 + alg.ϵ)

        # PPO's clipped objective
        L_CLIP = mean(min.(r .* batch_advantages, clamped_r .* batch_advantages))
        
        # Entropy bonus
        entropy = mean(sum(beta_entropy.(α, β), dims=1))
        
        return -L_CLIP - alg.c2 * entropy
    end
    # define value loss function
    function value_loss(m)
        state_values = dropdims(m(states), dims=1)
        return Flux.Losses.mse(batch_bellman_targets, state_values)
    end
    
    # get actor and critic gradients
    ∇_actor = gradient(act_m -> policy_loss_calculation(act_m), act_m) 
    ∇_critic = gradient(crit_m -> value_loss(crit_m), crit_m)
    
    robust_update!(agent.actor_model, ∇_actor[1])
    robust_update!(agent.critic_model, ∇_critic[1])
end

# OK
function gradient_calculation_and_update!(alg::PPO{MultiContinuousAct}, agent::CombinedActorCritic, states::AbstractArray, 
    actions::AbstractArray, batch_advantages::AbstractArray, batch_probabilities::AbstractArray, 
    batch_bellman_targets::AbstractArray)
    combined_m = agent.combined_model.model
    # define explicity policy loss function 
    function loss_calculation(m)
        α_raw, β_raw, state_values = m(states)
        α = Flux.softplus.(α_raw) .+ 1.0f0
        β = Flux.softplus.(β_raw) .+ 1.0f0

        u = (actions .+ 1.0f0) ./ 2.0f0

        # HPC Optimization: Use shared beta_logpdf utility and sum over heads
        new_log_probs = dropdims(sum(beta_logpdf(α, β, u), dims=1), dims=1)
        old_log_probs = batch_probabilities
        r = exp.(new_log_probs .- old_log_probs)
        clamped_r = clamp.(r, 1 - alg.ϵ, 1 + alg.ϵ)

        # PPO's clipped objective
        L_CLIP = mean(min.(r .* batch_advantages, clamped_r .* batch_advantages))
        
        L_value_loss = Flux.Losses.mse(batch_bellman_targets, dropdims(state_values, dims=1))
        
        # Entropy bonus
        entropy = mean(sum(beta_entropy.(α, β), dims=1)) 
        return alg.c1 * L_value_loss - L_CLIP  - alg.c2 *entropy
    end
    # get combined and critic gradients
    ∇_combined = gradient(combined_m -> loss_calculation(combined_m), combined_m) 
    
    robust_update!(agent.combined_model, ∇_combined[1])
end

# OK
function gradient_calculation_and_update!(alg::PPO{MultiDiscreteAct}, agent::StandardActorCritic, states::AbstractArray, 
    actions::AbstractArray, batch_advantages::AbstractArray, batch_probabilities::AbstractArray, 
    batch_bellman_targets::AbstractArray)
    
    act_m = agent.actor_model.model
    crit_m = agent.critic_model.model

    # MultiDiscrete Configuration (Hardcoded for 2 heads matching Utilities.jl)
    num_heads = 2
    head_size = Int(size(act_m.layers[end].bias, 1) / num_heads)

    # Pre-calculate old joint log probabilities and action masks
    old_probs_reshaped = reshape(batch_probabilities, head_size, num_heads, :)
    old_joint_log_probs = zeros(Float32, size(states, ndims(states)))
    action_masks = []
    for h in 1:num_heads
        mask = indicatormat(actions[h, :], head_size)
        push!(action_masks, mask)
        old_joint_log_probs .+= log.(dropdims(sum(old_probs_reshaped[:, h, :] .* mask, dims=1), dims=1) .+ 1f-10)
    end

    function policy_loss_calculation(m)
        logits = m(states) # (head_size * num_heads, batch)
        
        # Reshape to (head_size, num_heads, batch) for multi-head processing
        logits_reshaped = reshape(logits, head_size, num_heads, :)
        probs = softmax(logits_reshaped; dims=1)
        
        # Calculate new joint log-probabilities using pre-calculated masks
        new_joint_log_probs = reduce(+, [
            log.(dropdims(sum(probs[:, h, :] .* action_masks[h], dims=1), dims=1) .+ 1f-10)
            for h in 1:num_heads
        ])

        r = exp.(new_joint_log_probs .- old_joint_log_probs)
        clamped_r = clamp.(r, 1 - alg.ϵ, 1 + alg.ϵ)
        L_CLIP = mean(min.(r .* batch_advantages, clamped_r .* batch_advantages))
        
        # Entropy bonus: sum of entropies of all heads
        entropy = -1 * mean(sum(probs .* log.(probs .+ 1f-10), dims=(1, 2)))
        
        return -1 * L_CLIP - alg.c2 * entropy
    end

    function value_loss(m)
        state_values = dropdims(m(states), dims=1)
        return Flux.Losses.mse(batch_bellman_targets, state_values)
    end

    ∇_actor = gradient(act_m -> policy_loss_calculation(act_m), act_m) 
    ∇_critic = gradient(crit_m -> value_loss(crit_m), crit_m)
    
    robust_update!(agent.actor_model, ∇_actor[1])
    robust_update!(agent.critic_model, ∇_critic[1])
end

function gradient_calculation_and_update!(alg::PPO{MultiDiscreteAct}, agent::CombinedActorCritic, states::AbstractArray, 
    actions::AbstractArray, batch_advantages::AbstractArray, batch_probabilities::AbstractArray, 
    batch_bellman_targets::AbstractArray)
    
    combined_m = agent.combined_model.model

    # MultiDiscrete Configuration (Hardcoded for 2 heads matching Utilities.jl)
    num_heads = 2
    head_size = Int(size(combined_m.layers[end].paths[1].bias, 1) / num_heads)

    # Pre-calculate old joint log probabilities and action masks
    old_probs_reshaped = reshape(batch_probabilities, head_size, num_heads, :)
    old_joint_log_probs = zeros(Float32, size(states, ndims(states)))
    action_masks = []
    for h in 1:num_heads
        mask = indicatormat(actions[h, :], head_size)
        push!(action_masks, mask)
        old_joint_log_probs .+= log.(dropdims(sum(old_probs_reshaped[:, h, :] .* mask, dims=1), dims=1) .+ 1f-10)
    end

    function loss_calculation(m)
        logits, state_values = m(states)
        
        # Reshape to (head_size, num_heads, batch)
        logits_reshaped = reshape(logits, head_size, num_heads, :)
        probs = softmax(logits_reshaped; dims=1)
        
        # Calculate new joint log-probabilities using pre-calculated masks
        new_joint_log_probs = reduce(+, [
            log.(dropdims(sum(probs[:, h, :] .* action_masks[h], dims=1), dims=1) .+ 1f-10)
            for h in 1:num_heads
        ])

        r = exp.(new_joint_log_probs .- old_joint_log_probs)
        clamped_r = clamp.(r, 1 - alg.ϵ, 1 + alg.ϵ)
        L_CLIP = mean(min.(r .* batch_advantages, clamped_r .* batch_advantages))
        
        L_value_loss = Flux.Losses.mse(batch_bellman_targets, dropdims(state_values, dims=1))
        
        # Entropy bonus: sum of entropies of all heads
        entropy = -1 * mean(sum(probs .* log.(probs .+ 1f-10), dims=(1, 2)))
        
        return alg.c1 * L_value_loss - L_CLIP - alg.c2 * entropy
    end

    ∇_combined = gradient(combined_m -> loss_calculation(combined_m), combined_m) 
    
    robust_update!(agent.combined_model, ∇_combined[1])
end

function collect_trajectory_segment!(env::E, agent::A, info::Dict{Symbol, Any}; random_policy::Bool=false) where {E <: AbstractEnv, A <: AbstractAgent}
    # Using a dictionary here is a hacky means of using distributed processing and passing alg level parameters to each worker_agents
    # without needing to explicity transfer the alg struct which is much more light weight
    T::Int = info[:T] # Trajectory length
    γ::Float32 = info[:γ] # Discount factor
    advantage_coefficients::Vector{Float32} = info[:advantage_coefficients] # Advantage coefficients for GAE
    action_type = info[:action_type] # Action type
    algtype = info[:algtype] # Algorithm type
    obs_normalizer::RunningStat = info[:obs_normalizer] # For consistent normalization

    # initialize vectors to store all the transitions encountered
    local_segment_count = 0 # This keeps an internal count on the worker as to which step we are at
    
    # HPC Optimization: Pre-allocate vectors of known length T
    trajectory_states = Vector{Any}(undef, T) 
    trajectory_actions = Vector{action_type}(undef, T)
    all_targets = Vector{Float32}(undef, T)
    all_errors = Vector{Float32}(undef, T)
    all_advantages = Vector{Float32}(undef, T)
    all_terminals = Vector{Bool}(undef, T) # To reset GAE at boundaries
    
    # Initialize with dummy values, will be type-refined on first step
    trajectory_probabilities = Vector{Any}(undef, T)

    while local_segment_count < T # while we still haven't fully collected a segment
        if env.terminal
            state = reset!(env)
        else
            state = env.state
        end
        while env.terminal == false && local_segment_count < T 

            # Normalize observation before passing to actor
            norm_state = normalize(obs_normalizer, state)
            
            action, state_value, probs = get_action(algtype, agent, norm_state; random_policy=random_policy)
            
            new_state, reward, terminal = step!(env, action)

            # Normalize new state for next value estimate
            norm_new_state = normalize(obs_normalizer, new_state)
            _, next_state_value, _ = get_action(algtype, agent, norm_new_state)
            
            # Proper bootstrapping: Only stop bootstrapping if it's a real terminal (not truncation)
            is_truncated = hasfield(typeof(env), :truncated) ? env.truncated : false
            is_real_terminal = terminal && !is_truncated
            
            target = reward .+ (1 .- Int.(is_real_terminal)) .* γ .* next_state_value
            
            # Write directly to pre-allocated indices
            idx = local_segment_count + 1
            trajectory_states[idx] = state # store raw state, will be normalized in train!
            trajectory_actions[idx] = action
            trajectory_probabilities[idx] = probs
            all_targets[idx] = target[1]
            all_errors[idx] = target[1] - state_value[1]
            all_terminals[idx] = terminal
            
            state = new_state
            local_segment_count += 1
            env.terminal = terminal
        end
    end
    
    # Backward pass for GAE advantage estimation (O(T))
    # Corrected to respect episode boundaries
    gae = 0.0f0
    γλ = γ * info[:λ]
    for t in T:-1:1
        if all_terminals[t]
            gae = 0.0f0
        end
        gae = all_errors[t] + γλ * gae
        all_advantages[t] = gae
    end
    
    return trajectory_states, trajectory_actions, trajectory_probabilities, all_advantages, all_targets, all_errors
end

function get_trajectories!(alg::PPO{G}, env::E; random_policy::Bool=false) where {E <: AbstractEnv, G <: AbstractAction}
    info = Dict{Symbol, Any}(:T => alg.T, 
            :γ => alg.γ, :λ => alg.λ, :advantage_coefficients => alg.advantage_coefficients, 
            :action_type => G, :algtype => typeof(alg), :obs_normalizer => alg.obs_normalizer)

    agents = alg.worker_agents # get agents from algorithm

    # Distributed fix: Create the environment LOCALLY on each worker.
    # We use a closure that captures the necessary data to rebuild the environment.
    # This avoids sending the 'env' object (which contains un-picklable PyObjects) over the wire.
    
    futures = []
    for (i, p) in enumerate(workers())
        # DEFINITIVE FIX: Reconstruct the environment on the worker.
        # We pass 'env' as a variable, but 'clone(env)' is called LOCALLY on the worker.
        f = @spawnat p let worker_env = clone(env)
            collect_trajectory_segment!(worker_env, agents[i], info, random_policy=random_policy)
        end
        push!(futures, f)
    end
    
    results = fetch.(futures)
    all_states = vcat([r[1] for r in results]...)
    all_actions = vcat([r[2] for r in results]...)
    all_probabilities = vcat([r[3] for r in results]...)
    all_advantages = vcat([r[4] for r in results]...)
    all_targets = vcat([r[5] for r in results]...)
    all_errors = vcat([r[6] for r in results]...)
    return all_states, all_actions, all_probabilities, all_advantages, all_targets, all_errors
end

function validation_episode!(alg::PPO{G}, env::E, agent::A; render::Bool=false) where {E<:AbstractEnv, A <: AbstractAgent, G <: AbstractAction}
    # initialise vectors to store the history of states actions and rewards for the entire episode
    state = reset!(env)
    term = false
    episode_reward = Vector{Float64}()
    step=0
    while env.terminal == false # bool flag to denote whether routing has finished
        # Normalize observation before passing to actor
        norm_state = normalize(alg.obs_normalizer, state)
        
        # calculate the mode outputs based on the current graph
        action, _, _ = get_action(typeof(alg), agent, norm_state; det=true)
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
     

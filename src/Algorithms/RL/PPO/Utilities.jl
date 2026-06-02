
# ------------ get_actions functions ------- #
# ------------------------  Unambiguous entry-point dispatches ----------------------- #
# Discrete
function get_action(::Type{PPO{DiscreteAct}}, agent::A, obs_or_env; det=false, random_policy=false) where {A <: AbstractAgent}
    if obs_or_env isa AbstractEnv
        mask = hasfield(typeof(obs_or_env), :action_mask) ? obs_or_env.action_mask : nothing
        return _get_action_discrete(agent, obs_or_env.state; det=det, random_policy=random_policy, mask=mask)
    else
        return _get_action_discrete(agent, obs_or_env; det=det, random_policy=random_policy, mask=nothing)
    end
end
# Continuous
function get_action(::Type{PPO{ContinuousAct}}, agent::A, obs_or_env; det=false, random_policy=false) where {A <: AbstractAgent}
    obs = obs_or_env isa AbstractEnv ? obs_or_env.state : obs_or_env
    return _get_action_continuous(agent, obs; det=det, random_policy=random_policy)
end
# MultiDiscrete
function get_action(::Type{PPO{MultiDiscreteAct}}, agent::A, obs_or_env; det=false, random_policy=false) where {A <: AbstractAgent}
    if obs_or_env isa AbstractEnv
        mask = hasfield(typeof(obs_or_env), :action_mask) ? obs_or_env.action_mask : nothing
        return _get_action_multidiscrete(agent, obs_or_env.state; det=det, random_policy=random_policy, mask=mask)
    else
        return _get_action_multidiscrete(agent, obs_or_env; det=det, random_policy=random_policy, mask=nothing)
    end
end
# MultiContinuous
function get_action(::Type{PPO{MultiContinuousAct}}, agent::A, obs_or_env; det=false, random_policy=false) where {A <: AbstractAgent}
    obs = obs_or_env isa AbstractEnv ? obs_or_env.state : obs_or_env
    return _get_action_multicontinuous(agent, obs; det=det, random_policy=random_policy)
end

## ---------------------------- Internal Core Logic (Type-Stable) ---------------------------- ###
# DISCRETE
function _get_action_discrete(agent::CombinedActorCritic, obs; det=false, random_policy=false, mask=nothing)
    outputs, value = agent.combined_model(obs)
    if isnothing(mask)
        mask = zeros(Float32, length(outputs))
    end
    ws = dropdims(masked_probabilities(mask, outputs), dims=2)
    indices = 1:length(ws)
    action = det ? argmax(ws) : sample(indices, Weights(ws))
    return action, value, ws
end
function _get_action_discrete(agent::StandardActorCritic, obs; det=false, random_policy=false, mask=nothing)
    outputs = agent.actor_model(obs)
    value = agent.critic_model(obs)
    if isnothing(mask)
        mask = zeros(Float32, length(outputs))
    end
    ws = dropdims(masked_probabilities(mask, outputs), dims=2)
    indices = 1:length(ws)
    action = det ? argmax(ws) : sample(indices, Weights(ws))
    return action, value, ws
end
# CONTINUOUS
function _get_action_continuous(agent::StandardActorCritic, obs; det=false, random_policy=false)
    α_raw, β_raw = agent.actor_model(obs)
    value = agent.critic_model(obs)
    α = Flux.softplus.(α_raw) .+ 1.0f0 
    β = Flux.softplus.(β_raw) .+ 1.0f0
    dist = Beta(α[1], β[1])
    if det
        u = α[1] / (α[1] + β[1]) # Stable Mean
    elseif random_policy
        u = rand(Float32)
    else
        u = rand(dist)
    end
    action = 2.0f0 * u - 1.0f0 # Scale to [-1, 1] this is an arbitrary design choice but one that should be well documented and consistent.
    return action, value[1], logpdf(dist, u)
end
function _get_action_continuous(agent::CombinedActorCritic, obs; det=false, random_policy=false)
    α_raw, β_raw, value = agent.combined_model(obs)
    α = Flux.softplus.(α_raw) .+ 1.0f0
    β = Flux.softplus.(β_raw) .+ 1.0f0
    dist = Beta(α[1], β[1])
    if det
        u = α[1] / (α[1] + β[1]) # Stable Mean
    elseif random_policy
        u = rand(Float32)
    else
        u = rand(dist)
    end
    action = 2.0f0 * u - 1.0f0
    return action, value[1], logpdf(dist, u)
end
# MULTI-CONTINUOUS
function _get_action_multicontinuous(agent::StandardActorCritic, obs; det=false, random_policy=false)
    α_raw_vec, β_raw_vec = dropdims.(agent.actor_model(obs), dims=2)
    value = agent.critic_model(obs)
    α_vec = Flux.softplus.(α_raw_vec) .+ 1.0f0
    β_vec = Flux.softplus.(β_raw_vec) .+ 1.0f0
    dists = Beta.(α_vec, β_vec)
    if det
        u_vec = α_vec ./ (α_vec .+ β_vec)
    elseif random_policy
        u_vec = rand(Float32, length(α_vec))
    else
        u_vec = Float32.(rand.(dists))
    end
    action_vec = 2.0f0 .* u_vec .- 1.0f0
    return action_vec, value[1], sum(logpdf.(dists, u_vec))
end
function _get_action_multicontinuous(agent::CombinedActorCritic, obs; det=false, random_policy=false)
    α_raw_vec, β_raw_vec, value = dropdims.(agent.combined_model(obs), dims=2)
    α_vec = Flux.softplus.(α_raw_vec) .+ 1.0f0
    β_vec = Flux.softplus.(β_raw_vec) .+ 1.0f0
    dists = Beta.(α_vec, β_vec)
    if det
        u_vec = α_vec ./ (α_vec .+ β_vec)
    elseif random_policy
        u_vec = rand(Float32, length(α_vec))
    else
        u_vec = Float32.(rand.(dists))
    end
    action_vec = 2.0f0 .* u_vec .- 1.0f0
    return action_vec, value[1], sum(logpdf.(dists, u_vec))
end

# MULTI-DISCRETE
function _get_action_multidiscrete(agent::StandardActorCritic, obs; det=false, random_policy=false, mask=nothing)
    outputs = agent.actor_model(obs)
    value = agent.critic_model(obs)
    
    # We assume equal sized heads for now, i.e each action 
    # dimension has the same number of action options.
    # This should really be configurable/
    # For ParticleChase: 6 outputs -> 2 heads of size 3
    num_heads = 2
    head_size = Int(length(outputs) / num_heads)
    
    actions = Vector{Int}(undef, num_heads)
    all_ws = []
    
    for h in 1:num_heads
        head_logits = outputs[((h-1)*head_size + 1):(h*head_size)]
        # Simple mask for each head if provided, else zeros
        h_mask = zeros(Float32, head_size)
        ws = vec(masked_probabilities(h_mask, head_logits))
        indices = 1:head_size
        actions[h] = det ? argmax(ws) : sample(indices, Weights(ws))
        push!(all_ws, ws)
    end
    
    # Return actions as vector, value, and the combined weights for the buffer
    return actions, value[1], vcat(all_ws...)
end

function _get_action_multidiscrete(agent::CombinedActorCritic, obs; det=false, random_policy=false, mask=nothing)
    outputs, value = agent.combined_model(obs)
    
    num_heads = 2
    head_size = Int(length(outputs) / num_heads)
    
    actions = Vector{Int}(undef, num_heads)
    all_ws = []
    
    for h in 1:num_heads
        head_logits = outputs[((h-1)*head_size + 1):(h*head_size)]
        h_mask = zeros(Float32, head_size)
        ws = vec(masked_probabilities(h_mask, head_logits))
        indices = 1:head_size
        actions[h] = det ? argmax(ws) : sample(indices, Weights(ws))
        push!(all_ws, ws)
    end
    
    return actions, value[1], vcat(all_ws...)
end

# ---------------------------- functions for calculating the log probability -------------------- #

function beta_logpdf(α, β, u)
    # logpdf(Beta(α, β), u) = (α-1)*log(u) + (β-1)*log(1-u) - lbeta(α, β)
    # Clip u to avoid log(0)
    u_clipped = clamp.(u, 1f-6, 1f0 - 1f-6)
    return (α .- 1.0f0) .* log.(u_clipped) .+ (β .- 1.0f0) .* log.(1.0f0 .- u_clipped) .- logbeta.(α, β)
end

function beta_entropy(α, β)
    return logbeta(α, β) - (α - 1f0) * digamma(α) - (β - 1f0) * digamma(β) + (α + β - 2f0) * digamma(α + β)
end

# ------ functions for updating the actor learners by copying the centralized model --------------------- #

function update_actor_learners!(agent::CombinedActorCritic, alg::PPO{G}) where {G <: AbstractAction}
    cpu_combined_model = agent.combined_model.model |> cpu
    for worker in alg.worker_agents
        Flux.loadmodel!(worker.combined_model.model, cpu_combined_model)
    end
end

function update_actor_learners!(agent::StandardActorCritic, alg::PPO{G}) where {G <: AbstractAction}
    cpu_actor_model = agent.actor_model.model |> cpu
    cpu_critic_model = agent.critic_model.model |> cpu

    for worker in alg.worker_agents
        Flux.loadmodel!(worker.actor_model.model, cpu_actor_model)
        Flux.loadmodel!(worker.critic_model.model, cpu_critic_model)
    end
end

# ------------ misc utils ----------------------- #

function unzip(a; dims = 1)
    return map(fieldnames(eltype(a))) do x
        segments = getfield.(a, x)
        flat_data = reduce(vcat, segments)
        return flat_data
    end
end

function masked_probabilities(mask::Array{Float32}, outputs::Array{Float32})
    # Given an action mask in the case of discrete actions with state dependent
    # action spaces, mask out those forbidden actions.
    # Add small epsilon for numerical stability
    weights = softmax(mask .+ outputs; dims = 1) .+ 1f-10
    # Re-normalize
    return weights ./ sum(weights, dims=1)
end

function infer_mask(probabilities::AbstractVector{Float32})
    # given a probability infer the mask used. This might
    # define an over complete mask if a model learns to 
    # completely zero certain actions itself. Unlikely
    idxs = findall(iszero, probabilities)
    N = length(probabilities)
    mask = zeros(Float32, N) 
    mask[idxs] .= mask[idxs] .- Inf32
    return mask
end

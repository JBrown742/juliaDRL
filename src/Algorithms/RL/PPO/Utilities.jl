
# ------------ get_actions functions ------- #

# ------------------------  Unambiguous entry-point dispatches ----------------------- #

function get_action(::Type{PPO{DiscreteAct}}, agent::A, obs_or_env; det=false, random_policy=false) where {A <: AbstractAgent}
    if obs_or_env isa AbstractEnv
        mask = hasfield(typeof(obs_or_env), :action_mask) ? obs_or_env.action_mask : nothing
        return _get_action_discrete(agent, obs_or_env.state; det=det, random_policy=random_policy, mask=mask)
    else
        return _get_action_discrete(agent, obs_or_env; det=det, random_policy=random_policy, mask=nothing)
    end
end

function get_action(::Type{PPO{ContinuousAct}}, agent::A, obs_or_env; det=false, random_policy=false) where {A <: AbstractAgent}
    obs = obs_or_env isa AbstractEnv ? obs_or_env.state : obs_or_env
    return _get_action_continuous(agent, obs; det=det, random_policy=random_policy)
end

function get_action(::Type{PPO{MultiDiscreteAct}}, agent::A, obs_or_env; det=false, random_policy=false) where {A <: AbstractAgent}
    if obs_or_env isa AbstractEnv
        mask = hasfield(typeof(obs_or_env), :action_mask) ? obs_or_env.action_mask : nothing
        return _get_action_multidiscrete(agent, obs_or_env.state; det=det, random_policy=random_policy, mask=mask)
    else
        return _get_action_multidiscrete(agent, obs_or_env; det=det, random_policy=random_policy, mask=nothing)
    end
end

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

function _get_action_continuous(agent::StandardActorCritic, obs; det=false, random_policy=false)
    μ_raw, log_σ_raw = agent.actor_model(obs)
    value = agent.critic_model(obs)
    # STABILITY FIX: Clamp log_sigma to a reasonable range
    log_σ = clamp.(log_σ_raw, -2.0f0, 0.0f0)
    σ = exp.(log_σ)
    μ = tanh.(μ_raw) # Squash mean to [-1, 1]
    if det
        action = Float32.(μ)
    elseif random_policy
        action = Float32.(rand(Uniform(-1, 1), length(σ)))
    else
        action = Float32.(μ .+ σ .* randn())
    end
    return action[1], value[1], log_gauss_pdf(action[1], μ[1], σ[1])
end

function _get_action_continuous(agent::CombinedActorCritic, obs; det=false, random_policy=false)
    μ_raw, log_σ_raw, value = agent.combined_model(obs)
    log_σ = clamp.(log_σ_raw, -2.0f0, 0.0f0)
    σ = exp.(log_σ)
    μ = tanh.(μ_raw) # Squash mean
    if det
        action = Float32.(μ)
    elseif random_policy
        action = Float32.(rand(Uniform(-1, 1), length(σ)))
    else    
        action = Float32.(μ .+ σ .* randn())
    end
    return action[1], value[1], log_gauss_pdf(action[1], μ[1], σ[1])
end

# MULTI-CONTINUOUS
function _get_action_multicontinuous(agent::StandardActorCritic, obs; det=false, random_policy=false)
    μ_raw_vec, log_σ_raw_vec = dropdims.(agent.actor_model(obs), dims=2)
    value = agent.critic_model(obs)
    log_σ_vec = clamp.(log_σ_raw_vec, -2.0f0, 0.0f0)
    σ_vec = exp.(log_σ_vec)
    μ_vec = tanh.(μ_raw_vec) # Squash mean
    if det
        action_vec = Float32.(μ_vec)
    elseif random_policy
        action_vec = Float32.(rand(Uniform(-1, 1), length(σ_vec)))
    else
        action_vec = Float32.(μ_vec .+ σ_vec .* randn(length(σ_vec)))
    end
    return action_vec, value[1], log_gauss_pdf_multi(action_vec, μ_vec, σ_vec)
end

function _get_action_multicontinuous(agent::CombinedActorCritic, obs; det=false, random_policy=false)
    μ_raw_vec, log_σ_raw_vec, value = dropdims.(agent.combined_model(obs), dims=2)
    log_σ_vec = clamp.(log_σ_raw_vec, -2.0f0, 0.0f0)
    σ_vec = exp.(log_σ_vec)
    μ_vec = tanh.(μ_raw_vec) # Squash mean
    if det
        action_vec = Float32.(μ_vec)
    elseif random_policy
        action_vec = Float32.(rand(Uniform(-1, 1), length(σ_vec)))
    else
        action_vec = Float32.(μ_vec .+ σ_vec .* randn(length(σ_vec)))
    end
    return action_vec, value[1], log_gauss_pdf_multi(action_vec, μ_vec, σ_vec)
end

# MULTI-DISCRETE
function _get_action_multidiscrete(agent::StandardActorCritic, obs; det=false, random_policy=false, mask=nothing)
    outputs = agent.actor_model(obs)
    value = agent.critic_model(obs)
    
    # We assume equal sized heads for now (DeepMind Standard: should be configurable)
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

function log_gauss_pdf(x::R, μ::R, σ::R=0.05f0) where {R <: Real}
    return -log(σ) - log(sqrt(2f0 * π))  - 0.5f0 * (((x - μ)/σ) ^ 2) 
end

function log_gauss_pdf(x::Vector{R}, μ::Vector{R}, σ::Vector{R}) where {R <: Real}
    return log_gauss_pdf.(x, μ, σ)
end

function log_gauss_pdf_multi(x::Vector{R}, μ::Vector{R}, σ::Vector{R}) where {R <: Real}
    vals = log_gauss_pdf.(x, μ, σ)
    return sum(vals)[1] 
end

function log_gauss_pdf_multi(x::Vector{Vector{R}}, μ::Matrix{R}, σ::Matrix{R}) where {R <: Real}
    x_mat = hcat(x...)
    vals = log_gauss_pdf.(x_mat, μ, σ)
    return dropdims(sum(vals, dims=1), dims=1)
end

function log_gauss_pdf_multi(x::Matrix{R}, μ::Matrix{R}, σ::Matrix{R}) where {R <: Real}
    vals = log_gauss_pdf.(x, μ, σ)
    return dropdims(sum(vals, dims=1), dims=1)
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
    # Add small epsilon for numerical stability
    weights = softmax(mask .+ outputs; dims = 1) .+ 1f-10
    # Re-normalize
    return weights ./ sum(weights, dims=1)
end

function infer_mask(probabilities::Vector{Float32})
    idxs = findall(iszero, probabilities)
    N = length(probabilities)
    mask = zeros(Float32, N) 
    mask[idxs] .= mask[idxs] .- Inf32
    return mask
end

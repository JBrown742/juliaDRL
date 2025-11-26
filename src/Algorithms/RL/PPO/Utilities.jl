
# ------------ get_actions functions ------- #

## ---------------------------- Discrete Actions ---------------------------------- ###
function get_action(::Type{PPO{DiscreteAct}}, agent::CombinedActorCritic, obs::O; det=false, random_policy=false, mask=nothing) where {O <: AbstractObservation}
    outputs, value = agent.combined_model(obs)
    if isnothing(mask)
        # the mask is additive so -inf is masked and 0 is not
        mask = zeros(Float32, length(outputs))
    end
    ws = dropdims(masked_probabilities(mask, outputs), dims=2)
    indices = 1:length(ws)
    if det == true
        action = argmax(ws)
    else
        action = sample(indices, Weights(ws))
    end
    return action, value, ws
end


function get_action(::Type{PPO{DiscreteAct}}, agent::StandardActorCritic, obs::O; det=false, random_policy=false, mask=nothing) where {O <: AbstractObservation}
    outputs = agent.actor_model(obs)
    value = agent.critic_model(obs)
    if isnothing(mask)
        mask = zeros(Float32, length(outputs))
    end
    ws = dropdims(masked_probabilities(mask, outputs), dims=2)
    indices = collect(1:length(ws))
    if det == true
        action = argmax(ws)
    else
        action = sample(indices, Weights(ws))
    end
    return action, value, ws
end

## ------------------------------- Continuous Actions --------------------------------- ##

function get_action(::Type{PPO{ContinuousAct}}, agent::StandardActorCritic, obs::O; det=false, random_policy=false) where {O <: AbstractObservation}
    μ, log_σ = agent.actor_model(obs)
    value = agent.critic_model(obs)
    σ = exp.(log_σ)
    if det == true
        action_raw = Float32.(μ)
    elseif random_policy == true
        action_raw = Float32.(rand(Uniform(-1, 1), length(σ)))
    else
        action_raw = Float32.(μ .+ σ .* randn())
    end
    action = action_raw
    return action[1], value[1], log_gauss_pdf(action[1], μ[1], σ[1])
end



function get_action(::Type{PPO{ContinuousAct}}, agent::CombinedActorCritic, obs::O; det=false, random_policy=false) where {O <: AbstractObservation}
    μ, log_σ, value = agent.combined_model(obs)
    σ = exp.(log_σ)
    if det == true
        action_raw = Float32.(μ)
    elseif random_policy == true
        action_raw = Float32.(rand(Uniform(-1, 1), length(σ)))
    else    
        action_raw = Float32.(μ .+ σ .* randn())
    end
    action = action_raw
    return action[1], value[1], log_gauss_pdf(action[1], μ[1], σ[1])
end

## ------------------------------- Multicontinuous Actions ---------------------------- ##
function get_action(::Type{PPO{MultiContinuousAct}}, agent::StandardActorCritic, obs::O; det=false, random_policy=false) where {O <: AbstractObservation}
    # Need to decide here how we expect 
    μ_vec, log_σ_vec = dropdims.(agent.actor_model(obs), dims=2) # dropdims here so that each is a vector. This function will be called exclusively by a single agent
    value = agent.critic_model(obs)
    σ_vec = exp.(log_σ_vec)

    if det == true
        action_vec_raw = Float32.(μ_vec)
    elseif random_policy == true
        action_vec_raw = Float32.(rand(Uniform(-1, 1), length(σ_vec)))
    else
        action_vec_raw = Float32.(μ_vec .+ σ_vec .* randn(length(σ_vec)))
    end
    action_vec = action_vec_raw
    return action_vec, value[1], log_gauss_pdf_multi(action_vec, μ_vec, σ_vec)
end

function get_action(::Type{PPO{MultiContinuousAct}}, agent::CombinedActorCritic, obs::O; det=false, random_policy=false) where {O <: AbstractObservation}
    μ_vec, log_σ_vec, value = dropdims.(agent.combined_model(obs), dims=2)
    σ_vec = exp.(log_σ_vec)
    if det == true
        action_vec_raw = Float32.(μ_vec)
    elseif random_policy == true
        action_vec_raw = Float32.(rand(Uniform(-1, 1), length(σ_vec)))
    else
        action_vec_raw = Float32.(μ_vec .+ σ_vec .* randn(length(σ_vec)))
    end
    action_vec = action_vec_raw
    return action_vec, value[1], log_gauss_pdf_multi(action_vec, μ_vec, σ_vec)
end

## --------------------------- MultiDiscrete actions ----------------------------------- ##
function get_action(::Type{PPO{MultiDiscreteAct}}, agent::StandardActorCritic, obs::O; det=false, random_policy=false) where {O <: AbstractObservation}
    μ, log_σ = agent.actor_model(obs)
    value = agent.critic_model(obs)
    σ = exp.(log_σ)    
    if det == true
        action = μ[1]
    else
        d = Normal(Float64(μ[1]), σ[1])
        action = Float32(rand(d, 1)[1])
    end
    return action, value[1], log_gauss_pdf(action, μ[1], σ[1])
end

function get_action(::Type{PPO{MultiDiscreteAct}}, agent::CombinedActorCritic, obs::O; det=false, random_policy=false) where {O <: AbstractObservation}
    μ, log_σ, value = agent.combined_model(obs)
    σ = exp.(log_σ)

    if det == true
        action = μ[1]
    else
        d = Normal(Float64(μ[1]), σ[1])
        action = Float32(rand(d, 1)[1])
    end
    return action, value[1], log_gauss_pdf(action, μ[1], σ[1])
end

# ------------------------  Dispatches for use within data collection ------- #
function get_action(::Type{PPO{DiscreteAct}}, agent::A, env::E; det=false, random_policy=false) where {E <: AbstractEnv, A <: AbstractAgent}
    return get_action(PPO{DiscreteAct}, agent, env.state; det=det, random_policy=random_policy, mask=env.action_mask)
end

function get_action(::Type{PPO{ContinuousAct}}, agent::A, env::E; det=false, random_policy=false) where {E <: AbstractEnv, A <: AbstractAgent}
    return get_action(PPO{ContinuousAct}, agent, env.state; det=det, random_policy=random_policy)
end

function get_action(::Type{PPO{MultiDiscreteAct}}, agent::A, env::E; det=false, random_policy=false) where {E <: AbstractEnv, A <: AbstractAgent}
    return get_action(PPO{MultiDiscreteAct}, agent, env.state; det=det, random_policy=random_policy, mask=env.action_mask)
end

function get_action(::Type{PPO{MultiContinuousAct}}, agent::A, env::E; det=false, random_policy=false) where {E <: AbstractEnv, A <: AbstractAgent}
    return get_action(PPO{MultiContinuousAct}, agent, env.state; det=det, random_policy=random_policy)
end

# ---------------------------- functions for calculating the log probability -------------------- #
#
# We use log probabilities here for continuous actions in order to avoid numberical underflow.
# We simple need to exp(log(p1) - log(p2)) to recover r.
#

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
    return map(x -> cat(getfield.(a, x)..., dims=dims), fieldnames(eltype(a)))
end

function masked_probabilities(mask::Array{Float32}, outputs::Array{Float32})
    # Here the mask is additive as opposed to multiplicative
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

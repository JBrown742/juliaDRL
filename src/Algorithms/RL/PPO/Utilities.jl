
# ------------ get_actions functions ------- #

## ---------------------------- Discrete Actions ---------------------------------- ###
function get_action(::Type{PPO{DiscreteAct}}, agent::CombinedActorCritic, obs::O; det=false, mask=nothing) where {O <: AbstractObservation}
    outputs, value = agent.combined_model(obs)
    if isnothing(mask)
        mask = ones(length(outputs))
    end
    masked_outputs = outputs .+ mask
    ws = Float32.(softmax(masked_outputs; dims = 1))
    indices = collect(1:length(
        funws))
    if det == true
        action = argmax(ws)
    else
        action = sample(indices, Weights(ws))
    end
    return action, value, ws
end


function get_action(::Type{PPO{DiscreteAct}}, agent::StandardActorCritic, obs::O; det=false, mask=nothing) where {O <: AbstractObservation}
    outputs = agent.actor_model(obs)
    value = agent.critic_model(obs)
    if isnothing(mask)
        mask = ones(length(outputs))
    end
    masked_outputs = outputs .+ mask
    ws = Float32.(softmax(masked_outputs; dims = 1))
    indices = collect(1:length(ws))
    if det == true
        action = argmax(ws)
    else
        action = sample(indices, Weights(ws))
    end
    return action, value, ws
end

## ------------------------------- Continuous Actions --------------------------------- ##

function get_action(::Type{PPO{ContinuousAct}}, agent::StandardActorCritic, obs::O; det=false) where {O <: AbstractObservation}
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



function get_action(::Type{PPO{ContinuousAct}}, agent::CombinedActorCritic, obs::O; det=false) where {O <: AbstractObservation}
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

## ------------------------------- Multicontinuous Actions ---------------------------- ##
function get_action(::Type{PPO{MultiContinuousAct}}, agent::StandardActorCritic, obs::O; det=false) where {O <: AbstractObservation}
    # Need to decide here how we expect 
    μ_vec, log_σ_vec = dropdims.(agent.actor_model(obs), dims=2)
    value = agent.critic_model(obs)
    σ_vec = exp.(log_σ_vec)
    if det == true
        action_vec = μ_vec
    else
        d_vec = [Normal(Float64.(μ), Float64.(σ)) for (μ, σ) in zip(μ_vec, σ_vec)]
        action_vec = Float32.(rand.(d_vec))
    end
    return action_vec, value[1], prod(log_gauss_pdf_multi(action_vec, μ_vec, σ_vec))
end

function get_action(::Type{PPO{MultiContinuousAct}}, agent::CombinedActorCritic, obs::O; det=false) where {O <: AbstractObservation}
    μ, log_σ, value = agent.combined_model(obs)
    σ = exp.(log_σ)
    if det == true
        action = μ[1]
    else
        d = Normal(Float64.(μ[1]), σ[1])
        action = Float32(rand(d, 1)[1])
    end
    return action, value[1], log_gauss_pdf_multi(action, μ[1], σ[1])
end

## --------------------------- MultiDiscrete actions ----------------------------------- ##
function get_action(::Type{PPO{MultiDiscreteAct}}, agent::StandardActorCritic, obs::O; det=false) where {O <: AbstractObservation}
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

function get_action(::Type{PPO{MultiDiscreteAct}}, agent::CombinedActorCritic, obs::O; det=false) where {O <: AbstractObservation}
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
function get_action(::Type{PPO{DiscreteAct}}, agent::A, env::E; det=false) where {E <: AbstractEnv, A <: AbstractAgent}
    return get_action(PPO{DiscreteAct}, agent, env.state; det=det, mask=env.action_mask)
end

function get_action(::Type{PPO{ContinuousAct}}, agent::A, env::E; det=false) where {E <: AbstractEnv, A <: AbstractAgent}
    return get_action(PPO{ContinuousAct}, agent, env.state; det=det)
end

function get_action(::Type{PPO{MultiDiscreteAct}}, agent::A, env::E; det=false) where {E <: AbstractEnv, A <: AbstractAgent}
    return get_action(PPO{MultiDiscreteAct}, agent, env.state; det=det, mask=env.action_mask)
end

function get_action(::Type{PPO{MultiContinuousAct}}, agent::A, env::E; det=false) where {E <: AbstractEnv, A <: AbstractAgent}
    return get_action(PPO{MultiContinuousAct}, agent, env.state; det=det)
end

function log_gauss_pdf(x::Float32, μ::Float32, σ::Float32=0.05f0)
    return -log(σ) - log(sqrt(2 * π))  - 0.5 * (((x - μ)/σ) ^ 2)
end

function log_gauss_pdf(x::Vector{Float32}, μ::Vector{Float32}, σ::Vector{Float32})
    return -log.(σ) .- log.(sqrt.(2 .* π))  .- 0.5 .* (((x .- μ)./σ) .^ 2)
end

function log_gauss_pdf_multi(x::Vector{Float32}, μ::Vector{Float32}, σ::Vector{Float32})
    vals = -log.(σ) .- log.(sqrt.(2 .* π))  .- 0.5 .* (((x .- μ)./σ) .^ 2)
    return prod(vals)
end

function log_gauss_pdf_multi(x::Vector{Vector{Float32}}, μ::Matrix{Float32}, σ::Matrix{Float32})
    x_mat = cat(x..., dims=2)
    vals = -log.(σ) .- log.(sqrt.(2 .* π))  .- 0.5 .* (((x_mat .- μ)./σ) .^ 2)
    return dropdims(prod(vals, dims=1), dims=1)
end


function update_actor_learners!(agent::CombinedActorCritic, alg::PPO{G}) where {G <: AbstractAction}
    for (idx,p) in enumerate(Flux.params(agent.combined_model.model))
        for agent_idx in 1:alg.N
            Flux.params(alg.worker_agents[agent_idx].combined_model.model)[idx] .= copy(p |> cpu)
        end
    end
end

function update_actor_learners!(agent::StandardActorCritic, alg::PPO{G}) where {G <: AbstractAction}
    for (idx,p) in enumerate(Flux.params(agent.actor_model.model))
        for agent_idx in 1:alg.N
            Flux.params(alg.worker_agents[agent_idx].actor_model.model)[idx] .= copy(p |> cpu)
        end
    end
    for (idx,p) in enumerate(Flux.params(agent.critic_model.model))
        for agent_idx in 1:alg.N
            Flux.params(alg.worker_agents[agent_idx].critic_model.model)[idx] .= copy(p |> cpu)
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
    fun
end

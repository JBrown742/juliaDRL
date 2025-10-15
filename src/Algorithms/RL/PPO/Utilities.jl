
# ------------ get_actions functions ------- #

## ---------------------------- Discrete Actions ---------------------------------- ###
function get_action(::Type{PPO{DiscreteAct}}, agent::CombinedActorCritic, obs::O; det=false, mask=nothing) where {O <: AbstractObservation}
    outputs, value = agent.combined_model(obs)
    if isnothing(mask)
        mask = ones(length(outputs))
    end
    masked_outputs = outputs .+ mask
    ws = dropdims(Float32.(softmax(masked_outputs; dims = 1)), dims=2)
    indices = collect(1:length(ws))
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
    ws = dropdims(Float32.(softmax(masked_outputs; dims = 1)), dims=2)
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
    μ_vec, log_σ_vec = dropdims.(agent.actor_model(obs), dims=2) # dropdims here so that each is a vector. This function will be called exclusively by a single agent
    value = agent.critic_model(obs)
    σ_vec = exp.(log_σ_vec)
    if det == true
        action_vec = μ_vec
    else
        d_vec = [Normal(Float64.(μ), Float64.(σ)) for (μ, σ) in zip(μ_vec, σ_vec)]
        action_vec = Float32.(rand.(d_vec))
    end
    return action_vec, value[1], log_gauss_pdf_multi(action_vec, μ_vec, σ_vec)
end

function get_action(::Type{PPO{MultiContinuousAct}}, agent::CombinedActorCritic, obs::O; det=false) where {O <: AbstractObservation}
    μ_vec, log_σ_vec, value = dropdims.(agent.combined_model(obs), dims=2)
    σ_vec = exp.(log_σ_vec)
    if det == true
        action_vec = μ_vec
    else
        d_vec = [Normal(Float64.(μ), Float64.(σ)) for (μ, σ) in zip(μ_vec, σ_vec)]
        action_vec = Float32.(rand.(d_vec))
    end
    return action_vec, value[1], log_gauss_pdf_multi(action_vec, μ_vec, σ_vec)
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
    return -log(σ) - log(sqrt(2f0 * π))  - 0.5f0 * (((x - μ)/σ) ^ 2)
end

function log_gauss_pdf(x::Vector{Float32}, μ::Vector{Float32}, σ::Vector{Float32})
    return -log.(σ) .- log(sqrt(2f0 .* π))  .- 0.5f0 .* (((x .- μ)./σ) .^ 2)
end

function log_gauss_pdf_multi(x::Vector{Float32}, μ::Vector{Float32}, σ::Vector{Float32})
    vals = -log.(σ) .- log.(sqrt.(2f0 .* π))  .- 0.5f0 .* (((x .- μ)./σ) .^ 2)
    return sum(vals)[1]
end

function log_gauss_pdf_multi(x::Vector{Vector{Float32}}, μ::Matrix{Float32}, σ::Matrix{Float32})
    x_mat = cat(x..., dims=2)
    vals = -log.(σ) .- log(sqrt(2f0 .* π))  .- 0.5f0 .* (((x_mat .- μ)./σ) .^ 2)
    return dropdims(sum(vals, dims=1), dims=1)
end

function log_gauss_pdf_multi(x::Matrix{Float32}, μ::Matrix{Float32}, σ::Matrix{Float32})
    vals = -log.(σ) .- log.(sqrt.(2f0 .* π))  .- 0.5f0 .* (((x .- μ)./σ) .^ 2)
    return dropdims(sum(vals, dims=1), dims=1)
end


# function update_actor_learners!(agent::CombinedActorCritic, alg::PPO{G}) where {G <: AbstractAction}
#     cpu_combined_model = agent.combined_model.model |> cpu
#     for (idx,l) in enumerate(Flux.trainable(cpu_combined_model).layers)
#         if typeof(l) <: Split
#             for (i, path) in enumerate(l.paths)
#                 for agent_idx in 1:alg.N
#                     Flux.trainable(alg.worker_agents[agent_idx].combined_model.model).layers[idx].paths[i].weight .= copy(path.weight)
#                     Flux.trainable(alg.worker_agents[agent_idx].combined_model.model).layers[idx].paths[i].bias .= copy(path.bias)
#                 end
#             end
#         else
#             for agent_idx in 1:alg.N
#                 Flux.trainable(alg.worker_agents[agent_idx].combined_model.model).layers[idx].weight .= copy(l.weight)
#                 Flux.trainable(alg.worker_agents[agent_idx].combined_model.model).layers[idx].bias .= copy(l.bias)
#             end
#         end
#     end
# end

function update_actor_learners!(agent::CombinedActorCritic, alg::PPO{G}) where {G <: AbstractAction}
    cpu_combined_model = agent.combined_model.model |> cpu
    for worker in alg.worker_agents
        Flux.loadmodel!(worker.combined_model.model, cpu_combined_model)
    end
end

# function update_actor_learners!(agent::StandardActorCritic, alg::PPO{G}) where {G <: AbstractAction} # Can we make this a function that is implemented on each worker?
#     cpu_actor_model = agent.actor_model.model |> cpu
#     cpu_critic_model = agent.critic_model.model |> cpu
#     for (idx,l) in enumerate(Flux.trainable(cpu_actor_model).layers)
#         if typeof(l) <: Split
#             for (i, path) in enumerate(l.paths)
#                 for agent_idx in 1:alg.N
#                     Flux.trainable(alg.worker_agents[agent_idx].actor_model.model).layers[idx].paths[i].weight .= copy(path.weight)
#                     Flux.trainable(alg.worker_agents[agent_idx].actor_model.model).layers[idx].paths[i].bias .= copy(path.bias)
#                 end
#             end
#         else
#             for agent_idx in 1:alg.N
#                 Flux.trainable(alg.worker_agents[agent_idx].actor_model.model).layers[idx].weight .= copy(l.weight)
#                 Flux.trainable(alg.worker_agents[agent_idx].actor_model.model).layers[idx].bias .= copy(l.bias)
#             end
#         end
#     end
#     for (idx,l) in enumerate(Flux.trainable(cpu_critic_model).layers)
#         if typeof(l) <: Split
#             for (i, path) in enumerate(l.paths)
#                 for agent_idx in 1:alg.N
#                     Flux.trainable(alg.worker_agents[agent_idx].critic_model.model).layers[idx].paths[i].weight .= copy(path.weight)
#                     Flux.trainable(alg.worker_agents[agent_idx].critic_model.model).layers[idx].paths[i].bias .= copy(path.bias)
#                 end
#             end
#         else
#             for agent_idx in 1:alg.N
#                 Flux.trainable(alg.worker_agents[agent_idx].critic_model.model).layers[idx].weight .= copy(l.weight)
#                 Flux.trainable(alg.worker_agents[agent_idx].critic_model.model).layers[idx].bias .= copy(l.bias)
#             end
#         end
#     end
# end

function update_actor_learners!(agent::StandardActorCritic, alg::PPO{G}) where {G <: AbstractAction}
    cpu_actor_model = agent.actor_model.model |> cpu
    cpu_critic_model = agent.critic_model.model |> cpu

    for worker in alg.worker_agents
        Flux.loadmodel!(worker.actor_model.model, cpu_actor_model)
        Flux.loadmodel!(worker.critic_model.model, cpu_critic_model)
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

#
# Nowhere should the algorithm be dependent on the state type
# as long as the state is of one of the 3 defined types, then it only depends on the action type
# and in that case an algorithm which suContinuouspports two different action types may differ enough
# to warrant two sparate structs
mutable struct ContinuousPPO <: AbstractAlgorithm
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
    optimizers::Vector{AbstractRule}
    optimizer_s::Vector{NamedTuple}
    function ContinuousPPO(N::Int, T::Int, K::Int, agent::A, batch_size::Int, γ::Float64, λ::Float64, 
        ϵ::Float64, c1::Float64, c2::Float64, sync_frequency::Int, optimizer::Vector{O}) where {A <: AbstractAgent, O <: AbstractRule}
        m = agent.model
        # |> gpu -- removing GPU functionality for now
        als = [AbstractAgent(deepcopy(m)) for i in 1:N]
        exponents = collect(0:T*100)
        advantage_coefficients = (λ * γ) .^ exponents
        su = [Flux.setup(op, mod) for (op, mod) in zip(optimizer, agent.model)]
        return new(N, T, K, agent, als, batch_size, γ, λ, ϵ, c1, c2, sync_frequency, advantage_coefficients, optimizer, su)
    end
    function ContinuousPPO(N::Int, T::Int, K::Int,  agent::A,
        optimizer::Vector{O}) where {A <: AbstractAgent, O <: AbstractRule}
        m = agent.model
        # |> gpu -- removing GPU functionality for now
        als = [AbstractAgent(deepcopy(m)) for i in 1:N]
        exponents = collect(0:T*100)
        advantage_coefficients = (0.95 * 0.99) .^ exponents
        su = [Flux.setup(op, mod) for (op, mod) in zip(optimizer, agent.model)]
        return new(N, T, K, agent, als, 64, 0.99, 0.95, 0.2, 0.8, 0.001, 2, advantage_coefficients, optimizer, su)
    end

end

## The environment is still baked into this algo...
function train!(alg::ContinuousPPO, transitions::Vector{Experience}, probabilities::Vector{Float32},
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
        actions = Vector{ContinuousAct}() # currently ContinuousPPO will only support Continuous actions
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

        ∇ = Flux.gradient(alg.central_agent.model[1]) do m # track gradients
            μ, log_σ = dropdims.(m(reduce(hcat, states)), dims=1)
            σ = exp.(log_σ)
            new_log_probs = log_gauss_pdf(actions, μ, σ)
            old_log_probs = batch_probabilities
            r = exp.(new_log_probs .- old_log_probs)
            clamped_r = clamp.(r, 1 - alg.ϵ, 1 + alg.ϵ)
            vals = minimum.(batch_advantages .* clamped_r)
            L_CLIP = 1 * mean(vals)
            # entropy = 0.5 .* log.(2 * π .* σ .^ 2 .+ 0.5) # there is a closed form for the entropy of a gaussian
            -1 * L_CLIP
        end
        Flux.update!(alg.optimizer_s[1], alg.central_agent.model[1], ∇[1])
        ∇2 = Flux.gradient(alg.central_agent.model[2]) do m # track gradients
            state_values = m(reduce(hcat, states))
            Flux.Losses.mse(batch_bellman_targets, state_values)
        end
        Flux.update!(alg.optimizer_s[2], alg.central_agent.model[2], ∇2[1])
    end
end

function collect_trajectory_segment_continuous!(env::E, agent::A, info::Dict{Symbol, Any}) where {E <: AbstractEnv, A <: AbstractAgent}
    T::Int = info[:T]
    γ::Float32 = info[:γ]
    advantage_coefficients::Vector{Float32} = info[:advantage_coefficients]
    # initialize vectors to store all the transitions encountered
    local_segment_count = 0
    total_trajectory = Vector{Experience}()
    all_targets = Vector{Float32}()
    all_errors = Vector{Float32}()
    all_advantages = Vector{Float32}()
    trajectory_probabilities = Vector{Float32}()
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
            action, state_value, log_prob = get_action(ContinuousPPO, agent, state) # get the action, the value and the probability
            push!(trajectory_probabilities, log_prob)
            new_state, reward, terminal = step!(env, action) # take a step of the hopper envs
            experience = Experience(state, action, new_state, reward, terminal)
            push!(current_ep_trajectory, experience)
            _, next_state_value, _ = get_action(ContinuousPPO, agent, new_state) # get the value of the next state to calculate bellman error & advantage
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

function full_training_procedure!(alg::ContinuousPPO, envs::Vector{E}) where {E <: AbstractEnv}
    info = Dict{Symbol, Any}(:T => alg.T, :γ => alg.γ, :advantage_coefficients => alg.advantage_coefficients)
    agents = alg.actor_learners
    results = pmap(collect_trajectory_segment_continuous!, WorkerPool(workers()), envs, agents, fill(info, alg.N))
    all_transitions, all_probabilities, all_advantages, all_targets, all_errors = unzip(results)
    train!(alg, all_transitions, all_probabilities, all_advantages, all_targets, all_errors)
end


function validation_episode!(env::E, alg::ContinuousPPO; render::Bool=false) where {E<:AbstractEnv}
    # initialise vectors to store the history of states actions and rewards for the entire episode
    return validation_episode!(ContinuousPPO, env, alg.central_agent; render=render)
end

function validation_episode!(::Type{ContinuousPPO}, env::E, agent::T; render::Bool=false) where {E<:AbstractEnv, T <: AbstractAgent}
    # initialise vectors to store the history of states actions and rewards for the entire episode
    state = reset!(env)
    term = false
    episode_reward = Vector{Float64}()
    step=0
    while env.terminal == false # bool flag to denote whether routing has finished
        # calculate the mode outputs based on the current graph
        action, value, probs = get_action(ContinuousPPO, agent, state)
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

function update_actor_learners!(alg::ContinuousPPO)
    for (idx,p) in enumerate(Flux.params(alg.central_agent.model[1]))
        for agent_idx in 1:alg.N
            Flux.params(alg.actor_learners[agent_idx].model[1])[idx] .= copy(p |> cpu)
        end
    end
    for (idx,p) in enumerate(Flux.params(alg.central_agent.model[2]))
        for agent_idx in 1:alg.N
            Flux.params(alg.actor_learners[agent_idx].model[2])[idx] .= copy(p |> cpu)
        end
    end
end

# ------------ get_actions functions ------- #
function get_action(::Type{ContinuousPPO}, agent::A, obs::O; det=false, mask=nothing) where {A <: AbstractAgent, O <: AbstractObservation}
    μ, log_σ = agent.model[1](obs)
    value = agent.model[2](obs)
    σ = exp.(log_σ)
    if det == true
        action = μ[1]
    else
        d = Normal(Float64(μ[1]), σ[1])
        action = Float32(rand(d, 1)[1])
    end
    return action, value[1], log_gauss_pdf(action, μ[1], σ[1])
end

function log_gauss_pdf(x::Float32, μ::Float32, σ::Float32=0.05f0)
    return -log(σ) - log(sqrt(2 * π))  - 0.5 * (((x - μ)/σ) ^ 2)
end

function log_gauss_pdf(x::Vector{Float32}, μ::Vector{Float32}, σ::Vector{Float32})
    return -log.(σ) .- log(sqrt(2 * π))  .- 0.5 * (((x .- μ) ./ σ) .^ 2)
end
     
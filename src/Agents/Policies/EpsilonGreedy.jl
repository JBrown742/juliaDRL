mutable struct EpsilonGreedy <: AbstractPolicy
    epsilon::Float64
end



# ------------ get_actions functions ------- #
function get_action(agent::AbstractAgent, obs::O; mask=nothing, det=false) where {O <: AbstractObservation}
    return get_action(obs, agent.model, agent.policy; mask=mask, det=det)
end 

function get_action(obs::O, model::AbstractModel, policy::EpsilonGreedy; det=false, mask=nothing) where {O <: AbstractObservation}
    outputs = model(obs)
    if isnothing(mask)
        mask = ones(length(outputs))
    end
    masked_outputs = outputs .* mask
    if det == true
        return argmax(masked_outputs)
    else
        if rand(Float64) > policy.epsilon
            action = argmax(masked_outputs)
        else
            action_indices = findall(!iszero, masked_outputs)
            action = rand(action_indices)
        end
        return action
    end
end


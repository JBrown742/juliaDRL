mutable struct StandardActorCritic <: AbstractAgent
    actor_model::FluxModel
    critic_model::FluxModel
    function StandardActorCritic(actor::FluxModel, critic::FluxModel)
        return new(actor, critic)
    end
end

mutable struct CombinedActorCritic <: AbstractAgent
    combined_model::FluxModel
    function CombinedActorCritic(m::FluxModel)
        return new(m)
    end
end

mutable struct StandardPolicy <: AbstractAgent
    model::FluxModel
end
mutable struct StandardValue <: AbstractAgent
    model::FluxModel
end

function save_agent(A::StandardActorCritic, save_dir::String; agent_info::String="")
    if !isdir(save_dir)
        mkdir(save_dir)
    end
    num_saved_agents= length(readdir(save_dir))
    actor_model = A.actor_model
    critic_model = A.critic_model
    if agent_info==""
        agent_save_dir = save_dir * "/agent_$(num_saved_agents+1)/"
    else
        agent_save_dir = save_dir * "/agent_" * agent_info * "/"
    end
    if !isdir(agent_save_dir)
        mkdir(agent_save_dir)
    end
    save_model(actor_model, agent_save_dir; model_info="actor")
    save_model(critic_model, agent_save_dir; model_info="critic")
end

function save_agent(A::CombinedActorCritic, save_dir::String; agent_info::String="")
    if !isdir(save_dir)
        mkdir(save_dir)
    end
    num_saved_agents= length(readdir(save_dir))
    combined_model = A.combined_model
    if agent_info==""
        agent_save_dir = save_dir * "/agent_$(num_saved_agents+1)/"
    else
        agent_save_dir = save_dir * "/agent_" * agent_info * "/"
    end
    if !isdir(agent_save_dir)
        mkdir(agent_save_dir)
    end
    save_model(combined_model, agent_save_dir; model_info="combined")
end

function load_agent(::Type{StandardActorCritic}, agent_dir::String)
    actor = load_model(FluxModel, agent_dir * "/actor_model.jls")
    critic = load_model(FluxModel, agent_dir * "/critic_model.jls")
    return StandardActorCritic(actor, critic)
end

function load_agent(::Type{CombinedActorCritic}, agent_dir::String)
    combined = load_model(FluxModel, agent_dir * "/combined_model.jls")
    return CombinedActorCritic(combined)
end


struct Split{T}
    paths::T
end
  
Split(paths...) = Split(paths)
  
Flux.@layer Split
  
(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)
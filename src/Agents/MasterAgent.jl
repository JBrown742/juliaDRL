abstract type AbstractModel end
abstract type AbstractAgent end

mutable struct StandardActorCritic <: AbstractAgent
    actor_model::AbstractModel
    critic_model::AbstractModel
    model_type::Type
    function StandardActorCritic(actor::AbstractModel, critic::AbstractModel)
        return new(actor, critic, typeof(actor))
    end
end

mutable struct CombinedActorCritic <: AbstractAgent
    combined_model::AbstractModel
    model_type::Type
    function CombinedActorCritic(m::AbstractModel)
        return new(m, typeof(m))
    end
end

mutable struct StandardPolicy <: AbstractAgent
    model::AbstractModel
end
mutable struct StandardValue <: AbstractAgent
    model::AbstractModel
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

function load_agent(::Type{StandardActorCritic}, ::Type{M}, agent_dir::String) where {M <: AbstractModel}
    actor = load_model(M, agent_dir * "/actor_model.bson")
    critic = load_model(M, agent_dir * "/critic_model.bson")
    return StandardActorCritic(actor, critic)
end

function load_agent(::Type{CombinedActorCritic}, ::Type{M}, agent_dir::String) where {M <: AbstractModel}
    combined = load_model(M, agent_dir * "/combined_model.bson")
    return CombinedActorCritic(combined)
end


const VectorObs = Union{Vector{Float64}, Vector{Float32}}
const MatrixObs = Union{Matrix{Float64}, Matrix{Float32}}
const ArrayObs = Union{Array{Float64}, Array{Float32}}
mutable struct GraphObs 
    features::Union{Matrix{Float64}, Matrix{Float32}}
    adjacency::Union{Matrix{Float64}, Matrix{Float32}}
end
const AbstractObservation = Union{VectorObs, MatrixObs, ArrayObs, GraphObs}

const DiscreteAct = Int
const ContinuousAct = Float32
const MultiDiscreteAct = Vector{DiscreteAct}
const MultiContinuousAct = Vector{ContinuousAct}

const AbstractAction = Union{DiscreteAct, ContinuousAct, MultiDiscreteAct, MultiContinuousAct}


struct Split{T}
    paths::T
end
  
Split(paths...) = Split(paths)
  
Flux.@layer Split
  
(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)
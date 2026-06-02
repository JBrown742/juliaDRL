module ProximalPolicy

# --- Modules ---
include("Agents.jl")
using .Agents

include("Environments.jl")
using .Environments

include("Algorithms.jl")
using .Algorithms

# --- Re-exports for a clean External API ---

# Core Types & Structures
export
    AbstractAgent,
    StandardActorCritic,
    CombinedActorCritic,
    FluxModel,
    Split

# Action Types (Multiple Dispatch keys)
export
    DiscreteAct,
    ContinuousAct,
    MultiDiscreteAct,
    MultiContinuousAct

# Environments
# We export the abstract type and the built-in benchmarks
export
    AbstractEnv,
    Cartpole,
    Pendulum,
    CarRacing,
    BipedalWalker,
    ParticleChase

# The Training API
# The user should primarily interact with 'learn'
export
    learn,
    PPO,
    get_action,
    validation_episode!,
    visualise_learning


# Persistence & Utilities
export
    save_agent,
    load_agent,
    save_model,
    load_model

end # module ProximalPolicy

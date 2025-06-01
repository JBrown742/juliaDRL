abstract type AbstractEnv end

function distribute_worker_envs(env::E) where {E <: AbstractEnv}
    key = typeof(env)
    if haskey(environments, key)
        # Return the existing environment if already initialized.
        return environments[key]
    else
        worker_env = clone(env)
        environments[key] = worker_env
        return env
    end
end
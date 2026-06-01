# 🏛️ Environment Design Protocol

Every environment in `juliaDRL` must strictly adhere to the `AbstractEnv` contract. This ensures that any model or algorithm can interact with any environment without modification.

## 📋 The AbstractEnv Contract

All environments **must** implement the following fields and methods:

### Mandatory Fields
| Field | Type | Description |
| :--- | :--- | :--- |
| `state` | `AbstractArray{Float32}` | The current observation. |
| `terminal` | `Bool` | Whether the current episode has ended. |
| `episode_step` | `Int` | Current step count in the episode. |
| `episode_length` | `Int` | Maximum allowed steps. |

### Mandatory Methods

#### 1. `reset!(env::E)::Vector{Float32}`
Resets the environment to its initial state and returns the observation.
```julia
function reset!(env::MyEnv)
    # ... logic to reset simulation ...
    env.state = initial_obs
    env.terminal = false
    env.episode_step = 0
    return env.state
end
```

#### 2. `step!(env::E, action::A)::Tuple{Vector{Float32}, Float32, Bool}`
Advances the simulation by one step. Returns `(next_state, reward, is_terminal)`.
```julia
function step!(env::MyEnv, action::MyActionType)
    # 1. Apply action
    # 2. Update env.state and env.terminal
    # 3. Increment env.episode_step
    return env.state, reward, env.terminal
end
```

#### 3. `clone(env::E)::E`
Returns a fresh instance of the environment with the same configuration. This is critical for distributed experience collection.
```julia
function clone(env::MyEnv)
    return MyEnv(env.episode_length; config=env.config)
end
```

## 🚀 Best Practices

1. **Type Homogeneity:** All tensors must be `Float32`. Convert `PyCall` outputs immediately.
2. **Zero-Allocation Inner Loops:** Avoid `cat`, `vcat`, or `reshape` inside `step!` if possible.
3. **Lazy Pre-processing:** The environment should return the "Raw" observation. Use separate wrapper structs for cropping, grayscale, or frame-stacking.
4. **Independent Randomness:** Ensure that `clone(env)` results in environments with different random seeds if they are used on different workers.

## 🛠️ Implementation Example: Pure Julia

```julia
mutable struct MyEnv <: AbstractEnv
    state::Vector{Float32}
    terminal::Bool
    episode_step::Int
    episode_length::Int
    # ... internal sim state ...

    function MyEnv(len::Int)
        return new(zeros(Float32, 4), false, 0, len)
    end
end

function reset!(env::MyEnv)
    env.state = randn(Float32, 4)
    env.terminal = false
    env.episode_step = 0
    return env.state
end

function step!(env::MyEnv, action::Int)
    env.state .+= (action == 1 ? 0.1f0 : -0.1f0)
    env.episode_step += 1
    env.terminal = env.episode_step >= env.episode_length
    return env.state, 1.0f0, env.terminal
end

function clone(env::MyEnv)
    return MyEnv(env.episode_length)
end
```

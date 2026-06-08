# ProximalPolicy: Performance-Optimized Deep Reinforcement Learning in Julia

[![Julia Version](https://img.shields.io/badge/julia-1.10+-9558B2.svg)](https://julialang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Coverage](https://codecov.io/gh/JBrown742/ProximalPolicy.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JBrown742/ProximalPolicy.jl)

A high-performance, modular library for Deep Reinforcement Learning (DRL) implemented in Julia. This package is designed for **Research Engineers** who need to bridge the gap between advanced mathematical literature and scalable, compute-efficient infrastructure.

## Key Features

- **Architectural Separation:** Complete decoupling of RL algorithms from neural network models. Seamlessly swap any Flux.jl-compatible model into the learner.
- **Multi-Dispatch Core:** Specialized implementations for Discrete, Continuous, and Multi-Continuous action spaces using Julia's powerful multiple dispatch.
- **Performance Optimized:** 
    - **O(T) GAE Backward Pass:** Linear-time Generalized Advantage Estimation for high-throughput experience collection.
    - **Zero-Copy Batching:** Optimized training loops that minimize heap allocations and garbage collection pressure.
    - **Distributed Support:** Native integration with Julia's `Distributed.jl` for parallel trajectory collection across multiple workers.
- **Reproduction Ready:** Clear scripts and DVC integration for tracking experiments and results.

## Annotated Mathematical Implementation

### Proximal Policy Optimization (PPO)
This library implements the PPO-Clip algorithm as described in [Schulman et al. (2017)](https://arxiv.org/abs/1707.06347). The core surrogate objective is implemented with numerical stability in mind:

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{clamp}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) \right]$$

### Generalized Advantage Estimation (GAE)
We leverage GAE ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438)) for variance reduction. Our implementation uses an **O(T) backward pass** to compute advantages recursively, ensuring minimal computational overhead:

$$A_t = \delta_t + (\gamma\lambda) A_{t+1}$$
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

### Beta Distribution for Constrained Action Spaces
Unlike standard Gaussian policies that require $tanh$ squashing or aggressive clipping, `ProximalPolicy` leverages the **Beta Distribution** to naturally enforce the $[-1, 1]$ action space. This avoids the "boundary bias" and gradient saturation common in Gaussian-based continuous control.

An action $u \in [0, 1]$ is sampled from a Beta distribution, and then affinely transformed to the target space:
$$u_t \sim \text{Beta}(\alpha_t, \beta_t)$$
$$a_t = 2u_t - 1$$

To ensure numerical stability and a unimodal distribution, our implementation enforces $\alpha, \beta > 1$ via a softplus transformation of the network outputs:
$$\alpha, \beta = \text{softplus}(z) + 1.0$$

## Usage Example (Cartpole)

The library provides a clean API for training agents. For details on how to add your own custom environments, see our [Environment Design Protocol](docs/Environment_Design.md).

```julia
using ProximalPolicy
using Flux

# 1. Initialize environment
env = Cartpole(200)

# 2. Define Actor-Critic architecture
combined_network = Chain(
    Dense(length(env.state), 64, relu),
    Dense(64, 64, relu),
    Split((Dense(64, 2), Dense(64, 1))) 
)
model = FluxModel(combined_network, Flux.Adam(1e-4))
agent = CombinedActorCritic(model)

# 3. Configure PPO
alg = PPO(DiscreteAct, nworkers(), 2048, 10, agent)

# 4. Start training
learn(env, alg; training_iters=100)
```

## Performance Benchmarks

A Performance-optimized implementation of PPO. Some key drivers of performance include:
1. Eliminating splatting overhead in batch preparation.
2. Hoisting matrix concatenations out of the gradient inner loop.
3. Leveraging linear-time advantage estimation.

## Verification & Integration Learning Tests (ILTs)

The package implements both unit tests (which verify code correctness) and **Integration Learning Tests (ILTs)**, which verify that the agent actually *learns*. This library includes a suite of ILTs to ensure algorithmic convergence across all supported action spaces.

| Test Script | Action Space | Environment | Success Metric |
| :--- | :--- | :--- | :--- |
| `test_scripts/Cartpole/PPO.jl` | **Discrete** | CartPole-v1 | Reward -> 500 |
| `test_scripts/Pendulum/PPO.jl` | **Continuous** | Pendulum-v1 | Reward -> -200 |
| `test_scripts/BipedalWalker/PPO.jl` | **Multi-Continuous** | BipedalWalker-v3 | Positive Reward |
| `test_scripts/ParticleChase/PPO_MultiDiscrete.jl` | **Multi-Discrete** | ParticleChase | Learning Signal |

### Running the ILT Suite
To verify the entire library, run the provided scripts:
```bash
julia --project test_scripts/Cartpole/PPO.jl
```

## Unit Testing
We are incrementally adding unit tests to verify the core mathematical kernels and Performance utilities. These tests ensure that optimizations (like the $O(T)$ GAE pass) remain mathematically sound.

To run unit tests:
```bash
julia --project test/runtests.jl
```

The intention here is to use both the unit tests and ILTs to confirm the current learning stablity of the implementation for each of the four supported action spaces. These will be run by Github Actions on each PR to ensure breaking bugs aren't introduced into the stable develop branch.

---
*Developed by Jonathon Brown -jonathonbrown742@gmail.com*

*

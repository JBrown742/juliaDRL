# juliaDRL: High-Performance Deep Reinforcement Learning in Julia

[![Julia Version](https://img.shields.io/badge/julia-1.12+-9558B2.svg)](https://julialang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance, modular library for Deep Reinforcement Learning (DRL) implemented in Julia. This package is designed for **Research Engineers** who need to bridge the gap between advanced mathematical literature and scalable, compute-efficient infrastructure.

## 🚀 Key Features

- **Architectural Separation:** Complete decoupling of RL algorithms from neural network models. Seamlessly swap any Flux.jl-compatible model into the learner.
- **Multi-Dispatch Core:** Specialized implementations for Discrete, Continuous, and Multi-Continuous action spaces using Julia's powerful multiple dispatch.
- **HPC Optimized:** 
    - **O(T) GAE Backward Pass:** Linear-time Generalized Advantage Estimation for high-throughput experience collection.
    - **Zero-Copy Batching:** Optimized training loops that minimize heap allocations and garbage collection pressure.
    - **Distributed Support:** Native integration with Julia's `Distributed.jl` for parallel trajectory collection across multiple workers.
- **Reproduction Ready:** Clear scripts and DVC integration for tracking experiments and results.

## 📚 Annotated Mathematical Implementation

### Proximal Policy Optimization (PPO)
This library implements the PPO-Clip algorithm as described in [Schulman et al. (2017)](https://arxiv.org/abs/1707.06347). The core surrogate objective is implemented with numerical stability in mind:

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{clamp}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) \right]$$

### Generalized Advantage Estimation (GAE)
We leverage GAE ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438)) for variance reduction. Our implementation uses an **O(T) backward pass** to compute advantages recursively, ensuring minimal computational overhead:

$$A_t = \delta_t + (\gamma\lambda) A_{t+1}$$
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

## 🛠️ Usage Example (Cartpole)

The library provides a clean API for training agents. Below is a snippet of how to initialize and train a PPO agent on the Cartpole environment.

```julia
using juliaDRL
using Flux

# 1. Define your environment
env = Cartpole(200)

# 2. Define your Actor-Critic model
combined_network = Chain(
    Dense(length(env.state), 64, leakyrelu),
    Dense(64, 64, leakyrelu),
    Split((Dense(64, 2), Dense(64, 1))) # Split into Action Logits and State Value
)
model = DNN(combined_network, Adam(1e-4))
agent = CombinedActorCritic(model)

# 3. Configure the PPO algorithm
alg = PPO(DiscreteAct, nworkers(), 200, 10, agent; 
          batch_size=64, γ=0.99, λ=0.95, ϵ=0.2)

# 4. Start learning
learn(env, alg; training_iters=100)
```

## 📈 Performance Benchmarks

Our HPC-optimized implementation achieves significant speedups over naive Julia implementations by:
1. Eliminating splatting overhead in batch preparation.
2. Hoisting matrix concatenations out of the gradient inner loop.
3. Leveraging linear-time advantage estimation.

## 🧪 Verification & Integration Learning Tests (ILTs)

At DeepMind, we distinguish between unit tests (which verify code correctness) and **Integration Learning Tests (ILTs)**, which verify that the agent actually *learns*. This library includes a suite of ILTs to ensure algorithmic convergence across all supported action spaces.

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

## 🛠️ Unit Testing
We are incrementally adding unit tests to verify the core mathematical kernels and HPC utilities. These tests ensure that optimizations (like the $O(T)$ GAE pass) remain mathematically sound.

To run unit tests:
```bash
julia --project test/runtests.jl
```

---
*Developed by Jonathon Brown - PhD in Theoretical Physics & ML Engineer.*

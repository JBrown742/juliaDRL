# ProximalPolicy.jl

A high-performance, modular library for Deep Reinforcement Learning (DRL) implemented in Julia.

## Introduction

ProximalPolicy is designed for research engineers who need to bridge the gap between advanced mathematical literature and scalable, compute-efficient infrastructure.

## Core Features

- **Architectural Separation:** Decoupling of RL algorithms from neural network models.
- **Multi-Dispatch Core:** Specialized implementations for Discrete, Continuous, and Multi-Continuous action spaces.
- **Performance Optimized:** Linear-time GAE, zero-copy batching, and distributed support.

## Installation

```julia
using Pkg
Pkg.add("ProximalPolicy")
```

## Quick Start

```julia
using ProximalPolicy
using Flux

env = Cartpole(200)
combined_network = Chain(
    Dense(length(env.state), 64, relu),
    Dense(64, 64, relu),
    Split((Dense(64, 2), Dense(64, 1))) 
)
model = FluxModel(combined_network, Flux.Adam(1e-4))
agent = CombinedActorCritic(model)
alg = PPO(DiscreteAct, nworkers(), 2048, 10, agent)
learn(env, alg; training_iters=100)
```

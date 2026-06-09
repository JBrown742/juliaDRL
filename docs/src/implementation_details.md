# PPO Implementation Details

This document provides a comprehensive technical overview of the Proximal Policy Optimization (PPO) implementation in `ProximalPolicy`.

## 1. Core Architecture

The implementation follows a **Centralized Training, Decentralized Execution (CTDE)** pattern adapted for performance-oriented environments.

- **Central Agent:** Resides on the master process. It holds the definitive weights and is the only agent updated via backpropagation.
- **Worker Agents:** Distributed across multiple processes (Julia workers). They are used exclusively for parallel trajectory collection.
- **Synchronization:** Worker agents are periodically synchronized with the central agent's weights (controlled by `sync_frequency`).

## 2. PPO Algorithm Components

### 2.1 The Clipped Objective
The implementation uses the standard PPO-Clip objective to ensure the policy update stays within a "trust region":
$$L^{CLIP} = \mathbb{E}[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$$

- **Clipped Value Function:** The value loss also includes a clipping term to prevent the value function from changing too drastically in a single update.
- 
### 2.2 Generalized Advantage Estimation (GAE)
Advantages are calculated using GAE($\gamma, \lambda$) in an $O(T)$ backward pass over the collected trajectory segments. This balances the bias-variance tradeoff in advantage estimation.

### 2.3 Normalization and Scaling
For stability across diverse environments, the following techniques are implemented:
- **Observation Normalization:** A `RunningStat` object maintains a running mean and variance of observations, allowing the model to process inputs at a consistent scale.
- **Reward Scaling:** A `RewardScaler` adjusts rewards based on a running estimate of return variance, preventing value function gradients from dominating policy gradients.
- **Advantage Normalization:** Advantages are normalized to zero mean and unit variance per batch before the policy update.

## 3. Action Space Handling

The implementation uses Julia's multiple dispatch to handle different action types via the `AbstractAction` hierarchy:

### 3.1 Continuous Actions (`ContinuousAct`)
- **Distribution:** Uses a Beta distribution to ensure actions are naturally in the range [-1,1].

- **Entropy:** Closed-form expression for the beta distribution entropy is used.

### 3.2 Discrete Actions (`DiscreteAct`)
- **Distribution:** Categorical distribution using logits.
- **Action Masking:** Supports additive masks to zero out invalid actions in specific states (e.g., in complex strategic games or constrained environments).

### 3.3 Multi-Discrete and Multi-Continuous
Extensions of the above logic to handle vector-valued action spaces, commonly found in multi-agent or high-degree-of-freedom environments.
THIS IS STILL EXPERIMENTAL.

## 4. Stability Features

- **NaN Shield:** Gradients are checked for `NaN` values before every update. If detected, the update is skipped to prevent model corruption. This probably shouldn't be necessary but is a measure that was born out of necessity. Having originally used a Guassian distribution for continuous actions several hacks were required to truncate the action space to [-1, 1]. 

## 5. Parallel Execution

The `get_trajectories!` function utilizes Julia's `@spawnat` and `fetch` to gather experience from workers in parallel. 

- **Environment Cloning:** Each worker reconstructs its own local version of the environment to avoid serialization issues with un-picklable objects (like Python's `gym` environments accessed via `PythonCall`).

## 6. Training API (`learn`)

The `learn` function orchestrates the entire process:
1. Parallel trajectory collection.
2. Normalizer updates.
3. Advantage calculation (GAE).
4. Optimization (SGD with shuffled batches for $K$ epochs).
5. Periodic validation and checkpointing.
6. Learning curve generation (Max/Min and Std-Dev ribbons).

# 🚀 DeepMind-Grade Stability & Convergence TODO

This roadmap outlines the surgical changes required to ensure the `juliaDRL` package meets the standards of high-performance research engineering. Implementing these will fix silent mathematical bugs, eliminate training instability, and enable convergence in < 100 iterations.

---

## 1. Mathematical Consistency (Priority: Critical)

### [ ] Fix Continuous Distribution Mismatch (Gaussian)
*   **What:** Remove `tanh` squashing from `_get_action_continuous` and ensure the training loop (`gradient_calculation_and_update!`) uses the same raw Gaussian parameters.
*   **Why:** Currently, the agent collects data using a squashed/clamped distribution but calculates gradients against an unconstrained Gaussian. Removing the squashing ensures the math is symmetrical. We will rely on environment-level clamping for now.

### [ ] Fix Multi-Discrete Head Coordination
*   **What:** Refactor `gradient_calculation_and_update!{MultiDiscreteAct}` to calculate and sum the log-probabilities across *all* action heads, not just the first one.
*   **Why:** In environments like `ParticleChase`, the agent needs to coordinate multiple axes of movement. The current implementation only optimizes the first head, leaving other heads to behave as random noise, which prevents convergence.

### [ ] Implement Standard Gaussian Entropy
*   **What:** Verify and strictly implement the closed-form entropy for Gaussians: `H = 0.5 * log(2 * π * e * σ²)`.
*   **Why:** Entropy is the primary driver of exploration in PPO. If the entropy calculation is off (or missing the constant factors), the `c2` coefficient will not effectively prevent the policy from collapsing prematurely.

---

## 2. Implementation Correctness (Priority: High)

### [ ] Proper GAE Bootstrapping & Truncation Handling
*   **What:** Distinguish between `terminated` (true end of task) and `truncated` (max steps reached). Use the Value function of the next state to bootstrap the advantage for truncated transitions, while setting it to zero only for true terminations.
*   **Why:** In `BipedalWalker`, the agent often reaches the step limit. Treating a timeout as a "death" (terminal) introduces a massive negative bias in the advantage estimation, as the agent thinks it failed rather than simply running out of time.

### [ ] Orthogonal Initialization
*   **What:** Replace `glorot_uniform` with Orthogonal Initialization for all Dense layers. Set the gain of the final policy layer to `0.01` and the final value layer to `1.0`.
*   **Why:** PPO is highly sensitive to initial weights. Orthogonal initialization ensures the initial policy is close to a random walk and gradients are well-behaved, which is a key trick for achieving fast convergence in low-iteration regimes.

---

## 3. Robustness & HPC (Priority: Medium)

### [ ] Global Gradient Norm Clipping
*   **What:** Replace the element-wise `clamp!` in the gradient update with `Flux.clip_gradient_norm!`. A standard value is `0.5`.
*   **Why:** Element-wise clamping can change the *direction* of the gradient vector. Global norm clipping preserves the direction while scaling the magnitude, which is mathematically sound and essential for preventing "catastrophic forgetting" during high-advantage updates.

### [ ] Minimize Training Loop Allocations
*   **What:** Move the state stacking/concatenation outside the inner epoch loop in `train!`. Prepare the tensors once and slice them in-place during the `K` epochs.
*   **Why:** Repeatedly allocating memory for tensors inside the training loop triggers the Julia GC frequently, slowing down training and reducing GPU utilization (HPC Standard).

### [ ] Advantage Whitening (Batch Normalization)
*   **What:** Ensure Advantages are normalized across the *entire batch* of collected trajectories before the training epochs begin, ensuring a mean of 0 and std of 1.
*   **Why:** This stabilizes the scale of the policy update regardless of the environment's reward magnitude, allowing a single set of hyperparameters to work across multiple ILTs.

---

## 4. Advanced: Beta Distribution (Priority: DeepMind Portfolio Feature)

### [ ] Implement Beta Distribution Actor
*   **What:** Create a new actor architecture that outputs $\alpha$ and $\beta$ parameters (using `softplus(x) + 1.0`) instead of $\mu$ and $\sigma$.
*   **Why:** The Beta distribution is naturally bounded on [0, 1]. By scaling to [-1, 1], we eliminate the "boundary problem" entirely. This is mathematically superior to clipped Gaussians for physical control tasks like `BipedalWalker` and shows high-level research expertise.

---

## 5. Verification Workflow
For each item above:
1.  **Implement** the change.
2.  **Run the relevant ILT** (e.g., `test_scripts/Pendulum/PPO.jl` for continuous fixes).
3.  **Confirm Convergence:** Ensure the reward curve shows clear improvement within 100 iterations.
4.  **Commit** with a descriptive message (e.g., `fix(ppo): align train/inference distribution logic`).

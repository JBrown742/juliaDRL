using Distributed
if nworkers() == 1
    addprocs(3)
end
@everywhere using ProximalPolicy
using Flux
using LinearAlgebra

# ----------------- Custom Orthogonal Initialization ------------------ #
# Standard PPO practice: ensures initial gradients are stable and well-behaved.
function orthogonal_init(out, in; gain=1.0)
    W = randn(Float32, out, in)
    U, S, V = svd(W)
    return Float32.(U * V' .* gain)
end

# ----------------- BipedalWalker ILT (Aggressive) ------------------ # 
env = BipedalWalker(2000)

# 256x256 with Orthogonal Init and tanh for locomotion
actor_network = Chain(
    Dense(length(env.state), 256, tanh; init=(out, in) -> orthogonal_init(out, in, gain=sqrt(2))),
    Dense(256, 256, tanh; init=(out, in) -> orthogonal_init(out, in, gain=sqrt(2))),
    Split((
        Dense(256, 4; init=(out, in) -> orthogonal_init(out, in, gain=0.01)), # Alpha
        Dense(256, 4; init=(out, in) -> orthogonal_init(out, in, gain=0.01))  # Beta
    ))
)
actor_model = FluxModel(actor_network, Flux.Optimisers.Adam(3e-4))

critic_network = Chain(
    Dense(length(env.state), 256, leakyrelu; init=(out, in) -> orthogonal_init(out, in, gain=sqrt(2))),
    Dense(256, 256, leakyrelu; init=(out, in) -> orthogonal_init(out, in, gain=sqrt(2))),
    Dense(256, 1; init=(out, in) -> orthogonal_init(out, in, gain=1.0))
)
critic_model = FluxModel(critic_network, Flux.Optimisers.Adam(3e-4))

agent = StandardActorCritic(actor_model, critic_model)

# Aggressive Convergence settings (T=1024, K=10, batch=64)
# Increased K and reduced T to update more frequently and squeeze more info from data.
# c2 reduced to 0.005 to refine gait patterns discovered in earlier exploration.
alg = PPO(MultiContinuousAct, nworkers(), 1024, 10, agent; 
          batch_size=64, γ=0.99, λ=0.95, ϵ=0.2, c2=0.005)

# ILT: Convergence should be visible within 50-80 iterations.
learn(env, alg; training_iters=200, 
      save_dir="/home/johnny/Documents/PersonalCode/ProximalPolicy/example_scripts/save_data/BipedalWalker", 
      test_name="BipedalWalker_test_1")

# ----------------- Visualisation ------------------ # 
# To watch the trained agent, uncomment the line below:
visualise_learning(alg, env, "/home/johnny/Documents/PersonalCode/ProximalPolicy/example_scripts/save_data/BipedalWalker/BipedalWalker_test_1")

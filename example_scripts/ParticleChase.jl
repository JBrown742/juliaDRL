using Distributed
if nworkers() == 1
    addprocs(3)
end
@everywhere using juliaDRL
using Flux
using LinearAlgebra
using Random

# ----------------- Custom Orthogonal Initialization ------------------ #
function orthogonal_init(out, in; gain=1.0)
    W = randn(Float32, out, in)
    U, S, V = svd(W)
    return Float32.(U * V' .* gain)
end

# ----------------- ParticleChase MultiDiscrete ILT ------------------ # 
# Goal: reach target in 2D space using discrete (X, Y) controls.
env = ParticleChase(200, 2; max_speed=2.0f0)

# ARCHITECTURE: 
# Input: distance vector (length 2)
# Output: 2 heads of size 3 (Total 6 units). 
# Heads represent movement in [-1, 0, 1] for each axis.
actor_network = Chain(
    Dense(2, 64, relu; init=(out, in) -> orthogonal_init(out, in, gain=sqrt(2))),
    Dense(64, 64, relu; init=(out, in) -> orthogonal_init(out, in, gain=sqrt(2))),
    Dense(64, 6; init=(out, in) -> orthogonal_init(out, in, gain=0.01))
)
actor_model = FluxModel(actor_network, Flux.Optimisers.Adam(1e-3)) # Higher LR for simple task

critic_network = Chain(
    Dense(2, 64, relu; init=(out, in) -> orthogonal_init(out, in, gain=sqrt(2))),
    Dense(64, 64, relu; init=(out, in) -> orthogonal_init(out, in, gain=sqrt(2))),
    Dense(64, 1; init=(out, in) -> orthogonal_init(out, in, gain=1.0))
)
critic_model = FluxModel(critic_network, Flux.Optimisers.Adam(1e-3))

agent = StandardActorCritic(actor_model, critic_model)

# HYPERPARAMETERS: Balanced for MultiDiscrete
# T=512 is sufficient for this simple 200-step task.
alg = PPO(MultiDiscreteAct, nworkers(), 512, 10, agent; 
          batch_size=64, γ=0.99, λ=0.95, ϵ=0.2, c2=0.01)

# ILT: Convergence should be visible within 30-50 iterations.
learn(env, alg; training_iters=100, 
      save_dir="/home/johnny/Documents/PersonalCode/juliaDRL/test_data/ParticleChase", 
      test_name="ILT_ParticleChase_MultiDiscrete")

# ----------------- Visualisation ------------------ # 
# To watch the trained agent, uncomment the line below:
# visualise_learning(alg, env, "/home/johnny/Documents/PersonalCode/juliaDRL/test_data/ParticleChase/ILT_ParticleChase_MultiDiscrete")

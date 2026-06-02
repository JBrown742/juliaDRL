using Distributed
if nworkers() == 1
    addprocs(3)
end
@everywhere using ProximalPolicy
using Flux

# ----------------- Pendulum ILT (Continuous) ------------------ # 
env = Pendulum(200)

# Use tanh for activation in hidden layers for better control stability
actor_network = Chain(
    Dense(length(env.state), 64, tanh; init=Flux.glorot_uniform),
    Dense(64, 64, tanh; init=Flux.glorot_uniform),
    # Output: raw Alpha and raw Beta (will be transformed via softplus + 1 internally)
    Split((Dense(64, 1; init=Flux.glorot_uniform), Dense(64, 1; init=Flux.glorot_uniform)))
)
actor_model = FluxModel(actor_network, Flux.Optimisers.Adam(3e-4))

critic_network = Chain(
    Dense(length(env.state), 64, tanh; init=Flux.glorot_uniform),
    Dense(64, 64, tanh; init=Flux.glorot_uniform),
    Dense(64, 1; init=Flux.glorot_uniform)
)
critic_model = FluxModel(critic_network, Flux.Optimisers.Adam(1e-3))

agent = StandardActorCritic(actor_model, critic_model)

# Configuration for Beta PPO
# Added c2=0.001 for entropy bonus to aid exploration with Beta distribution
alg = PPO(ContinuousAct, nworkers(), 2048, 10, agent; 
          batch_size=64, γ=0.99, λ=0.95, ϵ=0.2, c2=0.001)

# ILT: Run for 50 iterations to verify reward signal improvement
learn(env, alg; training_iters=50, 
      save_dir="/home/johnny/Documents/PersonalCode/ProximalPolicy/test_data/Pendulum", 
      test_name="ILT_Pendulum")

# ----------------- Visualisation ------------------ # 
# To watch the trained agent, uncomment the line below:
# visualise_learning(alg, env, "/home/johnny/Documents/PersonalCode/ProximalPolicy/test_data/Pendulum/ILT_Pendulum")

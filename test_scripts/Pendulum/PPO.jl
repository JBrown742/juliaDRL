using Distributed
if nworkers() == 1
    addprocs(3)
end
@everywhere using juliaDRL
using Flux

# ----------------- Pendulum ILT (Continuous) ------------------ # 
env = Pendulum(200)

# Use tanh for activation in hidden layers for better control stability
actor_network = Chain(
    Dense(length(env.state), 64, tanh; init=Flux.glorot_uniform),
    Dense(64, 64, tanh; init=Flux.glorot_uniform),
    # Output: mean (mu) and log_std (log_sigma)
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

# Configuration for Continuous PPO
alg = PPO(ContinuousAct, nworkers(), 2048, 10, agent; 
          batch_size=64, γ=0.99, λ=0.95, ϵ=0.2, c2=0.0)

# ILT: Run for 50 iterations to verify reward signal improvement
learn(env, alg; training_iters=50, 
      save_dir="/home/johnny/Documents/PersonalCode/juliaDRL/test_data/Pendulum", 
      test_name="ILT_Pendulum")

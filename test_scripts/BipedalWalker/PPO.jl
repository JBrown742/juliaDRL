using Distributed
if nworkers() == 1
    addprocs(3)
end
@everywhere using juliaDRL
using Flux

# ----------------- BipedalWalker ILT (Hardened) ------------------ # 
env = BipedalWalker(2000)

# DeepMind Standard: Deep networks with tanh for locomotion stability
actor_network = Chain(
    Dense(length(env.state), 256, tanh; init=Flux.glorot_uniform),
    Dense(256, 256, tanh; init=Flux.glorot_uniform),
    Split((Dense(256, 4; init=Flux.glorot_uniform), Dense(256, 4; init=Flux.glorot_uniform)))
)
# Smaller learning rate for locomotion stability
actor_model = FluxModel(actor_network, Flux.Optimisers.Adam(1e-4))

critic_network = Chain(
    Dense(length(env.state), 256, relu; init=Flux.glorot_uniform),
    Dense(256, 256, relu; init=Flux.glorot_uniform),
    Dense(256, 1; init=Flux.glorot_uniform)
)
critic_model = FluxModel(critic_network, Flux.Optimisers.Adam(1e-4))

agent = StandardActorCritic(actor_model, critic_model)

# Hardened Hyperparameters: Large T, smaller batch, lower entropy (c2)
alg = PPO(MultiContinuousAct, nworkers(), 2048, 10, agent; 
          batch_size=128, γ=0.99, λ=0.95, ϵ=0.2, c2=0.001)

# ILT: Run for 200 iterations to verify conclusive convergence
learn(env, alg; training_iters=200, 
      save_dir="/home/johnny/Documents/PersonalCode/juliaDRL/test_data/BipedalWalker", 
      test_name="ILT_BipedalWalker_Hardened")

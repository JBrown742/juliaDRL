using Distributed
if nworkers() == 1
    addprocs(3)
end
@everywhere using juliaDRL
using Flux

# ----------------- ParticleChase ILT (Multi-Discrete) ------------------ # 
env = ParticleChase(200, 2)

# MultiDiscrete: 2 independent heads (X and Y), each with 3 actions (-1, 0, 1)
# The model outputs a flat vector of 6 logits.
actor_network = Chain(
    Dense(length(env.state), 64, relu; init=Flux.glorot_uniform),
    Dense(64, 64, relu; init=Flux.glorot_uniform),
    Dense(64, 6; init=Flux.glorot_uniform) 
)
actor_model = FluxModel(actor_network, Flux.Optimisers.Adam(1e-3))

critic_network = Chain(
    Dense(length(env.state), 64, relu; init=Flux.glorot_uniform),
    Dense(64, 1; init=Flux.glorot_uniform)
)
critic_model = FluxModel(critic_network, Flux.Optimisers.Adam(1e-3))

agent = StandardActorCritic(actor_model, critic_model)

alg = PPO(MultiDiscreteAct, nworkers(), 256, 10, agent; 
          batch_size=64, γ=0.99, λ=0.95, ϵ=0.2, c2=0.01)

# ILT: Run for 100 iterations to verify multi-discrete head coordination
learn(env, alg; training_iters=100, 
      save_dir="/home/johnny/Documents/PersonalCode/juliaDRL/test_data/ParticleChase", 
      test_name="ILT_MultiDiscrete")

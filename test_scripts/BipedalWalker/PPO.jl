using Distributed
if nworkers() == 1
  addprocs(8)
end
using Revise
@everywhere using juliaDRL
# using juliaDRL
using Flux
workers()
CLIP_THRESHOLD= 0.5

env = BipedalWalker(200, stuck_threshold=20)

actor_network = Chain(
Dense(length(env.observation_highs), 64, tanh; init=Flux.glorot_normal), 
  Dense(64, 64, tanh; init=Flux.glorot_normal),   
  Dense(64, 64, tanh; init=Flux.glorot_normal),
  Split(Dense(64, 4, sigmoid;init=Flux.glorot_normal), Dense(64, 4;init=Flux.glorot_normal))
)         # Output layer (for 10 classes)
base_actor_optim = Flux.Optimisers.Adam(1e-3)
actor_optim = OptimiserChain(ClipNorm(CLIP_THRESHOLD), base_actor_optim)
actor_model = DNN(actor_network, actor_optim)

critic_network = Chain(
  Dense(length(env.observation_highs), 64, tanh; init=Flux.glorot_normal), 
  Dense(64, 64, tanh; init=Flux.glorot_normal),  
  Dense(64, 64, tanh; init=Flux.glorot_normal),    
  Dense(64, 1;init=Flux.glorot_normal)
)     # Output layer (for 10 classes)
base_critic_optim = Flux.Optimisers.Adam(1e-3)
critic_optim = OptimiserChain(ClipNorm(CLIP_THRESHOLD), base_critic_optim)
critic_model = DNN(critic_network, critic_optim)


agent = StandardActorCritic(actor_model, critic_model)
alg = PPO(MultiContinuousAct, nworkers(), 256, 10, agent; batch_size=64, γ=0.999, λ=0.95, ϵ=0.2, c1=1., c2=0.001, sync_frequency=1)

## ----------------------------------------------------------------------------------------------------
## Combined Actor critic stuff

# combined_network = Chain(
# Dense(length(env.observation_highs), 64, leakyrelu; init=Flux.orthogonal), 
#   Dense(64, 64, leakyrelu; init=Flux.orthogonal),   
#   Split(Dense(64, 4, leakyrelu;init=Flux.orthogonal), Dense(64, 4, leakyrelu;init=Flux.orthogonal), Dense(64, 1; init=Flux.glorot_normal))
# )      # Output layer (for 10 classes)
# combined_optim = Flux.Optimisers.Adam(1e-4)
# combined_model = DNN(combined_network, combined_optim)

# agent = CombinedActorCritic(combined_model)

# alg = PPO(MultiContinuousAct, nworkers(), 256, nworkers(), agent, 256, 0.99, 0.95, 0.15, 1., 0.002, 1)

learn(env, alg; training_iters=500, checkpoint_freq=10, save_dir="/home/johnny/Documents/PersonalCode/juliaDRL/test_data/BipedalWalker/", test_name="PPO_StandardActorCritic_1")
env.episode_length += 4 * 256 
alg = PPO(MultiContinuousAct, nworkers(), 2048, 10, agent; batch_size=128, γ=0.999, λ=0.95, ϵ=0.2, c1=1., c2=0.001, sync_frequency=1)

learn(env, alg; training_iters=500, checkpoint_freq=10, save_dir="/home/johnny/Documents/PersonalCode/juliaDRL/test_data/BipedalWalker/", test_name="PPO_StandardActorCritic_2")



using Distributed
if nworkers() == 1
  addprocs(2)
end
using Revise
@everywhere using juliaDRL
# using juliaDRL
using Flux
workers()

env = BipedalWalker(2000, stuck_threshold=20)

actor_network = Chain(
Dense(length(env.observation_highs), 64, tanh; init=Flux.glorot_normal), 
  Dense(64, 64, tanh; init=Flux.glorot_normal),   
  Dense(64, 64, tanh; init=Flux.glorot_normal),
  Split(Dense(64, 4, tanh;init=Flux.glorot_normal), Dense(64, 4, tanh;init=Flux.glorot_normal))
)         # Output layer (for 10 classes)
actor_optim = Flux.Optimisers.Adam(1e-4)
actor_model = DNN(actor_network, actor_optim)

critic_network = Chain(
Dense(length(env.observation_highs), 64, tanh; init=Flux.glorot_normal), 
  Dense(64, 64, tanh; init=Flux.glorot_normal),  
  Dense(64, 64, tanh; init=Flux.glorot_normal),    
  Dense(64, 1;init=Flux.glorot_normal))                  # Output layer (for 10 classes)                        # Output layer (for 10 classes)
critic_optim = Flux.Optimisers.Adam(eta=1e-4)
critic_model = DNN(critic_network, critic_optim)
agent = StandardActorCritic(actor_model, critic_model)


agent = StandardActorCritic(actor_model, critic_model)
alg = PPO(MultiContinuousAct, nworkers(), 2048, 10, agent, 64, 0.999, 0.95, 0.18, 0.1, 0.001, 1)

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

learn(env, alg; training_iters=500, checkpoint_freq=10, save_dir="/home/johnny/Documents/PersonalCode/juliaDRL/test_data/BipedalWalker/", test_name="PPO_StandardActorCritic_1", random_policy=true)
learn(env, alg; training_iters=500, checkpoint_freq=10, save_dir="/home/johnny/Documents/PersonalCode/juliaDRL/test_data/BipedalWalker/", test_name="PPO_StandardActorCritic_2")
env.episode_length += 50
learn(env, alg; training_iters=200, checkpoint_freq=10, save_dir="/home/johnny/Documents/PersonalCode/juliaDRL/test_data/BipedalWalker/", test_name="PPO_StandardActorCritic_3")
env.episode_length += 50
learn(env, alg; training_iters=200, checkpoint_freq=10, save_dir="/home/johnny/Documents/PersonalCode/juliaDRL/test_data/BipedalWalker/", test_name="PPO_StandardActorCritic_4")
env.episode_length += 50
learn(env, alg; training_iters=200, checkpoint_freq=10, save_dir="/home/johnny/Documents/PersonalCode/juliaDRL/test_data/BipedalWalker/", test_name="PPO_StandardActorCritic_5")
env.episode_length += 50
learn(env, alg; training_iters=200, checkpoint_freq=10, save_dir="/home/johnny/Documents/PersonalCode/juliaDRL/test_data/BipedalWalker/", test_name="PPO_StandardActorCritic_6")
env.episode_length += 50
learn(env, alg; training_iters=200, checkpoint_freq=10, save_dir="/home/johnny/Documents/PersonalCode/juliaDRL/test_data/BipedalWalker/", test_name="PPO_StandardActorCritic_7")
env.episode_length += 50
learn(env, alg; training_iters=200, checkpoint_freq=10, save_dir="/home/johnny/Documents/PersonalCode/juliaDRL/test_data/BipedalWalker/", test_name="PPO_StandardActorCritic_8")
env.episode_length += 50

step!(env, Float32[0.790377, 0.8312113, -0.8979085, -0.93469393])




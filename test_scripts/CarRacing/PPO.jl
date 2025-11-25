using Distributed
if nworkers() == 1
  addprocs(6)
end

@everywhere using Revise
@everywhere using juliaDRL
using Flux
using Profile
using Plots

CLIP_THRESHOLD=0.5
env = CarRacing(100)
actor_network = Chain(
  Conv((6, 4), 4 => 8, stride=3, leakyrelu), # First convolution layer                       # First max pooling layer
  Conv((3, 3), 8 => 16, stride=2, leakyrelu), # Second convolution layer  
  Conv((3, 3), 16 => 16, stride=1, leakyrelu),                   # Second max pooling layer
  Flux.flatten,          # Flatten the tensor
  Dense(7 * 4 * 16, 128 , relu; init=Flux.glorot_uniform),                # First dense layer                 # Second dense layer
  Split((Dense(128, 3, sigmoid;init=Flux.glorot_uniform), Dense(128, 3, tanh;init=Flux.glorot_uniform)))                          # Output layer (for 10 classes)
)

base_actor_optim = Flux.Optimisers.Adam(1e-3)
actor_optim = OptimiserChain(ClipNorm(CLIP_THRESHOLD), base_actor_optim)
actor_model = CNN(actor_network, actor_optim)

# if image nxn, kernal k and stride s
# new dims = ((n - k) / s) + 1
critic_network = Chain(
  Conv((6, 4), 4 => 8, stride=3, leakyrelu), # First convolution layer                       # First max pooling layer
  Conv((3, 3), 8 => 16, stride=2, leakyrelu), # Second convolution layer  
  Conv((3, 3), 16 => 16, stride=1, leakyrelu),                   # Second max pooling layer
  Flux.flatten,          # Flatten the tensor
  Dense(7 * 4 * 16, 128 , relu; init=Flux.glorot_uniform),             # First dense layer                 # Second dense layer
  Dense(128, 1;init=Flux.glorot_uniform)   
)                      # Output layer (for 10 classes)                        # Output layer (for 10 classes)
base_critic_optim = Flux.Optimisers.Adam(1e-3)
critic_optim = OptimiserChain(ClipNorm(CLIP_THRESHOLD), base_critic_optim)
critic_model = CNN(critic_network, critic_optim)




agent = StandardActorCritic(actor_model, critic_model)
alg = PPO(MultiContinuousAct, nworkers(), 256, 10, agent; batch_size=32, γ=0.99, λ=0.95, ϵ=0.2, c1=1., c2=0.001, sync_frequency=1)

learn(env, alg; training_iters=500, checkpoint_freq=10, save_dir="/home/johnny/Documents/PersonalCode/juliaDRL/test_data/CarRacing/", test_name="PPO_StandardActorCritic_1")

visualise_learning(alg, vizenv, "juliaDRL/test_data/CarRacing/PPODemoTest2")

a = load_agent(typeof(agent), typeof(agent.actor_model), "/home/johnny/Documents/PersonalCode/juliaDRL/test_data/CarRacing/PPODemoTest2/checkpointed_agents/agent_iter_150")

vizenv = CarRacing(200, render=true)
validation_episode!(alg, vizenv, agent)

close!(vizenv)
close!(env)
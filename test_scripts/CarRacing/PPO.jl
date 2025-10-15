using Distributed
if nworkers() == 1
  addprocs(10)
end

@everywhere using Revise
@everywhere using juliaDRL
using Flux
using StatsBase
using PyCall
using Profile
using ProfileView
using Plots

env = CarRacing(1000)
actor_network = Chain(
  Conv((6, 6), 3 => 4, stride=3, leakyrelu), # First convolution layer                       # First max pooling layer
  Conv((3, 3), 4 => 8, stride=2, leakyrelu), # Second convolution layer  
  Conv((2, 2), 8 => 16, stride=2, leakyrelu),                   # Second max pooling layer
  Flux.flatten,          # Flatten the tensor
  Dense(7 * 7 * 16, 256 , relu; init=Flux.glorot_uniform),             # First dense layer                 # Second dense layer
  Split((Dense(256, 3, sigmoid;init=Flux.glorot_uniform), Dense(256, 3, tanh;init=Flux.glorot_uniform)))                          # Output layer (for 10 classes)
)
actor_optim = Flux.Optimisers.Adam(1e-3)
actor_model = CNN(actor_network, actor_optim)

# if image nxn, kernal k and stride s
# new dims = ((n - k) / s) + 1
critic_network = Chain(
  Conv((6, 6), 3 => 4, stride=3, leakyrelu), # First convolution layer                       # First max pooling layer
  Conv((3, 3), 4 => 8, stride=2, leakyrelu), # Second convolution layer  
  Conv((2, 2), 8 => 16, stride=2, leakyrelu),                   # Second max pooling layer
  Flux.flatten,          # Flatten the tensor
  Dense(7 * 7 * 16, 256 , relu; init=Flux.glorot_uniform),                 # Second dense layer
  Dense(256, 1;init=Flux.glorot_uniform))                          # Output layer (for 10 classes)                        # Output layer (for 10 classes)
critic_optim = Flux.Optimisers.Adam(1e-3)
critic_model = CNN(critic_network, critic_optim)


agent = StandardActorCritic(actor_model, critic_model)
alg = PPO(MultiContinuousAct, nworkers(), 256, nworkers(), agent, 256, 0.99, 0.95, 0.2, 1., 0.001, 3)

nworkers()

learn(env, alg; training_iters=10000, checkpoint_freq=5, test_name="test_data/CarRacing/PPODemoTest2")

visualise_learning(alg, vizenv, "juliaDRL/test_data/CarRacing/PPODemoTest2")

a = load_agent(typeof(agent), typeof(agent.actor_model), "/home/johnny/Documents/PersonalCode/juliaDRL/test_data/CarRacing/PPODemoTest2/checkpointed_agents/agent_iter_150")

vizenv = CarRacing(200, render=true)
validation_episode!(alg, vizenv, agent)

close!(vizenv)
close!(env)


const gym = PyNULL()
copy!(gym,  pyimport("gymnasium"))
env = gym.make("BipedalWalker-v3", hardcore=true, render_mode="human")

env.unwrapped.action_space
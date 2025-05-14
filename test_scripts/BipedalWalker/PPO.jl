using Revise
using juliaDRL
using Flux
using StatsBase
using PyCall
using Profile
using ProfileView
using Plots

env = BipedalWalker(400)
actor_network = Chain(
  Dense(length(env.observation_highs), 256, leakyrelu; init=Flux.glorot_uniform),         
  Dense(256, 256, leakyrelu; init=Flux.glorot_uniform),   
  Split(Dense(256, 4, sigmoid;init=Flux.glorot_uniform), Dense(256, 4, tanh;init=Flux.glorot_uniform))
)             # Output layer (for 10 classes)
actor_optim = Flux.Optimisers.Adam(1e-3)
actor_model = DNN(actor_network, actor_optim)

critic_network = Chain(
  Dense(length(env.observation_highs), 256, leakyrelu; init=Flux.glorot_uniform),         
  Dense(256, 256, leakyrelu; init=Flux.glorot_uniform),       
  Dense(256, 1;init=Flux.glorot_uniform))                          # Output layer (for 10 classes)                        # Output layer (for 10 classes)
critic_optim = Flux.Optimisers.Adam(1e-3)
critic_model = DNN(critic_network, critic_optim)


agent = StandardActorCritic(actor_model, critic_model)
alg = PPO(MultiContinuousAct, 16, 512, 4, agent, 32, 0.999, 0.95, 0.1, 1., 0.02, 5)


learn(env, alg; training_iters=1000, checkpoint_freq=5, test_name="juliaDRL/test_data/BipedalWalker/PPODemoTest1")


vizenv = BipedalWalker(200, render=true)
visualise_learning(alg, vizenv, "juliaDRL/test_data/BipedalWalker/PPODemoTest1")

a = load_agent(typeof(agent), typeof(agent.actor_model), "/home/johnny/Documents/PersonalCode/juliaDRL/test_data/CarRacing/PPODemoTest2/checkpointed_agents/agent_iter_150")

vizenv = BipedalWalker(2000, render=true)
validation_episode!(alg, vizenv, agent)

close!(vizenv)
close!(env)

agent.critic_model(vizenv.state)
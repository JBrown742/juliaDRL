using Revise
using juliaDRL
using Flux
using StatsBase
using PyCall
using Profile
using ProfileView
using Plots
function invrelu(x)
  return min(x, 0)
end


env = BipedalWalker(1600)
actor_network = Chain(
  Dense(length(env.observation_highs), 128, tanh; init=Flux.orthogonal),         
  Dense(128, 128, tanh; init=Flux.orthogonal),   
  Dense(128, 4, tanh;init=Flux.orthogonal)
)             # Output layer (for 10 classes)
actor_optim = Flux.Optimisers.Adam(1e-4)
actor_model = DNN(actor_network, actor_optim)

critic_network = Chain(
  Dense(length(env.observation_highs), 128, tanh; init=Flux.orthogonal),         
  Dense(128, 128, tanh; init=Flux.orthogonal),   
  Dense(128, 1;init=Flux.orthogonal))                          # Output layer (for 10 classes)                        # Output layer (for 10 classes)
critic_optim = Flux.Optimisers.Adam(eta=2e-4)
critic_model = DNN(critic_network, critic_optim)


agent = StandardActorCritic(actor_model, critic_model)
alg = PPO(MultiContinuousAct, 5, 2048, 10, agent, 128, 0.99, 0.95, 0.18, 1., 0., 3)


learn(env, alg; training_iters=10000, checkpoint_freq=5, test_name="juliaDRL/test_data/BipedalWalker/PPODemoTest3")


vizenv = BipedalWalker(2000, render=true)
visualise_learning(alg, vizenv, "juliaDRL/test_data/BipedalWalker/PPODemoTest2")

a = load_agent(typeof(agent), typeof(agent.actor_model), "/home/johnny/Documents/PersonalCode/juliaDRL/test_data/CarRacing/PPODemoTest2/agent_best")

vizenv = BipedalWalker(2000, render=true)
validation_episode!(alg, vizenv, agent)

close!(vizenv)
close!(env)

agent.critic_model(vizenv.state)
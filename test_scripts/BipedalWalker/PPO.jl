using Distributed
if nworkers() == 1
  addprocs(12)
end

@everywhere using Revise
@everywhere using juliaDRL
using Flux
workers()

env = BipedalWalker(500)

env.observation_highs
# actor_network = Chain(
# Dense(length(env.observation_highs), 64, leakyrelu; init=Flux.orthogonal), 
#   Dense(64, 64, leakyrelu; init=Flux.orthogonal),   
#   Split(Dense(64, 4, tanh;init=Flux.orthogonal), Dense(64, 4, tanh;init=Flux.orthogonal))
# )         # Output layer (for 10 classes)
# actor_optim = Flux.Optimisers.Adam(1e-4)
# actor_model = DNN(actor_network, actor_optim)

# critic_network = Chain(
# Dense(length(env.observation_highs), 64, leakyrelu; init=Flux.orthogonal), 
#   Dense(64, 128, leakyrelu; init=Flux.orthogonal),   
#   Dense(128, 1;init=Flux.orthogonal))                  # Output layer (for 10 classes)                        # Output layer (for 10 classes)
# critic_optim = Flux.Optimisers.Adam(eta=1e-4)
# critic_model = DNN(critic_network, critic_optim)
# agent = StandardActorCritic(actor_model, critic_model)


# agent = StandardActorCritic(actor_model, critic_model)
# alg = PPO(MultiContinuousAct, nworkers(), 256, nworkers(), agent, 256, 0.99, 0.95, 0.15, 1., 0.002, 1)


## ----------------------------------------------------------------------------------------------------
## Combined Actor critic stuff

combined_network = Chain(
Dense(length(env.observation_highs), 64, leakyrelu; init=Flux.orthogonal), 
  Dense(64, 64, leakyrelu; init=Flux.orthogonal),   
  Split(Dense(64, 4, tanh;init=Flux.orthogonal), Dense(64, 4, tanh;init=Flux.orthogonal), Dense(64, 1; init=Flux.glorot_normal))
)      # Output layer (for 10 classes)
combined_optim = Flux.Optimisers.Adam(1e-4)
combined_model = DNN(combined_network, combined_optim)

agent = CombinedActorCritic(combined_model)

alg = PPO(MultiContinuousAct, nworkers(), 256, nworkers(), agent, 256, 0.99, 0.95, 0.15, 1., 0.002, 1)

learn(env, alg; training_iters=5000, checkpoint_freq=10, test_name="juliaDRL/test_data/BipedalWalker/PPO_CombinedActorCritic_1")


x = cu(actor_model)

typeof(x) == typeof(actor_model)

old = Flux.params(agent.actor_model.model)
new = Flux.trainable(agent.actor_model.model)
x = Flux.trainable(agent.actor_model.model).layers[3]
Flux.trainable(agent.actor_model.model).layers[1].weight .= rand(Float32, (64,24))


typeof(x)

Flux.trainable(agent.actor_model.model).layers[1].weight == x






















actiontypevizenv = BipedalWalker(2000, render=true)
visualise_learning(alg, vizenv, "juliaDRL/test_data/BipedalWalker/PPODemoTestNew2")

a = load_agent(StandardActorCritic, DNN, "/home/johnny/Documents/PersonalCode/juliaDRL/test_data/BipedalWalker/PPODemoTestNew3/agent_best")
alg = PPO(MultiContinuousAct, nworkers(), 2048, 10, a, 64, 0.99, 0.95, 0.18, 1., 0., 3)

vizenv = BipedalWalker(2000, render=true)
validation_episode!(alg, vizenv, a)

close!(vizenv)
close!(env)

agent.critic_model(vizenv.state)
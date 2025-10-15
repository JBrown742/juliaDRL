using Distributed
if nworkers() == 1
    addprocs(10)
end

@everywhere using juliaDRL
using Flux
## ----------------- Standard actor-critic part ------------------ ## 
env = Pendulum(200)

actor_network = Chain(
    Dense(length(env.state), 64, leakyrelu; init=Flux.glorot_uniform),
    Dense(64, 64, leakyrelu; init=Flux.glorot_uniform),
    Split((Dense(64, 1, tanh;init=Flux.glorot_uniform), Dense(64, 1, tanh;init=Flux.glorot_uniform)))
)
actor_optim = Flux.Optimisers.Adam(1e-3)
actor_model = DNN(actor_network, actor_optim)

critic_network = Chain(
    Dense(length(env.state), 64, leakyrelu; init=Flux.glorot_uniform),
    Dense(64, 64, leakyrelu; init=Flux.glorot_uniform),
    Dense(64, 1; init=Flux.glorot_uniform)
)
critic_optim = Flux.Optimisers.Adam(2e-3)
critic_model = DNN(critic_network, critic_optim)

agent = StandardActorCritic(actor_model, critic_model)

## ------------------------------- Parameter Sharing part ------------------------ ##
#
# Here gradient interference and scaling disparity between the actor heads and the value head
# present a challenge. Can tackled using reward normalisation and adding complexiy to the value head.
#

# combined_network = Chain(
#     Dense(length(env.state), 256, leakyrelu; init=Flux.glorot_uniform),
#     Dense(256, 64, leakyrelu; init=Flux.glorot_uniform),
#     Split((Dense(64, 1;init=Flux.glorot_uniform), Dense(64, 1, tanh;init=Flux.glorot_uniform), Chain(Dense(64, 64, leakyrelu;init=Flux.glorot_uniform), Dense(64, 1;init=Flux.glorot_uniform))))
# )
# combined_optim = Flux.Optimisers.Adam(1e-3)
# combined_model = DNN(combined_network, combined_optim)


# agent = CombinedActorCritic(combined_model)
alg = PPO(ContinuousAct, nworkers(), 256, nworkers(), agent, 256, 0.99, 0.95, 0.2, 1., 0.002, 5)

learn(env, alg; training_iters=500, save_dir="/home/johnny/Documents/PersonalCode/juliaDRL/test_data/Pendulum", test_name="PPO_CombinedActorCritic_1")
close!(env)


vizenv = Pendulum(500; render=true)
visualise_learning(alg,  vizenv, "/home/johnny/Documents/PersonalCode/juliaDRL/DemoTest2")
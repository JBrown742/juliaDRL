using juliaDRL
using Flux

env = Cartpole(200)

# -------------------- standard agent ----------------------- #

actor_network = Chain(
    Dense(length(env.state), 64, leakyrelu; init=Flux.glorot_uniform),
    Dense(64, 64, leakyrelu; init=Flux.glorot_uniform),
    Dense(64, 2;init=Flux.glorot_uniform)
)
actor_optim = Flux.Optimisers.Adam(1e-3)
actor_model = DNN(actor_network, actor_optim)

critic_network = Chain(
    Dense(length(env.state), 64, leakyrelu; init=Flux.glorot_uniform),
    Dense(64, 64, leakyrelu; init=Flux.glorot_uniform),
    Dense(64, 1; init=Flux.glorot_uniform)
)
critic_optim = Flux.Optimisers.Adam(1e-3)
critic_model = DNN(critic_network, critic_optim)

agent = StandardActorCritic(actor_model, critic_model)
alg = PPO(DiscreteAct, 4, 512, 10, agent, 32, 0.9, 0.95, 0.2, 0.7, 0.0, 5)

#-------------------- Parameter shared agent -------------------- #
# combined_network = Chain(
#     Dense(length(env.state), 128, leakyrelu; init=Flux.glorot_uniform),
#     Dense(128, 256, leakyrelu; init=Flux.glorot_uniform),
#     Dense(256, 128, leakyrelu; init=Flux.glorot_uniform),using Revise
#     Split((Dense(128, 1;init=Flux.glorot_uniform), Dense(128, 1, tanh;init=Flux.glorot_uniform), Dense(128, 1;init=Flux.glorot_uniform)))
# )
# combined_optim = Flux.Optimisers.Adam(1e-3)
# combined_model = DNN(combined_network, combined_optim)


# agent = CombinedActorCritic(combined_model)

# ------------------------------------------------------- #
# alg = ContinuousPPO(4, 512, 10, agent, 128, 0.9, 0.95, 0.2, 0.6, 0.0, 5)
learn(env, alg; training_iters=100, test_name="PPOTest")
close!(env)
fill(deepcopy(env), alg.N)

typeof(alg) <: PPO

vizenv = Cartpole(200, render=true)
visualise_learning(alg, vizenv, "/home/johnny/Documents/PersonalCode/PPOTest")
close!(env)
using Distributed
if nworkers() == 1
    addprocs(6)
end
@everywhere using juliaDRL
using Flux
## ----------------- Standard actor-critic part ------------------ ## 
env = Pendulum(200)

CLIP_THRESHOLD= 1f0

actor_network = Chain(
    Dense(length(env.state), 64, tanh; init=Flux.glorot_uniform),
    Dense(64, 64, tanh; init=Flux.glorot_uniform),
    Split((Dense(64, 1;init=Flux.glorot_uniform), Dense(64, 1, tanh;init=Flux.glorot_uniform)))
)
base_actor_optim = Flux.Optimisers.Adam(1e-3)
actor_optim = OptimiserChain(ClipNorm(CLIP_THRESHOLD), base_actor_optim)
actor_model = DNN(actor_network, actor_optim)

critic_network = Chain(
    Dense(length(env.state), 64, tanh; init=Flux.glorot_uniform),
    Dense(64, 64, tanh; init=Flux.glorot_uniform),
    Dense(64, 1; init=Flux.glorot_uniform)
)
base_critic_optim = Flux.Optimisers.Adam(1e-3)
critic_optim = OptimiserChain(ClipNorm(CLIP_THRESHOLD), base_critic_optim)
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
alg = PPO(ContinuousAct, nworkers(), 256, 10, agent; batch_size=64, γ=0.9, λ=0.95, ϵ=0.2, c1=0.5, c2=0.0001, sync_frequency=1)

learn(env, alg; training_iters=100, save_dir="/home/johnny/Documents/PersonalCode/juliaDRL/test_data/Pendulum", test_name="PPO_CombinedActorCritic_1")

# close!(env)


vizenv = Pendulum(500; render=true)
visualise_learning(alg,  vizenv, "/home/johnny/Documents/PersonalCode/juliaDRL/DemoTest2")
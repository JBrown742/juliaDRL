using Distributed
if nworkers() == 1
  addprocs(10)
end

using Revise
using juliaDRL
using Flux
using Profile
using Plots

CLIP_THRESHOLD = 0.5

env = ParticleChase(200, 2; hard=true, max_speed=1f0)
# ----------------  Uncomment for separate Actor and critic networks. --------------------------- # 
actor_network = Chain(
  Dense(length(env.state), 64, leakyrelu; init=Flux.orthogonal),         
  Dense(64, 64, leakyrelu; init=Flux.orthogonal),   
  Split(Dense(64, env.dims, tanh;init=Flux.orthogonal), Dense(64, env.dims, tanh;init=Flux.orthogonal))
)             # Output layer (for 10 classes)
base_actor_optim = Flux.Optimisers.Adam(1e-3)
actor_optim = OptimiserChain(ClipNorm(CLIP_THRESHOLD), base_actor_optim)
actor_model = DNN(actor_network, actor_optim)

critic_network = Chain(
  Dense(length(env.state), 64, leakyrelu; init=Flux.orthogonal),         
  Dense(64, 64, leakyrelu; init=Flux.orthogonal),   
  Dense(64, 1;init=Flux.orthogonal)
)                                  # Output layer (for 10 classes)                        # Output layer (for 10 classes)
base_critic_optim = Flux.Optimisers.Adam(1e-3)
critic_optim = OptimiserChain(ClipNorm(CLIP_THRESHOLD), base_critic_optim)
critic_model = DNN(critic_network, critic_optim)


agent = StandardActorCritic(actor_model, critic_model)

# combined_network = Chain(
#   Dense(length(env.state), 64, tanh; init=Flux.glorot_uniform),         
#   Dense(64, 64, tanh; init=Flux.glorot_uniform),   
#   Split(Dense(64, env.dims, tanh;init=Flux.glorot_uniform), Dense(64, env.dims, inverse_relu;init=Flux.glorot_uniform), Dense(64, 1;init=Flux.glorot_uniform))
# )             # Output layer (for 10 classes)
# combined_optim = Flux.Optimisers.Adam(1e-3)
# combined_model = DNN(combined_network, combined_optim)

# agent = CombinedActorCritic(combined_model)


alg = PPO(MultiContinuousAct, nworkers(), 2048, 10, agent; batch_size=128, γ=0.99, λ=0.95, ϵ=0.2, c1=1., c2=0.01, sync_frequency=1)


learn(env, alg; training_iters=500, checkpoint_freq=10, save_dir="/home/johnny/Documents/PersonalCode/juliaDRL/test_data/ParticleChase/", test_name="PPO_StandardActorCritic_1")


plot(alg.clipfrac)
plot(alg.average_policy_loss)
plot(alg.average_value_loss)
plot(alg.approxkl)
vizenv = ParticleChase(200, 2; render=true, hard=true)
visualise_learning(alg, vizenv, "juliaDRL/test_data/ParticleChase/PPODemoTest1")



vizenv = ParticleChase(200, env.dims; render=true)
validation_episode!(alg, vizenv, agent; render=true)

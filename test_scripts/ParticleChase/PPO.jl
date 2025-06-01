using Revise
using juliaDRL
using Flux
using StatsBase
using PyCall
using Profile
using ProfileView
using Plots

function inverse_relu(x)
  return min(x, 0)
end


env = ParticleChase(200, 2; hard=true, max_speed=3f0)
# ----------------  Uncomment for separate Actor and critic networks. --------------------------- # 
actor_network = Chain(
  Dense(length(env.state), 64, leakyrelu; init=Flux.orthogonal),         
  Dense(64, 64, leakyrelu; init=Flux.orthogonal),   
  Split(Dense(64, env.dims, tanh;init=Flux.orthogonal), Dense(64, env.dims, inverse_relu;init=Flux.orthogonal))
)             # Output layer (for 10 classes)
actor_optim = Flux.Optimisers.Adam(1e-4)
actor_model = DNN(actor_network, actor_optim)

critic_network = Chain(
  Dense(length(env.state), 64, leakyrelu; init=Flux.orthogonal),         
  Dense(64, 64, leakyrelu; init=Flux.orthogonal),   
  Dense(64, 1;init=Flux.orthogonal)
)                                  # Output layer (for 10 classes)                        # Output layer (for 10 classes)
critic_optim = Flux.Optimisers.Adam(1e-4)
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


alg = PPO(MultiContinuousAct, 10, 1024, 5, agent, 128, 0.99, 0.95, 0.2, 1., 0.01, 4)


learn(env, alg; training_iters=1000, checkpoint_freq=5, test_name="juliaDRL/test_data/ParticleChase/PPODemoTest1")


plot(alg.clipfrac)
plot(alg.average_policy_loss)
plot(alg.average_value_loss)
plot(alg.approxkl)
vizenv = ParticleChase(200, 2; render=true, hard=true)
visualise_learning(alg, vizenv, "juliaDRL/test_data/ParticleChase/PPODemoTest1")



vizenv = ParticleChase(200, env.dims; render=true)
validation_episode!(alg, vizenv, agent; render=true)

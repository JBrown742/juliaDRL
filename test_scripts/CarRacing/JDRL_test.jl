using Revise
using juliaDRL
using Flux
using StatsBase

# env = Pendulum(200)

model = Chain(
  Conv((3, 3), 3 => 32, pad=(1, 1), relu), # First convolution layer
  MaxPool((2, 2)),                        # First max pooling layer
  Conv((3, 3), 32 => 64, pad=(1, 1), relu), # Second convolution layer
  MaxPool((2, 2)),                        # Second max pooling layer
  Conv((3, 3), 64 => 128, pad=(1, 1), relu),# Third convolution layer
  MaxPool((2, 2)),                        # Third max pooling layer
  x -> reshape(x, :, size(x, 4)),          # Flatten the tensor
  Dense(12*12*128, 256, relu),             # First dense layer
  Dense(256, 128, relu),                   # Second dense layer
  Dense(128, 10),                          # Output layer (for 10 classes)
  softmax                                 # Apply softmax to the outputs
)
# actor_optim = Flux.Optimisers.Adam(1e-3)
# actor_model = DNN(actor_network, actor_optim)

# critic_network = Chain(
#     Dense(length(env.state), 64, leakyrelu; init=Flux.glorot_uniform),
#     Dense(64, 64, leakyrelu; init=Flux.glorot_uniform),
#     Dense(64, 1; init=Flux.glorot_uniform)
# )
# critic_optim = Flux.Optimisers.Adam(1e-3)
# critic_model = DNN(critic_network, critic_optim)

# agent = StandardActorCritic(actor_model, critic_model)
# alg = ContinuousPPO(4, 1024, 10, agent, 128, 0.9, 0.95, 0.2, 0.5, 0.0, 5)


env = Pendulum(200)

combined_network = Chain(
    Dense(length(env.state), 128, leakyrelu; init=Flux.glorot_uniform),
    Dense(128, 256, leakyrelu; init=Flux.glorot_uniform),
    Dense(256, 128, leakyrelu; init=Flux.glorot_uniform),
    Split((Dense(128, 1;init=Flux.glorot_uniform), Dense(128, 1, tanh;init=Flux.glorot_uniform), Dense(128, 1;init=Flux.glorot_uniform)))
)
combined_optim = Flux.Optimisers.Adam(1e-3)
combined_model = DNN(combined_network, combined_optim)


agent = CombinedActorCritic(combined_model)
alg = ContinuousPPO(4, 512, 10, agent, 128, 0.9, 0.95, 0.2, 0.6, 0.0, 5)

learn(env, alg; training_iters=1000, test_name="PPOContinuousTest2")
close!(env)
fill(deepcopy(env), alg.N)

vizenv = Pendulum(200, render=true)
R = validation_episode!(ContinuousPPO, vizenv, alg.central_agent)
close!(env)
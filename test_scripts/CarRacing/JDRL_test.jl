using Revise
using juliaDRL
using Flux
using StatsBase
using PyCall
using Profile
using ProfileView

env = CarRacing(200)


actor_network = Chain(
  Conv((3, 3), 3 => 4, stride=3, relu), # First convolution layer
  MaxPool((2, 2)),                        # First max pooling layer
  Conv((4, 4), 4 => 8, stride=4, relu), # Second convolution layer
  MaxPool((2, 2)),                        # Second max pooling layer
  x -> reshape(x, :, size(x, 4)),          # Flatten the tensor
  Dense(2 * 2 * 8, 64, relu; init=Flux.glorot_uniform),             # First dense layer
  Dense(64, 32, relu; init=Flux.glorot_uniform),                   # Second dense layer
  Split((Dense(32, 3, tanh;init=Flux.glorot_uniform), Dense(32, 3, tanh;init=Flux.glorot_uniform)))                          # Output layer (for 10 classes)
)
actor_optim = Flux.Optimisers.Adam(1e-3)
actor_model = CNN(actor_network, actor_optim)


critic_network = Chain(
  Conv((3, 3), 3 => 4, stride=3, relu), # First convolution layer
  MaxPool((2, 2)),                        # First max pooling layer
  Conv((4, 4), 4 => 8, stride=4, relu), # Second convolution layer
  MaxPool((2, 2)),                        # Second max pooling layer
  x -> reshape(x, :, size(x, 4)),          # Flatten the tensor
  Dense(2 * 2 * 8, 64, relu; init=Flux.glorot_uniform),             # First dense layer
  Dense(64, 32, relu; init=Flux.glorot_uniform),                   # Second dense layer
  Dense(32, 1, tanh;init=Flux.glorot_uniform)
)                          # Output layer (for 10 classes)
critic_optim = Flux.Optimisers.Adam(1e-3)
critic_model = CNN(critic_network, critic_optim)


agent = StandardActorCritic(actor_model, critic_model)
alg = PPO(MultiContinuousAct, 4, 200, 3, agent, 16, 0.99, 0.95, 0.2, 1., 0.0, 5)

learn(env, alg; training_iters=100, checkpoint_freq=5, test_name="PPOMultiContinuousCarRacingTest")

vizenv = CarRacing(200, render=true)
visualise_learning(alg, vizenv, "/home/johnny/Documents/PersonalCode/PPOMultiContinuousCarRacingTest")
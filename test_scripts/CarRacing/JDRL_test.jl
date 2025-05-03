using Revise
using juliaDRL
using Flux
using StatsBase
using PyCall

env = CarRacing(200)


actor_network = Chain(
  Conv((3, 3), 3 => 32, pad=(1, 1), relu), # First convolution layer
  MaxPool((2, 2)),                        # First max pooling layer
  Conv((3, 3), 32 => 64, pad=(1, 1), relu), # Second convolution layer
  MaxPool((2, 2)),                        # Second max pooling layer
  Conv((3, 3), 64 => 128, pad=(1, 1), relu),# Third convolution layer
  MaxPool((2, 2)),                        # Third max pooling layer
  x -> reshape(x, :, size(x, 4)),          # Flatten the tensor
  Dense(12*12*128, 256, relu; init=Flux.glorot_uniform),             # First dense layer
  Dense(256, 128, relu; init=Flux.glorot_uniform),                   # Second dense layer
  Split((Dense(128, 3, tanh;init=Flux.glorot_uniform), Dense(128, 3, tanh;init=Flux.glorot_uniform)))                          # Output layer (for 10 classes)
)
actor_optim = Flux.Optimisers.Adam(1e-3)
actor_model = CNN(actor_network, actor_optim)


critic_network = Chain(
  Conv((3, 3), 3 => 32, pad=(1, 1), relu), # First convolution layer
  MaxPool((2, 2)),                        # First max pooling layer
  Conv((3, 3), 32 => 64, pad=(1, 1), relu), # Second convolution layer
  MaxPool((2, 2)),                        # Second max pooling layer
  Conv((3, 3), 64 => 128, pad=(1, 1), relu),# Third convolution layer
  MaxPool((2, 2)),                        # Third max pooling layer
  x -> reshape(x, :, size(x, 4)),          # Flatten the tensor
  Dense(12*12*128, 256, relu; init=Flux.glorot_uniform),             # First dense layer
  Dense(256, 128, relu; init=Flux.glorot_uniform),                   # Second dense layer
  Dense(128, 1; init=Flux.glorot_uniform)                         # Output layer (for 10 classes)
)
critic_optim = Flux.Optimisers.Adam(1e-3)
critic_model = CNN(critic_network, critic_optim)


agent = StandardActorCritic(actor_model, critic_model)
alg = PPO(MultiContinuousAct, 1, 256, 1, agent, 256, 0.99, 0.95, 0.2, 1., 0.0, 5)
show(IOContext(stdout, :limit=>false), subtypes(Any))
env.pyenv.car
repr(PyCall.inspect[:getmembers](env.pyenv))
renderize!(env)
obs = reset!(env)
for i in 1:10000
  actions, values, probs = juliaDRL.Algorithms.get_action(typeof(alg), agent, obs)
  obs, reward, done = step!(env, Float32.([0,1,0]))
end





learn(env, alg; training_iters=1000, test_name="PPOMultiContinuousCarRacingTest")
close!(env)
fill(deepcopy(env), alg.N)

vizenv = CarRacing(200, render=true)
R = validation_episode!(ContinuousPPO, vizenv, alg.central_agent)
close!(env)
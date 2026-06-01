using Distributed
if nworkers() == 1
  addprocs(6)
end

@everywhere using Revise
@everywhere using juliaDRL
using Flux
using Profile
using Plots

CLIP_THRESHOLD=0.5
env = CarRacing(100)
actor_network = Chain(
  Conv((6, 4), 4 => 8, stride=3, leakyrelu), # First convolution layer                       
  Conv((3, 3), 8 => 16, stride=2, leakyrelu), # Second convolution layer  
  Conv((3, 3), 16 => 16, stride=1, leakyrelu),  # Second max pooling layer
  Flux.flatten, # Flatten the tensor
  Dense(7 * 4 * 16, 128 , relu; init=Flux.glorot_uniform),                # First dense layer                 
  Split((Dense(128, 3, sigmoid;init=Flux.glorot_uniform), Dense(128, 3, tanh;init=Flux.glorot_uniform)))                          
)

base_actor_optim = Flux.Optimisers.Adam(1e-3)
actor_optim = OptimiserChain(ClipNorm(CLIP_THRESHOLD), base_actor_optim) # Using Clipnorm within the model should be a modelling consideration
actor_model = CNN(actor_network, actor_optim)

# if image nxn, kernal k and stride s
# new dims = ((n - k) / s) + 1
critic_network = Chain(
  Conv((6, 4), 4 => 8, stride=3, leakyrelu), # First convolution layer 
  Conv((3, 3), 8 => 16, stride=2, leakyrelu), # Second convolution layer  
  Conv((3, 3), 16 => 16, stride=1, leakyrelu),                  
  Flux.flatten,          # Flatten the tensor
  Dense(7 * 4 * 16, 128 , relu; init=Flux.glorot_uniform),
  Dense(128, 1;init=Flux.glorot_uniform)   
)                                              
base_critic_optim = Flux.Optimisers.Adam(1e-3)
critic_optim = OptimiserChain(ClipNorm(CLIP_THRESHOLD), base_critic_optim)
critic_model = CNN(critic_network, critic_optim)




agent = StandardActorCritic(actor_model, critic_model)
alg = PPO(MultiContinuousAct, nworkers(), 256, 10, agent; batch_size=32, γ=0.99, λ=0.95, ϵ=0.2, c1=1., c2=0.001, sync_frequency=1)

learn(env, alg; training_iters=500, checkpoint_freq=10, save_dir="/home/johnny/Documents/PersonalCode/juliaDRL/example_scripts/save_data/CarRacing/", test_name="PPO_StandardActorCritic_1")

# ----------------- Visualisation ------------------ # 
# To watch the trained agent, uncomment the line below:
visualise_learning(alg, env, "/home/johnny/Documents/PersonalCode/juliaDRL/example_scripts/save_data/CarRacing/PPO_StandardActorCritic_1")

close!(env)
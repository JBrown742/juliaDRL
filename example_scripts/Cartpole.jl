using Distributed
if nworkers() == 1
    addprocs(3)
end
@everywhere using ProximalPolicy
using Flux
using Flux.Optimisers

CLIP_THRESHOLD = 0.5
MODE="combined"

nworkers()
env = Cartpole(200)
if MODE=="standard"

    # -------------------- standard agent ----------------------- #

    actor_network = Chain(
        Dense(length(env.state), 32, leakyrelu; init=Flux.glorot_uniform),
        Dense(32, 64, leakyrelu; init=Flux.glorot_uniform),
        Dense(64, 2;init=Flux.glorot_uniform)
    )
    actor_optim = Flux.Optimisers.Adam(1e-4)
    actor_model = DNN(actor_network, actor_optim)

    critic_network = Chain(
        Dense(length(env.state), 32, leakyrelu; init=Flux.glorot_uniform),
        Dense(32, 64, leakyrelu; init=Flux.glorot_uniform),
        Dense(64, 1; init=Flux.glorot_uniform)
    )
    critic_optim = Flux.Optimisers.Adam(1e-4)
    critic_model = DNN(critic_network, critic_optim)

    agent = StandardActorCritic(actor_model, critic_model)
    alg = PPO(DiscreteAct, nworkers(), 128, nworkers(), agent, 128, 0.9, 0.95, 0.15, 1., 0.001, 3)
else
    #-------------------- Parameter shared agent -------------------- #
    combined_network = Chain(
        Dense(length(env.state), 64, leakyrelu; init=Flux.glorot_uniform),
        Dense(64, 64, leakyrelu; init=Flux.glorot_uniform),
        Dense(64, 64, leakyrelu; init=Flux.glorot_uniform),
        Split((Dense(64, 2;init=Flux.glorot_uniform), Dense(64, 1;init=Flux.glorot_uniform)))
    )
    base_optim = Flux.Optimisers.Adam(1e-4)
    combined_optim = OptimiserChain(ClipNorm(CLIP_THRESHOLD), base_optim)
    combined_model = FluxModel(combined_network, combined_optim)


    agent = CombinedActorCritic(combined_model)
    alg = PPO(DiscreteAct, nworkers(), 200, 10, agent; batch_size=64, γ=0.99, λ=0.95, ϵ=0.2, c1=0.5, c2=0.01, sync_frequency=1)
end
# ------------------------------------------------------- #
learn(env, alg; training_iters=100, save_dir="/home/johnny/Documents/PersonalCode/ProximalPolicy/example_scripts/save_data/Cartpole", test_name="PPO_CombinedActorCritic_2", average_window=10)

# ----------------- Visualisation ------------------ # 
# To watch the trained agent, uncomment the line below:


visualise_learning(alg, env, "/home/johnny/Documents/PersonalCode/ProximalPolicy/example_scripts/save_data/Cartpole/PPO_CombinedActorCritic_2"; agent_type=CombinedActorCritic)

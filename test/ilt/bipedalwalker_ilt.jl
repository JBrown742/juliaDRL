using Distributed
if nworkers() == 1
    addprocs(3)
end

@everywhere using ProximalPolicy
include("utils.jl")
@everywhere include("utils.jl")
@everywhere set_seed(42 + myid())

using Flux
using LinearAlgebra
using Test

@testset "ILT: BipedalWalker-v3 (Multi-Continuous)" begin
    # 1. Setup
    # The main env seed
    env = BipedalWalker(2000; seed=42)

    # Architectures with Orthogonal Init
    function orthogonal_init(out, in; gain=1.0)
        W = randn(Float32, out, in)
        U, S, V = svd(W)
        return Float32.(U * V' .* gain)
    end

    actor_network = Chain(
        Dense(length(env.state), 256, tanh; init=(out, in) -> orthogonal_init(out, in, gain=sqrt(2))),
        Dense(256, 256, tanh; init=(out, in) -> orthogonal_init(out, in, gain=sqrt(2))),
        Split((
            Dense(256, 4; init=(out, in) -> orthogonal_init(out, in, gain=0.01)),
            Dense(256, 4; init=(out, in) -> orthogonal_init(out, in, gain=0.01))
        ))
    )
    actor_model = FluxModel(actor_network, Flux.Optimisers.Adam(3e-4))

    critic_network = Chain(
        Dense(length(env.state), 256, relu; init=(out, in) -> orthogonal_init(out, in, gain=sqrt(2))),
        Dense(256, 256, relu; init=(out, in) -> orthogonal_init(out, in, gain=sqrt(2))),
        Dense(256, 1; init=(out, in) -> orthogonal_init(out, in, gain=1.0))
    )
    critic_model = FluxModel(critic_network, Flux.Optimisers.Adam(3e-4))

    agent = StandardActorCritic(actor_model, critic_model)

    # Algorithm: Conservative settings for CI stability
    # We use T=1024 and K=4 to keep it fast enough for a PR check.
    alg = PPO(MultiContinuousAct, nworkers(), 1024, 4, agent; 
              batch_size=128, γ=0.99, λ=0.95, ϵ=0.2, c2=0.01)

    # Run Learning
    # 50 iterations is enough to see the agent stop falling immediately.
    rewards = learn(env, alg; training_iters=50,  ephemeral=true)

    # verification (Lenient)
    # Random flailing or falling results in ~ -110. 
    # Learning to stand or shuffle forward slightly gets us to > -80.
    # Success here means the implementation is stable enough to allow improvement.
    @test verify_ilt(rewards, -50.0, window=5)
    
end

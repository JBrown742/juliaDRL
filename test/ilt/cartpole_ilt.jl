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

@testset "ILT: CartPole-v1 (Discrete)" begin
    # 1. Setup
    env = Cartpole(500; seed=42)

    # 2. Architectures with Orthogonal Init
    function orthogonal_init(out, in; gain=1.0)
        W = randn(Float32, out, in)
        U, S, V = svd(W)
        return Float32.(U * V' .* gain)
    end

    actor_network = Chain(
        Dense(4, 64, relu; init=(out, in) -> orthogonal_init(out, in, gain=sqrt(2))),
        Dense(64, 64, relu; init=(out, in) -> orthogonal_init(out, in, gain=sqrt(2))),
        Dense(64, 2; init=(out, in) -> orthogonal_init(out, in, gain=0.01))
    )
    actor_model = FluxModel(actor_network, Flux.Optimisers.Adam(1e-3))

    critic_network = Chain(
        Dense(4, 64, relu; init=(out, in) -> orthogonal_init(out, in, gain=sqrt(2))),
        Dense(64, 64, relu; init=(out, in) -> orthogonal_init(out, in, gain=sqrt(2))),
        Dense(64, 1; init=(out, in) -> orthogonal_init(out, in, gain=1.0))
    )
    critic_model = FluxModel(critic_network, Flux.Optimisers.Adam(1e-3))

    agent = StandardActorCritic(actor_model, critic_model)

    # 3. Algorithm
    alg = PPO(DiscreteAct, nworkers(), 512, 10, agent; 
              batch_size=64, γ=0.99, λ=0.95, ϵ=0.2, c2=0.01)

    # 4. Run Learning
    rewards = learn(env, alg; training_iters=30, ephemeral=true)

    # 5. Verification
    # Cartpole-v1 is solved at 475. 
    @test verify_ilt(rewards, 300.0, window=3)
end

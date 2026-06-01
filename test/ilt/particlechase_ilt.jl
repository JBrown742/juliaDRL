using Distributed
if nworkers() == 1
    addprocs(3)
end

@everywhere using juliaDRL
using Flux
using LinearAlgebra
using Test

include("utils.jl")

@testset "ILT: ParticleChase (Multi-Discrete)" begin
    # 1. Setup & Seeding
    set_seed(42)
    env = ParticleChase(200, 2; max_speed=2.0f0)

    # 2. Architectures with Orthogonal Init
    function orthogonal_init(out, in; gain=1.0)
        W = randn(Float32, out, in)
        U, S, V = svd(W)
        return Float32.(U * V' .* gain)
    end

    actor_network = Chain(
        Dense(2, 64, relu; init=(out, in) -> orthogonal_init(out, in, gain=sqrt(2))),
        Dense(64, 64, relu; init=(out, in) -> orthogonal_init(out, in, gain=sqrt(2))),
        Dense(64, 6; init=(out, in) -> orthogonal_init(out, in, gain=0.01))
    )
    actor_model = FluxModel(actor_network, Flux.Optimisers.Adam(1e-3))

    critic_network = Chain(
        Dense(2, 64, relu; init=(out, in) -> orthogonal_init(out, in, gain=sqrt(2))),
        Dense(64, 64, relu; init=(out, in) -> orthogonal_init(out, in, gain=sqrt(2))),
        Dense(64, 1; init=(out, in) -> orthogonal_init(out, in, gain=1.0))
    )
    critic_model = FluxModel(critic_network, Flux.Optimisers.Adam(1e-3))

    agent = StandardActorCritic(actor_model, critic_model)

    # 3. Algorithm
    alg = PPO(MultiDiscreteAct, nworkers(), 512, 10, agent; 
              batch_size=64, γ=0.99, λ=0.95, ϵ=0.2, c2=0.01)

    # 4. Run Learning
    rewards = learn(env, alg; training_iters=100, test_name="ILT_ParticleChase")

    # 5. Verification
    # ParticleChase is solved when distance to target is small.
    # Positive reward means it's making progress.
    @test verify_ilt(rewards, 5.0, window=3)
end

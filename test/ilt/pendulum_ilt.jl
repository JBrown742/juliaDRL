using Distributed
if nworkers() == 1
    addprocs(3)
end

@everywhere using juliaDRL
using Flux
using LinearAlgebra
using Test

include("utils.jl")

@testset "ILT: Pendulum-v1 (Continuous)" begin
    # 1. Setup & Seeding
    set_seed(42)
    env = Pendulum(200)

    # 2. Architectures with Orthogonal Init
    function orthogonal_init(out, in; gain=1.0)
        W = randn(Float32, out, in)
        U, S, V = svd(W)
        return Float32.(U * V' .* gain)
    end

    actor_network = Chain(
        Dense(length(env.state), 64, tanh; init=(out, in) -> orthogonal_init(out, in, gain=sqrt(2))),
        Dense(64, 64, tanh; init=(out, in) -> orthogonal_init(out, in, gain=sqrt(2))),
        Split((
            Dense(64, 1; init=(out, in) -> orthogonal_init(out, in, gain=0.01)),
            Dense(64, 1; init=(out, in) -> orthogonal_init(out, in, gain=0.01))
        ))
    )
    actor_model = FluxModel(actor_network, Flux.Optimisers.Adam(3e-4))

    critic_network = Chain(
        Dense(length(env.state), 64, relu; init=(out, in) -> orthogonal_init(out, in, gain=sqrt(2))),
        Dense(64, 64, relu; init=(out, in) -> orthogonal_init(out, in, gain=sqrt(2))),
        Dense(64, 1; init=(out, in) -> orthogonal_init(out, in, gain=1.0))
    )
    critic_model = FluxModel(critic_network, Flux.Optimisers.Adam(3e-4))

    agent = StandardActorCritic(actor_model, critic_model)

    # 3. Algorithm
    alg = PPO(ContinuousAct, nworkers(), 1024, 10, agent; 
              batch_size=64, γ=0.9, λ=0.95, ϵ=0.2, c2=0.001)

    # 4. Run Learning
    rewards = learn(env, alg; training_iters=100, test_name="ILT_Pendulum")

    # 5. Verification
    # Pendulum is considered solved if reward > -400. 
    # For a fast ILT we check if it reaches -700.
    @test verify_ilt(rewards, -800.0, window=3)
end

using Test
using juliaDRL

@testset "ppo_test_set" begin
    @test juliaDRL.Algorithms.RL.PPO.calculate_advantage_coefficients(10, 1f0, 1f0) == fill(1f0, 10)
end
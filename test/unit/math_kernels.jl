using Test
using juliaDRL
using Statistics

@testset "Mathematical Kernels" begin

    @testset "Log-Gaussian PDF" begin
        # The probability density of N(0, 1) at x=0 is 1/sqrt(2pi)
        # log(1/sqrt(2pi)) approx -0.9189
        μ = 0.0f0
        σ = 1.0f0
        x = 0.0f0
        expected = -0.9189385f0
        @test juliaDRL.Algorithms._get_action_continuous !== nothing # Ensure internal functions are accessible
        
        # Access the kernel through the exported module structure
        # (Assuming it's available in Algorithms)
        res = juliaDRL.Algorithms.log_gauss_pdf(x, μ, σ)
        @test res ≈ expected atol=1f-5
        
        # Test vectorization
        xs = [0.0f0, 1.0f0]
        res_vec = juliaDRL.Algorithms.log_gauss_pdf(xs, [μ, μ], [σ, σ])
        @test length(res_vec) == 2
        @test res_vec[1] ≈ expected atol=1f-5
    end

    @testset "GAE Advantage Coefficients" begin
        T = 5
        γ = 0.9f0
        λ = 0.95f0
        # Formula: (gamma * lambda) .^ (0:T)
        coeffs = juliaDRL.Algorithms.calculate_advantage_coefficients(T, γ, λ)
        
        @test length(coeffs) == T + 1
        @test coeffs[1] == 1.0f0 # (gamma * lambda)^0
        @test coeffs[2] ≈ γ * λ
        @test coeffs[end] ≈ (γ * λ)^T
    end

end

using Test
using ProximalPolicy
using Statistics
using Distributions

@testset "Mathematical Kernels" begin

    @testset "Beta Distribution Kernels" begin
        # Test Beta Log-PDF
        α = 2.0f0
        β = 5.0f0
        u = 0.3f0
        
        # Reference value from Distributions.jl
        d = Beta(α, β)
        expected_lp = logpdf(d, u)
        
        # Test our closed-form implementation
        res_lp = ProximalPolicy.Algorithms.beta_logpdf(α, β, u)
        @test res_lp ≈ expected_lp atol=1f-5
        
        # Test vectorization
        us = [0.1f0, 0.5f0, 0.9f0]
        res_vec = ProximalPolicy.Algorithms.beta_logpdf([α, α, α], [β, β, β], us)
        @test length(res_vec) == 3
        @test res_vec ≈ logpdf.(d, us) atol=1f-5
        
        # Test Beta Entropy
        # logbeta(α, β) - (α - 1)digamma(α) - (β - 1)digamma(β) + (α + β - 2)digamma(α + β)
        expected_ent = entropy(d)
        res_ent = ProximalPolicy.Algorithms.beta_entropy(α, β)
        @test res_ent ≈ expected_ent atol=1f-5
    end

    @testset "GAE Advantage Coefficients" begin
        T = 5
        γ = 0.9f0
        λ = 0.95f0
        # Formula: (gamma * lambda) .^ (0:T)
        coeffs = ProximalPolicy.Algorithms.calculate_advantage_coefficients(T, γ, λ)
        
        @test length(coeffs) == T + 1
        @test coeffs[1] == 1.0f0 # (gamma * lambda)^0
        @test coeffs[2] ≈ γ * λ
        @test coeffs[end] ≈ (γ * λ)^T
    end

end

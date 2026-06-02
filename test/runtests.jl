using Test
using ProximalPolicy

@testset "ProximalPolicy Unit Tests" begin
    include("unit/math_kernels.jl")
    include("unit/hpc_utils.jl")
end

using Test
using juliaDRL

@testset "juliaDRL Unit Tests" begin
    include("unit/math_kernels.jl")
    include("unit/hpc_utils.jl")
end

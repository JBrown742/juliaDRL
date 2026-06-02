using Test

@testset "Integration Learning Tests (ILTs)" begin
    include("ilt/pendulum_ilt.jl")
    include("ilt/cartpole_ilt.jl")
    include("ilt/particlechase_ilt.jl")
    include("ilt/bipedalwalker_ilt.jl")
end

using Distributed
using Random
using ProximalPolicy
using Test
using Statistics
using PyCall

"""
    set_seed(seed::Int)

Sets seeds for Julia, Random, and any other relevant libraries (like Python via PyCall) 
to ensure reproducibility in ILTs.
"""
function set_seed(seed::Int)
    # 1. Julia Seed
    Random.seed!(seed)

    # 2. Python Global Seeds
    pyimport("random").seed(seed)
    pyimport("numpy.random").seed(seed)

    # Note: gymnasium environment seeds are handled in reset! or constructors
    # println("Seeding PID $(getpid()) (myid $(Distributed.myid())) with seed $seed")
end

"""
    verify_ilt(rewards, threshold; window=5)

Checks if the average reward in a `window` of iterations meets the `threshold`.
"""
function verify_ilt(rewards, threshold; window=5)
    running_averages = Vector{Float32}()
    for i in 1:length(rewards)-window+1
        push!(running_averages, mean(rewards[i:i+window-1]))
    end
    if length(rewards) < window
        return mean(rewards) >= threshold
    end
    return any(running_averages .>= threshold)
end

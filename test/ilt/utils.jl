using Random
using juliaDRL
using Test
using Statistics

"""
    set_seed(seed::Int)

Sets seeds for Julia, Random, and any other relevant libraries (like Python via PyCall) 
to ensure reproducibility in ILTs.
"""
function set_seed(seed::Int)
    Random.seed!(seed)
    # If using PyCall/Gym, we might need to set seeds there too, 
    # but that's usually done on env.reset(seed=seed).
end

"""
    verify_ilt(rewards, threshold; window=5)

Checks if the average reward in the last `window` iterations meets the `threshold`.
"""
function verify_ilt(rewards, threshold; window=5)
    if length(rewards) < window
        return mean(rewards) >= threshold
    end
    return mean(rewards[end-window+1:end]) >= threshold
end

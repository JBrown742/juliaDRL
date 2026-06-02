using Random
using ProximalPolicy
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

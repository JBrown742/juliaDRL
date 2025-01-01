

"""
    buffer_pop!(buffer::Buffer{S})

Function for popping elements from buffer.
"""

#import Base: buffer_pop!

#=

CH: pop! generally returns items. If we don't return, it might 
be better to use delete! instead (pedantic sorry lol)

The functionality around the buffer seems like it will not 
change wrt the state type. Suppose we realise in a month that DQNState
isn't the only thing we want to use this for - it's really quite useful 
to then open this functionality up to an abstract state type. 

Finally -
Instead of using `buffer_FUNCTION`, such as `buffer_add!`, `buffer_pop!`, etc.,
why don't we just use add!, pop!, etc. ? The functionality is clear from context,
and is well defined and safe via mdispatch. Purely an aesthetic choice and not 
consequential lol
=#


# ==================== pop! functionalities ==================== #
function buffer_pop!(buffer::Buffer{S, A}, n::Int) where {S, A}
    if buffer.length >= n
        deleteat!(buffer.experiences, 1:n)
        if !isempty(buffer.bellman_errors)
            deleteat!(buffer.bellman_errors, 1:n)
            deleteat!(buffer.ranks, 1:n)
        end 
        buffer.length -= n
    else
        ErrorException("Fewer elements in buffer than number requested for removal")
    end
end

# ==================== add! functionalities ==================== #

function buffer_size_check(buffer::Buffer{S, A},l::Int64) where {S, A}
    check::Bool = buffer.length + l <= buffer.maximum_length 
    !check && ErrorException("Adding this many experiences would exceed maximum buffer length. Use buffer_pop! to free up space")
    return check
end

function buffer_add!(buffer::Buffer{S, A}, collected_exp::E; bellman_error::Union{Float64}=0.) where {S, A, E <: AbstractExperience}
    check::Bool = buffer_size_check(buffer, 1)
    if !check 
        push!(buffer.experiences, collected_exp)
        buffer.length += 1
    end
    if bellman_error > 0.
        push!(buffer.bellman_errors, bellman_error)
        push!(buffer.ranks, 0)
    end
end

function buffer_add!(buffer::Buffer{S, A}, collected_exp::Vector{E}; bellman_error::Vector{Float64}=Vector{Float64}()) where {S, A, E <: AbstractExperience}
    num_exps = length(collected_exp)
    check::Bool = buffer_size_check(buffer, num_exps)
    if check
        push!(buffer.experiences, collected_exp...)
        buffer.length += num_exps
    end
    if !isempty(bellman_error)
        push!(buffer.bellman_errors, bellman_error...)
        push!(buffer.ranks, zeros(Int, length(bellman_error))...)
    end
end

# ==================== update! functionalities ==================== #
buffer_shuffle!(buffer::Buffer{S, A}) where {S, A} = shuffle!(buffer.experiences) 


function buffer_update!(buffer::Buffer{S, A}, collected_experience::Vector{E}; bellman_errors::Vector{Float64}=Vector{Float64}(), shuffle::Bool=true)  where {S, A, E <: AbstractExperience}
    # Do some checks on the buffer lengths and act accordingly
    if buffer.length + length(collected_experience) <= buffer.maximum_length
        buffer_add!(buffer, collected_experience; bellman_error = bellman_errors)
    elseif buffer.length == buffer.maximum_length
        # shuffle && buffer_shuffle!(buffer)
        buffer_pop!(buffer, length(collected_experience))
        buffer_add!(buffer, collected_experience; bellman_error = bellman_errors)
    else # Does this situation happen? Do we safeguard against this with the prior checks?
        # shuffle && buffer_shuffle!(buffer)
        buffer_pop!(buffer, (buffer.length+length(collected_experience)-buffer.maximum_length))
        buffer_add!(buffer, collected_experience; bellman_error = bellman_errors)
    end
end

buffer_update!(buffer::Buffer{S, A}, collected_exp::E; bellman_error::Float64=0., shuffle::Bool=true) where {S, A, E <: AbstractExperience} = buffer_update!(buffer,[collected_exp]; bellman_errors = [bellman_error], shuffle=shuffle)

# ==================== sample functionalities ==================== #


import StatsBase: sample

function sample(buffer::Buffer{S, A}, sample_size::Int) where {S, A}# add functionality for changing the distribution?
    if !isempty(buffer.bellman_errors)
        sample_probs = buffer.bellman_errors .^ buffer.α ./ sum(buffer.bellman_errors .^ buffer.α)
        IS_weights = ( buffer.length .* sample_probs ) .^ (1*buffer.β)
        IS_weights ./= maximum(IS_weights)
        # println(sum(buffer.bellman_errors .^ buffer.α))
        # println(sample_probs)
        idxs = sample(1:buffer.length, Weights(sample_probs), sample_size; replace=false) 
        # println(idxs)
        buffer.ranks[idxs] .+= 1
    else
        idxs = rand(1:buffer.length, sample_size)
    end
    return buffer.experiences[idxs], idxs, IS_weights
end

function states(buffer::Buffer{S, A}) where {S, A}
    states = Vector{S}()
    for exp in buffer.experiences
        push!(states, exp.state)
    end
    return states
end

function rewards(buffer::Buffer{S, A}) where {S, A}
    rewards = Vector{S}()
    for exp in buffer.experiences
        push!(rewards, exp.reward)
    end
    return rewards
end

function next_states(buffer::Buffer{S, A}) where {S, A}
    next_states = Vector{S}()
    for exp in buffer.experiences
        push!(next_states, exp.next_state)
    end
    return next_states
end

function terminals(buffer::Buffer{S, A}) where {S, A}
    terminals = Vector{S}()
    for exp in buffer.experiences
        push!(terminals, exp.terminal)
    end
    return terminals
end

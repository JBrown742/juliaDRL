
# It's good to define these types with 'Abstract-' prefix, as it leaves a lot of 
# room for refinement in structs (and makes code nice and clear when you're super
# generally dispatching), i.e methods that dispatch on Abstract_____ clearly apply to 
# all subtypes. 

abstract type AbstractAlgorithm end

abstract type AbstractExperience end

"""
States will share common functionality and common dispatches, 
regardless of their specific application. 

Of course we can 
also specify our state as a specific constant type per application, 
but this misses out on the ability to reuse general functionality.
Furthermore, we miss out on the ability to make our code really 
easy to use and read by stacking specific functionality that only 
custom state structs could provide. 

I would suggest we build an overarching state construct which can
be readily subtyped. e.g., 

State{ Env }

so that we have more flexibility. BUT THEY ONLY NEED BUILT ONCE WE SPECIFY
AN ENV. So while we describe general practices, we can refer to abstract types. 

What is the basic structure of a State type?? If it needs more flexibility
we can build a macro that will generally build them. 
"""



mutable struct Experience{S, A} <: AbstractExperience
    state::S
    action::A
    next_state::S
    reward::Float32
    done::Bool
end

abstract type AbstractBuffer end

mutable struct Buffer{S, A} <: AbstractBuffer
    maximum_length::Int
    length::Int
    experiences::Vector{Experience{S,A}}
    bellman_errors::Vector{Float64}
    ranks::Vector{Int}
    α::Float64
    β::Float64
    # Very loose in the constructor but we can refine.
    Buffer{S,A}(max_length::Int) where {S, A}  = new{S,A}(max_length, 0, Vector{Experience{S,A}}(), Vector{Float64}(), Vector{Int}(), 1., 0.)
end

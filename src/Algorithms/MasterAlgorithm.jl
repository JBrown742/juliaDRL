
# It's good to define these types with 'Abstract-' prefix, as it leaves a lot of 
# room for refinement in structs (and makes code nice and clear when you're super
# generally dispatching), i.e methods that dispatch on Abstract_____ clearly apply to 
# all subtypes. 
abstract type AbstractAlgorithm end
abstract type AbstractExperience end
abstract type AbstractBuffer end

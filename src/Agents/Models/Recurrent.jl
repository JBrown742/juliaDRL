struct Recurrent <: AbstractModel
    model::Chain
end
function (m::Recurrent)(state::Vector{Float32})
    recurrent_layers = filter(layer -> layer isa Flux.Recur, m.model.layers)
    for r in recurrent_layers; Flux.reset!(r); end 
    output = m.model(state)
    return output
end
function (m::Recurrent)(state::Vector{Vector{Float32}})
    recurrent_layers = filter(layer -> layer isa Flux.Recur, m.model.layers)
    for r in recurrent_layers; Flux.reset!(r); end 
    reshaped_state = reduce(hcat, state)
    output = m.model(reshaped_state)
    return output
end
function (m::Recurrent)(state::Matrix{Float32})
    recurrent_layers = filter(layer -> layer isa Flux.Recur, m.model.layers)
    for r in recurrent_layers; Flux.reset!(r); end 
    output = m.model(state)
    return output
end
Functors.@functor Recurrent

function save_model(Recurrent_model::Recurrent, save_dir::String; model_info::String="")
    if !isdir(save_dir)
        mkdir(save_dir)
    end
    num_saved_models = length(readdir(save_dir))
    model = Recurrent_model.model
    if model_info==""
        @save save_dir * "/model_$(num_saved_models+1).bson" model
    else
        @save save_dir * "/model_" * model_info * ".bson" model
    end
end

function load_model(::Type{Recurrent}, load_path::String)
    @load load_path model
    Recurrent_model = Recurrent(model)
    return Recurrent_model
end


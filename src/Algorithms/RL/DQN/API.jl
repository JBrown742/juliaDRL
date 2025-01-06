function learn(env::Cartpole, alg::DQN; 
        training_iters = 4000, warmup_iters = 1000,
        checkpoint_freq=1, vals_per_checkpoint=10, 
        save_dir=pwd(), test_name="test"
    )
    info_dict = Dict{String, Any}("Agent" => Dict{String, Any}("model"=> repr(alg.agent.model)), 
        "Algorithm" => Dict{String, Any}("Buffer size" => alg.n, "batch_size" => alg.batch_size, 
        "γ" => alg.γ, "τ"=> alg.τ, "Optimizer"=> repr(alg.optimizer)), 
        "Environment" => repr(env), "training_iters" => training_iters, "warmup_iters" => warmup_iters,
        "checkpoint_freq" => checkpoint_freq, "vals_per_checkpoint" => vals_per_checkpoint
        )
    save_dir *= "/"*test_name
    if isdir(save_dir)
        rm(save_dir, recursive=true)
    end
    mkdir(save_dir)
    model_dir = save_dir*"/checkpointed_models"
    if isdir(model_dir)
       rm(model_dir, recursive=true)
    end
    mkdir(model_dir)
    reward_history = Vector{Float64}()
    best_reward = 0
    best_model = deepcopy(alg.agent.model)
    # execute several learning episodes to fill the buffer
    for i in 1:warmup_iters
        learning_episode!(env, alg)
    end
    println("learning begin")
    # then repeat for the number of training iterations
    for i in 1:training_iters 
        learning_episode!(env, alg)
        train!(alg)
        if i % checkpoint_freq == 0
            reward_av = mean([validation_episode!(env, alg) for _ in 1:vals_per_checkpoint])
            if reward_av > best_reward
                best_reward = reward_av
                best_model = deepcopy(alg.agent.model)
                vizenv = Cartpole(500, render=true)
                _ = validation_episode!(vizenv, alg, render=true)
                close!(vizenv)
                save_model(best_model, model_dir; model_info="iter_$(i)")
            end
            push!(reward_history, reward_av)
            println("Episode $(i):: Average reward is $(reward_av)")
        end
    end
    save_model(best_model, save_dir; model_info="best")
    plot(reward_history, title="average return for checkpointed models", linesize=2, legend=false);
    xlabel!("learning iteration", fontsize=20);
    ylabel!("avg reward", fontsize=20);
    savefig(save_dir*"/learning_curve")
    info_dict["best reward"] = best_reward
    json_string = JSON.json(info_dict)
    open(save_dir*"/metadata.json","w") do f 
        write(f, json_string) 
    end
end

function visualise_learning(env::Cartpole, test_dir::String)
    checkpoint_dir = test_dir*"/"*"checkpointed_models"
    model_list = readdir(checkpoint_dir)
    ordered_indices = sortperm(first.(split.(last.(split.(model_list, "_")), ".")))
    for mod in model_list[ordered_indices]
        iter = split(split(mod, "_")[end], ".")[1]
        model = load_model(checkpoint_dir * "/" *mod)
        agent = AbstractAgent(model)
        println("Model checkpointed at $(iter)")
        R = validation_episode!(env, agent)
        println("Achieved reward = $(R)")
    end
    close!(env)
end

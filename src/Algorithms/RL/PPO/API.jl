function juliaDRL.learn(env::Cartpole, alg::PPO; 
    training_iters = 100,
    checkpoint_freq=1, vals_per_checkpoint=10, 
    save_dir=pwd(), test_name="test"
)
    info_dict = Dict{String, Any}(
        "Agent" => Dict{String, Any}("model"=> repr(alg.central_agent.model)), 
        "Algorithm" => Dict{String, Any}(
            "batch_size" => alg.batch_size, 
            "Optimizer"=> repr(alg.optimizer),
            "N" => alg.N, "T" => alg.T, "K" => alg.K, 
            "λ" => alg.λ, "ϵ" => alg.ϵ, "γ" => alg.γ, 
            "c1" => alg.c1, "c2" => alg.c2
            ), 
        "Environment" => repr(env), 
        "training_iters" => training_iters,
        "checkpoint_freq" => checkpoint_freq, 
        "vals_per_checkpoint" => vals_per_checkpoint
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
    best_model = deepcopy(alg.central_agent.model)
    # execute several learning episodes to fill the buffer
    # then repeat for the number of training iterations
    for i in 1:training_iters 
        full_training_procedure!(alg, [env])
        if i % checkpoint_freq == 0
            reward_av = mean([validation_episode!(env, alg) for _ in 1:vals_per_checkpoint])
            if reward_av > best_reward
                best_reward = reward_av
                best_model = deepcopy(alg.central_agent.model)
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

function learn(env::E, alg::PPO; 
    training_iters = 1000,
    checkpoint_freq=1, vals_per_checkpoint=10, 
    save_dir=pwd(), test_name="test"
) where {E <: AbstractEnv}
    info_dict = Dict{String, Any}(
        "Agent" => repr(alg.central_agent), 
        "Algorithm" => Dict{String, Any}(
            "batch_size" => alg.batch_size,
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
    # If a save already exists
    println("Overwriting previous save...")
    if isdir(save_dir)
        rm(save_dir, recursive=true)
    end
    mkdir(save_dir)
    agent_dir = save_dir*"/checkpointed_agents"
    if isdir(agent_dir)
        rm(agent_dir, recursive=true)
    end
    mkdir(agent_dir)
    reward_history = Vector{Float64}()
    best_reward = -1e6
    best_agent = deepcopy(alg.central_agent)
    # execute several learning episodes to fill the buffer
    # then repeat for the number of training iterations
    for i in 1:training_iters 
        full_training_procedure!(alg, fill(deepcopy(env), alg.N))
        if i % checkpoint_freq == 0
            reward_av = mean([validation_episode!(env, alg) for _ in 1:vals_per_checkpoint])
            if reward_av > best_reward
                best_reward = reward_av
                best_agent = deepcopy(alg.central_agent)
                vizenv = deepcopy(env)
                vizenv.renderize = true
                _ = validation_episode!(vizenv, alg, render=true)
                close!(vizenv)
                save_agent(best_agent, agent_dir; agent_info="iter_$(i)")
            end
            push!(reward_history, reward_av)
            println("Episode $(i):: Average reward is $(reward_av)")
        end
        if i % alg.sync_frequency == 0
            update_actor_learners!(alg.central_agent, alg)
        end
    end
    save_agent(best_agent, save_dir; agent_info="best")
    plot(reward_history, title="average return for checkpointed agents", linesize=2, legend=false);
    xlabel!("learning iteration", fontsize=20);
    ylabel!("avg reward", fontsize=20);
    savefig(save_dir*"/learning_curve")
    info_dict["best reward"] = best_reward
    json_string = JSON.json(info_dict)
    open(save_dir*"/metadata.json","w") do f 
        write(f, json_string) 
    end
end

function visualise_learning(alg::PPO, env::E, test_dir::String) where {E <: AbstractEnv}
    checkpoint_dir = test_dir*"/"*"checkpointed_agents"
    agent_list = readdir(checkpoint_dir)
    ordered_indices = sortperm(parse.(Int, first.(split.(last.(split.(agent_list, "_")), "."))))
    println(parse.(Int, first.(split.(last.(split.(agent_list, "_")), "."))))
    println(agent_list[ordered_indices])
    for ag in agent_list[ordered_indices]
        iter = split(split(ag, "_")[end], ".")[1]
        agent = load_agent(typeof(alg.central_agent), checkpoint_dir * "/" * ag)
        println("Model checkpointed at $(iter)")
        R = validation_episode!(alg, env, agent)
        println("Achieved reward = $(R)")
    end
    close!(env)
end

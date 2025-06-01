function learn(env::E, alg::PPO; 
    training_iters = 1000,
    checkpoint_freq=1, vals_per_checkpoint=10, 
    save_dir=pwd(), test_name="test", average_window=2,
    plot_type="max_min"
) where {E <: AbstractEnv}
    info_dict = Dict{String, Any}(
        "Agent" => repr(alg.central_agent), 
        "Algorithm" => Dict{String, Any}(
            "batch_size" => alg.batch_size,
            "N" => alg.N, "T" => alg.T, "K" => alg.K, 
            "λ" => alg.λ, "ϵ" => alg.ϵ, "γ" => alg.γ, 
            "c1" => alg.c1, "c2" => alg.c2
            ), 
        # "Environment" => repr(env), 
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
    reward_history = Vector{Float32}()
    best_reward = -1e6
    best_agent = deepcopy(alg.central_agent)
    reset!(env)
    vizenv = deepcopy(env)
    renderize!(vizenv)
    @sync for p in workers()
       @spawnat p distribute_worker_envs(env)
    end
    # execute several learning episodes to fill the buffer
    # then repeat for the number of training iterations
    # initially fill the buffer 
    trajectories = get_trajectories!(alg, env)
    for i in 1:training_iters 
        # asynchronously launch training
        training_future = @async train!(alg, trajectories...)        
        # while master is training set workers off collctiong trajectories
        collection_future = @async get_trajectories!(alg, env)
        # wait for training to finish
        # update trajectories
        if i % checkpoint_freq == 0
            reward_av = mean([validation_episode!(env, alg) for _ in 1:vals_per_checkpoint])
            if reward_av > best_reward
                best_reward = reward_av
                best_agent = deepcopy(alg.central_agent)
                _ = validation_episode!(vizenv, alg, render=true)
                save_agent(best_agent, agent_dir; agent_info="iter_$(i)")
            end
            push!(reward_history, reward_av)
            println("Episode $(i):: Average reward is $(reward_av)")
        end
        if i % alg.sync_frequency == 0
            update_actor_learners!(alg.central_agent, alg)
        end
        wait(training_future)
        trajectories = fetch(collection_future)
    end
    close!(vizenv)
    save_agent(best_agent, save_dir; agent_info="best")
    # get running average_reward
    running_av = Vector{Float32}()
    running_std = Vector{Float32}()
    running_max = Vector{Float32}()
    running_min = Vector{Float32}()
    for (i, r) in enumerate(reward_history[1:end-average_window])
        window = reward_history[i:i + average_window]
        push!(running_av, mean(window))
        push!(running_std, std(window))
        push!(running_max, maximum(window) - mean(window))
        push!(running_min, mean(window) - minimum(window))
    end
    if plot_type=="max_min"
        plot(running_av, ribbon = (running_min, running_max), title="average return for checkpointed agents", linesize=2, legend=false);
    elseif plot_type=="std"
        plot(running_av, ribbon = running_std, title="average return for checkpointed agents", linesize=2, legend=false);
    end
    xlabel!("learning iteration", fontsize=20);
    ylabel!("avg reward", fontsize=20)

    savefig(save_dir*"/learning_curve")
    info_dict["best reward"] = best_reward
    json_string = JSON.json(info_dict)
    open(save_dir*"/metadata.json","w") do f 
        write(f, json_string) 
    end
end

function visualise_learning(alg::PPO{G}, env::E, test_dir::String) where {E <: AbstractEnv, G <: AbstractAction}
    checkpoint_dir = test_dir*"/"*"checkpointed_agents"
    agent_list = readdir(checkpoint_dir)
    ordered_indices = sortperm(parse.(Int, first.(split.(last.(split.(agent_list, "_")), "."))))
    println(parse.(Int, first.(split.(last.(split.(agent_list, "_")), "."))))
    println(agent_list[ordered_indices])
    for ag in agent_list[ordered_indices]
        iter = split(split(ag, "_")[end], ".")[1]
        agent = load_agent(typeof(alg.central_agent), alg.central_agent.model_type, checkpoint_dir * "/" * ag)
        println("Model checkpointed at $(iter)")
        R = validation_episode!(alg, env, agent; render=true)
        println("Achieved reward = $(R)")
    end
    close!(env)
end

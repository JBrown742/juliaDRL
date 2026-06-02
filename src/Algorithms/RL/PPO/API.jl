function learn(env::E, alg::PPO; 
    training_iters = 1000,
    checkpoint_freq=1, vals_per_checkpoint=10, 
    save_dir=pwd(), test_name="test", average_window=2,
    plot_type="max_min",
    random_policy=false,
    ephemeral=false
) where {E <: AbstractEnv}
    # setup directories
    if ephemeral==false
        if !isdir(save_dir)
            mkdir(save_dir)
        end
        agent_dir = save_dir * "/" * test_name * "/checkpointed_agents"
        if !isdir(save_dir * "/" * test_name)
            mkdir(save_dir * "/" * test_name)
        end
        if !isdir(agent_dir)
            mkdir(agent_dir)
        end
    end
    info_dict = Dict()
    reward_history = Vector{Float32}()
    best_reward = -1e6
    best_agent = deepcopy(alg.central_agent)
    reset!(env)
    
    for i in 1:training_iters 
        # Collect trajectories (normalized)
        trajectories = get_trajectories!(alg, env, random_policy=random_policy)
        
        # Extract and Normalize states for training
        states, actions, probs, advantages, targets, errors = trajectories
        
        # STABILITY FIX: Update normalizer with the whole batch
        for s in states
            update!(alg.obs_normalizer, s)
        end
        
        # Normalize states for training
        norm_states = [normalize(alg.obs_normalizer, s) for s in states]
        
        # Update our centralization model
        try
            train!(alg, norm_states, actions, probs, advantages, targets, errors)
        catch e
            println("Warning: Training step failed (likely NaN). Skipping. Error: ", e)
        end
        
        if i % alg.sync_frequency == 0
            update_actor_learners!(alg.central_agent, alg)
        end

        # checkpoint and validate
        if i % checkpoint_freq == 0
            reward_av = mean([validation_episode!(env, alg) for _ in 1:vals_per_checkpoint])
            push!(reward_history, reward_av)
            println("Episode $(i):: Average reward is $(reward_av)")
            if ephemeral==false
                if reward_av > best_reward
                    best_reward = reward_av
                    agent_save_path = save_agent(alg.central_agent, agent_dir; agent_info="iter_$(i)")
                    # Save normalizer state in the same directory
                    serialize(joinpath(agent_save_path, "normalizer.jls"), alg.obs_normalizer)
                end
            end
        end
    end
    
    # save_agent(best_agent, save_dir; agent_info="best")
    
    # learning curve plotting logic
    if length(reward_history) > average_window
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
            plot(running_av, ribbon = (running_min, running_max), title="Average Return", linesize=2, legend=false);
        elseif plot_type=="std"
            plot(running_av, ribbon = running_std, title="Average Return", linesize=2, legend=false);
        end
        xlabel!("Learning Iteration");
        ylabel!("Avg Reward")
        if ephemeral==false
            savefig(joinpath(save_dir, test_name, "learning_curve.png"))
        end
    end
    return reward_history
end

function _run_visualisation(alg::PPO{G}, env::E, agent_type::Type{A}, agent_dir::String) where {E <: AbstractEnv, G <: AbstractAction, A <: AbstractAgent}
    println("Visualising agent from: ", agent_dir)

    # Load the agent
    loaded_agent = load_agent(agent_type, agent_dir)

    # Load the normalizer if it exists
    norm_path = joinpath(agent_dir, "normalizer.jls")
    if isfile(norm_path)
        alg.obs_normalizer = deserialize(norm_path)
        println("Loaded observation normalizer.")
    end

    # Prepare environment for rendering
    renderize!(env)

    # Run a rendered episode
    try
        reward = validation_episode!(alg, env, loaded_agent; render=true)
        println("Visualisation complete. Episode Reward: ", reward)
        return reward
    finally
        # We don't close the env here to allow reuse in loops, 
        # but we might need to reset it.
    end
end

function visualise_learning(alg::PPO{G}, env::E, test_dir::String; agent_type::Type{A}=StandardActorCritic) where {E <: AbstractEnv, G <: AbstractAction, A <: AbstractAgent}
    checkpoint_dir = joinpath(test_dir, "checkpointed_agents")
    if !isdir(checkpoint_dir)
        println("Error: Checkpoint directory not found at ", checkpoint_dir)
        return
    end

    agent_list = readdir(checkpoint_dir)
    if isempty(agent_list)
        println("Error: No agents found in ", checkpoint_dir)
        return
    end

    # Sort by iteration number
    sorted_agents = sort(agent_list, by=x->begin
        m = match(r"iter_(\d+)", x)
        m === nothing ? 0 : parse(Int, m.captures[1])
    end)

    println("Visualising all $(length(sorted_agents)) checkpoints...")

    try
        for agent_name in sorted_agents
            _run_visualisation(alg, env, agent_type, joinpath(checkpoint_dir, agent_name))
        end
    finally
        close!(env)
    end
end

function visualise_best(alg::PPO{G}, env::E, test_dir::String; agent_type::Type{A}=StandardActorCritic) where {E <: AbstractEnv, G <: AbstractAction, A <: AbstractAgent}
    checkpoint_dir = joinpath(test_dir, "checkpointed_agents")
    if !isdir(checkpoint_dir)
        println("Error: Checkpoint directory not found at ", checkpoint_dir)
        return
    end

    agent_list = readdir(checkpoint_dir)
    if isempty(agent_list)
        println("Error: No agents found in ", checkpoint_dir)
        return
    end

    # The best agent is the one with the highest iteration index (since we only save on improvement)
    best_agent_name = sort(agent_list, by=x->begin
        m = match(r"iter_(\d+)", x)
        m === nothing ? 0 : parse(Int, m.captures[1])
    end)[end]

    println("Visualising BEST agent...")
    try
        _run_visualisation(alg, env, agent_type, joinpath(checkpoint_dir, best_agent_name))
    finally
        close!(env)
    end
end


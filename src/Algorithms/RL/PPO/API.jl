function learn(env::E, alg::PPO; 
    training_iters = 1000,
    checkpoint_freq=1, vals_per_checkpoint=10, 
    save_dir=pwd(), test_name="test", average_window=2,
    plot_type="max_min",
    random_policy=false
) where {E <: AbstractEnv}
    # setup directories
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
    info_dict = Dict()
    reward_history = Vector{Float32}()
    best_reward = -1e6
    best_agent = deepcopy(alg.central_agent)
    reset!(env)
    
    for i in 1:training_iters 
        # 1. Collect trajectories (normalized)
        trajectories = get_trajectories!(alg, env, random_policy=random_policy)
        
        # 2. Extract and Normalize states for training
        states, actions, probs, advantages, targets, errors = trajectories
        
        # STABILITY FIX: Reward Scaling
        # Large negative rewards in BipedalWalker can cause exploding gradients in the value function
        # scaled_targets = targets .* 1.0f0
        # scaled_errors = errors .* 1.0f0
        # scaled_advantages = advantages .* 1.0f0

        # # Update normalizer with the whole batch
        # for s in states
        #     update!(alg.obs_normalizer, s)
        # end
        
        # # STABILITY FIX: Update normalizer with the batch BEFORE training
        # for s in states
        #     update!(alg.obs_normalizer, s)
        # end
        # Normalize states for training
        # norm_states = [normalize!(alg.obs_normalizer, s) for s in states]
        
        # 3. Update our centralization model
        try
            train!(alg, states, actions, probs, advantages, targets, errors)
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
            
            if reward_av > best_reward
                best_reward = reward_av
                save_agent(alg.central_agent, agent_dir; agent_info="iter_$(i)")
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
        savefig(joinpath(save_dir, test_name, "learning_curve.png"))
    end
end

function visualise_learning(alg::PPO{G}, env::E, test_dir::String) where {E <: AbstractEnv, G <: AbstractAction}
    checkpoint_dir = test_dir*"/"*"checkpointed_agents"
    agent_list = readdir(checkpoint_dir)
    # Re-enable if needed, but ensure load_agent path is correct
    println("Visualisation directory: ", checkpoint_dir)
end

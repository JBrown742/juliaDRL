using Test
using ProximalPolicy

@testset "HPC Utilities" begin

    @testset "Unzip & Flatten Pipeline" begin
        # Mock trajectory data from 2 workers
        # Each worker returns a NamedTuple (or struct) with fields as vectors
        worker1_data = (
            states = [[1.0f0, 2.0f0], [3.0f0, 4.0f0]],
            actions = [1, 2],
            rewards = [0.1f0, 0.5f0]
        )
        worker2_data = (
            states = [[5.0f0, 6.0f0]],
            actions = [3],
            rewards = [1.0f0]
        )
        
        raw_results = [worker1_data, worker2_data]
        
        # unzip should flatten these into:
        # states: Vector of 3 vectors
        # actions: Vector of 3 Ints
        # rewards: Vector of 3 Float32s
        unzipped = ProximalPolicy.Algorithms.unzip(raw_results)
        
        @test length(unzipped) == 3 # states, actions, rewards
        
        unzipped_states = unzipped[1]
        unzipped_actions = unzipped[2]
        unzipped_rewards = unzipped[3]
        
        @test length(unzipped_states) == 3
        @test unzipped_states[1] == [1.0f0, 2.0f0]
        @test unzipped_states[3] == [5.0f0, 6.0f0]
        
        @test unzipped_actions == [1, 2, 3]
        @test unzipped_rewards == [0.1f0, 0.5f0, 1.0f0]
    end

end

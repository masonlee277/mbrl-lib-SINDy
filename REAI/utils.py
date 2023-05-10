import torch
import numpy as np
import matplotlib.pyplot as plt
from REAI.physics_models import trajectories_from_replay_buffer
from copy import deepcopy


import mbrl.planning as planning
import mbrl.util.common as common_util
#check physics model
#check physics model
def check_physics_model(replay_buffer, physics_model):
    '''
    physics_model = dynamics_model.model.physics_model
    '''

    trajectories_list, action_list  = trajectories_from_replay_buffer(replay_buffer)
    test_trajectory = trajectories_list[0]
    test_actions = action_list[0]

    predicted_states = []
    predict_recursively = []
    init_state = deepcopy(test_trajectory[0])

    cur_state = init_state
    #for i in range(len(test_trajectory)):
    for i in range(len(test_trajectory)):
        state = torch.tensor(test_trajectory[i])
        action = torch.tensor(test_actions[i])
        
        #print(state, init_state)
        #if i == 0: assert state == init_state, print(state,init_state)
        
        #predicting recursively (from its own prediction)
        if physics_model.predict_delta:
            next_state = np.array(physics_model.predict(torch.tensor(cur_state), action) + cur_state)
        else:
            next_state = np.array(physics_model.predict(torch.tensor(cur_state), action))
        predict_recursively.append(next_state)
        cur_state = next_state
        
        #predicting from actual state
        if physics_model.predict_delta:
            pred_state = np.array(physics_model.predict(state, action) + state)
        else:
            pred_state = np.array(physics_model.predict(state, action))
        predicted_states.append(pred_state)

    predicted_states = np.array(predicted_states)
    predict_recursively = np.array(predict_recursively)

    plt.figure(figsize=(10,15))
    state_dims = state.shape[0]
    for j in range(state_dims):
        plt.subplot(state_dims, 2, 2*j + 1)
        plt.plot( predicted_states[:-1, j] ,  label='model prediction from state')
        plt.plot( predict_recursively[:-1, j] ,  label='model prediction recursive')        
        plt.plot( test_trajectory[1:, j],  label='true trajectory')
        if j== state_dims - 1:
            plt.legend()


        plt.subplot(state_dims, 2, 2 *j + 2)
        plt.plot( np.abs(predicted_states[:-1, j] - test_trajectory[1:, j])  ,  label='model prediction from state')
        #plt.plot( np.abs(predicted_states_own[:-1, j]- test_trajectory[1:, j]) ,  label='model prediction recursive')        
        plt.title('Errors')
    plt.show()

def create_fake_replay_buffer(
    cfg, 
    obs_shape, 
    act_shape, 
    rng, 
    env, 
    dynamics_model, 
    num_samples=7000,
    num_actions=10,
    num_steps=8,
    input_mean=0.0,
    input_stddev=0.001,
    output_mean=0.0,
    output_stddev=0.001
):
    replay_buffer_fake = common_util.create_replay_buffer(cfg, obs_shape, act_shape, rng=rng)
    common_util.rollout_agent_trajectories(
            env,
            num_samples,  # initial exploration steps
            planning.RandomAgent(env),
            {},  # keyword arguments to pass to agent.act()
            replay_buffer=replay_buffer_fake,
            trial_length=num_samples,
        )

    trajectories_list, action_list = dynamics_model.model.physics_model.extract_data_from_buffer(replay_buffer_fake)
    total_states = np.concatenate(trajectories_list)

    action_space = env.action_space
    lower_bound = action_space.low
    upper_bound = action_space.high

    # Create temporary lists to store trajectories and actions
    temp_obs = []
    temp_action = []
    temp_next_obs = []

    for init_s in total_states:
        actions = [np.random.uniform(lower_bound, upper_bound) for x in range(num_actions)]
        actions = np.squeeze(actions)
        sim_trajectory = dynamics_model.model.physics_model.simulate(init_s, actions, num_steps)

        for i in range(len(sim_trajectory) - 1):
            cur_state = sim_trajectory[i]
            next_state = sim_trajectory[i + 1]

            cur_action = actions[i]

            # Add noise to the input state
            cur_state_noisy = cur_state + np.random.normal(loc=input_mean, scale=input_stddev, size=cur_state.shape)
            cur_state_noisy = np.clip(cur_state_noisy, lower_bound, upper_bound)
            temp_obs.append(cur_state_noisy)

            # Add noise to the output state
            next_state_noisy = next_state + np.random.normal(loc=output_mean, scale=output_stddev, size=next_state.shape)
            next_state_noisy = np.clip(next_state_noisy, lower_bound, upper_bound)
            temp_next_obs.append(next_state_noisy)

            temp_action.append(cur_action)

    # Set the actual replay buffer to the temporary lists
    temp_obs = np.array(temp_obs)
    temp_action = np.array(temp_action)
    temp_next_obs = np.array(temp_next_obs)

    # Expand dimensions of temp_action to match the shape of replay_buffer_fake.action
    temp_action = np.expand_dims(temp_action, axis=1)

    # Set the first values in replay_buffer_fake to the values in the respective temp lists
    replay_buffer_fake.obs[:replay_buffer_fake.num_stored] = temp_obs[:replay_buffer_fake.num_stored]
    replay_buffer_fake.action[:replay_buffer_fake.num_stored] = temp_action[:replay_buffer_fake.num_stored]
    replay_buffer_fake.next_obs[:replay_buffer_fake.num_stored] = temp_next_obs[:replay_buffer_fake.num_stored]

    return replay_buffer_fake



    


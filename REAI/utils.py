import torch
import numpy as np
import matplotlib.pyplot as plt
from REAI.physics_models import trajectories_from_replay_buffer
from copy import deepcopy

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



    


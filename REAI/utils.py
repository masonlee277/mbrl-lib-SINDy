import torch
import numpy as np
import matplotlib.pyplot as plt
from REAI.physics_models import trajectories_from_replay_buffer


#check physics model
def check_physics_model(replay_buffer, physics_model):
    '''
    physics_model = dynamics_model.model.physics_model
    '''

    trajectories_list, action_list  = trajectories_from_replay_buffer(replay_buffer)
    test_trajectory = trajectories_list[1]
    test_actions = action_list[1]

    predicted_states = []
    predict_recursively = []
    init_state = test_trajectory[0]
    for i in range(len(test_trajectory)):
        state = torch.tensor(test_trajectory[i])
        action = torch.tensor(test_actions[i])
        
        #predicting recursively (from its own prediction)
        predict_recursively.append(np.array(physics_model.predict(torch.tensor(init_state), action)))
        
        #predicting from actual state
        next_state = np.array(physics_model.predict(state, action))
        predicted_states.append(next_state)
        init_state = next_state

    predicted_states = np.array(predicted_states)
    predict_recursively = np.array(predict_recursively)

    plt.figure()
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

    


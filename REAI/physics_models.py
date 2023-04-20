import torch
import math

import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
# class PhysicsModel():

#     def __init__(self) -> None:
        
#     def predict(self, state, action):
#         raise NotImplementedError


class CartpoleModel():
    def __init__(self, 
                 gravity = 9.8,
                masscart = 1.0,
                masspole = 0.1,
                length = 0.5,       # actually half the pole's length
                force_mag = 10.0,
                tau = 1, #0.02,         # seconds between state updates
                kinematics_integrator = 'euler'):
                
        self.gravity = gravity
        self.masscart = masscart
        self.masspole = masspole
        self.total_mass = self.masspole + self.masscart
        self.length = length
        self.polemass_length = self.masspole * self.length
        self.force_mag = force_mag
        self.tau =  tau
        self.kinematics_integrator = kinematics_integrator


    def predict(self, state, action):
        '''
        returns delta on the state
        '''


        x, x_dot, theta, theta_dot = state[..., 0:1], state[...,1:2], state[...,2:3], state[...,3:4]
        force = (action*(action==1)*self.force_mag - action*(action!=1)*self.force_mag).unsqueeze(-1)
        # print('force', force.shape)
        # print('action', action.shape)
        # print('state', state.shape)
        
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)
        temp = (
            force + self.polemass_length * theta_dot ** 2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            dx = self.tau * x_dot
            dx_dot = self.tau * xacc
            d_theta = self.tau * theta_dot
            d_theta_dot = self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.dstate = torch.cat((dx, dx_dot, d_theta, d_theta_dot), dim = -1)

        return self.dstate
    

class SINDyModel():
    '''SINDy model for the cartpole system
    der : pysindy differentiation method: options are pysindy.differentiation.FiniteDifference, pysindy.differentiation.SINDyDerivative, pysindy.differentiation.SmoothedFiniteDifference
    
    '''
    def __init__(self, 
                    sparsity_threshold = 0.01, 
                    der = ps.SmoothedFiniteDifference(),
                    functions = [lambda x : 1, 
                                 lambda x : x, 
                                 lambda x : x**2, 
                                 lambda x,y : x*y, 
                                 lambda x: np.sin(x), 
                                 lambda x : np.cos(x)], 
                    print_model = False):
        

        self.der = der
        self.functions = functions
        self.lib = ps.CustomLibrary(library_functions=functions)
        self.optimizer = ps.STLSQ(threshold=sparsity_threshold)
        self.model = ps.SINDy(discrete_time=True, 
                              feature_library=self.lib, 
                              differentiation_method=self.der,
                              optimizer=self.optimizer)
        self.print_model = print_model

    def extract_data_from_buffer(self, replay_buffer_test):
        d = replay_buffer_test.get_all()
        #print("# samples stored", replay_buffer_test.num_stored)
        tup = d.astuple() # self.obs, self.act, self.next_obs, self.rewards, self.dones
        observations = np.array(tup[0])
        actions = np.array(tup[1])
        dones = np.array(tup[-1])

        #print(observations.shape)
        #print(dones.shape)
        #print(actions.shape)

        trajectory_splits = np.where(dones)[0] + 1
        trajectories = np.split(observations, trajectory_splits[:-1])
        u = np.split(actions, trajectory_splits[:-1])

        total_steps = 0
        # Print the individual trajectories
        #print('# of different trajectories: ', len(trajectories))
        trajectories_list = []
        action_list = []
        for i, (traj, act_seq) in enumerate(zip(trajectories,u)):
            #print(f'Trajectory {i + 1}:')
            total_steps += traj.shape[0]
            #if traj.shape[0] >0:
            #    trajectories_list.append(traj)
            #    action_list.append(act_seq)
            #print(traj.shape, act_seq.shape)
            #print()
        print('total steps: ', total_steps)

        # Convert the NumPy array of arrays to a list of arrays
        trajectories_list = [traj for traj in trajectories]
        action_list = [a for a in u]

        return trajectories_list, action_list


    def train(self, replay_buffer, num_trails = None):

        m  = ps.SINDy(discrete_time=True, 
                              feature_library=self.lib, 
                              differentiation_method=self.der,
                              optimizer=self.optimizer)

        trajectories_list, action_list = self.extract_data_from_buffer(replay_buffer)
        if num_trails is not None: 
            trajectories_list = trajectories_list[:num_trails]
            action_list = action_list[:num_trails]# this is for measuring sindy lengths

        print('training trajecotries: ', len(trajectories_list), num_trails)


        m.fit(trajectories_list,u=action_list, multiple_trajectories=True)
        self.model = m
        
        self.print_model=False
        if self.print_model:
            self.model.print()


    def simulate(self, inital_state, action_list, num_steps):
        return self.model.simulate(inital_state, num_steps, u = action_list)
    

    def predict(self, state, action, num_steps = 1, plot = False):
        
        action = action.unsqueeze(-1)
        action_tmp = action.cpu().numpy().reshape(-1, action.shape[-1])
        state_temp = state.cpu().numpy().reshape(-1,state.shape[-1])
        rollouts = np.array([self.model.simulate(state_i, num_steps, u=action_i) for state_i, action_i in zip(state_temp, action_tmp)]) 
        rollouts = rollouts.reshape(state.shape)

        if plot:
            nftrs = state.shape[-1]

            plt.figure()
            for i in range(nftrs):
                plt.subplot(nftrs,1, i+1)
                plt.plot(rollouts[0,:, i], label = 'SINDy')
                plt.plot(state[0, :,  i], '-o', label = 'true')
                plt.legend()
            plt.show()


        return torch.from_numpy(rollouts).float().to(state.device)





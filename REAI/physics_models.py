import torch
import math
import time
import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt

from dask import delayed
import dask.bag as db

# class PhysicsModel():

#     def __init__(self) -> None:
        
#     def predict(self, state, action):
#         raise NotImplementedError

def trajectories_from_replay_buffer(replay_buffer):
        d = replay_buffer.get_all()
        print("# samples stored", replay_buffer.num_stored)
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
        trajectories = np.split(observations, trajectory_splits)
        u = np.split(actions, trajectory_splits)

        total_steps = 0
        # Print the individual trajectories
        #print('# of different trajectories: ', len(trajectories))
        for i, (traj, act_seq) in enumerate(zip(trajectories,u)):
            #print(f'Trajectory {i + 1}:')
            total_steps += traj.shape[0]
            #print(traj.shape, act_seq.shape)
            #print()
        #print('total steps: ', total_steps)

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
        
    
    def predict(self, state, action, num_steps = 1, plot = False, overflow_clipp = 1e6, run_with_dask = True):

        state_np = state.cpu().numpy().reshape(-1, state.shape[-1])
        batch_size = state_np.shape[0]

        action_np = action.cpu().numpy().reshape(batch_size,-1)
        t0 = time.time()
        #rollouts = np.array([self.model.simulate(state_i, num_steps + 1, u=action_i)[1:] for state_i, action_i in zip(state_temp, action_tmp)]) 
        #rollouts = rollouts.reshape(state.shape)

        if batch_size>1:
            multiple_trjectories = True
            state_np = state_np[:, np.newaxis, :]
            action_np = action_np[:, np.newaxis, :]
        else:
            multiple_trjectories = False

        def f(x, u):
            if len(x.shape)==1:
                x = x[np.newaxis, :]
            if len(u.shape)==1:
                u = u[np.newaxis, :]
            #print(x.shape, u.shape)

            batch_size = x.shape[0]
            if batch_size>1:
                multiple_trjectories = True
            else:
                multiple_trjectories = False


            dx = self.model.predict(x, u=u, multiple_trajectories= multiple_trjectories)
            #convert to array and replace inf with 1e6
            dx = np.array(dx, dtype=np.float32)
            dx[dx>overflow_clipp] = overflow_clipp
            dx[dx<-overflow_clipp] = -overflow_clipp
            return dx

        if batch_size>8:
            print('Running with dask')
            if run_with_dask:
                u_bag = db.from_sequence(action_np)
                x_bag = db.from_sequence(state_np)

                dx_bag = x_bag.map(f, u_bag)
                #bag = db.from_sequence(zip(action_np, state_np), npartitions=8)
                dx = dx_bag.compute()
            
        else:
            dx = [f(x, u) for x, u in zip(state_np, action_np) ]

        dx = np.array(dx, dtype=np.float32)

        #dx = self.model.predict(state_np, u=action_np, multiple_trajectories= multiple_trjectories)
        #convert to array and replace inf with 1e6


        #    return dx
        
        #delayed_f = delayed(f)
        #dx = da.map()
        # rollouts0 = np.array([self.model.simulate(state_i, num_steps + 1, u=action_i)[:1] for state_i, action_i in zip(state_temp, action_tmp)]) 
        # rollouts0 = rollouts0.reshape(state.shape)
        
        ts = time.time() - t0
        print(batch_size)
        print('Sindy simulation time: ', ts)

        # if len(rollouts.shape) > 1: 
        #     print(rollouts.reshape(-1, 4)[0, :])
        #     print(rollouts0.reshape(-1, 4)[0, :])
        #     print(state.reshape(-1, 4)[0, :]) 
        # else:

        #     print(rollouts)
        #     print(rollouts0)
        #     print(state)

        if plot:
            nftrs = state.shape[-1]

            plt.figure()
            for i in range(nftrs):
                plt.subplot(nftrs,1, i+1)
                plt.plot(rollouts[0,:, i], label = 'SINDy')
                plt.plot(state[0, :,  i], '-o', label = 'true')
                plt.legend()
            plt.show()

        new_state = torch.from_numpy(dx.reshape(state.shape)).float().to(state.device)
        return new_state





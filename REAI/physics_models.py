import torch
import math
import time
import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt



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
                tau = 0.02,         # seconds between state updates
                kinematics_integrator = 'euler', 
                noise_level = 0, 
                predict_delta = False,
                **kwargs ):
                
        self.gravity = gravity
        self.masscart = masscart
        self.masspole = masspole
        self.total_mass = self.masspole + self.masscart
        self.length = length
        self.polemass_length = self.masspole * self.length
        self.force_mag = force_mag
        self.tau =  tau
        self.kinematics_integrator = kinematics_integrator
        self.noise_level = noise_level
        self.predict_delta = predict_delta
        self.inputs_normalizer = None
    def predict(self, state, action):
        '''
        returns the next state prediction
        '''

        #print(state.shape)
        x, x_dot, theta, theta_dot = state[..., 0:1], state[...,1:2], state[...,2:3], state[...,3:4]
        #force = (action*(action==1)*self.force_mag - action*(action!=1)*self.force_mag).unsqueeze(-1)
        force = action * self.force_mag

        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)
        temp = (
            force.reshape(theta_dot.shape) + self.polemass_length * theta_dot ** 2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        #Calculateing the derivatives
        if self.kinematics_integrator == "euler":
            dx = self.tau * x_dot
            dx_dot = self.tau * xacc
            d_theta = self.tau * theta_dot
            d_theta_dot = self.tau * thetaacc



            if self.predict_delta:
                dx = (dx).reshape(x.shape)
                dx_dot = (dx_dot).reshape(x_dot.shape)
                dtheta = (d_theta).reshape(theta.shape)
                dtheta_dot = (d_theta_dot).reshape(theta_dot.shape)
                output = torch.cat((dx, dx_dot, dtheta, dtheta_dot), dim = -1)

            else: 
                x = (x + dx).reshape(x.shape)
                x_dot = (x_dot + dx_dot).reshape(x_dot.shape)
                theta = (theta + d_theta).reshape(theta.shape)
                theta_dot = (theta_dot + d_theta_dot).reshape(theta_dot.shape)
                output = torch.cat((x, x_dot, theta, theta_dot), dim = -1)
            
        # semi-implicit euler
        # else:  
        #     x_dot = x_dot + self.tau * xacc
        #     x = x + self.tau * x_dot
        #     theta_dot = theta_dot + self.tau * thetaacc
        #     theta = theta + self.tau * theta_dot
        #     state = torch.cat((x, x_dot, theta, theta_dot), dim = -1)

        if self.noise_level > 0:
            noise = torch.randn(state.shape) * self.noise_level
            output = output + noise.to(state.device) 

        return output
    

class SINDyModel():
    '''SINDy model for the cartpole system
 
    der: pysindy.differentiation.SmoothedFiniteDifference
        The differentiation method to use
    functions: list of functions
        The library of functions to use
    print_model: bool
        Whether to print the model after training
    backend: str
        The backend to use for the model
        options: 'torch', 'sindy', 'dask'
    
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
                    print_model = False, 
                    backend = 'torch',
                    noise_level = 0,
                    predict_delta = True,
                    **kwargs):
    

        self.der = der
        self.functions = functions
        self.torch_functions = [lambda x : 1, 
                                 lambda x : x, 
                                 lambda x : x**2, 
                                 lambda x,y : x*y, 
                                 lambda x: torch.sin(x), 
                                 lambda x : torch.cos(x)]

                                
        self.lib = ps.CustomLibrary(library_functions=functions)
        self.optimizer = ps.STLSQ(threshold=sparsity_threshold)
        self.model = ps.SINDy(discrete_time=True, 
                              feature_library=self.lib, 
                              differentiation_method=self.der,
                              optimizer=self.optimizer)
        self.print_model = print_model

        self.backend = backend
        self.noise_level = noise_level
        self.predict_delta = predict_delta
        self.inputs_normalizer = None

    def extract_data_from_buffer(self, replay_buffer_test):
        d = replay_buffer_test.get_all()
        tup = d.astuple() 
        observations = np.array(tup[0])
        actions = np.array(tup[1])
        dones = np.array(tup[-1])

        trajectory_splits = np.where(dones)[0] + 1
        trajectories = np.split(observations, trajectory_splits)
        u = np.split(actions, trajectory_splits)

        total_steps = 0
        for i, (traj, act_seq) in enumerate(zip(trajectories,u)):
            #print(f'Trajectory {i + 1}:')
            total_steps += traj.shape[0]

        # Convert the NumPy array of arrays to a list of arrays
        trajectories_list = []
        action_list = []

        for traj, act in zip(trajectories, u):
            if len(traj) > 0:
              trajectories_list.append(traj)
              action_list.append(act)  

        return trajectories_list, action_list

    def equations_to_pytorch(self):
        equations = self.model.equations()
        
        x_dim = len(equations) #since there is 1 equation per variable
        num_functions = len(self.functions)
        torch_equations = []
        for eq_string in equations:
            #replacing xs
            for i in range(x_dim):
                eq_string = eq_string.replace(f'x{i}[k]', f'x[:, {i}]')

            #replacing functions
            for i in range(num_functions):
                eq_string = eq_string.replace(f'f{i}', f'* self.torch_functions[{i}]')

            eq_string = eq_string.replace(f'u0[k]', f'u[:, 0]')
            torch_equations.append(eq_string)
        
        return torch_equations


    def train(self, replay_buffer, num_trails = None):

        m  = ps.SINDy(discrete_time=True, 
                              feature_library=self.lib, 
                              differentiation_method=self.der,
                              optimizer=self.optimizer)

        trajectories_list, action_list = self.extract_data_from_buffer(replay_buffer)
        if num_trails is not None: 
            trajectories_list = trajectories_list[:num_trails]
            action_list = action_list[:num_trails]# this is for measuring sindy lengths

        #print('training trajecotries: ', len(trajectories_list), num_trails)


        m.fit(trajectories_list,u=action_list, multiple_trajectories=True)
        self.model = m
        
        self.print_model=False
        if self.print_model:
            self.model.print()
        
        #update torch equations
        self.torch_equations = self.equations_to_pytorch()


    def _predict_with_torch(self, state, action):
        '''
        Calculate the Sindy prediction using sindy eautions converted to 
        pytorch functiosn 
        '''

        if self.torch_equations is None:
            print('Converting equations to torch functions')
            self.torch_equations = self.equations_to_pytorch()
        x = state

        u = action.unsqueeze(-1)

        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        if len(x.shape) >2:
            x = x.reshape(-1, x.shape[-1])
        if len(u.shape) >2:
            u = u.reshape(-1, u.shape[-1])

        dx = torch.zeros(x.shape, device = x.device)
        for i in range(len(self.torch_equations)):
            dx[:, i] = eval(self.torch_equations[i])

        return dx.reshape(state.shape)
    
    def _run_with_dask(self, state, action, f):
        import dask.bag as db

        u_bag = db.from_sequence(action)
        x_bag = db.from_sequence(state)

        dx_bag = x_bag.map(f, u_bag)
        dx = dx_bag.compute()
        return dx
    

    def simulate(self, inital_state, action_list, num_steps):
        return self.model.simulate(inital_state, num_steps, u = action_list)

    def predict(self, state,
                 action, overflow_clipp = 1e6):
        
        '''
        Runs model prediction:
        state: torch tensor of shape (batch_size, state_dim)
        action: torch tensor of shape (batch_size, action_dim)
        num_steps: number of steps to predict
        backend: 'torch' or 'sindy' or 'dask', default 'torch'


        returns:
        next state prediction
        '''
        
        t0 = time.time()
        #self.sindy_calls += 1

        if self.backend == 'torch':
            dx = self._predict_with_torch(state, action)

        else:
            state_np = state.cpu().numpy().reshape(-1, state.shape[-1])
            batch_size = state_np.shape[0]

            action_np = action.cpu().numpy().reshape(batch_size,-1)

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
            

            if self.backend == 'dask':   
                print('Running with dask')
                dx = self._run_with_dask(state_np, action_np, f)

            elif self.backend == 'sindy':
                dx = f(state_np, action_np)
                #dx = np.array([f(x, u) for x, u in zip(state_np, action_np) ])
            
            #converting back to torch tensor, back to cuda
            dx = torch.from_numpy(dx.reshape(state.shape)).float().to(state.device)

        ts = time.time() - t0
        #print('Sindy simulation time: ', ts)

        #this is the correct delta prediction
        if self.predict_delta:
            output =  dx.reshape(state.shape) - state

        #output is by default the next-state prediction
        else:
            output = dx.reshape(state.shape) #  + state

        if self.noise_level > 0:
            noise = torch.randn(state.shape) * self.noise_level
            output = output + noise.to(state.device)
    
        return output
    



from IPython import display

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
import omegaconf

import mbrl.env.cartpole_continuous as cartpole_env
import mbrl.env.reward_fns as reward_fns
import mbrl.env.termination_fns as termination_fns

# all the mbrl.models sub-class the
import mbrl.models as models
import mbrl.planning as planning
import mbrl.util.common as common_util
import mbrl.util as util
from mbrl.util.math import powerlaw_psd_gaussian

from REAI.physics_models import SINDyModel, CartpoleModel
from REAI.physics_models import trajectories_from_replay_buffer
from itertools import product

from utils import check_physics_model

## TO do:
#exhaustive action trajectory search -> validate physics models
# compare distriobutions reward vs n actions (how narro it is )
# for different time horizins, different noise levels

device = "cuda:0" if torch.cuda.is_available() else "cpu"

seed = 0
trial_length = 200
num_trials = 10
ensemble_size = 5
rendering = True
num_cem_particles = 5 # number of particles for CEM, default 20


env = cartpole_env.CartPoleEnv()
env.seed(seed)
rng = np.random.default_rng(seed=seed)
generator = torch.Generator(device=device)
generator.manual_seed(seed)
obs_shape = env.observation_space.shape
act_shape = env.action_space.shape

# This functions allows the model to evaluate the true rewards given an observation
reward_fn = reward_fns.cartpole

# This function allows the model to know if an observation should make the episode end
term_fn = termination_fns.cartpole







# Everything with "???" indicates an option with a missing value.
# Our utility functions will fill in these details using the
# environment information
cfg_dict = {
    # dynamics model configuration
    "dynamics_model": {
        "_target_": "mbrl.models.GaussianMLP",  # NOTE: important we are using a GAUSSIANMLP Here --
        "device": device,
        "num_layers": 3,
        "ensemble_size": ensemble_size,
        "hid_size": 200,
        "in_size": "???",
        "in_features": "???",
        "out_size": "???",
        "deterministic": False,  # probabilistic model
        "propagation_method": "fixed_model",
        # can also configure activation function for GaussianMLP
        "activation_fn_cfg": {"_target_": "torch.nn.LeakyReLU", "negative_slope": 0.01},
    },
    # options for training the dynamics model
    "algorithm": {
        "learned_rewards": False,
        "target_is_delta": True,
        "normalize": True,
    },
    # these are experiment specific options
    "overrides": {
        "trial_length": trial_length,
        "num_steps": num_trials * trial_length,
        "model_batch_size": 32,
        "validation_ratio": 0.05,
    },
}

# CONFIGURATIONS PHYSICS MODEL + NN

# phys_nn_config = 0
# 0: no physics model, only pets

# phys_nn_config = 1:
# mean = NN(state, action) + physics_model.predict(state, action)
# logvar = NN(state, action)

# phys_nn_config = 2
# mean, logvar = NN(concat(physics_model.predict(state, action), state, action)
# here hidden layers must be doubled

# phys_nn_config = 3
# only physics model

phys_nn_config = 3

if phys_nn_config == 2:
    cfg_dict["dynamics_model"]["in_features"] = 2 * obs_shape[0] + (
        act_shape[0] if act_shape else 1
    )
    
if phys_nn_config == 3:
    cfg_dict["dynamics_model"]["deterministic"] = True


#initializing dynamics model

cfg = omegaconf.OmegaConf.create(cfg_dict)
# Create a 1-D dynamics model for this environment
dynamics_model = common_util.create_one_dim_tr_model(cfg, obs_shape, act_shape)
dynamics_model.model.physics_model = CartpoleModel() #SINDyModel(backend='torch') # change backend to 'torch' to run on GPU

    # CartpoleModel() 
dynamics_model.model.phys_nn_config = phys_nn_config

# Create a gym-like environment to encapsulate the model
model_env = models.ModelEnv(
    env, dynamics_model, term_fn, reward_fn, generator=generator
)

#initialize replay buffer
replay_buffer = common_util.create_replay_buffer(cfg, obs_shape, act_shape, rng=rng)
common_util.rollout_agent_trajectories(
    env,
    trial_length,  # initial exploration steps
    planning.RandomAgent(env),
    {},  # keyword arguments to pass to agent.act()
    replay_buffer=replay_buffer,
    trial_length=trial_length,
)

# pretrain Sindy model on random trajectories
if isinstance(dynamics_model.model.physics_model, SINDyModel):
    dynamics_model.model.physics_model.train(replay_buffer)


check_physics_model(replay_buffer, dynamics_model.model.physics_model)
print("num stored", replay_buffer.num_stored)

print("# samples stored", replay_buffer.num_stored)

agent_cfg = omegaconf.OmegaConf.create(
    {
        # this class evaluates many trajectories and picks the best one
        "_target_": "mbrl.planning.TrajectoryOptimizerAgent",
        "planning_horizon": 15,
        "replan_freq": 1,
        "verbose": False,
        "action_lb": "???",
        "action_ub": "???",
        # this is the optimizer to generate and choose a trajectory
        "optimizer_cfg": {
            "_target_": "mbrl.planning.CEMOptimizer",
            "num_iterations": 5,
            "elite_ratio": 0.1,
            "population_size": 500,
            "alpha": 0.1,
            "device": device,
            "lower_bound": "???",
            "upper_bound": "???",
            "return_mean_elites": True,
            "clipped_normal": False,
            #for iCEM
            #"keep_elite_frac" : 0.1,
            #"population_decay_factor":  0.9, 
            #"colored_noise_exponent": 0.5,
        },
    }
)




def physics_model_validation(time_horizon = 20, 
                             n_action_sequences = 1000):


    #action_sequences = powerlaw_psd_gaussian(exponent = 0.9, size = [n_action_sequences, time_horizon, 1], device = device)
    action_sequences = torch.rand([n_action_sequences, time_horizon, 1], dtype=torch.float32)*(-2) + 1

    x0 = torch.rand([4])*0.05

    noise_levels = [1, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 0]

    rewards_stats = []

    plt.figure()
    for i, ne in enumerate(noise_levels):
        
        #cfg = omegaconf.OmegaConf.create(cfg_dict)
        # Create a 1-D dynamics model for this environment
        dynamics_model = common_util.create_one_dim_tr_model(cfg, obs_shape, act_shape)
        dynamics_model.model.physics_model = CartpoleModel(noise_level=ne) #SINDyModel(backend='torch') # change backend to 'torch' to run on GPU

            # CartpoleModel() 
        dynamics_model.model.phys_nn_config = phys_nn_config

        # Create a gym-like environment to encapsulate the model
        model_env = models.ModelEnv(
            env, dynamics_model, term_fn, reward_fn, generator=generator
        )

        agent = planning.create_trajectory_optim_agent_for_model(
            model_env, agent_cfg, num_particles=num_cem_particles)


        #for i, x0_i in enumerate(x0):
        rewards = agent.trajectory_eval_fn(x0, action_sequences).cpu().detach().numpy()
        rewards_stats.append(rewards)
        #for debug break point -> model_env.py, line 196
        #    
        #plot histogram of rewards
        plt.subplot(3, 3, i+1)
        plt.hist(rewards, bins=20)
        plt.xlabel('reward')
        plt.ylabel('count')
        plt.title('Noise level: {:.3e}'.format(ne))
        plt.tight_layout()

    plt.show()
    return 

#x0_test = replay_buffer.get_all().astuple()[0][50, :]

physics_model_validation()


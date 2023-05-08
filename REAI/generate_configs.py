import torch
import numpy as np
import omegaconf




device = "cuda:0" if torch.cuda.is_available() else "cpu"

#Physics config options:  
# 0: no physics model, only pets
# 1: additive composition: 
#    mean = physics_model(x) + Pets(x), logvar = Pets(x) 
# 2: physics model prediction pass through pets
#   mean, logvar = NN(concat(physics_model.predict(state, action), state, action)
#3: only physics model, no NN


exp_config = omegaconf.DictConfig({
        'seed' : 0, 
        'phys_nn_config' : 3,                                  
        'physics_model' : 'sindy',  #sindy, cartpole
            'predict_delta' : False, 
        'model_kwargs' : {'backend' : 'torch', #'sindy', 'dask',  
                            'predict_delta' : False,
                            'noise_level' : 0.0},
        'trial_length' : 200,
        'num_trials' : 10,
        'num_particles' : 20,             
        'ensemble_size' : 5,
    })

with open('REAI/configs/main.yaml', 'w') as f:
    f.write(omegaconf.OmegaConf.to_yaml(exp_config))
    f.write('defaults: \n')
    f.write(' - dynamics: GaussianMLP\n')
    f.write(' - optimizer: CEM\n')
    f.write(' - agent: TrajectoryOptimizerAgent\n')
    f.write(' - env: cartpole\n')


env_cfg = {
    'gravity': 9.8,
    'masscart': 1.0,
    'masspole': 0.1,
    'length': 0.5,
    'force_mag': 10.0,
    'track_friction': 0.0,
    'joint_friction': 0.0,
    # Standard deviation on the observation of the states
    # DOES NOT AFFECT INTERNAL STATE PROPOGATION 
    # can be scalar or 1-d array
    # 'obs_noise': [0.003, 0.01, 0.003, 0.01], 
    'obs_noise': 0.0, 
   }

with open('REAI/configs/env/cartpole.yaml', 'w') as f:
    f.write('# @package _group_\n')
    f.write(omegaconf.OmegaConf.to_yaml(env_cfg))


optimizer_config = {
    "_target_": "mbrl.planning.CEMOptimizer",
    "num_iterations": 20,
    "elite_ratio": 0.1,
    "population_size": 1000,
    "alpha": 0.1,
    "device": device,
    "lower_bound": "???",
    "upper_bound": "???",
    "return_mean_elites": True,
    "clipped_normal": True,
}


with open('REAI/configs/optimizer/CEM.yaml', 'w') as f:
    f.write('# @package _group_\n')
    f.write(omegaconf.OmegaConf.to_yaml(optimizer_config))


optimizer_cfg = {
    "_target_": "mbrl.planning.ICEMOptimizer",
    "num_iterations": 20,
    "elite_ratio": 0.1,
    "population_size": 5000,
    "alpha": 0.1,
    "device": device,
    "lower_bound": "???",
    "upper_bound": "???",
    "return_mean_elites": True,
    "keep_elite_frac" : 0.1,
    "population_decay_factor":  2., 
    "colored_noise_exponent": 2,
    }

with open('REAI/configs/optimizer/ICEM.yaml', 'w') as f:
    f.write('# @package _group_\n')
    f.write(omegaconf.OmegaConf.to_yaml(optimizer_cfg))


optimizer_cfg = {
    "_target_": "mbrl.planning.MPPIOptimizer",
    "num_iterations": 5,
    "population_size": 350,
    "gamma" : 0.9, 
    "sigma" : 1.0,
    "beta" : 0.9,
    "lower_bound": "???",
    "upper_bound": "???",
    "device": device}

with open('REAI/configs/optimizer/MPPI.yaml', 'w') as f:
    f.write('# @package _group_\n')
    f.write(omegaconf.OmegaConf.to_yaml(optimizer_cfg))




agent_cfg= { "_target_": "mbrl.planning.TrajectoryOptimizerAgent",
            "planning_horizon": 15,
            "replan_freq": 1,
            "verbose": False,
            "action_lb": "???",
            "action_ub": "???",
            "optimizer_cfg": "???"
}
with open('REAI/configs/agent/TrajectoryOptimizerAgent.yaml', 'w') as f:
    f.write('# @package _group_\n')
    f.write(omegaconf.OmegaConf.to_yaml(agent_cfg))

dynamics_cfg = {
            "_target_": "mbrl.models.GaussianMLP",  # NOTE: important we are using a GAUSSIANMLP Here --
            "device": device,
            "num_layers": 3,
            "ensemble_size": 5,
            "hid_size": 200,
            "in_size": "???",
            "in_features": "???",
            "out_size": "???",
            "deterministic": False,  # probabilistic model
            "propagation_method": "fixed_model",
            # can also configure activation function for GaussianMLP
            "activation_fn_cfg": {"_target_": "torch.nn.LeakyReLU", "negative_slope": 0.01},
}

with open('REAI/configs/dynamics/GaussianMLP.yaml', 'w') as f:
    f.write('# @package _group_\n')
    f.write(omegaconf.OmegaConf.to_yaml(dynamics_cfg))

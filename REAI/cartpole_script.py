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

from REAI.physics_models import SINDyModel, CartpoleModel
from REAI.physics_models import trajectories_from_replay_buffer


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    seed = 0
    env = cartpole_env.CartPoleEnv()
    env.seed(seed)
    rng = np.random.default_rng(seed=0)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    # This functions allows the model to evaluate the true rewards given an observation
    reward_fn = reward_fns.cartpole

    # This function allows the model to know if an observation should make the episode end
    term_fn = termination_fns.cartpole

    trial_length = 200
    num_trials = 10
    ensemble_size = 5
    rendering = True

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

    cfg = omegaconf.OmegaConf.create(cfg_dict)

    # Create a 1-D dynamics model for this environment
    dynamics_model = common_util.create_one_dim_tr_model(cfg, obs_shape, act_shape)
    dynamics_model.model.physics_model = SINDyModel(backend='torch') # change backend to 'torch' to run on GPU
    
     # CartpoleModel() 
    dynamics_model.model.phys_nn_config = phys_nn_config

    # Create a gym-like environment to encapsulate the model
    model_env = models.ModelEnv(
        env, dynamics_model, term_fn, reward_fn, generator=generator
    )

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



    #check physics model

    def check_physics_model(replay_buffer):
        trajectories_list, action_list  = trajectories_from_replay_buffer(replay_buffer)
        test_trajectory = trajectories_list[1]
        test_actions = action_list[1]

        predicted_states = []
        predicted_states_own = []
        next_state = test_trajectory[0]
        for i in range(len(test_trajectory)):
            state = torch.tensor(test_trajectory[i])
            action = torch.tensor(test_actions[i])
            
            #predicting recursively (from its own prediction)
            predicted_states_own.append(np.array(dynamics_model.model.physics_model.predict(torch.tensor(next_state), action)))
            
            #predicting from actual state
            next_state = np.array(dynamics_model.model.physics_model.predict(state, action))
            predicted_states.append(next_state)

        predicted_states = np.array(predicted_states)
        predicted_states_own = np.array(predicted_states_own)

        plt.figure()
        state_dims = state.shape[0]
        for j in range(state_dims):
            plt.subplot(state_dims, 2, 2*j + 1)
            plt.plot( predicted_states[:-1, j] ,  label='model prediction from state')
            plt.plot( predicted_states_own[:-1, j] ,  label='model prediction recursive')        
            plt.plot( test_trajectory[1:, j],  label='true trajectory')
            if j== state_dims - 1:
                plt.legend()


            plt.subplot(state_dims, 2, 2 *j + 2)
            plt.plot( np.abs(predicted_states[:-1, j] - test_trajectory[1:, j])  ,  label='model prediction from state')
            plt.plot( np.abs(predicted_states_own[:-1, j]- test_trajectory[1:, j]) ,  label='model prediction recursive')        
            plt.title('Errors')
        plt.show()

    check_physics_model(replay_buffer)
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

    agent = planning.create_trajectory_optim_agent_for_model(
        model_env, agent_cfg, num_particles=20
    )

    train_losses = []
    val_scores = []

    def train_callback(_model, _total_calls, _epoch, tr_loss, val_score, _best_val):
        train_losses.append(tr_loss)
        val_scores.append(
            val_score.mean().item()
        )  # this returns val score per ensemble model


    def update_axes(
        _axs, _frame, _text, _trial, _steps_trial, _all_rewards, force_update=False
    ):
        if not force_update and (_steps_trial % 10 != 0):
            return
        _axs[0].imshow(_frame)
        _axs[0].set_xticks([])
        _axs[0].set_yticks([])
        _axs[1].clear()
        _axs[1].set_xlim([0, num_trials + 0.1])
        _axs[1].set_ylim([0, 200])
        _axs[1].set_xlabel("Trial")
        _axs[1].set_ylabel("Trial reward")
        _axs[1].plot(_all_rewards, "bs-")
        _text.set_text(f"Trial {_trial + 1}: {_steps_trial} steps")
        display.display(plt.gcf())
        display.clear_output(wait=True)


    a = np.array(np.arange(45).reshape(5, 3, 3))
    # c,d,e,f = a[:-1]
    # print(c,d,e,f)
    # print(a[:,:,1:2])
    c = a[:, :, 1:2]
    d = a[:, :, 2:3]
    np.concatenate((c, d), axis=-1)


    # Create a trainer for the model
    model_trainer = models.ModelTrainer(dynamics_model, optim_lr=1e-3, weight_decay=5e-5)

    # Create visualization objects
    fig, axs = plt.subplots(1, 2, figsize=(14, 3.75), gridspec_kw={"width_ratios": [1, 1]})
    ax_text = axs[0].text(300, 50, "")

    # Main PETS loop
    all_rewards = [0]
    for trial in range(num_trials):
        obs = env.reset()
        agent.reset()

        done = False
        total_reward = 0.0
        steps_trial = 0
        if rendering:
            update_axes(
                axs, env.render(mode="rgb_array"), ax_text, trial, steps_trial, all_rewards
            )

        # dataset

        while not done:
            # --------------- Model Training -----------------
            if steps_trial == 0:
                dynamics_model.update_normalizer(
                    replay_buffer.get_all()
                )  # update normalizer stats

                # bootsrapped
                dataset_train, dataset_val = common_util.get_basic_buffer_iterators(
                    replay_buffer,
                    batch_size=cfg.overrides.model_batch_size,
                    val_ratio=cfg.overrides.validation_ratio,
                    ensemble_size=ensemble_size,
                    shuffle_each_epoch=True,
                    bootstrap_permutes=False,  # build bootstrap dataset using sampling with replacement
                )
                
                #Train the model (don't if only using physics model)
                if phys_nn_config != 3:
                    model_trainer.train(
                        dataset_train,
                        dataset_val=dataset_val,
                        num_epochs=50,
                        patience=50,
                        callback=train_callback,
                        silent=False,
                    )
            
            # Train SINDy model outside of gradient step, it is not differentiable
            if isinstance(dynamics_model.model.physics_model, SINDyModel):
                dynamics_model.model.physics_model.train(replay_buffer)

            # --- Doing env step using the agent and adding to model dataset ---
            next_obs, reward, done, _ = common_util.step_env_and_add_to_buffer(
                env, obs, agent, {}, replay_buffer
            )


            if rendering:
                update_axes(
                    axs,
                    env.render(mode="rgb_array"),
                    ax_text,
                    trial,
                    steps_trial,
                    all_rewards,
                    force_update=True,
                )

            obs = next_obs
            total_reward += reward
            steps_trial += 1

            if steps_trial == trial_length:
                break
        
        if isinstance(dynamics_model.model.physics_model, SINDyModel):
            dynamics_model.model.physics_model.train(replay_buffer)

        all_rewards.append(total_reward)
        print("Total reward:", total_reward)

        if rendering:
            update_axes(
                axs,
                env.render(mode="rgb_array"),
                ax_text,
                trial,
                steps_trial,
                all_rewards,
                force_update=True,
            )

    fig, ax = plt.subplots(2, 1, figsize=(12, 10))
    ax[0].plot(train_losses)
    ax[0].set_xlabel("Total training epochs")
    ax[0].set_ylabel("Training loss (avg. NLL)")
    ax[1].plot(val_scores)
    ax[1].set_xlabel("Total training epochs")
    ax[1].set_ylabel("Validation score (avg. MSE)")
    plt.show()

    if not rendering:
        fig, ax = plt.subplots(1, 1)
        ax.set_xlim([0, num_trials + 0.1])
        ax.set_ylim([0, 200])
        ax.set_xlabel("Trial")
        ax.set_ylabel("Trial reward")
        ax.plot(all_rewards, "bs-")
        plt.show()
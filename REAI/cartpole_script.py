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
from REAI.utils import check_physics_model

import logging
from omegaconf import DictConfig, OmegaConf
import hydra
import pandas as pd

log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="main")
def run(exp_config : DictConfig):

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    log.info('Device : {}'.format(device))

    rendering  = False
    plotting = False

    seed=  exp_config['seed']
    trial_length = exp_config['trial_length']
    num_particles = exp_config['num_particles']
    num_trials = exp_config['num_trials']
    ensemble_size = exp_config['ensemble_size']

    cfg_dict = {
        'dynamics_model' : exp_config['dynamics'], 
        # options for training the dynamics model
        "algorithm": {
            "learned_rewards": False,
            "target_is_delta": True,
            "normalize": True,
        },
        # these are experiment specific options
        "overrides": {
            "trial_length": exp_config['trial_length'],
            "num_steps": exp_config['num_trials'] * exp_config['trial_length'],
            "model_batch_size": 32,
            "validation_ratio": 0.05,
        },
    }
    env_cfg = exp_config['env']#.to_container()

    #     env_cfg = {
    #     'gravity': 9.8,
    #     'masscart': 1.0,
    #     'masspole': 0.1,
    #     'length': 0.5,
    #     'force_mag': 10.0,
    #     'track_friction': 0.0,
    #     'joint_friction': 0.0,
    #     # Standard deviation on the observation of the states
    #     # DOES NOT AFFECT INTERNAL STATE PROPOGATION 
    #     # can be scalar or 1-d array
    #     # 'obs_noise': [0.003, 0.01, 0.003, 0.01], 
    #     'obs_noise': 0.0, 
    #    }

    #agent_cfg["optimizer_cfg"] =  exp_config['optimizer'].to_container()
    OmegaConf.update(exp_config['agent'], 'optimizer_cfg', exp_config['optimizer'])
    agent_cfg = omegaconf.OmegaConf.create(exp_config['agent'])

    env = cartpole_env.CartPoleEnv(**env_cfg)
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


    phys_nn_config = exp_config['phys_nn_config']
    physics_model = exp_config['physics_model']



    if phys_nn_config == 2:
        cfg_dict["dynamics_model"]["in_features"] = 2 * obs_shape[0] + (act_shape[0] if act_shape else 1)
        log.info('overriding in_features to {}'.format(cfg_dict["dynamics_model"]["in_features"]))

    if phys_nn_config == 3:
        cfg_dict["dynamics_model"]["deterministic"] = True
        log.info('overriding deterministic to True')

    cfg = omegaconf.OmegaConf.create(cfg_dict)

    # Create a 1-D dynamics model for this environment
    dynamics_model = common_util.create_one_dim_tr_model(cfg, obs_shape, act_shape)
    dynamics_model.model.phys_nn_config = phys_nn_config


    if physics_model == 'sindy':
        dynamics_model.model.physics_model = SINDyModel(**exp_config['model_kwargs']) # change backend to 'torch' to run on GPU

    elif physics_model == 'cartpole':
        dynamics_model.model.physics_model = CartpoleModel(**exp_config['model_kwargs'])



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
    #check_physics_model(replay_buffer, dynamics_model.model.physics_model)
    log.info("num stored: {}".format(replay_buffer.num_stored))
    log.info("# samples stored: {}".format(replay_buffer.num_stored))

    agent = planning.create_trajectory_optim_agent_for_model(
        model_env, agent_cfg, num_particles=num_particles
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


    # Create a trainer for the model
    model_trainer = models.ModelTrainer(dynamics_model, optim_lr=1e-3, weight_decay=5e-5)

    # Create visualization objects
    if plotting:
        fig, axs = plt.subplots(1, 2, figsize=(14, 3.75), gridspec_kw={"width_ratios": [1, 1]})
        ax_text = axs[0].text(300, 50, "")

    # Main PETS loop
    all_rewards = [0]
    for trial in range(num_trials):

        try:
            obs = env.reset()
            agent.reset()

            done = False
            total_reward = 0.0
            steps_trial = 0
            if rendering:
                update_axes(
                    axs, env.render(mode="rgb_array"), ax_text, trial, steps_trial, all_rewards
                )

            while not done:
                # --------------- Model Training -----------------
                if steps_trial == 0:
                    dynamics_model.update_normalizer(
                        replay_buffer.get_all()
                    )  # update normalizer stats

                    dynamics_model.model.physics_model.inputs_normalizer = dynamics_model.input_normalizer

                    # bootsrapped
                    dataset_train, dataset_val = common_util.get_basic_buffer_iterators(
                        replay_buffer,
                        batch_size=cfg.overrides.model_batch_size,
                        val_ratio=cfg.overrides.validation_ratio,
                        ensemble_size=ensemble_size,
                        shuffle_each_epoch=True,
                        bootstrap_permutes=False,  # build bootstrap dataset using sampling with replacement
                    )
                    if phys_nn_config != 3:
                        model_trainer.train(
                            dataset_train,
                            dataset_val=dataset_val,
                            num_epochs=150,
                            patience=50,
                            callback=train_callback,
                            silent=False,
                        )

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
            
            #retrain the physics model after each trial
            if isinstance(dynamics_model.model.physics_model, SINDyModel):
                dynamics_model.model.physics_model.train(replay_buffer)

            all_rewards.append(total_reward)
            log.info("Total reward: {}".format(total_reward))

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
        except Exception as e:
            # Log the error message to a file
            log.info("Error in trial {}".format(trial))
            log.error(str(e))
            
    if plotting:
        fig, ax = plt.subplots(2, 1, figsize=(12, 10))
        ax[0].plot(train_losses)
        ax[0].set_xlabel("Total training epochs")
        ax[0].set_ylabel("Training loss (avg. NLL)")
        ax[1].plot(val_scores)
        ax[1].set_xlabel("Total training epochs")
        ax[1].set_ylabel("Validation score (avg. MSE)")
        plt.show()

    if not rendering:
        if plotting:
            fig, ax = plt.subplots(1, 1)
            ax.set_xlim([0, num_trials + 0.1])
            ax.set_ylim([0, 200])
            ax.set_xlabel("Trial")
            ax.set_ylabel("Trial reward")
            ax.plot(all_rewards, "bs-")
            plt.show()

        #mean_and_logvar
        # 
    results  = np.array(all_rewards).reshape(-1, 1)
    df = pd.DataFrame({
    "optimizer": [exp_config['optimizer']['_target_'].split('.')[-1][:-9]],
    "seed": [seed],
    "phys_nn_config": [phys_nn_config],
    "rewards": [results],})

    # Save the DataFrame to a file
    df.to_csv("rewards.csv", index=False)


    return all_rewards

if __name__ == "__main__":
    run()
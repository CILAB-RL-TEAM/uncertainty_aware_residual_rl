#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import numpy as np
import pickle as pkl
import tqdm
from absl import app, flags
from flax.training import checkpoints

from serl_launcher.agents.continuous.assymetric_sac import SACAgent
from serl_launcher.common.evaluation import evaluate_with_stats
from serl_launcher.utils.timer_utils import Timer

import torch

from serl_launcher.utils.launcher import (
    make_wandb_logger,
    make_sac_agent,
)
from serl_launcher.data.data_store import ReplayBufferDataStore

import robosuite as suite
from robosuite.controllers import load_controller_config
from serl_launcher.wrappers.robosuite_state import RobosuiteStateGym
from serl_launcher.wrappers.mujoco_state import MujocoStateGym

#from octo.model.octo_model import OctoModel
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

import copy
import os
import random
import hydra
from collections import namedtuple
from omegaconf import OmegaConf
import gym
import d4rl
from gym.envs import make as make_gym_env_

Batch = namedtuple("Batch", "actions conditions")


FLAGS = flags.FLAGS

flags.DEFINE_string("env", "RobomimicState", "Category of environment.")
flags.DEFINE_string("env_name", "Lift", "Name of specific environment.")
flags.DEFINE_string("agent", "sac", "Name of agent.")
flags.DEFINE_string("exp_name", None, "Name of the experiment for wandb logging.")
flags.DEFINE_integer("max_traj_length", 100, "Maximum length of trajectory.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("batch_size", 4, "Batch size.")
flags.DEFINE_integer("utd_ratio", 4, "UTD ratio.")
flags.DEFINE_integer("max_steps", 10000, "Maximum number of training steps.")
flags.DEFINE_integer("replay_buffer_capacity", 20000, "Replay buffer capacity.")
flags.DEFINE_integer("learner_update", 2, "Number of learner steps per env step.")
flags.DEFINE_integer("log_period", 10, "Logging period.")
flags.DEFINE_integer("eval_period", 2000, "Evaluation period.")
flags.DEFINE_integer("eval_n_trajs", 5, "Number of trajectories for evaluation.")
flags.DEFINE_boolean("render", False, "Render the environment.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_string("robomimic_checkpoint_path", None, "Path to robomimic checkpoints to load the environment.")
flags.DEFINE_float("uncertainty_threshold", 1e-6, "Threshold for the uncertainty metric")
flags.DEFINE_float("uncertainty_decay_rate", 3e5, "Decay rate for the uncertainty metric")
flags.DEFINE_integer("burn_in_steps", 10000, "Number of steps to burn in")
flags.DEFINE_string("wandb_group_name", "test_group", "group for the wandb run")
flags.DEFINE_string("data_file", "", "data file to compute the distance to data metric")
flags.DEFINE_string("normalization_path", None, "Normalization file for diffusion policy")
flags.DEFINE_string("diffuision_config_path", None, "Config file for diffusion policy")
flags.DEFINE_integer("normalize_reward", 1, "If we want to normalize reward for kitchen envs")
flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  

devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)

def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def train_loop(agent: SACAgent, replay_buffer, env, eval_env, sampling_rng, wandb_logger):
    """Main training loop for uncertainty-aware residual reinforcement learning.
    
    This loop combines a base diffusion policy with a learned residual policy.
    The residual policy is applied based on distance to training data distribution.
    Uses exponentially decaying uncertainty thresholds over training steps.
    
    Args:
        agent: SAC agent for learning the residual policy.
        replay_buffer: Replay buffer for storing transitions.
        env: Training environment.
        eval_env: Evaluation environment.
        sampling_rng: JAX random number generator for sampling actions.
        wandb_logger: Logger for weights and biases experiment tracking.
    """

    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size,
        },
        device=sharding.replicate(),
    )
    
    update_info = None 
    critics_info = None

    #Load base policy from dppo diffusion model
    cfg = OmegaConf.load(FLAGS.diffuision_config_path)
    diffusion_policy = hydra.utils.instantiate(cfg.model)
    normalization = np.load(FLAGS.normalization_path)
    obs_min = normalization["obs_min"]
    obs_max = normalization["obs_max"]
    action_min = normalization["action_min"]
    action_max = normalization["action_max"]
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    ## Taken from dppo codebase
    def normalize_obs(obs):
        obs = 2 * (
            (obs - obs_min) / (obs_max - obs_min + 1e-6) - 0.5
        )
        return obs

    def unnormalize_action(action):
        action = (action + 1) / 2  # [-1, 1] -> [0, 1]
        return action * (action_max - action_min) + action_min

    
    ## Load data for distance to data metric
    training_data = np.load(FLAGS.data_file)
    final_data = training_data["states"]

    def get_base_action(obs):
        """Get action from diffusion policy given observation.
        
        Args:
            obs: Current observation from the environment.
            
        Returns:
            Action sampled from the diffusion policy.
        """
        ## Get action from the diffusion policy
        state = normalize_obs(obs)
        cond = {"state": torch.from_numpy(state).float().to(device)}
        cond["state"] = torch.unsqueeze(cond["state"], 0)
        cond["state"] = torch.unsqueeze(cond["state"], 0)
        samples = diffusion_policy(cond=cond, deterministic=True)
        act = samples.trajectories[0,0]
        act = torch.Tensor.cpu(act)
        act = act.numpy()
        base_action = unnormalize_action(act)
        return base_action
    

    obs, _ = env.reset()
    next_obs = obs
    done = False

    #training loop
    timer = Timer()
    running_return = 0.0
    next_base_action = None
  
    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True):
        timer.tick("total")
        ## Set exponentially decaying threshold
        threshold = FLAGS.uncertainty_threshold * np.exp(-step/FLAGS.uncertainty_decay_rate)

        with timer.context("sample_actions"):
            #get base action  
            base_action = get_base_action(obs)

            #We initially run only the base policy without any residual for some "burn in" steps.
            if step < FLAGS.burn_in_steps:
                policy_actions = np.zeros(base_action.shape)
                actions = base_action
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                policy_actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    actions=jax.device_put(base_action),
                    seed=key,
                    deterministic=False,
                )
                policy_actions = np.asarray(jax.device_get(policy_actions))

                ## Calculate distance to data metric
                dist = np.square(final_data-obs)
                dist = np.sum(dist, axis=1)
                dist = dist/dist.shape[0]
                ## Select residual action for uncertain states
                if np.min(dist) > threshold:
                    actions = policy_actions + base_action
                else:
                    actions =  base_action

        #Step environment
        with timer.context("step_env"):
            next_obs, reward, done, truncated, info = env.step(actions)
            reward = np.asarray(reward, dtype=np.float32)
            info = np.asarray(info)
            running_return += reward
            next_base_action = get_base_action(next_obs)
            
            transition = dict(
                observations=obs,
                actions=actions, ## Critic learns with the actions taken in the environment
                base_actions=base_action,
                next_base_actions=next_base_action,
                next_base_log_prob=0,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=done,
            )
            
            replay_buffer.insert(transition)

            obs = next_obs
            if done or truncated:
                running_return = 0.0
                obs, _ = env.reset()
                next_base_action = None
                diffusion_policy.eval()
        
        ## Actor-Critic Update
        with timer.context("learner"):
            for learner_step in range(FLAGS.learner_update):
                for critic_step in range(FLAGS.utd_ratio - 1):
                    with timer.context("sample_replay_buffer"):
                        batch = next(replay_iterator)

                    with timer.context("train_critics"):
                        agent, critics_info = agent.update_critics_state(
                            batch,
                        )
                        agent = jax.block_until_ready(agent)

                ## We only update the critics during the burn in period, after which we update both actor and critics
                if step > FLAGS.burn_in_steps:
                    with timer.context("train_actor"):
                        batch = next(replay_iterator)
                        agent, update_info = agent.update_high_utd(batch, utd_ratio=1)
                        agent = jax.block_until_ready(agent)
        

        ## For evaluation
        def residual_action(obs):
            """Compute final action using residual policy based on distance to training data.
            
            If observation is close to training data distribution, uses base policy only.
            Otherwise, adds residual policy action to base action.
            
            Args:
                obs: Current observation from the environment.
                
            Returns:
                Tuple of (action, base_action_used) where action is the final action
                to take and base_action_used is 1 if only base policy was used, 0 otherwise.
            """
            base_action = get_base_action(obs)
            
            policy_actions = agent.sample_actions(
                observations=jax.device_put(obs), 
                actions=jax.device_put(base_action),
                argmax=True,
            )
            policy_actions = np.asarray(jax.device_get(policy_actions))

            ##Same conditional logic as above for calculating final action
            dist = np.square(final_data-obs)
            dist = np.sum(dist, axis=1)
            dist = dist/dist.shape[0]
            if np.min(dist) > threshold:
                actions = policy_actions + base_action
                base_action_used = 0
            else:
                actions =  base_action
                base_action_used = 1
            
            return actions, base_action_used

        if step % FLAGS.eval_period == 0:
            with timer.context("eval"):
                evaluate_info = evaluate_with_stats(
                    policy_fn=residual_action,
                    env=eval_env,
                    num_episodes=FLAGS.eval_n_trajs,
                    base_policy=diffusion_policy, 
                    normalize_reward=FLAGS.normalize_reward
                )
            stats = {"eval_env_steps": evaluate_info, "env_step":step, "threshold":threshold}
            wandb_logger.log(stats, step=step)
            checkpoints.save_checkpoint(
                FLAGS.checkpoint_path, agent.state, step=step, keep=100
            )

        timer.tock("total")

        if step % FLAGS.log_period == 0:
            if update_info:
                wandb_logger.log(update_info, step=step)
            elif critics_info:
                wandb_logger.log(critics_info, step=step)

            stats = {"timer": timer.get_average_times()}
            wandb_logger.log(stats, step=step)
            action_stats = {}
            for i in range(base_action.shape[0]):
                action_stats[f"base_action_{i}"] = base_action[i]
                action_stats[f"policy_action_{i}"] = policy_actions[i]
                action_stats[f"final_action_{i}"] = actions[i]

            stats = {"actor_actions": action_stats}
            wandb_logger.log(stats, step=step)
        


##############################################################################


def main(_):
    """Main entry point for training uncertainty-aware residual RL agent.
    
    Initializes the environment, agent, and replay buffer, then runs the training loop.
    """
    print("pid ", os.getpid())
    assert FLAGS.batch_size % num_devices == 0
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)

    # create env
    if FLAGS.env == "MujocoState":
        mj_env = make_gym_env_(FLAGS.env_name)
        env = MujocoStateGym(env=mj_env)
        mj_eval_env = make_gym_env_(FLAGS.env_name)
        eval_env = MujocoStateGym(env=mj_eval_env)
    
    elif FLAGS.env == "RobomimicState":
        torch_device = TorchUtils.get_torch_device(try_to_use_cuda=True)

        path = FLAGS.robomimic_checkpoint_path

        _, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=path, device=torch_device, verbose=True)
        rm_env, _ = FileUtils.env_from_checkpoint(
            ckpt_dict=ckpt_dict, 
            render=FLAGS.render, 
            render_offscreen=True, 
            verbose=True,
        )
        rm_env.env.ignore_done = (
            False  # Fix a hardcoded ignore_done=True in robomimic env init
        )
        rm_env.env.reward_shaping = (
            False  #Reward shaping should be false for non-eval env
        )
        rm_env.env.horizon = (
            FLAGS.max_traj_length  #Changing horizon of the env
        )
        suite_env = rm_env.env
        env = RobosuiteStateGym(suite_env)

        ##Make eval env
        _, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=path, device=torch_device, verbose=True)
        eval_rm_env, _ = FileUtils.env_from_checkpoint(
            ckpt_dict=ckpt_dict, 
            render=FLAGS.render, 
            render_offscreen=True, 
            verbose=True,
        )
        eval_rm_env.env.ignore_done = (
            False  # Fix a hardcoded ignore_done=True in robomimic env init
        )
        eval_rm_env.env.reward_shaping = (
            True  #Reward shaping should be true for eval env
        )
        eval_rm_env.env.horizon = (
            FLAGS.max_traj_length  #Changing horizon of the env
        )
        eval_suite_env = eval_rm_env.env
        eval_env = RobosuiteStateGym(eval_suite_env)


    agent: SACAgent = make_sac_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
    )
    
    #Zero out the last layer of the actor network for residual policy
    shape_last_layer = agent.state.params["modules_actor"]["Dense_1"]["kernel"].shape
    agent.state.params["modules_actor"]["Dense_0"]["kernel"] =  jnp.zeros(shape_last_layer)
        
    rng, sampling_rng = jax.random.split(rng)

    agent: SACAgent = jax.device_put(
        jax.tree_map(jnp.array, agent), sharding.replicate()
    )


    def create_replay_buffer_and_wandb_logger():
        """Create replay buffer and wandb logger for experiment tracking.
        
        Returns:
            Tuple of (replay_buffer, wandb_logger) for storing transitions and logging metrics.
        """
        replay_buffer = ReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=FLAGS.replay_buffer_capacity,
        )
        # set up wandb and logging
        wandb_logger = make_wandb_logger(
            project="uarl",
            description=FLAGS.exp_name or FLAGS.env,
            debug=FLAGS.debug,
            group=FLAGS.wandb_group_name
        )
        return replay_buffer, wandb_logger
    
    sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
    replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()

   
    print_green("replay buffer created")
    print_green(f"replay_buffer size: {len(replay_buffer)}")
    
    
    train_loop(agent, replay_buffer, env, eval_env, sampling_rng, wandb_logger)

if __name__ == "__main__":
    app.run(main)
import copy
from typing import OrderedDict

import gym
import numpy as np
from gym import spaces
import cv2


class MujocoStateGym(gym.core.Env):
    #metadata = {"render_modes": ["rgb_array"]}
    metadata = None
    render_mode = None

    def __init__(self, env, keys=None):
        # Run super method
        self.env = env
        # set up observation and action spaces
        self.action_space = env.action_space
        
        self.observation_space = spaces.Dict()
        obs_example = self.env.reset()
        low = np.full_like(obs_example, fill_value=-1)
        high = np.full_like(obs_example, fill_value=1)
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype,
        )



    def reset(self, seed=None, options=None):
        """
        Extends env reset method to return flattened observation instead of normal OrderedDict and optionally resets seed

        Returns:
            np.array: Flattened environment observation space after reset occurs
        """
        obs = self.env.reset()
        return obs, {}
    
    def step(self, action):
        """
        Extends vanilla step() function call to return flattened observation instead of normal OrderedDict.

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (np.array) fslattened observations from the environment
                - (float) reward from the environment
                - (bool) episode ending after reaching an env terminal state
                - (bool) episode ending after an externally defined condition
                - (dict) misc information
        """
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        obs, reward, done, info = self.env.step(action)
        truncated = False
      
        return obs, reward, done, truncated, info

    def render(self):
        self.env.render()

    def is_success(self):
        self.env._check_success()

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Dummy function to be compatible with gym interface that simply returns environment reward

        Args:
            achieved_goal: [NOT USED]
            desired_goal: [NOT USED]
            info: [NOT USED]

        Returns:
            float: environment reward
        """
        # Dummy args used to mimic Wrapper interface
        return self.env.reward()

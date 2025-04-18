## From https://levelup.gitconnected.com/drawing-and-coding-18-rl-algorithms-from-scratch-714ec2f581e5

# Cloning and navigating to dir
git clone https://github.com/fareedkhan-dev/all-rl-algorithms.git
cd all-rl-algorithms

# Installing the required dependencies
pip install -r requirements.txt

# --- Core Python Libraries ---
import random
import math
from collections import defaultdict, deque, namedtuple
from typing import List, Tuple, Dict, Optional, Any, DefaultDict # For type hinting used in the code

# --- Numerical Computation ---
import numpy as np

# --- Machine Learning Framework (PyTorch - used extensively from REINFORCE onwards) ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal # Used in policy gradients, SAC, PlaNet etc.

# --- Environment ---
# For loading standard environments like Pendulum
import gymnasium as gym
# Note: The SimpleGridWorld class definition needs to be included directly in the code
# as it's a custom environment defined in the blog post.

# --- Visualization (Implied by the plots shown in the blog) ---
import matplotlib.pyplot as plt
import seaborn as sns # Often used for heatmaps

# --- Potentially for Asynchronous Methods (A3C) ---
# Although not explicitly shown in snippets, A3C implementations often use these
# import torch.multiprocessing as mp # Or standard 'multiprocessing'/'threading'

# --- Setup for PyTorch (Optional but good practice) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Disable Warnings (Optional) ---
import warnings
warnings.filterwarnings('ignore') # To suppress potential deprecation warnings etc.

# -------------------------------------
# 1. Simple Custom Grid World
# -------------------------------------

class SimpleGridWorld:
    """ A basic grid world environment. """
    def __init__(self, size=5):
        self.size = size
        self.start_state = (0, 0)
        self.goal_state = (size - 1, size - 1)
        self.state = self.start_state
        # Actions: 0:Up, 1:Down, 2:Left, 3:Right
        self.action_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        self.action_space_size = 4

    def reset(self) -> Tuple[int, int]:
        """ Resets to start state. """
        self.state = self.start_state
        return self.state

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """ Takes an action, returns next_state, reward, done. """
        if self.state == self.goal_state:
            return self.state, 0.0, True # Stay at goal

        # Calculate potential next state
        dr, dc = self.action_map[action]
        r, c = self.state
        next_r, next_c = r + dr, c + dc

        # Apply boundaries (stay in place if hitting wall)
        if not (0 <= next_r < self.size and 0 <= next_c < self.size):
            next_r, next_c = r, c # Stay in current state
            reward = -1.0         # Wall penalty
        else:
            reward = -0.1         # Step cost

        # Update state
        self.state = (next_r, next_c)

        # Check if goal reached
        done = (self.state == self.goal_state)
        if done:
            reward = 10.0         # Goal reward

        return self.state, reward, done

# -------------------------------------
# 2. Loading Gymnasium Pendulum
# -------------------------------------

pendulum_env = gym.make('Pendulum-v1')
print("Pendulum-v1 environment loaded.")

# Reset environment
observation, info = pendulum_env.reset(seed=42)
print(f"Initial Observation: {observation}")
print(f"Observation Space: {pendulum_env.observation_space}")
print(f"Action Space: {pendulum_env.action_space}")

# Take a random step
random_action = pendulum_env.action_space.sample()
observation, reward, terminated, truncated, info = pendulum_env.step(random_action)
done = terminated or truncated
print(f"Step with action {random_action}:")
print(f"  Next Obs: {observation}\n  Reward: {reward}\n  Done: {done}")

# Close environment (important if rendering was used)
pendulum_env.close()



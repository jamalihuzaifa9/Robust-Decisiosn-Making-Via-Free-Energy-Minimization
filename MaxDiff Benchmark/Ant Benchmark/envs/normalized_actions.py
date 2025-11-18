#!/usr/bin/env python3

import numpy as np

# Try gymnasium first, fallback to gym
try:
    import gymnasium as gym
    from gymnasium import ActionWrapper
    USE_GYMNASIUM = True
except ImportError:
    import gym
    from gym import ActionWrapper
    USE_GYMNASIUM = False


class NormalizedActions(ActionWrapper):
    """Wrapper to normalize action space to [-1, 1]."""
    
    def __init__(self, env):
        super().__init__(env)
        
        # Get original action space bounds
        self._original_action_space = env.action_space
        low_bound = self._original_action_space.low
        upper_bound = self._original_action_space.high
        
        # Create normalized action space [-1, 1]
        if USE_GYMNASIUM:
            self.action_space = gym.spaces.Box(
                low=np.full_like(low_bound, -1.0, dtype=np.float32),
                high=np.full_like(upper_bound, 1.0, dtype=np.float32),
                dtype=np.float32
            )
        else:
            self.action_space = gym.spaces.Box(
                low=np.full_like(low_bound, -1.0),
                high=np.full_like(upper_bound, 1.0)
            )

    def action(self, action):
        """Transform normalized action [-1, 1] to original action space."""
        low_bound = self._original_action_space.low
        upper_bound = self._original_action_space.high

        # Map from [-1, 1] to [low_bound, upper_bound]
        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)

        return action

    def reverse_action(self, action):
        """Transform action from original space to normalized space [-1, 1]."""
        low_bound = self._original_action_space.low
        upper_bound = self._original_action_space.high

        # Map from [low_bound, upper_bound] to [-1, 1]
        action = 2.0 * (action - low_bound) / (upper_bound - low_bound) - 1.0
        action = np.clip(action, -1.0, 1.0)

        return action
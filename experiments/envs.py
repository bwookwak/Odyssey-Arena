"""
Environment wrappers for Odyssey-Arena tasks.

Currently implements LightEnv wrapper with normalized observations.
Designed to be easily extended to other Odyssey-Arena environments.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from LightEnv.TextEnv_v2 import LightBulbEnv


class LightEnvWrapper:
    """
    Wrapper for LightBulbEnv with configurable observation format.
    
    Observation formats:
    - 'bitstring': "010101" (string of 0s and 1s) - default
    - 'emoji': "💡 ○ 💡 ○ 💡 ○" (emojis, compatible with original Odyssey-Arena)
    
    Action: integer in [0, num_bulbs)
    done: True when all bulbs are on (success)
    """
    
    def __init__(self, custom_logic, num_bulbs, seed=None, obs_format='bitstring'):
        """
        Args:
            custom_logic: Dict mapping bulb names to boolean expressions
            num_bulbs: Number of bulbs (level)
            seed: Random seed for environment
            obs_format: Observation format ('bitstring' or 'emoji')
        """
        self.env = LightBulbEnv(
            num_bulbs=num_bulbs,
            custom_logic=custom_logic,
            seed=seed,
            expose_logic=False
        )
        self.num_bulbs = num_bulbs
        self.custom_logic = custom_logic
        self.obs_format = obs_format
        
    def reset(self):
        """Reset environment and return initial observation."""
        obs_list = self.env.reset()
        return self._normalize_obs(obs_list)
    
    def step(self, action):
        """
        Take action and return (obs, feedback, done, info).
        
        Returns:
            obs: Normalized bit-string observation
            feedback: Text feedback from environment
            done: Whether task is complete (success)
            info: Additional information dict
        """
        obs_list, hint, done, info = self.env.step(action)
        obs = self._normalize_obs(obs_list)
        
        # Calculate progress (number of bulbs that are on)
        progress = sum(obs_list)
        info['progress'] = progress
        
        return obs, hint, done, info
    
    def _normalize_obs(self, obs_list):
        """
        Convert boolean list to string based on obs_format.
        
        - bitstring: [True, False, True] -> '101'
        - emoji: [True, False, True] -> '💡 ○ 💡'
        """
        if self.obs_format == 'emoji':
            return ' '.join('💡' if b else '○' for b in obs_list)
        else:  # bitstring
            return ''.join('1' if b else '0' for b in obs_list)
    
    def get_num_actions(self):
        """Return number of valid actions."""
        return self.num_bulbs
    
    def get_custom_logic(self):
        """Return custom logic (ground truth rules) for oracle verification."""
        return self.custom_logic


def load_light_tasks(task_file):
    """
    Load LightEnv tasks from JSON file.
    
    Args:
        task_file: Path to task JSON file
        
    Returns:
        List of task dicts with 'custom_logic' and 'level' keys
    """
    import json
    with open(task_file, 'r') as f:
        tasks = json.load(f)
    return tasks


def create_env(env_type, task_data, seed=None, obs_format='bitstring'):
    """
    Factory function to create environment wrappers.
    
    Args:
        env_type: Environment type ('light', 'energy', 'repo', 'trade')
        task_data: Task configuration dict
        seed: Random seed
        obs_format: Observation format ('bitstring' or 'emoji')
        
    Returns:
        Environment wrapper instance
    """
    if env_type == 'light':
        return LightEnvWrapper(
            custom_logic=task_data['custom_logic'],
            num_bulbs=task_data['level'],
            seed=seed,
            obs_format=obs_format
        )
    else:
        raise NotImplementedError(f"Environment {env_type} not yet implemented")

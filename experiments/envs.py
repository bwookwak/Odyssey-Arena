"""
Environment wrappers for Odyssey-Arena tasks.

Provides a common interface (BaseEnvWrapper) so that metrics and runners
can rely on info['progress'] and get_ground_truth_for_verification().
Currently implements LightEnv wrapper with normalized observations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from LightEnv.TextEnv_v2 import LightBulbEnv


class BaseEnvWrapper:
    """
    Base interface for environment wrappers.
    
    Subclasses must implement:
    - reset() -> obs
    - step(action) -> (obs, feedback, done, info)
    - get_num_actions() -> int
    info must include 'progress' (numeric) for loop/stagnation metrics.
    get_ground_truth_for_verification() returns env-specific ground truth or None.
    """
    
    def reset(self):
        raise NotImplementedError
    
    def step(self, action):
        raise NotImplementedError
    
    def get_num_actions(self) -> int:
        raise NotImplementedError
    
    def get_ground_truth_for_verification(self):
        """
        Return environment-specific ground truth for hypothesis verification, or None.
        E.g. for Light: custom_logic dict; for other envs: None or different structure.
        """
        return None


class LightEnvWrapper(BaseEnvWrapper):
    """
    Wrapper for LightBulbEnv with configurable observation format.
    
    Observation formats:
    - 'bitstring': "010101" (string of 0s and 1s) - default
    - 'emoji': "💡 ○ 💡 ○ 💡 ○" (emojis, compatible with original Odyssey-Arena)
    
    Action: integer in [0, num_bulbs)
    done: True when current state matches goal_state (success)
    goal_state: list[bool] target state. None defaults to all-True (original task).
    """
    
    def __init__(self, custom_logic, num_bulbs, seed=None, obs_format='bitstring', goal_state=None):
        """
        Args:
            custom_logic: Dict mapping bulb names to boolean expressions
            num_bulbs: Number of bulbs (level)
            seed: Random seed for environment
            obs_format: Observation format ('bitstring' or 'emoji')
            goal_state: Target state as list[bool]. None → all True (original task).
        """
        self.env = LightBulbEnv(
            num_bulbs=num_bulbs,
            custom_logic=custom_logic,
            seed=seed,
            expose_logic=False,
            goal_state=goal_state,
        )
        self.num_bulbs = num_bulbs
        self.custom_logic = custom_logic
        self.obs_format = obs_format
        # goal_state as list[bool] (resolved inside LightBulbEnv)
        self.goal_state = self.env.goal_state
        # goal_state as normalized string (same format as observations)
        self.goal_state_str = self._normalize_obs(self.goal_state)
        
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
        
        # progress: goal 상태와 일치하는 전구 수
        progress = sum(a == b for a, b in zip(obs_list, self.goal_state))
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
    
    def get_ground_truth_for_verification(self):
        """Standard interface: ground truth for verification (custom_logic for light)."""
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
            obs_format=obs_format,
            goal_state=task_data.get('goal_state', None),
        )
    else:
        raise NotImplementedError(f"Environment {env_type} not yet implemented")

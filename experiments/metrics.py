"""
Metrics for evaluating agent performance.

Computes per-episode and aggregate metrics:
- Success rate
- Loop ratio (repetitive behavior)
- Number of steps
- Memory usage rate
- Invalid action rate
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict


class EpisodeMetrics:
    """
    Tracks metrics for a single episode.
    """
    
    def __init__(self):
        self.steps = []
        self.success = False
        self.num_steps = 0
        
        # For loop detection
        self.seen_states = {}  # (obs, action) -> max_progress_seen
        self.loop_count = 0
        self.total_actions = 0
        
        # For memory usage
        self.memory_used_count = 0
        
        # For invalid actions
        self.invalid_action_count = 0
    
    def record_step(
        self,
        step: int,
        obs_before: str,
        action: Optional[int],
        obs_after: str,
        feedback: str,
        progress: int,
        llm_raw_output: str,
        memory_context_len: int,
        memory_used: bool,
        invalid_action: bool,
        memory_context: str = "",
        model_input: str = "",
        model_output: str = "",
        memory_ops: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Record a single step.
        
        Args:
            step: Step number
            obs_before: Observation before action
            action: Action taken (or None if invalid)
            obs_after: Observation after action
            feedback: Environment feedback
            progress: Progress value from env info (e.g. bulbs ON for light env)
            llm_raw_output: Raw LLM output
            memory_context_len: Length of memory context string
            memory_used: Whether memory was used this step
            invalid_action: Whether action was invalid
            memory_context: Actual memory context string shown to agent (for qualitative logs / slide 6)
            model_input: Full prompt / input sent to model (for logging and dashboard)
            model_output: Raw model output (same as llm_raw_output, for clarity next to model_input)
            memory_ops: Per-step memory operations (reflection, curation, verify, etc.) for dashboard
        """
        step_data = {
            'step': step,
            'obs_before': obs_before,
            'action': action,
            'obs_after': obs_after,
            'feedback': feedback,
            'progress': progress,
            'llm_raw_output': llm_raw_output,
            'memory_context_len': memory_context_len,
            'memory_used': memory_used,
            'invalid_action': invalid_action,
            'memory_context': memory_context,
            'model_input': model_input,
            'model_output': model_output or llm_raw_output,
            'memory_ops': memory_ops or [],
        }
        self.steps.append(step_data)
        self.num_steps += 1
        
        # Track loop behavior
        if action is not None:
            state_action = (obs_before, action)
            if state_action in self.seen_states:
                prev_max_progress = self.seen_states[state_action]
                if progress <= prev_max_progress:
                    self.loop_count += 1
            self.seen_states[state_action] = max(
                self.seen_states.get(state_action, -1),
                progress
            )
            self.total_actions += 1
        
        # Track memory usage
        if memory_used:
            self.memory_used_count += 1
        
        # Track invalid actions
        if invalid_action:
            self.invalid_action_count += 1
    
    def finalize(self, success: bool):
        """Mark episode as complete."""
        self.success = success
    
    def compute_metrics(self) -> Dict[str, Any]:
        """
        Compute episode-level metrics.
        
        Returns:
            Dict with:
            - success: bool
            - num_steps: int
            - loop_ratio: float
            - memory_usage_rate: float
            - invalid_action_rate: float
        """
        loop_ratio = self.loop_count / self.total_actions if self.total_actions > 0 else 0.0
        memory_usage_rate = self.memory_used_count / self.num_steps if self.num_steps > 0 else 0.0
        invalid_action_rate = self.invalid_action_count / self.num_steps if self.num_steps > 0 else 0.0
        
        return {
            'success': self.success,
            'num_steps': self.num_steps,
            'loop_ratio': loop_ratio,
            'memory_usage_rate': memory_usage_rate,
            'invalid_action_rate': invalid_action_rate,
            'loop_count': self.loop_count,
            'total_actions': self.total_actions
        }
    
    def get_steps(self) -> List[Dict[str, Any]]:
        """Get list of step data."""
        return self.steps


class AggregateMetrics:
    """
    Aggregates metrics across multiple episodes.
    """
    
    def __init__(self):
        self.episodes = []
    
    def add_episode(self, episode_metrics: Dict[str, Any]):
        """Add episode metrics to aggregate."""
        self.episodes.append(episode_metrics)
    
    def compute_aggregate(self) -> Dict[str, Any]:
        """
        Compute aggregate metrics.
        
        Returns:
            Dict with:
            - num_episodes: int
            - success_rate: float
            - avg_steps_all: float
            - avg_steps_success: float
            - avg_loop_ratio: float
            - avg_memory_usage_rate: float
            - avg_invalid_action_rate: float
        """
        if not self.episodes:
            return {
                'num_episodes': 0,
                'success_rate': 0.0,
                'avg_steps_all': 0.0,
                'avg_steps_success': 0.0,
                'avg_loop_ratio': 0.0,
                'avg_memory_usage_rate': 0.0,
                'avg_invalid_action_rate': 0.0
            }
        
        num_episodes = len(self.episodes)
        num_success = sum(1 for ep in self.episodes if ep['success'])
        success_rate = num_success / num_episodes
        
        avg_steps_all = sum(ep['num_steps'] for ep in self.episodes) / num_episodes
        
        success_episodes = [ep for ep in self.episodes if ep['success']]
        avg_steps_success = (
            sum(ep['num_steps'] for ep in success_episodes) / len(success_episodes)
            if success_episodes else 0.0
        )
        
        avg_loop_ratio = sum(ep['loop_ratio'] for ep in self.episodes) / num_episodes
        avg_memory_usage_rate = sum(ep['memory_usage_rate'] for ep in self.episodes) / num_episodes
        avg_invalid_action_rate = sum(ep['invalid_action_rate'] for ep in self.episodes) / num_episodes
        
        return {
            'num_episodes': num_episodes,
            'num_success': num_success,
            'success_rate': success_rate,
            'avg_steps_all': avg_steps_all,
            'avg_steps_success': avg_steps_success,
            'avg_loop_ratio': avg_loop_ratio,
            'avg_memory_usage_rate': avg_memory_usage_rate,
            'avg_invalid_action_rate': avg_invalid_action_rate
        }


def compute_summary_statistics(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute summary statistics from episode list.
    
    Args:
        episodes: List of episode metric dicts
        
    Returns:
        Summary statistics dict
    """
    agg = AggregateMetrics()
    for ep in episodes:
        agg.add_episode(ep)
    return agg.compute_aggregate()

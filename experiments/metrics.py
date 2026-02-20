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

        # For progress ratio (progress = number of goal-matching bulbs at each step)
        # _num_bulbs is the denominator; inferred from obs length on first step.
        self._progress_values: List[int] = []
        self._num_bulbs: int = 0
    
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
        invalid_output: str = "",
        progress_before: int = 0,
    ) -> bool:
        """
        Record a single step and return whether a loop was detected.

        Args:
            step: Step number
            obs_before: Observation before action
            action: Action taken (or None if invalid)
            obs_after: Observation after action
            feedback: Environment feedback
            progress: Progress value AFTER the action (from env info)
            progress_before: Progress value of obs_before, i.e. progress(state_bits)
                             Used for seen-max loop detection.
                             Caller should pass the previous step's progress value.
            llm_raw_output: Raw LLM output
            memory_context_len: Length of memory context string
            memory_used: Whether memory was used this step
            invalid_action: Whether action was invalid
            memory_context: Actual memory context string shown to agent
            model_input: Full prompt / input sent to model
            model_output: Raw model output
            memory_ops: Per-step memory operations for dashboard

        Returns:
            is_loop (bool): True if this step is classified as a loop by seen-max.

        Loop detection — seen-max algorithm:
            seen[(state_bits, action)] = max progress(state_bits) seen at each occurrence.
            A step is a loop when the same (state, action) pair was visited before AND the
            current progress p = progress(state_bits) is no higher than the best seen so far.
            Intuition: "you're repeating this pair without being in a better state."
        """
        # Infer num_bulbs from observation length (first step wins)
        if self._num_bulbs == 0 and obs_before:
            self._num_bulbs = len(obs_before)

        # ── Seen-max loop detection ────────────────────────────────────────────
        # p = progress(state_bits) = progress of obs_before (the entering state)
        is_loop = False
        if action is not None:
            p = progress_before
            key = (obs_before, action)
            if key in self.seen_states and p <= self.seen_states[key]:
                self.loop_count += 1
                is_loop = True
            self.seen_states[key] = max(self.seen_states.get(key, -1), p)
            self.total_actions += 1

        step_data = {
            'step': step,
            'obs_before': obs_before,
            'action': action,
            'obs_after': obs_after,
            'feedback': feedback,
            'progress': progress,
            'progress_before': progress_before,
            'is_loop': is_loop,
            'llm_raw_output': llm_raw_output,
            'memory_context_len': memory_context_len,
            'memory_used': memory_used,
            'invalid_action': invalid_action,
            'invalid_output': invalid_output,
            'memory_context': memory_context,
            'model_input': model_input,
            'model_output': model_output or llm_raw_output,
            'memory_ops': memory_ops or [],
        }
        self.steps.append(step_data)
        self.num_steps += 1

        # Track bulb-lit progress (after-action values)
        self._progress_values.append(progress)

        # Track memory usage
        if memory_used:
            self.memory_used_count += 1

        # Track invalid actions
        if invalid_action:
            self.invalid_action_count += 1

        return is_loop
    
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

        # Progress ratio (0.0–1.0): divide raw counts by total number of bulbs
        denom = self._num_bulbs if self._num_bulbs > 0 else 1
        raw_max = max(self._progress_values) if self._progress_values else 0
        raw_avg = (
            sum(self._progress_values) / len(self._progress_values)
            if self._progress_values else 0.0
        )
        max_progress = round(raw_max / denom, 4)
        avg_progress = round(raw_avg / denom, 4)

        return {
            'success': self.success,
            'num_steps': self.num_steps,
            'loop_ratio': loop_ratio,
            'memory_usage_rate': memory_usage_rate,
            'invalid_action_rate': invalid_action_rate,
            'loop_count': self.loop_count,
            'total_actions': self.total_actions,
            'max_progress': max_progress,
            'avg_progress': avg_progress,
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
                'avg_invalid_action_rate': 0.0,
                'avg_max_progress': 0.0,
                'avg_avg_progress': 0.0,
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

        # Bulb-lit progress ratio aggregates (episode values are already 0.0–1.0)
        avg_max_progress = (
            sum(ep.get('max_progress', 0.0) for ep in self.episodes) / num_episodes
        )
        avg_avg_progress = (
            sum(ep.get('avg_progress', 0.0) for ep in self.episodes) / num_episodes
        )

        return {
            'num_episodes': num_episodes,
            'num_success': num_success,
            'success_rate': success_rate,
            'avg_steps_all': avg_steps_all,
            'avg_steps_success': avg_steps_success,
            'avg_loop_ratio': avg_loop_ratio,
            'avg_memory_usage_rate': avg_memory_usage_rate,
            'avg_invalid_action_rate': avg_invalid_action_rate,
            'avg_max_progress': round(avg_max_progress, 4),
            'avg_avg_progress': round(avg_avg_progress, 4),
        }


def compute_pass_at_k(episodes_data: List[Dict[str, Any]], pass_k: int) -> Dict[str, Any]:
    """
    Compute pass@k statistics from episode data.

    Groups episodes by task_level (or task_idx if present) and checks whether
    at least one trial succeeded per task.

    Args:
        episodes_data: List of episode_data dicts (must contain 'task_level' and 'success')
        pass_k: Number of trials per task (k)

    Returns:
        Dict with:
        - pass_at_k: float  (fraction of tasks where ≥1 trial succeeded)
        - num_tasks: int
        - num_tasks_passed: int
        - pass_k: int
        - per_task: List[Dict] with task_level, trials_succeeded, passed
    """
    task_results: Dict[Any, List[bool]] = defaultdict(list)
    task_level_map: Dict[Any, Any] = {}

    for ep in episodes_data:
        # Use (task_idx, task_level) as key if task_idx present, else task_level alone
        task_idx = ep.get('task_idx')
        task_level = ep.get('task_level')
        key = task_idx if task_idx is not None else task_level
        task_results[key].append(bool(ep.get('success', False)))
        task_level_map[key] = task_level

    per_task = []
    for key, successes in task_results.items():
        passed = any(successes)
        per_task.append({
            'task_level': task_level_map[key],
            'trials_succeeded': sum(successes),
            'trials_total': len(successes),
            'passed': passed,
        })

    num_tasks = len(per_task)
    num_tasks_passed = sum(1 for t in per_task if t['passed'])
    pass_at_k_score = num_tasks_passed / num_tasks if num_tasks > 0 else 0.0

    return {
        'pass_at_k': pass_at_k_score,
        'pass_k': pass_k,
        'num_tasks': num_tasks,
        'num_tasks_passed': num_tasks_passed,
        'per_task': per_task,
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

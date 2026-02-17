"""
Main runner and CLI for mis-evolved memory experiments.

Usage:
    python -m experiments.run --env light --agent llm --memory gated --num_episodes 30
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import time

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.envs import load_light_tasks, create_env
from experiments.llm import create_llm_client
from experiments.agents import create_agent
from experiments.memory import create_memory_system
from experiments.interventions import create_intervention
from experiments.metrics import EpisodeMetrics, compute_summary_statistics


class ExperimentRunner:
    """
    Runs experiments with specified configuration.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize runner with configuration.
        
        Args:
            config: Configuration dict with experiment parameters
        """
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load tasks
        self.tasks = self._load_tasks()
        
        # Initialize LLM client (if needed)
        self.llm_client = None
        if config['agent'] == 'llm' or config['memory'] == 'reflector':
            self.llm_client = create_llm_client(
                provider=config.get('provider', 'vllm'),
                model_name=config['model'],
                api_key=config.get('api_key'),
                use_dummy=config.get('use_dummy_llm', False)
            )
        
        # Results storage
        self.episodes_data = []
        self.episode_metrics = []
    
    def _load_tasks(self):
        """Load tasks from file."""
        env_type = self.config['env']
        
        if env_type == 'light':
            task_file = self.config.get('task_file')
            if task_file is None:
                task_file = 'test_data/turnonlights/test_turnonlights_lite_251030.json'
            return load_light_tasks(task_file)
        else:
            raise NotImplementedError(f"Environment {env_type} not yet implemented")
    
    def run(self):
        """Run all episodes."""
        num_episodes = self.config['num_episodes']
        seed = self.config['seed']
        
        print(f"\n{'='*60}")
        print(f"Starting experiment: {self.config['agent']} + {self.config['memory']} + {self.config['intervention']}")
        print(f"Episodes: {num_episodes}, Max steps: {self.config['max_steps']}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}\n")
        
        for episode_idx in range(num_episodes):
            episode_seed = seed + episode_idx if seed is not None else None
            
            # Select task (cycle through available tasks)
            task_data = self.tasks[episode_idx % len(self.tasks)]
            
            print(f"Episode {episode_idx + 1}/{num_episodes} (level={task_data['level']})...")
            
            # Run episode
            episode_result = self._run_episode(
                episode_idx=episode_idx,
                task_data=task_data,
                seed=episode_seed
            )
            
            # Store results
            self.episodes_data.append(episode_result['episode_data'])
            self.episode_metrics.append(episode_result['metrics'])
            
            # Print episode summary
            metrics = episode_result['metrics']
            status = "✓ SUCCESS" if metrics['success'] else "✗ FAILED"
            print(f"  {status} | Steps: {metrics['num_steps']} | Loop ratio: {metrics['loop_ratio']:.3f}")
            print()
        
        # Save results
        self._save_results()
        
        # Print summary
        self._print_summary()
    
    def _run_episode(self, episode_idx: int, task_data: Dict, seed: Optional[int]):
        """
        Run a single episode.
        
        Returns:
            Dict with 'episode_data' and 'metrics'
        """
        # Create environment
        env = create_env(
            env_type=self.config['env'],
            task_data=task_data,
            seed=seed
        )
        
        # Create memory system
        # Pass llm_client for reflector memory type
        memory_kwargs = {}
        if self.config['memory'] == 'reflector':
            memory_kwargs['llm_client'] = self.llm_client
        memory = create_memory_system(self.config['memory'], **memory_kwargs)
        
        # Create intervention
        intervention_type = self.config['intervention']
        intervention_kwargs = {}
        if intervention_type == 'noise':
            intervention_kwargs['noise_prob'] = self.config.get('noise_p', 0.05)
        elif intervention_type == 'surgery':
            intervention_kwargs['suppress_after_step'] = self.config.get('surgery_step', 80)
        
        intervention = create_intervention(
            intervention_type,
            seed=seed,
            **intervention_kwargs
        )
        
        # Create agent
        agent = create_agent(
            agent_type=self.config['agent'],
            llm_client=self.llm_client,
            env_type=self.config['env'],
            seed=seed
        )
        
        # Reset agent history (for LLM agent)
        if hasattr(agent, 'reset_episode'):
            agent.reset_episode()
        
        # Apply start-of-episode interventions
        intervention.apply_at_start(memory, env)
        
        # Initialize episode tracking
        episode_tracker = EpisodeMetrics()
        
        # Reset environment
        obs = env.reset()
        done = False
        step = 0
        
        # Episode data for logging
        episode_data = {
            'episode_idx': episode_idx,
            'task_level': task_data['level'],
            'steps': []
        }
        
        # Run episode
        while not done and step < self.config['max_steps']:
            # Apply observation intervention (e.g., noise)
            obs_for_agent = intervention.apply_to_observation(obs, step)
            
            # Get memory context
            memory_suppressed = intervention.should_suppress_memory(step)
            memory_context = memory.get_context(obs_for_agent, suppressed=memory_suppressed)
            memory_used = len(memory_context) > 0
            
            # Select action
            action, raw_output = agent.select_action(
                obs=obs_for_agent,
                num_actions=env.get_num_actions(),
                step=step,
                memory_context=memory_context
            )
            
            # Check if action is valid
            invalid_action = (action is None)
            if invalid_action:
                # Fallback to random action
                import random
                action = random.randint(0, env.get_num_actions() - 1)
            
            # Take action
            obs_after, feedback, done, info = env.step(action)
            progress = info.get('progress', 0)
            
            # Update memory (use true observation, not noisy one)
            memory.update(obs, action, obs_after, feedback)
            
            # Record step
            episode_tracker.record_step(
                step=step,
                obs_before=obs,
                action=action,
                obs_after=obs_after,
                feedback=feedback,
                progress=progress,
                llm_raw_output=raw_output,
                memory_context_len=len(memory_context),
                memory_used=memory_used,
                invalid_action=invalid_action
            )
            
            # Record in agent history (for LLM agent)
            if hasattr(agent, 'record_step'):
                agent.record_step(step, obs, action, obs_after, feedback, done)
            
            # Update observation
            obs = obs_after
            step += 1
        
        # Finalize episode
        episode_tracker.finalize(success=done)
        metrics = episode_tracker.compute_metrics()
        
        # Store episode data
        episode_data['steps'] = episode_tracker.get_steps()
        episode_data['success'] = done
        episode_data['num_steps'] = step
        episode_data['metrics'] = metrics
        
        return {
            'episode_data': episode_data,
            'metrics': metrics
        }
    
    def _save_results(self):
        """Save results to JSON files."""
        # Save episodes
        episodes_file = self.output_dir / 'episodes.json'
        with open(episodes_file, 'w') as f:
            json.dump(self.episodes_data, f, indent=2)
        print(f"Saved episodes to: {episodes_file}")
        
        # Save summary
        summary = compute_summary_statistics(self.episode_metrics)
        summary_file = self.output_dir / 'summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to: {summary_file}")
        
        # Save config
        config_file = self.output_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"Saved config to: {config_file}")
    
    def _print_summary(self):
        """Print summary statistics."""
        summary = compute_summary_statistics(self.episode_metrics)
        
        print(f"\n{'='*60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        print(f"Episodes: {summary['num_episodes']}")
        print(f"Success rate: {summary['success_rate']:.2%} ({summary['num_success']}/{summary['num_episodes']})")
        print(f"Avg steps (all): {summary['avg_steps_all']:.1f}")
        print(f"Avg steps (success): {summary['avg_steps_success']:.1f}")
        print(f"Avg loop ratio: {summary['avg_loop_ratio']:.3f}")
        print(f"Avg memory usage: {summary['avg_memory_usage_rate']:.3f}")
        print(f"Avg invalid action rate: {summary['avg_invalid_action_rate']:.3f}")
        print(f"{'='*60}\n")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run mis-evolved memory experiments on Odyssey-Arena"
    )
    
    # Environment
    parser.add_argument('--env', type=str, default='light',
                        choices=['light', 'energy', 'repo', 'trade'],
                        help='Environment type')
    parser.add_argument('--task_file', type=str, default=None,
                        help='Path to task file (default: test_turnonlights_lite_251030.json)')
    
    # Agent
    parser.add_argument('--agent', type=str, default='llm',
                        choices=['llm', 'random', 'heuristic'],
                        help='Agent type')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                        help='LLM model name (e.g., gpt-4, Qwen/Qwen2.5-7B-Instruct)')
    parser.add_argument('--provider', type=str, default='vllm',
                        choices=['vllm', 'openai', 'openrouter', 'dummy'],
                        help='LLM provider (vllm for local, openai/openrouter for API)')
    parser.add_argument('--api_key', type=str, default=None,
                        help='API key for cloud providers (or use env vars)')
    parser.add_argument('--use_dummy_llm', action='store_true',
                        help='Use dummy LLM for testing (no GPU/API required)')
    
    # Memory
    parser.add_argument('--memory', type=str, default='nomem',
                        choices=['nomem', 'naive', 'gated', 'reflector'],
                        help='Memory system type')
    
    # Intervention
    parser.add_argument('--intervention', type=str, default='none',
                        choices=['none', 'injection', 'surgery', 'noise'],
                        help='Intervention type')
    parser.add_argument('--noise_p', type=float, default=0.05,
                        help='Noise probability (for noise intervention)')
    parser.add_argument('--surgery_step', type=int, default=80,
                        help='Step to suppress memory (for surgery intervention)')
    
    # Experiment settings
    parser.add_argument('--num_episodes', type=int, default=30,
                        help='Number of episodes to run')
    parser.add_argument('--max_steps', type=int, default=200,
                        help='Maximum steps per episode')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Output
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Convert args to config dict
    config = vars(args)
    
    # Run experiment
    runner = ExperimentRunner(config)
    runner.run()


if __name__ == '__main__':
    main()

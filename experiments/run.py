"""
Main runner and CLI for mis-evolved memory experiments.

Usage:
    python -m experiments.run --env light --agent llm --memory gated --num_episodes 30
"""

import argparse
import json
import os
import signal
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
        memory_needs_llm = config['memory'] in ['text', 'hypothesis', 'gated', 'naive', 'reflector']
        if config['agent'] == 'llm' or memory_needs_llm:
            self.llm_client = create_llm_client(
                provider=config.get('provider', 'vllm'),
                model_name=config['model'],
                api_key=config.get('api_key'),
                use_dummy=config.get('use_dummy_llm', False)
            )
        
        # Results storage
        self.episodes_data = []
        self.episode_metrics = []
        
        # Memory persistence: create once and reuse across episodes (optional)
        self._persistent_memory = None
        if config.get('memory_persistence', False):
            self._persistent_memory = self._create_memory_system()
    
    def _create_memory_system(self):
        """Create a memory system from config (used for persistent or per-episode memory)."""
        memory_type = self.config['memory']
        memory_needs_llm = memory_type in ['text', 'hypothesis', 'gated', 'naive', 'reflector']
        return create_memory_system(
            memory_type,
            llm_client=self.llm_client if memory_needs_llm else None,
            obs_format=self.config.get('obs_format', 'bitstring'),
            verification_mode=self.config.get('verification_mode', 'oracle'),
            env_type=self.config.get('env', 'light'),
            reflection_frequency=self.config.get('reflection_frequency', 1),
            max_text_memories=self.config.get('max_text_memories', 20),
            curation_context_size=self.config.get('curation_context_size', 10),
            min_insight_length=self.config.get('min_insight_length', 10),
            max_context_items=self.config.get('max_context_items'),
            initial_confidence=self.config.get('initial_confidence', 0.73),
            confidence_support_weight=self.config.get('confidence_support_weight', 1),
            confidence_contradict_weight=self.config.get('confidence_contradict_weight', 2),
            partial_match_support_delta=self.config.get('partial_match_support_delta', 0.5),
            min_confidence=self.config.get('min_confidence', 0.75),
            min_support=self.config.get('min_support', 2),
            max_contradictions=self.config.get('max_contradictions', 0),
            min_confidence_llm=self.config.get('min_confidence_llm', 0.70),
            min_support_llm=self.config.get('min_support_llm', 1),
        )
    
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
    
    def _sample_task(self, episode_idx: int) -> Dict:
        """Select task for episode based on task_sampling config (cyclic or random)."""
        sampling = self.config.get('task_sampling', 'cyclic')
        if sampling == 'random':
            import random
            rng = random.Random(self.config.get('seed') + episode_idx)
            idx = rng.randint(0, len(self.tasks) - 1)
            return self.tasks[idx]
        # default: cyclic
        return self.tasks[episode_idx % len(self.tasks)]
    
    def _print_experiment_setup(self):
        """Print full experiment setup before running (for real-time verification)."""
        c = self.config
        print("\n" + "=" * 60)
        print("EXPERIMENT SETUP")
        print("=" * 60)
        print(f"  env:              {c.get('env', 'light')}")
        print(f"  agent:            {c.get('agent', 'llm')}")
        print(f"  model:            {c.get('model', 'N/A')}")
        print(f"  provider:         {c.get('provider', 'N/A')}")
        print(f"  memory:           {c.get('memory', 'nomem')}")
        print(f"  verification_mode:{c.get('verification_mode', 'oracle')}")
        print(f"  intervention:     {c.get('intervention', 'none')}")
        print(f"  num_episodes:      {c.get('num_episodes', 30)}")
        print(f"  max_steps:        {c.get('max_steps', 200)}")
        print(f"  seed:             {c.get('seed')}")
        print(f"  reflection_freq:   {c.get('reflection_frequency', 1)}")
        print(f"  max_context_items:{c.get('max_context_items')}")
        print(f"  task_sampling:    {c.get('task_sampling', 'cyclic')}")
        print(f"  memory_persistence:{c.get('memory_persistence', False)}")
        print(f"  output_dir:       {self.output_dir}")
        print("=" * 60 + "\n")
    
    def _write_progress(self, episode_idx: int, step: int, max_steps: int,
                        num_episodes: int, num_success: int, env_progress: int):
        """Write progress.json for dashboard polling."""
        try:
            progress_file = self.output_dir / "progress.json"
            data = {
                "episode": episode_idx + 1,
                "total_episodes": num_episodes,
                "step": step,
                "max_steps": max_steps,
                "env_progress": env_progress,
                "num_success": num_success,
                "success_rate": round(num_success / (episode_idx + 1), 4) if episode_idx >= 0 else 0.0,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            tmp = progress_file.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            tmp.replace(progress_file)
        except Exception:
            pass

    def run(self):
        """Run all episodes."""
        num_episodes = self.config['num_episodes']
        seed = self.config['seed']
        memory_persistent = self.config.get('memory_persistence', False)

        # Save config.json immediately so dashboard can display it before run completes
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            config_file = self.output_dir / 'config.json'
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Warning: could not save config.json early: {e}")

        self._print_experiment_setup()

        print(f"Starting: {self.config['agent']} + {self.config['memory']} + {self.config['intervention']}")
        print(f"Episodes: {num_episodes}, Max steps: {self.config['max_steps']}\n")
        
        for episode_idx in range(num_episodes):
            episode_seed = seed + episode_idx if seed is not None else None
            
            task_data = self._sample_task(episode_idx)
            task_level = task_data.get('level', task_data.get('idx', episode_idx))
            
            print(f"\n{'#'*70}")
            print(f"##  EPISODE {episode_idx + 1}/{num_episodes}  |  level={task_level}")
            print(f"{'#'*70}")
            
            # Write progress at episode start
            num_success_so_far = sum(1 for m in self.episode_metrics if m.get('success'))
            self._write_progress(
                episode_idx=episode_idx, step=0,
                max_steps=self.config['max_steps'],
                num_episodes=num_episodes,
                num_success=num_success_so_far,
                env_progress=0,
            )
            
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
            print(f"\n{'#'*70}")
            print(f"##  EPISODE {episode_idx + 1} RESULT: {status}")
            print(f"##  Steps={metrics['num_steps']}  Loop={metrics['loop_ratio']:.3f}  Invalid={metrics['invalid_action_rate']:.3f}")
            print(f"{'#'*70}")
            
            # Write progress after episode ends
            num_success_so_far = sum(1 for m in self.episode_metrics if m.get('success'))
            self._write_progress(
                episode_idx=episode_idx, step=metrics['num_steps'],
                max_steps=self.config['max_steps'],
                num_episodes=num_episodes,
                num_success=num_success_so_far,
                env_progress=0,
            )
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
            seed=seed,
            obs_format=self.config.get('obs_format', 'bitstring')
        )
        
        # Use persistent memory if enabled, else create per-episode memory
        if self._persistent_memory is not None:
            memory = self._persistent_memory
        else:
            memory = self._create_memory_system()
        
        # Create intervention (pass domain for injection so it uses env-appropriate patterns)
        intervention_type = self.config['intervention']
        intervention_kwargs = {'obs_format': self.config.get('obs_format', 'bitstring')}
        if intervention_type == 'injection':
            from experiments.domain import get_domain_adapter
            intervention_kwargs['domain'] = get_domain_adapter(self.config.get('env', 'light'))
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
            seed=seed,
            prompt_template=self.config.get('prompt_template', 'research'),
            max_history=self.config.get('history_window', None),
            think_in_history=self.config.get('think_in_history', 'last'),
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
            
            # Select action (agent may return 2-tuple or 3-tuple with model_input)
            result = agent.select_action(
                obs=obs_for_agent,
                num_actions=env.get_num_actions(),
                step=step,
                memory_context=memory_context
            )
            if len(result) == 3:
                action, raw_output, extra = result
                model_input = extra.get("model_input", "")
                think_text = extra.get("think", "")
            else:
                action, raw_output = result[0], result[1]
                model_input = ""
                think_text = ""
            
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
            
            # Oracle verification if enabled (use standard env interface)
            if (self.config.get('verification_mode') == 'oracle' and 
                hasattr(memory, 'verify_with_oracle')):
                ground_truth = env.get_ground_truth_for_verification() if hasattr(env, 'get_ground_truth_for_verification') else getattr(env, 'get_custom_logic', lambda: None)()
                if ground_truth:
                    use_llm_oracle = self.config.get('oracle_use_llm', True)
                    memory.verify_with_oracle(ground_truth, use_llm=use_llm_oracle)
            
            # Collect per-step memory ops (reflection, curation, verify) for dashboard
            memory_ops = memory.get_recent_ops() if hasattr(memory, 'get_recent_ops') else []
            
            # Per-step header (always printed)
            log_interval = self.config.get('log_step_interval', 25)
            mem_tag = "mem=ON" if memory_used else "mem=OFF"
            inv_tag = " [INVALID→RANDOM]" if invalid_action else ""
            _num_ep = self.config.get('num_episodes', '?')
            print(f"\n{'─'*70}")
            print(f"  Step {step:>3}  |  Ep {episode_idx + 1}/{_num_ep}  |  action={action}{inv_tag}  |  progress={progress}  |  {mem_tag}")
            print(f"{'─'*70}")

            # Detailed input/output logging every N steps
            if log_interval > 0 and (step <= 1 or step % log_interval == 0):
                print("[ INPUT ]")
                print(model_input if model_input else "(no prompt)")
                if think_text:
                    print("\n[ THINK ]")
                    print(think_text)
                    print("\n[ ACTION ]")
                    print(raw_output[len(think_text):].strip() if think_text in raw_output else raw_output)
                else:
                    print("\n[ OUTPUT ]")
                    print(raw_output)
                if memory_ops:
                    print("\n[ MEMORY OPS ]")
                    for op in memory_ops:
                        print(json.dumps(op, ensure_ascii=False, indent=2))
            
            # Record step (include memory_context, model in/out, memory_ops for dashboard)
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
                invalid_action=invalid_action,
                memory_context=memory_context,
                model_input=model_input,
                model_output=raw_output,
                memory_ops=memory_ops,
            )
            
            # Periodic progress.json update every log_interval
            if log_interval > 0 and (step <= 1 or step % log_interval == 0):
                num_success_so_far = sum(1 for m in self.episode_metrics if m.get('success'))
                self._write_progress(
                    episode_idx=episode_idx,
                    step=step,
                    max_steps=self.config['max_steps'],
                    num_episodes=self.config['num_episodes'],
                    num_success=num_success_so_far,
                    env_progress=progress,
                )
            
            # Record in agent history (for LLM agent)
            if hasattr(agent, 'record_step'):
                agent.record_step(step, obs, action, obs_after, feedback, done, think=think_text)
            
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
    parser.add_argument('--reflection_frequency', type=int, default=1,
                        help='Reflector: trigger reflection every N experiences (default: 1)')
    parser.add_argument('--max_context_items', type=int, default=None, metavar='N',
                        help='Max hypotheses/memories in context (default: None = no limit)')
    parser.add_argument('--memory', type=str, default='nomem',
                        choices=['nomem', 'text', 'hypothesis', 'gated', 'naive', 'reflector'],
                        help='Memory system type (text=raw text, hypothesis=structured, gated=verified)')
    parser.add_argument('--verification_mode', type=str, default='oracle',
                        choices=['oracle', 'llm', 'observation', 'hybrid', 'none'],
                        help='How to verify hypotheses (oracle=ground truth rules, observation=empirical, llm=ask LLM, hybrid=oracle+llm)')
    parser.add_argument('--oracle_use_llm', action='store_true',
                        help='Use LLM to analyze custom_logic for oracle verification (more accurate)')
    
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
    parser.add_argument('--memory_persistence', action='store_true',
                        help='Keep memory across episodes (continuous task stream)')
    parser.add_argument('--task_sampling', type=str, default='cyclic',
                        choices=['cyclic', 'random'],
                        help='Task selection: cyclic (default) or random')
    parser.add_argument('--log_step_interval', type=int, default=25,
                        help='Log progress every N steps (0=disable, default: 25)')
    
    # Prompt and format options
    parser.add_argument('--prompt_template', type=str, default='research',
                        choices=['research', 'original'],
                        help='Prompt template (research=concise, original=Odyssey-Arena style)')
    parser.add_argument('--history_window', type=int, default=None, metavar='N',
                        help='Number of recent steps to include in LLM prompt history (default: None = all history)')
    parser.add_argument('--think_in_history', type=str, default='last',
                        choices=['all', 'last', 'none'],
                        help='How model reasoning appears in history: '
                             'all=every step, last=only most recent step (default), none=excluded')
    parser.add_argument('--obs_format', type=str, default='bitstring',
                        choices=['bitstring', 'emoji'],
                        help='Observation format (bitstring="010101", emoji="💡 ○ 💡")')
    
    # Output
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Display name for dashboard (default: experiment_id)')
    parser.add_argument('--dashboard', action='store_true', default=True,
                        help='Register run with experiment dashboard and write run.log (default: True)')
    parser.add_argument('--no-dashboard', action='store_false', dest='dashboard',
                        help='Disable dashboard registration and run.log')

    # Email notification
    parser.add_argument('--notify_email', type=str, default=None,
                        help='Send completion email to this address (env: NOTIFY_TO). '
                             'Requires NOTIFY_PASSWORD and NOTIFY_FROM env vars (or Gmail App Password).')

    return parser.parse_args()


def _tee_stdout_stderr_to_file(log_path: Path):
    """Redirect stdout/stderr to also write to log_path. Returns (original_stdout, original_stderr, log_file)."""
    class Tee:
        def __init__(self, orig, f):
            self._orig = orig
            self._f = f
        def write(self, s):
            self._orig.write(s)
            if s:
                self._f.write(s)
                self._f.flush()
        def flush(self):
            self._orig.flush()
            self._f.flush()
    log_file = open(log_path, "w", encoding="utf-8")
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(orig_stdout, log_file)
    sys.stderr = Tee(orig_stderr, log_file)
    return orig_stdout, orig_stderr, log_file


def main():
    """Main entry point."""
    args = parse_args()
    config = vars(args)
    runner = ExperimentRunner(config)
    use_dashboard = args.dashboard
    log_file = None
    orig_stdout = orig_stderr = None
    _dashboard_finished = [False]  # list so handler can mutate

    def _on_exit_signal(signum, frame):
        # Restore default handlers immediately so a second Ctrl+C kills instantly
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        if use_dashboard and not _dashboard_finished[0]:
            _dashboard_finished[0] = True
            try:
                from experiments.dashboard.registry import register_finish
                register_finish(str(runner.output_dir), success=False)
            except Exception:
                pass
        # Close log file before hard exit
        if log_file is not None and orig_stdout is not None:
            try:
                sys.stdout = orig_stdout
                sys.stderr = orig_stderr
                log_file.close()
            except Exception:
                pass
        # os._exit() bypasses Python I/O wait and terminates immediately
        os._exit(130 if signum == signal.SIGINT else 1)

    if use_dashboard:
        try:
            from experiments.dashboard.registry import register_start
            register_start(str(runner.output_dir), config, name=args.experiment_name)
            orig_stdout, orig_stderr, log_file = _tee_stdout_stderr_to_file(runner.output_dir / "run.log")
            signal.signal(signal.SIGTERM, _on_exit_signal)
            signal.signal(signal.SIGINT, _on_exit_signal)
        except Exception as e:
            print(f"Dashboard register start failed: {e}", file=sys.__stderr__)

    notify_to = args.notify_email or os.environ.get("NOTIFY_TO", "")

    def _send_notify(success: bool, summary=None):
        if not notify_to:
            return
        try:
            from experiments.notify import send_experiment_email
            send_experiment_email(
                to=notify_to,
                experiment_name=args.experiment_name,
                experiment_id=config.get("experiment_id"),
                output_dir=str(runner.output_dir),
                success=success,
                summary=summary,
                config_summary={
                    k: config.get(k) for k in
                    ("env", "agent", "model", "memory", "verification_mode",
                     "intervention", "num_episodes", "max_steps", "seed")
                },
            )
        except Exception as e:
            print(f"Email notification failed: {e}", file=sys.__stderr__)

    try:
        runner.run()
        summary_for_finish = None
        summary_path = runner.output_dir / "summary.json"
        if summary_path.exists():
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary_for_finish = json.load(f)
            except Exception:
                pass
        if use_dashboard and not _dashboard_finished[0]:
            _dashboard_finished[0] = True
            try:
                from experiments.dashboard.registry import register_finish
                register_finish(str(runner.output_dir), success=True, summary=summary_for_finish)
            except Exception as e:
                print(f"Dashboard register_finish failed: {e}", file=sys.__stderr__)
        _send_notify(success=True, summary=summary_for_finish)
    except Exception:
        if use_dashboard and not _dashboard_finished[0]:
            _dashboard_finished[0] = True
            try:
                from experiments.dashboard.registry import register_finish
                register_finish(str(runner.output_dir), success=False)
            except Exception as e:
                print(f"Dashboard register_finish failed: {e}", file=sys.__stderr__)
        _send_notify(success=False)
        raise
    finally:
        if log_file is not None and orig_stdout is not None:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            log_file.close()


if __name__ == '__main__':
    main()

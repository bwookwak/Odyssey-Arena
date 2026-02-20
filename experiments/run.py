"""
Main runner and CLI for mis-evolved memory experiments.

Usage:
    python -m experiments.run --env light --agent llm --memory gated --num_episodes 30
"""

import argparse
import asyncio
import json
import os
import re
import signal
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import time

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.envs import load_light_tasks, create_env
from experiments.llm import create_llm_client
from experiments.agents import create_agent
from experiments.memory import create_memory_system
from experiments.interventions import create_intervention
from experiments.metrics import EpisodeMetrics, compute_summary_statistics, compute_pass_at_k


def _format_prompt_for_display(prompt: str, max_history: int = 5) -> str:
    """
    Return a display-friendly version of the prompt where history entries
    beyond the most recent `max_history` are replaced with a summary line.
    Supports both 'research' (=== RECENT HISTORY ===) and 'original'
    (### History Action and Feedback:) prompt templates.
    """
    # --- research template ---
    m = re.search(
        r'(=== RECENT HISTORY ===\n)(.*?)(\n=== CURRENT SITUATION ===)',
        prompt, re.DOTALL,
    )
    if m:
        header, block, footer = m.group(1), m.group(2), m.group(3)
        # Each entry starts with "Step N:"; continuation lines (Reasoning) are indented
        entries = re.split(r'(?=^Step \d+:)', block, flags=re.MULTILINE)
        entries = [e for e in entries if e.strip()]
        if len(entries) > max_history:
            n_hidden = len(entries) - max_history
            new_block = (
                f"  ... [{n_hidden} earlier step(s) omitted] ...\n"
                + ''.join(entries[-max_history:])
            )
            return prompt[: m.start()] + header + new_block + footer + prompt[m.end() :]
        return prompt

    # --- original template ---
    m = re.search(
        r'(### History Action and Feedback:\n)(.*?)(\n\n### Current State:)',
        prompt, re.DOTALL,
    )
    if m:
        header, block, footer = m.group(1), m.group(2), m.group(3)
        lines = [l for l in block.split('\n') if l.strip() and l.strip() != '(No history yet)']
        if len(lines) > max_history:
            n_hidden = len(lines) - max_history
            new_block = (
                f"  ... [{n_hidden} earlier step(s) omitted] ...\n"
                + '\n'.join(lines[-max_history:])
            )
            return prompt[: m.start()] + header + new_block + footer + prompt[m.end() :]
        return prompt

    return prompt


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

        # Checkpoint tracking (updated after each save; included in progress.json)
        self._last_checkpoint_info: Optional[Dict[str, Any]] = None

        # Memory persistence: create once and reuse across episodes (optional)
        self._persistent_memory = None
        if config.get('memory_persistence', False):
            self._persistent_memory = self._create_memory_system()
            # Pipeline: load explore-stage memory snapshot into infer-stage memory
            snapshot = config.get('_initial_memory_snapshot')
            if snapshot and self._persistent_memory is not None:
                from experiments.memory import load_memory_snapshot
                load_memory_snapshot(snapshot, self._persistent_memory)
    
    def _config_for_save(self) -> Dict[str, Any]:
        """Return config dict safe for JSON serialization (strips runtime-only _ keys)."""
        return {k: v for k, v in self.config.items() if not k.startswith('_')}

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
            mid_episode_reflection=self.config.get('mid_episode_reflection', True),
            episode_reflection=self.config.get('episode_reflection', True),
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
            # Append checkpoint info as a separate field; does not affect dashboard fields above
            if self._last_checkpoint_info is not None:
                data["checkpoint"] = self._last_checkpoint_info
            tmp = progress_file.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            tmp.replace(progress_file)
        except Exception:
            pass

    def _save_checkpoint(self, next_episode_idx: int, extra: Optional[Dict[str, Any]] = None):
        """
        Save a checkpoint so the experiment can be resumed later.

        Writes checkpoint.json atomically (tmp → replace).
        Updates self._last_checkpoint_info so the next _write_progress call
        includes the checkpoint field in progress.json.

        Args:
            next_episode_idx: The episode index to start from on resume
                              (i.e. number of completed episodes so far).
            extra: Optional dict of additional state for pass@k mode
                   (e.g. {'task_idx': 3, 'trial_idx': 1}).
        """
        checkpoint_file = self.output_dir / "checkpoint.json"
        saved_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        data: Dict[str, Any] = {
            "next_episode_idx": next_episode_idx,
            "episodes_data": self.episodes_data,
            "episode_metrics": self.episode_metrics,
            "saved_at": saved_at,
        }
        if extra:
            data.update(extra)

        tmp = checkpoint_file.with_suffix(".json.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            tmp.replace(checkpoint_file)
        except Exception as e:
            print(f"Warning: checkpoint save failed: {e}")
            return

        self._last_checkpoint_info = {
            "episode": next_episode_idx,
            "file": str(checkpoint_file.name),
            "saved_at": saved_at,
        }
        if extra:
            for k in ("task_idx", "trial_idx"):
                if k in extra:
                    self._last_checkpoint_info[k] = extra[k]

        print(f"  [checkpoint] saved at episode {next_episode_idx} → {checkpoint_file.name}")

        # episodes.json / summary.json 갱신
        self._save_interim()

    def _save_interim(self):
        """
        Atomically write episodes.json and summary.json to output_dir.

        Called after every episode / batch so the dashboard detail page always
        shows up-to-date data without waiting for the full run to finish.
        """
        # episodes.json
        episodes_file = self.output_dir / "episodes.json"
        tmp_ep = episodes_file.with_suffix(".json.tmp")
        try:
            with open(tmp_ep, "w", encoding="utf-8") as f:
                json.dump(self.episodes_data, f, indent=2, ensure_ascii=False)
            tmp_ep.replace(episodes_file)
        except Exception as e:
            print(f"Warning: episodes.json interim save failed: {e}")

        # summary.json
        if not self.episode_metrics:
            return
        try:
            summary = compute_summary_statistics(self.episode_metrics)
            pass_k = self.config.get('pass_k', 1)
            if pass_k > 1:
                summary['pass_at_k'] = compute_pass_at_k(self.episodes_data, pass_k)
            summary_file = self.output_dir / "summary.json"
            tmp_su = summary_file.with_suffix(".json.tmp")
            with open(tmp_su, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            tmp_su.replace(summary_file)
        except Exception as e:
            print(f"Warning: summary.json interim save failed: {e}")

    def _print_step_io(
        self,
        model_input: str,
        think_text: str,
        raw_output: str,
        memory_ops: List[Dict],
        tag: str = "",
        max_history: int = 5,
        invalid_action: bool = False,
    ) -> None:
        """Print LLM input/output for a step with history abbreviated for readability."""
        display_input = _format_prompt_for_display(model_input, max_history) if model_input else ""
        print(f"{tag}[ INPUT ]")
        print(display_input if display_input else "(no prompt)")
        if invalid_action:
            print(f"\n{tag}[ INVALID OUTPUT — parse failed, no action taken ]")
            print(raw_output if raw_output else "(empty output)")
        elif think_text:
            print(f"\n{tag}[ THINK ]")
            print(think_text)
            print(f"\n{tag}[ ACTION ]")
            action_part = raw_output[len(think_text):].strip() if think_text in raw_output else raw_output
            print(action_part)
        else:
            print(f"\n{tag}[ OUTPUT ]")
            print(raw_output)
        if memory_ops:
            print(f"\n{tag}[ MEMORY OPS ]")
            for op in memory_ops:
                print(json.dumps(op, ensure_ascii=False, indent=2))

    def get_memory_snapshot(self) -> Optional[Dict[str, Any]]:
        """
        Serialize the current persistent memory to a dict for pipeline stage handoff.
        Returns None if no persistent memory or memory type does not support to_dict().
        """
        if self._persistent_memory is None:
            return None
        if not hasattr(self._persistent_memory, 'to_dict'):
            return None
        snapshot = self._persistent_memory.to_dict()
        snapshot['stats'] = self._persistent_memory.get_stats()
        return snapshot

    def _load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint.json from output_dir if it exists.

        Returns:
            Checkpoint dict or None if not found / invalid.
        """
        checkpoint_file = self.output_dir / "checkpoint.json"
        if not checkpoint_file.exists():
            return None
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                ckpt = json.load(f)
            required = {"next_episode_idx", "episodes_data", "episode_metrics"}
            if not required.issubset(ckpt):
                print(f"Warning: checkpoint.json is missing required fields; ignoring.")
                return None
            return ckpt
        except Exception as e:
            print(f"Warning: could not load checkpoint.json: {e}")
            return None

    def run(self):
        """Run all episodes (standard / pass@k / parallel mode)."""
        pass_k = self.config.get('pass_k', 1)
        parallel = self.config.get('parallel_episodes', 1)

        if self.config.get('memory_persistence', False) and parallel > 1:
            print("Warning: --parallel_episodes is incompatible with --memory_persistence. "
                  "Falling back to sequential execution.")
            parallel = 1

        if parallel > 1:
            import asyncio as _asyncio
            if pass_k > 1:
                _asyncio.run(self._run_pass_k_async(pass_k, parallel))
            else:
                _asyncio.run(self._run_standard_async(parallel))
        elif pass_k > 1:
            self._run_pass_k(pass_k)
        else:
            self._run_standard()

    def _run_standard(self):
        """Run num_episodes episodes with task_sampling (original behavior)."""
        num_episodes = self.config['num_episodes']
        seed = self.config['seed']
        checkpoint_interval = self.config.get('checkpoint_interval', 0)

        # --- Auto-resume if checkpoint.json exists ---
        start_episode_idx = 0
        ckpt = self._load_checkpoint()
        if ckpt is not None:
            self.episodes_data = ckpt['episodes_data']
            self.episode_metrics = ckpt['episode_metrics']
            start_episode_idx = ckpt['next_episode_idx']
            self._last_checkpoint_info = {
                "episode": start_episode_idx,
                "file": "checkpoint.json",
                "saved_at": ckpt.get('saved_at', ''),
            }
            print(f"[resume] Loaded checkpoint: resuming from episode {start_episode_idx + 1}/{num_episodes}")

        # Save config.json immediately so dashboard can display it before run completes
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            config_file = self.output_dir / 'config.json'
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self._config_for_save(), f, indent=2)
        except Exception as e:
            print(f"Warning: could not save config.json early: {e}")

        self._print_experiment_setup()

        print(f"Starting: {self.config['agent']} + {self.config['memory']} + {self.config['intervention']}")
        print(f"Episodes: {num_episodes}, Max steps: {self.config['max_steps']}\n")
        if checkpoint_interval > 0:
            print(f"Checkpointing every {checkpoint_interval} episodes → {self.output_dir / 'checkpoint.json'}\n")

        for episode_idx in range(start_episode_idx, num_episodes):
            episode_seed = seed + episode_idx if seed is not None else None

            task_data = self._sample_task(episode_idx)
            task_level = task_data.get('level', task_data.get('idx', episode_idx))

            print(f"\n{'#'*70}")
            print(f"##  EPISODE {episode_idx + 1}/{num_episodes}  |  level={task_level}")
            print(f"{'#'*70}")

            num_success_so_far = sum(1 for m in self.episode_metrics if m.get('success'))
            self._write_progress(
                episode_idx=episode_idx, step=0,
                max_steps=self.config['max_steps'],
                num_episodes=num_episodes,
                num_success=num_success_so_far,
                env_progress=0,
            )

            episode_result = self._run_episode(
                episode_idx=episode_idx,
                task_data=task_data,
                seed=episode_seed
            )

            self.episodes_data.append(episode_result['episode_data'])
            self.episode_metrics.append(episode_result['metrics'])

            metrics = episode_result['metrics']
            status = "✓ SUCCESS" if metrics['success'] else "✗ FAILED"
            lc = metrics.get('loop_count', 0)
            ta = metrics.get('total_actions', 0)
            print(f"\n{'#'*70}")
            print(f"##  EPISODE {episode_idx + 1} RESULT: {status}")
            print(f"##  Steps={metrics['num_steps']}  "
                  f"Loop(seen-max)={metrics['loop_ratio']:.3f} ({lc}/{ta})  "
                  f"Invalid={metrics['invalid_action_rate']:.3f}")
            print(f"{'#'*70}")

            # --- Checkpoint ---
            completed = episode_idx + 1
            if checkpoint_interval > 0 and completed % checkpoint_interval == 0:
                self._save_checkpoint(next_episode_idx=completed)
            else:
                self._save_interim()

            num_success_so_far = sum(1 for m in self.episode_metrics if m.get('success'))
            self._write_progress(
                episode_idx=episode_idx, step=metrics['num_steps'],
                max_steps=self.config['max_steps'],
                num_episodes=num_episodes,
                num_success=num_success_so_far,
                env_progress=0,
            )
            print()

        self._save_results()
        self._print_summary()

    def _run_pass_k(self, pass_k: int):
        """Run pass@k evaluation: K independent trials per task."""
        seed = self.config['seed']
        total_episodes = len(self.tasks) * pass_k
        checkpoint_interval = self.config.get('checkpoint_interval', 0)

        # Update config so dashboard/logs show correct total
        self.config['num_episodes'] = total_episodes
        self.config['_pass_k_mode'] = True

        # --- Auto-resume if checkpoint.json exists ---
        start_global_idx = 0
        start_task_idx = 0
        start_trial_idx = 0
        ckpt = self._load_checkpoint()
        if ckpt is not None:
            self.episodes_data = ckpt['episodes_data']
            self.episode_metrics = ckpt['episode_metrics']
            start_global_idx = ckpt['next_episode_idx']
            start_task_idx = ckpt.get('task_idx', start_global_idx // pass_k)
            start_trial_idx = ckpt.get('trial_idx', start_global_idx % pass_k)
            self._last_checkpoint_info = {
                "episode": start_global_idx,
                "file": "checkpoint.json",
                "saved_at": ckpt.get('saved_at', ''),
                "task_idx": start_task_idx,
                "trial_idx": start_trial_idx,
            }
            print(f"[resume] Loaded checkpoint: resuming from "
                  f"task {start_task_idx + 1}, trial {start_trial_idx + 1} "
                  f"(global ep {start_global_idx + 1}/{total_episodes})")

        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            config_file = self.output_dir / 'config.json'
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self._config_for_save(), f, indent=2)
        except Exception as e:
            print(f"Warning: could not save config.json early: {e}")

        self._print_experiment_setup()

        print(f"Starting [pass@{pass_k} mode]: {self.config['agent']} + {self.config['memory']} + {self.config['intervention']}")
        print(f"Tasks: {len(self.tasks)}, Trials per task: {pass_k}, Total episodes: {total_episodes}")
        print(f"Max steps: {self.config['max_steps']}\n")
        if checkpoint_interval > 0:
            print(f"Checkpointing every {checkpoint_interval} episodes → {self.output_dir / 'checkpoint.json'}\n")

        global_episode_idx = start_global_idx

        for task_idx, task_data in enumerate(self.tasks):
            # Skip tasks already completed in checkpoint
            if task_idx < start_task_idx:
                global_episode_idx += pass_k
                continue

            task_level = task_data.get('level', task_data.get('idx', task_idx))

            print(f"\n{'='*70}")
            print(f"==  TASK {task_idx + 1}/{len(self.tasks)}  |  level={task_level}  |  running {pass_k} trials")
            print(f"{'='*70}")

            resume_trial = start_trial_idx if task_idx == start_task_idx else 0

            for trial_idx in range(pass_k):
                # Skip trials already completed in checkpoint (for the resumed task)
                if trial_idx < resume_trial:
                    global_episode_idx += 1
                    continue

                episode_seed = seed + task_idx * pass_k + trial_idx if seed is not None else None

                print(f"\n{'#'*70}")
                print(f"##  TASK {task_idx + 1}/{len(self.tasks)}  TRIAL {trial_idx + 1}/{pass_k}  "
                      f"|  level={task_level}  |  ep={global_episode_idx + 1}/{total_episodes}")
                print(f"{'#'*70}")

                num_success_so_far = sum(1 for m in self.episode_metrics if m.get('success'))
                self._write_progress(
                    episode_idx=global_episode_idx, step=0,
                    max_steps=self.config['max_steps'],
                    num_episodes=total_episodes,
                    num_success=num_success_so_far,
                    env_progress=0,
                )

                episode_result = self._run_episode(
                    episode_idx=global_episode_idx,
                    task_data=task_data,
                    seed=episode_seed
                )

                episode_result['episode_data']['trial_idx'] = trial_idx
                episode_result['episode_data']['task_idx'] = task_idx

                self.episodes_data.append(episode_result['episode_data'])
                self.episode_metrics.append(episode_result['metrics'])

                metrics = episode_result['metrics']
                status = "✓ SUCCESS" if metrics['success'] else "✗ FAILED"
                print(f"\n{'#'*70}")
                print(f"##  TRIAL {trial_idx + 1}/{pass_k} RESULT: {status}")
                print(f"##  Steps={metrics['num_steps']}  Loop={metrics['loop_ratio']:.3f}  Invalid={metrics['invalid_action_rate']:.3f}")
                print(f"{'#'*70}")

                global_episode_idx += 1

                # --- Checkpoint (after completing a trial) ---
                if checkpoint_interval > 0 and global_episode_idx % checkpoint_interval == 0:
                    next_trial = trial_idx + 1
                    next_task = task_idx
                    if next_trial >= pass_k:
                        next_trial = 0
                        next_task = task_idx + 1
                    self._save_checkpoint(
                        next_episode_idx=global_episode_idx,
                        extra={'task_idx': next_task, 'trial_idx': next_trial},
                    )
                else:
                    self._save_interim()

                num_success_so_far = sum(1 for m in self.episode_metrics if m.get('success'))
                self._write_progress(
                    episode_idx=global_episode_idx - 1, step=metrics['num_steps'],
                    max_steps=self.config['max_steps'],
                    num_episodes=total_episodes,
                    num_success=num_success_so_far,
                    env_progress=0,
                )
                print()

            # Per-task summary after all trials
            completed_eps = [ep for ep in self.episodes_data
                             if ep.get('task_idx') == task_idx]
            task_success = any(ep['success'] for ep in completed_eps)
            task_status = "✓ PASS" if task_success else "✗ FAIL"
            successes = sum(ep['success'] for ep in completed_eps)
            print(f"  → Task {task_idx + 1} (level={task_level}): {task_status}  ({successes}/{pass_k} trials succeeded)")

        self._save_results()
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
        # goal_state_str: task에 명시적 goal_state가 있을 때만 prompt_builder에 주입.
        # 없으면 None → prompt는 기본 "light on all the bulbs" 설명 사용.
        _goal_state_str = (
            getattr(env, 'goal_state_str', None)
            if task_data.get('goal_state') is not None
            else None
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
            prompt_template=self.config.get('prompt_template', 'original'),
            max_history=self.config.get('history_window', None),
            think_in_history=self.config.get('think_in_history', 'last'),
            latent_rule_guide=self.config.get('latent_rule_guide', True),
            include_raw_output=self.config.get('include_raw_output', False),
            parse_action_retries=self.config.get('max_parse_errors', 10),
        )
        
        # Reset agent history (for LLM agent)
        if hasattr(agent, 'reset_episode'):
            agent.reset_episode()
        
        # 에피소드별 goal_state를 prompt_builder에 주입
        if hasattr(agent, 'prompt_builder') and _goal_state_str is not None:
            agent.prompt_builder.goal_state_str = _goal_state_str
        
        # Apply start-of-episode interventions
        intervention.apply_at_start(memory, env)
        
        # Initialize episode tracking
        episode_tracker = EpisodeMetrics()
        
        # Reset environment
        obs = env.reset()
        done = False
        step = 0
        max_state_visits = self.config.get('max_state_visits', None)
        _consecutive_same_state = 0
        _prev_obs: Optional[str] = None
        # progress_before: progress(obs_before) for seen-max loop detection
        # Starts at 0 (initial state is all-OFF in light env; first action records it).
        _progress_before = 0

        # Episode data for logging
        episode_data = {
            'episode_idx': episode_idx,
            'task_level': task_data['level'],
            'steps': []
        }
        
        # Run episode
        while not done and step < self.config['max_steps']:
            # State loop detection: stop if obs unchanged for N consecutive steps
            if max_state_visits is not None:
                if obs == _prev_obs:
                    _consecutive_same_state += 1
                else:
                    _consecutive_same_state = 1
                _prev_obs = obs
                if _consecutive_same_state >= max_state_visits:
                    print(f"\n  [stop] State '{obs}' unchanged for {_consecutive_same_state} consecutive steps "
                          f"(≥{max_state_visits}) — episode stopped (failure).")
                    break

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
            
            # If all retries exhausted in agent, abort episode immediately (no step recorded)
            if action is None:
                print(f"\n  [EPISODE ABORT] All {agent.parse_action_retries} parse retries exhausted "
                      f"at step {step} — marking episode as failed.")
                break

            # Take action
            obs_after, feedback, done, info = env.step(action)
            progress = info.get('progress', 0)

            # Update memory
            memory.update(obs, action, obs_after, feedback, think=think_text)
            
            # Oracle verification if enabled (use standard env interface)
            if (self.config.get('verification_mode') == 'oracle' and 
                hasattr(memory, 'verify_with_oracle')):
                ground_truth = env.get_ground_truth_for_verification() if hasattr(env, 'get_ground_truth_for_verification') else getattr(env, 'get_custom_logic', lambda: None)()
                if ground_truth:
                    use_llm_oracle = self.config.get('oracle_use_llm', True)
                    memory.verify_with_oracle(ground_truth, use_llm=use_llm_oracle)
            
            # Collect per-step memory ops (reflection, curation, verify) for dashboard
            memory_ops = memory.get_recent_ops() if hasattr(memory, 'get_recent_ops') else []

            # Record step — seen-max loop detection uses _progress_before = progress(obs_before)
            is_loop = episode_tracker.record_step(
                step=step,
                obs_before=obs,
                action=action,
                obs_after=obs_after,
                feedback=feedback,
                progress=progress,
                progress_before=_progress_before,
                llm_raw_output=raw_output,
                memory_context_len=len(memory_context),
                memory_used=memory_used,
                invalid_action=False,
                invalid_output="",
                memory_context=memory_context,
                model_input=model_input,
                model_output=raw_output,
                memory_ops=memory_ops,
            )

            # Per-step header
            log_interval = self.config.get('log_step_interval', 25)
            mem_tag = "mem=ON" if memory_used else "mem=OFF"
            loop_tag = " [LOOP]" if is_loop else ""
            _num_ep = self.config.get('num_episodes', '?')
            lc = episode_tracker.loop_count
            ta = episode_tracker.total_actions
            lr_now = f"{lc/ta:.2f}" if ta > 0 else "0.00"
            print(f"\n{'─'*70}")
            print(f"  Step {step:>3}  |  Ep {episode_idx + 1}/{_num_ep}  "
                  f"|  action={action}  |  progress={progress}  "
                  f"|  {mem_tag}  |  loop={lr_now}({lc}/{ta}){loop_tag}")
            print(f"{'─'*70}")

            # Detailed input/output logging every step (history abbreviated to last 5)
            self._print_step_io(model_input, think_text, raw_output, memory_ops,
                                invalid_action=False)

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
                agent.record_step(step, obs, action, obs_after, feedback, done,
                                  think=think_text, invalid_output="",
                                  raw_output=raw_output)

            # Update state for next step
            obs = obs_after
            _progress_before = progress  # progress(obs_after) = progress(next obs_before)
            step += 1

        # Finalize episode
        episode_tracker.finalize(success=done)
        metrics = episode_tracker.compute_metrics()

        # Post-episode reflection: extract multiple insights from full trajectory
        _task_ctx = ""
        if hasattr(env, 'goal_state_str') and env.goal_state_str:
            _task_ctx = (
                f"Light Bulb Puzzle: reach state {env.goal_state_str} "
                f"by toggling bulbs (action = bulb index, 0-based). "
                f"Hidden toggle rules govern which actions take effect. "
                f"Max steps = {self.config.get('max_steps', '?')}."
            )
        memory.reflect_on_episode(
            trajectory=episode_tracker.get_steps(),
            task_context=_task_ctx,
            success=done,
        )
        post_ep_ops = memory.get_recent_ops() if hasattr(memory, 'get_recent_ops') else []

        # Store episode data
        episode_data['steps'] = episode_tracker.get_steps()
        episode_data['success'] = done
        episode_data['num_steps'] = step
        episode_data['metrics'] = metrics
        if post_ep_ops:
            episode_data['post_episode_ops'] = post_ep_ops

        return {
            'episode_data': episode_data,
            'metrics': metrics
        }

    # ------------------------------------------------------------------ #
    #  Async / parallel helpers                                           #
    # ------------------------------------------------------------------ #

    async def _run_episode_async(
        self,
        episode_idx: int,
        task_data: Dict,
        seed: Optional[int],
        ep_tag: str = "",
        is_leader: bool = False,
    ) -> Dict:
        """
        Async version of _run_episode.

        The only difference is that the LLM call is awaited via
        agent.async_select_action(), allowing multiple episodes to run
        concurrently inside an asyncio event loop.

        is_leader: only the leader episode (highest episode_idx in the batch)
                   writes per-step progress updates, preventing the dashboard
                   from jumping backwards when concurrent episodes write.
        """
        env = create_env(
            env_type=self.config['env'],
            task_data=task_data,
            seed=seed,
            obs_format=self.config.get('obs_format', 'bitstring'),
        )

        # Always create fresh memory per episode in parallel mode
        memory = self._create_memory_system()

        intervention_type = self.config['intervention']
        intervention_kwargs = {'obs_format': self.config.get('obs_format', 'bitstring')}
        if intervention_type == 'injection':
            from experiments.domain import get_domain_adapter
            intervention_kwargs['domain'] = get_domain_adapter(self.config.get('env', 'light'))
        if intervention_type == 'noise':
            intervention_kwargs['noise_prob'] = self.config.get('noise_p', 0.05)
        elif intervention_type == 'surgery':
            intervention_kwargs['suppress_after_step'] = self.config.get('surgery_step', 80)

        intervention = create_intervention(intervention_type, seed=seed, **intervention_kwargs)

        agent = create_agent(
            agent_type=self.config['agent'],
            llm_client=self.llm_client,
            env_type=self.config['env'],
            seed=seed,
            prompt_template=self.config.get('prompt_template', 'original'),
            max_history=self.config.get('history_window', None),
            think_in_history=self.config.get('think_in_history', 'last'),
            latent_rule_guide=self.config.get('latent_rule_guide', True),
            include_raw_output=self.config.get('include_raw_output', False),
            parse_action_retries=self.config.get('max_parse_errors', 10),
        )
        if hasattr(agent, 'reset_episode'):
            agent.reset_episode()

        intervention.apply_at_start(memory, env)

        episode_tracker = EpisodeMetrics()
        obs = env.reset()
        done = False
        step = 0
        max_state_visits = self.config.get('max_state_visits', None)
        _consecutive_same_state = 0
        _prev_obs: Optional[str] = None
        _progress_before = 0  # progress(obs_before) for seen-max loop detection

        episode_data: Dict[str, Any] = {
            'episode_idx': episode_idx,
            'task_level': task_data['level'],
            'steps': [],
        }

        log_interval = self.config.get('log_step_interval', 25)
        _num_ep = self.config.get('num_episodes', '?')
        tag = f"[{ep_tag}] " if ep_tag else ""

        while not done and step < self.config['max_steps']:
            # State loop detection: stop if obs unchanged for N consecutive steps
            if max_state_visits is not None:
                if obs == _prev_obs:
                    _consecutive_same_state += 1
                else:
                    _consecutive_same_state = 1
                _prev_obs = obs
                if _consecutive_same_state >= max_state_visits:
                    print(f"\n  {tag}[stop] State '{obs}' unchanged for {_consecutive_same_state} consecutive steps "
                          f"(≥{max_state_visits}) — episode stopped (failure).")
                    break

            obs_for_agent = intervention.apply_to_observation(obs, step)
            memory_suppressed = intervention.should_suppress_memory(step)
            memory_context = memory.get_context(obs_for_agent, suppressed=memory_suppressed)
            memory_used = len(memory_context) > 0

            # Async LLM call — yields control to other coroutines while waiting
            result = await agent.async_select_action(
                obs=obs_for_agent,
                num_actions=env.get_num_actions(),
                step=step,
                memory_context=memory_context,
            )
            if len(result) == 3:
                action, raw_output, extra = result
                model_input = extra.get("model_input", "")
                think_text = extra.get("think", "")
            else:
                action, raw_output = result[0], result[1]
                model_input = ""
                think_text = ""

            # If all retries exhausted in agent, abort episode immediately (no step recorded)
            if action is None:
                print(f"\n  {tag}[EPISODE ABORT] All {agent.parse_action_retries} parse retries exhausted "
                      f"at step {step} — marking episode as failed.")
                break

            obs_after, feedback, done, info = env.step(action)
            progress = info.get('progress', 0)

            memory.update(obs, action, obs_after, feedback, think=think_text)

            if (self.config.get('verification_mode') == 'oracle'
                    and hasattr(memory, 'verify_with_oracle')):
                ground_truth = (
                    env.get_ground_truth_for_verification()
                    if hasattr(env, 'get_ground_truth_for_verification') else None
                )
                if ground_truth:
                    memory.verify_with_oracle(
                        ground_truth, use_llm=self.config.get('oracle_use_llm', True)
                    )

            memory_ops = memory.get_recent_ops() if hasattr(memory, 'get_recent_ops') else []

            # Record step — seen-max loop detection uses _progress_before = progress(obs_before)
            is_loop = episode_tracker.record_step(
                step=step, obs_before=obs, action=action,
                obs_after=obs_after, feedback=feedback, progress=progress,
                progress_before=_progress_before,
                llm_raw_output=raw_output, memory_context_len=len(memory_context),
                memory_used=memory_used, invalid_action=False,
                invalid_output="",
                memory_context=memory_context, model_input=model_input,
                model_output=raw_output, memory_ops=memory_ops,
            )

            mem_tag = "mem=ON" if memory_used else "mem=OFF"
            loop_tag = " [LOOP]" if is_loop else ""
            lc = episode_tracker.loop_count
            ta = episode_tracker.total_actions
            lr_now = f"{lc/ta:.2f}" if ta > 0 else "0.00"
            print(f"\n{'─'*70}")
            print(f"  {tag}Step {step:>3}  |  Ep {episode_idx + 1}/{_num_ep}  "
                  f"|  action={action}  |  progress={progress}  "
                  f"|  {mem_tag}  |  loop={lr_now}({lc}/{ta}){loop_tag}")
            print(f"{'─'*70}")

            self._print_step_io(model_input, think_text, raw_output, memory_ops,
                                tag=tag, invalid_action=False)

            # Only the leader (highest episode_idx in batch) writes per-step
            # progress to prevent the dashboard from jumping backwards.
            if is_leader and log_interval > 0 and (step <= 1 or step % log_interval == 0):
                num_success_so_far = sum(1 for m in self.episode_metrics if m.get('success'))
                self._write_progress(
                    episode_idx=episode_idx,
                    step=step,
                    max_steps=self.config['max_steps'],
                    num_episodes=self.config.get('num_episodes', '?'),
                    num_success=num_success_so_far,
                    env_progress=progress,
                )

            if hasattr(agent, 'record_step'):
                agent.record_step(step, obs, action, obs_after, feedback, done,
                                  think=think_text, invalid_output="",
                                  raw_output=raw_output)

            obs = obs_after
            _progress_before = progress  # progress(obs_after) = progress(next obs_before)
            step += 1

        episode_tracker.finalize(success=done)
        metrics = episode_tracker.compute_metrics()

        # Post-episode reflection: extract multiple insights from full trajectory
        _task_ctx = ""
        if hasattr(env, 'goal_state_str') and env.goal_state_str:
            _task_ctx = (
                f"Light Bulb Puzzle: reach state {env.goal_state_str} "
                f"by toggling bulbs (action = bulb index, 0-based). "
                f"Hidden toggle rules govern which actions take effect. "
                f"Max steps = {self.config.get('max_steps', '?')}."
            )
        memory.reflect_on_episode(
            trajectory=episode_tracker.get_steps(),
            task_context=_task_ctx,
            success=done,
        )
        post_ep_ops = memory.get_recent_ops() if hasattr(memory, 'get_recent_ops') else []

        episode_data['steps'] = episode_tracker.get_steps()
        episode_data['success'] = done
        episode_data['num_steps'] = step
        episode_data['metrics'] = metrics
        if post_ep_ops:
            episode_data['post_episode_ops'] = post_ep_ops
        return {'episode_data': episode_data, 'metrics': metrics}

    async def _run_standard_async(self, parallel_episodes: int):
        """
        Run num_episodes episodes in batches of parallel_episodes concurrently.

        Episodes in a batch share the asyncio event loop; the LLM await points
        let other coroutines run while one is waiting for an API response.
        Checkpointing and progress writing happen after each batch.
        """
        num_episodes = self.config['num_episodes']
        seed = self.config['seed']
        checkpoint_interval = self.config.get('checkpoint_interval', 0)

        # Auto-resume
        start_episode_idx = 0
        ckpt = self._load_checkpoint()
        if ckpt is not None:
            self.episodes_data = ckpt['episodes_data']
            self.episode_metrics = ckpt['episode_metrics']
            start_episode_idx = ckpt['next_episode_idx']
            self._last_checkpoint_info = {
                "episode": start_episode_idx,
                "file": "checkpoint.json",
                "saved_at": ckpt.get('saved_at', ''),
            }
            print(f"[resume] Loaded checkpoint: resuming from episode "
                  f"{start_episode_idx + 1}/{num_episodes}")

        print(f"[parallel] {parallel_episodes} concurrent episodes  "
              f"| total={num_episodes}  max_steps={self.config['max_steps']}\n")

        for batch_start in range(start_episode_idx, num_episodes, parallel_episodes):
            batch_end = min(batch_start + parallel_episodes, num_episodes)
            batch_indices = list(range(batch_start, batch_end))

            print(f"\n{'#'*70}")
            print(f"##  BATCH  episodes {batch_start + 1}..{batch_end}/{num_episodes}")
            print(f"{'#'*70}")

            # Write progress at batch start so dashboard shows episode advancing
            num_success_so_far = sum(1 for m in self.episode_metrics if m.get('success'))
            self._write_progress(
                episode_idx=batch_start, step=0,
                max_steps=self.config['max_steps'],
                num_episodes=num_episodes,
                num_success=num_success_so_far,
                env_progress=0,
            )

            tasks = [
                self._run_episode_async(
                    episode_idx=ep_idx,
                    task_data=self._sample_task(ep_idx),
                    seed=seed + ep_idx if seed is not None else None,
                    ep_tag=f"Ep{ep_idx + 1}",
                    is_leader=(ep_idx == batch_indices[-1]),
                )
                for ep_idx in batch_indices
            ]
            batch_results = await asyncio.gather(*tasks)

            for result in batch_results:
                self.episodes_data.append(result['episode_data'])
                self.episode_metrics.append(result['metrics'])

            # Print batch summary
            print(f"\n{'#'*70}")
            print(f"##  BATCH RESULTS  episodes {batch_start + 1}..{batch_end}")
            for result in batch_results:
                ep_idx = result['episode_data']['episode_idx']
                m = result['metrics']
                status = "✓ SUCCESS" if m['success'] else "✗ FAILED"
                print(f"##  Ep{ep_idx + 1}: {status}  "
                      f"steps={m['num_steps']}  loop={m['loop_ratio']:.3f}  "
                      f"invalid={m['invalid_action_rate']:.3f}")
            print(f"{'#'*70}")

            completed = len(self.episodes_data)
            if checkpoint_interval > 0 and completed % checkpoint_interval == 0:
                self._save_checkpoint(next_episode_idx=completed)
            else:
                self._save_interim()

            num_success = sum(1 for m in self.episode_metrics if m.get('success'))
            self._write_progress(
                episode_idx=completed - 1, step=0,
                max_steps=self.config['max_steps'],
                num_episodes=num_episodes,
                num_success=num_success,
                env_progress=0,
            )
            print()

        self._save_results()
        self._print_summary()

    async def _run_pass_k_async(self, pass_k: int, parallel_episodes: int):
        """
        Run pass@k in batches of parallel_episodes concurrently.

        Iterates over all tasks; for each task runs pass_k trials,
        scheduling up to parallel_episodes at once.
        """
        seed = self.config['seed']
        total_episodes = len(self.tasks) * pass_k
        checkpoint_interval = self.config.get('checkpoint_interval', 0)

        self.config['num_episodes'] = total_episodes
        self.config['_pass_k_mode'] = True

        # Auto-resume
        start_global_idx = 0
        ckpt = self._load_checkpoint()
        if ckpt is not None:
            self.episodes_data = ckpt['episodes_data']
            self.episode_metrics = ckpt['episode_metrics']
            start_global_idx = ckpt['next_episode_idx']
            self._last_checkpoint_info = {
                "episode": start_global_idx,
                "file": "checkpoint.json",
                "saved_at": ckpt.get('saved_at', ''),
            }
            print(f"[resume] Loaded checkpoint: resuming from global ep "
                  f"{start_global_idx + 1}/{total_episodes}")

        print(f"[parallel pass@{pass_k}] {parallel_episodes} concurrent episodes  "
              f"| tasks={len(self.tasks)}  total={total_episodes}  "
              f"max_steps={self.config['max_steps']}\n")

        # Flatten all (task_idx, trial_idx) pairs and skip already completed ones
        all_jobs = [
            (task_idx, trial_idx)
            for task_idx in range(len(self.tasks))
            for trial_idx in range(pass_k)
        ]
        all_jobs = all_jobs[start_global_idx:]  # resume skip

        global_episode_idx = start_global_idx

        for batch_start in range(0, len(all_jobs), parallel_episodes):
            batch_jobs = all_jobs[batch_start: batch_start + parallel_episodes]

            ep_indices = list(range(global_episode_idx, global_episode_idx + len(batch_jobs)))
            print(f"\n{'#'*70}")
            print(f"##  BATCH  global eps {ep_indices[0] + 1}..{ep_indices[-1] + 1}/{total_episodes}")
            print(f"{'#'*70}")

            # Write progress at batch start so dashboard shows episode advancing
            num_success_so_far = sum(1 for m in self.episode_metrics if m.get('success'))
            self._write_progress(
                episode_idx=ep_indices[0], step=0,
                max_steps=self.config['max_steps'],
                num_episodes=total_episodes,
                num_success=num_success_so_far,
                env_progress=0,
            )

            tasks = []
            leader_ep_idx = ep_indices[-1]
            for (task_idx, trial_idx), ep_idx in zip(batch_jobs, ep_indices):
                task_data = self.tasks[task_idx]
                ep_seed = seed + task_idx * pass_k + trial_idx if seed is not None else None
                tasks.append(
                    self._run_episode_async(
                        episode_idx=ep_idx,
                        task_data=task_data,
                        seed=ep_seed,
                        ep_tag=f"T{task_idx + 1}t{trial_idx + 1}",
                        is_leader=(ep_idx == leader_ep_idx),
                    )
                )

            batch_results = await asyncio.gather(*tasks)

            for (task_idx, trial_idx), result in zip(batch_jobs, batch_results):
                result['episode_data']['task_idx'] = task_idx
                result['episode_data']['trial_idx'] = trial_idx
                self.episodes_data.append(result['episode_data'])
                self.episode_metrics.append(result['metrics'])

            # Print batch summary
            print(f"\n{'#'*70}")
            print(f"##  BATCH RESULTS")
            for (task_idx, trial_idx), result in zip(batch_jobs, batch_results):
                m = result['metrics']
                status = "✓ SUCCESS" if m['success'] else "✗ FAILED"
                task_level = self.tasks[task_idx].get('level', task_idx)
                print(f"##  T{task_idx + 1}(lv={task_level}) trial{trial_idx + 1}: {status}  "
                      f"steps={m['num_steps']}  loop={m['loop_ratio']:.3f}")
            print(f"{'#'*70}")

            global_episode_idx += len(batch_jobs)
            completed = len(self.episodes_data)

            if checkpoint_interval > 0 and completed % checkpoint_interval == 0:
                next_task, next_trial = 0, 0
                if batch_start + parallel_episodes < len(all_jobs):
                    next_task, next_trial = all_jobs[batch_start + parallel_episodes]
                self._save_checkpoint(
                    next_episode_idx=global_episode_idx,
                    extra={'task_idx': next_task, 'trial_idx': next_trial},
                )
            else:
                self._save_interim()

            num_success = sum(1 for m in self.episode_metrics if m.get('success'))
            self._write_progress(
                episode_idx=global_episode_idx - 1, step=0,
                max_steps=self.config['max_steps'],
                num_episodes=total_episodes,
                num_success=num_success,
                env_progress=0,
            )
            print()

        self._save_results()
        self._print_summary()

    def _save_results(self):
        """Save results to JSON files."""
        episodes_file = self.output_dir / 'episodes.json'
        with open(episodes_file, 'w') as f:
            json.dump(self.episodes_data, f, indent=2)
        print(f"Saved episodes to: {episodes_file}")

        summary = compute_summary_statistics(self.episode_metrics)

        pass_k = self.config.get('pass_k', 1)
        if pass_k > 1:
            pass_k_stats = compute_pass_at_k(self.episodes_data, pass_k)
            summary['pass_at_k'] = pass_k_stats

        summary_file = self.output_dir / 'summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to: {summary_file}")

        config_file = self.output_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(self._config_for_save(), f, indent=2)
        print(f"Saved config to: {config_file}")
    
    def _print_summary(self):
        """Print summary statistics."""
        summary = compute_summary_statistics(self.episode_metrics)
        pass_k = self.config.get('pass_k', 1)

        print(f"\n{'='*60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        print(f"Episodes: {summary['num_episodes']}")
        print(f"Success rate: {summary['success_rate']:.2%} ({summary['num_success']}/{summary['num_episodes']})")
        print(f"Avg steps (all): {summary['avg_steps_all']:.1f}")
        print(f"Avg steps (success): {summary['avg_steps_success']:.1f}")
        print(f"Avg loop ratio (seen-max): {summary['avg_loop_ratio']:.3f}")
        print(f"Avg memory usage: {summary['avg_memory_usage_rate']:.3f}")
        print(f"Avg invalid action rate: {summary['avg_invalid_action_rate']:.3f}")
        print(f"Avg max progress:   {summary.get('avg_max_progress', 0.0):.3f}  (ratio 0–1)")
        print(f"Avg mean progress:  {summary.get('avg_avg_progress', 0.0):.3f}  (ratio 0–1)")

        if pass_k > 1:
            pass_k_stats = compute_pass_at_k(self.episodes_data, pass_k)
            print(f"{'─'*60}")
            print(f"pass@{pass_k}: {pass_k_stats['pass_at_k']:.2%}  "
                  f"({pass_k_stats['num_tasks_passed']}/{pass_k_stats['num_tasks']} tasks passed)")
            print(f"  (task passes if ≥1 of {pass_k} trials succeeded)")

        print(f"{'='*60}\n")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run mis-evolved memory experiments on Odyssey-Arena"
    )
    
    # Pipeline mode
    parser.add_argument('--mode', type=str, default='infer',
                        choices=['infer', 'explore', 'pipeline'],
                        help=(
                            'Run mode: '
                            '"infer" = single-stage inference (default, backward-compat); '
                            '"explore" = exploration only, saves memory.json; '
                            '"pipeline" = explore then infer sequentially.'
                        ))
    parser.add_argument('--explore_episodes', type=int, default=20,
                        help='[explore/pipeline] Number of exploration episodes (default: 20)')
    parser.add_argument('--explore_task_file', type=str, default=None,
                        help='[explore/pipeline] Task file for exploration stage '
                             '(default: same as --task_file)')
    parser.add_argument('--explore_memory_file', type=str, default=None,
                        help='[infer] Path to a memory.json from a previous explore run. '
                             'Pre-loads memory before inference episodes begin.')

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
                        choices=['vllm', 'openai', 'openrouter', 'gemini', 'dummy'],
                        help='LLM provider (vllm for local, openai/openrouter/gemini for API)')
    parser.add_argument('--api_key', type=str, default=None,
                        help='API key for cloud providers (or use env vars)')
    parser.add_argument('--use_dummy_llm', action='store_true',
                        help='Use dummy LLM for testing (no GPU/API required)')
    
    # Memory
    parser.add_argument('--reflection_mode', type=str, default='traj',
                        choices=['none', 'mid', 'traj'],
                        help=(
                            'Reflection strategy for memory-based agents. '
                            'none: no reflection; '
                            'mid: k-step mid-episode Reflector-Curator only (requires --reflection_frequency); '
                            'traj: post-episode (full trajectory) reflection only. '
                            'Default: traj.'
                        ))
    parser.add_argument('--reflection_frequency', type=int, default=None,
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
    parser.add_argument('--max_state_visits', type=int, default=None, metavar='N',
                        help='Stop episode early (as failure) if the same observation is '
                             'visited N or more times. Default: None (disabled, always runs to max_steps)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--memory_persistence', action='store_true',
                        help='Keep memory across episodes (continuous task stream)')
    parser.add_argument('--task_sampling', type=str, default='cyclic',
                        choices=['cyclic', 'random'],
                        help='Task selection: cyclic (default) or random')
    parser.add_argument('--pass_k', type=int, default=1, metavar='K',
                        help='Pass@k evaluation: run K independent trials per task. '
                             'Overrides --num_episodes (total = num_tasks * K). '
                             'Default: 1 (standard single-trial mode)')
    parser.add_argument('--log_step_interval', type=int, default=25,
                        help='Log progress every N steps (0=disable, default: 25)')

    parser.add_argument('--parallel_episodes', type=int, default=1, metavar='N',
                        help='Number of episodes to run concurrently via asyncio (default: 1 = sequential). '
                             'Uses async LLM calls with rate-limit retry. '
                             'Incompatible with --memory_persistence.')

    # Checkpointing
    parser.add_argument('--checkpoint_interval', type=int, default=0, metavar='N',
                        help='Save checkpoint every N completed episodes to checkpoint.json '
                             '(0=disabled, default: 0). Checkpoint info is stored in progress.json '
                             'under the "checkpoint" field. If checkpoint.json already exists in '
                             '--output_dir when the experiment starts, it is automatically resumed.')
    
    # Prompt and format options
    parser.add_argument('--prompt_template', type=str, default='original',
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
    parser.add_argument('--include_raw_output', action='store_true', default=False,
                        help='Include the previous step\'s full raw LLM output (think + action tag) '
                             'in the prompt. Default: False.')
    parser.add_argument('--max_parse_errors', type=int, default=10, metavar='N',
                        help='Max LLM call retries per step when response is empty or unparseable. '
                             'Episode is aborted if all retries fail. Default: 10.')

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


def _tee_stdout_stderr_to_file(log_path: Path, mode: str = "w"):
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
    log_file = open(log_path, mode, encoding="utf-8")
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(orig_stdout, log_file)
    sys.stderr = Tee(orig_stderr, log_file)
    return orig_stdout, orig_stderr, log_file


def main():
    """Main entry point."""
    args = parse_args()
    config = vars(args)

    # ── reflection_mode 검증 및 내부 플래그 변환 ───────────────────────────
    reflection_mode = config.get('reflection_mode', 'traj')
    if reflection_mode == 'mid':
        if config.get('reflection_frequency') is None:
            import sys as _sys
            print("[ERROR] --reflection_mode mid 사용 시 --reflection_frequency N 이 필수입니다.")
            _sys.exit(1)
        config['mid_episode_reflection'] = True
        config['episode_reflection'] = False
    elif reflection_mode == 'traj':
        config['mid_episode_reflection'] = False
        config['episode_reflection'] = True
    else:  # 'none'
        config['mid_episode_reflection'] = False
        config['episode_reflection'] = False

    # reflection_frequency 미지정 시 기본값
    if config.get('reflection_frequency') is None:
        config['reflection_frequency'] = 5  # mid 아닌 모드에서는 의미 없으나 생성자에 필요

    mode = config.get('mode', 'infer')

    # ── Pipeline mode: delegate to PipelineRunner ──────────────────────
    if mode == 'pipeline':
        from experiments.pipeline import PipelineRunner
        runner = PipelineRunner(config)
        use_dashboard = args.dashboard
        log_file = None
        orig_stdout = orig_stderr = None
        _dashboard_finished = [False]

        def _on_exit_signal(signum, frame):
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            if use_dashboard and not _dashboard_finished[0]:
                _dashboard_finished[0] = True
                try:
                    from experiments.dashboard.registry import register_finish
                    register_finish(str(runner.output_dir), success=False)
                except Exception:
                    pass
            if log_file is not None and orig_stdout is not None:
                try:
                    sys.stdout = orig_stdout
                    sys.stderr = orig_stderr
                    log_file.close()
                except Exception:
                    pass
            os._exit(130 if signum == signal.SIGINT else 1)

        if use_dashboard:
            try:
                from experiments.dashboard.registry import register_start
                register_start(str(runner.output_dir), config, name=args.experiment_name)
                orig_stdout, orig_stderr, log_file = _tee_stdout_stderr_to_file(
                    runner.output_dir / "run.log", mode="w"
                )
                signal.signal(signal.SIGTERM, _on_exit_signal)
                signal.signal(signal.SIGINT, _on_exit_signal)
            except Exception as e:
                print(f"Dashboard register start failed: {e}", file=sys.__stderr__)

        notify_to = args.notify_email or os.environ.get("NOTIFY_TO", "")

        def _send_notify_pl(success: bool, summary=None):
            if not notify_to:
                return
            try:
                from experiments.notify import send_experiment_email
                send_experiment_email(
                    to=notify_to, experiment_name=args.experiment_name,
                    experiment_id=config.get("experiment_id"),
                    output_dir=str(runner.output_dir), success=success,
                    summary=summary,
                    config_summary={k: config.get(k) for k in (
                        "env", "agent", "model", "memory", "verification_mode",
                        "intervention", "num_episodes", "max_steps", "seed",
                        "mode", "explore_episodes")},
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
                    register_finish(str(runner.output_dir), success=True,
                                    summary=summary_for_finish)
                except Exception as e:
                    print(f"Dashboard register_finish failed: {e}", file=sys.__stderr__)
            _send_notify_pl(success=True, summary=summary_for_finish)
        except Exception:
            if use_dashboard and not _dashboard_finished[0]:
                _dashboard_finished[0] = True
                try:
                    from experiments.dashboard.registry import register_finish
                    register_finish(str(runner.output_dir), success=False)
                except Exception as e:
                    print(f"Dashboard register_finish failed: {e}", file=sys.__stderr__)
            _send_notify_pl(success=False)
            raise
        finally:
            if log_file is not None and orig_stdout is not None:
                sys.stdout = orig_stdout
                sys.stderr = orig_stderr
                log_file.close()
        return  # pipeline done

    # ── Explore-only mode: run ExperimentRunner with memory_persistence ─
    if mode == 'explore':
        config['memory_persistence'] = True
        config['num_episodes'] = config.get('explore_episodes', config.get('num_episodes', 20))
        if config.get('explore_task_file'):
            config['task_file'] = config['explore_task_file']

    # ── Infer-only mode: optionally pre-load memory from file ───────────
    if mode == 'infer' and config.get('explore_memory_file'):
        memory_file = Path(config['explore_memory_file'])
        if memory_file.exists():
            try:
                with open(memory_file, 'r', encoding='utf-8') as f:
                    config['_initial_memory_snapshot'] = json.load(f)
                config['memory_persistence'] = True
                print(f"[infer] Pre-loaded memory snapshot from {memory_file}")
            except Exception as e:
                print(f"Warning: could not load explore_memory_file: {e}", file=sys.__stderr__)

    runner = ExperimentRunner(config)

    # ── After explore run: save memory snapshot ─────────────────────────
    _mode_is_explore = (mode == 'explore')
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

    # Detect auto-resume: checkpoint.json exists → append log instead of overwriting
    _checkpoint_exists = (runner.output_dir / "checkpoint.json").exists()
    _log_mode = "a" if _checkpoint_exists else "w"

    if use_dashboard:
        try:
            from experiments.dashboard.registry import register_start
            register_start(str(runner.output_dir), config, name=args.experiment_name)
            orig_stdout, orig_stderr, log_file = _tee_stdout_stderr_to_file(
                runner.output_dir / "run.log", mode=_log_mode
            )
            if _checkpoint_exists:
                resume_marker = (
                    f"\n{'#'*70}\n"
                    f"##  RESUMED FROM CHECKPOINT  |  {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
                    f"{'#'*70}\n"
                )
                print(resume_marker)
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
        # Save memory snapshot after explore-only run
        if _mode_is_explore:
            snapshot = runner.get_memory_snapshot()
            if snapshot:
                memory_file = runner.output_dir / 'memory.json'
                try:
                    with open(memory_file, 'w', encoding='utf-8') as f:
                        json.dump(snapshot, f, indent=2, ensure_ascii=False)
                    print(f"\n[explore] Memory snapshot saved → {memory_file}")
                    print(f"[explore] Stats: {snapshot.get('stats', {})}")
                except Exception as e:
                    print(f"Warning: could not save memory.json: {e}")
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

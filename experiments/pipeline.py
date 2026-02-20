"""
Two-stage pipeline runner for Odyssey-Arena experiments.

Stage 1 (Explore): Agent freely explores the environment and builds memory
                   from train tasks (e.g. test_turnonlights_lite_251030_augmented.json).
Stage 2 (Infer):   Agent uses built memory to reach goal_state from the same
                   or a different task file.

Usage (via run.py --mode pipeline):
    python -m experiments.run \\
        --mode pipeline \\
        --explore_episodes 20 \\
        --num_episodes 30 \\
        --memory gated \\
        --output_dir results/pipeline_test \\
        ...

Usage (standalone stages):
    # Explore only (saves memory.json):
    python -m experiments.run --mode explore --explore_episodes 20 --output_dir results/explore ...

    # Infer only (loads memory.json from previous explore):
    python -m experiments.run --mode infer --explore_memory_file results/explore/memory.json ...

Output structure:
    output_dir/
        config.json           # full pipeline config
        run.log               # combined log
        progress.json         # stage-aware combined progress
        experiment_meta.json
        explore/
            episodes.json     # explore stage episodes
            summary.json      # explore stage metrics
            memory.json       # serialized memory snapshot
            progress.json     # explore stage progress
        infer/
            episodes.json     # infer stage episodes
            summary.json      # infer stage metrics (success rate)
            progress.json     # infer stage progress
        episodes.json         # copy of infer/episodes.json (backward compat)
        summary.json          # copy of infer/summary.json  (backward compat)
"""

import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from experiments.run import ExperimentRunner
from experiments.memory import load_memory_snapshot


# --------------------------------------------------------------------------- #
#  StageRunner: ExperimentRunner subclass with pipeline progress integration  #
# --------------------------------------------------------------------------- #

class StageRunner(ExperimentRunner):
    """
    ExperimentRunner that additionally:
    - Labels its output clearly with stage name (explore / infer)
    - Calls a pipeline_progress_hook after every _write_progress() so the
      parent PipelineRunner can update the combined progress.json atomically.
    """

    def __init__(self, config: Dict[str, Any],
                 pipeline_progress_hook: Optional[Callable] = None):
        """
        Args:
            config: ExperimentRunner config dict (with extra _stage key)
            pipeline_progress_hook: callable(episode_idx, step, max_steps,
                                             num_episodes, num_success)
                                    called after each local _write_progress().
        """
        self._pipeline_progress_hook = pipeline_progress_hook
        super().__init__(config)

    def _write_progress(self, episode_idx: int, step: int, max_steps: int,
                        num_episodes: int, num_success: int, env_progress: int):
        super()._write_progress(episode_idx, step, max_steps,
                                num_episodes, num_success, env_progress)
        if self._pipeline_progress_hook:
            try:
                self._pipeline_progress_hook(
                    episode_idx=episode_idx,
                    step=step,
                    max_steps=max_steps,
                    num_episodes=num_episodes,
                    num_success=num_success,
                )
            except Exception:
                pass


# --------------------------------------------------------------------------- #
#  PipelineRunner                                                              #
# --------------------------------------------------------------------------- #

class PipelineRunner:
    """
    Orchestrates Explore → Infer two-stage pipeline.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.explore_dir = self.output_dir / 'explore'
        self.infer_dir = self.output_dir / 'infer'
        self.explore_dir.mkdir(parents=True, exist_ok=True)
        self.infer_dir.mkdir(parents=True, exist_ok=True)

        n_explore = config.get('explore_episodes', 20)
        n_infer = config.get('num_episodes', 30)
        max_steps = config.get('max_steps', 200)

        # Pipeline-level progress state (written to output_dir/progress.json)
        self._progress: Dict[str, Any] = {
            'mode': 'pipeline',
            'current_stage': 'explore',
            'explore': {
                'status': 'pending',
                'episode': 0,
                'total_episodes': n_explore,
                'step': 0,
                'max_steps': max_steps,
                'elapsed_sec': 0.0,
            },
            'infer': {
                'status': 'pending',
                'episode': 0,
                'total_episodes': n_infer,
                'step': 0,
                'max_steps': max_steps,
                'num_success': 0,
                'success_rate': 0.0,
                'elapsed_sec': 0.0,
            },
            'updated_at': '',
        }

    # ------------------------------------------------------------------ #
    #  Progress helpers                                                    #
    # ------------------------------------------------------------------ #

    def _write_progress(self) -> None:
        self._progress['updated_at'] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        stage = self._progress['current_stage']
        stage_data = self._progress.get(stage, {})

        # Top-level fields kept for dashboard backward-compat (index.html JS)
        combined: Dict[str, Any] = {
            **self._progress,
            'episode': stage_data.get('episode', 0),
            'total_episodes': stage_data.get('total_episodes', 1),
            'step': stage_data.get('step', 0),
            'max_steps': stage_data.get('max_steps', self.config.get('max_steps', 200)),
            'success_rate': stage_data.get('success_rate', 0.0),
            'num_success': stage_data.get('num_success', 0),
        }
        progress_file = self.output_dir / 'progress.json'
        tmp = progress_file.with_suffix('.json.tmp')
        try:
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(combined, f)
            tmp.replace(progress_file)
        except Exception:
            pass

    def _make_explore_hook(self, start_time: float) -> Callable:
        def hook(episode_idx, step, max_steps, num_episodes, num_success, **_):
            self._progress['explore'].update({
                'status': 'running',
                'episode': episode_idx + 1,
                'total_episodes': num_episodes,
                'step': step,
                'max_steps': max_steps,
                'elapsed_sec': round(time.time() - start_time, 1),
            })
            self._progress['current_stage'] = 'explore'
            self._write_progress()
        return hook

    def _make_infer_hook(self, start_time: float) -> Callable:
        def hook(episode_idx, step, max_steps, num_episodes, num_success, **_):
            success_rate = round(num_success / (episode_idx + 1), 4) if episode_idx >= 0 else 0.0
            self._progress['infer'].update({
                'status': 'running',
                'episode': episode_idx + 1,
                'total_episodes': num_episodes,
                'step': step,
                'max_steps': max_steps,
                'num_success': num_success,
                'success_rate': success_rate,
                'elapsed_sec': round(time.time() - start_time, 1),
            })
            self._progress['current_stage'] = 'infer'
            self._write_progress()
        return hook

    # ------------------------------------------------------------------ #
    #  Stage config builders                                              #
    # ------------------------------------------------------------------ #

    def _explore_config(self) -> Dict[str, Any]:
        cfg = dict(self.config)
        cfg['output_dir'] = str(self.explore_dir)
        cfg['num_episodes'] = self.config.get('explore_episodes', 20)
        cfg['memory_persistence'] = True   # memory must grow across explore eps
        if self.config.get('explore_task_file'):
            cfg['task_file'] = self.config['explore_task_file']
        return cfg

    def _infer_config(self, memory_snapshot: Optional[Dict]) -> Dict[str, Any]:
        cfg = dict(self.config)
        cfg['output_dir'] = str(self.infer_dir)
        cfg['num_episodes'] = self.config.get('num_episodes', 30)
        cfg['memory_persistence'] = True   # keep memory across infer eps too
        cfg['_initial_memory_snapshot'] = memory_snapshot
        return cfg

    # ------------------------------------------------------------------ #
    #  run()                                                              #
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        # ── Stage 1: Explore ───────────────────────────────────────────
        print(f"\n{'#'*70}")
        print(f"##  PIPELINE  |  STAGE 1: EXPLORE  "
              f"({self.config.get('explore_episodes', 20)} episodes)")
        print(f"{'#'*70}\n")

        self._progress['explore']['status'] = 'running'
        self._write_progress()

        explore_start = time.time()
        explore_runner = StageRunner(
            config=self._explore_config(),
            pipeline_progress_hook=self._make_explore_hook(explore_start),
        )
        explore_runner.run()

        explore_elapsed = time.time() - explore_start
        self._progress['explore']['status'] = 'done'
        self._progress['explore']['elapsed_sec'] = round(explore_elapsed, 1)
        self._write_progress()

        # ── Memory handoff ─────────────────────────────────────────────
        memory_snapshot = explore_runner.get_memory_snapshot()
        if memory_snapshot:
            memory_file = self.explore_dir / 'memory.json'
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(memory_snapshot, f, indent=2, ensure_ascii=False)
            print(f"\n[Pipeline] Memory snapshot saved → {memory_file}")
            print(f"[Pipeline] Memory stats: {memory_snapshot.get('stats', {})}")
        else:
            print("\n[Pipeline] Warning: no persistent memory snapshot from explore stage "
                  "(check --memory type and --memory_persistence).")

        # ── Stage 2: Infer ─────────────────────────────────────────────
        print(f"\n{'#'*70}")
        print(f"##  PIPELINE  |  STAGE 2: INFER  "
              f"({self.config.get('num_episodes', 30)} episodes)")
        print(f"{'#'*70}\n")

        self._progress['infer']['status'] = 'running'
        self._progress['current_stage'] = 'infer'
        self._write_progress()

        infer_start = time.time()
        infer_runner = StageRunner(
            config=self._infer_config(memory_snapshot),
            pipeline_progress_hook=self._make_infer_hook(infer_start),
        )
        infer_runner.run()

        infer_elapsed = time.time() - infer_start
        self._progress['infer']['status'] = 'done'
        self._progress['infer']['elapsed_sec'] = round(infer_elapsed, 1)
        self._write_progress()

        # ── Propagate infer results to parent dir (backward compat) ────
        for fname in ('episodes.json', 'summary.json'):
            src = self.infer_dir / fname
            if src.exists():
                shutil.copy2(src, self.output_dir / fname)

        self._print_pipeline_summary(explore_elapsed, infer_elapsed)

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _print_pipeline_summary(self, explore_elapsed: float, infer_elapsed: float) -> None:
        infer_data = self._progress.get('infer', {})
        print(f"\n{'='*70}")
        print("PIPELINE SUMMARY")
        print(f"{'='*70}")
        print(f"  Explore stage: {self.config.get('explore_episodes', 20)} episodes "
              f"| elapsed: {explore_elapsed:.1f}s")
        print(f"  Infer  stage: {self.config.get('num_episodes', 30)} episodes "
              f"| elapsed: {infer_elapsed:.1f}s")
        print(f"  Infer success rate: {infer_data.get('success_rate', 0.0):.2%} "
              f"({infer_data.get('num_success', 0)}/{self.config.get('num_episodes', 30)})")
        print(f"{'='*70}\n")

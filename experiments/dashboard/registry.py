"""
Experiment registry for the dashboard.
Stores experiment_id, output_dir, status, config, summary in a JSON file.
"""

import fcntl
import json
import contextlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Default registry path: next to this file
DASHBOARD_DIR = Path(__file__).resolve().parent
DATA_DIR = DASHBOARD_DIR / "data"
REGISTRY_FILE = DATA_DIR / "registry.json"
LOCK_FILE = DATA_DIR / "registry.lock"


def _ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


@contextlib.contextmanager
def _registry_lock():
    """Exclusive file lock so parallel processes don't clobber the registry."""
    _ensure_data_dir()
    with open(LOCK_FILE, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def _load_registry() -> List[Dict[str, Any]]:
    _ensure_data_dir()
    if not REGISTRY_FILE.exists():
        return []
    try:
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def _save_registry(entries: List[Dict[str, Any]]):
    _ensure_data_dir()
    tmp = REGISTRY_FILE.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    tmp.replace(REGISTRY_FILE)


def generate_experiment_id() -> str:
    """Generate a unique experiment id (timestamp + microseconds to avoid collision on parallel starts)."""
    return datetime.utcnow().strftime("exp_%Y%m%d_%H%M%S_%f")


def register_start(output_dir: str, config: Dict[str, Any], name: Optional[str] = None) -> str:
    """
    Register experiment start. Returns experiment_id.
    output_dir: absolute or relative path to experiment output directory.
    name: optional display name; if not set, experiment_id is used.
    """
    output_path = Path(output_dir).resolve()
    experiment_id = generate_experiment_id()
    display_name = (name or "").strip() or experiment_id
    config_summary = {
        "env": config.get("env"),
        "agent": config.get("agent"),
        "model": config.get("model"),
        "memory": config.get("memory"),
        "verification_mode": config.get("verification_mode"),
        "intervention": config.get("intervention"),
        "num_episodes": config.get("num_episodes"),
        "max_steps": config.get("max_steps"),
        "seed": config.get("seed"),
        "mode": config.get("mode", "infer"),
        "explore_episodes": config.get("explore_episodes"),
    }
    started_at = datetime.utcnow().isoformat() + "Z"
    entry = {
        "experiment_id": experiment_id,
        "name": display_name,
        "output_dir": str(output_path),
        "status": "running",
        "started_at": started_at,
        "finished_at": None,
        "config_summary": config_summary,
        "config": config,
        "summary": None,
    }
    with _registry_lock():
        entries = _load_registry()
        entries.insert(0, entry)
        _save_registry(entries)
    # Write experiment_meta.json into output_dir for this run
    output_path.mkdir(parents=True, exist_ok=True)
    meta_file = output_path / "experiment_meta.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump({
            "experiment_id": experiment_id,
            "name": display_name,
            "status": "running",
            "started_at": started_at,
            "config_summary": config_summary,
        }, f, indent=2, ensure_ascii=False)
    return experiment_id


def register_finish(output_dir: str, success: bool, summary: Optional[Dict[str, Any]] = None):
    """
    Update registry and experiment_meta.json with final status and summary.
    """
    output_path = Path(output_dir).resolve()
    finished_at = datetime.utcnow().isoformat() + "Z"
    with _registry_lock():
        entries = _load_registry()
        for e in entries:
            if Path(e["output_dir"]).resolve() == output_path and e["status"] == "running":
                e["status"] = "completed" if success else "failed"
                e["finished_at"] = finished_at
                if summary is not None:
                    e["summary"] = summary
                _save_registry(entries)
                break
    meta_file = output_path / "experiment_meta.json"
    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (json.JSONDecodeError, IOError):
            meta = {}
        meta["status"] = "completed" if success else "failed"
        meta["finished_at"] = finished_at
        if summary is not None:
            meta["summary"] = summary
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)


def get_all_experiments() -> List[Dict[str, Any]]:
    """Return all experiments (newest first). Heal running entries if output has summary.json."""
    import os
    entries = _load_registry()
    healed = False
    for e in entries:
        if e.get("name") is None:
            e["name"] = e.get("experiment_id", "")
        if e.get("status") != "running":
            continue
        out_path = Path(e["output_dir"])
        summary_file = out_path / "summary.json"
        if not summary_file.exists():
            continue
        # Only heal if summary.json was written AFTER this run started.
        # Prevents old summary.json from a previous run marking a new (failed) entry as completed.
        started_at_str = e.get("started_at") or ""
        try:
            mtime = os.path.getmtime(summary_file)
            mtime_dt = datetime.utcfromtimestamp(mtime)
            if started_at_str:
                started_dt = datetime.fromisoformat(started_at_str.replace("Z", "+00:00"))
                # Remove tz info for naive comparison
                started_dt = started_dt.replace(tzinfo=None)
                if mtime_dt < started_dt:
                    continue  # summary.json is from a previous run, skip
        except Exception:
            continue  # can't determine, skip to be safe
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                e["summary"] = json.load(f)
            e["status"] = "completed"
            e["finished_at"] = e.get("finished_at") or datetime.utcnow().isoformat() + "Z"
            healed = True
        except (json.JSONDecodeError, IOError):
            pass
    if healed:
        _save_registry(entries)
    return entries


def delete_experiments(experiment_ids: List[str]) -> int:
    """
    Remove experiments from dashboard registry (does not delete output files).
    Returns number of entries removed.
    """
    if not experiment_ids:
        return 0
    ids_set = set(experiment_ids)
    with _registry_lock():
        entries = _load_registry()
        original_len = len(entries)
        entries = [e for e in entries if e.get("experiment_id") not in ids_set]
        removed = original_len - len(entries)
        if removed:
            _save_registry(entries)
    return removed

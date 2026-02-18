"""
Experiment dashboard: list experiments, view log and output per run.
Run with: python -m experiments.dashboard.app
For external access: host=0.0.0.0 and use ngrok/cloudflare tunnel.
"""

import json
from pathlib import Path
from flask import Flask, render_template, request, json as flask_json, abort, url_for

from experiments.dashboard.registry import get_all_experiments, delete_experiments

app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))
app.config["JSON_AS_ASCII"] = False


@app.context_processor
def inject_sort_url():
    """Build index URL with current filter params and given sort."""
    def sort_url(sort_by_val, sort_order_val):
        args = dict(request.args)
        args["sort_by"] = sort_by_val
        args["sort_order"] = sort_order_val
        return url_for("index", **args)
    return dict(sort_url=sort_url)

# Limit log file read size (last N bytes) for very long logs
MAX_LOG_TAIL_BYTES = 512 * 1024  # 512KB


def _read_log_tail(output_dir: str) -> str:
    log_path = Path(output_dir) / "run.log"
    if not log_path.exists():
        return "(no run.log yet)"
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(0, 2)
            size = f.tell()
            if size <= MAX_LOG_TAIL_BYTES:
                f.seek(0)
                return f.read()
            f.seek(size - MAX_LOG_TAIL_BYTES)
            return "... (truncated)\n" + f.read()
    except (IOError, OSError) as e:
        return f"(error reading log: {e})"


def _read_json_safe(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _filter_experiments(experiments: list, request) -> list:
    """Filter by status, name, started_*, finished_* query params. Dates compared as ISO strings."""
    status = request.args.get("status", "").strip().lower()
    name_contains = request.args.get("name", "").strip()
    started_after = request.args.get("started_after", "").strip()
    started_before = request.args.get("started_before", "").strip()
    finished_after = request.args.get("finished_after", "").strip()
    finished_before = request.args.get("finished_before", "").strip()

    out = experiments
    if status and status in ("running", "completed", "failed"):
        out = [e for e in out if (e.get("status") or "").lower() == status]
    if name_contains:
        key = name_contains.lower()
        out = [e for e in out if key in (e.get("name") or e.get("experiment_id") or "").lower()]
    if started_after:
        out = [e for e in out if (e.get("started_at") or "") >= started_after]
    if started_before:
        out = [e for e in out if (e.get("started_at") or "") <= started_before]
    if finished_after:
        out = [e for e in out if (e.get("finished_at") or "") >= finished_after]
    if finished_before:
        out = [e for e in out if (e.get("finished_at") or "") <= finished_before]
    return out


def _sort_experiments(experiments: list, sort_by: str, sort_order: str) -> list:
    """Sort by name, experiment_id, status, started_at, finished_at. order: asc or desc."""
    key = (sort_by or "started_at").strip().lower()
    if key not in ("name", "experiment_id", "status", "started_at", "finished_at"):
        key = "started_at"
    rev = (sort_order or "desc").strip().lower() == "desc"

    def sort_key(e):
        v = e.get(key) or e.get("experiment_id") or ""
        if key in ("started_at", "finished_at"):
            return v or ""  # ISO string, empty last
        return (v or "").lower() if isinstance(v, str) else str(v)

    return sorted(experiments, key=sort_key, reverse=rev)


def _load_progress(output_dir: str) -> dict:
    """Read progress.json for a given output_dir. Returns {} if missing."""
    return _read_json_safe(Path(output_dir) / "progress.json")


@app.route("/")
def index():
    experiments = get_all_experiments()
    # Attach progress snapshot for non-running experiments (failed/completed shown statically)
    for e in experiments:
        if e.get("status") != "running":
            e["_progress"] = _load_progress(e.get("output_dir", ""))
        else:
            e["_progress"] = None
    filtered = _filter_experiments(experiments, request)
    sort_by = request.args.get("sort_by", "started_at")
    sort_order = request.args.get("sort_order", "desc")
    sorted_list = _sort_experiments(filtered, sort_by, sort_order)
    return render_template(
        "index.html",
        experiments=sorted_list,
        filter_status=request.args.get("status", ""),
        filter_name=request.args.get("name", ""),
        filter_started_after=request.args.get("started_after", ""),
        filter_started_before=request.args.get("started_before", ""),
        filter_finished_after=request.args.get("finished_after", ""),
        filter_finished_before=request.args.get("finished_before", ""),
        sort_by=sort_by,
        sort_order=sort_order,
    )


def _format_json(obj: dict) -> str:
    if not obj:
        return ""
    return json.dumps(obj, indent=2, ensure_ascii=False)


@app.route("/experiment/<experiment_id>")
def experiment_detail(experiment_id):
    experiments = get_all_experiments()
    entry = next((e for e in experiments if e.get("experiment_id") == experiment_id), None)
    if not entry:
        abort(404)
    output_dir = entry["output_dir"]
    log_content = _read_log_tail(output_dir)
    config = _read_json_safe(Path(output_dir) / "config.json")
    summary = _read_json_safe(Path(output_dir) / "summary.json")
    if not summary and entry.get("summary"):
        summary = entry["summary"]
    episodes = _read_json_safe(Path(output_dir) / "episodes.json")
    if not isinstance(episodes, list):
        episodes = []
    return render_template(
        "detail.html",
        entry=entry,
        log_content=log_content,
        config_str=_format_json(config),
        summary_str=_format_json(summary),
        episodes=episodes,
    )


@app.route("/api/experiments")
def api_experiments():
    """JSON list for polling."""
    experiments = get_all_experiments()
    return flask_json.jsonify(experiments)


@app.route("/api/experiment/<experiment_id>/progress")
def api_progress(experiment_id):
    """Return progress.json content for a running experiment."""
    experiments = get_all_experiments()
    entry = next((e for e in experiments if e.get("experiment_id") == experiment_id), None)
    if not entry:
        abort(404)
    progress = _read_json_safe(Path(entry["output_dir"]) / "progress.json")
    return flask_json.jsonify({"progress": progress, "status": entry.get("status")})


@app.route("/api/experiment/<experiment_id>/log")
def api_log_tail(experiment_id):
    experiments = get_all_experiments()
    entry = next((e for e in experiments if e.get("experiment_id") == experiment_id), None)
    if not entry:
        abort(404)
    log_content = _read_log_tail(entry["output_dir"])
    return flask_json.jsonify({"log": log_content, "status": entry.get("status")})


@app.route("/api/experiments/delete", methods=["POST"])
def api_delete_experiments():
    """Remove experiments from dashboard (registry only; output files are not deleted)."""
    data = request.get_json(force=True, silent=True) or {}
    ids = data.get("experiment_ids") or []
    if not isinstance(ids, list):
        ids = [ids]
    removed = delete_experiments(ids)
    return flask_json.jsonify({"removed": removed})


def main():
    import argparse
    import os
    default_port = int(os.environ.get("PORT", "5000"))
    p = argparse.ArgumentParser(description="Experiment dashboard")
    p.add_argument("--host", default="0.0.0.0", help="Bind host (0.0.0.0 for external access)")
    p.add_argument("--port", type=int, default=default_port, help="Port (default: PORT env or 5000)")
    p.add_argument("--debug", action="store_true", help="Flask debug mode")
    args = p.parse_args()
    print(f"Dashboard: http://{args.host}:{args.port}/ (external: use ngrok or tunnel)")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

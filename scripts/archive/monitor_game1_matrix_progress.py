#!/usr/bin/env python3
"""Summarize live Game 1 multi-agent sample progress from result artifacts."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def parse_started_at(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def fmt_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, _ = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def slurm_states(job_ids: str | None) -> dict[str, dict[str, str]]:
    if not job_ids:
        return {}
    cmd = ["squeue", "-j", job_ids, "-h", "-o", "%i|%T|%M|%R"]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except OSError:
        return {}
    states: dict[str, dict[str, str]] = {}
    for line in proc.stdout.splitlines():
        parts = line.split("|", 3)
        if len(parts) == 4:
            job_id, state, elapsed, reason = parts
            states[job_id] = {"state": state, "elapsed": elapsed, "reason": reason}
    return states


def find_slurm_state(states: dict[str, dict[str, str]], config_id: int) -> dict[str, str] | None:
    suffix = f"_{config_id}"
    for job_id, state in states.items():
        if job_id.endswith(suffix) or job_id.endswith(f"[{config_id}]"):
            return {"job_id": job_id, **state}
    return None


def interaction_summary(output_dir: Path) -> tuple[int, dict[str, Any]]:
    interactions_path = output_dir / "all_interactions.json"
    if not interactions_path.exists():
        return 0, {}
    data = load_json(interactions_path)
    interactions = data.get("interactions", data) if isinstance(data, dict) else data
    if not interactions:
        return 0, {}
    return len(interactions), interactions[-1]


def progress_fraction(config: dict[str, Any], interaction_count: int, state: str) -> float:
    n_agents = int(config.get("n_agents") or len(config.get("models", [])) or 10)
    max_rounds = int(config.get("max_rounds") or 10)
    discussion_turns = int(config.get("discussion_turns") or 2)

    setup_interactions = n_agents
    per_round = (
        n_agents * discussion_turns
        + n_agents  # private thinking
        + n_agents  # proposals
        + n_agents * n_agents  # each agent votes on each proposal
        + n_agents  # reflection when no consensus
    )
    expected_max = setup_interactions + max_rounds * per_round
    if expected_max <= 0:
        return 0.0
    progress = min(interaction_count / expected_max, 1.0)
    if state == "COMPLETED":
        return 1.0
    return min(progress, 0.995)


def summarize(root: Path, job_ids: str | None) -> dict[str, Any]:
    now = datetime.now()
    states = slurm_states(job_ids)
    configs = sorted((root / "configs").glob("config_*.json"))
    rows = []

    for config_path in configs:
        config = load_json(config_path)
        config_id = int(config["config_id"])
        status_path = root / "status" / f"config_{config_id:04d}.json"
        status = load_json(status_path) if status_path.exists() else {}
        output_dir = Path(config["output_dir"])
        if not output_dir.is_absolute():
            output_dir = root.parent.parent.parent / output_dir
            if not output_dir.exists():
                output_dir = Path.cwd() / config["output_dir"]

        interaction_count, last = interaction_summary(output_dir)
        state = status.get("state", "UNKNOWN")
        slurm_state = find_slurm_state(states, config_id)
        progress = progress_fraction(config, interaction_count, state)
        started = parse_started_at(status.get("started_at"))
        elapsed_seconds = (now - started).total_seconds() if started else None
        remaining_seconds = None
        if elapsed_seconds and progress > 0.01 and state == "RUNNING":
            remaining_seconds = elapsed_seconds * (1.0 - progress) / progress

        rows.append(
            {
                "config_id": config_id,
                "family": config.get("experiment_family"),
                "model_order": config.get("model_order"),
                "state": state,
                "slurm": slurm_state,
                "interactions": interaction_count,
                "last_phase": last.get("phase"),
                "last_round": last.get("round"),
                "last_agent": last.get("agent_id"),
                "progress_pct_estimate": round(progress * 100, 1),
                "elapsed_estimate": fmt_duration(elapsed_seconds),
                "remaining_estimate": fmt_duration(remaining_seconds),
                "output_dir": str(output_dir),
            }
        )

    return {"timestamp": now.isoformat(timespec="seconds"), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--job-ids", default=None, help="Comma-separated Slurm job IDs to query")
    parser.add_argument("--write-log", action="store_true")
    args = parser.parse_args()

    root = Path(args.results_root)
    summary = summarize(root, args.job_ids)

    if args.write_log:
        monitor_dir = root / "monitor"
        monitor_dir.mkdir(parents=True, exist_ok=True)
        with (monitor_dir / "progress.jsonl").open("a") as f:
            f.write(json.dumps(summary, sort_keys=True) + "\n")
        with (monitor_dir / "progress_latest.json").open("w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

    print(f"Progress snapshot {summary['timestamp']}")
    for row in summary["rows"]:
        slurm = row.get("slurm") or {}
        slurm_bits = f" slurm={slurm.get('state', 'not-in-squeue')}"
        if slurm.get("job_id"):
            slurm_bits += f"({slurm['job_id']})"
        print(
            f"config_{row['config_id']:04d} {row['state']}{slurm_bits} "
            f"{row['progress_pct_estimate']}% "
            f"elapsed={row['elapsed_estimate']} remaining={row['remaining_estimate']} "
            f"round={row['last_round']} phase={row['last_phase']} "
            f"agent={row['last_agent']} interactions={row['interactions']}"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Print and optionally log progress for Game 3 sample batches."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOTAL_INTERACTIONS = 10 + 10 * 60


def resolve_root(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def load_configs(root: Path) -> List[Dict[str, Any]]:
    return [
        load_json(path)
        for path in sorted((root / "configs").glob("config_*.json"))
    ]


def result_path(output_dir: Path) -> Path | None:
    for name in ("experiment_results.json", "run_1_experiment_results.json"):
        candidate = output_dir / name
        if candidate.exists():
            return candidate
    return None


def interaction_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    return len(payload) if isinstance(payload, list) else 0


def summarize_config(root: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    config_id = int(config["config_id"])
    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir = output_dir.resolve()
    status_payload = load_json(root / "status" / f"config_{config_id:04d}.json")
    progress = load_json(output_dir / "progress.json")
    result = result_path(output_dir)
    interactions = output_dir / "all_interactions.json"

    status = status_payload.get("state") or "PENDING"
    if result is not None:
        status = "SUCCESS"

    total_estimate = 10 + int(config.get("max_rounds", 10)) * (
        int(config.get("discussion_turns", 2)) * int(config.get("n_agents", 10))
        + 4 * int(config.get("n_agents", 10))
    )
    count = int(progress.get("interaction_count") or interaction_count(interactions))
    pct = 100.0 if status == "SUCCESS" else min(99.0, count / max(total_estimate, 1) * 100.0)

    last = progress.get("last_interaction", {})
    if not last and interactions.exists():
        try:
            payload = json.loads(interactions.read_text(encoding="utf-8"))
            if payload:
                last_item = payload[-1]
                last = {
                    "round": last_item.get("round"),
                    "phase": last_item.get("phase"),
                    "agent_id": last_item.get("agent_id"),
                    "model_name": last_item.get("model_name"),
                }
        except json.JSONDecodeError:
            pass

    return {
        "config_id": config_id,
        "status": status,
        "experiment_type": config.get("experiment_family") or config.get("experiment_type"),
        "model_order": config.get("model_order"),
        "adversary_model": config.get("adversary_model"),
        "interactions": count,
        "estimated_total_interactions": total_estimate,
        "percent": pct,
        "last": last,
        "result_path": str(result) if result else None,
        "returncode": status_payload.get("returncode"),
        "updated_at": progress.get("updated_at") or status_payload.get("updated_at"),
    }


def format_report(root: Path, rows: List[Dict[str, Any]]) -> str:
    now = dt.datetime.now().isoformat(timespec="seconds")
    completed = sum(1 for row in rows if row["status"] == "SUCCESS")
    failed = sum(1 for row in rows if row["status"] == "FAILED")
    running = sum(1 for row in rows if row["status"] == "RUNNING")
    avg_pct = sum(row["percent"] for row in rows) / max(len(rows), 1)
    lines = [
        f"[{now}] Game 3 sample progress: {completed}/{len(rows)} SUCCESS, "
        f"{running} RUNNING, {failed} FAILED, mean estimated progress={avg_pct:.1f}%",
        f"root={root}",
    ]
    for row in rows:
        last = row.get("last") or {}
        lines.append(
            "  config_{config_id:04d} {status:<8} {percent:5.1f}% "
            "interactions={interactions}/{estimated_total_interactions} "
            "last=round {round} {phase} {agent} {model} order={model_order} adversary={adversary}".format(
                config_id=row["config_id"],
                status=row["status"],
                percent=row["percent"],
                interactions=row["interactions"],
                estimated_total_interactions=row["estimated_total_interactions"],
                round=last.get("round"),
                phase=last.get("phase"),
                agent=last.get("agent_id"),
                model=last.get("model_name"),
                model_order=row.get("model_order"),
                adversary=row.get("adversary_model"),
            )
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--write-log", action="store_true")
    args = parser.parse_args()

    root = resolve_root(args.results_root)
    rows = [summarize_config(root, config) for config in load_configs(root)]
    report = format_report(root, rows)
    print(report)
    if args.write_log:
        log_dir = root / "progress"
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / "progress_monitor.log").open("a", encoding="utf-8") as handle:
            handle.write(report + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

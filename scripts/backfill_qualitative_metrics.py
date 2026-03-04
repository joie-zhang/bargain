#!/usr/bin/env python3
"""Backfill structured qualitative metrics into existing co-funding result JSONs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strong_models_experiment.analysis.qualitative_metrics import (
    compute_qualitative_metrics_v1,
    extract_pledge_history_from_logs,
)


def _is_cofunding_result(data: Dict[str, Any]) -> bool:
    config = data.get("config", {})
    return config.get("game_type") == "co_funding"


def process_file(path: Path, include_events: bool, dry_run: bool) -> tuple[bool, bool]:
    """Process one result file.

    Returns:
        (processed, updated)
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False, False

    if not _is_cofunding_result(data):
        return False, False

    logs = data.get("conversation_logs", [])
    config = data.get("config", {})
    game_state = {
        "projects": config.get("items", []),
        "round_pledges": extract_pledge_history_from_logs(logs),
        "agent_budgets": config.get("agent_budgets", {}),
    }

    metrics, events = compute_qualitative_metrics_v1(logs, game_state)

    before = data.get("qualitative_metrics_v1")
    data["qualitative_metrics_v1"] = metrics
    if include_events:
        data["qualitative_events"] = events

    updated = before != metrics
    if updated and not dry_run:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    return True, updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments/results/cofunding_latest",
        help="Results directory containing *experiment_results.json files.",
    )
    parser.add_argument(
        "--include-events",
        action="store_true",
        help="Also write qualitative_events (larger files).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute stats only; do not modify files.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise SystemExit(f"Results directory not found: {results_dir}")

    files = sorted(results_dir.rglob("*experiment_results.json"))
    processed = 0
    updated = 0
    for path in files:
        p, u = process_file(path, include_events=args.include_events, dry_run=args.dry_run)
        processed += int(p)
        updated += int(u)

    print(f"Found files: {len(files)}")
    print(f"Processed co-funding results: {processed}")
    print(f"{'Would update' if args.dry_run else 'Updated'} files: {updated}")


if __name__ == "__main__":
    main()

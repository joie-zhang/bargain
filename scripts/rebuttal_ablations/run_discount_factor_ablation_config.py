#!/usr/bin/env python3
"""Idempotently run one discount-factor ablation manifest config."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def expected_result(config: dict[str, Any]) -> Path:
    output_dir = PROJECT_ROOT / config["output_dir"]
    return output_dir / f"run_{config['run_number']}_experiment_results.json"


def valid_existing_result(config: dict[str, Any]) -> bool:
    path = expected_result(config)
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    result_config = payload.get("config")
    if not isinstance(result_config, dict):
        return False
    final_utilities = payload.get("final_utilities")
    agent_performance = payload.get("agent_performance")
    has_terminal_utilities = (
        isinstance(final_utilities, dict)
        and len(final_utilities) == 2
    ) or (
        isinstance(agent_performance, dict)
        and len(agent_performance) == 2
        and all(
            isinstance(record, dict) and "final_utility" in record
            for record in agent_performance.values()
        )
    )
    return (
        payload.get("experiment_id") is not None
        and isinstance(payload.get("consensus_reached"), bool)
        and has_terminal_utilities
        and float(result_config.get("gamma_discount", -1)) == float(config["gamma_discount"])
        and float(result_config.get("competition_level", -1))
        == float(config["competition_level"])
        and int(result_config.get("random_seed", -1)) == int(config["random_seed"])
    )


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} CONFIG.json", file=sys.stderr)
        return 2

    config_path = Path(sys.argv[1]).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if valid_existing_result(config):
        print(f"Already complete: {expected_result(config)}")
        return 0

    output_dir = PROJECT_ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "experiment_name": config["experiment_name"],
        "ablation": config["ablation"],
        "config_id": config["config_id"],
        "baseline_model": config["baseline_model"],
        "adversary_model": config["adversary_model"],
        "seed_replicate": config["seed_replicate"],
        "conceptual_order": config["conceptual_order"],
    }
    env = os.environ.copy()
    env["EXPERIMENT_RUN_METADATA_JSON"] = json.dumps(metadata, sort_keys=True)

    command = [
        str(PROJECT_ROOT / ".venv" / "bin" / "python"),
        str(PROJECT_ROOT / "run_strong_models_experiment.py"),
        "--models",
        *[str(model) for model in config["models"]],
        "--batch",
        "--num-runs",
        "1",
        "--run-number",
        str(config["run_number"]),
        "--game-type",
        "item_allocation",
        "--num-items",
        str(config["num_items"]),
        "--max-rounds",
        str(config["max_rounds"]),
        "--competition-level",
        str(config["competition_level"]),
        "--gamma-discount",
        str(config["gamma_discount"]),
        "--random-seed",
        str(config["random_seed"]),
        "--discussion-turns",
        str(config["discussion_turns"]),
        "--model-order",
        str(config["model_order"]),
        "--output-dir",
        str(output_dir),
        "--job-id",
        str(config["config_id"]),
    ]
    print("Running config:", config_path)
    print("Expected result:", expected_result(config))
    print("Command:", " ".join(command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
    if completed.returncode != 0:
        return completed.returncode
    if not valid_existing_result(config):
        print(
            f"Runner returned success but expected result is absent or invalid: "
            f"{expected_result(config)}",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

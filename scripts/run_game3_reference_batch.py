#!/usr/bin/env python3
"""
Run a generated Game 3 reference batch from its config directory.

This is intentionally resume-friendly: completed configs are skipped by default.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNNER = PROJECT_ROOT / "run_strong_models_experiment.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--start-at", type=int, default=0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_results_root(raw_value: str) -> Path:
    candidate = Path(raw_value)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (Path.cwd() / candidate).resolve()
    if resolved.name == "configs":
        return resolved.parent
    return resolved


def discover_configs(results_root: Path) -> List[Path]:
    config_dir = results_root / "configs"
    return sorted(config_dir.glob("config_*.json"))


def output_files(config: dict) -> List[Path]:
    output_dir = (PROJECT_ROOT / config["output_dir"]).resolve()
    return [
        output_dir / "experiment_results.json",
        output_dir / "run_1_experiment_results.json",
    ]


def is_completed(config: dict) -> bool:
    return any(path.exists() for path in output_files(config))


def build_command(config: dict, config_index: int) -> List[str]:
    cmd = [
        sys.executable,
        str(RUNNER),
        "--game-type",
        "co_funding",
        "--models",
        *config["models"],
        "--batch",
        "--num-runs",
        str(config["num_runs"]),
        "--run-number",
        str(config["run_number"]),
        "--job-id",
        str(config_index),
        "--m-projects",
        str(config["m_projects"]),
        "--alpha",
        str(config["alpha"]),
        "--sigma",
        str(config["sigma"]),
        "--c-min",
        str(config["c_min"]),
        "--c-max",
        str(config["c_max"]),
        "--max-rounds",
        str(config["max_rounds"]),
        "--discussion-turns",
        str(config["discussion_turns"]),
        "--model-order",
        str(config["model_order"]),
        "--max-tokens-per-phase",
        str(config["max_tokens_per_phase"]),
        "--random-seed",
        str(config["random_seed"]),
        "--cofunding-discussion-transparency",
        str(config["cofunding_discussion_transparency"]),
        "--cofunding-time-discount",
        str(config["cofunding_time_discount"]),
        "--output-dir",
        str(config["output_dir"]),
    ]
    if not config.get("cofunding_enable_commit_vote", True):
        cmd.append("--cofunding-disable-commit-vote")
    if not config.get("cofunding_enable_time_discount", True):
        cmd.append("--cofunding-disable-time-discount")
    return cmd


def main() -> int:
    args = parse_args()
    results_root = resolve_results_root(args.results_dir)
    config_paths = discover_configs(results_root)
    if not config_paths:
        raise SystemExit(f"No config_*.json files found under {results_root / 'configs'}")

    log_dir = results_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    remaining_budget = args.max_configs
    executed = 0

    for config_path in config_paths[args.start_at:]:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        exp_id = int(config["experiment_id"])
        if not args.force and is_completed(config):
            print(f"Skipping completed config {config_path.name} (experiment_id={exp_id})")
            continue

        log_path = log_dir / f"{config_path.stem}.log"
        cmd = build_command(config, exp_id)
        env = os.environ.copy()
        env.setdefault("OPENROUTER_TRANSPORT", "auto")

        print(f"Running {config_path.name} -> {config['output_dir']}")
        print(" ".join(cmd))
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"$ {' '.join(cmd)}\n")
            handle.flush()
            completed = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        if completed.returncode != 0:
            print(
                f"Config {config_path.name} failed with exit code {completed.returncode}. "
                f"See {log_path}",
                file=sys.stderr,
            )
            return completed.returncode

        executed += 1
        if remaining_budget is not None and executed >= remaining_budget:
            break
        if args.sleep_seconds > 0:
            import time

            time.sleep(args.sleep_seconds)

    print(f"Executed {executed} configs from {results_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

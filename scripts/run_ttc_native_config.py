#!/usr/bin/env python3
"""Run one native test-time-compute scaling config."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAX_TOKENS_PER_PHASE_LIMIT = 16_384
LEGACY_MAX_TOKENS_PER_PHASE = 10_500
# Anthropic's output ceiling for claude-sonnet-4-6 / claude-opus-4-6, mirroring the
# clamp in strong_models_experiment/agents/agent_factory.py. Only reachable when a
# config opts in explicitly via allow_extended_max_tokens_per_phase.
EXTENDED_MAX_TOKENS_PER_PHASE_LIMIT = 65_536


def load_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_max_tokens_per_phase(config: Dict[str, Any]) -> int:
    """Resolve the per-phase output cap for native TTC configs.

    Older generated TTC configs used 10500. Treat that as the old default and
    upgrade it to the current cap, while preserving smaller intentional caps.
    Seed-replication configs can opt out of that migration so that their cap
    exactly matches the archived source experiment.

    A config may set allow_extended_max_tokens_per_phase=true to raise its ceiling
    to the provider maximum. This exists for adaptive-thinking models at max effort,
    which can consume the whole standard 16384 cap on reasoning and return empty
    visible content. A run using the extended ceiling is NOT comparable to the
    standard grid; the resolved cap is recorded in the result file so such runs stay
    identifiable.
    """
    limit = (
        EXTENDED_MAX_TOKENS_PER_PHASE_LIMIT
        if config.get("allow_extended_max_tokens_per_phase", False)
        else MAX_TOKENS_PER_PHASE_LIMIT
    )

    raw_value = config.get("max_tokens_per_phase", MAX_TOKENS_PER_PHASE_LIMIT)
    if raw_value is None:
        return limit

    value = int(raw_value)
    if config.get("preserve_config_max_tokens_per_phase", False):
        return min(value, limit)
    if value == LEGACY_MAX_TOKENS_PER_PHASE:
        return limit
    return min(value, limit)


def add_game_args(cmd: List[str], config: Dict[str, Any]) -> None:
    game_type = config["game_type"]
    cmd.extend(["--game-type", game_type])
    if game_type == "item_allocation":
        cmd.extend(["--competition-level", str(config["competition_level"])])
        cmd.extend(["--num-items", str(config.get("m_items", 5))])
    elif game_type == "diplomacy":
        cmd.extend(["--rho", str(config["rho"])])
        cmd.extend(["--theta", str(config["theta"])])
        cmd.extend(["--n-issues", str(config.get("n_issues", 5))])
    elif game_type == "co_funding":
        cmd.extend(["--alpha", str(config["alpha"])])
        cmd.extend(["--sigma", str(config["sigma"])])
        cmd.extend(["--m-projects", str(config.get("m_projects", 5))])
        cmd.extend(["--c-min", str(config.get("c_min", 10.0))])
        cmd.extend(["--c-max", str(config.get("c_max", 30.0))])
        if not config.get("cofunding_enable_commit_vote", True):
            cmd.append("--cofunding-disable-commit-vote")
        if not config.get("cofunding_enable_time_discount", True):
            cmd.append("--cofunding-disable-time-discount")
        cmd.extend([
            "--cofunding-discussion-transparency",
            str(config.get("cofunding_discussion_transparency", "own")),
            "--cofunding-time-discount",
            str(config.get("cofunding_time_discount", 0.9)),
        ])
    else:
        raise ValueError(f"Unsupported game_type: {game_type}")


def build_command(config: Dict[str, Any]) -> List[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "run_strong_models_experiment.py"),
        "--models",
        *config["models"],
        "--batch",
        "--num-runs",
        "1",
        "--run-number",
        str(config.get("run_number", 1)),
        "--max-rounds",
        str(config.get("max_rounds", 10)),
        "--gamma-discount",
        str(config.get("gamma_discount", 0.9)),
        "--random-seed",
        str(config["random_seed"]),
        "--discussion-turns",
        str(config.get("discussion_turns", 2)),
        "--model-order",
        config["model_order"],
        "--max-tokens-per-phase",
        str(resolve_max_tokens_per_phase(config)),
        "--output-dir",
        config["output_dir"],
        "--job-id",
        str(config["config_id"]),
    ]
    add_game_args(cmd, config)
    return cmd


def experiment_metadata(config: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "config_id",
        "experiment_name",
        "target_provider",
        "target_model_family",
        "target_model",
        "target_model_id",
        "target_reasoning_level_requested",
        "target_reasoning_level_index",
        "baseline_model",
        "baseline_reasoning_level_requested",
        "order",
        "target_position",
        "reasoning_agent_index",
        "game_cell_label",
        "game_cell_id",
        "game_label",
        "seed_label",
    ]
    return {key: config[key] for key in keys if key in config}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    cmd = build_command(config)

    env = os.environ.copy()
    env["EXPERIMENT_RUN_METADATA_JSON"] = json.dumps(experiment_metadata(config), sort_keys=True)
    env.setdefault("PYTHONUNBUFFERED", "1")

    print(f"Config: {config_path}")
    print("Command:")
    print(" ".join(cmd))
    print("Metadata:")
    print(env["EXPERIMENT_RUN_METADATA_JSON"])

    if args.dry_run:
        return 0

    completed = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

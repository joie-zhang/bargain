#!/usr/bin/env python3
"""Build the matched 100-run Game 1 GPT-5.4 baseline-team treatment."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path("/scratch/gpfs/DANQIC/jz4391/bargain")
DEFAULT_CONTROL_ROOT = (
    PROJECT_ROOT
    / "experiments/results/full_games123_multiagent_production_20260428_085255"
)
EXPECTED_N = (2, 4, 6, 8, 10)
EXPECTED_COMPETITION = (0.0, 0.25, 0.5, 0.75, 1.0)
EXPECTED_POSITIONS = ("first", "last")
EXPECTED_SEEDS = (1, 2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def token(value: Any) -> str:
    rendered = str(value).replace(".", "p").replace("-", "_").replace("/", "_")
    return re.sub(r"[^A-Za-z0-9_]+", "_", rendered).strip("_")


def load_control_configs(control_root: Path) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for path in sorted((control_root / "configs").glob("config_*.json")):
        config = json.loads(path.read_text(encoding="utf-8"))
        if (
            config.get("game_label") == "game1"
            and config.get("experiment_family") == "homogeneous_adversary"
            and config.get("adversary_model") == "gpt-5.4-high"
        ):
            config["_source_config_path"] = str(path.resolve())
            selected.append(config)
    selected.sort(
        key=lambda row: (
            int(row["n_agents"]),
            float(row["competition_level"]),
            str(row["adversary_position"]),
            int(row["seed_replicate"]),
        )
    )
    return selected


def validate_factorial(configs: list[dict[str, Any]]) -> None:
    if len(configs) != 100:
        raise ValueError(f"Expected 100 GPT-5.4 Game 1 controls, found {len(configs)}")
    observed = Counter(
        (
            int(row["n_agents"]),
            float(row["competition_level"]),
            str(row["adversary_position"]),
            int(row["seed_replicate"]),
        )
        for row in configs
    )
    expected = Counter(
        (n_agents, competition, position, seed)
        for n_agents in EXPECTED_N
        for competition in EXPECTED_COMPETITION
        for position in EXPECTED_POSITIONS
        for seed in EXPECTED_SEEDS
    )
    if observed != expected:
        missing = sorted((expected - observed).elements())
        extra = sorted((observed - expected).elements())
        raise ValueError(f"Factorial mismatch; missing={missing[:5]}, extra={extra[:5]}")


def build_treatment_config(
    source: dict[str, Any],
    *,
    config_id: int,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_config_path = Path(source.pop("_source_config_path"))
    source_output_dir = Path(source["output_dir"])
    source_result_path = source_output_dir / "experiment_results.json"
    if not source_result_path.exists():
        raise FileNotFoundError(f"Missing control result: {source_result_path}")
    source_result = json.loads(source_result_path.read_text(encoding="utf-8"))
    fixed_agent_preferences = source_result.get("agent_preferences")
    if not isinstance(fixed_agent_preferences, dict):
        raise ValueError(f"Control result has no agent preference table: {source_result_path}")

    config = json.loads(json.dumps(source))
    n_agents = int(config["n_agents"])
    baseline_ids = [
        agent_id
        for agent_id, role in config["agent_role_map"].items()
        if role == "baseline"
    ]
    adversary_ids = [
        agent_id
        for agent_id, role in config["agent_role_map"].items()
        if role == "adversary"
    ]
    if len(baseline_ids) != n_agents - 1 or len(adversary_ids) != 1:
        raise ValueError(
            f"Bad role map for source config {source['config_id']}: "
            f"baseline={baseline_ids}, adversary={adversary_ids}"
        )
    if set(fixed_agent_preferences) != set(config["agent_role_map"]):
        raise ValueError(
            f"Control preference IDs do not match roles for source config {source['config_id']}"
        )
    if any(
        not isinstance(values, list) or len(values) != int(config["num_items"])
        for values in fixed_agent_preferences.values()
    ):
        raise ValueError(
            f"Control preference dimensions are invalid for source config {source['config_id']}"
        )

    run_name = (
        f"config_{config_id:04d}_game1_team_coordination_n{n_agents}_"
        f"comp_{token(config['competition_level'])}_gpt_5p4_high_"
        f"{config['adversary_position']}_seed{config['seed_replicate']}"
    )
    output_dir = output_root / "runs" / run_name

    config.update(
        {
            "config_id": config_id,
            "job_id": config_id,
            "batch_type": "game1_gpt54_team_coordination",
            "experiment_family": "homogeneous_adversary_team_coordination",
            "experiment_type": "homogeneous_adversary_team_coordination",
            "treatment_arm": "baseline_team_coordination",
            "control_config_id": int(source["config_id"]),
            "control_config_path": str(source_config_path.resolve()),
            "control_result_path": str(source_result_path.resolve()),
            "fixed_agent_preferences": fixed_agent_preferences,
            "fixed_agent_preferences_sha256": canonical_json_sha256(
                fixed_agent_preferences
            ),
            "output_dir": str(output_dir.resolve()),
            "team_coordination": {
                "enabled": True,
                "protocol_version": "baseline-private-team-v1",
                "member_ids": baseline_ids,
                "captain_id": baseline_ids[0],
                "share_full_team_preferences": True,
                "share_private_thinking": True,
                "share_private_voting": True,
                "share_reflection": True,
                "extra_api_calls": 0,
                "singleton_policy": "true_noop_negative_control",
                "joint_objective": "maximize_sum_discounted_baseline_utility",
            },
            "model_config_overrides": {
                "gpt-5.4-high": {
                    "api_type": "openai",
                    "provider": "OpenAI",
                    "model_id": "gpt-5.4",
                    "reasoning_effort": "high",
                    "custom_parameters": {
                        "phase_token_cap_policy": "prefer_model_cap_when_experiment_default"
                    },
                }
            },
        }
    )

    protected_fields = (
        "game_label",
        "game_type",
        "n_agents",
        "num_items",
        "competition_level",
        "competition_id",
        "random_seed",
        "seed",
        "seed_replicate",
        "models",
        "agent_model_map",
        "agent_role_map",
        "adversary_model",
        "adversary_position",
        "model_order",
        "max_rounds",
        "discussion_turns",
        "gamma_discount",
        "parallel_phases",
    )
    for field in protected_fields:
        if config[field] != source[field]:
            raise AssertionError(f"Treatment changed protected field {field}")

    lineage = {
        "config_id": config_id,
        "control_config_id": int(source["config_id"]),
        "n_agents": n_agents,
        "team_size": n_agents - 1,
        "competition_level": float(config["competition_level"]),
        "adversary_position": str(config["adversary_position"]),
        "seed_replicate": int(config["seed_replicate"]),
        "random_seed": int(config["random_seed"]),
        "control_config_path": str(source_config_path.resolve()),
        "control_config_sha256": sha256(source_config_path),
        "control_result_path": str(source_result_path.resolve()),
        "control_result_sha256": sha256(source_result_path),
        "fixed_agent_preferences_sha256": canonical_json_sha256(
            fixed_agent_preferences
        ),
        "treatment_output_dir": str(output_dir.resolve()),
    }
    return config, lineage


def write_sbatch(output_root: Path, max_concurrent: int) -> Path:
    path = output_root / "slurm" / "run_team_gpt54.sbatch"
    script = f"""#!/bin/bash
#SBATCH --job-name=team54
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --partition=cpu
#SBATCH --output={output_root}/slurm/team54_%A_%a.out
#SBATCH --error={output_root}/slurm/team54_%A_%a.err

set -eo pipefail

BASE_DIR="{PROJECT_ROOT}"
RUN_DIR="{output_root}"
cd "$BASE_DIR"

module purge
module load anaconda3/2024.2
module load proxy/default

KEY_ENV_FILE="${{BARGAIN_API_KEYS_ENV:-/home/jz4391/.config/bargain/api_keys.env}}"
if [[ -f "$KEY_ENV_FILE" ]]; then
  set -a
  source "$KEY_ENV_FILE"
  set +a
fi

export OPENROUTER_PROVIDER_FALLBACK=0
export OPENAI_TRANSPORT=proxy
export OPENAI_PROXY_POLL_DIR=/home/jz4391/openrouter_proxy
export LLM_FAILURE_REPORT_PATH="$RUN_DIR/monitoring/provider_failures.md"
export PYTHONUNBUFFERED=1

"{PROJECT_ROOT}/.venv/bin/python" \
  "{PROJECT_ROOT}/scripts/full_games123_multiagent_batch.py" \
  run-one --results-root "$RUN_DIR" --config-id "$SLURM_ARRAY_TASK_ID"
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, default=DEFAULT_CONTROL_ROOT)
    parser.add_argument("--max-concurrent", type=int, default=20)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    control_root = args.control_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output root: {output_root}")

    controls = load_control_configs(control_root)
    validate_factorial(controls)
    treatment_configs: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    for config_id, source in enumerate(controls, start=1):
        treatment, lineage = build_treatment_config(
            source,
            config_id=config_id,
            output_root=output_root,
        )
        treatment_configs.append(treatment)
        lineage_rows.append(lineage)

    for directory in (
        "configs",
        "logs",
        "monitoring",
        "runs",
        "status",
        "slurm",
        "analysis",
    ):
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    for config in treatment_configs:
        write_json(
            output_root / "configs" / f"config_{config['config_id']:04d}.json",
            config,
        )

    fieldnames = list(lineage_rows[0])
    with (output_root / "control_lineage.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(lineage_rows)

    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest = {
        "generated_at": generated_at,
        "results_root": str(output_root),
        "control_root": str(control_root),
        "design": "matched_game1_gpt54_baseline_team_coordination",
        "n_configs": len(treatment_configs),
        "n_values": list(EXPECTED_N),
        "team_sizes": [value - 1 for value in EXPECTED_N],
        "competition_levels": list(EXPECTED_COMPETITION),
        "adversary_positions": list(EXPECTED_POSITIONS),
        "seed_replicates": list(EXPECTED_SEEDS),
        "max_concurrent": int(args.max_concurrent),
        "slurm_time": "08:00:00",
        "gpt54_route": "direct_openai",
        "openai_transport": "file_proxy_on_slurm",
        "gpt54_reasoning_effort": "high",
        "openrouter_provider_fallback": False,
        "coordination_protocol_version": "baseline-private-team-v1",
        "control_lineage": str((output_root / "control_lineage.csv").resolve()),
    }
    write_json(output_root / "manifest.json", manifest)
    sbatch_path = write_sbatch(output_root, int(args.max_concurrent))

    print(json.dumps({
        "output_root": str(output_root),
        "configs": len(treatment_configs),
        "sbatch_path": str(sbatch_path),
        "max_concurrent": int(args.max_concurrent),
    }, indent=2))


if __name__ == "__main__":
    main()

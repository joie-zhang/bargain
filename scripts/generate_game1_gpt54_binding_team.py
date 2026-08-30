#!/usr/bin/env python3
"""Build the matched 100-run Game 1 GPT-5.4 binding-team treatment."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path("/scratch/gpfs/DANQIC/jz4391/bargain")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_game1_gpt54_team_coordination import (  # noqa: E402
    DEFAULT_CONTROL_ROOT,
    EXPECTED_COMPETITION,
    EXPECTED_N,
    EXPECTED_POSITIONS,
    EXPECTED_SEEDS,
    build_treatment_config,
    load_control_configs,
    validate_factorial,
    write_json,
)


PROTOCOL_VERSION = "binding-team-three-turn-v3-env-utility"
SOURCE_FILES = (
    "negotiation/llm_agents.py",
    "strong_models_experiment/experiment.py",
    "strong_models_experiment/phases/phase_handlers.py",
    "scripts/full_games123_multiagent_batch.py",
    "scripts/generate_game1_gpt54_binding_team.py",
    "scripts/analyze_game1_gpt54_team_coordination.py",
    "scripts/analyze_game1_gpt54_binding_team.py",
    "scripts/audit_export_game1_gpt54_binding_team.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_text(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def build_binding_config(
    source: dict[str, Any],
    *,
    config_id: int,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config, lineage = build_treatment_config(
        source,
        config_id=config_id,
        output_root=output_root,
    )
    members = list(config["team_coordination"]["member_ids"])
    captain_id = members[(config_id - 1) % len(members)]
    config.update(
        {
            "batch_type": "game1_gpt54_binding_team_v3",
            "experiment_family": "homogeneous_adversary_binding_team",
            "experiment_type": "homogeneous_adversary_binding_team",
            "treatment_arm": "binding_nano_team_three_turn_env_utility",
            "max_tokens_discussion": 16384,
            "max_tokens_proposal": 16384,
            "max_tokens_voting": 16384,
            "max_tokens_reflection": 16384,
            "max_tokens_thinking": 16384,
            "max_tokens_default": 16384,
            "max_tokens_per_phase": 16384,
            "sampling_temperature": 1.0,
            "team_coordination": {
                "enabled": True,
                "protocol_version": PROTOCOL_VERSION,
                "member_ids": members,
                "captain_id": captain_id,
                "rotate_captain_each_round": True,
                "share_full_team_preferences": True,
                "share_private_thinking": False,
                "share_private_voting": False,
                "share_reflection": False,
                "share_team_planning": True,
                "planning_turns": 3,
                "planning_max_tokens": 8192,
                "ballot_max_tokens": 4096,
                "max_action_repairs": 1,
                "hard_fail_on_action_error": True,
                "synthetic_actions_allowed": False,
                "formal_coalition_proposal": True,
                "binding_ballot": True,
                "singleton_policy": "true_noop_negative_control",
                "joint_objective": "maximize_expected_discounted_sum_nano_utility",
                "utility_evaluator": "environment",
                "utility_reporting_required": False,
                "tie_break": "seeded_existing_tabulation_rule",
            },
            "model_config_overrides": {
                "gpt-5-nano": {"temperature": 1.0},
                "gpt-5.4-high": {
                    "api_type": "openai",
                    "provider": "OpenAI",
                    "model_id": "gpt-5.4",
                    "temperature": 1.0,
                    "reasoning_effort": "high",
                    "custom_parameters": {
                        "phase_token_cap_policy": "prefer_model_cap_when_experiment_default"
                    },
                },
            },
        }
    )
    lineage.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "initial_captain_id": captain_id,
            "planning_turns": 3,
            "binding_ballot": True,
            "synthetic_actions_allowed": False,
        }
    )
    return config, lineage


def write_sbatch(output_root: Path) -> Path:
    path = output_root / "slurm" / "run_binding_team_gpt54.sbatch"
    script = f"""#!/bin/bash
#SBATCH --job-name=bind54
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=cpu
#SBATCH --output={output_root}/slurm/bind54_%A_%a.out
#SBATCH --error={output_root}/slurm/bind54_%A_%a.err

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
export OPENROUTER_TRANSPORT=proxy
export OPENROUTER_PROXY_POLL_DIR=/home/jz4391/openrouter_proxy
export OPENROUTER_PROXY_CLIENT_TIMEOUT=9000
export OPENAI_TRANSPORT=proxy
export OPENAI_PROXY_POLL_DIR=/home/jz4391/openrouter_proxy
export OPENAI_PROXY_CLIENT_TIMEOUT=9000
export EXPERIMENT_EXTERNALIZE_PROMPTS=1
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
    configs: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    for config_id, source in enumerate(controls, start=1):
        config, lineage = build_binding_config(
            source,
            config_id=config_id,
            output_root=output_root,
        )
        configs.append(config)
        lineage_rows.append(lineage)

    for directory in (
        "configs",
        "logs",
        "monitoring",
        "runs",
        "status",
        "slurm",
        "analysis",
        "transcripts",
        "rollouts",
    ):
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    for config in configs:
        write_json(
            output_root / "configs" / f"config_{config['config_id']:04d}.json",
            config,
        )

    with (output_root / "control_lineage.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(lineage_rows[0]))
        writer.writeheader()
        writer.writerows(lineage_rows)

    status_text = git_text("status", "--short")
    source_hashes = {
        relative_path: sha256(PROJECT_ROOT / relative_path)
        for relative_path in SOURCE_FILES
    }
    manifest = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "results_root": str(output_root),
        "control_root": str(control_root),
        "design": "matched_game1_gpt54_binding_nano_team",
        "n_configs": len(configs),
        "n_values": list(EXPECTED_N),
        "team_sizes": [value - 1 for value in EXPECTED_N],
        "competition_levels": list(EXPECTED_COMPETITION),
        "adversary_positions": list(EXPECTED_POSITIONS),
        "seed_replicates": list(EXPECTED_SEEDS),
        "max_concurrent": int(args.max_concurrent),
        "slurm_time": "12:00:00",
        "gpt54_route": "direct_openai_through_file_proxy",
        "nano_route": "direct_openai_through_file_proxy",
        "gpt54_reasoning_effort": "high",
        "openrouter_provider_fallback": False,
        "coordination_protocol_version": PROTOCOL_VERSION,
        "objective": "maximize_expected_discounted_sum_nano_utility",
        "planning_turns_per_round": 3,
        "binding_coalition_proposal": True,
        "binding_team_ballot": True,
        "synthetic_actions_allowed": False,
        "historical_controls_are_noncontemporaneous": True,
        "git_head": git_text("rev-parse", "HEAD"),
        "git_worktree_dirty": bool(status_text),
        "git_status_sha256": hashlib.sha256(status_text.encode("utf-8")).hexdigest(),
        "source_file_sha256": source_hashes,
        "control_lineage": str((output_root / "control_lineage.csv").resolve()),
    }
    write_json(output_root / "manifest.json", manifest)
    sbatch_path = write_sbatch(output_root)
    print(json.dumps({
        "output_root": str(output_root),
        "configs": len(configs),
        "sbatch_path": str(sbatch_path),
        "max_concurrent": int(args.max_concurrent),
    }, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate native test-time-compute scaling configs and CPU Slurm wrappers."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_MODEL = "gpt-5-nano"
BASELINE_REASONING_LEVEL = "low"
MAX_TOKENS_PER_PHASE = 16384
MAX_ROUNDS = 10
DISCUSSION_TURNS = 2
GAMMA_DISCOUNT = 0.9
SEED = 42

MODEL_CONDITIONS = [
    {
        "provider": "OpenAI",
        "family": "gpt-5",
        "model": "gpt-5-minimal-effort",
        "model_id": "gpt-5-2025-08-07",
        "level": "minimal",
        "level_index": 0,
    },
    {
        "provider": "OpenAI",
        "family": "gpt-5",
        "model": "gpt-5-low-effort",
        "model_id": "gpt-5-2025-08-07",
        "level": "low",
        "level_index": 1,
    },
    {
        "provider": "OpenAI",
        "family": "gpt-5",
        "model": "gpt-5-medium-effort",
        "model_id": "gpt-5-2025-08-07",
        "level": "medium",
        "level_index": 2,
    },
    {
        "provider": "OpenAI",
        "family": "gpt-5",
        "model": "gpt-5-high-effort",
        "model_id": "gpt-5-2025-08-07",
        "level": "high",
        "level_index": 3,
    },
    {
        "provider": "Anthropic",
        "family": "claude-sonnet-4-6",
        "model": "claude-sonnet-4-6-effort-low",
        "model_id": "claude-sonnet-4-6",
        "level": "low",
        "level_index": 0,
    },
    {
        "provider": "Anthropic",
        "family": "claude-sonnet-4-6",
        "model": "claude-sonnet-4-6-effort-medium",
        "model_id": "claude-sonnet-4-6",
        "level": "medium",
        "level_index": 1,
    },
    {
        "provider": "Anthropic",
        "family": "claude-sonnet-4-6",
        "model": "claude-sonnet-4-6-effort-high",
        "model_id": "claude-sonnet-4-6",
        "level": "high",
        "level_index": 2,
    },
    {
        "provider": "Anthropic",
        "family": "claude-sonnet-4-6",
        "model": "claude-sonnet-4-6-effort-max",
        "model_id": "claude-sonnet-4-6",
        "level": "max",
        "level_index": 3,
    },
    {
        "provider": "Google",
        "family": "gemini-3-flash",
        "model": "gemini-3-flash-thinking-minimal",
        "model_id": "google/gemini-3-flash-preview",
        "level": "minimal",
        "level_index": 0,
    },
    {
        "provider": "Google",
        "family": "gemini-3-flash",
        "model": "gemini-3-flash-thinking-low",
        "model_id": "google/gemini-3-flash-preview",
        "level": "low",
        "level_index": 1,
    },
    {
        "provider": "Google",
        "family": "gemini-3-flash",
        "model": "gemini-3-flash-thinking-medium",
        "model_id": "google/gemini-3-flash-preview",
        "level": "medium",
        "level_index": 2,
    },
    {
        "provider": "Google",
        "family": "gemini-3-flash",
        "model": "gemini-3-flash-thinking-high",
        "model_id": "google/gemini-3-flash-preview",
        "level": "high",
        "level_index": 3,
    },
]

GAME_CELLS = [
    {
        "game_cell_id": "game1_comp_0p0",
        "game_cell_label": "game1_cooperative",
        "game_label": "game1",
        "game_type": "item_allocation",
        "competition_level": 0.0,
        "m_items": 5,
    },
    {
        "game_cell_id": "game1_comp_0p5",
        "game_cell_label": "game1_mixed",
        "game_label": "game1",
        "game_type": "item_allocation",
        "competition_level": 0.5,
        "m_items": 5,
    },
    {
        "game_cell_id": "game1_comp_1p0",
        "game_cell_label": "game1_competitive",
        "game_label": "game1",
        "game_type": "item_allocation",
        "competition_level": 1.0,
        "m_items": 5,
    },
    {
        "game_cell_id": "game2_rho_1_theta_1",
        "game_cell_label": "game2_cooperative",
        "game_label": "game2",
        "game_type": "diplomacy",
        "rho": 1.0,
        "theta": 1.0,
        "n_issues": 5,
    },
    {
        "game_cell_id": "game2_rho_0_theta_1",
        "game_cell_label": "game2_mixed",
        "game_label": "game2",
        "game_type": "diplomacy",
        "rho": 0.0,
        "theta": 1.0,
        "n_issues": 5,
    },
    {
        "game_cell_id": "game2_rho_n1_theta_1",
        "game_cell_label": "game2_competitive",
        "game_label": "game2",
        "game_type": "diplomacy",
        "rho": -1.0,
        "theta": 1.0,
        "n_issues": 5,
    },
    {
        "game_cell_id": "game3_alpha_1p0_sigma_1p0",
        "game_cell_label": "game3_easy",
        "game_label": "game3",
        "game_type": "co_funding",
        "alpha": 1.0,
        "sigma": 1.0,
        "m_projects": 5,
        "c_min": 10.0,
        "c_max": 30.0,
        "cofunding_discussion_transparency": "own",
        "cofunding_enable_commit_vote": True,
        "cofunding_enable_time_discount": True,
        "cofunding_time_discount": 0.9,
    },
    {
        "game_cell_id": "game3_alpha_0p5_sigma_0p6",
        "game_cell_label": "game3_mixed",
        "game_label": "game3",
        "game_type": "co_funding",
        "alpha": 0.5,
        "sigma": 0.6,
        "m_projects": 5,
        "c_min": 10.0,
        "c_max": 30.0,
        "cofunding_discussion_transparency": "own",
        "cofunding_enable_commit_vote": True,
        "cofunding_enable_time_discount": True,
        "cofunding_time_discount": 0.9,
    },
    {
        "game_cell_id": "game3_alpha_0p0_sigma_0p2",
        "game_cell_label": "game3_hard",
        "game_label": "game3",
        "game_type": "co_funding",
        "alpha": 0.0,
        "sigma": 0.2,
        "m_projects": 5,
        "c_min": 10.0,
        "c_max": 30.0,
        "cofunding_discussion_transparency": "own",
        "cofunding_enable_commit_vote": True,
        "cofunding_enable_time_discount": True,
        "cofunding_time_discount": 0.9,
    },
]

ORDERS = [
    {"order": "target_first", "target_position": 0},
    {"order": "baseline_first", "target_position": 1},
]


def timestamp_now() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize(value: Any) -> str:
    token = str(value).replace(".", "p").replace("-", "_").replace("/", "_")
    return re.sub(r"[^A-Za-z0-9_]+", "_", token).strip("_")


def resolve_results_root(raw: str | None) -> Path:
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()
    return (PROJECT_ROOT / "experiments" / "results" / f"ttc_native_scaling_{timestamp_now()}").resolve()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_configs(results_root: Path) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    config_id = 1
    for model_condition in MODEL_CONDITIONS:
        for game_cell in GAME_CELLS:
            for order in ORDERS:
                target_first = order["target_position"] == 0
                models = (
                    [model_condition["model"], BASELINE_MODEL]
                    if target_first
                    else [BASELINE_MODEL, model_condition["model"]]
                )
                rel_output_dir = (
                    Path("experiments")
                    / "results"
                    / results_root.name
                    / model_condition["family"]
                    / f"level_{sanitize(model_condition['level'])}"
                    / game_cell["game_cell_id"]
                    / order["order"]
                    / "seed_42"
                )
                config: Dict[str, Any] = {
                    "config_id": config_id,
                    "experiment_name": "ttc_native_scaling",
                    "target_provider": model_condition["provider"],
                    "target_model_family": model_condition["family"],
                    "target_model": model_condition["model"],
                    "target_model_id": model_condition["model_id"],
                    "target_reasoning_level_requested": model_condition["level"],
                    "target_reasoning_level_index": model_condition["level_index"],
                    "baseline_model": BASELINE_MODEL,
                    "baseline_reasoning_level_requested": BASELINE_REASONING_LEVEL,
                    "models": models,
                    "order": order["order"],
                    "model_order": order["order"],
                    "target_position": order["target_position"],
                    "reasoning_agent_index": order["target_position"],
                    "seed_label": "seed_42",
                    "random_seed": SEED,
                    "run_number": 1,
                    "max_rounds": MAX_ROUNDS,
                    "discussion_turns": DISCUSSION_TURNS,
                    "gamma_discount": GAMMA_DISCOUNT,
                    "max_tokens_per_phase": MAX_TOKENS_PER_PHASE,
                    "output_dir": str(rel_output_dir),
                }
                config.update(game_cell)
                configs.append(config)
                config_id += 1
    return configs


def make_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)


def write_slurm_script(results_root: Path) -> Path:
    slurm_dir = results_root / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    script_path = slurm_dir / "run_one.sbatch"
    script_path.write_text(
        f"""#!/bin/bash
#SBATCH --job-name=ttc_native
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --partition=cpu
#SBATCH --output={results_root}/slurm/logs/%x_%j.out
#SBATCH --error={results_root}/slurm/logs/%x_%j.err

set -eo pipefail

BASE_DIR="{PROJECT_ROOT}"
cd "$BASE_DIR"
mkdir -p "{results_root}/slurm/logs" "{results_root}/monitoring"

echo "============================================================"
echo "TTC native scaling job"
echo "SLURM_JOB_ID=${{SLURM_JOB_ID:-none}}"
echo "Node=${{SLURM_NODELIST:-none}}"
echo "Started=$(date)"
echo "Config=${{1:-}}"
echo "============================================================"

module purge
module load anaconda3/2024.2
module load proxy/default

KEY_ENV_FILE="${{BARGAIN_API_KEYS_ENV:-/home/jz4391/.config/bargain/api_keys.env}}"
if [[ -f "$KEY_ENV_FILE" ]]; then
  set -a
  source "$KEY_ENV_FILE"
  set +a
fi

source "$BASE_DIR/.venv/bin/activate"

export OPENROUTER_TRANSPORT="${{OPENROUTER_TRANSPORT:-proxy}}"
export OPENROUTER_PROXY_POLL_DIR="${{OPENROUTER_PROXY_POLL_DIR:-/home/jz4391/openrouter_proxy}}"
export OPENROUTER_PROXY_CLIENT_TIMEOUT="${{OPENROUTER_PROXY_CLIENT_TIMEOUT:-9000}}"
export OPENROUTER_API_TIMEOUT="${{OPENROUTER_API_TIMEOUT:-1800}}"
export LLM_FAILURE_REPORT_PATH="${{LLM_FAILURE_REPORT_PATH:-{results_root}/monitoring/provider_failures.md}}"
export PYTHONUNBUFFERED=1

CONFIG_FILE="${{1:?config path is required}}"
"$BASE_DIR/.venv/bin/python" "$BASE_DIR/scripts/run_ttc_native_config.py" --config "$CONFIG_FILE"

echo "Finished=$(date)"
""",
        encoding="utf-8",
    )
    make_executable(script_path)
    return script_path


def write_submit_script(results_root: Path, configs: List[Dict[str, Any]], slurm_script: Path) -> Path:
    submit_path = results_root / "slurm" / "submit_all.sh"
    submit_path.write_text(
        f"""#!/bin/bash
set -eo pipefail

BASE_DIR="{PROJECT_ROOT}"
RUN_DIR="{results_root}"
CONFIG_DIR="$RUN_DIR/configs"
SBATCH_SCRIPT="{slurm_script}"
SUBMITTED="$RUN_DIR/slurm/submitted_jobs.tsv"

cd "$BASE_DIR"
mkdir -p "$RUN_DIR/slurm/logs" "$RUN_DIR/monitoring"

KEY_ENV_FILE="${{BARGAIN_API_KEYS_ENV:-/home/jz4391/.config/bargain/api_keys.env}}"
if [[ -f "$KEY_ENV_FILE" ]]; then
  set -a
  source "$KEY_ENV_FILE"
  set +a
fi

"$BASE_DIR/.venv/bin/python" - <<'PY'
from negotiation.provider_key_rotation import discover_provider_keys
required = ["openai", "anthropic", "openrouter"]
missing = []
for provider in required:
    labels = [key.label for key in discover_provider_keys(provider)]
    print(f"{{provider}} keys: {{', '.join(labels) if labels else 'MISSING'}}")
    if not labels:
        missing.append(provider)
if missing:
    raise SystemExit(f"Missing provider keys: {{', '.join(missing)}}")
PY

printf "config_id\\tjob_id\\tconfig_file\\n" > "$SUBMITTED"
count=0
for config_file in "$CONFIG_DIR"/config_*.json; do
  config_id="$(basename "$config_file" .json | sed 's/config_//')"
  job_name="ttc${{config_id}}"
  job_id="$(sbatch --parsable --job-name="$job_name" "$SBATCH_SCRIPT" "$config_file")"
  printf "%s\\t%s\\t%s\\n" "$config_id" "$job_id" "$config_file" >> "$SUBMITTED"
  count=$((count + 1))
  echo "Submitted config $config_id as job $job_id"
done

echo "Submitted $count jobs"
echo "Submission manifest: $SUBMITTED"
""",
        encoding="utf-8",
    )
    make_executable(submit_path)
    return submit_path


def write_manifest(results_root: Path, configs: List[Dict[str, Any]]) -> None:
    rows = [
        {
            "config_id": cfg["config_id"],
            "target_provider": cfg["target_provider"],
            "target_model": cfg["target_model"],
            "reasoning_level": cfg["target_reasoning_level_requested"],
            "game_cell_id": cfg["game_cell_id"],
            "order": cfg["order"],
            "output_dir": cfg["output_dir"],
        }
        for cfg in configs
    ]
    write_json(
        results_root / "manifest.json",
        {
            "experiment_name": "ttc_native_scaling",
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
            "num_configs": len(configs),
            "baseline_model": BASELINE_MODEL,
            "baseline_reasoning_level_requested": BASELINE_REASONING_LEVEL,
            "seed": SEED,
            "max_tokens_per_phase": MAX_TOKENS_PER_PHASE,
            "model_conditions": MODEL_CONDITIONS,
            "game_cells": GAME_CELLS,
            "orders": ORDERS,
            "configs": rows,
        },
    )


def generate(results_root: Path) -> List[Dict[str, Any]]:
    configs = build_configs(results_root)
    if len(configs) != 216:
        raise RuntimeError(f"Expected 216 configs, generated {len(configs)}")
    configs_dir = results_root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    for cfg in configs:
        write_json(configs_dir / f"config_{cfg['config_id']:04d}.json", cfg)
    slurm_script = write_slurm_script(results_root)
    write_submit_script(results_root, configs, slurm_script)
    write_manifest(results_root, configs)
    return configs


def validate_configs(configs: Iterable[Dict[str, Any]]) -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    from strong_models_experiment.configs import STRONG_MODELS_CONFIG

    missing: List[str] = []
    for cfg in configs:
        for model in cfg["models"]:
            if model not in STRONG_MODELS_CONFIG:
                missing.append(model)
    if missing:
        raise RuntimeError(f"Missing model configs: {sorted(set(missing))}")


def submit(results_root: Path) -> None:
    submit_script = results_root / "slurm" / "submit_all.sh"
    if not submit_script.exists():
        raise FileNotFoundError(submit_script)
    subprocess.run([str(submit_script)], cwd=PROJECT_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=None)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_root = resolve_results_root(args.results_root)
    configs = build_configs(results_root)
    validate_configs(configs)
    print(f"Results root: {results_root}")
    print(f"Configs: {len(configs)}")
    if args.dry_run:
        print(json.dumps(configs[:3], indent=2, sort_keys=True))
        return 0
    generated = generate(results_root)
    print(f"Wrote {len(generated)} configs")
    print(f"Submit script: {results_root / 'slurm' / 'submit_all.sh'}")
    if args.submit:
        submit(results_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

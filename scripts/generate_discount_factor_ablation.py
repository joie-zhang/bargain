#!/usr/bin/env python3
"""Generate Game 1 discount-factor rebuttal experiments."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_NAME = "discount_factor_ablation_game1_20260725"

BASELINE_MODEL = "gpt-5-nano"
ADVERSARY_MODELS = [
    "amazon-nova-micro-v1.0",
    "claude-3-haiku-20240307",
    "amazon-nova-pro-v1.0",
    "gpt-4o-mini-2024-07-18",
    "deepseek-v3",
    "claude-sonnet-4-20250514",
    "deepseek-r1-0528",
    "gemini-2.5-pro",
    "gpt-5.4-high",
    "claude-opus-4-6-thinking",
]
GAMMA_DISCOUNTS = [0.9, 1.0]
COMPETITION_LEVELS = [0.0, 0.5, 1.0]
MODEL_ORDERS = ["weak_first", "strong_first"]
SEEDS = [42, 123]
DISCUSSION_TURNS = 2
NUM_ITEMS = 5
MAX_ROUNDS = 10

def expected_config_count(gammas: list[float]) -> int:
    return (
        len(ADVERSARY_MODELS)
        * len(gammas)
        * len(COMPETITION_LEVELS)
        * len(MODEL_ORDERS)
        * len(SEEDS)
    )


def slug_number(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def result_file(config: dict[str, Any]) -> Path:
    output_dir = PROJECT_ROOT / config["output_dir"]
    return output_dir / f"run_{config['run_number']}_experiment_results.json"


def result_is_complete(config: dict[str, Any]) -> bool:
    path = result_file(config)
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


def build_configs(
    run_name: str,
    gammas: list[float] | None = None,
) -> list[dict[str, Any]]:
    selected_gammas = GAMMA_DISCOUNTS if gammas is None else gammas
    configs: list[dict[str, Any]] = []
    experiment_id = 0
    # Model-first ordering makes it easy to select a representative smoke task
    # for every provider/model while preserving a deterministic manifest.
    for adversary, gamma, competition, order, seed in product(
        ADVERSARY_MODELS,
        selected_gammas,
        COMPETITION_LEVELS,
        MODEL_ORDERS,
        SEEDS,
    ):
        models = (
            [BASELINE_MODEL, adversary]
            if order == "weak_first"
            else [adversary, BASELINE_MODEL]
        )
        run_number = SEEDS.index(seed) + 1
        output_dir = (
            f"experiments/results/{run_name}/runs/"
            f"config_{experiment_id:03d}_{adversary}/"
            f"gamma_{slug_number(gamma)}/comp_{slug_number(competition)}/"
            f"{order}/seed_{seed}"
        )
        configs.append(
            {
                "experiment_id": experiment_id,
                "config_id": experiment_id,
                "experiment_name": run_name,
                "ablation": "game1_discount_factor",
                "game_label": "game1",
                "game_type": "item_allocation",
                "baseline_model": BASELINE_MODEL,
                "weak_model": BASELINE_MODEL,
                "adversary_model": adversary,
                "strong_model": adversary,
                "models": models,
                "model_order": order,
                "conceptual_order": (
                    "baseline_first" if order == "weak_first" else "adversary_first"
                ),
                "competition_level": competition,
                "gamma_discount": gamma,
                "seed_replicate": SEEDS.index(seed) + 1,
                "run_number": run_number,
                "num_items": NUM_ITEMS,
                "max_rounds": MAX_ROUNDS,
                "random_seed": seed,
                "discussion_turns": DISCUSSION_TURNS,
                "output_dir": output_dir,
            }
        )
        experiment_id += 1
    expected = expected_config_count(selected_gammas)
    if len(configs) != expected:
        raise AssertionError(f"Expected {expected} configs, generated {len(configs)}")
    return configs


def write_sbatch(run_root: Path) -> None:
    sbatch_path = run_root / "slurm" / "run_array.sbatch"
    text = f"""#!/bin/bash
#SBATCH --job-name=g1-gamma
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output={run_root}/logs/%A_%a.out
#SBATCH --error={run_root}/logs/%A_%a.err

set -eo pipefail

BASE_DIR="{PROJECT_ROOT}"
RUN_ROOT="{run_root}"
cd "$BASE_DIR"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/monitoring"

module purge || true
module load anaconda3/2024.2 || true
module load proxy/default || true

KEY_ENV_FILE="${{BARGAIN_API_KEYS_ENV:-/home/jz4391/.config/bargain/api_keys.env}}"
if [[ -f "$KEY_ENV_FILE" ]]; then
  set -a
  source "$KEY_ENV_FILE"
  set +a
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="$BASE_DIR${{PYTHONPATH:+:$PYTHONPATH}}"
export OPENROUTER_TRANSPORT="${{OPENROUTER_TRANSPORT:-proxy}}"
export OPENROUTER_PROXY_POLL_DIR="${{OPENROUTER_PROXY_POLL_DIR:-/home/jz4391/openrouter_proxy}}"
export OPENROUTER_PROXY_CLIENT_TIMEOUT="${{OPENROUTER_PROXY_CLIENT_TIMEOUT:-9000}}"
export OPENROUTER_PROVIDER_FALLBACK="${{OPENROUTER_PROVIDER_FALLBACK:-true}}"
export LLM_FAILURE_REPORT_PATH="${{LLM_FAILURE_REPORT_PATH:-$RUN_ROOT/monitoring/provider_failures.md}}"

CONFIG_FILE=$(printf "$RUN_ROOT/configs/config_%03d.json" "$SLURM_ARRAY_TASK_ID")
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Missing config: $CONFIG_FILE" >&2
  exit 2
fi

"$BASE_DIR/.venv/bin/python" "$BASE_DIR/scripts/run_discount_factor_ablation_config.py" "$CONFIG_FILE"
"""
    sbatch_path.write_text(text, encoding="utf-8")
    sbatch_path.chmod(0o755)


def write_artifacts(
    run_name: str,
    configs: list[dict[str, Any]],
    gammas: list[float] | None = None,
) -> Path:
    selected_gammas = GAMMA_DISCOUNTS if gammas is None else gammas
    expected = expected_config_count(selected_gammas)
    run_root = PROJECT_ROOT / "experiments" / "results" / run_name
    config_dir = run_root / "configs"
    for path in (config_dir, run_root / "logs", run_root / "slurm", run_root / "monitoring"):
        path.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "experiment_id",
        "config_file",
        "baseline_model",
        "adversary_model",
        "gamma_discount",
        "competition_level",
        "model_order",
        "conceptual_order",
        "seed",
        "seed_replicate",
        "run_number",
        "discussion_turns",
        "output_dir",
        "result_file",
    ]
    manifest_rows: list[dict[str, Any]] = []
    smoke_ids: list[int] = []
    seen_models: set[str] = set()
    for config in configs:
        config_file = config_dir / f"config_{config['experiment_id']:03d}.json"
        config_file.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        adversary = str(config["adversary_model"])
        if adversary not in seen_models:
            smoke_ids.append(int(config["experiment_id"]))
            seen_models.add(adversary)
        manifest_rows.append(
            {
                "experiment_id": config["experiment_id"],
                "config_file": config_file.name,
                "baseline_model": config["baseline_model"],
                "adversary_model": adversary,
                "gamma_discount": config["gamma_discount"],
                "competition_level": config["competition_level"],
                "model_order": config["model_order"],
                "conceptual_order": config["conceptual_order"],
                "seed": config["random_seed"],
                "seed_replicate": config["seed_replicate"],
                "run_number": config["run_number"],
                "discussion_turns": config["discussion_turns"],
                "output_dir": config["output_dir"],
                "result_file": str(result_file(config).relative_to(PROJECT_ROOT)),
            }
        )

    with (run_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    (run_root / "smoke_ids.txt").write_text(
        ",".join(str(value) for value in smoke_ids) + "\n",
        encoding="utf-8",
    )
    (run_root / "README.md").write_text(
        "\n".join(
            [
                "# Game 1 discount-factor ablation",
                "",
                f"- Expected runs: {expected}",
                f"- Baseline: `{BASELINE_MODEL}`",
                f"- Adversaries: {len(ADVERSARY_MODELS)}",
                f"- Gamma: {selected_gammas}",
                f"- Competition: {COMPETITION_LEVELS}",
                f"- Orders: {MODEL_ORDERS}",
                f"- Seeds: {SEEDS}",
                f"- Discussion turns: {DISCUSSION_TURNS}",
                "",
                "The manifest is the source of truth. A result is complete only when its",
                "JSON exists and matches gamma, competition level, and random seed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    write_sbatch(run_root)
    return run_root


def audit(run_name: str, gammas: list[float] | None = None) -> int:
    selected_gammas = GAMMA_DISCOUNTS if gammas is None else gammas
    expected = expected_config_count(selected_gammas)
    configs = build_configs(run_name, selected_gammas)
    complete = [config for config in configs if result_is_complete(config)]
    missing = [config for config in configs if not result_is_complete(config)]
    print(f"complete={len(complete)} expected={expected} missing={len(missing)}")
    if missing:
        ids = [str(config["experiment_id"]) for config in missing]
        print("missing_ids=" + ",".join(ids))
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument(
        "--gammas",
        nargs="+",
        type=float,
        default=GAMMA_DISCOUNTS,
        help="One or more discount factors to cross with the fixed design.",
    )
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    gammas = list(dict.fromkeys(args.gammas))
    if not gammas or any(gamma <= 0 or gamma > 1 for gamma in gammas):
        parser.error("--gammas values must lie in (0, 1]")

    if args.audit:
        return audit(args.run_name, gammas)

    configs = build_configs(args.run_name, gammas)
    run_root = write_artifacts(args.run_name, configs, gammas)
    print(f"run_root={run_root}")
    print(f"configs={len(configs)}")
    print(f"smoke_ids={(run_root / 'smoke_ids.txt').read_text().strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate GPT-5.2 reasoning-effort sweep configs and Slurm helpers."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments" / "results"
DEFAULT_BASELINE_MODEL = "gpt-5-nano"
DEFAULT_EFFORTS = ("low", "medium", "high", "xhigh")
DEFAULT_MODEL_ORDERS = ("weak_first", "strong_first")
MODEL_ID = "gpt-5.2-2025-12-11"
GAME_TYPES = ("item_allocation", "diplomacy", "co_funding")
MODEL_ORDERS = ("weak_first", "strong_first")


def parse_csv_list(raw: str) -> List[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def effort_alias(effort: str) -> str:
    return f"gpt-5.2-{effort}"


def validate_requested_models(efforts: Iterable[str], baseline_model: str) -> None:
    from strong_models_experiment.configs import STRONG_MODELS_CONFIG

    missing = [baseline_model] if baseline_model not in STRONG_MODELS_CONFIG else []
    for effort in efforts:
        alias = effort_alias(effort)
        cfg = STRONG_MODELS_CONFIG.get(alias)
        if cfg is None:
            missing.append(alias)
            continue
        if cfg.get("model_id") != MODEL_ID:
            raise ValueError(f"{alias} must use model_id {MODEL_ID}, found {cfg.get('model_id')!r}")
        if cfg.get("api_type") != "openai":
            raise ValueError(f"{alias} must use api_type=openai, found {cfg.get('api_type')!r}")
        if cfg.get("reasoning_effort") != effort:
            raise ValueError(
                f"{alias} must set reasoning_effort={effort!r}, "
                f"found {cfg.get('reasoning_effort')!r}"
            )
    if missing:
        raise ValueError("Missing model aliases in STRONG_MODELS_CONFIG: " + ", ".join(missing))


def ordered_models(model_order: str, baseline_model: str, reasoning_model: str) -> List[str]:
    if model_order == "weak_first":
        return [baseline_model, reasoning_model]
    if model_order == "strong_first":
        return [reasoning_model, baseline_model]
    raise ValueError(f"Unsupported model_order: {model_order}")


def build_config(
    *,
    experiment_id: int,
    effort: str,
    model_order: str,
    run_number: int,
    seed: int,
    run_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    reasoning_model = effort_alias(effort)
    output_dir = (
        run_dir
        / "results"
        / args.game_type
        / f"{reasoning_model}_vs_{args.baseline_model}"
        / model_order
        / f"seed_{seed}"
    )
    if args.runs_per_condition > 1:
        output_dir = output_dir / f"run_{run_number}"
    try:
        output_dir_value = str(output_dir.relative_to(REPO_ROOT))
    except ValueError:
        output_dir_value = str(output_dir)

    max_tokens_per_phase = args.max_tokens_per_phase
    if max_tokens_per_phase is not None and max_tokens_per_phase <= 0:
        max_tokens_per_phase = None

    return {
        "experiment_id": experiment_id,
        "condition_id": f"{reasoning_model}_{model_order}_seed_{seed}",
        "treatment": "openai_reasoning_effort",
        "reasoning_model": reasoning_model,
        "baseline_model": args.baseline_model,
        "models": ordered_models(model_order, args.baseline_model, reasoning_model),
        "model_order": model_order,
        "reasoning_effort": effort,
        "model_id": MODEL_ID,
        "run_number": run_number,
        "runs_per_condition": args.runs_per_condition,
        "random_seed": seed,
        "game_type": args.game_type,
        "num_runs": 1,
        "max_rounds": args.max_rounds,
        "num_items": args.num_items,
        "competition_level": args.competition_level,
        "gamma_discount": args.gamma_discount,
        "discussion_turns": args.discussion_turns,
        "disable_discussion": False,
        "disable_thinking": False,
        "disable_reflection": False,
        "n_issues": args.n_issues,
        "rho": args.rho,
        "theta": args.theta,
        "m_projects": args.m_projects,
        "alpha": args.alpha,
        "sigma": args.sigma,
        "c_min": args.c_min,
        "c_max": args.c_max,
        "cofunding_discussion_transparency": args.cofunding_discussion_transparency,
        "cofunding_enable_commit_vote": not args.cofunding_disable_commit_vote,
        "cofunding_enable_time_discount": not args.cofunding_disable_time_discount,
        "cofunding_time_discount": args.cofunding_time_discount,
        "max_tokens_per_phase": max_tokens_per_phase,
        "api_reasoning_control": {
            "type": "model_alias_reasoning_effort",
            "source": "STRONG_MODELS_CONFIG",
            "prompt_reasoning_budget": None,
        },
        "output_dir": output_dir_value,
    }


def write_configs(config_dir: Path, configs: List[Dict[str, Any]], padding_width: int) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    for cfg in configs:
        config_path = config_dir / f"config_{cfg['experiment_id']:0{padding_width}d}.json"
        config_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    all_configs = "\n".join(
        str(config_dir / f"config_{cfg['experiment_id']:0{padding_width}d}.json")
        for cfg in configs
    )
    (config_dir / "all_configs.txt").write_text(all_configs + "\n", encoding="utf-8")


def write_index(config_dir: Path, configs: List[Dict[str, Any]], padding_width: int) -> None:
    fieldnames = [
        "experiment_id",
        "reasoning_model",
        "baseline_model",
        "reasoning_effort",
        "model_order",
        "game_type",
        "run_number",
        "seed",
        "config_file",
        "output_dir",
    ]
    with (config_dir / "experiment_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cfg in configs:
            writer.writerow(
                {
                    "experiment_id": cfg["experiment_id"],
                    "reasoning_model": cfg["reasoning_model"],
                    "baseline_model": cfg["baseline_model"],
                    "reasoning_effort": cfg["reasoning_effort"],
                    "model_order": cfg["model_order"],
                    "game_type": cfg["game_type"],
                    "run_number": cfg["run_number"],
                    "seed": cfg["random_seed"],
                    "config_file": f"config_{cfg['experiment_id']:0{padding_width}d}.json",
                    "output_dir": cfg["output_dir"],
                }
            )


def write_summary(
    config_dir: Path,
    run_dir: Path,
    configs: List[Dict[str, Any]],
    efforts: List[str],
    orders: List[str],
    args: argparse.Namespace,
) -> None:
    summary = f"""GPT-5.2 Reasoning-Effort Sweep
================================
Created at: {datetime.now().isoformat(timespec="seconds")}
Run directory: {run_dir}

Treatment:
  Varied: OpenAI provider-native reasoning_effort
  Values: {", ".join(efforts)}
  Model snapshot: {MODEL_ID}
  Prompt reasoning budget: none

Matrix:
  Reasoning models: {", ".join(effort_alias(effort) for effort in efforts)}
  Baseline model: {args.baseline_model}
  Model orders: {", ".join(orders)}
  Runs per effort/order: {args.runs_per_condition}
  Total configs: {len(configs)}

Game:
  Type: {args.game_type}
  Max rounds: {args.max_rounds}
  Discussion turns: {args.discussion_turns}
  Item allocation: num_items={args.num_items}, competition_level={args.competition_level}
  Diplomacy: n_issues={args.n_issues}, rho={args.rho}, theta={args.theta}
  Co-funding: m_projects={args.m_projects}, alpha={args.alpha}, sigma={args.sigma}
  Max tokens per phase: {args.max_tokens_per_phase}

Outputs:
  Configs: {config_dir}
  Index: {config_dir / "experiment_index.csv"}
  Slurm: {run_dir / "slurm" / "run_api.sbatch"}
  Logs: {REPO_ROOT / "logs" / "cluster"}

Run policy:
  The generator does not submit Slurm jobs.
  The generated run scripts do not pass --reasoning-token-budget.
  The first live API smoke should be run manually after approval, using config_0000.
"""
    (config_dir / "summary.txt").write_text(summary, encoding="utf-8")


def run_config_py() -> str:
    return f'''#!/usr/bin/env python3
"""Run one generated GPT-5.2 effort config."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path({str(REPO_ROOT)!r})


def append_if_present(cmd, cfg, option, key):
    value = cfg.get(key)
    if value is not None:
        cmd.extend([option, str(value)])


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: run_config.py CONFIG_FILE", file=sys.stderr)
        return 2

    config_path = Path(sys.argv[1]).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    output_dir = REPO_ROOT / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "run_strong_models_experiment.py"),
        "--game-type",
        cfg["game_type"],
        "--models",
        *cfg["models"],
        "--batch",
        "--num-runs",
        "1",
        "--run-number",
        str(cfg["run_number"]),
        "--max-rounds",
        str(cfg["max_rounds"]),
        "--num-items",
        str(cfg["num_items"]),
        "--competition-level",
        str(cfg["competition_level"]),
        "--gamma-discount",
        str(cfg["gamma_discount"]),
        "--discussion-turns",
        str(cfg["discussion_turns"]),
        "--model-order",
        cfg["model_order"],
        "--random-seed",
        str(cfg["random_seed"]),
        "--n-issues",
        str(cfg["n_issues"]),
        "--rho",
        str(cfg["rho"]),
        "--theta",
        str(cfg["theta"]),
        "--m-projects",
        str(cfg["m_projects"]),
        "--alpha",
        str(cfg["alpha"]),
        "--sigma",
        str(cfg["sigma"]),
        "--c-min",
        str(cfg["c_min"]),
        "--c-max",
        str(cfg["c_max"]),
        "--cofunding-discussion-transparency",
        cfg["cofunding_discussion_transparency"],
        "--output-dir",
        cfg["output_dir"],
        "--job-id",
        str(cfg["experiment_id"]),
    ]

    append_if_present(cmd, cfg, "--max-tokens-per-phase", "max_tokens_per_phase")

    if cfg.get("disable_discussion"):
        cmd.append("--disable-discussion")
    if cfg.get("disable_thinking"):
        cmd.append("--disable-thinking")
    if cfg.get("disable_reflection"):
        cmd.append("--disable-reflection")
    if not cfg.get("cofunding_enable_commit_vote", True):
        cmd.append("--cofunding-disable-commit-vote")
    if not cfg.get("cofunding_enable_time_discount", True):
        cmd.append("--cofunding-disable-time-discount")

    print("Config:", config_path, flush=True)
    print("Reasoning model:", cfg["reasoning_model"], flush=True)
    print("Reasoning effort:", cfg["reasoning_effort"], flush=True)
    print("Baseline model:", cfg["baseline_model"], flush=True)
    print("Model order:", cfg["model_order"], flush=True)
    print("Command:", " ".join(shlex.quote(part) for part in cmd), flush=True)

    completed = subprocess.run(cmd, cwd=REPO_ROOT)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
'''


def write_slurm_files(run_dir: Path, total_configs: int, padding_width: int, args: argparse.Namespace) -> None:
    slurm_dir = run_dir / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    log_dir = REPO_ROOT / "logs" / "cluster"
    partition_line = f"#SBATCH --partition={args.slurm_partition}\n" if args.slurm_partition else ""

    (slurm_dir / "run_config.py").write_text(run_config_py(), encoding="utf-8")
    os.chmod(slurm_dir / "run_config.py", 0o755)

    sbatch = f"""#!/bin/bash
#SBATCH --job-name=gpt52-effort
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={args.cpus_per_task}
#SBATCH --mem={args.mem}
#SBATCH --time={args.slurm_time}
{partition_line}#SBATCH --output={log_dir}/gpt52_effort_%A_%a.out
#SBATCH --error={log_dir}/gpt52_effort_%A_%a.err

set -eo pipefail

BASE_DIR="{REPO_ROOT}"
RUN_DIR="{run_dir}"
cd "$BASE_DIR"
mkdir -p "{log_dir}" "$RUN_DIR/monitoring"

module purge
module load anaconda3/2024.2
module load proxy/default

KEY_ENV_FILE="${{BARGAIN_API_KEYS_ENV:-/home/jz4391/.config/bargain/api_keys.env}}"
if [[ -f "$KEY_ENV_FILE" ]]; then
  set -a
  source "$KEY_ENV_FILE"
  set +a
fi

export PYTHONUNBUFFERED=1
export OPENROUTER_TRANSPORT="${{OPENROUTER_TRANSPORT:-proxy}}"
export OPENROUTER_PROXY_POLL_DIR="${{OPENROUTER_PROXY_POLL_DIR:-/home/jz4391/openrouter_proxy}}"
export OPENROUTER_PROXY_CLIENT_TIMEOUT="${{OPENROUTER_PROXY_CLIENT_TIMEOUT:-9000}}"
export LLM_FAILURE_REPORT_PATH="${{LLM_FAILURE_REPORT_PATH:-$RUN_DIR/monitoring/provider_failures.md}}"

PYTHON_BIN="$BASE_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

CONFIG_ID="${{SLURM_ARRAY_TASK_ID:-0}}"
CONFIG_ID_PADDED=$(printf "%0{padding_width}d" "$CONFIG_ID")
CONFIG_FILE="$RUN_DIR/configs/config_${{CONFIG_ID_PADDED}}.json"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "ERROR: Config file not found: $CONFIG_FILE"
  exit 1
fi

echo "GPT-5.2 effort Slurm task"
echo "Job: ${{SLURM_JOB_ID:-local}}, array task: $CONFIG_ID"
echo "Node: ${{SLURM_NODELIST:-unknown}}"
echo "Started: $(date)"

"$PYTHON_BIN" "$RUN_DIR/slurm/run_config.py" "$CONFIG_FILE"
"""
    (slurm_dir / "run_api.sbatch").write_text(sbatch, encoding="utf-8")
    os.chmod(slurm_dir / "run_api.sbatch", 0o755)

    submit = f"""#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
RUN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TOTAL_CONFIGS={total_configs}
MAX_CONCURRENT="{args.max_concurrent}"
TEST_MODE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --test)
      TEST_MODE=true
      shift
      ;;
    --max-concurrent)
      MAX_CONCURRENT="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

mkdir -p "{log_dir}"

if [[ "$TEST_MODE" == "true" ]]; then
  ARRAY_SPEC="0"
else
  ARRAY_SPEC="0-$((TOTAL_CONFIGS - 1))"
fi

if [[ -n "$MAX_CONCURRENT" && "$MAX_CONCURRENT" != "0" ]]; then
  ARRAY_SPEC="${{ARRAY_SPEC}}%${{MAX_CONCURRENT}}"
fi

echo "Run dir: $RUN_DIR"
echo "Total configs: $TOTAL_CONFIGS"
echo "Submitting array: $ARRAY_SPEC"
sbatch --array="$ARRAY_SPEC" "$SCRIPT_DIR/run_api.sbatch"
"""
    (slurm_dir / "submit_all.sh").write_text(submit, encoding="utf-8")
    os.chmod(slurm_dir / "submit_all.sh", 0o755)

    local = f"""#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
RUN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BASE_DIR="{REPO_ROOT}"
cd "$BASE_DIR"

KEY_ENV_FILE="${{BARGAIN_API_KEYS_ENV:-/home/jz4391/.config/bargain/api_keys.env}}"
if [[ -f "$KEY_ENV_FILE" ]]; then
  set -a
  source "$KEY_ENV_FILE"
  set +a
fi

PYTHON_BIN="$BASE_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

CONFIG_ID="${{1:-0}}"
CONFIG_ID_PADDED=$(printf "%0{padding_width}d" "$CONFIG_ID")
CONFIG_FILE="$RUN_DIR/configs/config_${{CONFIG_ID_PADDED}}.json"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "ERROR: Config file not found: $CONFIG_FILE"
  exit 1
fi

echo "This runs a live API experiment for config $CONFIG_ID."
"$PYTHON_BIN" "$RUN_DIR/slurm/run_config.py" "$CONFIG_FILE"
"""
    (slurm_dir / "run_local.sh").write_text(local, encoding="utf-8")
    os.chmod(slurm_dir / "run_local.sh", 0o755)


def update_latest_symlink(run_dir: Path, enabled: bool) -> None:
    if not enabled:
        return
    latest = run_dir.parent / "gpt52_effort_derisk"
    if latest.is_symlink() or latest.is_file():
        latest.unlink()
    elif latest.exists():
        print(f"Skipping latest symlink because non-symlink path exists: {latest}")
        return
    latest.symlink_to(run_dir.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate GPT-5.2 reasoning_effort sweep configs without submitting jobs."
    )
    parser.add_argument("--run-name", default=None, help="Run directory name under --output-root")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output root for generated run")
    parser.add_argument("--baseline-model", default=DEFAULT_BASELINE_MODEL)
    parser.add_argument("--efforts", default=",".join(DEFAULT_EFFORTS), help="Comma-separated efforts")
    parser.add_argument("--model-orders", default=",".join(DEFAULT_MODEL_ORDERS), help="Comma-separated model orders")
    parser.add_argument("--weak-first-only", action="store_true", help="Generate only weak_first order")
    parser.add_argument("--strong-first-only", action="store_true", help="Generate only strong_first order")
    parser.add_argument("--runs-per-condition", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=5200)
    parser.add_argument("--game-type", choices=GAME_TYPES, default="item_allocation")
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--num-items", type=int, default=5)
    parser.add_argument("--competition-level", type=float, default=1.0)
    parser.add_argument("--gamma-discount", type=float, default=0.9)
    parser.add_argument("--discussion-turns", type=int, default=2)
    parser.add_argument("--n-issues", type=int, default=5)
    parser.add_argument("--rho", type=float, default=0.0)
    parser.add_argument("--theta", type=float, default=0.5)
    parser.add_argument("--m-projects", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--c-min", type=float, default=10.0)
    parser.add_argument("--c-max", type=float, default=50.0)
    parser.add_argument("--cofunding-discussion-transparency", choices=("aggregate", "own", "full"), default="own")
    parser.add_argument("--cofunding-disable-commit-vote", action="store_true")
    parser.add_argument("--cofunding-disable-time-discount", action="store_true")
    parser.add_argument("--cofunding-time-discount", type=float, default=0.9)
    parser.add_argument("--max-tokens-per-phase", type=int, default=12000)
    parser.add_argument("--slurm-time", default="02:00:00")
    parser.add_argument("--slurm-partition", default="cpu")
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--mem", default="8G")
    parser.add_argument("--max-concurrent", default="4")
    parser.add_argument("--no-latest-symlink", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.runs_per_condition < 1:
        raise ValueError("--runs-per-condition must be >= 1")
    if args.weak_first_only and args.strong_first_only:
        raise ValueError("--weak-first-only and --strong-first-only cannot both be set")

    efforts = parse_csv_list(args.efforts)
    unknown_efforts = [effort for effort in efforts if effort not in DEFAULT_EFFORTS]
    if unknown_efforts:
        raise ValueError("Unsupported efforts: " + ", ".join(unknown_efforts))

    if args.weak_first_only:
        orders = ["weak_first"]
    elif args.strong_first_only:
        orders = ["strong_first"]
    else:
        orders = parse_csv_list(args.model_orders)
    unknown_orders = [order for order in orders if order not in MODEL_ORDERS]
    if unknown_orders:
        raise ValueError("Unsupported model orders: " + ", ".join(unknown_orders))

    validate_requested_models(efforts, args.baseline_model)

    output_root = resolve_path(args.output_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"gpt52_effort_derisk_{timestamp}"
    run_dir = output_root / run_name
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")

    configs: List[Dict[str, Any]] = []
    experiment_id = 0
    for effort in efforts:
        for run_number in range(1, args.runs_per_condition + 1):
            seed = args.base_seed + run_number - 1
            for model_order in orders:
                configs.append(
                    build_config(
                        experiment_id=experiment_id,
                        effort=effort,
                        model_order=model_order,
                        run_number=run_number,
                        seed=seed,
                        run_dir=run_dir,
                        args=args,
                    )
                )
                experiment_id += 1

    padding_width = max(4, len(str(max(len(configs) - 1, 0))))
    config_dir = run_dir / "configs"
    write_configs(config_dir, configs, padding_width)
    write_index(config_dir, configs, padding_width)
    write_summary(config_dir, run_dir, configs, efforts, orders, args)
    write_slurm_files(run_dir, len(configs), padding_width, args)
    update_latest_symlink(run_dir, enabled=not args.no_latest_symlink)

    print(run_dir)
    print(f"Generated {len(configs)} configs")
    print(f"Configs: {config_dir}")
    print(f"Index: {config_dir / 'experiment_index.csv'}")
    print(f"Summary: {config_dir / 'summary.txt'}")
    print(f"Slurm: {run_dir / 'slurm' / 'run_api.sbatch'}")
    print(f"Local API smoke script: {run_dir / 'slurm' / 'run_local.sh'} 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

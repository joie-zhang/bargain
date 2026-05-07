#!/usr/bin/env python3
"""Generate the Llama 3.3 70B appendix baseline configs and CPU Slurm jobs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import stat
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "experiments" / "results"

BASELINE_MODEL = "llama-3.3-70b-instruct"
ADVERSARIES = [
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
MODEL_ORDERS = ["weak_first", "strong_first"]
GAME1_COMPETITION_LEVELS = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]
GAME2_RHOS = [-1.0, 0.0, 1.0]
GAME2_THETAS = [0.0, 0.5, 1.0]
GAME3_ALPHAS = [0.0, 0.5, 1.0]
GAME3_SIGMAS = [0.2, 0.6, 1.0]


@dataclass(frozen=True)
class GameSettings:
    game_label: str
    run_name: str
    total_configs: int
    slurm_job_prefix: str


def _float_slug(value: float) -> str:
    raw = f"{value:.2f}"
    if raw.endswith("00"):
        raw = f"{value:.1f}"
    elif raw.endswith("0"):
        raw = raw[:-1]
    return raw.replace("-", "n").replace(".", "_")


def _models_for_order(adversary: str, model_order: str) -> List[str]:
    if model_order == "weak_first":
        return [BASELINE_MODEL, adversary]
    if model_order == "strong_first":
        return [adversary, BASELINE_MODEL]
    raise ValueError(f"Unsupported model_order: {model_order}")


def _conceptual_order(model_order: str) -> str:
    return "baseline_first" if model_order == "weak_first" else "adversary_first"


def _game1_configs(run_name: str) -> Iterable[Dict[str, Any]]:
    experiment_id = 0
    for adversary in ADVERSARIES:
        for competition_level in GAME1_COMPETITION_LEVELS:
            for model_order in MODEL_ORDERS:
                run_number = 1 if model_order == "weak_first" else 2
                output_dir = (
                    f"experiments/results/{run_name}/"
                    f"{BASELINE_MODEL}_vs_{adversary}/{model_order}/"
                    f"comp_{_float_slug(competition_level)}/turns_2/run_{run_number}"
                )
                yield {
                    "experiment_id": experiment_id,
                    "experiment_type": "appendix_llama33_baseline",
                    "game_label": "game1",
                    "game_type": "item_allocation",
                    "baseline_model": BASELINE_MODEL,
                    "adversary_model": adversary,
                    "model1": BASELINE_MODEL,
                    "model2": adversary,
                    "weak_model": BASELINE_MODEL,
                    "strong_model": adversary,
                    "models": _models_for_order(adversary, model_order),
                    "model_order": model_order,
                    "conceptual_order": _conceptual_order(model_order),
                    "run_number": run_number,
                    "num_runs": 1,
                    "max_tokens_per_phase": 16384,
                    "num_items": 5,
                    "m_items": 5,
                    "max_rounds": 10,
                    "gamma_discount": 0.9,
                    "competition_level": competition_level,
                    "discussion_turns": 2,
                    "random_seed": 42,
                    "output_dir": output_dir,
                }
                experiment_id += 1


def _game2_configs(run_name: str) -> Iterable[Dict[str, Any]]:
    experiment_id = 0
    for adversary in ADVERSARIES:
        for rho in GAME2_RHOS:
            for theta in GAME2_THETAS:
                for model_order in MODEL_ORDERS:
                    output_dir = (
                        f"experiments/results/{run_name}/model_scale/"
                        f"{BASELINE_MODEL}_vs_{adversary}/{model_order}/"
                        f"rho_{_float_slug(rho)}_theta_{_float_slug(theta)}"
                    )
                    yield {
                        "experiment_id": experiment_id,
                        "experiment_type": "appendix_llama33_baseline",
                        "game_label": "game2",
                        "game_type": "diplomacy",
                        "baseline_model": BASELINE_MODEL,
                        "adversary_model": adversary,
                        "model1": BASELINE_MODEL,
                        "model2": adversary,
                        "models": _models_for_order(adversary, model_order),
                        "model_order": model_order,
                        "conceptual_order": _conceptual_order(model_order),
                        "run_number": 1,
                        "num_runs": 1,
                        "max_tokens_per_phase": 16384,
                        "n_issues": 5,
                        "rho": rho,
                        "theta": theta,
                        "competition_index": theta * (1.0 - rho) / 2.0,
                        "max_rounds": 10,
                        "gamma_discount": 0.9,
                        "discussion_turns": 2,
                        "random_seed": 42 + experiment_id,
                        "output_dir": output_dir,
                    }
                    experiment_id += 1


def _game3_configs(run_name: str) -> Iterable[Dict[str, Any]]:
    experiment_id = 0
    for adversary in ADVERSARIES:
        for alpha in GAME3_ALPHAS:
            for sigma in GAME3_SIGMAS:
                for model_order in MODEL_ORDERS:
                    output_dir = (
                        f"experiments/results/{run_name}/model_scale/"
                        f"{BASELINE_MODEL}_vs_{adversary}/{model_order}/"
                        f"alpha_{_float_slug(alpha)}_sigma_{_float_slug(sigma)}"
                    )
                    yield {
                        "experiment_id": experiment_id,
                        "experiment_type": "appendix_llama33_baseline",
                        "game_label": "game3",
                        "game_type": "co_funding",
                        "baseline_model": BASELINE_MODEL,
                        "adversary_model": adversary,
                        "model1": BASELINE_MODEL,
                        "model2": adversary,
                        "models": _models_for_order(adversary, model_order),
                        "model_order": model_order,
                        "conceptual_order": _conceptual_order(model_order),
                        "run_number": 1,
                        "num_runs": 1,
                        "max_tokens_per_phase": 16384,
                        "m_projects": 5,
                        "alpha": alpha,
                        "sigma": sigma,
                        "competition_index": (1.0 - alpha) * (1.0 - sigma),
                        "c_min": 10.0,
                        "c_max": 30.0,
                        "cofunding_discussion_transparency": "own",
                        "cofunding_enable_commit_vote": True,
                        "cofunding_enable_time_discount": True,
                        "cofunding_time_discount": 0.9,
                        "max_rounds": 10,
                        "gamma_discount": 0.9,
                        "discussion_turns": 2,
                        "random_seed": 42 + experiment_id,
                        "output_dir": output_dir,
                    }
                    experiment_id += 1


def _validate_models() -> None:
    sys.path.insert(0, str(REPO_ROOT))
    from strong_models_experiment.configs import STRONG_MODELS_CONFIG

    missing = [
        model
        for model in [BASELINE_MODEL, *ADVERSARIES]
        if model not in STRONG_MODELS_CONFIG
    ]
    if missing:
        raise ValueError("Missing models in STRONG_MODELS_CONFIG: " + ", ".join(missing))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_index(config_dir: Path, configs: List[Dict[str, Any]], padding: int) -> None:
    fields = [
        "experiment_id",
        "experiment_type",
        "game_type",
        "baseline_model",
        "adversary_model",
        "model_order",
        "conceptual_order",
        "competition_level",
        "rho",
        "theta",
        "alpha",
        "sigma",
        "competition_index",
        "run_number",
        "seed",
        "discussion_turns",
        "config_file",
        "output_dir",
    ]
    with (config_dir / "experiment_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cfg in configs:
            writer.writerow(
                {
                    "experiment_id": cfg["experiment_id"],
                    "experiment_type": cfg["experiment_type"],
                    "game_type": cfg["game_type"],
                    "baseline_model": cfg["baseline_model"],
                    "adversary_model": cfg["adversary_model"],
                    "model_order": cfg["model_order"],
                    "conceptual_order": cfg["conceptual_order"],
                    "competition_level": cfg.get("competition_level", ""),
                    "rho": cfg.get("rho", ""),
                    "theta": cfg.get("theta", ""),
                    "alpha": cfg.get("alpha", ""),
                    "sigma": cfg.get("sigma", ""),
                    "competition_index": cfg.get("competition_index", ""),
                    "run_number": cfg["run_number"],
                    "seed": cfg["random_seed"],
                    "discussion_turns": cfg["discussion_turns"],
                    "config_file": f"config_{cfg['experiment_id']:0{padding}d}.json",
                    "output_dir": cfg["output_dir"],
                }
            )


def _write_summary(run_dir: Path, configs: List[Dict[str, Any]], settings: GameSettings) -> None:
    game_type = configs[0]["game_type"] if configs else "unknown"
    summary = [
        "Appendix Llama 3.3 Baseline Batch",
        "==================================",
        f"Created at: {datetime.now().isoformat(timespec='seconds')}",
        f"Run directory: {run_dir}",
        f"Game label: {settings.game_label}",
        f"Game type: {game_type}",
        f"Baseline model: {BASELINE_MODEL}",
        f"Adversary models: {', '.join(ADVERSARIES)}",
        f"Total configs: {len(configs)}",
        "Discussion turns: 2",
        "Max rounds: 10",
        "Max tokens per phase: 16384",
        "Slurm mode: one CPU sbatch job per config",
        "",
        "Do not use model_order labels as capability labels in analysis.",
        "weak_first means baseline_first; strong_first means adversary_first.",
        "",
    ]
    (run_dir / "configs" / "summary.txt").write_text("\n".join(summary), encoding="utf-8")


def _run_config_py() -> str:
    return f'''#!/usr/bin/env python3
"""Run one generated appendix Llama 3.3 baseline config."""

from __future__ import annotations

import json
import os
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
        "--gamma-discount",
        str(cfg["gamma_discount"]),
        "--discussion-turns",
        str(cfg["discussion_turns"]),
        "--model-order",
        cfg["model_order"],
        "--random-seed",
        str(cfg["random_seed"]),
        "--output-dir",
        cfg["output_dir"],
        "--job-id",
        str(cfg["experiment_id"]),
    ]

    game_type = cfg["game_type"]
    if game_type == "item_allocation":
        cmd.extend([
            "--num-items",
            str(cfg.get("num_items", cfg.get("m_items", 5))),
            "--competition-level",
            str(cfg["competition_level"]),
        ])
    elif game_type == "diplomacy":
        cmd.extend([
            "--n-issues",
            str(cfg["n_issues"]),
            "--rho",
            str(cfg["rho"]),
            "--theta",
            str(cfg["theta"]),
        ])
    elif game_type == "co_funding":
        cmd.extend([
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
            "--cofunding-time-discount",
            str(cfg["cofunding_time_discount"]),
        ])
        if not cfg.get("cofunding_enable_commit_vote", True):
            cmd.append("--cofunding-disable-commit-vote")
        if not cfg.get("cofunding_enable_time_discount", True):
            cmd.append("--cofunding-disable-time-discount")
    else:
        raise ValueError(f"Unsupported game_type: {{game_type}}")

    append_if_present(cmd, cfg, "--max-tokens-per-phase", "max_tokens_per_phase")

    env = os.environ.copy()
    metadata = {{
        "experiment_type": cfg.get("experiment_type"),
        "appendix_batch": "llama33_baseline_202605",
        "baseline_model": cfg.get("baseline_model"),
        "adversary_model": cfg.get("adversary_model"),
        "model1": cfg.get("model1"),
        "model2": cfg.get("model2"),
        "conceptual_order": cfg.get("conceptual_order"),
        "config_file": str(config_path),
    }}
    env["EXPERIMENT_RUN_METADATA_JSON"] = json.dumps(metadata)

    print("Config:", config_path, flush=True)
    print("Game:", cfg["game_label"], cfg["game_type"], flush=True)
    print("Baseline:", cfg["baseline_model"], flush=True)
    print("Adversary:", cfg["adversary_model"], flush=True)
    print("Model order:", cfg["model_order"], f"({{cfg['conceptual_order']}})", flush=True)
    print("Command:", " ".join(shlex.quote(part) for part in cmd), flush=True)

    completed = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _write_slurm_files(
    run_dir: Path,
    settings: GameSettings,
    total_configs: int,
    padding: int,
    args: argparse.Namespace,
) -> None:
    slurm_dir = run_dir / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "slurm_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    run_config = slurm_dir / "run_config.py"
    run_config.write_text(_run_config_py(), encoding="utf-8")
    run_config.chmod(run_config.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)

    sbatch = f"""#!/bin/bash
#SBATCH --partition={args.slurm_partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={args.cpus_per_task}
#SBATCH --mem={args.mem}
#SBATCH --time={args.slurm_time}
#SBATCH --output={log_dir}/%x_%j.out
#SBATCH --error={log_dir}/%x_%j.err

set -eo pipefail

BASE_DIR="{REPO_ROOT}"
RUN_DIR="{run_dir}"
cd "$BASE_DIR"
mkdir -p "$RUN_DIR/monitoring" "{log_dir}"

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
export OPENROUTER_TRANSPORT="${{OPENROUTER_TRANSPORT:-proxy}}"
export OPENROUTER_PROXY_POLL_DIR="${{OPENROUTER_PROXY_POLL_DIR:-/home/jz4391/openrouter_proxy}}"
export OPENROUTER_PROXY_CLIENT_TIMEOUT="${{OPENROUTER_PROXY_CLIENT_TIMEOUT:-9000}}"
export LLM_FAILURE_REPORT_PATH="${{LLM_FAILURE_REPORT_PATH:-$RUN_DIR/monitoring/provider_failures.md}}"

PYTHON_BIN="$BASE_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

if [[ -z "${{CONFIG_FILE:-}}" ]]; then
  echo "ERROR: CONFIG_FILE was not exported to the Slurm job."
  exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "ERROR: Config file not found: $CONFIG_FILE"
  exit 1
fi

echo "Appendix Llama 3.3 baseline Slurm task"
echo "Job: ${{SLURM_JOB_ID:-local}}"
echo "Node: ${{SLURM_NODELIST:-unknown}}"
echo "Config: $CONFIG_FILE"
echo "OpenRouter transport: $OPENROUTER_TRANSPORT"
echo "Started: $(date)"

"$PYTHON_BIN" "$RUN_DIR/slurm/run_config.py" "$CONFIG_FILE"
"""
    run_one = slurm_dir / "run_one.sbatch"
    run_one.write_text(sbatch, encoding="utf-8")
    run_one.chmod(run_one.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)

    submit = f"""#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
RUN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TOTAL_CONFIGS={total_configs}
PADDING={padding}
DELAY_SECONDS=0
START_ID=0
END_ID=$((TOTAL_CONFIGS - 1))

while [[ $# -gt 0 ]]; do
  case "$1" in
    --delay-seconds)
      DELAY_SECONDS="$2"
      shift 2
      ;;
    --start-id)
      START_ID="$2"
      shift 2
      ;;
    --end-id)
      END_ID="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

mkdir -p "$RUN_DIR/slurm_logs"
SUBMITTED="$RUN_DIR/slurm/submitted_jobs.txt"
: > "$SUBMITTED"

echo "Run dir: $RUN_DIR"
echo "Submitting individual CPU jobs for configs $START_ID through $END_ID"

for CONFIG_ID in $(seq "$START_ID" "$END_ID"); do
  CONFIG_PADDED=$(printf "%0${{PADDING}}d" "$CONFIG_ID")
  CONFIG_FILE="$RUN_DIR/configs/config_${{CONFIG_PADDED}}.json"
  if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
  fi

  JOB_NAME="{settings.slurm_job_prefix}_${{CONFIG_PADDED}}"
  JOB_ID=$(sbatch --parsable \\
    --job-name="$JOB_NAME" \\
    --export=ALL,CONFIG_FILE="$CONFIG_FILE" \\
    "$SCRIPT_DIR/run_one.sbatch")
  echo "${{CONFIG_PADDED}},${{JOB_ID}},${{CONFIG_FILE}}" | tee -a "$SUBMITTED"

  if [[ "$DELAY_SECONDS" != "0" ]]; then
    sleep "$DELAY_SECONDS"
  fi
done

echo "Submitted $(wc -l < "$SUBMITTED") jobs."
echo "Job manifest: $SUBMITTED"
"""
    submit_all = slurm_dir / "submit_individual.sh"
    submit_all.write_text(submit, encoding="utf-8")
    submit_all.chmod(submit_all.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)


def _materialize_game(settings: GameSettings, configs: List[Dict[str, Any]], args: argparse.Namespace) -> Path:
    run_dir = RESULTS_ROOT / settings.run_name
    if run_dir.exists() and not args.force:
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    config_dir = run_dir / "configs"
    analysis_dir = run_dir / "analysis"
    config_dir.mkdir(parents=True, exist_ok=args.force)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    padding = 4
    config_paths: List[Path] = []
    for cfg in configs:
        path = config_dir / f"config_{cfg['experiment_id']:0{padding}d}.json"
        _write_json(path, cfg)
        config_paths.append(path)

    (config_dir / "all_configs.txt").write_text(
        "".join(f"{path}\n" for path in config_paths),
        encoding="utf-8",
    )
    _write_index(config_dir, configs, padding)
    _write_summary(run_dir, configs, settings)
    _write_slurm_files(run_dir, settings, len(configs), padding, args)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suffix", default="202605")
    parser.add_argument("--slurm-partition", default="cpu")
    parser.add_argument("--slurm-time", default="06:00:00")
    parser.add_argument("--cpus-per-task", type=int, default=1)
    parser.add_argument("--mem", default="8G")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_models()

    game_settings = [
        GameSettings(
            game_label="game1",
            run_name=f"appendix_llama33_baseline_game1_{args.suffix}",
            total_configs=140,
            slurm_job_prefix="apx_l33_g1",
        ),
        GameSettings(
            game_label="game2",
            run_name=f"appendix_llama33_baseline_game2_{args.suffix}",
            total_configs=180,
            slurm_job_prefix="apx_l33_g2",
        ),
        GameSettings(
            game_label="game3",
            run_name=f"appendix_llama33_baseline_game3_{args.suffix}",
            total_configs=180,
            slurm_job_prefix="apx_l33_g3",
        ),
    ]
    builders = {
        "game1": _game1_configs,
        "game2": _game2_configs,
        "game3": _game3_configs,
    }

    run_dirs = []
    for settings in game_settings:
        configs = list(builders[settings.game_label](settings.run_name))
        if len(configs) != settings.total_configs:
            raise AssertionError(
                f"{settings.game_label}: expected {settings.total_configs}, got {len(configs)}"
            )
        run_dir = _materialize_game(settings, configs, args)
        run_dirs.append(run_dir)
        print(f"{settings.game_label}: wrote {len(configs)} configs to {run_dir}")

    print("Generated appendix Llama 3.3 baseline batches:")
    for run_dir in run_dirs:
        print(f"  {run_dir}")
    print("Submit with each run_dir/slurm/submit_individual.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

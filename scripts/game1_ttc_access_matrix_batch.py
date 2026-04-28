#!/usr/bin/env python3
"""Run Game 1 access-scaling matrix experiments.

Curves:
- GPT-5 nano scaled first vs GPT-5 nano
- GPT-5 nano scaled second vs GPT-5 nano
- Claude Opus 4.6 scaled first vs GPT-5 nano
- Claude Opus 4.6 scaled second vs GPT-5 nano
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_BASE = PROJECT_ROOT / "experiments" / "results"

BASELINE_MODEL = "gpt-5-nano"
STRONG_MODEL = "claude-opus-4-6"
K_VALUES = [1, 2, 4, 8]
SEEDS = [41001, 41002, 41003]
NUM_ITEMS = 5
MAX_ROUNDS = 10
COMPETITION_LEVEL = 1.0
GAMMA_DISCOUNT = 0.9
DISCUSSION_TURNS = 2
ACCESS_PHASES = ["discussion", "proposal", "voting", "reflection"]

CURVES = [
    {
        "curve_id": "nano_scaled_first",
        "curve_label": "GPT-5 nano scaled first",
        "scaled_model": BASELINE_MODEL,
        "opponent_model": BASELINE_MODEL,
        "models": [BASELINE_MODEL, BASELINE_MODEL],
        "access_agent_index": 0,
        "focal_agent_id": "Agent_1",
        "opponent_agent_id": "Agent_2",
        "scaled_position": "first",
        "model_order": "access_first",
    },
    {
        "curve_id": "nano_scaled_second",
        "curve_label": "GPT-5 nano scaled second",
        "scaled_model": BASELINE_MODEL,
        "opponent_model": BASELINE_MODEL,
        "models": [BASELINE_MODEL, BASELINE_MODEL],
        "access_agent_index": 1,
        "focal_agent_id": "Agent_2",
        "opponent_agent_id": "Agent_1",
        "scaled_position": "second",
        "model_order": "access_second",
    },
    {
        "curve_id": "opus_scaled_first",
        "curve_label": "Claude Opus 4.6 scaled first",
        "scaled_model": STRONG_MODEL,
        "opponent_model": BASELINE_MODEL,
        "models": [STRONG_MODEL, BASELINE_MODEL],
        "access_agent_index": 0,
        "focal_agent_id": "Agent_1",
        "opponent_agent_id": "Agent_2",
        "scaled_position": "first",
        "model_order": "strong_first",
    },
    {
        "curve_id": "opus_scaled_second",
        "curve_label": "Claude Opus 4.6 scaled second",
        "scaled_model": STRONG_MODEL,
        "opponent_model": BASELINE_MODEL,
        "models": [BASELINE_MODEL, STRONG_MODEL],
        "access_agent_index": 1,
        "focal_agent_id": "Agent_2",
        "opponent_agent_id": "Agent_1",
        "scaled_position": "second",
        "model_order": "weak_first",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--results-root", type=str, default=None)

    submit = subparsers.add_parser("submit")
    submit.add_argument("--results-root", type=str, required=True)
    submit.add_argument("--max-concurrent", type=int, default=16)
    submit.add_argument("--time", type=str, default="08:00:00")

    run_one = subparsers.add_parser("run-one")
    run_one.add_argument("--results-root", type=str, required=True)
    run_one.add_argument("--config-id", type=int, required=True)

    status = subparsers.add_parser("status")
    status.add_argument("--results-root", type=str, required=True)
    status.add_argument("--job-id", type=str, default=None)
    status.add_argument("--json", action="store_true")

    report = subparsers.add_parser("report")
    report.add_argument("--results-root", type=str, required=True)
    report.add_argument("--job-id", type=str, default=None)

    plot = subparsers.add_parser("plot")
    plot.add_argument("--results-root", type=str, required=True)
    plot.add_argument("--allow-partial", action="store_true")

    return parser.parse_args()


def timestamp_now() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_results_root(raw_value: Optional[str], *, create_new: bool = False) -> Path:
    if raw_value:
        path = Path(raw_value)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        return path
    if not create_new:
        raise ValueError("--results-root is required for this command")
    return (RESULTS_BASE / f"game1_ttc_access_matrix_{timestamp_now()}").resolve()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def create_latest_symlink(results_root: Path) -> None:
    symlink = results_root.parent / "game1_ttc_access_matrix_latest"
    if symlink.is_symlink() or symlink.exists():
        symlink.unlink()
    symlink.symlink_to(results_root.name)


def build_configs(results_root: Path) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    config_id = 1
    for curve in CURVES:
        for k in K_VALUES:
            for seed_index, seed in enumerate(SEEDS, start=1):
                output_dir = (
                    results_root
                    / "runs"
                    / curve["curve_id"]
                    / f"k{k}_seed{seed}"
                )
                config = {
                    "config_id": config_id,
                    "experiment_family": "game1_ttc_access_matrix",
                    "curve_id": curve["curve_id"],
                    "curve_label": curve["curve_label"],
                    "scaled_model": curve["scaled_model"],
                    "opponent_model": curve["opponent_model"],
                    "scaled_position": curve["scaled_position"],
                    "models": curve["models"],
                    "game_type": "item_allocation",
                    "n_agents": 2,
                    "model_order": curve["model_order"],
                    "focal_agent_id": curve["focal_agent_id"],
                    "opponent_agent_id": curve["opponent_agent_id"],
                    "access_k": k,
                    "access_agent_index": curve["access_agent_index"],
                    "access_agent_id": curve["focal_agent_id"],
                    "access_phases": ACCESS_PHASES,
                    "access_mechanism": "K total calls per scaled phase: K-1 private drafts plus one self-selector",
                    "seed_index": seed_index,
                    "random_seed": seed,
                    "num_items": NUM_ITEMS,
                    "max_rounds": MAX_ROUNDS,
                    "competition_level": COMPETITION_LEVEL,
                    "gamma_discount": GAMMA_DISCOUNT,
                    "discussion_turns": DISCUSSION_TURNS,
                    "parallel_phases": True,
                    "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
                }
                configs.append(config)
                config_id += 1
    return configs


def write_generated_files(results_root: Path, configs: List[Dict[str, Any]]) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    config_dir = results_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "results_root": str(results_root),
        "batch_type": "game1_ttc_access_matrix",
        "created_at": dt.datetime.now().isoformat(),
        "baseline_model": BASELINE_MODEL,
        "strong_model": STRONG_MODEL,
        "curves": CURVES,
        "k_values": K_VALUES,
        "seeds": SEEDS,
        "num_configs": len(configs),
        "game_type": "item_allocation",
        "num_items": NUM_ITEMS,
        "max_rounds": MAX_ROUNDS,
        "competition_level": COMPETITION_LEVEL,
        "gamma_discount": GAMMA_DISCOUNT,
        "discussion_turns": DISCUSSION_TURNS,
        "access_phases": ACCESS_PHASES,
        "access_mechanism": "black-box private candidate drafts plus self-selection; K is total model calls per scaled phase",
        "error_bars": "SEM across three matched seeds",
    }
    write_json(results_root / "manifest.json", manifest)

    fieldnames = [
        "config_id",
        "curve_id",
        "curve_label",
        "scaled_model",
        "scaled_position",
        "access_k",
        "seed_index",
        "random_seed",
        "models",
        "model_order",
        "focal_agent_id",
        "access_agent_index",
        "access_phases",
        "output_dir",
        "config_file",
    ]
    with (config_dir / "experiment_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for config in configs:
            path = config_dir / f"config_{config['config_id']:04d}.json"
            write_json(path, config)
            row = {key: config.get(key) for key in fieldnames if key not in {"models", "access_phases", "config_file"}}
            row["models"] = "+".join(config["models"])
            row["access_phases"] = "+".join(config["access_phases"])
            row["config_file"] = path.name
            writer.writerow(row)
    (config_dir / "all_configs.txt").write_text(
        "\n".join(str(config_dir / f"config_{config['config_id']:04d}.json") for config in configs) + "\n",
        encoding="utf-8",
    )


def load_config(results_root: Path, config_id: int) -> Dict[str, Any]:
    path = results_root / "configs" / f"config_{config_id:04d}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    return load_json(path)


def load_configs(results_root: Path) -> List[Dict[str, Any]]:
    return [
        load_json(path)
        for path in sorted((results_root / "configs").glob("config_*.json"))
    ]


def result_path_for(config: Dict[str, Any]) -> Path:
    return (PROJECT_ROOT / config["output_dir"]).resolve() / "experiment_results.json"


def interactions_path_for(config: Dict[str, Any]) -> Path:
    return (PROJECT_ROOT / config["output_dir"]).resolve() / "all_interactions.json"


def load_dotenv_env() -> Dict[str, str]:
    env = dict(os.environ)
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return env
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env.setdefault(key.strip(), value.strip().strip("'").strip('"'))
    return env


def build_command(config: Dict[str, Any]) -> List[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "run_strong_models_experiment.py"),
        "--models",
        *config["models"],
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
        "--discussion-turns",
        str(config["discussion_turns"]),
        "--model-order",
        str(config["model_order"]),
        "--random-seed",
        str(config["random_seed"]),
        "--job-id",
        str(config["config_id"]),
        "--access-k",
        str(config["access_k"]),
        "--access-agent-index",
        str(config["access_agent_index"]),
        "--access-phases",
        *config["access_phases"],
        "--output-dir",
        str((PROJECT_ROOT / config["output_dir"]).resolve()),
    ]
    if config.get("parallel_phases"):
        cmd.append("--parallel-phases")
    return cmd


def enrich_result_metadata(config: Dict[str, Any]) -> None:
    path = result_path_for(config)
    if not path.exists():
        return
    payload = load_json(path)
    payload.setdefault("config", {})
    for key in [
        "config_id",
        "experiment_family",
        "curve_id",
        "curve_label",
        "scaled_model",
        "opponent_model",
        "scaled_position",
        "models",
        "model_order",
        "focal_agent_id",
        "opponent_agent_id",
        "access_k",
        "access_agent_index",
        "access_agent_id",
        "access_phases",
        "access_mechanism",
        "seed_index",
    ]:
        payload["config"][key] = config[key]
    write_json(path, payload)


def validate_result(config: Dict[str, Any]) -> Optional[str]:
    path = result_path_for(config)
    if not path.exists():
        return "missing result file"
    try:
        payload = load_json(path)
    except Exception as exc:
        return f"invalid result JSON: {exc}"
    utilities = payload.get("final_utilities") or {}
    if config["focal_agent_id"] not in utilities:
        return f"missing focal utility for {config['focal_agent_id']}"
    if config["opponent_agent_id"] not in utilities:
        return f"missing opponent utility for {config['opponent_agent_id']}"
    if payload.get("config", {}).get("random_seed") != config["random_seed"]:
        return "result seed does not match config"
    observed_access = payload.get("config", {}).get("access_config") or {}
    observed_k = int(observed_access.get("k", config["access_k"]))
    if observed_k != int(config["access_k"]):
        return "result access_k does not match config"
    observed_index = int(observed_access.get("agent_index", config["access_agent_index"]))
    if observed_index != int(config["access_agent_index"]):
        return "result access_agent_index does not match config"
    return None


def status_path_for(results_root: Path, config: Dict[str, Any]) -> Path:
    return results_root / "status" / f"config_{config['config_id']:04d}.json"


def run_config(results_root: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    output_dir = (PROJECT_ROOT / config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    monitoring_dir = results_root / "monitoring"
    monitoring_dir.mkdir(parents=True, exist_ok=True)
    log_dir = results_root / "logs"
    status_dir = results_root / "status"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"config_{config['config_id']:04d}.log"
    status_path = status_path_for(results_root, config)

    if validate_result(config) is None:
        enrich_result_metadata(config)
        status = {
            "config_id": config["config_id"],
            "curve_id": config["curve_id"],
            "access_k": config["access_k"],
            "state": "SUCCESS",
            "returncode": 0,
            "skipped_existing": True,
            "result_path": str(result_path_for(config)),
            "updated_at": dt.datetime.now().isoformat(),
        }
        write_json(status_path, status)
        return status

    cmd = build_command(config)
    write_json(
        status_path,
        {
            "config_id": config["config_id"],
            "curve_id": config["curve_id"],
            "access_k": config["access_k"],
            "seed": config["random_seed"],
            "state": "RUNNING",
            "command": shlex.join(cmd),
            "started_at": dt.datetime.now().isoformat(),
            "log_path": str(log_path),
        },
    )
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Started at {dt.datetime.now().isoformat()}\n")
        handle.write(f"Config: {json.dumps(config, sort_keys=True)}\n")
        handle.write(f"Command: {shlex.join(cmd)}\n\n")
        handle.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env={
                **load_dotenv_env(),
                "PYTHONUNBUFFERED": "1",
                "LLM_FAILURE_REPORT_PATH": str(monitoring_dir / "provider_failures.md"),
            },
        )
        returncode = proc.wait()
        handle.write(f"\nReturncode: {returncode}\n")

    result_error = validate_result(config)
    state = "SUCCESS" if returncode == 0 and result_error is None else "FAILED"
    if state == "SUCCESS":
        enrich_result_metadata(config)
    status = {
        "config_id": config["config_id"],
        "curve_id": config["curve_id"],
        "access_k": config["access_k"],
        "seed": config["random_seed"],
        "state": state,
        "returncode": returncode,
        "result_error": result_error,
        "log_path": str(log_path),
        "result_path": str(result_path_for(config)) if result_path_for(config).exists() else None,
        "updated_at": dt.datetime.now().isoformat(),
    }
    write_json(status_path, status)
    return status


def write_slurm_file(results_root: Path, num_configs: int, max_concurrent: int, slurm_time: str) -> Path:
    slurm_dir = results_root / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    python_bin = PROJECT_ROOT / ".venv" / "bin" / "python"
    script_path = PROJECT_ROOT / "scripts" / "game1_ttc_access_matrix_batch.py"
    sbatch_path = slurm_dir / "run_access_matrix.sbatch"
    sbatch_text = f"""#!/bin/bash
#SBATCH --job-name=g1ttcmat
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time={slurm_time}
#SBATCH --partition=cpu
#SBATCH --array=1-{num_configs}%{max_concurrent}
#SBATCH --output={PROJECT_ROOT}/slurm/game1_ttc_matrix_%A_%a.out
#SBATCH --error={PROJECT_ROOT}/slurm/game1_ttc_matrix_%A_%a.err

set -eo pipefail

BASE_DIR="{PROJECT_ROOT}"
RUN_DIR="{results_root}"
cd "$BASE_DIR"

mkdir -p "{PROJECT_ROOT / 'slurm'}"

module purge
module load anaconda3/2024.2
module load proxy/default

KEY_ENV_FILE="${{BARGAIN_API_KEYS_ENV:-/home/jz4391/.config/bargain/api_keys.env}}"
if [[ -f "$KEY_ENV_FILE" ]]; then
  set -a
  source "$KEY_ENV_FILE"
  set +a
fi

export LLM_FAILURE_REPORT_PATH="${{LLM_FAILURE_REPORT_PATH:-$RUN_DIR/monitoring/provider_failures.md}}"
export PYTHONUNBUFFERED=1

"{python_bin}" "{script_path}" run-one --results-root "$RUN_DIR" --config-id "$SLURM_ARRAY_TASK_ID"
"""
    sbatch_path.write_text(sbatch_text, encoding="utf-8")
    os.chmod(sbatch_path, 0o755)
    return sbatch_path


def squeue_states(job_id: Optional[str]) -> List[str]:
    if not job_id:
        return []
    try:
        completed = subprocess.run(
            ["squeue", "-h", "-j", str(job_id), "-o", "%i %T %M"],
            cwd=str(PROJECT_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError:
        return []
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def status_payload(results_root: Path, job_id: Optional[str] = None) -> Dict[str, Any]:
    configs = load_configs(results_root)
    rows = []
    counts = {"SUCCESS": 0, "FAILED": 0, "RUNNING": 0, "PENDING": 0}
    for config in configs:
        result_error = validate_result(config)
        status_path = status_path_for(results_root, config)
        saved_status = load_json(status_path) if status_path.exists() else {}
        saved_state = saved_status.get("state")
        if result_error is None:
            state = "SUCCESS"
        elif saved_state in {"FAILED", "RUNNING"}:
            state = saved_state
        else:
            state = "PENDING"
        counts[state] = counts.get(state, 0) + 1
        rows.append(
            {
                "config_id": config["config_id"],
                "curve_id": config["curve_id"],
                "curve_label": config["curve_label"],
                "scaled_model": config["scaled_model"],
                "scaled_position": config["scaled_position"],
                "k": config["access_k"],
                "seed": config["random_seed"],
                "state": state,
                "result_error": result_error,
                "started_at": saved_status.get("started_at"),
                "updated_at": saved_status.get("updated_at"),
                "log_path": saved_status.get("log_path"),
                "result_path": str(result_path_for(config)) if result_path_for(config).exists() else None,
            }
        )
    manifest_path = results_root / "manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    effective_job_id = job_id or manifest.get("slurm_job_id")
    return {
        "results_root": str(results_root),
        "job_id": effective_job_id,
        "counts": counts,
        "total": len(configs),
        "squeue": squeue_states(effective_job_id),
        "rows": rows,
        "updated_at": dt.datetime.now().isoformat(),
    }


def parse_dt(value: Optional[str]) -> Optional[dt.datetime]:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value)
    except ValueError:
        return None


def estimate_eta(payload: Dict[str, Any]) -> str:
    rows = payload["rows"]
    now = dt.datetime.now()
    completed = [row for row in rows if row["state"] == "SUCCESS"]
    started_times = [parse_dt(row.get("started_at")) for row in rows if row.get("started_at")]
    started_times = [value for value in started_times if value is not None]
    if not completed or not started_times:
        return "ETA unavailable until first completions"
    first_start = min(started_times)
    elapsed = max((now - first_start).total_seconds(), 1.0)
    rate = len(completed) / elapsed
    remaining = payload["total"] - len(completed)
    if rate <= 0 or remaining <= 0:
        return "ETA now"
    eta_seconds = remaining / rate
    eta_time = now + dt.timedelta(seconds=eta_seconds)
    return f"ETA about {eta_seconds / 60:.1f} min ({eta_time.strftime('%H:%M')})"


def token_totals(config: Dict[str, Any], agent_id: str) -> Dict[str, int]:
    path = interactions_path_for(config)
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0}
    if not path.exists():
        return totals
    try:
        interactions = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return totals
    for interaction in interactions:
        if interaction.get("agent_id") != agent_id:
            continue
        usage = interaction.get("token_usage") or {}
        for key in totals:
            totals[key] += int(usage.get(key, 0) or 0)
    return totals


def load_result_rows(results_root: Path, *, allow_partial: bool = False) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    missing = []
    for config in load_configs(results_root):
        result_error = validate_result(config)
        if result_error is not None:
            missing.append((config["config_id"], result_error))
            continue
        payload = load_json(result_path_for(config))
        utilities = payload.get("final_utilities") or {}
        focal_tokens = token_totals(config, config["focal_agent_id"])
        rows.append(
            {
                "config_id": config["config_id"],
                "curve_id": config["curve_id"],
                "curve_label": config["curve_label"],
                "scaled_model": config["scaled_model"],
                "scaled_position": config["scaled_position"],
                "k": int(config["access_k"]),
                "seed_index": config["seed_index"],
                "random_seed": config["random_seed"],
                "focal_agent_id": config["focal_agent_id"],
                "opponent_agent_id": config["opponent_agent_id"],
                "focal_utility": float(utilities.get(config["focal_agent_id"], 0.0)),
                "opponent_utility": float(utilities.get(config["opponent_agent_id"], 0.0)),
                "consensus_reached": bool(payload.get("consensus_reached")),
                "final_round": int(payload.get("final_round") or 0),
                **{f"focal_{key}": value for key, value in focal_tokens.items()},
                "result_path": str(result_path_for(config)),
            }
        )
    if missing and not allow_partial:
        details = "; ".join(f"config {config_id}: {error}" for config_id, error in missing[:8])
        raise RuntimeError(f"Cannot plot before all configs are complete: {details}")
    return rows


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def sample_std(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summary_rows_from_results(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    keys = sorted({(row["curve_id"], row["k"]) for row in rows})
    for curve_id, k in keys:
        group = [row for row in rows if row["curve_id"] == curve_id and row["k"] == k]
        focal_values = [row["focal_utility"] for row in group]
        opponent_values = [row["opponent_utility"] for row in group]
        n = len(group)
        focal_std = sample_std(focal_values)
        opponent_std = sample_std(opponent_values)
        sample = group[0]
        summary.append(
            {
                "curve_id": curve_id,
                "curve_label": sample["curve_label"],
                "scaled_model": sample["scaled_model"],
                "scaled_position": sample["scaled_position"],
                "k": k,
                "n": n,
                "mean_focal_utility": mean(focal_values),
                "std_focal_utility": focal_std,
                "sem_focal_utility": focal_std / math.sqrt(n) if n else float("nan"),
                "mean_opponent_utility": mean(opponent_values),
                "std_opponent_utility": opponent_std,
                "sem_opponent_utility": opponent_std / math.sqrt(n) if n else float("nan"),
                "consensus_rate": mean(1.0 if row["consensus_reached"] else 0.0 for row in group),
                "mean_final_round": mean(row["final_round"] for row in group),
                "mean_focal_total_tokens": mean(row["focal_total_tokens"] for row in group),
            }
        )
    return summary


def parse_json_object(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}


def preferences_from_interactions(interactions: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    prefs: Dict[str, List[float]] = {}
    for interaction in interactions:
        if interaction.get("phase") != "game_setup":
            continue
        aid = interaction.get("agent_id")
        entries = re.findall(
            r"^\s*(\d+):\s*[^\n]+?->\s*([0-9.]+)\s*$",
            interaction.get("prompt", ""),
            flags=re.M,
        )
        if aid and entries:
            values = [0.0] * (max(int(index) for index, _ in entries) + 1)
            for index, value in entries:
                values[int(index)] = float(value)
            prefs[aid] = values
    return prefs


def allocation_utility(allocation: Dict[str, Any], agent_id: str, prefs: Dict[str, List[float]]) -> float:
    values = prefs.get(agent_id, [])
    total = 0.0
    for raw_index in allocation.get(agent_id, []) or []:
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            continue
        if 0 <= index < len(values):
            total += values[index]
    return total


def qualitative_metrics_for_run(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    path = interactions_path_for(config)
    if not path.exists():
        return None
    try:
        interactions = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    prefs = preferences_from_interactions(interactions)
    focal = config["focal_agent_id"]
    opponent = config["opponent_agent_id"]
    focal_props = []
    votes = []
    for interaction in interactions:
        phase = interaction.get("phase", "")
        if phase.startswith("proposal_round_"):
            payload = parse_json_object(interaction.get("response", ""))
            allocation = payload.get("allocation", {})
            if interaction.get("agent_id") == focal:
                focal_props.append(
                    {
                        "focal_u": allocation_utility(allocation, focal, prefs),
                        "opponent_u": allocation_utility(allocation, opponent, prefs),
                    }
                )
        elif phase.startswith("voting_round_"):
            payload = parse_json_object(interaction.get("response", ""))
            votes.append(
                {
                    "voter": interaction.get("agent_id"),
                    "proposal_by": payload.get("proposal_by"),
                    "vote": payload.get("vote_decision"),
                }
            )
    focal_on_opp = [
        vote for vote in votes
        if vote["voter"] == focal and vote["proposal_by"] == opponent
    ]
    opp_on_focal = [
        vote for vote in votes
        if vote["voter"] == opponent and vote["proposal_by"] == focal
    ]
    return {
        "focal_prop_mean_self": mean(prop["focal_u"] for prop in focal_props) if focal_props else float("nan"),
        "focal_prop_mean_opponent": mean(prop["opponent_u"] for prop in focal_props) if focal_props else float("nan"),
        "focal_accepts_opponent_rate": mean(1.0 if vote["vote"] == "accept" else 0.0 for vote in focal_on_opp) if focal_on_opp else float("nan"),
        "opponent_accepts_focal_rate": mean(1.0 if vote["vote"] == "accept" else 0.0 for vote in opp_on_focal) if opp_on_focal else float("nan"),
    }


def qualitative_summary(results_root: Path, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    config_by_id = {config["config_id"]: config for config in load_configs(results_root)}
    enriched = []
    for row in rows:
        config = config_by_id.get(row["config_id"])
        if not config:
            continue
        metrics = qualitative_metrics_for_run(config)
        if metrics is None:
            continue
        enriched.append({**row, **metrics})
    out = []
    for curve_id, k in sorted({(row["curve_id"], row["k"]) for row in enriched}):
        group = [row for row in enriched if row["curve_id"] == curve_id and row["k"] == k]
        if not group:
            continue
        out.append(
            {
                "curve_id": curve_id,
                "k": k,
                "n": len(group),
                "focal_prop_mean_self": mean(row["focal_prop_mean_self"] for row in group),
                "focal_prop_mean_opponent": mean(row["focal_prop_mean_opponent"] for row in group),
                "focal_accepts_opponent_rate": mean(row["focal_accepts_opponent_rate"] for row in group),
                "opponent_accepts_focal_rate": mean(row["opponent_accepts_focal_rate"] for row in group),
            }
        )
    return out


def plot_results(results_root: Path, *, allow_partial: bool = False) -> Dict[str, Any]:
    rows = load_result_rows(results_root, allow_partial=allow_partial)
    if not rows:
        raise RuntimeError("No completed rows available to plot")
    analysis_dir = results_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = analysis_dir / "game1_ttc_access_matrix_raw.csv"
    summary_csv = analysis_dir / "game1_ttc_access_matrix_summary.csv"
    qualitative_csv = analysis_dir / "game1_ttc_access_matrix_qualitative.csv"
    plot_png = analysis_dir / "game1_ttc_access_matrix_payoff_vs_k.png"
    plot_pdf = analysis_dir / "game1_ttc_access_matrix_payoff_vs_k.pdf"

    raw_fields = [
        "config_id",
        "curve_id",
        "curve_label",
        "scaled_model",
        "scaled_position",
        "k",
        "seed_index",
        "random_seed",
        "focal_agent_id",
        "opponent_agent_id",
        "focal_utility",
        "opponent_utility",
        "consensus_reached",
        "final_round",
        "focal_input_tokens",
        "focal_output_tokens",
        "focal_total_tokens",
        "focal_reasoning_tokens",
        "result_path",
    ]
    write_csv(raw_csv, rows, raw_fields)

    summary_rows = summary_rows_from_results(rows)
    summary_fields = [
        "curve_id",
        "curve_label",
        "scaled_model",
        "scaled_position",
        "k",
        "n",
        "mean_focal_utility",
        "std_focal_utility",
        "sem_focal_utility",
        "mean_opponent_utility",
        "std_opponent_utility",
        "sem_opponent_utility",
        "consensus_rate",
        "mean_final_round",
        "mean_focal_total_tokens",
    ]
    write_csv(summary_csv, summary_rows, summary_fields)

    q_rows = qualitative_summary(results_root, rows)
    write_csv(
        qualitative_csv,
        q_rows,
        [
            "curve_id",
            "k",
            "n",
            "focal_prop_mean_self",
            "focal_prop_mean_opponent",
            "focal_accepts_opponent_rate",
            "opponent_accepts_focal_rate",
        ],
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for curve in CURVES:
        curve_rows = [
            row for row in summary_rows
            if row["curve_id"] == curve["curve_id"]
        ]
        if not curve_rows:
            continue
        curve_rows.sort(key=lambda row: row["k"])
        ax.errorbar(
            [row["k"] for row in curve_rows],
            [row["mean_focal_utility"] for row in curve_rows],
            yerr=[row["sem_focal_utility"] for row in curve_rows],
            marker="o",
            linewidth=2,
            capsize=4,
            label=curve["curve_label"],
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_xlabel("Access scaling K (total model calls per scaled phase)")
    ax.set_ylabel("Scaled agent final discounted utility")
    ax.set_title("Game 1 Access Scaling: Model and Position Ablation")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(plot_png, dpi=220)
    fig.savefig(plot_pdf)
    plt.close(fig)

    payload = {
        "plot_png": str(plot_png),
        "plot_pdf": str(plot_pdf),
        "raw_csv": str(raw_csv),
        "summary_csv": str(summary_csv),
        "qualitative_csv": str(qualitative_csv),
        "rows": len(rows),
        "summary": summary_rows,
    }
    write_json(analysis_dir / "plot_manifest.json", payload)
    return payload


def report_text(results_root: Path, job_id: Optional[str] = None) -> str:
    payload = status_payload(results_root, job_id=job_id)
    counts = payload["counts"]
    lines = [
        (
            f"{counts.get('SUCCESS', 0)}/{payload['total']} complete, "
            f"{counts.get('RUNNING', 0)} running, "
            f"{counts.get('PENDING', 0)} pending, "
            f"{counts.get('FAILED', 0)} failed"
        ),
        estimate_eta(payload),
    ]
    rows = load_result_rows(results_root, allow_partial=True)
    if rows:
        lines.append("")
        lines.append("Partial utility trends, scaled agent mean +/- SEM:")
        for summary in summary_rows_from_results(rows):
            lines.append(
                f"- {summary['curve_label']} K={summary['k']} "
                f"n={summary['n']}: {summary['mean_focal_utility']:.2f} "
                f"+/- {summary['sem_focal_utility']:.2f}, "
                f"consensus={summary['consensus_rate']:.2f}, "
                f"mean_round={summary['mean_final_round']:.1f}"
            )
        q_rows = qualitative_summary(results_root, rows)
        if q_rows:
            lines.append("")
            lines.append("Transcript-derived bargaining signals:")
            for q in q_rows:
                lines.append(
                    f"- {q['curve_id']} K={q['k']} n={q['n']}: "
                    f"focal proposals avg self/other="
                    f"{q['focal_prop_mean_self']:.1f}/{q['focal_prop_mean_opponent']:.1f}; "
                    f"focal accepts opponent={q['focal_accepts_opponent_rate']:.2f}; "
                    f"opponent accepts focal={q['opponent_accepts_focal_rate']:.2f}"
                )
    else:
        lines.append("No completed result rows yet.")
    if counts.get("FAILED", 0):
        lines.append("")
        lines.append("Failed configs:")
        for row in payload["rows"]:
            if row["state"] == "FAILED":
                lines.append(f"- config {row['config_id']:04d} {row['curve_id']} K={row['k']}: {row['result_error']}")
    return "\n".join(lines)


def cmd_generate(args: argparse.Namespace) -> int:
    results_root = resolve_results_root(args.results_root, create_new=True)
    configs = build_configs(results_root)
    write_generated_files(results_root, configs)
    create_latest_symlink(results_root)
    print(results_root)
    print(f"Generated {len(configs)} configs")
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    results_root = resolve_results_root(args.results_root)
    configs = load_configs(results_root)
    sbatch_path = write_slurm_file(results_root, len(configs), args.max_concurrent, args.time)
    completed = subprocess.run(
        ["sbatch", str(sbatch_path)],
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        print(completed.stdout, end="")
        print(completed.stderr, end="", file=sys.stderr)
        return completed.returncode
    output = completed.stdout.strip()
    job_id = output.split()[-1] if output else None
    manifest_path = results_root / "manifest.json"
    manifest = load_json(manifest_path)
    manifest["slurm_job_id"] = job_id
    manifest["slurm_submit_output"] = output
    manifest["slurm_sbatch_path"] = str(sbatch_path)
    manifest["slurm_max_concurrent"] = args.max_concurrent
    manifest["submitted_at"] = dt.datetime.now().isoformat()
    write_json(manifest_path, manifest)
    print(output)
    print(f"results_root={results_root}")
    print(f"job_id={job_id}")
    return 0


def cmd_run_one(args: argparse.Namespace) -> int:
    results_root = resolve_results_root(args.results_root)
    config = load_config(results_root, args.config_id)
    status = run_config(results_root, config)
    print(json.dumps(status, indent=2, default=str))
    return 0 if status["state"] == "SUCCESS" else 1


def cmd_status(args: argparse.Namespace) -> int:
    results_root = resolve_results_root(args.results_root)
    payload = status_payload(results_root, job_id=args.job_id)
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return 0
    counts = payload["counts"]
    print(
        f"{counts.get('SUCCESS', 0)}/{payload['total']} complete, "
        f"{counts.get('RUNNING', 0)} running, "
        f"{counts.get('PENDING', 0)} pending, "
        f"{counts.get('FAILED', 0)} failed"
    )
    print(estimate_eta(payload))
    if payload["squeue"]:
        print("Slurm:")
        for line in payload["squeue"][:30]:
            print(f"  {line}")
    for row in payload["rows"]:
        if row["state"] != "SUCCESS":
            print(
                f"  config {row['config_id']:04d} {row['curve_id']} "
                f"K={row['k']} seed={row['seed']}: {row['state']} "
                f"{row.get('result_error') or ''}"
            )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    results_root = resolve_results_root(args.results_root)
    print(report_text(results_root, job_id=args.job_id))
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    results_root = resolve_results_root(args.results_root)
    payload = plot_results(results_root, allow_partial=args.allow_partial)
    print(json.dumps(payload, indent=2, default=str))
    return 0


def main() -> int:
    args = parse_args()
    return {
        "generate": cmd_generate,
        "submit": cmd_submit,
        "run-one": cmd_run_one,
        "status": cmd_status,
        "report": cmd_report,
        "plot": cmd_plot,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run and plot Game 1 black-box access-scaling experiments.

This batch fixes the checkpoint pair to gpt-5-nano vs gpt-5-nano, scales only
Agent_1's black-box access budget, and repeats each K on matched seeds.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_BASE = PROJECT_ROOT / "experiments" / "results"
MODEL = "gpt-5-nano"
K_VALUES = [1, 2, 4, 8]
SEEDS = [41001, 41002, 41003]
NUM_ITEMS = 5
MAX_ROUNDS = 10
COMPETITION_LEVEL = 1.0
GAMMA_DISCOUNT = 0.9
DISCUSSION_TURNS = 2
ACCESS_AGENT_INDEX = 0
FOCAL_AGENT_ID = "Agent_1"
OPPONENT_AGENT_ID = "Agent_2"
ACCESS_PHASES = ["proposal", "voting", "reflection"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--results-root", type=str, default=None)

    submit = subparsers.add_parser("submit")
    submit.add_argument("--results-root", type=str, required=True)
    submit.add_argument("--max-concurrent", type=int, default=12)
    submit.add_argument("--time", type=str, default="04:00:00")

    run_one = subparsers.add_parser("run-one")
    run_one.add_argument("--results-root", type=str, required=True)
    run_one.add_argument("--config-id", type=int, required=True)

    status = subparsers.add_parser("status")
    status.add_argument("--results-root", type=str, required=True)
    status.add_argument("--job-id", type=str, default=None)
    status.add_argument("--json", action="store_true")

    plot = subparsers.add_parser("plot")
    plot.add_argument("--results-root", type=str, required=True)
    plot.add_argument("--allow-partial", action="store_true")

    monitor = subparsers.add_parser("monitor")
    monitor.add_argument("--results-root", type=str, required=True)
    monitor.add_argument("--job-id", type=str, default=None)
    monitor.add_argument("--interval-seconds", type=int, default=300)

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
    return (RESULTS_BASE / f"game1_ttc_access_scaling_{timestamp_now()}").resolve()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def create_latest_symlink(results_root: Path) -> None:
    symlink = results_root.parent / "game1_ttc_access_scaling_latest"
    if symlink.is_symlink() or symlink.exists():
        symlink.unlink()
    symlink.symlink_to(results_root.name)


def build_configs(results_root: Path) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    config_id = 1
    for k in K_VALUES:
        for seed_index, seed in enumerate(SEEDS, start=1):
            output_dir = results_root / "runs" / f"k{k}_seed{seed}"
            configs.append(
                {
                    "config_id": config_id,
                    "experiment_family": "game1_ttc_access_scaling",
                    "models": [MODEL, MODEL],
                    "game_type": "item_allocation",
                    "n_agents": 2,
                    "model_order": "access_first",
                    "focal_agent_id": FOCAL_AGENT_ID,
                    "opponent_agent_id": OPPONENT_AGENT_ID,
                    "access_k": k,
                    "access_agent_index": ACCESS_AGENT_INDEX,
                    "access_agent_id": FOCAL_AGENT_ID,
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
            )
            config_id += 1
    return configs


def write_generated_files(results_root: Path, configs: List[Dict[str, Any]]) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    config_dir = results_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "results_root": str(results_root),
        "batch_type": "game1_ttc_access_scaling",
        "created_at": dt.datetime.now().isoformat(),
        "model_pair": [MODEL, MODEL],
        "focal_agent_id": FOCAL_AGENT_ID,
        "opponent_agent_id": OPPONENT_AGENT_ID,
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
        "access_k",
        "seed_index",
        "random_seed",
        "model_order",
        "num_items",
        "max_rounds",
        "competition_level",
        "discussion_turns",
        "access_phases",
        "output_dir",
        "config_file",
    ]
    with (config_dir / "experiment_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for config in configs:
            config_path = config_dir / f"config_{config['config_id']:04d}.json"
            write_json(config_path, config)
            row = {key: config.get(key) for key in fieldnames if key not in {"access_phases", "config_file"}}
            row["access_phases"] = "+".join(config["access_phases"])
            row["config_file"] = config_path.name
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
    result_path = result_path_for(config)
    if not result_path.exists():
        return
    payload = load_json(result_path)
    payload.setdefault("config", {})
    for key in [
        "config_id",
        "experiment_family",
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
    write_json(result_path, payload)


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
    observed_k = int((payload.get("config", {}).get("access_config") or {}).get("k", config["access_k"]))
    if observed_k != int(config["access_k"]):
        return "result access_k does not match config"
    return None


def run_config(results_root: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    output_dir = (PROJECT_ROOT / config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = results_root / "logs"
    status_dir = results_root / "status"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"config_{config['config_id']:04d}.log"
    status_path = status_dir / f"config_{config['config_id']:04d}.json"

    existing_error = validate_result(config)
    if existing_error is None:
        enrich_result_metadata(config)
        status = {
            "config_id": config["config_id"],
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
                "LLM_FAILURE_REPORT_PATH": str(results_root / "monitoring" / "provider_failures.md"),
                "EXPERIMENT_RUN_METADATA_JSON": json.dumps(config, default=str),
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
    script_path = PROJECT_ROOT / "scripts" / "game1_ttc_access_batch.py"
    sbatch_path = slurm_dir / "run_access_scaling.sbatch"
    sbatch_text = f"""#!/bin/bash
#SBATCH --job-name=g1ttcacc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time={slurm_time}
#SBATCH --partition=cpu
#SBATCH --array=1-{num_configs}%{max_concurrent}
#SBATCH --output={PROJECT_ROOT}/slurm/game1_ttc_access_%A_%a.out
#SBATCH --error={PROJECT_ROOT}/slurm/game1_ttc_access_%A_%a.err

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
            ["squeue", "-h", "-j", str(job_id), "-o", "%i %T"],
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
    status_dir = results_root / "status"
    rows = []
    counts = {"SUCCESS": 0, "FAILED": 0, "RUNNING": 0, "PENDING": 0}
    for config in configs:
        result_error = validate_result(config)
        status_path = status_dir / f"config_{config['config_id']:04d}.json"
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
                "k": config["access_k"],
                "seed": config["random_seed"],
                "state": state,
                "result_error": result_error,
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
        details = "; ".join(f"config {config_id}: {error}" for config_id, error in missing[:6])
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


def plot_results(results_root: Path, *, allow_partial: bool = False) -> Dict[str, Any]:
    rows = load_result_rows(results_root, allow_partial=allow_partial)
    if not rows:
        raise RuntimeError("No completed rows available to plot")

    analysis_dir = results_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = analysis_dir / "game1_ttc_access_raw.csv"
    summary_csv = analysis_dir / "game1_ttc_access_summary.csv"
    plot_png = analysis_dir / "game1_ttc_access_payoff_vs_k.png"
    plot_pdf = analysis_dir / "game1_ttc_access_payoff_vs_k.pdf"

    raw_fields = [
        "config_id",
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

    summary_rows: List[Dict[str, Any]] = []
    for k in sorted({row["k"] for row in rows}):
        group = [row for row in rows if row["k"] == k]
        focal_values = [row["focal_utility"] for row in group]
        opponent_values = [row["opponent_utility"] for row in group]
        n = len(group)
        focal_std = sample_std(focal_values)
        opponent_std = sample_std(opponent_values)
        summary_rows.append(
            {
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
    write_csv(
        summary_csv,
        summary_rows,
        [
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
        ],
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = [row["k"] for row in summary_rows]
    means = [row["mean_focal_utility"] for row in summary_rows]
    sems = [row["sem_focal_utility"] for row in summary_rows]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.errorbar(
        ks,
        means,
        yerr=sems,
        marker="o",
        linewidth=2,
        capsize=4,
        color="#1f77b4",
        label=f"{FOCAL_AGENT_ID} mean +/- SEM",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_xlabel("Access scaling K (total model calls per scaled phase)")
    ax.set_ylabel("Final discounted utility")
    ax.set_title("Game 1: GPT-5 Nano vs GPT-5 Nano Access Scaling")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_png, dpi=220)
    fig.savefig(plot_pdf)
    plt.close(fig)

    payload = {
        "plot_png": str(plot_png),
        "plot_pdf": str(plot_pdf),
        "raw_csv": str(raw_csv),
        "summary_csv": str(summary_csv),
        "rows": len(rows),
        "summary": summary_rows,
    }
    write_json(analysis_dir / "plot_manifest.json", payload)
    return payload


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
    if payload["squeue"]:
        print("Slurm:")
        for line in payload["squeue"][:20]:
            print(f"  {line}")
    for row in payload["rows"]:
        if row["state"] != "SUCCESS":
            print(f"  config {row['config_id']:04d} k={row['k']} seed={row['seed']}: {row['state']} {row.get('result_error') or ''}")
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    results_root = resolve_results_root(args.results_root)
    payload = plot_results(results_root, allow_partial=args.allow_partial)
    print(json.dumps(payload, indent=2, default=str))
    return 0


def cmd_monitor(args: argparse.Namespace) -> int:
    import time

    results_root = resolve_results_root(args.results_root)
    while True:
        payload = status_payload(results_root, job_id=args.job_id)
        counts = payload["counts"]
        print(
            f"[{dt.datetime.now().isoformat(timespec='seconds')}] "
            f"{counts.get('SUCCESS', 0)}/{payload['total']} complete, "
            f"{counts.get('RUNNING', 0)} running, "
            f"{counts.get('PENDING', 0)} pending, "
            f"{counts.get('FAILED', 0)} failed",
            flush=True,
        )
        if counts.get("FAILED", 0):
            return 1
        if counts.get("SUCCESS", 0) == payload["total"]:
            plot_payload = plot_results(results_root)
            print(json.dumps(plot_payload, indent=2, default=str))
            return 0
        time.sleep(args.interval_seconds)


def main() -> int:
    args = parse_args()
    return {
        "generate": cmd_generate,
        "submit": cmd_submit,
        "run-one": cmd_run_one,
        "status": cmd_status,
        "plot": cmd_plot,
        "monitor": cmd_monitor,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())

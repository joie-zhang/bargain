#!/usr/bin/env python3
"""Backfill Game 1 access scaling with Claude Opus 4.6 vs Claude Opus 4.6."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_SCRIPT = PROJECT_ROOT / "scripts" / "game1_ttc_access_matrix_batch.py"
RESULTS_BASE = PROJECT_ROOT / "experiments" / "results"
DEFAULT_NANO_ROOT = RESULTS_BASE / "game1_ttc_access_matrix_20260427_010142"

spec = importlib.util.spec_from_file_location("game1_ttc_access_matrix_batch", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(base)

OPUS_MODEL = "claude-opus-4-6"
OPUS_OPUS_CURVES = [
    {
        "curve_id": "opus_opus_scaled_first",
        "curve_label": "Claude Opus 4.6 scaled first",
        "scaled_model": OPUS_MODEL,
        "opponent_model": OPUS_MODEL,
        "models": [OPUS_MODEL, OPUS_MODEL],
        "access_agent_index": 0,
        "focal_agent_id": "Agent_1",
        "opponent_agent_id": "Agent_2",
        "scaled_position": "first",
        "model_order": "access_first",
    },
    {
        "curve_id": "opus_opus_scaled_second",
        "curve_label": "Claude Opus 4.6 scaled second",
        "scaled_model": OPUS_MODEL,
        "opponent_model": OPUS_MODEL,
        "models": [OPUS_MODEL, OPUS_MODEL],
        "access_agent_index": 1,
        "focal_agent_id": "Agent_2",
        "opponent_agent_id": "Agent_1",
        "scaled_position": "second",
        "model_order": "access_second",
    },
]

COMBINED_CURVES = [
    base.CURVES[0],
    base.CURVES[1],
    OPUS_OPUS_CURVES[0],
    OPUS_OPUS_CURVES[1],
]


def patch_base_for_opus_opus() -> None:
    base.BASELINE_MODEL = OPUS_MODEL
    base.STRONG_MODEL = OPUS_MODEL
    base.CURVES = OPUS_OPUS_CURVES


def timestamp_now() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_path(raw: str | None, *, default: Optional[Path] = None) -> Path:
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        return path
    if default is None:
        raise ValueError("path is required")
    return default.resolve()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_slurm_file(results_root: Path, num_configs: int, max_concurrent: int, slurm_time: str) -> Path:
    slurm_dir = results_root / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    python_bin = PROJECT_ROOT / ".venv" / "bin" / "python"
    sbatch_path = slurm_dir / "run_opus_opus_backfill.sbatch"
    sbatch_text = f"""#!/bin/bash
#SBATCH --job-name=g1ttcoo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time={slurm_time}
#SBATCH --partition=cpu
#SBATCH --array=1-{num_configs}%{max_concurrent}
#SBATCH --output={PROJECT_ROOT}/slurm/game1_ttc_opus_opus_%A_%a.out
#SBATCH --error={PROJECT_ROOT}/slurm/game1_ttc_opus_opus_%A_%a.err

set -eo pipefail

BASE_DIR="{PROJECT_ROOT}"
RUN_DIR="{results_root}"
cd "$BASE_DIR"

mkdir -p "{PROJECT_ROOT / 'slurm'}"

module purge
module load anaconda3/2024.2
module load proxy/default

export PYTHONUNBUFFERED=1

"{python_bin}" "{Path(__file__).resolve()}" run-one --results-root "$RUN_DIR" --config-id "$SLURM_ARRAY_TASK_ID"
"""
    sbatch_path.write_text(sbatch_text, encoding="utf-8")
    os.chmod(sbatch_path, 0o755)
    return sbatch_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--results-root", type=str, default=None)

    submit = subparsers.add_parser("submit")
    submit.add_argument("--results-root", type=str, required=True)
    submit.add_argument("--max-concurrent", type=int, default=8)
    submit.add_argument("--time", type=str, default="08:00:00")

    run_one = subparsers.add_parser("run-one")
    run_one.add_argument("--results-root", type=str, required=True)
    run_one.add_argument("--config-id", type=int, required=True)

    status = subparsers.add_parser("status")
    status.add_argument("--results-root", type=str, required=True)
    status.add_argument("--job-id", type=str, default=None)

    report = subparsers.add_parser("report")
    report.add_argument("--results-root", type=str, required=True)
    report.add_argument("--job-id", type=str, default=None)

    plot = subparsers.add_parser("plot")
    plot.add_argument("--results-root", type=str, required=True)
    plot.add_argument("--nano-results-root", type=str, default=str(DEFAULT_NANO_ROOT))

    return parser.parse_args()


def cmd_generate(args: argparse.Namespace) -> int:
    patch_base_for_opus_opus()
    results_root = resolve_path(
        args.results_root,
        default=RESULTS_BASE / f"game1_ttc_access_opus_opus_{timestamp_now()}",
    )
    configs = base.build_configs(results_root)
    base.write_generated_files(results_root, configs)
    manifest_path = results_root / "manifest.json"
    manifest = load_json(manifest_path)
    manifest["batch_type"] = "game1_ttc_access_opus_opus_backfill"
    manifest["description"] = "Backfill Claude Opus 4.6 vs Claude Opus 4.6 access-scaling curves."
    manifest["replacement_for"] = "Previous opus_scaled_first/second curves used GPT-5 nano as opponent."
    write_json(manifest_path, manifest)
    latest = results_root.parent / "game1_ttc_access_opus_opus_latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(results_root.name)
    print(results_root)
    print(f"Generated {len(configs)} configs")
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    results_root = resolve_path(args.results_root)
    configs = base.load_configs(results_root)
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
    patch_base_for_opus_opus()
    results_root = resolve_path(args.results_root)
    config = base.load_config(results_root, args.config_id)
    status = base.run_config(results_root, config)
    print(json.dumps(status, indent=2, default=str))
    return 0 if status["state"] == "SUCCESS" else 1


def cmd_status(args: argparse.Namespace) -> int:
    results_root = resolve_path(args.results_root)
    return base.cmd_status(argparse.Namespace(results_root=str(results_root), job_id=args.job_id, json=False))


def cmd_report(args: argparse.Namespace) -> int:
    results_root = resolve_path(args.results_root)
    print(base.report_text(results_root, job_id=args.job_id))
    return 0


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def sample_std(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def summary_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for curve_id, k in sorted({(row["curve_id"], int(row["k"])) for row in rows}):
        group = [row for row in rows if row["curve_id"] == curve_id and int(row["k"]) == k]
        vals = [float(row["focal_utility"]) for row in group]
        opp_vals = [float(row["opponent_utility"]) for row in group]
        std = sample_std(vals)
        opp_std = sample_std(opp_vals)
        sample = group[0]
        out.append(
            {
                "curve_id": curve_id,
                "curve_label": sample["curve_label"],
                "scaled_model": sample["scaled_model"],
                "scaled_position": sample["scaled_position"],
                "k": k,
                "n": len(group),
                "mean_focal_utility": mean(vals),
                "std_focal_utility": std,
                "sem_focal_utility": std / math.sqrt(len(group)) if group else float("nan"),
                "mean_opponent_utility": mean(opp_vals),
                "std_opponent_utility": opp_std,
                "sem_opponent_utility": opp_std / math.sqrt(len(group)) if group else float("nan"),
                "consensus_rate": mean([1.0 if row["consensus_reached"] else 0.0 for row in group]),
                "mean_final_round": mean([float(row["final_round"]) for row in group]),
            }
        )
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def cmd_plot(args: argparse.Namespace) -> int:
    patch_base_for_opus_opus()
    results_root = resolve_path(args.results_root)
    nano_root = resolve_path(args.nano_results_root)

    nano_rows = [
        row for row in base.load_result_rows(nano_root, allow_partial=False)
        if row["curve_id"] in {"nano_scaled_first", "nano_scaled_second"}
    ]
    opus_rows = base.load_result_rows(results_root, allow_partial=False)
    rows = nano_rows + opus_rows
    summaries = summary_rows(rows)

    analysis_dir = results_root / "analysis"
    raw_csv = analysis_dir / "game1_ttc_access_matrix_opus_opus_replacement_raw.csv"
    summary_csv = analysis_dir / "game1_ttc_access_matrix_opus_opus_replacement_summary.csv"
    plot_png = analysis_dir / "game1_ttc_access_matrix_opus_opus_replacement_payoff_vs_k.png"
    plot_pdf = analysis_dir / "game1_ttc_access_matrix_opus_opus_replacement_payoff_vs_k.pdf"

    raw_fields = [
        "config_id", "curve_id", "curve_label", "scaled_model", "scaled_position", "k",
        "seed_index", "random_seed", "focal_agent_id", "opponent_agent_id",
        "focal_utility", "opponent_utility", "consensus_reached", "final_round",
        "result_path",
    ]
    write_csv(raw_csv, [{key: row.get(key) for key in raw_fields} for row in rows], raw_fields)

    summary_fields = [
        "curve_id", "curve_label", "scaled_model", "scaled_position", "k", "n",
        "mean_focal_utility", "std_focal_utility", "sem_focal_utility",
        "mean_opponent_utility", "std_opponent_utility", "sem_opponent_utility",
        "consensus_rate", "mean_final_round",
    ]
    write_csv(summary_csv, summaries, summary_fields)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for curve in COMBINED_CURVES:
        curve_rows = [row for row in summaries if row["curve_id"] == curve["curve_id"]]
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
    ax.set_xticks(base.K_VALUES)
    ax.set_xticklabels([str(k) for k in base.K_VALUES])
    ax.set_xlabel("Access scaling K (total model calls per scaled phase)")
    ax.set_ylabel("Scaled agent final discounted utility")
    ax.set_title("Game 1 Access Scaling: Opus-vs-Opus Backfill")
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
        "nano_results_root": str(nano_root),
        "opus_opus_results_root": str(results_root),
        "rows": len(rows),
        "summary": summaries,
    }
    write_json(analysis_dir / "plot_manifest.json", payload)
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

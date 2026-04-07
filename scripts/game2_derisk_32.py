#!/usr/bin/env python3
"""Launch and collect a minimal Game 2 diplomacy de-risk batch for the final 32 models."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "experiments" / "results"
SLURM_LOG_DIR = REPO_ROOT / "slurm"
CANONICAL_MARKDOWN = REPO_ROOT / "docs" / "guides" / "chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md"
BASELINE_MODEL = "gpt-5-nano"

SMOKE_CONFIG = {
    "game_type": "diplomacy",
    "num_runs": 1,
    "max_rounds": 1,
    "n_issues": 5,
    "rho": 0.0,
    "theta": 0.5,
    "discussion_turns": 1,
    "disable_thinking": True,
    "disable_reflection": True,
    "disable_discussion": False,
    "max_tokens_per_phase": 1200,
}

FINAL_32_MODELS: List[Dict[str, Any]] = [
    {"rank": 1, "model": "claude-opus-4-6-thinking", "elo": 1504, "provider": "anthropic", "requires_gpu": False},
    {"rank": 2, "model": "claude-opus-4-6", "elo": 1499, "provider": "anthropic", "requires_gpu": False},
    {"rank": 3, "model": "gemini-3-pro", "elo": 1486, "provider": "google", "requires_gpu": False},
    {"rank": 4, "model": "gpt-5.4-high", "elo": 1484, "provider": "openai", "requires_gpu": False},
    {"rank": 5, "model": "gpt-5.2-chat-latest-20260210", "elo": 1478, "provider": "openai", "requires_gpu": False},
    {"rank": 6, "model": "claude-opus-4-5-20251101-thinking-32k", "elo": 1474, "provider": "anthropic", "requires_gpu": False},
    {"rank": 7, "model": "claude-opus-4-5-20251101", "elo": 1468, "provider": "anthropic", "requires_gpu": False},
    {"rank": 8, "model": "gemini-2.5-pro", "elo": 1448, "provider": "google", "requires_gpu": False},
    {"rank": 9, "model": "qwen3-max-preview", "elo": 1435, "provider": "openrouter", "requires_gpu": False},
    {"rank": 10, "model": "deepseek-r1-0528", "elo": 1422, "provider": "openrouter", "requires_gpu": False},
    {"rank": 11, "model": "claude-haiku-4-5-20251001", "elo": 1407, "provider": "anthropic", "requires_gpu": False},
    {"rank": 12, "model": "deepseek-r1", "elo": 1398, "provider": "openrouter", "requires_gpu": False},
    {"rank": 13, "model": "claude-sonnet-4-20250514", "elo": 1389, "provider": "anthropic", "requires_gpu": False},
    {"rank": 14, "model": "claude-3-5-sonnet-20241022", "elo": 1372, "provider": "openrouter", "requires_gpu": False},
    {"rank": 15, "model": "gemma-3-27b-it", "elo": 1365, "provider": "openrouter", "requires_gpu": False},
    {"rank": 16, "model": "o3-mini-high", "elo": 1363, "provider": "openai", "requires_gpu": False},
    {"rank": 17, "model": "deepseek-v3", "elo": 1358, "provider": "openrouter", "requires_gpu": False},
    {"rank": 18, "model": "gpt-4o-2024-05-13", "elo": 1345, "provider": "openai", "requires_gpu": False},
    {"rank": 19, "model": "qwq-32b", "elo": 1336, "provider": "openrouter", "requires_gpu": False},
    {"rank": 20, "model": "gpt-4.1-nano-2025-04-14", "elo": 1322, "provider": "openai", "requires_gpu": False},
    {"rank": 21, "model": "llama-3.3-70b-instruct", "elo": 1318, "provider": "openrouter", "requires_gpu": False},
    {"rank": 22, "model": "gpt-4o-mini-2024-07-18", "elo": 1317, "provider": "openai", "requires_gpu": False},
    {"rank": 23, "model": "qwen2.5-72b-instruct", "elo": 1302, "provider": "openrouter", "requires_gpu": False},
    {"rank": 24, "model": "amazon-nova-pro-v1.0", "elo": 1290, "provider": "openrouter", "requires_gpu": False},
    {"rank": 25, "model": "command-r-plus-08-2024", "elo": 1276, "provider": "openrouter", "requires_gpu": False},
    {"rank": 26, "model": "claude-3-haiku-20240307", "elo": 1260, "provider": "anthropic", "requires_gpu": False},
    {"rank": 27, "model": "amazon-nova-micro-v1.0", "elo": 1240, "provider": "openrouter", "requires_gpu": False},
    {"rank": 28, "model": "llama-3.1-8b-instruct", "elo": 1211, "provider": "openrouter", "requires_gpu": False},
    {"rank": 29, "model": "llama-3.2-3b-instruct", "elo": 1166, "provider": "openrouter", "requires_gpu": False},
    {"rank": 30, "model": "qwq-32b-preview", "elo": 1156, "provider": "openrouter", "requires_gpu": False},
    {"rank": 31, "model": "phi-3-mini-128k-instruct", "elo": 1128, "provider": "princeton_cluster", "requires_gpu": True},
    {"rank": 32, "model": "llama-3.2-1b-instruct", "elo": 1110, "provider": "openrouter", "requires_gpu": False},
]

STATIC_AUDIT_NOTES: Dict[str, Dict[str, str]] = {
    "claude-opus-4-6-thinking": {
        "level": "note",
        "message": "Uses the same Anthropic base route as claude-opus-4-6 with thinking enabled; shared base is intentional.",
    },
    "claude-opus-4-5-20251101-thinking-32k": {
        "level": "note",
        "message": "Uses the same Anthropic base route as claude-opus-4-5-20251101 with a thinking budget override; shared base is intentional.",
    },
    "qwen3-max-preview": {
        "level": "note",
        "message": "Alias route in this repo points to the current OpenRouter qwen/qwen3-max endpoint.",
    },
    "claude-3-5-sonnet-20241022": {
        "level": "fail",
        "message": "Repo route anthropic/claude-3.5-sonnet is retired and should not be used for new experiments.",
    },
    "qwq-32b-preview": {
        "level": "fail",
        "message": "Repo alias points to qwen/qwq-32b, so this is not a distinct live preview endpoint and should not be treated as a separate runnable model.",
    },
}


def timestamp_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def slugify(value: str) -> str:
    chars = []
    for ch in value.lower():
        if ch.isalnum():
            chars.append(ch)
        else:
            chars.append("_")
    slug = "".join(chars)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def load_manifest(run_dir: Path) -> Dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def build_manifest(run_dir: Path) -> Dict[str, Any]:
    return {
        "created_at": datetime.now().isoformat(),
        "repo_root": str(REPO_ROOT),
        "run_dir": str(run_dir),
        "canonical_markdown": str(CANONICAL_MARKDOWN),
        "baseline_model": BASELINE_MODEL,
        "smoke_config": SMOKE_CONFIG,
        "models": FINAL_32_MODELS,
    }


def make_cpu_sbatch(run_dir: Path) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name=g2d32-api
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output={SLURM_LOG_DIR}/game2_derisk_api_%A_%a.out
#SBATCH --error={SLURM_LOG_DIR}/game2_derisk_api_%A_%a.err

set -euo pipefail

BASE_DIR="{REPO_ROOT}"
RUN_DIR="{run_dir}"
cd "$BASE_DIR"
mkdir -p "{SLURM_LOG_DIR}"

module purge
module load anaconda3/2024.2
module load proxy/default

PYTHON_BIN="$BASE_DIR/.venv/bin/python"
export OPENROUTER_TRANSPORT="${{OPENROUTER_TRANSPORT:-proxy}}"
export OPENROUTER_PROXY_CLIENT_TIMEOUT="${{OPENROUTER_PROXY_CLIENT_TIMEOUT:-6000}}"

"$PYTHON_BIN" "$BASE_DIR/scripts/game2_derisk_32.py" run-one --run-dir "$RUN_DIR" --index "$SLURM_ARRAY_TASK_ID"
"""


def make_gpu_sbatch(run_dir: Path) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name=g2d32-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output={SLURM_LOG_DIR}/game2_derisk_gpu_%A_%a.out
#SBATCH --error={SLURM_LOG_DIR}/game2_derisk_gpu_%A_%a.err
#SBATCH --constraint=gpu80
#SBATCH --gres=gpu:a100:1

set -euo pipefail

BASE_DIR="{REPO_ROOT}"
RUN_DIR="{run_dir}"
cd "$BASE_DIR"
mkdir -p "{SLURM_LOG_DIR}"

module purge
module load anaconda3/2024.2
module load proxy/default

PYTHON_BIN="$BASE_DIR/.venv/bin/python"
export OPENROUTER_TRANSPORT="${{OPENROUTER_TRANSPORT:-proxy}}"
export OPENROUTER_PROXY_CLIENT_TIMEOUT="${{OPENROUTER_PROXY_CLIENT_TIMEOUT:-6000}}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$PYTHON_BIN" "$BASE_DIR/scripts/game2_derisk_32.py" run-one --run-dir "$RUN_DIR" --index "$SLURM_ARRAY_TASK_ID"
"""


def make_submit_script(run_dir: Path, api_indices: List[int], gpu_indices: List[int], api_max_concurrent: int) -> str:
    api_spec = ",".join(str(idx) for idx in api_indices)
    gpu_spec = ",".join(str(idx) for idx in gpu_indices)
    gpu_submit = ""
    if gpu_spec:
        gpu_submit = f"""
GPU_JOB_ID=$(sbatch --parsable --array={shlex.quote(gpu_spec)} "$SCRIPT_DIR/run_gpu.sbatch")
echo "$GPU_JOB_ID" | tee "$RUN_DIR/gpu_job_id.txt"
"""
    return f"""#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
RUN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

API_JOB_ID=$(sbatch --parsable --array={shlex.quote(api_spec)}%{api_max_concurrent} "$SCRIPT_DIR/run_api.sbatch")
echo "$API_JOB_ID" | tee "$RUN_DIR/api_job_id.txt"
{gpu_submit}
echo "Submitted run dir: $RUN_DIR"
echo "Monitor with: squeue -u $USER"
"""


def command_for_entry(entry: Dict[str, Any], output_dir: Path) -> List[str]:
    baseline_model = BASELINE_MODEL
    if entry["requires_gpu"]:
        # Local Princeton-cluster models should be smoked against themselves on
        # GPU nodes so the de-risk path does not depend on outbound API access.
        baseline_model = entry["model"]

    cmd = [
        sys.executable,
        "run_strong_models_experiment.py",
        "--game-type",
        SMOKE_CONFIG["game_type"],
        "--models",
        baseline_model,
        entry["model"],
        "--batch",
        "--num-runs",
        str(SMOKE_CONFIG["num_runs"]),
        "--run-number",
        "1",
        "--max-rounds",
        str(SMOKE_CONFIG["max_rounds"]),
        "--n-issues",
        str(SMOKE_CONFIG["n_issues"]),
        "--rho",
        str(SMOKE_CONFIG["rho"]),
        "--theta",
        str(SMOKE_CONFIG["theta"]),
        "--discussion-turns",
        str(SMOKE_CONFIG["discussion_turns"]),
        "--max-tokens-per-phase",
        str(SMOKE_CONFIG["max_tokens_per_phase"]),
        "--output-dir",
        str(output_dir),
    ]
    if SMOKE_CONFIG["disable_thinking"]:
        cmd.append("--disable-thinking")
    if SMOKE_CONFIG["disable_reflection"]:
        cmd.append("--disable-reflection")
    if SMOKE_CONFIG["disable_discussion"]:
        cmd.append("--disable-discussion")
    return cmd


def init_run(args: argparse.Namespace) -> int:
    run_name = args.run_name or f"game2_derisk_32_{timestamp_now()}"
    run_dir = RESULTS_ROOT / run_name
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")

    manifest = build_manifest(run_dir)
    api_indices = [idx for idx, item in enumerate(FINAL_32_MODELS) if not item["requires_gpu"]]
    gpu_indices = [idx for idx, item in enumerate(FINAL_32_MODELS) if item["requires_gpu"]]

    write_text(run_dir / "manifest.json", json.dumps(manifest, indent=2))
    write_text(run_dir / "slurm" / "run_api.sbatch", make_cpu_sbatch(run_dir))
    write_text(run_dir / "slurm" / "run_gpu.sbatch", make_gpu_sbatch(run_dir))
    write_text(
        run_dir / "slurm" / "submit_all.sh",
        make_submit_script(run_dir, api_indices, gpu_indices, args.api_max_concurrent),
    )
    os.chmod(run_dir / "slurm" / "run_api.sbatch", 0o755)
    os.chmod(run_dir / "slurm" / "run_gpu.sbatch", 0o755)
    os.chmod(run_dir / "slurm" / "submit_all.sh", 0o755)

    print(run_dir)
    return 0


def run_one(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    manifest = load_manifest(run_dir)
    models = manifest["models"]
    index = int(args.index)
    if index < 0 or index >= len(models):
        raise IndexError(f"Index {index} out of bounds for manifest with {len(models)} entries")

    entry = models[index]
    slug = slugify(entry["model"])
    output_dir = run_dir / "results" / f"{entry['rank']:02d}_{slug}"
    status_dir = run_dir / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    command_log = output_dir / "command.log"
    result_file = output_dir / "experiment_results.json"

    start = time.time()
    cmd = command_for_entry(entry, output_dir)
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=os.environ.copy(),
    )
    duration = time.time() - start

    combined_log = ""
    if completed.stdout:
        combined_log += completed.stdout
    if completed.stderr:
        if combined_log:
            combined_log += "\n"
        combined_log += completed.stderr
    command_log.write_text(combined_log, encoding="utf-8")

    runtime_success = completed.returncode == 0 and result_file.exists()
    audit = STATIC_AUDIT_NOTES.get(entry["model"])
    overall_success = runtime_success
    overall_note = "Runtime diplomacy smoke succeeded."
    if not runtime_success:
        overall_success = False
        overall_note = "Runtime diplomacy smoke failed."
    if audit:
        if audit["level"] == "fail":
            overall_success = False
            overall_note = audit["message"]
        elif overall_success:
            overall_note = f"{overall_note} {audit['message']}"

    status: Dict[str, Any] = {
        "rank": entry["rank"],
        "model": entry["model"],
        "elo": entry["elo"],
        "provider": entry["provider"],
        "requires_gpu": entry["requires_gpu"],
        "baseline_model": manifest["baseline_model"],
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "command": cmd,
        "returncode": completed.returncode,
        "runtime_success": runtime_success,
        "overall_success": overall_success,
        "duration_seconds": round(duration, 3),
        "note": overall_note,
        "command_log": str(command_log),
        "result_file": str(result_file) if result_file.exists() else None,
        "log_tail": combined_log.splitlines()[-40:],
        "audit": audit,
    }

    if result_file.exists():
        payload = json.loads(result_file.read_text(encoding="utf-8"))
        status["consensus_reached"] = payload.get("consensus_reached")
        status["final_round"] = payload.get("final_round")
        status["final_utilities"] = payload.get("final_utilities")

    status_path = status_dir / f"status_{entry['rank']:02d}_{slug}.json"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

    print(json.dumps(status, indent=2))
    return 0 if runtime_success else 1


def collect_status_map(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    status_dir = run_dir / "status"
    status_map: Dict[str, Dict[str, Any]] = {}
    if not status_dir.exists():
        return status_map
    for path in sorted(status_dir.glob("status_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        status_map[payload["model"]] = payload
    return status_map


def report_run(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    manifest = load_manifest(run_dir)
    status_map = collect_status_map(run_dir)
    report_rows = []
    counts = {"passed": 0, "failed": 0, "pending": 0}

    for entry in manifest["models"]:
        status = status_map.get(entry["model"])
        if status is None:
            row = {
                "rank": entry["rank"],
                "model": entry["model"],
                "elo": entry["elo"],
                "provider": entry["provider"],
                "status": "PENDING",
                "note": "No status file yet.",
            }
            counts["pending"] += 1
        elif status["overall_success"]:
            row = {
                "rank": entry["rank"],
                "model": entry["model"],
                "elo": entry["elo"],
                "provider": entry["provider"],
                "status": "PASS",
                "note": status["note"],
                "runtime_success": status["runtime_success"],
                "duration_seconds": status.get("duration_seconds"),
            }
            counts["passed"] += 1
        else:
            reason = status["note"]
            if not status["runtime_success"] and status.get("log_tail"):
                reason = f"{reason} Log tail: {' | '.join(status['log_tail'][-3:])}"
            row = {
                "rank": entry["rank"],
                "model": entry["model"],
                "elo": entry["elo"],
                "provider": entry["provider"],
                "status": "FAIL",
                "note": reason,
                "runtime_success": status["runtime_success"],
                "duration_seconds": status.get("duration_seconds"),
                "command_log": status.get("command_log"),
            }
            counts["failed"] += 1
        report_rows.append(row)

    report_payload = {
        "created_at": datetime.now().isoformat(),
        "run_dir": str(run_dir),
        "canonical_markdown": manifest["canonical_markdown"],
        "summary": counts,
        "models": report_rows,
    }
    write_text(run_dir / "report.json", json.dumps(report_payload, indent=2))

    lines = [
        f"# Game 2 De-risk Report ({run_dir.name})",
        "",
        f"- Canonical markdown: `{manifest['canonical_markdown']}`",
        f"- Passed: {counts['passed']}",
        f"- Failed: {counts['failed']}",
        f"- Pending: {counts['pending']}",
        "",
    ]
    for row in report_rows:
        lines.append(f"{row['rank']}. `{row['model']}` (Elo {row['elo']}): {row['status']}. {row['note']}")
    lines.append("")
    write_text(run_dir / "report.md", "\n".join(lines))

    print(json.dumps(report_payload, indent=2))
    return 0 if counts["failed"] == 0 and counts["pending"] == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create a new run directory and Slurm launch scripts")
    init_parser.add_argument("--run-name", type=str, default=None)
    init_parser.add_argument("--api-max-concurrent", type=int, default=4)
    init_parser.set_defaults(func=init_run)

    run_one_parser = subparsers.add_parser("run-one", help="Run a single manifest entry")
    run_one_parser.add_argument("--run-dir", type=str, required=True)
    run_one_parser.add_argument("--index", type=int, required=True)
    run_one_parser.set_defaults(func=run_one)

    report_parser = subparsers.add_parser("report", help="Aggregate status files into a report")
    report_parser.add_argument("--run-dir", type=str, required=True)
    report_parser.set_defaults(func=report_run)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Clone an archived 216-config TTC experiment at a new random seed."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_SOURCE_ROOT = (
    PROJECT_ROOT / "experiments" / "results" / "ttc_native_scaling_20260502_212943"
)
EXPECTED_CONFIGS = 216
ARCHIVED_SEED = 42
ARCHIVED_CAP = 10_500


def timestamp_now() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> Dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        status = run("status", "--porcelain")
        return {
            "commit": run("rev-parse", "HEAD"),
            "worktree_dirty": bool(status),
            "status_porcelain": status.splitlines(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "worktree_dirty": None, "status_porcelain": []}


def load_source_configs(source_root: Path) -> List[Dict[str, Any]]:
    config_paths = sorted((source_root / "configs").glob("config_*.json"))
    if len(config_paths) != EXPECTED_CONFIGS:
        raise RuntimeError(
            f"Expected {EXPECTED_CONFIGS} archived configs, found {len(config_paths)} "
            f"under {source_root / 'configs'}"
        )
    configs = [json.loads(path.read_text(encoding="utf-8")) for path in config_paths]
    actual_ids = [int(config["config_id"]) for config in configs]
    expected_ids = list(range(1, EXPECTED_CONFIGS + 1))
    if actual_ids != expected_ids:
        raise RuntimeError("Archived config IDs are not exactly 1..216 in filename order")
    return configs


def clone_config(
    source: Dict[str, Any],
    results_root: Path,
    seed: int,
) -> Dict[str, Any]:
    clone = dict(source)
    clone["random_seed"] = seed
    clone["seed_label"] = f"seed_{seed}"
    clone["preserve_config_max_tokens_per_phase"] = True
    clone["replication_source_seed"] = source["random_seed"]
    clone["replication_source_config_id"] = source["config_id"]
    clone["output_dir"] = str(
        results_root
        / source["target_model_family"]
        / f"level_{source['target_reasoning_level_requested']}"
        / source["game_cell_id"]
        / source["order"]
        / f"seed_{seed}"
    )
    return clone


def validate_configs(
    source_configs: Iterable[Dict[str, Any]],
    cloned_configs: Iterable[Dict[str, Any]],
    seed: int,
) -> None:
    source_list = list(source_configs)
    clone_list = list(cloned_configs)
    if len(source_list) != EXPECTED_CONFIGS or len(clone_list) != EXPECTED_CONFIGS:
        raise RuntimeError("Replication must contain exactly 216 configs")

    allowed_changed = {"random_seed", "seed_label", "output_dir"}
    allowed_added = {
        "preserve_config_max_tokens_per_phase",
        "replication_source_seed",
        "replication_source_config_id",
    }
    output_dirs = set()
    for source, clone in zip(source_list, clone_list):
        source_keys = set(source)
        clone_keys = set(clone)
        if clone_keys - source_keys != allowed_added:
            raise RuntimeError(
                f"Config {source['config_id']} has unexpected added fields: "
                f"{sorted((clone_keys - source_keys) ^ allowed_added)}"
            )
        changed = {
            key for key in source_keys if source.get(key) != clone.get(key)
        }
        if changed != allowed_changed:
            raise RuntimeError(
                f"Config {source['config_id']} changed unexpected fields: {sorted(changed)}"
            )
        if source["random_seed"] != ARCHIVED_SEED:
            raise RuntimeError(
                f"Config {source['config_id']} source seed is {source['random_seed']}, "
                f"expected {ARCHIVED_SEED}"
            )
        if clone["random_seed"] != seed:
            raise RuntimeError(f"Config {source['config_id']} seed was not rewritten")
        if int(clone["max_tokens_per_phase"]) != ARCHIVED_CAP:
            raise RuntimeError(
                f"Config {source['config_id']} cap is not archived cap {ARCHIVED_CAP}"
            )
        output_dir = clone["output_dir"]
        if output_dir in output_dirs:
            raise RuntimeError(f"Duplicate output directory: {output_dir}")
        output_dirs.add(output_dir)

    from strong_models_experiment.configs import STRONG_MODELS_CONFIG

    missing = sorted(
        {
            model
            for config in clone_list
            for model in config["models"]
            if model not in STRONG_MODELS_CONFIG
        }
    )
    if missing:
        raise RuntimeError(f"Missing model configs: {missing}")


def write_slurm_script(results_root: Path, seed: int) -> Path:
    slurm_dir = results_root / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    script_path = slurm_dir / "run_one.sbatch"
    script_path.write_text(
        f"""#!/bin/bash
#SBATCH --job-name=ttc_seed{seed}
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
RUN_DIR="{results_root}"
cd "$BASE_DIR"
mkdir -p "$RUN_DIR/slurm/logs" "$RUN_DIR/monitoring"

echo "============================================================"
echo "TTC native seed replication"
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

export OPENROUTER_TRANSPORT="${{OPENROUTER_TRANSPORT:-auto}}"
export OPENROUTER_PROXY_POLL_DIR="${{OPENROUTER_PROXY_POLL_DIR:-/home/jz4391/openrouter_proxy}}"
export OPENROUTER_PROXY_CLIENT_TIMEOUT="${{OPENROUTER_PROXY_CLIENT_TIMEOUT:-9000}}"
export OPENROUTER_API_TIMEOUT="${{OPENROUTER_API_TIMEOUT:-1800}}"
export LLM_FAILURE_REPORT_PATH="${{LLM_FAILURE_REPORT_PATH:-$RUN_DIR/monitoring/provider_failures.md}}"
export PYTHONUNBUFFERED=1

CONFIG_FILE="${{1:?config path is required}}"
"$BASE_DIR/.venv/bin/python" "$BASE_DIR/scripts/run_ttc_native_config.py" --config "$CONFIG_FILE"

echo "Finished=$(date)"
""",
        encoding="utf-8",
    )
    make_executable(script_path)
    return script_path


def write_submit_script(results_root: Path, slurm_script: Path, seed: int) -> Path:
    submit_path = results_root / "slurm" / "submit.sh"
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

if [[ ! -f "$SUBMITTED" ]]; then
  printf "submitted_at\\tconfig_id\\tjob_id\\tconfig_file\\n" > "$SUBMITTED"
fi

if (( $# == 0 )); then
  mapfile -t requested_ids < <(seq 1 {EXPECTED_CONFIGS})
else
  requested_ids=("$@")
fi

count=0
for raw_id in "${{requested_ids[@]}}"; do
  if [[ ! "$raw_id" =~ ^[0-9]+$ ]] || (( raw_id < 1 || raw_id > {EXPECTED_CONFIGS} )); then
    echo "Invalid config ID: $raw_id" >&2
    exit 2
  fi
  printf -v config_file "%s/config_%04d.json" "$CONFIG_DIR" "$raw_id"
  printf -v config_tag "%04d" "$raw_id"
  job_name="ttc{seed}_${{config_tag}}"
  job_id="$(sbatch --parsable --job-name="$job_name" "$SBATCH_SCRIPT" "$config_file")"
  printf "%s\\t%s\\t%s\\t%s\\n" "$(date --iso-8601=seconds)" "$raw_id" "$job_id" "$config_file" >> "$SUBMITTED"
  count=$((count + 1))
  echo "Submitted config $raw_id as job $job_id"
done

echo "Submitted $count jobs"
echo "Submission manifest: $SUBMITTED"
""",
        encoding="utf-8",
    )
    make_executable(submit_path)
    return submit_path


def write_manifest(
    source_root: Path,
    results_root: Path,
    source_configs: List[Dict[str, Any]],
    cloned_configs: List[Dict[str, Any]],
    seed: int,
) -> None:
    source_hashes = {
        f"config_{config['config_id']:04d}.json": file_sha256(
            source_root / "configs" / f"config_{config['config_id']:04d}.json"
        )
        for config in source_configs
    }
    rows = [
        {
            "config_id": config["config_id"],
            "target_provider": config["target_provider"],
            "target_model": config["target_model"],
            "reasoning_level": config["target_reasoning_level_requested"],
            "game_cell_id": config["game_cell_id"],
            "order": config["order"],
            "output_dir": config["output_dir"],
        }
        for config in cloned_configs
    ]
    write_json(
        results_root / "manifest.json",
        {
            "experiment_name": "ttc_native_scaling_seed_replication",
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
            "num_configs": len(cloned_configs),
            "source_results_root": str(source_root),
            "source_seed": ARCHIVED_SEED,
            "replication_seed": seed,
            "max_tokens_per_phase": ARCHIVED_CAP,
            "cap_policy": (
                "Start every config at the archived 10500-token cap. Only a documented "
                "cap-related hard failure may be recovered at 16384 tokens."
            ),
            "config_equivalence_policy": (
                "Archived configs are byte-parsed and cloned. Only random_seed, "
                "seed_label, output_dir, and replication/cap-control metadata differ."
            ),
            "temporal_alias_note": (
                "GPT-5 is pinned to gpt-5-2025-08-07. Claude Sonnet 4.6, "
                "Gemini 3 Flash Preview, and gpt-5-nano use the archived aliases and "
                "can be affected by provider-side alias drift."
            ),
            "source_config_sha256": source_hashes,
            "code_state": git_state(),
            "configs": rows,
        },
    )


def generate(source_root: Path, results_root: Path, seed: int) -> List[Dict[str, Any]]:
    if results_root.exists() and any(results_root.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty results root: {results_root}")
    source_configs = load_source_configs(source_root)
    cloned_configs = [
        clone_config(source, results_root, seed) for source in source_configs
    ]
    validate_configs(source_configs, cloned_configs, seed)

    configs_dir = results_root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    for config in cloned_configs:
        write_json(configs_dir / f"config_{config['config_id']:04d}.json", config)
    slurm_script = write_slurm_script(results_root, seed)
    write_submit_script(results_root, slurm_script, seed)
    write_manifest(source_root, results_root, source_configs, cloned_configs, seed)
    return cloned_configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=984)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    results_root = (
        args.results_root.resolve()
        if args.results_root
        else (
            PROJECT_ROOT
            / "experiments"
            / "results"
            / f"ttc_native_scaling_seed{args.seed}_{timestamp_now()}"
        ).resolve()
    )
    source_configs = load_source_configs(source_root)
    cloned_configs = [
        clone_config(source, results_root, args.seed) for source in source_configs
    ]
    validate_configs(source_configs, cloned_configs, args.seed)
    print(f"Source root: {source_root}")
    print(f"Results root: {results_root}")
    print(f"Seed: {args.seed}")
    print(f"Configs: {len(cloned_configs)}")
    if args.dry_run:
        print(json.dumps(cloned_configs[:2], indent=2, sort_keys=True))
        return 0
    generate(source_root, results_root, args.seed)
    print(f"Wrote {len(cloned_configs)} configs")
    print(f"Submit script: {results_root / 'slurm' / 'submit.sh'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

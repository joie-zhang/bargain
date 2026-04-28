#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import random
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strong_models_experiment.analysis.active_model_roster import (  # noqa: E402
    ACTIVE_ADVERSARY_MODELS,
    active_model_elo_map,
    canonical_model_name,
    elo_for_model,
    short_model_name,
    tier_from_elo,
)
from strong_models_experiment.analysis.game1_multiagent_smoke import (  # noqa: E402
    write_analysis_outputs,
)


DEFAULT_BASELINE_MODEL = "gpt-5-nano"
DEFAULT_FOCAL_MODELS = list(ACTIVE_ADVERSARY_MODELS)
DEFAULT_COMPETITION_LEVELS = [0.0, 0.5, 0.9, 1.0]
DEFAULT_N_VALUES = [3, 5]
DEFAULT_PROPOSAL1_REPS = 3
DEFAULT_PROPOSAL2_FIELDS = 10
DEFAULT_DISCUSSION_TURNS = 2
DEFAULT_NUM_ITEMS = 5
DEFAULT_MAX_ROUNDS = 10
DEFAULT_GAMMA_DISCOUNT = 0.9
DEFAULT_BASE_SEED = 4242
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_MAX_CONCURRENT = 32
DEFAULT_SLURM_TIME = "08:00:00"
JOB_NAME = "g1mfull"
MAX_ARRAY_CHUNK_SIZE = 2000

# Proposal 2 uses a broad tiered ecology, but excludes the known context-risk
# low-end OpenRouter routes that already failed in the n>2 smoke batch.
ECOLOGY_EXCLUSIONS = {
    "llama-3.2-1b-instruct",
    "llama-3.2-3b-instruct",
    "llama-3.1-8b-instruct",
    "qwen2.5-72b-instruct",
}

NON_RETRYABLE_PATTERNS = (
    "maximum context length",
    "prompt length",
    "should not exceed 32768",
    "should not exceed 60000",
    "context window",
    "invalid_request_error",
    "400 bad request",
    "could not parse the json body",
)

RETRYABLE_PATTERNS = (
    "429",
    "rate limit",
    "rate-limit",
    "timed out",
    "timeout",
    "temporarily unavailable",
    "service unavailable",
    "overloaded",
    "connection error",
    "internal server error",
    "500",
    "502",
    "503",
    "504",
    "proxy error",
    "readtimeout",
    "apiconnectionerror",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate, submit, monitor, and analyze the full Game 1 multi-agent Proposal 1/2 batch."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--results-root", type=str, default=None)
    generate.add_argument("--baseline-model", type=str, default=DEFAULT_BASELINE_MODEL)
    generate.add_argument("--proposal1-reps", type=int, default=DEFAULT_PROPOSAL1_REPS)
    generate.add_argument("--proposal2-fields", type=int, default=DEFAULT_PROPOSAL2_FIELDS)
    generate.add_argument("--discussion-turns", type=int, default=DEFAULT_DISCUSSION_TURNS)
    generate.add_argument("--num-items", type=int, default=DEFAULT_NUM_ITEMS)
    generate.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    generate.add_argument("--gamma-discount", type=float, default=DEFAULT_GAMMA_DISCOUNT)
    generate.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    generate.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    generate.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    generate.add_argument("--slurm-time", type=str, default=DEFAULT_SLURM_TIME)

    run_one = subparsers.add_parser("run-one")
    run_one.add_argument("--results-root", type=str, required=True)
    run_one.add_argument("--config-id", type=int, required=True)
    run_one.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)

    submit_pending = subparsers.add_parser("submit-pending")
    submit_pending.add_argument("--results-root", type=str, required=True)
    submit_pending.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    submit_pending.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    submit_pending.add_argument("--rerun-failed", action="store_true")

    summary = subparsers.add_parser("summary")
    summary.add_argument("--results-root", type=str, required=True)
    summary.add_argument("--json", action="store_true")

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--results-root", type=str, required=True)

    return parser.parse_args()


def sanitize_float(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".").replace("-", "n").replace(".", "p")


def latest_root_or_new(raw_value: Optional[str]) -> Path:
    if raw_value:
        path = Path(raw_value)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        return path
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return (PROJECT_ROOT / "experiments" / "results" / f"game1_multiagent_full_{timestamp}").resolve()


def resolve_results_root(raw_value: str) -> Path:
    path = Path(raw_value)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def create_latest_symlink(results_root: Path) -> None:
    symlink = results_root.parent / "game1_multiagent_full_latest"
    if symlink.is_symlink() or symlink.exists():
        symlink.unlink()
    symlink.symlink_to(results_root.name)


def build_ecology_pools() -> Dict[str, List[str]]:
    elo_map = active_model_elo_map()
    pools = {"weak": [], "medium": [], "strong": []}
    for model_name in ACTIVE_ADVERSARY_MODELS:
        canonical = canonical_model_name(model_name)
        if canonical in ECOLOGY_EXCLUSIONS:
            continue
        elo = elo_map[canonical]
        tier = tier_from_elo(elo).lower()
        pools[tier].append(canonical)
    return pools


def build_field_templates(n_agents: int) -> List[List[str]]:
    if n_agents == 3:
        return [
            ["weak", "medium"],
            ["weak", "strong"],
            ["medium", "strong"],
            ["weak", "medium"],
            ["medium", "strong"],
        ]
    if n_agents == 5:
        return [
            ["weak", "weak", "medium", "strong"],
            ["weak", "medium", "medium", "strong"],
            ["weak", "medium", "strong", "strong"],
            ["weak", "weak", "strong", "strong"],
            ["medium", "medium", "strong", "strong"],
        ]
    raise ValueError(f"Unsupported n_agents={n_agents}")


def build_mixed_fields(
    n_agents: int,
    num_fields: int,
    ecology_pools: Dict[str, Sequence[str]],
    seed: int,
) -> List[Dict]:
    rng = random.Random(seed + n_agents * 7919)
    templates = build_field_templates(n_agents)
    fields: List[Dict] = []
    for field_idx in range(num_fields):
        template = templates[field_idx % len(templates)]
        opponents: List[str] = []
        for tier in ("weak", "medium", "strong"):
            count = sum(1 for item in template if item == tier)
            if count == 0:
                continue
            pool = list(ecology_pools[tier])
            if count > len(pool):
                raise ValueError(f"Tier {tier} has only {len(pool)} models; need {count} for n={n_agents}")
            sampled = rng.sample(pool, count)
            opponents.extend(sampled)
        rng.shuffle(opponents)
        fields.append(
            {
                "field_id": field_idx + 1,
                "template": template,
                "opponents": opponents,
            }
        )
    return fields


def build_manifest(args: argparse.Namespace, results_root: Path) -> Dict:
    ecology_pools = build_ecology_pools()
    ecology_fields = {
        str(n_agents): build_mixed_fields(
            n_agents=n_agents,
            num_fields=args.proposal2_fields,
            ecology_pools=ecology_pools,
            seed=args.base_seed,
        )
        for n_agents in DEFAULT_N_VALUES
    }
    return {
        "results_root": str(results_root),
        "game_type": "item_allocation",
        "baseline_model": args.baseline_model,
        "focal_models": DEFAULT_FOCAL_MODELS,
        "competition_levels": DEFAULT_COMPETITION_LEVELS,
        "n_values": DEFAULT_N_VALUES,
        "proposal1_reps": int(args.proposal1_reps),
        "proposal2_fields": int(args.proposal2_fields),
        "discussion_turns": int(args.discussion_turns),
        "num_items": int(args.num_items),
        "max_rounds": int(args.max_rounds),
        "gamma_discount": float(args.gamma_discount),
        "base_seed": int(args.base_seed),
        "max_attempts": int(args.max_attempts),
        "max_concurrent": int(args.max_concurrent),
        "slurm_time": str(args.slurm_time),
        "ecology_exclusions": sorted(ECOLOGY_EXCLUSIONS),
        "ecology_pools": ecology_pools,
        "ecology_fields": ecology_fields,
        "notes": (
            "Full Game 1 multi-agent batch for Proposal 1 and Proposal 2. "
            "Proposal 2 ecology excludes the known context-risk low-end OpenRouter routes."
        ),
    }


def pairing_id(parts: Iterable[object]) -> str:
    return "|".join(str(part) for part in parts)


def make_config_record(
    *,
    config_id: int,
    proposal_family: str,
    condition_role: str,
    pairing: str,
    focal_model: str,
    baseline_model: str,
    models: List[str],
    n_agents: int,
    competition_level: float,
    random_seed: int,
    replicate_id: Optional[int],
    field_id: Optional[int],
    ecology_models: List[str],
    results_root: Path,
    manifest: Dict,
) -> Dict:
    comp_token = sanitize_float(competition_level)
    run_token = f"rep_{replicate_id:02d}" if replicate_id is not None else f"field_{field_id:02d}"
    output_dir = (
        results_root
        / proposal_family
        / f"focal_{focal_model}"
        / f"n_{n_agents}"
        / f"comp_{comp_token}"
        / run_token
        / condition_role
    )
    return {
        "config_id": config_id,
        "proposal_family": proposal_family,
        "condition_role": condition_role,
        "pairing_id": pairing,
        "game_type": "item_allocation",
        "focal_model": focal_model,
        "baseline_model": baseline_model,
        "models": models,
        "ecology_models": ecology_models,
        "n_agents": n_agents,
        "competition_level": competition_level,
        "num_items": manifest["num_items"],
        "max_rounds": manifest["max_rounds"],
        "gamma_discount": manifest["gamma_discount"],
        "discussion_turns": manifest["discussion_turns"],
        "random_seed": random_seed,
        "replicate_id": replicate_id,
        "field_id": field_id,
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
    }


def build_config_records(manifest: Dict, results_root: Path) -> List[Dict]:
    configs: List[Dict] = []
    config_id = 1
    baseline = manifest["baseline_model"]

    for focal_model in manifest["focal_models"]:
        for n_agents in manifest["n_values"]:
            for competition_level in manifest["competition_levels"]:
                for replicate_id in range(1, manifest["proposal1_reps"] + 1):
                    seed = manifest["base_seed"] + config_id * 13
                    pair_id = pairing_id(
                        [
                            "proposal1",
                            focal_model,
                            f"n{n_agents}",
                            f"comp{competition_level}",
                            f"rep{replicate_id}",
                        ]
                    )
                    configs.extend(
                        [
                            make_config_record(
                                config_id=config_id,
                                proposal_family="proposal1_invasion",
                                condition_role="focal",
                                pairing=pair_id,
                                focal_model=focal_model,
                                baseline_model=baseline,
                                models=[focal_model] + [baseline] * (n_agents - 1),
                                n_agents=n_agents,
                                competition_level=competition_level,
                                random_seed=seed,
                                replicate_id=replicate_id,
                                field_id=None,
                                ecology_models=[],
                                results_root=results_root,
                                manifest=manifest,
                            ),
                            make_config_record(
                                config_id=config_id + 1,
                                proposal_family="proposal1_invasion",
                                condition_role="control",
                                pairing=pair_id,
                                focal_model=focal_model,
                                baseline_model=baseline,
                                models=[baseline] * n_agents,
                                n_agents=n_agents,
                                competition_level=competition_level,
                                random_seed=seed,
                                replicate_id=replicate_id,
                                field_id=None,
                                ecology_models=[],
                                results_root=results_root,
                                manifest=manifest,
                            ),
                        ]
                    )
                    config_id += 2

    for focal_model in manifest["focal_models"]:
        for n_agents in manifest["n_values"]:
            ecology_fields = manifest["ecology_fields"][str(n_agents)]
            for competition_level in manifest["competition_levels"]:
                for field in ecology_fields:
                    seed = manifest["base_seed"] + config_id * 13
                    pair_id = pairing_id(
                        [
                            "proposal2",
                            focal_model,
                            f"n{n_agents}",
                            f"comp{competition_level}",
                            f"field{field['field_id']}",
                        ]
                    )
                    opponents = list(field["opponents"])
                    configs.extend(
                        [
                            make_config_record(
                                config_id=config_id,
                                proposal_family="proposal2_mixed_ecology",
                                condition_role="focal",
                                pairing=pair_id,
                                focal_model=focal_model,
                                baseline_model=baseline,
                                models=[focal_model] + opponents,
                                n_agents=n_agents,
                                competition_level=competition_level,
                                random_seed=seed,
                                replicate_id=None,
                                field_id=field["field_id"],
                                ecology_models=opponents,
                                results_root=results_root,
                                manifest=manifest,
                            ),
                            make_config_record(
                                config_id=config_id + 1,
                                proposal_family="proposal2_mixed_ecology",
                                condition_role="control",
                                pairing=pair_id,
                                focal_model=focal_model,
                                baseline_model=baseline,
                                models=[baseline] + opponents,
                                n_agents=n_agents,
                                competition_level=competition_level,
                                random_seed=seed,
                                replicate_id=None,
                                field_id=field["field_id"],
                                ecology_models=opponents,
                                results_root=results_root,
                                manifest=manifest,
                            ),
                        ]
                    )
                    config_id += 2

    return configs


def write_generated_files(results_root: Path, manifest: Dict, configs: List[Dict]) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    config_dir = results_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    write_json(results_root / "manifest.json", manifest)

    for config in configs:
        config_path = config_dir / f"config_{config['config_id']:04d}.json"
        write_json(config_path, config)

    all_configs = config_dir / "all_configs.txt"
    all_config_lines = [str(config_dir / f"config_{cfg['config_id']:04d}.json") for cfg in configs]
    all_configs.write_text("\n".join(all_config_lines) + "\n", encoding="utf-8")

    index_path = config_dir / "experiment_index.csv"
    fieldnames = [
        "config_id",
        "proposal_family",
        "condition_role",
        "pairing_id",
        "focal_model",
        "baseline_model",
        "n_agents",
        "competition_level",
        "replicate_id",
        "field_id",
        "random_seed",
        "models",
        "ecology_models",
        "output_dir",
        "config_file",
    ]
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for config in configs:
            row = {key: config.get(key) for key in fieldnames if key not in {"models", "ecology_models", "config_file"}}
            row["models"] = "+".join(config["models"])
            row["ecology_models"] = "+".join(config["ecology_models"])
            row["config_file"] = f"config_{config['config_id']:04d}.json"
            writer.writerow(row)


def build_command(config: Dict) -> List[str]:
    output_dir = (PROJECT_ROOT / config["output_dir"]).resolve()
    return [
        sys.executable,
        "run_strong_models_experiment.py",
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
        "weak_first",
        "--random-seed",
        str(config["random_seed"]),
        "--output-dir",
        str(output_dir),
        "--job-id",
        str(config["config_id"]),
        "--max-tokens-voting",
        str(config.get("max_tokens_voting", 768)),
    ]


def result_file_for_output(output_dir: Path) -> Optional[Path]:
    candidates = [
        output_dir / "experiment_results.json",
        output_dir / "run_1_experiment_results.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def config_result_exists(config: Dict) -> bool:
    output_dir = (PROJECT_ROOT / config["output_dir"]).resolve()
    return result_file_for_output(output_dir) is not None


def status_path_for(results_root: Path, config_id: int) -> Path:
    return results_root / "status" / f"config_{config_id:04d}.json"


def log_path_for(results_root: Path, config_id: int) -> Path:
    return results_root / "logs" / f"config_{config_id:04d}.log"


def load_configs(results_root: Path) -> List[Dict]:
    rows = []
    for path in sorted((results_root / "configs").glob("config_*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def load_config(results_root: Path, config_id: int) -> Dict:
    path = results_root / "configs" / f"config_{config_id:04d}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_status(results_root: Path, config_id: int) -> Optional[Dict]:
    path = status_path_for(results_root, config_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_status(results_root: Path, config_id: int, payload: Dict) -> None:
    payload["config_id"] = config_id
    payload["updated_at"] = dt.datetime.now().isoformat()
    write_json(status_path_for(results_root, config_id), payload)


def classify_failure(log_text: str) -> Tuple[str, bool]:
    lowered = log_text.lower()
    if any(pattern in lowered for pattern in NON_RETRYABLE_PATTERNS):
        return ("non_retryable_context_or_request_failure", False)
    if any(pattern in lowered for pattern in RETRYABLE_PATTERNS):
        return ("retryable_provider_or_network_failure", True)
    if "traceback (most recent call last):" in lowered:
        return ("python_traceback", False)
    return ("unknown_failure", True)


def active_array_tasks() -> Dict[int, Dict[str, str]]:
    try:
        output = subprocess.check_output(
            ["squeue", "-h", "-u", os.getenv("USER", ""), "-o", "%A|%a|%T|%j"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}

    active: Dict[int, Dict[str, str]] = {}
    for line in output.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 4:
            continue
        job_id, array_id, state, job_name = parts
        if job_name != JOB_NAME:
            continue
        try:
            config_id = int(array_id)
        except ValueError:
            continue
        active[config_id] = {"job_id": job_id, "state": state}
    return active


def summarize_progress(results_root: Path, max_attempts: int) -> Dict:
    configs = load_configs(results_root)
    summary = {
        "total": len(configs),
        "successful": 0,
        "running": 0,
        "pending_in_queue": 0,
        "skipped": 0,
        "failed_retryable": 0,
        "failed_terminal": 0,
        "not_started": 0,
        "unfinished": 0,
        "pair_total": 0,
        "pair_successful": 0,
        "pair_skipped": 0,
        "pair_incomplete": 0,
    }

    status_by_pair: Dict[str, List[str]] = {}

    for config in configs:
        config_id = int(config["config_id"])
        if config_result_exists(config):
            state = "SUCCESS"
            summary["successful"] += 1
        else:
            status_payload = read_status(results_root, config_id)
            if status_payload is None:
                state = "NOT_STARTED"
                summary["not_started"] += 1
            else:
                status_state = status_payload.get("state")
                attempts = int(status_payload.get("attempts", 0))
                retryable = bool(status_payload.get("retryable", False))
                if status_state == "SUCCESS":
                    state = "SUCCESS"
                    summary["successful"] += 1
                elif status_state == "RUNNING":
                    state = "RUNNING"
                    summary["running"] += 1
                elif status_state == "SUBMITTED":
                    state = "PENDING"
                    summary["pending_in_queue"] += 1
                elif status_state == "SKIPPED":
                    state = "SKIPPED"
                    summary["skipped"] += 1
                elif attempts >= max_attempts or not retryable:
                    state = "FAILED_TERMINAL"
                    summary["failed_terminal"] += 1
                else:
                    state = "FAILED_RETRYABLE"
                    summary["failed_retryable"] += 1

        if state not in {"SUCCESS", "SKIPPED"}:
            summary["unfinished"] += 1
        status_by_pair.setdefault(config["pairing_id"], []).append(state)

    summary["pair_total"] = len(status_by_pair)
    for states in status_by_pair.values():
        if len(states) == 2 and all(state == "SUCCESS" for state in states):
            summary["pair_successful"] += 1
        elif len(states) == 2 and all(state == "SKIPPED" for state in states):
            summary["pair_skipped"] += 1
        else:
            summary["pair_incomplete"] += 1
    return summary


def select_resubmittable_config_ids(results_root: Path, max_attempts: int, rerun_failed: bool) -> List[int]:
    configs = load_configs(results_root)
    config_ids: List[int] = []
    for config in configs:
        config_id = int(config["config_id"])
        if config_result_exists(config):
            continue
        status_payload = read_status(results_root, config_id)
        if status_payload is None:
            config_ids.append(config_id)
            continue
        attempts = int(status_payload.get("attempts", 0))
        retryable = bool(status_payload.get("retryable", False))
        state = status_payload.get("state")
        if attempts >= max_attempts:
            continue
        if state == "SUCCESS":
            continue
        if state == "SKIPPED":
            continue
        if state in {"RUNNING", "SUBMITTED"}:
            continue
        if rerun_failed and retryable:
            config_ids.append(config_id)
    return config_ids


def chunk_config_ids(ids: Sequence[int], chunk_size: int = MAX_ARRAY_CHUNK_SIZE) -> List[List[int]]:
    ordered = sorted(set(int(item) for item in ids))
    return [ordered[idx : idx + chunk_size] for idx in range(0, len(ordered), chunk_size)]


def chunk_array(ids: Sequence[int]) -> str:
    if not ids:
        return ""
    sorted_ids = sorted(set(int(item) for item in ids))
    ranges: List[str] = []
    start = sorted_ids[0]
    prev = start
    for item in sorted_ids[1:]:
        if item == prev + 1:
            prev = item
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = item
        prev = item
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def write_slurm_files(results_root: Path, manifest: Dict) -> None:
    slurm_dir = results_root / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    python_bin = PROJECT_ROOT / ".venv" / "bin" / "python"
    script_path = PROJECT_ROOT / "scripts" / "game1_multiagent_full_batch.py"

    sbatch = f"""#!/bin/bash
#SBATCH --job-name={JOB_NAME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time={manifest['slurm_time']}
#SBATCH --partition=cpu
#SBATCH --requeue
#SBATCH --output={PROJECT_ROOT}/slurm/game1_multiagent_full_%A_%a.out
#SBATCH --error={PROJECT_ROOT}/slurm/game1_multiagent_full_%A_%a.err

set -eo pipefail

BASE_DIR="{PROJECT_ROOT}"
mkdir -p "{PROJECT_ROOT / 'slurm'}"
cd "$BASE_DIR"

export PS1="${{PS1:-}}"

module purge
module load anaconda3/2024.2
module load proxy/default

KEY_ENV_FILE="${{BARGAIN_API_KEYS_ENV:-/home/jz4391/.config/bargain/api_keys.env}}"
if [[ -f "$KEY_ENV_FILE" ]]; then
  set -a
  source "$KEY_ENV_FILE"
  set +a
fi

: "${{RUN_DIR:?RUN_DIR is required}}"
: "${{SUBMISSION_FILE:?SUBMISSION_FILE is required}}"

export OPENROUTER_TRANSPORT="${{OPENROUTER_TRANSPORT:-proxy}}"
export OPENROUTER_PROXY_POLL_DIR="${{OPENROUTER_PROXY_POLL_DIR:-/home/jz4391/openrouter_proxy}}"
export OPENROUTER_PROXY_CLIENT_TIMEOUT="${{OPENROUTER_PROXY_CLIENT_TIMEOUT:-9000}}"
export LLM_FAILURE_REPORT_PATH="${{LLM_FAILURE_REPORT_PATH:-$RUN_DIR/monitoring/provider_failures.md}}"
export PYTHONUNBUFFERED=1

CONFIG_ID=$("{python_bin}" - <<'PY'
import json
import os
from pathlib import Path

submission_file = Path(os.environ["SUBMISSION_FILE"])
task_index = int(os.environ["SLURM_ARRAY_TASK_ID"]) - 1
payload = json.loads(submission_file.read_text(encoding="utf-8"))
config_ids = payload["config_ids"]
if task_index < 0 or task_index >= len(config_ids):
    raise SystemExit(f"Invalid task index {{task_index + 1}} for {{submission_file}}")
print(config_ids[task_index])
PY
)

"{python_bin}" "{script_path}" run-one --results-root "$RUN_DIR" --config-id "$CONFIG_ID" --max-attempts {manifest['max_attempts']}
"""
    (slurm_dir / "run_api.sbatch").write_text(sbatch, encoding="utf-8")
    os.chmod(slurm_dir / "run_api.sbatch", 0o755)

    submit = f"""#!/bin/bash
set -euo pipefail

BASE_DIR="{PROJECT_ROOT}"
RUN_DIR="{results_root}"
cd "$BASE_DIR"

"{python_bin}" "{script_path}" submit-pending --results-root "$RUN_DIR" --max-concurrent {manifest['max_concurrent']} --max-attempts {manifest['max_attempts']} --rerun-failed
"""
    (slurm_dir / "submit_pending.sh").write_text(submit, encoding="utf-8")
    os.chmod(slurm_dir / "submit_pending.sh", 0o755)

    supervise = f"""#!/bin/bash
set -euo pipefail

BASE_DIR="{PROJECT_ROOT}"
RUN_DIR="{results_root}"
PYTHON_BIN="{python_bin}"
SCRIPT="{script_path}"
cd "$BASE_DIR"

mkdir -p "$RUN_DIR/monitoring"

while true; do
    echo "============================================================" | tee -a "$RUN_DIR/monitoring/supervisor.log"
    echo "Supervisor tick: $(date)" | tee -a "$RUN_DIR/monitoring/supervisor.log"
    "$PYTHON_BIN" "$SCRIPT" summary --results-root "$RUN_DIR" --json | tee "$RUN_DIR/monitoring/latest_summary.json" | tee -a "$RUN_DIR/monitoring/supervisor.log"
    "$PYTHON_BIN" "$SCRIPT" analyze --results-root "$RUN_DIR" >> "$RUN_DIR/monitoring/supervisor.log" 2>&1 || true

    unfinished=$("$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
path = Path("{results_root}") / "monitoring" / "latest_summary.json"
payload = json.loads(path.read_text(encoding="utf-8"))
print(payload["unfinished"])
PY
)
    active=$("$PYTHON_BIN" - <<'PY'
import subprocess
import os
try:
    out = subprocess.check_output(["squeue", "-h", "-u", os.getenv("USER", ""), "-o", "%j"], text=True, stderr=subprocess.DEVNULL)
except Exception:
    print(0)
    raise SystemExit(0)
count = sum(1 for line in out.splitlines() if line.strip() == "{JOB_NAME}")
print(count)
PY
)
    echo "Active {JOB_NAME} jobs: $active" | tee -a "$RUN_DIR/monitoring/supervisor.log"

    if [[ "$unfinished" -eq 0 ]]; then
        echo "Batch complete at $(date)" | tee -a "$RUN_DIR/monitoring/supervisor.log"
        exit 0
    fi

    if [[ "$active" -eq 0 ]]; then
        echo "No active Slurm arrays; submitting pending work." | tee -a "$RUN_DIR/monitoring/supervisor.log"
        "$PYTHON_BIN" "$SCRIPT" submit-pending --results-root "$RUN_DIR" --max-concurrent {manifest['max_concurrent']} --max-attempts {manifest['max_attempts']} --rerun-failed | tee -a "$RUN_DIR/monitoring/supervisor.log"
    fi

    sleep 1200
done
"""
    (slurm_dir / "supervise.sh").write_text(supervise, encoding="utf-8")
    os.chmod(slurm_dir / "supervise.sh", 0o755)


def cmd_generate(args: argparse.Namespace) -> int:
    results_root = latest_root_or_new(args.results_root)
    manifest = build_manifest(args, results_root)
    configs = build_config_records(manifest, results_root)
    write_generated_files(results_root, manifest, configs)
    write_slurm_files(results_root, manifest)
    create_latest_symlink(results_root)
    print(results_root)
    print(f"Generated {len(configs)} single-run configs")
    return 0


def cmd_run_one(args: argparse.Namespace) -> int:
    results_root = resolve_results_root(args.results_root)
    config = load_config(results_root, args.config_id)
    output_dir = (PROJECT_ROOT / config["output_dir"]).resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path = log_path_for(results_root, int(config["config_id"]))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    existing_status = read_status(results_root, int(config["config_id"])) or {}
    attempts_completed = int(existing_status.get("attempts", 0))
    if config_result_exists(config):
        write_status(
            results_root,
            int(config["config_id"]),
            {
                "state": "SUCCESS",
                "attempts": attempts_completed,
                "retryable": False,
                "returncode": 0,
            },
        )
        return 0

    cmd = build_command(config)
    retryable = True
    returncode = 1
    failure_reason = "not_run"

    for attempt in range(attempts_completed + 1, args.max_attempts + 1):
        write_status(
            results_root,
            int(config["config_id"]),
            {
                "state": "RUNNING",
                "attempts": attempt - 1,
                "retryable": True,
                "command": shlex.join(cmd),
                "slurm_job_id": os.getenv("SLURM_JOB_ID"),
                "slurm_array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"),
                "current_attempt": attempt,
            },
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n" + "=" * 72 + "\n")
            handle.write(f"Attempt {attempt}/{args.max_attempts} started at {dt.datetime.now().isoformat()}\n")
            handle.write(f"Command: {shlex.join(cmd)}\n")
            handle.write("=" * 72 + "\n")
            handle.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                env={
                    **os.environ,
                    "PYTHONUNBUFFERED": "1",
                    "LLM_FAILURE_REPORT_PATH": str(results_root / "monitoring" / "provider_failures.md"),
                },
                text=True,
            )
            returncode = proc.wait()
            handle.write(f"\nAttempt {attempt} returncode: {returncode}\n")

        if config_result_exists(config):
            write_status(
                results_root,
                int(config["config_id"]),
                {
                    "state": "SUCCESS",
                    "attempts": attempt,
                    "retryable": False,
                    "returncode": returncode,
                    "command": shlex.join(cmd),
                },
            )
            return 0

        log_text = log_path.read_text(encoding="utf-8", errors="ignore")
        failure_reason, retryable = classify_failure(log_text[-25000:])
        write_status(
            results_root,
            int(config["config_id"]),
            {
                "state": "FAILED",
                "attempts": attempt,
                "retryable": retryable,
                "returncode": returncode,
                "failure_reason": failure_reason,
                "command": shlex.join(cmd),
            },
        )
        if not retryable:
            break
        if attempt < args.max_attempts:
            time.sleep(30 * attempt)

    return 0 if config_result_exists(config) else max(returncode, 1)


def cmd_submit_pending(args: argparse.Namespace) -> int:
    results_root = resolve_results_root(args.results_root)
    config_ids = select_resubmittable_config_ids(
        results_root=results_root,
        max_attempts=args.max_attempts,
        rerun_failed=args.rerun_failed,
    )
    if not config_ids:
        print("No resubmittable configs found.")
        return 0

    sbatch_path = results_root / "slurm" / "run_api.sbatch"
    if not sbatch_path.exists():
        raise FileNotFoundError(f"Missing sbatch file: {sbatch_path}")
    submissions_dir = results_root / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)

    chunked = chunk_config_ids(config_ids)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    submitted_total = 0
    chunk = chunked[0]
    chunk_idx = 1
    payload = {
        "created_at": dt.datetime.now().isoformat(),
        "results_root": str(results_root),
        "config_ids": chunk,
    }
    submission_file = submissions_dir / f"submission_{stamp}_{chunk_idx:02d}.json"
    write_json(submission_file, payload)
    array_range = f"1-{len(chunk)}" if len(chunk) > 1 else "1"
    array_spec = f"{array_range}%{args.max_concurrent}"
    out = subprocess.check_output(
        [
            "sbatch",
            "--export",
            f"ALL,RUN_DIR={results_root},SUBMISSION_FILE={submission_file}",
            "--array",
            array_spec,
            str(sbatch_path),
        ],
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    print(out.strip())
    submitted_total += len(chunk)
    for config_id in chunk:
        status_payload = read_status(results_root, config_id) or {}
        write_status(
            results_root,
            config_id,
            {
                "state": "SUBMITTED",
                "attempts": int(status_payload.get("attempts", 0)),
                "retryable": bool(status_payload.get("retryable", True)),
                "submission_file": str(submission_file),
                "submission_chunk": chunk_idx,
            },
        )
    print(
        f"Submitted {submitted_total} configs in 1 Slurm array "
        f"(remaining chunks after this: {max(0, len(chunked) - 1)})"
    )
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    results_root = resolve_results_root(args.results_root)
    manifest_path = results_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = summarize_progress(results_root, int(manifest.get("max_attempts", DEFAULT_MAX_ATTEMPTS)))
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for key in [
            "total",
            "successful",
            "running",
            "pending_in_queue",
            "failed_retryable",
            "failed_terminal",
            "not_started",
            "unfinished",
            "pair_total",
            "pair_successful",
            "pair_incomplete",
        ]:
            print(f"{key}: {payload[key]}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    results_root = resolve_results_root(args.results_root)
    outputs = write_analysis_outputs(results_root)
    for key, value in outputs.items():
        print(f"{key}: {value}")
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "generate":
        return cmd_generate(args)
    if args.command == "run-one":
        return cmd_run_one(args)
    if args.command == "submit-pending":
        return cmd_submit_pending(args)
    if args.command == "summary":
        return cmd_summary(args)
    if args.command == "analyze":
        return cmd_analyze(args)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate, submit, run, and summarize the 15-sample Games 1-3 derisk batch."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import random
import re
import secrets
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strong_models_experiment.analysis.active_model_roster import (  # noqa: E402
    context_filtered_model_pool,
    elo_for_model,
    quantile_elo_bucket_map,
)


BASELINE_MODEL = "gpt-5-nano"
ADVERSARY_MODELS = ["amazon-nova-micro-v1.0", "claude-opus-4-6-thinking"]
N_AGENTS = 10
MAX_ROUNDS = 2
DISCUSSION_TURNS = 2
GAMMA_DISCOUNT = 0.9
SLURM_TIME = "08:00:00"
MAX_CONCURRENT = 15

GAME_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "game_label": "game1",
        "game_type": "item_allocation",
        "num_items": 25,
        "competition_level": 0.9,
    },
    {
        "game_label": "game2",
        "game_type": "diplomacy",
        "n_issues": 10,
        "rho": (6.0 / 3.141592653589793) * __import__("math").asin(-1.0 / (2.0 * (N_AGENTS - 1))),
        "theta": 0.9,
    },
    {
        "game_label": "game3",
        "game_type": "co_funding",
        "m_projects": 25,
        "alpha": 0.1,
        "sigma": 0.5,
        "c_min": 10.0,
        "c_max": 30.0,
        "cofunding_discussion_transparency": "own",
        "cofunding_enable_commit_vote": True,
        "cofunding_enable_time_discount": True,
        "cofunding_time_discount": 0.9,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--results-root", type=str, default=None)

    submit = subparsers.add_parser("submit")
    submit.add_argument("--results-root", type=str, required=True)

    run_one = subparsers.add_parser("run-one")
    run_one.add_argument("--results-root", type=str, required=True)
    run_one.add_argument("--config-id", type=int, required=True)

    summary = subparsers.add_parser("summary")
    summary.add_argument("--results-root", type=str, required=True)
    summary.add_argument("--json", action="store_true")

    return parser.parse_args()


def timestamp_now() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def latest_root_or_new(raw_value: Optional[str]) -> Path:
    if raw_value:
        path = Path(raw_value)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        return path
    return (
        PROJECT_ROOT
        / "experiments"
        / "results"
        / f"derisk_15_samples_{timestamp_now()}"
    ).resolve()


def resolve_results_root(raw_value: str) -> Path:
    path = Path(raw_value)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def sanitize_token(value: Any) -> str:
    token = str(value).replace(".", "p").replace("-", "_").replace("/", "_")
    return re.sub(r"[^A-Za-z0-9_]+", "_", token).strip("_")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def agent_ids(n_agents: int) -> List[str]:
    return [f"Agent_{idx}" for idx in range(1, n_agents + 1)]


def build_agent_maps(models: List[str], adversary_model: Optional[str]) -> Dict[str, Dict[str, Any]]:
    ids = agent_ids(len(models))
    agent_model_map = dict(zip(ids, models))
    agent_elo_map = {agent_id: elo_for_model(model) for agent_id, model in agent_model_map.items()}
    if adversary_model is None:
        agent_role_map = {agent_id: "heterogeneous_agent" for agent_id in ids}
    else:
        agent_role_map = {
            agent_id: ("adversary" if model == adversary_model else "baseline")
            for agent_id, model in agent_model_map.items()
        }
    return {
        "agent_model_map": agent_model_map,
        "agent_elo_map": agent_elo_map,
        "agent_role_map": agent_role_map,
    }


def common_config(
    *,
    config_id: int,
    results_root: Path,
    game_def: Dict[str, Any],
    sample_kind: str,
    sample_name: str,
    model_order: str,
    models: List[str],
    random_seed: int,
    adversary_model: Optional[str] = None,
    adversary_position: Optional[str] = None,
    model_pool: Optional[List[str]] = None,
    elo_bucket_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    game_label = game_def["game_label"]
    output_dir = (
        results_root
        / "runs"
        / f"config_{config_id:04d}_{game_label}_{sanitize_token(sample_name)}"
    )
    config: Dict[str, Any] = {
        "config_id": config_id,
        "experiment_id": config_id,
        "batch_type": "derisk_15_samples",
        "experiment_family": sample_kind,
        "experiment_type": sample_kind,
        "sample_name": sample_name,
        "game_label": game_label,
        "game_type": game_def["game_type"],
        "baseline_model": BASELINE_MODEL,
        "models": models,
        "model1": BASELINE_MODEL if adversary_model else "heterogeneous_pool",
        "model2": adversary_model if adversary_model else "random_10_models",
        "model_order": model_order,
        "n_agents": len(models),
        "max_rounds": MAX_ROUNDS,
        "discussion_turns": DISCUSSION_TURNS,
        "gamma_discount": GAMMA_DISCOUNT,
        "parallel_phases": True,
        "random_seed": random_seed,
        "seed": random_seed,
        "run_number": 1,
        "adversary_model": adversary_model,
        "adversary_position": adversary_position,
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        **build_agent_maps(models, adversary_model),
    }
    for key, value in game_def.items():
        if key not in {"game_label", "game_type"}:
            config[key] = value
    if model_pool is not None:
        config["model_pool"] = model_pool
        config["model_pool_size"] = len(model_pool)
    if elo_bucket_map is not None:
        config["elo_bucket_method"] = "quantile_terciles"
        config["model_elo_bucket_map"] = elo_bucket_map
        config["agent_elo_bucket_map"] = {
            agent_id: elo_bucket_map.get(model)
            for agent_id, model in config["agent_model_map"].items()
        }
    return config


def build_configs(results_root: Path) -> List[Dict[str, Any]]:
    config_id = 1
    configs: List[Dict[str, Any]] = []
    model_pool = list(context_filtered_model_pool())
    elo_bucket_map = quantile_elo_bucket_map(model_pool, n_buckets=3)

    for game_def in GAME_DEFINITIONS:
        for adversary in ADVERSARY_MODELS:
            first_models = [adversary] + [BASELINE_MODEL] * (N_AGENTS - 1)
            configs.append(
                common_config(
                    config_id=config_id,
                    results_root=results_root,
                    game_def=game_def,
                    sample_kind="homogeneous_adversary_sample",
                    sample_name=f"{game_def['game_label']}_{sanitize_token(adversary)}_first",
                    model_order="adversary_first",
                    models=first_models,
                    random_seed=secrets.randbelow(2**31 - 1),
                    adversary_model=adversary,
                    adversary_position="first",
                )
            )
            config_id += 1

            last_models = [BASELINE_MODEL] * (N_AGENTS - 1) + [adversary]
            configs.append(
                common_config(
                    config_id=config_id,
                    results_root=results_root,
                    game_def=game_def,
                    sample_kind="homogeneous_adversary_sample",
                    sample_name=f"{game_def['game_label']}_{sanitize_token(adversary)}_last",
                    model_order="adversary_last",
                    models=last_models,
                    random_seed=secrets.randbelow(2**31 - 1),
                    adversary_model=adversary,
                    adversary_position="last",
                )
            )
            config_id += 1

        draw_seed = secrets.randbits(64)
        rng = random.Random(draw_seed)
        sampled_models = rng.sample(model_pool, N_AGENTS)
        rng.shuffle(sampled_models)
        configs.append(
            common_config(
                config_id=config_id,
                results_root=results_root,
                game_def=game_def,
                sample_kind="heterogeneous_random_sample",
                sample_name=f"{game_def['game_label']}_heterogeneous_random",
                model_order="sampled_random_order",
                models=sampled_models,
                random_seed=secrets.randbelow(2**31 - 1),
                model_pool=model_pool,
                elo_bucket_map=elo_bucket_map,
            )
        )
        configs[-1]["heterogeneous_draw_seed"] = draw_seed
        config_id += 1

    return configs


def write_generated_files(results_root: Path, configs: List[Dict[str, Any]]) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    config_dir = results_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    slurm_dir = results_root / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    (results_root / "logs").mkdir(parents=True, exist_ok=True)
    (results_root / "status").mkdir(parents=True, exist_ok=True)

    manifest = {
        "results_root": str(results_root),
        "batch_type": "derisk_15_samples",
        "generated_at": dt.datetime.now().isoformat(),
        "n_configs": len(configs),
        "n_agents": N_AGENTS,
        "max_rounds": MAX_ROUNDS,
        "discussion_turns": DISCUSSION_TURNS,
        "parallel_phases": True,
        "slurm_time": SLURM_TIME,
        "max_concurrent": MAX_CONCURRENT,
        "baseline_model": BASELINE_MODEL,
        "adversary_models": ADVERSARY_MODELS,
        "context_filtered_model_pool": list(context_filtered_model_pool()),
        "context_filtered_model_pool_count": len(context_filtered_model_pool()),
        "game_definitions": GAME_DEFINITIONS,
        "notes": "Three games x four homogeneous adversary positions plus one independently sampled heterogeneous roster per game.",
    }
    write_json(results_root / "manifest.json", manifest)

    for config in configs:
        write_json(config_dir / f"config_{config['config_id']:04d}.json", config)

    (config_dir / "all_configs.txt").write_text(
        "\n".join(str(config_dir / f"config_{cfg['config_id']:04d}.json") for cfg in configs) + "\n",
        encoding="utf-8",
    )

    fieldnames = [
        "config_id",
        "game_label",
        "game_type",
        "experiment_type",
        "sample_name",
        "model_order",
        "adversary_model",
        "adversary_position",
        "n_agents",
        "random_seed",
        "models",
        "output_dir",
    ]
    with (config_dir / "experiment_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for config in configs:
            row = {key: config.get(key) for key in fieldnames if key != "models"}
            row["models"] = "+".join(config["models"])
            writer.writerow(row)

    latest = results_root.parent / "derisk_15_samples_latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(results_root.name)


def load_config(results_root: Path, config_id: int) -> Dict[str, Any]:
    path = results_root / "configs" / f"config_{config_id:04d}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_configs(results_root: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((results_root / "configs").glob("config_*.json"))
    ]


def result_path_for(config: Dict[str, Any]) -> Optional[Path]:
    output_dir = (PROJECT_ROOT / config["output_dir"]).resolve()
    for candidate in [
        output_dir / "experiment_results.json",
        output_dir / "run_1_experiment_results.json",
    ]:
        if candidate.exists():
            return candidate
    return None


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
    python_bin = PROJECT_ROOT / ".venv" / "bin" / "python"
    output_dir = (PROJECT_ROOT / config["output_dir"]).resolve()
    cmd = [
        str(python_bin),
        str(PROJECT_ROOT / "run_strong_models_experiment.py"),
        "--models",
        *config["models"],
        "--game-type",
        config["game_type"],
        "--max-rounds",
        str(config["max_rounds"]),
        "--discussion-turns",
        str(config["discussion_turns"]),
        "--model-order",
        config["model_order"],
        "--random-seed",
        str(config["random_seed"]),
        "--output-dir",
        str(output_dir),
        "--job-id",
        str(config["config_id"]),
        "--run-number",
        str(config["run_number"]),
        "--parallel-phases",
        "--max-tokens-voting",
        str(config.get("max_tokens_voting", 4096)),
    ]

    if config["game_type"] == "item_allocation":
        cmd.extend([
            "--num-items",
            str(config["num_items"]),
            "--competition-level",
            str(config["competition_level"]),
            "--gamma-discount",
            str(config["gamma_discount"]),
        ])
    elif config["game_type"] == "diplomacy":
        cmd.extend([
            "--n-issues",
            str(config["n_issues"]),
            "--rho",
            str(config["rho"]),
            "--theta",
            str(config["theta"]),
        ])
    elif config["game_type"] == "co_funding":
        cmd.extend([
            "--m-projects",
            str(config["m_projects"]),
            "--alpha",
            str(config["alpha"]),
            "--sigma",
            str(config["sigma"]),
            "--c-min",
            str(config["c_min"]),
            "--c-max",
            str(config["c_max"]),
            "--cofunding-discussion-transparency",
            str(config["cofunding_discussion_transparency"]),
            "--cofunding-time-discount",
            str(config["cofunding_time_discount"]),
        ])
    else:
        raise ValueError(f"Unknown game_type: {config['game_type']}")

    return cmd


def enrich_result_metadata(config: Dict[str, Any]) -> None:
    result_path = result_path_for(config)
    if result_path is None:
        return
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload.setdefault("config", {})
    metadata_keys = [
        "config_id",
        "batch_type",
        "experiment_family",
        "experiment_type",
        "sample_name",
        "game_label",
        "baseline_model",
        "models",
        "agent_model_map",
        "agent_elo_map",
        "agent_role_map",
        "model_order",
        "adversary_model",
        "adversary_position",
        "model_pool",
        "model_pool_size",
        "elo_bucket_method",
        "model_elo_bucket_map",
        "agent_elo_bucket_map",
        "heterogeneous_draw_seed",
    ]
    payload["config"].update({key: config[key] for key in metadata_keys if key in config})
    result_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def run_config(results_root: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    output_dir = (PROJECT_ROOT / config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_root / "logs" / f"config_{config['config_id']:04d}.log"
    status_path = results_root / "status" / f"config_{config['config_id']:04d}.json"

    existing_result = result_path_for(config)
    if existing_result is not None:
        enrich_result_metadata(config)
        status = {
            "config_id": config["config_id"],
            "state": "SUCCESS",
            "returncode": 0,
            "skipped_existing": True,
            "result_path": str(existing_result),
            "updated_at": dt.datetime.now().isoformat(),
        }
        write_json(status_path, status)
        return status

    cmd = build_command(config)
    start_time = dt.datetime.now()
    write_json(
        status_path,
        {
            "config_id": config["config_id"],
            "state": "RUNNING",
            "game_label": config["game_label"],
            "sample_name": config["sample_name"],
            "command": shlex.join(cmd),
            "started_at": start_time.isoformat(),
        },
    )

    env = load_dotenv_env()
    env.update({
        "PYTHONUNBUFFERED": "1",
        "OPENROUTER_TRANSPORT": env.get("OPENROUTER_TRANSPORT", "proxy"),
        "OPENROUTER_PROXY_POLL_DIR": env.get("OPENROUTER_PROXY_POLL_DIR", "/home/jz4391/openrouter_proxy"),
        "OPENROUTER_PROXY_CLIENT_TIMEOUT": env.get("OPENROUTER_PROXY_CLIENT_TIMEOUT", "9000"),
        "OPENROUTER_API_TIMEOUT": env.get("OPENROUTER_API_TIMEOUT", "1800"),
        "LLM_FAILURE_REPORT_PATH": env.get(
            "LLM_FAILURE_REPORT_PATH",
            str(results_root / "monitoring" / "provider_failures.md"),
        ),
    })
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Started at {start_time.isoformat()}\n")
        handle.write(f"Config: config_{config['config_id']:04d} {config['game_label']} {config['sample_name']}\n")
        handle.write(f"Command: {shlex.join(cmd)}\n\n")
        handle.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        returncode = proc.wait()
        handle.write(f"\nReturncode: {returncode}\n")

    end_time = dt.datetime.now()
    result_path = result_path_for(config)
    state = "SUCCESS" if returncode == 0 and result_path is not None else "FAILED"
    if state == "SUCCESS":
        enrich_result_metadata(config)
    status = {
        "config_id": config["config_id"],
        "state": state,
        "returncode": returncode,
        "started_at": start_time.isoformat(),
        "finished_at": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "log_path": str(log_path),
        "result_path": str(result_path) if result_path else None,
        "updated_at": end_time.isoformat(),
    }
    write_json(status_path, status)
    return status


def write_slurm_file(results_root: Path) -> Path:
    slurm_dir = results_root / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    python_bin = PROJECT_ROOT / ".venv" / "bin" / "python"
    script_path = PROJECT_ROOT / "scripts" / "derisk_15_samples_batch.py"
    sbatch_path = slurm_dir / "run_one.sbatch"
    sbatch = f"""#!/bin/bash
#SBATCH --job-name=drisk15
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time={SLURM_TIME}
#SBATCH --partition=cpu
#SBATCH --output={PROJECT_ROOT}/slurm/derisk15_%j.out
#SBATCH --error={PROJECT_ROOT}/slurm/derisk15_%j.err

set -eo pipefail

BASE_DIR="{PROJECT_ROOT}"
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

: "${{RUN_DIR:?RUN_DIR is required}}"
: "${{CONFIG_ID:?CONFIG_ID is required}}"

export OPENROUTER_TRANSPORT="${{OPENROUTER_TRANSPORT:-proxy}}"
export OPENROUTER_PROXY_POLL_DIR="${{OPENROUTER_PROXY_POLL_DIR:-/home/jz4391/openrouter_proxy}}"
export OPENROUTER_PROXY_CLIENT_TIMEOUT="${{OPENROUTER_PROXY_CLIENT_TIMEOUT:-9000}}"
export OPENROUTER_API_TIMEOUT="${{OPENROUTER_API_TIMEOUT:-1800}}"
export LLM_FAILURE_REPORT_PATH="${{LLM_FAILURE_REPORT_PATH:-$RUN_DIR/monitoring/provider_failures.md}}"
export PYTHONUNBUFFERED=1

"{python_bin}" "{script_path}" run-one --results-root "$RUN_DIR" --config-id "$CONFIG_ID"
"""
    sbatch_path.write_text(sbatch, encoding="utf-8")
    os.chmod(sbatch_path, 0o755)
    return sbatch_path


def cmd_generate(args: argparse.Namespace) -> int:
    results_root = latest_root_or_new(args.results_root)
    configs = build_configs(results_root)
    write_generated_files(results_root, configs)
    print(results_root)
    print(f"Generated {len(configs)} configs")
    return 0


def cmd_run_one(args: argparse.Namespace) -> int:
    results_root = resolve_results_root(args.results_root)
    config = load_config(results_root, args.config_id)
    status = run_config(results_root, config)
    print(json.dumps(status, indent=2, default=str))
    return 0 if status["state"] == "SUCCESS" else 1


def parse_job_id(sbatch_output: str) -> Optional[str]:
    match = re.search(r"Submitted batch job\s+(\d+)", sbatch_output)
    return match.group(1) if match else None


def cmd_submit(args: argparse.Namespace) -> int:
    results_root = resolve_results_root(args.results_root)
    configs = load_configs(results_root)
    sbatch_path = write_slurm_file(results_root)
    submissions = []
    for config in configs:
        output = subprocess.check_output(
            [
                "sbatch",
                f"--export=ALL,RUN_DIR={results_root},CONFIG_ID={config['config_id']}",
                str(sbatch_path),
            ],
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        job_id = parse_job_id(output)
        submissions.append({
            "config_id": config["config_id"],
            "job_id": job_id,
            "sbatch_output": output.strip(),
            "submitted_at": dt.datetime.now().isoformat(),
        })
        print(f"config_{config['config_id']:04d}: {output.strip()}")
    write_json(
        results_root / "submissions.json",
        {
            "submitted_at": dt.datetime.now().isoformat(),
            "slurm_time": SLURM_TIME,
            "submissions": submissions,
        },
    )
    return 0


def read_status(results_root: Path, config_id: int) -> Dict[str, Any]:
    path = results_root / "status" / f"config_{config_id:04d}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"state": "STATUS_PARSE_ERROR"}


def load_submissions(results_root: Path) -> Dict[int, str]:
    path = results_root / "submissions.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(row["config_id"]): str(row["job_id"])
        for row in payload.get("submissions", [])
        if row.get("job_id")
    }


def squeue_states(job_ids: List[str]) -> Dict[str, Dict[str, str]]:
    if not job_ids:
        return {}
    try:
        output = subprocess.check_output(
            ["squeue", "-h", "-j", ",".join(job_ids), "-o", "%A|%T|%M|%R|%j"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return {}
    states = {}
    for line in output.splitlines():
        parts = line.split("|", 4)
        if len(parts) == 5:
            states[parts[0]] = {
                "state": parts[1],
                "elapsed": parts[2],
                "reason": parts[3],
                "name": parts[4],
            }
    return states


def detect_phase(log_path: Path) -> str:
    if not log_path.exists():
        return "not_started"
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "log_unreadable"
    phase_markers = [
        ("Strong models experiment completed successfully", "completed"),
        ("VOTE TABULATION PHASE", "vote_tabulation"),
        ("PRIVATE VOTING PHASE", "voting"),
        ("PROPOSAL ENUMERATION PHASE", "proposal_enumeration"),
        ("PROPOSAL PHASE", "proposal"),
        ("PRIVATE THINKING PHASE", "private_thinking"),
        ("PUBLIC DISCUSSION PHASE", "discussion"),
        ("DISCUSSION PHASE", "discussion"),
        ("GAME SETUP PHASE", "setup"),
        ("SETUP PHASE", "setup"),
    ]
    for line in reversed(lines[-400:]):
        for marker, phase in phase_markers:
            if marker in line:
                return phase
    return "starting"


def detect_phase_from_interactions(run_dir: Path) -> Optional[str]:
    path = run_dir / "all_interactions.json"
    if not path.exists():
        return None
    try:
        interactions = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return "interactions_unreadable"
    if not isinstance(interactions, list) or not interactions:
        return None

    last_phase = str(interactions[-1].get("phase", "unknown"))
    if last_phase.startswith("voting_round_"):
        return last_phase
    if last_phase.startswith("proposal_round_"):
        return last_phase
    if last_phase.startswith("private_thinking_round_"):
        return last_phase
    if last_phase.startswith("discussion_round_"):
        return last_phase
    if last_phase == "game_setup":
        return "setup"
    return last_phase


def validate_result(config: Dict[str, Any], result_path: Path) -> Dict[str, Any]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    config_payload = payload.get("config", {})
    logs = payload.get("conversation_logs", [])
    tabulation_logs = [
        log for log in logs
        if log.get("phase") == "vote_tabulation"
    ]
    all_interactions = result_path.parent / "all_interactions.json"
    return {
        "parallel_phases_configured": bool(config_payload.get("parallel_phases")),
        "n_agents": config_payload.get("n_agents"),
        "model_count": len(config_payload.get("models", [])),
        "final_round": payload.get("final_round"),
        "consensus_reached": payload.get("consensus_reached"),
        "has_strict_majority_tabulation_text": any(
            "strict-majority" in str(log.get("content", "")).lower()
            or "strict majority" in str(log.get("content", "")).lower()
            for log in tabulation_logs
        ),
        "conversation_log_count": len(logs),
        "all_interactions_exists": all_interactions.exists(),
    }


def collect_summary(results_root: Path) -> Dict[str, Any]:
    configs = load_configs(results_root)
    submissions = load_submissions(results_root)
    queue = squeue_states(list(submissions.values()))
    rows = []
    durations = []
    for config in configs:
        config_id = int(config["config_id"])
        status = read_status(results_root, config_id)
        result_path = result_path_for(config)
        job_id = submissions.get(config_id)
        queue_state = queue.get(job_id, {}) if job_id else {}
        state = status.get("state")
        if result_path is not None and state != "FAILED":
            state = "SUCCESS"
        elif state is None:
            state = queue_state.get("state", "SUBMITTED" if job_id else "NOT_SUBMITTED")
        if state == "COMPLETED" and result_path is None:
            state = "MISSING_RESULT"
        if state in {"FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY", "MISSING_RESULT"}:
            normalized = "ERROR"
        elif state == "SUCCESS":
            normalized = "SUCCESS"
        elif queue_state.get("state") == "PENDING":
            normalized = "QUEUED"
        elif state == "RUNNING" or queue_state.get("state") == "RUNNING":
            normalized = "RUNNING"
        else:
            normalized = state

        log_path = results_root / "logs" / f"config_{config_id:04d}.log"
        run_dir = Path(config["output_dir"])
        phase = detect_phase_from_interactions(run_dir) or detect_phase(log_path)
        row = {
            "config_id": config_id,
            "job_id": job_id,
            "game_label": config["game_label"],
            "sample_name": config["sample_name"],
            "state": normalized,
            "raw_state": state,
            "queue_state": queue_state.get("state"),
            "phase": phase,
            "result_path": str(result_path) if result_path else None,
            "log_path": str(log_path),
        }
        if result_path is not None:
            try:
                row["validation"] = validate_result(config, result_path)
            except Exception as exc:  # noqa: BLE001
                row["validation"] = {"error": str(exc)}
        if status.get("duration_seconds") is not None and normalized == "SUCCESS":
            durations.append(float(status["duration_seconds"]))
        rows.append(row)

    counts = {
        "total": len(rows),
        "finished": sum(1 for row in rows if row["state"] == "SUCCESS"),
        "started": sum(1 for row in rows if row["state"] in {"RUNNING", "SUCCESS", "ERROR"}),
        "queued": sum(1 for row in rows if row["state"] == "QUEUED"),
        "running": sum(1 for row in rows if row["state"] == "RUNNING"),
        "errored": sum(1 for row in rows if row["state"] == "ERROR"),
    }
    eta = None
    if durations and counts["running"]:
        avg_duration = sum(durations) / len(durations)
        started_times = []
        now = dt.datetime.now()
        for config in configs:
            status = read_status(results_root, int(config["config_id"]))
            if status.get("state") == "RUNNING" and status.get("started_at"):
                try:
                    started_times.append((now - dt.datetime.fromisoformat(status["started_at"])).total_seconds())
                except ValueError:
                    pass
        remaining_running = [max(0.0, avg_duration - elapsed) for elapsed in started_times]
        if remaining_running:
            eta_seconds = max(remaining_running)
            eta = {
                "basis": f"mean successful runtime over {len(durations)} completed sample(s)",
                "seconds": eta_seconds,
                "eta_at": (now + dt.timedelta(seconds=eta_seconds)).isoformat(),
            }
    return {
        "generated_at": dt.datetime.now().isoformat(),
        "counts": counts,
        "eta": eta,
        "rows": rows,
    }


def cmd_summary(args: argparse.Namespace) -> int:
    results_root = resolve_results_root(args.results_root)
    summary = collect_summary(results_root)
    if args.json:
        print(json.dumps(summary, indent=2, default=str))
        return 0

    counts = summary["counts"]
    print(
        f"finished={counts['finished']}/{counts['total']} "
        f"started={counts['started']} queued={counts['queued']} "
        f"running={counts['running']} errored={counts['errored']}"
    )
    eta = summary.get("eta")
    if eta:
        print(f"eta={eta['eta_at']} ({eta['basis']})")
    else:
        print("eta=not yet estimable")
    for row in summary["rows"]:
        print(
            f"config_{row['config_id']:04d} {row['game_label']} {row['state']} "
            f"job={row.get('job_id')} phase={row['phase']} {row['sample_name']}"
        )
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "generate":
        return cmd_generate(args)
    if args.command == "submit":
        return cmd_submit(args)
    if args.command == "run-one":
        return cmd_run_one(args)
    if args.command == "summary":
        return cmd_summary(args)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

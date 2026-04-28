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
from pathlib import Path
from typing import Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strong_models_experiment.analysis.active_model_roster import (  # noqa: E402
    active_model_elo_map,
    context_filtered_model_pool,
    elo_for_model,
    quantile_elo_bucket_map,
)


BASELINE_MODEL = "gpt-5-nano"
SAMPLE_ADVERSARIES = ["amazon-nova-micro-v1.0", "claude-opus-4-6-thinking"]
DEFAULT_N = 10
DEFAULT_NUM_ITEMS = 25
DEFAULT_COMPETITION_LEVEL = 1.0
DEFAULT_MAX_ROUNDS = 10
DEFAULT_DISCUSSION_TURNS = 2
DEFAULT_GAMMA_DISCOUNT = 0.9
DEFAULT_SAMPLE_SEED = 10001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and run Game 1 n-agent matrix sample configs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--results-root", type=str, default=None)
    generate.add_argument("--seed", type=int, default=DEFAULT_SAMPLE_SEED)

    run_one = subparsers.add_parser("run-one")
    run_one.add_argument("--results-root", type=str, required=True)
    run_one.add_argument("--config-id", type=int, required=True)

    run_all = subparsers.add_parser("run-all")
    run_all.add_argument("--results-root", type=str, required=True)
    run_all.add_argument("--start-at", type=int, default=1)

    submit = subparsers.add_parser("submit")
    submit.add_argument("--results-root", type=str, required=True)
    submit.add_argument("--max-concurrent", type=int, default=5)
    submit.add_argument("--time", type=str, default="12:00:00")

    summary = subparsers.add_parser("summary")
    summary.add_argument("--results-root", type=str, required=True)
    summary.add_argument("--json", action="store_true")

    return parser.parse_args()


def latest_root_or_new(raw_value: Optional[str]) -> Path:
    if raw_value:
        path = Path(raw_value)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        return path
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        PROJECT_ROOT
        / "experiments"
        / "results"
        / f"game1_multiagent_matrix_sample_{timestamp}"
    ).resolve()


def resolve_results_root(raw_value: str) -> Path:
    path = Path(raw_value)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def sanitize_token(value: object) -> str:
    return str(value).replace(".", "p").replace("-", "_").replace("/", "_")


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def create_latest_symlink(results_root: Path) -> None:
    symlink = results_root.parent / "game1_multiagent_matrix_sample_latest"
    if symlink.is_symlink() or symlink.exists():
        symlink.unlink()
    symlink.symlink_to(results_root.name)


def agent_ids(n_agents: int) -> List[str]:
    return [f"Agent_{index}" for index in range(1, n_agents + 1)]


def build_agent_maps(models: List[str], adversary_model: Optional[str] = None) -> Dict[str, Dict]:
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


def make_config(
    *,
    config_id: int,
    results_root: Path,
    experiment_family: str,
    model_order: str,
    models: List[str],
    random_seed: int,
    adversary_model: Optional[str] = None,
    adversary_position: Optional[str] = None,
    model_pool: Optional[List[str]] = None,
    elo_bucket_map: Optional[Dict[str, str]] = None,
) -> Dict:
    n_agents = len(models)
    output_dir = (
        results_root
        / experiment_family
        / f"config_{config_id:04d}_{sanitize_token(model_order)}"
    )
    maps = build_agent_maps(models, adversary_model=adversary_model)
    config = {
        "config_id": config_id,
        "experiment_family": experiment_family,
        "game_type": "item_allocation",
        "baseline_model": BASELINE_MODEL,
        "models": models,
        "model_order": model_order,
        "n_agents": n_agents,
        "num_items": DEFAULT_NUM_ITEMS,
        "competition_level": DEFAULT_COMPETITION_LEVEL,
        "max_rounds": DEFAULT_MAX_ROUNDS,
        "discussion_turns": DEFAULT_DISCUSSION_TURNS,
        "gamma_discount": DEFAULT_GAMMA_DISCOUNT,
        "random_seed": random_seed,
        "adversary_model": adversary_model,
        "adversary_position": adversary_position,
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        **maps,
    }
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


def build_sample_configs(results_root: Path, seed: int) -> List[Dict]:
    configs: List[Dict] = []
    config_id = 1

    for adversary in SAMPLE_ADVERSARIES:
        first_models = [adversary] + [BASELINE_MODEL] * (DEFAULT_N - 1)
        configs.append(
            make_config(
                config_id=config_id,
                results_root=results_root,
                experiment_family="homogeneous_adversary_sample",
                model_order="adversary_first",
                models=first_models,
                random_seed=seed,
                adversary_model=adversary,
                adversary_position="first",
            )
        )
        config_id += 1

        last_models = [BASELINE_MODEL] * (DEFAULT_N - 1) + [adversary]
        configs.append(
            make_config(
                config_id=config_id,
                results_root=results_root,
                experiment_family="homogeneous_adversary_sample",
                model_order="adversary_last",
                models=last_models,
                random_seed=seed,
                adversary_model=adversary,
                adversary_position="last",
            )
        )
        config_id += 1

    model_pool = list(context_filtered_model_pool())
    rng = random.Random(seed)
    sampled_models = rng.sample(model_pool, DEFAULT_N)
    rng.shuffle(sampled_models)
    elo_bucket_map = quantile_elo_bucket_map(model_pool, n_buckets=3)
    configs.append(
        make_config(
            config_id=config_id,
            results_root=results_root,
            experiment_family="heterogeneous_random_sample",
            model_order="sampled_random_order",
            models=sampled_models,
            random_seed=seed,
            model_pool=model_pool,
            elo_bucket_map=elo_bucket_map,
        )
    )

    return configs


def write_generated_files(results_root: Path, manifest: Dict, configs: List[Dict]) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    config_dir = results_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    write_json(results_root / "manifest.json", manifest)
    for config in configs:
        write_json(config_dir / f"config_{config['config_id']:04d}.json", config)

    (config_dir / "all_configs.txt").write_text(
        "\n".join(str(config_dir / f"config_{cfg['config_id']:04d}.json") for cfg in configs) + "\n",
        encoding="utf-8",
    )

    fieldnames = [
        "config_id",
        "experiment_family",
        "model_order",
        "adversary_model",
        "adversary_position",
        "n_agents",
        "num_items",
        "competition_level",
        "random_seed",
        "models",
        "output_dir",
        "config_file",
    ]
    with (config_dir / "experiment_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for config in configs:
            row = {key: config.get(key) for key in fieldnames if key not in {"models", "config_file"}}
            row["models"] = "+".join(config["models"])
            row["config_file"] = f"config_{config['config_id']:04d}.json"
            writer.writerow(row)


def write_slurm_file(results_root: Path, max_concurrent: int, slurm_time: str) -> Path:
    slurm_dir = results_root / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    python_bin = PROJECT_ROOT / ".venv" / "bin" / "python"
    script_path = PROJECT_ROOT / "scripts" / "game1_multiagent_matrix_batch.py"
    sbatch_path = slurm_dir / "run_samples.sbatch"
    sbatch_text = f"""#!/bin/bash
#SBATCH --job-name=g1matrix
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time={slurm_time}
#SBATCH --partition=cpu
#SBATCH --output={PROJECT_ROOT}/slurm/game1_matrix_sample_%A_%a.out
#SBATCH --error={PROJECT_ROOT}/slurm/game1_matrix_sample_%A_%a.err

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

export OPENROUTER_TRANSPORT="${{OPENROUTER_TRANSPORT:-proxy}}"
export OPENROUTER_PROXY_POLL_DIR="${{OPENROUTER_PROXY_POLL_DIR:-/home/jz4391/openrouter_proxy}}"
export OPENROUTER_PROXY_CLIENT_TIMEOUT="${{OPENROUTER_PROXY_CLIENT_TIMEOUT:-9000}}"
export OPENROUTER_API_TIMEOUT="${{OPENROUTER_API_TIMEOUT:-1800}}"
export LLM_FAILURE_REPORT_PATH="${{LLM_FAILURE_REPORT_PATH:-$RUN_DIR/monitoring/provider_failures.md}}"
export PYTHONUNBUFFERED=1

"{python_bin}" "{script_path}" run-one --results-root "$RUN_DIR" --config-id "$SLURM_ARRAY_TASK_ID"
"""
    sbatch_path.write_text(sbatch_text, encoding="utf-8")
    os.chmod(sbatch_path, 0o755)
    return sbatch_path


def cmd_generate(args: argparse.Namespace) -> int:
    results_root = latest_root_or_new(args.results_root)
    configs = build_sample_configs(results_root, seed=args.seed)
    manifest = {
        "results_root": str(results_root),
        "batch_type": "game1_multiagent_matrix_sample",
        "game_type": "item_allocation",
        "baseline_model": BASELINE_MODEL,
        "sample_adversaries": SAMPLE_ADVERSARIES,
        "context_filtered_model_pool": list(context_filtered_model_pool()),
        "context_filtered_model_pool_count": len(context_filtered_model_pool()),
        "n_agents": DEFAULT_N,
        "num_items": DEFAULT_NUM_ITEMS,
        "competition_level": DEFAULT_COMPETITION_LEVEL,
        "max_rounds": DEFAULT_MAX_ROUNDS,
        "discussion_turns": DEFAULT_DISCUSSION_TURNS,
        "gamma_discount": DEFAULT_GAMMA_DISCOUNT,
        "sample_seed": args.seed,
        "config_count": len(configs),
        "notes": "Approved Game 1 fail-fast samples: two adversaries in first/last positions plus one true-random heterogeneous sample.",
    }
    write_generated_files(results_root, manifest, configs)
    create_latest_symlink(results_root)
    print(results_root)
    print(f"Generated {len(configs)} sample configs")
    return 0


def load_config(results_root: Path, config_id: int) -> Dict:
    path = results_root / "configs" / f"config_{config_id:04d}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_configs(results_root: Path) -> List[Dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((results_root / "configs").glob("config_*.json"))
    ]


def result_path_for(config: Dict) -> Optional[Path]:
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
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        env.setdefault(key, value)
    return env


def build_command(config: Dict) -> List[str]:
    output_dir = (PROJECT_ROOT / config["output_dir"]).resolve()
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
        "--output-dir",
        str(output_dir),
        "--job-id",
        str(config["config_id"]),
    ]
    if config.get("max_tokens_voting") is not None:
        cmd.extend(["--max-tokens-voting", str(config["max_tokens_voting"])])
    if config.get("parallel_phases", False):
        cmd.append("--parallel-phases")
    return cmd


def enrich_result_metadata(config: Dict) -> None:
    result_path = result_path_for(config)
    if result_path is None:
        return
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload.setdefault("config", {})
    payload["config"].update(
        {
            key: config[key]
            for key in [
                "config_id",
                "experiment_family",
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
            ]
            if key in config
        }
    )
    result_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def run_config(results_root: Path, config: Dict) -> Dict:
    output_dir = (PROJECT_ROOT / config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    monitoring_dir = results_root / "monitoring"
    monitoring_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = results_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    status_dir = results_root / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"config_{config['config_id']:04d}.log"
    status_path = status_dir / f"config_{config['config_id']:04d}.json"

    if result_path_for(config) is not None:
        enrich_result_metadata(config)
        status = {
            "config_id": config["config_id"],
            "state": "SUCCESS",
            "returncode": 0,
            "skipped_existing": True,
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
        },
    )
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Started at {dt.datetime.now().isoformat()}\n")
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

    state = "SUCCESS" if returncode == 0 and result_path_for(config) is not None else "FAILED"
    if state == "SUCCESS":
        enrich_result_metadata(config)
    status = {
        "config_id": config["config_id"],
        "state": state,
        "returncode": returncode,
        "log_path": str(log_path),
        "result_path": str(result_path_for(config)) if result_path_for(config) else None,
        "updated_at": dt.datetime.now().isoformat(),
    }
    write_json(status_path, status)
    return status


def cmd_run_one(args: argparse.Namespace) -> int:
    results_root = resolve_results_root(args.results_root)
    config = load_config(results_root, args.config_id)
    status = run_config(results_root, config)
    print(json.dumps(status, indent=2))
    return 0 if status["state"] == "SUCCESS" else 1


def cmd_run_all(args: argparse.Namespace) -> int:
    results_root = resolve_results_root(args.results_root)
    configs = [config for config in load_configs(results_root) if int(config["config_id"]) >= args.start_at]
    failed = 0
    for config in configs:
        print(f"Running config_{config['config_id']:04d}: {config['experiment_family']} {config['model_order']}")
        status = run_config(results_root, config)
        print(json.dumps(status, indent=2))
        if status["state"] != "SUCCESS":
            failed += 1
            break
    return 0 if failed == 0 else 1


def collect_summary(results_root: Path) -> List[Dict]:
    rows: List[Dict] = []
    for config in load_configs(results_root):
        result_path = result_path_for(config)
        row = {
            "config_id": config["config_id"],
            "experiment_family": config["experiment_family"],
            "model_order": config["model_order"],
            "adversary_model": config.get("adversary_model"),
            "adversary_position": config.get("adversary_position"),
            "models": config["models"],
            "status": "SUCCESS" if result_path is not None else "MISSING",
            "result_path": str(result_path) if result_path else None,
        }
        if result_path is not None:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            row["consensus_reached"] = bool(payload.get("consensus_reached", False))
            row["final_round"] = payload.get("final_round")
            row["final_utilities"] = payload.get("final_utilities")
            row["pairwise_cosine_summary"] = payload.get("config", {}).get("pairwise_cosine_summary")
        rows.append(row)
    return rows


def cmd_summary(args: argparse.Namespace) -> int:
    results_root = resolve_results_root(args.results_root)
    rows = collect_summary(results_root)
    if args.json:
        print(json.dumps(rows, indent=2, default=str))
    else:
        for row in rows:
            print(
                f"config_{row['config_id']:04d} {row['status']} "
                f"{row['experiment_family']} {row['model_order']} "
                f"consensus={row.get('consensus_reached')} round={row.get('final_round')}"
            )
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    results_root = resolve_results_root(args.results_root)
    configs = load_configs(results_root)
    if not configs:
        raise FileNotFoundError(f"No configs found under {results_root / 'configs'}")
    sbatch_path = write_slurm_file(results_root, args.max_concurrent, args.time)
    first_id = min(int(config["config_id"]) for config in configs)
    last_id = max(int(config["config_id"]) for config in configs)
    array_spec = f"{first_id}-{last_id}%{args.max_concurrent}"
    output = subprocess.check_output(
        ["sbatch", "--export=ALL", "--array", array_spec, str(sbatch_path)],
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    print(output.strip())
    print(f"Submitted config IDs {first_id}-{last_id} with max_concurrent={args.max_concurrent}")
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "generate":
        return cmd_generate(args)
    if args.command == "run-one":
        return cmd_run_one(args)
    if args.command == "run-all":
        return cmd_run_all(args)
    if args.command == "summary":
        return cmd_summary(args)
    if args.command == "submit":
        return cmd_submit(args)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

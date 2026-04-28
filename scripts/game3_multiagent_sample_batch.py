#!/usr/bin/env python3
"""Generate and run the approved Game 3 N=10 sample configs."""

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
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strong_models_experiment.analysis.active_model_roster import (  # noqa: E402
    context_filtered_model_pool,
    elo_for_model,
    quantile_elo_bucket_map,
)


BASELINE_MODEL = "gpt-5-nano"
SAMPLE_ADVERSARIES = ["claude-opus-4-6-thinking", "amazon-nova-micro-v1.0"]
HETEROGENEOUS_RUNTIME_EXCLUDED_MODELS = {
    # Confirmed fail-fast issue during Game 3 sample setup: Anthropic returns
    # 404 model_not_found for this retired native route.
    "claude-3-haiku-20240307",
    # Confirmed fail-fast issue in the heterogeneous sample: repeated request
    # timeouts with no new interactions after the preceding qwen3-max turn.
    "qwq-32b",
}
DEFAULT_N = 10
DEFAULT_M_PROJECTS = 25
DEFAULT_ALPHA = 0.1
DEFAULT_SIGMA = 0.2
DEFAULT_C_MIN = 10.0
DEFAULT_C_MAX = 30.0
DEFAULT_MAX_ROUNDS = 10
DEFAULT_DISCUSSION_TURNS = 2
DEFAULT_TIME_DISCOUNT = 0.9
DEFAULT_MAX_TOKENS_PER_PHASE = 10500
DEFAULT_SAMPLE_SEED = 37000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and run Game 3 multi-agent sample configs."
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
        / f"game3_multiagent_sample_{timestamp}"
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
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def create_latest_symlink(results_root: Path) -> None:
    symlink = results_root.parent / "game3_multiagent_sample_latest"
    if symlink.is_symlink() or symlink.exists():
        symlink.unlink()
    symlink.symlink_to(results_root.name)


def agent_ids(n_agents: int) -> List[str]:
    return [f"Agent_{index}" for index in range(1, n_agents + 1)]


def build_agent_maps(models: List[str], adversary_model: Optional[str] = None) -> Dict[str, Dict]:
    ids = agent_ids(len(models))
    agent_model_map = dict(zip(ids, models))
    agent_elo_map = {
        agent_id: elo_for_model(model)
        for agent_id, model in agent_model_map.items()
    }
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
        "experiment_id": config_id,
        "experiment_family": experiment_family,
        "experiment_type": experiment_family,
        "game_type": "co_funding",
        "baseline_model": BASELINE_MODEL,
        "models": models,
        "model1": BASELINE_MODEL if adversary_model else "heterogeneous_pool",
        "model2": adversary_model if adversary_model else "random_10_models",
        "model_order": model_order,
        "n_agents": n_agents,
        "m_projects": DEFAULT_M_PROJECTS,
        "alpha": DEFAULT_ALPHA,
        "sigma": DEFAULT_SIGMA,
        "c_min": DEFAULT_C_MIN,
        "c_max": DEFAULT_C_MAX,
        "max_rounds": DEFAULT_MAX_ROUNDS,
        "discussion_turns": DEFAULT_DISCUSSION_TURNS,
        "cofunding_discussion_transparency": "own",
        "cofunding_enable_commit_vote": True,
        "cofunding_enable_time_discount": True,
        "cofunding_time_discount": DEFAULT_TIME_DISCOUNT,
        "max_tokens_per_phase": DEFAULT_MAX_TOKENS_PER_PHASE,
        "random_seed": random_seed,
        "seed": random_seed,
        "run_number": 1,
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
                random_seed=seed + config_id,
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
                random_seed=seed + config_id,
                adversary_model=adversary,
                adversary_position="last",
            )
        )
        config_id += 1

    model_pool = [
        model
        for model in context_filtered_model_pool()
        if model not in HETEROGENEOUS_RUNTIME_EXCLUDED_MODELS
    ]
    rng = random.Random(seed + 999)
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
            random_seed=seed + config_id,
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
        "experiment_id",
        "experiment_type",
        "model1",
        "model2",
        "model_order",
        "run_number",
        "seed",
        "adversary_model",
        "adversary_position",
        "n_agents",
        "m_projects",
        "alpha",
        "sigma",
        "models",
        "output_dir",
        "config_file",
    ]
    with (config_dir / "experiment_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for config in configs:
            row = {
                key: config.get(key)
                for key in fieldnames
                if key not in {"models", "config_file"}
            }
            row["models"] = "+".join(config["models"])
            row["config_file"] = f"config_{config['config_id']:04d}.json"
            writer.writerow(row)


def cmd_generate(args: argparse.Namespace) -> int:
    results_root = latest_root_or_new(args.results_root)
    configs = build_sample_configs(results_root, seed=args.seed)
    canonical_model_pool = list(context_filtered_model_pool())
    model_pool = [
        model
        for model in canonical_model_pool
        if model not in HETEROGENEOUS_RUNTIME_EXCLUDED_MODELS
    ]
    manifest = {
        "results_root": str(results_root),
        "batch_type": "game3_multiagent_sample",
        "game_type": "co_funding",
        "baseline_model": BASELINE_MODEL,
        "sample_adversaries": SAMPLE_ADVERSARIES,
        "canonical_context_filtered_model_pool": canonical_model_pool,
        "canonical_context_filtered_model_pool_count": len(canonical_model_pool),
        "context_filtered_model_pool": model_pool,
        "context_filtered_model_pool_count": len(model_pool),
        "heterogeneous_runtime_excluded_models": sorted(HETEROGENEOUS_RUNTIME_EXCLUDED_MODELS),
        "n_agents": DEFAULT_N,
        "m_projects": DEFAULT_M_PROJECTS,
        "alpha": DEFAULT_ALPHA,
        "sigma": DEFAULT_SIGMA,
        "c_min": DEFAULT_C_MIN,
        "c_max": DEFAULT_C_MAX,
        "max_rounds": DEFAULT_MAX_ROUNDS,
        "discussion_turns": DEFAULT_DISCUSSION_TURNS,
        "cofunding_discussion_transparency": "own",
        "cofunding_enable_commit_vote": True,
        "cofunding_enable_time_discount": True,
        "cofunding_time_discount": DEFAULT_TIME_DISCOUNT,
        "max_tokens_per_phase": DEFAULT_MAX_TOKENS_PER_PHASE,
        "sample_seed": args.seed,
        "config_count": len(configs),
        "notes": "Approved Game 3 fail-fast samples: Opus/Nova adversary first and last, plus one true-random heterogeneous sample.",
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
    return [
        sys.executable,
        str(PROJECT_ROOT / "run_strong_models_experiment.py"),
        "--models",
        *config["models"],
        "--game-type",
        "co_funding",
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
        "--max-rounds",
        str(config["max_rounds"]),
        "--discussion-turns",
        str(config["discussion_turns"]),
        "--cofunding-discussion-transparency",
        str(config["cofunding_discussion_transparency"]),
        "--cofunding-time-discount",
        str(config["cofunding_time_discount"]),
        "--max-tokens-per-phase",
        str(config["max_tokens_per_phase"]),
        "--model-order",
        str(config["model_order"]),
        "--random-seed",
        str(config["random_seed"]),
        "--output-dir",
        str(output_dir),
        "--job-id",
        str(config["config_id"]),
        "--run-number",
        str(config["run_number"]),
    ]


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
                "experiment_id",
                "experiment_family",
                "experiment_type",
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
                "EXPERIMENT_EXTERNALIZE_PROMPTS": "1",
                "NEGOTIATION_CONTEXT_HISTORY_MAX_ENTRIES": "40",
                "NEGOTIATION_CONTEXT_HISTORY_MAX_CHARS": "45000",
                "NEGOTIATION_STRATEGIC_NOTES_MAX_ENTRIES": "4",
                "NEGOTIATION_STRATEGIC_NOTES_MAX_CHARS": "20000",
                "LLM_FAILURE_REPORT_PATH": str(results_root / "monitoring" / "provider_failures.md"),
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
    configs = [
        config
        for config in load_configs(results_root)
        if int(config["config_id"]) >= args.start_at
    ]
    failed = 0
    for config in configs:
        print(
            f"Running config_{config['config_id']:04d}: "
            f"{config['experiment_family']} {config['model_order']}",
            flush=True,
        )
        status = run_config(results_root, config)
        print(json.dumps(status, indent=2), flush=True)
        if status["state"] != "SUCCESS":
            failed += 1
    return 0 if failed == 0 else 1


def collect_summary(results_root: Path) -> List[Dict]:
    rows: List[Dict] = []
    for config in load_configs(results_root):
        result_path = result_path_for(config)
        status_path = results_root / "status" / f"config_{config['config_id']:04d}.json"
        status_payload = {}
        if status_path.exists():
            status_payload = json.loads(status_path.read_text(encoding="utf-8"))
        state = status_payload.get("state")
        row = {
            "config_id": config["config_id"],
            "experiment_family": config["experiment_family"],
            "model_order": config["model_order"],
            "adversary_model": config.get("adversary_model"),
            "adversary_position": config.get("adversary_position"),
            "models": config["models"],
            "status": "SUCCESS" if result_path is not None else (state or "MISSING"),
            "returncode": status_payload.get("returncode"),
            "result_path": str(result_path) if result_path else None,
        }
        if result_path is not None:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            row["consensus_reached"] = bool(payload.get("consensus_reached", False))
            row["final_round"] = payload.get("final_round")
            row["final_utilities"] = payload.get("final_utilities")
            row["funded_projects"] = payload.get("final_allocation")
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
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

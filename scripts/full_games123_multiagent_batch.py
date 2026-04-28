#!/usr/bin/env python3
"""Generate, submit, run, summarize, and report the full Games 1-3 batch."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import shlex
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strong_models_experiment.analysis.active_model_roster import (  # noqa: E402
    context_filtered_model_pool,
    elo_for_model,
    quantile_elo_bucket_map,
)


BASELINE_MODEL = "gpt-5-nano"
ADVERSARY_MODELS = [
    "amazon-nova-micro-v1.0",
    "gpt-4o-mini-2024-07-18",
    "claude-sonnet-4-20250514",
    "gemini-2.5-pro",
    "claude-opus-4-6-thinking",
]
N_VALUES = [2, 4, 6, 8, 10]
MAX_AGENT_COLUMNS = max(N_VALUES)
GAME1_COMPETITION_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
GAME2_THETAS = [0.2, 0.8]
GAME3_SIGMAS = [0.2, 0.5]
GAME3_ALPHAS = [0.2, 0.8]
HOMOGENEOUS_SEED_REPLICATES = [1, 2]
HETEROGENEOUS_RUNS_PER_CELL = 20
HETEROGENEOUS_EXCLUDED_MODELS = {"qwq-32b"}
ELO_BUCKET_COUNT = 3
MAX_ROUNDS = 10
DISCUSSION_TURNS = 2
GAMMA_DISCOUNT = 0.9
MAX_TOKENS_VOTING = None
SLURM_TIME = "08:00:00"
MAX_CONCURRENT = 50
MASTER_SEED = 20260427
JOB_NAME = "fg123"
MAX_ARRAY_TASKS_PER_SUBMISSION = 2000
EXCLUDED_MODEL_REFERENCE_PATTERNS = tuple(
    sorted({
        model.lower()
        for model in HETEROGENEOUS_EXCLUDED_MODELS
    } | {
        model.split("-", 1)[0].lower()
        for model in HETEROGENEOUS_EXCLUDED_MODELS
    })
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--results-root", type=str, default=None)
    generate.add_argument("--master-seed", type=int, default=MASTER_SEED)
    generate.add_argument("--slurm-time", type=str, default=SLURM_TIME)
    generate.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    generate.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    generate.add_argument("--discussion-turns", type=int, default=DISCUSSION_TURNS)
    generate.add_argument("--max-tokens-voting", type=int, default=MAX_TOKENS_VOTING)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--results-root", type=str, required=True)

    run_one = subparsers.add_parser("run-one")
    run_one.add_argument("--results-root", type=str, required=True)
    run_one.add_argument("--config-id", type=int, required=True)

    submit = subparsers.add_parser("submit")
    submit.add_argument("--results-root", type=str, required=True)

    summary = subparsers.add_parser("summary")
    summary.add_argument("--results-root", type=str, required=True)
    summary.add_argument("--json", action="store_true")

    report = subparsers.add_parser("report")
    report.add_argument("--results-root", type=str, required=True)

    return parser.parse_args()


def timestamp_now() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def latest_root_or_new(raw_value: Optional[str]) -> Path:
    if raw_value:
        path = Path(raw_value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()
    return (
        PROJECT_ROOT
        / "experiments"
        / "results"
        / f"full_games123_multiagent_{timestamp_now()}"
    ).resolve()


def resolve_results_root(raw_value: str) -> Path:
    path = Path(raw_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def sanitize_token(value: Any) -> str:
    token = str(value).replace(".", "p").replace("-", "_").replace("/", "_")
    return re.sub(r"[^A-Za-z0-9_]+", "_", token).strip("_")


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def config_output_dir(config: Dict[str, Any]) -> Path:
    path = Path(config["output_dir"])
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def read_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def config_status_path(results_root: Path, config_id: int) -> Path:
    return results_root / "status" / f"config_{config_id:04d}.json"


def config_latest_log_path(results_root: Path, config_id: int) -> Path:
    return results_root / "logs" / f"config_{config_id:04d}.log"


def config_attempt_log_path(results_root: Path, config_id: int, start: dt.datetime) -> Tuple[Path, str]:
    job_id = os.getenv("SLURM_JOB_ID", "nojob")
    task_id = os.getenv("SLURM_ARRAY_TASK_ID", "notask")
    timestamp = start.strftime("%Y%m%dT%H%M%S_%f")
    attempt_id = f"{timestamp}_job{sanitize_token(job_id)}_task{sanitize_token(task_id)}_pid{os.getpid()}"
    return results_root / "logs" / f"config_{config_id:04d}_attempt_{attempt_id}.log", attempt_id


def legacy_attempts_from_status(status: Dict[str, Any]) -> List[Dict[str, Any]]:
    attempts = list(status.get("attempts") or [])
    if attempts or not status:
        return attempts
    if status.get("started_at") or status.get("log_path"):
        attempts.append(
            {
                "attempt_id": status.get("attempt_id", "legacy"),
                "state": status.get("state"),
                "returncode": status.get("returncode"),
                "started_at": status.get("started_at"),
                "finished_at": status.get("finished_at"),
                "duration_seconds": status.get("duration_seconds"),
                "log_path": status.get("attempt_log_path") or status.get("log_path"),
                "result_path": status.get("result_path"),
            }
        )
    return attempts


def update_latest_log_pointer(latest_path: Path, attempt_path: Path) -> None:
    """Point logs/config_XXXX.log at the newest attempt without overwriting attempts."""
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    if latest_path.exists() and not latest_path.is_symlink():
        try:
            legacy_name = (
                latest_path.parent
                / f"{latest_path.stem}_attempt_legacy_{int(latest_path.stat().st_mtime)}{latest_path.suffix}"
            )
            if not legacy_name.exists():
                latest_path.rename(legacy_name)
            else:
                latest_path.unlink()
        except OSError:
            latest_path.unlink(missing_ok=True)
    elif latest_path.is_symlink():
        latest_path.unlink()

    try:
        latest_path.symlink_to(attempt_path)
    except OSError:
        latest_path.write_text(
            f"Latest attempt log: {attempt_path}\n",
            encoding="utf-8",
        )


def rho_lower_bound(n_agents: int) -> float:
    return (6.0 / math.pi) * math.asin(-1.0 / (2.0 * (n_agents - 1)))


def stable_seed(master_seed: int, *parts: Any) -> int:
    raw = "|".join([str(master_seed), *(str(part) for part in parts)]).encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def agent_ids(n_agents: int) -> List[str]:
    return [f"Agent_{idx}" for idx in range(1, n_agents + 1)]


def build_agent_maps(models: List[str], adversary_model: Optional[str], family: str) -> Dict[str, Dict[str, Any]]:
    ids = agent_ids(len(models))
    agent_model_map = dict(zip(ids, models))
    agent_elo_map = {agent_id: elo_for_model(model) for agent_id, model in agent_model_map.items()}
    if family == "homogeneous_control":
        agent_role_map = {agent_id: "baseline_control" for agent_id in ids}
    elif family == "homogeneous_adversary":
        agent_role_map = {
            agent_id: ("adversary" if model == adversary_model else "baseline")
            for agent_id, model in agent_model_map.items()
        }
    else:
        agent_role_map = {agent_id: "heterogeneous_random_agent" for agent_id in ids}
    return {
        "agent_model_map": agent_model_map,
        "agent_elo_map": agent_elo_map,
        "agent_role_map": agent_role_map,
    }


def filtered_heterogeneous_pool() -> List[str]:
    pool = [
        model
        for model in context_filtered_model_pool()
        if model not in HETEROGENEOUS_EXCLUDED_MODELS
    ]
    if len(pool) != 24:
        raise ValueError(f"Expected 24 heterogeneous models after exclusions, got {len(pool)}")
    return pool


def find_excluded_model_references(value: Any, path: str = "$") -> List[str]:
    """Find recursive references to excluded heterogeneous models in generated payloads."""
    matches: List[str] = []
    if isinstance(value, str):
        lowered = value.lower()
        if any(pattern and pattern in lowered for pattern in EXCLUDED_MODEL_REFERENCE_PATTERNS):
            matches.append(path)
        return matches
    if isinstance(value, dict):
        for key, child in value.items():
            key_path = f"{path}.{key}"
            matches.extend(find_excluded_model_references(str(key), f"{key_path}<key>"))
            matches.extend(find_excluded_model_references(child, key_path))
        return matches
    if isinstance(value, list):
        for index, child in enumerate(value):
            matches.extend(find_excluded_model_references(child, f"{path}[{index}]"))
    return matches


def game_parameter_grid(game_label: str, n_agents: int) -> List[Dict[str, Any]]:
    if game_label == "game1":
        return [
            {
                "competition_id": f"comp_{sanitize_token(level)}",
                "competition_level": level,
                "num_items": int(2.5 * n_agents),
            }
            for level in GAME1_COMPETITION_LEVELS
        ]
    if game_label == "game2":
        return [
            {
                "competition_id": (
                    f"rho_{sanitize_token(rho)}_theta_{sanitize_token(theta)}"
                ),
                "n_issues": 10,
                "rho": rho,
                "theta": theta,
                "rho_label": "negative_lower_bound" if rho < 0 else "high_alignment",
            }
            for rho in [rho_lower_bound(n_agents), 0.9]
            for theta in GAME2_THETAS
        ]
    if game_label == "game3":
        return [
            {
                "competition_id": (
                    f"sigma_{sanitize_token(sigma)}_alpha_{sanitize_token(alpha)}"
                ),
                "m_projects": int(2.5 * n_agents),
                "alpha": alpha,
                "sigma": sigma,
                "c_min": 10.0,
                "c_max": 30.0,
                "cofunding_discussion_transparency": "own",
                "cofunding_enable_commit_vote": True,
                "cofunding_enable_time_discount": True,
                "cofunding_time_discount": GAMMA_DISCOUNT,
            }
            for sigma in GAME3_SIGMAS
            for alpha in GAME3_ALPHAS
        ]
    raise ValueError(f"Unknown game label: {game_label}")


def game_type_for_label(game_label: str) -> str:
    return {
        "game1": "item_allocation",
        "game2": "diplomacy",
        "game3": "co_funding",
    }[game_label]


def common_config(
    *,
    config_id: int,
    results_root: Path,
    game_label: str,
    family: str,
    models: List[str],
    model_order: str,
    random_seed: int,
    max_rounds: int,
    discussion_turns: int,
    max_tokens_voting: Optional[int],
    game_params: Dict[str, Any],
    seed_replicate: Optional[int] = None,
    heterogeneous_run_index: Optional[int] = None,
    adversary_model: Optional[str] = None,
    adversary_position: Optional[str] = None,
    heterogeneous_model_pool: Optional[List[str]] = None,
    heterogeneous_draw_seed: Optional[int] = None,
    elo_bucket_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    n_agents = len(models)
    name_parts = [
        f"config_{config_id:04d}",
        game_label,
        family,
        f"n{n_agents}",
        game_params["competition_id"],
    ]
    if adversary_model:
        name_parts.extend([sanitize_token(adversary_model), str(adversary_position)])
    if seed_replicate is not None:
        name_parts.append(f"seed{seed_replicate}")
    if heterogeneous_run_index is not None:
        name_parts.append(f"run{heterogeneous_run_index:02d}")
    output_dir = results_root / "runs" / "_".join(name_parts)

    config: Dict[str, Any] = {
        "config_id": config_id,
        "experiment_id": config_id,
        "batch_type": "full_games123_multiagent",
        "experiment_family": family,
        "experiment_type": family,
        "game_label": game_label,
        "game_type": game_type_for_label(game_label),
        "competition_id": game_params["competition_id"],
        "baseline_model": BASELINE_MODEL,
        "models": models,
        "model_order": model_order,
        "n_agents": n_agents,
        "max_rounds": int(max_rounds),
        "discussion_turns": int(discussion_turns),
        "gamma_discount": GAMMA_DISCOUNT,
        "parallel_phases": True,
        "random_seed": int(random_seed),
        "seed": int(random_seed),
        "seed_replicate": seed_replicate,
        "run_number": 1,
        "heterogeneous_run_index": heterogeneous_run_index,
        "adversary_model": adversary_model,
        "adversary_position": adversary_position,
        "output_dir": relative_or_absolute(output_dir),
        **build_agent_maps(models, adversary_model, family),
    }
    if max_tokens_voting is not None:
        config["max_tokens_voting"] = int(max_tokens_voting)
    config.update({key: value for key, value in game_params.items() if key != "competition_id"})

    if heterogeneous_model_pool is not None:
        config["model_pool"] = heterogeneous_model_pool
        config["model_pool_size"] = len(heterogeneous_model_pool)
        config["heterogeneous_excluded_model_count"] = len(HETEROGENEOUS_EXCLUDED_MODELS)
        config["heterogeneous_sampling"] = {
            "without_replacement_within_run": True,
            "with_replacement_across_runs": True,
            "random_order": True,
        }
    if heterogeneous_draw_seed is not None:
        config["heterogeneous_draw_seed"] = int(heterogeneous_draw_seed)
    if elo_bucket_map is not None:
        config["elo_bucket_method"] = "quantile_terciles_after_explicit_exclusions"
        config["model_elo_bucket_map"] = elo_bucket_map
        config["agent_elo_bucket_map"] = {
            agent_id: elo_bucket_map.get(model)
            for agent_id, model in config["agent_model_map"].items()
        }
    return config


def build_configs(
    *,
    results_root: Path,
    master_seed: int,
    max_rounds: int,
    discussion_turns: int,
    max_tokens_voting: Optional[int],
) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    config_id = 1
    hetero_pool = filtered_heterogeneous_pool()
    elo_bucket_map = quantile_elo_bucket_map(hetero_pool, n_buckets=ELO_BUCKET_COUNT)

    for game_label in ["game1", "game2", "game3"]:
        for n_agents in N_VALUES:
            for game_params in game_parameter_grid(game_label, n_agents):
                for seed_replicate in HOMOGENEOUS_SEED_REPLICATES:
                    random_seed = stable_seed(
                        master_seed,
                        "homogeneous_control",
                        game_label,
                        n_agents,
                        game_params["competition_id"],
                        seed_replicate,
                    )
                    configs.append(
                        common_config(
                            config_id=config_id,
                            results_root=results_root,
                            game_label=game_label,
                            family="homogeneous_control",
                            models=[BASELINE_MODEL] * n_agents,
                            model_order="homogeneous_control",
                            random_seed=random_seed,
                            max_rounds=max_rounds,
                            discussion_turns=discussion_turns,
                            max_tokens_voting=max_tokens_voting,
                            game_params=game_params,
                            seed_replicate=seed_replicate,
                        )
                    )
                    config_id += 1

                for adversary in ADVERSARY_MODELS:
                    for position in ["first", "last"]:
                        for seed_replicate in HOMOGENEOUS_SEED_REPLICATES:
                            models = (
                                [adversary] + [BASELINE_MODEL] * (n_agents - 1)
                                if position == "first"
                                else [BASELINE_MODEL] * (n_agents - 1) + [adversary]
                            )
                            random_seed = stable_seed(
                                master_seed,
                                "homogeneous_adversary",
                                game_label,
                                n_agents,
                                game_params["competition_id"],
                                adversary,
                                position,
                                seed_replicate,
                            )
                            configs.append(
                                common_config(
                                    config_id=config_id,
                                    results_root=results_root,
                                    game_label=game_label,
                                    family="homogeneous_adversary",
                                    models=models,
                                    model_order=f"adversary_{position}",
                                    random_seed=random_seed,
                                    max_rounds=max_rounds,
                                    discussion_turns=discussion_turns,
                                    max_tokens_voting=max_tokens_voting,
                                    game_params=game_params,
                                    seed_replicate=seed_replicate,
                                    adversary_model=adversary,
                                    adversary_position=position,
                                )
                            )
                            config_id += 1

                for run_index in range(1, HETEROGENEOUS_RUNS_PER_CELL + 1):
                    draw_seed = stable_seed(
                        master_seed,
                        "heterogeneous_draw",
                        game_label,
                        n_agents,
                        game_params["competition_id"],
                        run_index,
                    )
                    rng = random.Random(draw_seed)
                    models = rng.sample(hetero_pool, n_agents)
                    rng.shuffle(models)
                    random_seed = stable_seed(
                        master_seed,
                        "heterogeneous_environment",
                        game_label,
                        n_agents,
                        game_params["competition_id"],
                        run_index,
                    )
                    configs.append(
                        common_config(
                            config_id=config_id,
                            results_root=results_root,
                            game_label=game_label,
                            family="heterogeneous_random",
                            models=models,
                            model_order="sampled_random_order",
                            random_seed=random_seed,
                            max_rounds=max_rounds,
                            discussion_turns=discussion_turns,
                            max_tokens_voting=max_tokens_voting,
                            game_params=game_params,
                            heterogeneous_run_index=run_index,
                            heterogeneous_model_pool=hetero_pool,
                            heterogeneous_draw_seed=draw_seed,
                            elo_bucket_map=elo_bucket_map,
                        )
                    )
                    config_id += 1

    return configs


def write_generated_files(results_root: Path, manifest: Dict[str, Any], configs: List[Dict[str, Any]]) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    config_dir = results_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (results_root / "logs").mkdir(exist_ok=True)
    (results_root / "status").mkdir(exist_ok=True)
    (results_root / "slurm").mkdir(exist_ok=True)

    write_json(results_root / "manifest.json", manifest)
    for config in configs:
        write_json(config_dir / f"config_{config['config_id']:04d}.json", config)

    (config_dir / "all_configs.txt").write_text(
        "\n".join(str(config_dir / f"config_{config['config_id']:04d}.json") for config in configs) + "\n",
        encoding="utf-8",
    )

    agent_fieldnames = []
    for agent_idx in range(1, MAX_AGENT_COLUMNS + 1):
        agent_fieldnames.extend([
            f"agent_{agent_idx}_id",
            f"agent_{agent_idx}_model",
            f"agent_{agent_idx}_elo",
            f"agent_{agent_idx}_elo_bucket",
            f"agent_{agent_idx}_role",
        ])

    fieldnames = [
        "config_id", "game_label", "game_type", "experiment_family", "competition_id",
        "n_agents", "num_items", "n_issues", "m_projects", "competition_level",
        "rho", "theta", "sigma", "alpha", "adversary_model", "adversary_position",
        "seed_replicate", "random_seed", "seed", "model_order", "models",
        "heterogeneous_run_index", "heterogeneous_draw_seed", "model_pool_size",
        "model_pool", "heterogeneous_sampling", "elo_bucket_method",
        "output_dir",
    ] + agent_fieldnames
    with (config_dir / "experiment_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for config in configs:
            row = {field: config.get(field) for field in fieldnames if field != "models"}
            row["models"] = "+".join(config["models"])
            model_pool = config.get("model_pool") or []
            row["model_pool"] = "+".join(model_pool)
            heterogeneous_sampling = config.get("heterogeneous_sampling")
            row["heterogeneous_sampling"] = (
                json.dumps(heterogeneous_sampling, sort_keys=True)
                if heterogeneous_sampling
                else ""
            )
            for agent_idx in range(1, MAX_AGENT_COLUMNS + 1):
                agent_id = f"Agent_{agent_idx}"
                active_agent = agent_idx <= int(config["n_agents"])
                row[f"agent_{agent_idx}_id"] = agent_id if active_agent else ""
                row[f"agent_{agent_idx}_model"] = config.get("agent_model_map", {}).get(agent_id, "")
                row[f"agent_{agent_idx}_elo"] = config.get("agent_elo_map", {}).get(agent_id, "")
                row[f"agent_{agent_idx}_elo_bucket"] = config.get("agent_elo_bucket_map", {}).get(agent_id, "")
                row[f"agent_{agent_idx}_role"] = config.get("agent_role_map", {}).get(agent_id, "")
            writer.writerow(row)

    latest = results_root.parent / "full_games123_multiagent_latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(results_root.name)


def load_configs(results_root: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((results_root / "configs").glob("config_*.json"))
    ]


def load_config(results_root: Path, config_id: int) -> Dict[str, Any]:
    path = results_root / "configs" / f"config_{config_id:04d}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def result_path_for(config: Dict[str, Any]) -> Optional[Path]:
    output_dir = config_output_dir(config)
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
    if not python_bin.exists():
        python_bin = Path(sys.executable)
    output_dir = config_output_dir(config)
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
        str(config["model_order"]),
        "--random-seed",
        str(config["random_seed"]),
        "--output-dir",
        str(output_dir),
        "--job-id",
        str(config["config_id"]),
        "--run-number",
        str(config.get("run_number", 1)),
        "--gamma-discount",
        str(config.get("gamma_discount", GAMMA_DISCOUNT)),
        "--parallel-phases",
    ]
    if config.get("max_tokens_voting") is not None:
        cmd.extend(["--max-tokens-voting", str(config["max_tokens_voting"])])
    if config["game_type"] == "item_allocation":
        cmd.extend([
            "--num-items",
            str(config["num_items"]),
            "--competition-level",
            str(config["competition_level"]),
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
        if not config.get("cofunding_enable_commit_vote", True):
            cmd.append("--cofunding-disable-commit-vote")
        if not config.get("cofunding_enable_time_discount", True):
            cmd.append("--cofunding-disable-time-discount")
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
        "config_id", "batch_type", "experiment_family", "experiment_type",
        "game_label", "game_type", "competition_id", "baseline_model",
        "models", "agent_model_map", "agent_elo_map", "agent_role_map",
        "model_order", "adversary_model", "adversary_position",
        "seed_replicate", "heterogeneous_run_index", "heterogeneous_draw_seed",
        "model_pool", "model_pool_size",
        "heterogeneous_sampling", "heterogeneous_excluded_model_count",
        "elo_bucket_method", "model_elo_bucket_map",
        "agent_elo_bucket_map", "max_tokens_voting", "parallel_phases",
    ]
    payload["config"].update({key: config[key] for key in metadata_keys if key in config})
    result_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def run_config(results_root: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    output_dir = config_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    monitoring_dir = results_root / "monitoring"
    monitoring_dir.mkdir(parents=True, exist_ok=True)
    config_id = int(config["config_id"])
    (results_root / "logs").mkdir(parents=True, exist_ok=True)
    latest_log_path = config_latest_log_path(results_root, config_id)
    status_path = config_status_path(results_root, config_id)
    previous_status = read_json_file(status_path)
    attempts = legacy_attempts_from_status(previous_status)

    existing = result_path_for(config)
    if existing is not None:
        enrich_result_metadata(config)
        status = {
            "config_id": config_id,
            "state": "SUCCESS",
            "returncode": 0,
            "skipped_existing": True,
            "result_path": str(existing),
            "log_path": str(latest_log_path),
            "attempts": attempts,
            "updated_at": dt.datetime.now().isoformat(),
        }
        write_json(status_path, status)
        return status

    cmd = build_command(config)
    start = dt.datetime.now()
    log_path, attempt_id = config_attempt_log_path(results_root, config_id, start)
    update_latest_log_pointer(latest_log_path, log_path)
    attempt_record = {
        "attempt_id": attempt_id,
        "state": "RUNNING",
        "started_at": start.isoformat(),
        "log_path": str(log_path),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "slurm_array_job_id": os.getenv("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"),
        "pid": os.getpid(),
    }
    attempts.append(attempt_record)
    write_json(
        status_path,
        {
            "config_id": config_id,
            "state": "RUNNING",
            "attempt_id": attempt_id,
            "game_label": config["game_label"],
            "experiment_family": config["experiment_family"],
            "command": shlex.join(cmd),
            "started_at": start.isoformat(),
            "log_path": str(latest_log_path),
            "attempt_log_path": str(log_path),
            "attempts": attempts,
        },
    )

    env = load_dotenv_env()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "OPENROUTER_TRANSPORT": env.get("OPENROUTER_TRANSPORT", "proxy"),
            "OPENROUTER_PROXY_POLL_DIR": env.get(
                "OPENROUTER_PROXY_POLL_DIR",
                "/home/jz4391/openrouter_proxy",
            ),
            "OPENROUTER_PROXY_CLIENT_TIMEOUT": env.get("OPENROUTER_PROXY_CLIENT_TIMEOUT", "9000"),
            "OPENROUTER_API_TIMEOUT": env.get("OPENROUTER_API_TIMEOUT", "1800"),
            "LLM_FAILURE_REPORT_PATH": env.get(
                "LLM_FAILURE_REPORT_PATH",
                str(monitoring_dir / "provider_failures.md"),
            ),
        }
    )
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Started at {start.isoformat()}\n")
        handle.write(f"Config: config_{config['config_id']:04d}\n")
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

    end = dt.datetime.now()
    result_path = result_path_for(config)
    state = "SUCCESS" if returncode == 0 and result_path is not None else "FAILED"
    if state == "SUCCESS":
        enrich_result_metadata(config)
    attempts[-1].update(
        {
            "state": state,
            "returncode": returncode,
            "finished_at": end.isoformat(),
            "duration_seconds": (end - start).total_seconds(),
            "result_path": str(result_path) if result_path else None,
        }
    )
    status = {
        "config_id": config_id,
        "state": state,
        "returncode": returncode,
        "attempt_id": attempt_id,
        "started_at": start.isoformat(),
        "finished_at": end.isoformat(),
        "duration_seconds": (end - start).total_seconds(),
        "log_path": str(latest_log_path),
        "attempt_log_path": str(log_path),
        "result_path": str(result_path) if result_path else None,
        "attempts": attempts,
        "updated_at": end.isoformat(),
    }
    write_json(status_path, status)
    return status


def write_slurm_file(results_root: Path, manifest: Dict[str, Any]) -> Path:
    slurm_dir = results_root / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    python_bin = PROJECT_ROOT / ".venv" / "bin" / "python"
    script_path = PROJECT_ROOT / "scripts" / "full_games123_multiagent_batch.py"
    sbatch_path = slurm_dir / "run_full_games123.sbatch"
    sbatch = f"""#!/bin/bash
#SBATCH --job-name={JOB_NAME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time={manifest['slurm_time']}
#SBATCH --partition=cpu
#SBATCH --output={PROJECT_ROOT}/slurm/full_games123_%A_%a.out
#SBATCH --error={PROJECT_ROOT}/slurm/full_games123_%A_%a.err

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
: "${{SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}}"
CONFIG_OFFSET="${{CONFIG_OFFSET:-0}}"
CONFIG_ID=$((SLURM_ARRAY_TASK_ID + CONFIG_OFFSET))

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


def validate_configs(configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    errors: List[str] = []
    ids = [int(config["config_id"]) for config in configs]
    if ids != list(range(1, len(configs) + 1)):
        errors.append("Config IDs are not contiguous from 1")

    family_counts = Counter(config["experiment_family"] for config in configs)
    game_family_counts = Counter((config["game_label"], config["experiment_family"]) for config in configs)
    expected_game_family = {
        ("game1", "homogeneous_control"): 50,
        ("game1", "homogeneous_adversary"): 500,
        ("game1", "heterogeneous_random"): 500,
        ("game2", "homogeneous_control"): 40,
        ("game2", "homogeneous_adversary"): 400,
        ("game2", "heterogeneous_random"): 400,
        ("game3", "homogeneous_control"): 40,
        ("game3", "homogeneous_adversary"): 400,
        ("game3", "heterogeneous_random"): 400,
    }
    for key, expected in expected_game_family.items():
        actual = game_family_counts.get(key, 0)
        if actual != expected:
            errors.append(f"{key} expected {expected}, got {actual}")
    if len(configs) != 2730:
        errors.append(f"Expected 2730 configs, got {len(configs)}")

    hetero_pool = filtered_heterogeneous_pool()
    if "qwq-32b" in hetero_pool:
        errors.append("qwq-32b remains in heterogeneous pool")

    for config in configs:
        n_agents = int(config["n_agents"])
        if len(config["models"]) != n_agents:
            errors.append(f"config_{config['config_id']:04d}: model count does not match n_agents")
        if not config.get("parallel_phases"):
            errors.append(f"config_{config['config_id']:04d}: parallel_phases is not true")
        if int(config.get("max_rounds", 0)) != 10:
            errors.append(f"config_{config['config_id']:04d}: max_rounds is not 10")
        if int(config.get("discussion_turns", 0)) != 2:
            errors.append(f"config_{config['config_id']:04d}: discussion_turns is not 2")
        if config.get("max_tokens_voting") is not None:
            errors.append(
                f"config_{config['config_id']:04d}: max_tokens_voting must be absent/null "
                "for the full batch"
            )
        excluded_refs = find_excluded_model_references(config)
        if excluded_refs:
            errors.append(
                f"config_{config['config_id']:04d}: contains excluded model reference(s) "
                f"at {excluded_refs[:5]}"
            )

        if config["experiment_family"] == "heterogeneous_random":
            if len(set(config["models"])) != len(config["models"]):
                errors.append(f"config_{config['config_id']:04d}: heterogeneous models are not unique")
            if any(model in HETEROGENEOUS_EXCLUDED_MODELS for model in config["models"]):
                errors.append(f"config_{config['config_id']:04d}: contains excluded heterogeneous model")
            if int(config.get("model_pool_size", 0)) != 24:
                errors.append(f"config_{config['config_id']:04d}: model_pool_size is not 24")

        if config["game_label"] == "game1":
            if int(config["num_items"]) != int(2.5 * n_agents):
                errors.append(f"config_{config['config_id']:04d}: Game 1 num_items mismatch")
            if float(config["competition_level"]) not in GAME1_COMPETITION_LEVELS:
                errors.append(f"config_{config['config_id']:04d}: invalid Game 1 competition")
        elif config["game_label"] == "game2":
            if int(config["n_issues"]) != 10:
                errors.append(f"config_{config['config_id']:04d}: Game 2 n_issues is not 10")
            if float(config["theta"]) not in GAME2_THETAS:
                errors.append(f"config_{config['config_id']:04d}: invalid Game 2 theta")
            if not (math.isclose(float(config["rho"]), rho_lower_bound(n_agents), abs_tol=1e-12) or math.isclose(float(config["rho"]), 0.9)):
                errors.append(f"config_{config['config_id']:04d}: invalid Game 2 rho")
        elif config["game_label"] == "game3":
            if int(config["m_projects"]) != int(2.5 * n_agents):
                errors.append(f"config_{config['config_id']:04d}: Game 3 m_projects mismatch")
            if float(config["sigma"]) not in GAME3_SIGMAS:
                errors.append(f"config_{config['config_id']:04d}: invalid Game 3 sigma")
            if float(config["alpha"]) not in GAME3_ALPHAS:
                errors.append(f"config_{config['config_id']:04d}: invalid Game 3 alpha")

    return {
        "valid": not errors,
        "errors": errors,
        "total": len(configs),
        "family_counts": dict(family_counts),
        "game_family_counts": {f"{game}:{family}": count for (game, family), count in game_family_counts.items()},
        "heterogeneous_pool_count": len(hetero_pool),
        "heterogeneous_pool": hetero_pool,
    }


def parse_job_id(output: str) -> Optional[str]:
    match = re.search(r"Submitted batch job\s+(\d+)", output)
    return match.group(1) if match else None


def read_status(results_root: Path, config_id: int) -> Dict[str, Any]:
    path = results_root / "status" / f"config_{config_id:04d}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"state": "STATUS_PARSE_ERROR"}


def load_submission_records(results_root: Path) -> List[Dict[str, Any]]:
    path = results_root / "submissions.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("job_id"):
        return [{"job_id": str(payload["job_id"]), "config_offset": 0}]
    return [
        {
            "job_id": str(row["job_id"]),
            "config_offset": int(row.get("config_offset", 0)),
        }
        for row in payload.get("submissions", [])
        if row.get("job_id")
    ]


def load_pending_submission_ranges(results_root: Path) -> List[Tuple[int, int]]:
    path = results_root / "submissions.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    ranges = []
    for row in payload.get("pending_submissions", []):
        try:
            ranges.append((int(row["config_start"]), int(row["config_end"])))
        except (KeyError, TypeError, ValueError):
            continue
    return ranges


def is_pending_submission(config_id: int, pending_ranges: Iterable[Tuple[int, int]]) -> bool:
    return any(start <= config_id <= end for start, end in pending_ranges)


def squeue_array_states(submission_records: Iterable[Dict[str, Any]]) -> Dict[int, Dict[str, str]]:
    records = [record for record in submission_records if record.get("job_id")]
    if not records:
        return {}
    states: Dict[int, Dict[str, str]] = {}
    for record in records:
        job_id = str(record["job_id"])
        config_offset = int(record.get("config_offset", 0))
        try:
            output = subprocess.check_output(
                ["squeue", "-r", "-h", "-j", job_id, "-o", "%i|%T|%M|%R|%j"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
        for line in output.splitlines():
            parts = line.split("|", 4)
            if len(parts) != 5:
                continue
            slurm_id, state, elapsed, reason, name = parts
            if "_" not in slurm_id:
                continue
            try:
                task_id = int(slurm_id.rsplit("_", 1)[1])
            except ValueError:
                continue
            config_id = task_id + config_offset
            states[config_id] = {
                "state": state,
                "elapsed": elapsed,
                "reason": reason,
                "name": name,
                "job_id": job_id,
                "task_id": str(task_id),
            }
    return states


def detect_phase(log_path: Path) -> str:
    if not log_path.exists():
        return "not_started"
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "log_unreadable"
    markers = [
        ("Strong models experiment completed successfully", "completed"),
        ("POST-PLEDGE COMMIT VOTE", "cofunding_commit_vote"),
        ("VOTE TABULATION PHASE", "vote_tabulation"),
        ("PRIVATE VOTING PHASE", "voting"),
        ("PROPOSAL ENUMERATION PHASE", "proposal_enumeration"),
        ("PROPOSAL PHASE", "proposal"),
        ("PRIVATE THINKING PHASE", "private_thinking"),
        ("DISCUSSION PHASE", "discussion"),
        ("GAME SETUP PHASE", "setup"),
        ("SETUP PHASE", "setup"),
    ]
    for line in reversed(lines[-500:]):
        for marker, phase in markers:
            if marker in line:
                return phase
    return "starting"


def validate_result_payload(config: Dict[str, Any], result_path: Path) -> Dict[str, Any]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    cfg = payload.get("config", {})
    tabulations = [
        log for log in payload.get("conversation_logs", [])
        if log.get("phase") == "vote_tabulation"
    ]
    return {
        "parallel_phases": bool(cfg.get("parallel_phases")),
        "max_tokens_voting": cfg.get("max_tokens_voting"),
        "pairwise_cosine_summary": cfg.get("pairwise_cosine_summary"),
        "acceptance_rule_seen": any(
            "two-thirds supermajority" in str(log.get("content", "")).lower()
            for log in tabulations
        ),
        "consensus_reached": payload.get("consensus_reached"),
        "final_round": payload.get("final_round"),
    }


def collect_summary(results_root: Path) -> Dict[str, Any]:
    configs = load_configs(results_root)
    submission_records = load_submission_records(results_root)
    pending_ranges = load_pending_submission_ranges(results_root)
    job_ids = [str(record["job_id"]) for record in submission_records]
    queue_states = squeue_array_states(submission_records)
    rows: List[Dict[str, Any]] = []
    counts = Counter()
    breakdown: Dict[str, Counter] = defaultdict(Counter)
    phase_counts = Counter()
    durations: List[float] = []

    for config in configs:
        config_id = int(config["config_id"])
        status = read_status(results_root, config_id)
        result_path = result_path_for(config)
        queue_state = queue_states.get(config_id, {})
        raw_state = status.get("state") or queue_state.get("state") or "NOT_STARTED"

        if result_path is not None and raw_state != "FAILED":
            state = "FINISHED"
        elif raw_state == "RUNNING" or queue_state.get("state") == "RUNNING":
            state = "IN_PROGRESS"
        elif queue_state.get("state") == "PENDING":
            state = "QUEUED"
        elif raw_state in {"FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY", "STATUS_PARSE_ERROR"}:
            state = "ERRORED"
        elif is_pending_submission(config_id, pending_ranges):
            state = "PENDING_SUBMISSION"
        elif job_ids and not status and config_id not in queue_states:
            state = "NOT_STARTED_OR_DISAPPEARED"
        else:
            state = raw_state

        log_path = results_root / "logs" / f"config_{config_id:04d}.log"
        phase = "completed" if state == "FINISHED" else detect_phase(log_path)
        phase_counts[phase] += 1

        if state == "FINISHED" and status.get("duration_seconds") is not None:
            durations.append(float(status["duration_seconds"]))

        key = f"{config['game_label']}:{config['experiment_family']}"
        counts[state] += 1
        breakdown[key][state] += 1

        row = {
            "config_id": config_id,
            "game_label": config["game_label"],
            "experiment_family": config["experiment_family"],
            "state": state,
            "raw_state": raw_state,
            "queue_state": queue_state.get("state"),
            "phase": phase,
            "result_path": str(result_path) if result_path else None,
            "log_path": str(log_path),
        }
        rows.append(row)

    health_samples = []
    for row in rows:
        if len(health_samples) >= 5:
            break
        if row["state"] != "FINISHED" or not row["result_path"]:
            continue
        config = load_config(results_root, int(row["config_id"]))
        try:
            health_samples.append({
                "config_id": row["config_id"],
                "game_label": row["game_label"],
                "experiment_family": row["experiment_family"],
                **validate_result_payload(config, Path(row["result_path"])),
            })
        except Exception as exc:  # noqa: BLE001
            health_samples.append({
                "config_id": row["config_id"],
                "error": str(exc),
            })

    total = len(rows)
    finished = counts.get("FINISHED", 0)
    in_progress = counts.get("IN_PROGRESS", 0)
    queued = counts.get("QUEUED", 0)
    errored = counts.get("ERRORED", 0)
    pending_submission = counts.get("PENDING_SUBMISSION", 0)
    started = finished + in_progress + errored
    eta_seconds = None
    if durations and (queued or in_progress):
        avg_duration = sum(durations) / len(durations)
        waves_remaining = math.ceil((queued + in_progress) / MAX_CONCURRENT)
        eta_seconds = waves_remaining * avg_duration

    return {
        "generated_at": dt.datetime.now().isoformat(),
        "results_root": str(results_root),
        "job_ids": job_ids,
        "job_id": ",".join(job_ids) if job_ids else None,
        "total": total,
        "started": started,
        "finished": finished,
        "queued": queued,
        "pending_submission": pending_submission,
        "errored": errored,
        "in_progress": in_progress,
        "counts": dict(counts),
        "breakdown": {key: dict(counter) for key, counter in sorted(breakdown.items())},
        "phase_counts": dict(phase_counts),
        "health_samples": health_samples,
        "eta_seconds": eta_seconds,
        "rows": rows,
    }


def print_summary(summary: Dict[str, Any]) -> None:
    print(f"results_root: {summary['results_root']}")
    print(f"job_id: {summary.get('job_id')}")
    print(
        "overall: "
        f"started={summary['started']} finished={summary['finished']} "
        f"queued={summary['queued']} errored={summary['errored']} "
        f"in_progress={summary['in_progress']} "
        f"pending_submission={summary.get('pending_submission', 0)} "
        f"total={summary['total']}"
    )
    if summary.get("eta_seconds") is not None:
        eta = dt.timedelta(seconds=int(summary["eta_seconds"]))
        print(f"eta_estimate: {eta}")
    print("breakdown:")
    for key, counter in summary["breakdown"].items():
        print(
            f"  {key}: "
            f"finished={counter.get('FINISHED', 0)} "
            f"in_progress={counter.get('IN_PROGRESS', 0)} "
            f"queued={counter.get('QUEUED', 0)} "
            f"pending_submission={counter.get('PENDING_SUBMISSION', 0)} "
            f"errored={counter.get('ERRORED', 0)}"
        )
    print("top phases:")
    for phase, count in Counter(summary["phase_counts"]).most_common(10):
        print(f"  {phase}: {count}")
    if summary["health_samples"]:
        print("health samples:")
        for sample in summary["health_samples"]:
            cosine_summary = sample.get("pairwise_cosine_summary") or {}
            print(
                f"  config_{int(sample['config_id']):04d} {sample.get('game_label')} "
                f"{sample.get('experiment_family')}: "
                f"parallel={sample.get('parallel_phases')} "
                f"vote_tokens={sample.get('max_tokens_voting')} "
                f"supermajority_text={sample.get('acceptance_rule_seen')} "
                f"cosine_max_err={cosine_summary.get('max_abs_error')}"
            )


def cmd_generate(args: argparse.Namespace) -> int:
    results_root = latest_root_or_new(args.results_root)
    configs = build_configs(
        results_root=results_root,
        master_seed=args.master_seed,
        max_rounds=args.max_rounds,
        discussion_turns=args.discussion_turns,
        max_tokens_voting=args.max_tokens_voting,
    )
    manifest = {
        "results_root": str(results_root),
        "batch_type": "full_games123_multiagent",
        "generated_at": dt.datetime.now().isoformat(),
        "n_configs": len(configs),
        "baseline_model": BASELINE_MODEL,
        "adversary_models": ADVERSARY_MODELS,
        "n_values": N_VALUES,
        "game1_competition_levels": GAME1_COMPETITION_LEVELS,
        "game2_rho_values": "n_specific_negative_lower_bound_and_0.9",
        "game2_theta_values": GAME2_THETAS,
        "game2_n_issues": 10,
        "game3_sigma_values": GAME3_SIGMAS,
        "game3_alpha_values": GAME3_ALPHAS,
        "game3_m_projects": "int(2.5 * n_agents)",
        "homogeneous_seed_replicates": HOMOGENEOUS_SEED_REPLICATES,
        "heterogeneous_runs_per_cell": HETEROGENEOUS_RUNS_PER_CELL,
        "heterogeneous_pool": filtered_heterogeneous_pool(),
        "heterogeneous_pool_count": len(filtered_heterogeneous_pool()),
        "heterogeneous_excluded_model_count": len(HETEROGENEOUS_EXCLUDED_MODELS),
        "elo_bucket_count": ELO_BUCKET_COUNT,
        "master_seed": int(args.master_seed),
        "max_rounds": int(args.max_rounds),
        "discussion_turns": int(args.discussion_turns),
        "max_tokens_voting": args.max_tokens_voting,
        "parallel_phases": True,
        "slurm_time": str(args.slurm_time),
        "max_concurrent": int(args.max_concurrent),
    }
    validation = validate_configs(configs)
    manifest["validation"] = validation
    if not validation["valid"]:
        raise RuntimeError("Generated configs failed validation: " + "; ".join(validation["errors"][:10]))
    write_generated_files(results_root, manifest, configs)
    print(results_root)
    print(f"Generated {len(configs)} configs")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    results_root = resolve_results_root(args.results_root)
    validation = validate_configs(load_configs(results_root))
    print(json.dumps(validation, indent=2, default=str))
    return 0 if validation["valid"] else 1


def cmd_run_one(args: argparse.Namespace) -> int:
    results_root = resolve_results_root(args.results_root)
    config = load_config(results_root, args.config_id)
    status = run_config(results_root, config)
    print(json.dumps(status, indent=2, default=str))
    return 0 if status["state"] == "SUCCESS" else 1


def cmd_submit(args: argparse.Namespace) -> int:
    results_root = resolve_results_root(args.results_root)
    configs = load_configs(results_root)
    manifest = json.loads((results_root / "manifest.json").read_text(encoding="utf-8"))
    validation = validate_configs(configs)
    if not validation["valid"]:
        raise RuntimeError("Refusing to submit invalid configs: " + "; ".join(validation["errors"][:10]))
    sbatch_path = write_slurm_file(results_root, manifest)
    chunk_size = min(MAX_ARRAY_TASKS_PER_SUBMISSION, len(configs))
    chunks = [
        (start, min(start + chunk_size - 1, len(configs)))
        for start in range(1, len(configs) + 1, chunk_size)
    ]
    per_chunk_concurrency = max(1, int(manifest["max_concurrent"]) // len(chunks))
    submissions = []
    for start, end in chunks:
        config_offset = start - 1
        task_count = end - start + 1
        array_spec = f"1-{task_count}%{per_chunk_concurrency}"
        output = subprocess.check_output(
            [
                "sbatch",
                "--export=ALL,RUN_DIR=" + str(results_root) + f",CONFIG_OFFSET={config_offset}",
                "--array",
                array_spec,
                str(sbatch_path),
            ],
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        job_id = parse_job_id(output)
        submissions.append(
            {
                "job_id": job_id,
                "sbatch_output": output.strip(),
                "array_spec": array_spec,
                "config_start": start,
                "config_end": end,
                "config_offset": config_offset,
                "max_concurrent": per_chunk_concurrency,
                "submitted_at": dt.datetime.now().isoformat(),
            }
        )
        print(output.strip())
        print(f"array_spec={array_spec}")
    write_json(
        results_root / "submissions.json",
        {
            "submitted_at": dt.datetime.now().isoformat(),
            "submissions": submissions,
            "job_ids": [row["job_id"] for row in submissions],
            "array_specs": [row["array_spec"] for row in submissions],
            "slurm_time": manifest["slurm_time"],
            "max_concurrent": manifest["max_concurrent"],
            "per_chunk_concurrency": per_chunk_concurrency,
            "sbatch_path": str(sbatch_path),
        },
    )
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    results_root = resolve_results_root(args.results_root)
    summary = collect_summary(results_root)
    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        print_summary(summary)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    results_root = resolve_results_root(args.results_root)
    summary = collect_summary(results_root)
    lines = [
        "# Full Games 1-3 Multi-Agent Batch Report",
        "",
        f"- Results root: `{results_root}`",
        f"- Generated at: {summary['generated_at']}",
        f"- Slurm job ID: `{summary.get('job_id')}`",
        f"- Total configs: {summary['total']}",
        f"- Started: {summary['started']}",
        f"- Finished: {summary['finished']}",
        f"- In progress: {summary['in_progress']}",
        f"- Queued: {summary['queued']}",
        f"- Pending submission: {summary.get('pending_submission', 0)}",
        f"- Errored: {summary['errored']}",
        "",
        "## Breakdown",
        "",
        "| Slice | Finished | In Progress | Queued | Pending Submission | Errored |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, counter in summary["breakdown"].items():
        lines.append(
            f"| {key} | {counter.get('FINISHED', 0)} | {counter.get('IN_PROGRESS', 0)} | "
            f"{counter.get('QUEUED', 0)} | {counter.get('PENDING_SUBMISSION', 0)} | "
            f"{counter.get('ERRORED', 0)} |"
        )
    lines.extend(["", "## Phase Counts", ""])
    for phase, count in Counter(summary["phase_counts"]).most_common():
        lines.append(f"- `{phase}`: {count}")
    lines.extend(["", "## Health Samples", ""])
    for sample in summary["health_samples"]:
        lines.append(f"- `config_{int(sample['config_id']):04d}`: `{sample}`")
    report_path = results_root / "full_batch_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report_path)
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "generate":
        return cmd_generate(args)
    if args.command == "validate":
        return cmd_validate(args)
    if args.command == "run-one":
        return cmd_run_one(args)
    if args.command == "submit":
        return cmd_submit(args)
    if args.command == "summary":
        return cmd_summary(args)
    if args.command == "report":
        return cmd_report(args)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

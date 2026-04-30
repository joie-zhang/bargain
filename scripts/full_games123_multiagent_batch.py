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
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
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
    "gpt-5.4-high",
]
GEMINI_MODEL = "gemini-2.5-pro"
GEMINI_MODELS = {"gemini-2.5-pro", "gemini-3.1-pro"}
N_VALUES = [2, 4, 6, 8, 10]
MAX_AGENT_COLUMNS = max(N_VALUES)
GAME1_COMPETITION_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
GAME2_THETAS = [0.2, 0.8]
GAME3_SIGMAS = [0.2, 0.5]
GAME3_ALPHAS = [0.2, 0.8]
HOMOGENEOUS_SEED_REPLICATES = [1, 2]
HETEROGENEOUS_RUNS_PER_CELL = 20
HETEROGENEOUS_POOL_SIZE = 24
HETEROGENEOUS_EXCLUDED_MODELS = {"qwq-32b", "gemini-3-pro"}
HETEROGENEOUS_STRATA_COUNT = 5
HETEROGENEOUS_RUNS_PER_STRATUM = HETEROGENEOUS_RUNS_PER_CELL // HETEROGENEOUS_STRATA_COUNT
HETEROGENEOUS_STRATUM_LABELS = ["very_low", "low", "mid", "high", "very_high"]
HETEROGENEOUS_STRATEGY_PURE_RANDOM = "pure_random"
HETEROGENEOUS_STRATEGY_ELO_STDDEV_EQUAL_WIDTH = "elo_stddev_equal_width_stratified"
HETEROGENEOUS_SAMPLING_STRATEGY = HETEROGENEOUS_STRATEGY_ELO_STDDEV_EQUAL_WIDTH
HETEROGENEOUS_SAMPLING_STRATEGIES = [
    HETEROGENEOUS_STRATEGY_PURE_RANDOM,
    HETEROGENEOUS_STRATEGY_ELO_STDDEV_EQUAL_WIDTH,
]
HETEROGENEOUS_SUBSET_MAP_DIRNAME = "heterogeneous_subset_maps"
if HETEROGENEOUS_RUNS_PER_CELL % HETEROGENEOUS_STRATA_COUNT != 0:
    raise ValueError("HETEROGENEOUS_RUNS_PER_CELL must divide evenly across strata")
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
SELECTION_PRESETS = [
    "all",
    "extended_derisk_n10_amazon_gpt4o",
    "non_gemini_homogeneous",
    "gemini_homogeneous",
    "non_gemini_heterogeneous",
    "gemini_heterogeneous",
]
EXCLUDED_MODEL_REFERENCE_PATTERNS = tuple(
    sorted({
        model.lower()
        for model in HETEROGENEOUS_EXCLUDED_MODELS
    })
)
EXCLUDED_MODEL_REFERENCE_ALLOWED_KEYS = {
    "heterogeneous_excluded_model_count",
    "heterogeneous_excluded_models",
}


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
    generate.add_argument(
        "--heterogeneous-sampling-strategy",
        choices=HETEROGENEOUS_SAMPLING_STRATEGIES,
        default=HETEROGENEOUS_SAMPLING_STRATEGY,
        help=(
            "Heterogeneous roster sampler. Default uses equal-width Elo-stddev "
            "stratification; pure_random preserves the older behavior."
        ),
    )

    validate = subparsers.add_parser("validate")
    validate.add_argument("--results-root", type=str, required=True)

    run_one = subparsers.add_parser("run-one")
    run_one.add_argument("--results-root", type=str, required=True)
    run_one.add_argument("--config-id", type=int, required=True)

    submit = subparsers.add_parser("submit")
    submit.add_argument("--results-root", type=str, required=True)

    select = subparsers.add_parser("select")
    select.add_argument("--results-root", type=str, required=True)
    select.add_argument("--selection-name", type=str, required=True)
    select.add_argument("--preset", choices=SELECTION_PRESETS, default=None)
    select.add_argument("--game-label", action="append", default=[])
    select.add_argument("--experiment-family", action="append", default=[])
    select.add_argument("--n-agents", action="append", type=int, default=[])
    select.add_argument("--adversary-model", action="append", default=[])
    select.add_argument("--adversary-position", action="append", default=[])
    select.add_argument("--seed-replicate", action="append", type=int, default=[])
    select.add_argument("--contains-model", action="append", default=[])
    select.add_argument("--exclude-model", action="append", default=[])
    select.add_argument("--config-id", action="append", type=int, default=[])

    submit_selection = subparsers.add_parser("submit-selection")
    submit_selection.add_argument("--results-root", type=str, required=True)
    submit_selection.add_argument("--selection-name", type=str, default=None)
    submit_selection.add_argument("--selection-file", type=str, default=None)
    submit_selection.add_argument("--max-concurrent", type=int, default=None)
    submit_selection.add_argument("--rerun-existing", action="store_true")
    submit_selection.add_argument("--dry-run", action="store_true")

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


def split_csv_values(values: Iterable[Any]) -> List[str]:
    tokens: List[str] = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if token:
                tokens.append(token)
    return tokens


def split_csv_int_values(values: Iterable[Any]) -> List[int]:
    return [int(value) for value in split_csv_values(values)]


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


@dataclass(frozen=True, slots=True)
class HeterogeneousSubset:
    n_agents: int
    subset_key: str
    models: Tuple[str, ...]
    model_elos: Tuple[int, ...]
    elo_mean: float
    elo_variance: float
    elo_stddev: float
    subset_rank_by_stddev: int
    subset_rank_within_stratum: int
    stratum_index: int
    stratum_label: str
    stratum_population_size: int
    stratum_stddev_min: float
    stratum_stddev_max: float
    stratum_variance_min: float
    stratum_variance_max: float


@dataclass(frozen=True, slots=True)
class HeterogeneousSubsetMap:
    pool: Tuple[str, ...]
    rows_by_n: Dict[int, Tuple[HeterogeneousSubset, ...]]
    rows_by_n_stratum: Dict[Tuple[int, int], Tuple[HeterogeneousSubset, ...]]


def validate_heterogeneous_sampling_strategy(strategy: str) -> str:
    if strategy not in HETEROGENEOUS_SAMPLING_STRATEGIES:
        raise ValueError(
            f"Unknown heterogeneous sampling strategy {strategy!r}; "
            f"expected one of {HETEROGENEOUS_SAMPLING_STRATEGIES}"
        )
    return strategy


def model_elo_values(models: Iterable[str]) -> Tuple[int, ...]:
    elos: List[int] = []
    for model in models:
        elo = elo_for_model(model)
        if elo is None:
            raise ValueError(f"Missing Elo for model {model!r}")
        elos.append(int(elo))
    return tuple(elos)


def elo_summary_for_elos(elos: Iterable[int | float]) -> Dict[str, float]:
    values = [float(value) for value in elos]
    if not values:
        raise ValueError("Cannot summarize Elo values for an empty model set")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "elo_mean": mean,
        "elo_variance": variance,
        "elo_stddev": math.sqrt(variance),
    }


def subset_key_for_models(models: Iterable[str]) -> str:
    return "+".join(models)


@lru_cache(maxsize=None)
def random_state_start_hash(seed: int) -> str:
    rng = random.Random(int(seed))
    state_json = json.dumps(rng.getstate(), separators=(",", ":"))
    return hashlib.sha256(state_json.encode("utf-8")).hexdigest()


def random_state_start_metadata(seed: int) -> Dict[str, Any]:
    rng = random.Random(int(seed))
    state = rng.getstate()
    state_json = json.dumps(state, separators=(",", ":"))
    return {
        "rng": "python_random.Random",
        "python_version": sys.version,
        "seed": int(seed),
        "state": state,
        "state_sha256": hashlib.sha256(state_json.encode("utf-8")).hexdigest(),
    }


def heterogeneous_generation_seed(master_seed: int, strategy: str) -> int:
    return stable_seed(master_seed, "heterogeneous_generation_rng", strategy)


@lru_cache(maxsize=8)
def build_heterogeneous_subset_map(pool: Tuple[str, ...]) -> HeterogeneousSubsetMap:
    rows_by_n: Dict[int, Tuple[HeterogeneousSubset, ...]] = {}
    rows_by_n_stratum: Dict[Tuple[int, int], Tuple[HeterogeneousSubset, ...]] = {}
    pool_elos = model_elo_values(pool)

    for n_agents in N_VALUES:
        raw_rows: List[Tuple[Tuple[str, ...], Tuple[int, ...], float, float, float, str]] = []
        for indices in combinations(range(len(pool)), n_agents):
            models = tuple(pool[index] for index in indices)
            elos = tuple(pool_elos[index] for index in indices)
            summary = elo_summary_for_elos(elos)
            raw_rows.append(
                (
                    models,
                    elos,
                    summary["elo_mean"],
                    summary["elo_variance"],
                    summary["elo_stddev"],
                    subset_key_for_models(models),
                )
            )

        raw_rows.sort(key=lambda row: (row[4], row[5]))
        min_stddev = raw_rows[0][4]
        max_stddev = raw_rows[-1][4]
        width = (max_stddev - min_stddev) / HETEROGENEOUS_STRATA_COUNT
        bins: List[List[Tuple[int, Tuple[str, ...], Tuple[int, ...], float, float, float, str]]] = [
            [] for _ in range(HETEROGENEOUS_STRATA_COUNT)
        ]

        for rank, (models, elos, mean, variance, stddev, subset_key) in enumerate(raw_rows, start=1):
            if width <= 0:
                stratum_index = 0
            elif stddev >= max_stddev:
                stratum_index = HETEROGENEOUS_STRATA_COUNT - 1
            else:
                stratum_index = min(
                    HETEROGENEOUS_STRATA_COUNT - 1,
                    int((stddev - min_stddev) / width),
                )
            bins[stratum_index].append((rank, models, elos, mean, variance, stddev, subset_key))

        empty_bins = [index for index, rows in enumerate(bins) if not rows]
        if empty_bins:
            raise ValueError(
                f"Equal-width Elo-stddev strata are empty for N={n_agents}: {empty_bins}"
            )

        finalized: List[HeterogeneousSubset] = []
        for stratum_index, bin_rows in enumerate(bins):
            stddev_min = min_stddev + width * stratum_index
            stddev_max = (
                max_stddev
                if stratum_index == HETEROGENEOUS_STRATA_COUNT - 1
                else min_stddev + width * (stratum_index + 1)
            )
            stratum_population_size = len(bin_rows)
            finalized_bin: List[HeterogeneousSubset] = []
            for within_rank, (rank, models, elos, mean, variance, stddev, subset_key) in enumerate(
                sorted(bin_rows, key=lambda row: (row[5], row[6])),
                start=1,
            ):
                finalized_bin.append(
                    HeterogeneousSubset(
                        n_agents=n_agents,
                        subset_key=subset_key,
                        models=models,
                        model_elos=elos,
                        elo_mean=mean,
                        elo_variance=variance,
                        elo_stddev=stddev,
                        subset_rank_by_stddev=rank,
                        subset_rank_within_stratum=within_rank,
                        stratum_index=stratum_index,
                        stratum_label=HETEROGENEOUS_STRATUM_LABELS[stratum_index],
                        stratum_population_size=stratum_population_size,
                        stratum_stddev_min=stddev_min,
                        stratum_stddev_max=stddev_max,
                        stratum_variance_min=stddev_min ** 2,
                        stratum_variance_max=stddev_max ** 2,
                    )
                )
            rows_by_n_stratum[(n_agents, stratum_index)] = tuple(finalized_bin)
            finalized.extend(finalized_bin)

        finalized.sort(key=lambda row: row.subset_rank_by_stddev)
        rows_by_n[n_agents] = tuple(finalized)

    return HeterogeneousSubsetMap(
        pool=pool,
        rows_by_n=rows_by_n,
        rows_by_n_stratum=rows_by_n_stratum,
    )


def candidate_metadata(candidate: HeterogeneousSubset) -> Dict[str, Any]:
    return {
        "subset_key": candidate.subset_key,
        "subset_rank_by_stddev": candidate.subset_rank_by_stddev,
        "subset_rank_within_stratum": candidate.subset_rank_within_stratum,
        "stratum_index": candidate.stratum_index,
        "stratum_label": candidate.stratum_label,
        "stratum_method": "equal_width_stddev",
        "stratum_population_size": candidate.stratum_population_size,
        "stratum_stddev_min": candidate.stratum_stddev_min,
        "stratum_stddev_max": candidate.stratum_stddev_max,
        "stratum_variance_min": candidate.stratum_variance_min,
        "stratum_variance_max": candidate.stratum_variance_max,
        "subset_model_ids_unordered": list(candidate.models),
        "subset_model_elos_unordered": list(candidate.model_elos),
        "elo_mean": candidate.elo_mean,
        "elo_variance": candidate.elo_variance,
        "elo_stddev": candidate.elo_stddev,
    }


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
    if len(pool) != HETEROGENEOUS_POOL_SIZE:
        raise ValueError(
            f"Expected {HETEROGENEOUS_POOL_SIZE} heterogeneous models after exclusions, got {len(pool)}"
        )
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
            if str(key) in EXCLUDED_MODEL_REFERENCE_ALLOWED_KEYS:
                continue
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
    heterogeneous_order_seed: Optional[int] = None,
    heterogeneous_sampling_strategy: str = HETEROGENEOUS_SAMPLING_STRATEGY,
    heterogeneous_metadata: Optional[Dict[str, Any]] = None,
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
        config["heterogeneous_excluded_models"] = sorted(HETEROGENEOUS_EXCLUDED_MODELS)
        config["heterogeneous_sampling_strategy"] = heterogeneous_sampling_strategy
        config["heterogeneous_sampling"] = {
            "strategy": heterogeneous_sampling_strategy,
            "without_replacement_within_run": True,
            "with_replacement_across_runs": True,
            "sampling_with_replacement": True,
            "planned_roster_replication": False,
            "random_order": True,
        }
    if heterogeneous_draw_seed is not None:
        config["heterogeneous_draw_seed"] = int(heterogeneous_draw_seed)
    if heterogeneous_order_seed is not None:
        config["heterogeneous_order_seed"] = int(heterogeneous_order_seed)
    if heterogeneous_metadata:
        config.update(heterogeneous_metadata)
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
    heterogeneous_sampling_strategy: str = HETEROGENEOUS_SAMPLING_STRATEGY,
) -> List[Dict[str, Any]]:
    heterogeneous_sampling_strategy = validate_heterogeneous_sampling_strategy(
        heterogeneous_sampling_strategy
    )
    configs: List[Dict[str, Any]] = []
    config_id = 1
    hetero_pool = filtered_heterogeneous_pool()
    hetero_pool_tuple = tuple(hetero_pool)
    subset_map = (
        build_heterogeneous_subset_map(hetero_pool_tuple)
        if heterogeneous_sampling_strategy == HETEROGENEOUS_STRATEGY_ELO_STDDEV_EQUAL_WIDTH
        else None
    )
    elo_bucket_map = quantile_elo_bucket_map(hetero_pool, n_buckets=ELO_BUCKET_COUNT)
    generation_seed = heterogeneous_generation_seed(master_seed, heterogeneous_sampling_strategy)
    generation_rng_state_start_hash = random_state_start_hash(generation_seed)
    subset_reuse_global: Counter[str] = Counter()

    for game_label in ["game1", "game2", "game3"]:
        for n_agents in N_VALUES:
            for game_params in game_parameter_grid(game_label, n_agents):
                competition_id = game_params["competition_id"]
                for seed_replicate in HOMOGENEOUS_SEED_REPLICATES:
                    random_seed = stable_seed(
                        master_seed,
                        "homogeneous_control",
                        game_label,
                        n_agents,
                        competition_id,
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
                                competition_id,
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

                subset_reuse_within_cell: Counter[str] = Counter()
                if heterogeneous_sampling_strategy == HETEROGENEOUS_STRATEGY_PURE_RANDOM:
                    for run_index in range(1, HETEROGENEOUS_RUNS_PER_CELL + 1):
                        draw_seed = stable_seed(
                            master_seed,
                            "heterogeneous_draw",
                            heterogeneous_sampling_strategy,
                            game_label,
                            n_agents,
                            competition_id,
                            run_index,
                        )
                        order_seed = stable_seed(
                            master_seed,
                            "heterogeneous_order",
                            heterogeneous_sampling_strategy,
                            game_label,
                            n_agents,
                            competition_id,
                            run_index,
                        )
                        draw_rng = random.Random(draw_seed)
                        order_rng = random.Random(order_seed)
                        drawn_models = set(draw_rng.sample(hetero_pool, n_agents))
                        unordered_models = tuple(model for model in hetero_pool if model in drawn_models)
                        models = list(unordered_models)
                        order_rng.shuffle(models)
                        elos = model_elo_values(unordered_models)
                        elo_summary = elo_summary_for_elos(elos)
                        subset_key = subset_key_for_models(unordered_models)
                        global_before = subset_reuse_global[subset_key]
                        cell_before = subset_reuse_within_cell[subset_key]
                        subset_reuse_global[subset_key] += 1
                        subset_reuse_within_cell[subset_key] += 1
                        metadata = {
                            "generation_rng_state_start_hash": generation_rng_state_start_hash,
                            "sampling_rng_state_start_hash": random_state_start_hash(draw_seed),
                            "order_rng_state_start_hash": random_state_start_hash(order_seed),
                            "sampling_with_replacement": True,
                            "planned_roster_replication": False,
                            "subset_key": subset_key,
                            "subset_model_ids_unordered": list(unordered_models),
                            "subset_model_elos_unordered": list(elos),
                            "subset_reuse_count_global_before_this_draw": global_before,
                            "subset_reuse_count_global_after_this_draw": subset_reuse_global[subset_key],
                            "subset_reuse_count_within_cell_before_this_draw": cell_before,
                            "subset_reuse_count_within_cell_after_this_draw": subset_reuse_within_cell[subset_key],
                            **elo_summary,
                        }
                        random_seed = stable_seed(
                            master_seed,
                            "heterogeneous_environment",
                            game_label,
                            n_agents,
                            competition_id,
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
                                heterogeneous_order_seed=order_seed,
                                heterogeneous_sampling_strategy=heterogeneous_sampling_strategy,
                                heterogeneous_metadata=metadata,
                                elo_bucket_map=elo_bucket_map,
                            )
                        )
                        config_id += 1
                else:
                    if subset_map is None:
                        raise AssertionError("subset_map must be built for stratified sampling")
                    run_index = 1
                    for stratum_index in range(HETEROGENEOUS_STRATA_COUNT):
                        candidates = subset_map.rows_by_n_stratum[(n_agents, stratum_index)]
                        for stratum_draw_index in range(1, HETEROGENEOUS_RUNS_PER_STRATUM + 1):
                            draw_seed = stable_seed(
                                master_seed,
                                "heterogeneous_draw",
                                heterogeneous_sampling_strategy,
                                game_label,
                                n_agents,
                                competition_id,
                                stratum_index,
                                stratum_draw_index,
                            )
                            order_seed = stable_seed(
                                master_seed,
                                "heterogeneous_order",
                                heterogeneous_sampling_strategy,
                                game_label,
                                n_agents,
                                competition_id,
                                run_index,
                            )
                            draw_rng = random.Random(draw_seed)
                            order_rng = random.Random(order_seed)
                            candidate = draw_rng.choice(candidates)
                            models = list(candidate.models)
                            order_rng.shuffle(models)
                            global_before = subset_reuse_global[candidate.subset_key]
                            cell_before = subset_reuse_within_cell[candidate.subset_key]
                            subset_reuse_global[candidate.subset_key] += 1
                            subset_reuse_within_cell[candidate.subset_key] += 1
                            metadata = {
                                **candidate_metadata(candidate),
                                "stratum_draw_index": stratum_draw_index,
                                "generation_rng_state_start_hash": generation_rng_state_start_hash,
                                "sampling_rng_state_start_hash": random_state_start_hash(draw_seed),
                                "order_rng_state_start_hash": random_state_start_hash(order_seed),
                                "sampling_with_replacement": True,
                                "planned_roster_replication": False,
                                "subset_reuse_count_global_before_this_draw": global_before,
                                "subset_reuse_count_global_after_this_draw": subset_reuse_global[candidate.subset_key],
                                "subset_reuse_count_within_cell_before_this_draw": cell_before,
                                "subset_reuse_count_within_cell_after_this_draw": subset_reuse_within_cell[candidate.subset_key],
                            }
                            random_seed = stable_seed(
                                master_seed,
                                "heterogeneous_environment",
                                game_label,
                                n_agents,
                                competition_id,
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
                                    heterogeneous_order_seed=order_seed,
                                    heterogeneous_sampling_strategy=heterogeneous_sampling_strategy,
                                    heterogeneous_metadata=metadata,
                                    elo_bucket_map=elo_bucket_map,
                                )
                            )
                            config_id += 1
                            run_index += 1

    return configs


def format_float(value: float) -> str:
    return f"{float(value):.12g}"


def write_heterogeneous_subset_maps(
    output_dir: Path,
    subset_map: HeterogeneousSubsetMap,
    *,
    master_seed: int,
    strategy: str,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generation_seed = heterogeneous_generation_seed(master_seed, strategy)
    generation_state = random_state_start_metadata(generation_seed)
    map_manifest = {
        "generated_at": dt.datetime.now().isoformat(),
        "strategy": strategy,
        "stratum_method": "equal_width_stddev",
        "n_strata": HETEROGENEOUS_STRATA_COUNT,
        "stratum_labels": HETEROGENEOUS_STRATUM_LABELS,
        "n_values": N_VALUES,
        "master_seed": int(master_seed),
        "generation_rng_state_start": generation_state,
        "model_pool_size": len(subset_map.pool),
        "model_pool_file": f"model_pool_{len(subset_map.pool)}.csv",
        "excluded_models": sorted(HETEROGENEOUS_EXCLUDED_MODELS),
        "source_files": [
            "scripts/full_games123_multiagent_batch.py",
            "strong_models_experiment/analysis/active_model_roster.py",
        ],
    }
    write_json(output_dir / "manifest.json", map_manifest)

    with (output_dir / f"model_pool_{len(subset_map.pool)}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pool_index", "model", "arena_elo"])
        writer.writeheader()
        for index, model in enumerate(subset_map.pool):
            writer.writerow({
                "pool_index": index,
                "model": model,
                "arena_elo": elo_for_model(model),
            })

    boundary_fieldnames = [
        "n_agents",
        "stratum_index",
        "stratum_label",
        "stratum_method",
        "stratum_population_size",
        "stratum_stddev_min",
        "stratum_stddev_max",
        "stratum_variance_min",
        "stratum_variance_max",
        "observed_stddev_min",
        "observed_stddev_max",
    ]
    with (output_dir / "strata_boundaries.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=boundary_fieldnames)
        writer.writeheader()
        for n_agents in N_VALUES:
            for stratum_index in range(HETEROGENEOUS_STRATA_COUNT):
                rows = subset_map.rows_by_n_stratum[(n_agents, stratum_index)]
                first = rows[0]
                writer.writerow({
                    "n_agents": n_agents,
                    "stratum_index": stratum_index,
                    "stratum_label": first.stratum_label,
                    "stratum_method": "equal_width_stddev",
                    "stratum_population_size": len(rows),
                    "stratum_stddev_min": format_float(first.stratum_stddev_min),
                    "stratum_stddev_max": format_float(first.stratum_stddev_max),
                    "stratum_variance_min": format_float(first.stratum_variance_min),
                    "stratum_variance_max": format_float(first.stratum_variance_max),
                    "observed_stddev_min": format_float(rows[0].elo_stddev),
                    "observed_stddev_max": format_float(rows[-1].elo_stddev),
                })

    summary_fieldnames = [
        "n_agents",
        "subset_count",
        "elo_stddev_min",
        "elo_stddev_max",
        "elo_variance_min",
        "elo_variance_max",
        "stratum_population_sizes",
    ]
    with (output_dir / "exact_subset_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fieldnames)
        writer.writeheader()
        for n_agents in N_VALUES:
            rows = subset_map.rows_by_n[n_agents]
            writer.writerow({
                "n_agents": n_agents,
                "subset_count": len(rows),
                "elo_stddev_min": format_float(min(row.elo_stddev for row in rows)),
                "elo_stddev_max": format_float(max(row.elo_stddev for row in rows)),
                "elo_variance_min": format_float(min(row.elo_variance for row in rows)),
                "elo_variance_max": format_float(max(row.elo_variance for row in rows)),
                "stratum_population_sizes": "+".join(
                    str(len(subset_map.rows_by_n_stratum[(n_agents, stratum_index)]))
                    for stratum_index in range(HETEROGENEOUS_STRATA_COUNT)
                ),
            })

    subset_fieldnames = [
        "n_agents",
        "subset_key",
        "subset_rank_by_stddev",
        "subset_rank_within_stratum",
        "stratum_index",
        "stratum_label",
        "stratum_method",
        "stratum_population_size",
        "stratum_stddev_min",
        "stratum_stddev_max",
        "stratum_variance_min",
        "stratum_variance_max",
        "model_ids_unordered",
        "model_elos_unordered",
        "elo_mean",
        "elo_variance",
        "elo_stddev",
    ]
    for n_agents in N_VALUES:
        with (output_dir / f"n_{n_agents:02d}_subsets.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=subset_fieldnames)
            writer.writeheader()
            for row in subset_map.rows_by_n[n_agents]:
                writer.writerow({
                    "n_agents": row.n_agents,
                    "subset_key": row.subset_key,
                    "subset_rank_by_stddev": row.subset_rank_by_stddev,
                    "subset_rank_within_stratum": row.subset_rank_within_stratum,
                    "stratum_index": row.stratum_index,
                    "stratum_label": row.stratum_label,
                    "stratum_method": "equal_width_stddev",
                    "stratum_population_size": row.stratum_population_size,
                    "stratum_stddev_min": format_float(row.stratum_stddev_min),
                    "stratum_stddev_max": format_float(row.stratum_stddev_max),
                    "stratum_variance_min": format_float(row.stratum_variance_min),
                    "stratum_variance_max": format_float(row.stratum_variance_max),
                    "model_ids_unordered": "+".join(row.models),
                    "model_elos_unordered": "+".join(str(elo) for elo in row.model_elos),
                    "elo_mean": format_float(row.elo_mean),
                    "elo_variance": format_float(row.elo_variance),
                    "elo_stddev": format_float(row.elo_stddev),
                })

    return {
        "subset_map_dir": str(output_dir),
        "subset_map_manifest": str(output_dir / "manifest.json"),
        "generation_rng_state_start_hash": generation_state["state_sha256"],
    }


def write_generated_files(results_root: Path, manifest: Dict[str, Any], configs: List[Dict[str, Any]]) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    config_dir = results_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (results_root / "logs").mkdir(exist_ok=True)
    (results_root / "status").mkdir(exist_ok=True)
    (results_root / "slurm").mkdir(exist_ok=True)
    (results_root / "selections").mkdir(exist_ok=True)
    (results_root / "submissions").mkdir(exist_ok=True)

    write_json(results_root / "manifest.json", manifest)
    if manifest.get("write_heterogeneous_subset_maps"):
        strategy = validate_heterogeneous_sampling_strategy(
            str(manifest.get("heterogeneous_sampling_strategy", HETEROGENEOUS_SAMPLING_STRATEGY))
        )
        if strategy == HETEROGENEOUS_STRATEGY_ELO_STDDEV_EQUAL_WIDTH:
            subset_map_info = write_heterogeneous_subset_maps(
                config_dir / HETEROGENEOUS_SUBSET_MAP_DIRNAME,
                build_heterogeneous_subset_map(tuple(filtered_heterogeneous_pool())),
                master_seed=int(manifest.get("master_seed", MASTER_SEED)),
                strategy=strategy,
            )
            manifest["heterogeneous_subset_map"] = subset_map_info
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
        "heterogeneous_run_index", "heterogeneous_draw_seed", "heterogeneous_order_seed",
        "heterogeneous_sampling_strategy", "generation_rng_state_start_hash",
        "sampling_rng_state_start_hash", "order_rng_state_start_hash",
        "model_pool_size", "model_pool", "heterogeneous_excluded_models",
        "heterogeneous_sampling", "sampling_with_replacement", "planned_roster_replication",
        "subset_key", "subset_rank_by_stddev", "subset_rank_within_stratum",
        "subset_reuse_count_global_before_this_draw",
        "subset_reuse_count_global_after_this_draw",
        "subset_reuse_count_within_cell_before_this_draw",
        "subset_reuse_count_within_cell_after_this_draw",
        "stratum_index", "stratum_label", "stratum_method", "stratum_draw_index",
        "stratum_population_size", "stratum_stddev_min", "stratum_stddev_max",
        "stratum_variance_min", "stratum_variance_max",
        "subset_model_ids_unordered", "subset_model_elos_unordered",
        "elo_mean", "elo_variance", "elo_stddev", "elo_bucket_method",
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
            row["heterogeneous_excluded_models"] = "+".join(
                config.get("heterogeneous_excluded_models") or []
            )
            row["subset_model_ids_unordered"] = "+".join(
                config.get("subset_model_ids_unordered") or []
            )
            row["subset_model_elos_unordered"] = "+".join(
                str(elo) for elo in (config.get("subset_model_elos_unordered") or [])
            )
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


def validate_result_file(config: Dict[str, Any], result_path: Path) -> Optional[str]:
    """Return None when a result file is complete enough to count as success."""
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return f"invalid result JSON: {exc}"
    except OSError as exc:
        return f"result file unreadable: {exc}"

    if not isinstance(payload, dict):
        return "result payload is not a JSON object"

    final_utilities = payload.get("final_utilities")
    if not isinstance(final_utilities, dict):
        return "missing or non-dict final_utilities"

    try:
        n_agents = int(config.get("n_agents") or len(config.get("models") or []))
    except (TypeError, ValueError):
        return "cannot infer expected agent count"
    expected_agents = agent_ids(n_agents)
    if not expected_agents:
        return "cannot infer expected agent IDs"

    missing_agents = [agent_id for agent_id in expected_agents if agent_id not in final_utilities]
    if missing_agents:
        return "missing final_utilities for " + ", ".join(missing_agents[:5])

    for agent_id in expected_agents:
        try:
            float(final_utilities[agent_id])
        except (TypeError, ValueError):
            return f"non-numeric final_utility for {agent_id}: {final_utilities[agent_id]!r}"

    payload_config = payload.get("config", {})
    if payload_config is None:
        payload_config = {}
    if not isinstance(payload_config, dict):
        return "result config is not a JSON object"

    for key in ("config_id", "random_seed"):
        if key not in payload_config or payload_config[key] is None or key not in config:
            continue
        try:
            if int(payload_config[key]) != int(config[key]):
                return f"result config {key} mismatch: {payload_config[key]!r} != {config[key]!r}"
        except (TypeError, ValueError):
            return f"result config {key} is non-integer: {payload_config[key]!r}"

    if payload_config.get("game_type") is not None and payload_config.get("game_type") != config.get("game_type"):
        return f"result config game_type mismatch: {payload_config.get('game_type')!r} != {config.get('game_type')!r}"

    return None


def selection_slug(selection_name: str) -> str:
    slug = sanitize_token(selection_name)
    if not slug:
        raise ValueError("Selection name cannot be empty after sanitization")
    return slug


def selection_ids_path(results_root: Path, selection_name: str) -> Path:
    return results_root / "selections" / f"{selection_slug(selection_name)}_config_ids.txt"


def selection_manifest_path(results_root: Path, selection_name: str) -> Path:
    return results_root / "selections" / f"{selection_slug(selection_name)}_selection.json"


def read_config_id_file(path: Path) -> List[int]:
    ids: List[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            ids.append(int(stripped))
    return ids


def config_succeeded(config: Dict[str, Any]) -> bool:
    result_path = result_path_for(config)
    return result_path is not None and validate_result_file(config, result_path) is None


def chunked(values: List[int], chunk_size: int = MAX_ARRAY_TASKS_PER_SUBMISSION) -> List[List[int]]:
    return [values[idx:idx + chunk_size] for idx in range(0, len(values), chunk_size)]


def selection_preset_criteria(preset: Optional[str]) -> Dict[str, set]:
    if not preset or preset == "all":
        return {}
    if preset == "extended_derisk_n10_amazon_gpt4o":
        return {
            "experiment_families": {"homogeneous_adversary"},
            "n_agents": {10},
            "adversary_models": {
                "amazon-nova-micro-v1.0",
                "gpt-4o-mini-2024-07-18",
            },
            "adversary_positions": {"first", "last"},
            "seed_replicates": {1},
        }
    if preset == "non_gemini_homogeneous":
        return {
            "experiment_families": {"homogeneous_control", "homogeneous_adversary"},
            "exclude_models": set(GEMINI_MODELS),
        }
    if preset == "gemini_homogeneous":
        return {
            "experiment_families": {"homogeneous_adversary"},
            "contains_models": set(GEMINI_MODELS),
        }
    if preset == "non_gemini_heterogeneous":
        return {
            "experiment_families": {"heterogeneous_random"},
            "exclude_models": set(GEMINI_MODELS),
        }
    if preset == "gemini_heterogeneous":
        return {
            "experiment_families": {"heterogeneous_random"},
            "contains_models": set(GEMINI_MODELS),
        }
    raise ValueError(f"Unknown selection preset: {preset}")


def merge_selection_criteria(base: Dict[str, set], extra: Dict[str, set]) -> Dict[str, set]:
    merged = {key: set(value) for key, value in base.items()}
    intersect_keys = {
        "config_ids",
        "game_labels",
        "experiment_families",
        "n_agents",
        "adversary_models",
        "adversary_positions",
        "seed_replicates",
    }
    union_keys = {"contains_models", "exclude_models"}
    for key, values in extra.items():
        if not values:
            continue
        incoming = set(values)
        if key in union_keys:
            merged[key] = set(merged.get(key, set())) | incoming
        elif key in intersect_keys and merged.get(key):
            merged[key] = set(merged[key]) & incoming
        else:
            merged[key] = incoming
    return merged


def config_matches_selection(config: Dict[str, Any], criteria: Dict[str, set]) -> bool:
    models = set(config.get("models") or [])
    checks = [
        ("config_ids", int(config["config_id"])),
        ("game_labels", config.get("game_label")),
        ("experiment_families", config.get("experiment_family")),
        ("n_agents", int(config.get("n_agents", 0))),
        ("adversary_models", config.get("adversary_model")),
        ("adversary_positions", config.get("adversary_position")),
        ("seed_replicates", config.get("seed_replicate")),
    ]
    for key, value in checks:
        allowed = criteria.get(key)
        if allowed and value not in allowed:
            return False
    contains_models = criteria.get("contains_models") or set()
    if contains_models and not (models & contains_models):
        return False
    exclude_models = criteria.get("exclude_models") or set()
    if exclude_models and models & exclude_models:
        return False
    return True


def select_config_ids(configs: List[Dict[str, Any]], criteria: Dict[str, set]) -> List[int]:
    return [
        int(config["config_id"])
        for config in configs
        if config_matches_selection(config, criteria)
    ]


def selection_breakdown(configs: List[Dict[str, Any]], config_ids: List[int]) -> Dict[str, Dict[str, int]]:
    selected = {int(config_id) for config_id in config_ids}
    counters: Dict[str, Counter] = {
        "by_game_family": Counter(),
        "by_game": Counter(),
        "by_family": Counter(),
        "by_n_agents": Counter(),
        "by_adversary_model": Counter(),
        "by_contains_gemini": Counter(),
    }
    for config in configs:
        if int(config["config_id"]) not in selected:
            continue
        counters["by_game_family"][f"{config['game_label']}:{config['experiment_family']}"] += 1
        counters["by_game"][str(config["game_label"])] += 1
        counters["by_family"][str(config["experiment_family"])] += 1
        counters["by_n_agents"][str(config["n_agents"])] += 1
        if config.get("adversary_model"):
            counters["by_adversary_model"][str(config["adversary_model"])] += 1
        counters["by_contains_gemini"][str(bool(GEMINI_MODELS & set(config.get("models") or [])))] += 1
    return {key: dict(counter) for key, counter in counters.items()}


def write_selection_files(
    results_root: Path,
    configs: List[Dict[str, Any]],
    selection_name: str,
    criteria: Dict[str, set],
    config_ids: List[int],
) -> Dict[str, Any]:
    selections_dir = results_root / "selections"
    selections_dir.mkdir(parents=True, exist_ok=True)
    ids_path = selection_ids_path(results_root, selection_name)
    ids_path.write_text(
        "\n".join(str(config_id) for config_id in config_ids) + ("\n" if config_ids else ""),
        encoding="utf-8",
    )
    selected = {int(config_id) for config_id in config_ids}
    csv_path = selections_dir / f"{selection_slug(selection_name)}_index.csv"
    fieldnames = [
        "config_id", "game_label", "experiment_family", "n_agents",
        "competition_id", "adversary_model", "adversary_position",
        "seed_replicate", "heterogeneous_run_index", "contains_gemini",
        "models",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for config in configs:
            if int(config["config_id"]) not in selected:
                continue
            row = {field: config.get(field) for field in fieldnames if field != "models"}
            row["models"] = "+".join(config.get("models") or [])
            row["contains_gemini"] = bool(GEMINI_MODELS & set(config.get("models") or []))
            writer.writerow(row)
    manifest = {
        "selection_name": selection_slug(selection_name),
        "created_at": dt.datetime.now().isoformat(),
        "results_root": str(results_root),
        "count": len(config_ids),
        "config_ids_file": str(ids_path),
        "index_csv": str(csv_path),
        "criteria": {key: sorted(value) for key, value in criteria.items()},
        "breakdown": selection_breakdown(configs, config_ids),
        "config_ids": config_ids,
    }
    write_json(selection_manifest_path(results_root, selection_name), manifest)
    return manifest


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
        "heterogeneous_order_seed", "model_pool", "model_pool_size",
        "heterogeneous_sampling", "heterogeneous_sampling_strategy",
        "heterogeneous_excluded_model_count", "heterogeneous_excluded_models",
        "generation_rng_state_start_hash", "sampling_rng_state_start_hash",
        "order_rng_state_start_hash", "sampling_with_replacement",
        "planned_roster_replication", "subset_key", "subset_rank_by_stddev",
        "subset_rank_within_stratum", "subset_reuse_count_global_before_this_draw",
        "subset_reuse_count_global_after_this_draw",
        "subset_reuse_count_within_cell_before_this_draw",
        "subset_reuse_count_within_cell_after_this_draw",
        "stratum_index", "stratum_label", "stratum_method", "stratum_draw_index",
        "stratum_population_size", "stratum_stddev_min", "stratum_stddev_max",
        "stratum_variance_min", "stratum_variance_max",
        "subset_model_ids_unordered", "subset_model_elos_unordered",
        "elo_mean", "elo_variance", "elo_stddev",
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
    existing_result_error = validate_result_file(config, existing) if existing is not None else None
    if existing is not None and existing_result_error is None:
        enrich_result_metadata(config)
        status = {
            "config_id": config_id,
            "state": "SUCCESS",
            "returncode": 0,
            "skipped_existing": True,
            "result_path": str(existing),
            "result_validation_error": None,
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
            "EXPERIMENT_RUN_METADATA_JSON": json.dumps(config, default=str),
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
    result_error = validate_result_file(config, result_path) if result_path is not None else "missing result file"
    state = "SUCCESS" if returncode == 0 and result_path is not None and result_error is None else "FAILED"
    if state == "SUCCESS":
        enrich_result_metadata(config)
    attempts[-1].update(
        {
            "state": state,
            "returncode": returncode,
            "finished_at": end.isoformat(),
            "duration_seconds": (end - start).total_seconds(),
            "result_path": str(result_path) if result_path else None,
            "result_validation_error": result_error,
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
        "result_validation_error": result_error,
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
if [[ -n "${{SELECTED_CONFIG_IDS_FILE:-}}" ]]; then
  if [[ ! -f "$SELECTED_CONFIG_IDS_FILE" ]]; then
    echo "Missing SELECTED_CONFIG_IDS_FILE: $SELECTED_CONFIG_IDS_FILE" >&2
    exit 2
  fi
  CONFIG_ID="$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "$SELECTED_CONFIG_IDS_FILE" | tr -d '[:space:]')"
  if [[ -z "$CONFIG_ID" ]]; then
    echo "No config ID for task $SLURM_ARRAY_TASK_ID in $SELECTED_CONFIG_IDS_FILE" >&2
    exit 2
  fi
else
  CONFIG_OFFSET="${{CONFIG_OFFSET:-0}}"
  CONFIG_ID=$((SLURM_ARRAY_TASK_ID + CONFIG_OFFSET))
fi

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
    for excluded_model in HETEROGENEOUS_EXCLUDED_MODELS:
        if excluded_model in hetero_pool:
            errors.append(f"{excluded_model} remains in heterogeneous pool")
    hetero_cell_counts: Counter[Tuple[str, int, str]] = Counter()
    hetero_cell_stratum_counts: Counter[Tuple[str, int, str, int]] = Counter()
    hetero_cell_strategies: Dict[Tuple[str, int, str], set] = defaultdict(set)

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
            hetero_cell = (config["game_label"], n_agents, config["competition_id"])
            hetero_cell_counts[hetero_cell] += 1
            if len(set(config["models"])) != len(config["models"]):
                errors.append(f"config_{config['config_id']:04d}: heterogeneous models are not unique")
            if any(model in HETEROGENEOUS_EXCLUDED_MODELS for model in config["models"]):
                errors.append(f"config_{config['config_id']:04d}: contains excluded heterogeneous model")
            if int(config.get("model_pool_size", 0)) != HETEROGENEOUS_POOL_SIZE:
                errors.append(
                    f"config_{config['config_id']:04d}: model_pool_size is not {HETEROGENEOUS_POOL_SIZE}"
                )
            strategy = config.get("heterogeneous_sampling_strategy")
            hetero_cell_strategies[hetero_cell].add(strategy)
            if strategy not in HETEROGENEOUS_SAMPLING_STRATEGIES:
                errors.append(f"config_{config['config_id']:04d}: invalid heterogeneous sampling strategy")
            if not config.get("sampling_with_replacement", False):
                errors.append(f"config_{config['config_id']:04d}: sampling_with_replacement is not true")
            if config.get("planned_roster_replication", True):
                errors.append(f"config_{config['config_id']:04d}: planned_roster_replication is not false")
            unordered_models = config.get("subset_model_ids_unordered") or config["models"]
            try:
                logged_elos = [int(value) for value in config.get("subset_model_elos_unordered", [])]
                computed_elos = list(model_elo_values(tuple(unordered_models)))
                if logged_elos and logged_elos != computed_elos:
                    errors.append(f"config_{config['config_id']:04d}: subset Elo list does not match model IDs")
                summary = elo_summary_for_elos(computed_elos)
                for key, expected in summary.items():
                    if key not in config or not math.isclose(
                        float(config[key]),
                        float(expected),
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    ):
                        errors.append(f"config_{config['config_id']:04d}: {key} does not round-trip from model IDs")
            except (TypeError, ValueError) as exc:
                errors.append(f"config_{config['config_id']:04d}: failed Elo metadata validation: {exc}")
            for reuse_key in [
                "subset_reuse_count_global_before_this_draw",
                "subset_reuse_count_global_after_this_draw",
                "subset_reuse_count_within_cell_before_this_draw",
                "subset_reuse_count_within_cell_after_this_draw",
            ]:
                if reuse_key not in config:
                    errors.append(f"config_{config['config_id']:04d}: missing {reuse_key}")
            if strategy == HETEROGENEOUS_STRATEGY_ELO_STDDEV_EQUAL_WIDTH:
                required_keys = [
                    "stratum_index",
                    "stratum_label",
                    "stratum_method",
                    "stratum_population_size",
                    "stratum_stddev_min",
                    "stratum_stddev_max",
                    "stratum_variance_min",
                    "stratum_variance_max",
                    "subset_rank_by_stddev",
                    "subset_rank_within_stratum",
                    "generation_rng_state_start_hash",
                    "sampling_rng_state_start_hash",
                    "order_rng_state_start_hash",
                ]
                for key in required_keys:
                    if key not in config:
                        errors.append(f"config_{config['config_id']:04d}: missing {key}")
                try:
                    stratum_index = int(config["stratum_index"])
                    hetero_cell_stratum_counts[(*hetero_cell, stratum_index)] += 1
                    stddev = float(config["elo_stddev"])
                    lower = float(config["stratum_stddev_min"])
                    upper = float(config["stratum_stddev_max"])
                    if not (lower - 1e-9 <= stddev <= upper + 1e-9):
                        errors.append(
                            f"config_{config['config_id']:04d}: elo_stddev outside stratum boundaries"
                        )
                    if config.get("stratum_method") != "equal_width_stddev":
                        errors.append(f"config_{config['config_id']:04d}: invalid stratum_method")
                    if not (0 <= stratum_index < HETEROGENEOUS_STRATA_COUNT):
                        errors.append(f"config_{config['config_id']:04d}: invalid stratum_index")
                    if int(config.get("stratum_population_size", 0)) <= 0:
                        errors.append(f"config_{config['config_id']:04d}: empty stratum_population_size")
                except (KeyError, TypeError, ValueError) as exc:
                    errors.append(f"config_{config['config_id']:04d}: invalid stratum metadata: {exc}")

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

    for cell, count in hetero_cell_counts.items():
        if count != HETEROGENEOUS_RUNS_PER_CELL:
            errors.append(f"heterogeneous cell {cell} expected {HETEROGENEOUS_RUNS_PER_CELL}, got {count}")
    for cell in hetero_cell_counts:
        if HETEROGENEOUS_STRATEGY_ELO_STDDEV_EQUAL_WIDTH not in hetero_cell_strategies.get(cell, set()):
            continue
        for stratum_index in range(HETEROGENEOUS_STRATA_COUNT):
            key = (*cell, stratum_index)
            count = hetero_cell_stratum_counts.get(key, 0)
            if count != HETEROGENEOUS_RUNS_PER_STRATUM:
                errors.append(
                    f"heterogeneous cell {cell} stratum {stratum_index} "
                    f"expected {HETEROGENEOUS_RUNS_PER_STRATUM}, got {count}"
                )

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
    records = []
    for row in payload.get("submissions", []):
        if not row.get("job_id"):
            continue
        record = {
            "job_id": str(row["job_id"]),
            "config_offset": int(row.get("config_offset", 0)),
        }
        selected_file = row.get("selected_config_ids_file")
        if selected_file:
            selected_path = Path(selected_file)
            if not selected_path.is_absolute():
                selected_path = results_root / selected_path
            if selected_path.exists():
                record["config_ids"] = read_config_id_file(selected_path)
        records.append(record)
    return records


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
            config_ids = record.get("config_ids") or []
            if config_ids:
                if task_id < 1 or task_id > len(config_ids):
                    continue
                config_id = int(config_ids[task_id - 1])
            else:
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
        "result_validation_error": validate_result_file(config, result_path),
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
        result_error = validate_result_file(config, result_path) if result_path is not None else None
        queue_state = queue_states.get(config_id, {})
        raw_state = status.get("state") or queue_state.get("state") or "NOT_STARTED"

        if result_path is not None and result_error is None and raw_state != "FAILED":
            state = "FINISHED"
        elif raw_state == "RUNNING" or queue_state.get("state") == "RUNNING":
            state = "IN_PROGRESS"
        elif queue_state.get("state") == "PENDING":
            state = "QUEUED"
        elif result_path is not None and result_error is not None:
            state = "ERRORED"
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
            "result_validation_error": result_error,
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
        heterogeneous_sampling_strategy=args.heterogeneous_sampling_strategy,
    )
    generation_seed = heterogeneous_generation_seed(
        args.master_seed,
        args.heterogeneous_sampling_strategy,
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
        "heterogeneous_runs_per_stratum": HETEROGENEOUS_RUNS_PER_STRATUM,
        "heterogeneous_strata_count": HETEROGENEOUS_STRATA_COUNT,
        "heterogeneous_stratum_labels": HETEROGENEOUS_STRATUM_LABELS,
        "heterogeneous_sampling_strategy": args.heterogeneous_sampling_strategy,
        "heterogeneous_generation_rng_state_start": random_state_start_metadata(generation_seed),
        "write_heterogeneous_subset_maps": (
            args.heterogeneous_sampling_strategy == HETEROGENEOUS_STRATEGY_ELO_STDDEV_EQUAL_WIDTH
        ),
        "heterogeneous_pool": filtered_heterogeneous_pool(),
        "heterogeneous_pool_count": len(filtered_heterogeneous_pool()),
        "heterogeneous_excluded_model_count": len(HETEROGENEOUS_EXCLUDED_MODELS),
        "heterogeneous_excluded_models": sorted(HETEROGENEOUS_EXCLUDED_MODELS),
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


def selection_criteria_from_args(args: argparse.Namespace) -> Dict[str, set]:
    criteria = selection_preset_criteria(args.preset)
    explicit = {
        "config_ids": set(split_csv_int_values(args.config_id)),
        "game_labels": set(split_csv_values(args.game_label)),
        "experiment_families": set(split_csv_values(args.experiment_family)),
        "n_agents": set(split_csv_int_values(args.n_agents)),
        "adversary_models": set(split_csv_values(args.adversary_model)),
        "adversary_positions": set(split_csv_values(args.adversary_position)),
        "seed_replicates": set(split_csv_int_values(args.seed_replicate)),
        "contains_models": set(split_csv_values(args.contains_model)),
        "exclude_models": set(split_csv_values(args.exclude_model)),
    }
    return merge_selection_criteria(criteria, explicit)


def cmd_select(args: argparse.Namespace) -> int:
    results_root = resolve_results_root(args.results_root)
    configs = load_configs(results_root)
    validation = validate_configs(configs)
    if not validation["valid"]:
        raise RuntimeError("Refusing to select from invalid configs: " + "; ".join(validation["errors"][:10]))
    criteria = selection_criteria_from_args(args)
    config_ids = select_config_ids(configs, criteria)
    manifest = write_selection_files(
        results_root=results_root,
        configs=configs,
        selection_name=args.selection_name,
        criteria=criteria,
        config_ids=config_ids,
    )
    print(f"selection_name={manifest['selection_name']}")
    print(f"count={manifest['count']}")
    print(f"config_ids_file={manifest['config_ids_file']}")
    print(f"index_csv={manifest['index_csv']}")
    return 0


def resolve_selection_file(results_root: Path, selection_name: Optional[str], selection_file: Optional[str]) -> Path:
    if selection_file and selection_name:
        raise ValueError("Use either --selection-name or --selection-file, not both")
    if selection_file:
        path = Path(selection_file)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        return path
    if selection_name:
        return selection_ids_path(results_root, selection_name)
    raise ValueError("submit-selection requires --selection-name or --selection-file")


def append_submission_records(
    results_root: Path,
    new_submissions: List[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    path = results_root / "submissions.json"
    existing = read_json_file(path)
    existing_submissions = list(existing.get("submissions") or [])
    if existing.get("job_id") and not existing_submissions:
        existing_submissions.append({
            "job_id": existing.get("job_id"),
            "config_offset": int(existing.get("config_offset", 0)),
            "array_spec": existing.get("array_spec"),
            "submitted_at": existing.get("submitted_at"),
        })
    submissions = existing_submissions + new_submissions
    payload = {
        "updated_at": dt.datetime.now().isoformat(),
        "submissions": submissions,
        "job_ids": [row["job_id"] for row in submissions if row.get("job_id")],
        "array_specs": [row["array_spec"] for row in submissions if row.get("array_spec")],
        **metadata,
    }
    write_json(path, payload)
    return payload


def cmd_submit_selection(args: argparse.Namespace) -> int:
    results_root = resolve_results_root(args.results_root)
    configs = load_configs(results_root)
    configs_by_id = {int(config["config_id"]): config for config in configs}
    manifest = json.loads((results_root / "manifest.json").read_text(encoding="utf-8"))
    validation = validate_configs(configs)
    if not validation["valid"]:
        raise RuntimeError("Refusing to submit invalid configs: " + "; ".join(validation["errors"][:10]))

    selected_file = resolve_selection_file(results_root, args.selection_name, args.selection_file)
    if not selected_file.exists():
        raise FileNotFoundError(f"Missing selection config-id file: {selected_file}")
    selected_ids = read_config_id_file(selected_file)
    missing_ids = [config_id for config_id in selected_ids if config_id not in configs_by_id]
    if missing_ids:
        raise ValueError(f"Selection contains unknown config IDs: {missing_ids[:10]}")

    skipped_existing = [
        config_id
        for config_id in selected_ids
        if not args.rerun_existing and config_succeeded(configs_by_id[config_id])
    ]
    skipped_existing_set = set(skipped_existing)
    submit_ids = [
        config_id
        for config_id in selected_ids
        if args.rerun_existing or config_id not in skipped_existing_set
    ]
    print(f"selection_file={selected_file}")
    print(f"selected={len(selected_ids)}")
    print(f"skipped_existing={len(skipped_existing)}")
    print(f"to_submit={len(submit_ids)}")
    if not submit_ids:
        print("No configs to submit.")
        return 0

    max_concurrent = int(args.max_concurrent or manifest.get("max_concurrent") or MAX_CONCURRENT)
    submit_chunks = chunked(submit_ids)
    per_chunk_concurrency = max(1, max_concurrent // len(submit_chunks))
    sbatch_path = write_slurm_file(results_root, manifest)
    submissions_dir = results_root / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = selection_slug(args.selection_name or selected_file.stem)
    submissions = []

    for chunk_idx, chunk in enumerate(submit_chunks, start=1):
        chunk_ids_path = submissions_dir / f"{base_name}_{stamp}_{chunk_idx:02d}_config_ids.txt"
        chunk_ids_path.write_text("\n".join(str(config_id) for config_id in chunk) + "\n", encoding="utf-8")
        array_range = f"1-{len(chunk)}" if len(chunk) > 1 else "1"
        array_spec = f"{array_range}%{per_chunk_concurrency}"
        if args.dry_run:
            output = "DRY RUN: sbatch not called"
            job_id = None
        else:
            output = subprocess.check_output(
                [
                    "sbatch",
                    "--export=ALL,RUN_DIR=" + str(results_root) + f",SELECTED_CONFIG_IDS_FILE={chunk_ids_path}",
                    "--array",
                    array_spec,
                    str(sbatch_path),
                ],
                text=True,
                cwd=str(PROJECT_ROOT),
            )
            job_id = parse_job_id(output)
        submission = {
            "job_id": job_id,
            "sbatch_output": output.strip(),
            "array_spec": array_spec,
            "selection_name": base_name,
            "selection_source_file": str(selected_file),
            "selected_config_ids_file": str(chunk_ids_path),
            "config_count": len(chunk),
            "max_concurrent": per_chunk_concurrency,
            "rerun_existing": bool(args.rerun_existing),
            "dry_run": bool(args.dry_run),
            "submitted_at": dt.datetime.now().isoformat(),
        }
        submissions.append(submission)
        print(output.strip())
        print(f"array_spec={array_spec}")
        print(f"selected_config_ids_file={chunk_ids_path}")

    per_selection_payload = {
        "created_at": dt.datetime.now().isoformat(),
        "selection_name": base_name,
        "selection_source_file": str(selected_file),
        "selected_count": len(selected_ids),
        "skipped_existing_count": len(skipped_existing),
        "submitted_count": len(submit_ids),
        "skipped_existing_config_ids": skipped_existing,
        "submitted_config_ids": submit_ids,
        "submissions": submissions,
        "slurm_time": manifest["slurm_time"],
        "max_concurrent": max_concurrent,
        "per_chunk_concurrency": per_chunk_concurrency,
        "sbatch_path": str(sbatch_path),
    }
    per_selection_path = submissions_dir / f"{base_name}_{stamp}_submission.json"
    write_json(per_selection_path, per_selection_payload)
    if not args.dry_run:
        append_submission_records(
            results_root,
            submissions,
            {
                "last_submission_file": str(per_selection_path),
                "last_selection_name": base_name,
                "slurm_time": manifest["slurm_time"],
                "max_concurrent": max_concurrent,
                "per_chunk_concurrency": per_chunk_concurrency,
                "sbatch_path": str(sbatch_path),
            },
        )
    print(f"submission_manifest={per_selection_path}")
    return 0


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
    append_submission_records(
        results_root,
        submissions,
        {
            "last_submission_type": "full_contiguous",
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
    if args.command == "select":
        return cmd_select(args)
    if args.command == "submit-selection":
        return cmd_submit_selection(args)
    if args.command == "summary":
        return cmd_summary(args)
    if args.command == "report":
        return cmd_report(args)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Export all-success preliminary plots for full Games 1-3 multi-agent results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = (
    PROJECT_ROOT / "experiments" / "results" / "full_games123_multiagent_production_20260428_085255"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis" / "full_games123_all_success_preliminary_20260428"

GPT5_NANO_ELO = 1337
DEFAULT_MODEL_ELOS: Dict[str, int] = {
    "gpt-5-nano": GPT5_NANO_ELO,
    "gpt-5-nano-high": GPT5_NANO_ELO,
}

MODEL_SHORT_NAMES: Dict[str, str] = {
    "amazon-nova-micro-v1.0": "Nova Micro",
    "amazon-nova-pro-v1.0": "Nova Pro",
    "claude-3-haiku-20240307": "Claude 3 Haiku",
    "claude-haiku-4-5-20251001": "Haiku 4.5",
    "claude-opus-4-5-20251101": "Opus 4.5",
    "claude-opus-4-5-20251101-thinking-32k": "Opus 4.5 Thinking",
    "claude-opus-4-6": "Opus 4.6",
    "claude-opus-4-6-thinking": "Opus 4.6 Thinking",
    "claude-sonnet-4-20250514": "Sonnet 4",
    "command-r-plus-08-2024": "Command R+",
    "deepseek-r1-0528": "DeepSeek R1-0528",
    "deepseek-v3": "DeepSeek V3",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-3-flash": "Gemini 3 Flash",
    "gemini-3-pro": "Gemini 3 Pro",
    "gemma-3-27b-it": "Gemma 3 27B",
    "gpt-4.1-nano-2025-04-14": "GPT-4.1 nano",
    "gpt-4o-2024-05-13": "GPT-4o",
    "gpt-4o-mini-2024-07-18": "GPT-4o mini",
    "gpt-5-nano": "GPT-5-nano",
    "gpt-5-nano-high": "GPT-5-nano",
    "gpt-5.2-chat-latest-20260210": "GPT-5.2 Chat",
    "gpt-5.4-high": "GPT-5.4 High",
    "grok-4": "Grok-4",
    "llama-3.3-70b-instruct": "Llama 3.3 70B",
    "o3-mini-high": "o3-mini-high",
    "qwen3-max-preview": "Qwen3 Max",
}

TOKEN_LIMIT_PATTERNS = (
    "MAX_TOKENS",
    "finish_reason=length",
    "OpenAI returned empty content",
    "Response truncated",
    "max_tokens must be greater than thinking.budget_tokens",
)
VOTE_FALLBACK_LOG_PATTERNS = (
    "defaulted all proposal votes to reject",
    "Structured batch vote request",
    "Structured batch vote retry",
    "Structured compact vote retry",
)
SYNTHETIC_VOTE_PATTERNS = (
    "Default reject because structured vote recovery failed",
    "Missing or invalid vote entry",
    "Failed to parse vote response",
    "structured_vote_recovery_failed",
    "No JSON found in batch vote response",
)
SYNTHETIC_PROPOSAL_PATTERNS = (
    "Failed to parse response - defaulting to proposer gets all",
    "No allocation in proposal",
    "No JSON found in proposal response",
    "Proposal invalid after retry",
)
PROVIDER_DEGRADATION_PATTERNS = (
    "ResourceExhausted",
    "quota",
    "rate limit",
    "RateLimitError",
    "insufficient_quota",
    "TimeoutError",
)

MARKERS_BY_N = {
    2: "o",
    4: "s",
    6: "^",
    8: "D",
    10: "P",
}
COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#7f7f7f",
]
GINI_SMALL_N_CORRECTION_THRESHOLD = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot preliminary full Games 1-3 all-success metrics."
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--exclude-synthetic-proposals",
        action="store_true",
        help=(
            "Further exclude runs with proposal parse/validation fallback markers. "
            "This keeps all-success outputs but drops proposal-fallback runs."
        ),
    )
    parser.add_argument(
        "--exclude-controls",
        action="store_true",
        help="Exclude homogeneous control runs from the exported subset CSVs.",
    )
    return parser.parse_args()


def short_model_name(model: str) -> str:
    return MODEL_SHORT_NAMES.get(model, model)


def parse_config_id(path: Path) -> Optional[int]:
    for part in path.parts:
        match = re.match(r"config_(\d{4})", part)
        if match:
            return int(match.group(1))
    return None


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def count_any(text: str, patterns: Sequence[str]) -> int:
    lowered = text.lower()
    return sum(lowered.count(pattern.lower()) for pattern in patterns)


def load_status(results_root: Path, config_id: int) -> Dict[str, Any]:
    path = results_root / "status" / f"config_{config_id:04d}.json"
    if not path.exists():
        return {"state": "NO_STATUS"}
    try:
        return load_json(path)
    except json.JSONDecodeError:
        return {"state": "INVALID_STATUS_JSON"}


def result_files(results_root: Path) -> List[Path]:
    paths = list((results_root / "runs").glob("config_*/experiment_results.json"))
    paths.extend((results_root / "runs").glob("config_*/run_1_experiment_results.json"))
    return sorted(paths, key=lambda path: parse_config_id(path) or 10**9)


def agent_sort_key(agent_id: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)$", agent_id)
    if match:
        return int(match.group(1)), agent_id
    return 10**9, agent_id


def gini(values: Sequence[float]) -> float:
    clean = np.asarray([float(value) for value in values if value is not None], dtype=float)
    if clean.size == 0:
        return math.nan
    if np.allclose(clean, clean[0]):
        return 0.0
    mean_value = float(np.mean(clean))
    if math.isclose(mean_value, 0.0):
        return 0.0
    diffs = np.abs(clean[:, None] - clean[None, :])
    return float(np.mean(diffs) / (2.0 * mean_value))


def gini_small_n_correction_factor(value_count: int) -> float:
    if 1 < value_count < GINI_SMALL_N_CORRECTION_THRESHOLD:
        return float(value_count / (value_count - 1))
    return 1.0


def linear_fit_stats(xs: Sequence[float], ys: Sequence[float]) -> Dict[str, float]:
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if x is not None
        and y is not None
        and math.isfinite(float(x))
        and math.isfinite(float(y))
    ]
    if len(pairs) < 2:
        return {
            "slope": math.nan,
            "intercept": math.nan,
            "r_squared": math.nan,
            "correlation": math.nan,
            "fit_count": float(len(pairs)),
        }

    x_arr = np.asarray([pair[0] for pair in pairs], dtype=float)
    y_arr = np.asarray([pair[1] for pair in pairs], dtype=float)
    if np.allclose(x_arr, x_arr[0]):
        return {
            "slope": math.nan,
            "intercept": math.nan,
            "r_squared": math.nan,
            "correlation": math.nan,
            "fit_count": float(len(pairs)),
        }

    slope, intercept = np.polyfit(x_arr, y_arr, deg=1)
    fitted = slope * x_arr + intercept
    ss_res = float(np.sum((y_arr - fitted) ** 2))
    ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    r_squared = 1.0 - (ss_res / ss_tot) if not math.isclose(ss_tot, 0.0) else 1.0
    correlation = 0.0 if np.allclose(y_arr, y_arr[0]) else float(np.corrcoef(x_arr, y_arr)[0, 1])
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "correlation": correlation,
        "fit_count": float(len(pairs)),
    }


def resolve_elo(agent_id: str, model: Optional[str], agent_elo_map: Dict[str, Any]) -> Optional[int]:
    value = agent_elo_map.get(agent_id)
    if value is not None and value != "":
        return int(value)
    if model in DEFAULT_MODEL_ELOS:
        return DEFAULT_MODEL_ELOS[model]
    return None


def competition_key(config: Dict[str, Any]) -> str:
    game = config.get("game_label") or config.get("game_type") or "unknown"
    if game == "game2" or (game == "unknown" and "rho" in config and "theta" in config):
        rho = float(config.get("rho", 0.0))
        theta = float(config.get("theta", 0.0))
        return f"rho={rho:.3f}, theta={theta:.2f}"
    if game == "game3" or (game == "unknown" and "sigma" in config and "alpha" in config):
        sigma = float(config.get("sigma", 0.0))
        alpha = float(config.get("alpha", 0.0))
        return f"sigma={sigma:.2f}, alpha={alpha:.2f}"
    if game == "game1" or "competition_level" in config:
        return f"comp={float(config.get('competition_level', 0.0)):.2f}"
    return "unknown"


def numeric_competition_value(config: Dict[str, Any]) -> Optional[float]:
    game = config.get("game_label")
    if game == "game2" and config.get("rho") is not None and config.get("theta") is not None:
        return float(config["rho"]) * float(config["theta"])
    if game == "game3" and config.get("sigma") is not None and config.get("alpha") is not None:
        return float(config["sigma"]) * float(config["alpha"])
    if config.get("competition_level") is not None:
        return float(config["competition_level"])
    return None


def row_base(
    config_id: int,
    result_path: Path,
    payload: Dict[str, Any],
    status: Dict[str, Any],
    interaction_text: str,
    log_text: str,
) -> Dict[str, Any]:
    config = payload.get("config") or {}
    token_limit_count = count_any(log_text, TOKEN_LIMIT_PATTERNS)
    vote_fallback_log_count = count_any(log_text, VOTE_FALLBACK_LOG_PATTERNS)
    synthetic_vote_count = count_any(interaction_text, SYNTHETIC_VOTE_PATTERNS)
    synthetic_proposal_count = count_any(interaction_text, SYNTHETIC_PROPOSAL_PATTERNS)
    provider_degradation_count = count_any(log_text + "\n" + interaction_text, PROVIDER_DEGRADATION_PATTERNS)

    recomputed_strict = (
        status.get("state") == "SUCCESS"
        and result_path.exists()
        and synthetic_vote_count == 0
        and token_limit_count == 0
        and vote_fallback_log_count == 0
    )
    return {
        "config_id": config_id,
        "result_path": str(result_path),
        "run_dir": str(result_path.parent),
        "status": status.get("state"),
        "started_at": status.get("started_at"),
        "finished_at": status.get("finished_at"),
        "duration_seconds": status.get("duration_seconds"),
        "game_label": config.get("game_label"),
        "game_type": config.get("game_type"),
        "experiment_family": config.get("experiment_family"),
        "n_agents": config.get("n_agents"),
        "m_items": config.get("m_items"),
        "n_issues": config.get("n_issues"),
        "m_projects": config.get("m_projects"),
        "competition_key": competition_key(config),
        "competition_value": numeric_competition_value(config),
        "competition_level": config.get("competition_level"),
        "rho": config.get("rho"),
        "theta": config.get("theta"),
        "sigma": config.get("sigma"),
        "alpha": config.get("alpha"),
        "random_seed": config.get("random_seed") or config.get("seed"),
        "seed_replicate": config.get("seed_replicate"),
        "heterogeneous_run_index": config.get("heterogeneous_run_index"),
        "heterogeneous_draw_seed": config.get("heterogeneous_draw_seed"),
        "adversary_model": config.get("adversary_model"),
        "adversary_position": config.get("adversary_position"),
        "model_order": config.get("model_order"),
        "max_tokens_voting": config.get("max_tokens_voting"),
        "parallel_phases": config.get("parallel_phases"),
        "consensus_reached": payload.get("consensus_reached"),
        "final_round": payload.get("final_round"),
        "strict_voting_clean": bool(recomputed_strict),
        "token_limit_marker_count": token_limit_count,
        "vote_fallback_log_marker_count": vote_fallback_log_count,
        "synthetic_vote_marker_count": synthetic_vote_count,
        "synthetic_proposal_marker_count": synthetic_proposal_count,
        "provider_degradation_marker_count": provider_degradation_count,
    }


def build_tables(results_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    run_rows: List[Dict[str, Any]] = []
    agent_rows: List[Dict[str, Any]] = []

    for result_path in result_files(results_root):
        config_id = parse_config_id(result_path)
        if config_id is None:
            continue
        payload = load_json(result_path)
        config = payload.get("config") or {}
        status = load_status(results_root, config_id)
        interaction_path = result_path.with_name("all_interactions.json")
        log_path = results_root / "logs" / f"config_{config_id:04d}.log"
        interaction_text = maybe_read_text(interaction_path)
        log_text = maybe_read_text(log_path)

        base = row_base(
            config_id=config_id,
            result_path=result_path,
            payload=payload,
            status=status,
            interaction_text=interaction_text,
            log_text=log_text,
        )

        final_utilities = payload.get("final_utilities") or {}
        agent_performance = payload.get("agent_performance") or {}
        agent_model_map = config.get("agent_model_map") or {}
        agent_elo_map = config.get("agent_elo_map") or {}
        agent_role_map = config.get("agent_role_map") or {}
        agent_ids = sorted(
            set(final_utilities) | set(agent_model_map) | set(agent_role_map),
            key=agent_sort_key,
        )

        utilities: List[float] = []
        elos: List[int] = []
        payoff_fit_elos: List[float] = []
        payoff_fit_utilities: List[float] = []
        adversary_utilities: List[float] = []
        baseline_utilities: List[float] = []
        adversary_elos: List[int] = []
        model_counts: Counter[str] = Counter()

        for agent_id in agent_ids:
            model = agent_model_map.get(agent_id)
            role = agent_role_map.get(agent_id)
            utility = final_utilities.get(agent_id)
            if utility is None and agent_id in agent_performance:
                utility = agent_performance[agent_id].get("final_utility")
            if utility is not None:
                utility = float(utility)
                utilities.append(utility)
            elo = resolve_elo(agent_id, model, agent_elo_map)
            if elo is not None:
                elos.append(int(elo))
            if utility is not None and elo is not None:
                payoff_fit_elos.append(float(elo))
                payoff_fit_utilities.append(float(utility))
            if model:
                model_counts[model] += 1
            if role == "adversary":
                if utility is not None:
                    adversary_utilities.append(float(utility))
                if elo is not None:
                    adversary_elos.append(int(elo))
            if role == "baseline":
                if utility is not None:
                    baseline_utilities.append(float(utility))

            agent_row = {
                **base,
                "agent_id": agent_id,
                "agent_index": agent_sort_key(agent_id)[0],
                "model": model,
                "model_short": short_model_name(model) if model else None,
                "role": role,
                "elo": elo,
                "final_utility": utility,
            }
            agent_rows.append(agent_row)

        utility_count = len(utilities)
        utility_gini_raw = gini(utilities)
        utility_gini_correction_factor = gini_small_n_correction_factor(utility_count)
        utility_gini_corrected = (
            utility_gini_raw * utility_gini_correction_factor
            if math.isfinite(utility_gini_raw)
            else math.nan
        )
        payoff_elo_fit = linear_fit_stats(payoff_fit_elos, payoff_fit_utilities)
        payoff_elo_slope = payoff_elo_fit["slope"]
        payoff_elo_abs_slope = abs(payoff_elo_slope) if math.isfinite(payoff_elo_slope) else math.nan

        base.update(
            {
                "agent_count_in_result": len(agent_ids),
                "model_list": "+".join(config.get("models") or []),
                "unique_model_count": len(model_counts),
                "utility_count": utility_count,
                "utility_min": min(utilities) if utilities else math.nan,
                "utility_max": max(utilities) if utilities else math.nan,
                "utility_range": (max(utilities) - min(utilities)) if utilities else math.nan,
                "mean_utility": mean(utilities) if utilities else math.nan,
                "sum_utility": sum(utilities) if utilities else math.nan,
                "utility_variance": float(np.var(utilities)) if utilities else math.nan,
                "utility_std": float(np.std(utilities)) if utilities else math.nan,
                "utility_gini": utility_gini_corrected,
                "utility_gini_raw": utility_gini_raw,
                "utility_gini_corrected": utility_gini_corrected,
                "utility_gini_correction_factor": utility_gini_correction_factor,
                "utility_gini_small_n_corrected": utility_gini_correction_factor != 1.0,
                "payoff_elo_fit_agent_count": int(payoff_elo_fit["fit_count"]),
                "payoff_elo_slope": payoff_elo_slope,
                "payoff_elo_slope_per_100_elo": (
                    payoff_elo_slope * 100.0 if math.isfinite(payoff_elo_slope) else math.nan
                ),
                "payoff_elo_abs_slope": payoff_elo_abs_slope,
                "payoff_elo_abs_slope_per_100_elo": (
                    payoff_elo_abs_slope * 100.0
                    if math.isfinite(payoff_elo_abs_slope)
                    else math.nan
                ),
                "payoff_elo_intercept": payoff_elo_fit["intercept"],
                "payoff_elo_r_squared": payoff_elo_fit["r_squared"],
                "payoff_elo_correlation": payoff_elo_fit["correlation"],
                "elo_variance": float(np.var(elos)) if elos else math.nan,
                "elo_std": float(np.std(elos)) if elos else math.nan,
                "elo_min": min(elos) if elos else math.nan,
                "elo_max": max(elos) if elos else math.nan,
                "adversary_elo": mean(adversary_elos) if adversary_elos else math.nan,
                "adversary_utility": (
                    mean(adversary_utilities) if adversary_utilities else math.nan
                ),
                "baseline_mean_utility": (
                    mean(baseline_utilities) if baseline_utilities else math.nan
                ),
                "baseline_sum_utility": (
                    sum(baseline_utilities) if baseline_utilities else math.nan
                ),
                "baseline_agent_count": len(baseline_utilities),
            }
        )
        run_rows.append(base)

    return pd.DataFrame(run_rows), pd.DataFrame(agent_rows)


def filter_subset(
    runs: pd.DataFrame,
    agents: pd.DataFrame,
    exclude_synthetic_proposals: bool,
    exclude_controls: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mask = pd.Series(True, index=runs.index)
    if exclude_synthetic_proposals:
        mask &= runs["synthetic_proposal_marker_count"].fillna(0).astype(int).eq(0)
    if exclude_controls:
        mask &= runs["experiment_family"].ne("homogeneous_control")
    filtered_runs = runs[mask].copy()
    keep_ids = set(filtered_runs["config_id"].astype(int).tolist())
    filtered_agents = agents[agents["config_id"].astype(int).isin(keep_ids)].copy()
    return filtered_runs, filtered_agents


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def group_count(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[*columns, "run_count"])
    return (
        frame.groupby(list(columns), dropna=False)
        .size()
        .reset_index(name="run_count")
        .sort_values([*columns])
    )


def aggregate_homogeneous_adversary(runs: pd.DataFrame) -> pd.DataFrame:
    hom = runs[runs["experiment_family"].eq("homogeneous_adversary")].copy()
    if hom.empty:
        return pd.DataFrame()
    group_cols = [
        "game_label",
        "n_agents",
        "competition_key",
        "competition_value",
        "adversary_model",
        "adversary_elo",
        "adversary_position",
    ]
    return (
        hom.groupby(group_cols, dropna=False)
        .agg(
            run_count=("config_id", "count"),
            adversary_utility_mean=("adversary_utility", "mean"),
            adversary_utility_std=("adversary_utility", "std"),
            baseline_mean_utility_mean=("baseline_mean_utility", "mean"),
            baseline_mean_utility_std=("baseline_mean_utility", "std"),
            mean_utility=("mean_utility", "mean"),
            consensus_rate=("consensus_reached", "mean"),
        )
        .reset_index()
        .sort_values(["game_label", "n_agents", "competition_key", "adversary_elo"])
    )


def aggregate_homogeneous_adversary_pooled_positions(runs: pd.DataFrame) -> pd.DataFrame:
    hom = runs[runs["experiment_family"].eq("homogeneous_adversary")].copy()
    if hom.empty:
        return pd.DataFrame()
    group_cols = [
        "game_label",
        "n_agents",
        "competition_key",
        "competition_value",
        "adversary_model",
        "adversary_elo",
    ]
    return (
        hom.groupby(group_cols, dropna=False)
        .agg(
            run_count=("config_id", "count"),
            adversary_utility_mean=("adversary_utility", "mean"),
            adversary_utility_std=("adversary_utility", "std"),
            baseline_mean_utility_mean=("baseline_mean_utility", "mean"),
            baseline_mean_utility_std=("baseline_mean_utility", "std"),
            mean_utility=("mean_utility", "mean"),
            consensus_rate=("consensus_reached", "mean"),
        )
        .reset_index()
        .sort_values(["game_label", "n_agents", "competition_key", "adversary_elo"])
    )


def aggregate_heterogeneous_agents(agents: pd.DataFrame) -> pd.DataFrame:
    hetero = agents[agents["experiment_family"].eq("heterogeneous_random")].copy()
    if hetero.empty:
        return pd.DataFrame()
    group_cols = ["game_label", "n_agents", "competition_key", "model", "model_short", "elo"]
    return (
        hetero.groupby(group_cols, dropna=False)
        .agg(
            agent_run_count=("config_id", "count"),
            final_utility_mean=("final_utility", "mean"),
            final_utility_std=("final_utility", "std"),
        )
        .reset_index()
        .sort_values(["game_label", "n_agents", "competition_key", "elo"])
    )


def aggregate_heterogeneous_agents_custom(
    agents: pd.DataFrame,
    group_cols: Sequence[str],
) -> pd.DataFrame:
    hetero = agents[agents["experiment_family"].eq("heterogeneous_random")].copy()
    if hetero.empty:
        return pd.DataFrame()

    full_group_cols = ["game_label", *group_cols, "model", "model_short", "elo"]
    full_group_cols = list(dict.fromkeys(full_group_cols))
    summary = (
        hetero.groupby(full_group_cols, dropna=False)
        .agg(
            agent_run_count=("config_id", "count"),
            final_utility_mean=("final_utility", "mean"),
            final_utility_std=("final_utility", "std"),
        )
        .reset_index()
    )
    sort_cols = [col for col in ["game_label", *group_cols, "elo"] if col in summary.columns]
    return summary.sort_values(sort_cols)


def aggregate_homogeneous_adversary_custom(
    runs: pd.DataFrame,
    group_cols: Sequence[str],
) -> pd.DataFrame:
    hom = runs[runs["experiment_family"].eq("homogeneous_adversary")].copy()
    if hom.empty:
        return pd.DataFrame()

    full_group_cols = ["game_label", *group_cols, "adversary_model", "adversary_elo"]
    full_group_cols = list(dict.fromkeys(full_group_cols))
    summary = (
        hom.groupby(full_group_cols, dropna=False)
        .agg(
            run_count=("config_id", "count"),
            adversary_utility_mean=("adversary_utility", "mean"),
            adversary_utility_std=("adversary_utility", "std"),
            baseline_mean_utility_mean=("baseline_mean_utility", "mean"),
            baseline_mean_utility_std=("baseline_mean_utility", "std"),
            mean_utility=("mean_utility", "mean"),
            consensus_rate=("consensus_reached", "mean"),
        )
        .reset_index()
    )
    sort_cols = [col for col in ["game_label", *group_cols, "adversary_elo"] if col in summary.columns]
    return summary.sort_values(sort_cols)


def aggregate_heterogeneous_gini(
    runs: pd.DataFrame,
    group_cols: Sequence[str],
) -> pd.DataFrame:
    hetero = runs[runs["experiment_family"].eq("heterogeneous_random")].copy()
    if hetero.empty:
        return pd.DataFrame()
    effective_group_cols = list(group_cols)
    if not effective_group_cols:
        hetero["overall_group"] = "overall"
        effective_group_cols = ["overall_group"]
    summary = (
        hetero.groupby(effective_group_cols, dropna=False)
        .agg(
            run_count=("config_id", "count"),
            elo_variance_mean=("elo_variance", "mean"),
            elo_variance_std=("elo_variance", "std"),
            elo_std_mean=("elo_std", "mean"),
            elo_std_std=("elo_std", "std"),
            utility_gini_mean=("utility_gini", "mean"),
            utility_gini_std=("utility_gini", "std"),
            utility_gini_raw_mean=("utility_gini_raw", "mean"),
            utility_gini_raw_std=("utility_gini_raw", "std"),
            utility_gini_correction_factor_mean=("utility_gini_correction_factor", "mean"),
            utility_std_mean=("utility_std", "mean"),
            utility_std_std=("utility_std", "std"),
            utility_variance_mean=("utility_variance", "mean"),
            utility_variance_std=("utility_variance", "std"),
            utility_range_mean=("utility_range", "mean"),
            utility_range_std=("utility_range", "std"),
            payoff_elo_slope_mean=("payoff_elo_slope_per_100_elo", "mean"),
            payoff_elo_slope_std=("payoff_elo_slope_per_100_elo", "std"),
            payoff_elo_abs_slope_mean=("payoff_elo_abs_slope_per_100_elo", "mean"),
            payoff_elo_abs_slope_std=("payoff_elo_abs_slope_per_100_elo", "std"),
            payoff_elo_r_squared_mean=("payoff_elo_r_squared", "mean"),
            payoff_elo_correlation_mean=("payoff_elo_correlation", "mean"),
            payoff_elo_fit_agent_count_mean=("payoff_elo_fit_agent_count", "mean"),
            mean_utility=("mean_utility", "mean"),
            consensus_rate=("consensus_reached", "mean"),
        )
        .reset_index()
    )
    counts = summary["run_count"].clip(lower=1).astype(float)
    summary["elo_variance_sem"] = summary["elo_variance_std"] / np.sqrt(counts)
    summary["elo_std_sem"] = summary["elo_std_std"] / np.sqrt(counts)
    summary["utility_gini_sem"] = summary["utility_gini_std"] / np.sqrt(counts)
    summary["utility_gini_raw_sem"] = summary["utility_gini_raw_std"] / np.sqrt(counts)
    summary["utility_std_sem"] = summary["utility_std_std"] / np.sqrt(counts)
    summary["utility_variance_sem"] = summary["utility_variance_std"] / np.sqrt(counts)
    summary["utility_range_sem"] = summary["utility_range_std"] / np.sqrt(counts)
    summary["payoff_elo_slope_sem"] = summary["payoff_elo_slope_std"] / np.sqrt(counts)
    summary["payoff_elo_abs_slope_sem"] = summary["payoff_elo_abs_slope_std"] / np.sqrt(counts)
    return summary.sort_values(effective_group_cols)


def ordered_values(frame: pd.DataFrame, column: str) -> List[Any]:
    if column not in frame.columns:
        return []
    values = frame[column].dropna().unique().tolist()
    if column == "n_agents":
        return sorted(values, key=lambda value: int(value))
    if column == "competition_key" and "competition_value" in frame.columns:
        value_rows = (
            frame[[column, "competition_value"]]
            .drop_duplicates()
            .sort_values(["competition_value", column], na_position="last")
        )
        return value_rows[column].dropna().tolist()
    return sorted(values, key=lambda value: str(value))


def grid_shape(panel_count: int, max_cols: int = 3) -> Tuple[int, int]:
    if panel_count <= 0:
        return 0, 0
    cols = min(max_cols, panel_count)
    rows = int(math.ceil(panel_count / cols))
    return rows, cols


def label_for_group(column: str, value: Any) -> str:
    if column == "n_agents":
        return f"N={int(value)}"
    if column == "competition_key":
        return str(value)
    if column == "overall_group":
        return "overall"
    return f"{column}={value}"


def color_map_for(values: Sequence[Any]) -> Dict[Any, str]:
    return {value: COLORS[index % len(COLORS)] for index, value in enumerate(values)}


def scatter_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: Optional[str] = None,
) -> None:
    if frame.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return
    if group_col is None or group_col not in frame.columns:
        ax.scatter(
            frame[x_col],
            frame[y_col],
            s=52,
            alpha=0.78,
            color="#1f77b4",
            edgecolor="white",
            linewidth=0.55,
        )
        return

    values = ordered_values(frame, group_col)
    colors = color_map_for(values)
    for value in values:
        group = frame[frame[group_col].eq(value)]
        marker = MARKERS_BY_N.get(int(value), "o") if group_col == "n_agents" else "o"
        ax.scatter(
            group[x_col],
            group[y_col],
            s=52,
            alpha=0.78,
            marker=marker,
            color=colors.get(value, "#1f77b4"),
            edgecolor="white",
            linewidth=0.55,
            label=label_for_group(group_col, value),
        )
    if len(values) > 1:
        ax.legend(fontsize=7, frameon=False, loc="best")


def line_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: Optional[str] = None,
) -> None:
    if frame.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return
    if group_col is None or group_col not in frame.columns:
        group = frame.sort_values(x_col)
        ax.plot(group[x_col], group[y_col], marker="o", linewidth=1.25, markersize=5)
        return

    values = ordered_values(frame, group_col)
    colors = color_map_for(values)
    for value in values:
        group = frame[frame[group_col].eq(value)].sort_values(x_col)
        marker = MARKERS_BY_N.get(int(value), "o") if group_col == "n_agents" else "o"
        ax.plot(
            group[x_col],
            group[y_col],
            marker=marker,
            linewidth=1.25,
            markersize=5,
            color=colors.get(value, "#1f77b4"),
            label=label_for_group(group_col, value),
        )
    if len(values) > 1:
        ax.legend(fontsize=7, frameon=False, loc="best")


def scatter_by_setting(
    frame: pd.DataFrame,
    output_path: Path,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    plot_df = frame.dropna(subset=[x_col, y_col]).copy()
    if plot_df.empty:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    comp_values = sorted(plot_df["competition_key"].dropna().unique().tolist())
    color_by_comp = {value: COLORS[index % len(COLORS)] for index, value in enumerate(comp_values)}

    for (n_agents, comp_key), group in plot_df.groupby(["n_agents", "competition_key"], dropna=False):
        marker = MARKERS_BY_N.get(int(n_agents), "o") if not pd.isna(n_agents) else "o"
        color = color_by_comp.get(comp_key, "#1f77b4")
        ax.scatter(
            group[x_col],
            group[y_col],
            s=58,
            alpha=0.78,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            label=f"N={int(n_agents)} | {comp_key}",
        )

    if plot_df[x_col].nunique() >= 2 and len(plot_df) >= 3:
        xs = plot_df[x_col].astype(float).to_numpy()
        ys = plot_df[y_col].astype(float).to_numpy()
        slope, intercept = np.polyfit(xs, ys, deg=1)
        line_x = np.linspace(float(xs.min()), float(xs.max()), 200)
        ax.plot(line_x, slope * line_x + intercept, color="#222222", linewidth=1.4, alpha=0.75)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.grid(True, alpha=0.22)
    ax.legend(fontsize=8, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def line_summary_plot(
    summary: pd.DataFrame,
    output_path: Path,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    plot_df = summary.dropna(subset=[x_col, y_col]).copy()
    if plot_df.empty:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    comp_values = sorted(plot_df["competition_key"].dropna().unique().tolist())
    color_by_comp = {value: COLORS[index % len(COLORS)] for index, value in enumerate(comp_values)}
    for (n_agents, comp_key), group in plot_df.groupby(["n_agents", "competition_key"], dropna=False):
        group = group.sort_values(x_col)
        marker = MARKERS_BY_N.get(int(n_agents), "o") if not pd.isna(n_agents) else "o"
        ax.plot(
            group[x_col],
            group[y_col],
            marker=marker,
            linewidth=1.4,
            markersize=6,
            color=color_by_comp.get(comp_key, "#1f77b4"),
            label=f"N={int(n_agents)} | {comp_key}",
        )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.grid(True, alpha=0.22)
    ax.legend(fontsize=8, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def faceted_scatter_plot(
    frame: pd.DataFrame,
    output_path: Path,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
    mode: str,
) -> None:
    plot_df = frame.dropna(subset=[x_col, y_col]).copy()
    if plot_df.empty:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "n_comp_grid":
        n_values = ordered_values(plot_df, "n_agents")
        comp_values = ordered_values(plot_df, "competition_key")
        if not n_values or not comp_values:
            return
        fig, axes = plt.subplots(
            len(n_values),
            len(comp_values),
            figsize=(3.7 * len(comp_values), 2.9 * len(n_values)),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        for row_idx, n_value in enumerate(n_values):
            for col_idx, comp_value in enumerate(comp_values):
                ax = axes[row_idx][col_idx]
                subset = plot_df[
                    plot_df["n_agents"].eq(n_value) & plot_df["competition_key"].eq(comp_value)
                ]
                scatter_panel(ax, subset, x_col, y_col)
                if row_idx == 0:
                    ax.set_title(str(comp_value), fontsize=9)
                if col_idx == 0:
                    ax.set_ylabel(f"N={int(n_value)}\n{y_label}", fontsize=9)
                if row_idx == len(n_values) - 1:
                    ax.set_xlabel(x_label, fontsize=9)
                ax.grid(True, alpha=0.18)
        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        return

    if mode == "by_n":
        facet_col = "n_agents"
        group_col = "competition_key"
        max_cols = 3
    elif mode == "by_competition":
        facet_col = "competition_key"
        group_col = "n_agents"
        max_cols = 3
    else:
        raise ValueError(f"Unknown facet mode: {mode}")

    values = ordered_values(plot_df, facet_col)
    rows, cols = grid_shape(len(values), max_cols=max_cols)
    if rows == 0:
        return
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4.3 * cols, 3.25 * rows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for index, value in enumerate(values):
        row_idx, col_idx = divmod(index, cols)
        ax = axes[row_idx][col_idx]
        subset = plot_df[plot_df[facet_col].eq(value)]
        scatter_panel(ax, subset, x_col, y_col, group_col=group_col)
        ax.set_title(label_for_group(facet_col, value), fontsize=10)
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.grid(True, alpha=0.18)
    for index in range(len(values), rows * cols):
        row_idx, col_idx = divmod(index, cols)
        axes[row_idx][col_idx].axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def faceted_line_plot(
    frame: pd.DataFrame,
    output_path: Path,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
    mode: str,
) -> None:
    plot_df = frame.dropna(subset=[x_col, y_col]).copy()
    if plot_df.empty:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "n_comp_grid":
        n_values = ordered_values(plot_df, "n_agents")
        comp_values = ordered_values(plot_df, "competition_key")
        if not n_values or not comp_values:
            return
        fig, axes = plt.subplots(
            len(n_values),
            len(comp_values),
            figsize=(3.7 * len(comp_values), 2.9 * len(n_values)),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        for row_idx, n_value in enumerate(n_values):
            for col_idx, comp_value in enumerate(comp_values):
                ax = axes[row_idx][col_idx]
                subset = plot_df[
                    plot_df["n_agents"].eq(n_value) & plot_df["competition_key"].eq(comp_value)
                ]
                line_panel(ax, subset, x_col, y_col)
                if row_idx == 0:
                    ax.set_title(str(comp_value), fontsize=9)
                if col_idx == 0:
                    ax.set_ylabel(f"N={int(n_value)}\n{y_label}", fontsize=9)
                if row_idx == len(n_values) - 1:
                    ax.set_xlabel(x_label, fontsize=9)
                ax.grid(True, alpha=0.18)
        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        return

    if mode == "by_n":
        facet_col = "n_agents"
        group_col = "competition_key"
        max_cols = 3
    elif mode == "by_competition":
        facet_col = "competition_key"
        group_col = "n_agents"
        max_cols = 3
    else:
        raise ValueError(f"Unknown facet mode: {mode}")

    values = ordered_values(plot_df, facet_col)
    rows, cols = grid_shape(len(values), max_cols=max_cols)
    if rows == 0:
        return
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4.3 * cols, 3.25 * rows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for index, value in enumerate(values):
        row_idx, col_idx = divmod(index, cols)
        ax = axes[row_idx][col_idx]
        subset = plot_df[plot_df[facet_col].eq(value)]
        line_panel(ax, subset, x_col, y_col, group_col=group_col)
        ax.set_title(label_for_group(facet_col, value), fontsize=10)
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.grid(True, alpha=0.18)
    for index in range(len(values), rows * cols):
        row_idx, col_idx = divmod(index, cols)
        axes[row_idx][col_idx].axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def error_scatter_summary_plot(
    summary: pd.DataFrame,
    output_path: Path,
    title: str,
    group_col: Optional[str] = None,
    x_mean_col: str = "elo_variance_mean",
    x_sem_col: str = "elo_variance_sem",
    x_label: str = "Mean within-run Elo variance",
    y_mean_col: str = "utility_gini_mean",
    y_sem_col: str = "utility_gini_sem",
    y_label: str = "Mean final utility Gini coefficient",
) -> None:
    plot_df = summary.dropna(subset=[x_mean_col, y_mean_col]).copy()
    if plot_df.empty:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.3, 6.1))
    if group_col is None or group_col not in plot_df.columns:
        groups = [(None, plot_df)]
        colors = {None: "#1f77b4"}
    else:
        values = ordered_values(plot_df, group_col)
        groups = [(value, plot_df[plot_df[group_col].eq(value)]) for value in values]
        colors = color_map_for(values)

    for value, group in groups:
        marker = MARKERS_BY_N.get(int(value), "o") if group_col == "n_agents" else "o"
        label = label_for_group(group_col, value) if group_col else None
        ax.errorbar(
            group[x_mean_col],
            group[y_mean_col],
            xerr=group[x_sem_col].fillna(0.0),
            yerr=group[y_sem_col].fillna(0.0),
            fmt=marker,
            markersize=6,
            alpha=0.82,
            color=colors.get(value, "#1f77b4"),
            ecolor=colors.get(value, "#1f77b4"),
            elinewidth=0.9,
            capsize=2,
            label=label,
        )
        for _, row in group.iterrows():
            if "run_count" in row and int(row["run_count"]) > 1:
                ax.annotate(
                    f"n={int(row['run_count'])}",
                    (row[x_mean_col], row[y_mean_col]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                    alpha=0.75,
                )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.grid(True, alpha=0.22)
    if group_col:
        ax.legend(fontsize=8, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def grouped_elo_line_plot(
    summary: pd.DataFrame,
    output_path: Path,
    x_col: str,
    y_col: str,
    title: str,
    y_label: str,
    group_col: Optional[str] = None,
    x_label: str = "Adversary Elo",
) -> None:
    plot_df = summary.dropna(subset=[x_col, y_col]).copy()
    if plot_df.empty:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.7, 6.0))
    if group_col is None or group_col not in plot_df.columns:
        line_panel(ax, plot_df, x_col, y_col)
    else:
        line_panel(ax, plot_df, x_col, y_col, group_col=group_col)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def iter_game_groups(frame: pd.DataFrame):
    if frame.empty or "game_label" not in frame.columns:
        return []
    return frame.groupby("game_label", dropna=False)


def plot_outputs(runs: pd.DataFrame, agents: pd.DataFrame, output_dir: Path, prefix: str) -> List[Path]:
    plot_paths: List[Path] = []

    hetero = runs[runs["experiment_family"].eq("heterogeneous_random")].copy()
    hetero_x_specs = [
        ("elo_stddev", "elo_std", "within-run Elo stddev", "Within-run Elo stddev"),
        ("elo_variance", "elo_variance", "within-run Elo variance", "Within-run Elo variance"),
    ]
    hetero_y_specs = [
        (
            "gini",
            "utility_gini",
            "utility Gini (small-N corrected)",
            "Final utility Gini coefficient (small-N corrected)",
        ),
        (
            "utility_std",
            "utility_std",
            "utility stddev",
            "Final utility stddev",
        ),
        (
            "payoff_elo_slope",
            "payoff_elo_slope_per_100_elo",
            "payoff-vs-Elo slope",
            "Final utility slope per 100 Elo",
        ),
        (
            "payoff_elo_abs_slope",
            "payoff_elo_abs_slope_per_100_elo",
            "absolute payoff-vs-Elo slope",
            "Absolute final utility slope per 100 Elo",
        ),
    ]
    for game_label, game_df in hetero.groupby("game_label", dropna=False):
        for y_slug, y_col, y_title_label, y_axis_label in hetero_y_specs:
            for metric_slug, x_col, title_label, axis_label in hetero_x_specs:
                path = output_dir / f"{prefix}_{game_label}_heterogeneous_{y_slug}_vs_{metric_slug}.png"
                scatter_by_setting(
                    game_df,
                    path,
                    x_col=x_col,
                    y_col=y_col,
                    title=f"{game_label}: {y_title_label} vs {title_label}",
                    x_label=axis_label,
                    y_label=y_axis_label,
                )
                if path.exists():
                    plot_paths.append(path)

                for mode, suffix in [
                    ("n_comp_grid", "faceted_n_by_competition"),
                    ("by_n", "faceted_by_n"),
                    ("by_competition", "faceted_by_competition"),
                ]:
                    path = (
                        output_dir
                        / f"{prefix}_{game_label}_heterogeneous_{y_slug}_vs_{metric_slug}_{suffix}.png"
                    )
                    faceted_scatter_plot(
                        game_df,
                        path,
                        x_col=x_col,
                        y_col=y_col,
                        title=f"{game_label}: {y_title_label} vs {title_label}",
                        x_label=axis_label,
                        y_label=y_axis_label,
                        mode=mode,
                    )
                    if path.exists():
                        plot_paths.append(path)

    hetero_gini_by_n_comp = aggregate_heterogeneous_gini(
        runs,
        ["game_label", "n_agents", "competition_key", "competition_value"],
    )
    hetero_gini_by_n = aggregate_heterogeneous_gini(runs, ["game_label", "n_agents"])
    hetero_gini_by_comp = aggregate_heterogeneous_gini(
        runs,
        ["game_label", "competition_key", "competition_value"],
    )
    hetero_gini_overall = aggregate_heterogeneous_gini(runs, ["game_label"])
    summary_x_specs = [
        (
            "elo_stddev",
            "elo_std_mean",
            "elo_std_sem",
            "mean Elo stddev",
            "Mean within-run Elo stddev",
        ),
        (
            "elo_variance",
            "elo_variance_mean",
            "elo_variance_sem",
            "mean Elo variance",
            "Mean within-run Elo variance",
        ),
    ]
    summary_y_specs = [
        (
            "gini",
            "utility_gini_mean",
            "utility_gini_sem",
            "mean utility Gini (small-N corrected)",
            "Mean final utility Gini coefficient (small-N corrected)",
        ),
        (
            "utility_std",
            "utility_std_mean",
            "utility_std_sem",
            "mean utility stddev",
            "Mean final utility stddev",
        ),
        (
            "payoff_elo_slope",
            "payoff_elo_slope_mean",
            "payoff_elo_slope_sem",
            "mean payoff-vs-Elo slope",
            "Mean final utility slope per 100 Elo",
        ),
        (
            "payoff_elo_abs_slope",
            "payoff_elo_abs_slope_mean",
            "payoff_elo_abs_slope_sem",
            "mean absolute payoff-vs-Elo slope",
            "Mean absolute final utility slope per 100 Elo",
        ),
    ]
    summary_frames = [
        (
            hetero_gini_by_n_comp,
            "by_n_competition",
            "by N and competition",
            "n_agents",
        ),
        (
            hetero_gini_by_n,
            "avg_over_competition_by_n",
            "averaged over competition",
            "n_agents",
        ),
        (
            hetero_gini_by_comp,
            "avg_over_n_by_competition",
            "averaged over N",
            "competition_key",
        ),
        (
            hetero_gini_overall,
            "avg_over_n_and_competition",
            "averaged over N and competition",
            None,
        ),
    ]
    for y_slug, y_mean_col, y_sem_col, title_y, y_axis_label in summary_y_specs:
        for metric_slug, x_mean_col, x_sem_col, title_metric, x_axis_label in summary_x_specs:
            for summary, suffix, title_suffix, group_col in summary_frames:
                for game_label, game_df in iter_game_groups(summary):
                    path = (
                        output_dir
                        / f"{prefix}_{game_label}_heterogeneous_mean_{y_slug}_vs_{metric_slug}_{suffix}.png"
                    )
                    error_scatter_summary_plot(
                        game_df,
                        path,
                        title=f"{game_label}: {title_y} vs {title_metric} {title_suffix}",
                        group_col=group_col,
                        x_mean_col=x_mean_col,
                        x_sem_col=x_sem_col,
                        x_label=x_axis_label,
                        y_mean_col=y_mean_col,
                        y_sem_col=y_sem_col,
                        y_label=y_axis_label,
                    )
                    if path.exists():
                        plot_paths.append(path)

    hom = runs[runs["experiment_family"].eq("homogeneous_adversary")].copy()
    hom_summary = aggregate_homogeneous_adversary_pooled_positions(hom)
    for game_label, game_df in hom.groupby("game_label", dropna=False):
        path = output_dir / f"{prefix}_{game_label}_hom_adversary_payoff_vs_elo_raw.png"
        scatter_by_setting(
            game_df,
            path,
            x_col="adversary_elo",
            y_col="adversary_utility",
            title=f"{game_label}: adversary payoff vs adversary Elo",
            x_label="Adversary Elo",
            y_label="Adversary final utility",
        )
        if path.exists():
            plot_paths.append(path)

        path = output_dir / f"{prefix}_{game_label}_hom_baseline_payoff_vs_adversary_elo_raw.png"
        scatter_by_setting(
            game_df,
            path,
            x_col="adversary_elo",
            y_col="baseline_mean_utility",
            title=f"{game_label}: baseline payoff vs adversary Elo",
            x_label="Adversary Elo",
            y_label="Mean GPT-5-nano baseline final utility",
        )
        if path.exists():
            plot_paths.append(path)

        for mode, suffix in [
            ("n_comp_grid", "faceted_n_by_competition"),
            ("by_n", "faceted_by_n"),
            ("by_competition", "faceted_by_competition"),
        ]:
            path = output_dir / f"{prefix}_{game_label}_hom_adversary_payoff_vs_elo_raw_{suffix}.png"
            faceted_scatter_plot(
                game_df,
                path,
                x_col="adversary_elo",
                y_col="adversary_utility",
                title=f"{game_label}: adversary payoff vs adversary Elo",
                x_label="Adversary Elo",
                y_label="Adversary final utility",
                mode=mode,
            )
            if path.exists():
                plot_paths.append(path)

            path = output_dir / f"{prefix}_{game_label}_hom_baseline_payoff_vs_adversary_elo_raw_{suffix}.png"
            faceted_scatter_plot(
                game_df,
                path,
                x_col="adversary_elo",
                y_col="baseline_mean_utility",
                title=f"{game_label}: baseline payoff vs adversary Elo",
                x_label="Adversary Elo",
                y_label="Mean GPT-5-nano baseline final utility",
                mode=mode,
            )
            if path.exists():
                plot_paths.append(path)

    for game_label, game_df in iter_game_groups(hom_summary):
        path = output_dir / f"{prefix}_{game_label}_hom_adversary_payoff_vs_elo_mean.png"
        line_summary_plot(
            game_df,
            path,
            x_col="adversary_elo",
            y_col="adversary_utility_mean",
            title=f"{game_label}: mean adversary payoff vs adversary Elo",
            x_label="Adversary Elo",
            y_label="Mean adversary final utility",
        )
        if path.exists():
            plot_paths.append(path)

        path = output_dir / f"{prefix}_{game_label}_hom_baseline_payoff_vs_adversary_elo_mean.png"
        line_summary_plot(
            game_df,
            path,
            x_col="adversary_elo",
            y_col="baseline_mean_utility_mean",
            title=f"{game_label}: mean baseline payoff vs adversary Elo",
            x_label="Adversary Elo",
            y_label="Mean GPT-5-nano baseline final utility",
        )
        if path.exists():
            plot_paths.append(path)

        for mode, suffix in [
            ("n_comp_grid", "faceted_n_by_competition"),
            ("by_n", "faceted_by_n"),
            ("by_competition", "faceted_by_competition"),
        ]:
            path = output_dir / f"{prefix}_{game_label}_hom_adversary_payoff_vs_elo_mean_{suffix}.png"
            faceted_line_plot(
                game_df,
                path,
                x_col="adversary_elo",
                y_col="adversary_utility_mean",
                title=f"{game_label}: mean adversary payoff vs adversary Elo",
                x_label="Adversary Elo",
                y_label="Mean adversary final utility",
                mode=mode,
            )
            if path.exists():
                plot_paths.append(path)

            path = output_dir / f"{prefix}_{game_label}_hom_baseline_payoff_vs_adversary_elo_mean_{suffix}.png"
            faceted_line_plot(
                game_df,
                path,
                x_col="adversary_elo",
                y_col="baseline_mean_utility_mean",
                title=f"{game_label}: mean baseline payoff vs adversary Elo",
                x_label="Adversary Elo",
                y_label="Mean GPT-5-nano baseline final utility",
                mode=mode,
            )
            if path.exists():
                plot_paths.append(path)

    hom_avg_over_comp = aggregate_homogeneous_adversary_custom(hom, ["n_agents"])
    hom_avg_over_n = aggregate_homogeneous_adversary_custom(
        hom,
        ["competition_key", "competition_value"],
    )
    hom_avg_over_n_comp = aggregate_homogeneous_adversary_custom(hom, [])
    for summary, suffix, group_col, title_suffix in [
        (hom_avg_over_comp, "avg_over_competition_by_n", "n_agents", "averaged over competition"),
        (hom_avg_over_n, "avg_over_n_by_competition", "competition_key", "averaged over N"),
        (hom_avg_over_n_comp, "avg_over_n_and_competition", None, "averaged over N and competition"),
    ]:
        if summary.empty:
            continue
        for game_label, game_df in iter_game_groups(summary):
            path = output_dir / f"{prefix}_{game_label}_hom_adversary_payoff_vs_elo_mean_{suffix}.png"
            grouped_elo_line_plot(
                game_df,
                path,
                x_col="adversary_elo",
                y_col="adversary_utility_mean",
                title=f"{game_label}: mean adversary payoff vs adversary Elo ({title_suffix})",
                y_label="Mean adversary final utility",
                group_col=group_col,
            )
            if path.exists():
                plot_paths.append(path)

            path = output_dir / f"{prefix}_{game_label}_hom_baseline_payoff_vs_adversary_elo_mean_{suffix}.png"
            grouped_elo_line_plot(
                game_df,
                path,
                x_col="adversary_elo",
                y_col="baseline_mean_utility_mean",
                title=f"{game_label}: mean baseline payoff vs adversary Elo ({title_suffix})",
                y_label="Mean GPT-5-nano baseline final utility",
                group_col=group_col,
            )
            if path.exists():
                plot_paths.append(path)

    hetero_agent_summary = aggregate_heterogeneous_agents(agents)
    for game_label, game_df in iter_game_groups(hetero_agent_summary):
        path = output_dir / f"{prefix}_{game_label}_heterogeneous_agent_payoff_vs_elo_mean.png"
        line_summary_plot(
            game_df,
            path,
            x_col="elo",
            y_col="final_utility_mean",
            title=f"{game_label}: heterogeneous mean agent payoff vs Elo",
            x_label="Agent Elo",
            y_label="Mean final utility",
        )
        if path.exists():
            plot_paths.append(path)

        for mode, suffix in [
            ("n_comp_grid", "faceted_n_by_competition"),
            ("by_n", "faceted_by_n"),
            ("by_competition", "faceted_by_competition"),
        ]:
            path = output_dir / f"{prefix}_{game_label}_heterogeneous_agent_payoff_vs_elo_mean_{suffix}.png"
            faceted_line_plot(
                game_df,
                path,
                x_col="elo",
                y_col="final_utility_mean",
                title=f"{game_label}: heterogeneous mean agent payoff vs Elo",
                x_label="Agent Elo",
                y_label="Mean final utility",
                mode=mode,
            )
            if path.exists():
                plot_paths.append(path)

    hetero_agent_avg_over_comp = aggregate_heterogeneous_agents_custom(agents, ["n_agents"])
    hetero_agent_avg_over_n = aggregate_heterogeneous_agents_custom(
        agents,
        ["competition_key", "competition_value"],
    )
    hetero_agent_avg_over_n_comp = aggregate_heterogeneous_agents_custom(agents, [])
    for summary, suffix, group_col, title_suffix in [
        (
            hetero_agent_avg_over_comp,
            "avg_over_competition_by_n",
            "n_agents",
            "averaged over competition",
        ),
        (
            hetero_agent_avg_over_n,
            "avg_over_n_by_competition",
            "competition_key",
            "averaged over N",
        ),
        (
            hetero_agent_avg_over_n_comp,
            "avg_over_n_and_competition",
            None,
            "averaged over N and competition",
        ),
    ]:
        if summary.empty:
            continue
        for game_label, game_df in iter_game_groups(summary):
            path = output_dir / f"{prefix}_{game_label}_heterogeneous_agent_payoff_vs_elo_mean_{suffix}.png"
            grouped_elo_line_plot(
                game_df,
                path,
                x_col="elo",
                y_col="final_utility_mean",
                title=f"{game_label}: heterogeneous mean agent payoff vs Elo ({title_suffix})",
                y_label="Mean final utility",
                group_col=group_col,
                x_label="Agent Elo",
            )
            if path.exists():
                plot_paths.append(path)

    return plot_paths


def markdown_table(frame: pd.DataFrame, max_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    return frame.head(max_rows).to_markdown(index=False)


def write_report(
    output_dir: Path,
    prefix: str,
    results_root: Path,
    all_runs: pd.DataFrame,
    runs: pd.DataFrame,
    agents: pd.DataFrame,
    plot_paths: Sequence[Path],
    args: argparse.Namespace,
) -> None:
    composition = group_count(
        runs,
        ["game_label", "experiment_family", "n_agents", "competition_key"],
    )
    models = (
        agents.groupby(["experiment_family", "model", "model_short", "elo"], dropna=False)
        .size()
        .reset_index(name="agent_appearances")
        .sort_values(["experiment_family", "elo", "model"])
        if not agents.empty
        else pd.DataFrame()
    )
    proposal_clean_count = int(runs["synthetic_proposal_marker_count"].fillna(0).eq(0).sum())
    synthetic_proposal_count = int(runs["synthetic_proposal_marker_count"].fillna(0).gt(0).sum())
    plot_rel = runs[runs["experiment_family"].isin(["homogeneous_adversary", "heterogeneous_random"])]

    lines = [
        "# Full Games 1-3 All-Success Preliminary Analysis",
        "",
        f"- Results root: `{results_root}`",
        f"- Exclude synthetic proposals: `{args.exclude_synthetic_proposals}`",
        f"- Exclude controls: `{args.exclude_controls}`",
        f"- Total successful result rows loaded: `{len(all_runs)}`",
        f"- Exported subset run rows: `{len(runs)}`",
        f"- Plot-relevant non-control rows: `{len(plot_rel)}`",
        f"- Rows with zero proposal fallback markers: `{proposal_clean_count}`",
        f"- Rows with proposal fallback markers: `{synthetic_proposal_count}`",
        f"- Gini small-N correction: `raw_gini * N/(N-1)` for complete runs with `N < {GINI_SMALL_N_CORRECTION_THRESHOLD}`; raw and correction factor columns are also exported.",
        "- Payoff-vs-Elo slope: per-run OLS fit of `final_utility ~ elo`; signed and absolute slopes are exported per 100 Elo points.",
        "",
        "## Composition",
        "",
        markdown_table(composition, max_rows=80),
        "",
        "## Model Appearances",
        "",
        markdown_table(models, max_rows=120),
        "",
        "## Plots",
        "",
    ]
    if plot_paths:
        lines.extend(f"- `{path}`" for path in plot_paths)
    else:
        lines.append("_No plots were generated for the selected subset._")
    lines.append("")
    (output_dir / f"{prefix}_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    results_root = args.results_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_runs, all_agents = build_tables(results_root)
    prefix = "all_success"

    runs, agents = filter_subset(
        all_runs,
        all_agents,
        exclude_synthetic_proposals=args.exclude_synthetic_proposals,
        exclude_controls=args.exclude_controls,
    )

    write_csv(output_dir / "all_success_runs.csv", all_runs)
    write_csv(output_dir / "all_success_agents.csv", all_agents)
    write_csv(output_dir / f"{prefix}_runs.csv", runs)
    write_csv(output_dir / f"{prefix}_agents.csv", agents)
    write_csv(
        output_dir / f"{prefix}_composition_by_game_family_setting.csv",
        group_count(runs, ["game_label", "experiment_family", "n_agents", "competition_key"]),
    )
    write_csv(
        output_dir / f"{prefix}_composition_by_model.csv",
        (
            agents.groupby(["experiment_family", "model", "model_short", "elo"], dropna=False)
            .size()
            .reset_index(name="agent_appearances")
            .sort_values(["experiment_family", "elo", "model"])
            if not agents.empty
            else pd.DataFrame()
        ),
    )
    write_csv(
        output_dir / f"{prefix}_homogeneous_adversary_summary_by_position.csv",
        aggregate_homogeneous_adversary(runs),
    )
    write_csv(
        output_dir / f"{prefix}_homogeneous_adversary_summary_pooled_positions.csv",
        aggregate_homogeneous_adversary_pooled_positions(runs),
    )
    write_csv(
        output_dir / f"{prefix}_heterogeneous_agent_payoff_by_model.csv",
        aggregate_heterogeneous_agents(agents),
    )
    write_csv(
        output_dir / f"{prefix}_heterogeneous_agent_payoff_by_model_averaged_over_competition_by_n.csv",
        aggregate_heterogeneous_agents_custom(agents, ["n_agents"]),
    )
    write_csv(
        output_dir / f"{prefix}_heterogeneous_agent_payoff_by_model_averaged_over_n_by_competition.csv",
        aggregate_heterogeneous_agents_custom(
            agents,
            ["competition_key", "competition_value"],
        ),
    )
    write_csv(
        output_dir / f"{prefix}_heterogeneous_agent_payoff_by_model_averaged_over_n_and_competition.csv",
        aggregate_heterogeneous_agents_custom(agents, []),
    )
    write_csv(
        output_dir / f"{prefix}_heterogeneous_gini_summary_by_n_competition.csv",
        aggregate_heterogeneous_gini(
            runs,
            ["game_label", "n_agents", "competition_key", "competition_value"],
        ),
    )
    write_csv(
        output_dir / f"{prefix}_heterogeneous_gini_summary_averaged_over_competition_by_n.csv",
        aggregate_heterogeneous_gini(runs, ["game_label", "n_agents"]),
    )
    write_csv(
        output_dir / f"{prefix}_heterogeneous_gini_summary_averaged_over_n_by_competition.csv",
        aggregate_heterogeneous_gini(
            runs,
            ["game_label", "competition_key", "competition_value"],
        ),
    )
    write_csv(
        output_dir / f"{prefix}_heterogeneous_gini_summary_averaged_over_n_and_competition.csv",
        aggregate_heterogeneous_gini(runs, ["game_label"]),
    )
    write_csv(
        output_dir / f"{prefix}_heterogeneous_utility_dispersion_summary_by_n_competition.csv",
        aggregate_heterogeneous_gini(
            runs,
            ["game_label", "n_agents", "competition_key", "competition_value"],
        ),
    )
    write_csv(
        output_dir
        / f"{prefix}_heterogeneous_utility_dispersion_summary_averaged_over_competition_by_n.csv",
        aggregate_heterogeneous_gini(runs, ["game_label", "n_agents"]),
    )
    write_csv(
        output_dir
        / f"{prefix}_heterogeneous_utility_dispersion_summary_averaged_over_n_by_competition.csv",
        aggregate_heterogeneous_gini(
            runs,
            ["game_label", "competition_key", "competition_value"],
        ),
    )
    write_csv(
        output_dir
        / f"{prefix}_heterogeneous_utility_dispersion_summary_averaged_over_n_and_competition.csv",
        aggregate_heterogeneous_gini(runs, ["game_label"]),
    )
    write_csv(
        output_dir / f"{prefix}_heterogeneous_payoff_elo_slope_summary_by_n_competition.csv",
        aggregate_heterogeneous_gini(
            runs,
            ["game_label", "n_agents", "competition_key", "competition_value"],
        ),
    )
    write_csv(
        output_dir
        / f"{prefix}_heterogeneous_payoff_elo_slope_summary_averaged_over_competition_by_n.csv",
        aggregate_heterogeneous_gini(runs, ["game_label", "n_agents"]),
    )
    write_csv(
        output_dir
        / f"{prefix}_heterogeneous_payoff_elo_slope_summary_averaged_over_n_by_competition.csv",
        aggregate_heterogeneous_gini(
            runs,
            ["game_label", "competition_key", "competition_value"],
        ),
    )
    write_csv(
        output_dir
        / f"{prefix}_heterogeneous_payoff_elo_slope_summary_averaged_over_n_and_competition.csv",
        aggregate_heterogeneous_gini(runs, ["game_label"]),
    )
    write_csv(
        output_dir / f"{prefix}_homogeneous_adversary_summary_averaged_over_competition_by_n.csv",
        aggregate_homogeneous_adversary_custom(runs, ["n_agents"]),
    )
    write_csv(
        output_dir / f"{prefix}_homogeneous_adversary_summary_averaged_over_n_by_competition.csv",
        aggregate_homogeneous_adversary_custom(
            runs,
            ["competition_key", "competition_value"],
        ),
    )
    write_csv(
        output_dir / f"{prefix}_homogeneous_adversary_summary_averaged_over_n_and_competition.csv",
        aggregate_homogeneous_adversary_custom(runs, []),
    )

    plot_paths = plot_outputs(runs, agents, output_dir, prefix)
    write_report(output_dir, prefix, results_root, all_runs, runs, agents, plot_paths, args)

    print(f"loaded_success_results={len(all_runs)}")
    print(f"selected_runs={len(runs)}")
    print(f"selected_agents={len(agents)}")
    print(f"output_dir={output_dir}")
    for path in plot_paths:
        print(f"wrote_plot={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

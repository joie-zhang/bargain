#!/usr/bin/env python3
"""Combined N=2 baseline comparison analysis.

This script reads the standard GPT-5-nano N=2 batches and the appendix
Llama-3.3 baseline batches, computes the requested payoff, competition,
order, consensus-round, optimality, and fairness metrics, and emits a compact
plot/report bundle.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from game_environments.cofunding_metrics import (  # noqa: E402
    lindahl_distance,
    lindahl_equilibrium,
    optimal_funded_set,
)
from game_environments.metrics import (  # noqa: E402
    compute_utility as game2_compute_utility,
    optimal_social_welfare as game2_optimal_social_welfare,
)
from strong_models_experiment.analysis.active_model_roster import (  # noqa: E402
    active_model_elo_map,
    canonical_model_name,
    short_model_name,
)


OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "n2_baseline_comparison_analysis_20260505"
FULL_ELO_MARKDOWN = PROJECT_ROOT / "docs" / "guides" / "chatbot_arena_elo_scores_2026_03_31.md"


@dataclass(frozen=True)
class BaselineSpec:
    key: str
    label: str
    baseline_model: str
    game_roots: dict[str, Path]


BASELINES: tuple[BaselineSpec, ...] = (
    BaselineSpec(
        key="gpt5_nano",
        label="GPT-5-nano baseline",
        baseline_model="gpt-5-nano",
        game_roots={
            "game1": PROJECT_ROOT / "experiments/results/scaling_experiment_20260404_064451",
            "game2": PROJECT_ROOT / "experiments/results/diplomacy_20260405_082215",
            "game3": PROJECT_ROOT / "experiments/results/cofunding_20260405_083548",
        },
    ),
    BaselineSpec(
        key="llama33",
        label="Llama-3.3-70B baseline",
        baseline_model="llama-3.3-70b-instruct",
        game_roots={
            "game1": PROJECT_ROOT / "experiments/results/appendix_llama33_baseline_game1_202605",
            "game2": PROJECT_ROOT / "experiments/results/appendix_llama33_baseline_game2_202605",
            "game3": PROJECT_ROOT / "experiments/results/appendix_llama33_baseline_game3_202605",
        },
    ),
)


GAME_LABELS = {
    "game1": "Game 1: item allocation",
    "game2": "Game 2: diplomacy",
    "game3": "Game 3: cofunding",
}


ORDER_LABELS = {
    "baseline_first": "Baseline first",
    "adversary_first": "Adversary first",
}


PALETTE = {
    "adversary": "#b45309",
    "baseline": "#2563eb",
    "fit": "#111827",
    "baseline_first": "#2563eb",
    "adversary_first": "#b45309",
}


_GAME2_NBS_CACHE: dict[tuple[Any, ...], dict[str, float]] = {}


def load_combined_elo_map() -> dict[str, int]:
    """Active roster Elo map, extended with legacy rows from the full Arena guide."""
    elo_map = dict(active_model_elo_map())
    if FULL_ELO_MARKDOWN.exists():
        pattern = re.compile(r"^\|\s*\d+\s*\|[^|]*\|\s*`?([^`|]+?)`?\s*\|\s*(\d+)\s*\|")
        for line in FULL_ELO_MARKDOWN.read_text(encoding="utf-8").splitlines():
            match = pattern.match(line)
            if not match:
                continue
            model = match.group(1).strip()
            elo = int(match.group(2))
            elo_map.setdefault(model, elo)
    return elo_map


def fmt_float(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if math.isnan(number):
        return "NA"
    text = f"{number:.{digits}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def fmt_signed(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if math.isnan(number):
        return "NA"
    return f"{number:+.{digits}f}"


def sem_series(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) <= 1:
        return 0.0
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


def finite_yerr(values: Any) -> np.ndarray | float:
    if isinstance(values, pd.Series):
        arr = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        return np.maximum(arr, 0.0)
    try:
        value = float(values)
    except (TypeError, ValueError):
        return 0.0
    return max(value, 0.0) if math.isfinite(value) else 0.0


def errorbar_points(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    x_col: str = "adversary_elo",
    y_col: str = "mean",
    yerr_col: str = "sem",
    color: Any,
    label: str | None = None,
    marker: str = "o",
    linestyle: str = "none",
    linewidth: float = 0.9,
    markersize: float = 4.0,
    alpha: float = 0.85,
) -> None:
    plot_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, y_col])
    if plot_df.empty:
        return
    ax.errorbar(
        plot_df[x_col],
        plot_df[y_col],
        yerr=finite_yerr(plot_df[yerr_col] if yerr_col in plot_df.columns else 0.0),
        fmt=marker,
        linestyle=linestyle,
        linewidth=linewidth,
        markersize=markersize,
        capsize=2.0,
        capthick=0.7,
        elinewidth=0.75,
        color=color,
        ecolor=color,
        alpha=alpha,
        label=label,
    )


def rel(path: Path | str) -> str:
    path_obj = Path(path)
    if path_obj.is_absolute():
        try:
            return str(path_obj.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path_obj)
    return str(path_obj)


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def result_file(output_dir: Path, run_number: int | None = None) -> Path | None:
    candidates: list[Path] = []
    if run_number is not None:
        candidates.append(output_dir / f"run_{run_number}_experiment_results.json")
    candidates.extend(
        [
            output_dir / "run_1_experiment_results.json",
            output_dir / "experiment_results.json",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    run_files = sorted(output_dir.glob("run_*_experiment_results.json"))
    return run_files[0] if run_files else None


def ordered_agent_ids(final_utilities: dict[str, Any]) -> list[str]:
    if {"Agent_1", "Agent_2"}.issubset(final_utilities):
        return ["Agent_1", "Agent_2"]
    if {"Agent_Alpha", "Agent_Beta"}.issubset(final_utilities):
        return ["Agent_Alpha", "Agent_Beta"]
    return sorted(final_utilities)


def infer_agents_for_empty_utilities(payload: dict[str, Any], config_payload: dict[str, Any]) -> list[str]:
    models = config_payload.get("models") or []
    if isinstance(models, list) and len(models) == 2:
        return ["Agent_1", "Agent_2"]

    prefs = payload.get("agent_preferences") or {}
    if isinstance(prefs, dict) and prefs:
        return sorted(str(agent) for agent in prefs)

    agents = config_payload.get("agents") or []
    if isinstance(agents, list) and agents:
        agent_ids: list[str] = []
        for idx, agent in enumerate(agents, start=1):
            if isinstance(agent, dict) and agent.get("id"):
                agent_ids.append(str(agent["id"]))
            elif isinstance(agent, str):
                agent_ids.append(agent)
            else:
                agent_ids.append(f"Agent_{idx}")
        return sorted(agent_ids)

    return ["Agent_1", "Agent_2"]


def final_utilities_with_zero_for_no_consensus(
    payload: dict[str, Any],
    config_payload: dict[str, Any],
) -> dict[str, Any] | None:
    final_utilities = payload.get("final_utilities")
    if not isinstance(final_utilities, dict):
        return None
    if payload.get("consensus_reached", False):
        return final_utilities
    agent_ids = infer_agents_for_empty_utilities(payload, config_payload)
    filled = {agent: 0.0 for agent in agent_ids}
    filled.update({str(agent): float(value) for agent, value in final_utilities.items()})
    return filled


def clean_model_for_identity(model_name: str) -> str:
    return str(model_name).strip()


def resolve_elo(model_name: str, elo_map: dict[str, int]) -> int | None:
    canonical = canonical_model_name(model_name)
    return elo_map.get(canonical) or elo_map.get(str(model_name).strip())


def display_model_name(model_name: str) -> str:
    short = short_model_name(model_name)
    if short != canonical_model_name(model_name):
        return short
    return {
        "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet",
        "phi-3-mini-128k-instruct": "Phi-3 Mini",
        "qwq-32b-preview": "QwQ Preview",
    }.get(str(model_name), short)


def conceptual_order(ordered_models: list[str], baseline_model: str, adversary_model: str) -> str:
    if ordered_models and clean_model_for_identity(ordered_models[0]) == clean_model_for_identity(baseline_model):
        return "baseline_first"
    if ordered_models and clean_model_for_identity(ordered_models[0]) == clean_model_for_identity(adversary_model):
        return "adversary_first"
    return "unknown"


def infer_role_agents(
    *,
    payload: dict[str, Any],
    config_payload: dict[str, Any],
    baseline_model: str,
    adversary_model: str,
    model_order: str,
) -> tuple[str | None, str | None, list[str]]:
    final_utilities = payload.get("final_utilities") or {}
    if not isinstance(final_utilities, dict):
        return None, None, []

    agent_ids = ordered_agent_ids(final_utilities)
    ordered_models = [clean_model_for_identity(model) for model in config_payload.get("models", [])]
    baseline_clean = clean_model_for_identity(baseline_model)
    adversary_clean = clean_model_for_identity(adversary_model)

    baseline_agent = None
    adversary_agent = None
    if len(agent_ids) == len(ordered_models):
        for agent_id, model_name in zip(agent_ids, ordered_models, strict=False):
            if model_name == baseline_clean:
                baseline_agent = agent_id
            if model_name == adversary_clean:
                adversary_agent = agent_id

    if baseline_agent and adversary_agent:
        return baseline_agent, adversary_agent, ordered_models

    if len(agent_ids) == 2:
        if model_order in {"weak_first", "baseline_first"}:
            return agent_ids[0], agent_ids[1], [baseline_clean, adversary_clean]
        if model_order in {"strong_first", "adversary_first"}:
            return agent_ids[1], agent_ids[0], [adversary_clean, baseline_clean]

    return baseline_agent, adversary_agent, ordered_models


def competition_fields(game_id: str, row: dict[str, str], config_payload: dict[str, Any]) -> dict[str, Any]:
    if game_id == "game1":
        value = float(row.get("competition_level") or config_payload.get("competition_level", 0.0))
        return {
            "competition_value": value,
            "competition_label": f"c={fmt_float(value)}",
            "competition_dimension": "competition_level",
            "competition_setting": f"competition_level={fmt_float(value)}",
            "competition_level": value,
        }
    if game_id == "game2":
        rho = float(row.get("rho") or config_payload.get("rho", 0.0))
        theta = float(row.get("theta") or config_payload.get("theta", 0.0))
        ci = theta * (1.0 - rho) / 2.0
        return {
            "competition_value": ci,
            "competition_label": f"CI2={fmt_float(ci)}",
            "competition_dimension": "competition_index",
            "competition_setting": f"rho={fmt_float(rho)};theta={fmt_float(theta)}",
            "rho": rho,
            "theta": theta,
        }
    if game_id == "game3":
        alpha = float(row.get("alpha") or config_payload.get("alpha", 0.0))
        sigma = float(row.get("sigma") or config_payload.get("sigma", 0.0))
        ci = (1.0 - alpha) * (1.0 - sigma)
        return {
            "competition_value": ci,
            "competition_label": f"CI3={fmt_float(ci)}",
            "competition_dimension": "competition_index",
            "competition_setting": f"alpha={fmt_float(alpha)};sigma={fmt_float(sigma)}",
            "alpha": alpha,
            "sigma": sigma,
        }
    raise ValueError(f"Unknown game id: {game_id}")


def game1_utility_for_allocation(
    prefs: dict[str, list[float]],
    allocation: dict[str, list[int]],
) -> dict[str, float]:
    result = {agent: 0.0 for agent in prefs}
    for agent, item_indices in allocation.items():
        if agent not in prefs:
            continue
        result[agent] = float(sum(float(prefs[agent][int(idx)]) for idx in item_indices))
    return result


def game1_optimal_welfare(prefs: dict[str, list[float]]) -> float:
    agents = sorted(prefs)
    if not agents:
        return 0.0
    n_items = len(prefs[agents[0]])
    return float(sum(max(float(prefs[agent][idx]) for agent in agents) for idx in range(n_items)))


def game1_nbs(prefs: dict[str, list[float]]) -> dict[str, float]:
    agents = sorted(prefs)
    if len(agents) != 2:
        return {agent: 0.0 for agent in agents}
    n_items = len(prefs[agents[0]])
    best_product = -1.0
    best_utils = {agent: 0.0 for agent in agents}
    for mask in range(1 << n_items):
        alloc = {
            agents[0]: [idx for idx in range(n_items) if (mask >> idx) & 1],
            agents[1]: [idx for idx in range(n_items) if not ((mask >> idx) & 1)],
        }
        utils = game1_utility_for_allocation(prefs, alloc)
        product = utils[agents[0]] * utils[agents[1]]
        if product > best_product:
            best_product = product
            best_utils = utils
    return best_utils


def game2_state(config_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "agent_positions": config_payload["agent_positions"],
        "agent_weights": config_payload["agent_weights"],
        "n_issues": int(config_payload.get("n_issues", 5)),
    }


def fast_game2_nbs(state: dict[str, Any]) -> dict[str, float]:
    """Fast cached NBS approximation for batch plotting.

    The full helper in game_environments.metrics is accurate but too slow for
    hundreds of repeated N=2 states. This mirrors the existing exploitation
    plotting script's 5-restart L-BFGS-B routine.
    """
    agents = sorted(state["agent_positions"])
    n_issues = int(state["n_issues"])
    positions = np.array([state["agent_positions"][agent] for agent in agents], dtype=float)
    weights = np.array([state["agent_weights"][agent] for agent in agents], dtype=float)

    def neg_log_product(agreement: np.ndarray) -> float:
        utilities = 100.0 * np.sum(weights * (1.0 - np.abs(positions - agreement)), axis=1)
        if np.any(utilities <= 1e-12):
            return 1e10
        return float(-np.sum(np.log(utilities)))

    bounds = [(0.0, 1.0)] * n_issues
    rng = np.random.RandomState(42)
    best_fun = float("inf")
    best_agreement: np.ndarray | None = None
    for _ in range(5):
        start = rng.uniform(0.0, 1.0, n_issues)
        result = minimize(
            neg_log_product,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-10},
        )
        if result.fun < best_fun:
            best_fun = float(result.fun)
            best_agreement = np.asarray(result.x, dtype=float)

    if best_agreement is None:
        return {agent: np.nan for agent in agents}
    return {
        agent: float(100.0 * np.sum(weights[idx] * (1.0 - np.abs(positions[idx] - best_agreement))))
        for idx, agent in enumerate(agents)
    }


def cached_game2_nbs(state: dict[str, Any]) -> dict[str, float]:
    agents = sorted(state["agent_positions"])
    key = tuple(
        (
            agent,
            tuple(float(x) for x in state["agent_positions"][agent]),
            tuple(float(x) for x in state["agent_weights"][agent]),
        )
        for agent in agents
    ) + (int(state["n_issues"]),)
    if key not in _GAME2_NBS_CACHE:
        _GAME2_NBS_CACHE[key] = fast_game2_nbs(state)
    return _GAME2_NBS_CACHE[key]


def game2_actual_utilities(payload: dict[str, Any], state: dict[str, Any]) -> dict[str, float]:
    agreement = payload.get("final_allocation")
    if not payload.get("consensus_reached") or not isinstance(agreement, list):
        return {agent: 0.0 for agent in state["agent_positions"]}
    return {
        agent: game2_compute_utility(agent, agreement, state)
        for agent in state["agent_positions"]
    }


def extract_cofunding_contributions(payload: dict[str, Any]) -> dict[str, list[float]] | None:
    final_round = int(payload.get("final_round") or 0)
    target_funded = sorted(int(x) for x in (payload.get("final_allocation") or []))
    logs = payload.get("conversation_logs") or []
    if not isinstance(logs, list):
        return None

    for log in logs:
        if log.get("phase") != "proposal_enumeration":
            continue
        if final_round and int(log.get("round") or 0) != final_round:
            continue
        for entry in log.get("enumerated_proposals") or []:
            if not isinstance(entry, dict):
                continue
            funded = sorted(int(x) for x in (entry.get("funded_projects") or []))
            if target_funded and funded != target_funded:
                continue
            contrib = entry.get("contributions_by_agent")
            if not contrib and isinstance(entry.get("original_proposal"), dict):
                contrib = entry["original_proposal"].get("contributions_by_agent")
            if isinstance(contrib, dict):
                return {
                    str(agent): [float(x) for x in values]
                    for agent, values in contrib.items()
                    if isinstance(values, list)
                }

    latest: dict[str, list[float]] = {}
    for log in logs:
        if log.get("phase") != "proposal":
            continue
        proposal = log.get("proposal") or {}
        if not isinstance(proposal, dict) or "contributions" not in proposal:
            continue
        latest[str(log.get("from"))] = [float(x) for x in proposal["contributions"]]
    return latest or None


def cofunding_actual_utilities(
    valuations: dict[str, list[float]],
    contributions: dict[str, list[float]],
    funded_set: list[int],
) -> dict[str, float]:
    return {
        agent: float(sum(float(valuations[agent][j]) - float(contributions.get(agent, [0.0] * len(valuations[agent]))[j]) for j in funded_set))
        for agent in valuations
    }


def cofunding_optimal_welfare(
    valuations: dict[str, list[float]],
    costs: list[float],
    total_budget: float,
) -> tuple[list[int], float]:
    opt_set = optimal_funded_set(valuations, costs, total_budget)
    opt_sw = float(
        sum(sum(float(valuations[agent][j]) for agent in valuations) - float(costs[j]) for j in opt_set)
    )
    return opt_set, max(opt_sw, 0.0)


def safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return np.nan
    return float(numerator / denominator)


def compute_solution_metrics(
    *,
    game_id: str,
    payload: dict[str, Any],
    config_payload: dict[str, Any],
    baseline_agent: str,
    adversary_agent: str,
) -> dict[str, float]:
    consensus = bool(payload.get("consensus_reached", False))
    final_round = float(payload.get("final_round")) if payload.get("final_round") is not None else np.nan
    metrics: dict[str, float] = {
        "rounds_to_consensus": final_round if consensus else np.nan,
        "actual_social_welfare_undiscounted": np.nan,
        "optimal_social_welfare": np.nan,
        "optimality_ratio": np.nan,
        "fairness_distance": np.nan,
        "adversary_fairness_excess": np.nan,
        "baseline_fairness_excess": np.nan,
        "adversary_fairness_excess_ratio": np.nan,
        "baseline_fairness_excess_ratio": np.nan,
        "nash_product_ratio": np.nan,
        "actual_funded_project_count": np.nan,
        "optimal_funded_project_count": np.nan,
        "optimal_project_recall": np.nan,
        "adversary_cost_shift_from_lindahl": np.nan,
        "baseline_cost_shift_from_lindahl": np.nan,
    }

    if game_id == "game1":
        raw_prefs = payload.get("agent_preferences") or {}
        allocation = payload.get("final_allocation") or {}
        if not isinstance(raw_prefs, dict):
            return metrics
        prefs = {
            str(agent): [float(x) for x in values]
            for agent, values in raw_prefs.items()
            if isinstance(values, list)
        }
        actual = game1_utility_for_allocation(prefs, allocation if consensus and isinstance(allocation, dict) else {})
        nbs = game1_nbs(prefs)
        sw_actual = float(sum(actual.values()))
        sw_opt = game1_optimal_welfare(prefs)
        nbs_product = float(np.prod([max(value, 0.0) for value in nbs.values()]))
        actual_product = float(np.prod([max(value, 0.0) for value in actual.values()]))
        metrics.update(
            {
                "actual_social_welfare_undiscounted": sw_actual,
                "optimal_social_welfare": sw_opt,
                "optimality_ratio": safe_ratio(sw_actual, sw_opt),
                "fairness_distance": float(
                    math.sqrt(sum((actual.get(agent, 0.0) - nbs.get(agent, 0.0)) ** 2 for agent in nbs))
                ),
                "adversary_fairness_excess": actual.get(adversary_agent, 0.0) - nbs.get(adversary_agent, 0.0),
                "baseline_fairness_excess": actual.get(baseline_agent, 0.0) - nbs.get(baseline_agent, 0.0),
                "adversary_fairness_excess_ratio": safe_ratio(
                    actual.get(adversary_agent, 0.0) - nbs.get(adversary_agent, 0.0),
                    nbs.get(adversary_agent, 0.0),
                ),
                "baseline_fairness_excess_ratio": safe_ratio(
                    actual.get(baseline_agent, 0.0) - nbs.get(baseline_agent, 0.0),
                    nbs.get(baseline_agent, 0.0),
                ),
                "nash_product_ratio": safe_ratio(actual_product, nbs_product),
            }
        )
        return metrics

    if game_id == "game2":
        if "agent_positions" not in config_payload or "agent_weights" not in config_payload:
            return metrics
        state = game2_state(config_payload)
        actual = game2_actual_utilities(payload, state)
        nbs = cached_game2_nbs(state)
        sw_actual = float(sum(actual.values()))
        sw_opt = float(game2_optimal_social_welfare(state))
        nbs_product = float(np.prod([max(value, 0.0) for value in nbs.values()]))
        actual_product = float(np.prod([max(value, 0.0) for value in actual.values()]))
        metrics.update(
            {
                "actual_social_welfare_undiscounted": sw_actual,
                "optimal_social_welfare": sw_opt,
                "optimality_ratio": safe_ratio(sw_actual, sw_opt),
                "fairness_distance": float(
                    math.sqrt(sum((actual.get(agent, 0.0) - nbs.get(agent, 0.0)) ** 2 for agent in nbs))
                ),
                "adversary_fairness_excess": actual.get(adversary_agent, 0.0) - nbs.get(adversary_agent, 0.0),
                "baseline_fairness_excess": actual.get(baseline_agent, 0.0) - nbs.get(baseline_agent, 0.0),
                "adversary_fairness_excess_ratio": safe_ratio(
                    actual.get(adversary_agent, 0.0) - nbs.get(adversary_agent, 0.0),
                    nbs.get(adversary_agent, 0.0),
                ),
                "baseline_fairness_excess_ratio": safe_ratio(
                    actual.get(baseline_agent, 0.0) - nbs.get(baseline_agent, 0.0),
                    nbs.get(baseline_agent, 0.0),
                ),
                "nash_product_ratio": safe_ratio(actual_product, nbs_product),
            }
        )
        return metrics

    if game_id == "game3":
        raw_prefs = payload.get("agent_preferences") or {}
        items = config_payload.get("items") or []
        if not isinstance(raw_prefs, dict) or not items:
            return metrics
        valuations = {
            str(agent): [float(x) for x in values]
            for agent, values in raw_prefs.items()
            if isinstance(values, list)
        }
        costs = [float(item["cost"]) for item in items]
        funded_set = [int(x) for x in (payload.get("final_allocation") or [])] if consensus else []
        contributions = extract_cofunding_contributions(payload) if consensus else None
        if contributions is None:
            contributions = {agent: [0.0] * len(costs) for agent in valuations}
        for agent in valuations:
            contributions.setdefault(agent, [0.0] * len(costs))
        actual = cofunding_actual_utilities(valuations, contributions, funded_set)
        sw_actual = float(sum(actual.values()))
        total_budget = float(config_payload.get("total_budget") or sum(config_payload.get("agent_budgets", {}).values()) or sum(costs))
        opt_set, sw_opt = cofunding_optimal_welfare(valuations, costs, total_budget)
        recall = (
            len(set(funded_set) & set(opt_set)) / len(opt_set)
            if opt_set
            else (1.0 if not funded_set else 0.0)
        )
        metrics.update(
            {
                "actual_social_welfare_undiscounted": sw_actual,
                "optimal_social_welfare": sw_opt,
                "optimality_ratio": safe_ratio(sw_actual, sw_opt),
                "actual_funded_project_count": float(len(funded_set)),
                "optimal_funded_project_count": float(len(opt_set)),
                "optimal_project_recall": float(recall),
            }
        )
        if funded_set:
            lindahl = lindahl_equilibrium(valuations, costs, funded_set)
            fair_actual = {
                agent: float(sum(float(valuations[agent][j]) - float(lindahl[agent][j]) for j in funded_set))
                for agent in valuations
            }
            total_funded_cost = sum(costs[j] for j in funded_set)
            actual_product = float(np.prod([max(value, 0.0) for value in actual.values()]))
            fair_product = float(np.prod([max(value, 0.0) for value in fair_actual.values()]))
            adv_actual_cost = sum(contributions.get(adversary_agent, [0.0] * len(costs))[j] for j in funded_set)
            base_actual_cost = sum(contributions.get(baseline_agent, [0.0] * len(costs))[j] for j in funded_set)
            adv_lindahl_cost = sum(lindahl.get(adversary_agent, [0.0] * len(costs))[j] for j in funded_set)
            base_lindahl_cost = sum(lindahl.get(baseline_agent, [0.0] * len(costs))[j] for j in funded_set)
            metrics.update(
                {
                    "fairness_distance": safe_ratio(lindahl_distance(contributions, lindahl), total_funded_cost),
                    "adversary_fairness_excess": actual.get(adversary_agent, 0.0) - fair_actual.get(adversary_agent, 0.0),
                    "baseline_fairness_excess": actual.get(baseline_agent, 0.0) - fair_actual.get(baseline_agent, 0.0),
                    "adversary_fairness_excess_ratio": safe_ratio(
                        actual.get(adversary_agent, 0.0) - fair_actual.get(adversary_agent, 0.0),
                        abs(fair_actual.get(adversary_agent, 0.0)),
                    ),
                    "baseline_fairness_excess_ratio": safe_ratio(
                        actual.get(baseline_agent, 0.0) - fair_actual.get(baseline_agent, 0.0),
                        abs(fair_actual.get(baseline_agent, 0.0)),
                    ),
                    "nash_product_ratio": safe_ratio(actual_product, fair_product),
                    "adversary_cost_shift_from_lindahl": adv_lindahl_cost - adv_actual_cost,
                    "baseline_cost_shift_from_lindahl": base_lindahl_cost - base_actual_cost,
                }
            )
        return metrics

    return metrics


def load_baseline_rows(spec: BaselineSpec, elo_map: dict[str, int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []

    for game_id, root in spec.game_roots.items():
        index_path = root / "configs" / "experiment_index.csv"
        with index_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if game_id == "game1":
                    baseline_model = row.get("baseline_model") or row.get("weak_model") or spec.baseline_model
                    adversary_model = row.get("adversary_model") or row.get("strong_model")
                else:
                    baseline_model = row.get("baseline_model") or row.get("model1") or spec.baseline_model
                    adversary_model = row.get("adversary_model") or row.get("model2")
                if not adversary_model:
                    skipped.append(f"{game_id}:{row.get('config_file')}:missing adversary")
                    continue

                adversary_canonical = canonical_model_name(adversary_model)
                adversary_elo = resolve_elo(adversary_model, elo_map)
                if adversary_elo is None:
                    skipped.append(f"{game_id}:{row.get('config_file')}:missing Elo {adversary_model}")
                    continue

                config_path = root / "configs" / row["config_file"]
                config_payload = json.loads(config_path.read_text(encoding="utf-8"))
                output_dir = resolve_path(config_payload["output_dir"])
                run_number = int(row["run_number"]) if row.get("run_number") else None
                path = result_file(output_dir, run_number=run_number)
                if path is None:
                    skipped.append(f"{game_id}:{row.get('config_file')}:missing result")
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                final_utilities = final_utilities_with_zero_for_no_consensus(payload, config_payload)
                if final_utilities is None:
                    skipped.append(f"{game_id}:{row.get('config_file')}:bad final utilities")
                    continue
                payload["final_utilities"] = final_utilities

                model_order = str(row.get("model_order") or config_payload.get("model_order") or "")
                baseline_agent, adversary_agent, ordered_models = infer_role_agents(
                    payload=payload,
                    config_payload=config_payload,
                    baseline_model=baseline_model,
                    adversary_model=adversary_model,
                    model_order=model_order,
                )
                if baseline_agent not in final_utilities or adversary_agent not in final_utilities:
                    skipped.append(f"{game_id}:{row.get('config_file')}:role mapping failed")
                    continue

                comp = competition_fields(game_id, row, config_payload)
                order = row.get("conceptual_order") or conceptual_order(
                    ordered_models,
                    baseline_model,
                    adversary_model,
                )
                metrics = compute_solution_metrics(
                    game_id=game_id,
                    payload=payload,
                    config_payload=payload.get("config") or config_payload,
                    baseline_agent=str(baseline_agent),
                    adversary_agent=str(adversary_agent),
                )

                baseline_utility = float(final_utilities[str(baseline_agent)])
                adversary_utility = float(final_utilities[str(adversary_agent)])
                rows.append(
                    {
                        "baseline_key": spec.key,
                        "baseline_label": spec.label,
                        "baseline_model": baseline_model,
                        "game_id": game_id,
                        "game_label": GAME_LABELS[game_id],
                        "config_file": row["config_file"],
                        "experiment_id": row.get("experiment_id"),
                        "result_path": rel(path),
                        "output_dir": rel(output_dir),
                        "baseline_agent": baseline_agent,
                        "adversary_agent": adversary_agent,
                        "adversary_model": adversary_canonical,
                        "adversary_raw_model": adversary_model,
                        "adversary_short": display_model_name(adversary_model),
                        "adversary_elo": int(adversary_elo),
                        "model_order": model_order,
                        "conceptual_order": order,
                        "order_label": ORDER_LABELS.get(str(order), str(order)),
                        "run_number": run_number,
                        "seed": row.get("seed") or row.get("random_seed") or config_payload.get("random_seed"),
                        "discussion_turns": row.get("discussion_turns") or config_payload.get("discussion_turns"),
                        "consensus_reached": bool(payload.get("consensus_reached", False)),
                        "final_round": float(payload.get("final_round")) if payload.get("final_round") is not None else np.nan,
                        "adversary_utility": adversary_utility,
                        "baseline_utility": baseline_utility,
                        "utility_gap_adv_minus_base": adversary_utility - baseline_utility,
                        "payoff_social_welfare": adversary_utility + baseline_utility,
                        **comp,
                        **metrics,
                    }
                )

    frame = pd.DataFrame(rows)
    if skipped:
        skipped_path = OUT_DIR / f"{spec.key}_skipped_rows.txt"
        skipped_path.parent.mkdir(parents=True, exist_ok=True)
        skipped_path.write_text("\n".join(skipped) + "\n", encoding="utf-8")
    return frame


def regression_metrics(x: pd.Series, y: pd.Series) -> dict[str, float]:
    clean = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2 or clean["x"].nunique() < 2:
        return {"slope_per_100_elo": np.nan, "pearson_r": np.nan, "spearman_r": np.nan}
    slope, _intercept = np.polyfit(clean["x"], clean["y"], deg=1)
    return {
        "slope_per_100_elo": float(slope * 100.0),
        "pearson_r": float(clean["x"].corr(clean["y"], method="pearson")),
        "spearman_r": float(clean["x"].corr(clean["y"], method="spearman")),
    }


def aggregate_metric(df: pd.DataFrame, group_cols: list[str], metric: str) -> pd.DataFrame:
    agg = (
        df.groupby(group_cols, as_index=False)
        .agg(
            n=(metric, "size"),
            mean=(metric, "mean"),
            std=(metric, "std"),
            sem=(metric, sem_series),
            consensus_rate=("consensus_reached", "mean"),
        )
        .replace([np.inf, -np.inf], np.nan)
    )
    return agg


def fit_line(ax: plt.Axes, df: pd.DataFrame, x_col: str, y_col: str, color: str, linestyle: str = "--") -> float:
    clean = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2 or clean[x_col].nunique() < 2:
        return np.nan
    slope, intercept = np.polyfit(clean[x_col], clean[y_col], deg=1)
    xs = np.linspace(float(clean[x_col].min()), float(clean[x_col].max()), 100)
    ax.plot(xs, slope * xs + intercept, color=color, linestyle=linestyle, linewidth=1.3, alpha=0.9)
    return float(slope * 100.0)


def annotate_models(ax: plt.Axes, df: pd.DataFrame, y_col: str, fontsize: float = 5.5, alpha: float = 0.7) -> None:
    for _, row in df.iterrows():
        if pd.isna(row[y_col]):
            continue
        ax.annotate(
            str(row["adversary_short"]),
            (row["adversary_elo"], row[y_col]),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            alpha=alpha,
        )


def defined_metric_rows(df: pd.DataFrame, x_col: str = "adversary_elo", y_col: str = "mean") -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, y_col])


def undefined_metric_rows(df: pd.DataFrame, x_col: str = "adversary_elo", y_col: str = "mean") -> pd.DataFrame:
    clean = df.replace([np.inf, -np.inf], np.nan)
    return clean[clean[x_col].notna() & clean[y_col].isna()].copy()


def mark_undefined_metric_rows(
    ax: plt.Axes,
    batches: list[tuple[pd.Series, Any]],
    *,
    label: str = "x = undefined metric",
) -> None:
    if not batches:
        return
    ymin, ymax = ax.get_ylim()
    if not math.isfinite(ymin) or not math.isfinite(ymax) or abs(ymax - ymin) < 1e-12:
        ymin, ymax = 0.0, 1.0
    marker_y = ymin + 0.035 * (ymax - ymin)
    for xs, color in batches:
        clean_xs = pd.to_numeric(xs, errors="coerce").dropna()
        if clean_xs.empty:
            continue
        ax.scatter(
            clean_xs,
            [marker_y] * len(clean_xs),
            marker="x",
            s=24,
            linewidths=0.9,
            color=color,
            alpha=0.9,
            zorder=5,
        )
    ax.scatter([], [], marker="x", s=24, linewidths=0.9, color="#111827", label=label)
    ax.set_ylim(ymin, ymax)


def style_axis(ax: plt.Axes, ylabel: str) -> None:
    ax.set_xlabel("Adversary Chatbot Arena Elo", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.22)
    ax.tick_params(labelsize=8)


def plot_overall_metric(
    df: pd.DataFrame,
    spec: BaselineSpec,
    metric: str,
    ylabel: str,
    filename: str,
    *,
    point_color: str,
    label_points: bool = True,
) -> pd.DataFrame:
    out_path = OUT_DIR / spec.key / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharey=False)
    slope_rows: list[dict[str, Any]] = []

    for ax, game_id in zip(axes, ["game1", "game2", "game3"], strict=True):
        subset = df[(df["baseline_key"] == spec.key) & (df["game_id"] == game_id)].copy()
        agg = aggregate_metric(
            subset,
            ["adversary_model", "adversary_short", "adversary_elo"],
            metric,
        ).sort_values("adversary_elo")
        missing = undefined_metric_rows(agg)
        errorbar_points(ax, agg, color=point_color, markersize=4.4, alpha=0.9)
        slope = fit_line(ax, agg, "adversary_elo", "mean", PALETTE["fit"])
        if label_points:
            annotate_models(ax, agg, "mean", fontsize=5.4, alpha=0.8)
        ax.set_title(f"{GAME_LABELS[game_id]}\nslope={fmt_signed(slope)} / 100 Elo", fontsize=10)
        style_axis(ax, ylabel)
        if not missing.empty:
            mark_undefined_metric_rows(ax, [(missing["adversary_elo"], "#6b7280")])
            ax.legend(fontsize=6.5, frameon=True)
        trend = regression_metrics(agg["adversary_elo"], agg["mean"])
        slope_rows.append(
            {
                "baseline_key": spec.key,
                "plot": filename,
                "metric": metric,
                "game_id": game_id,
                "group": "overall",
                "n_points": len(agg),
                **trend,
            }
        )

    fig.suptitle(f"{spec.label}: {ylabel} vs adversary Elo", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(slope_rows)


def plot_by_competition_metric(
    df: pd.DataFrame,
    spec: BaselineSpec,
    metric: str,
    ylabel: str,
    filename: str,
    *,
    label_points: bool = False,
) -> pd.DataFrame:
    out_path = OUT_DIR / spec.key / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4), sharey=False)
    slope_rows: list[dict[str, Any]] = []

    for ax, game_id in zip(axes, ["game1", "game2", "game3"], strict=True):
        subset = df[(df["baseline_key"] == spec.key) & (df["game_id"] == game_id)].copy()
        agg = aggregate_metric(
            subset,
            ["competition_value", "competition_label", "adversary_model", "adversary_short", "adversary_elo"],
            metric,
        ).sort_values(["competition_value", "adversary_elo"])
        comp_values = sorted(agg["competition_value"].dropna().unique().tolist())
        colors = plt.cm.viridis(np.linspace(0.05, 0.95, max(len(comp_values), 1)))
        missing_batches: list[tuple[pd.Series, Any]] = []
        for color, comp_value in zip(colors, comp_values, strict=False):
            comp_df = agg[agg["competition_value"].eq(comp_value)].sort_values("adversary_elo")
            defined_df = defined_metric_rows(comp_df)
            missing_df = undefined_metric_rows(comp_df)
            if not missing_df.empty:
                missing_batches.append((missing_df["adversary_elo"], color))
            slope = fit_line(ax, comp_df, "adversary_elo", "mean", color)
            label = str(comp_df["competition_label"].iloc[0]) if not comp_df.empty else str(comp_value)
            errorbar_points(ax, defined_df, color=color, markersize=3.6, alpha=0.75)
            ax.plot(defined_df["adversary_elo"], defined_df["mean"], color=color, linewidth=0.7, alpha=0.35)
            if label_points:
                annotate_models(ax, defined_df, "mean", fontsize=3.2, alpha=0.35)
            ax.plot([], [], color=color, linestyle="--", label=f"{label} ({fmt_signed(slope)})")
            slope_rows.append(
                {
                    "baseline_key": spec.key,
                    "plot": filename,
                    "metric": metric,
                    "game_id": game_id,
                    "group": label,
                    "n_points": len(comp_df),
                    **regression_metrics(comp_df["adversary_elo"], comp_df["mean"]),
                }
            )
        ax.set_title(GAME_LABELS[game_id], fontsize=10)
        style_axis(ax, ylabel)
        mark_undefined_metric_rows(ax, missing_batches)
        ax.legend(title="Competition (slope)", fontsize=6.4, title_fontsize=7, frameon=True, ncol=1)

    fig.suptitle(f"{spec.label}: {ylabel} vs Elo by competition level/index", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(slope_rows)


def plot_by_order_metric(
    df: pd.DataFrame,
    spec: BaselineSpec,
    metric: str,
    ylabel: str,
    filename: str,
) -> pd.DataFrame:
    out_path = OUT_DIR / spec.key / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.2), sharey=False)
    slope_rows: list[dict[str, Any]] = []

    for ax, game_id in zip(axes, ["game1", "game2", "game3"], strict=True):
        subset = df[(df["baseline_key"] == spec.key) & (df["game_id"] == game_id)].copy()
        agg = aggregate_metric(
            subset,
            ["conceptual_order", "order_label", "adversary_model", "adversary_short", "adversary_elo"],
            metric,
        ).sort_values(["conceptual_order", "adversary_elo"])
        missing_batches: list[tuple[pd.Series, Any]] = []
        for order in ["adversary_first", "baseline_first"]:
            order_df = agg[agg["conceptual_order"].eq(order)].sort_values("adversary_elo")
            if order_df.empty:
                continue
            color = PALETTE.get(order, "#475569")
            defined_df = defined_metric_rows(order_df)
            missing_df = undefined_metric_rows(order_df)
            if not missing_df.empty:
                missing_batches.append((missing_df["adversary_elo"], color))
            slope = fit_line(ax, order_df, "adversary_elo", "mean", color)
            errorbar_points(ax, defined_df, color=color, markersize=4.2, alpha=0.85)
            ax.plot(defined_df["adversary_elo"], defined_df["mean"], color=color, linewidth=0.8, alpha=0.4)
            label = str(order_df["order_label"].iloc[0])
            ax.plot([], [], color=color, linestyle="--", label=f"{label} ({fmt_signed(slope)})")
            slope_rows.append(
                {
                    "baseline_key": spec.key,
                    "plot": filename,
                    "metric": metric,
                    "game_id": game_id,
                    "group": label,
                    "n_points": len(order_df),
                    **regression_metrics(order_df["adversary_elo"], order_df["mean"]),
                }
            )
        ax.set_title(GAME_LABELS[game_id], fontsize=10)
        style_axis(ax, ylabel)
        mark_undefined_metric_rows(ax, missing_batches)
        ax.legend(title="Order (slope)", fontsize=7, title_fontsize=8)

    fig.suptitle(f"{spec.label}: {ylabel} vs Elo by model order", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(slope_rows)


def plot_by_order_and_competition_metric(
    df: pd.DataFrame,
    spec: BaselineSpec,
    metric: str,
    ylabel: str,
    filename: str,
) -> pd.DataFrame:
    out_path = OUT_DIR / spec.key / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5), sharey=False)
    slope_rows: list[dict[str, Any]] = []
    linestyles = {"adversary_first": "-", "baseline_first": ":"}

    for ax, game_id in zip(axes, ["game1", "game2", "game3"], strict=True):
        subset = df[(df["baseline_key"] == spec.key) & (df["game_id"] == game_id)].copy()
        agg = aggregate_metric(
            subset,
            [
                "competition_value",
                "competition_label",
                "conceptual_order",
                "order_label",
                "adversary_model",
                "adversary_short",
                "adversary_elo",
            ],
            metric,
        ).sort_values(["competition_value", "conceptual_order", "adversary_elo"])
        comp_values = sorted(agg["competition_value"].dropna().unique().tolist())
        colors = plt.cm.viridis(np.linspace(0.05, 0.95, max(len(comp_values), 1)))
        missing_batches: list[tuple[pd.Series, Any]] = []
        for color, comp_value in zip(colors, comp_values, strict=False):
            for order in ["adversary_first", "baseline_first"]:
                sub = agg[agg["competition_value"].eq(comp_value) & agg["conceptual_order"].eq(order)]
                if sub.empty:
                    continue
                sub = sub.sort_values("adversary_elo")
                defined_sub = defined_metric_rows(sub)
                missing_sub = undefined_metric_rows(sub)
                if not missing_sub.empty:
                    missing_batches.append((missing_sub["adversary_elo"], color))
                slope = fit_line(ax, sub, "adversary_elo", "mean", color, linestyle=linestyles[order])
                errorbar_points(ax, defined_sub, color=color, markersize=3.0, alpha=0.62)
                ax.plot(
                    defined_sub["adversary_elo"],
                    defined_sub["mean"],
                    color=color,
                    linestyle=linestyles[order],
                    linewidth=0.8,
                    alpha=0.42,
                )
                label = f"{sub['competition_label'].iloc[0]}, {sub['order_label'].iloc[0]} ({fmt_signed(slope)})"
                ax.plot([], [], color=color, linestyle=linestyles[order], label=label)
                slope_rows.append(
                    {
                        "baseline_key": spec.key,
                        "plot": filename,
                        "metric": metric,
                        "game_id": game_id,
                        "group": label,
                        "n_points": len(sub),
                        **regression_metrics(sub["adversary_elo"], sub["mean"]),
                    }
            )
        ax.set_title(GAME_LABELS[game_id], fontsize=10)
        style_axis(ax, ylabel)
        mark_undefined_metric_rows(ax, missing_batches)
        ax.legend(fontsize=4.9, title_fontsize=6, frameon=True, ncol=1)

    fig.suptitle(f"{spec.label}: {ylabel} vs Elo by order and competition", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(slope_rows)


def plot_fairness_excess_by_role(df: pd.DataFrame, spec: BaselineSpec, filename: str) -> pd.DataFrame:
    out_path = OUT_DIR / spec.key / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.2), sharey=False)
    slope_rows: list[dict[str, Any]] = []
    metrics = [
        ("adversary_fairness_excess", "Adversary", PALETTE["adversary"]),
        ("baseline_fairness_excess", "Baseline", PALETTE["baseline"]),
    ]

    for ax, game_id in zip(axes, ["game1", "game2", "game3"], strict=True):
        subset = df[(df["baseline_key"] == spec.key) & (df["game_id"] == game_id)].copy()
        missing_batches: list[tuple[pd.Series, Any]] = []
        for metric, label, color in metrics:
            agg = aggregate_metric(
                subset,
                ["adversary_model", "adversary_short", "adversary_elo"],
                metric,
            ).sort_values("adversary_elo")
            defined_df = defined_metric_rows(agg)
            missing_df = undefined_metric_rows(agg)
            if not missing_df.empty:
                missing_batches.append((missing_df["adversary_elo"], color))
            slope = fit_line(ax, agg, "adversary_elo", "mean", color)
            errorbar_points(ax, defined_df, color=color, markersize=4.2, alpha=0.82)
            ax.plot(defined_df["adversary_elo"], defined_df["mean"], color=color, linewidth=0.8, alpha=0.35)
            ax.plot([], [], color=color, linestyle="--", label=f"{label} ({fmt_signed(slope)})")
            slope_rows.append(
                {
                    "baseline_key": spec.key,
                    "plot": filename,
                    "metric": metric,
                    "game_id": game_id,
                    "group": label,
                    "n_points": len(agg),
                    **regression_metrics(agg["adversary_elo"], agg["mean"]),
                }
            )
        ax.axhline(0, color="#111827", linewidth=0.7, alpha=0.6)
        ax.set_title(GAME_LABELS[game_id], fontsize=10)
        style_axis(ax, "Utility points above fair benchmark")
        mark_undefined_metric_rows(ax, missing_batches)
        ax.legend(title="Role (slope)", fontsize=7, title_fontsize=8)

    fig.suptitle(
        f"{spec.label}: who is above/below NBS or Lindahl benchmark?",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(slope_rows)


def make_all_plots(df: pd.DataFrame) -> pd.DataFrame:
    slope_frames: list[pd.DataFrame] = []
    for spec in BASELINES:
        plot_specs = [
            ("adversary_utility", "Mean adversary payoff", "01_adversary_payoff_overall.png", PALETTE["adversary"], True),
            ("baseline_utility", "Mean baseline payoff", "03_baseline_payoff_overall.png", PALETTE["baseline"], True),
            ("rounds_to_consensus", "Mean rounds to consensus", "07_rounds_to_consensus_overall.png", "#0f766e", True),
            ("optimality_ratio", "Mean SW / optimal SW", "09_optimality_ratio_overall.png", "#7c3aed", True),
            ("fairness_distance", "Mean fairness distance", "11_fairness_distance_overall.png", "#be123c", True),
        ]
        for metric, ylabel, filename, color, label_points in plot_specs:
            slope_frames.append(
                plot_overall_metric(df, spec, metric, ylabel, filename, point_color=color, label_points=label_points)
            )
        comp_specs = [
            ("adversary_utility", "Mean adversary payoff", "02_adversary_payoff_by_competition.png", True),
            ("baseline_utility", "Mean baseline payoff", "04_baseline_payoff_by_competition.png", True),
            ("rounds_to_consensus", "Mean rounds to consensus", "08_rounds_to_consensus_by_competition.png", False),
            ("optimality_ratio", "Mean SW / optimal SW", "10_optimality_ratio_by_competition.png", False),
            ("fairness_distance", "Mean fairness distance", "12_fairness_distance_by_competition.png", False),
        ]
        for metric, ylabel, filename, label_points in comp_specs:
            slope_frames.append(
                plot_by_competition_metric(df, spec, metric, ylabel, filename, label_points=label_points)
            )
        slope_frames.append(
            plot_by_order_metric(
                df,
                spec,
                "adversary_utility",
                "Mean adversary payoff",
                "05_adversary_payoff_by_order.png",
            )
        )
        slope_frames.append(
            plot_by_order_and_competition_metric(
                df,
                spec,
                "adversary_utility",
                "Mean adversary payoff",
                "06_adversary_payoff_by_order_and_competition.png",
            )
        )
        slope_frames.append(plot_fairness_excess_by_role(df, spec, "13_fairness_excess_by_role_overall.png"))
    return pd.concat(slope_frames, ignore_index=True)


def write_summary_tables(df: pd.DataFrame, slopes: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "all_runs_with_metrics.csv", index=False)
    slopes.to_csv(OUT_DIR / "slope_summary.csv", index=False)

    overall = (
        df.groupby(
            [
                "baseline_key",
                "baseline_label",
                "game_id",
                "game_label",
                "adversary_model",
                "adversary_short",
                "adversary_elo",
            ],
            as_index=False,
        )
        .agg(
            n=("adversary_utility", "size"),
            adversary_utility=("adversary_utility", "mean"),
            adversary_utility_sem=("adversary_utility", sem_series),
            baseline_utility=("baseline_utility", "mean"),
            baseline_utility_sem=("baseline_utility", sem_series),
            utility_gap_adv_minus_base=("utility_gap_adv_minus_base", "mean"),
            utility_gap_adv_minus_base_sem=("utility_gap_adv_minus_base", sem_series),
            rounds_to_consensus=("rounds_to_consensus", "mean"),
            rounds_to_consensus_sem=("rounds_to_consensus", sem_series),
            consensus_rate=("consensus_reached", "mean"),
            optimality_ratio=("optimality_ratio", "mean"),
            optimality_ratio_sem=("optimality_ratio", sem_series),
            fairness_distance=("fairness_distance", "mean"),
            fairness_distance_sem=("fairness_distance", sem_series),
            adversary_fairness_excess=("adversary_fairness_excess", "mean"),
            adversary_fairness_excess_sem=("adversary_fairness_excess", sem_series),
            baseline_fairness_excess=("baseline_fairness_excess", "mean"),
            baseline_fairness_excess_sem=("baseline_fairness_excess", sem_series),
        )
        .sort_values(["baseline_key", "game_id", "adversary_elo"])
    )
    overall.to_csv(OUT_DIR / "overall_by_model_game.csv", index=False)

    by_comp = (
        df.groupby(
            [
                "baseline_key",
                "baseline_label",
                "game_id",
                "game_label",
                "competition_value",
                "competition_label",
                "adversary_model",
                "adversary_short",
                "adversary_elo",
            ],
            as_index=False,
        )
        .agg(
            n=("adversary_utility", "size"),
            adversary_utility=("adversary_utility", "mean"),
            adversary_utility_sem=("adversary_utility", sem_series),
            baseline_utility=("baseline_utility", "mean"),
            baseline_utility_sem=("baseline_utility", sem_series),
            rounds_to_consensus=("rounds_to_consensus", "mean"),
            rounds_to_consensus_sem=("rounds_to_consensus", sem_series),
            consensus_rate=("consensus_reached", "mean"),
            optimality_ratio=("optimality_ratio", "mean"),
            optimality_ratio_sem=("optimality_ratio", sem_series),
            fairness_distance=("fairness_distance", "mean"),
            fairness_distance_sem=("fairness_distance", sem_series),
            adversary_fairness_excess=("adversary_fairness_excess", "mean"),
            adversary_fairness_excess_sem=("adversary_fairness_excess", sem_series),
            baseline_fairness_excess=("baseline_fairness_excess", "mean"),
            baseline_fairness_excess_sem=("baseline_fairness_excess", sem_series),
        )
        .sort_values(["baseline_key", "game_id", "competition_value", "adversary_elo"])
    )
    by_comp.to_csv(OUT_DIR / "by_competition_model_game.csv", index=False)

    roster = (
        df.groupby(
            [
                "baseline_key",
                "baseline_label",
                "adversary_model",
                "adversary_short",
                "adversary_elo",
                "game_id",
            ]
        )
        .size()
        .unstack("game_id", fill_value=0)
        .reset_index()
    )
    for game_id in ["game1", "game2", "game3"]:
        if game_id not in roster.columns:
            roster[game_id] = 0
    game_cols = ["game1", "game2", "game3"]
    roster["games_present"] = roster[game_cols].gt(0).sum(axis=1)
    roster["present_in_games"] = roster.apply(
        lambda row: ",".join(game_id for game_id in game_cols if int(row[game_id]) > 0),
        axis=1,
    )
    roster = roster.sort_values(["baseline_key", "adversary_elo", "adversary_model"])
    roster.to_csv(OUT_DIR / "model_roster_by_game.csv", index=False)
    roster[roster["games_present"].lt(3)].to_csv(
        OUT_DIR / "model_roster_presence_inconsistencies.csv",
        index=False,
    )

    raw_rows: list[dict[str, Any]] = []
    for spec in BASELINES:
        for game_id, root in spec.game_roots.items():
            index_path = root / "configs" / "experiment_index.csv"
            with index_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    adversary_model = (
                        row.get("adversary_model")
                        or (row.get("strong_model") if game_id == "game1" else row.get("model2"))
                    )
                    if not adversary_model:
                        continue
                    raw_rows.append(
                        {
                            "baseline_key": spec.key,
                            "baseline_label": spec.label,
                            "adversary_model": canonical_model_name(adversary_model),
                            "adversary_short": display_model_name(adversary_model),
                            "game_id": game_id,
                        }
                    )
    raw_roster = (
        pd.DataFrame(raw_rows)
        .groupby(["baseline_key", "baseline_label", "adversary_model", "adversary_short", "game_id"])
        .size()
        .unstack("game_id", fill_value=0)
        .reset_index()
    )
    for game_id in ["game1", "game2", "game3"]:
        if game_id not in raw_roster.columns:
            raw_roster[game_id] = 0
    raw_roster["games_present"] = raw_roster[game_cols].gt(0).sum(axis=1)
    raw_roster["present_in_games"] = raw_roster.apply(
        lambda row: ",".join(game_id for game_id in game_cols if int(row[game_id]) > 0),
        axis=1,
    )
    raw_roster = raw_roster.sort_values(["baseline_key", "adversary_model"])
    raw_roster.to_csv(OUT_DIR / "raw_config_model_roster_by_game.csv", index=False)
    raw_roster[raw_roster["games_present"].lt(3)].to_csv(
        OUT_DIR / "raw_config_model_presence_inconsistencies.csv",
        index=False,
    )


def report_plot_lines(spec: BaselineSpec) -> list[str]:
    base_dir = Path(spec.key)
    plots = [
        ("Does capability scale?", "01_adversary_payoff_overall.png"),
        ("Capability scaling by competition", "02_adversary_payoff_by_competition.png"),
        ("Baseline payoff against stronger adversaries", "03_baseline_payoff_overall.png"),
        ("Baseline payoff by competition", "04_baseline_payoff_by_competition.png"),
        ("Model order", "05_adversary_payoff_by_order.png"),
        ("Model order by competition", "06_adversary_payoff_by_order_and_competition.png"),
        ("Rounds to consensus", "07_rounds_to_consensus_overall.png"),
        ("Rounds by competition", "08_rounds_to_consensus_by_competition.png"),
        ("Social-welfare optimality", "09_optimality_ratio_overall.png"),
        ("Optimality by competition", "10_optimality_ratio_by_competition.png"),
        ("NBS/Lindahl fairness distance", "11_fairness_distance_overall.png"),
        ("Fairness distance by competition", "12_fairness_distance_by_competition.png"),
        ("Which role is above/below fair benchmark", "13_fairness_excess_by_role_overall.png"),
    ]
    lines = []
    for title, filename in plots:
        lines.extend([f"### {title}", "", f"![{title}]({base_dir / filename})", ""])
    return lines


def qualitative_report_lines() -> list[str]:
    return [
        "## Qualitative QA: Why Higher-Elo Models Do Better",
        "",
        "Across both baselines, the basic payoff slopes are positive for the adversary in every game. The mechanism is not simply that stronger models are more aggressive. The samples point to three repeated advantages: they identify complementary trades faster, keep proposals machine-parseable and vote-aligned, and use competitive settings to anchor the negotiation around their high-value issues.",
        "",
        "The baseline-payoff plots are therefore not purely zero-sum. Against GPT-5-nano, baseline payoff rises overall in Games 1-3, but the competition-stratified plots show losses in the hardest cells: Game 1 at `c=1`, Game 2 at `CI2=1`, and Game 3 at `CI3=0.8`. Against Llama-3.3, baseline payoff is flat or negative in Games 1-2 and positive in Game 3 overall, with sharper losses in high-competition Game 3. This matches the rollouts: cooperative structure can let both sides improve, while high-conflict structure turns model capability into surplus extraction.",
        "",
        "### Game 1: Item Allocation",
        "",
        "The core mechanism is package recognition. In a complementary-preference run, DeepSeek R1 immediately recognized that the other side wanted the zero-value-to-it items and accepted the clean split: [DeepSeek R1 cooperative split](../scaling_experiment_20260404_064451/gpt-5-nano_vs_deepseek-r1/weak_first/comp_0.0/turns_1/run_1/experiment_results.json). The key line is: \"Your Option C looks promising\" because each agent gets exactly its high-value bundle. This explains why optimality rises with Elo: higher-Elo models more often find the welfare-maximizing allocation rather than wasting items on the wrong side.",
        "",
        "In more competitive cells, higher-Elo models often win by making the contested item the only unresolved issue and then forcing a yes/no commitment. GPT-5.4 High did this at `c=0.95`: [GPT-5.4 High high-competition split](../scaling_experiment_20260404_064451/gpt-5-nano_vs_gpt-5.4-high/strong_first/comp_0.95/turns_2/run_2/experiment_results.json). It asked directly whether Stone was non-negotiable, then obtained the exact proposal it wanted in Round 2. This is why the order plot matters: going first lets stronger models set the frame, and the GPT-5-nano baseline order slopes are steeper when the adversary starts in Games 1 and 2.",
        "",
        "Lower-Elo losses are often protocol losses, not principled concessions. A Llama-3.2-3B run repeatedly generated the fallback allocation with the reason \"Failed to parse response - defaulting to proposer gets all\": [Llama-3.2-3B parser failure](../scaling_experiment_20260404_064451/gpt-5-nano_vs_llama-3.2-3b-instruct/strong_first/comp_0.9/turns_1/run_2/experiment_results.json). This produces no consensus and pulls down low-Elo payoff, rounds-to-consensus, optimality, and NBS proximity.",
        "",
        "Fairness interpretation: Game 1 fairness distance falls with Elo, but adversary fairness excess rises. Stronger models are not merely finding fairer splits; they are finding efficient splits and often placing themselves above the NBS share when conflict leaves a bargaining margin.",
        "",
        "### Game 2: Diplomacy",
        "",
        "The cooperative mechanism is cross-issue trade. In a low-conflict case, Claude Opus 4.6 Thinking explicitly saw that priorities were complementary: [Opus 4.6 Thinking issue trade](../diplomacy_20260405_082215/model_scale/gpt-5-nano_vs_claude-opus-4-6-thinking/weak_first/rho_n1_0_theta_0_0/run_1_experiment_results.json). The quote \"your priorities for mine\" captures the main dynamic: high-Elo models preserve the other side's top issues while asking for their own top issues. This produces high payoffs for both sides and explains the positive optimality slope.",
        "",
        "In high-conflict settings, stronger models are better at making their anchors explicit and diagnosing whether a package exists. GPT-5.4 High at `CI2=1` stated, \"AI chips and carbon are the anchors,\" then forced the package into its preferred region: [GPT-5.4 High high-CI package](../diplomacy_20260405_082215/model_scale/gpt-5-nano_vs_gpt-5.4-high/strong_first/rho_n1_0_theta_1_0/run_1_experiment_results.json). The final agreement gave the adversary 88.42 utility and the baseline 33.22, which is the competitive version of capability scaling.",
        "",
        "The low-Elo failure mode is diffuse, recursive negotiation. In a Llama-3.2-1B high-CI run, the model repeatedly restated broad middle-ground language and never converted it into mutually accepted proposals: [Llama-3.2-1B no-consensus treaty](../diplomacy_20260405_082215/model_scale/gpt-5-nano_vs_llama-3.2-1b-instruct/strong_first/rho_n1_0_theta_1_0/run_1_experiment_results.json). This explains why stronger models do not need more rounds on average; they generally converge slightly faster because they translate priorities into concrete numeric packages.",
        "",
        "Fairness interpretation: Game 2 has the clearest extraction pattern. Fairness distance falls with Elo, while adversary fairness excess rises. Stronger models are closer to the NBS frontier in absolute distance, but they also capture more of the surplus above their own NBS benchmark, especially in high-CI cells where the baseline payoff slope turns negative.",
        "",
        "### Game 3: Cofunding",
        "",
        "Game 3 adds a feasibility and trust problem: a model can win either by identifying a jointly fundable high-surplus set, or by shifting contributions onto the other side. A GPT-5.4 High run at `CI3=0.8` shows the latter: [GPT-5.4 High cost-shifted Harborview](../cofunding_20260405_083548/model_scale/gpt-5-nano_vs_gpt-5.4-high/weak_first/alpha_0_0_sigma_0_2/run_1_experiment_results.json). The adversary said, \"Rounds reset\" and \"Parkside still gives me zero value,\" then got Harborview funded with both agents paying 11. The final utilities were -8.91 for the baseline and 44.55 for the adversary, a clean cost-shifting example.",
        "",
        "The hardest cofunding cases also create deadlocks when both agents try to use cross-round promises that the game does not bind. In a Claude Opus 4.6 Thinking run, the stronger model eventually says, \"you keep proposing Market Street ... then rejecting it at the vote\": [Opus 4.6 Thinking cofunding deadlock](../cofunding_20260405_083548/model_scale/gpt-5-nano_vs_claude-opus-4-6-thinking/weak_first/alpha_0_0_sigma_0_2/run_1_experiment_results.json). This is why the `CI3=0.8` adversary-payoff slope is weaker than the lower-CI slopes: hard scarcity can cap realized gains through no-consensus outcomes.",
        "",
        "In cooperative cofunding cells, capability looks pro-social. In an Amazon Nova Micro run, both sides converged on equal contributions to shared high-value projects: [Nova Micro equal-share funding](../cofunding_20260405_083548/model_scale/gpt-5-nano_vs_amazon-nova-micro-v1.0/strong_first/alpha_1_0_sigma_1_0/run_1_experiment_results.json). The relevant rationale was: \"Simple, equal contributions maintain fairness and reduce deadlock risk.\" This is the cooperative counterpart to the cost-shifting examples and explains why Game 3 baseline payoff can rise with adversary Elo in the aggregate.",
        "",
        "Optimality and Lindahl interpretation: stronger models improve social-welfare optimality in Game 3 for both baselines, except that the highest-CI stratum is bottlenecked by deadlock. Lindahl distance falls with Elo overall, but cost-shift samples show that lower distance is not the same as no exploitation; stronger models can both fund more optimal projects and allocate more of the cost burden to the baseline.",
        "",
        "### Synthesis",
        "",
        "Capability scaling is robust: adversary payoff slopes are positive in all three games for both baselines. Competition determines who captures the gains. In cooperative cells, higher Elo mostly improves package discovery and social welfare; in high-competition cells, it improves anchoring, commitment tests, and cost shifting. Rounds-to-consensus are flat or slightly decreasing with Elo, so the payoff advantage is not coming from longer bargaining. It comes from cleaner proposals, better recognition of feasible trades, and stronger control over which proposal reaches the vote.",
        "",
    ]


def write_report(df: pd.DataFrame, slopes: pd.DataFrame) -> None:
    lines: list[str] = [
        "# N=2 Baseline Comparison: GPT-5-nano and Llama-3.3",
        "",
        "This report compares the standard `gpt-5-nano` baseline experiments with the appendix `llama-3.3-70b-instruct` baseline experiments across the three N=2 games.",
        "",
        "Competition-index convention recovered from prior local Codex history and existing scripts: Game 2 uses `CI2 = theta * (1 - rho) / 2`; Game 3 uses `CI3 = (1 - alpha) * (1 - sigma)`.",
        "",
        "Error bars in plotted aggregate points are `mean ± 1 SEM` across repeated runs in that plotted group.",
        "",
        "Generated files:",
        "",
        "- `all_runs_with_metrics.csv`: one row per loaded run with payoff, order, competition, rounds, optimality, and fairness metrics.",
        "- `overall_by_model_game.csv`: model/game means.",
        "- `by_competition_model_game.csv`: model/game/competition means.",
        "- `model_roster_by_game.csv`: loaded plotted-model counts by game.",
        "- `model_roster_presence_inconsistencies.csv`: loaded plotted models present in only a subset of games.",
        "- `raw_config_model_roster_by_game.csv`: configured model counts by game before result-file loading.",
        "- `raw_config_model_presence_inconsistencies.csv`: configured models present in only a subset of games.",
        "- `slope_summary.csv`: all plotted linear slopes, expressed as utility or metric units per 100 Elo.",
        "",
        "Fairness definitions: Game 1 and Game 2 use the Nash Bargaining Solution with disagreement utility 0. Game 3 uses Lindahl-style proportional cost sharing over the actually funded project set; the optimality plot for Game 3 uses the surplus-maximizing funded set under the total budget.",
        "",
        "Rounds-to-consensus plots average `final_round` only among runs that actually reached consensus; no-consensus runs are reflected separately in `consensus_rate` columns.",
        "",
        "When a plotted aggregate cell has loaded runs but the y-metric is undefined, the point is shown as a small `x` at the bottom of the panel rather than being silently dropped. This occurs mainly for rounds-to-consensus after no-consensus runs and for Game 3 Lindahl fairness when no project is funded. Fitted lines and displayed slopes use only defined y-values.",
        "",
        "## Loaded Runs",
        "",
        "| Baseline | Game | Runs | Models | Consensus rate |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    inventory = (
        df.groupby(["baseline_label", "game_id", "game_label"], as_index=False)
        .agg(
            runs=("adversary_utility", "size"),
            models=("adversary_model", "nunique"),
            consensus=("consensus_reached", "mean"),
        )
        .sort_values(["baseline_label", "game_id"])
    )
    for _, row in inventory.iterrows():
        lines.append(
            f"| {row['baseline_label']} | {row['game_label']} | {int(row['runs'])} | "
            f"{int(row['models'])} | {float(row['consensus']):.2f} |"
        )

    lines.extend(["", "## Headline Slopes", ""])
    for spec in BASELINES:
        lines.extend([f"### {spec.label}", ""])
        for plot_file, title in [
            ("01_adversary_payoff_overall.png", "Adversary payoff"),
            ("03_baseline_payoff_overall.png", "Baseline payoff"),
            ("07_rounds_to_consensus_overall.png", "Rounds to consensus"),
            ("09_optimality_ratio_overall.png", "Optimality ratio"),
            ("11_fairness_distance_overall.png", "Fairness distance"),
        ]:
            sub = slopes[(slopes["baseline_key"].eq(spec.key)) & (slopes["plot"].eq(plot_file))]
            if sub.empty:
                continue
            lines.extend([f"**{title}**", "", "| Game | Slope / 100 Elo | Pearson r | n |", "| --- | ---: | ---: | ---: |"])
            for _, row in sub.sort_values("game_id").iterrows():
                lines.append(
                    f"| {GAME_LABELS[row['game_id']]} | {fmt_signed(row['slope_per_100_elo'])} | "
                    f"{fmt_float(row['pearson_r'])} | {int(row['n_points'])} |"
                )
            lines.append("")

    lines.extend(qualitative_report_lines())

    for spec in BASELINES:
        lines.extend(["", f"## {spec.label} Plots", ""])
        lines.extend(report_plot_lines(spec))

    report_path = OUT_DIR / "n2_baseline_comparison_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    elo_map = load_combined_elo_map()
    frames = [load_baseline_rows(spec, elo_map) for spec in BASELINES]
    all_runs = pd.concat(frames, ignore_index=True)
    slopes = make_all_plots(all_runs)
    write_summary_tables(all_runs, slopes)
    write_report(all_runs, slopes)
    print(f"Wrote N=2 baseline comparison bundle to {OUT_DIR}")
    print(f"Rows loaded: {len(all_runs)}")


if __name__ == "__main__":
    main()

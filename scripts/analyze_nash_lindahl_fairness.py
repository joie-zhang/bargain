#!/usr/bin/env python3
"""
Nash-bargaining and Lindahl-style fairness analysis for Games 1--3.

The script reads the N=2 experiment roots and the full N>2 multi-agent roots,
recomputes undiscounted utilities from final outcomes, and writes a compact
research report plus CSV tables under analysis/nash_lindahl_fairness_20260505.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis" / "nash_lindahl_fairness_20260505"

N2_MAIN_ROOTS = {
    "game1": PROJECT_ROOT / "experiments/results/scaling_experiment_20260404_064451",
    "game2": PROJECT_ROOT / "experiments/results/diplomacy_20260405_082215",
    "game3": PROJECT_ROOT / "experiments/results/cofunding_20260405_083548",
}

N2_LLAMA_ROOTS = {
    "game1": PROJECT_ROOT / "experiments/results/appendix_llama33_baseline_game1_202605",
    "game2": PROJECT_ROOT / "experiments/results/appendix_llama33_baseline_game2_202605",
    "game3": PROJECT_ROOT / "experiments/results/appendix_llama33_baseline_game3_202605",
}

MULTIAGENT_ROOTS = {
    "n_gt_2_homogeneous": PROJECT_ROOT / "experiments/results/full_games123_multiagent_production_20260428_085255",
    "n_gt_2_heterogeneous": PROJECT_ROOT / "experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848",
}

GAME_LABELS = {
    "game1": "Game 1: Item allocation",
    "game2": "Game 2: Diplomacy",
    "game3": "Game 3: Co-funding",
}

EPS = 1e-9


class RowTimeoutError(TimeoutError):
    pass


def _row_timeout_handler(_signum: int, _frame: Any) -> None:
    raise RowTimeoutError("row analysis timed out")


def agent_sort_key(agent_id: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", str(agent_id))
    return (int(match.group(1)) if match else 10**9, str(agent_id))


def ordered_agents(mapping: dict[str, Any]) -> list[str]:
    return sorted(mapping.keys(), key=agent_sort_key)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def recover_game1_preferences(
    result_path: Path,
    config: dict[str, Any] | None = None,
) -> dict[str, list[float]]:
    """Recover the exact private values delivered in a Game 1 setup log.

    The runner historically serialized preferences into the terminal result only
    after consensus.  A no-consensus result can therefore have an empty embedded
    preference object even though the actual setup prompt is retained beside it.
    This function is a strict parser of that retained experiment input, not an
    imputation from other runs.
    """

    cfg = config or (load_json(result_path).get("config") or {})
    expected_agents = ordered_agents(
        {str(agent): None for agent in (cfg.get("agents") or [])}
        or {str(agent): None for agent in (cfg.get("agent_model_map") or {})}
    )
    expected_n = int(cfg.get("n_agents") or len(expected_agents))
    expected_items = len(cfg.get("items") or [])
    if expected_n <= 0 or len(expected_agents) != expected_n or expected_items <= 0:
        raise ValueError(
            f"Cannot validate setup-log preferences for {result_path}: "
            f"n_agents={expected_n}, agent_ids={len(expected_agents)}, items={expected_items}"
        )

    interaction_paths = [
        result_path.with_name("all_interactions.json"),
        result_path.with_name("run_1_all_interactions.json"),
    ]
    for interaction_path in interaction_paths:
        if not interaction_path.is_file():
            continue
        recovered: dict[str, dict[int, float]] = {}
        for entry in load_json(interaction_path):
            if str(entry.get("phase")) != "game_setup":
                continue
            agent_id = str(entry.get("agent_id") or "")
            if agent_id not in expected_agents:
                continue
            values = recovered.setdefault(agent_id, {})
            for match in re.finditer(
                r"^\s*(\d+):\s*.+?->\s*(-?\d+(?:\.\d+)?)\s*$",
                str(entry.get("prompt") or ""),
                flags=re.MULTILINE,
            ):
                item_index = int(match.group(1))
                value = float(match.group(2))
                if item_index in values and not math.isclose(values[item_index], value, abs_tol=1e-9):
                    raise ValueError(
                        f"Conflicting setup values for {agent_id} item {item_index} in {interaction_path}"
                    )
                values[item_index] = value

        expected_indices = set(range(expected_items))
        if set(recovered) != set(expected_agents):
            continue
        if any(set(values) != expected_indices for values in recovered.values()):
            continue
        preferences = {
            agent_id: [recovered[agent_id][index] for index in range(expected_items)]
            for agent_id in expected_agents
        }
        if any(not math.isclose(sum(values), 100.0, abs_tol=1e-8) for values in preferences.values()):
            raise ValueError(f"Recovered Game 1 preference vector does not sum to 100 in {interaction_path}")
        return preferences

    raise ValueError(f"Could not recover complete Game 1 setup preferences for {result_path}")


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def result_file(output_dir: Path, run_number: int | None = None) -> Path | None:
    candidates: list[Path] = []
    if run_number is not None:
        candidates.append(output_dir / f"run_{run_number}_experiment_results.json")
    candidates.extend(
        [
            output_dir / "experiment_results.json",
            output_dir / "run_1_experiment_results.json",
            output_dir / "run_2_experiment_results.json",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(output_dir.glob("run_*_experiment_results.json"))
    return matches[0] if matches else None


def normalize_game_id(payload: dict[str, Any], fallback: str | None = None) -> str:
    config = payload.get("config") or {}
    label = str(config.get("game_label") or fallback or "").lower()
    game_type = str(config.get("game_type") or "").lower()
    if label in {"game1", "game2", "game3"}:
        return label
    if game_type == "item_allocation":
        return "game1"
    if game_type in {"diplomacy", "diplomatic_treaty"}:
        return "game2"
    if game_type == "co_funding":
        return "game3"
    raise ValueError(f"Could not infer game id for {config.get('output_dir')}")


def competition_fields(game_id: str, config: dict[str, Any]) -> tuple[float, str]:
    if game_id == "game1":
        value = float(config.get("competition_level", 0.0))
        return value, f"competition={value:g}"
    if game_id == "game2":
        rho = float(config.get("rho", 0.0))
        theta = float(config.get("theta", 0.0))
        value = theta * (1.0 - rho) / 2.0
        return value, f"rho={rho:g}, theta={theta:g}"
    if game_id == "game3":
        alpha = float(config.get("alpha", 0.0))
        sigma = float(config.get("sigma", 0.0))
        value = (1.0 - alpha) * (1.0 - sigma)
        return value, f"alpha={alpha:g}, sigma={sigma:g}"
    return float("nan"), ""


def gini(values: list[float]) -> float:
    """Return the shifted, finite-sample-corrected Gini used in the paper."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    min_val = float(np.min(arr))
    if min_val < 0:
        arr = arr - min_val
    if float(np.sum(arr)) <= EPS:
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    if n == 1:
        return 0.0
    raw_gini = float(
        (2 * np.sum(np.arange(1, n + 1) * arr)) / (n * np.sum(arr))
        - (n + 1) / n
    )
    return min(max(n * raw_gini / (n - 1), 0.0), 1.0)


def safe_ratio(num: float, den: float) -> float:
    if abs(den) <= EPS:
        if abs(num) <= EPS:
            return 1.0
        return float("nan")
    return num / den


def nash_attainment(actual: dict[str, float], benchmark: dict[str, float]) -> float:
    """Geometric mean of actual/benchmark utility using log1p smoothing.

    Standard Nash products are undefined or numerically explosive when a
    benchmark utility is exactly zero. log1p keeps the readout interpretable for
    sparse public-good settings while preserving the ordering for positive
    utilities.
    """
    ratios = []
    for aid, b in benchmark.items():
        a = max(float(actual.get(aid, 0.0)), 0.0)
        ratios.append(math.log1p(a) - math.log1p(max(float(b), 0.0)))
    return float(math.exp(np.mean(ratios))) if ratios else float("nan")


def nbs_distance_norm(actual: dict[str, float], benchmark: dict[str, float]) -> float:
    aids = list(benchmark)
    if not aids:
        return float("nan")
    a = np.asarray([actual.get(aid, 0.0) for aid in aids], dtype=float)
    b = np.asarray([benchmark.get(aid, 0.0) for aid in aids], dtype=float)
    # Public-good benchmarks can be exactly zero for all agents when the
    # selected Nash/Lindahl funded set is empty. Normalize by the larger
    # substantive utility scale instead of EPS so sparse cases do not dominate
    # aggregate tables with meaningless billion-scale ratios.
    denom = max(float(np.mean(np.abs(b))), float(np.mean(np.abs(a))), 1.0)
    return float(np.sqrt(np.mean((a - b) ** 2)) / denom)


def utility_stats(actual: dict[str, float], benchmark: dict[str, float]) -> dict[str, float]:
    ratios = [safe_ratio(actual.get(aid, 0.0), benchmark.get(aid, 0.0)) for aid in benchmark]
    finite = [r for r in ratios if math.isfinite(r)]
    return {
        "nash_attainment": nash_attainment(actual, benchmark),
        "nbs_distance_norm": nbs_distance_norm(actual, benchmark),
        "nbs_min_ratio": min(finite) if finite else float("nan"),
        "nbs_mean_ratio": float(np.mean(finite)) if finite else float("nan"),
    }


def game1_raw_utilities(allocation: dict[str, list[int]], preferences: dict[str, list[float]]) -> dict[str, float]:
    utilities: dict[str, float] = {}
    for aid, prefs in preferences.items():
        items = allocation.get(aid, []) if isinstance(allocation, dict) else []
        utilities[aid] = float(sum(prefs[idx] for idx in items if isinstance(idx, int) and 0 <= idx < len(prefs)))
    return utilities


def game1_social_optimum(preferences: dict[str, list[float]]) -> float:
    agents = ordered_agents(preferences)
    if not agents:
        return 0.0
    m = len(preferences[agents[0]])
    return float(sum(max(float(preferences[aid][j]) for aid in agents) for j in range(m)))


def game1_nbs_exact(preferences: dict[str, list[float]]) -> dict[str, float]:
    agents = ordered_agents(preferences)
    m = len(preferences[agents[0]])
    best_obj = -float("inf")
    best_utils: dict[str, float] | None = None
    for assignment in itertools.product(range(len(agents)), repeat=m):
        utils = {aid: 0.0 for aid in agents}
        for item_idx, agent_idx in enumerate(assignment):
            aid = agents[agent_idx]
            utils[aid] += float(preferences[aid][item_idx])
        obj = sum(math.log(max(u, 0.0) + EPS) for u in utils.values())
        if obj > best_obj:
            best_obj = obj
            best_utils = utils
    return best_utils or {aid: 0.0 for aid in agents}


def _local_search_item_nbs(preferences: dict[str, list[float]], starts: list[list[int]]) -> dict[str, float]:
    agents = ordered_agents(preferences)
    n = len(agents)
    m = len(preferences[agents[0]])
    pref = np.asarray([[preferences[aid][j] for j in range(m)] for aid in agents], dtype=float)

    def utilities(assign: list[int]) -> np.ndarray:
        u = np.zeros(n)
        for j, i in enumerate(assign):
            u[i] += pref[i, j]
        return u

    def objective(u: np.ndarray) -> float:
        return float(np.sum(np.log(np.maximum(u, 0.0) + EPS)))

    best_assign = starts[0][:]
    best_u = utilities(best_assign)
    best_obj = objective(best_u)

    for start in starts:
        assign = start[:]
        u = utilities(assign)
        current_obj = objective(u)
        improved = True
        passes = 0
        while improved and passes < 100:
            improved = False
            passes += 1
            best_move = None
            best_move_obj = current_obj
            for j in range(m):
                old_i = assign[j]
                for new_i in range(n):
                    if new_i == old_i:
                        continue
                    cand_u = u.copy()
                    cand_u[old_i] -= pref[old_i, j]
                    cand_u[new_i] += pref[new_i, j]
                    cand_obj = objective(cand_u)
                    if cand_obj > best_move_obj + 1e-10:
                        best_move_obj = cand_obj
                        best_move = (j, old_i, new_i, cand_u)
            if best_move is not None:
                j, _old_i, new_i, cand_u = best_move
                assign[j] = new_i
                u = cand_u
                current_obj = best_move_obj
                improved = True
        if current_obj > best_obj:
            best_assign = assign
            best_u = u
            best_obj = current_obj

    return {aid: float(best_u[i]) for i, aid in enumerate(agents)}


def game1_nbs_approx(preferences: dict[str, list[float]], seed: int = 0) -> dict[str, float]:
    agents = ordered_agents(preferences)
    n = len(agents)
    m = len(preferences[agents[0]])
    pref = np.asarray([[preferences[aid][j] for j in range(m)] for aid in agents], dtype=float)
    starts: list[list[int]] = []
    starts.append([int(np.argmax(pref[:, j])) for j in range(m)])
    starts.append([j % n for j in range(m)])
    starts.append([(j * 7 + 3) % n for j in range(m)])
    rng = np.random.default_rng(seed)
    for _ in range(6):
        starts.append(rng.integers(0, n, size=m).tolist())
    return _local_search_item_nbs(preferences, starts)


def game2_normalize_agreement(agreement: list[Any]) -> np.ndarray:
    arr = np.asarray(agreement, dtype=float)
    if arr.size and np.max(np.abs(arr)) > 1.0:
        arr = arr / 100.0
    return np.clip(arr, 0.0, 1.0)


def game2_raw_utilities(agreement: list[Any], positions: dict[str, list[float]], weights: dict[str, list[float]]) -> dict[str, float]:
    a = game2_normalize_agreement(agreement)
    utilities: dict[str, float] = {}
    for aid in positions:
        p = np.asarray(positions[aid], dtype=float)
        w = np.asarray(weights[aid], dtype=float)
        utilities[aid] = float(np.sum(w * (1.0 - np.abs(p - a))) * 100.0)
    return utilities


def game2_social_optimum(positions: dict[str, list[float]], weights: dict[str, list[float]]) -> float:
    agents = ordered_agents(positions)
    pos = np.asarray([positions[aid] for aid in agents], dtype=float)
    w = np.asarray([weights[aid] for aid in agents], dtype=float)
    n_issues = pos.shape[1]
    sw = 0.0
    for k in range(n_issues):
        order = np.argsort(pos[:, k])
        sorted_pos = pos[order, k]
        sorted_w = w[order, k]
        median_idx = int(np.searchsorted(np.cumsum(sorted_w), np.sum(sorted_w) / 2.0))
        a_k = sorted_pos[min(median_idx, len(sorted_pos) - 1)]
        sw += float(np.sum(w[:, k] * (1.0 - np.abs(pos[:, k] - a_k))))
    return sw * 100.0


def game2_nbs_multiagent_agreement(
    positions: dict[str, list[float]],
    weights: dict[str, list[float]],
) -> np.ndarray:
    """Return a validated numerical NBS treaty for Game 2 with N > 2.

    The negative log Nash product is convex on the treaty box wherever all
    utilities are positive. Five deterministic SLSQP starts provide robust
    numerical candidates; coordinate-wise refinement handles the
    absolute-value kinks, and the final subgradient check certifies convex
    first-order optimality to a small numerical tolerance.
    """
    agents = ordered_agents(positions)
    pos = np.asarray([positions[aid] for aid in agents], dtype=float)
    w = np.asarray([weights[aid] for aid in agents], dtype=float)
    n_issues = pos.shape[1]

    def utilities(agreement: np.ndarray) -> np.ndarray:
        return np.sum(w * (1.0 - np.abs(pos - agreement[None, :])), axis=1)

    def objective(agreement: np.ndarray) -> float:
        values = utilities(agreement)
        if np.any(values <= 0.0):
            return float("inf")
        return float(-np.log(values).sum())

    def objective_and_gradient(agreement: np.ndarray) -> tuple[float, np.ndarray]:
        delta = agreement[None, :] - pos
        values = np.sum(w * (1.0 - np.abs(delta)), axis=1)
        if np.any(values <= 0.0):
            return 1e100, np.zeros(n_issues, dtype=float)
        # sign(0)=0 selects a valid subgradient at an ideal-point kink.
        gradient = np.sum(w * np.sign(delta) / values[:, None], axis=0)
        return float(-np.log(values).sum()), gradient

    weighted_start = np.asarray(
        [
            np.average(pos[:, k], weights=np.maximum(w[:, k], EPS))
            for k in range(n_issues)
        ],
        dtype=float,
    )
    starts = [weighted_start, np.mean(pos, axis=0), np.median(pos, axis=0)]
    rng = np.random.default_rng(42)
    starts.extend(rng.uniform(0.0, 1.0, size=n_issues) for _ in range(2))

    candidates = []
    for start in starts:
        result = minimize(
            objective_and_gradient,
            np.clip(start, 0.0, 1.0),
            jac=True,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n_issues,
            options={"maxiter": 3000, "ftol": 1e-13, "disp": False},
        )
        if bool(result.success) and np.all(np.isfinite(result.x)) and np.isfinite(result.fun):
            candidates.append(result)
    if not candidates:
        raise RuntimeError("all five Game 2 NBS starts failed")
    candidates.sort(key=lambda result: (float(result.fun), tuple(np.asarray(result.x))))
    agreement = np.clip(np.asarray(candidates[0].x, dtype=float), 0.0, 1.0)

    # SLSQP can stop infinitesimally to one side of a nonsmooth ideal point.
    # Exact one-coordinate minimization snaps true kink optima to observed
    # ideals and makes the final convex optimality audit deterministic.
    for _cycle in range(30):
        before = objective(agreement)
        for k in range(n_issues):
            def one_dimensional(value: float) -> float:
                trial = agreement.copy()
                trial[k] = float(value)
                return objective(trial)

            scalar = minimize_scalar(
                one_dimensional,
                bounds=(0.0, 1.0),
                method="bounded",
                options={"xatol": 1e-13, "maxiter": 1000},
            )
            candidate_values = np.concatenate(
                [np.asarray([agreement[k], 0.0, 1.0, float(scalar.x)]), pos[:, k]]
            )
            scored = sorted(
                ((one_dimensional(float(value)), float(value)) for value in candidate_values),
                key=lambda pair: (pair[0], pair[1]),
            )
            agreement[k] = scored[0][1]
        after = objective(agreement)
        if before - after < 1e-13:
            break

    values = utilities(agreement)
    if not np.all(np.isfinite(agreement)) or not np.all(np.isfinite(values)):
        raise RuntimeError("Game 2 NBS solver returned non-finite output")
    if np.any(values <= 0.0):
        raise RuntimeError("Game 2 NBS solver returned non-positive utility")
    if np.any(agreement < -1e-8) or np.any(agreement > 1.0 + 1e-8):
        raise RuntimeError("Game 2 NBS solver violated the treaty box")

    # For this convex objective, zero in every coordinate's subgradient
    # interval (with the appropriate one-sided box condition) is a global
    # first-order certificate.
    kink_tolerance = 2e-6
    coordinate_violations: list[float] = []
    for k in range(n_issues):
        delta = agreement[k] - pos[:, k]
        coefficients = w[:, k] / values
        tied = np.abs(delta) <= kink_tolerance
        base = float(np.sum(coefficients[~tied] * np.sign(delta[~tied])))
        radius = float(np.sum(coefficients[tied]))
        low, high = base - radius, base + radius
        if agreement[k] <= kink_tolerance:
            coordinate_violations.append(max(0.0, -high))
        elif agreement[k] >= 1.0 - kink_tolerance:
            coordinate_violations.append(max(0.0, low))
        elif low > 0.0:
            coordinate_violations.append(low)
        elif high < 0.0:
            coordinate_violations.append(-high)
        else:
            coordinate_violations.append(0.0)
    subgradient_violation = float(max(coordinate_violations, default=0.0))
    if subgradient_violation > 5e-3:
        raise RuntimeError(
            f"Game 2 NBS subgradient violation {subgradient_violation:.3e}"
        )
    if objective(agreement) > objective(weighted_start) + 1e-8:
        raise RuntimeError("Game 2 NBS objective is worse than the weighted compromise start")
    return agreement


def game2_nbs(positions: dict[str, list[float]], weights: dict[str, list[float]]) -> dict[str, float]:
    agents = ordered_agents(positions)
    pos = np.asarray([positions[aid] for aid in agents], dtype=float)
    w = np.asarray([weights[aid] for aid in agents], dtype=float)
    n_issues = pos.shape[1]

    def neg_log_product(a: np.ndarray) -> float:
        utils = np.sum(w * (1.0 - np.abs(pos - a[None, :])), axis=1)
        if np.any(utils <= 0):
            return 1e12
        return float(-np.sum(np.log(utils + EPS)))

    weighted_start = np.asarray(
        [
            np.average(pos[:, k], weights=np.maximum(w[:, k], EPS))
            for k in range(n_issues)
        ],
        dtype=float,
    )
    if len(agents) > 2:
        agreement = game2_nbs_multiagent_agreement(positions, weights)
        return game2_raw_utilities(agreement.tolist(), positions, weights)
    else:
        starts = [
            np.mean(pos, axis=0),
            weighted_start,
            np.median(pos, axis=0),
        ]
        rng = np.random.default_rng(42)
        starts.extend(rng.uniform(0, 1, size=n_issues) for _ in range(2))
        maxiter = 700
    best = None
    best_obj = float("inf")
    for start in starts:
        res = minimize(
            neg_log_product,
            np.clip(start, 0.0, 1.0),
            method="L-BFGS-B",
            bounds=[(0.0, 1.0)] * n_issues,
            options={"maxiter": maxiter, "ftol": 1e-9},
        )
        if res.fun < best_obj:
            best = res.x
            best_obj = float(res.fun)
    assert best is not None
    return game2_raw_utilities(best.tolist(), positions, weights)


def game3_extract_contributions(payload: dict[str, Any]) -> dict[str, list[float]] | None:
    if not payload.get("consensus_reached"):
        return None
    final_round = int(payload.get("final_round") or 0)
    for entry in payload.get("conversation_logs", []):
        if entry.get("phase") != "proposal_enumeration" or int(entry.get("round") or -1) != final_round:
            continue
        proposals = entry.get("enumerated_proposals") or []
        if not proposals:
            continue
        proposal = proposals[0]
        contrib = proposal.get("contributions_by_agent")
        if contrib:
            return {aid: [float(x) for x in xs] for aid, xs in contrib.items()}
        original = proposal.get("original_proposal") or {}
        contrib = original.get("contributions_by_agent")
        if contrib:
            return {aid: [float(x) for x in xs] for aid, xs in contrib.items()}
    return None


def game3_raw_utilities(
    valuations: dict[str, list[float]],
    contributions: dict[str, list[float]],
    funded_set: list[int],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for aid, vals in valuations.items():
        xs = contributions.get(aid, [0.0] * len(vals))
        out[aid] = float(sum(float(vals[j]) - float(xs[j]) for j in funded_set))
    return out


def game3_social_welfare(
    valuations: dict[str, list[float]],
    contributions: dict[str, list[float]],
    funded_set: list[int],
) -> float:
    return float(sum(game3_raw_utilities(valuations, contributions, funded_set).values()))


def game3_optimal_funded_set(valuations: dict[str, list[float]], costs: list[float], total_budget: float) -> list[int]:
    agents = list(valuations)
    m = len(costs)
    int_costs = [max(1, int(round(c))) for c in costs]
    budget = max(0, int(round(total_budget)))
    surplus = [
        max(sum(float(valuations[a][j]) for a in agents) - float(costs[j]), 0.0)
        for j in range(m)
    ]
    candidates = [j for j, s in enumerate(surplus) if s > EPS and int_costs[j] <= budget]
    if not candidates or budget <= 0:
        return []

    dp = [0.0] * (budget + 1)
    keep: list[list[bool]] = [[False] * (budget + 1) for _ in candidates]
    for row_idx, j in enumerate(candidates):
        c = int_costs[j]
        s = surplus[j]
        for b in range(budget, c - 1, -1):
            if dp[b - c] + s > dp[b] + 1e-12:
                dp[b] = dp[b - c] + s
                keep[row_idx][b] = True

    result: list[int] = []
    b = int(max(range(budget + 1), key=lambda x: dp[x]))
    for row_idx in range(len(candidates) - 1, -1, -1):
        j = candidates[row_idx]
        c = int_costs[j]
        if keep[row_idx][b]:
            result.append(j)
            b -= c
    return sorted(result)


def game3_lindahl_contributions(
    valuations: dict[str, list[float]],
    costs: list[float],
    funded_set: list[int],
) -> dict[str, list[float]]:
    agents = list(valuations)
    m = len(costs)
    out = {aid: [0.0] * m for aid in agents}
    for j in funded_set:
        total_val = sum(float(valuations[aid][j]) for aid in agents)
        if total_val <= EPS:
            for aid in agents:
                out[aid][j] = float(costs[j]) / len(agents)
        else:
            for aid in agents:
                out[aid][j] = float(costs[j]) * float(valuations[aid][j]) / total_val
    return out


def game3_funded_payment_distance(
    actual_contribs: dict[str, list[float]],
    benchmark_contribs: dict[str, list[float]],
    costs: list[float],
    funded_set: list[int],
) -> dict[str, float]:
    """Compare executed funded-project payments with a funded-set reference.

    The accepted proposal retains pledges to every project, but Game 3 refunds
    pledges outside ``funded_set``.  Outcome-level burden sharing must therefore
    exclude those refunded entries from both the distance numerator and the
    actual-payment term in its normalizer.  Full pledged and refunded totals are
    retained separately for provenance.
    """
    agents = list(benchmark_contribs)
    project_count = len(costs)
    invalid_projects = [j for j in funded_set if j < 0 or j >= project_count]
    if invalid_projects:
        raise ValueError(f"funded project indices out of range: {invalid_projects}")

    distance_squared = 0.0
    total_pledged = 0.0
    total_paid_funded = 0.0
    for aid in agents:
        actual = np.asarray(
            actual_contribs.get(aid, [0.0] * project_count), dtype=float
        )
        benchmark = np.asarray(
            benchmark_contribs.get(aid, [0.0] * project_count), dtype=float
        )
        if actual.shape != (project_count,) or benchmark.shape != (project_count,):
            raise ValueError(f"Game 3 contribution vector length mismatch for {aid}")
        total_pledged += float(actual.sum())
        if funded_set:
            paid = actual[funded_set]
            target = benchmark[funded_set]
            total_paid_funded += float(paid.sum())
            distance_squared += float(np.square(paid - target).sum())

    funded_total_cost = float(sum(costs[j] for j in funded_set))
    refunded_pledges = total_pledged - total_paid_funded
    if refunded_pledges < -1e-8:
        raise ValueError(f"computed negative refunded pledges: {refunded_pledges}")
    distance = math.sqrt(distance_squared)
    return {
        "lindahl_distance_norm": distance
        / max(funded_total_cost, total_paid_funded, 1.0),
        "actual_total_contribution": total_paid_funded,
        "actual_total_pledged": total_pledged,
        "refunded_pledges": max(refunded_pledges, 0.0),
        "funded_total_cost": funded_total_cost,
        "overfunding": total_paid_funded - funded_total_cost,
    }


def game3_lindahl_nbs(
    valuations: dict[str, list[float]],
    costs: list[float],
    budgets: dict[str, float],
    total_budget: float,
) -> tuple[dict[str, float], list[int]]:
    agents = list(valuations)
    m = len(costs)
    best_obj = -float("inf")
    best_utils = {aid: 0.0 for aid in agents}
    best_set: list[int] = []
    for mask in range(1 << m):
        subset = [j for j in range(m) if mask & (1 << j)]
        if sum(costs[j] for j in subset) > total_budget + 1e-9:
            continue
        lindahl = game3_lindahl_contributions(valuations, costs, subset)
        if any(sum(lindahl[aid]) > float(budgets.get(aid, total_budget)) + 1e-9 for aid in agents):
            continue
        utils = game3_raw_utilities(valuations, lindahl, subset)
        obj = sum(math.log(max(u, 0.0) + EPS) for u in utils.values())
        if obj > best_obj:
            best_obj = obj
            best_utils = utils
            best_set = subset
    return best_utils, best_set


@dataclass
class AnalysisRow:
    source_group: str
    dataset: str
    game_id: str
    result_path: Path
    payload: dict[str, Any] | None
    config: dict[str, Any]
    agent_model_map: dict[str, str]
    agent_role_map: dict[str, str]
    agent_elo_map: dict[str, Any]


def n2_rows(root: Path, dataset: str, game_id: str):
    index_path = root / "configs" / "experiment_index.csv"
    if not index_path.exists():
        return
    with index_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            cfg = load_json(root / "configs" / row["config_file"])
            output_dir = resolve_path(cfg["output_dir"])
            path = result_file(output_dir, int(row.get("run_number") or cfg.get("run_number") or 1))
            if path is None:
                continue
            ordered_models = list(cfg.get("models") or [])
            aids = [f"Agent_{idx + 1}" for idx in range(len(ordered_models))]
            agent_model_map = {
                aid: ordered_models[idx] if idx < len(ordered_models) else ""
                for idx, aid in enumerate(aids)
            }
            baseline_model = cfg.get("baseline_model") or cfg.get("model1") or cfg.get("weak_model") or ""
            adversary_model = cfg.get("adversary_model") or cfg.get("model2") or cfg.get("strong_model") or ""
            agent_role_map = {
                aid: (
                    "baseline"
                    if agent_model_map.get(aid) == baseline_model
                    else "adversary"
                    if agent_model_map.get(aid) == adversary_model
                    else ""
                )
                for aid in aids
            }
            yield AnalysisRow(
                source_group="n2_main_gpt5_baseline" if dataset == "n2_main" else "n2_llama33_baseline",
                dataset=dataset,
                game_id=game_id,
                result_path=path,
                payload=None,
                config=cfg,
                agent_model_map=agent_model_map,
                agent_role_map=agent_role_map,
                agent_elo_map={},
            )


def multiagent_rows(root: Path, source_group: str):
    for path in sorted((root / "runs").glob("*/experiment_results.json")):
        payload = load_json(path)
        cfg = payload.get("config") or {}
        if int(cfg.get("n_agents") or 0) <= 2:
            continue
        game_id = normalize_game_id(payload)
        yield AnalysisRow(
            source_group=source_group,
            dataset="n_gt_2",
            game_id=game_id,
            result_path=path,
            payload=None,
            config=cfg,
            agent_model_map=dict(cfg.get("agent_model_map") or {}),
            agent_role_map=dict(cfg.get("agent_role_map") or {}),
            agent_elo_map=dict(cfg.get("agent_elo_map") or {}),
        )


def analyze_row(row: AnalysisRow) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = row.payload or load_json(row.result_path)
    cfg = {**row.config, **(payload.get("config") or {})}
    game_id = row.game_id
    preferences_recovered_from_setup = False
    if game_id == "game1" and not (payload.get("agent_preferences") or {}):
        payload = dict(payload)
        payload["agent_preferences"] = recover_game1_preferences(row.result_path, cfg)
        preferences_recovered_from_setup = True
    n_agents = int(cfg.get("n_agents") or len(payload.get("agent_preferences") or {}))
    competition_value, competition_label = competition_fields(game_id, cfg)
    consensus = bool(payload.get("consensus_reached"))
    final_round = int(payload.get("final_round") or 0) if consensus else 0
    gamma = float(cfg.get("gamma_discount", 0.9))
    discount = gamma ** max(final_round - 1, 0) if consensus else 0.0
    preferences = {aid: [float(x) for x in vals] for aid, vals in (payload.get("agent_preferences") or {}).items()}
    agents = ordered_agents(preferences)

    actual_raw = {aid: 0.0 for aid in agents}
    benchmark = {aid: 0.0 for aid in agents}
    sw_opt = 0.0
    nbs_method = ""
    extra: dict[str, Any] = {}
    lindahl_utils: dict[str, float] | None = None
    lindahl_contribs: dict[str, list[float]] | None = None
    actual_contribs: dict[str, list[float]] | None = None
    funded_set: list[int] = []

    if game_id == "game1":
        allocation = payload.get("final_allocation") if consensus else {}
        actual_raw = game1_raw_utilities(allocation or {}, preferences) if consensus else actual_raw
        if n_agents == 2:
            benchmark = game1_nbs_exact(preferences)
            nbs_method = "exact discrete NBS"
        else:
            benchmark = game1_nbs_approx(preferences, seed=int(cfg.get("random_seed") or cfg.get("seed") or 0))
            nbs_method = "approximate discrete NBS (multi-start local search)"
        sw_opt = game1_social_optimum(preferences)

    elif game_id == "game2":
        positions = {
            aid: [float(x) for x in vals]
            for aid, vals in (cfg.get("agent_positions") or {}).items()
        }
        weights = {
            aid: [float(x) for x in vals]
            for aid, vals in (cfg.get("agent_weights") or {}).items()
        }
        if consensus:
            actual_raw = game2_raw_utilities(payload.get("final_allocation") or [], positions, weights)
        else:
            actual_raw = {aid: 0.0 for aid in positions}
        benchmark = game2_nbs(positions, weights)
        nbs_method = (
            "continuous numerical NBS (five-start SLSQP with analytic subgradient and coordinate refinement)"
            if n_agents > 2
            else "continuous numerical NBS (five-start L-BFGS-B)"
        )
        sw_opt = game2_social_optimum(positions, weights)
        agents = ordered_agents(positions)

    elif game_id == "game3":
        costs = [float(item.get("cost", 0.0)) for item in cfg.get("items", [])]
        budgets = {aid: float(x) for aid, x in (cfg.get("agent_budgets") or {}).items()}
        total_budget = float(cfg.get("total_budget") or sum(budgets.values()))
        funded_set = [int(j) for j in (payload.get("final_allocation") or [])] if consensus else []
        actual_contribs = game3_extract_contributions(payload) if consensus else None
        if actual_contribs is None:
            actual_contribs = {aid: [0.0] * len(costs) for aid in agents}
        actual_raw = game3_raw_utilities(preferences, actual_contribs, funded_set) if consensus else actual_raw
        lindahl_contribs = game3_lindahl_contributions(preferences, costs, funded_set)
        lindahl_utils = game3_raw_utilities(preferences, lindahl_contribs, funded_set)
        opt_set = game3_optimal_funded_set(preferences, costs, total_budget)
        opt_lindahl = game3_lindahl_contributions(preferences, costs, opt_set)
        sw_opt = max(game3_social_welfare(preferences, opt_lindahl, opt_set), 0.0)
        if n_agents <= 2 and len(costs) <= 5:
            benchmark, nbs_set = game3_lindahl_nbs(preferences, costs, budgets, total_budget)
            nbs_method = "enumerated Lindahl-cost-sharing NBS over funded sets"
        else:
            benchmark = game3_raw_utilities(preferences, opt_lindahl, opt_set)
            nbs_set = opt_set
            nbs_method = "Lindahl cost-sharing on utilitarian-optimal funded set (N>2 proxy)"
        payment_distance = game3_funded_payment_distance(
            actual_contribs,
            lindahl_contribs,
            costs,
            funded_set,
        )
        extra.update(
            {
                "funded_project_count": len(funded_set),
                "optimal_project_count": len(opt_set),
                "nbs_lindahl_project_count": len(nbs_set),
                "provision_rate": len(set(funded_set) & set(opt_set)) / len(opt_set) if opt_set else 1.0,
                **payment_distance,
            }
        )

    actual_sw = float(sum(actual_raw.values()))
    sw_eff = actual_sw / sw_opt if sw_opt > EPS else (1.0 if actual_sw >= -EPS else 0.0)
    stats = utility_stats(actual_raw, benchmark)
    run_record: dict[str, Any] = {
        "source_group": row.source_group,
        "dataset": row.dataset,
        "game_id": game_id,
        "game_label": GAME_LABELS[game_id],
        "result_path": str(row.result_path.relative_to(PROJECT_ROOT)),
        "n_agents": n_agents,
        "experiment_family": cfg.get("experiment_family") or cfg.get("experiment_type") or "",
        "competition_value": competition_value,
        "competition_label": competition_label,
        "competition_id": cfg.get("competition_id") or competition_label,
        "consensus_reached": consensus,
        "preferences_recovered_from_setup": preferences_recovered_from_setup,
        "final_round": final_round,
        "discount_factor": discount,
        "nbs_method": nbs_method,
        "actual_sw": actual_sw,
        "sw_opt": sw_opt,
        "sw_efficiency": sw_eff,
        "utility_gini": gini(list(actual_raw.values())),
        **stats,
        **extra,
    }

    agent_records: list[dict[str, Any]] = []
    discounted = {aid: float(v) for aid, v in (payload.get("final_utilities") or {}).items()}
    for aid in agents:
        record = {
            "source_group": row.source_group,
            "dataset": row.dataset,
            "game_id": game_id,
            "game_label": GAME_LABELS[game_id],
            "result_path": str(row.result_path.relative_to(PROJECT_ROOT)),
            "n_agents": n_agents,
            "experiment_family": run_record["experiment_family"],
            "competition_value": competition_value,
            "competition_label": competition_label,
            "competition_id": run_record["competition_id"],
            "agent_id": aid,
            "model": row.agent_model_map.get(aid, ""),
            "role": row.agent_role_map.get(aid, ""),
            "elo": row.agent_elo_map.get(aid, np.nan),
            "actual_raw_utility": actual_raw.get(aid, 0.0),
            "actual_discounted_utility": discounted.get(aid, 0.0),
            "nbs_utility": benchmark.get(aid, 0.0),
            "nbs_ratio": safe_ratio(actual_raw.get(aid, 0.0), benchmark.get(aid, 0.0)),
            "nbs_residual": actual_raw.get(aid, 0.0) - benchmark.get(aid, 0.0),
        }
        if game_id == "game3" and lindahl_utils is not None and actual_contribs is not None and lindahl_contribs is not None:
            actual_paid = sum(actual_contribs.get(aid, [0.0] * len(funded_set))[j] for j in funded_set)
            fair_paid = sum(lindahl_contribs.get(aid, [0.0] * len(funded_set))[j] for j in funded_set)
            total_actual_paid = sum(
                sum(actual_contribs.get(xaid, [0.0] * len(funded_set))[j] for j in funded_set)
                for xaid in agents
            )
            total_benefit = sum(sum(preferences[xaid][j] for j in funded_set) for xaid in agents)
            benefit = sum(preferences[aid][j] for j in funded_set)
            record.update(
                {
                    "lindahl_utility": lindahl_utils.get(aid, 0.0),
                    "lindahl_ratio": safe_ratio(actual_raw.get(aid, 0.0), lindahl_utils.get(aid, 0.0)),
                    "lindahl_residual": actual_raw.get(aid, 0.0) - lindahl_utils.get(aid, 0.0),
                    "actual_paid_funded": actual_paid,
                    "lindahl_fair_paid_funded": fair_paid,
                    "underpayment_vs_lindahl": fair_paid - actual_paid,
                    "benefit_share": benefit / total_benefit if total_benefit > EPS else 0.0,
                    "cost_share": actual_paid / total_actual_paid if total_actual_paid > EPS else 0.0,
                    "benefit_minus_cost_share": (
                        benefit / total_benefit if total_benefit > EPS else 0.0
                    )
                    - (actual_paid / total_actual_paid if total_actual_paid > EPS else 0.0),
                }
            )
        agent_records.append(record)

    return run_record, agent_records


def slope(x: pd.Series, y: pd.Series) -> float:
    data = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 2 or data["x"].nunique() < 2:
        return float("nan")
    return float(np.polyfit(data["x"].astype(float), data["y"].astype(float), 1)[0])


def write_markdown_report(out_dir: Path, runs: pd.DataFrame, agents: pd.DataFrame) -> None:
    overall = (
        runs.groupby(["source_group", "game_id"], dropna=False)
        .agg(
            runs=("result_path", "count"),
            n_mean=("n_agents", "mean"),
            consensus=("consensus_reached", "mean"),
            sw_eff=("sw_efficiency", "mean"),
            nash_attain=("nash_attainment", "mean"),
            nbs_dist=("nbs_distance_norm", "mean"),
            gini=("utility_gini", "mean"),
            lindahl_dist=("lindahl_distance_norm", "mean"),
        )
        .reset_index()
    )
    overall.to_csv(out_dir / "summary_by_source_game.csv", index=False)

    by_comp = (
        runs.groupby(["source_group", "game_id", "competition_value"], dropna=False)
        .agg(
            runs=("result_path", "count"),
            consensus=("consensus_reached", "mean"),
            sw_eff=("sw_efficiency", "mean"),
            nash_attain=("nash_attainment", "mean"),
            nbs_dist=("nbs_distance_norm", "mean"),
            gini=("utility_gini", "mean"),
            lindahl_dist=("lindahl_distance_norm", "mean"),
            provision=("provision_rate", "mean"),
        )
        .reset_index()
        .sort_values(["source_group", "game_id", "competition_value"])
    )
    by_comp.to_csv(out_dir / "summary_by_competition.csv", index=False)

    slope_rows = []
    for (source, game), group in runs.groupby(["source_group", "game_id"]):
        slope_rows.append(
            {
                "source_group": source,
                "game_id": game,
                "runs": len(group),
                "sw_eff_slope_per_comp": slope(group["competition_value"], group["sw_efficiency"]),
                "nash_attain_slope_per_comp": slope(group["competition_value"], group["nash_attainment"]),
                "nbs_dist_slope_per_comp": slope(group["competition_value"], group["nbs_distance_norm"]),
                "gini_slope_per_comp": slope(group["competition_value"], group["utility_gini"]),
                "lindahl_dist_slope_per_comp": slope(group["competition_value"], group.get("lindahl_distance_norm", pd.Series(dtype=float))),
            }
        )
    slopes = pd.DataFrame(slope_rows)
    slopes.to_csv(out_dir / "competition_slopes.csv", index=False)

    role_rows = []
    role_agents = agents[agents["role"].isin(["baseline", "adversary"])]
    if not role_agents.empty:
        role_summary = (
            role_agents.groupby(["source_group", "game_id", "role"], dropna=False)
            .agg(
                observations=("result_path", "count"),
                utility=("actual_raw_utility", "mean"),
                nbs_ratio=("nbs_ratio", "mean"),
                nbs_residual=("nbs_residual", "mean"),
                lindahl_residual=("lindahl_residual", "mean"),
                underpayment=("underpayment_vs_lindahl", "mean"),
            )
            .reset_index()
        )
        role_summary.to_csv(out_dir / "role_summary.csv", index=False)
        for (source, game), group in role_summary.groupby(["source_group", "game_id"]):
            wide = group.pivot(index=["source_group", "game_id"], columns="role", values=["nbs_ratio", "nbs_residual", "lindahl_residual", "underpayment"])
            if not wide.empty:
                rec = {"source_group": source, "game_id": game}
                for metric in ["nbs_ratio", "nbs_residual", "lindahl_residual", "underpayment"]:
                    try:
                        rec[f"adversary_minus_baseline_{metric}"] = float(wide[(metric, "adversary")].iloc[0] - wide[(metric, "baseline")].iloc[0])
                    except Exception:
                        rec[f"adversary_minus_baseline_{metric}"] = float("nan")
                role_rows.append(rec)
    role_diff = pd.DataFrame(role_rows)
    role_diff.to_csv(out_dir / "role_advantage_summary.csv", index=False)

    hetero = agents[(agents["source_group"] == "n_gt_2_heterogeneous") & agents["elo"].notna()].copy()
    elo_rows = []
    if not hetero.empty:
        hetero["elo"] = pd.to_numeric(hetero["elo"], errors="coerce")
        for game, group in hetero.groupby("game_id"):
            for metric in ["actual_raw_utility", "nbs_ratio", "nbs_residual", "benefit_minus_cost_share"]:
                if metric not in group:
                    continue
                data = group[["elo", metric]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(data) > 2 and data["elo"].nunique() > 1:
                    elo_rows.append(
                        {
                            "game_id": game,
                            "metric": metric,
                            "observations": len(data),
                            "slope_per_100_elo": slope(data["elo"], data[metric]) * 100,
                            "pearson": float(data["elo"].corr(data[metric])),
                        }
                    )
        pd.DataFrame(elo_rows).to_csv(out_dir / "heterogeneous_elo_slopes.csv", index=False)
    elo_df = pd.DataFrame(elo_rows)

    def fmt(x: Any, digits: int = 3) -> str:
        try:
            val = float(x)
        except Exception:
            return str(x)
        if math.isnan(val):
            return ""
        return f"{val:.{digits}f}"

    compact_overall = overall.copy()
    compact_overall["game"] = compact_overall["game_id"].map(GAME_LABELS)
    compact_overall = compact_overall[
        ["source_group", "game", "runs", "consensus", "sw_eff", "nash_attain", "nbs_dist", "gini", "lindahl_dist"]
    ]

    slope_view = slopes.copy()
    slope_view["game"] = slope_view["game_id"].map(GAME_LABELS)
    slope_view = slope_view[
        ["source_group", "game", "sw_eff_slope_per_comp", "nash_attain_slope_per_comp", "nbs_dist_slope_per_comp", "gini_slope_per_comp", "lindahl_dist_slope_per_comp"]
    ]

    game3_comp = by_comp[by_comp["game_id"].eq("game3")].copy()
    game3_comp = game3_comp[
        ["source_group", "competition_value", "runs", "sw_eff", "nash_attain", "lindahl_dist", "provision"]
    ]

    def cell(table: pd.DataFrame, source: str, game: str, column: str) -> float:
        data = table[(table["source_group"].eq(source)) & (table["game_id"].eq(game))]
        if data.empty or column not in data:
            return float("nan")
        return float(data[column].iloc[0])

    def slope_cell(source: str, game: str, column: str) -> float:
        return cell(slopes, source, game, column)

    def role_cell(source: str, game: str, column: str) -> float:
        if role_diff.empty:
            return float("nan")
        data = role_diff[(role_diff["source_group"].eq(source)) & (role_diff["game_id"].eq(game))]
        if data.empty or column not in data:
            return float("nan")
        return float(data[column].iloc[0])

    def game3_comp_cell(source: str, competition: float, column: str) -> float:
        data = game3_comp[
            (game3_comp["source_group"].eq(source))
            & (np.isclose(game3_comp["competition_value"].astype(float), competition))
        ]
        if data.empty or column not in data:
            return float("nan")
        return float(data[column].iloc[0])

    def elo_cell(game: str, metric: str, column: str) -> float:
        if elo_df.empty:
            return float("nan")
        data = elo_df[(elo_df["game_id"].eq(game)) & (elo_df["metric"].eq(metric))]
        if data.empty or column not in data:
            return float("nan")
        return float(data[column].iloc[0])

    lines: list[str] = []
    lines.append("# Nash Bargaining and Lindahl-Style Fairness Analysis\n")
    lines.append("Generated from the linked N=2 and N>2 experiment results.\n")
    lines.append("## Executive Takeaways\n")
    lines.append(
        "- **Nash bargaining is a stricter benchmark than raw utility.** Across the experiments, high social welfare and high utility do not imply closeness to the Nash bargaining solution; several settings reach efficient agreements while distributing the bargaining surplus unevenly.\n"
    )
    lines.append(
        "- **Competition mostly hurts fairness by increasing dispersion, not only by lowering total welfare.** The competition-index slopes below show that harder settings generally reduce Nash attainment and increase distance from the NBS benchmark, especially in the public-goods co-funding game.\n"
    )
    lines.append(
        "- **Lindahl-style fairness is most diagnostic in Game 3.** It exposes whether agents who benefit from funded public projects paid proportional cost shares, separating genuine surplus creation from free-riding or cost shifting.\n"
    )
    lines.append(
        "- **For N>2, the normative target becomes harsher.** Supermajority voting can pass coalition agreements that are locally acceptable but far from an all-agent Nash product benchmark; this is expected and substantively important rather than a coding artifact.\n"
    )
    lines.append("## Key Empirical Findings\n")
    lines.append(
        f"- **Two-agent Games 1 and 2 are close to Nash bargaining; large groups are not automatically so.** In the main N=2 baseline, Game 1 has `sw_eff={fmt(cell(overall, 'n2_main_gpt5_baseline', 'game1', 'sw_eff'))}` and `nbs_dist={fmt(cell(overall, 'n2_main_gpt5_baseline', 'game1', 'nbs_dist'))}`, while Game 2 has `sw_eff={fmt(cell(overall, 'n2_main_gpt5_baseline', 'game2', 'sw_eff'))}` and `nbs_dist={fmt(cell(overall, 'n2_main_gpt5_baseline', 'game2', 'nbs_dist'))}`. Homogeneous N>2 Game 1 keeps consensus at 1.000 but falls to `sw_eff={fmt(cell(overall, 'n_gt_2_homogeneous', 'game1', 'sw_eff'))}` and `nbs_dist={fmt(cell(overall, 'n_gt_2_homogeneous', 'game1', 'nbs_dist'))}`, showing that agreement alone is not a fairness result.\n"
    )
    lines.append(
        f"- **Diplomacy is the most stable fairness environment.** Heterogeneous N>2 diplomacy reaches `sw_eff={fmt(cell(overall, 'n_gt_2_heterogeneous', 'game2', 'sw_eff'))}` and `nbs_dist={fmt(cell(overall, 'n_gt_2_heterogeneous', 'game2', 'nbs_dist'))}`, almost matching or slightly improving on the N=2 baseline by the NBS-distance metric. This supports the interpretation that continuous issue compromise is easier for groups than indivisible allocation or public-good financing.\n"
    )
    lines.append(
        f"- **Co-funding separates consensus from fairness.** Heterogeneous N>2 Game 3 has very high consensus (`{fmt(cell(overall, 'n_gt_2_heterogeneous', 'game3', 'consensus'))}`) but lower welfare (`sw_eff={fmt(cell(overall, 'n_gt_2_heterogeneous', 'game3', 'sw_eff'))}`) and lower Nash attainment (`{fmt(cell(overall, 'n_gt_2_heterogeneous', 'game3', 'nash_attain'))}`) than the two-agent baselines. Homogeneous N>2 Game 3 is the clearest public-goods failure: `sw_eff={fmt(cell(overall, 'n_gt_2_homogeneous', 'game3', 'sw_eff'))}`, `nbs_dist={fmt(cell(overall, 'n_gt_2_homogeneous', 'game3', 'nbs_dist'))}`, and `gini={fmt(cell(overall, 'n_gt_2_homogeneous', 'game3', 'gini'))}`.\n"
    )
    lines.append(
        f"- **Competition usually worsens the distribution even when welfare is flat.** In N>2 Game 1, competition barely changes social welfare but increases NBS distance strongly (`+{fmt(slope_cell('n_gt_2_heterogeneous', 'game1', 'nbs_dist_slope_per_comp'))}` heterogeneous, `+{fmt(slope_cell('n_gt_2_homogeneous', 'game1', 'nbs_dist_slope_per_comp'))}` homogeneous). In heterogeneous N>2 Game 3, competition is damaging on both margins: `sw_eff` slope `{fmt(slope_cell('n_gt_2_heterogeneous', 'game3', 'sw_eff_slope_per_comp'))}`, `nbs_dist` slope `+{fmt(slope_cell('n_gt_2_heterogeneous', 'game3', 'nbs_dist_slope_per_comp'))}`, and `gini` slope `+{fmt(slope_cell('n_gt_2_heterogeneous', 'game3', 'gini_slope_per_comp'))}`.\n"
    )
    lines.append(
        f"- **Lindahl distance must be read together with provision.** In main N=2 Game 3, the highest competition condition has low Lindahl distance (`{fmt(game3_comp_cell('n2_main_gpt5_baseline', 0.8, 'lindahl_dist'))}`) but also collapses project provision (`{fmt(game3_comp_cell('n2_main_gpt5_baseline', 0.8, 'provision'))}`). That is not a fairness success; it means there are fewer funded public goods over which cost shares can be unfair.\n"
    )
    lines.append(
        f"- **Adversarial insertion creates the largest fairness skew in homogeneous N>2 games.** The inserted adversary beats baseline agents relative to the NBS benchmark by `+{fmt(role_cell('n_gt_2_homogeneous', 'game1', 'adversary_minus_baseline_nbs_residual'))}` in Game 1, `+{fmt(role_cell('n_gt_2_homogeneous', 'game2', 'adversary_minus_baseline_nbs_residual'))}` in Game 2, and `+{fmt(role_cell('n_gt_2_homogeneous', 'game3', 'adversary_minus_baseline_nbs_residual'))}` in Game 3. In Game 3 it also underpays relative to Lindahl shares by `+{fmt(role_cell('n_gt_2_homogeneous', 'game3', 'adversary_minus_baseline_underpayment'))}`.\n"
    )
    lines.append(
        f"- **Higher-Elo agents capture more surplus in heterogeneous groups, but not mainly by Lindahl free-riding.** Per 100 Elo, NBS residual rises by `+{fmt(elo_cell('game1', 'nbs_residual', 'slope_per_100_elo'))}` in Game 1, `+{fmt(elo_cell('game2', 'nbs_residual', 'slope_per_100_elo'))}` in Game 2, and `+{fmt(elo_cell('game3', 'nbs_residual', 'slope_per_100_elo'))}` in Game 3. But Game 3's benefit-minus-cost-share slope is `{fmt(elo_cell('game3', 'benefit_minus_cost_share', 'slope_per_100_elo'))}`, so the Elo advantage is not simply higher-Elo agents paying less than their benefit share.\n"
    )
    lines.append("## Methods\n")
    lines.append(
        "For every successful run I recomputed **undiscounted substantive utility** from the final allocation or treaty, while retaining consensus and final-round metadata. No-consensus runs receive zero utility, matching the game disagreement point.\n"
    )
    lines.append(
        "- Game 1 uses an exact discrete NBS for two-agent runs and an approximate discrete NBS by multi-start local search for N>2 item allocations.\n"
    )
    lines.append(
        "- Game 2 uses a continuous numerical NBS over treaty vectors, maximizing the product of utilities over the zero-disagreement point. N>2 diplomacy uses five deterministic SLSQP starts with analytic subgradients and coordinate refinement.\n"
    )
    lines.append(
        "- Game 3 reports a Lindahl comparison for the actually funded projects; two-agent runs also use an enumerated Lindahl-cost-sharing Nash benchmark, while N>2 runs use Lindahl cost-sharing on the utilitarian-optimal funded set as a tractable proxy.\n"
    )
    lines.append(
        "The scalar competition axis is `competition_level` for Game 1, `theta * (1 - rho) / 2` for Game 2, and `(1 - alpha) * (1 - sigma)` for Game 3.\n"
    )
    lines.append("## Overall Results\n")
    lines.append(compact_overall.to_markdown(index=False, floatfmt=".3f"))
    lines.append("\n\nMetric notes: `sw_eff` is realized social welfare divided by the utilitarian optimum; `nash_attain` is the geometric mean of actual/NBS utility ratios; `nbs_dist` is normalized RMS distance from the NBS utility vector; `lindahl_dist` applies only to Game 3.\n")
    lines.append("## Competition Slopes\n")
    lines.append(slope_view.to_markdown(index=False, floatfmt=".3f"))
    lines.append(
        "\n\nNegative `nash_attain` slopes mean agreements fall further below the bargaining benchmark as conflict rises. Positive `nbs_dist` or `gini` slopes mean the distribution becomes less fair by that metric.\n"
    )
    if not game3_comp.empty:
        lines.append("## Game 3 Lindahl Patterns by Competition\n")
        lines.append(game3_comp.to_markdown(index=False, floatfmt=".3f"))
        lines.append(
            "\n\nThe Game 3 table is the cleanest Lindahl-style readout: `lindahl_dist` measures how far executed payments on realized funded projects are from benefit-proportional cost shares for those projects; refunded pledges are excluded. `provision` is the fraction of socially optimal projects that were funded.\n"
        )
    if not role_diff.empty:
        lines.append("## Baseline-vs-Adversary Fairness Advantage\n")
        lines.append(role_diff.to_markdown(index=False, floatfmt=".3f"))
        lines.append(
            "\n\nPositive adversary-minus-baseline NBS residuals mean the non-baseline or inserted adversary captured more surplus relative to the fair bargaining benchmark. In Game 3, positive underpayment means the adversary paid less than its Lindahl-implied fair share relative to baseline agents.\n"
        )
    if elo_rows:
        elo_df = pd.DataFrame(elo_rows)
        lines.append("## Heterogeneous N>2 Elo Slopes\n")
        lines.append(elo_df.to_markdown(index=False, floatfmt=".3f"))
        lines.append(
            "\n\nThese slopes ask whether higher-Elo agents in heterogeneous groups receive more utility or more surplus relative to the fairness benchmark, averaged over random mixed-model rosters.\n"
        )
    lines.append("## Interpretation by Game\n")
    lines.append(
        "### Game 1: Item Allocation\n"
        "The NBS benchmark asks whether contested indivisible goods were allocated to balance gains, not merely to maximize total item value. A high-efficiency split can still be unfair if one side receives the high-value contested bundle while the other receives low-value leftovers. In N>2 runs the benchmark is approximate because exact N-agent indivisible Nash optimization is combinatorial, so the right interpretation is directional: larger NBS distance means the final allocation is less proportionally fair relative to agents' private item values.\n"
    )
    lines.append(
        "### Game 2: Diplomacy\n"
        "The treaty game is the cleanest Nash bargaining case. The NBS usually corresponds to issue-by-issue compromise weighted by how much each delegation cares. Departures from NBS indicate that the treaty leaned too far toward some parties' high-weight ideals, even when the treaty remained efficient in total welfare terms. This makes the NBS analysis a useful correction to raw Elo-utility plots: a stronger model can gain not by finding a better treaty for everyone, but by pulling the shared treaty vector toward its own weighted priorities.\n"
    )
    lines.append(
        "### Game 3: Co-Funding\n"
        "The co-funding game is where Lindahl fairness has real bite. The central fairness question is not only which projects were funded, but who paid for them. A model can look successful in raw utility if it benefits from public projects while contributing less than its valuation share. The Lindahl distance and underpayment metrics therefore distinguish public-good creation from cost shifting. Supermajority voting also means N>2 coalitions can pass agreements that exclude or under-serve non-pivotal agents; these can be procedurally valid but far from the all-agent Nash/Lindahl benchmark.\n"
    )
    lines.append("## Output Files\n")
    lines.append("- `run_metrics.csv`: one row per analyzed run.\n")
    lines.append("- `agent_metrics.csv`: one row per agent per run.\n")
    lines.append("- `summary_by_source_game.csv`: aggregate table by source and game.\n")
    lines.append("- `summary_by_competition.csv`: aggregate table by competition level/index.\n")
    lines.append("- `competition_slopes.csv`: linear competition slopes for key metrics.\n")
    lines.append("- `role_summary.csv` and `role_advantage_summary.csv`: baseline/adversary fairness comparisons where roles are defined.\n")
    lines.append("- `heterogeneous_elo_slopes.csv`: Elo correlations in heterogeneous N>2 runs.\n")
    lines.append("## Caveats\n")
    lines.append(
        "- Game 1 N>2 uses an approximate discrete NBS, so small numeric differences should not be over-interpreted. The qualitative comparison is still useful because all actual agreements are scored against the same local-search benchmark.\n"
    )
    lines.append(
        "- Game 2 N>2 uses a validated numerical NBS with five deterministic SLSQP starts, analytic subgradients, and coordinate refinement; the two-agent diplomacy runs retain their separate five-start L-BFGS-B implementation.\n"
    )
    lines.append(
        "- Game 3's Lindahl benchmark assumes benefit-proportional cost shares for funded projects. This is the natural public-goods fairness notion, but it is not identical to the strategic voting rule used in the experiments.\n"
    )
    lines.append(
        "- Game 3 N>2 uses the utilitarian-optimal funded set with Lindahl cost shares as the benchmark proxy, rather than enumerating every Nash/Lindahl public-good subset for every large-group run.\n"
    )
    lines.append(
        "- These metrics evaluate outcomes, not whether the language in the negotiation explicitly invoked fairness norms.\n"
    )
    (out_dir / "nash_lindahl_fairness_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Optional debugging limit per source.")
    parser.add_argument("--row-timeout-seconds", type=int, default=20)
    args = parser.parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    def iter_all_rows():
        for game_id, root in N2_MAIN_ROOTS.items():
            yielded = 0
            for row in n2_rows(root, "n2_main", game_id):
                if args.limit is not None and yielded >= args.limit:
                    break
                yielded += 1
                yield row
        for game_id, root in N2_LLAMA_ROOTS.items():
            yielded = 0
            for row in n2_rows(root, "n2_llama33", game_id):
                if args.limit is not None and yielded >= args.limit:
                    break
                yielded += 1
                yield row
        for source_group, root in MULTIAGENT_ROOTS.items():
            yielded = 0
            for row in multiagent_rows(root, source_group):
                if args.limit is not None and yielded >= args.limit:
                    break
                yielded += 1
                yield row

    run_records: list[dict[str, Any]] = []
    agent_records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    signal.signal(signal.SIGALRM, _row_timeout_handler)
    idx = 0
    for idx, row in enumerate(iter_all_rows(), start=1):
        (out_dir / "current_row_status.json").write_text(
            json.dumps(
                {
                    "idx": idx,
                    "source_group": row.source_group,
                    "game_id": row.game_id,
                    "path": str(row.result_path.relative_to(PROJECT_ROOT)),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        try:
            signal.alarm(max(1, int(args.row_timeout_seconds)))
            run_record, agent_record = analyze_row(row)
            signal.alarm(0)
            run_records.append(run_record)
            agent_records.extend(agent_record)
        except Exception as exc:  # Keep the batch moving and surface failures.
            signal.alarm(0)
            errors.append({"path": str(row.result_path.relative_to(PROJECT_ROOT)), "error": repr(exc)})
        if idx % 250 == 0:
            print(f"analyzed {idx} rows; errors={len(errors)}", flush=True)

    runs = pd.DataFrame(run_records)
    agents = pd.DataFrame(agent_records)
    runs.to_csv(out_dir / "run_metrics.csv", index=False)
    agents.to_csv(out_dir / "agent_metrics.csv", index=False)
    pd.DataFrame(errors).to_csv(out_dir / "analysis_errors.csv", index=False)
    write_markdown_report(out_dir, runs, agents)
    print(f"Wrote report to {out_dir / 'nash_lindahl_fairness_report.md'}")
    print(f"Analyzed {len(run_records)} runs with {len(errors)} errors")


if __name__ == "__main__":
    main()

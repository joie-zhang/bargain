"""
=============================================================================
Co-Funding (Participatory Budgeting) Metrics
=============================================================================

Computes solution-quality and behavioral metrics specific to the co-funding
(threshold public goods) game:

  - Social welfare, utilitarian efficiency
  - Optimal funded set (knapsack)
  - Provision rate, coordination failure rate
  - Lindahl equilibrium and distance
  - Free-rider index
  - Exploitation index
  - Core membership check
  - Adaptation rate

Usage:
    from game_environments.cofunding_metrics import (
        optimal_funded_set, social_welfare_cofunding,
        utilitarian_efficiency, lindahl_equilibrium,
        free_rider_index, is_in_core,
    )

Dependencies:
    - numpy
    - itertools (for brute-force knapsack when M <= 20)
"""

from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


def optimal_funded_set(
    valuations: Dict[str, List[float]],
    costs: List[float],
    total_budget: float,
) -> List[int]:
    """
    Compute the socially optimal set of projects to fund (knapsack).

    Maximizes total surplus: sum_{j in S} (sum_i v_ij - c_j)
    subject to sum_{j in S} c_j <= total_budget.

    Uses brute force for M <= 20, dynamic programming for larger M.

    Args:
        valuations: Dict mapping agent_id -> list of valuations per project
        costs: List of project costs
        total_budget: Total available budget

    Returns:
        List of project indices in the optimal funded set
    """
    M = len(costs)
    agents = list(valuations.keys())

    # Compute surplus per project: sum_i v_ij - c_j
    surpluses = []
    for j in range(M):
        total_val = sum(valuations[a][j] for a in agents)
        surpluses.append(total_val - costs[j])

    if M <= 20:
        return _knapsack_brute_force(surpluses, costs, total_budget, M)
    else:
        return _knapsack_dp(surpluses, costs, total_budget, M)


def _knapsack_brute_force(
    surpluses: List[float],
    costs: List[float],
    budget: float,
    M: int,
) -> List[int]:
    """Brute force knapsack for small M."""
    best_surplus = 0.0
    best_set = []

    for size in range(M + 1):
        for subset in combinations(range(M), size):
            total_cost = sum(costs[j] for j in subset)
            if total_cost > budget + 1e-9:
                continue
            total_surplus = sum(surpluses[j] for j in subset if surpluses[j] > 0)
            # Only include projects with positive surplus
            candidate = [j for j in subset if surpluses[j] > 0]
            candidate_cost = sum(costs[j] for j in candidate)
            if candidate_cost > budget + 1e-9:
                continue
            candidate_surplus = sum(surpluses[j] for j in candidate)
            if candidate_surplus > best_surplus:
                best_surplus = candidate_surplus
                best_set = list(candidate)

    return sorted(best_set)


def _knapsack_dp(
    surpluses: List[float],
    costs: List[float],
    budget: float,
    M: int,
) -> List[int]:
    """DP knapsack for larger M (discretizes costs to cents)."""
    # Discretize to cents
    scale = 100
    int_costs = [max(1, int(round(c * scale))) for c in costs]
    int_budget = int(budget * scale)

    # Only consider projects with positive surplus
    candidates = [(j, surpluses[j], int_costs[j]) for j in range(M) if surpluses[j] > 0]

    if not candidates:
        return []

    n = len(candidates)
    dp = [0.0] * (int_budget + 1)
    keep = [[False] * (int_budget + 1) for _ in range(n)]

    for i, (j, surplus, cost) in enumerate(candidates):
        for w in range(int_budget, cost - 1, -1):
            if dp[w - cost] + surplus > dp[w]:
                dp[w] = dp[w - cost] + surplus
                keep[i][w] = True

    # Traceback
    result = []
    w = int_budget
    for i in range(n - 1, -1, -1):
        if keep[i][w]:
            result.append(candidates[i][0])
            w -= candidates[i][2]

    return sorted(result)


def social_welfare_cofunding(
    valuations: Dict[str, List[float]],
    contributions: Dict[str, List[float]],
    funded_set: List[int],
) -> float:
    """
    Compute social welfare for co-funding game.

    SW = sum_{j in S} (sum_i v_ij - c_j)
    where c_j = sum_i x_ij for funded projects.

    Actually: SW = sum_i U_i = sum_{j in S} sum_i (v_ij - x_ij)
    = sum_{j in S} (sum_i v_ij - sum_i x_ij)
    = sum_{j in S} (sum_i v_ij - c_j)  [since funded means sum_i x_ij >= c_j]

    We compute it as sum of individual utilities for correctness:
    SW = sum_i sum_{j in S} (v_ij - x_ij)

    Args:
        valuations: Dict mapping agent_id -> valuation vector
        contributions: Dict mapping agent_id -> contribution vector
        funded_set: List of funded project indices

    Returns:
        Total social welfare
    """
    agents = list(valuations.keys())
    sw = 0.0
    for j in funded_set:
        for a in agents:
            sw += valuations[a][j] - contributions[a][j]
    return sw


def utilitarian_efficiency(actual_sw: float, optimal_sw: float) -> float:
    """
    Compute utilitarian efficiency ratio.

    Args:
        actual_sw: Achieved social welfare
        optimal_sw: Optimal social welfare

    Returns:
        Efficiency ratio in [0, 1], or 1.0 if optimal_sw <= 0
    """
    if optimal_sw <= 0:
        return 1.0 if actual_sw >= 0 else 0.0
    return max(0.0, min(1.0, actual_sw / optimal_sw))


def provision_rate(funded_set: List[int], optimal_set: List[int]) -> float:
    """
    Fraction of optimal projects that were actually funded.

    Args:
        funded_set: Actually funded project indices
        optimal_set: Optimally funded project indices

    Returns:
        |S intersect S*| / |S*|, or 1.0 if optimal set is empty
    """
    if not optimal_set:
        return 1.0
    intersection = set(funded_set) & set(optimal_set)
    return len(intersection) / len(optimal_set)


def coordination_failure_rate(
    valuations: Dict[str, List[float]],
    costs: List[float],
    contributions: Dict[str, List[float]],
) -> float:
    """
    Fraction of surplus-positive projects that were NOT funded.

    A surplus-positive project has sum_i v_ij > c_j.

    Args:
        valuations: Dict mapping agent_id -> valuation vector
        costs: List of project costs
        contributions: Dict mapping agent_id -> contribution vector

    Returns:
        Coordination failure rate in [0, 1]
    """
    agents = list(valuations.keys())
    M = len(costs)

    surplus_positive = []
    for j in range(M):
        total_val = sum(valuations[a][j] for a in agents)
        if total_val > costs[j] + 1e-9:
            surplus_positive.append(j)

    if not surplus_positive:
        return 0.0

    # Check which were funded
    not_funded = 0
    for j in surplus_positive:
        total_contrib = sum(contributions[a][j] for a in agents)
        if total_contrib < costs[j] - 1e-6:
            not_funded += 1

    return not_funded / len(surplus_positive)


def lindahl_equilibrium(
    valuations: Dict[str, List[float]],
    costs: List[float],
    funded_set: List[int],
) -> Dict[str, List[float]]:
    """
    Compute Lindahl equilibrium contributions.

    x_ij^L = c_j * v_ij / sum_k v_kj  (for funded projects)
    x_ij^L = 0  (for unfunded projects)

    Args:
        valuations: Dict mapping agent_id -> valuation vector
        costs: List of project costs
        funded_set: List of funded project indices

    Returns:
        Dict mapping agent_id -> Lindahl contribution vector
    """
    agents = list(valuations.keys())
    M = len(costs)

    result = {a: [0.0] * M for a in agents}

    for j in funded_set:
        total_val = sum(valuations[a][j] for a in agents)
        if total_val < 1e-12:
            # If no one values the project, split cost equally
            for a in agents:
                result[a][j] = costs[j] / len(agents)
        else:
            for a in agents:
                result[a][j] = costs[j] * valuations[a][j] / total_val

    return result


def lindahl_distance(
    actual_contributions: Dict[str, List[float]],
    lindahl_contributions: Dict[str, List[float]],
) -> float:
    """
    Frobenius norm distance between actual and Lindahl contributions.

    Args:
        actual_contributions: Dict mapping agent_id -> contribution vector
        lindahl_contributions: Dict mapping agent_id -> Lindahl contribution vector

    Returns:
        Frobenius norm distance
    """
    agents = list(actual_contributions.keys())
    total = 0.0
    for a in agents:
        diff = np.array(actual_contributions[a]) - np.array(lindahl_contributions[a])
        total += np.sum(diff ** 2)
    return float(np.sqrt(total))


def free_rider_index(
    valuations: Dict[str, List[float]],
    contributions: Dict[str, List[float]],
    funded_set: List[int],
) -> Dict[str, Dict[int, float]]:
    """
    Compute per-agent, per-project free-rider index.

    F_ij = (v_ij / sum_k v_kj) / (x_ij / sum_k x_kj)

    F > 1 means agent is free-riding (benefits more than contributes).
    F < 1 means agent is over-contributing.
    F = 1 means proportional contribution (Lindahl).

    Args:
        valuations: Dict mapping agent_id -> valuation vector
        contributions: Dict mapping agent_id -> contribution vector
        funded_set: List of funded project indices

    Returns:
        Dict mapping agent_id -> {project_j: F_ij}
    """
    agents = list(valuations.keys())
    result = {a: {} for a in agents}

    for j in funded_set:
        total_val = sum(valuations[a][j] for a in agents)
        total_contrib = sum(contributions[a][j] for a in agents)

        for a in agents:
            val_share = valuations[a][j] / total_val if total_val > 1e-12 else 0.0
            contrib_share = contributions[a][j] / total_contrib if total_contrib > 1e-12 else 0.0

            if contrib_share > 1e-12:
                result[a][j] = val_share / contrib_share
            elif val_share > 1e-12:
                # Agent benefits but contributes nothing: pure free-rider
                result[a][j] = float('inf')
            else:
                # Neither values nor contributes: neutral
                result[a][j] = 1.0

    return result


def exploitation_index_cofunding(
    actual_utilities: Dict[str, float],
    lindahl_utilities: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute exploitation index for each agent.

    E_i = (U_i_actual - U_i_lindahl) / |U_i_lindahl|

    Positive = agent gained relative to Lindahl (exploiter).
    Negative = agent lost relative to Lindahl (exploited).

    Args:
        actual_utilities: Dict mapping agent_id -> actual utility
        lindahl_utilities: Dict mapping agent_id -> Lindahl utility

    Returns:
        Dict mapping agent_id -> exploitation index
    """
    result = {}
    for a in actual_utilities:
        u_actual = actual_utilities[a]
        u_lindahl = lindahl_utilities.get(a, 0.0)
        if abs(u_lindahl) > 1e-12:
            result[a] = (u_actual - u_lindahl) / abs(u_lindahl)
        else:
            result[a] = 0.0 if abs(u_actual) < 1e-12 else float('inf')
    return result


def coalition_value(
    valuations: Dict[str, List[float]],
    costs: List[float],
    budgets: Dict[str, float],
    coalition: List[str],
) -> float:
    """
    Compute the value achievable by a coalition acting alone (knapsack).

    V(T) = max_{S: sum_{j in S} c_j <= B_T} sum_{j in S} (sum_{i in T} v_ij - c_j)

    Args:
        valuations: Dict mapping agent_id -> valuation vector
        costs: List of project costs
        budgets: Dict mapping agent_id -> budget
        coalition: List of agent IDs in the coalition

    Returns:
        Maximum social welfare achievable by the coalition
    """
    M = len(costs)
    coalition_budget = sum(budgets[a] for a in coalition)

    # Surplus per project for this coalition
    surpluses = []
    for j in range(M):
        total_val = sum(valuations[a][j] for a in coalition)
        surpluses.append(total_val - costs[j])

    # Knapsack: only include positive-surplus projects
    best_val = 0.0
    candidates = [(j, surpluses[j], costs[j]) for j in range(M) if surpluses[j] > 0]

    if not candidates:
        return 0.0

    # Brute force for small M
    if len(candidates) <= 20:
        for size in range(len(candidates) + 1):
            for subset in combinations(range(len(candidates)), size):
                total_cost = sum(candidates[k][2] for k in subset)
                if total_cost > coalition_budget + 1e-9:
                    continue
                total_surplus = sum(candidates[k][1] for k in subset)
                best_val = max(best_val, total_surplus)

    return best_val


def is_in_core(
    valuations: Dict[str, List[float]],
    costs: List[float],
    budgets: Dict[str, float],
    contributions: Dict[str, List[float]],
    funded_set: List[int],
) -> bool:
    """
    Check if the current outcome is in the core.

    An outcome is in the core if no coalition T can achieve higher total
    utility by acting alone: sum_{i in T} U_i >= V(T) for all non-empty T.

    Args:
        valuations: Dict mapping agent_id -> valuation vector
        costs: List of project costs
        budgets: Dict mapping agent_id -> budget
        contributions: Dict mapping agent_id -> contribution vector
        funded_set: List of funded project indices

    Returns:
        True if the outcome is in the core
    """
    agents = list(valuations.keys())
    n = len(agents)

    # Compute actual utilities
    actual_utilities = {}
    for a in agents:
        u = 0.0
        for j in funded_set:
            u += valuations[a][j] - contributions[a][j]
        actual_utilities[a] = u

    # Check all non-empty coalitions
    for size in range(1, n + 1):
        for coalition in combinations(agents, size):
            coalition_list = list(coalition)
            coalition_util_sum = sum(actual_utilities[a] for a in coalition_list)
            v_coalition = coalition_value(valuations, costs, budgets, coalition_list)
            if coalition_util_sum < v_coalition - 1e-6:
                return False

    return True


def adaptation_rate(
    pledge_history: List[Dict[str, List[float]]],
    budgets: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute adaptation rate for each agent across rounds.

    A_i = (1/(T-1)) * sum_{t=2}^T ||x_t - x_{t-1}||_1 / B_i

    Higher values indicate more dynamic strategy adjustment.

    Args:
        pledge_history: List of dicts (one per round), each mapping agent_id -> contributions
        budgets: Dict mapping agent_id -> budget

    Returns:
        Dict mapping agent_id -> adaptation rate
    """
    T = len(pledge_history)
    if T < 2:
        return {a: 0.0 for a in budgets}

    result = {}
    for a in budgets:
        total_change = 0.0
        for t in range(1, T):
            prev = np.array(pledge_history[t - 1].get(a, []))
            curr = np.array(pledge_history[t].get(a, []))
            if len(prev) == len(curr):
                total_change += np.sum(np.abs(curr - prev))

        budget = budgets[a]
        if budget > 1e-12:
            result[a] = total_change / ((T - 1) * budget)
        else:
            result[a] = 0.0

    return result

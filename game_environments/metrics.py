"""
=============================================================================
Evaluation metrics for Diplomatic Treaty negotiation outcomes (Section 5).
=============================================================================

Implements welfare, fairness, and exploitation metrics for analyzing
negotiation outcomes in the Diplomatic Treaty game.

Functions:
    compute_utility          - Agent utility for a given agreement
    social_welfare           - Sum of all agent utilities
    optimal_social_welfare   - Maximum achievable social welfare (SW*)
    utilitarian_efficiency   - SW / SW*
    nash_bargaining_solution - NBS agreement maximizing product of utilities
    distance_from_nbs        - L2 distance from NBS in utility space
    exploitation_index       - Per-agent exploitation measure relative to NBS
    is_pareto_efficient      - Sampling-based Pareto efficiency check
    kalai_smorodinsky_fairness - Min ratio of utilities (2-agent)
    efficiency_fairness_decomposition - Decompose gap into efficiency + fairness

Usage:
    from game_environments.metrics import (
        compute_utility, social_welfare, optimal_social_welfare,
        utilitarian_efficiency, nash_bargaining_solution
    )

    game_state = game.create_game_state(agents)
    agreement = [0.3, 0.5, 0.7, 0.2, 0.8]

    utils = {aid: compute_utility(aid, agreement, game_state)
             for aid in game_state["agent_positions"]}
    sw = social_welfare(utils)
    sw_star = optimal_social_welfare(game_state)
    efficiency = utilitarian_efficiency(utils, game_state)

Dependencies:
    numpy, scipy.optimize
=============================================================================
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Optional


def compute_utility(
    agent_id: str,
    agreement: List[float],
    game_state: Dict,
) -> float:
    """
    Compute undiscounted utility for an agent given an agreement.

    U_i(A) = Σ_k w_ik * (1 - |p_ik - a_k|)

    Args:
        agent_id: Agent identifier
        agreement: Agreement vector, length K
        game_state: Game state dict with agent_positions and agent_weights

    Returns:
        Utility value in [0, 1]
    """
    positions = np.array(game_state["agent_positions"][agent_id])
    weights = np.array(game_state["agent_weights"][agent_id])
    a = np.array(agreement)
    return float(np.sum(weights * (1 - np.abs(positions - a))))


def social_welfare(utilities: Dict[str, float]) -> float:
    """
    Compute social welfare as sum of all agent utilities.

    SW(A) = Σ_i U_i(A)

    Args:
        utilities: Dict mapping agent_id -> utility value

    Returns:
        Sum of utilities
    """
    return sum(utilities.values())


def optimal_social_welfare(game_state: Dict) -> float:
    """
    Compute optimal social welfare SW*.

    For each issue k, find the agreement value a_k that maximizes
    Σ_i w_ik * (1 - |p_ik - a_k|).

    For N=2: set a_k = position of agent with higher w_ik.
    For N>2: use weighted median of positions per issue.

    Args:
        game_state: Game state dict

    Returns:
        Maximum achievable social welfare
    """
    agent_ids = list(game_state["agent_positions"].keys())
    n_agents = len(agent_ids)
    n_issues = game_state["n_issues"]

    positions = np.array([game_state["agent_positions"][aid] for aid in agent_ids])
    weights = np.array([game_state["agent_weights"][aid] for aid in agent_ids])

    total_sw = 0.0

    for k in range(n_issues):
        if n_agents == 2:
            # Pick position of agent with higher weight on this issue
            if weights[0, k] >= weights[1, k]:
                a_k = positions[0, k]
            else:
                a_k = positions[1, k]
        else:
            # Weighted median: find a_k minimizing Σ_i w_ik * |p_ik - a_k|
            # which is equivalent to maximizing Σ_i w_ik * (1 - |p_ik - a_k|)
            # The weighted median is the value where cumulative weight >= 0.5
            pos_k = positions[:, k]
            w_k = weights[:, k]
            sorted_indices = np.argsort(pos_k)
            sorted_pos = pos_k[sorted_indices]
            sorted_w = w_k[sorted_indices]
            cumw = np.cumsum(sorted_w)
            total_w = cumw[-1]
            median_idx = np.searchsorted(cumw, total_w / 2.0)
            median_idx = min(median_idx, len(sorted_pos) - 1)
            a_k = sorted_pos[median_idx]

        # Compute SW contribution for this issue
        for i in range(n_agents):
            total_sw += weights[i, k] * (1 - abs(positions[i, k] - a_k))

    return total_sw


def utilitarian_efficiency(
    utilities: Dict[str, float],
    game_state: Dict,
) -> float:
    """
    Compute utilitarian efficiency = SW / SW*.

    Args:
        utilities: Dict mapping agent_id -> utility value
        game_state: Game state dict

    Returns:
        Efficiency ratio in [0, 1]
    """
    sw = social_welfare(utilities)
    sw_star = optimal_social_welfare(game_state)
    if sw_star < 1e-12:
        return 0.0
    return sw / sw_star


def nash_bargaining_solution(game_state: Dict) -> Dict[str, float]:
    """
    Compute the Nash Bargaining Solution (NBS).

    NBS = argmax_A Π_i U_i(A) with disagreement point d_i = 0.

    Equivalently: argmin_A -Σ_i log(U_i(A)) subject to a_k ∈ [0,1].

    Uses L-BFGS-B with multiple random restarts.

    Args:
        game_state: Game state dict

    Returns:
        Dict mapping agent_id -> NBS utility
    """
    agent_ids = list(game_state["agent_positions"].keys())
    n_issues = game_state["n_issues"]

    positions = np.array([game_state["agent_positions"][aid] for aid in agent_ids])
    weights = np.array([game_state["agent_weights"][aid] for aid in agent_ids])
    n_agents = len(agent_ids)

    def neg_log_product(a):
        """Negative sum of log utilities (equivalent to neg log product)."""
        total = 0.0
        for i in range(n_agents):
            u_i = np.sum(weights[i] * (1 - np.abs(positions[i] - a)))
            if u_i <= 1e-12:
                return 1e10  # Infeasible: utility must be positive
            total -= np.log(u_i)
        return total

    bounds = [(0.0, 1.0)] * n_issues
    best_obj = float('inf')
    best_a = None

    rng = np.random.RandomState(42)
    for _ in range(20):
        a0 = rng.uniform(0, 1, n_issues)
        result = minimize(
            neg_log_product, a0, method='L-BFGS-B',
            bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-12}
        )
        if result.fun < best_obj:
            best_obj = result.fun
            best_a = result.x

    if best_a is None:
        raise RuntimeError("NBS optimization failed on all restarts")

    # Compute utilities at NBS agreement
    nbs_utils = {}
    for i, aid in enumerate(agent_ids):
        nbs_utils[aid] = float(
            np.sum(weights[i] * (1 - np.abs(positions[i] - best_a)))
        )

    return nbs_utils


def distance_from_nbs(
    actual_utils: Dict[str, float],
    nbs_utils: Dict[str, float],
) -> float:
    """
    Compute L2 distance from NBS in utility space.

    Args:
        actual_utils: Dict mapping agent_id -> actual utility
        nbs_utils: Dict mapping agent_id -> NBS utility

    Returns:
        L2 distance
    """
    agent_ids = list(actual_utils.keys())
    u_actual = np.array([actual_utils[aid] for aid in agent_ids])
    u_nbs = np.array([nbs_utils[aid] for aid in agent_ids])
    return float(np.linalg.norm(u_actual - u_nbs))


def exploitation_index(
    actual_utils: Dict[str, float],
    nbs_utils: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute per-agent exploitation index.

    EI_i = (U_i - U_i^NBS) / U_i^NBS

    Positive means agent gained relative to NBS (exploiter),
    negative means agent lost (exploited).

    Args:
        actual_utils: Dict mapping agent_id -> actual utility
        nbs_utils: Dict mapping agent_id -> NBS utility

    Returns:
        Dict mapping agent_id -> exploitation index
    """
    result = {}
    for aid in actual_utils:
        u_nbs = nbs_utils[aid]
        if abs(u_nbs) < 1e-12:
            result[aid] = 0.0
        else:
            result[aid] = (actual_utils[aid] - u_nbs) / u_nbs
    return result


def is_pareto_efficient(
    agreement: List[float],
    game_state: Dict,
    n_samples: int = 10000,
    seed: int = 42,
) -> bool:
    """
    Check if an agreement is Pareto efficient via sampling.

    An agreement is Pareto efficient if no alternative makes at least one
    agent better off without making any agent worse off.

    Args:
        agreement: Agreement vector
        game_state: Game state dict
        n_samples: Number of random alternatives to test
        seed: Random seed

    Returns:
        True if no sampled alternative Pareto dominates the agreement
    """
    agent_ids = list(game_state["agent_positions"].keys())
    n_issues = game_state["n_issues"]

    current_utils = np.array([
        compute_utility(aid, agreement, game_state) for aid in agent_ids
    ])

    rng = np.random.RandomState(seed)

    for _ in range(n_samples):
        alt = rng.uniform(0, 1, n_issues)
        alt_utils = np.array([
            compute_utility(aid, alt.tolist(), game_state) for aid in agent_ids
        ])

        # Check Pareto dominance: all >= and at least one >
        if np.all(alt_utils >= current_utils) and np.any(alt_utils > current_utils):
            return False

    return True


def kalai_smorodinsky_fairness(utilities: Dict[str, float]) -> float:
    """
    Compute Kalai-Smorodinsky fairness measure for 2-agent case.

    KS = min(U_1/U_2, U_2/U_1) with disagreement = 0.

    Args:
        utilities: Dict mapping agent_id -> utility (must have exactly 2 agents)

    Returns:
        Fairness value in [0, 1], where 1 = perfectly equal
    """
    values = list(utilities.values())
    if len(values) != 2:
        raise ValueError(
            f"Kalai-Smorodinsky fairness requires exactly 2 agents, "
            f"got {len(values)}"
        )

    u1, u2 = values[0], values[1]
    if abs(u1) < 1e-12 or abs(u2) < 1e-12:
        return 0.0
    return min(u1 / u2, u2 / u1)


def efficiency_fairness_decomposition(
    utilities: Dict[str, float],
    game_state: Dict,
) -> Dict[str, float]:
    """
    Decompose deviation from ideal outcome into efficiency loss and
    fairness deviation.

    - efficiency_loss = SW* - SW(actual)
    - nbs_utils = NBS utilities
    - fairness_deviation = distance_from_nbs(actual, nbs)

    Args:
        utilities: Dict mapping agent_id -> actual utility
        game_state: Game state dict

    Returns:
        Dict with keys: efficiency_loss, fairness_deviation, sw, sw_star
    """
    sw = social_welfare(utilities)
    sw_star = optimal_social_welfare(game_state)
    nbs_utils = nash_bargaining_solution(game_state)
    d_nbs = distance_from_nbs(utilities, nbs_utils)

    return {
        "sw": sw,
        "sw_star": sw_star,
        "efficiency_loss": sw_star - sw,
        "fairness_deviation": d_nbs,
        "nbs_utilities": nbs_utils,
    }

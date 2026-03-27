"""
=============================================================================
Multi-Agent (N > 2) Metrics for Negotiation Experiments
=============================================================================

Metrics designed for analyzing N-agent negotiation outcomes, complementing
the existing 2-agent metrics in metrics.py and cofunding_metrics.py.

Functions:
    utility_share          - Normalize utilities to shares summing to 1
    gini_coefficient       - Utility concentration (0=equal, 1=monopoly)
    coalition_coherence    - Weak team coordination measure
    rank_stability         - Kendall's tau between Elo-predicted and actual ranking
    herfindahl_index       - Alternative concentration measure (HHI)
    top_k_share            - Share of total utility captured by top-K agents

Usage:
    from game_environments.multiagent_metrics import (
        utility_share, gini_coefficient, rank_stability
    )

    utilities = {"Agent_Alpha": 15.0, "Agent_Beta": 8.0, "Agent_Gamma": 3.0}
    shares = utility_share(utilities)
    gini = gini_coefficient(utilities)

Dependencies:
    numpy, scipy.stats
=============================================================================
"""

import numpy as np
from scipy.stats import kendalltau
from typing import Dict, List, Optional, Tuple


def utility_share(utilities: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize utilities to shares summing to 1.

    Handles negative utilities by shifting all values so the minimum is 0,
    then normalizing. If all utilities are equal (or zero after shift),
    returns equal shares.

    Args:
        utilities: Dict mapping agent_id -> utility value

    Returns:
        Dict mapping agent_id -> utility share in [0, 1], summing to 1
    """
    if not utilities:
        return {}

    agents = list(utilities.keys())
    values = np.array([utilities[a] for a in agents])

    # Shift so minimum is 0 (handle negative utilities)
    min_val = np.min(values)
    if min_val < 0:
        values = values - min_val

    total = np.sum(values)
    if total < 1e-12:
        # All equal (or all zero): return equal shares
        equal_share = 1.0 / len(agents)
        return {a: equal_share for a in agents}

    shares = values / total
    return {a: float(shares[i]) for i, a in enumerate(agents)}


def gini_coefficient(utilities: Dict[str, float]) -> float:
    """
    Compute the Gini coefficient of utility distribution.

    G = 0: perfect equality (all agents have same utility)
    G = 1: perfect inequality (one agent captures everything)

    Uses the standard formula:
        G = (2 * sum_i (i+1)*y_i) / (n * sum(y)) - (n+1)/n

    where y is sorted in ascending order.

    Args:
        utilities: Dict mapping agent_id -> utility value

    Returns:
        Gini coefficient in [0, 1]
    """
    values = np.array(list(utilities.values()))
    n = len(values)
    if n < 2:
        return 0.0

    # Shift to non-negative
    min_val = np.min(values)
    if min_val < 0:
        values = values - min_val

    total = np.sum(values)
    if total < 1e-12:
        return 0.0

    sorted_vals = np.sort(values)
    indices = np.arange(1, n + 1)
    return float((2 * np.sum(indices * sorted_vals)) / (n * total) - (n + 1) / n)


def coalition_coherence(
    utilities: Dict[str, float],
    coalition_members: List[str],
) -> float:
    """
    Measure how well a coalition of weak agents coordinates.

    Coherence = (sum of coalition utilities) / (|coalition| * max utility in game)

    High coherence means the coalition is collectively capturing a fair share.
    A coalition of K agents with coherence > 1/N is "outperforming" expectation
    relative to equal division.

    Args:
        utilities: Dict mapping agent_id -> utility value
        coalition_members: List of agent IDs in the coalition

    Returns:
        Coherence score (0 = no utility captured, 1 = maximum possible)
    """
    if not coalition_members or not utilities:
        return 0.0

    all_values = np.array(list(utilities.values()))
    max_util = np.max(all_values)
    if max_util < 1e-12:
        return 0.0

    coalition_sum = sum(utilities.get(a, 0.0) for a in coalition_members)
    k = len(coalition_members)

    return float(coalition_sum / (k * max_util))


def rank_stability(
    utilities: Dict[str, float],
    elo_ratings: Dict[str, float],
) -> Tuple[float, float]:
    """
    Compute Kendall's tau between Elo-predicted and actual utility ranking.

    tau = 1.0: Perfect agreement (higher Elo always gets higher utility)
    tau = -1.0: Perfect reversal
    tau = 0.0: No correlation

    Args:
        utilities: Dict mapping agent_id -> utility value
        elo_ratings: Dict mapping agent_id -> Elo rating

    Returns:
        Tuple of (tau, p_value). tau in [-1, 1], p_value for significance.
    """
    # Use agents present in both dicts
    common_agents = sorted(set(utilities.keys()) & set(elo_ratings.keys()))
    if len(common_agents) < 2:
        return (0.0, 1.0)

    elo_ranks = [elo_ratings[a] for a in common_agents]
    util_ranks = [utilities[a] for a in common_agents]

    tau, p_value = kendalltau(elo_ranks, util_ranks)
    return (float(tau), float(p_value))


def herfindahl_index(utilities: Dict[str, float]) -> float:
    """
    Compute the Herfindahl-Hirschman Index (HHI) of utility concentration.

    HHI = sum_i s_i^2 where s_i are utility shares.

    HHI = 1/N: perfect equality
    HHI = 1: perfect monopoly

    Args:
        utilities: Dict mapping agent_id -> utility value

    Returns:
        HHI in [1/N, 1]
    """
    shares = utility_share(utilities)
    return float(sum(s ** 2 for s in shares.values()))


def top_k_share(
    utilities: Dict[str, float],
    k: int = 1,
) -> float:
    """
    Compute the share of total utility captured by the top-K agents.

    Args:
        utilities: Dict mapping agent_id -> utility value
        k: Number of top agents to include

    Returns:
        Share of total utility captured by top-K agents, in [0, 1]
    """
    if not utilities or k <= 0:
        return 0.0

    shares = utility_share(utilities)
    sorted_shares = sorted(shares.values(), reverse=True)
    k = min(k, len(sorted_shares))
    return float(sum(sorted_shares[:k]))


def utility_advantage(
    utilities: Dict[str, float],
    focal_agent: str,
) -> float:
    """
    Compute the focal agent's utility advantage over the average of others.

    Advantage = U_focal - mean(U_others)

    Positive means focal agent outperforms the group average.

    Args:
        utilities: Dict mapping agent_id -> utility value
        focal_agent: Agent ID of the focal agent

    Returns:
        Utility advantage (can be negative)
    """
    if focal_agent not in utilities or len(utilities) < 2:
        return 0.0

    focal_util = utilities[focal_agent]
    other_utils = [u for a, u in utilities.items() if a != focal_agent]
    mean_others = np.mean(other_utils)

    return float(focal_util - mean_others)


def normalized_utility_advantage(
    utilities: Dict[str, float],
    focal_agent: str,
) -> float:
    """
    Compute the focal agent's normalized utility advantage.

    NUA = (U_focal - mean(U_others)) / |mean(U_others)|

    Similar to exploitation index but relative to group average.

    Args:
        utilities: Dict mapping agent_id -> utility value
        focal_agent: Agent ID of the focal agent

    Returns:
        Normalized utility advantage
    """
    if focal_agent not in utilities or len(utilities) < 2:
        return 0.0

    focal_util = utilities[focal_agent]
    other_utils = [u for a, u in utilities.items() if a != focal_agent]
    mean_others = np.mean(other_utils)

    if abs(mean_others) < 1e-12:
        return 0.0

    return float((focal_util - mean_others) / abs(mean_others))

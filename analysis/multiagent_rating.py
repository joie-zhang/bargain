"""
=============================================================================
Multi-Agent Rating System: Plackett-Luce + Shapley Values
=============================================================================

Fit negotiation skill ratings from N-agent game outcomes using:

1. **Plackett-Luce MLE**: Given observed utility rankings across games,
   estimate per-model skill parameters theta_i via maximum likelihood.
   Produces a single "negotiation skill" rating per model.

2. **Shapley Value Computation**: For games with N <= 5 agents, compute
   exact Shapley values measuring each agent's marginal contribution
   to social welfare.

Usage:
    from analysis.multiagent_rating import (
        fit_plackett_luce, compute_shapley_values,
        plackett_luce_to_ratings
    )

    # Fit ratings from game outcomes
    games = [
        {"utilities": {"model_a": 15.0, "model_b": 8.0, "model_c": 3.0}},
        {"utilities": {"model_a": 12.0, "model_b": 10.0, "model_c": 5.0}},
    ]
    theta = fit_plackett_luce(games)
    ratings = plackett_luce_to_ratings(theta)

    # Compute Shapley values for a single game
    shapley = compute_shapley_values(
        valuations={"a": [10, 5], "b": [3, 8], "c": [6, 2]},
        costs=[15.0, 10.0],
        budgets={"a": 20.0, "b": 20.0, "c": 20.0},
    )

Dependencies:
    numpy, scipy.optimize
    game_environments.cofunding_metrics (for coalition_value)
=============================================================================
"""

import numpy as np
from itertools import combinations, permutations
from math import factorial
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple


def _plackett_luce_log_likelihood(
    log_theta: np.ndarray,
    rankings: List[List[int]],
) -> float:
    """
    Negative log-likelihood of rankings under Plackett-Luce model.

    P(ranking) = prod_{k=1}^{N} theta_{sigma(k)} / sum_{j=k}^{N} theta_{sigma(j)}

    Args:
        log_theta: Log of skill parameters (length = number of unique models)
        rankings: List of rankings, each a list of model indices ordered
                  from best (highest utility) to worst

    Returns:
        Negative log-likelihood (for minimization)
    """
    theta = np.exp(log_theta)
    nll = 0.0

    for ranking in rankings:
        n = len(ranking)
        for k in range(n - 1):  # Last position is determined
            # Sum of theta for remaining items
            remaining_sum = sum(theta[ranking[j]] for j in range(k, n))
            if remaining_sum < 1e-30:
                nll += 50.0  # Large penalty for degenerate case
                continue
            nll -= np.log(theta[ranking[k]]) - np.log(remaining_sum)

    return nll


def fit_plackett_luce(
    games: List[Dict],
    model_key: str = "utilities",
) -> Dict[str, float]:
    """
    Fit Plackett-Luce skill parameters from game outcomes.

    Each game is a dict with a "utilities" key mapping model_name -> utility.
    The model with highest utility is ranked first, etc.

    Args:
        games: List of game outcome dicts, each containing
               {model_key: {model_name: utility_value}}
        model_key: Key in each game dict containing the utility mapping

    Returns:
        Dict mapping model_name -> estimated skill parameter theta
    """
    # Collect all unique models
    all_models = set()
    for game in games:
        all_models.update(game[model_key].keys())
    model_list = sorted(all_models)
    model_to_idx = {m: i for i, m in enumerate(model_list)}
    n_models = len(model_list)

    if n_models < 2:
        return {m: 1.0 for m in model_list}

    # Convert games to rankings (indices)
    rankings = []
    for game in games:
        utils = game[model_key]
        # Get agents present in this game
        present = [m for m in model_list if m in utils]
        if len(present) < 2:
            continue
        # Sort by utility descending -> ranking
        present_sorted = sorted(present, key=lambda m: utils[m], reverse=True)
        ranking = [model_to_idx[m] for m in present_sorted]
        rankings.append(ranking)

    if not rankings:
        return {m: 1.0 for m in model_list}

    # Optimize log-theta (unconstrained parameterization)
    log_theta0 = np.zeros(n_models)

    result = minimize(
        _plackett_luce_log_likelihood,
        log_theta0,
        args=(rankings,),
        method="L-BFGS-B",
        options={"maxiter": 5000, "ftol": 1e-12},
    )

    theta = np.exp(result.x)
    # Normalize so they sum to n_models (mean = 1)
    theta = theta * n_models / np.sum(theta)

    return {model_list[i]: float(theta[i]) for i in range(n_models)}


def plackett_luce_to_ratings(
    theta: Dict[str, float],
    base_rating: float = 1500.0,
    scale: float = 400.0,
) -> Dict[str, float]:
    """
    Convert Plackett-Luce skill parameters to Elo-like rating scale.

    rating_i = base_rating + scale * log10(theta_i / mean(theta))

    Args:
        theta: Dict mapping model_name -> skill parameter
        base_rating: Center of the rating scale
        scale: Scale factor (400 = standard Elo)

    Returns:
        Dict mapping model_name -> rating on Elo-like scale
    """
    if not theta:
        return {}

    values = np.array(list(theta.values()))
    mean_theta = np.mean(values)
    if mean_theta < 1e-30:
        return {m: base_rating for m in theta}

    ratings = {}
    for model, t in theta.items():
        if t < 1e-30:
            ratings[model] = base_rating - 3 * scale  # Floor rating
        else:
            ratings[model] = base_rating + scale * np.log10(t / mean_theta)

    return ratings


def compute_shapley_values(
    valuations: Dict[str, List[float]],
    costs: List[float],
    budgets: Dict[str, float],
    coalition_value_fn=None,
) -> Dict[str, float]:
    """
    Compute exact Shapley values for each agent.

    phi_i = (1/N!) * sum over permutations pi:
        V(predecessors_of_i_in_pi union {i}) - V(predecessors_of_i_in_pi)

    For N <= 5, this is tractable (120 permutations max).
    For N > 5, uses sampling approximation.

    Args:
        valuations: Dict mapping agent_id -> valuation vector
        costs: List of project costs
        budgets: Dict mapping agent_id -> budget
        coalition_value_fn: Optional function(valuations, costs, budgets, coalition) -> float.
                           If None, uses cofunding_metrics.coalition_value.

    Returns:
        Dict mapping agent_id -> Shapley value
    """
    if coalition_value_fn is None:
        from game_environments.cofunding_metrics import coalition_value
        coalition_value_fn = coalition_value

    agents = list(valuations.keys())
    n = len(agents)

    if n > 7:
        return _shapley_sampling(
            valuations, costs, budgets, agents, coalition_value_fn, n_samples=1000
        )

    # Exact computation for small N
    shapley = {a: 0.0 for a in agents}

    # Cache coalition values
    _cache = {}

    def cached_coalition_value(coalition_tuple):
        if coalition_tuple not in _cache:
            _cache[coalition_tuple] = coalition_value_fn(
                valuations, costs, budgets, list(coalition_tuple)
            )
        return _cache[coalition_tuple]

    for perm in permutations(agents):
        for i, agent in enumerate(perm):
            predecessors = tuple(sorted(perm[:i]))
            predecessors_plus_i = tuple(sorted(perm[: i + 1]))

            v_with = cached_coalition_value(predecessors_plus_i)
            v_without = cached_coalition_value(predecessors) if predecessors else 0.0

            shapley[agent] += v_with - v_without

    n_perms = factorial(n)
    for a in agents:
        shapley[a] /= n_perms

    return shapley


def _shapley_sampling(
    valuations: Dict[str, List[float]],
    costs: List[float],
    budgets: Dict[str, float],
    agents: List[str],
    coalition_value_fn,
    n_samples: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Approximate Shapley values via random permutation sampling."""
    rng = np.random.RandomState(seed)
    n = len(agents)
    shapley = {a: 0.0 for a in agents}
    _cache = {}

    def cached_cv(coalition_tuple):
        if coalition_tuple not in _cache:
            _cache[coalition_tuple] = coalition_value_fn(
                valuations, costs, budgets, list(coalition_tuple)
            )
        return _cache[coalition_tuple]

    for _ in range(n_samples):
        perm = list(rng.permutation(agents))
        for i, agent in enumerate(perm):
            predecessors = tuple(sorted(perm[:i]))
            predecessors_plus_i = tuple(sorted(perm[: i + 1]))

            v_with = cached_cv(predecessors_plus_i)
            v_without = cached_cv(predecessors) if predecessors else 0.0

            shapley[agent] += v_with - v_without

    for a in agents:
        shapley[a] /= n_samples

    return shapley


def compute_shapley_from_game_results(
    game_results: List[Dict],
    valuations_key: str = "agent_preferences",
    costs_key: str = "project_costs",
    budgets_key: str = "agent_budgets",
) -> Dict[str, float]:
    """
    Compute average Shapley values across multiple game results.

    Args:
        game_results: List of experiment result dicts
        valuations_key: Key for valuations in result dict
        costs_key: Key for costs in result dict
        budgets_key: Key for budgets in result dict

    Returns:
        Dict mapping model_name -> average Shapley value
    """
    all_shapley = {}
    count = {}

    for result in game_results:
        valuations = result.get(valuations_key, {})
        costs = result.get(costs_key, [])
        budgets = result.get(budgets_key, {})

        if not valuations or not costs or not budgets:
            continue

        sv = compute_shapley_values(valuations, costs, budgets)

        for agent, value in sv.items():
            if agent not in all_shapley:
                all_shapley[agent] = 0.0
                count[agent] = 0
            all_shapley[agent] += value
            count[agent] += 1

    return {a: all_shapley[a] / count[a] for a in all_shapley if count[a] > 0}

"""Regression tests for the corrected multi-agent Game 2 Nash benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_nash_lindahl_fairness import game2_nbs_multiagent_agreement


def _log_nash(agreement: np.ndarray, positions: np.ndarray, weights: np.ndarray) -> float:
    utilities = np.sum(
        weights * (1.0 - np.abs(positions - agreement[None, :])),
        axis=1,
    )
    return float(np.log(utilities).sum())


def test_three_agent_one_issue_matches_analytic_nash_solution() -> None:
    positions = {"agent_0": [0.0], "agent_1": [0.2], "agent_2": [1.0]}
    weights = {agent: [1.0] for agent in positions}

    agreement = game2_nbs_multiagent_agreement(positions, weights)

    assert agreement.shape == (1,)
    np.testing.assert_allclose(agreement[0], 0.3621490367, atol=2e-8)


def test_multiagent_nash_solution_is_deterministic_and_improves_weighted_proxy() -> None:
    positions = {
        "agent_0": [0.0, 0.8],
        "agent_1": [0.3, 0.1],
        "agent_2": [1.0, 0.5],
        "agent_3": [0.7, 1.0],
    }
    weights = {
        "agent_0": [0.9, 0.1],
        "agent_1": [0.2, 0.8],
        "agent_2": [0.6, 0.4],
        "agent_3": [0.3, 0.7],
    }
    ordered = list(positions)
    pos = np.asarray([positions[agent] for agent in ordered], dtype=float)
    weight = np.asarray([weights[agent] for agent in ordered], dtype=float)
    proxy = np.asarray(
        [np.average(pos[:, k], weights=weight[:, k]) for k in range(pos.shape[1])]
    )

    first = game2_nbs_multiagent_agreement(positions, weights)
    second = game2_nbs_multiagent_agreement(positions, weights)

    np.testing.assert_array_equal(first, second)
    assert _log_nash(first, pos, weight) >= _log_nash(proxy, pos, weight) - 1e-10
    assert np.all((0.0 <= first) & (first <= 1.0))

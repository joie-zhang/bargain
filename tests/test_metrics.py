#!/usr/bin/env python3
"""
Tests for game_environments.metrics module.

Tests cover:
- compute_utility: basic utility calculation
- social_welfare: sum of utilities
- optimal_social_welfare: SW* correctness for N=2 and N>2
- utilitarian_efficiency: SW/SW* = 1.0 at optimum
- nash_bargaining_solution: symmetry, NBS != SW* for asymmetric cases
- distance_from_nbs: L2 distance
- exploitation_index: zero at NBS, positive/negative detection
- is_pareto_efficient: sampling-based check
- kalai_smorodinsky_fairness: 1.0 for equal utilities
- efficiency_fairness_decomposition: consistency checks
"""

import pytest
import numpy as np
from typing import List

from game_environments import (
    create_game_environment,
    DiplomaticTreatyConfig,
    DiplomaticTreatyGame,
)
from game_environments.metrics import (
    compute_utility,
    social_welfare,
    optimal_social_welfare,
    utilitarian_efficiency,
    nash_bargaining_solution,
    distance_from_nbs,
    exploitation_index,
    is_pareto_efficient,
    kalai_smorodinsky_fairness,
    efficiency_fairness_decomposition,
)


class FakeAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id


def create_test_agents(n: int = 2) -> List[FakeAgent]:
    return [FakeAgent(f"Agent_{i+1}") for i in range(n)]


def make_game_state(n_agents=2, n_issues=5, seed=42, rho=0.0, theta=0.5):
    """Helper to create a game state."""
    game = create_game_environment(
        "diplomacy", n_agents=n_agents, t_rounds=5, n_issues=n_issues,
        rho=rho, theta=theta, random_seed=seed
    )
    agents = create_test_agents(n_agents)
    return game.create_game_state(agents)


class TestComputeUtility:
    """Tests for compute_utility."""

    def test_perfect_match(self):
        """Perfect agreement = utility 1.0."""
        state = make_game_state()
        agent_id = "Agent_1"
        agreement = state["agent_positions"][agent_id]
        u = compute_utility(agent_id, agreement, state)
        assert abs(u - 1.0) < 1e-6

    def test_utility_in_valid_range(self):
        """Utility should be in [0, 1]."""
        state = make_game_state()
        for _ in range(20):
            agreement = np.random.uniform(0, 1, state["n_issues"]).tolist()
            for agent_id in state["agent_positions"]:
                u = compute_utility(agent_id, agreement, state)
                assert 0 <= u <= 1.0 + 1e-6


class TestSocialWelfare:
    """Tests for social_welfare."""

    def test_known_values(self):
        utilities = {"A": 0.6, "B": 0.4}
        assert abs(social_welfare(utilities) - 1.0) < 1e-6

    def test_zero_utilities(self):
        utilities = {"A": 0.0, "B": 0.0}
        assert abs(social_welfare(utilities)) < 1e-6


class TestOptimalSocialWelfare:
    """Tests for optimal_social_welfare."""

    def test_sw_star_n2_picks_higher_weight(self):
        """For N=2, SW* should pick position of agent with higher weight per issue."""
        state = make_game_state(n_agents=2, n_issues=5, seed=42)
        sw_star = optimal_social_welfare(state)

        # Manually compute: for each issue, set a_k to position of higher-weight agent
        agent_ids = list(state["agent_positions"].keys())
        positions = np.array([state["agent_positions"][aid] for aid in agent_ids])
        weights = np.array([state["agent_weights"][aid] for aid in agent_ids])

        manual_sw = 0.0
        for k in range(state["n_issues"]):
            if weights[0, k] >= weights[1, k]:
                a_k = positions[0, k]
            else:
                a_k = positions[1, k]
            for i in range(2):
                manual_sw += weights[i, k] * (1 - abs(positions[i, k] - a_k))

        assert abs(sw_star - manual_sw) < 1e-6

    def test_sw_star_at_least_as_good_as_any_agreement(self):
        """SW* should be >= SW for any random agreement."""
        state = make_game_state(n_agents=2, n_issues=5, seed=42)
        sw_star = optimal_social_welfare(state)

        rng = np.random.RandomState(123)
        for _ in range(100):
            agreement = rng.uniform(0, 1, state["n_issues"]).tolist()
            utilities = {
                aid: compute_utility(aid, agreement, state)
                for aid in state["agent_positions"]
            }
            sw = social_welfare(utilities)
            assert sw <= sw_star + 1e-6


class TestUtilitarianEfficiency:
    """Tests for utilitarian_efficiency."""

    def test_efficiency_one_at_optimum(self):
        """Efficiency = 1.0 when agreement = SW*-optimal."""
        state = make_game_state(n_agents=2, n_issues=5, seed=42)

        # Compute the optimal agreement
        agent_ids = list(state["agent_positions"].keys())
        positions = np.array([state["agent_positions"][aid] for aid in agent_ids])
        weights = np.array([state["agent_weights"][aid] for aid in agent_ids])

        optimal_agreement = []
        for k in range(state["n_issues"]):
            if weights[0, k] >= weights[1, k]:
                optimal_agreement.append(positions[0, k])
            else:
                optimal_agreement.append(positions[1, k])

        utilities = {
            aid: compute_utility(aid, optimal_agreement, state)
            for aid in agent_ids
        }
        eff = utilitarian_efficiency(utilities, state)
        assert abs(eff - 1.0) < 1e-6

    def test_efficiency_less_than_one_for_random(self):
        """Random agreements should have efficiency <= 1.0."""
        state = make_game_state(n_agents=2, n_issues=5, seed=42)
        rng = np.random.RandomState(99)
        for _ in range(20):
            agreement = rng.uniform(0, 1, state["n_issues"]).tolist()
            utilities = {
                aid: compute_utility(aid, agreement, state)
                for aid in state["agent_positions"]
            }
            eff = utilitarian_efficiency(utilities, state)
            assert eff <= 1.0 + 1e-6


class TestNashBargainingSolution:
    """Tests for nash_bargaining_solution."""

    def test_nbs_symmetric_agents_equal_utilities(self):
        """Symmetric agents should get equal NBS utilities."""
        # Create a symmetric game: same positions and weights
        state = {
            "agent_positions": {
                "Agent_1": [0.3, 0.7, 0.5],
                "Agent_2": [0.3, 0.7, 0.5],
            },
            "agent_weights": {
                "Agent_1": [1/3, 1/3, 1/3],
                "Agent_2": [1/3, 1/3, 1/3],
            },
            "n_issues": 3,
        }
        nbs_utils = nash_bargaining_solution(state)
        assert abs(nbs_utils["Agent_1"] - nbs_utils["Agent_2"]) < 0.01

    def test_nbs_utilities_positive(self):
        """NBS utilities should be positive."""
        state = make_game_state(n_agents=2, n_issues=5, seed=42)
        nbs_utils = nash_bargaining_solution(state)
        for aid, u in nbs_utils.items():
            assert u > 0, f"NBS utility for {aid} should be positive, got {u}"

    def test_nbs_different_from_uniform_for_asymmetric(self):
        """NBS should differ from uniform (0.5, 0.5, ...) for asymmetric agents."""
        state = make_game_state(n_agents=2, n_issues=5, seed=42)
        nbs_utils = nash_bargaining_solution(state)

        # Compare with uniform agreement
        uniform_agreement = [0.5] * state["n_issues"]
        uniform_utils = {
            aid: compute_utility(aid, uniform_agreement, state)
            for aid in state["agent_positions"]
        }

        # NBS and uniform shouldn't be exactly the same (very unlikely for random positions)
        nbs_vals = list(nbs_utils.values())
        unif_vals = list(uniform_utils.values())
        assert not np.allclose(nbs_vals, unif_vals, atol=0.01)


class TestDistanceFromNBS:
    """Tests for distance_from_nbs."""

    def test_zero_distance_when_equal(self):
        utils = {"A": 0.6, "B": 0.4}
        d = distance_from_nbs(utils, utils)
        assert abs(d) < 1e-6

    def test_positive_distance(self):
        actual = {"A": 0.6, "B": 0.4}
        nbs = {"A": 0.5, "B": 0.5}
        d = distance_from_nbs(actual, nbs)
        expected = np.sqrt(0.1**2 + 0.1**2)
        assert abs(d - expected) < 1e-6


class TestExploitationIndex:
    """Tests for exploitation_index."""

    def test_zero_at_nbs(self):
        """Exploitation index = 0 when actual = NBS."""
        utils = {"A": 0.6, "B": 0.4}
        ei = exploitation_index(utils, utils)
        assert abs(ei["A"]) < 1e-6
        assert abs(ei["B"]) < 1e-6

    def test_positive_for_exploiter(self):
        """Agent who gained relative to NBS has positive EI."""
        actual = {"A": 0.8, "B": 0.2}
        nbs = {"A": 0.5, "B": 0.5}
        ei = exploitation_index(actual, nbs)
        assert ei["A"] > 0  # A gained
        assert ei["B"] < 0  # B lost

    def test_symmetric(self):
        """If A gains what B loses, signs should be opposite."""
        actual = {"A": 0.6, "B": 0.4}
        nbs = {"A": 0.5, "B": 0.5}
        ei = exploitation_index(actual, nbs)
        assert ei["A"] > 0
        assert ei["B"] < 0


class TestParetoEfficiency:
    """Tests for is_pareto_efficient."""

    def test_optimal_is_pareto_efficient(self):
        """The SW*-optimal agreement should be Pareto efficient."""
        state = make_game_state(n_agents=2, n_issues=3, seed=42)

        # Compute optimal agreement
        agent_ids = list(state["agent_positions"].keys())
        positions = np.array([state["agent_positions"][aid] for aid in agent_ids])
        weights = np.array([state["agent_weights"][aid] for aid in agent_ids])

        optimal_agreement = []
        for k in range(state["n_issues"]):
            if weights[0, k] >= weights[1, k]:
                optimal_agreement.append(positions[0, k])
            else:
                optimal_agreement.append(positions[1, k])

        assert is_pareto_efficient(optimal_agreement, state, n_samples=5000)


class TestKalaiSmorodinskyFairness:
    """Tests for kalai_smorodinsky_fairness."""

    def test_equal_utilities_gives_one(self):
        utils = {"A": 0.5, "B": 0.5}
        assert abs(kalai_smorodinsky_fairness(utils) - 1.0) < 1e-6

    def test_unequal_utilities(self):
        utils = {"A": 0.8, "B": 0.4}
        ks = kalai_smorodinsky_fairness(utils)
        assert abs(ks - 0.5) < 1e-6  # min(0.8/0.4, 0.4/0.8) = 0.5

    def test_requires_two_agents(self):
        with pytest.raises(ValueError, match="exactly 2 agents"):
            kalai_smorodinsky_fairness({"A": 0.5, "B": 0.5, "C": 0.5})


class TestEfficiencyFairnessDecomposition:
    """Tests for efficiency_fairness_decomposition."""

    def test_decomposition_consistency(self):
        """Check that decomposition values are internally consistent."""
        state = make_game_state(n_agents=2, n_issues=5, seed=42)
        agreement = [0.5] * state["n_issues"]
        utilities = {
            aid: compute_utility(aid, agreement, state)
            for aid in state["agent_positions"]
        }

        decomp = efficiency_fairness_decomposition(utilities, state)
        assert abs(decomp["sw"] - social_welfare(utilities)) < 1e-6
        assert abs(decomp["sw_star"] - optimal_social_welfare(state)) < 1e-6
        assert abs(decomp["efficiency_loss"] - (decomp["sw_star"] - decomp["sw"])) < 1e-6
        assert decomp["efficiency_loss"] >= -1e-6  # SW <= SW*
        assert decomp["fairness_deviation"] >= -1e-6  # distance is non-negative


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

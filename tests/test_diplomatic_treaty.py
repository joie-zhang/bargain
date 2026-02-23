#!/usr/bin/env python3
"""
Comprehensive tests for DiplomaticTreatyGame.

Tests cover:
- Configuration validation (rho, theta bounds, PSD feasibility)
- Gaussian copula position generation (uniformity, correlation, full range)
- SLSQP weight generation (simplex, exact cosine similarity)
- Utility calculation (perfect match = max utility)
- Proposal validation (correct number of issues, values in range)
- Proposal parsing (JSON extraction, fallback handling)
- Control parameter effects
"""

import pytest
import numpy as np
from typing import List
from scipy.stats import kstest

from game_environments import (
    create_game_environment,
    DiplomaticTreatyConfig,
    DiplomaticTreatyGame,
    GameType,
)


class FakeAgent:
    """Simple agent for testing."""
    def __init__(self, agent_id: str):
        self.agent_id = agent_id


def create_test_agents(n: int = 2) -> List[FakeAgent]:
    """Create a list of fake agents for testing."""
    return [FakeAgent(f"Agent_{i+1}") for i in range(n)]


class TestDiplomaticTreatyConfig:
    """Tests for DiplomaticTreatyConfig validation."""

    def test_valid_config(self):
        """Test that valid config parameters are accepted."""
        config = DiplomaticTreatyConfig(
            n_agents=2,
            t_rounds=10,
            n_issues=5,
            rho=0.0,
            theta=0.5,
        )
        assert config.rho == 0.0
        assert config.theta == 0.5

    def test_rho_bounds(self):
        """Test that rho must be in [-1, 1]."""
        DiplomaticTreatyConfig(n_agents=2, t_rounds=5, rho=-1.0)
        DiplomaticTreatyConfig(n_agents=2, t_rounds=5, rho=1.0)
        DiplomaticTreatyConfig(n_agents=2, t_rounds=5, rho=0.0)

        with pytest.raises(ValueError, match="rho must be in"):
            DiplomaticTreatyConfig(n_agents=2, t_rounds=5, rho=-1.5)
        with pytest.raises(ValueError, match="rho must be in"):
            DiplomaticTreatyConfig(n_agents=2, t_rounds=5, rho=1.5)

    def test_theta_bounds(self):
        """Test that theta must be in [0, 1]."""
        DiplomaticTreatyConfig(n_agents=2, t_rounds=5, theta=0.0)
        DiplomaticTreatyConfig(n_agents=2, t_rounds=5, theta=1.0)
        DiplomaticTreatyConfig(n_agents=2, t_rounds=5, theta=0.5)

        with pytest.raises(ValueError, match="theta must be in"):
            DiplomaticTreatyConfig(n_agents=2, t_rounds=5, theta=-0.1)
        with pytest.raises(ValueError, match="theta must be in"):
            DiplomaticTreatyConfig(n_agents=2, t_rounds=5, theta=1.5)

    def test_psd_n2_rho_negative_passes(self):
        """N=2, rho=-0.9 should pass (no PSD issue for 2 agents)."""
        DiplomaticTreatyConfig(n_agents=2, t_rounds=5, rho=-0.9)

    def test_psd_n3_rho_very_negative_fails(self):
        """N=3, rho=-0.9 should raise ValueError (PSD violation)."""
        with pytest.raises(ValueError, match="infeasible"):
            DiplomaticTreatyConfig(n_agents=3, t_rounds=5, rho=-0.9)

    def test_psd_n3_rho_moderate_negative_passes(self):
        """N=3, rho=-0.4 should pass (rho_z ~ -0.407 >= -0.5)."""
        DiplomaticTreatyConfig(n_agents=3, t_rounds=5, rho=-0.4)

    def test_psd_n5_rho_small_negative_passes(self):
        """N=5, rho=-0.2 should pass."""
        DiplomaticTreatyConfig(n_agents=5, t_rounds=5, rho=-0.2)


class TestGaussianCopulaPositions:
    """Tests for Gaussian copula position generation."""

    def test_positions_in_valid_range(self):
        """All positions must be in [0, 1]."""
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=10, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        for agent_id, positions in state["agent_positions"].items():
            for pos in positions:
                assert 0 <= pos <= 1, f"Position {pos} out of range for {agent_id}"

    def test_marginal_uniformity_ks_test(self):
        """With rho=0, marginals should be Uniform[0,1] (KS test)."""
        n_issues = 500
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=n_issues, rho=0.0,
            random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        for agent_id in state["agent_positions"]:
            positions = np.array(state["agent_positions"][agent_id])
            stat, p_value = kstest(positions, 'uniform')
            assert p_value > 0.01, (
                f"KS test failed for {agent_id}: stat={stat:.4f}, p={p_value:.4f}. "
                f"Positions may not be uniformly distributed."
            )

    def test_correlation_monotonicity(self):
        """Higher rho should produce higher position correlation."""
        rho_values = [-0.5, 0.0, 0.5, 1.0]
        correlations = []

        for rho in rho_values:
            game = create_game_environment(
                "diplomacy", n_agents=2, t_rounds=5, n_issues=200, rho=rho,
                random_seed=42
            )
            agents = create_test_agents(2)
            state = game.create_game_state(agents)

            pos1 = np.array(state["agent_positions"]["Agent_1"])
            pos2 = np.array(state["agent_positions"]["Agent_2"])
            corr = np.corrcoef(pos1, pos2)[0, 1]
            correlations.append(corr)

        # Check monotonicity
        for i in range(len(correlations) - 1):
            assert correlations[i] < correlations[i + 1], (
                f"Correlation not monotonic: rho={rho_values[i]}->{rho_values[i+1]}, "
                f"corr={correlations[i]:.4f}->{correlations[i+1]:.4f}"
            )

    def test_rho_zero_near_zero_correlation(self):
        """With rho=0, positions should have near-zero correlation."""
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=500, rho=0.0,
            random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        pos1 = np.array(state["agent_positions"]["Agent_1"])
        pos2 = np.array(state["agent_positions"]["Agent_2"])
        corr = np.corrcoef(pos1, pos2)[0, 1]
        assert abs(corr) < 0.15, f"Expected near-zero correlation at rho=0, got {corr:.4f}"

    def test_rho_one_identical_positions(self):
        """With rho=1, positions should be nearly identical."""
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=100, rho=1.0,
            random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        pos1 = np.array(state["agent_positions"]["Agent_1"])
        pos2 = np.array(state["agent_positions"]["Agent_2"])
        assert np.allclose(pos1, pos2, atol=1e-6), "rho=1.0 should give identical positions"

    def test_positions_cover_full_range(self):
        """Positions should cover the full [0, 1] range (not clustering around 0.5)."""
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=200, rho=0.0,
            random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        all_positions = []
        for agent_id in state["agent_positions"]:
            all_positions.extend(state["agent_positions"][agent_id])
        all_positions = np.array(all_positions)

        assert np.min(all_positions) < 0.1, "Positions should have values near 0"
        assert np.max(all_positions) > 0.9, "Positions should have values near 1"

    def test_reproducibility_with_seed(self):
        """Same seed should produce identical positions."""
        game1 = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=5, random_seed=42
        )
        game2 = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=5, random_seed=42
        )
        agents = create_test_agents(2)

        state1 = game1.create_game_state(agents)
        state2 = game2.create_game_state(agents)

        assert state1["agent_positions"] == state2["agent_positions"]
        assert state1["agent_weights"] == state2["agent_weights"]


class TestSLSQPWeights:
    """Tests for SLSQP weight generation."""

    def test_weights_sum_to_one(self):
        """Weights must sum to 1 for each agent."""
        game = create_game_environment(
            "diplomacy", n_agents=3, t_rounds=5, n_issues=5, random_seed=42
        )
        agents = create_test_agents(3)
        state = game.create_game_state(agents)

        for agent_id, weights in state["agent_weights"].items():
            weight_sum = sum(weights)
            assert abs(weight_sum - 1.0) < 1e-6, (
                f"Weights sum to {weight_sum} for {agent_id}"
            )

    def test_weights_non_negative(self):
        """All weights must be non-negative."""
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=5, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        for agent_id, weights in state["agent_weights"].items():
            for w in weights:
                assert w >= -1e-10, f"Negative weight {w} for {agent_id}"

    @pytest.mark.parametrize("theta", [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95])
    def test_exact_cosine_similarity(self, theta):
        """Cosine similarity of weights should match theta within tolerance."""
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=5, theta=theta,
            random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        w1 = np.array(state["agent_weights"]["Agent_1"])
        w2 = np.array(state["agent_weights"]["Agent_2"])

        cos_sim = np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2))
        assert abs(cos_sim - theta) < 0.01, (
            f"theta={theta}, actual cosine={cos_sim:.4f}, error={abs(cos_sim-theta):.4f}"
        )

    def test_theta_one_identical_weights(self):
        """theta=1.0 should give identical weights."""
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=5, theta=1.0,
            random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        w1 = np.array(state["agent_weights"]["Agent_1"])
        w2 = np.array(state["agent_weights"]["Agent_2"])
        assert np.allclose(w1, w2), "theta=1.0 should give identical weights"

    def test_multi_agent_pairwise_cosine(self):
        """For N>2 agents, ALL pairwise cosine similarities should be near theta."""
        theta = 0.5
        for n_agents in [3, 4]:
            game = create_game_environment(
                "diplomacy", n_agents=n_agents, t_rounds=5, n_issues=8,
                theta=theta, random_seed=42
            )
            agents = create_test_agents(n_agents)
            state = game.create_game_state(agents)

            agent_ids = list(state["agent_weights"].keys())
            weights = [np.array(state["agent_weights"][aid]) for aid in agent_ids]
            for i in range(n_agents):
                for j in range(i + 1, n_agents):
                    cos_sim = np.dot(weights[i], weights[j]) / (
                        np.linalg.norm(weights[i]) * np.linalg.norm(weights[j])
                    )
                    assert abs(cos_sim - theta) < 0.05, (
                        f"N={n_agents}, pair ({i},{j}): cosine={cos_sim:.4f}, "
                        f"expected {theta}"
                    )


class TestGameStateCreation:
    """Tests for game state creation."""

    def test_game_state_structure(self):
        """Game state should have all required keys (no issue_types)."""
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=5, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        assert "issues" in state
        assert "n_issues" in state
        assert "agent_positions" in state
        assert "agent_weights" in state
        assert "parameters" in state
        assert "game_type" in state
        # issue_types should NOT be present
        assert "issue_types" not in state
        # lambda should NOT be in parameters
        assert "lambda" not in state["parameters"]

    def test_correct_number_of_issues(self):
        """Correct number of issues should be created."""
        for n_issues in [3, 5, 8]:
            game = create_game_environment(
                "diplomacy", n_agents=2, t_rounds=5, n_issues=n_issues, random_seed=42
            )
            agents = create_test_agents(2)
            state = game.create_game_state(agents)

            assert state["n_issues"] == n_issues
            assert len(state["issues"]) == n_issues
            for agent_id in state["agent_positions"]:
                assert len(state["agent_positions"][agent_id]) == n_issues
                assert len(state["agent_weights"][agent_id]) == n_issues


class TestUtilityCalculation:
    """Tests for utility calculation."""

    def test_perfect_match_max_utility(self):
        """Perfect position match gives maximum utility."""
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=5, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        agent_id = "Agent_1"
        positions = state["agent_positions"][agent_id]
        proposal = {"agreement": positions}

        utility = game.calculate_utility(agent_id, proposal, state, round_num=1)
        assert abs(utility - 1.0) < 1e-6, f"Expected utility 1.0, got {utility}"

    def test_worst_case_utility(self):
        """Opposite positions give minimum utility."""
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=5, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        agent_id = "Agent_1"
        positions = state["agent_positions"][agent_id]
        worst_agreement = [0.0 if p >= 0.5 else 1.0 for p in positions]
        proposal = {"agreement": worst_agreement}

        utility = game.calculate_utility(agent_id, proposal, state, round_num=1)
        assert utility >= 0, f"Utility should be non-negative, got {utility}"
        assert utility < 0.5, f"Utility should be low for worst case, got {utility}"

    def test_discount_factor_applied(self):
        """Discount factor should be applied correctly."""
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, gamma_discount=0.9, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        agent_id = "Agent_1"
        positions = state["agent_positions"][agent_id]
        proposal = {"agreement": positions}

        utility_r1 = game.calculate_utility(agent_id, proposal, state, round_num=1)
        utility_r2 = game.calculate_utility(agent_id, proposal, state, round_num=2)
        utility_r3 = game.calculate_utility(agent_id, proposal, state, round_num=3)

        assert abs(utility_r2 - utility_r1 * 0.9) < 1e-6
        assert abs(utility_r3 - utility_r1 * 0.81) < 1e-6

    def test_utility_formula_correctness(self):
        """Utility formula: U = Σ w_k * (1 - |p_k - a_k|)."""
        config = DiplomaticTreatyConfig(
            n_agents=2, t_rounds=5, n_issues=3, gamma_discount=1.0, random_seed=42
        )
        game = DiplomaticTreatyGame(config)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        agent_id = "Agent_1"
        positions = np.array(state["agent_positions"][agent_id])
        weights = np.array(state["agent_weights"][agent_id])
        agreement = [0.3, 0.6, 0.9]

        proposal = {"agreement": agreement}
        utility = game.calculate_utility(agent_id, proposal, state, round_num=1)

        agreement_arr = np.array(agreement)
        expected = np.sum(weights * (1 - np.abs(positions - agreement_arr)))

        assert abs(utility - expected) < 1e-6


class TestProposalValidation:
    """Tests for proposal validation."""

    def test_valid_proposal(self):
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=5, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        proposal = {"agreement": [0.3, 0.5, 0.7, 0.2, 0.9]}
        assert game.validate_proposal(proposal, state) is True

    def test_wrong_number_of_issues(self):
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=5, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        assert game.validate_proposal({"agreement": [0.3, 0.5, 0.7]}, state) is False
        assert game.validate_proposal({"agreement": [0.3]*7}, state) is False

    def test_values_out_of_range(self):
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=5, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        assert game.validate_proposal({"agreement": [-0.1, 0.5, 0.7, 0.2, 0.9]}, state) is False
        assert game.validate_proposal({"agreement": [0.3, 1.5, 0.7, 0.2, 0.9]}, state) is False

    def test_boundary_values(self):
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=5, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        assert game.validate_proposal({"agreement": [0.0, 1.0, 0.0, 1.0, 0.5]}, state) is True


class TestProposalParsing:
    """Tests for proposal parsing."""

    def test_valid_json_parsing(self):
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=3, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        response = '{"agreement": [0.3, 0.5, 0.7], "reasoning": "Test"}'
        proposal = game.parse_proposal(response, "Agent_1", state, ["Agent_1", "Agent_2"])
        assert proposal["agreement"] == [0.3, 0.5, 0.7]
        assert proposal["proposed_by"] == "Agent_1"

    def test_json_in_text(self):
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=3, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        response = 'Here: {"agreement": [0.4, 0.6, 0.8], "reasoning": "ok"}'
        proposal = game.parse_proposal(response, "Agent_1", state, ["Agent_1", "Agent_2"])
        assert proposal["agreement"] == [0.4, 0.6, 0.8]

    def test_fallback_on_invalid_json(self):
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=3, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        response = "Not valid JSON at all!"
        proposal = game.parse_proposal(response, "Agent_1", state, ["Agent_1", "Agent_2"])
        assert proposal["agreement"] == [0.5, 0.5, 0.5]

    def test_value_clipping(self):
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=3, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        response = '{"agreement": [-0.5, 1.5, 0.5], "reasoning": "Test"}'
        proposal = game.parse_proposal(response, "Agent_1", state, ["Agent_1", "Agent_2"])
        assert proposal["agreement"][0] == 0.0
        assert proposal["agreement"][1] == 1.0
        assert proposal["agreement"][2] == 0.5


class TestControlParameterEffects:
    """Tests for control parameter effects."""

    def test_high_rho_correlated_positions(self):
        """High rho should produce more correlated positions than low rho."""
        game_high = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=20, rho=0.9, random_seed=42
        )
        agents = create_test_agents(2)
        state_high = game_high.create_game_state(agents)

        game_low = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=20, rho=-0.5, random_seed=42
        )
        state_low = game_low.create_game_state(agents)

        pos1_high = np.array(state_high["agent_positions"]["Agent_1"])
        pos2_high = np.array(state_high["agent_positions"]["Agent_2"])
        corr_high = np.corrcoef(pos1_high, pos2_high)[0, 1]

        pos1_low = np.array(state_low["agent_positions"]["Agent_1"])
        pos2_low = np.array(state_low["agent_positions"]["Agent_2"])
        corr_low = np.corrcoef(pos1_low, pos2_low)[0, 1]

        assert corr_high > corr_low

    def test_high_theta_similar_weights(self):
        """High theta should produce more similar weights (higher cosine similarity)."""
        game_high = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=10, theta=0.95, random_seed=42
        )
        agents = create_test_agents(2)
        state_high = game_high.create_game_state(agents)

        game_low = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=10, theta=0.1, random_seed=42
        )
        state_low = game_low.create_game_state(agents)

        w1_h = np.array(state_high["agent_weights"]["Agent_1"])
        w2_h = np.array(state_high["agent_weights"]["Agent_2"])
        sim_high = np.dot(w1_h, w2_h) / (np.linalg.norm(w1_h) * np.linalg.norm(w2_h))

        w1_l = np.array(state_low["agent_weights"]["Agent_1"])
        w2_l = np.array(state_low["agent_weights"]["Agent_2"])
        sim_low = np.dot(w1_l, w2_l) / (np.linalg.norm(w1_l) * np.linalg.norm(w2_l))

        assert sim_high > sim_low


class TestGameType:
    """Tests for game type identification."""

    def test_game_type(self):
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=5, random_seed=42
        )
        assert game.get_game_type() == GameType.DIPLOMATIC_TREATY

    def test_factory_creates_correct_type(self):
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=5
        )
        assert isinstance(game, DiplomaticTreatyGame)


class TestPromptGeneration:
    """Tests for prompt generation methods."""

    def test_game_rules_prompt(self):
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=10, n_issues=5, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        prompt = game.get_game_rules_prompt(state)
        assert "Diplomatic" in prompt
        assert "10" in prompt

    def test_preference_prompt_contains_positions_and_weights(self):
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=3, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        prompt = game.get_preference_assignment_prompt("Agent_1", state)

        for pos in state["agent_positions"]["Agent_1"]:
            assert f"{pos:.3f}" in prompt
        for weight in state["agent_weights"]["Agent_1"]:
            assert f"{weight:.3f}" in prompt

    def test_preference_prompt_no_issue_types(self):
        """Preference prompt should not mention win-win/zero-sum."""
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=3, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        prompt = game.get_preference_assignment_prompt("Agent_1", state)
        assert "win-win" not in prompt
        assert "zero-sum" not in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

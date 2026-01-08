#!/usr/bin/env python3
"""
Comprehensive tests for DiplomaticTreatyGame.

Tests cover:
- Configuration validation (rho, theta, lambda bounds)
- Game state creation (positions in [0,1], weights sum to 1)
- Utility calculation (perfect match = max utility)
- Proposal validation (correct number of issues, values in range)
- Proposal parsing (JSON extraction, fallback handling)
- Control parameter effects
"""

import pytest
import numpy as np
from typing import List

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
            lam=0.0
        )
        assert config.rho == 0.0
        assert config.theta == 0.5
        assert config.lam == 0.0

    def test_rho_bounds(self):
        """Test that rho must be in [-1, 1]."""
        # Valid bounds
        DiplomaticTreatyConfig(n_agents=2, t_rounds=5, rho=-1.0)
        DiplomaticTreatyConfig(n_agents=2, t_rounds=5, rho=1.0)
        DiplomaticTreatyConfig(n_agents=2, t_rounds=5, rho=0.0)

        # Invalid bounds
        with pytest.raises(ValueError, match="rho must be in"):
            DiplomaticTreatyConfig(n_agents=2, t_rounds=5, rho=-1.5)
        with pytest.raises(ValueError, match="rho must be in"):
            DiplomaticTreatyConfig(n_agents=2, t_rounds=5, rho=1.5)

    def test_theta_bounds(self):
        """Test that theta must be in [0, 1]."""
        # Valid bounds
        DiplomaticTreatyConfig(n_agents=2, t_rounds=5, theta=0.0)
        DiplomaticTreatyConfig(n_agents=2, t_rounds=5, theta=1.0)
        DiplomaticTreatyConfig(n_agents=2, t_rounds=5, theta=0.5)

        # Invalid bounds
        with pytest.raises(ValueError, match="theta must be in"):
            DiplomaticTreatyConfig(n_agents=2, t_rounds=5, theta=-0.1)
        with pytest.raises(ValueError, match="theta must be in"):
            DiplomaticTreatyConfig(n_agents=2, t_rounds=5, theta=1.5)

    def test_lambda_bounds(self):
        """Test that lam must be in [-1, 1]."""
        # Valid bounds
        DiplomaticTreatyConfig(n_agents=2, t_rounds=5, lam=-1.0)
        DiplomaticTreatyConfig(n_agents=2, t_rounds=5, lam=1.0)
        DiplomaticTreatyConfig(n_agents=2, t_rounds=5, lam=0.0)

        # Invalid bounds
        with pytest.raises(ValueError, match="lam must be in"):
            DiplomaticTreatyConfig(n_agents=2, t_rounds=5, lam=-1.5)
        with pytest.raises(ValueError, match="lam must be in"):
            DiplomaticTreatyConfig(n_agents=2, t_rounds=5, lam=1.5)


class TestGameStateCreation:
    """Tests for game state creation."""

    def test_game_state_structure(self):
        """Test that game state has all required keys."""
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=5, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        assert "issues" in state
        assert "n_issues" in state
        assert "agent_positions" in state
        assert "agent_weights" in state
        assert "issue_types" in state
        assert "parameters" in state
        assert "game_type" in state

    def test_positions_in_valid_range(self):
        """Test that all positions are in [0, 1]."""
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=10, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        for agent_id, positions in state["agent_positions"].items():
            for pos in positions:
                assert 0 <= pos <= 1, f"Position {pos} out of range for {agent_id}"

    def test_weights_sum_to_one(self):
        """Test that weights sum to 1 for each agent."""
        game = create_game_environment(
            "diplomacy", n_agents=3, t_rounds=5, n_issues=5, random_seed=42
        )
        agents = create_test_agents(3)
        state = game.create_game_state(agents)

        for agent_id, weights in state["agent_weights"].items():
            weight_sum = sum(weights)
            assert abs(weight_sum - 1.0) < 1e-6, f"Weights sum to {weight_sum} for {agent_id}"

    def test_weights_non_negative(self):
        """Test that all weights are non-negative."""
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=5, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        for agent_id, weights in state["agent_weights"].items():
            for w in weights:
                assert w >= 0, f"Negative weight {w} for {agent_id}"

    def test_correct_number_of_issues(self):
        """Test that correct number of issues are created."""
        for n_issues in [3, 5, 8]:
            game = create_game_environment(
                "diplomacy", n_agents=2, t_rounds=5, n_issues=n_issues, random_seed=42
            )
            agents = create_test_agents(2)
            state = game.create_game_state(agents)

            assert state["n_issues"] == n_issues
            assert len(state["issues"]) == n_issues
            assert len(state["issue_types"]) == n_issues
            for agent_id in state["agent_positions"]:
                assert len(state["agent_positions"][agent_id]) == n_issues
                assert len(state["agent_weights"][agent_id]) == n_issues

    def test_issue_types_valid(self):
        """Test that issue types are either -1 or 1."""
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=10, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        for issue_type in state["issue_types"]:
            assert issue_type in [-1, 1], f"Invalid issue type: {issue_type}"

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same game state."""
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


class TestUtilityCalculation:
    """Tests for utility calculation."""

    def test_perfect_match_max_utility(self):
        """Test that perfect position match gives maximum utility."""
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=5, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        agent_id = "Agent_1"
        positions = state["agent_positions"][agent_id]

        # Proposal exactly matching agent's positions
        proposal = {"agreement": positions}

        utility = game.calculate_utility(agent_id, proposal, state, round_num=1)

        # Maximum utility is 1.0 (sum of weights × 1.0 = 1.0)
        assert abs(utility - 1.0) < 1e-6, f"Expected utility 1.0, got {utility}"

    def test_worst_case_utility(self):
        """Test that opposite positions give minimum utility."""
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=5, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        agent_id = "Agent_1"
        positions = state["agent_positions"][agent_id]

        # Proposal at maximum distance from agent's positions
        # For position p, the worst agreement is 0 if p >= 0.5, else 1
        worst_agreement = [0.0 if p >= 0.5 else 1.0 for p in positions]
        proposal = {"agreement": worst_agreement}

        utility = game.calculate_utility(agent_id, proposal, state, round_num=1)

        # Utility should be low but non-negative
        assert utility >= 0, f"Utility should be non-negative, got {utility}"
        assert utility < 0.5, f"Utility should be low for worst case, got {utility}"

    def test_discount_factor_applied(self):
        """Test that discount factor is applied correctly."""
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

        # Check discount is applied
        assert abs(utility_r2 - utility_r1 * 0.9) < 1e-6
        assert abs(utility_r3 - utility_r1 * 0.81) < 1e-6

    def test_utility_formula_correctness(self):
        """Test utility formula: U = Σ w_k × (1 - |p_k - a_k|)."""
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

        # Manual calculation
        agreement_arr = np.array(agreement)
        expected = np.sum(weights * (1 - np.abs(positions - agreement_arr)))

        assert abs(utility - expected) < 1e-6


class TestProposalValidation:
    """Tests for proposal validation."""

    def test_valid_proposal(self):
        """Test that valid proposal passes validation."""
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=5, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        proposal = {"agreement": [0.3, 0.5, 0.7, 0.2, 0.9]}
        assert game.validate_proposal(proposal, state) is True

    def test_wrong_number_of_issues(self):
        """Test that wrong number of issues fails validation."""
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=5, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        # Too few
        proposal = {"agreement": [0.3, 0.5, 0.7]}
        assert game.validate_proposal(proposal, state) is False

        # Too many
        proposal = {"agreement": [0.3, 0.5, 0.7, 0.2, 0.9, 0.4, 0.6]}
        assert game.validate_proposal(proposal, state) is False

    def test_values_out_of_range(self):
        """Test that values outside [0, 1] fail validation."""
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=5, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        # Negative value
        proposal = {"agreement": [-0.1, 0.5, 0.7, 0.2, 0.9]}
        assert game.validate_proposal(proposal, state) is False

        # Value > 1
        proposal = {"agreement": [0.3, 1.5, 0.7, 0.2, 0.9]}
        assert game.validate_proposal(proposal, state) is False

    def test_boundary_values(self):
        """Test that boundary values 0 and 1 are valid."""
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=5, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        proposal = {"agreement": [0.0, 1.0, 0.0, 1.0, 0.5]}
        assert game.validate_proposal(proposal, state) is True


class TestProposalParsing:
    """Tests for proposal parsing."""

    def test_valid_json_parsing(self):
        """Test parsing valid JSON response."""
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=3, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        response = '{"agreement": [0.3, 0.5, 0.7], "reasoning": "Test"}'
        proposal = game.parse_proposal(response, "Agent_1", state, ["Agent_1", "Agent_2"])

        assert proposal["agreement"] == [0.3, 0.5, 0.7]
        assert proposal["reasoning"] == "Test"
        assert proposal["proposed_by"] == "Agent_1"

    def test_json_in_text(self):
        """Test extracting JSON from surrounding text."""
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=3, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        response = 'Here is my proposal: {"agreement": [0.4, 0.6, 0.8], "reasoning": "Compromise"}'
        proposal = game.parse_proposal(response, "Agent_1", state, ["Agent_1", "Agent_2"])

        assert proposal["agreement"] == [0.4, 0.6, 0.8]

    def test_fallback_on_invalid_json(self):
        """Test fallback to neutral proposal on invalid JSON."""
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=3, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        response = "This is not valid JSON at all!"
        proposal = game.parse_proposal(response, "Agent_1", state, ["Agent_1", "Agent_2"])

        # Should fallback to neutral 0.5 for all issues
        assert proposal["agreement"] == [0.5, 0.5, 0.5]
        assert proposal["proposed_by"] == "Agent_1"

    def test_value_clipping(self):
        """Test that out-of-range values are clipped."""
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=3, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        response = '{"agreement": [-0.5, 1.5, 0.5], "reasoning": "Test"}'
        proposal = game.parse_proposal(response, "Agent_1", state, ["Agent_1", "Agent_2"])

        # Values should be clipped to [0, 1]
        assert proposal["agreement"][0] == 0.0
        assert proposal["agreement"][1] == 1.0
        assert proposal["agreement"][2] == 0.5

    def test_padding_short_agreement(self):
        """Test that short agreement arrays are padded with 0.5."""
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=5, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        response = '{"agreement": [0.3, 0.7], "reasoning": "Test"}'
        proposal = game.parse_proposal(response, "Agent_1", state, ["Agent_1", "Agent_2"])

        assert len(proposal["agreement"]) == 5
        assert proposal["agreement"][:2] == [0.3, 0.7]
        assert proposal["agreement"][2:] == [0.5, 0.5, 0.5]


class TestControlParameterEffects:
    """Tests for control parameter effects."""

    def test_high_rho_correlated_positions(self):
        """Test that high rho produces more correlated positions."""
        # High rho (cooperative)
        game_high = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=20, rho=0.9, random_seed=42
        )
        agents = create_test_agents(2)
        state_high = game_high.create_game_state(agents)

        # Low rho (competitive)
        game_low = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=20, rho=-0.9, random_seed=42
        )
        state_low = game_low.create_game_state(agents)

        # Calculate position correlations
        pos1_high = np.array(state_high["agent_positions"]["Agent_1"])
        pos2_high = np.array(state_high["agent_positions"]["Agent_2"])
        corr_high = np.corrcoef(pos1_high, pos2_high)[0, 1]

        pos1_low = np.array(state_low["agent_positions"]["Agent_1"])
        pos2_low = np.array(state_low["agent_positions"]["Agent_2"])
        corr_low = np.corrcoef(pos1_low, pos2_low)[0, 1]

        # High rho should have higher correlation than low rho
        assert corr_high > corr_low, f"Expected high rho to have higher correlation: {corr_high} vs {corr_low}"

    def test_high_theta_similar_weights(self):
        """Test that high theta produces more similar weights."""
        # High theta (overlapping interests)
        game_high = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=10, theta=0.95, random_seed=42
        )
        agents = create_test_agents(2)
        state_high = game_high.create_game_state(agents)

        # Low theta (different interests)
        game_low = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=10, theta=0.05, random_seed=42
        )
        state_low = game_low.create_game_state(agents)

        # Calculate weight similarity (cosine similarity)
        w1_high = np.array(state_high["agent_weights"]["Agent_1"])
        w2_high = np.array(state_high["agent_weights"]["Agent_2"])
        sim_high = np.dot(w1_high, w2_high) / (np.linalg.norm(w1_high) * np.linalg.norm(w2_high))

        w1_low = np.array(state_low["agent_weights"]["Agent_1"])
        w2_low = np.array(state_low["agent_weights"]["Agent_2"])
        sim_low = np.dot(w1_low, w2_low) / (np.linalg.norm(w1_low) * np.linalg.norm(w2_low))

        # High theta should have more similar weights
        assert sim_high > sim_low, f"Expected high theta to have more similar weights: {sim_high} vs {sim_low}"

    def test_lambda_affects_issue_types(self):
        """Test that lambda affects distribution of issue types."""
        n_trials = 5
        n_issues = 20

        # High lambda (mostly compatible)
        compatible_counts_high = []
        for seed in range(n_trials):
            game = create_game_environment(
                "diplomacy", n_agents=2, t_rounds=5, n_issues=n_issues, lam=0.9, random_seed=seed
            )
            agents = create_test_agents(2)
            state = game.create_game_state(agents)
            compatible_counts_high.append(sum(1 for t in state["issue_types"] if t == 1))

        # Low lambda (mostly conflicting)
        compatible_counts_low = []
        for seed in range(n_trials):
            game = create_game_environment(
                "diplomacy", n_agents=2, t_rounds=5, n_issues=n_issues, lam=-0.9, random_seed=seed
            )
            agents = create_test_agents(2)
            state = game.create_game_state(agents)
            compatible_counts_low.append(sum(1 for t in state["issue_types"] if t == 1))

        avg_high = np.mean(compatible_counts_high)
        avg_low = np.mean(compatible_counts_low)

        # High lambda should have more compatible issues
        assert avg_high > avg_low, f"Expected high lambda to have more compatible issues: {avg_high} vs {avg_low}"


class TestGameType:
    """Tests for game type identification."""

    def test_game_type(self):
        """Test that game type is correctly identified."""
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=5, random_seed=42
        )
        assert game.get_game_type() == GameType.DIPLOMATIC_TREATY

    def test_factory_creates_correct_type(self):
        """Test that factory creates DiplomaticTreatyGame."""
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=5
        )
        assert isinstance(game, DiplomaticTreatyGame)


class TestPromptGeneration:
    """Tests for prompt generation methods."""

    def test_game_rules_prompt(self):
        """Test that game rules prompt contains key information."""
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=10, n_issues=5, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        prompt = game.get_game_rules_prompt(state)

        assert "Diplomatic" in prompt
        assert "5" in prompt or "five" in prompt.lower()  # number of issues
        assert "10" in prompt  # number of rounds
        assert "[0" in prompt and "1]" in prompt  # value range

    def test_preference_prompt_contains_positions_and_weights(self):
        """Test that preference prompt contains agent's positions and weights."""
        game = create_game_environment(
            "diplomacy", n_agents=2, t_rounds=5, n_issues=3, random_seed=42
        )
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        prompt = game.get_preference_assignment_prompt("Agent_1", state)

        # Should contain position values
        for pos in state["agent_positions"]["Agent_1"]:
            assert f"{pos:.3f}" in prompt

        # Should contain weight values
        for weight in state["agent_weights"]["Agent_1"]:
            assert f"{weight:.3f}" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

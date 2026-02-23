#!/usr/bin/env python3
"""
Tests for CoFundingGame core logic.

Tests cover:
- Valuation generation (sum=100, non-negative, cosine similarity)
- Game state creation (all required keys, budgets, costs)
- Utility calculation (funded vs unfunded, edge cases)
- Proposal parsing and validation
- Protocol type
- Voting prompt raises RuntimeError
- Early termination detection
- Seed reproducibility
"""

import json

import pytest
import numpy as np
from typing import List

from game_environments import CoFundingConfig, GameType
from game_environments.co_funding import CoFundingGame


class FakeAgent:
    """Simple agent for testing."""
    def __init__(self, agent_id: str):
        self.agent_id = agent_id


def create_test_agents(n: int = 2) -> List[FakeAgent]:
    return [FakeAgent(f"Agent_{i+1}") for i in range(n)]


def make_game(seed=42, alpha=0.5, sigma=0.5, m_projects=5, n_agents=2, t_rounds=5):
    config = CoFundingConfig(
        n_agents=n_agents,
        t_rounds=t_rounds,
        m_projects=m_projects,
        alpha=alpha,
        sigma=sigma,
        random_seed=seed,
    )
    return CoFundingGame(config)


class TestValuationGeneration:
    """Tests for valuation vector generation."""

    def test_valuations_sum_to_100(self):
        """Each agent's valuation vector should sum to 100."""
        game = make_game(alpha=0.5)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        for aid, vals in state["agent_valuations"].items():
            assert abs(sum(vals) - 100.0) < 0.01, f"{aid} valuations sum to {sum(vals)}"

    def test_valuations_non_negative(self):
        """All valuations should be non-negative."""
        game = make_game(alpha=0.5)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        for aid, vals in state["agent_valuations"].items():
            assert all(v >= -1e-9 for v in vals), f"{aid} has negative valuations"

    @pytest.mark.parametrize("alpha", [0.0, 0.3, 0.5, 0.7, 1.0])
    def test_cosine_similarity_matches_alpha(self, alpha):
        """Cosine similarity between agent valuations should match alpha."""
        game = make_game(alpha=alpha, seed=42)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        v1 = np.array(state["agent_valuations"]["Agent_1"])
        v2 = np.array(state["agent_valuations"]["Agent_2"])
        cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        assert abs(cos_sim - alpha) < 0.02, f"Expected cos_sim={alpha}, got {cos_sim}"

    def test_identical_valuations_alpha_1(self):
        """alpha=1 should produce identical valuations."""
        game = make_game(alpha=1.0)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        v1 = state["agent_valuations"]["Agent_1"]
        v2 = state["agent_valuations"]["Agent_2"]
        np.testing.assert_allclose(v1, v2, atol=1e-6)

    def test_3agent_valuations(self):
        """3-agent valuation generation should produce all-pairs cosine similarity matching alpha."""
        game = make_game(alpha=0.5, n_agents=3)
        agents = create_test_agents(3)
        state = game.create_game_state(agents)
        vecs = []
        for aid in ["Agent_1", "Agent_2", "Agent_3"]:
            vals = state["agent_valuations"][aid]
            assert abs(sum(vals) - 100.0) < 0.01
            assert all(v >= -1e-9 for v in vals)
            vecs.append(np.array(vals))
        # Verify ALL pairwise cosine similarities match alpha
        for i in range(3):
            for j in range(i + 1, 3):
                cos_sim = np.dot(vecs[i], vecs[j]) / (np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[j]))
                assert abs(cos_sim - 0.5) < 0.05, (
                    f"Pair ({i},{j}): expected cos_sim=0.5, got {cos_sim:.4f}"
                )


class TestGameStateCreation:
    """Tests for create_game_state."""

    def test_game_state_has_required_keys(self):
        """Game state should contain all required fields."""
        game = make_game()
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        required_keys = [
            "projects", "m_projects", "project_costs", "total_cost",
            "agent_budgets", "total_budget", "agent_valuations",
            "parameters", "game_type", "round_pledges",
            "current_pledges", "aggregate_totals", "funded_projects",
        ]
        for key in required_keys:
            assert key in state, f"Missing key: {key}"

    def test_budget_equals_sigma_times_cost(self):
        """Total budget should equal sigma * total_cost."""
        game = make_game(sigma=0.6)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        expected_budget = 0.6 * state["total_cost"]
        assert abs(state["total_budget"] - expected_budget) < 0.1

    def test_per_agent_budget_equal(self):
        """Each agent should get equal budget."""
        game = make_game(n_agents=3)
        agents = create_test_agents(3)
        state = game.create_game_state(agents)
        budgets = list(state["agent_budgets"].values())
        assert abs(budgets[0] - budgets[1]) < 0.01
        assert abs(budgets[1] - budgets[2]) < 0.01

    def test_project_costs_in_range(self):
        """Project costs should be within [c_min, c_max]."""
        game = make_game()
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        for cost in state["project_costs"]:
            assert 10.0 <= cost <= 30.0

    def test_m_projects_correct(self):
        """Number of projects should match config."""
        game = make_game(m_projects=7)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        assert state["m_projects"] == 7
        assert len(state["projects"]) == 7
        assert len(state["project_costs"]) == 7

    def test_game_type_string(self):
        """game_type field should be 'co_funding'."""
        game = make_game()
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        assert state["game_type"] == "co_funding"


class TestUtilityCalculation:
    """Tests for calculate_utility."""

    def test_all_funded_no_contribution(self):
        """If all projects funded and agent contributes 0, utility = sum(valuations)."""
        game = make_game(m_projects=3)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        state["funded_projects"] = [0, 1, 2]

        proposal = {"contributions": [0.0, 0.0, 0.0], "proposed_by": "Agent_1"}
        utility = game.calculate_utility("Agent_1", proposal, state, 1)

        expected = sum(state["agent_valuations"]["Agent_1"])
        assert abs(utility - expected) < 0.01

    def test_unfunded_project_zero_utility(self):
        """Contributions to unfunded projects should not affect utility."""
        game = make_game(m_projects=3)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        state["funded_projects"] = []  # nothing funded

        proposal = {"contributions": [10.0, 10.0, 10.0], "proposed_by": "Agent_1"}
        utility = game.calculate_utility("Agent_1", proposal, state, 1)
        assert utility == 0.0

    def test_utility_manual_example(self):
        """Manual utility calculation check."""
        game = make_game(m_projects=3)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        # Override valuations for predictable test
        state["agent_valuations"]["Agent_1"] = [50.0, 30.0, 20.0]
        state["funded_projects"] = [0, 2]  # projects 0 and 2 funded

        proposal = {"contributions": [15.0, 5.0, 8.0], "proposed_by": "Agent_1"}
        utility = game.calculate_utility("Agent_1", proposal, state, 1)

        # U = (50 - 15) + (20 - 8) = 35 + 12 = 47
        assert abs(utility - 47.0) < 0.01

    def test_no_discount_factor(self):
        """Utility should NOT change with round number (no discounting)."""
        game = make_game(m_projects=3)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        state["funded_projects"] = [0]

        proposal = {"contributions": [5.0, 0.0, 0.0], "proposed_by": "Agent_1"}
        u1 = game.calculate_utility("Agent_1", proposal, state, 1)
        u5 = game.calculate_utility("Agent_1", proposal, state, 5)
        u10 = game.calculate_utility("Agent_1", proposal, state, 10)
        assert u1 == u5 == u10


class TestProposalParsing:
    """Tests for parse_proposal and validate_proposal."""

    def test_parse_valid_json(self):
        """Valid JSON response should be parsed correctly."""
        game = make_game(m_projects=3)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        response = '{"contributions": [5.0, 10.0, 3.0], "reasoning": "test"}'
        parsed = game.parse_proposal(response, "Agent_1", state, ["Agent_1", "Agent_2"])
        assert parsed["contributions"] == [5.0, 10.0, 3.0]
        assert parsed["proposed_by"] == "Agent_1"

    def test_parse_json_embedded_in_text(self):
        """JSON embedded in text should be extracted."""
        game = make_game(m_projects=3)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        response = 'Here is my pledge: {"contributions": [1.0, 2.0, 3.0], "reasoning": "test"} thank you'
        parsed = game.parse_proposal(response, "Agent_1", state, ["Agent_1", "Agent_2"])
        assert parsed["contributions"] == [1.0, 2.0, 3.0]

    def test_parse_malformed_json(self):
        """Malformed JSON should fall back to zero vector."""
        game = make_game(m_projects=3)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        response = "this is not JSON at all"
        parsed = game.parse_proposal(response, "Agent_1", state, ["Agent_1", "Agent_2"])
        assert parsed["contributions"] == [0.0, 0.0, 0.0]

    def test_validate_valid_proposal(self):
        """Valid proposal should pass validation."""
        game = make_game(m_projects=3)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        budget = state["agent_budgets"]["Agent_1"]

        proposal = {
            "contributions": [budget * 0.3, budget * 0.3, budget * 0.3],
            "proposed_by": "Agent_1",
        }
        assert game.validate_proposal(proposal, state) is True

    def test_validate_over_budget(self):
        """Over-budget proposal should fail validation."""
        game = make_game(m_projects=3)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        budget = state["agent_budgets"]["Agent_1"]

        proposal = {
            "contributions": [budget, budget, budget],
            "proposed_by": "Agent_1",
        }
        assert game.validate_proposal(proposal, state) is False

    def test_validate_negative_contributions(self):
        """Negative contributions should fail validation."""
        game = make_game(m_projects=3)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        proposal = {
            "contributions": [-1.0, 5.0, 5.0],
            "proposed_by": "Agent_1",
        }
        assert game.validate_proposal(proposal, state) is False

    def test_validate_wrong_length(self):
        """Wrong-length contribution vector should fail validation."""
        game = make_game(m_projects=3)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        proposal = {
            "contributions": [1.0, 2.0],  # only 2 instead of 3
            "proposed_by": "Agent_1",
        }
        assert game.validate_proposal(proposal, state) is False


class TestProtocolAndGameType:
    """Tests for protocol type, game type, and voting prompt."""

    def test_protocol_type(self):
        """Co-funding should return 'talk_pledge_revise'."""
        game = make_game()
        assert game.get_protocol_type() == "talk_pledge_revise"

    def test_game_type(self):
        """Should return CO_FUNDING enum."""
        game = make_game()
        assert game.get_game_type() == GameType.CO_FUNDING

    def test_voting_prompt_raises(self):
        """get_voting_prompt should raise RuntimeError."""
        game = make_game()
        with pytest.raises(RuntimeError, match="voting"):
            game.get_voting_prompt("Agent_1", {}, {}, 1)


class TestEarlyTermination:
    """Tests for check_early_termination (individual mode)."""

    def _make_individual_game(self, m_projects=3):
        config = CoFundingConfig(
            n_agents=2, t_rounds=5, m_projects=m_projects,
            alpha=0.5, sigma=0.5, random_seed=42,
            pledge_mode="individual",
        )
        return CoFundingGame(config)

    def test_early_termination_identical_pledges(self):
        """Identical pledges for 2 rounds should trigger early termination."""
        game = self._make_individual_game()
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        pledges = {
            "Agent_1": [5.0, 3.0, 2.0],
            "Agent_2": [4.0, 6.0, 1.0],
        }
        state["round_pledges"] = [pledges, pledges]
        assert game.check_early_termination(state) is True

    def test_no_early_termination_different_pledges(self):
        """Different pledges across rounds should NOT trigger termination."""
        game = self._make_individual_game()
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        round1 = {"Agent_1": [5.0, 3.0, 2.0], "Agent_2": [4.0, 6.0, 1.0]}
        round2 = {"Agent_1": [6.0, 2.0, 2.0], "Agent_2": [4.0, 6.0, 1.0]}
        state["round_pledges"] = [round1, round2]
        assert game.check_early_termination(state) is False

    def test_no_early_termination_one_round(self):
        """Only 1 round of history should NOT trigger termination."""
        game = self._make_individual_game()
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        pledges = {"Agent_1": [5.0, 3.0, 2.0], "Agent_2": [4.0, 6.0, 1.0]}
        state["round_pledges"] = [pledges]
        assert game.check_early_termination(state) is False

    def test_early_termination_within_tolerance(self):
        """Pledges within atol=0.01 should trigger termination."""
        game = self._make_individual_game()
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        round1 = {"Agent_1": [5.0, 3.0, 2.0], "Agent_2": [4.0, 6.0, 1.0]}
        round2 = {"Agent_1": [5.005, 3.003, 2.002], "Agent_2": [4.001, 6.002, 1.004]}
        state["round_pledges"] = [round1, round2]
        assert game.check_early_termination(state) is True


class TestSeedReproducibility:
    """Tests for reproducibility."""

    def test_same_seed_same_state(self):
        """Same seed should produce identical game states."""
        game1 = make_game(seed=123)
        game2 = make_game(seed=123)
        agents = create_test_agents(2)
        state1 = game1.create_game_state(agents)
        state2 = game2.create_game_state(agents)

        assert state1["project_costs"] == state2["project_costs"]
        np.testing.assert_allclose(
            state1["agent_valuations"]["Agent_1"],
            state2["agent_valuations"]["Agent_1"],
            atol=1e-6,
        )

    def test_different_seed_different_state(self):
        """Different seeds should produce different game states."""
        game1 = make_game(seed=123)
        game2 = make_game(seed=456)
        agents = create_test_agents(2)
        state1 = game1.create_game_state(agents)
        state2 = game2.create_game_state(agents)

        # Costs should differ (with overwhelming probability)
        assert state1["project_costs"] != state2["project_costs"]


class TestUpdateGameState:
    """Tests for update_game_state_with_pledges and compute_final_outcome."""

    def test_update_aggregates(self):
        """Aggregate totals should be sum of all agent contributions."""
        game = make_game(m_projects=3)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        pledges = {
            "Agent_1": {"contributions": [10.0, 5.0, 0.0]},
            "Agent_2": {"contributions": [5.0, 10.0, 3.0]},
        }
        game.update_game_state_with_pledges(state, pledges)

        assert state["aggregate_totals"] == [15.0, 15.0, 3.0]

    def test_update_funded_projects(self):
        """Projects meeting cost threshold should be marked funded."""
        game = make_game(m_projects=3)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        # Set known costs
        state["project_costs"] = [10.0, 20.0, 30.0]

        pledges = {
            "Agent_1": {"contributions": [6.0, 5.0, 0.0]},
            "Agent_2": {"contributions": [5.0, 15.0, 10.0]},
        }
        game.update_game_state_with_pledges(state, pledges)

        # Project 0: 11.0 >= 10.0 -> funded
        # Project 1: 20.0 >= 20.0 -> funded
        # Project 2: 10.0 < 30.0 -> not funded
        assert 0 in state["funded_projects"]
        assert 1 in state["funded_projects"]
        assert 2 not in state["funded_projects"]

    def test_compute_final_outcome_utilities(self):
        """Final outcome should compute correct utilities."""
        game = make_game(m_projects=3)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        state["agent_valuations"] = {
            "Agent_1": [40.0, 30.0, 30.0],
            "Agent_2": [20.0, 50.0, 30.0],
        }
        state["project_costs"] = [10.0, 20.0, 30.0]

        pledges = {
            "Agent_1": {"contributions": [6.0, 12.0, 0.0]},
            "Agent_2": {"contributions": [5.0, 8.0, 0.0]},
        }
        game.update_game_state_with_pledges(state, pledges)

        outcome = game.compute_final_outcome(state)

        # Funded: [0, 1] (11>=10, 20>=20)
        # Agent_1: (40-6) + (30-12) = 34 + 18 = 52
        # Agent_2: (20-5) + (50-8) = 15 + 42 = 57
        assert abs(outcome["utilities"]["Agent_1"] - 52.0) < 0.01
        assert abs(outcome["utilities"]["Agent_2"] - 57.0) < 0.01


class TestPrompts:
    """Test that prompts are generated without errors."""

    def test_game_rules_prompt(self):
        game = make_game()
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        prompt = game.get_game_rules_prompt(state)
        assert "Participatory Budgeting" in prompt
        assert "Talk-Pledge-Revise" in prompt

    def test_preference_assignment_prompt(self):
        game = make_game()
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        prompt = game.get_preference_assignment_prompt("Agent_1", state)
        assert "BUDGET" in prompt
        assert "Agent_1" in prompt

    def test_proposal_prompt(self):
        game = make_game()
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        prompt = game.get_proposal_prompt("Agent_1", state, 1, ["Agent_1", "Agent_2"])
        assert "contributions" in prompt

    def test_discussion_prompt(self):
        game = make_game()
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        prompt = game.get_discussion_prompt("Agent_1", state, 1, 5, [])
        assert "DISCUSSION" in prompt

    def test_thinking_prompt(self):
        game = make_game()
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        prompt = game.get_thinking_prompt("Agent_1", state, 1, 5, [])
        assert "STRATEGIC" in prompt

    def test_reflection_prompt(self):
        game = make_game()
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        state["funded_projects"] = []
        prompt = game.get_reflection_prompt("Agent_1", state, 1, 5, {})
        assert "Reflect" in prompt

    def test_feedback_prompt(self):
        game = make_game()
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        state["funded_projects"] = []
        prompt = game.get_feedback_prompt("Agent_1", state)
        assert "Aggregate" in prompt

    def test_format_proposal_display(self):
        game = make_game(m_projects=3)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        proposal = {"contributions": [5.0, 3.0, 2.0], "proposed_by": "Agent_1", "reasoning": "test"}
        display = game.format_proposal_display(proposal, state)
        assert "PLEDGE" in display
        assert "Agent_1" in display

    def test_agent_preferences_summary(self):
        game = make_game()
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        summary = game.get_agent_preferences_summary("Agent_1", state)
        assert "valuations" in summary
        assert "budget" in summary
        assert "projects" in summary


class TestJointPledgeMode:
    """Tests for joint pledge mode (Change 4)."""

    def _make_joint_game(self, m_projects=3, pledge_mode="joint"):
        from game_environments import CoFundingConfig
        from game_environments.co_funding import CoFundingGame
        config = CoFundingConfig(
            n_agents=2, t_rounds=5, m_projects=m_projects,
            alpha=0.5, sigma=0.5, random_seed=42,
            pledge_mode=pledge_mode,
        )
        return CoFundingGame(config)

    def test_game_state_has_pledge_mode(self):
        """Game state should include pledge_mode."""
        game = self._make_joint_game()
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        assert state["pledge_mode"] == "joint"

    def test_parse_joint_proposal(self):
        """Valid joint pledge should be parsed correctly."""
        game = self._make_joint_game(m_projects=3)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        response = json.dumps({
            "contributions": {
                "Agent_1": [5.0, 3.0, 2.0],
                "Agent_2": [4.0, 6.0, 1.0],
            },
            "reasoning": "test joint plan",
        })
        parsed = game.parse_proposal(response, "Agent_1", state, ["Agent_1", "Agent_2"])

        # Self-assignment should be Agent_1's row
        assert parsed["contributions"] == [5.0, 3.0, 2.0]
        assert parsed["proposed_by"] == "Agent_1"
        assert "joint_plan" in parsed
        assert parsed["joint_plan"]["Agent_1"] == [5.0, 3.0, 2.0]
        assert parsed["joint_plan"]["Agent_2"] == [4.0, 6.0, 1.0]

    def test_validate_joint_proposal_valid(self):
        """Valid joint proposal should pass validation."""
        game = self._make_joint_game(m_projects=3)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        budget = state["agent_budgets"]["Agent_1"]

        proposal = {
            "contributions": [budget * 0.2, budget * 0.3, 0.0],
            "joint_plan": {
                "Agent_1": [budget * 0.2, budget * 0.3, 0.0],
                "Agent_2": [budget * 0.1, budget * 0.4, budget * 0.1],
            },
            "proposed_by": "Agent_1",
        }
        assert game.validate_proposal(proposal, state) is True

    def test_validate_joint_proposal_over_budget(self):
        """Joint plan with over-budget entry should fail."""
        game = self._make_joint_game(m_projects=3)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        budget = state["agent_budgets"]["Agent_2"]

        proposal = {
            "contributions": [1.0, 1.0, 1.0],
            "joint_plan": {
                "Agent_1": [1.0, 1.0, 1.0],
                "Agent_2": [budget, budget, budget],  # way over budget
            },
            "proposed_by": "Agent_1",
        }
        assert game.validate_proposal(proposal, state) is False

    def test_early_termination_joint_agreement(self):
        """Agents with identical joint plans should trigger early termination."""
        game = self._make_joint_game(m_projects=3)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        agreed_plan = {
            "Agent_1": [5.0, 3.0, 2.0],
            "Agent_2": [4.0, 6.0, 1.0],
        }
        state["joint_plans"] = {
            "Agent_1": agreed_plan,
            "Agent_2": agreed_plan,
        }
        assert game.check_early_termination(state) is True

    def test_early_termination_joint_disagreement(self):
        """Agents with different joint plans should NOT trigger termination."""
        game = self._make_joint_game(m_projects=3)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        state["joint_plans"] = {
            "Agent_1": {
                "Agent_1": [5.0, 3.0, 2.0],
                "Agent_2": [4.0, 6.0, 1.0],
            },
            "Agent_2": {
                "Agent_1": [3.0, 5.0, 2.0],  # differs from Agent_1's plan
                "Agent_2": [4.0, 6.0, 1.0],
            },
        }
        assert game.check_early_termination(state) is False

    def test_proposal_prompt_joint_mode(self):
        """Joint mode prompt should mention dictionary format."""
        game = self._make_joint_game(m_projects=3)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        prompt = game.get_proposal_prompt("Agent_1", state, 1, ["Agent_1", "Agent_2"])
        assert "JOINT FUNDING PLAN" in prompt
        assert "Agent_1" in prompt
        assert "Agent_2" in prompt

    def test_legacy_individual_mode(self):
        """pledge_mode='individual' should preserve old behavior."""
        game = self._make_joint_game(m_projects=3, pledge_mode="individual")
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        assert state["pledge_mode"] == "individual"

        # Prompt should NOT mention joint plan
        prompt = game.get_proposal_prompt("Agent_1", state, 1, ["Agent_1", "Agent_2"])
        assert "JOINT FUNDING PLAN" not in prompt
        assert "contributions" in prompt

        # Parse individual-format response
        response = '{"contributions": [5.0, 3.0, 2.0], "reasoning": "test"}'
        parsed = game.parse_proposal(response, "Agent_1", state, ["Agent_1", "Agent_2"])
        assert parsed["contributions"] == [5.0, 3.0, 2.0]
        assert "joint_plan" not in parsed

        # Early termination uses 2-consecutive-round logic
        pledges = {"Agent_1": [5.0, 3.0, 2.0], "Agent_2": [4.0, 6.0, 1.0]}
        state["round_pledges"] = [pledges, pledges]
        assert game.check_early_termination(state) is True

    def test_invalid_pledge_mode_raises(self):
        """Invalid pledge_mode should raise ValueError."""
        from game_environments import CoFundingConfig
        with pytest.raises(ValueError, match="pledge_mode"):
            CoFundingConfig(
                n_agents=2, t_rounds=5, m_projects=3,
                pledge_mode="invalid",
            )

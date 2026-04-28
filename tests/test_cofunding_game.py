#!/usr/bin/env python3
"""
Tests for CoFundingGame core logic.

Tests cover:
- Valuation generation (sum=100, non-negative, cosine similarity)
- Game state creation (all required keys, budgets, costs)
- Utility calculation (funded vs unfunded, edge cases)
- Proposal parsing and validation
- Protocol type
- Joint proposal preparation and voting prompt generation
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


def make_game(
    seed=42,
    alpha=0.5,
    sigma=0.5,
    m_projects=5,
    n_agents=2,
    t_rounds=5,
    discussion_transparency="own",
    enable_commit_vote=True,
    enable_time_discount=True,
    gamma_discount=0.9,
):
    config = CoFundingConfig(
        n_agents=n_agents,
        t_rounds=t_rounds,
        m_projects=m_projects,
        alpha=alpha,
        sigma=sigma,
        discussion_transparency=discussion_transparency,
        enable_commit_vote=enable_commit_vote,
        enable_time_discount=enable_time_discount,
        gamma_discount=gamma_discount,
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
            assert sum(vals) == 100, f"{aid} valuations sum to {sum(vals)}"
            assert all(float(v).is_integer() for v in vals), (
                f"{aid} valuations are not all integers: {vals}"
            )

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
            assert sum(vals) == 100
            assert all(float(v).is_integer() for v in vals)
            assert all(v >= -1e-9 for v in vals)
            vecs.append(np.array(vals))
        # Verify ALL pairwise cosine similarities match alpha
        for i in range(3):
            for j in range(i + 1, 3):
                cos_sim = np.dot(vecs[i], vecs[j]) / (np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[j]))
                assert abs(cos_sim - 0.5) < 0.05, (
                    f"Pair ({i},{j}): expected cos_sim=0.5, got {cos_sim:.4f}"
                )

    @pytest.mark.parametrize("alpha", [0.2, 0.8])
    def test_10agent_valuations_match_alpha_for_full_batch_shape(self, alpha):
        """Full-batch Game 3 shape should keep all-pairs cosine near alpha."""
        game = make_game(alpha=alpha, n_agents=10, m_projects=25, seed=42)
        agents = create_test_agents(10)
        state = game.create_game_state(agents)
        vecs = [
            np.array(state["agent_valuations"][f"Agent_{idx}"], dtype=float)
            for idx in range(1, 11)
        ]
        errors = []
        for i in range(10):
            assert sum(vecs[i]) == 100
            assert all(float(value).is_integer() for value in vecs[i])
            for j in range(i + 1, 10):
                cos_sim = np.dot(vecs[i], vecs[j]) / (
                    np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[j])
                )
                errors.append(abs(cos_sim - alpha))
        assert max(errors) < 0.02


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
            "accepted_proposal",
        ]
        for key in required_keys:
            assert key in state, f"Missing key: {key}"

    def test_budget_scales_between_half_and_full_cost(self):
        """Total budget should track sigma * total_cost after integer equal-budget rounding."""
        game = make_game(sigma=0.6)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        expected_budget = 0.6 * state["total_cost"]
        assert abs(state["total_budget"] - expected_budget) <= (len(agents) / 2) + 0.1

    def test_per_agent_budget_equal(self):
        """Each agent should get equal budget."""
        game = make_game(n_agents=3)
        agents = create_test_agents(3)
        state = game.create_game_state(agents)
        budgets = list(state["agent_budgets"].values())
        assert abs(budgets[0] - budgets[1]) < 0.01
        assert abs(budgets[1] - budgets[2]) < 0.01
        assert all(float(b).is_integer() for b in budgets)
        assert float(state["total_budget"]).is_integer()
        assert sum(int(b) for b in budgets) == state["total_budget"]

    def test_project_costs_in_range(self):
        """Project costs should be integer-valued and within [c_min, c_max]."""
        game = make_game()
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        for project, cost in zip(state["projects"], state["project_costs"]):
            assert 10.0 <= cost <= 30.0
            assert float(cost).is_integer()
            assert float(project["cost"]).is_integer()
            assert project["cost"] == cost
        assert float(state["total_cost"]).is_integer()
        assert sum(int(cost) for cost in state["project_costs"]) == state["total_cost"]

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

    def test_discount_factor_applied_by_default(self):
        """Utility should decrease over rounds when time discounting is enabled."""
        game = make_game(m_projects=3)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        state["funded_projects"] = [0]

        proposal = {"contributions": [5.0, 0.0, 0.0], "proposed_by": "Agent_1"}
        u1 = game.calculate_utility("Agent_1", proposal, state, 1)
        u5 = game.calculate_utility("Agent_1", proposal, state, 5)
        u10 = game.calculate_utility("Agent_1", proposal, state, 10)
        assert u1 > u5 > u10

    def test_time_discount_can_be_disabled(self):
        """Utility should be round-invariant when time discounting is disabled."""
        game = make_game(m_projects=3, enable_time_discount=False)
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
        assert parsed["raw_response"] == response
        assert parsed["parse_error"]["type"] == "ValueError"
        assert parsed["parse_error"]["message"] == "No JSON found in response"

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
        """Co-funding should return 'propose_and_vote'."""
        game = make_game()
        assert game.get_protocol_type() == "propose_and_vote"

    def test_game_type(self):
        """Should return CO_FUNDING enum."""
        game = make_game()
        assert game.get_game_type() == GameType.CO_FUNDING

    def test_voting_prompt_renders_joint_proposal(self):
        """Voting prompt should describe the aggregated joint proposal."""
        game = make_game(m_projects=3)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        proposal = {
            "contributions_by_agent": {
                "Agent_1": [5.0, 3.0, 2.0],
                "Agent_2": [4.0, 6.0, 1.0],
            },
            "aggregate_totals": [9.0, 9.0, 3.0],
            "funded_projects": [],
            "proposed_by": "system",
        }
        prompt = game.get_voting_prompt("Agent_1", proposal, state, 1)
        assert "JOINT FUNDING PROPOSAL" in prompt
        assert '"vote": "accept"' in prompt


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
        state["agent_budgets"] = {"Agent_1": 100.0, "Agent_2": 100.0}

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
        state["agent_budgets"] = {"Agent_1": 100.0, "Agent_2": 100.0}

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

    def test_update_scales_over_budget_individual_pledge(self):
        """State update should scale an over-budget pledge instead of failing."""
        game = make_game(m_projects=3)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        budget = state["agent_budgets"]["Agent_1"]

        pledges = {
            "Agent_1": {"contributions": [budget, budget, budget]},
            "Agent_2": {"contributions": [0.0, 0.0, 0.0]},
        }
        game.update_game_state_with_pledges(state, pledges)

        corrected = state["current_pledges"]["Agent_1"]
        assert sum(corrected["contributions"]) == pytest.approx(budget, abs=1e-6)
        assert corrected["auto_corrected"]["reason"] == "scaled_to_budget"
        assert corrected["auto_corrected"]["original_total"] == pytest.approx(budget * 3, abs=1e-6)
        assert state["round_pledges"][-1]["Agent_1"] == corrected["contributions"]

    def test_update_scales_over_budget_joint_plan(self):
        """State update should sanitize joint-plan rows to each agent's budget."""
        game = make_game(m_projects=3)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        budget_1 = state["agent_budgets"]["Agent_1"]
        budget_2 = state["agent_budgets"]["Agent_2"]

        pledges = {
            "Agent_1": {
                "contributions": [1.0, 1.0, 1.0],
                "joint_plan": {
                    "Agent_1": [budget_1, budget_1, budget_1],
                    "Agent_2": [budget_2, budget_2, budget_2],
                },
            },
            "Agent_2": {"contributions": [0.0, 0.0, 0.0]},
        }
        game.update_game_state_with_pledges(state, pledges)

        corrected = state["current_pledges"]["Agent_1"]
        assert sum(corrected["contributions"]) == pytest.approx(budget_1, abs=1e-6)
        assert corrected["joint_plan"]["Agent_1"] == corrected["contributions"]
        assert sum(corrected["joint_plan"]["Agent_2"]) == pytest.approx(budget_2, abs=1e-6)
        assert corrected["joint_plan_auto_corrected"]["Agent_1"]["reason"] == "scaled_to_budget"
        assert corrected["joint_plan_auto_corrected"]["Agent_2"]["reason"] == "scaled_to_budget"
        assert state["joint_plans"]["Agent_1"] == corrected["joint_plan"]

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
        state["agent_budgets"] = {"Agent_1": 100.0, "Agent_2": 100.0}

        pledges = {
            "Agent_1": {"contributions": [6.0, 12.0, 0.0]},
            "Agent_2": {"contributions": [5.0, 8.0, 0.0]},
        }
        prepared = game.prepare_proposals_for_voting(
            [
                {"contributions": [6.0, 12.0, 0.0], "proposed_by": "Agent_1"},
                {"contributions": [5.0, 8.0, 0.0], "proposed_by": "Agent_2"},
            ],
            state,
            1,
        )
        game.record_accepted_proposal(state, prepared[0])

        outcome = game.compute_final_outcome(state)

        # Funded: [0, 1] (11>=10, 20>=20)
        # Agent_1: (40-6) + (30-12) = 34 + 18 = 52
        # Agent_2: (20-5) + (50-8) = 15 + 42 = 57
        assert abs(outcome["utilities"]["Agent_1"] - 52.0) < 0.01
        assert abs(outcome["utilities"]["Agent_2"] - 57.0) < 0.01

    def test_compute_final_outcome_without_accepted_proposal_is_zero(self):
        """No accepted joint proposal should yield zero utility."""
        game = make_game(m_projects=3)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        outcome = game.compute_final_outcome(state)

        assert outcome["funded_projects"] == []
        assert outcome["utilities"]["Agent_1"] == 0.0
        assert outcome["utilities"]["Agent_2"] == 0.0

    def test_prepare_proposals_for_voting_builds_single_joint_proposal(self):
        """Raw per-agent proposals should be aggregated into one joint proposal."""
        game = make_game(m_projects=3)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        prepared = game.prepare_proposals_for_voting(
            [
                {"contributions": [5.0, 3.0, 2.0], "proposed_by": "Agent_1"},
                {"contributions": [4.0, 6.0, 1.0], "proposed_by": "Agent_2"},
            ],
            state,
            1,
        )

        assert len(prepared) == 1
        assert prepared[0]["contributions_by_agent"]["Agent_1"] == [5.0, 3.0, 2.0]
        assert prepared[0]["contributions_by_agent"]["Agent_2"] == [4.0, 6.0, 1.0]
        assert prepared[0]["aggregate_totals"] == [9.0, 9.0, 3.0]

    def test_prepare_proposals_for_voting_scales_over_budget_vectors(self):
        """Joint proposal preparation should return sanitized per-agent contributions."""
        game = make_game(m_projects=3)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        budget_1 = state["agent_budgets"]["Agent_1"]
        budget_2 = state["agent_budgets"]["Agent_2"]

        prepared = game.prepare_proposals_for_voting(
            [
                {"contributions": [budget_1, budget_1, budget_1], "proposed_by": "Agent_1"},
                {"contributions": [budget_2, 0.0, 0.0], "proposed_by": "Agent_2"},
            ],
            state,
            1,
        )

        assert len(prepared) == 1
        assert sum(prepared[0]["contributions_by_agent"]["Agent_1"]) == pytest.approx(budget_1, abs=1e-6)
        assert prepared[0]["contributions_by_agent"]["Agent_1"] == state["current_pledges"]["Agent_1"]["contributions"]

    def test_record_accepted_proposal_scales_over_budget_vectors(self):
        """Accepted proposal storage should sanitize contributions before final outcome."""
        game = make_game(m_projects=3)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        state["project_costs"] = [10.0, 20.0, 30.0]
        budget_1 = state["agent_budgets"]["Agent_1"]
        budget_2 = state["agent_budgets"]["Agent_2"]

        proposal = {
            "contributions_by_agent": {
                "Agent_1": [budget_1, budget_1, budget_1],
                "Agent_2": [budget_2, 0.0, 0.0],
            },
            "aggregate_totals": [999.0, 999.0, 999.0],
            "funded_projects": [0, 1, 2],
        }

        game.record_accepted_proposal(state, proposal)

        accepted = state["accepted_proposal"]
        assert sum(accepted["contributions_by_agent"]["Agent_1"]) == pytest.approx(budget_1, abs=1e-6)
        assert sum(accepted["contributions_by_agent"]["Agent_2"]) == pytest.approx(budget_2, abs=1e-6)
        assert accepted["aggregate_totals"] != [999.0, 999.0, 999.0]
        assert accepted["auto_corrected"]["Agent_1"]["reason"] == "scaled_to_budget"


class TestPrompts:
    """Test that prompts are generated without errors."""

    def test_game_rules_prompt(self):
        game = make_game()
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        prompt = game.get_game_rules_prompt(state)
        assert "Participatory Budgeting" in prompt
        assert "Propose-and-Vote" in prompt

    def test_game_rules_prompt_omits_trailing_point_zero_zero_for_integer_project_costs(self):
        game = make_game(m_projects=3)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        state["project_costs"] = [10, 20, 30]
        for project, cost in zip(state["projects"], state["project_costs"]):
            project["cost"] = cost

        prompt = game.get_game_rules_prompt(state)

        assert "cost = 10" in prompt
        assert "cost = 20" in prompt
        assert "cost = 30" in prompt
        assert "cost = 10.00" not in prompt
        assert "cost = 20.00" not in prompt
        assert "cost = 30.00" not in prompt

    def test_preference_assignment_prompt(self):
        game = make_game()
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        prompt = game.get_preference_assignment_prompt("Agent_1", state)
        assert "BUDGET" in prompt
        assert "Agent_1" in prompt

    def test_cofunding_uses_combined_setup_phase(self):
        """Co-funding should merge private preferences into setup."""
        game = make_game()
        assert game.uses_combined_setup_phase() is True

    def test_combined_setup_prompt_contains_rules_budget_and_valuations(self):
        """Combined setup prompt should include both shared and private information."""
        game = make_game()
        agents = create_test_agents(2)
        state = game.create_game_state(agents)

        prompt = game.get_combined_setup_prompt("Agent_1", state)

        assert "GAME STRUCTURE" in prompt
        assert "HOW IT WORKS" in prompt
        assert "YOUR PRIVATE BUDGET" in prompt
        assert "PROJECT DETAILS AND YOUR VALUATIONS" in prompt
        assert "Please do not initiate the discussion or proposal phase yet." in prompt
        assert "summarize the game structure and rules" in prompt
        assert (
            "reiterate the private budget that was assigned to you, along with the project costs and your project valuations"
            in prompt
        )
        budget_text = game._format_display_number(state["agent_budgets"]["Agent_1"])
        assert f"**YOUR PRIVATE BUDGET:** {budget_text} " in prompt

        for val in state["agent_valuations"]["Agent_1"]:
            assert f"Your valuation = {game._format_display_number(val)}" in prompt

    def test_preference_assignment_prompt_omits_trailing_point_zero_zero_for_integer_budget_valuations_and_costs(self):
        game = make_game()
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        state["agent_budgets"]["Agent_1"] = 27
        state["total_budget"] = 54
        state["agent_valuations"]["Agent_1"] = [40, 30, 20, 10, 0]
        state["project_costs"] = [10, 20, 30, 40, 50]
        for project, cost in zip(state["projects"], state["project_costs"]):
            project["cost"] = cost
        state["total_cost"] = 150

        prompt = game.get_preference_assignment_prompt("Agent_1", state)

        assert "**YOUR PRIVATE BUDGET:** 27 " in prompt
        assert "(cost: 10): Your valuation = 40" in prompt
        assert "Your valuation = 40 (" not in prompt
        assert "**TOTAL VALUATIONS:** 100" in prompt
        assert "**TOTAL PROJECT COSTS:** 150" in prompt
        assert "**YOUR PRIVATE BUDGET:** 27.00" not in prompt
        assert "(cost: 10.00):" not in prompt
        assert "Your valuation = 40.00" not in prompt
        assert "**TOTAL VALUATIONS:** 100.00" not in prompt
        assert "**TOTAL PROJECT COSTS:** 150.00" not in prompt

    def test_proposal_prompt(self):
        game = make_game()
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        prompt = game.get_proposal_prompt("Agent_1", state, 1, ["Agent_1", "Agent_2"])
        assert "contributions" in prompt

    def test_proposal_prompt_omits_trailing_point_zero_zero_for_integer_budget_valuations_and_costs(self):
        game = make_game(m_projects=3)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        state["agent_budgets"]["Agent_1"] = 27
        state["agent_valuations"]["Agent_1"] = [40, 30, 30]
        state["project_costs"] = [10, 20, 30]
        for project, cost in zip(state["projects"], state["project_costs"]):
            project["cost"] = cost

        prompt = game.get_proposal_prompt("Agent_1", state, 1, ["Agent_1", "Agent_2"])

        assert "**YOUR FIXED TOTAL BUDGET:** 27" in prompt
        assert "cost=10, your_val=40" in prompt
        assert "your_val=40" in prompt
        assert "your budget (27)" in prompt
        assert "**YOUR FIXED TOTAL BUDGET:** 27.00" not in prompt
        assert "cost=10.00, your_val=40" not in prompt
        assert "your_val=40.00" not in prompt
        assert "your budget (27.00)" not in prompt

    def test_discussion_prompt(self):
        game = make_game()
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        prompt = game.get_discussion_prompt("Agent_1", state, 1, 5, [])
        assert "DISCUSSION" in prompt
        assert "PREVIOUS ROUND PROPOSAL SNAPSHOT" in prompt

    def test_discussion_prompt_omits_trailing_point_zero_zero_for_integer_budgets_and_costs(self):
        game = make_game(m_projects=3)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        state["agent_budgets"]["Agent_1"] = 27
        state["agent_budgets"]["Agent_2"] = 28
        state["project_costs"] = [10, 20, 30]
        for project, cost in zip(state["projects"], state["project_costs"]):
            project["cost"] = cost

        prompt = game.get_discussion_prompt("Agent_1", state, 1, 5, [])

        assert "budget=27" in prompt
        assert "budget=28" in prompt
        assert "/ cost=10" in prompt
        assert "budget=27.00" not in prompt
        assert "budget=28.00" not in prompt
        assert "/ cost=10.00" not in prompt

    def test_discussion_prompt_aggregate_mode(self):
        game = make_game(discussion_transparency="aggregate")
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        prompt = game.get_discussion_prompt("Agent_1", state, 1, 5, [])
        assert "PREVIOUS ROUND PROPOSAL SNAPSHOT" not in prompt

    def test_discussion_prompt_with_own_transparency(self):
        config = CoFundingConfig(
            n_agents=2,
            t_rounds=5,
            m_projects=3,
            discussion_transparency="own",
            random_seed=42,
        )
        game = CoFundingGame(config)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        pledges = {
            "Agent_1": {"contributions": [5.0, 3.0, 2.0]},
            "Agent_2": {"contributions": [4.0, 6.0, 1.0]},
        }
        game.update_game_state_with_pledges(state, pledges)
        prompt = game.get_discussion_prompt("Agent_1", state, 2, 5, [])
        assert "PREVIOUS ROUND PROPOSAL SNAPSHOT" in prompt
        assert "your_prev_proposed=" in prompt
        assert "others_prev_proposed=" in prompt

    def test_thinking_prompt(self):
        game = make_game()
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        prompt = game.get_thinking_prompt("Agent_1", state, 1, 5, [])
        assert "STRATEGIC" in prompt
        assert "**YOUR TOP PRIORITIES:**" not in prompt
        assert "**YOUR FULL PREFERENCE REMINDER:**" in prompt
        assert '"key_priorities"' in prompt
        assert '"potential_concessions"' in prompt

        for project, valuation in zip(
            state["projects"],
            state["agent_valuations"]["Agent_1"],
        ):
            assert (
                f"{project['name']} "
                f"(val={game._format_display_number(valuation)}, cost={game._format_display_number(project['cost'])})"
            ) in prompt

    def test_thinking_prompt_omits_trailing_point_zero_zero_for_integer_budget_valuations_and_costs(self):
        game = make_game(m_projects=3)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        state["agent_budgets"]["Agent_1"] = 27
        state["agent_valuations"]["Agent_1"] = [40, 30, 30]
        state["project_costs"] = [10, 20, 30]
        for project, cost in zip(state["projects"], state["project_costs"]):
            project["cost"] = cost

        prompt = game.get_thinking_prompt("Agent_1", state, 1, 5, [])

        assert "- Fixed total budget for the whole game: 27" in prompt
        assert "**YOUR TOP PRIORITIES:**" not in prompt
        assert "**YOUR FULL PREFERENCE REMINDER:**" in prompt
        for project, valuation, cost in zip(
            state["projects"],
            state["agent_valuations"]["Agent_1"],
            state["project_costs"],
        ):
            assert f"{project['name']} (val={valuation}, cost={cost})" in prompt
        assert "- Fixed total budget for the whole game: 27.00" not in prompt
        assert "cost=10.00" not in prompt
        assert "val=40.00" not in prompt

    def test_reflection_prompt(self):
        game = make_game()
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        state["funded_projects"] = []
        prompt = game.get_reflection_prompt("Agent_1", state, 1, 5, {})
        assert "Reflect" in prompt
        assert "Vote outcome this round" in prompt

    def test_format_proposal_display(self):
        game = make_game(m_projects=3)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        proposal = {"contributions": [5.0, 3.0, 2.0], "proposed_by": "Agent_1", "reasoning": "test"}
        display = game.format_proposal_display(proposal, state)
        assert "PROPOSAL" in display
        assert "Agent_1" in display

    def test_format_joint_proposal_display(self):
        game = make_game(m_projects=3)
        agents = create_test_agents(2)
        state = game.create_game_state(agents)
        proposal = {
            "contributions_by_agent": {
                "Agent_1": [5.0, 3.0, 2.0],
                "Agent_2": [4.0, 6.0, 1.0],
            },
            "aggregate_totals": [9.0, 9.0, 3.0],
            "funded_projects": [],
            "proposed_by": "system",
        }
        display = game.format_proposal_display(proposal, state)
        assert "JOINT PROPOSAL" in display
        assert "Agent_1" in display
        assert "Agent_2" in display

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
        """pledge_mode='individual' should preserve per-agent proposal behavior."""
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

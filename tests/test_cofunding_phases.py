#!/usr/bin/env python3
"""
Tests for co-funding phase handlers and orchestrator integration.

Tests cover:
- Pledge submission phase returns correct structure
- Feedback phase computes correct aggregates
- Early termination triggers after 2 identical rounds
- Full round loop runs without error with mock agents
- Game state is correctly mutated across rounds
"""

import asyncio
import json
import pytest
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from game_environments import CoFundingConfig
from game_environments.co_funding import CoFundingGame


# ---- Fake agent for testing ----

@dataclass
class FakeAgentResponse:
    """Mimics the response object returned by agent.generate_response()."""
    content: str
    metadata: Optional[Dict[str, Any]] = None
    tokens_used: Optional[int] = None


class FakeAgent:
    """
    Fake agent that returns predetermined JSON pledge responses.
    Implements the minimal interface needed by PhaseHandler.
    """
    def __init__(self, agent_id: str, pledge_responses: List[str] = None):
        self.agent_id = agent_id
        self._pledge_responses = pledge_responses or []
        self._call_count = 0
        self._max_tokens = None

    async def generate_response(self, context, prompt) -> FakeAgentResponse:
        """Return the next predetermined response."""
        if self._call_count < len(self._pledge_responses):
            response = self._pledge_responses[self._call_count]
        else:
            # Default: zero contributions
            m = 5
            response = json.dumps({"contributions": [0.0] * m, "reasoning": "default"})
        self._call_count += 1
        return FakeAgentResponse(content=response)

    def update_max_tokens(self, max_tokens):
        self._max_tokens = max_tokens

    def get_model_info(self) -> Dict[str, Any]:
        return {"model_name": "fake-model"}

    async def think_strategy(self, prompt, context) -> Dict[str, Any]:
        return {
            "reasoning": "test reasoning",
            "strategy": "test strategy",
            "key_priorities": [],
            "potential_concessions": [],
        }


def make_game_and_state(n_agents=2, m_projects=3, seed=42, sigma=0.5, alpha=0.5):
    """Helper to create a game and its state."""
    config = CoFundingConfig(
        n_agents=n_agents,
        t_rounds=5,
        m_projects=m_projects,
        alpha=alpha,
        sigma=sigma,
        random_seed=seed,
    )
    game = CoFundingGame(config)
    agents = [FakeAgent(f"Agent_{i+1}") for i in range(n_agents)]
    state = game.create_game_state(agents)
    return game, agents, state


class TestPledgeSubmissionPhase:
    """Tests for the pledge submission flow."""

    def test_pledge_parse_and_validate(self):
        """Pledges should be parsed and validated correctly."""
        game, agents, state = make_game_and_state(m_projects=3)

        budget = state["agent_budgets"]["Agent_1"]
        response = json.dumps({
            "contributions": [budget * 0.3, budget * 0.3, budget * 0.2],
            "reasoning": "balanced",
        })

        parsed = game.parse_proposal(response, "Agent_1", state, ["Agent_1", "Agent_2"])
        assert len(parsed["contributions"]) == 3
        assert game.validate_proposal(parsed, state) is True

    def test_invalid_pledge_fallback(self):
        """Invalid pledges should be caught by validation."""
        game, agents, state = make_game_and_state(m_projects=3)

        # Way over budget
        budget = state["agent_budgets"]["Agent_1"]
        response = json.dumps({
            "contributions": [budget * 10, budget * 10, budget * 10],
            "reasoning": "too much",
        })

        parsed = game.parse_proposal(response, "Agent_1", state, ["Agent_1", "Agent_2"])
        assert game.validate_proposal(parsed, state) is False


class TestFeedbackPhase:
    """Tests for the feedback phase flow."""

    def test_feedback_computes_aggregates(self):
        """Feedback phase should correctly compute aggregate totals."""
        game, agents, state = make_game_and_state(m_projects=3)
        state["project_costs"] = [10.0, 20.0, 30.0]

        pledges = {
            "Agent_1": {"contributions": [5.0, 10.0, 0.0], "proposed_by": "Agent_1"},
            "Agent_2": {"contributions": [6.0, 10.0, 5.0], "proposed_by": "Agent_2"},
        }

        game.update_game_state_with_pledges(state, pledges)

        assert state["aggregate_totals"] == [11.0, 20.0, 5.0]
        assert 0 in state["funded_projects"]  # 11 >= 10
        assert 1 in state["funded_projects"]  # 20 >= 20
        assert 2 not in state["funded_projects"]  # 5 < 30

    def test_feedback_appends_history(self):
        """Each feedback round should append to round_pledges history."""
        game, agents, state = make_game_and_state(m_projects=3)

        pledges_r1 = {
            "Agent_1": {"contributions": [5.0, 5.0, 0.0]},
            "Agent_2": {"contributions": [3.0, 3.0, 0.0]},
        }
        pledges_r2 = {
            "Agent_1": {"contributions": [6.0, 4.0, 0.0]},
            "Agent_2": {"contributions": [4.0, 2.0, 0.0]},
        }

        game.update_game_state_with_pledges(state, pledges_r1)
        assert len(state["round_pledges"]) == 1

        game.update_game_state_with_pledges(state, pledges_r2)
        assert len(state["round_pledges"]) == 2


class TestEarlyTerminationIntegration:
    """Tests for early termination in the round loop context."""

    def test_early_termination_after_2_identical_rounds(self):
        """Early termination should trigger when pledges are identical for 2 rounds."""
        game, agents, state = make_game_and_state(m_projects=3)

        pledges = {
            "Agent_1": {"contributions": [5.0, 3.0, 2.0]},
            "Agent_2": {"contributions": [4.0, 6.0, 1.0]},
        }

        # Round 1
        game.update_game_state_with_pledges(state, pledges)
        assert not game.check_early_termination(state)

        # Round 2 (same pledges)
        game.update_game_state_with_pledges(state, pledges)
        assert game.check_early_termination(state)

    def test_no_termination_with_different_pledges(self):
        """No early termination when pledges change."""
        game, agents, state = make_game_and_state(m_projects=3)

        pledges_r1 = {
            "Agent_1": {"contributions": [5.0, 3.0, 2.0]},
            "Agent_2": {"contributions": [4.0, 6.0, 1.0]},
        }
        pledges_r2 = {
            "Agent_1": {"contributions": [6.0, 2.0, 2.0]},
            "Agent_2": {"contributions": [4.0, 6.0, 1.0]},
        }

        game.update_game_state_with_pledges(state, pledges_r1)
        game.update_game_state_with_pledges(state, pledges_r2)
        assert not game.check_early_termination(state)


class TestFullRoundLoop:
    """Test the full co-funding round loop logic (without actual LLM calls)."""

    def test_simulated_round_loop(self):
        """Simulate a full round loop and verify state mutation."""
        game, agents, state = make_game_and_state(m_projects=3, sigma=1.0)

        # Set predictable costs
        state["project_costs"] = [10.0, 15.0, 20.0]
        budget = state["agent_budgets"]["Agent_1"]

        for round_num in range(1, 4):
            # Simulate pledges
            pledges = {
                "Agent_1": {"contributions": [5.0 + round_num, 3.0, 2.0]},
                "Agent_2": {"contributions": [5.0, 3.0 + round_num, 2.0]},
            }

            game.update_game_state_with_pledges(state, pledges)

            # Check state is mutated
            assert len(state["round_pledges"]) == round_num
            assert len(state["aggregate_totals"]) == 3

        # Compute final outcome
        outcome = game.compute_final_outcome(state)
        assert "utilities" in outcome
        assert "funded_projects" in outcome
        assert "Agent_1" in outcome["utilities"]
        assert "Agent_2" in outcome["utilities"]

    def test_game_state_preserves_structure(self):
        """Game state should maintain all keys after multiple rounds."""
        game, agents, state = make_game_and_state(m_projects=3)

        required_keys = [
            "projects", "m_projects", "project_costs", "total_cost",
            "agent_budgets", "total_budget", "agent_valuations",
            "parameters", "game_type", "round_pledges",
            "current_pledges", "aggregate_totals", "funded_projects",
        ]

        # After initial creation
        for key in required_keys:
            assert key in state

        # After a round of pledges
        pledges = {
            "Agent_1": {"contributions": [1.0, 1.0, 1.0]},
            "Agent_2": {"contributions": [1.0, 1.0, 1.0]},
        }
        game.update_game_state_with_pledges(state, pledges)

        for key in required_keys:
            assert key in state


class TestFactoryIntegration:
    """Test that co_funding game can be created via the factory."""

    def test_create_via_factory(self):
        """create_game_environment should work for co_funding."""
        from game_environments import create_game_environment

        game = create_game_environment(
            game_type="co_funding",
            n_agents=2,
            t_rounds=5,
            m_projects=4,
            alpha=0.7,
            sigma=0.6,
        )
        assert game.get_protocol_type() == "talk_pledge_revise"
        assert game.get_game_type().value == "co_funding"

    def test_create_via_config(self):
        """create_game_from_config should work for CoFundingConfig."""
        from game_environments import create_game_from_config, CoFundingConfig

        config = CoFundingConfig(
            n_agents=3,
            t_rounds=10,
            m_projects=6,
            alpha=0.3,
            sigma=0.8,
            random_seed=42,
        )
        game = create_game_from_config(config)
        assert game.get_protocol_type() == "talk_pledge_revise"

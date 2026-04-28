#!/usr/bin/env python3
"""
Tests for co-funding phase handlers and orchestrator integration.

Tests cover:
- Proposal aggregation yields a single joint proposal for voting
- Voting/tabulation records accepted joint proposals
- Legacy early termination helper still behaves consistently
- Full round loop state mutations remain valid
- Game state is correctly mutated across rounds
"""

import asyncio
import json
import pytest
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from game_environments import CoFundingConfig
from game_environments.co_funding import CoFundingGame
from strong_models_experiment.phases.phase_handlers import PhaseHandler, VoteIntegrityError


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


class FakeThinkingAgent(FakeAgent):
    """Fake agent that returns a supplied private-thinking payload."""

    def __init__(self, agent_id: str, thinking_response: Dict[str, Any]):
        super().__init__(agent_id)
        self._thinking_response = dict(thinking_response)

    async def think_strategy(self, prompt, context) -> Dict[str, Any]:
        return dict(self._thinking_response)


def make_game_and_state(n_agents=2, m_projects=3, seed=42, sigma=0.5, alpha=0.5, pledge_mode="individual"):
    """Helper to create a game and its state."""
    config = CoFundingConfig(
        n_agents=n_agents,
        t_rounds=5,
        m_projects=m_projects,
        alpha=alpha,
        sigma=sigma,
        random_seed=seed,
        pledge_mode=pledge_mode,
    )
    game = CoFundingGame(config)
    agents = [FakeAgent(f"Agent_{i+1}") for i in range(n_agents)]
    state = game.create_game_state(agents)
    return game, agents, state


class TestProposalValidation:
    """Tests for co-funding proposal parsing and validation."""

    def test_run_private_thinking_phase_saves_raw_fallback_diagnostics(self):
        """Saved private-thinking interactions should retain raw fallback provenance."""
        game, _, state = make_game_and_state(n_agents=2, m_projects=3)
        raw_response = '{"reasoning": "missing comma" "strategy": "fallback"}'
        fallback_payload = {
            "reasoning": raw_response,
            "strategy": "Basic preference-driven approach",
            "key_priorities": [],
            "potential_concessions": [],
            "target_items": [],
            "anticipated_resistance": [],
            "raw_response": raw_response,
            "parse_error": {
                "type": "JSONDecodeError",
                "message": "Expecting ',' delimiter",
                "pos": 29,
            },
            "parsed_or_fallback_response": {
                "reasoning": raw_response,
                "strategy": "Basic preference-driven approach",
                "key_priorities": [],
                "potential_concessions": [],
                "target_items": [],
                "anticipated_resistance": [],
            },
            "used_fallback": True,
        }
        agents = [FakeThinkingAgent("Agent_1", fallback_payload)]
        preferences = {
            "agent_preferences": state["agent_valuations"],
            "game_state": state,
        }
        saved = []

        def save_interaction(*args, **kwargs):
            saved.append((args, kwargs))

        handler = PhaseHandler(
            save_interaction_callback=save_interaction,
            game_environment=game,
        )

        asyncio.run(
            handler.run_private_thinking_phase(
                agents=agents,
                items=state["projects"],
                preferences=preferences,
                round_num=1,
                max_rounds=game.config.t_rounds,
                discussion_messages=[],
            )
        )

        saved_response = json.loads(saved[0][0][3])
        assert saved[0][0][1] == "private_thinking_round_1"
        assert saved_response["used_fallback"] is True
        assert saved_response["raw_response"] == raw_response
        assert saved_response["parse_error"]["type"] == "JSONDecodeError"
        assert saved_response["parsed_or_fallback_response"]["strategy"] == "Basic preference-driven approach"

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

    def test_parse_failure_preserves_raw_response(self):
        """Parse failures should retain the raw response for later inspection."""
        game, agents, state = make_game_and_state(m_projects=3)

        response = "definitely not valid JSON"
        parsed = game.parse_proposal(response, "Agent_1", state, ["Agent_1", "Agent_2"])

        assert parsed["contributions"] == [0.0, 0.0, 0.0]
        assert parsed["raw_response"] == response
        assert parsed["parse_error"]["type"] == "ValueError"

    def test_repairs_literal_newlines_inside_json_strings(self):
        """Game 3 should parse model JSON with unescaped newlines in reasoning."""
        game, agents, state = make_game_and_state(m_projects=3)

        response = """{
          "contributions": [1.0, 2.0, 3.0],
          "reasoning": "First line.

Second line with detail."
        }"""
        parsed = game.parse_proposal(response, "Agent_1", state, ["Agent_1", "Agent_2"])

        assert parsed["contributions"] == [1.0, 2.0, 3.0]
        assert "parse_error" not in parsed
        assert "Second line" in parsed["reasoning"]

    def test_run_proposal_phase_saves_parse_failure_diagnostics(self):
        """Saved proposal interactions should include raw responses on parse failure."""
        game, _, state = make_game_and_state(m_projects=3)
        items = state["projects"]
        preferences = {
            "agent_preferences": state["agent_valuations"],
            "game_state": state,
        }
        bad_response = "not json"
        agents = [
            FakeAgent("Agent_1", [bad_response]),
            FakeAgent("Agent_2", [json.dumps({"contributions": [0.0, 0.0, 0.0], "reasoning": "ok"})]),
        ]
        saved = []

        def save_interaction(*args, **kwargs):
            saved.append((args, kwargs))

        handler = PhaseHandler(
            save_interaction_callback=save_interaction,
            game_environment=game,
        )

        asyncio.run(
            handler.run_proposal_phase(
                agents=agents,
                items=items,
                preferences=preferences,
                round_num=1,
                max_rounds=game.config.t_rounds,
            )
        )

        saved_response = json.loads(
            next(args[3] for args, _kwargs in saved if args[1] == "proposal_round_1_invalid_attempt_0")
        )
        assert saved_response["raw_response"] == bad_response
        assert saved_response["raw_proposal"] == bad_response
        assert saved_response["parse_error"]["type"] == "ValueError"

    def test_run_proposal_phase_saves_validation_failure_diagnostics(self):
        """Saved proposal interactions should keep the last invalid raw response."""
        game, _, state = make_game_and_state(m_projects=3)
        items = state["projects"]
        preferences = {
            "agent_preferences": state["agent_valuations"],
            "game_state": state,
        }
        budget = state["agent_budgets"]["Agent_1"]
        invalid_first = json.dumps({
            "contributions": [budget * 10, budget * 10, budget * 10],
            "reasoning": "too much",
        })
        invalid_retry = json.dumps({
            "contributions": [budget * 20, budget * 20, budget * 20],
            "reasoning": "still too much",
        })
        agents = [
            FakeAgent("Agent_1", [invalid_first, invalid_retry]),
            FakeAgent("Agent_2", [json.dumps({"contributions": [0.0, 0.0, 0.0], "reasoning": "ok"})]),
        ]
        saved = []

        def save_interaction(*args, **kwargs):
            saved.append((args, kwargs))

        handler = PhaseHandler(
            save_interaction_callback=save_interaction,
            game_environment=game,
        )

        asyncio.run(
            handler.run_proposal_phase(
                agents=agents,
                items=items,
                preferences=preferences,
                round_num=1,
                max_rounds=game.config.t_rounds,
            )
        )

        invalid_response = json.loads(
            next(args[3] for args, _kwargs in saved if args[1] == "proposal_round_1_invalid_attempt_0")
        )
        assert invalid_response["raw_response"] == invalid_first
        assert invalid_response["raw_proposal"] == invalid_first
        assert "over budget" in invalid_response["validation_error"]

        saved_response = json.loads(
            next(args[3] for args, _kwargs in saved if args[1] == "proposal_round_1")
        )
        assert saved_response["contributions"] == [0.0, 0.0, 0.0]
        assert saved_response["raw_response"] == invalid_retry
        assert saved_response["validation_error"] == "Proposal invalid after retry"


class TestJointProposalPreparation:
    """Tests for aggregation into the round's joint proposal."""

    def test_prepare_proposals_for_voting_computes_aggregates(self):
        """Joint proposal preparation should compute aggregate totals."""
        game, agents, state = make_game_and_state(m_projects=3)
        state["project_costs"] = [10.0, 20.0, 30.0]
        state["agent_budgets"] = {"Agent_1": 100.0, "Agent_2": 100.0}

        proposals = [
            {"contributions": [5.0, 10.0, 0.0], "proposed_by": "Agent_1"},
            {"contributions": [6.0, 10.0, 5.0], "proposed_by": "Agent_2"},
        ]

        prepared = game.prepare_proposals_for_voting(proposals, state, 1)

        assert len(prepared) == 1
        assert state["aggregate_totals"] == [11.0, 20.0, 5.0]
        assert 0 in state["funded_projects"]  # 11 >= 10
        assert 1 in state["funded_projects"]  # 20 >= 20
        assert 2 not in state["funded_projects"]  # 5 < 30
        assert prepared[0]["contributions_by_agent"]["Agent_1"] == [5.0, 10.0, 0.0]

    def test_prepare_proposals_for_voting_appends_history(self):
        """Each prepared round should append to round_pledges history."""
        game, agents, state = make_game_and_state(m_projects=3)

        proposals_r1 = [
            {"contributions": [5.0, 5.0, 0.0], "proposed_by": "Agent_1"},
            {"contributions": [3.0, 3.0, 0.0], "proposed_by": "Agent_2"},
        ]
        proposals_r2 = [
            {"contributions": [6.0, 4.0, 0.0], "proposed_by": "Agent_1"},
            {"contributions": [4.0, 2.0, 0.0], "proposed_by": "Agent_2"},
        ]

        game.prepare_proposals_for_voting(proposals_r1, state, 1)
        assert len(state["round_pledges"]) == 1

        game.prepare_proposals_for_voting(proposals_r2, state, 2)
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


class TestVotingPhases:
    """Tests for the active propose-and-vote flow."""

    def test_all_accept_records_accepted_joint_proposal(self):
        game, _, state = make_game_and_state(m_projects=3)
        items = state["projects"]
        preferences = {
            "agent_preferences": state["agent_valuations"],
            "game_state": state,
        }

        raw_proposals = [
            {"contributions": [5.0, 3.0, 2.0], "proposed_by": "Agent_1"},
            {"contributions": [4.0, 6.0, 1.0], "proposed_by": "Agent_2"},
        ]
        handler = PhaseHandler(game_environment=game)
        proposal_agents = [FakeAgent("Agent_1"), FakeAgent("Agent_2")]
        enumeration = asyncio.run(
            handler.run_proposal_enumeration_phase(
                agents=proposal_agents,
                items=items,
                preferences=preferences,
                round_num=2,
                max_rounds=5,
                proposals=raw_proposals,
            )
        )

        agents = [
            FakeAgent("Agent_1", ['{"vote": "accept", "reasoning": "works"}']),
            FakeAgent("Agent_2", ['{"vote": "accept", "reasoning": "agree"}']),
        ]
        handler = PhaseHandler(game_environment=game)

        voting = asyncio.run(
            handler.run_private_voting_phase(
                agents=agents,
                items=items,
                preferences=preferences,
                proposals=raw_proposals,
                enumerated_proposals=enumeration["enumerated_proposals"],
                round_num=2,
                max_rounds=5,
            )
        )
        result = asyncio.run(
            handler.run_vote_tabulation_phase(
                agents=agents,
                items=items,
                preferences=preferences,
                round_num=2,
                max_rounds=5,
                private_votes=voting["private_votes"],
                enumerated_proposals=enumeration["enumerated_proposals"],
            )
        )
        assert result["consensus_reached"] is True
        assert state["accepted_proposal"] is not None

    def test_reject_blocks_consensus(self):
        game, _, state = make_game_and_state(m_projects=3)
        items = state["projects"]
        preferences = {
            "agent_preferences": state["agent_valuations"],
            "game_state": state,
        }

        raw_proposals = [
            {"contributions": [5.0, 3.0, 2.0], "proposed_by": "Agent_1"},
            {"contributions": [4.0, 6.0, 1.0], "proposed_by": "Agent_2"},
        ]
        handler = PhaseHandler(game_environment=game)
        proposal_agents = [FakeAgent("Agent_1"), FakeAgent("Agent_2")]
        enumeration = asyncio.run(
            handler.run_proposal_enumeration_phase(
                agents=proposal_agents,
                items=items,
                preferences=preferences,
                round_num=2,
                max_rounds=5,
                proposals=raw_proposals,
            )
        )

        agents = [
            FakeAgent("Agent_1", ['{"vote": "accept", "reasoning": "works"}']),
            FakeAgent("Agent_2", ['{"vote": "reject", "reasoning": "revise"}']),
        ]
        handler = PhaseHandler(game_environment=game)

        voting = asyncio.run(
            handler.run_private_voting_phase(
                agents=agents,
                items=items,
                preferences=preferences,
                proposals=raw_proposals,
                enumerated_proposals=enumeration["enumerated_proposals"],
                round_num=2,
                max_rounds=5,
            )
        )
        result = asyncio.run(
            handler.run_vote_tabulation_phase(
                agents=agents,
                items=items,
                preferences=preferences,
                round_num=2,
                max_rounds=5,
                private_votes=voting["private_votes"],
                enumerated_proposals=enumeration["enumerated_proposals"],
            )
        )
        assert result["consensus_reached"] is False
        assert state["accepted_proposal"] is None

    def test_prefinal_commit_vote_failure_hard_fails_without_synthetic_nay(self):
        game, _, state = make_game_and_state(m_projects=3)
        preferences = {
            "agent_preferences": state["agent_valuations"],
            "game_state": state,
        }
        agents = [
            FakeAgent("Agent_1", ["not json"]),
            FakeAgent("Agent_2", ['{"commit_vote": "yay", "reasoning": "ok"}']),
        ]
        handler = PhaseHandler(game_environment=game)

        with pytest.raises(VoteIntegrityError):
            asyncio.run(
                handler.run_cofunding_commit_vote_phase(
                    agents=agents,
                    items=state["projects"],
                    preferences=preferences,
                    round_num=1,
                    max_rounds=3,
                )
            )

        integrity = handler.get_vote_integrity()
        assert integrity["hard_failed"] is True
        assert integrity["synthetic_vote_count"] == 0
        assert integrity["contaminated"] is False

    def test_final_commit_vote_failure_uses_audited_synthetic_nay(self):
        game, _, state = make_game_and_state(m_projects=3)
        preferences = {
            "agent_preferences": state["agent_valuations"],
            "game_state": state,
        }
        agents = [
            FakeAgent("Agent_1", ["not json"]),
            FakeAgent("Agent_2", ['{"commit_vote": "yay", "reasoning": "ok"}']),
        ]
        handler = PhaseHandler(game_environment=game)

        result = asyncio.run(
            handler.run_cofunding_commit_vote_phase(
                agents=agents,
                items=state["projects"],
                preferences=preferences,
                round_num=3,
                max_rounds=3,
            )
        )

        assert result["commit_votes"][0]["commit_vote"] == "nay"
        assert result["commit_votes"][0]["synthetic_vote"] is True
        integrity = handler.get_vote_integrity()
        assert integrity["contaminated"] is True
        assert integrity["synthetic_vote_count"] == 1

    def test_supermajority_accept_records_accepted_joint_proposal_for_n3(self):
        game, agents, state = make_game_and_state(n_agents=3, m_projects=4)
        items = state["projects"]
        preferences = {
            "agent_preferences": state["agent_valuations"],
            "game_state": state,
        }

        raw_proposals = []
        for agent in agents:
            budget = state["agent_budgets"][agent.agent_id]
            raw_proposals.append({
                "contributions": [budget * 0.5, 0.0, 0.0, 0.0],
                "proposed_by": agent.agent_id,
            })

        handler = PhaseHandler(game_environment=game)
        enumeration = asyncio.run(
            handler.run_proposal_enumeration_phase(
                agents=agents,
                items=items,
                preferences=preferences,
                round_num=1,
                max_rounds=5,
                proposals=raw_proposals,
            )
        )
        private_votes = [
            {"voter_id": "Agent_1", "proposal_number": 1, "vote": "accept"},
            {"voter_id": "Agent_2", "proposal_number": 1, "vote": "accept"},
            {"voter_id": "Agent_3", "proposal_number": 1, "vote": "reject"},
        ]

        result = asyncio.run(
            handler.run_vote_tabulation_phase(
                agents=agents,
                items=items,
                preferences=preferences,
                round_num=1,
                max_rounds=5,
                private_votes=private_votes,
                enumerated_proposals=enumeration["enumerated_proposals"],
            )
        )

        assert result["consensus_reached"] is True
        assert result["supermajority_threshold"] == 2
        assert result["majority_threshold"] == 2
        assert result["acceptance_rule"] == "two_thirds_supermajority"
        assert result["accepted_proposal_number"] == 1
        assert state["accepted_proposal"] is not None


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

        accepted = game.prepare_proposals_for_voting(
            [
                {"contributions": [8.0, 3.0, 2.0], "proposed_by": "Agent_1"},
                {"contributions": [5.0, 6.0, 2.0], "proposed_by": "Agent_2"},
            ],
            state,
            3,
        )[0]
        game.record_accepted_proposal(state, accepted)

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
        assert game.get_protocol_type() == "propose_and_vote"
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
        assert game.get_protocol_type() == "propose_and_vote"

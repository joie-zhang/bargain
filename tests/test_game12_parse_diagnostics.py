#!/usr/bin/env python3
"""Tests for preserving raw parse diagnostics in games 1 and 2."""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from game_environments import create_game_environment
from strong_models_experiment.phases.phase_handlers import PhaseHandler


@dataclass
class FakeAgentResponse:
    """Mimics the response object returned by agent.generate_response()."""

    content: str
    metadata: Optional[Dict[str, Any]] = None
    tokens_used: Optional[int] = None


class FakeAgent:
    """Fake agent that returns predetermined responses."""

    def __init__(self, agent_id: str, responses: List[str]):
        self.agent_id = agent_id
        self._responses = responses
        self._call_count = 0
        self._max_tokens = None

    async def generate_response(self, context, prompt) -> FakeAgentResponse:
        response = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1
        return FakeAgentResponse(content=response)

    def update_max_tokens(self, max_tokens):
        self._max_tokens = max_tokens

    def get_model_info(self) -> Dict[str, Any]:
        return {"model_name": "fake-model"}


def test_item_allocation_parse_failure_preserves_raw_response():
    """Game 1 proposal parse failures should retain the raw response."""
    game = create_game_environment(
        "item_allocation",
        n_agents=2,
        t_rounds=3,
        m_items=3,
        random_seed=42,
    )
    state = game.create_game_state([FakeAgent("Agent_1", ["unused"]), FakeAgent("Agent_2", ["unused"])])

    response = "not valid json"
    parsed = game.parse_proposal(response, "Agent_1", state, ["Agent_1", "Agent_2"])

    assert parsed["allocation"]["Agent_1"] == [0, 1, 2]
    assert parsed["raw_response"] == response
    assert parsed["parse_error"]["type"] == "ValueError"


def test_diplomatic_treaty_parse_failure_preserves_raw_response():
    """Game 2 proposal parse failures should retain the raw response."""
    game = create_game_environment(
        "diplomacy",
        n_agents=2,
        t_rounds=3,
        n_issues=3,
        random_seed=42,
    )
    state = game.create_game_state([FakeAgent("Agent_1", ["unused"]), FakeAgent("Agent_2", ["unused"])])

    response = "not valid json"
    parsed = game.parse_proposal(response, "Agent_1", state, ["Agent_1", "Agent_2"])

    assert parsed["agreement"] == [0.5, 0.5, 0.5]
    assert parsed["raw_response"] == response
    assert parsed["parse_error"]["type"] == "ValueError"


def test_item_allocation_batch_vote_parse_failure_is_saved_with_raw_response():
    """Game 1 saved vote artifacts should preserve invalid batch vote responses."""

    async def run_test():
        bad_response = "not valid json"
        game = create_game_environment(
            "item_allocation",
            n_agents=2,
            t_rounds=3,
            m_items=3,
            random_seed=42,
        )
        agents = [
            FakeAgent("Agent_1", [bad_response]),
            FakeAgent("Agent_2", [json.dumps({"votes": [{"proposal_number": 1, "vote": "accept"}]})]),
        ]
        state = game.create_game_state(agents)
        preferences = {
            "agent_preferences": state["agent_preferences"],
            "game_state": state,
        }
        proposal = {
            "allocation": {"Agent_1": [0], "Agent_2": [1, 2]},
            "reasoning": "Proposal one",
            "proposed_by": "Agent_1",
            "round": 1,
        }
        enumerated_proposals = [
            {
                "proposal_number": 1,
                "proposer": "Agent_1",
                "reasoning": "Proposal one",
                "original_proposal": proposal,
                "allocation": proposal["allocation"],
            }
        ]
        saved = []

        def save_interaction(*args, **kwargs):
            saved.append((args, kwargs))

        handler = PhaseHandler(save_interaction_callback=save_interaction, game_environment=game)

        result = await handler.run_private_voting_phase(
            agents=agents,
            items=state["items"],
            preferences=preferences,
            round_num=1,
            max_rounds=3,
            proposals=[proposal],
            enumerated_proposals=enumerated_proposals,
        )

        assert result["private_votes"][0]["vote"] == "reject"
        saved_response = json.loads(saved[0][0][3])
        assert saved_response["raw_response"] == bad_response
        assert saved_response["parse_error"]["type"] == "ValueError"

    asyncio.run(run_test())


def test_diplomatic_treaty_batch_vote_parse_failure_is_saved_with_raw_response():
    """Game 2 saved vote artifacts should preserve invalid batch vote responses."""

    async def run_test():
        bad_response = "not valid json"
        game = create_game_environment(
            "diplomacy",
            n_agents=2,
            t_rounds=3,
            n_issues=3,
            random_seed=42,
        )
        agents = [
            FakeAgent("Agent_1", [bad_response]),
            FakeAgent("Agent_2", [json.dumps({"votes": [{"proposal_number": 1, "vote": "accept"}]})]),
        ]
        state = game.create_game_state(agents)
        preferences = {
            "agent_preferences": state["agent_positions"],
            "agent_weights": state["agent_weights"],
            "game_state": state,
        }
        items = [{"name": issue} for issue in state["issues"]]
        proposal = {
            "agreement": [65, 20, 55],
            "reasoning": "Proposal one",
            "proposed_by": "Agent_1",
            "round": 1,
        }
        enumerated_proposals = [
            {
                "proposal_number": 1,
                "proposer": "Agent_1",
                "reasoning": "Proposal one",
                "original_proposal": proposal,
                "agreement": proposal["agreement"],
            }
        ]
        saved = []

        def save_interaction(*args, **kwargs):
            saved.append((args, kwargs))

        handler = PhaseHandler(save_interaction_callback=save_interaction, game_environment=game)

        result = await handler.run_private_voting_phase(
            agents=agents,
            items=items,
            preferences=preferences,
            round_num=1,
            max_rounds=3,
            proposals=[proposal],
            enumerated_proposals=enumerated_proposals,
        )

        assert result["private_votes"][0]["vote"] == "reject"
        saved_response = json.loads(saved[0][0][3])
        assert saved_response["raw_response"] == bad_response
        assert saved_response["parse_error"]["type"] == "ValueError"

    asyncio.run(run_test())

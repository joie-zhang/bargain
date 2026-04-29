#!/usr/bin/env python3
"""Tests for preserving raw parse diagnostics in games 1 and 2."""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

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
        self.prompts = []

    async def generate_response(self, context, prompt) -> FakeAgentResponse:
        self.prompts.append(prompt)
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

    assert parsed["allocation"] == {"Agent_1": [], "Agent_2": []}
    assert parsed["raw_response"] == response
    assert parsed["parse_error"]["type"] == "ValueError"
    assert game.validate_proposal(parsed, state) is False


def test_item_allocation_repairs_literal_newlines_inside_json_strings():
    """Game 1 should parse model JSON with unescaped newlines in string fields."""
    game = create_game_environment(
        "item_allocation",
        n_agents=2,
        t_rounds=3,
        m_items=3,
        random_seed=42,
    )
    state = game.create_game_state([FakeAgent("Agent_1", ["unused"]), FakeAgent("Agent_2", ["unused"])])

    response = """{
      "allocation": {
        "Agent_1": [0, 2],
        "Agent_2": [1]
      },
      "reasoning": "First line.

Second line with detail."
    }"""
    parsed = game.parse_proposal(response, "Agent_1", state, ["Agent_1", "Agent_2"])

    assert parsed["allocation"] == {"Agent_1": [0, 2], "Agent_2": [1]}
    assert "parse_error" not in parsed
    assert "Second line" in parsed["reasoning"]


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


def test_diplomatic_treaty_repairs_literal_newlines_inside_json_strings():
    """Game 2 should parse model JSON with unescaped newlines in string fields."""
    game = create_game_environment(
        "diplomacy",
        n_agents=2,
        t_rounds=3,
        n_issues=3,
        random_seed=42,
    )
    state = game.create_game_state([FakeAgent("Agent_1", ["unused"]), FakeAgent("Agent_2", ["unused"])])

    response = """{
      "agreement": [65, 20, 55],
      "reasoning": "First line.

Second line with detail."
    }"""
    parsed = game.parse_proposal(response, "Agent_1", state, ["Agent_1", "Agent_2"])

    assert parsed["agreement"] == [0.65, 0.2, 0.55]
    assert "parse_error" not in parsed
    assert "Second line" in parsed["reasoning"]


def test_item_allocation_final_round_batch_vote_parse_failure_is_saved_with_raw_response():
    """Game 1 final-round synthetic vote artifacts should preserve invalid raw responses."""

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
            round_num=3,
            max_rounds=3,
            proposals=[proposal],
            enumerated_proposals=enumerated_proposals,
        )

        assert result["private_votes"][0]["vote"] == "reject"
        assert result["private_votes"][0]["synthetic_vote"] is True
        assert result["voting_summary"]["vote_integrity"]["contaminated"] is True
        saved_response = json.loads(saved[0][0][3])
        assert saved_response["raw_response"] == bad_response
        assert saved_response["parse_error"]["type"] == "ValueError"

    asyncio.run(run_test())


def test_item_allocation_batch_vote_repairs_literal_newlines_inside_json_strings():
    """Game 1 batch voting should parse unescaped newlines in vote reasoning."""
    game = create_game_environment(
        "item_allocation",
        n_agents=2,
        t_rounds=3,
        m_items=3,
        random_seed=42,
    )

    response = """{
      "votes": [
        {
          "proposal_number": 1,
          "vote": "accept",
          "reasoning": "First line.

Second line."
        }
      ]
    }"""
    votes = game.parse_batch_voting_response(response, [1], "Agent_1", 1)

    assert votes[0]["vote"] == "accept"
    assert "parse_error" not in votes[0]
    assert "Second line" in votes[0]["reasoning"]


def test_item_allocation_proposal_repair_prompt_uses_allocation_schema():
    """Game 1 repair prompts should not ask for Game 2-style agreement vectors."""

    async def run_test():
        game = create_game_environment(
            "item_allocation",
            n_agents=2,
            t_rounds=3,
            m_items=3,
            random_seed=42,
        )
        bad_legacy_response = '{"agreement": [36, 0, 12], "reasoning": "old schema"}'
        valid_repair = json.dumps(
            {
                "allocation": {"Agent_1": [0, 2], "Agent_2": [1]},
                "reasoning": "valid repaired allocation",
            }
        )
        agents = [
            FakeAgent("Agent_1", [bad_legacy_response, valid_repair]),
            FakeAgent(
                "Agent_2",
                [
                    json.dumps(
                        {
                            "allocation": {"Agent_1": [0], "Agent_2": [1, 2]},
                            "reasoning": "valid first try",
                        }
                    )
                ],
            ),
        ]
        state = game.create_game_state(agents)
        preferences = {
            "agent_preferences": state["agent_preferences"],
            "game_state": state,
        }
        saved = []

        def save_interaction(*args, **kwargs):
            saved.append((args, kwargs))

        handler = PhaseHandler(save_interaction_callback=save_interaction, game_environment=game)

        result = await handler.run_proposal_phase(
            agents=agents,
            items=state["items"],
            preferences=preferences,
            round_num=1,
            max_rounds=3,
        )

        assert len(agents[0].prompts) == 2
        repair_prompt = agents[0].prompts[1]
        assert "allocation object" in repair_prompt
        assert "\"allocation\"" in repair_prompt
        assert "agreement" not in repair_prompt.lower()
        assert result["proposals"][0]["allocation"] == {"Agent_1": [0, 2], "Agent_2": [1]}
        invalid_response = json.loads(
            next(args[3] for args, _kwargs in saved if args[1] == "proposal_round_1_invalid_attempt_0")
        )
        assert invalid_response["raw_proposal"] == bad_legacy_response
        assert invalid_response["raw_response"] == bad_legacy_response
        assert invalid_response["parse_error"]["type"] == "ValueError"
        assert invalid_response["will_retry"] is True
        assert invalid_response["hard_failed"] is False

        saved_response = json.loads(
            next(args[3] for args, _kwargs in saved if args[1] == "proposal_round_1")
        )
        assert saved_response["recovered_after_error"] == "parse error"
        assert saved_response["raw_response"] == bad_legacy_response

    asyncio.run(run_test())


def test_proposal_repair_prompts_are_schema_isolated_by_game():
    """Repair prompts should preserve each game's native proposal schema."""
    handler = PhaseHandler()
    agent_ids = ["Agent_1", "Agent_2"]
    items = [{"name": f"Item {idx}"} for idx in range(3)]

    game1_prompts = handler._build_proposal_retry_prompts(
        game_type="item_allocation",
        proposal_prompt="BASE PROMPT",
        game_state={"game_type": "item_allocation", "items": items},
        agent_id="Agent_1",
        agent_ids=agent_ids,
        items=items,
    )
    game1_text = "\n".join(game1_prompts).lower()
    assert "allocation object" in game1_text
    assert '"allocation"' in game1_text
    assert "agreement" not in game1_text
    assert "contributions" not in game1_text

    game2_prompts = handler._build_proposal_retry_prompts(
        game_type="diplomacy",
        proposal_prompt="BASE PROMPT",
        game_state={"game_type": "diplomacy", "n_issues": 3},
        agent_id="Agent_1",
        agent_ids=agent_ids,
        items=items,
    )
    game2_text = "\n".join(game2_prompts).lower()
    assert "agreement array" in game2_text
    assert '"agreement"' in game2_text
    assert "allocation object" not in game2_text
    assert "contributions" not in game2_text

    game3_prompts = handler._build_proposal_retry_prompts(
        game_type="co_funding",
        proposal_prompt="BASE PROMPT",
        game_state={
            "game_type": "co_funding",
            "m_projects": 3,
            "agent_budgets": {"Agent_1": 25.0},
        },
        agent_id="Agent_1",
        agent_ids=agent_ids,
        items=items,
    )
    game3_text = "\n".join(game3_prompts).lower()
    assert "contributions array" in game3_text
    assert '"contributions"' in game3_text
    assert "agreement" not in game3_text
    assert "allocation" not in game3_text


def test_item_allocation_unrepaired_legacy_agreement_vector_hard_fails():
    """Repeated Game 2-style vectors should fail instead of being saved as proposals."""

    async def run_test():
        game = create_game_environment(
            "item_allocation",
            n_agents=2,
            t_rounds=3,
            m_items=3,
            random_seed=42,
        )
        bad_legacy_response = '{"agreement": [36, 0, 12], "reasoning": "old schema"}'
        agents = [
            FakeAgent("Agent_1", [bad_legacy_response, bad_legacy_response, bad_legacy_response]),
            FakeAgent(
                "Agent_2",
                [
                    json.dumps(
                        {
                            "allocation": {"Agent_1": [0], "Agent_2": [1, 2]},
                            "reasoning": "valid first try",
                        }
                    )
                ],
            ),
        ]
        state = game.create_game_state(agents)
        preferences = {
            "agent_preferences": state["agent_preferences"],
            "game_state": state,
        }
        saved = []

        def save_interaction(*args, **kwargs):
            saved.append((args, kwargs))

        handler = PhaseHandler(save_interaction_callback=save_interaction, game_environment=game)

        with pytest.raises(ValueError, match="item_allocation proposal from Agent_1 remained invalid"):
            await handler.run_proposal_phase(
                agents=agents,
                items=state["items"],
                preferences=preferences,
                round_num=1,
                max_rounds=3,
            )

        assert [args[1] for args, _kwargs in saved] == [
            "proposal_round_1_invalid_attempt_0",
            "proposal_round_1_invalid_attempt_1",
            "proposal_round_1_invalid_attempt_2",
        ]
        saved_prompts = [args[2] for args, _kwargs in saved]
        assert all("agreement" not in prompt.lower() for prompt in saved_prompts)
        final_diagnostic = json.loads(saved[-1][0][3])
        assert final_diagnostic["raw_proposal"] == bad_legacy_response
        assert final_diagnostic["raw_response"] == bad_legacy_response
        assert final_diagnostic["parse_error"]["type"] == "ValueError"
        assert final_diagnostic["will_retry"] is False
        assert final_diagnostic["hard_failed"] is True
        assert "skipped" not in final_diagnostic

    asyncio.run(run_test())


def test_item_allocation_repair_prompt_without_persisted_game_state_stays_allocation_schema():
    """Game 1 reconstructed game state should not fall back to Game 2 repair prompts."""

    async def run_test():
        game = create_game_environment(
            "item_allocation",
            n_agents=2,
            t_rounds=3,
            m_items=3,
            random_seed=42,
        )
        bad_legacy_response = '{"agreement": [36, 0, 12], "reasoning": "old schema"}'
        agents = [
            FakeAgent("Agent_1", [bad_legacy_response, bad_legacy_response, bad_legacy_response]),
            FakeAgent(
                "Agent_2",
                [
                    json.dumps(
                        {
                            "allocation": {"Agent_1": [0], "Agent_2": [1, 2]},
                            "reasoning": "valid first try",
                        }
                    )
                ],
            ),
        ]
        state = game.create_game_state(agents)
        preferences = {
            "agent_preferences": state["agent_preferences"],
        }
        saved = []

        def save_interaction(*args, **kwargs):
            saved.append((args, kwargs))

        handler = PhaseHandler(save_interaction_callback=save_interaction, game_environment=game)

        with pytest.raises(ValueError, match="item_allocation proposal from Agent_1 remained invalid"):
            await handler.run_proposal_phase(
                agents=agents,
                items=state["items"],
                preferences=preferences,
                round_num=1,
                max_rounds=3,
            )

        saved_prompts = [args[2] for args, _kwargs in saved]
        assert len(saved_prompts) == 3
        assert all("allocation" in prompt.lower() for prompt in saved_prompts)
        assert all("agreement" not in prompt.lower() for prompt in saved_prompts)
        final_diagnostic = json.loads(saved[-1][0][3])
        assert final_diagnostic["game_type"] == "item_allocation"

    asyncio.run(run_test())


def test_diplomatic_treaty_final_round_batch_vote_parse_failure_is_saved_with_raw_response():
    """Game 2 final-round synthetic vote artifacts should preserve invalid raw responses."""

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
            round_num=3,
            max_rounds=3,
            proposals=[proposal],
            enumerated_proposals=enumerated_proposals,
        )

        assert result["private_votes"][0]["vote"] == "reject"
        assert result["private_votes"][0]["synthetic_vote"] is True
        assert result["voting_summary"]["vote_integrity"]["contaminated"] is True
        saved_response = json.loads(saved[0][0][3])
        assert saved_response["raw_response"] == bad_response
        assert saved_response["parse_error"]["type"] == "ValueError"

    asyncio.run(run_test())


def test_diplomatic_treaty_batch_vote_repairs_literal_newlines_inside_json_strings():
    """Game 2 batch voting should parse unescaped newlines in vote reasoning."""
    game = create_game_environment(
        "diplomacy",
        n_agents=2,
        t_rounds=3,
        n_issues=3,
        random_seed=42,
    )

    response = """{
      "votes": [
        {
          "proposal_number": 1,
          "vote": "reject",
          "reasoning": "First line.

Second line."
        }
      ]
    }"""
    votes = game.parse_batch_voting_response(response, [1], "Agent_1", 1)

    assert votes[0]["vote"] == "reject"
    assert "parse_error" not in votes[0]
    assert "Second line" in votes[0]["reasoning"]

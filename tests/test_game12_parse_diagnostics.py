#!/usr/bin/env python3
"""Tests for preserving raw parse diagnostics in games 1 and 2."""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from game_environments import create_game_environment
from negotiation.provider_key_rotation import ProviderTransientRetryExhaustedError
from strong_models_experiment.phases.phase_handlers import PhaseHandler


@dataclass
class FakeAgentResponse:
    """Mimics the response object returned by agent.generate_response()."""

    content: str
    metadata: Optional[Dict[str, Any]] = None
    tokens_used: Optional[int] = None


class FakeAgent:
    """Fake agent that returns predetermined responses."""

    def __init__(self, agent_id: str, responses: List[Any]):
        self.agent_id = agent_id
        self._responses = responses
        self._call_count = 0
        self._max_tokens = None
        self.prompts = []

    async def generate_response(self, context, prompt) -> FakeAgentResponse:
        self.prompts.append(prompt)
        response = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1
        if isinstance(response, BaseException):
            raise response
        return FakeAgentResponse(content=response)

    def update_max_tokens(self, max_tokens):
        self._max_tokens = max_tokens

    def get_model_info(self) -> Dict[str, Any]:
        return {"model_name": "fake-model"}


def assert_strict_json_format_rules(prompt: str) -> None:
    """Prompt should explicitly ban common model JSON formatting failures."""
    assert "JSON FORMAT REQUIREMENTS" in prompt
    assert "first non-whitespace character" in prompt
    assert "double quotes" in prompt
    assert "markdown code fences" in prompt
    assert "JSON comments" in prompt
    assert "prose before/after" in prompt
    assert "placeholders inside arrays" in prompt
    assert "literal line breaks inside quoted string values" in prompt


def extract_additional_example(prompt: str) -> Dict[str, Any]:
    """Return the second valid JSON example from a proposal prompt."""
    example_text = prompt.split(
        "Additional valid JSON example using the same schema:\n",
        1,
    )[1].split("\n\n**JSON FORMAT REQUIREMENTS:**", 1)[0]
    return json.loads(example_text)


def test_json_response_prompts_ban_known_parse_error_patterns():
    """Proposal, vote, and private-thinking prompts should all include strict JSON rules."""
    agents = [FakeAgent("Agent_1", ["unused"]), FakeAgent("Agent_2", ["unused"])]
    agent_ids = [agent.agent_id for agent in agents]

    item_game = create_game_environment(
        "item_allocation",
        n_agents=2,
        t_rounds=3,
        m_items=3,
        random_seed=42,
    )
    item_state = item_game.create_game_state(agents)
    item_proposal = {
        "proposal_number": 1,
        "allocation": {"Agent_1": [0, 2], "Agent_2": [1]},
        "proposed_by": "Agent_1",
    }
    for prompt in [
        item_game.get_proposal_prompt("Agent_1", item_state, 1, agent_ids),
        item_game.get_batch_voting_prompt("Agent_1", [item_proposal], item_state, 1),
        item_game.get_thinking_prompt("Agent_1", item_state, 1, 3, []),
    ]:
        assert_strict_json_format_rules(prompt)

    diplomacy_game = create_game_environment(
        "diplomacy",
        n_agents=2,
        t_rounds=3,
        n_issues=3,
        random_seed=42,
    )
    diplomacy_state = diplomacy_game.create_game_state(agents)
    diplomacy_proposal = {
        "proposal_number": 1,
        "agreement": [50, 50, 50],
        "proposed_by": "Agent_1",
    }
    for prompt in [
        diplomacy_game.get_proposal_prompt("Agent_1", diplomacy_state, 1, agent_ids),
        diplomacy_game.get_batch_voting_prompt("Agent_1", [diplomacy_proposal], diplomacy_state, 1),
        diplomacy_game.get_thinking_prompt("Agent_1", diplomacy_state, 1, 3, []),
    ]:
        assert_strict_json_format_rules(prompt)

    cofunding_game = create_game_environment(
        "co_funding",
        n_agents=2,
        t_rounds=3,
        m_projects=3,
        random_seed=42,
    )
    cofunding_state = cofunding_game.create_game_state(agents)
    cofunding_proposal = {
        "aggregate_totals": [0.0, 0.0, 0.0],
        "funded_projects": [],
        "contributions_by_agent": {
            "Agent_1": [0.0, 0.0, 0.0],
            "Agent_2": [0.0, 0.0, 0.0],
        },
    }
    for prompt in [
        cofunding_game.get_proposal_prompt("Agent_1", cofunding_state, 1, agent_ids),
        cofunding_game.get_voting_prompt("Agent_1", cofunding_proposal, cofunding_state, 1),
        cofunding_game.get_commit_vote_prompt("Agent_1", cofunding_state, 1, 3),
        cofunding_game.get_thinking_prompt("Agent_1", cofunding_state, 1, 3, []),
    ]:
        assert_strict_json_format_rules(prompt)


def test_item_allocation_proposal_prompt_highlights_complete_ownership_invariant():
    """The proposal prompt should make the all-items-exactly-once rule prominent."""
    agents = [FakeAgent("Agent_1", ["unused"]), FakeAgent("Agent_2", ["unused"])]
    agent_ids = [agent.agent_id for agent in agents]
    game = create_game_environment(
        "item_allocation",
        n_agents=2,
        t_rounds=3,
        m_items=3,
        random_seed=42,
    )
    state = game.create_game_state(agents)

    prompt = game.get_proposal_prompt("Agent_1", state, 1, agent_ids)

    assert "Complete-Ownership Invariant" in prompt
    assert "The union of all agent arrays must be exactly [0, 1, 2]" in prompt
    assert "Do not omit any item index" in prompt
    assert "Do not duplicate an item index" in prompt


def test_item_allocation_validation_error_names_missing_and_duplicate_items():
    """Validation diagnostics should name exact item-index invariant violations."""
    agents = [FakeAgent("Agent_1", ["unused"]), FakeAgent("Agent_2", ["unused"])]
    game = create_game_environment(
        "item_allocation",
        n_agents=2,
        t_rounds=3,
        m_items=4,
        random_seed=42,
    )
    state = game.create_game_state(agents)
    proposal = {
        "allocation": {
            "Agent_1": [0, 1, 1],
            "Agent_2": [2],
        }
    }

    error = game.proposal_validation_error(proposal, state)

    assert "duplicate item indices" in error
    assert "1 assigned to ['Agent_1', 'Agent_1']" in error
    assert "missing item indices [3]" in error


def test_proposal_prompts_include_second_game_specific_valid_json_example():
    """Each game should show more than one valid proposal example in its own schema."""
    agents = [FakeAgent("Agent_1", ["unused"]), FakeAgent("Agent_2", ["unused"])]
    agent_ids = [agent.agent_id for agent in agents]

    item_game = create_game_environment(
        "item_allocation",
        n_agents=2,
        t_rounds=3,
        m_items=5,
        random_seed=42,
    )
    item_state = item_game.create_game_state(agents)
    item_prompt = item_game.get_proposal_prompt("Agent_1", item_state, 1, agent_ids)
    item_example = extract_additional_example(item_prompt)
    assert sorted(item_example) == ["allocation", "reasoning"]
    allocated_items = [
        item
        for allocation in item_example["allocation"].values()
        for item in allocation
    ]
    assert sorted(allocated_items) == list(range(5))

    diplomacy_game = create_game_environment(
        "diplomacy",
        n_agents=2,
        t_rounds=3,
        n_issues=7,
        random_seed=42,
    )
    diplomacy_state = diplomacy_game.create_game_state(agents)
    diplomacy_prompt = diplomacy_game.get_proposal_prompt("Agent_1", diplomacy_state, 1, agent_ids)
    diplomacy_example = extract_additional_example(diplomacy_prompt)
    assert sorted(diplomacy_example) == ["agreement", "reasoning"]
    assert len(diplomacy_example["agreement"]) == 7
    assert all(0 <= value <= 100 for value in diplomacy_example["agreement"])

    cofunding_game = create_game_environment(
        "co_funding",
        n_agents=2,
        t_rounds=3,
        m_projects=8,
        random_seed=42,
    )
    cofunding_state = cofunding_game.create_game_state(agents)
    cofunding_prompt = cofunding_game.get_proposal_prompt("Agent_1", cofunding_state, 1, agent_ids)
    cofunding_example = extract_additional_example(cofunding_prompt)
    assert sorted(cofunding_example) == ["contributions", "reasoning"]
    assert len(cofunding_example["contributions"]) == 8
    assert sum(cofunding_example["contributions"]) <= cofunding_state["agent_budgets"]["Agent_1"]


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


def test_item_allocation_prefinal_batch_vote_failure_defaults_to_reject_after_logging_raw_attempts():
    """Game 1 pre-final invalid batch votes should persist raw responses and continue."""

    async def run_test():
        bad_initial = "not valid json"
        bad_retry = "still not valid json"
        bad_compact = "compact repair also not json"
        game = create_game_environment(
            "item_allocation",
            n_agents=2,
            t_rounds=3,
            m_items=3,
            random_seed=42,
        )
        agents = [
            FakeAgent("Agent_1", [bad_initial, bad_retry, bad_compact]),
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

        invalid_saved = [(args, kwargs) for args, kwargs in saved if "_batch_invalid_attempt_" in args[1]]
        assert [args[1] for args, _kwargs in invalid_saved] == [
            "voting_round_1_batch_invalid_attempt_0",
            "voting_round_1_batch_invalid_attempt_1",
            "voting_round_1_batch_invalid_attempt_2",
        ]
        payloads = [json.loads(args[3]) for args, _kwargs in invalid_saved]
        assert [payload["raw_response"] for payload in payloads] == [
            bad_initial,
            bad_retry,
            bad_compact,
        ]
        assert [payload["missing_proposal_numbers"] for payload in payloads] == [[1], [1], [1]]
        assert [payload["will_retry"] for payload in payloads] == [True, True, False]
        assert [payload["hard_failed"] for payload in payloads] == [False, False, False]
        assert result["private_votes"][0]["vote"] == "reject"
        assert result["private_votes"][0]["synthetic_vote"] is True
        assert result["voting_summary"]["vote_integrity"]["synthetic_vote_count"] == 1

    asyncio.run(run_test())


def test_item_allocation_hard_batch_vote_failure_defaults_to_reject_without_raising():
    """Hard provider failures in batch voting should contaminate/audit, not fail the sample."""

    async def run_test():
        hard_error = ProviderTransientRetryExhaustedError("provider timeout budget exhausted")
        game = create_game_environment(
            "item_allocation",
            n_agents=2,
            t_rounds=3,
            m_items=3,
            random_seed=42,
        )
        agents = [
            FakeAgent("Agent_1", [hard_error, hard_error, hard_error]),
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

        agent_1_vote = next(v for v in result["private_votes"] if v["voter_id"] == "Agent_1")
        assert agent_1_vote["vote"] == "reject"
        assert agent_1_vote["synthetic_vote"] is True
        assert agent_1_vote["fallback_policy_version"] == "invalid-output-default-v1"
        assert agent_1_vote["parse_error"]["type"] == "ProviderTransientRetryExhaustedError"
        assert result["voting_summary"]["vote_integrity"]["contaminated"] is True
        assert result["voting_summary"]["vote_integrity"]["synthetic_vote_count"] == 1

        invalid_saved = [(args, kwargs) for args, kwargs in saved if "_batch_invalid_attempt_" in args[1]]
        assert [args[1] for args, _kwargs in invalid_saved] == [
            "voting_round_1_batch_invalid_attempt_0",
            "voting_round_1_batch_invalid_attempt_1",
            "voting_round_1_batch_invalid_attempt_2",
        ]
        assert json.loads(invalid_saved[0][0][3])["parse_error"]["type"] == "ProviderTransientRetryExhaustedError"

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
        assert "TARGETED REPAIR CONTEXT" in repair_prompt
        assert "INVALID RAW RESPONSE" in repair_prompt
        assert bad_legacy_response in repair_prompt
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
        saved_final_prompt = next(args[2] for args, _kwargs in saved if args[1] == "proposal_round_1")
        assert saved_final_prompt == repair_prompt
        assert saved_response["recovered_after_error"] == "parse error (ValueError: No allocation in proposal)"
        assert saved_response["raw_response"] == bad_legacy_response

    asyncio.run(run_test())


def test_proposal_repair_prompt_surfaces_specific_json_parse_reason():
    """Repair context should include the concrete parser exception, not only `parse error`."""

    async def run_test():
        game = create_game_environment(
            "item_allocation",
            n_agents=2,
            t_rounds=3,
            m_items=3,
            random_seed=42,
        )
        bad_json_response = '{"allocation": invalid, "reasoning": "bare identifier"}'
        valid_repair = json.dumps(
            {
                "allocation": {"Agent_1": [0, 2], "Agent_2": [1]},
                "reasoning": "valid repaired allocation",
            }
        )
        agents = [
            FakeAgent("Agent_1", [bad_json_response, valid_repair]),
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

        handler = PhaseHandler(game_environment=game)
        await handler.run_proposal_phase(
            agents=agents,
            items=state["items"],
            preferences=preferences,
            round_num=1,
            max_rounds=3,
        )

        repair_prompt = agents[0].prompts[1]
        assert "- Parser/validator error: parse error (JSONDecodeError:" in repair_prompt
        assert "Expecting value" in repair_prompt
        assert "line 1" in repair_prompt
        assert "column" in repair_prompt

    asyncio.run(run_test())


def test_item_allocation_repair_prompt_reports_exact_allocation_invariant_failures():
    """Repair context should include exact missing and duplicate item indices."""

    async def run_test():
        game = create_game_environment(
            "item_allocation",
            n_agents=2,
            t_rounds=3,
            m_items=3,
            random_seed=42,
        )
        bad_response = json.dumps(
            {
                "allocation": {"Agent_1": [0, 1], "Agent_2": [1]},
                "reasoning": "item 1 is duplicated and item 2 is omitted",
            }
        )
        valid_repair = json.dumps(
            {
                "allocation": {"Agent_1": [0, 2], "Agent_2": [1]},
                "reasoning": "valid repaired allocation",
            }
        )
        agents = [
            FakeAgent("Agent_1", [bad_response, valid_repair]),
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

        handler = PhaseHandler(game_environment=game)
        await handler.run_proposal_phase(
            agents=agents,
            items=state["items"],
            preferences=preferences,
            round_num=1,
            max_rounds=3,
        )

        repair_prompt = agents[0].prompts[1]
        assert "duplicate item indices" in repair_prompt
        assert "1 assigned to ['Agent_1', 'Agent_2']" in repair_prompt
        assert "missing item indices [2]" in repair_prompt
        assert "expected each item index in [0, 1, 2] to appear exactly once" in repair_prompt

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
    assert "}}." not in game2_text
    game2_repair_payload = json.loads(
        game2_prompts[1].split("Use this shape: ", 1)[1].split(". You may", 1)[0]
    )
    assert game2_repair_payload == {
        "agreement": [50, 50, 50],
        "reasoning": "brief compromise rationale",
    }

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
    assert "}}." not in game3_text
    game3_repair_payload = json.loads(
        game3_prompts[1].split("Use this shape: ", 1)[1].split(". You may", 1)[0]
    )
    assert game3_repair_payload == {
        "contributions": [0.0, 0.0, 0.0],
        "reasoning": "brief funding rationale",
    }


def test_item_allocation_missing_only_after_repair_exhaustion_defaults_to_proposer_gets_all():
    """Missing item ownership now falls back to the naive proposer-gets-all proposal."""

    async def run_test():
        game = create_game_environment(
            "item_allocation",
            n_agents=2,
            t_rounds=3,
            m_items=3,
            random_seed=42,
        )
        partial_response = json.dumps(
            {
                "allocation": {"Agent_1": [0], "Agent_2": [1]},
                "reasoning": "item 2 is intentionally left out",
            }
        )
        agents = [
            FakeAgent("Agent_1", [partial_response] * 5),
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

        saved_phases = [args[1] for args, _kwargs in saved]
        assert saved_phases[:5] == [
            "proposal_round_1_invalid_attempt_0",
            "proposal_round_1_invalid_attempt_1",
            "proposal_round_1_invalid_attempt_2",
            "proposal_round_1_invalid_attempt_3",
            "proposal_round_1_invalid_attempt_4",
        ]
        final_diagnostic = json.loads(saved[4][0][3])
        assert final_diagnostic["will_retry"] is False
        assert final_diagnostic["hard_failed"] is False
        assert "missing item indices [2]" in final_diagnostic["validation_error"]
        proposal = result["proposals"][0]
        assert proposal["allocation"] == {"Agent_1": [0, 1, 2], "Agent_2": []}
        assert proposal["synthetic_proposal"] is True
        assert proposal["fallback_policy_version"] == "invalid-output-default-v1"

    asyncio.run(run_test())


def test_item_allocation_duplicate_after_repair_exhaustion_defaults_to_proposer_gets_all():
    """Duplicate item ownership now falls back to the naive proposer-gets-all proposal."""

    async def run_test():
        game = create_game_environment(
            "item_allocation",
            n_agents=2,
            t_rounds=3,
            m_items=3,
            random_seed=42,
        )
        duplicate_response = json.dumps(
            {
                "allocation": {"Agent_1": [0, 1], "Agent_2": [1, 2]},
                "reasoning": "item 1 is duplicated",
            }
        )
        agents = [
            FakeAgent("Agent_1", [duplicate_response] * 5),
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

        invalid_saved = [(args, kwargs) for args, kwargs in saved if "_invalid_attempt_" in args[1]]
        final_diagnostic = json.loads(invalid_saved[-1][0][3])
        assert final_diagnostic["will_retry"] is False
        assert final_diagnostic["hard_failed"] is False
        assert "duplicate item indices" in final_diagnostic["validation_error"]
        proposal = result["proposals"][0]
        assert proposal["allocation"] == {"Agent_1": [0, 1, 2], "Agent_2": []}
        assert proposal["synthetic_proposal"] is True

    asyncio.run(run_test())


def test_item_allocation_unrepaired_legacy_agreement_vector_defaults_to_proposer_gets_all():
    """Repeated Game 2-style vectors should become the naive default proposal."""

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

        result = await handler.run_proposal_phase(
            agents=agents,
            items=state["items"],
            preferences=preferences,
            round_num=1,
            max_rounds=3,
        )

        invalid_saved = [(args, kwargs) for args, kwargs in saved if "_invalid_attempt_" in args[1]]
        assert [args[1] for args, _kwargs in invalid_saved] == [
            "proposal_round_1_invalid_attempt_0",
            "proposal_round_1_invalid_attempt_1",
            "proposal_round_1_invalid_attempt_2",
            "proposal_round_1_invalid_attempt_3",
            "proposal_round_1_invalid_attempt_4",
        ]
        saved_prompts = [args[2] for args, _kwargs in invalid_saved]
        assert all("\"allocation\"" in prompt for prompt in saved_prompts[1:])
        assert all("INVALID RAW RESPONSE" in prompt for prompt in saved_prompts[1:])
        final_diagnostic = json.loads(invalid_saved[-1][0][3])
        assert final_diagnostic["raw_proposal"] == bad_legacy_response
        assert final_diagnostic["raw_response"] == bad_legacy_response
        assert final_diagnostic["parse_error"]["type"] == "ValueError"
        assert final_diagnostic["will_retry"] is False
        assert final_diagnostic["hard_failed"] is False
        assert "skipped" not in final_diagnostic
        proposal = result["proposals"][0]
        assert proposal["allocation"] == {"Agent_1": [0, 1, 2], "Agent_2": []}
        assert proposal["synthetic_proposal"] is True

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

        result = await handler.run_proposal_phase(
            agents=agents,
            items=state["items"],
            preferences=preferences,
            round_num=1,
            max_rounds=3,
        )

        invalid_saved = [(args, kwargs) for args, kwargs in saved if "_invalid_attempt_" in args[1]]
        saved_prompts = [args[2] for args, _kwargs in invalid_saved]
        assert len(saved_prompts) == 5
        assert all("allocation" in prompt.lower() for prompt in saved_prompts)
        assert all("INVALID RAW RESPONSE" in prompt for prompt in saved_prompts[1:])
        final_diagnostic = json.loads(invalid_saved[-1][0][3])
        assert final_diagnostic["game_type"] == "item_allocation"
        assert result["proposals"][0]["synthetic_proposal"] is True

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


def test_diplomatic_treaty_prefinal_batch_vote_failure_defaults_to_reject_after_logging_raw_attempts():
    """Game 2 pre-final invalid batch votes should persist raw responses and continue."""

    async def run_test():
        bad_initial = "not valid json"
        bad_retry = "still not valid json"
        bad_compact = "compact repair also not json"
        game = create_game_environment(
            "diplomacy",
            n_agents=2,
            t_rounds=3,
            n_issues=3,
            random_seed=42,
        )
        agents = [
            FakeAgent("Agent_1", [bad_initial, bad_retry, bad_compact]),
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

        invalid_saved = [(args, kwargs) for args, kwargs in saved if "_batch_invalid_attempt_" in args[1]]
        assert [args[1] for args, _kwargs in invalid_saved] == [
            "voting_round_1_batch_invalid_attempt_0",
            "voting_round_1_batch_invalid_attempt_1",
            "voting_round_1_batch_invalid_attempt_2",
        ]
        payloads = [json.loads(args[3]) for args, _kwargs in invalid_saved]
        assert [payload["raw_response"] for payload in payloads] == [
            bad_initial,
            bad_retry,
            bad_compact,
        ]
        assert [payload["missing_proposal_numbers"] for payload in payloads] == [[1], [1], [1]]
        assert [payload["will_retry"] for payload in payloads] == [True, True, False]
        assert [payload["hard_failed"] for payload in payloads] == [False, False, False]
        assert result["private_votes"][0]["vote"] == "reject"
        assert result["private_votes"][0]["synthetic_vote"] is True
        assert result["voting_summary"]["vote_integrity"]["synthetic_vote_count"] == 1

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


def test_item_allocation_repairs_comments_in_json():
    """Game 1 should parse JSON comments before schema validation."""
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
        "Agent_1": [0, 2], // high value items
        "Agent_2": [1]
      },
      "reasoning": "comment should be stripped"
    }"""
    parsed = game.parse_proposal(response, "Agent_1", state, ["Agent_1", "Agent_2"])

    assert parsed["allocation"] == {"Agent_1": [0, 2], "Agent_2": [1]}
    assert "parse_error" not in parsed


def test_item_allocation_batch_vote_repairs_comments():
    """Game 1 batch vote parsing should tolerate comments in JSON arrays."""
    game = create_game_environment(
        "item_allocation",
        n_agents=2,
        t_rounds=3,
        m_items=3,
        random_seed=42,
    )

    response = """{
      "votes": [
        {"proposal_number": 1, "vote": "accept", "reasoning": "ok"}, // vote one
        {"proposal_number": 2, "vote": "reject", "reasoning": "no"}
      ]
    }"""
    votes = game.parse_batch_voting_response(response, [1, 2], "Agent_1", 1)

    assert [vote["vote"] for vote in votes] == ["accept", "reject"]
    assert all("parse_error" not in vote for vote in votes)


def test_cofunding_repairs_markdown_numbers_and_missing_comma():
    """Game 3 proposals should parse common markdown and comma issues."""
    game = create_game_environment(
        "co_funding",
        n_agents=2,
        t_rounds=3,
        m_projects=3,
        random_seed=42,
    )
    state = game.create_game_state([FakeAgent("Agent_1", ["unused"]), FakeAgent("Agent_2", ["unused"])])

    response = """{
      "contributions": [0.0, **10.0**, 5.0]
      "reasoning": "missing comma before reasoning"
    }"""
    parsed = game.parse_proposal(response, "Agent_1", state, ["Agent_1", "Agent_2"])

    assert parsed["contributions"] == [0.0, 10.0, 5.0]
    assert "parse_error" not in parsed

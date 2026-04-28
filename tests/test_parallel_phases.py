#!/usr/bin/env python3
"""Tests for deterministic parallel phase execution and interaction streaming."""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

from game_environments import create_game_environment
from strong_models_experiment.experiment import StrongModelsExperiment
from strong_models_experiment.phases.phase_handlers import PhaseHandler


@dataclass
class FakeAgentResponse:
    content: str
    metadata: Optional[Dict[str, Any]] = None
    tokens_used: Optional[int] = None
    response_time: Optional[float] = None


class DelayedVoteAgent:
    """Fake agent that lets a later agent finish first."""

    active_calls = 0
    max_active_calls = 0

    def __init__(self, agent_id: str, vote_response: str, delay: float):
        self.agent_id = agent_id
        self.vote_response = vote_response
        self.delay = delay
        self.prompts: List[str] = []
        self._max_tokens = None

    async def generate_response(self, context, prompt) -> FakeAgentResponse:
        self.prompts.append(prompt)
        DelayedVoteAgent.active_calls += 1
        DelayedVoteAgent.max_active_calls = max(
            DelayedVoteAgent.max_active_calls,
            DelayedVoteAgent.active_calls,
        )
        try:
            await asyncio.sleep(self.delay)
            return FakeAgentResponse(content=self.vote_response)
        finally:
            DelayedVoteAgent.active_calls -= 1

    def update_max_tokens(self, max_tokens):
        self._max_tokens = max_tokens

    def get_model_info(self) -> Dict[str, Any]:
        return {"model_name": "fake-model"}


def test_parallel_helper_launches_exactly_one_task_per_agent_and_preserves_order():
    async def run_test():
        class Agent:
            def __init__(self, agent_id):
                self.agent_id = agent_id

        agents = [Agent("Agent_1"), Agent("Agent_2"), Agent("Agent_3")]
        handler = PhaseHandler(parallel_phases=True)
        active = 0
        max_active = 0

        async def task(idx, agent):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            try:
                await asyncio.sleep(0.03 - idx * 0.01)
                return agent.agent_id
            finally:
                active -= 1

        results = await handler._run_agent_tasks_in_order(agents, task)
        assert results == ["Agent_1", "Agent_2", "Agent_3"]
        assert max_active == len(agents)

    asyncio.run(run_test())


def test_serial_helper_runs_one_agent_at_a_time():
    async def run_test():
        class Agent:
            def __init__(self, agent_id):
                self.agent_id = agent_id

        agents = [Agent("Agent_1"), Agent("Agent_2"), Agent("Agent_3")]
        handler = PhaseHandler(parallel_phases=False)
        active = 0
        max_active = 0

        async def task(_idx, agent):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            try:
                await asyncio.sleep(0)
                return agent.agent_id
            finally:
                active -= 1

        results = await handler._run_agent_tasks_in_order(agents, task)
        assert results == ["Agent_1", "Agent_2", "Agent_3"]
        assert max_active == 1

    asyncio.run(run_test())


def test_parallel_phases_rejects_numeric_partial_concurrency():
    with pytest.raises(TypeError, match="parallel_phases must be a boolean"):
        PhaseHandler(parallel_phases=3)


def test_extract_token_usage_preserves_response_time_seconds():
    response = FakeAgentResponse(
        content="ok",
        metadata={"usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}},
        response_time=1.25,
    )

    token_usage = PhaseHandler()._extract_token_usage(response)

    assert token_usage == {
        "input_tokens": 3,
        "output_tokens": 4,
        "total_tokens": 7,
        "response_time_seconds": 1.25,
    }


def test_extract_token_usage_preserves_direct_anthropic_metadata():
    response = FakeAgentResponse(
        content="ok",
        metadata={
            "input_tokens": 11,
            "output_tokens": 13,
            "total_tokens": 29,
            "reasoning_tokens": 5,
            "thinking_tokens": 5,
        },
        response_time=2.5,
    )

    token_usage = PhaseHandler()._extract_token_usage(response)

    assert token_usage == {
        "input_tokens": 11,
        "output_tokens": 13,
        "total_tokens": 29,
        "reasoning_tokens": 5,
        "thinking_tokens": 5,
        "response_time_seconds": 2.5,
    }


def test_legacy_vote_parsers_repair_literal_newlines_inside_json_strings():
    handler = PhaseHandler()
    vote = handler._parse_vote_response(
        """{
          "vote": "accept",
          "reasoning": "First line.

Second line."
        }""",
        "Agent_1",
        1,
    )
    commit_vote = handler._parse_commit_vote_response(
        """{
          "commit_vote": "yay",
          "reasoning": "First line.

Second line."
        }""",
        "Agent_1",
        1,
    )

    assert vote["vote"] == "accept"
    assert "Second line" in vote["reasoning"]
    assert commit_vote["commit_vote"] == "yay"
    assert "Second line" in commit_vote["reasoning"]


def test_parallel_voting_saves_in_agent_order_even_when_completion_order_differs():
    async def run_test():
        DelayedVoteAgent.active_calls = 0
        DelayedVoteAgent.max_active_calls = 0

        game = create_game_environment(
            "item_allocation",
            n_agents=2,
            t_rounds=3,
            m_items=3,
            random_seed=42,
        )
        agent_1 = DelayedVoteAgent(
            "Agent_1",
            json.dumps(
                {
                    "votes": [
                        {"proposal_number": 1, "vote": "accept", "reasoning": "A1 accepts one"},
                        {"proposal_number": 2, "vote": "reject", "reasoning": "A1 rejects two"},
                    ]
                }
            ),
            delay=0.05,
        )
        agent_2 = DelayedVoteAgent(
            "Agent_2",
            json.dumps(
                {
                    "votes": [
                        {"proposal_number": 1, "vote": "reject", "reasoning": "A2 rejects one"},
                        {"proposal_number": 2, "vote": "accept", "reasoning": "A2 accepts two"},
                    ]
                }
            ),
            delay=0.0,
        )
        agents = [agent_1, agent_2]
        state = game.create_game_state(agents)
        preferences = {
            "agent_preferences": state["agent_preferences"],
            "game_state": state,
        }

        proposal_1 = {
            "allocation": {"Agent_1": [0], "Agent_2": [1, 2]},
            "reasoning": "Proposal one",
            "proposed_by": "Agent_1",
            "round": 1,
        }
        proposal_2 = {
            "allocation": {"Agent_1": [1], "Agent_2": [0, 2]},
            "reasoning": "Proposal two",
            "proposed_by": "Agent_2",
            "round": 1,
        }
        enumerated_proposals = [
            {
                "proposal_number": 1,
                "proposer": "Agent_1",
                "reasoning": "Proposal one",
                "original_proposal": proposal_1,
                "allocation": proposal_1["allocation"],
            },
            {
                "proposal_number": 2,
                "proposer": "Agent_2",
                "reasoning": "Proposal two",
                "original_proposal": proposal_2,
                "allocation": proposal_2["allocation"],
            },
        ]

        saved_interactions = []

        def save_interaction(agent_id, phase, prompt, response, round_num=None, token_usage=None, model_name=None):
            saved_interactions.append((agent_id, phase))

        handler = PhaseHandler(
            save_interaction_callback=save_interaction,
            game_environment=game,
            parallel_phases=True,
        )
        result = await handler.run_private_voting_phase(
            agents=agents,
            items=state["items"],
            preferences=preferences,
            round_num=1,
            max_rounds=3,
            proposals=[proposal_1, proposal_2],
            enumerated_proposals=enumerated_proposals,
        )

        assert DelayedVoteAgent.max_active_calls == len(agents)
        assert saved_interactions == [
            ("Agent_1", "voting_round_1_proposal_1"),
            ("Agent_1", "voting_round_1_proposal_2"),
            ("Agent_2", "voting_round_1_proposal_1"),
            ("Agent_2", "voting_round_1_proposal_2"),
        ]
        assert [
            (vote["voter_id"], vote["proposal_number"], vote["vote"])
            for vote in result["private_votes"]
        ] == [
            ("Agent_1", 1, "accept"),
            ("Agent_1", 2, "reject"),
            ("Agent_2", 1, "reject"),
            ("Agent_2", 2, "accept"),
        ]

    asyncio.run(run_test())


def test_parallel_proposals_save_in_agent_order_even_when_completion_order_differs():
    async def run_test():
        DelayedVoteAgent.active_calls = 0
        DelayedVoteAgent.max_active_calls = 0

        game = create_game_environment(
            "item_allocation",
            n_agents=2,
            t_rounds=3,
            m_items=3,
            random_seed=42,
        )
        agents = [
            DelayedVoteAgent(
                "Agent_1",
                json.dumps(
                    {
                        "allocation": {"Agent_1": [0], "Agent_2": [1, 2]},
                        "reasoning": "Agent 1 proposal",
                    }
                ),
                delay=0.05,
            ),
            DelayedVoteAgent(
                "Agent_2",
                json.dumps(
                    {
                        "allocation": {"Agent_1": [2], "Agent_2": [0, 1]},
                        "reasoning": "Agent 2 proposal",
                    }
                ),
                delay=0.0,
            ),
        ]
        state = game.create_game_state(agents)
        preferences = {
            "agent_preferences": state["agent_preferences"],
            "game_state": state,
        }
        saved_interactions = []

        def save_interaction(agent_id, phase, prompt, response, round_num=None, token_usage=None, model_name=None):
            saved_interactions.append((agent_id, phase, json.loads(response)["proposed_by"]))

        handler = PhaseHandler(
            save_interaction_callback=save_interaction,
            game_environment=game,
            parallel_phases=True,
        )

        result = await handler.run_proposal_phase(
            agents=agents,
            items=state["items"],
            preferences=preferences,
            round_num=1,
            max_rounds=3,
        )

        assert DelayedVoteAgent.max_active_calls == len(agents)
        assert saved_interactions == [
            ("Agent_1", "proposal_round_1", "Agent_1"),
            ("Agent_2", "proposal_round_1", "Agent_2"),
        ]
        assert [proposal["proposed_by"] for proposal in result["proposals"]] == ["Agent_1", "Agent_2"]
        assert [message["from"] for message in result["messages"]] == ["Agent_1", "Agent_2"]

    asyncio.run(run_test())


def test_stream_save_overwrites_agent_files_without_duplicate_history(tmp_path):
    experiment = StrongModelsExperiment(output_dir=tmp_path)
    experiment.current_experiment_id = "test_run"

    experiment._save_interaction("Agent_1", "phase_a", "prompt 1", "response 1", 1)
    experiment._save_interaction("Agent_1", "phase_b", "prompt 2", "response 2", 1)
    experiment._save_interaction("Agent_2", "phase_a", "prompt 3", "response 3", 1)
    experiment._stream_save_json()

    all_interactions = json.loads((tmp_path / "all_interactions.json").read_text())
    assert len(all_interactions) == 3

    agent_1_payload = json.loads((tmp_path / "agent_Agent_1_interactions.json").read_text())
    assert agent_1_payload["total_interactions"] == 2
    assert [entry["phase"] for entry in agent_1_payload["interactions"]] == ["phase_a", "phase_b"]

    agent_2_payload = json.loads((tmp_path / "agent_Agent_2_interactions.json").read_text())
    assert agent_2_payload["total_interactions"] == 1
    assert [entry["phase"] for entry in agent_2_payload["interactions"]] == ["phase_a"]

#!/usr/bin/env python3
"""Regression tests for Game 3 context propagation across phases and rounds."""

import asyncio
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strong_models_experiment import StrongModelsExperiment


@dataclass
class FakeAgentResponse:
    """Minimal response object for experiment tests."""

    content: str
    metadata: Optional[Dict[str, Any]] = None
    tokens_used: Optional[int] = None


class RecordingAgent:
    """Fake agent that records every context it receives."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.calls: List[Dict[str, Any]] = []
        self._max_tokens = None

    def _record_call(self, context, prompt: str) -> None:
        self.calls.append(
            {
                "turn_type": context.turn_type,
                "current_round": context.current_round,
                "context": copy.deepcopy(context.to_dict()),
                "prompt": prompt,
            }
        )

    async def generate_response(self, context, prompt) -> FakeAgentResponse:
        self._record_call(context, prompt)

        if context.turn_type == "setup":
            return FakeAgentResponse(content=f"{self.agent_id} acknowledges setup.")

        if context.turn_type == "discussion":
            return FakeAgentResponse(
                content=f"{self.agent_id} discussion for round {context.current_round}."
            )

        if context.turn_type == "proposal":
            return FakeAgentResponse(
                content=json.dumps(
                    {
                        "contributions": [1.0, 0.0, 0.0],
                        "reasoning": f"{self.agent_id} proposal reasoning",
                    }
                )
            )

        if context.turn_type == "private_voting":
            vote = "reject" if context.current_round == 1 else "accept"
            return FakeAgentResponse(
                content=json.dumps(
                    {
                        "vote": vote,
                        "reasoning": f"{self.agent_id} {vote}s the joint proposal",
                    }
                )
            )

        if context.turn_type == "reflection":
            return FakeAgentResponse(
                content=f"{self.agent_id} reflection for round {context.current_round}."
            )

        raise AssertionError(f"Unexpected turn_type: {context.turn_type}")

    async def think_strategy(self, prompt, context) -> Dict[str, Any]:
        self._record_call(context, prompt)
        return {
            "reasoning": f"{self.agent_id} private reasoning for round {context.current_round}",
            "strategy": f"{self.agent_id} strategy for round {context.current_round}",
            "key_priorities": ["Project Alpha"],
            "potential_concessions": ["Project Beta"],
        }

    def update_max_tokens(self, max_tokens):
        self._max_tokens = max_tokens

    def get_model_info(self) -> Dict[str, Any]:
        return {"model_name": f"fake-{self.agent_id.lower()}"}


def _find_call(agent: RecordingAgent, turn_type: str, round_num: int) -> Dict[str, Any]:
    for call in agent.calls:
        if call["turn_type"] == turn_type and call["current_round"] == round_num:
            return call
    raise AssertionError(f"No {turn_type} call recorded for round {round_num}")


def _history_phases(call: Dict[str, Any]) -> set[tuple[int, str]]:
    history = call["context"]["conversation_history"]
    return {
        (int(entry.get("round") or 0), str(entry.get("phase")))
        for entry in history
    }


def test_game3_round_context_includes_previous_phases_and_previous_rounds(monkeypatch, tmp_path):
    """Game 3 phases should receive prior public phases plus each agent's own private notes."""

    async def run_test():
        experiment = StrongModelsExperiment(output_dir=tmp_path)
        agents = [RecordingAgent("Agent_1"), RecordingAgent("Agent_2")]

        async def fake_create_agents(models, config):
            return agents

        monkeypatch.setattr(experiment.agent_factory, "create_agents", fake_create_agents)

        await experiment.run_single_experiment(
            models=["fake-model-a", "fake-model-b"],
            experiment_config={
                "game_type": "co_funding",
                "m_projects": 3,
                "t_rounds": 2,
                "random_seed": 42,
                "discussion_turns": 1,
                "pledge_mode": "individual",
                "disable_reflection": False,
            },
        )

        agent_1 = agents[0]

        thinking_round_1 = _find_call(agent_1, "thinking", 1)
        assert (1, "discussion") in _history_phases(thinking_round_1)

        proposal_round_1 = _find_call(agent_1, "proposal", 1)
        assert (1, "discussion") in _history_phases(proposal_round_1)
        assert any(
            "Round 1 | Private Thinking" in note
            for note in proposal_round_1["context"]["strategic_notes"]
        )

        voting_round_1 = _find_call(agent_1, "private_voting", 1)
        assert {
            (1, "discussion"),
            (1, "proposal"),
            (1, "proposal_enumeration"),
        }.issubset(_history_phases(voting_round_1))
        assert any(
            "Round 1 | Proposal" in note
            for note in voting_round_1["context"]["strategic_notes"]
        )

        reflection_round_1 = _find_call(agent_1, "reflection", 1)
        assert (1, "vote_tabulation") in _history_phases(reflection_round_1)
        assert any(
            "Round 1 | Private Voting" in note
            for note in reflection_round_1["context"]["strategic_notes"]
        )

        discussion_round_2 = _find_call(agent_1, "discussion", 2)
        assert {
            (1, "discussion"),
            (1, "proposal"),
            (1, "proposal_enumeration"),
            (1, "vote_tabulation"),
        }.issubset(_history_phases(discussion_round_2))
        assert any(
            "Round 1 | Reflection" in note
            for note in discussion_round_2["context"]["strategic_notes"]
        )

        proposal_round_2 = _find_call(agent_1, "proposal", 2)
        assert {
            (1, "discussion"),
            (1, "proposal"),
            (1, "proposal_enumeration"),
            (1, "vote_tabulation"),
            (2, "discussion"),
        }.issubset(_history_phases(proposal_round_2))
        assert any(
            "Round 2 | Private Thinking" in note
            for note in proposal_round_2["context"]["strategic_notes"]
        )

        voting_round_2 = _find_call(agent_1, "private_voting", 2)
        assert {
            (2, "discussion"),
            (2, "proposal"),
            (2, "proposal_enumeration"),
        }.issubset(_history_phases(voting_round_2))
        assert any(
            "Round 2 | Proposal" in note
            for note in voting_round_2["context"]["strategic_notes"]
        )

    asyncio.run(run_test())

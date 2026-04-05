#!/usr/bin/env python3
"""Regression tests for Game 2 context propagation and preference replay."""

import asyncio
import copy
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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

        if context.turn_type in {"setup", "preference_assignment"}:
            return FakeAgentResponse(content=f"{self.agent_id} acknowledges setup.")

        if context.turn_type == "discussion":
            return FakeAgentResponse(
                content=f"{self.agent_id} discussion for round {context.current_round}."
            )

        if context.turn_type == "proposal":
            if self.agent_id == "Agent_1":
                proposal = {
                    "agreement": [20, 40, 60],
                    "reasoning": "Agent 1 treaty proposal reasoning",
                }
            else:
                proposal = {
                    "agreement": [30, 50, 70],
                    "reasoning": "Agent 2 treaty proposal reasoning",
                }
            return FakeAgentResponse(content=json.dumps(proposal))

        if context.turn_type == "private_voting":
            return FakeAgentResponse(
                content=json.dumps(
                    {
                        "votes": [
                            {
                                "proposal_number": 1,
                                "vote": "reject",
                                "reasoning": f"{self.agent_id} rejects proposal 1",
                            },
                            {
                                "proposal_number": 2,
                                "vote": "reject",
                                "reasoning": f"{self.agent_id} rejects proposal 2",
                            },
                        ]
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
            "key_priorities": ["Issue A"],
            "potential_concessions": ["Issue C"],
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


def test_game2_later_phases_replay_positions_and_weights(monkeypatch, tmp_path):
    """Game 2 later phases should receive full treaty preferences, not positions alone."""

    async def run_test():
        experiment = StrongModelsExperiment(output_dir=tmp_path)
        agents = [RecordingAgent("Agent_1"), RecordingAgent("Agent_2")]

        async def fake_create_agents(models, config):
            return agents

        monkeypatch.setattr(experiment.agent_factory, "create_agents", fake_create_agents)

        await experiment.run_single_experiment(
            models=["fake-model-a", "fake-model-b"],
            experiment_config={
                "game_type": "diplomacy",
                "n_issues": 3,
                "t_rounds": 2,
                "gamma_discount": 0.9,
                "rho": 0.0,
                "theta": 0.5,
                "random_seed": 42,
                "discussion_turns": 1,
            },
        )

        agent_1 = agents[0]

        for turn_type, round_num in [
            ("discussion", 1),
            ("thinking", 1),
            ("proposal", 1),
            ("private_voting", 1),
            ("reflection", 1),
            ("discussion", 2),
            ("proposal", 2),
            ("private_voting", 2),
            ("reflection", 2),
        ]:
            call = _find_call(agent_1, turn_type, round_num)
            context_prefs = call["context"]["preferences"]
            assert set(context_prefs.keys()) >= {"positions", "weights", "issues"}
            assert len(context_prefs["positions"]) == 3
            assert len(context_prefs["weights"]) == 3
            assert len(context_prefs["issues"]) == 3

    asyncio.run(run_test())

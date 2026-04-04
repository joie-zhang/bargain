#!/usr/bin/env python3
"""Tests for Game 2's simultaneous multi-proposal voting flow."""

import asyncio
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
    """Fake agent that returns a predetermined batch vote response."""

    def __init__(self, agent_id: str, vote_response: str):
        self.agent_id = agent_id
        self.vote_response = vote_response
        self.prompts: List[str] = []
        self._max_tokens = None

    async def generate_response(self, context, prompt) -> FakeAgentResponse:
        self.prompts.append(prompt)
        return FakeAgentResponse(content=self.vote_response)

    def update_max_tokens(self, max_tokens):
        self._max_tokens = max_tokens

    def get_model_info(self) -> Dict[str, Any]:
        return {"model_name": "fake-model"}


def test_private_voting_phase_batches_diplomatic_treaty_votes():
    """Diplomatic treaty voting should prompt once per agent with all proposals."""

    async def run_test():
        game = create_game_environment(
            "diplomacy",
            n_agents=2,
            t_rounds=3,
            n_issues=3,
            random_seed=42,
        )

        agent_1 = FakeAgent(
            "Agent_1",
            """
            {
                "votes": [
                    {"proposal_number": 1, "vote": "accept", "reasoning": "Closer on chips and minerals"},
                    {"proposal_number": 2, "vote": "reject", "reasoning": "Oversight rate is too far from ideal"}
                ]
            }
            """,
        )
        agent_2 = FakeAgent(
            "Agent_2",
            """
            {
                "votes": [
                    {"proposal_number": 1, "vote": "reject", "reasoning": "Need stronger oversight"},
                    {"proposal_number": 2, "vote": "accept", "reasoning": "Closer overall package"}
                ]
            }
            """,
        )
        agents = [agent_1, agent_2]
        state = game.create_game_state(agents)
        preferences = {
            "agent_preferences": state["agent_positions"],
            "agent_weights": state["agent_weights"],
            "game_state": state,
        }
        items = [{"name": issue} for issue in state["issues"]]

        proposal_1 = {
            "agreement": [65, 20, 55],
            "reasoning": "Proposal one",
            "proposed_by": "Agent_1",
            "round": 1,
        }
        proposal_2 = {
            "agreement": [45, 60, 35],
            "reasoning": "Proposal two",
            "proposed_by": "Agent_2",
            "round": 1,
        }
        proposals = [proposal_1, proposal_2]
        enumerated_proposals = [
            {
                "proposal_number": 1,
                "proposer": "Agent_1",
                "reasoning": "Proposal one",
                "original_proposal": proposal_1,
                "agreement": proposal_1["agreement"],
            },
            {
                "proposal_number": 2,
                "proposer": "Agent_2",
                "reasoning": "Proposal two",
                "original_proposal": proposal_2,
                "agreement": proposal_2["agreement"],
            },
        ]

        saved_interactions = []

        def save_interaction(agent_id, phase, prompt, response, round_num=None, token_usage=None, model_name=None):
            saved_interactions.append(
                {
                    "agent_id": agent_id,
                    "phase": phase,
                    "prompt": prompt,
                    "response": response,
                    "round": round_num,
                    "token_usage": token_usage,
                    "model_name": model_name,
                }
            )

        handler = PhaseHandler(save_interaction_callback=save_interaction, game_environment=game)

        result = await handler.run_private_voting_phase(
            agents=agents,
            items=items,
            preferences=preferences,
            round_num=1,
            max_rounds=3,
            proposals=proposals,
            enumerated_proposals=enumerated_proposals,
        )

        assert len(agent_1.prompts) == 1
        assert len(agent_2.prompts) == 1
        assert "PROPOSAL #1" in agent_1.prompts[0]
        assert "PROPOSAL #2" in agent_1.prompts[0]
        assert "AI chip export quota" in agent_1.prompts[0]
        assert "REASONING:" not in agent_1.prompts[0]
        assert "Proposal one" not in agent_1.prompts[0]
        assert "Proposal two" not in agent_1.prompts[0]

        assert [
            (vote["voter_id"], vote["proposal_number"], vote["vote"], vote["reasoning"])
            for vote in result["private_votes"]
        ] == [
            ("Agent_1", 1, "accept", "Closer on chips and minerals"),
            ("Agent_1", 2, "reject", "Oversight rate is too far from ideal"),
            ("Agent_2", 1, "reject", "Need stronger oversight"),
            ("Agent_2", 2, "accept", "Closer overall package"),
        ]

        saved_phases = [entry["phase"] for entry in saved_interactions]
        assert saved_phases == [
            "voting_round_1_proposal_1",
            "voting_round_1_proposal_2",
            "voting_round_1_proposal_1",
            "voting_round_1_proposal_2",
        ]

    asyncio.run(run_test())

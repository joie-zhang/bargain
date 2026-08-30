import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from game_environments import create_game_environment
from strong_models_experiment.agents.agent_factory import StrongModelAgentFactory
from strong_models_experiment.phases.phase_handlers import PhaseHandler


def _team_handler(member_ids):
    return PhaseHandler(
        team_coordination={
            "enabled": True,
            "member_ids": member_ids,
            "captain_id": member_ids[0],
            "share_private_thinking": True,
            "share_private_voting": True,
            "share_reflection": True,
        }
    )


def test_direct_openai_override_replaces_openrouter_parameters():
    config = StrongModelAgentFactory.resolve_model_config(
        "gpt-5.4-high",
        {
            "model_config_overrides": {
                "gpt-5.4-high": {
                    "api_type": "openai",
                    "provider": "OpenAI",
                    "model_id": "gpt-5.4",
                    "reasoning_effort": "high",
                    "custom_parameters": {
                        "phase_token_cap_policy": "prefer_model_cap_when_experiment_default"
                    },
                }
            }
        },
    )
    assert config["api_type"] == "openai"
    assert config["model_id"] == "gpt-5.4"
    assert config["reasoning_effort"] == "high"
    assert "reasoning" not in config["custom_parameters"]


def test_team_briefing_is_private_and_contains_all_team_preferences():
    handler = _team_handler(["Agent_1", "Agent_2", "Agent_3"])
    items = [{"name": "Apple"}, {"name": "Book"}]
    preferences = {
        "agent_preferences": {
            "Agent_1": [9, 1],
            "Agent_2": [2, 8],
            "Agent_3": [4, 6],
            "Agent_4": [10, 10],
        }
    }
    briefing = handler.build_team_coordination_briefing("Agent_2", items, preferences)
    assert "TEAM MEMBERS: Agent_1, Agent_2, Agent_3" in briefing
    assert "DESIGNATED TEAM CAPTAIN: Agent_1" in briefing
    assert "Agent_1: 0:Apple=9, 1:Book=1" in briefing
    assert "Agent_3: 0:Apple=4, 1:Book=6" in briefing
    assert "Agent_4" not in briefing
    assert handler.build_team_coordination_briefing("Agent_4", items, preferences) == ""


def test_team_notes_share_only_selected_private_phases():
    handler = _team_handler(["Agent_2", "Agent_3", "Agent_4"])
    assert handler.team_note_recipients("Agent_3", "Private Thinking") == [
        "Agent_2",
        "Agent_3",
        "Agent_4",
    ]
    assert handler.team_note_recipients("Agent_3", "Private Voting") == [
        "Agent_2",
        "Agent_3",
        "Agent_4",
    ]
    assert handler.team_note_recipients("Agent_3", "Proposal") == ["Agent_3"]
    assert handler.team_note_recipients("Agent_1", "Private Thinking") == ["Agent_1"]


def test_singleton_team_is_a_true_noop():
    handler = _team_handler(["Agent_1"])
    items = [{"name": "Apple"}]
    preferences = {"agent_preferences": {"Agent_1": [10], "Agent_2": [0]}}
    assert handler.build_team_coordination_briefing("Agent_1", items, preferences) == ""
    assert handler.team_note_recipients("Agent_1", "Private Thinking") == ["Agent_1"]


def _binding_handler(member_ids):
    game_environment = create_game_environment(
        game_type="item_allocation",
        n_agents=4,
        t_rounds=10,
        gamma_discount=0.9,
        random_seed=17,
        m_items=4,
        competition_level=0.5,
    )
    return PhaseHandler(
        game_environment=game_environment,
        team_coordination={
            "enabled": True,
            "protocol_version": PhaseHandler.BINDING_TEAM_PROTOCOL_VERSION,
            "member_ids": member_ids,
            "captain_id": member_ids[0],
            "rotate_captain_each_round": True,
            "share_team_planning": True,
            "planning_turns": 3,
            "planning_max_tokens": 8192,
            "ballot_max_tokens": 4096,
        },
        parallel_phases=True,
    )


def test_binding_objective_is_pure_team_sum_and_captain_rotates():
    handler = _binding_handler(["Agent_1", "Agent_2", "Agent_3"])
    objective = handler.build_team_system_objective("Agent_2")
    assert "sole objective" in objective.lower()
    assert "discounted sum" in objective.lower()
    assert "no special priority" in objective.lower()
    assert "fairness objective" in objective.lower()
    discussion = handler.apply_team_objective_to_prompt(
        "Agent_2",
        "- Identify mutually beneficial trade possibilities",
        phase_label="public discussion",
    )
    assert "mutually beneficial" not in discussion
    assert "increase the Nano team's expected discounted utility" in discussion
    assert handler.team_captain_id(1) == "Agent_1"
    assert handler.team_captain_id(2) == "Agent_2"
    assert handler.team_captain_id(3) == "Agent_3"


@dataclass
class _FakeResponse:
    content: str
    metadata: Optional[Dict[str, Any]] = None
    tokens_used: Optional[int] = None
    response_time: Optional[float] = None


class _BindingAgent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.max_tokens = None

    def update_max_tokens(self, max_tokens):
        self.max_tokens = max_tokens

    def get_model_info(self):
        return {"model_name": "fake-model"}

    async def generate_response(self, context, prompt):
        if "You are the captain" in prompt:
            return _FakeResponse(content=json.dumps({
                "allocation": {
                    "Agent_1": ["Apple"],
                    "Agent_2": ["1:Jewel"],
                    "Agent_3": ["Stone"],
                    "Agent_4": ["3"],
                },
                "team_member_utilities": {
                    "Agent_1": 100,
                    "Agent_2": 100,
                    "Agent_3": 100,
                },
                "team_utility_sum": 100,
                "reasoning": "Highest Nano team sum found in the team room.",
            }))
        if "selected_proposal_number" in prompt:
            return _FakeResponse(content=json.dumps({
                "selected_proposal_number": 1,
                "reasoning": "Proposal 1 has the best expected discounted Nano sum.",
            }))
        if '"votes"' in prompt:
            return _FakeResponse(content=json.dumps({
                "votes": [
                    {"proposal_number": 1, "vote": "reject", "reasoning": "Low own value"},
                    {"proposal_number": 2, "vote": "accept", "reasoning": "Higher own value"},
                ]
            }))
        return _FakeResponse(content=json.dumps({
            "analysis": "Candidate or audit message",
            "candidate_allocation": {
                "Agent_1": [0], "Agent_2": [1], "Agent_3": [2], "Agent_4": [3]
            },
        }))


def test_three_turn_planning_and_binding_ballot_have_no_synthetic_actions():
    async def run_test():
        interactions = []
        handler = _binding_handler(["Agent_1", "Agent_2", "Agent_3"])
        handler.save_interaction = lambda *args, **kwargs: interactions.append((args, kwargs))
        agents = [_BindingAgent(f"Agent_{index}") for index in range(1, 5)]
        game_state = handler.game_environment.create_game_state(agents)
        game_state["agent_preferences"] = {
            "Agent_1": [9.0, 1.0, 1.0, 1.0],
            "Agent_2": [1.0, 9.0, 1.0, 1.0],
            "Agent_3": [1.0, 1.0, 9.0, 1.0],
            "Agent_4": [1.0, 1.0, 1.0, 9.0],
        }
        preferences = {
            "game_state": game_state,
            "agent_preferences": game_state["agent_preferences"],
        }
        private_context = {agent.agent_id: [] for agent in agents}
        plan = await handler.run_team_planning_phase(
            agents,
            game_state["items"],
            preferences,
            1,
            10,
            private_context_by_agent=private_context,
        )
        assert len(plan["planning_messages"]) == 9
        assert plan["captain_id"] == "Agent_1"
        assert plan["final_team_proposal"]["coalition_proposal"] is True
        assert plan["final_team_proposal"]["team_raw_utility"] == 27
        assert plan["final_team_proposal"]["team_utility_evaluator"] == "environment"
        assert handler.game_environment.validate_proposal(
            plan["final_team_proposal"], game_state
        )

        team_proposal = plan["final_team_proposal"]
        adversary_proposal = {
            "allocation": {
                "Agent_1": [3], "Agent_2": [0], "Agent_3": [1], "Agent_4": [2]
            },
            "reasoning": "Adversary option",
            "proposed_by": "Agent_4",
            "round": 1,
        }
        enumerated = [
            {
                "proposal_number": 1,
                "proposer": "Agent_1",
                "allocation": team_proposal["allocation"],
                "reasoning": team_proposal["reasoning"],
                "original_proposal": team_proposal,
            },
            {
                "proposal_number": 2,
                "proposer": "Agent_4",
                "allocation": adversary_proposal["allocation"],
                "reasoning": adversary_proposal["reasoning"],
                "original_proposal": adversary_proposal,
            },
        ]
        ballot = await handler.run_strict_binding_voting_phase(
            agents,
            game_state["items"],
            preferences,
            1,
            10,
            enumerated,
            private_context_by_agent=private_context,
        )
        assert len(ballot["private_votes"]) == 8
        team_votes = [
            vote for vote in ballot["private_votes"]
            if vote["voter_id"] in {"Agent_1", "Agent_2", "Agent_3"}
        ]
        assert all(
            vote["vote"] == ("accept" if vote["proposal_number"] == 1 else "reject")
            for vote in team_votes
        )
        assert not any(vote.get("synthetic_vote") for vote in ballot["private_votes"])

    asyncio.run(run_test())

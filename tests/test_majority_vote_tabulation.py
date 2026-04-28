import asyncio

import strong_models_experiment.phases.phase_handlers as phase_handlers_module
from strong_models_experiment.phases.phase_handlers import PhaseHandler


class FakeAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id


def _run_tabulation(n_agents, private_votes, enumerated_proposals):
    agents = [FakeAgent(f"Agent_{idx}") for idx in range(1, n_agents + 1)]
    preferences = {
        "agent_preferences": {
            agent.agent_id: [1.0, 1.0, 1.0] for agent in agents
        }
    }
    return asyncio.run(
        PhaseHandler().run_vote_tabulation_phase(
            agents=agents,
            items=[{"name": "A"}, {"name": "B"}, {"name": "C"}],
            preferences=preferences,
            round_num=1,
            max_rounds=3,
            private_votes=private_votes,
            enumerated_proposals=enumerated_proposals,
        )
    )


def _votes(n_agents, accepts_by_proposal):
    votes = []
    for proposal_number, accept_count in accepts_by_proposal.items():
        for agent_idx in range(1, n_agents + 1):
            votes.append({
                "voter_id": f"Agent_{agent_idx}",
                "proposal_number": proposal_number,
                "vote": "accept" if agent_idx <= accept_count else "reject",
                "reasoning": "test vote",
            })
    return votes


def test_supermajority_accepts_for_n4():
    proposals = [
        {"proposal_number": 1, "allocation": {"Agent_1": [0]}},
        {"proposal_number": 2, "allocation": {"Agent_1": [1]}},
    ]

    result = _run_tabulation(
        n_agents=4,
        private_votes=_votes(4, {1: 3, 2: 2}),
        enumerated_proposals=proposals,
    )

    assert result["consensus_reached"] is True
    assert result["supermajority_threshold"] == 3
    assert result["majority_threshold"] == 3
    assert result["acceptance_rule"] == "two_thirds_supermajority"
    assert result["accepted_proposal_number"] == 1
    assert result["accepted_vote_count"] == 3
    assert result["final_allocation"] == {"Agent_1": [0]}


def test_n2_supermajority_still_requires_both_accepts():
    proposals = [
        {"proposal_number": 1, "allocation": {"Agent_1": [0]}},
    ]

    result = _run_tabulation(
        n_agents=2,
        private_votes=_votes(2, {1: 1}),
        enumerated_proposals=proposals,
    )

    assert result["consensus_reached"] is False
    assert result["supermajority_threshold"] == 2
    assert result["majority_threshold"] == 2
    assert result["accepted_proposal_number"] is None


def test_most_accept_votes_wins_when_multiple_proposals_have_supermajority():
    proposals = [
        {"proposal_number": 1, "allocation": {"Agent_1": [0]}},
        {"proposal_number": 2, "allocation": {"Agent_1": [1]}},
    ]

    result = _run_tabulation(
        n_agents=10,
        private_votes=_votes(10, {1: 6, 2: 7}),
        enumerated_proposals=proposals,
    )

    assert result["consensus_reached"] is True
    assert result["supermajority_threshold"] == 7
    assert result["majority_threshold"] == 7
    assert result["accepted_proposal_number"] == 2
    assert result["accepted_vote_count"] == 7
    assert result["final_allocation"] == {"Agent_1": [1]}


def test_tied_top_supermajority_proposals_are_randomly_selected(monkeypatch):
    proposals = [
        {"proposal_number": 1, "allocation": {"Agent_1": [0]}},
        {"proposal_number": 2, "allocation": {"Agent_1": [1]}},
    ]

    def choose_second(options):
        assert options == [1, 2]
        return 2

    monkeypatch.setattr(phase_handlers_module.random, "choice", choose_second)
    result = _run_tabulation(
        n_agents=4,
        private_votes=_votes(4, {1: 3, 2: 3}),
        enumerated_proposals=proposals,
    )

    assert result["consensus_reached"] is True
    assert result["accepted_proposal_number"] == 2
    assert result["accepted_vote_count"] == 3
    assert result["tied_winning_proposals"] == [1, 2]
    assert result["final_allocation"] == {"Agent_1": [1]}

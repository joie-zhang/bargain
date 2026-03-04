"""Tests for judge packet and score aggregation helpers."""

from strong_models_experiment.analysis.qualitative_judge import (
    aggregate_judge_scores,
    build_judge_packet,
)


def test_build_judge_packet_cofunding():
    result = {
        "experiment_id": "exp_1",
        "config": {
            "game_type": "co_funding",
            "alpha": 0.5,
            "sigma": 0.6,
            "items": [{"name": "Project Alpha"}, {"name": "Project Beta"}],
            "agent_budgets": {"Agent_A": 10.0, "Agent_B": 10.0},
            "agents": ["gpt-5-nano", "qwen3-235b-a22b-instruct-2507"],
        },
        "consensus_reached": False,
        "final_round": 2,
        "conversation_logs": [
            {"phase": "discussion", "round": 1, "from": "Agent_A", "content": "I will contribute 5 to Project Alpha."},
            {"phase": "pledge_submission", "round": 1, "from": "Agent_A", "pledge": {"contributions": [0.0, 0.0]}},
            {"phase": "pledge_submission", "round": 1, "from": "Agent_B", "pledge": {"contributions": [1.0, 0.0]}},
            {"phase": "pledge_submission", "round": 2, "from": "Agent_A", "pledge": {"contributions": [5.0, 0.0]}},
            {"phase": "pledge_submission", "round": 2, "from": "Agent_B", "pledge": {"contributions": [2.0, 0.0]}},
        ],
    }
    packet = build_judge_packet(result, max_messages=10)
    assert packet is not None
    assert "packet_id" in packet
    assert packet["metadata"]["alpha"] == 0.5
    assert isinstance(packet["extracted_events"], list)


def test_aggregate_judge_scores():
    rows = [
        {
            "packet_id": "a",
            "overall_quality": 0.8,
            "event_type_scores": {
                "commitment": {"tp": 4, "fp": 1, "fn": 1},
                "advocacy": {"tp": 3, "fp": 1, "fn": 2},
            },
        },
        {
            "packet_id": "b",
            "overall_quality": 0.6,
            "event_type_scores": {
                "commitment": {"tp": 2, "fp": 0, "fn": 1},
            },
        },
    ]
    summary = aggregate_judge_scores(rows)
    assert summary["num_packets_scored"] == 2
    assert abs(summary["overall_quality_mean"] - 0.7) < 1e-9
    assert "commitment" in summary["event_metrics"]
    commitment = summary["event_metrics"]["commitment"]
    assert commitment["tp"] == 6
    assert commitment["fp"] == 1
    assert commitment["fn"] == 2

"""Tests for structured qualitative co-funding metrics."""

from strong_models_experiment.analysis.qualitative_metrics import (
    compute_coalition_formation_metrics,
    compute_persuasion_effectiveness_metrics,
    compute_promise_keeping_metrics,
    compute_qualitative_metrics_v1,
    extract_pledge_history_from_logs,
)


def test_extract_pledge_history_from_logs():
    logs = [
        {
            "phase": "pledge_submission",
            "round": 1,
            "from": "Agent_A",
            "pledge": {"contributions": [5.0, 0.0]},
        },
        {
            "phase": "pledge_submission",
            "round": 1,
            "from": "Agent_B",
            "pledge": {"contributions": [0.0, 4.0]},
        },
        {
            "phase": "pledge_submission",
            "round": 2,
            "from": "Agent_A",
            "pledge": {"contributions": [6.0, 0.0]},
        },
    ]

    history = extract_pledge_history_from_logs(logs)
    assert len(history) == 2
    assert history[0]["Agent_A"] == [5.0, 0.0]
    assert history[0]["Agent_B"] == [0.0, 4.0]
    assert history[1]["Agent_A"] == [6.0, 0.0]


def test_promise_keeping_metrics():
    commitments = [
        {"round": 1, "speaker": "Agent_A", "project_idx": 0, "amount": 10.0},
        {"round": 1, "speaker": "Agent_B", "project_idx": 1, "amount": 7.0},
    ]
    history = [
        {"Agent_A": [0.0, 0.0], "Agent_B": [0.0, 0.0]},
        {"Agent_A": [10.5, 0.0], "Agent_B": [0.0, 6.5]},
    ]
    result = compute_promise_keeping_metrics(commitments, history, tolerance=1.0)
    assert result["total_commitments"] == 2
    assert result["kept_commitments"] == 2
    assert result["overall_keep_rate"] == 1.0


def test_persuasion_effectiveness_metrics():
    advocacy = [
        {"round": 1, "speaker": "Agent_A", "project_idx": 0},
    ]
    history = [
        {"Agent_A": [1.0, 0.0], "Agent_B": [2.0, 0.0]},
        {"Agent_A": [1.0, 0.0], "Agent_B": [5.0, 0.0]},
    ]
    result = compute_persuasion_effectiveness_metrics(advocacy, history)
    assert result["event_count"] == 1
    assert abs(result["overall_other_agent_delta"] - 3.0) < 1e-9


def test_coalition_formation_metrics():
    history = [
        {"Agent_A": [2.0, 0.0], "Agent_B": [2.0, 0.0]},
        {"Agent_A": [3.0, 0.0], "Agent_B": [1.0, 0.0]},
        {"Agent_A": [1.0, 0.0], "Agent_B": [2.0, 0.0]},
        {"Agent_A": [2.0, 0.0], "Agent_B": [2.0, 0.0]},
    ]
    result = compute_coalition_formation_metrics(history)
    assert result["max_consecutive_rounds_any_project"] >= 2
    assert result["persistent_project_count"] >= 1


def test_compute_qualitative_metrics_v1_end_to_end():
    logs = [
        {
            "phase": "discussion",
            "round": 1,
            "from": "Agent_A",
            "content": "I will contribute 5 to Market Street Protected Bike Lane.",
        },
        {
            "phase": "discussion",
            "round": 1,
            "from": "Agent_A",
            "content": "Let's focus on Market Street Protected Bike Lane first.",
        },
        {
            "phase": "pledge_submission",
            "round": 1,
            "from": "Agent_A",
            "pledge": {"contributions": [0.0, 0.0]},
        },
        {
            "phase": "pledge_submission",
            "round": 1,
            "from": "Agent_B",
            "pledge": {"contributions": [1.0, 0.0]},
        },
        {
            "phase": "pledge_submission",
            "round": 2,
            "from": "Agent_A",
            "pledge": {"contributions": [5.0, 0.0]},
        },
        {
            "phase": "pledge_submission",
            "round": 2,
            "from": "Agent_B",
            "pledge": {"contributions": [3.0, 0.0]},
        },
    ]
    game_state = {
        "projects": [
            {"name": "Market Street Protected Bike Lane"},
            {"name": "Parkside Adventure Playground"},
        ],
        "round_pledges": [
            {"Agent_A": [0.0, 0.0], "Agent_B": [1.0, 0.0]},
            {"Agent_A": [5.0, 0.0], "Agent_B": [3.0, 0.0]},
        ],
        "agent_budgets": {"Agent_A": 10.0, "Agent_B": 10.0},
    }
    metrics, events = compute_qualitative_metrics_v1(logs, game_state)
    assert metrics["schema_version"] == "qualitative_metrics_v1"
    assert metrics["event_count"] >= 1
    assert "promise_keeping" in metrics
    assert "persuasion_effectiveness" in metrics
    assert "coalition_formation" in metrics
    assert "adaptation_rate" in metrics
    assert isinstance(events, list)

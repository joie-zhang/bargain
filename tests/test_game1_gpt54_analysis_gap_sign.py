import json
from pathlib import Path

from scripts import analyze_game1_gpt54_team_coordination as analysis
from scripts import analyze_game1_team_ceiling_normalization as ceiling_analysis


def test_primary_gap_is_adversary_minus_average_nano() -> None:
    metrics = analysis.utility_metrics(
        {
            "consensus_reached": True,
            "final_round": 1,
            "final_utilities": {
                "Agent_1": 80.0,
                "Agent_2": 40.0,
                "Agent_3": 60.0,
            },
        },
        {
            "Agent_1": "adversary",
            "Agent_2": "baseline",
            "Agent_3": "baseline",
        },
    )

    assert metrics["baseline_mean_payoff"] == 50.0
    assert metrics["adversary_minus_nano_gap"] == 30.0
    assert metrics["per_agent_gap"] == -30.0


def test_ceiling_normalized_gap_is_adversary_minus_average_nano(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps({
            "consensus_reached": True,
            "final_round": 1,
            "final_allocation": {"Agent_1": [0], "Agent_2": [1]},
            "final_utilities": {"Agent_1": 80.0, "Agent_2": 70.0},
        }),
        encoding="utf-8",
    )

    metrics, _ = ceiling_analysis.result_metrics(
        result_path,
        {"Agent_1": [80.0, 20.0], "Agent_2": [30.0, 70.0]},
        {"Agent_1": "adversary", "Agent_2": "baseline"},
        gamma=0.9,
        welfare_ceiling=150.0,
        per_capita_ceiling=75.0,
    )

    assert metrics["raw_adversary_minus_nano_gap"] == 10.0
    assert metrics["raw_adversary_minus_nano_ceiling_normalized_gap"] == 10.0 / 75.0
    assert metrics["raw_ceiling_normalized_gap"] == -10.0 / 75.0

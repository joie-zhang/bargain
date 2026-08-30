import json
from pathlib import Path

import pytest

from scripts.ttc_accounting import account_token_usage, resolve_final_interactions


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def interaction(experiment_id: str | None) -> dict[str, object]:
    return {"experiment_id": experiment_id, "phase": "discussion_round_1"}


def test_resolver_selects_timestamped_final_attempt(tmp_path: Path) -> None:
    result = tmp_path / "run_1_experiment_results.json"
    canonical = tmp_path / "run_1_all_interactions.json"
    final_attempt = tmp_path / "run_1_all_interactions_20260503_004335.json"
    write_json(result, {"experiment_id": "final"})
    write_json(canonical, [interaction("stale")])
    write_json(final_attempt, [interaction("final"), interaction("final")])

    selected, records = resolve_final_interactions(result)

    assert selected == final_attempt
    assert len(records) == 2


@pytest.mark.parametrize(
    "candidate_payloads",
    [
        [],
        [[interaction(None)]],
        [[interaction("final"), interaction("stale")]],
        [[interaction("stale")]],
        [[interaction("final")], [interaction("final")]],
    ],
)
def test_resolver_fails_closed_without_one_wholly_matching_log(
    tmp_path: Path, candidate_payloads: list[list[dict[str, object]]]
) -> None:
    result = tmp_path / "run_1_experiment_results.json"
    write_json(result, {"experiment_id": "final"})
    for index, payload in enumerate(candidate_payloads):
        write_json(tmp_path / f"run_1_all_interactions_{index}.json", payload)

    with pytest.raises(RuntimeError, match="Expected exactly one interaction log"):
        resolve_final_interactions(result)


def test_stored_total_is_authoritative_across_provider_semantics() -> None:
    accounted = account_token_usage(
        {
            "input_tokens": 100,
            "output_tokens": 25,
            "reasoning_tokens": 10,
            "total_tokens": 135,
            "output_tokens_includes_reasoning": False,
        }
    )

    assert accounted["total_tokens"] == 135
    assert accounted["token_total_source"] == "stored_total_tokens"
    assert accounted["stored_total_equals_input_plus_output"] is False


@pytest.mark.parametrize(
    ("includes_reasoning", "expected_total", "expected_source"),
    [
        (True, 125, "derived_input_plus_inclusive_output"),
        (False, 135, "derived_input_plus_visible_output_plus_reasoning"),
    ],
)
def test_fallback_requires_and_honors_explicit_output_semantics(
    includes_reasoning: bool, expected_total: int, expected_source: str
) -> None:
    accounted = account_token_usage(
        {
            "input_tokens": 100,
            "output_tokens": 25,
            "reasoning_tokens": 10,
            "output_tokens_includes_reasoning": includes_reasoning,
        }
    )

    assert accounted["total_tokens"] == expected_total
    assert accounted["token_total_source"] == expected_source


def test_unknown_fallback_semantics_and_conflicting_aliases_fail() -> None:
    with pytest.raises(ValueError, match="Cannot derive"):
        account_token_usage({"input_tokens": 100, "output_tokens": 25})
    with pytest.raises(ValueError, match="Conflicting"):
        account_token_usage(
            {
                "input_tokens": 100,
                "output_tokens": 25,
                "total_tokens": 125,
                "reasoning_tokens": 10,
                "thinking_tokens": 11,
            }
        )


def test_seed42_released_cohort_accounting() -> None:
    project_root = Path(__file__).resolve().parents[1]
    ttc_root = (
        project_root
        / "experiments/results/ttc_native_scaling_20260502_212943"
    )
    config_root = ttc_root / "configs"
    if not config_root.is_dir():
        pytest.skip("Released seed-42 TTC data are not installed")

    resolved = 0
    noncanonical = 0
    target_records = 0
    usage_events = 0
    stored_events = 0
    identity_events = 0
    total_tokens = 0
    reasoning_tokens = 0
    for config_path in sorted(config_root.glob("config_*.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        run_dir = project_root / config["output_dir"]
        selected, records = resolve_final_interactions(
            run_dir / "run_1_experiment_results.json"
        )
        resolved += 1
        noncanonical += selected.name != "run_1_all_interactions.json"
        target_agent = (
            "Agent_1" if int(config.get("target_position", 0)) == 0 else "Agent_2"
        )
        for record in records:
            if record.get("agent_id") != target_agent:
                continue
            target_records += 1
            usage = record.get("token_usage") or {}
            if not usage:
                continue
            accounted = account_token_usage(usage)
            usage_events += 1
            stored_events += accounted["token_total_source"] == "stored_total_tokens"
            identity_events += accounted["stored_total_equals_input_plus_output"] is True
            total_tokens += int(accounted["total_tokens"])
            reasoning_tokens += int(accounted["reasoning_tokens"])

    assert (resolved, noncanonical) == (216, 13)
    assert (target_records, usage_events) == (2737, 2542)
    assert (stored_events, identity_events) == (2542, 2542)
    assert total_tokens == 23_399_855
    assert reasoning_tokens == 707_776

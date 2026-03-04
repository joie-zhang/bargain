"""Tests for qualitative schema validation."""

from strong_models_experiment.analysis.qualitative_schema import (
    validate_qualitative_event,
    validate_qualitative_metrics_v1,
)


def test_validate_qualitative_event_ok():
    event = {
        "event_type": "commitment",
        "round": 2,
        "speaker": "Agent_A",
        "project_idx": 1,
        "amount": 5.0,
    }
    assert validate_qualitative_event(event) == []


def test_validate_qualitative_event_rejects_bad_payload():
    errors = validate_qualitative_event({"round": 0, "speaker": ""})
    assert errors


def test_validate_qualitative_metrics_ok():
    payload = {
        "schema_version": "qualitative_metrics_v1",
        "event_count": 1,
        "promise_keeping": {
            "total_commitments": 1,
            "kept_commitments": 1,
            "overall_keep_rate": 1.0,
        },
        "persuasion_effectiveness": {"event_count": 1},
        "coalition_formation": {
            "coalition_active_round_fraction": 0.5,
            "persistent_project_fraction": 0.5,
        },
        "adaptation_rate": {"Agent_A": 0.1},
    }
    assert validate_qualitative_metrics_v1(payload) == []


def test_validate_qualitative_metrics_rejects_legacy_shape():
    payload = {
        "schema_version": "legacy",
        "event_count": -1,
        "promise_keeping": {},
        "persuasion_effectiveness": {},
        "coalition_formation": {},
        "adaptation_rate": [],
    }
    errors = validate_qualitative_metrics_v1(payload)
    assert errors

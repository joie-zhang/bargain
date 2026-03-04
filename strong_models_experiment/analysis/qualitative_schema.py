"""Validation helpers for qualitative metrics and extracted events."""

from __future__ import annotations

from typing import Any, Dict, List


def validate_qualitative_event(event: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not isinstance(event, dict):
        return ["event must be a dict"]

    event_type = event.get("event_type")
    if not isinstance(event_type, str) or not event_type:
        errors.append("event_type must be a non-empty string")

    round_num = event.get("round")
    if not isinstance(round_num, int) or round_num < 1:
        errors.append("round must be an integer >= 1")

    speaker = event.get("speaker")
    if not isinstance(speaker, str) or not speaker:
        errors.append("speaker must be a non-empty string")

    project_idx = event.get("project_idx")
    if project_idx is not None and (not isinstance(project_idx, int) or project_idx < 0):
        errors.append("project_idx must be a non-negative integer when provided")

    amount = event.get("amount")
    if amount is not None and not isinstance(amount, (int, float)):
        errors.append("amount must be numeric when provided")

    return errors


def validate_qualitative_metrics_v1(metrics: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not isinstance(metrics, dict):
        return ["metrics must be a dict"]

    if metrics.get("schema_version") != "qualitative_metrics_v1":
        errors.append("schema_version must be 'qualitative_metrics_v1'")

    if not isinstance(metrics.get("event_count"), int) or metrics.get("event_count") < 0:
        errors.append("event_count must be a non-negative integer")

    promise = metrics.get("promise_keeping")
    if not isinstance(promise, dict):
        errors.append("promise_keeping must be a dict")
    else:
        for key in ("total_commitments", "kept_commitments"):
            if key not in promise or not isinstance(promise[key], int) or promise[key] < 0:
                errors.append(f"promise_keeping.{key} must be a non-negative integer")
        overall_keep_rate = promise.get("overall_keep_rate")
        if overall_keep_rate is not None:
            if not isinstance(overall_keep_rate, (int, float)) or not (0.0 <= float(overall_keep_rate) <= 1.0):
                errors.append("promise_keeping.overall_keep_rate must be in [0,1] or None")

    persuasion = metrics.get("persuasion_effectiveness")
    if not isinstance(persuasion, dict):
        errors.append("persuasion_effectiveness must be a dict")
    else:
        if "event_count" not in persuasion or not isinstance(persuasion["event_count"], int) or persuasion["event_count"] < 0:
            errors.append("persuasion_effectiveness.event_count must be a non-negative integer")

    coalition = metrics.get("coalition_formation")
    if not isinstance(coalition, dict):
        errors.append("coalition_formation must be a dict")
    else:
        for key in ("coalition_active_round_fraction", "persistent_project_fraction"):
            value = coalition.get(key)
            if value is not None and (not isinstance(value, (int, float)) or not (0.0 <= float(value) <= 1.0)):
                errors.append(f"coalition_formation.{key} must be in [0,1] or None")

    adaptation = metrics.get("adaptation_rate")
    if not isinstance(adaptation, dict):
        errors.append("adaptation_rate must be a dict")
    else:
        for agent_id, value in adaptation.items():
            if not isinstance(agent_id, str) or not agent_id:
                errors.append("adaptation_rate keys must be non-empty agent IDs")
            if not isinstance(value, (int, float)):
                errors.append(f"adaptation_rate[{agent_id}] must be numeric")
            elif float(value) < 0:
                errors.append(f"adaptation_rate[{agent_id}] must be >= 0")

    return errors

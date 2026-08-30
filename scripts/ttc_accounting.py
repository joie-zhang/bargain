#!/usr/bin/env python3
"""Fail-closed TTC attempt resolution and token accounting helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_final_interactions(
    result_path: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    """Return the unique interaction log whose records match the final result ID."""
    result = _load_json(result_path)
    if not isinstance(result, dict):
        raise RuntimeError(f"Final result is not a JSON object: {result_path}")
    raw_expected_id = result.get("experiment_id")
    expected_id = "" if raw_expected_id is None else str(raw_expected_id)
    if not expected_id:
        raise RuntimeError(f"Final result has no experiment_id: {result_path}")

    suffix = "_experiment_results.json"
    if not result_path.name.endswith(suffix):
        raise RuntimeError(f"Unsupported result filename: {result_path}")
    run_prefix = result_path.name[: -len(suffix)]
    candidates = sorted(result_path.parent.glob(f"{run_prefix}_all_interactions*.json"))
    matches: list[tuple[Path, list[dict[str, Any]]]] = []
    diagnostics: dict[str, list[str]] = {}
    for candidate in candidates:
        payload = _load_json(candidate)
        if not isinstance(payload, list) or not payload:
            diagnostics[candidate.name] = ["<empty-or-non-list>"]
            continue
        observed_ids: list[str] = []
        valid_records = True
        for record in payload:
            if not isinstance(record, dict):
                observed_ids.append("<non-object>")
                valid_records = False
                continue
            raw_observed_id = record.get("experiment_id")
            observed_id = "" if raw_observed_id is None else str(raw_observed_id)
            observed_ids.append(observed_id or "<missing>")
            if observed_id != expected_id:
                valid_records = False
        diagnostics[candidate.name] = sorted(set(observed_ids))
        if valid_records:
            matches.append((candidate, payload))

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one interaction log for experiment_id={expected_id}; "
            f"found {len(matches)} among {len(candidates)} candidates; "
            f"observed={diagnostics}"
        )
    return matches[0]


def _token_value(usage: dict[str, Any], key: str, *, required: bool) -> float | None:
    value = usage.get(key)
    if value is None:
        if required:
            raise ValueError(f"Missing required token field {key}: {usage}")
        return None
    if isinstance(value, bool):
        raise ValueError(f"Invalid boolean token field {key}: {usage}")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"Invalid token field {key}: {usage}")
    return numeric


def account_token_usage(usage: dict[str, Any]) -> dict[str, Any]:
    """Use a stored total when present; derive only from explicit output semantics."""
    input_tokens = _token_value(usage, "input_tokens", required=True)
    output_tokens = _token_value(usage, "output_tokens", required=True)
    assert input_tokens is not None and output_tokens is not None

    components = [
        value
        for key in ("reasoning_tokens", "thinking_tokens")
        if (value := _token_value(usage, key, required=False)) is not None
    ]
    if len(components) == 2 and not math.isclose(components[0], components[1]):
        raise ValueError(f"Conflicting reasoning/thinking token aliases: {usage}")
    reasoning_tokens = components[0] if components else 0.0

    stored_total = _token_value(usage, "total_tokens", required=False)
    if stored_total is not None:
        total_tokens = stored_total
        token_total_source = "stored_total_tokens"
        stored_identity: bool | None = math.isclose(
            stored_total, input_tokens + output_tokens
        )
    else:
        stored_identity = None
        includes_reasoning = usage.get("output_tokens_includes_reasoning")
        if includes_reasoning is True:
            total_tokens = input_tokens + output_tokens
            token_total_source = "derived_input_plus_inclusive_output"
        elif includes_reasoning is False:
            total_tokens = input_tokens + output_tokens + reasoning_tokens
            token_total_source = "derived_input_plus_visible_output_plus_reasoning"
        else:
            raise ValueError(
                "Cannot derive total_tokens without a stored total or explicit "
                f"output_tokens_includes_reasoning semantics: {usage}"
            )

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
        "token_total_source": token_total_source,
        "stored_total_equals_input_plus_output": stored_identity,
    }

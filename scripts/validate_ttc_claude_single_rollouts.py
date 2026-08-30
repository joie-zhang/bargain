#!/usr/bin/env python3
"""Validate and aggregate one-file-per-rollout Codex TTC annotations."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import validate_ttc_claude_codex_adjudication as base


ROOT = base.ROOT
OUTPUTS = ROOT / "single_rollout_outputs"
WRAPPER_FIELDS = {
    "rollout_id",
    "seed",
    "source_config_id",
    "analyzed",
    "events",
    "audit",
}
AUDIT_FIELDS = {"ambiguous", "unsupported", "new_tag_ideas"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--write-aggregate", action="store_true")
    parser.add_argument(
        "--strict-saved-schema",
        action="store_true",
        help="Reject legacy wrapper/chunk fields instead of normalizing them in memory",
    )
    args = parser.parse_args()

    chunks, manifests, views, titles = base.load_context()
    rollout_to_chunk = titles["__rollout_to_chunk__"]
    errors: list[str] = []
    warnings: list[str] = []
    events: list[dict[str, Any]] = []
    completed: set[str] = set()

    for path in sorted(OUTPUTS.glob("seed_*_config_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: invalid JSON: {exc}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"{path}: expected JSON object")
            continue
        wrapper_fields = set(payload)
        if args.strict_saved_schema and wrapper_fields != WRAPPER_FIELDS:
            missing = sorted(WRAPPER_FIELDS - wrapper_fields)
            extra = sorted(wrapper_fields - WRAPPER_FIELDS)
            errors.append(
                f"{path}: wrapper fields differ from schema; "
                f"missing={missing}, extra={extra}"
            )
        rollout_id = payload.get("rollout_id")
        if rollout_id not in manifests:
            errors.append(f"{path}: unknown rollout_id {rollout_id!r}")
            continue
        if path.stem != rollout_id:
            errors.append(f"{path}: filename does not match rollout_id")
        manifest = manifests[rollout_id]
        if payload.get("seed") != manifest["seed"]:
            errors.append(f"{path}: incorrect seed")
        if payload.get("source_config_id") != manifest["source_config_id"]:
            errors.append(f"{path}: incorrect source_config_id")
        if payload.get("analyzed") is not True:
            errors.append(f"{path}: analyzed must be true")
        audit = payload.get("audit")
        if not isinstance(audit, dict):
            errors.append(f"{path}: audit must be an object")
        else:
            audit_fields = set(audit)
            if audit_fields != AUDIT_FIELDS:
                missing = sorted(AUDIT_FIELDS - audit_fields)
                extra = sorted(audit_fields - AUDIT_FIELDS)
                errors.append(
                    f"{path}: audit fields differ from schema; "
                    f"missing={missing}, extra={extra}"
                )
            for key in ("ambiguous", "unsupported", "new_tag_ideas"):
                if not isinstance(audit.get(key), list):
                    errors.append(f"{path}: audit.{key} must be an array")
        rows = payload.get("events")
        if not isinstance(rows, list):
            errors.append(f"{path}: events must be an array")
            continue
        completed.add(rollout_id)
        expected_chunk = rollout_to_chunk[rollout_id]
        for line_number, row in enumerate(rows, 1):
            if not isinstance(row, dict):
                errors.append(f"{path}:events[{line_number}]: expected object")
                continue
            event_fields = set(row)
            required_event_fields = set(base.REQUIRED_FIELDS)
            if args.strict_saved_schema and event_fields != required_event_fields:
                missing = sorted(required_event_fields - event_fields)
                extra = sorted(event_fields - required_event_fields)
                errors.append(
                    f"{path}:events[{line_number}]: event fields differ from schema; "
                    f"missing={missing}, extra={extra}"
                )
            row["_source_file"] = str(path)
            row["_source_line"] = line_number
            if not args.strict_saved_schema:
                # Legacy workers used rollout-local chunk labels or omitted the
                # field. The canonical value is deterministic from the manifest.
                row["chunk_id"] = expected_chunk
            row_errors, row_warnings = base.validate_event(
                row, expected_chunk, manifests, views, titles
            )
            errors.extend(row_errors)
            warnings.extend(row_warnings)
            events.append(row)

    duplicate_keys = Counter(
        (
            row.get("rollout_id"),
            row.get("tag_code"),
            row.get("source_kind"),
            row.get("log_index"),
            row.get("interaction_index"),
            row.get("speaker_agent"),
        )
        for row in events
    )
    duplicate_count = sum(count > 1 for count in duplicate_keys.values())
    if duplicate_count:
        warnings.append(f"{duplicate_count} duplicate event identity keys")
    if args.require_all:
        missing = sorted(set(manifests) - completed)
        if missing:
            errors.append(f"missing {len(missing)} rollout outputs")
    if args.write_aggregate and not errors:
        base.aggregate(events, titles)

    report = {
        "source_rollouts": len(manifests),
        "completed_rollouts": len(completed),
        "event_rows": len(events),
        "errors": errors,
        "warnings": warnings,
    }
    (ROOT / "single_rollout_validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"source_rollouts={len(manifests)} completed={len(completed)} "
        f"events={len(events)} errors={len(errors)} warnings={len(warnings)}"
    )
    for error in errors[:20]:
        print(f"ERROR {error}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

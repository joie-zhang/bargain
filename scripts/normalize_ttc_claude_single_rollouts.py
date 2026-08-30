#!/usr/bin/env python3
"""Canonicalize deterministic metadata and duplicate event identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import validate_ttc_claude_codex_adjudication as base


ROOT = base.ROOT
OUTPUTS = ROOT / "single_rollout_outputs"
WRAPPER_FIELDS = (
    "rollout_id",
    "seed",
    "source_config_id",
    "analyzed",
    "events",
    "audit",
)
AUDIT_FIELDS = ("ambiguous", "unsupported", "new_tag_ideas")
DUPLICATE_FIELDS = (
    "rollout_id",
    "tag_code",
    "source_kind",
    "log_index",
    "interaction_index",
    "speaker_agent",
)
CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def preference(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        CONFIDENCE_RANK.get(row.get("confidence"), -1),
        len(str(row.get("quote", ""))),
        len(str(row.get("rationale", ""))),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    _, manifests, _, titles = base.load_context()
    rollout_to_chunk = titles["__rollout_to_chunk__"]
    files = changed = normalized_rows = duplicates_removed = 0

    for path in sorted(OUTPUTS.glob("seed_*_config_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        rollout_id = payload.get("rollout_id")
        rows = payload.get("events")
        audit = payload.get("audit")
        if (
            rollout_id not in manifests
            or not isinstance(rows, list)
            or not isinstance(audit, dict)
            or not all(field in payload for field in WRAPPER_FIELDS)
            or not all(field in audit for field in AUDIT_FIELDS)
        ):
            continue
        files += 1
        expected_chunk = rollout_to_chunk[rollout_id]
        chosen: dict[tuple[Any, ...], dict[str, Any]] = {}
        positions: list[tuple[Any, ...]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            canonical = dict(row)
            if canonical.get("chunk_id") != expected_chunk:
                normalized_rows += 1
            canonical["chunk_id"] = expected_chunk
            key = tuple(canonical.get(field) for field in DUPLICATE_FIELDS)
            if key not in chosen:
                chosen[key] = canonical
                positions.append(key)
            else:
                duplicates_removed += 1
                if preference(canonical) > preference(chosen[key]):
                    chosen[key] = canonical
        canonical_rows = [chosen[key] for key in positions]
        canonical_audit = {field: audit[field] for field in AUDIT_FIELDS}
        canonical_payload = {
            "rollout_id": payload["rollout_id"],
            "seed": payload["seed"],
            "source_config_id": payload["source_config_id"],
            "analyzed": payload["analyzed"],
            "events": canonical_rows,
            "audit": canonical_audit,
        }
        if canonical_payload != payload:
            changed += 1
            if args.write:
                path.write_text(
                    json.dumps(canonical_payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )

    mode = "wrote" if args.write else "would_change"
    print(
        f"files={files} {mode}={changed} normalized_chunk_rows={normalized_rows} "
        f"duplicates_removed={duplicates_removed}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate and aggregate TTC LLM strategic tag adjudication outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "analysis/ttc_llm_strategic_tag_adjudication_20260629"
CHUNK_INDEX = OUT_DIR / "chunk_index.jsonl"
CODEBOOK = OUT_DIR / "llm_tag_codebook.json"
OUTPUT_DIR = OUT_DIR / "subagent_outputs"

EVENT_TYPES = {
    "utterance",
    "private_thinking",
    "proposal_reasoning",
    "vote_reasoning",
    "reflection",
    "formal_outcome",
}
SOURCE_KINDS = {"conversation_log", "interaction", "formal_outcome"}
PHASES = {"discussion", "private_thinking", "proposal", "voting", "reflection", "final_outcome"}
CONFIDENCES = {"high", "medium", "low"}
REQUIRED_FIELDS = [
    "chunk_id",
    "config_id",
    "result_path",
    "interactions_path",
    "rollout_view_path",
    "family",
    "level",
    "level_index",
    "provider",
    "game_label",
    "game_cell",
    "game_type",
    "n_agents",
    "order",
    "target_agent",
    "baseline_agent",
    "speaker_agent",
    "speaker_model",
    "speaker_elo",
    "speaker_role",
    "speaker_is_target",
    "speaker_is_baseline",
    "tag_code",
    "tag_title",
    "evidence_type",
    "source_kind",
    "phase",
    "round",
    "discussion_turn",
    "log_index",
    "interaction_index",
    "speaker_order",
    "total_speakers",
    "quote",
    "rationale",
    "confidence",
    "negation_checked",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            value["_source_file"] = str(path)
            value["_source_line"] = line_number
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            clean = {key: value for key, value in row.items() if not key.startswith("_")}
            handle.write(json.dumps(clean, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def load_context() -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, dict[str, Any]]]:
    chunks = {row["chunk_id"]: row for row in read_jsonl(CHUNK_INDEX)}
    tag_titles = {row["tag_code"]: row["tag_title"] for row in json.loads(CODEBOOK.read_text())}
    rollout_manifest = {}
    for chunk_id, chunk in chunks.items():
        for row in read_jsonl(Path(chunk["manifest_path"])):
            row["_chunk_id"] = chunk_id
            rollout_manifest[str(row["config_id"])] = row
    return chunks, tag_titles, rollout_manifest


def loc(row: dict[str, Any]) -> str:
    return f"{row.get('_source_file')}:{row.get('_source_line')}"


def validate_row(
    row: dict[str, Any],
    *,
    expected_chunk_id: str,
    tag_titles: dict[str, str],
    rollout_manifest: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    where = loc(row)

    missing = [field for field in REQUIRED_FIELDS if field not in row]
    if missing:
        errors.append(f"{where}: missing fields: {', '.join(missing)}")

    if row.get("chunk_id") != expected_chunk_id:
        errors.append(f"{where}: chunk_id {row.get('chunk_id')!r} != file chunk {expected_chunk_id!r}")

    config_id = row.get("config_id")
    manifest = rollout_manifest.get(str(config_id))
    if not manifest:
        errors.append(f"{where}: config_id is not in manifest: {config_id!r}")
    elif manifest.get("_chunk_id") != expected_chunk_id:
        errors.append(
            f"{where}: config_id {config_id!r} belongs to {manifest.get('_chunk_id')}, not {expected_chunk_id}"
        )
    else:
        for field in [
            "result_path",
            "interactions_path",
            "rollout_view_path",
            "family",
            "level",
            "level_index",
            "provider",
            "game_label",
            "game_cell",
            "game_type",
            "n_agents",
            "order",
            "target_agent",
            "baseline_agent",
        ]:
            if str(row.get(field)) != str(manifest.get(field)):
                errors.append(
                    f"{where}: {field} {row.get(field)!r} does not match manifest {manifest.get(field)!r}"
                )

    tag_code = row.get("tag_code")
    if tag_code not in tag_titles:
        errors.append(f"{where}: unknown tag_code {tag_code!r}")
    elif row.get("tag_title") != tag_titles[tag_code]:
        warnings.append(
            f"{where}: tag_title {row.get('tag_title')!r} does not match {tag_titles[tag_code]!r}"
        )

    if row.get("evidence_type") not in EVENT_TYPES:
        errors.append(f"{where}: invalid evidence_type {row.get('evidence_type')!r}")
    if row.get("source_kind") not in SOURCE_KINDS:
        errors.append(f"{where}: invalid source_kind {row.get('source_kind')!r}")
    if row.get("phase") not in PHASES:
        errors.append(f"{where}: invalid phase {row.get('phase')!r}")
    if row.get("confidence") not in CONFIDENCES:
        errors.append(f"{where}: invalid confidence {row.get('confidence')!r}")
    if not isinstance(row.get("negation_checked"), bool):
        errors.append(f"{where}: negation_checked must be boolean")

    for field in ["speaker_is_target", "speaker_is_baseline"]:
        if not isinstance(row.get(field), bool):
            errors.append(f"{where}: {field} must be boolean")

    for field in ["quote", "rationale"]:
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{where}: {field} must be non-empty text")

    for field in [
        "config_id",
        "level_index",
        "n_agents",
        "round",
        "discussion_turn",
        "log_index",
        "interaction_index",
        "speaker_order",
        "total_speakers",
    ]:
        value = row.get(field)
        if value is not None and not isinstance(value, int):
            errors.append(f"{where}: {field} must be integer or null")

    speaker = row.get("speaker_agent")
    if manifest and speaker:
        agent_model_map = manifest.get("agent_model_map") or {}
        agent_elo_map = manifest.get("agent_elo_map") or {}
        agent_role_map = manifest.get("agent_role_map") or {}
        if speaker not in agent_role_map:
            warnings.append(f"{where}: speaker_agent {speaker!r} not in agent_role_map")
        else:
            if row.get("speaker_model") != agent_model_map.get(speaker):
                warnings.append(
                    f"{where}: speaker_model {row.get('speaker_model')!r} does not match "
                    f"{agent_model_map.get(speaker)!r}"
                )
            if row.get("speaker_elo") != agent_elo_map.get(speaker):
                warnings.append(
                    f"{where}: speaker_elo {row.get('speaker_elo')!r} does not match "
                    f"{agent_elo_map.get(speaker)!r}"
                )
            if row.get("speaker_role") != agent_role_map.get(speaker):
                warnings.append(
                    f"{where}: speaker_role {row.get('speaker_role')!r} does not match "
                    f"{agent_role_map.get(speaker)!r}"
                )
            if row.get("speaker_is_target") != (speaker == manifest.get("target_agent")):
                errors.append(f"{where}: speaker_is_target inconsistent with target_agent")
            if row.get("speaker_is_baseline") != (speaker == manifest.get("baseline_agent")):
                errors.append(f"{where}: speaker_is_baseline inconsistent with baseline_agent")

    return errors, warnings


def confidence_rank(value: str) -> int:
    return {"low": 1, "medium": 2, "high": 3}.get(value, 0)


def aggregate(events: list[dict[str, Any]], tag_titles: dict[str, str]) -> None:
    write_jsonl(OUT_DIR / "ttc_llm_event_tags.jsonl", events)

    tag_counts = Counter(row["tag_code"] for row in events)
    write_csv(
        OUT_DIR / "ttc_llm_event_tag_counts_by_tag.csv",
        [
            {
                "tag_code": tag_code,
                "tag_title": tag_titles[tag_code],
                "event_count": tag_counts.get(tag_code, 0),
            }
            for tag_code in sorted(tag_titles)
        ],
        ["tag_code", "tag_title", "event_count"],
    )

    grouped: dict[tuple[int, str], dict[str, Any]] = {}
    for row in events:
        key = (row["config_id"], row["tag_code"])
        current = grouped.setdefault(
            key,
            {
                "config_id": row["config_id"],
                "result_path": row["result_path"],
                "family": row["family"],
                "level": row["level"],
                "level_index": row["level_index"],
                "game_label": row["game_label"],
                "game_cell": row["game_cell"],
                "order": row["order"],
                "tag_code": row["tag_code"],
                "tag_title": row["tag_title"],
                "event_count": 0,
                "target_event_count": 0,
                "baseline_event_count": 0,
                "max_confidence": row["confidence"],
            },
        )
        current["event_count"] += 1
        if row.get("speaker_is_target"):
            current["target_event_count"] += 1
        if row.get("speaker_is_baseline"):
            current["baseline_event_count"] += 1
        if confidence_rank(row["confidence"]) > confidence_rank(current["max_confidence"]):
            current["max_confidence"] = row["confidence"]
    write_csv(
        OUT_DIR / "ttc_llm_rollout_tag_summary.csv",
        sorted(grouped.values(), key=lambda r: (r["config_id"], r["tag_code"])),
        [
            "config_id",
            "result_path",
            "family",
            "level",
            "level_index",
            "game_label",
            "game_cell",
            "order",
            "tag_code",
            "tag_title",
            "event_count",
            "target_event_count",
            "baseline_event_count",
            "max_confidence",
        ],
    )


def write_report(
    event_files: list[Path],
    completed_chunks: set[str],
    events: list[dict[str, Any]],
    errors: list[str],
    warnings: list[str],
) -> None:
    tag_counts = Counter(row.get("tag_code") for row in events if row.get("tag_code"))
    lines = [
        "# TTC LLM Strategic Tag Adjudication Validation",
        "",
        f"- Event files: {len(event_files)}",
        f"- Completed chunks with event files: {len(completed_chunks)}",
        f"- Event rows: {len(events)}",
        f"- Errors: {len(errors)}",
        f"- Warnings: {len(warnings)}",
        "",
        "## Top Tag Counts",
        "",
    ]
    for tag_code, count in tag_counts.most_common(30):
        lines.append(f"- `{tag_code}`: {count}")
    if errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in errors[:200])
        if len(errors) > 200:
            lines.append(f"- ... {len(errors) - 200} more")
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings[:200])
        if len(warnings) > 200:
            lines.append(f"- ... {len(warnings) - 200} more")
    (OUT_DIR / "ttc_llm_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-aggregate", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    _, tag_titles, rollout_manifest = load_context()
    event_files = sorted(OUTPUT_DIR.glob("chunk_*_events.jsonl"))
    events: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    completed_chunks: set[str] = set()

    for path in event_files:
        match = re.fullmatch(r"(chunk_\d{4})_events\.jsonl", path.name)
        if not match:
            errors.append(f"{path}: unexpected event filename")
            continue
        chunk_id = match.group(1)
        completed_chunks.add(chunk_id)
        try:
            file_events = read_jsonl(path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        for row in file_events:
            row_errors, row_warnings = validate_row(
                row,
                expected_chunk_id=chunk_id,
                tag_titles=tag_titles,
                rollout_manifest=rollout_manifest,
            )
            errors.extend(row_errors)
            warnings.extend(row_warnings)
        events.extend(file_events)

    write_report(event_files, completed_chunks, events, errors, warnings)
    if args.write_aggregate:
        aggregate(events, tag_titles)

    print(f"event_files={len(event_files)} completed_chunks={len(completed_chunks)} events={len(events)}")
    print(f"errors={len(errors)} warnings={len(warnings)}")
    if errors:
        print("first_errors:")
        for error in errors[:10]:
            print(f"- {error}")
    if warnings:
        print("first_warnings:")
        for warning in warnings[:10]:
            print(f"- {warning}")
    if args.strict and errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

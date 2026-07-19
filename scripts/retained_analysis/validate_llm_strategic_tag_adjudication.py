#!/usr/bin/env python3
"""Validate and aggregate LLM strategic tag adjudication chunk outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "analysis/llm_strategic_tag_adjudication_20260628"
CHUNK_INDEX = OUT_DIR / "chunk_index.jsonl"
CODEBOOK = OUT_DIR / "llm_tag_codebook.json"
OUTPUT_DIR = OUT_DIR / "subagent_outputs"

EVENT_TYPES = {"utterance", "proposal_reasoning", "formal_outcome"}
PHASES = {"discussion", "proposal", "final_outcome"}
CONFIDENCES = {"high", "medium", "low"}
REQUIRED_FIELDS = [
    "chunk_id",
    "result_path",
    "config_id",
    "experiment_family",
    "game_label",
    "n_agents",
    "tag_code",
    "tag_title",
    "evidence_type",
    "phase",
    "round",
    "discussion_turn",
    "log_index",
    "speaker_agent",
    "speaker_model",
    "speaker_elo",
    "speaker_role",
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
    tag_titles = {row["tag_code"]: row["tag_title"] for row in json.loads(CODEBOOK.read_text(encoding="utf-8"))}
    rollout_manifest: dict[str, dict[str, Any]] = {}
    for chunk_id, chunk in chunks.items():
        for row in read_jsonl(Path(chunk["manifest_path"])):
            row["_chunk_id"] = chunk_id
            rollout_manifest[row["result_path"]] = row
    return chunks, tag_titles, rollout_manifest


def format_location(row: dict[str, Any]) -> str:
    return f"{row.get('_source_file')}:{row.get('_source_line')}"


def validate_event(
    row: dict[str, Any],
    *,
    expected_chunk_id: str,
    tag_titles: dict[str, str],
    rollout_manifest: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    loc = format_location(row)

    missing = [field for field in REQUIRED_FIELDS if field not in row]
    if missing:
        errors.append(f"{loc}: missing fields: {', '.join(missing)}")

    if row.get("chunk_id") != expected_chunk_id:
        errors.append(f"{loc}: chunk_id {row.get('chunk_id')!r} != file chunk {expected_chunk_id!r}")

    result_path = row.get("result_path")
    manifest = rollout_manifest.get(result_path)
    if not manifest:
        errors.append(f"{loc}: result_path is not in any manifest: {result_path!r}")
    elif manifest.get("_chunk_id") != expected_chunk_id:
        errors.append(
            f"{loc}: result_path belongs to {manifest.get('_chunk_id')}, not {expected_chunk_id}"
        )
    else:
        for field in ["config_id", "experiment_family", "game_label", "n_agents"]:
            if str(row.get(field)) != str(manifest.get(field)):
                errors.append(
                    f"{loc}: {field} {row.get(field)!r} does not match manifest {manifest.get(field)!r}"
                )

    tag_code = row.get("tag_code")
    if tag_code not in tag_titles:
        errors.append(f"{loc}: unknown tag_code {tag_code!r}")
    elif row.get("tag_title") != tag_titles[tag_code]:
        warnings.append(
            f"{loc}: tag_title {row.get('tag_title')!r} does not match codebook {tag_titles[tag_code]!r}"
        )

    if row.get("evidence_type") not in EVENT_TYPES:
        errors.append(f"{loc}: invalid evidence_type {row.get('evidence_type')!r}")
    if row.get("phase") not in PHASES:
        errors.append(f"{loc}: invalid phase {row.get('phase')!r}")
    if row.get("confidence") not in CONFIDENCES:
        errors.append(f"{loc}: invalid confidence {row.get('confidence')!r}")
    if not isinstance(row.get("negation_checked"), bool):
        errors.append(f"{loc}: negation_checked must be boolean")

    for field in ["quote", "rationale"]:
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{loc}: {field} must be non-empty text")

    for field in ["round", "discussion_turn", "log_index", "speaker_order", "total_speakers"]:
        value = row.get(field)
        if value is not None and not isinstance(value, int):
            errors.append(f"{loc}: {field} must be integer or null")

    speaker = row.get("speaker_agent")
    if manifest and speaker:
        agent_model_map = manifest.get("agent_model_map") or {}
        agent_elo_map = manifest.get("agent_elo_map") or {}
        agent_role_map = manifest.get("agent_role_map") or {}
        if speaker not in agent_model_map:
            warnings.append(f"{loc}: speaker_agent {speaker!r} is not in manifest agent_model_map")
        else:
            if row.get("speaker_model") != agent_model_map.get(speaker):
                warnings.append(
                    f"{loc}: speaker_model {row.get('speaker_model')!r} does not match "
                    f"manifest {agent_model_map.get(speaker)!r}"
                )
            if row.get("speaker_elo") != agent_elo_map.get(speaker):
                warnings.append(
                    f"{loc}: speaker_elo {row.get('speaker_elo')!r} does not match "
                    f"manifest {agent_elo_map.get(speaker)!r}"
                )
            if row.get("speaker_role") != agent_role_map.get(speaker):
                warnings.append(
                    f"{loc}: speaker_role {row.get('speaker_role')!r} does not match "
                    f"manifest {agent_role_map.get(speaker)!r}"
                )

    return errors, warnings


def confidence_rank(value: str) -> int:
    return {"low": 1, "medium": 2, "high": 3}.get(value, 0)


def aggregate(events: list[dict[str, Any]], tag_titles: dict[str, str]) -> None:
    events_out = OUT_DIR / "llm_event_tags.jsonl"
    write_jsonl(events_out, events)

    tag_counts = Counter(row["tag_code"] for row in events)
    tag_rows = [
        {
            "tag_code": tag_code,
            "tag_title": tag_titles[tag_code],
            "event_count": tag_counts.get(tag_code, 0),
        }
        for tag_code in sorted(tag_titles)
    ]
    write_csv(
        OUT_DIR / "llm_event_tag_counts_by_tag.csv",
        tag_rows,
        ["tag_code", "tag_title", "event_count"],
    )

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in events:
        key = (row["result_path"], row["tag_code"])
        current = grouped.setdefault(
            key,
            {
                "result_path": row["result_path"],
                "config_id": row["config_id"],
                "experiment_family": row["experiment_family"],
                "game_label": row["game_label"],
                "n_agents": row["n_agents"],
                "tag_code": row["tag_code"],
                "tag_title": row["tag_title"],
                "event_count": 0,
                "max_confidence": row["confidence"],
            },
        )
        current["event_count"] += 1
        if confidence_rank(row["confidence"]) > confidence_rank(current["max_confidence"]):
            current["max_confidence"] = row["confidence"]
    rollout_rows = sorted(grouped.values(), key=lambda r: (r["result_path"], r["tag_code"]))
    write_csv(
        OUT_DIR / "llm_rollout_tag_summary.csv",
        rollout_rows,
        [
            "result_path",
            "config_id",
            "experiment_family",
            "game_label",
            "n_agents",
            "tag_code",
            "tag_title",
            "event_count",
            "max_confidence",
        ],
    )


def write_report(
    *,
    event_files: list[Path],
    event_count: int,
    completed_chunks: set[str],
    errors: list[str],
    warnings: list[str],
    tag_counts: Counter[str],
) -> None:
    lines = [
        "# LLM Strategic Tag Adjudication Validation",
        "",
        f"- Event files: {len(event_files)}",
        f"- Completed chunks: {len(completed_chunks)}",
        f"- Event rows: {event_count}",
        f"- Errors: {len(errors)}",
        f"- Warnings: {len(warnings)}",
        "",
        "## Top Tag Counts",
        "",
    ]
    for tag_code, count in tag_counts.most_common(25):
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
    (OUT_DIR / "llm_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    global OUT_DIR, CHUNK_INDEX, CODEBOOK, OUTPUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Adjudication output directory containing chunk_index.jsonl and subagent_outputs/.",
    )
    parser.add_argument("--write-aggregate", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    OUT_DIR = args.out_dir.resolve()
    CHUNK_INDEX = OUT_DIR / "chunk_index.jsonl"
    CODEBOOK = OUT_DIR / "llm_tag_codebook.json"
    OUTPUT_DIR = OUT_DIR / "subagent_outputs"

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
            row_errors, row_warnings = validate_event(
                row,
                expected_chunk_id=chunk_id,
                tag_titles=tag_titles,
                rollout_manifest=rollout_manifest,
            )
            errors.extend(row_errors)
            warnings.extend(row_warnings)
        events.extend(file_events)

    tag_counts = Counter(row.get("tag_code") for row in events if row.get("tag_code"))
    write_report(
        event_files=event_files,
        event_count=len(events),
        completed_chunks=completed_chunks,
        errors=errors,
        warnings=warnings,
        tag_counts=tag_counts,
    )
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

#!/usr/bin/env python3
"""Strictly validate and aggregate nine-seed Codex TTC labels."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT = PROJECT_ROOT / "analysis/ttc_claude_nine_seed_codex_adjudication_20260728"
OUTPUTS = ROOT / "subagent_outputs"
REPORT_TITLE = "Nine-seed Claude TTC Codex Adjudication Validation"
EVENT_TYPES = {
    "utterance",
    "private_thinking",
    "proposal_reasoning",
    "vote_reasoning",
    "reflection",
    "formal_outcome",
}
SOURCE_KINDS = {"conversation_log", "interaction", "formal_outcome"}
PHASES = {
    "discussion",
    "private_thinking",
    "proposal",
    "voting",
    "reflection",
    "final_outcome",
}
CONFIDENCES = {"high", "medium", "low"}
IDENTITY_FIELDS = [
    "rollout_id",
    "seed",
    "source_config_id",
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
]
REQUIRED_FIELDS = [
    "chunk_id",
    *IDENTITY_FIELDS,
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
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            row["_source_file"] = str(path)
            row["_source_line"] = line_number
            rows.append(row)
    return rows


def clean(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(clean(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def location(row: dict[str, Any]) -> str:
    return f"{row.get('_source_file')}:{row.get('_source_line')}"


def quote_is_in(quote: Any, source: Any) -> bool:
    if not isinstance(quote, str) or not quote.strip() or not isinstance(source, str):
        return False
    if quote in source:
        return True
    normalize = lambda text: re.sub(r"\s+", " ", text).strip()
    return normalize(quote) in normalize(source)


def canonical_interaction_phase(value: Any) -> Any:
    """Map saved retry/invalid-attempt phase labels to the public schema."""
    if not isinstance(value, str) or value in PHASES:
        return value
    for prefix, canonical in (
        ("discussion_", "discussion"),
        ("private_thinking_", "private_thinking"),
        ("proposal_", "proposal"),
        ("voting_", "voting"),
        ("reflection_", "reflection"),
    ):
        if value.startswith(prefix):
            return canonical
    return value


def load_context() -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, str],
]:
    chunks = {row["chunk_id"]: row for row in read_jsonl(ROOT / "chunk_index.jsonl")}
    manifests = {}
    rollout_to_chunk = {}
    views = {}
    for chunk_id, chunk in chunks.items():
        for row in read_jsonl(Path(chunk["manifest_path"])):
            rollout_id = row["rollout_id"]
            manifests[rollout_id] = row
            rollout_to_chunk[rollout_id] = chunk_id
            views[rollout_id] = json.loads(
                Path(row["rollout_view_path"]).read_text(encoding="utf-8")
            )
    titles = {
        row["tag_code"]: row["tag_title"]
        for row in json.loads((ROOT / "llm_tag_codebook.json").read_text())
    }
    return chunks, manifests, views, titles | {"__rollout_to_chunk__": rollout_to_chunk}


def validate_event(
    row: dict[str, Any],
    expected_chunk: str,
    manifests: dict[str, dict[str, Any]],
    views: dict[str, dict[str, Any]],
    title_context: dict[str, Any],
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    where = location(row)
    missing = [field for field in REQUIRED_FIELDS if field not in row]
    if missing:
        errors.append(f"{where}: missing fields: {', '.join(missing)}")
    if row.get("chunk_id") != expected_chunk:
        errors.append(f"{where}: wrong chunk_id {row.get('chunk_id')!r}")

    rollout_id = row.get("rollout_id")
    manifest = manifests.get(str(rollout_id))
    chunk_map = title_context["__rollout_to_chunk__"]
    if manifest is None:
        errors.append(f"{where}: unknown rollout_id {rollout_id!r}")
        return errors, warnings
    if chunk_map[rollout_id] != expected_chunk:
        errors.append(f"{where}: rollout belongs to {chunk_map[rollout_id]}")
    for field in IDENTITY_FIELDS:
        if str(row.get(field)) != str(manifest.get(field)):
            errors.append(
                f"{where}: {field}={row.get(field)!r} != manifest {manifest.get(field)!r}"
            )

    tag = row.get("tag_code")
    if tag not in title_context:
        errors.append(f"{where}: unknown tag_code {tag!r}")
    elif row.get("tag_title") != title_context[tag]:
        errors.append(f"{where}: tag_title does not match codebook for {tag!r}")
    if row.get("evidence_type") not in EVENT_TYPES:
        errors.append(f"{where}: invalid evidence_type {row.get('evidence_type')!r}")
    if row.get("source_kind") not in SOURCE_KINDS:
        errors.append(f"{where}: invalid source_kind {row.get('source_kind')!r}")
    if row.get("phase") not in PHASES:
        errors.append(f"{where}: invalid phase {row.get('phase')!r}")
    if row.get("confidence") not in CONFIDENCES:
        errors.append(f"{where}: invalid confidence {row.get('confidence')!r}")
    for field in ("speaker_is_target", "speaker_is_baseline", "negation_checked"):
        if not isinstance(row.get(field), bool):
            errors.append(f"{where}: {field} must be boolean")
    for field in ("quote", "rationale"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            errors.append(f"{where}: {field} must be non-empty text")
    for field in (
        "seed",
        "source_config_id",
        "config_id",
        "level_index",
        "n_agents",
        "round",
        "discussion_turn",
        "log_index",
        "interaction_index",
        "speaker_order",
        "total_speakers",
    ):
        if row.get(field) is not None and not isinstance(row[field], int):
            errors.append(f"{where}: {field} must be integer or null")

    speaker = row.get("speaker_agent")
    if speaker is not None:
        role_map = manifest["agent_role_map"]
        if speaker not in role_map:
            errors.append(f"{where}: unknown speaker_agent {speaker!r}")
        else:
            expected_model = manifest["agent_model_map"].get(speaker)
            expected_elo = manifest["agent_elo_map"].get(speaker)
            if row.get("speaker_model") != expected_model:
                errors.append(
                    f"{where}: speaker_model={row.get('speaker_model')!r}, "
                    f"expected {expected_model!r}"
                )
            if row.get("speaker_elo") != expected_elo:
                errors.append(
                    f"{where}: speaker_elo={row.get('speaker_elo')!r}, expected {expected_elo!r}"
                )
            if row.get("speaker_role") != role_map[speaker]:
                errors.append(f"{where}: incorrect speaker_role")
            if row.get("speaker_is_target") != (speaker == manifest["target_agent"]):
                errors.append(f"{where}: incorrect speaker_is_target")
            if row.get("speaker_is_baseline") != (speaker == manifest["baseline_agent"]):
                errors.append(f"{where}: incorrect speaker_is_baseline")

    view = views[rollout_id]
    source_kind = row.get("source_kind")
    if source_kind == "conversation_log":
        index = row.get("log_index")
        if not isinstance(index, int) or not (0 <= index < len(view["conversation_logs"])):
            errors.append(f"{where}: invalid conversation log_index {index!r}")
        else:
            source = view["conversation_logs"][index]
            for field in (
                "round",
                "discussion_turn",
                "speaker_order",
                "total_speakers",
            ):
                if row.get(field) != source.get(field):
                    errors.append(f"{where}: {field} does not match conversation log")
            if row.get("speaker_agent") != source.get("speaker_agent"):
                errors.append(f"{where}: speaker_agent does not match conversation log")
            if row.get("phase") != source.get("phase"):
                errors.append(f"{where}: phase does not match conversation log")
            if row.get("interaction_index") is not None:
                errors.append(f"{where}: conversation row must have null interaction_index")
            if row.get("evidence_type") != "utterance":
                errors.append(f"{where}: conversation row must use utterance evidence_type")
            if not quote_is_in(row.get("quote"), source.get("content")):
                errors.append(f"{where}: quote is not verbatim in cited conversation turn")
    elif source_kind == "interaction":
        index = row.get("interaction_index")
        sources = {
            entry["interaction_index"]: entry
            for entry in view["agent_authored_interactions"]
        }
        if not isinstance(index, int) or index not in sources:
            errors.append(f"{where}: invalid interaction_index {index!r}")
        else:
            source = sources[index]
            for field in ("round", "discussion_turn"):
                if row.get(field) != source.get(field):
                    errors.append(f"{where}: {field} does not match interaction")
            source_phase = canonical_interaction_phase(source.get("phase"))
            if row.get("phase") != source_phase:
                errors.append(f"{where}: phase does not match interaction")
            if row.get("speaker_agent") != source.get("agent_id"):
                errors.append(f"{where}: speaker_agent does not match interaction")
            if row.get("log_index") is not None:
                errors.append(f"{where}: interaction row must have null log_index")
            expected_type = {
                "discussion": "utterance",
                "private_thinking": "private_thinking",
                "proposal": "proposal_reasoning",
                "voting": "vote_reasoning",
                "reflection": "reflection",
            }.get(source_phase)
            if row.get("evidence_type") != expected_type:
                errors.append(
                    f"{where}: evidence_type does not match interaction phase "
                    f"({expected_type!r})"
                )
            if not quote_is_in(row.get("quote"), source.get("response")):
                errors.append(f"{where}: quote is not verbatim in cited interaction")
    elif source_kind == "formal_outcome":
        if row.get("evidence_type") != "formal_outcome":
            errors.append(f"{where}: formal outcome must use formal_outcome evidence_type")
        if row.get("phase") != "final_outcome":
            errors.append(f"{where}: formal outcome must use final_outcome phase")
        for field in (
            "discussion_turn",
            "log_index",
            "interaction_index",
            "speaker_order",
            "total_speakers",
        ):
            if row.get(field) is not None:
                errors.append(f"{where}: formal outcome {field} must be null")
    return errors, warnings


def aggregate(events: list[dict[str, Any]], tag_titles: dict[str, Any]) -> None:
    write_jsonl(ROOT / "ttc_codex_event_tags.jsonl", events)
    count_fields = [
        "tag_code",
        "tag_title",
        "event_count",
        "target_event_count",
        "baseline_event_count",
    ]
    counts: dict[str, Counter] = {}
    for row in events:
        counter = counts.setdefault(row["tag_code"], Counter())
        counter["event_count"] += 1
        counter["target_event_count"] += int(bool(row["speaker_is_target"]))
        counter["baseline_event_count"] += int(bool(row["speaker_is_baseline"]))
    write_csv(
        ROOT / "ttc_codex_event_tag_counts.csv",
        [
            {
                "tag_code": code,
                "tag_title": tag_titles[code],
                **counts.get(code, {}),
            }
            for code in sorted(key for key in tag_titles if not key.startswith("__"))
        ],
        count_fields,
    )
    grouped: dict[tuple[str, str], Counter] = {}
    metadata: dict[tuple[str, str], dict[str, Any]] = {}
    for row in events:
        key = (row["rollout_id"], row["tag_code"])
        grouped.setdefault(key, Counter())["event_count"] += 1
        grouped[key]["target_event_count"] += int(bool(row["speaker_is_target"]))
        grouped[key]["baseline_event_count"] += int(bool(row["speaker_is_baseline"]))
        metadata[key] = {
            field: row[field]
            for field in (
                "rollout_id",
                "seed",
                "source_config_id",
                "family",
                "level",
                "level_index",
                "game_label",
                "game_cell",
                "order",
                "tag_code",
                "tag_title",
            )
        }
    fields = [
        "rollout_id",
        "seed",
        "source_config_id",
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
    ]
    write_csv(
        ROOT / "ttc_codex_rollout_tag_summary.csv",
        [
            {**metadata[key], **grouped[key]}
            for key in sorted(grouped, key=lambda item: (item[0], item[1]))
        ],
        fields,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--write-aggregate", action="store_true")
    parser.add_argument(
        "--chunks",
        nargs="*",
        help="Optional chunk ids to validate, e.g. chunk_0011 chunk_0019.",
    )
    args = parser.parse_args()
    chunks, manifests, views, titles = load_context()
    selected_chunks = set(args.chunks or [])
    unknown_selected = sorted(selected_chunks - set(chunks))
    if unknown_selected:
        raise SystemExit(f"unknown chunk id(s): {', '.join(unknown_selected)}")
    errors: list[str] = []
    warnings: list[str] = []
    events: list[dict[str, Any]] = []
    completed = set()
    for path in sorted(OUTPUTS.glob("chunk_*_events.jsonl")):
        match = re.fullmatch(r"(chunk_\d{4})_events\.jsonl", path.name)
        if not match:
            errors.append(f"{path}: unexpected filename")
            continue
        chunk_id = match.group(1)
        if chunk_id not in chunks:
            errors.append(f"{path}: unknown chunk")
            continue
        if selected_chunks and chunk_id not in selected_chunks:
            continue
        completed.add(chunk_id)
        try:
            rows = read_jsonl(path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        for row in rows:
            row_errors, row_warnings = validate_event(
                row, chunk_id, manifests, views, titles
            )
            errors.extend(row_errors)
            warnings.extend(row_warnings)
        events.extend(rows)
        if not Path(chunks[chunk_id]["audit_path"]).exists():
            errors.append(f"{chunk_id}: missing audit file")

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
    duplicates = [key for key, count in duplicate_keys.items() if count > 1]
    if duplicates:
        warnings.append(f"{len(duplicates)} duplicate event identity keys")
    if args.require_all:
        required_chunks = selected_chunks or set(chunks)
        missing_chunks = sorted(required_chunks - completed)
        if missing_chunks:
            errors.append(
                f"missing {len(missing_chunks)} event files: {', '.join(missing_chunks)}"
            )
    if args.write_aggregate and not errors:
        aggregate(events, titles)
    report = [
        f"# {REPORT_TITLE}",
        "",
        f"- Source rollouts: {len(manifests)}",
        f"- Expected chunks: {len(chunks)}",
        f"- Completed event files: {len(completed)}",
        f"- Event rows: {len(events)}",
        f"- Errors: {len(errors)}",
        f"- Warnings: {len(warnings)}",
        "",
        "## Errors",
        "",
        *(f"- {item}" for item in errors[:300]),
        "",
        "## Warnings",
        "",
        *(f"- {item}" for item in warnings[:300]),
    ]
    (ROOT / "validation_report.md").write_text("\n".join(report) + "\n")
    print(
        f"source_rollouts={len(manifests)} chunks={len(chunks)} "
        f"completed={len(completed)} events={len(events)} "
        f"errors={len(errors)} warnings={len(warnings)}"
    )
    if errors:
        for item in errors[:20]:
            print(f"ERROR {item}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

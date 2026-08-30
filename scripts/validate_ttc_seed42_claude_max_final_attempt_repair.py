#!/usr/bin/env python3
"""Validate and aggregate the 13 repaired seed-42 Claude-max label files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT = PROJECT_ROOT / "analysis/ttc_seed42_claude_max_final_attempt_repair_20260814"
OUTPUTS = ROOT / "subagent_outputs"
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
    "chunk_id",
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
EVENT_FIELDS = [
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
REQUIRED_FIELDS = IDENTITY_FIELDS + EVENT_FIELDS


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


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def where(row: dict[str, Any]) -> str:
    return f"{row.get('_source_file')}:{row.get('_source_line')}"


def quote_is_in(quote: Any, source: Any) -> bool:
    if not isinstance(quote, str) or not quote.strip() or not isinstance(source, str):
        return False
    if quote in source:
        return True
    normalize = lambda text: re.sub(r"\s+", " ", text).strip()
    return normalize(quote) in normalize(source)


def load_context() -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    manifests = {
        row["rollout_id"]: row
        for row in read_jsonl(ROOT / "all_repair_rollouts_manifest.jsonl")
    }
    if len(manifests) != 13:
        raise ValueError(f"repair manifest has {len(manifests)} rollouts, expected 13")
    views = {
        rollout_id: json.loads(Path(row["rollout_view_path"]).read_text(encoding="utf-8"))
        for rollout_id, row in manifests.items()
    }
    codebook = {
        row["tag_code"]: row
        for row in json.loads((ROOT / "llm_tag_codebook.json").read_text(encoding="utf-8"))
    }
    if len(codebook) != 50:
        raise ValueError(f"repair codebook has {len(codebook)} tags, expected 50")
    return manifests, views, codebook


def validate_speaker(
    row: dict[str, Any], manifest: dict[str, Any]
) -> list[str]:
    errors = []
    speaker = row.get("speaker_agent")
    if speaker is None:
        expected = {
            "speaker_model": None,
            "speaker_elo": None,
            "speaker_role": None,
            "speaker_is_target": False,
            "speaker_is_baseline": False,
        }
    elif speaker not in manifest["agent_role_map"]:
        return [f"{where(row)}: unknown speaker_agent {speaker!r}"]
    else:
        expected = {
            "speaker_model": manifest["agent_model_map"].get(speaker),
            "speaker_elo": manifest["agent_elo_map"].get(speaker),
            "speaker_role": manifest["agent_role_map"][speaker],
            "speaker_is_target": speaker == manifest["target_agent"],
            "speaker_is_baseline": speaker == manifest["baseline_agent"],
        }
    for field, value in expected.items():
        if row.get(field) != value:
            errors.append(
                f"{where(row)}: {field}={row.get(field)!r}, expected {value!r}"
            )
    return errors


def validate_event(
    row: dict[str, Any],
    manifest: dict[str, Any],
    view: dict[str, Any],
    codebook: dict[str, dict[str, Any]],
) -> list[str]:
    errors = []
    missing = [field for field in REQUIRED_FIELDS if field not in row]
    extra = sorted(set(clean(row)) - set(REQUIRED_FIELDS))
    if missing:
        errors.append(f"{where(row)}: missing fields: {', '.join(missing)}")
    if extra:
        errors.append(f"{where(row)}: unexpected fields: {', '.join(extra)}")
    for field in IDENTITY_FIELDS:
        if row.get(field) != manifest.get(field):
            errors.append(
                f"{where(row)}: {field}={row.get(field)!r}, "
                f"expected {manifest.get(field)!r}"
            )

    definition = codebook.get(row.get("tag_code"))
    if definition is None:
        errors.append(f"{where(row)}: unknown tag_code {row.get('tag_code')!r}")
    else:
        if row.get("tag_title") != definition["tag_title"]:
            errors.append(f"{where(row)}: tag_title does not match the codebook")
        scope = definition.get("scope_hint") or {}
        minimum = scope.get("min_agents")
        if minimum is not None and int(minimum) > int(manifest["n_agents"]):
            errors.append(
                f"{where(row)}: {row.get('tag_code')} requires {minimum} agents"
            )
        games = scope.get("games") or []
        if games and manifest["game_label"] not in games:
            errors.append(
                f"{where(row)}: {row.get('tag_code')} is scoped to {games}"
            )

    if row.get("evidence_type") not in EVENT_TYPES:
        errors.append(f"{where(row)}: invalid evidence_type")
    if row.get("source_kind") not in SOURCE_KINDS:
        errors.append(f"{where(row)}: invalid source_kind")
    if row.get("phase") not in PHASES:
        errors.append(f"{where(row)}: invalid phase")
    if row.get("confidence") not in CONFIDENCES:
        errors.append(f"{where(row)}: invalid confidence")
    for field in ("speaker_is_target", "speaker_is_baseline", "negation_checked"):
        if not isinstance(row.get(field), bool):
            errors.append(f"{where(row)}: {field} must be boolean")
    if row.get("negation_checked") is not True:
        errors.append(f"{where(row)}: negation_checked must be true")
    for field in ("quote", "rationale"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            errors.append(f"{where(row)}: {field} must be non-empty text")
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
            errors.append(f"{where(row)}: {field} must be integer or null")
    errors.extend(validate_speaker(row, manifest))

    source_kind = row.get("source_kind")
    if source_kind == "conversation_log":
        index = row.get("log_index")
        logs = view["conversation_logs"]
        if not isinstance(index, int) or not 0 <= index < len(logs):
            errors.append(f"{where(row)}: invalid log_index {index!r}")
        else:
            source = logs[index]
            if source.get("phase") != "discussion":
                errors.append(f"{where(row)}: only public discussion can cite conversation_logs")
            for field in (
                "phase",
                "round",
                "discussion_turn",
                "speaker_order",
                "total_speakers",
            ):
                if row.get(field) != source.get(field):
                    errors.append(f"{where(row)}: {field} does not match the cited turn")
            if row.get("speaker_agent") != source.get("speaker_agent"):
                errors.append(f"{where(row)}: speaker_agent does not match the cited turn")
            if row.get("interaction_index") is not None:
                errors.append(f"{where(row)}: conversation row needs null interaction_index")
            if row.get("evidence_type") != "utterance":
                errors.append(f"{where(row)}: conversation evidence must be utterance")
            if not quote_is_in(row.get("quote"), source.get("content")):
                errors.append(f"{where(row)}: quote is not in the cited public turn")
    elif source_kind == "interaction":
        index = row.get("interaction_index")
        interactions = {
            source["interaction_index"]: source
            for source in view["agent_authored_interactions"]
        }
        if not isinstance(index, int) or index not in interactions:
            errors.append(f"{where(row)}: invalid interaction_index {index!r}")
        else:
            source = interactions[index]
            if source.get("phase") == "discussion":
                errors.append(f"{where(row)}: public discussion must cite conversation_logs")
            for field in ("phase", "round", "discussion_turn"):
                if row.get(field) != source.get(field):
                    errors.append(f"{where(row)}: {field} does not match the interaction")
            if row.get("speaker_agent") != source.get("agent_id"):
                errors.append(f"{where(row)}: speaker_agent does not match the interaction")
            if row.get("log_index") is not None:
                errors.append(f"{where(row)}: interaction row needs null log_index")
            if row.get("speaker_order") is not None or row.get("total_speakers") is not None:
                errors.append(f"{where(row)}: interaction speaker coordinates must be null")
            expected_type = {
                "private_thinking": "private_thinking",
                "proposal": "proposal_reasoning",
                "voting": "vote_reasoning",
                "reflection": "reflection",
            }.get(source.get("phase"))
            if row.get("evidence_type") != expected_type:
                errors.append(
                    f"{where(row)}: evidence_type must be {expected_type!r}"
                )
            if not quote_is_in(row.get("quote"), source.get("response")):
                errors.append(f"{where(row)}: quote is not in the cited interaction")
    elif source_kind == "formal_outcome":
        if definition is not None and not (
            definition.get("scope_hint") or {}
        ).get("structural"):
            errors.append(f"{where(row)}: formal_outcome needs a structural tag")
        if row.get("evidence_type") != "formal_outcome":
            errors.append(f"{where(row)}: formal outcome evidence_type mismatch")
        if row.get("phase") != "final_outcome":
            errors.append(f"{where(row)}: formal outcome phase mismatch")
        for field in (
            "round",
            "discussion_turn",
            "log_index",
            "interaction_index",
            "speaker_order",
            "total_speakers",
        ):
            if row.get(field) is not None:
                errors.append(f"{where(row)}: formal outcome {field} must be null")
    return errors


def aggregate(
    manifests: dict[str, dict[str, Any]],
    events: list[dict[str, Any]],
    codebook: dict[str, dict[str, Any]],
) -> None:
    write_jsonl(ROOT / "ttc_seed42_claude_max_repaired_event_tags.jsonl", events)
    per_rollout = Counter(row["rollout_id"] for row in events)
    per_target = Counter(
        row["rollout_id"] for row in events if row["speaker_is_target"]
    )
    per_baseline = Counter(
        row["rollout_id"] for row in events if row["speaker_is_baseline"]
    )
    rollout_fields = [
        "rollout_id",
        "seed",
        "config_id",
        "game_label",
        "game_cell",
        "order",
        "final_round",
        "consensus_reached",
        "target_utility",
        "baseline_utility",
        "event_count",
        "target_event_count",
        "baseline_event_count",
    ]
    write_csv(
        ROOT / "ttc_seed42_claude_max_repaired_rollout_summary.csv",
        [
            {
                **{field: manifest.get(field) for field in rollout_fields},
                "event_count": per_rollout[rollout_id],
                "target_event_count": per_target[rollout_id],
                "baseline_event_count": per_baseline[rollout_id],
            }
            for rollout_id, manifest in sorted(manifests.items())
        ],
        rollout_fields,
    )
    counts = Counter(row["tag_code"] for row in events)
    target_counts = Counter(
        row["tag_code"] for row in events if row["speaker_is_target"]
    )
    baseline_counts = Counter(
        row["tag_code"] for row in events if row["speaker_is_baseline"]
    )
    tag_fields = [
        "tag_code",
        "tag_title",
        "event_count",
        "target_event_count",
        "baseline_event_count",
    ]
    write_csv(
        ROOT / "ttc_seed42_claude_max_repaired_tag_counts.csv",
        [
            {
                "tag_code": code,
                "tag_title": definition["tag_title"],
                "event_count": counts[code],
                "target_event_count": target_counts[code],
                "baseline_event_count": baseline_counts[code],
            }
            for code, definition in sorted(codebook.items())
        ],
        tag_fields,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--write-aggregate", action="store_true")
    args = parser.parse_args()
    manifests, views, codebook = load_context()
    mapping = json.loads(
        (ROOT / "final_attempt_mapping_manifest.json").read_text(encoding="utf-8")
    )
    errors = []
    warnings = []
    events = []
    completed = set()

    if len(mapping) != 13 or not all(row.get("all_checks_pass") for row in mapping):
        errors.append("final-attempt mapping manifest is incomplete or has failed checks")
    for rollout_id, manifest in sorted(manifests.items()):
        if file_sha256(Path(manifest["result_path"])) != manifest["result_sha256"]:
            errors.append(f"{rollout_id}: result source hash changed")
        if file_sha256(Path(manifest["interactions_path"])) != manifest["interactions_sha256"]:
            errors.append(f"{rollout_id}: final-attempt interaction source hash changed")
        if file_sha256(Path(manifest["canonical_interactions_path"])) != manifest["canonical_interactions_sha256"]:
            errors.append(f"{rollout_id}: canonical interaction source hash changed")
        output_path = Path(manifest["output_path"])
        audit_path = Path(manifest["audit_path"])
        if not output_path.exists():
            if args.require_all:
                errors.append(f"{rollout_id}: missing event file")
            continue
        completed.add(rollout_id)
        if not audit_path.exists():
            errors.append(f"{rollout_id}: missing audit file")
        try:
            rows = read_jsonl(output_path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        for row in rows:
            errors.extend(validate_event(row, manifest, views[rollout_id], codebook))
        events.extend(rows)

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
        errors.append(f"{len(duplicates)} duplicate event identity keys")
    if args.require_all and completed != set(manifests):
        errors.append(f"only {len(completed)}/13 repaired rollouts have event files")
    if args.write_aggregate:
        if errors:
            errors.append("aggregate not written because validation failed")
        elif completed != set(manifests):
            errors.append("aggregate requires all 13 repaired rollouts")
        else:
            aggregate(manifests, events, codebook)

    report = [
        "# Seed-42 Claude-max final-attempt repair validation",
        "",
        f"- Source rollouts: {len(manifests)}",
        f"- Completed event files: {len(completed)}",
        f"- Event rows: {len(events)}",
        f"- Errors: {len(errors)}",
        f"- Warnings: {len(warnings)}",
        "",
        "## Errors",
        "",
        *(f"- {item}" for item in errors),
        "",
        "## Warnings",
        "",
        *(f"- {item}" for item in warnings),
    ]
    (ROOT / "validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(
        ROOT / "validation_status.json",
        {
            "source_rollouts": len(manifests),
            "completed_event_files": len(completed),
            "event_rows": len(events),
            "error_count": len(errors),
            "warning_count": len(warnings),
            "errors": errors,
            "warnings": warnings,
            "aggregate_written": bool(
                args.write_aggregate and not errors and completed == set(manifests)
            ),
        },
    )
    print(
        f"rollouts={len(manifests)} completed={len(completed)} events={len(events)} "
        f"errors={len(errors)} warnings={len(warnings)}"
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

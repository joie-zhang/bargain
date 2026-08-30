#!/usr/bin/env python3
"""Validate and aggregate 648 one-rollout GPT-5 TTC semantic tag files."""

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
ROOT = PROJECT_ROOT / "analysis/ttc_gpt5_nine_seed_codex_adjudication_20260809"
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
EVENT_FIELDS = [
    "tag_code",
    "tag_title",
    "evidence_type",
    "source_kind",
    "phase",
    "round",
    "discussion_turn",
    "log_index",
    "interaction_index",
    "speaker_agent",
    "speaker_model",
    "speaker_elo",
    "speaker_role",
    "speaker_is_target",
    "speaker_is_baseline",
    "speaker_order",
    "total_speakers",
    "quote",
    "rationale",
    "confidence",
    "negation_checked",
]
IDENTITY_FIELDS = [
    "rollout_id",
    "seed",
    "source_config_id",
    "config_id",
    "result_path",
    "interactions_path",
    "rollout_view_path",
    "provider",
    "family",
    "level",
    "level_index",
    "game_label",
    "game_cell",
    "game_type",
    "n_agents",
    "order",
    "target_agent",
    "baseline_agent",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
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


def quote_is_in(quote: Any, source: Any) -> bool:
    if not isinstance(quote, str) or not quote.strip() or not isinstance(source, str):
        return False
    if quote in source:
        return True
    normalize = lambda text: re.sub(r"\s+", " ", text).strip()
    return normalize(quote) in normalize(source)


def load_context() -> tuple[
    dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]
]:
    manifests = {
        row["rollout_id"]: row
        for row in read_jsonl(ROOT / "all_rollouts_manifest.jsonl")
    }
    if len(manifests) != 648:
        raise ValueError(f"manifest has {len(manifests)} unique rollouts, expected 648")
    views = {
        rollout_id: json.loads(Path(row["rollout_view_path"]).read_text())
        for rollout_id, row in manifests.items()
    }
    codebook = {
        row["tag_code"]: row
        for row in json.loads((ROOT / "llm_tag_codebook.json").read_text())
    }
    return manifests, views, codebook


def validate_speaker(
    event: dict[str, Any], manifest: dict[str, Any], where: str
) -> list[str]:
    errors = []
    speaker = event.get("speaker_agent")
    if speaker is None:
        expected = {
            "speaker_model": None,
            "speaker_elo": None,
            "speaker_role": None,
            "speaker_is_target": False,
            "speaker_is_baseline": False,
        }
    elif speaker not in manifest["agent_role_map"]:
        return [f"{where}: unknown speaker_agent {speaker!r}"]
    else:
        expected = {
            "speaker_model": manifest["agent_model_map"].get(speaker),
            "speaker_elo": manifest["agent_elo_map"].get(speaker),
            "speaker_role": manifest["agent_role_map"][speaker],
            "speaker_is_target": speaker == manifest["target_agent"],
            "speaker_is_baseline": speaker == manifest["baseline_agent"],
        }
    for field, value in expected.items():
        if event.get(field) != value:
            errors.append(f"{where}: {field}={event.get(field)!r}, expected {value!r}")
    return errors


def validate_event(
    event: Any,
    manifest: dict[str, Any],
    view: dict[str, Any],
    codebook: dict[str, dict[str, Any]],
    where: str,
) -> list[str]:
    if not isinstance(event, dict):
        return [f"{where}: event must be an object"]
    errors = []
    missing = [field for field in EVENT_FIELDS if field not in event]
    extra = sorted(set(event) - set(EVENT_FIELDS))
    if missing:
        errors.append(f"{where}: missing fields: {', '.join(missing)}")
    if extra:
        errors.append(f"{where}: unexpected fields: {', '.join(extra)}")
    tag = event.get("tag_code")
    definition = codebook.get(tag)
    if definition is None:
        errors.append(f"{where}: unknown tag_code {tag!r}")
    else:
        if event.get("tag_title") != definition["tag_title"]:
            errors.append(f"{where}: tag_title does not match codebook")
        minimum = definition.get("scope_hint", {}).get("min_agents")
        if minimum is not None and int(minimum) > int(manifest["n_agents"]):
            errors.append(f"{where}: {tag} requires at least {minimum} agents")
        games = definition.get("scope_hint", {}).get("games") or []
        if games and manifest["game_label"] not in games:
            errors.append(f"{where}: {tag} is scoped to {games}, not {manifest['game_label']}")

    if event.get("evidence_type") not in EVENT_TYPES:
        errors.append(f"{where}: invalid evidence_type {event.get('evidence_type')!r}")
    if event.get("source_kind") not in SOURCE_KINDS:
        errors.append(f"{where}: invalid source_kind {event.get('source_kind')!r}")
    if event.get("phase") not in PHASES:
        errors.append(f"{where}: invalid phase {event.get('phase')!r}")
    if event.get("confidence") not in CONFIDENCES:
        errors.append(f"{where}: invalid confidence {event.get('confidence')!r}")
    for field in ("speaker_is_target", "speaker_is_baseline", "negation_checked"):
        if not isinstance(event.get(field), bool):
            errors.append(f"{where}: {field} must be boolean")
    if event.get("negation_checked") is not True:
        errors.append(f"{where}: negation_checked must be true")
    for field in ("quote", "rationale"):
        if not isinstance(event.get(field), str) or not event[field].strip():
            errors.append(f"{where}: {field} must be non-empty text")
    for field in (
        "round",
        "discussion_turn",
        "log_index",
        "interaction_index",
        "speaker_elo",
        "speaker_order",
        "total_speakers",
    ):
        if event.get(field) is not None and not isinstance(event[field], int):
            errors.append(f"{where}: {field} must be integer or null")
    errors.extend(validate_speaker(event, manifest, where))

    source_kind = event.get("source_kind")
    if source_kind == "conversation_log":
        index = event.get("log_index")
        logs = view["conversation_logs"]
        if not isinstance(index, int) or not 0 <= index < len(logs):
            errors.append(f"{where}: invalid log_index {index!r}")
        else:
            source = logs[index]
            for field in (
                "phase",
                "round",
                "discussion_turn",
                "speaker_order",
                "total_speakers",
            ):
                if event.get(field) != source.get(field):
                    errors.append(f"{where}: {field} does not match conversation log")
            if event.get("speaker_agent") != source.get("speaker_agent"):
                errors.append(f"{where}: speaker_agent does not match conversation log")
            if event.get("interaction_index") is not None:
                errors.append(f"{where}: interaction_index must be null")
            if event.get("evidence_type") != "utterance":
                errors.append(f"{where}: conversation evidence_type must be utterance")
            if not quote_is_in(event.get("quote"), source.get("content")):
                errors.append(f"{where}: quote is not in cited conversation log")
    elif source_kind == "interaction":
        index = event.get("interaction_index")
        interactions = {
            row["interaction_index"]: row
            for row in view["agent_authored_interactions"]
        }
        if not isinstance(index, int) or index not in interactions:
            errors.append(f"{where}: invalid interaction_index {index!r}")
        else:
            source = interactions[index]
            if source.get("phase") == "discussion":
                errors.append(f"{where}: public discussion must cite conversation_logs")
            for field in ("phase", "round", "discussion_turn"):
                if event.get(field) != source.get(field):
                    errors.append(f"{where}: {field} does not match interaction")
            if event.get("speaker_agent") != source.get("agent_id"):
                errors.append(f"{where}: speaker_agent does not match interaction")
            if event.get("log_index") is not None:
                errors.append(f"{where}: log_index must be null")
            if event.get("speaker_order") is not None or event.get("total_speakers") is not None:
                errors.append(f"{where}: interaction speaker coordinates must be null")
            expected_type = {
                "private_thinking": "private_thinking",
                "proposal": "proposal_reasoning",
                "voting": "vote_reasoning",
                "reflection": "reflection",
            }.get(source.get("phase"))
            if event.get("evidence_type") != expected_type:
                errors.append(
                    f"{where}: evidence_type should be {expected_type!r} for this phase"
                )
            if not quote_is_in(event.get("quote"), source.get("response")):
                errors.append(f"{where}: quote is not in cited interaction")
    elif source_kind == "formal_outcome":
        if definition is not None and not definition.get("scope_hint", {}).get("structural"):
            errors.append(f"{where}: formal_outcome is only valid for structural tags")
        if event.get("evidence_type") != "formal_outcome":
            errors.append(f"{where}: formal outcome evidence_type mismatch")
        if event.get("phase") != "final_outcome":
            errors.append(f"{where}: formal outcome phase must be final_outcome")
        for field in (
            "round",
            "discussion_turn",
            "log_index",
            "interaction_index",
            "speaker_order",
            "total_speakers",
        ):
            if event.get(field) is not None:
                errors.append(f"{where}: formal outcome {field} must be null")
    return errors


def aggregate(
    manifests: dict[str, dict[str, Any]],
    outputs: dict[str, dict[str, Any]],
    events: list[dict[str, Any]],
    codebook: dict[str, dict[str, Any]],
) -> None:
    write_jsonl(ROOT / "ttc_gpt5_event_tags.jsonl", events)
    review_fields = [
        "rollout_id",
        "seed",
        "source_config_id",
        "family",
        "level",
        "level_index",
        "game_label",
        "game_cell",
        "order",
        "consensus_reached",
        "event_count",
        "distinct_tag_count",
        "target_event_count",
        "baseline_event_count",
        "formal_outcome_event_count",
        "reviewed",
        "output_path",
    ]
    review_rows = []
    for rollout_id, manifest in manifests.items():
        output = outputs[rollout_id]
        rollout_events = [row for row in events if row["rollout_id"] == rollout_id]
        review_rows.append(
            {
                **{field: manifest.get(field) for field in review_fields},
                "event_count": len(rollout_events),
                "distinct_tag_count": len({row["tag_code"] for row in rollout_events}),
                "target_event_count": sum(row["speaker_is_target"] for row in rollout_events),
                "baseline_event_count": sum(row["speaker_is_baseline"] for row in rollout_events),
                "formal_outcome_event_count": sum(
                    row["source_kind"] == "formal_outcome" for row in rollout_events
                ),
                "reviewed": output["reviewed"],
                "output_path": manifest["output_path"],
            }
        )
    write_csv(ROOT / "ttc_gpt5_rollout_review_summary.csv", review_rows, review_fields)

    counts = Counter(row["tag_code"] for row in events)
    role_counts = Counter(
        (row["tag_code"], row["speaker_role"] or "formal_outcome") for row in events
    )
    count_fields = [
        "tag_code",
        "tag_title",
        "event_count",
        "target_event_count",
        "baseline_event_count",
        "unattributed_event_count",
    ]
    count_rows = []
    for code in sorted(codebook):
        count_rows.append(
            {
                "tag_code": code,
                "tag_title": codebook[code]["tag_title"],
                "event_count": counts[code],
                "target_event_count": role_counts[(code, "target")],
                "baseline_event_count": role_counts[(code, "baseline")],
                "unattributed_event_count": role_counts[(code, "formal_outcome")],
            }
        )
    write_csv(ROOT / "ttc_gpt5_event_tag_counts.csv", count_rows, count_fields)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--write-aggregate", action="store_true")
    parser.add_argument("--rollout-id", action="append")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()

    manifests, views, codebook = load_context()
    selected = set(args.rollout_id or manifests)
    unknown = selected - set(manifests)
    if unknown:
        raise SystemExit(f"unknown rollout ids: {', '.join(sorted(unknown))}")

    errors = []
    warnings = []
    outputs: dict[str, dict[str, Any]] = {}
    events = []
    expected_files = {Path(manifests[rid]["output_path"]).resolve() for rid in selected}
    if not args.rollout_id:
        for path in OUTPUTS.glob("*.json"):
            if path.resolve() not in expected_files:
                errors.append(f"unexpected output file: {path}")

    for rollout_id in sorted(selected):
        manifest = manifests[rollout_id]
        if file_sha256(Path(manifest["result_path"])) != manifest["result_sha256"]:
            errors.append(f"{rollout_id}: result source hash changed")
        if file_sha256(Path(manifest["interactions_path"])) != manifest["interactions_sha256"]:
            errors.append(f"{rollout_id}: interactions source hash changed")
        path = Path(manifest["output_path"])
        if not path.exists():
            if args.require_all:
                errors.append(f"{rollout_id}: missing output {path}")
            continue
        try:
            output = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"{rollout_id}: cannot read output: {exc}")
            continue
        if not isinstance(output, dict):
            errors.append(f"{rollout_id}: top-level output must be an object")
            continue
        if set(output) != {"rollout_id", "reviewed", "events"}:
            errors.append(
                f"{rollout_id}: top-level keys must be rollout_id, reviewed, events"
            )
        if output.get("rollout_id") != rollout_id:
            errors.append(f"{rollout_id}: output rollout_id mismatch")
        if output.get("reviewed") is not True:
            errors.append(f"{rollout_id}: reviewed must be true")
        if not isinstance(output.get("events"), list):
            errors.append(f"{rollout_id}: events must be a list")
            continue
        outputs[rollout_id] = output
        rollout_events = []
        for index, event in enumerate(output["events"]):
            where = f"{path}:events[{index}]"
            errors.extend(validate_event(event, manifest, views[rollout_id], codebook, where))
            if isinstance(event, dict):
                rollout_events.append(
                    {
                        **{field: manifest.get(field) for field in IDENTITY_FIELDS},
                        **event,
                    }
                )
        duplicate_keys = Counter(
            (
                row.get("tag_code"),
                row.get("source_kind"),
                row.get("log_index"),
                row.get("interaction_index"),
                row.get("speaker_agent"),
            )
            for row in rollout_events
        )
        duplicates = [key for key, count in duplicate_keys.items() if count > 1]
        if duplicates:
            errors.append(f"{rollout_id}: {len(duplicates)} duplicate event identity keys")
        events.extend(rollout_events)

    if args.require_all:
        missing = selected - set(outputs)
        if missing:
            errors.append(f"missing or unreadable outputs: {len(missing)}")
    if args.write_aggregate:
        if errors:
            errors.append("aggregate not written because validation failed")
        elif set(outputs) != set(manifests):
            errors.append("aggregate requires all 648 outputs")
        else:
            aggregate(manifests, outputs, events, codebook)

    report = [
        "# Nine-seed GPT-5 TTC Codex adjudication validation",
        "",
        f"- Expected rollouts: {len(selected)}",
        f"- Completed valid-shape outputs read: {len(outputs)}",
        f"- Event rows: {len(events)}",
        f"- Errors: {len(errors)}",
        f"- Warnings: {len(warnings)}",
        "",
        "## Errors",
        "",
        *(f"- {item}" for item in errors[:500]),
        "",
        "## Warnings",
        "",
        *(f"- {item}" for item in warnings[:500]),
    ]
    if not args.no_report:
        (ROOT / "validation_report.md").write_text(
            "\n".join(report) + "\n", encoding="utf-8"
        )
    print(
        f"expected={len(selected)} outputs={len(outputs)} events={len(events)} "
        f"errors={len(errors)} warnings={len(warnings)}"
    )
    if errors:
        for error in errors[:30]:
            print(f"ERROR {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

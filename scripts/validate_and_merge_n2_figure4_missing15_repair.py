#!/usr/bin/env python3
"""Validate and merge the isolated Figure 4 missing-annotation repair."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path("/scratch/gpfs/DANQIC/jz4391/bargain")
SOURCE = ROOT / "analysis/llm_strategic_tag_adjudication_n2_gpt5_20260629"
REPAIR = ROOT / "analysis/n2_figure4_missing15_codex_repair_20260814"
SOURCE_MANIFEST = SOURCE / "all_rollouts_manifest.jsonl"
SOURCE_EVENTS = SOURCE / "llm_event_tags.jsonl"
CODEBOOK = SOURCE / "llm_tag_codebook.json"

EXPECTED_PATHS = (
    ROOT / "experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_gpt-5-nano-high/strong_first/comp_0.0/turns_2/run_2/run_2_experiment_results.json",
    ROOT / "experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_gpt-5-nano-high/weak_first/comp_0.0/turns_2/run_1/run_1_experiment_results.json",
    ROOT / "experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_gpt-5-nano-high/strong_first/comp_0.25/turns_2/run_2/run_2_experiment_results.json",
    ROOT / "experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_gpt-5-nano-high/weak_first/comp_0.25/turns_2/run_1/run_1_experiment_results.json",
    ROOT / "experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_gpt-5-nano-high/strong_first/comp_0.5/turns_2/run_2/run_2_experiment_results.json",
    ROOT / "experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_gpt-5-nano-high/weak_first/comp_0.5/turns_2/run_1/run_1_experiment_results.json",
    ROOT / "experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_gpt-5-nano-high/strong_first/comp_0.75/turns_2/run_2/run_2_experiment_results.json",
    ROOT / "experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_gpt-5-nano-high/weak_first/comp_0.75/turns_2/run_1/run_1_experiment_results.json",
    ROOT / "experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_gpt-5-nano-high/strong_first/comp_0.9/turns_2/run_2/run_2_experiment_results.json",
    ROOT / "experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_gpt-5-nano-high/weak_first/comp_0.9/turns_2/run_1/run_1_experiment_results.json",
    ROOT / "experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_gpt-5-nano-high/strong_first/comp_0.95/turns_2/run_2/run_2_experiment_results.json",
    ROOT / "experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_gpt-5-nano-high/weak_first/comp_0.95/turns_2/run_1/run_1_experiment_results.json",
    ROOT / "experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_gpt-5-nano-high/strong_first/comp_1.0/turns_2/run_2/run_2_experiment_results.json",
    ROOT / "experiments/results/scaling_experiment_20260404_064451/gpt-5-nano_vs_gpt-5-nano-high/weak_first/comp_1.0/turns_2/run_1/run_1_experiment_results.json",
    ROOT / "experiments/results/cofunding_20260405_083548/model_scale/gpt-5-nano_vs_qwen2.5-72b-instruct/weak_first/alpha_1_0_sigma_1_0/run_1_experiment_results.json",
)

REQUIRED_FIELDS = frozenset(
    {
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
    }
)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def event_identity(row: dict[str, object]) -> tuple[object, ...]:
    return (
        row["result_path"],
        row["tag_code"],
        row["evidence_type"],
        row["phase"],
        row["round"],
        row["discussion_turn"],
        row["log_index"],
        row["speaker_agent"],
    )


def prepare_manifest() -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    all_manifest = read_jsonl(SOURCE_MANIFEST)
    manifest_by_path = {str(row["result_path"]): row for row in all_manifest}
    if len(manifest_by_path) != len(all_manifest):
        raise ValueError("Source manifest has duplicate result paths")

    source_event_paths = {
        str(row["result_path"])
        for row in read_jsonl(SOURCE_EVENTS)
        if row.get("result_path")
    }
    rows: list[dict[str, object]] = []
    for index, path in enumerate(EXPECTED_PATHS):
        result_path = str(path)
        if result_path not in manifest_by_path:
            raise ValueError(f"Expected path is absent from source manifest: {result_path}")
        if result_path in source_event_paths:
            raise ValueError(f"Repair path already has a retained source event: {result_path}")
        source_row = manifest_by_path[result_path]
        rows.append(
            {
                "repair_id": f"repair_{index:04d}",
                "result_path": result_path,
                "result_sha256": sha256(path),
                "config_id": source_row["config_id"],
                "game_label": source_row["game_label"],
                "adversary_model": source_row["adversary_model"],
                "adversary_agent": source_row["adversary_agent"],
                "judge_model": "gpt-5.6-sol",
                "judge_reasoning_effort": "high",
                "annotation_unit": "one transcript per independent Codex subagent",
                "output_path": str(
                    REPAIR
                    / "single_rollout_outputs"
                    / f"repair_{index:04d}_events.jsonl"
                ),
                "audit_path": str(
                    REPAIR / "audits" / f"repair_{index:04d}_audit.md"
                ),
            }
        )
    write_jsonl(REPAIR / "repair_manifest.jsonl", rows)
    return rows, manifest_by_path


def validate_event(
    row: dict[str, object],
    repair_row: dict[str, object],
    manifest_row: dict[str, object],
    transcript: dict[str, object],
    codebook: dict[str, dict[str, object]],
) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_FIELDS - set(row)
    extra = set(row) - REQUIRED_FIELDS
    if missing or extra:
        errors.append(f"schema mismatch missing={sorted(missing)} extra={sorted(extra)}")
        return errors

    expected_scalars = {
        "chunk_id": repair_row["repair_id"],
        "result_path": repair_row["result_path"],
        "config_id": manifest_row["config_id"],
        "experiment_family": "n2_gpt5_bilateral",
        "game_label": manifest_row["game_label"],
        "n_agents": 2,
        "total_speakers": 2,
    }
    for field, expected in expected_scalars.items():
        if row[field] != expected:
            errors.append(f"{field}={row[field]!r}, expected {expected!r}")

    tag_code = str(row["tag_code"])
    if tag_code not in codebook:
        errors.append(f"unknown tag_code {tag_code!r}")
    elif row["tag_title"] != codebook[tag_code]["tag_title"]:
        errors.append("tag_title does not match codebook")
    if row["evidence_type"] not in {"utterance", "proposal_reasoning", "formal_outcome"}:
        errors.append(f"invalid evidence_type {row['evidence_type']!r}")
    if row["confidence"] not in {"high", "medium", "low"}:
        errors.append(f"invalid confidence {row['confidence']!r}")
    if row["negation_checked"] is not True:
        errors.append("negation_checked must be true")
    if not isinstance(row["quote"], str) or not row["quote"].strip():
        errors.append("quote must be non-empty")
    if not isinstance(row["rationale"], str) or not row["rationale"].strip():
        errors.append("rationale must be non-empty")

    role_map = manifest_row["agent_role_map"]
    model_map = manifest_row["agent_model_map"]
    elo_map = manifest_row["agent_elo_map"]
    speaker = row["speaker_agent"]
    if speaker is None:
        if any(row[field] is not None for field in ("speaker_model", "speaker_elo", "speaker_role", "speaker_order")):
            errors.append("null speaker must have null speaker metadata")
    elif speaker not in role_map:
        errors.append(f"unknown speaker_agent {speaker!r}")
    else:
        if row["speaker_model"] != model_map[speaker]:
            errors.append("speaker_model does not match manifest")
        if row["speaker_elo"] != elo_map[speaker]:
            errors.append("speaker_elo does not match manifest")
        if row["speaker_role"] != role_map[speaker]:
            errors.append("speaker_role does not match manifest")

    if row["evidence_type"] == "formal_outcome":
        if row["phase"] != "final_outcome":
            errors.append("formal_outcome row must use final_outcome phase")
        if row["discussion_turn"] is not None:
            errors.append("formal_outcome discussion_turn must be null")
    else:
        logs = transcript.get("conversation_logs")
        log_index = row["log_index"]
        if not isinstance(logs, list):
            errors.append("transcript conversation_logs is not a list")
        elif not isinstance(log_index, int) or not (0 <= log_index < len(logs)):
            errors.append(f"log_index {log_index!r} is outside conversation_logs")
        else:
            log = logs[log_index]
            inferred_speaker_order = log.get("speaker_order")
            if inferred_speaker_order is None and row["speaker_agent"] in {"Agent_1", "Agent_2"}:
                inferred_speaker_order = int(str(row["speaker_agent"]).split("_")[1])
            checks = {
                "phase": log.get("phase"),
                "round": log.get("round"),
                "discussion_turn": log.get("discussion_turn"),
                "speaker_agent": log.get("from"),
                "speaker_order": inferred_speaker_order,
                "total_speakers": log.get("total_speakers", 2),
            }
            for field, expected in checks.items():
                if row[field] != expected:
                    errors.append(
                        f"{field}={row[field]!r}, indexed log has {expected!r}"
                    )
            quote_sources = [str(log.get("content", ""))]
            proposal = log.get("proposal")
            if isinstance(proposal, dict):
                quote_sources.append(str(proposal.get("reasoning", "")))
            if not any(str(row["quote"]) in source for source in quote_sources):
                errors.append(
                    "quote is not an exact substring of indexed log content or proposal reasoning"
                )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    REPAIR.mkdir(parents=True, exist_ok=True)
    (REPAIR / "single_rollout_outputs").mkdir(exist_ok=True)
    (REPAIR / "audits").mkdir(exist_ok=True)
    repair_rows, manifest_by_path = prepare_manifest()
    if args.prepare_only:
        print(f"Prepared {len(repair_rows)} repair manifest rows")
        return

    codebook_rows = json.loads(CODEBOOK.read_text(encoding="utf-8"))
    codebook = {str(row["tag_code"]): row for row in codebook_rows}
    errors: list[str] = []
    all_repair_events: list[dict[str, object]] = []
    row_counts: dict[str, int] = {}
    for repair_row in repair_rows:
        repair_id = str(repair_row["repair_id"])
        output_path = Path(str(repair_row["output_path"]))
        audit_path = Path(str(repair_row["audit_path"]))
        if not output_path.is_file():
            errors.append(f"{repair_id}: missing output {output_path}")
            continue
        if not audit_path.is_file() or not audit_path.read_text(encoding="utf-8").strip():
            errors.append(f"{repair_id}: missing or empty audit {audit_path}")
        events = read_jsonl(output_path)
        if not events:
            errors.append(f"{repair_id}: empty output lacks a machine-readable reviewed-zero record")
            continue
        row_counts[repair_id] = len(events)
        transcript = json.loads(
            Path(str(repair_row["result_path"])).read_text(encoding="utf-8")
        )
        manifest_row = manifest_by_path[str(repair_row["result_path"])]
        for row_number, event in enumerate(events, start=1):
            for error in validate_event(
                event, repair_row, manifest_row, transcript, codebook
            ):
                errors.append(f"{repair_id}:{row_number}: {error}")
        identities = [event_identity(row) for row in events if REQUIRED_FIELDS <= set(row)]
        duplicates = [identity for identity, count in Counter(identities).items() if count > 1]
        if duplicates:
            errors.append(f"{repair_id}: {len(duplicates)} duplicate event identities")
        all_repair_events.extend(events)

    seen_paths = {str(row.get("result_path")) for row in all_repair_events}
    expected_paths = {str(path) for path in EXPECTED_PATHS}
    if seen_paths != expected_paths:
        errors.append(
            "repair event path coverage mismatch: "
            f"missing={sorted(expected_paths - seen_paths)} "
            f"extra={sorted(seen_paths - expected_paths)}"
        )

    report = {
        "status": "passed" if not errors else "failed",
        "expected_rollouts": len(EXPECTED_PATHS),
        "validated_rollouts": len(row_counts),
        "repair_event_rows": len(all_repair_events),
        "event_rows_by_repair_id": row_counts,
        "errors": errors,
    }
    (REPAIR / "validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown = [
        "# Figure 4 missing-annotation repair validation",
        "",
        f"- Status: {report['status']}",
        f"- Expected rollouts: {report['expected_rollouts']}",
        f"- Validated rollouts: {report['validated_rollouts']}",
        f"- Repair event rows: {report['repair_event_rows']}",
        f"- Errors: {len(errors)}",
    ]
    if errors:
        markdown.extend(["", "## Errors", ""] + [f"- {error}" for error in errors])
    (REPAIR / "validation_report.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    if errors:
        raise SystemExit(f"Repair validation failed with {len(errors)} errors")

    all_repair_events.sort(
        key=lambda row: (
            str(row["result_path"]),
            -1 if row["log_index"] is None else int(row["log_index"]),
            str(row["tag_code"]),
            str(row["speaker_agent"]),
        )
    )
    repair_events_path = REPAIR / "repair_event_tags.jsonl"
    write_jsonl(repair_events_path, all_repair_events)
    completion_rows: list[dict[str, object]] = []
    for repair_row in repair_rows:
        output_path = Path(str(repair_row["output_path"]))
        audit_path = Path(str(repair_row["audit_path"]))
        completion_rows.append(
            {
                **repair_row,
                "review_status": "completed_and_validated",
                "event_row_count": row_counts[str(repair_row["repair_id"])],
                "output_sha256": sha256(output_path),
                "audit_sha256": sha256(audit_path),
            }
        )
    completion_path = REPAIR / "completion_ledger.jsonl"
    write_jsonl(completion_path, completion_rows)
    source_events = read_jsonl(SOURCE_EVENTS)
    source_paths = {str(row["result_path"]) for row in source_events}
    overlap = expected_paths & source_paths
    if overlap:
        raise ValueError(f"Refusing merge because repair paths overlap source: {overlap}")
    merged_path = REPAIR / "merged_event_tags_with_missing15_patch.jsonl"
    write_jsonl(merged_path, source_events + all_repair_events)

    provenance = {
        "repair_scope": "fill only the 15 primary Figure 4 paths without retained event rows",
        "source_manifest": str(SOURCE_MANIFEST),
        "source_events": str(SOURCE_EVENTS),
        "codebook": str(CODEBOOK),
        "judge_model": "gpt-5.6-sol",
        "judge_reasoning_effort": "high",
        "annotation_unit": "one transcript per independent Codex subagent",
        "source_event_rows": len(source_events),
        "repair_rollouts": len(EXPECTED_PATHS),
        "repair_event_rows": len(all_repair_events),
        "merged_event_rows": len(source_events) + len(all_repair_events),
        "source_events_sha256": sha256(SOURCE_EVENTS),
        "repair_events_sha256": sha256(repair_events_path),
        "completion_ledger_sha256": sha256(completion_path),
        "merged_events_sha256": sha256(merged_path),
        "source_files_modified": False,
    }
    (REPAIR / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

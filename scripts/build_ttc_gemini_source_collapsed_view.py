#!/usr/bin/env python3
"""Build a non-destructive source-collapsed view of Gemini TTC labels.

The Gemini adjudication prompt permits multiple positive occurrences of one tag
inside one authored source record.  The validator's duplicate warning uses a
coarser identity that permits at most one row per tag and source record.  This
script retains the occurrence-level raw files and writes the coarser view to a
separate directory for analyses that count at most one tag event per source.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = (
    PROJECT_ROOT / "analysis/ttc_gemini_nine_seed_codex_adjudication_20260809"
)
DEFAULT_OUTPUT = DEFAULT_ROOT / "source_collapsed_analysis_20260814"
IDENTITY_FIELDS = (
    "rollout_id",
    "tag_code",
    "source_kind",
    "log_index",
    "interaction_index",
    "speaker_agent",
)
CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
SUMMARY_FIELDS = (
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
OCCURRENCE_IDENTITY_FIELDS = (*IDENTITY_FIELDS, "normalized_quote")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for source_line, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{source_line}: expected a JSON object")
                row["_source_file"] = path.name
                row["_source_line"] = source_line
                rows.append(row)
    return rows


def clean(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in IDENTITY_FIELDS)


def normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def choose_representative(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Max confidence is desirable.  The earliest stable file/line wins among
    # rows at the same confidence.
    best_confidence = max(CONFIDENCE_RANK.get(row.get("confidence"), -1) for row in rows)
    candidates = [
        row
        for row in rows
        if CONFIDENCE_RANK.get(row.get("confidence"), -1) == best_confidence
    ]
    return min(candidates, key=lambda row: (row["_source_file"], row["_source_line"]))


def classify_group(rows: list[dict[str, Any]]) -> str:
    quotes = [normalized_text(row.get("quote", "")) for row in rows]
    if len(set(quotes)) < len(quotes):
        return "repeated_identical_quote"
    if rows[0].get("source_kind") == "formal_outcome":
        return "distinct_formal_outcome_facts"
    return "distinct_verbatim_excerpts_in_one_source"


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def write_counts(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    tag_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    rollout_counts: dict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
    for row in rows:
        tag_key = (row["tag_code"], row["tag_title"])
        tag_counts[tag_key]["event_count"] += 1
        tag_counts[tag_key]["target_event_count"] += int(row["speaker_is_target"])
        tag_counts[tag_key]["baseline_event_count"] += int(row["speaker_is_baseline"])
        summary_key = tuple(row[field] for field in SUMMARY_FIELDS)
        rollout_counts[summary_key]["event_count"] += 1
        rollout_counts[summary_key]["target_event_count"] += int(
            row["speaker_is_target"]
        )
        rollout_counts[summary_key]["baseline_event_count"] += int(
            row["speaker_is_baseline"]
        )

    count_fields = [
        "tag_code",
        "tag_title",
        "event_count",
        "target_event_count",
        "baseline_event_count",
    ]
    with (output_dir / "ttc_codex_event_tag_counts_source_collapsed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=count_fields)
        writer.writeheader()
        for (tag_code, tag_title), counts in sorted(tag_counts.items()):
            writer.writerow(
                {"tag_code": tag_code, "tag_title": tag_title, **counts}
            )

    summary_fields = [
        *SUMMARY_FIELDS,
        "event_count",
        "target_event_count",
        "baseline_event_count",
    ]
    with (output_dir / "ttc_codex_rollout_tag_summary_source_collapsed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for key, counts in sorted(rollout_counts.items()):
            writer.writerow({**dict(zip(SUMMARY_FIELDS, key)), **counts})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    input_dir = args.root / "subagent_outputs"
    input_paths = sorted(input_dir.glob("chunk_*_events.jsonl"))
    if len(input_paths) != 648:
        raise SystemExit(f"expected 648 chunk event files, found {len(input_paths)}")
    if input_dir.resolve() == args.output_dir.resolve():
        raise SystemExit("output directory must differ from the raw input directory")

    raw_rows = read_rows(input_paths)
    occurrence_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        occurrence_groups[
            (*identity(row), normalized_text(row.get("quote", "")))
        ].append(row)
        groups[identity(row)].append(row)

    occurrence_rows = [
        clean(choose_representative(group)) for group in occurrence_groups.values()
    ]
    repeated_occurrence_groups = [
        group for group in occurrence_groups.values() if len(group) > 1
    ]
    collapsed_rows: list[dict[str, Any]] = []
    duplicate_records: list[dict[str, Any]] = []
    group_classes: Counter[str] = Counter()
    varying_fields: Counter[str] = Counter()
    multiplicities: Counter[int] = Counter()
    groups_by_seed: Counter[int] = Counter()
    extras_by_seed: Counter[int] = Counter()
    groups_by_source: Counter[str] = Counter()
    extras_by_source: Counter[str] = Counter()
    origin_stats: dict[str, Counter[str]] = defaultdict(Counter)
    origin_metadata: dict[str, dict[str, Any]] = {}
    for key, group in groups.items():
        selected = choose_representative(group)
        collapsed_rows.append(clean(selected))
        if len(group) == 1:
            continue
        multiplicities[len(group)] += 1
        classification = classify_group(group)
        group_classes[classification] += 1
        seed = int(selected["seed"])
        source_kind = str(selected["source_kind"])
        groups_by_seed[seed] += 1
        extras_by_seed[seed] += len(group) - 1
        groups_by_source[source_kind] += 1
        extras_by_source[source_kind] += len(group) - 1
        source_file = str(selected["_source_file"])
        origin_stats[source_file]["duplicate_identity_groups"] += 1
        origin_stats[source_file]["extra_rows"] += len(group) - 1
        origin_stats[source_file][classification] += 1
        origin_metadata[source_file] = {
            "chunk_file": source_file,
            "rollout_id": selected["rollout_id"],
            "seed": selected["seed"],
            "source_config_id": selected["source_config_id"],
            "level": selected["level"],
        }
        field_names = sorted(clean(group[0]))
        differences = [
            field
            for field in field_names
            if len(
                {
                    json.dumps(row.get(field), sort_keys=True, ensure_ascii=False)
                    for row in group
                }
            )
            > 1
        ]
        varying_fields.update(differences)
        duplicate_records.append(
            {
                "identity": dict(zip(IDENTITY_FIELDS, key)),
                "classification": classification,
                "raw_row_count": len(group),
                "extra_row_count": len(group) - 1,
                "varying_fields": differences,
                "retained_raw_reference": {
                    "file": selected["_source_file"],
                    "line": selected["_source_line"],
                },
                "raw_candidates": [
                    {
                        "file": row["_source_file"],
                        "line": row["_source_line"],
                        "quote": row.get("quote"),
                        "rationale": row.get("rationale"),
                        "confidence": row.get("confidence"),
                    }
                    for row in group
                ],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    occurrence_path = (
        args.output_dir / "ttc_codex_event_tags_occurrence_dedup.jsonl"
    )
    event_path = args.output_dir / "ttc_codex_event_tags_source_collapsed.jsonl"
    duplicate_path = args.output_dir / "duplicate_event_groups.jsonl"
    write_jsonl(occurrence_path, occurrence_rows)
    write_jsonl(event_path, collapsed_rows)
    write_jsonl(duplicate_path, duplicate_records)
    write_counts(args.output_dir, collapsed_rows)
    origin_fields = [
        "chunk_file",
        "rollout_id",
        "seed",
        "source_config_id",
        "level",
        "duplicate_identity_groups",
        "extra_rows",
        "distinct_verbatim_excerpts_in_one_source",
        "distinct_formal_outcome_facts",
        "repeated_identical_quote",
    ]
    with (args.output_dir / "duplicate_origin_by_chunk.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=origin_fields)
        writer.writeheader()
        for source_file in sorted(origin_stats):
            writer.writerow(
                {**origin_metadata[source_file], **origin_stats[source_file]}
            )

    input_digest = hashlib.sha256()
    for path in input_paths:
        input_digest.update(path.name.encode("utf-8"))
        input_digest.update(bytes.fromhex(sha256(path)))
    report = {
        "identity_fields": list(IDENTITY_FIELDS),
        "interpretation": (
            "One retained row per tag, authored source record, and speaker. "
            "Raw occurrence-level annotations remain unchanged."
        ),
        "representative_rule": (
            "Highest confidence, then earliest input file and source line. "
            "Representative choice cannot change collapsed event counts."
        ),
        "input_chunk_files": len(input_paths),
        "input_rows": len(raw_rows),
        "occurrence_identity_fields": list(OCCURRENCE_IDENTITY_FIELDS),
        "occurrence_deduplicated_rows": len(occurrence_rows),
        "repeated_occurrence_groups": len(repeated_occurrence_groups),
        "repeated_occurrence_extra_rows": sum(
            len(group) - 1 for group in repeated_occurrence_groups
        ),
        "output_rows": len(collapsed_rows),
        "duplicate_identity_groups": len(duplicate_records),
        "extra_rows_collapsed": len(raw_rows) - len(collapsed_rows),
        "multiplicity_distribution": {
            str(key): value for key, value in sorted(multiplicities.items())
        },
        "group_classification": dict(sorted(group_classes.items())),
        "duplicate_groups_by_seed": {
            str(key): value for key, value in sorted(groups_by_seed.items())
        },
        "extra_rows_by_seed": {
            str(key): value for key, value in sorted(extras_by_seed.items())
        },
        "duplicate_groups_by_source_kind": dict(sorted(groups_by_source.items())),
        "extra_rows_by_source_kind": dict(sorted(extras_by_source.items())),
        "groups_with_varying_field": dict(sorted(varying_fields.items())),
        "input_directory_digest": input_digest.hexdigest(),
        "output_sha256": sha256(event_path),
        "occurrence_output_sha256": sha256(occurrence_path),
        "raw_outputs_modified": False,
    }
    report_path = args.output_dir / "normalization_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown = [
        "# Gemini TTC source-collapsed analysis view",
        "",
        f"- Raw chunk files: {report['input_chunk_files']}",
        f"- Raw occurrence rows: {report['input_rows']}",
        f"- Occurrence-deduplicated rows: {report['occurrence_deduplicated_rows']}",
        f"- Repeated occurrence groups: {report['repeated_occurrence_groups']}",
        f"- Source-collapsed rows: {report['output_rows']}",
        f"- Repeated identity groups: {report['duplicate_identity_groups']}",
        f"- Extra rows collapsed: {report['extra_rows_collapsed']}",
        "- Raw outputs modified: no",
        "",
        "## Group origin",
        "",
        *(
            f"- {name}: {count} groups"
            for name, count in sorted(group_classes.items())
        ),
        "",
        "## Identity and selection",
        "",
        f"- Identity fields: {', '.join(IDENTITY_FIELDS)}",
        f"- Occurrence identity adds: normalized_quote",
        f"- Representative rule: {report['representative_rule']}",
        "- All candidate quotes, rationales, confidences, and raw locations are in duplicate_event_groups.jsonl.",
    ]
    (args.output_dir / "normalization_report.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    print(
        f"input_rows={len(raw_rows)} output_rows={len(collapsed_rows)} "
        f"duplicate_groups={len(duplicate_records)} "
        f"extra_rows={len(raw_rows) - len(collapsed_rows)}"
    )


if __name__ == "__main__":
    main()

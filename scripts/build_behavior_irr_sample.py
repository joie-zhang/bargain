#!/usr/bin/env python3
"""Build a reproducible, stratified sample for semantic label review.

The review unit is one behavior tag applied to one target-authored source
record. Public discussion records come from ``conversation_logs``. Private
thinking, proposal, voting, and reflection records come from
``agent_authored_interactions``. Discussion interactions are excluded because
they duplicate the public transcript.

Structural tags are excluded from this turn-level study. They require a
separate rollout-outcome review.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis" / "behavior_annotation_irr_review_20260814"
DEFAULT_SELECTED23 = (
    PROJECT_ROOT
    / "analysis"
    / "ttc_claude_seed_qualitative_adjudication_20260728"
    / "paper_tag_subset_23.json"
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    manifest_path: Path
    event_path: Path


DEFAULT_DATASETS = (
    DatasetSpec(
        "ttc_seed42",
        PROJECT_ROOT
        / "analysis"
        / "ttc_llm_strategic_tag_adjudication_20260629"
        / "all_ttc_rollouts_manifest.jsonl",
        PROJECT_ROOT
        / "analysis"
        / "ttc_llm_strategic_tag_adjudication_20260629"
        / "ttc_llm_event_tags.jsonl",
    ),
    DatasetSpec(
        "ttc_gpt5_extra_nine",
        PROJECT_ROOT
        / "analysis"
        / "ttc_gpt5_nine_seed_codex_adjudication_20260809"
        / "all_rollouts_manifest.jsonl",
        PROJECT_ROOT
        / "analysis"
        / "ttc_gpt5_nine_seed_codex_adjudication_20260809"
        / "ttc_gpt5_event_tags.jsonl",
    ),
    DatasetSpec(
        "ttc_claude_extra_nine",
        PROJECT_ROOT
        / "analysis"
        / "ttc_claude_nine_seed_codex_adjudication_20260728"
        / "all_available_rollouts_manifest.jsonl",
        PROJECT_ROOT
        / "analysis"
        / "ttc_claude_nine_seed_codex_adjudication_20260728"
        / "ttc_codex_event_tags.jsonl",
    ),
    DatasetSpec(
        "ttc_gemini_extra_nine",
        PROJECT_ROOT
        / "analysis"
        / "ttc_gemini_nine_seed_codex_adjudication_20260809"
        / "all_available_rollouts_manifest.jsonl",
        PROJECT_ROOT
        / "analysis"
        / "ttc_gemini_nine_seed_codex_adjudication_20260809"
        / "ttc_codex_event_tags.jsonl",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=480)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument(
        "--label-set",
        choices=("selected23", "full50"),
        default="selected23",
    )
    parser.add_argument(
        "--selected23-path",
        type=Path,
        default=DEFAULT_SELECTED23,
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=[dataset.name for dataset in DEFAULT_DATASETS],
        help="Include only this dataset. Repeat to include more than one.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(seed: int, value: str) -> int:
    raw = f"{seed}\0{value}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest(), "big")


def rollout_id(dataset: str, manifest: dict[str, Any]) -> str:
    explicit = manifest.get("rollout_id")
    if explicit:
        return str(explicit)
    return f"{dataset}_config_{int(manifest['config_id']):04d}"


def source_key(source_kind: str, source_index: int) -> str:
    return f"{source_kind}:{source_index}"


def event_source_key(event: dict[str, Any]) -> str | None:
    source_kind = event.get("source_kind")
    if source_kind == "conversation_log" and event.get("log_index") is not None:
        return source_key(source_kind, int(event["log_index"]))
    if source_kind == "interaction" and event.get("interaction_index") is not None:
        return source_key(source_kind, int(event["interaction_index"]))
    return None


def load_codebook(specs: tuple[DatasetSpec, ...], label_set: str, selected23: Path) -> list[dict[str, Any]]:
    codebook_path = specs[0].manifest_path.parent / "llm_tag_codebook.json"
    full = json.loads(codebook_path.read_text(encoding="utf-8"))
    if label_set == "selected23":
        selected = json.loads(selected23.read_text(encoding="utf-8"))
        selected_codes = {row["tag_code"] for row in selected}
        full = [row for row in full if row["tag_code"] in selected_codes]
    if not full:
        raise ValueError("The selected codebook is empty")
    return full


def eligible_tag(tag: dict[str, Any], manifest: dict[str, Any]) -> bool:
    scope = tag.get("scope_hint") or {}
    if scope.get("structural"):
        return False
    minimum_agents = scope.get("min_agents")
    if minimum_agents is not None and int(manifest.get("n_agents") or 0) < int(minimum_agents):
        return False
    games = scope.get("games") or []
    return not games or manifest.get("game_label") in games


def manifest_rows(spec: DatasetSpec) -> list[dict[str, Any]]:
    rows = list(iter_jsonl(spec.manifest_path))
    ids = [rollout_id(spec.name, row) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate rollout identifiers in {spec.manifest_path}")
    return rows


def load_positive_identities(
    specs: tuple[DatasetSpec, ...],
    allowed_codes: set[str],
) -> tuple[set[tuple[str, str, str, str]], dict[str, int]]:
    positives: set[tuple[str, str, str, str]] = set()
    raw_rows = 0
    eligible_rows = 0
    duplicate_rows = 0
    for spec in specs:
        for event in iter_jsonl(spec.event_path):
            raw_rows += 1
            if event.get("tag_code") not in allowed_codes or not event.get("speaker_is_target"):
                continue
            identity_source = event_source_key(event)
            if identity_source is None:
                continue
            manifest_stub = {"config_id": event["config_id"], "rollout_id": event.get("rollout_id")}
            identity = (
                spec.name,
                rollout_id(spec.name, manifest_stub),
                identity_source,
                str(event["tag_code"]),
            )
            eligible_rows += 1
            if identity in positives:
                duplicate_rows += 1
            positives.add(identity)
    return positives, {
        "raw_event_rows": raw_rows,
        "eligible_target_turn_event_rows": eligible_rows,
        "deduplicated_positive_identities": len(positives),
        "duplicate_eligible_event_rows": duplicate_rows,
    }


def iter_target_sources(view: dict[str, Any], target_agent: str) -> Iterator[dict[str, Any]]:
    for row in view.get("conversation_logs") or []:
        if row.get("speaker_agent") != target_agent:
            continue
        yield {
            "source_kind": "conversation_log",
            "source_index": int(row["log_index"]),
            "phase": row.get("phase") or "discussion",
            "round": row.get("round"),
            "discussion_turn": row.get("discussion_turn"),
            "speaker_agent": row.get("speaker_agent"),
            "source_text": str(row.get("content") or ""),
        }
    for row in view.get("agent_authored_interactions") or view.get("target_private_interactions") or []:
        phase = str(row.get("phase") or "")
        agent = row.get("agent_id") or target_agent
        if agent != target_agent or phase in {"discussion", "game_setup"}:
            continue
        yield {
            "source_kind": "interaction",
            "source_index": int(row["interaction_index"]),
            "phase": phase,
            "round": row.get("round"),
            "discussion_turn": row.get("discussion_turn"),
            "speaker_agent": agent,
            "source_text": str(row.get("response") or row.get("content") or ""),
        }


def candidate_rows(
    specs: tuple[DatasetSpec, ...],
    codebook: list[dict[str, Any]],
    positives: set[tuple[str, str, str, str]],
) -> Iterator[dict[str, Any]]:
    for spec in specs:
        for manifest in manifest_rows(spec):
            rid = rollout_id(spec.name, manifest)
            view_path = Path(manifest["rollout_view_path"]).resolve()
            view = json.loads(view_path.read_text(encoding="utf-8"))
            context_hash = sha256_file(view_path)
            target_agent = str(manifest["target_agent"])
            tags = [tag for tag in codebook if eligible_tag(tag, manifest)]
            for source in iter_target_sources(view, target_agent):
                skey = source_key(source["source_kind"], source["source_index"])
                for tag in tags:
                    positive = (spec.name, rid, skey, tag["tag_code"]) in positives
                    item_key = "|".join((spec.name, rid, skey, tag["tag_code"]))
                    item_id = hashlib.sha256(item_key.encode("utf-8")).hexdigest()[:24]
                    stratum = "|".join(
                        (
                            spec.name,
                            str(manifest["family"]),
                            str(manifest["level"]),
                            "positive" if positive else "negative",
                            str(tag["tag_code"]),
                        )
                    )
                    yield {
                        "item_id": item_id,
                        "dataset": spec.name,
                        "rollout_id": rid,
                        "config_id": manifest["config_id"],
                        "seed": manifest.get("seed", 42),
                        "family": manifest["family"],
                        "level": manifest["level"],
                        "level_index": manifest.get("level_index"),
                        "game_label": manifest.get("game_label"),
                        "game_cell": manifest.get("game_cell"),
                        "order": manifest.get("order"),
                        "target_agent": target_agent,
                        "rollout_view_path": str(view_path),
                        "rollout_view_sha256": context_hash,
                        "source_kind": source["source_kind"],
                        "source_index": source["source_index"],
                        "phase": source["phase"],
                        "round": source["round"],
                        "discussion_turn": source["discussion_turn"],
                        "speaker_agent": source["speaker_agent"],
                        "source_text_sha256": hashlib.sha256(
                            source["source_text"].encode("utf-8")
                        ).hexdigest(),
                        "tag_code": tag["tag_code"],
                        "tag_title": tag["tag_title"],
                        "tag_category": tag["category"],
                        "tag_definition": tag["definition"],
                        "machine_positive": positive,
                        "sampling_stratum": stratum,
                    }


def select_balanced_sample(
    specs: tuple[DatasetSpec, ...],
    codebook: list[dict[str, Any]],
    positives: set[tuple[str, str, str, str]],
    sample_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    counts: Counter[str] = Counter()
    for row in candidate_rows(specs, codebook, positives):
        counts[row["sampling_stratum"]] += 1
    if not counts:
        raise ValueError("No eligible label-turn candidates were found")
    target_size = min(sample_size, sum(counts.values()))
    per_stratum_cap = max(1, math.ceil(target_size / len(counts)) + 2)

    # Python's heap is a min-heap. Store negative priorities so each heap keeps
    # the rows with the lowest deterministic hash values.
    heaps: dict[str, list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    for row in candidate_rows(specs, codebook, positives):
        stratum = row["sampling_stratum"]
        priority = stable_hash(seed, row["item_id"])
        entry = (-priority, row["item_id"], row)
        heap = heaps[stratum]
        if len(heap) < per_stratum_cap:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)

    queues = {
        stratum: [entry[2] for entry in sorted(heap, key=lambda value: (-value[0], value[1]))]
        for stratum, heap in heaps.items()
    }
    ordered_strata = sorted(queues, key=lambda value: stable_hash(seed, f"stratum|{value}"))
    sample: list[dict[str, Any]] = []
    position = Counter()
    while len(sample) < target_size:
        added = False
        for stratum in ordered_strata:
            idx = position[stratum]
            if idx >= len(queues[stratum]):
                continue
            sample.append(queues[stratum][idx])
            position[stratum] += 1
            added = True
            if len(sample) == target_size:
                break
        if not added:
            raise RuntimeError("Candidate reservoir was too small for the requested sample")

    selected_counts = Counter(row["sampling_stratum"] for row in sample)
    for row in sample:
        stratum = row["sampling_stratum"]
        row["candidate_count_in_stratum"] = counts[stratum]
        row["sample_count_in_stratum"] = selected_counts[stratum]
        row["sampling_weight"] = counts[stratum] / selected_counts[stratum]
        row["sampling_seed"] = seed
    sample.sort(key=lambda row: stable_hash(seed, f"order|{row['item_id']}"))
    for index, row in enumerate(sample, start=1):
        row["sample_position"] = index
    return sample, counts


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def summarize_sample(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    dimensions = ("dataset", "family", "level", "tag_code", "machine_positive")
    return {
        dimension: dict(sorted(Counter(str(row[dimension]) for row in rows).items()))
        for dimension in dimensions
    }


def main() -> None:
    args = parse_args()
    selected_names = set(args.dataset or [dataset.name for dataset in DEFAULT_DATASETS])
    specs = tuple(dataset for dataset in DEFAULT_DATASETS if dataset.name in selected_names)
    for spec in specs:
        for path in (spec.manifest_path, spec.event_path):
            if not path.exists():
                raise FileNotFoundError(path)
    codebook = load_codebook(specs, args.label_set, args.selected23_path)
    turn_level_codebook = [row for row in codebook if not (row.get("scope_hint") or {}).get("structural")]
    structural_codes = [
        row["tag_code"] for row in codebook if (row.get("scope_hint") or {}).get("structural")
    ]
    possible_codes = {row["tag_code"] for row in turn_level_codebook}
    positives, event_stats = load_positive_identities(specs, possible_codes)
    sample, candidate_counts = select_balanced_sample(
        specs,
        turn_level_codebook,
        positives,
        args.sample_size,
        args.seed,
    )

    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "sampling_manifest.jsonl"
    manifest_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in sample)
    atomic_write(manifest_path, manifest_text)
    provenance = {
        "schema_version": 1,
        "study_unit": "one behavior tag on one target-authored source record",
        "label_set": args.label_set,
        "requested_sample_size": args.sample_size,
        "actual_sample_size": len(sample),
        "sampling_seed": args.seed,
        "sampling_method": (
            "deterministic round-robin across nonempty dataset x family x effort x "
            "machine class x tag strata"
        ),
        "sampling_manifest": str(manifest_path),
        "sampling_manifest_sha256": hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
        "datasets": [
            {
                "name": spec.name,
                "manifest_path": str(spec.manifest_path.resolve()),
                "manifest_sha256": sha256_file(spec.manifest_path),
                "event_path": str(spec.event_path.resolve()),
                "event_sha256": sha256_file(spec.event_path),
            }
            for spec in specs
        ],
        "turn_level_tag_count": len(turn_level_codebook),
        "turn_level_tag_codes": [row["tag_code"] for row in turn_level_codebook],
        "excluded_structural_tag_codes": structural_codes,
        "candidate_count": sum(candidate_counts.values()),
        "nonempty_stratum_count": len(candidate_counts),
        "event_input_stats": event_stats,
        "sample_counts": summarize_sample(sample),
        "analysis_note": (
            "The sample is intentionally class-balanced for semantic validation. "
            "Use sampling_weight for population-level agreement estimates."
        ),
    }
    atomic_write(output_dir / "sampling_provenance.json", json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize TTC LLM strategic tag adjudication outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("analysis/ttc_llm_strategic_tag_adjudication_20260629")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_codebook(path: Path) -> dict[str, dict]:
    return {row["tag_code"]: row for row in json.loads(path.read_text())}


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sort_key(value: str) -> tuple:
    level_order = {"minimal": 0, "low": 1, "medium": 2, "high": 3, "max": 4}
    if value in level_order:
        return (level_order[value], value)
    return (999, value)


def top_table(rows: list[dict], fields: list[str], limit: int) -> str:
    if not rows:
        return "(none)"
    rows = rows[:limit]
    out = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(out)


def collect_audit_themes(audit_dir: Path) -> dict[str, list[str]]:
    themes = {
        "accepted-deal or final-round continuation hallucination": [
            "future rounds after an accepted",
            "post_agreement_future_round_confusion",
            "game_continuation_after_acceptance",
            "final-round future-round hallucination",
        ],
        "budget refresh, rescope, or rule-change misconception": [
            "budget_refresh",
            "budget-refresh",
            "rescope",
            "rule-change",
            "rule changes",
            "budget changes",
        ],
        "payoff/preference perspective mismatch": [
            "payoff-perspective",
            "preference-perspective",
            "vote-reasoning mismatch",
            "preference inconsistencies",
            "perspective mismatch",
        ],
        "transcript/source divergence": [
            "transcript conflicts",
            "truncated or mismatched",
            "public-log vs compact-interaction divergence",
            "source concerns",
            "internal reasoning/vector mismatches",
        ],
        "strategic bluffing or misrepresentation gap": [
            "strategic_preference_misrepresentation",
            "bluffed",
            "misrepresentation",
        ],
        "discount/future-review mechanism gap": [
            "discount_arbitrage",
            "review_clause_trade",
            "verification_ladder_strategy",
            "future milestone",
            "staged future",
        ],
    }
    hits: dict[str, list[str]] = {name: [] for name in themes}
    for path in sorted(audit_dir.glob("chunk_*_audit.md")):
        text = path.read_text(errors="replace").lower()
        for name, patterns in themes.items():
            if any(pattern.lower() in text for pattern in patterns):
                hits[name].append(path.name)
    return hits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--expected-chunks", type=int, default=36)
    parser.add_argument("--expected-rollouts", type=int, default=216)
    args = parser.parse_args()

    root: Path = args.root
    event_path = root / "ttc_llm_event_tags.jsonl"
    manifest_path = root / "all_ttc_rollouts_manifest.jsonl"
    codebook_path = root / "llm_tag_codebook.json"
    output_dir = root / "final_summaries"
    output_dir.mkdir(exist_ok=True)

    events = read_jsonl(event_path)
    manifest = read_jsonl(manifest_path)
    codebook = read_codebook(codebook_path)

    all_configs = {row["config_id"] for row in manifest}
    event_configs = {row["config_id"] for row in events}
    chunks = {row["chunk_id"] for row in events}

    by_tag: dict[str, Counter] = defaultdict(Counter)
    configs_by_tag: dict[str, set[int]] = defaultdict(set)
    for row in events:
        tag = row["tag_code"]
        by_tag[tag]["events"] += 1
        by_tag[tag][f"{row.get('speaker_role')}_events"] += 1
        configs_by_tag[tag].add(row["config_id"])

    tag_rows: list[dict] = []
    for tag, meta in codebook.items():
        counts = by_tag[tag]
        tag_rows.append(
            {
                "tag_code": tag,
                "tag_title": meta["tag_title"],
                "category": meta.get("category", ""),
                "event_count": counts["events"],
                "rollout_count": len(configs_by_tag[tag]),
                "target_event_count": counts["target_events"],
                "baseline_event_count": counts["baseline_events"],
            }
        )
    tag_rows.sort(key=lambda r: (-int(r["event_count"]), -int(r["rollout_count"]), r["tag_code"]))
    write_csv(
        output_dir / "tag_frequency_summary.csv",
        tag_rows,
        [
            "tag_code",
            "tag_title",
            "category",
            "event_count",
            "rollout_count",
            "target_event_count",
            "baseline_event_count",
        ],
    )

    cell_events: dict[tuple[str, str, str, str], Counter] = defaultdict(Counter)
    cell_rollouts: dict[tuple[str, str, str, str], set[int]] = defaultdict(set)
    for row in events:
        key = (row["tag_code"], row["family"], row["level"], row["game_label"])
        cell_events[key]["event_count"] += 1
        cell_rollouts[key].add(row["config_id"])

    cell_rows = []
    for (tag, family, level, game), counts in sorted(
        cell_events.items(), key=lambda x: (x[0][0], x[0][1], sort_key(x[0][2]), x[0][3])
    ):
        cell_rows.append(
            {
                "tag_code": tag,
                "tag_title": codebook.get(tag, {}).get("tag_title", ""),
                "family": family,
                "level": level,
                "game_label": game,
                "event_count": counts["event_count"],
                "rollout_count": len(cell_rollouts[(tag, family, level, game)]),
            }
        )
    write_csv(
        output_dir / "tag_counts_by_family_level_game.csv",
        cell_rows,
        ["tag_code", "tag_title", "family", "level", "game_label", "event_count", "rollout_count"],
    )

    group_rows = []
    for group_name in ["family", "level", "game_label", "speaker_role", "evidence_type", "confidence"]:
        counter = Counter(row.get(group_name) for row in events)
        for value, count in sorted(counter.items(), key=lambda item: (-item[1], str(item[0]))):
            group_rows.append({"group": group_name, "value": value, "event_count": count})
    write_csv(output_dir / "event_counts_by_group.csv", group_rows, ["group", "value", "event_count"])

    role_counter = Counter(row["speaker_role"] for row in events)
    source_counter = Counter(row["source_kind"] for row in events)
    evidence_counter = Counter(row["evidence_type"] for row in events)
    confidence_counter = Counter(row["confidence"] for row in events)
    zero_tags = [row for row in tag_rows if int(row["event_count"]) == 0]
    audit_themes = collect_audit_themes(root / "subagent_outputs")

    top_examples = []
    for tag_row in tag_rows[:8]:
        tag = tag_row["tag_code"]
        example = next((row for row in events if row["tag_code"] == tag), None)
        if not example:
            continue
        quote = " ".join(str(example["quote"]).split())
        if len(quote) > 180:
            quote = quote[:177] + "..."
        top_examples.append(
            {
                "tag_code": tag,
                "config": example["config_id"],
                "role": example["speaker_role"],
                "round": example["round"],
                "quote": quote.replace("|", "\\|"),
            }
        )

    report = []
    report.append("# TTC LLM Strategic Tagging Final Summary")
    report.append("")
    report.append("## Coverage")
    report.append("")
    report.append(f"- Completed chunks represented in aggregate: {len(chunks)} / {args.expected_chunks}")
    report.append(f"- Rollouts with adjudicated events: {len(event_configs)} / {args.expected_rollouts}")
    report.append(f"- Rollouts in source manifest: {len(all_configs)}")
    report.append(f"- Event rows: {len(events)}")
    report.append(f"- Missing manifest configs with no event rows: {len(all_configs - event_configs)}")
    report.append("")
    report.append("## Output Files")
    report.append("")
    report.append(f"- Event-level tags: `{event_path}`")
    report.append(f"- Rollout/tag summary: `{root / 'ttc_llm_rollout_tag_summary.csv'}`")
    report.append(f"- Tag counts: `{root / 'ttc_llm_event_tag_counts_by_tag.csv'}`")
    report.append(f"- Validation report: `{root / 'ttc_llm_validation_report.md'}`")
    report.append(f"- Final tag summary CSV: `{output_dir / 'tag_frequency_summary.csv'}`")
    report.append(f"- Family/level/game CSV: `{output_dir / 'tag_counts_by_family_level_game.csv'}`")
    report.append(f"- Group-count CSV: `{output_dir / 'event_counts_by_group.csv'}`")
    report.append("")
    report.append("## Distribution")
    report.append("")
    report.append("- Speaker roles: " + ", ".join(f"{k}={v}" for k, v in role_counter.most_common()))
    report.append("- Sources: " + ", ".join(f"{k}={v}" for k, v in source_counter.most_common()))
    report.append("- Evidence types: " + ", ".join(f"{k}={v}" for k, v in evidence_counter.most_common()))
    report.append("- Confidence: " + ", ".join(f"{k}={v}" for k, v in confidence_counter.most_common()))
    report.append("")
    report.append("## Top Tags")
    report.append("")
    report.append(
        top_table(
            tag_rows,
            [
                "tag_code",
                "tag_title",
                "event_count",
                "rollout_count",
                "target_event_count",
                "baseline_event_count",
            ],
            20,
        )
    )
    report.append("")
    report.append("## Top-Tag Example Pointers")
    report.append("")
    report.append(top_table(top_examples, ["tag_code", "config", "role", "round", "quote"], 8))
    report.append("")
    report.append("## Zero-Event Tags")
    report.append("")
    if zero_tags:
        report.append(", ".join(f"`{row['tag_code']}`" for row in zero_tags))
    else:
        report.append("None.")
    report.append("")
    report.append("## Audit Caveat Themes")
    report.append("")
    for name, files in audit_themes.items():
        if files:
            preview = ", ".join(files[:8])
            suffix = "" if len(files) <= 8 else f", plus {len(files) - 8} more"
            report.append(f"- {name}: {len(files)} chunk audits ({preview}{suffix})")
    report.append("")
    report.append("These caveats are not validator failures. They are places where workers repeatedly noted TTC-specific behavior that may deserve a future tag or a separate data-quality flag.")
    report.append("")

    (root / "ttc_llm_strategic_tagging_final_report.md").write_text("\n".join(report))
    print(root / "ttc_llm_strategic_tagging_final_report.md")
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

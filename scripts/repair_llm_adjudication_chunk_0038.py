#!/usr/bin/env python3
"""Repair chunk_0038 LLM adjudication rows to the fixed 50-tag schema."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "analysis/llm_strategic_tag_adjudication_20260628"
CODEBOOK = OUT_DIR / "llm_tag_codebook.json"
EVENTS = OUT_DIR / "subagent_outputs/chunk_0038_events.jsonl"
AUDIT = OUT_DIR / "subagent_outputs/chunk_0038_audit.md"


def main() -> None:
    tag_titles = {row["tag_code"]: row["tag_title"] for row in json.loads(CODEBOOK.read_text())}
    remapped = {
        "multi_round_staging": "formal_artifact_frameworking",
        "consensus_checkpoint_invocation": "lock_in_confirmation_pressure",
    }
    remap_counts = {key: 0 for key in remapped}
    final_outcome_fixes = 0
    title_fixes = 0
    rows = []
    with EVENTS.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            old_code = row.get("tag_code")
            if old_code in remapped:
                new_code = remapped[old_code]
                row["tag_code"] = new_code
                row["tag_title"] = tag_titles[new_code]
                row["rationale"] = (
                    row.get("rationale", "").rstrip(".")
                    + f". Main-agent repair: mapped provisional `{old_code}` to existing codebook tag `{new_code}`."
                )
                remap_counts[old_code] += 1
            elif row.get("tag_code") in tag_titles and row.get("tag_title") != tag_titles[row["tag_code"]]:
                row["tag_title"] = tag_titles[row["tag_code"]]
                title_fixes += 1

            if row.get("phase") == "vote_tabulation":
                row["phase"] = "final_outcome"
                row["speaker_agent"] = None
                row["speaker_model"] = None
                row["speaker_elo"] = None
                row["speaker_role"] = None
                row["speaker_order"] = None
                final_outcome_fixes += 1

            if row.get("tag_code") in tag_titles:
                row["tag_title"] = tag_titles[row["tag_code"]]
            rows.append(row)

    with EVENTS.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    with AUDIT.open("a", encoding="utf-8") as handle:
        handle.write("\n## Main-Agent Repair\n")
        handle.write(
            f"- Remapped provisional `multi_round_staging` rows to `formal_artifact_frameworking`: "
            f"{remap_counts['multi_round_staging']}.\n"
        )
        handle.write(
            f"- Remapped provisional `consensus_checkpoint_invocation` rows to "
            f"`lock_in_confirmation_pressure`: {remap_counts['consensus_checkpoint_invocation']}.\n"
        )
        handle.write(
            f"- Normalized `vote_tabulation` formal rows to `phase: final_outcome` with null speaker "
            f"metadata: {final_outcome_fixes}.\n"
        )
        handle.write(f"- Normalized tag titles to the codebook for all rows; direct title fixes: {title_fixes}.\n")

    print(
        "remapped="
        f"{remap_counts} final_outcome_fixes={final_outcome_fixes} "
        f"title_fixes={title_fixes} rows={len(rows)}"
    )


if __name__ == "__main__":
    main()

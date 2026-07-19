#!/usr/bin/env python3
"""Small targeted repairs for LLM adjudication chunk outputs."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "analysis/llm_strategic_tag_adjudication_20260628"


def main() -> None:
    path = OUT_DIR / "subagent_outputs/chunk_0037_events.jsonl"
    rows = []
    removed = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("tag_code") == "trade_sequence_dependency":
                removed.append(row)
            else:
                rows.append(row)

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    audit = OUT_DIR / "subagent_outputs/chunk_0037_audit.md"
    with audit.open("a", encoding="utf-8") as handle:
        handle.write("\n## Main-Agent Repair\n")
        handle.write(
            f"- Removed {len(removed)} rows labeled `trade_sequence_dependency` because that tag is "
            "not in the current 50-tag codebook. The underlying behavior is retained here as a "
            "possible future-tag idea, not as an adjudicated positive row.\n"
        )
    print(f"removed={len(removed)} kept={len(rows)} path={path}")


if __name__ == "__main__":
    main()

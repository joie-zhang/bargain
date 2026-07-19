#!/usr/bin/env python3
"""Report progress for TTC LLM strategic tag adjudication chunks."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "analysis/ttc_llm_strategic_tag_adjudication_20260629"
CHUNK_INDEX = OUT_DIR / "chunk_index.jsonl"
OUTPUT_DIR = OUT_DIR / "subagent_outputs"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def main() -> None:
    chunks = read_jsonl(CHUNK_INDEX)
    completed = []
    missing = []
    event_rows = 0
    completed_rollouts = 0
    for chunk in chunks:
        event_path = Path(chunk["output_path"])
        audit_path = Path(chunk["audit_path"])
        if event_path.exists() and audit_path.exists():
            completed.append(chunk)
            event_rows += count_jsonl(event_path)
            completed_rollouts += chunk["rollout_count"]
        else:
            missing.append(chunk)

    print(f"completed_chunks={len(completed)} / {len(chunks)}")
    print(f"completed_rollouts={completed_rollouts} / {sum(c['rollout_count'] for c in chunks)}")
    print(f"event_rows={event_rows}")
    if completed:
        print("last_completed=" + ", ".join(c["chunk_id"] for c in completed[-10:]))
    if missing:
        print("next_missing=" + ", ".join(c["chunk_id"] for c in missing[:12]))


if __name__ == "__main__":
    main()

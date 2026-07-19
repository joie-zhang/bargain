#!/usr/bin/env python3
"""Report progress for LLM strategic tag adjudication chunks."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "analysis/llm_strategic_tag_adjudication_20260628"
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Adjudication output directory containing chunk_index.jsonl and subagent_outputs/.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    chunk_index = out_dir / "chunk_index.jsonl"
    chunks = read_jsonl(chunk_index)
    completed = []
    missing = []
    total_events = 0
    for chunk in chunks:
        event_path = Path(chunk["output_path"])
        audit_path = Path(chunk["audit_path"])
        if event_path.exists() and audit_path.exists():
            count = count_jsonl(event_path)
            completed.append((chunk["chunk_id"], count))
            total_events += count
        else:
            missing.append(chunk["chunk_id"])

    print(f"completed_chunks={len(completed)} / {len(chunks)}")
    print(f"completed_rollouts={sum(read_jsonl(Path(c['manifest_path'])).__len__() for c in chunks if c['chunk_id'] in {x[0] for x in completed})}")
    print(f"event_rows={total_events}")
    if completed:
        print("last_completed=" + ", ".join(chunk_id for chunk_id, _ in completed[-10:]))
    if missing:
        print("next_missing=" + ", ".join(missing[:20]))


if __name__ == "__main__":
    main()

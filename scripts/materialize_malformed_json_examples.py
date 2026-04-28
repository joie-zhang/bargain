#!/usr/bin/env python3
"""Convert malformed_json_examples.jsonl into a normal aggregate JSON file."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def resolve_jsonl_path(path: Path) -> Path:
    """Accept either a JSONL file path or a results root/run directory."""
    if path.is_dir():
        return path / "monitoring" / "malformed_json_examples.jsonl"
    return path


def materialize_jsonl(jsonl_path: Path, output_path: Path | None = None) -> Tuple[Path, Dict[str, Any]]:
    """Read JSONL examples and write a JSON aggregate next to the JSONL by default."""
    jsonl_path = resolve_jsonl_path(jsonl_path).resolve()
    if output_path is None:
        output_path = jsonl_path.with_suffix(".json")
    else:
        output_path = output_path.resolve()

    examples: List[Dict[str, Any]] = []
    malformed_lines: List[Dict[str, Any]] = []
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    malformed_lines.append(
                        {
                            "line_number": line_number,
                            "error": str(exc),
                            "line_preview": stripped[:500],
                        }
                    )
                    continue
                if isinstance(payload, dict):
                    examples.append(payload)
                else:
                    malformed_lines.append(
                        {
                            "line_number": line_number,
                            "error": "JSONL entry was not an object",
                            "line_preview": stripped[:500],
                        }
                    )

    output = {
        "schema_version": 1,
        "generated_at": dt.datetime.now().isoformat(),
        "source_jsonl": str(jsonl_path),
        "count": len(examples),
        "malformed_line_count": len(malformed_lines),
        "malformed_lines": malformed_lines,
        "examples": examples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, default=str, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path, output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        help=(
            "Path to malformed_json_examples.jsonl, or a results/run root containing "
            "monitoring/malformed_json_examples.jsonl."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to malformed_json_examples.json next to the JSONL.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path, output = materialize_jsonl(args.path, args.output)
    print(f"Wrote {output['count']} malformed JSON example(s) to {output_path}")
    if output["malformed_line_count"]:
        print(f"Skipped {output['malformed_line_count']} malformed JSONL line(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

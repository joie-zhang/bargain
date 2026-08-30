#!/usr/bin/env python3
"""
=============================================================================
Unreferenced ICML AIWILD Graphics Manifest Builder
=============================================================================

Identifies image files in overleaf/icml_aiwild_template/graphics/ that the
compiled ICML AIWILD paper does NOT reference, so they can be triaged for
deletion before the arXiv upload.

A file counts as REFERENCED if either source says so:

  1. An \\includegraphics{...} in a .tex file that the document actually
     inputs (icml_aiwild_2026.tex -> abstract, body -> 1_intro..5_conclusions,
     appendix). LaTeX comments are stripped first. Standalone .tex files that
     the document never inputs -- example_paper.tex, blah.tex -- are ignored.
  2. A path recorded in the compiled icml_aiwild_2026.log.

The union is deliberate and conservative. LaTeX wraps long paths across lines
in its log, so the log alone undercounts; the tex parse alone misses graphics
pulled in by macros. A file must be absent from BOTH to be called
unreferenced, so the delete candidate list never over-reaches.

Usage:
    python scripts/build_unreferenced_graphics_manifest.py
    python scripts/build_unreferenced_graphics_manifest.py --check

What it creates:
    docs/reproducibility/unreferenced_graphics_manifest.json
        {generated_at, template_dir, counts, referenced[], unreferenced[]}

Options:
    --check   Recompute and diff against the existing manifest. Exits 1 if
              they disagree. Use before acting on the triage decisions.

Dependencies:
    Standard library only.

=============================================================================
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = PROJECT_ROOT / "overleaf" / "icml_aiwild_template"
GRAPHICS_DIR = TEMPLATE_DIR / "graphics"
LOG_FILE = TEMPLATE_DIR / "icml_aiwild_2026.log"
MANIFEST = PROJECT_ROOT / "docs" / "reproducibility" / "unreferenced_graphics_manifest.json"

# .tex files the compiled document actually pulls in.
INPUT_TEX = [
    "icml_aiwild_2026.tex",
    "body.tex",
    "abstract.tex",
    "appendix.tex",
    "1_intro.tex",
    "2_background.tex",
    "3_approach.tex",
    "4_analysis.tex",
    "5_conclusions.tex",
]

IMAGE_SUFFIXES = (".png", ".pdf", ".jpg", ".jpeg")
INCLUDEGRAPHICS = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}")
LOG_PATHS = re.compile(r"[^<>{}\s]*graphics/[^<>{}\s]+?\.(?:png|pdf|jpe?g)", re.IGNORECASE)


def strip_tex_comments(text: str) -> str:
    """Drop % comments, honoring \\% escapes. Keeps line structure."""
    out = []
    for line in text.splitlines():
        cleaned, i = [], 0
        while i < len(line):
            ch = line[i]
            if ch == "\\" and i + 1 < len(line):
                cleaned.append(line[i : i + 2])
                i += 2
                continue
            if ch == "%":
                break
            cleaned.append(ch)
            i += 1
        out.append("".join(cleaned))
    return "\n".join(out)


def normalize(raw: str) -> list[str]:
    """Map an \\includegraphics argument to candidate graphics-relative paths.

    Handles the shared-Overleaf-root form ('icml_aiwild_template/graphics/x.png'),
    leading './', and extensionless references that LaTeX resolves by suffix search.
    """
    p = raw.strip().replace("\\", "/")
    for prefix in ("./", "icml_aiwild_template/"):
        while p.startswith(prefix):
            p = p[len(prefix) :]
    if not p.startswith("graphics/"):
        idx = p.find("graphics/")
        if idx == -1:
            return []
        p = p[idx:]
    if p.lower().endswith(IMAGE_SUFFIXES):
        return [p]
    return [p + suffix for suffix in IMAGE_SUFFIXES]


def referenced_from_tex() -> set[str]:
    found: set[str] = set()
    for name in INPUT_TEX:
        path = TEMPLATE_DIR / name
        if not path.exists():
            continue
        body = strip_tex_comments(path.read_text(encoding="utf-8", errors="replace"))
        for raw in INCLUDEGRAPHICS.findall(body):
            found.update(normalize(raw))
    return found


def referenced_from_log() -> set[str]:
    if not LOG_FILE.exists():
        return set()
    text = LOG_FILE.read_text(encoding="utf-8", errors="replace")
    # LaTeX hard-wraps log lines at ~79 chars mid-path; rejoin before matching.
    unwrapped = re.sub(r"\n(?! )", "", text)
    found: set[str] = set()
    for hit in LOG_PATHS.findall(unwrapped) + LOG_PATHS.findall(text):
        found.update(normalize(hit))
    return found


def all_graphics() -> list[str]:
    return sorted(
        str(p.relative_to(TEMPLATE_DIR)).replace("\\", "/")
        for p in GRAPHICS_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def build() -> dict:
    from_tex = referenced_from_tex()
    from_log = referenced_from_log()
    referenced = from_tex | from_log
    present = all_graphics()

    unreferenced = [p for p in present if p not in referenced]
    used = [p for p in present if p in referenced]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "template_dir": str(TEMPLATE_DIR.relative_to(PROJECT_ROOT)),
        "counts": {
            "present": len(present),
            "referenced": len(used),
            "unreferenced": len(unreferenced),
            "referenced_via_tex": len(from_tex & set(present)),
            "referenced_via_log": len(from_log & set(present)),
        },
        "referenced": used,
        "unreferenced": [
            {
                "path": p,
                "bytes": (TEMPLATE_DIR / p).stat().st_size,
                "dir": str(Path(p).parent),
            }
            for p in unreferenced
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="diff against existing manifest")
    args = ap.parse_args()

    data = build()
    c = data["counts"]
    print(
        f"present={c['present']}  referenced={c['referenced']} "
        f"(tex={c['referenced_via_tex']}, log={c['referenced_via_log']})  "
        f"unreferenced={c['unreferenced']}"
    )

    if args.check:
        if not MANIFEST.exists():
            print(f"FAIL: {MANIFEST} does not exist")
            return 1
        old = json.loads(MANIFEST.read_text())
        old_set = {e["path"] for e in old["unreferenced"]}
        new_set = {e["path"] for e in data["unreferenced"]}
        if old_set == new_set:
            print("OK: manifest matches the current tree")
            return 0
        for p in sorted(new_set - old_set):
            print(f"  + {p}  (newly unreferenced)")
        for p in sorted(old_set - new_set):
            print(f"  - {p}  (now referenced or gone)")
        return 1

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(data, indent=2) + "\n")
    total_mb = sum(e["bytes"] for e in data["unreferenced"]) / 1024 / 1024
    print(f"wrote {MANIFEST.relative_to(PROJECT_ROOT)}  ({total_mb:.1f} MB unreferenced)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

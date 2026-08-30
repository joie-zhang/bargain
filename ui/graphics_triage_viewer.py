#!/usr/bin/env python3
"""
=============================================================================
Unreferenced Graphics Triage Viewer
=============================================================================

Streamlit UI for deciding keep-or-delete on the image files in
overleaf/icml_aiwild_template/graphics/ that the compiled ICML AIWILD paper
does not reference. It shows one image at a time with the context needed to
judge it, and records every decision immediately so the session is resumable.

It operates on exactly the file list in
docs/reproducibility/unreferenced_graphics_manifest.json -- nothing else. Files
the paper references are never shown and can never be staged. Regenerate that
manifest with scripts/build_unreferenced_graphics_manifest.py whenever the
paper's figures change.

This viewer NEVER deletes or moves anything. It writes decisions to a CSV and,
on request, emits a staging shell script for you to review and run yourself.

Usage:
    ./ui/run_graphics_triage.sh
    ./ui/run_graphics_triage.sh --port 8080

    # or directly
    streamlit run ui/graphics_triage_viewer.py

What it creates:
    docs/reproducibility/graphics_triage_decisions.csv   # path,decision,note,decided_at
    docs/reproducibility/stage_unreferenced_graphics.sh  # written by the export button

Decision aid: for each unreferenced file the sidebar lists the most
similar-named files the paper DOES reference. Most unreferenced assets are
superseded variants of a live figure, so seeing the surviving sibling is
usually enough to decide.

Dependencies:
    streamlit, and a manifest built by
    scripts/build_unreferenced_graphics_manifest.py

=============================================================================
"""

from __future__ import annotations

import csv
import difflib
import json
from datetime import datetime
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = PROJECT_ROOT / "overleaf" / "icml_aiwild_template"
MANIFEST = PROJECT_ROOT / "docs" / "reproducibility" / "unreferenced_graphics_manifest.json"
DECISIONS = PROJECT_ROOT / "docs" / "reproducibility" / "graphics_triage_decisions.csv"
STAGE_SCRIPT = PROJECT_ROOT / "docs" / "reproducibility" / "stage_unreferenced_graphics.sh"
STAGE_DIR = "experiments/results/TO_DELETE_20260809/unreferenced_graphics"

KEEP, DELETE, UNDECIDED = "keep", "delete", "undecided"


# --------------------------------------------------------------------------- data


@st.cache_data
def load_manifest() -> tuple[list[dict], list[str]]:
    if not MANIFEST.exists():
        st.error(
            f"Manifest not found: {MANIFEST.relative_to(PROJECT_ROOT)}\n\n"
            "Build it first:\n\n"
            "    python scripts/build_unreferenced_graphics_manifest.py"
        )
        st.stop()
    data = json.loads(MANIFEST.read_text())
    return data["unreferenced"], data["referenced"]


def load_decisions() -> dict[str, dict]:
    if not DECISIONS.exists():
        return {}
    with DECISIONS.open(newline="", encoding="utf-8") as fh:
        return {row["path"]: row for row in csv.DictReader(fh)}


def save_decisions(decisions: dict[str, dict]) -> None:
    DECISIONS.parent.mkdir(parents=True, exist_ok=True)
    with DECISIONS.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "decision", "note", "decided_at"])
        writer.writeheader()
        for path in sorted(decisions):
            writer.writerow(decisions[path])


def record(path: str, decision: str, note: str = "") -> None:
    st.session_state.decisions[path] = {
        "path": path,
        "decision": decision,
        "note": note,
        "decided_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_decisions(st.session_state.decisions)


def similar_referenced(path: str, referenced: list[str], n: int = 4) -> list[str]:
    """Referenced files whose names most resemble this one, same directory first."""
    stem = Path(path).stem
    same_dir = [r for r in referenced if Path(r).parent == Path(path).parent]
    pool = same_dir or referenced
    ranked = sorted(
        pool, key=lambda r: difflib.SequenceMatcher(None, stem, Path(r).stem).ratio(), reverse=True
    )
    return ranked[:n]


# --------------------------------------------------------------------------- ui

st.set_page_config(page_title="Graphics Triage", page_icon="🖼️", layout="wide")

unreferenced, referenced = load_manifest()
if "decisions" not in st.session_state:
    st.session_state.decisions = load_decisions()
if "idx" not in st.session_state:
    st.session_state.idx = 0

decisions = st.session_state.decisions

# ---- sidebar: scope, filters, progress
with st.sidebar:
    st.header("Scope")
    total_mb = sum(e["bytes"] for e in unreferenced) / 1024 / 1024
    st.caption(
        f"**{len(unreferenced)} unreferenced files** ({total_mb:.1f} MB) in "
        f"`graphics/`. The {len(referenced)} files the paper references are "
        "excluded from this list and cannot be staged here."
    )

    st.header("Filter")
    dirs = sorted({e["dir"] for e in unreferenced})
    chosen_dirs = st.multiselect("Directory", dirs, default=[])
    only_undecided = st.checkbox("Only undecided", value=True)

    view = [
        e
        for e in unreferenced
        if (not chosen_dirs or e["dir"] in chosen_dirs)
        and (
            not only_undecided
            or decisions.get(e["path"], {}).get("decision", UNDECIDED) == UNDECIDED
        )
    ]

    st.header("Progress")
    n_keep = sum(1 for d in decisions.values() if d["decision"] == KEEP)
    n_del = sum(1 for d in decisions.values() if d["decision"] == DELETE)
    done = n_keep + n_del
    st.progress(done / max(len(unreferenced), 1))
    c1, c2, c3 = st.columns(3)
    c1.metric("Keep", n_keep)
    c2.metric("Delete", n_del)
    c3.metric("Left", len(unreferenced) - done)

    by_path = {e["path"]: e for e in unreferenced}
    freed = sum(by_path[p]["bytes"] for p, d in decisions.items()
                if d["decision"] == DELETE and p in by_path) / 1024 / 1024
    st.caption(f"Marked for deletion: **{freed:.1f} MB**")

    st.divider()
    if st.button("Export staging script", width='stretch'):
        to_delete = sorted(
            p for p, d in decisions.items() if d["decision"] == DELETE and p in by_path
        )
        lines = [
            "#!/usr/bin/env bash",
            "# Stage triaged unreferenced ICML AIWILD graphics.",
            "# Generated by ui/graphics_triage_viewer.py on "
            f"{datetime.now().isoformat(timespec='seconds')}.",
            "# MOVES files into a staging directory. It deletes nothing.",
            f"# Files staged: {len(to_delete)}",
            "set -euo pipefail",
            f'cd "{PROJECT_ROOT}"',
            f'S="{STAGE_DIR}"',
            "",
        ]
        for p in to_delete:
            dest = f'$S/{Path(p).parent}'
            lines += [f'mkdir -p "{dest}"', f'mv "overleaf/icml_aiwild_template/{p}" "{dest}/"']
        lines += ["", 'echo "Staged ' + str(len(to_delete)) + ' files into $S"']
        STAGE_SCRIPT.parent.mkdir(parents=True, exist_ok=True)
        STAGE_SCRIPT.write_text("\n".join(lines) + "\n")
        STAGE_SCRIPT.chmod(0o755)
        st.success(f"Wrote {STAGE_SCRIPT.relative_to(PROJECT_ROOT)} ({len(to_delete)} files)")

st.title("Unreferenced graphics triage")

if not view:
    st.success(
        "Nothing left in this filter. Uncheck *Only undecided* or clear the "
        "directory filter to review earlier decisions."
    )
    st.stop()

st.session_state.idx = max(0, min(st.session_state.idx, len(view) - 1))
entry = view[st.session_state.idx]
path, abs_path = entry["path"], TEMPLATE_DIR / entry["path"]
current = decisions.get(path, {}).get("decision", UNDECIDED)

# ---- navigation
nav = st.columns([1, 1, 6, 2])
if nav[0].button("← Prev", width='stretch', disabled=st.session_state.idx == 0):
    st.session_state.idx -= 1
    st.rerun()
if nav[1].button("Next →", width='stretch',
                 disabled=st.session_state.idx >= len(view) - 1):
    st.session_state.idx += 1
    st.rerun()
nav[2].markdown(
    f"**{st.session_state.idx + 1} / {len(view)}** in view &nbsp;·&nbsp; "
    f"`{path}` &nbsp;·&nbsp; {entry['bytes']/1024/1024:.2f} MB"
)
badge = {KEEP: ":green[KEEP]", DELETE: ":red[DELETE]", UNDECIDED: ":gray[undecided]"}[current]
nav[3].markdown(f"### {badge}")

image_col, info_col = st.columns([3, 1])

with image_col:
    if abs_path.suffix.lower() == ".pdf":
        st.info(
            f"`{path}` is a PDF and cannot be previewed inline. "
            "Judge it from the filename and its referenced siblings, or open it separately."
        )
    elif abs_path.exists():
        st.image(str(abs_path), width='stretch')
    else:
        st.error(f"File is missing from disk: {path}")

with info_col:
    st.subheader("Decide")
    if st.button("✅ Keep", width='stretch', type="primary" if current == KEEP else "secondary"):
        record(path, KEEP)
        if st.session_state.idx < len(view) - 1:
            st.session_state.idx += 1
        st.rerun()
    if st.button("🗑️ Delete", width='stretch', type="primary" if current == DELETE else "secondary"):
        record(path, DELETE)
        if st.session_state.idx < len(view) - 1:
            st.session_state.idx += 1
        st.rerun()
    if current != UNDECIDED and st.button("↩︎ Clear decision", width='stretch'):
        decisions.pop(path, None)
        save_decisions(decisions)
        st.rerun()

    note = st.text_input("Note (optional)", value=decisions.get(path, {}).get("note", ""))
    if note != decisions.get(path, {}).get("note", "") and current != UNDECIDED:
        record(path, current, note)

    st.divider()
    st.caption("**Closest figures the paper DOES use**")
    for sib in similar_referenced(path, referenced):
        st.caption(f"`{Path(sib).name}`")

st.divider()
with st.expander("All files in this directory (referenced vs not)"):
    d = entry["dir"]
    st.caption(f"**{d}**")
    used_here = [r for r in referenced if str(Path(r).parent) == d]
    unref_here = [e["path"] for e in unreferenced if e["dir"] == d]
    cols = st.columns(2)
    cols[0].caption(f"Referenced ({len(used_here)})")
    for r in used_here:
        cols[0].caption(f"✅ `{Path(r).name}`")
    cols[1].caption(f"Unreferenced ({len(unref_here)})")
    for r in unref_here:
        mark = {KEEP: "✅", DELETE: "🗑️", UNDECIDED: "·"}[
            decisions.get(r, {}).get("decision", UNDECIDED)
        ]
        cols[1].caption(f"{mark} `{Path(r).name}`")

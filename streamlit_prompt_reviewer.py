#!/usr/bin/env python3
"""
=============================================================================
Streamlit Prompt Audit Reviewer
=============================================================================

Interactive UI for reviewing proposed prompt changes one at a time.
Shows a git-diff-style before/after preview for each proposed change.
Accept, skip, or decline each change with optional narrative notes.
Saves decisions to docs/reference/prompt_change_decisions.json for
ingestion by scripts/apply_prompt_changes.py.

Usage:
    streamlit run streamlit_prompt_reviewer.py

What it creates:
    docs/reference/prompt_change_decisions.json  # decisions output

Dependencies:
    streamlit, difflib (stdlib)

=============================================================================
"""

import difflib
import html
import json
from pathlib import Path

import streamlit as st

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
CHANGES_FILE = BASE_DIR / "docs/reference/prompt_changes.json"
DECISIONS_FILE = BASE_DIR / "docs/reference/prompt_change_decisions.json"

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_changes():
    with open(CHANGES_FILE) as f:
        return json.load(f)


# ── Diff rendering ────────────────────────────────────────────────────────────
def render_unified_diff(before: str, after: str) -> str:
    """Return HTML for a unified-diff-style view."""
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    diff = list(difflib.unified_diff(before_lines, after_lines, lineterm="", n=100))

    if not diff:
        return "<em style='color:#888'>No change detected.</em>"

    rows = []
    for line in diff:
        escaped = html.escape(line)
        if line.startswith("+++") or line.startswith("---"):
            style = "background:#f0f0f0; color:#555; font-style:italic;"
        elif line.startswith("@@"):
            style = "background:#e8f4fd; color:#0366d6;"
        elif line.startswith("+"):
            style = "background:#e6ffed; color:#22863a;"
        elif line.startswith("-"):
            style = "background:#ffeef0; color:#cb2431;"
        else:
            style = "background:#fff; color:#24292e;"
        rows.append(
            f'<div style="{style} font-family:monospace; font-size:0.82em; '
            f'padding:2px 10px; white-space:pre-wrap; word-break:break-word;">'
            f"{escaped}</div>"
        )
    return (
        '<div style="border:1px solid #d0d7de; border-radius:6px; overflow:hidden;">'
        + "".join(rows)
        + "</div>"
    )


# ── Badges ────────────────────────────────────────────────────────────────────
def priority_badge(priority: str) -> str:
    colors = {
        "HIGH": ("#b31d28", "#ffeef0"),
        "MEDIUM": ("#7d6608", "#fffbdd"),
        "LOW": ("#2ea44f", "#e6ffed"),
    }
    fg, bg = colors.get(priority, ("#555", "#f0f0f0"))
    return (
        f'<span style="background:{bg}; color:{fg}; border:1px solid {fg}; '
        f"border-radius:4px; padding:2px 8px; font-size:0.78em; "
        f'font-weight:bold;">{priority}</span>'
    )


def game_badge(game: int) -> str:
    labels = {
        1: "Game 1 · Item Allocation",
        2: "Game 2 · Diplomatic Treaty",
        3: "Game 3 · Co-Funding",
    }
    bg = {1: "#ddf4ff", 2: "#f3e8ff", 3: "#fff8c5"}
    fg = {1: "#0550ae", 2: "#6e40c9", 3: "#7d4e00"}
    label = labels.get(game, f"Game {game}")
    b = bg.get(game, "#f0f0f0")
    f = fg.get(game, "#333")
    return (
        f'<span style="background:{b}; color:{f}; border:1px solid {f}; '
        f"border-radius:4px; padding:2px 8px; font-size:0.78em;"
        f'">{label}</span>'
    )


# ── Save decisions ────────────────────────────────────────────────────────────
def save_decisions(decisions: dict):
    DECISIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DECISIONS_FILE, "w") as f:
        json.dump(decisions, f, indent=2)


# ── Summary screen ────────────────────────────────────────────────────────────
def show_summary(changes, decisions):
    st.title("✅ Review Complete")

    accepted = [k for k, v in decisions.items() if v["decision"] == "accept"]
    skipped = [k for k, v in decisions.items() if v["decision"] == "skip"]
    declined = [k for k, v in decisions.items() if v["decision"] == "decline"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Accepted", len(accepted))
    col2.metric("Skipped", len(skipped))
    col3.metric("Declined", len(declined))

    st.divider()

    changes_by_id = {c["id"]: c for c in changes}
    for decision_label, ids, icon in [
        ("Accepted", accepted, "✅"),
        ("Skipped", skipped, "⏭"),
        ("Declined", declined, "❌"),
    ]:
        if ids:
            st.subheader(f"{icon} {decision_label}")
            for cid in ids:
                c = changes_by_id.get(cid, {})
                notes = decisions[cid].get("notes", "")
                note_str = f" *(Note: {notes})*" if notes else ""
                st.markdown(f"- **{cid}** — {c.get('title', '')}{note_str}")

    st.divider()

    col_save, col_restart = st.columns([2, 1])
    with col_save:
        if st.button("💾 Save decisions to JSON", type="primary", use_container_width=True):
            save_decisions(decisions)
            st.success(f"Saved to `{DECISIONS_FILE.relative_to(BASE_DIR)}`")
            st.code("python scripts/apply_prompt_changes.py", language="bash")
    with col_restart:
        if st.button("🔄 Start over", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Prompt Audit Reviewer",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        """
        <style>
        .block-container { max-width: 960px; padding-top: 2rem; }
        div[data-testid="stButton"] button { font-size: 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    changes = load_changes()
    n = len(changes)

    if "idx" not in st.session_state:
        st.session_state.idx = 0
    if "decisions" not in st.session_state:
        st.session_state.decisions = {}

    idx = st.session_state.idx

    if idx >= n:
        show_summary(changes, st.session_state.decisions)
        return

    change = changes[idx]

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="display:flex; align-items:center; gap:10px; margin-bottom:4px;">'
        f"{game_badge(change['game'])} &nbsp; {priority_badge(change['priority'])}"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.title(f"{change['id']}: {change['title']}")
    st.caption(
        f"Prompt: `{change['affected_prompt']}` &nbsp;·&nbsp; File: `{change['file']}`"
    )

    st.progress(idx / n, text=f"Change {idx + 1} of {n}")
    st.divider()

    # ── Issue description ─────────────────────────────────────────────────────
    with st.expander("📋 Why is this a problem?", expanded=True):
        st.markdown(change["description"])

    # ── Diff view ─────────────────────────────────────────────────────────────
    st.subheader("Proposed change (unified diff)")
    diff_html = render_unified_diff(change["before_rendered"], change["after_rendered"])
    st.markdown(diff_html, unsafe_allow_html=True)

    # ── Side-by-side ──────────────────────────────────────────────────────────
    with st.expander("🔍 Side-by-side view", expanded=False):
        col_b, col_a = st.columns(2)
        with col_b:
            st.markdown(
                '<p style="font-weight:bold; color:#cb2431;">BEFORE</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="background:#ffeef0; padding:12px; border-radius:6px; '
                f"font-family:monospace; font-size:0.82em; white-space:pre-wrap; "
                f'word-break:break-word;">{html.escape(change["before_rendered"])}</div>',
                unsafe_allow_html=True,
            )
        with col_a:
            st.markdown(
                '<p style="font-weight:bold; color:#22863a;">AFTER</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="background:#e6ffed; padding:12px; border-radius:6px; '
                f"font-family:monospace; font-size:0.82em; white-space:pre-wrap; "
                f'word-break:break-word;">{html.escape(change["after_rendered"])}</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Notes ─────────────────────────────────────────────────────────────────
    prev = st.session_state.decisions.get(change["id"], {})
    notes = st.text_area(
        "Notes (optional — reasoning, caveats, or follow-up ideas)",
        value=prev.get("notes", ""),
        key=f"notes_{idx}",
        height=80,
    )

    # ── Buttons ───────────────────────────────────────────────────────────────
    col_acc, col_skip, col_dec, col_back = st.columns([1.2, 1, 1.2, 1])

    def record_and_advance(decision: str):
        st.session_state.decisions[change["id"]] = {
            "decision": decision,
            "notes": st.session_state.get(f"notes_{idx}", ""),
        }
        st.session_state.idx += 1

    with col_acc:
        if st.button("✅  Accept", key=f"acc_{idx}", use_container_width=True, type="primary"):
            record_and_advance("accept")
            st.rerun()

    with col_skip:
        if st.button("⏭  Skip", key=f"skip_{idx}", use_container_width=True):
            record_and_advance("skip")
            st.rerun()

    with col_dec:
        if st.button("❌  Decline", key=f"dec_{idx}", use_container_width=True):
            record_and_advance("decline")
            st.rerun()

    with col_back:
        if idx > 0:
            if st.button("← Back", key=f"back_{idx}", use_container_width=True):
                st.session_state.idx -= 1
                st.rerun()

    # ── Jump-to sidebar ───────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### All changes")
        decision_icons = {"accept": "✅", "skip": "⏭", "decline": "❌"}
        for i, c in enumerate(changes):
            d = st.session_state.decisions.get(c["id"], {})
            icon = decision_icons.get(d.get("decision", ""), "⬜")
            short = c["title"][:38] + ("…" if len(c["title"]) > 38 else "")
            if st.button(f"{icon} {c['id']}: {short}", key=f"jump_{i}", use_container_width=True):
                st.session_state.idx = i
                st.rerun()


if __name__ == "__main__":
    main()

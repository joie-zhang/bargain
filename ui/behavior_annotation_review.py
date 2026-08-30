#!/usr/bin/env python3
"""Blind human review UI for turn-level semantic behavior labels."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from pathlib import Path
from typing import Any

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui.behavior_review_core import (  # noqa: E402
    agreement_csv,
    agreement_rows,
    append_decision,
    atomic_write_text,
    latest_decisions,
    load_jsonl,
    sha256_file,
    source_record,
    validate_reviewer_id,
)


DEFAULT_REVIEW_DIR = PROJECT_ROOT / "analysis" / "behavior_annotation_irr_review_20260814"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_REVIEW_DIR / "sampling_manifest.jsonl",
    )
    parser.add_argument(
        "--journal",
        type=Path,
        default=DEFAULT_REVIEW_DIR / "human_decisions.jsonl",
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=DEFAULT_REVIEW_DIR / "agreement_ready.csv",
    )
    args, _ = parser.parse_known_args()
    return args


@st.cache_data(show_spinner=False)
def load_manifest(path: str, digest: str) -> list[dict[str, Any]]:
    del digest
    rows = load_jsonl(Path(path))
    item_ids = [row["item_id"] for row in rows]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("The sampling manifest contains duplicate item IDs")
    return sorted(rows, key=lambda row: int(row["sample_position"]))


@st.cache_data(show_spinner=False)
def load_view(path: str, expected_digest: str) -> dict[str, Any]:
    resolved = Path(path)
    actual_digest = sha256_file(resolved)
    if actual_digest != expected_digest:
        raise ValueError("The rollout view changed after sampling")
    return json.loads(resolved.read_text(encoding="utf-8"))


def exact_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False)


def render_record(title: str, text: str, metadata: str, focused: bool) -> None:
    border = "3px solid #2563eb" if focused else "1px solid #d1d5db"
    background = "#eff6ff" if focused else "#ffffff"
    st.markdown(
        f"""
        <div style="border:{border};background:{background};border-radius:8px;padding:12px;margin:8px 0">
          <div style="font-weight:700">{html.escape(title)}</div>
          <div style="font-size:0.8rem;color:#6b7280;margin-bottom:8px">{html.escape(metadata)}</div>
          <div style="white-space:pre-wrap;word-break:break-word">{html.escape(text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def focus_matches(item: dict[str, Any], source_kind: str, index: int) -> bool:
    return item["source_kind"] == source_kind and int(item["source_index"]) == int(index)


def render_full_context(view: dict[str, Any], item: dict[str, Any]) -> None:
    transcript, interactions, setup = st.tabs(
        ["Public transcript", "Agent-authored interactions", "Setup and outcome"]
    )
    with transcript:
        for row in view.get("conversation_logs") or []:
            index = int(row["log_index"])
            render_record(
                f"Log {index}, {row.get('speaker_agent')}",
                exact_text(row.get("content") or ""),
                f"round {row.get('round')}, discussion turn {row.get('discussion_turn')}",
                focus_matches(item, "conversation_log", index),
            )
    with interactions:
        rows = view.get("agent_authored_interactions") or view.get("target_private_interactions") or []
        for row in rows:
            index = int(row["interaction_index"])
            render_record(
                f"Interaction {index}, {row.get('agent_id') or item['target_agent']}",
                exact_text(row.get("response") or row.get("content") or ""),
                f"phase {row.get('phase')}, round {row.get('round')}",
                focus_matches(item, "interaction", index),
            )
    with setup:
        st.caption("This is the same compact rollout context stored for semantic adjudication.")
        st.subheader("Configuration")
        st.json(view.get("config") or {}, expanded=False)
        st.subheader("Outcome")
        st.json(view.get("outcome") or {}, expanded=False)


args = parse_args()
manifest_path = args.manifest.resolve()
journal_path = args.journal.resolve()
export_path = args.export.resolve()

st.set_page_config(page_title="Behavior label review", page_icon="🔎", layout="wide")
st.title("Blind behavior label review")
st.caption(
    "Decide whether the named behavior occurs in the focused target-authored source record. "
    "Use the full rollout as context. The machine decision stays hidden until you respond."
)

if not manifest_path.exists():
    st.error(
        f"Sampling manifest not found: {manifest_path}\n\n"
        "Build it with `python scripts/build_behavior_irr_sample.py`."
    )
    st.stop()

manifest_digest = sha256_file(manifest_path)
try:
    items = load_manifest(str(manifest_path), manifest_digest)
except Exception as exc:
    st.error(str(exc))
    st.stop()

decision_rows = load_jsonl(journal_path)
latest = latest_decisions(decision_rows)

if "review_idx" not in st.session_state:
    st.session_state.review_idx = 0
if "reviewer_id" not in st.session_state:
    st.session_state.reviewer_id = ""

with st.sidebar:
    st.header("Reviewer")
    reviewer_id = st.text_input(
        "Reviewer ID",
        value=st.session_state.reviewer_id,
        placeholder="for example, jz4391",
        help="Use a stable pseudonym. Do not enter an email address or full name.",
    )
    st.session_state.reviewer_id = reviewer_id.strip()
    try:
        valid_reviewer = validate_reviewer_id(reviewer_id)
        reviewer_error = None
    except ValueError as exc:
        valid_reviewer = ""
        reviewer_error = str(exc)
    if reviewer_error and reviewer_id:
        st.error(reviewer_error)

    reviewer_decisions = {
        item_id: row
        for (saved_reviewer, item_id), row in latest.items()
        if saved_reviewer == valid_reviewer
    }
    completed = sum(
        row.get("reviewer_response") in {"yes", "no", "unsure", "skip"}
        for row in reviewer_decisions.values()
    )
    st.header("Progress")
    st.progress(completed / max(len(items), 1))
    st.metric("Reviewed", f"{completed} / {len(items)}")
    only_unreviewed = st.checkbox("Only unreviewed", value=True)
    reveal_after_response = st.checkbox(
        "Reveal machine result after each response",
        value=False,
        help="Leave this off for an agreement study because feedback can affect later decisions.",
    )

    export_rows = agreement_rows(items, decision_rows)
    export_text = agreement_csv(export_rows)
    st.download_button(
        "Download agreement table",
        data=export_text,
        file_name=export_path.name,
        mime="text/csv",
        width="stretch",
    )
    if st.button("Write agreement table on cluster", width="stretch"):
        atomic_write_text(export_path, export_text)
        st.success(f"Wrote {export_path}")

view_items = [
    item for item in items if not only_unreviewed or item["item_id"] not in reviewer_decisions
]
if not view_items:
    st.success("You reviewed every item in this view.")
    st.stop()

st.session_state.review_idx = max(0, min(st.session_state.review_idx, len(view_items) - 1))
item = view_items[st.session_state.review_idx]
saved = reviewer_decisions.get(item["item_id"])

nav = st.columns([1, 1, 6])
if nav[0].button("Previous", disabled=st.session_state.review_idx == 0, width="stretch"):
    st.session_state.review_idx -= 1
    st.rerun()
if nav[1].button(
    "Next",
    disabled=st.session_state.review_idx >= len(view_items) - 1,
    width="stretch",
):
    st.session_state.review_idx += 1
    st.rerun()
nav[2].markdown(
    f"**{st.session_state.review_idx + 1} / {len(view_items)}** in this view, "
    f"item `{item['item_id']}`"
)

st.subheader(item["tag_title"])
st.write(item["tag_definition"])
st.caption(
    f"Category: {item['tag_category']} | Model: {item['family']} | "
    f"Effort: {item['level']} | Dataset: {item['dataset']}"
)

try:
    rollout_view = load_view(item["rollout_view_path"], item["rollout_view_sha256"])
    focused = source_record(rollout_view, item)
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.markdown("#### Focused source record")
focused_text = exact_text(focused.get("content") or focused.get("response") or "")
render_record(
    f"{item['source_kind']} {item['source_index']}, {item['speaker_agent']}",
    focused_text,
    f"phase {item['phase']}, round {item['round']}, discussion turn {item['discussion_turn']}",
    True,
)

st.markdown("#### Your decision")
st.write("Does the target perform this behavior in the focused source record?")
note = st.text_input("Optional note", key=f"note_{item['item_id']}")
buttons = st.columns(4)
button_specs = (
    ("Yes", "yes"),
    ("No", "no"),
    ("Unsure", "unsure"),
    ("Skip", "skip"),
)
for column, (label, response) in zip(buttons, button_specs):
    if column.button(label, width="stretch", disabled=not valid_reviewer):
        append_decision(
            journal_path,
            manifest_path,
            item,
            valid_reviewer,
            response,
            note,
        )
        if only_unreviewed:
            st.session_state.review_idx = min(
                st.session_state.review_idx, max(len(view_items) - 2, 0)
            )
        st.rerun()

if not valid_reviewer:
    st.info("Enter a valid reviewer ID to enable the decision buttons.")

if saved:
    st.success(f"Your saved response is `{saved['reviewer_response']}`.")
    if reveal_after_response:
        machine_text = "YES" if item["machine_positive"] else "NO"
        st.info(f"Machine decision after your response: **{machine_text}**")

st.markdown("#### Full rollout context")
render_full_context(rollout_view, item)

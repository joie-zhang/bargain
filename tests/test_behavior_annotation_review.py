from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.analyze_behavior_irr import metric_block
from scripts.build_behavior_irr_sample import eligible_tag, iter_target_sources
from ui.behavior_review_core import (
    agreement_csv,
    agreement_rows,
    append_decision,
    latest_decisions,
    load_jsonl,
    source_record,
)


def test_eligible_tag_respects_turn_level_scope() -> None:
    manifest = {"n_agents": 2, "game_label": "game1"}
    assert eligible_tag({"scope_hint": {}}, manifest)
    assert not eligible_tag({"scope_hint": {"structural": True}}, manifest)
    assert not eligible_tag({"scope_hint": {"min_agents": 3}}, manifest)
    assert not eligible_tag({"scope_hint": {"games": ["game3"]}}, manifest)


def test_target_source_enumeration_excludes_duplicate_discussion_interactions() -> None:
    view = {
        "conversation_logs": [
            {"speaker_agent": "Agent_1", "log_index": 0, "content": "target"},
            {"speaker_agent": "Agent_2", "log_index": 1, "content": "baseline"},
        ],
        "agent_authored_interactions": [
            {
                "agent_id": "Agent_1",
                "interaction_index": 2,
                "phase": "discussion",
                "response": "target",
            },
            {
                "agent_id": "Agent_1",
                "interaction_index": 3,
                "phase": "private_thinking",
                "response": "private",
            },
            {
                "agent_id": "Agent_2",
                "interaction_index": 4,
                "phase": "private_thinking",
                "response": "baseline private",
            },
        ],
    }
    sources = list(iter_target_sources(view, "Agent_1"))
    assert [(row["source_kind"], row["source_index"]) for row in sources] == [
        ("conversation_log", 0),
        ("interaction", 3),
    ]


def test_journal_is_append_only_and_export_uses_latest_response(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    item = {
        "item_id": "item1",
        "machine_positive": True,
        "sampling_weight": 2.0,
        "dataset": "test",
        "family": "gpt-5",
        "level": "low",
        "tag_code": "example",
    }
    manifest_path.write_text(json.dumps(item) + "\n", encoding="utf-8")
    journal_path = tmp_path / "decisions.jsonl"

    append_decision(journal_path, manifest_path, item, "reviewer_1", "unsure")
    append_decision(journal_path, manifest_path, item, "reviewer_1", "yes")
    decisions = load_jsonl(journal_path)

    assert len(decisions) == 2
    assert latest_decisions(decisions)[("reviewer_1", "item1")]["reviewer_response"] == "yes"
    rows = agreement_rows([item], decisions)
    assert rows[0]["reviewer_binary"] == 1
    assert rows[0]["agrees_with_machine"] is True
    csv_text = agreement_csv(rows)
    assert "machine_positive" in csv_text
    assert "reviewer_1" in csv_text


def test_source_record_checks_text_hash() -> None:
    text = "A focused turn"
    item = {
        "source_kind": "conversation_log",
        "source_index": 7,
        "source_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    view = {"conversation_logs": [{"log_index": 7, "content": text}]}
    assert source_record(view, item)["content"] == text


def test_agreement_metrics_use_machine_against_human_confusion() -> None:
    rows = [
        {"machine_binary": 1, "reviewer_binary": 1, "sampling_weight": 1},
        {"machine_binary": 1, "reviewer_binary": 0, "sampling_weight": 1},
        {"machine_binary": 0, "reviewer_binary": 1, "sampling_weight": 1},
        {"machine_binary": 0, "reviewer_binary": 0, "sampling_weight": 1},
    ]
    metrics = metric_block(rows, weighted=False)
    assert metrics["confusion_machine_against_human"] == {
        "tp": 1.0,
        "fp": 1.0,
        "fn": 1.0,
        "tn": 1.0,
    }
    assert metrics["raw_agreement"] == 0.5
    assert metrics["cohen_kappa"] == 0.0

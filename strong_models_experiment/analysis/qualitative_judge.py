"""Utilities for judge-packet generation and judge-score aggregation."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Dict, List

from .qualitative_metrics import (
    compute_qualitative_metrics_v1,
    extract_pledge_history_from_logs,
)


RUBRIC_VERSION = "qualitative_judge_rubric_v1"


def build_judge_packet(result: Dict[str, Any], max_messages: int = 40) -> Dict[str, Any] | None:
    """Build one judge packet from a co-funding result record."""
    config = result.get("config", {})
    if config.get("game_type") != "co_funding":
        return None

    logs = result.get("conversation_logs", [])
    discussions = [m for m in logs if m.get("phase") == "discussion"][:max_messages]

    qualitative_events = result.get("qualitative_events")
    qualitative_metrics = result.get("qualitative_metrics_v1")

    if not qualitative_metrics:
        game_state = {
            "projects": config.get("items", []),
            "round_pledges": extract_pledge_history_from_logs(logs),
            "agent_budgets": config.get("agent_budgets", {}),
        }
        qualitative_metrics, qualitative_events = compute_qualitative_metrics_v1(logs, game_state)

    payload_id = f"{result.get('experiment_id','')}_{config.get('alpha','na')}_{config.get('sigma','na')}"
    packet_id = hashlib.sha1(payload_id.encode("utf-8")).hexdigest()[:16]

    return {
        "packet_id": packet_id,
        "rubric_version": RUBRIC_VERSION,
        "metadata": {
            "experiment_id": result.get("experiment_id"),
            "alpha": config.get("alpha"),
            "sigma": config.get("sigma"),
            "models": config.get("agents", []),
            "consensus_reached": result.get("consensus_reached"),
            "final_round": result.get("final_round"),
        },
        "transcript": discussions,
        "extracted_events": qualitative_events or [],
        "qualitative_metrics_v1": qualitative_metrics or {},
    }


def aggregate_judge_scores(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate packet-level judge responses into quality metrics."""
    by_event: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    overall_quality_scores: List[float] = []

    for row in rows:
        event_scores = row.get("event_type_scores", {})
        if isinstance(event_scores, dict):
            for event_type, counts in event_scores.items():
                if not isinstance(counts, dict):
                    continue
                by_event[event_type]["tp"] += int(counts.get("tp", 0) or 0)
                by_event[event_type]["fp"] += int(counts.get("fp", 0) or 0)
                by_event[event_type]["fn"] += int(counts.get("fn", 0) or 0)

        quality = row.get("overall_quality")
        if isinstance(quality, (int, float)):
            overall_quality_scores.append(float(quality))

    event_metrics: Dict[str, Dict[str, float]] = {}
    for event_type, counts in by_event.items():
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        event_metrics[event_type] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "rubric_version": RUBRIC_VERSION,
        "num_packets_scored": len(rows),
        "overall_quality_mean": (
            sum(overall_quality_scores) / len(overall_quality_scores)
            if overall_quality_scores
            else None
        ),
        "event_metrics": event_metrics,
    }

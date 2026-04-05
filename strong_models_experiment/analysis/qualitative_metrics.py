"""Structured qualitative metrics for co-funding transcripts.

This module intentionally avoids legacy keyword sentiment heuristics.
It links explicit communication events and contribution vectors.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple


_COMMITMENT_PATTERNS = (
    re.compile(
        r"\b(?:i|we)\s*(?:will|can|shall|'ll)\s*"
        r"(?:contribute|pledge|allocate|commit|put)\s*\$?\s*"
        r"([0-9]+(?:\.[0-9]+)?)\s*(?:to|toward|towards|for)\s*"
        r"(project\s+[a-z0-9_\- ]+)",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(project\s+[a-z0-9_\- ]+)\s*[:\-]?\s*"
        r"(?:i|we)\s*(?:will|can|shall|'ll)\s*"
        r"(?:contribute|pledge|allocate|commit|put)\s*\$?\s*"
        r"([0-9]+(?:\.[0-9]+)?)",
        flags=re.IGNORECASE,
    ),
)


def _adaptation_rate(
    pledge_history: List[Dict[str, List[float]]],
    budgets: Dict[str, float],
) -> Dict[str, float]:
    """Compute adaptation rate without importing heavy game modules."""
    rounds = len(pledge_history)
    if rounds < 2:
        return {aid: 0.0 for aid in budgets}

    result: Dict[str, float] = {}
    for aid, budget in budgets.items():
        total_change = 0.0
        for t in range(1, rounds):
            prev = pledge_history[t - 1].get(aid, [])
            curr = pledge_history[t].get(aid, [])
            if len(prev) != len(curr):
                continue
            total_change += sum(abs(float(c) - float(p)) for p, c in zip(prev, curr))

        result[aid] = total_change / ((rounds - 1) * budget) if budget > 1e-12 else 0.0
    return result


def _normalize_project_aliases(projects: List[Dict[str, Any]]) -> Dict[str, int]:
    """Build a robust alias map for project name lookup."""
    aliases: Dict[str, int] = {}
    for idx, project in enumerate(projects):
        name = str(project.get("name", f"Project_{idx + 1}")).strip()
        low = name.lower()
        aliases[low] = idx

        # If formatted like "Project Alpha", also allow "alpha".
        if low.startswith("project "):
            short = low.replace("project ", "", 1).strip()
            if short:
                aliases[short] = idx

    return aliases


def _resolve_project_index(raw_project: str, aliases: Dict[str, int]) -> int | None:
    text = raw_project.strip().lower()
    if text in aliases:
        return aliases[text]
    for alias, idx in aliases.items():
        if alias and re.search(rf"\b{re.escape(alias)}\b", text):
            return idx
    return None


def extract_pledge_history_from_logs(conversation_logs: List[Dict[str, Any]]) -> List[Dict[str, List[float]]]:
    """Extract pledge history from phase logs as round-indexed contribution maps."""
    by_round: Dict[int, Dict[str, List[float]]] = defaultdict(dict)
    for log in conversation_logs:
        phase = log.get("phase")
        if phase == "pledge_submission":
            payload = log.get("pledge", {})
        elif phase == "proposal":
            payload = log.get("proposal", {})
            if not isinstance(payload, dict) or "contributions" not in payload:
                continue
        else:
            continue

        contributions = payload.get("contributions")
        if not isinstance(contributions, list):
            continue
        round_num = int(log.get("round", 0))
        if round_num <= 0:
            continue
        agent_id = log.get("from") or payload.get("proposed_by")
        if not agent_id:
            continue
        by_round[round_num][agent_id] = [float(x) for x in contributions]

    if not by_round:
        return []
    return [by_round[r] for r in sorted(by_round.keys())]


def _extract_commitment_events(
    conversation_logs: List[Dict[str, Any]],
    project_aliases: Dict[str, int],
) -> List[Dict[str, Any]]:
    """Parse explicit numeric commitments from discussion utterances."""
    events: List[Dict[str, Any]] = []
    for log in conversation_logs:
        if log.get("phase") != "discussion":
            continue
        content = str(log.get("content", ""))
        if not content.strip():
            continue

        round_num = int(log.get("round", 0))
        speaker = str(log.get("from", ""))
        if round_num <= 0 or not speaker:
            continue

        for pattern in _COMMITMENT_PATTERNS:
            for match in pattern.finditer(content):
                g1, g2 = match.group(1), match.group(2)
                if g1.lower().startswith("project "):
                    project_raw, amount_raw = g1, g2
                else:
                    amount_raw, project_raw = g1, g2

                project_idx = _resolve_project_index(project_raw, project_aliases)
                if project_idx is None:
                    continue
                try:
                    amount = float(amount_raw)
                except ValueError:
                    continue

                events.append(
                    {
                        "event_type": "commitment",
                        "round": round_num,
                        "speaker": speaker,
                        "project_idx": project_idx,
                        "amount": amount,
                        "evidence_text": match.group(0),
                    }
                )
    return events


def _extract_advocacy_events(
    conversation_logs: List[Dict[str, Any]],
    project_aliases: Dict[str, int],
) -> List[Dict[str, Any]]:
    """Parse project advocacy mentions by explicit project references."""
    events: List[Dict[str, Any]] = []
    for log in conversation_logs:
        if log.get("phase") != "discussion":
            continue
        content = str(log.get("content", "")).lower()
        if not content:
            continue
        round_num = int(log.get("round", 0))
        speaker = str(log.get("from", ""))
        if round_num <= 0 or not speaker:
            continue

        for alias, project_idx in project_aliases.items():
            if not alias:
                continue
            if re.search(rf"\b{re.escape(alias)}\b", content):
                events.append(
                    {
                        "event_type": "advocacy",
                        "round": round_num,
                        "speaker": speaker,
                        "project_idx": project_idx,
                    }
                )
    return events


def _build_round_lookup(pledge_history: List[Dict[str, List[float]]]) -> Dict[int, Dict[str, List[float]]]:
    return {idx + 1: pledges for idx, pledges in enumerate(pledge_history)}


def compute_promise_keeping_metrics(
    commitments: List[Dict[str, Any]],
    pledge_history: List[Dict[str, List[float]]],
    tolerance: float = 2.0,
) -> Dict[str, Any]:
    """Compute promise-keeping quality: commitments vs next-round realized pledge."""
    by_round = _build_round_lookup(pledge_history)
    per_agent_total: Dict[str, int] = defaultdict(int)
    per_agent_kept: Dict[str, int] = defaultdict(int)
    abs_errors: List[float] = []

    for c in commitments:
        t = int(c["round"])
        speaker = c["speaker"]
        project_idx = int(c["project_idx"])
        amount = float(c["amount"])
        next_round = by_round.get(t + 1)
        if not next_round:
            continue
        speaker_pledge = next_round.get(speaker)
        if not speaker_pledge or project_idx >= len(speaker_pledge):
            continue

        realized = float(speaker_pledge[project_idx])
        err = abs(realized - amount)
        abs_errors.append(err)
        per_agent_total[speaker] += 1
        if err <= tolerance:
            per_agent_kept[speaker] += 1

    total = sum(per_agent_total.values())
    kept = sum(per_agent_kept.values())
    overall_rate = (kept / total) if total > 0 else None

    per_agent_rate = {
        aid: (per_agent_kept[aid] / count if count > 0 else None)
        for aid, count in per_agent_total.items()
    }

    return {
        "total_commitments": total,
        "kept_commitments": kept,
        "overall_keep_rate": overall_rate,
        "mean_abs_error": (sum(abs_errors) / len(abs_errors)) if abs_errors else None,
        "per_agent_keep_rate": per_agent_rate,
    }


def compute_persuasion_effectiveness_metrics(
    advocacy_events: List[Dict[str, Any]],
    pledge_history: List[Dict[str, List[float]]],
) -> Dict[str, Any]:
    """Compute whether advocacy predicts next-round increases by other agents."""
    by_round = _build_round_lookup(pledge_history)
    per_agent_deltas: Dict[str, List[float]] = defaultdict(list)
    all_deltas: List[float] = []

    for event in advocacy_events:
        t = int(event["round"])
        speaker = str(event["speaker"])
        project_idx = int(event["project_idx"])

        pledges_t = by_round.get(t)
        pledges_next = by_round.get(t + 1)
        if not pledges_t or not pledges_next:
            continue

        other_deltas = []
        for aid, vec_t in pledges_t.items():
            if aid == speaker:
                continue
            vec_next = pledges_next.get(aid)
            if not vec_next:
                continue
            if project_idx >= len(vec_t) or project_idx >= len(vec_next):
                continue
            other_deltas.append(float(vec_next[project_idx]) - float(vec_t[project_idx]))

        if other_deltas:
            mean_delta = sum(other_deltas) / len(other_deltas)
            per_agent_deltas[speaker].append(mean_delta)
            all_deltas.append(mean_delta)

    per_agent_mean = {
        aid: (sum(vals) / len(vals) if vals else None) for aid, vals in per_agent_deltas.items()
    }
    return {
        "event_count": len(all_deltas),
        "overall_other_agent_delta": (sum(all_deltas) / len(all_deltas)) if all_deltas else None,
        "per_agent_other_agent_delta": per_agent_mean,
    }


def compute_coalition_formation_metrics(
    pledge_history: List[Dict[str, List[float]]],
    min_contribution: float = 1e-6,
) -> Dict[str, Any]:
    """Detect repeated co-funding coalitions from contribution vectors."""
    if not pledge_history:
        return {
            "coalition_active_round_fraction": None,
            "persistent_project_fraction": None,
            "max_consecutive_rounds_any_project": 0,
            "persistent_project_count": 0,
        }

    total_rounds = len(pledge_history)
    threshold = max(1, math.ceil(total_rounds / 2))
    example_round = next(iter(pledge_history), {})
    m_projects = len(next(iter(example_round.values()))) if example_round else 0
    if m_projects == 0:
        return {
            "coalition_active_round_fraction": None,
            "persistent_project_fraction": None,
            "max_consecutive_rounds_any_project": 0,
            "persistent_project_count": 0,
        }

    project_active_series: List[List[bool]] = [[] for _ in range(m_projects)]
    coalition_rounds = 0

    for round_pledges in pledge_history:
        any_project_active = False
        for j in range(m_projects):
            contributors = 0
            for _, vec in round_pledges.items():
                if j < len(vec) and float(vec[j]) > min_contribution:
                    contributors += 1
            active = contributors >= 2
            project_active_series[j].append(active)
            any_project_active = any_project_active or active
        coalition_rounds += 1 if any_project_active else 0

    def _longest_streak(bits: List[bool]) -> int:
        best = 0
        curr = 0
        for b in bits:
            curr = curr + 1 if b else 0
            if curr > best:
                best = curr
        return best

    longest_per_project = [_longest_streak(bits) for bits in project_active_series]
    persistent_count = sum(1 for x in longest_per_project if x >= threshold)

    return {
        "coalition_active_round_fraction": coalition_rounds / total_rounds if total_rounds else None,
        "persistent_project_fraction": persistent_count / m_projects if m_projects else None,
        "max_consecutive_rounds_any_project": max(longest_per_project) if longest_per_project else 0,
        "persistent_project_count": persistent_count,
    }


def compute_qualitative_metrics_v1(
    conversation_logs: List[Dict[str, Any]],
    game_state: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Compute qualitative metrics for co-funding from structured logs + vectors."""
    projects = game_state.get("projects", [])
    project_aliases = _normalize_project_aliases(projects)

    pledge_history = game_state.get("round_pledges", [])
    if not pledge_history:
        pledge_history = extract_pledge_history_from_logs(conversation_logs)

    budgets = game_state.get("agent_budgets", {})
    commitments = _extract_commitment_events(conversation_logs, project_aliases)
    advocacy = _extract_advocacy_events(conversation_logs, project_aliases)

    promise_keeping = compute_promise_keeping_metrics(commitments, pledge_history)
    persuasion = compute_persuasion_effectiveness_metrics(advocacy, pledge_history)
    coalition = compute_coalition_formation_metrics(pledge_history)
    adapt = _adaptation_rate(pledge_history, budgets) if budgets else {}

    events: List[Dict[str, Any]] = []
    events.extend(commitments)
    events.extend(advocacy)

    metrics = {
        "schema_version": "qualitative_metrics_v1",
        "event_count": len(events),
        "promise_keeping": promise_keeping,
        "persuasion_effectiveness": persuasion,
        "coalition_formation": coalition,
        "adaptation_rate": adapt,
    }
    return metrics, events

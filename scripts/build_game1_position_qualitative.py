#!/usr/bin/env python3
"""Build qualitative transcript artifacts for Game 1 same-model controls."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OBSERVATIONS = (
    REPO_ROOT
    / "Figures/n_gt_2_game1_multiagent/position_effects_control/control_position_observations.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis/game1_position_qualitative"


ALIGNMENT_PATTERNS = (
    "aligned",
    "agree",
    "on board",
    "ready to lock",
    "confirm",
    "acceptable",
    "works for me",
    "satisfies",
    "support",
)
QUICK_PATTERNS = (
    "one-round",
    "round-1",
    "round 1",
    "quick",
    "clean",
    "no leftovers",
    "avoid deadlock",
    "lock",
    "finalize",
    "consensus",
)
EFFICIENCY_PATTERNS = (
    "pareto",
    "efficient",
    "value-maximizing",
    "maximizes",
    "top-priority",
    "top priority",
    "top items",
    "top picks",
    "highest-valued",
)
CONCESSION_PATTERNS = (
    "fallback",
    "minimal adjustment",
    "flexible",
    "open to",
    "adjust",
    "acceptable",
    "if needed",
    "if anyone wants",
)
ANCHOR_PATTERNS = (
    "i propose",
    "proposed",
    "preferred",
    "my top",
    "top priority",
    "i take",
    "i get",
    "i would receive",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create line-numbered transcript packets and coded features for position-effect analysis."
    )
    parser.add_argument("--observations", default=str(DEFAULT_OBSERVATIONS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_any(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def agent_position(agent_id: Optional[str]) -> Optional[int]:
    if not agent_id:
        return None
    match = re.fullmatch(r"Agent_(\d+)", str(agent_id))
    return int(match.group(1)) if match else None


def norm_allocation(allocation: Dict[str, Iterable[int]]) -> tuple:
    return tuple((agent, tuple(sorted(items))) for agent, items in sorted(allocation.items()))


def allocation_utility(allocation: Dict[str, Iterable[int]], preferences: Dict[str, List[float]]) -> Dict[str, float]:
    utilities: Dict[str, float] = {}
    for agent_id, items in allocation.items():
        values = preferences.get(agent_id) or []
        utilities[agent_id] = float(sum(values[item] for item in items if item < len(values)))
    return utilities


def top_items(values: List[float]) -> set[int]:
    if not values:
        return set()
    top_value = max(values)
    return {idx for idx, value in enumerate(values) if value == top_value}


def agent_received_top_item(agent_id: str, allocation: Dict[str, Iterable[int]], preferences: Dict[str, List[float]]) -> bool:
    tops = top_items(preferences.get(agent_id) or [])
    return bool(tops.intersection(set(allocation.get(agent_id) or [])))


def agent_max_utility(agent_id: str, preferences: Dict[str, List[float]]) -> Optional[float]:
    values = preferences.get(agent_id) or []
    return float(max(values)) if values else None


def accepted_proposal_index(logs: List[Dict[str, Any]]) -> Optional[int]:
    for entry in logs:
        if entry.get("phase") != "vote_tabulation":
            continue
        match = re.search(r"Proposal #(\d+) accepted", str(entry.get("content", "")))
        if match:
            return int(match.group(1))
    return None


def contains_any(text: str, patterns: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in patterns)


def add_line(lines: List[str], text: str) -> int:
    lines.append(text.rstrip())
    return len(lines)


def add_event(
    events: List[Dict[str, Any]],
    run_meta: Dict[str, Any],
    event_type: str,
    transcript_path: Path,
    line_number: int,
    quote: str,
    agent_id: Optional[str] = None,
    phase: Optional[str] = None,
) -> None:
    events.append(
        {
            **run_meta,
            "event_type": event_type,
            "agent_id": agent_id,
            "position": agent_position(agent_id),
            "phase": phase,
            "transcript_path": str(transcript_path),
            "line_number": line_number,
            "quote": quote.strip(),
        }
    )


def raw_interaction_category(phase: str) -> Optional[str]:
    if phase.startswith("private_thinking_round_"):
        return "private_thinking"
    if phase.startswith("proposal_round_"):
        return "formal_proposal_raw"
    if phase.startswith("voting_round_"):
        return "private_vote"
    if phase.startswith("reflection_round_"):
        return "reflection"
    return None


def decode_response(response: Any) -> Any:
    if not isinstance(response, str):
        return response
    stripped = response.strip()
    if not stripped:
        return response
    if stripped[0] not in "[{":
        return response
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return response


def render_response(response: Any) -> List[str]:
    decoded = decode_response(response)
    if isinstance(decoded, dict):
        rendered: List[str] = []
        for key, value in decoded.items():
            if isinstance(value, list):
                rendered.append(f"- {key}:")
                for item in value:
                    rendered.append(f"  - {item}")
            elif isinstance(value, dict):
                rendered.append(f"- {key}: {json.dumps(value, ensure_ascii=True, sort_keys=True)}")
            else:
                rendered.append(f"- {key}: {value}")
        return rendered
    if isinstance(decoded, list):
        return [json.dumps(decoded, ensure_ascii=True)]
    return str(response or "").strip().splitlines()


def write_transcript(
    run: pd.Series,
    result: Dict[str, Any],
    transcript_path: Path,
    events: List[Dict[str, Any]],
) -> Dict[str, bool]:
    logs = list(result.get("conversation_logs") or [])
    utilities = result.get("final_utilities") or {}
    allocation = result.get("final_allocation") or {}
    run_meta = {
        "run_id": run["run_id"],
        "config_id": int(run["config_id"]),
        "n_agents": int(run["n_agents"]),
        "competition_level": float(run["competition_level"]),
        "first_minus_last": float(run["first_minus_last"]),
        "result_path": run["result_path"],
    }

    flags = {
        "agent1_opening_anchor_language": False,
        "later_alignment_language": False,
        "quick_resolution_frame": False,
        "efficiency_top_items_frame": False,
        "later_concession_language": False,
        "raw_interactions_loaded": False,
        "private_thinking_entries": 0,
        "raw_formal_proposal_entries": 0,
        "private_voting_entries": 0,
        "reflection_entries": 0,
    }

    lines: List[str] = []
    add_line(lines, f"# {run['run_id']}")
    add_line(lines, "")
    add_line(lines, f"- source: {run['result_path']}")
    add_line(lines, f"- n_agents: {int(run['n_agents'])}")
    add_line(lines, f"- competition_level: {float(run['competition_level']):g}")
    add_line(lines, f"- final_round: {result.get('final_round')}")
    add_line(lines, f"- consensus_reached: {result.get('consensus_reached')}")
    add_line(lines, f"- final_utilities: {utilities}")
    add_line(lines, f"- final_allocation: {allocation}")
    add_line(lines, "")
    add_line(lines, "## Agent Preferences")
    for agent_id in sorted(result.get("agent_preferences") or {}):
        add_line(lines, f"- {agent_id}: {result['agent_preferences'][agent_id]}")
    add_line(lines, "")
    add_line(lines, "## Transcript")

    seen_agent_discussion: set[str] = set()
    for entry_index, entry in enumerate(logs, start=1):
        phase = str(entry.get("phase", ""))
        agent_id = entry.get("from") or entry.get("agent_id")
        position = agent_position(agent_id)
        header = (
            f"### Entry {entry_index:02d} | phase={phase} | round={entry.get('round')} "
            f"| from={agent_id}"
        )
        if entry.get("speaker_order") is not None:
            header += f" | speaker_order={entry.get('speaker_order')}/{entry.get('total_speakers')}"
        add_line(lines, "")
        add_line(lines, header)

        content = str(entry.get("content", "")).strip()
        if content:
            add_line(lines, "")
            for raw_line in content.splitlines():
                line_number = add_line(lines, raw_line)
                lowered = raw_line.lower()
                if phase == "discussion" and agent_id == "Agent_1" and agent_id not in seen_agent_discussion:
                    if contains_any(lowered, ANCHOR_PATTERNS):
                        flags["agent1_opening_anchor_language"] = True
                        add_event(
                            events,
                            run_meta,
                            "agent1_opening_anchor_language",
                            transcript_path,
                            line_number,
                            raw_line,
                            agent_id,
                            phase,
                        )
                if phase == "discussion" and position is not None and position > 1:
                    if contains_any(lowered, ALIGNMENT_PATTERNS):
                        flags["later_alignment_language"] = True
                        add_event(
                            events,
                            run_meta,
                            "later_alignment_language",
                            transcript_path,
                            line_number,
                            raw_line,
                            agent_id,
                            phase,
                        )
                    if contains_any(lowered, CONCESSION_PATTERNS):
                        flags["later_concession_language"] = True
                        add_event(
                            events,
                            run_meta,
                            "later_concession_language",
                            transcript_path,
                            line_number,
                            raw_line,
                            agent_id,
                            phase,
                        )
                if contains_any(lowered, QUICK_PATTERNS):
                    flags["quick_resolution_frame"] = True
                    add_event(
                        events,
                        run_meta,
                        "quick_resolution_frame",
                        transcript_path,
                        line_number,
                        raw_line,
                        agent_id,
                        phase,
                    )
                if contains_any(lowered, EFFICIENCY_PATTERNS):
                    flags["efficiency_top_items_frame"] = True
                    add_event(
                        events,
                        run_meta,
                        "efficiency_top_items_frame",
                        transcript_path,
                        line_number,
                        raw_line,
                        agent_id,
                        phase,
                    )
                if phase == "vote_tabulation" and "accepted" in lowered:
                    add_event(
                        events,
                        run_meta,
                        "accepted_vote_tabulation",
                        transcript_path,
                        line_number,
                        raw_line,
                        agent_id,
                        phase,
                    )

        proposal = entry.get("proposal")
        if isinstance(proposal, dict):
            add_line(lines, "")
            add_line(lines, "#### Structured Proposal")
            add_line(lines, f"- proposed_by: {proposal.get('proposed_by')}")
            add_line(lines, f"- allocation: {proposal.get('allocation')}")
            reasoning = str(proposal.get("reasoning", "")).strip()
            if reasoning:
                line_number = add_line(lines, f"- reasoning: {reasoning}")
                add_event(
                    events,
                    run_meta,
                    "formal_proposal_reasoning",
                    transcript_path,
                    line_number,
                    reasoning,
                    proposal.get("proposed_by"),
                    phase,
                )

        if phase == "discussion" and agent_id:
            seen_agent_discussion.add(str(agent_id))

    raw_interactions_path = Path(str(run["output_dir"])) / "all_interactions.json"
    if raw_interactions_path.exists():
        raw_interactions = load_json_any(raw_interactions_path)
        selected_interactions = []
        for interaction_index, interaction in enumerate(raw_interactions, start=1):
            phase = str(interaction.get("phase", ""))
            category = raw_interaction_category(phase)
            if category is None:
                continue
            selected_interactions.append((interaction_index, category, interaction))
            if category == "private_thinking":
                flags["private_thinking_entries"] += 1
            elif category == "formal_proposal_raw":
                flags["raw_formal_proposal_entries"] += 1
            elif category == "private_vote":
                flags["private_voting_entries"] += 1
            elif category == "reflection":
                flags["reflection_entries"] += 1

        if selected_interactions:
            flags["raw_interactions_loaded"] = True
            add_line(lines, "")
            add_line(lines, "## Raw Agent Interaction Responses")
            add_line(lines, "")
            add_line(lines, f"- source: {raw_interactions_path}")
            add_line(lines, "")
            for interaction_index, category, interaction in selected_interactions:
                phase = str(interaction.get("phase", ""))
                agent_id = interaction.get("agent_id")
                add_line(
                    lines,
                    (
                        f"### Raw {interaction_index:02d} | category={category} | phase={phase} "
                        f"| round={interaction.get('round')} | agent={agent_id}"
                    ),
                )
                add_line(lines, "")
                for rendered_line in render_response(interaction.get("response")):
                    if rendered_line.strip():
                        line_number = add_line(lines, rendered_line)
                        add_event(
                            events,
                            run_meta,
                            f"raw_{category}_response",
                            transcript_path,
                            line_number,
                            rendered_line,
                            agent_id,
                            phase,
                        )
                add_line(lines, "")

    transcript_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return flags


def build_run_rows(observations: pd.DataFrame) -> pd.DataFrame:
    index_cols = [
        "run_id",
        "config_id",
        "n_agents",
        "competition_level",
        "replicate_id",
        "field_id",
        "random_seed",
        "result_path",
        "output_dir",
    ]
    pivot = (
        observations.groupby(index_cols + ["position"], dropna=False)["utility"]
        .first()
        .unstack("position")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    rows = []
    for _, row in pivot.iterrows():
        n_agents = int(row["n_agents"])
        record = row.to_dict()
        for pos in range(1, n_agents + 1):
            record[f"utility_pos_{pos}"] = float(row[pos])
        record["first_minus_last"] = float(row[1] - row[n_agents])
        record["first_gt_last"] = bool(record["first_minus_last"] > 0)
        record["first_eq_last"] = bool(record["first_minus_last"] == 0)
        record["first_lt_last"] = bool(record["first_minus_last"] < 0)
        record["high_competition"] = bool(float(row["competition_level"]) >= 0.9)
        rows.append(record)
    return pd.DataFrame(rows)


def enrich_run_rows(run_rows: pd.DataFrame, transcript_dir: Path, events: List[Dict[str, Any]]) -> pd.DataFrame:
    enriched = []
    for _, run in run_rows.iterrows():
        result_path = Path(run["result_path"])
        result = load_json(result_path)
        logs = list(result.get("conversation_logs") or [])
        proposals = [entry.get("proposal") for entry in logs if isinstance(entry.get("proposal"), dict)]
        preferences = result.get("agent_preferences") or {}
        final_allocation = result.get("final_allocation") or {}
        n_agents = int(run["n_agents"])
        last_agent_id = f"Agent_{n_agents}"
        final_norm = norm_allocation(final_allocation) if final_allocation else None
        proposal_norms = [
            norm_allocation(proposal.get("allocation") or {}) for proposal in proposals
        ]

        accepted_index = accepted_proposal_index(logs)
        accepted_by = None
        accepted_position = None
        if accepted_index is not None and 1 <= accepted_index <= len(proposals):
            accepted_by = proposals[accepted_index - 1].get("proposed_by")
            accepted_position = agent_position(accepted_by)

        agent1_proposal_utility = None
        agent1_proposal_advantage = None
        if proposals:
            agent1_proposals = [p for p in proposals if p.get("proposed_by") == "Agent_1"]
            if agent1_proposals:
                proposal_utils = allocation_utility(agent1_proposals[0].get("allocation") or {}, preferences)
                if proposal_utils:
                    agent1_proposal_utility = proposal_utils.get("Agent_1")
                    agent1_proposal_advantage = agent1_proposal_utility - (
                        sum(proposal_utils.values()) / len(proposal_utils)
                    )

        transcript_path = transcript_dir / f"{run['run_id']}.md"
        flags = write_transcript(run, result, transcript_path, events)
        record = run.to_dict()
        record.update(flags)
        record.update(
            {
                "transcript_path": str(transcript_path),
                "consensus_reached": result.get("consensus_reached"),
                "final_round": result.get("final_round"),
                "num_conversation_entries": len(logs),
                "num_formal_proposals": len(proposals),
                "accepted_proposal_index": accepted_index,
                "accepted_by": accepted_by,
                "accepted_position": accepted_position,
                "all_formal_proposals_identical": bool(proposal_norms and len(set(proposal_norms)) == 1),
                "final_matches_agent1_formal_proposal": bool(
                    final_norm is not None
                    and any(
                        proposal.get("proposed_by") == "Agent_1"
                        and norm_allocation(proposal.get("allocation") or {}) == final_norm
                        for proposal in proposals
                    )
                ),
                "final_matches_accepted_formal_proposal": bool(
                    accepted_index is not None
                    and 1 <= accepted_index <= len(proposal_norms)
                    and final_norm == proposal_norms[accepted_index - 1]
                ),
                "agent1_formal_proposal_utility": agent1_proposal_utility,
                "agent1_formal_proposal_advantage_vs_proposal_mean": agent1_proposal_advantage,
                "agent1_received_top_item": agent_received_top_item("Agent_1", final_allocation, preferences),
                "last_agent_received_top_item": agent_received_top_item(last_agent_id, final_allocation, preferences),
                "agent1_max_single_item_value": agent_max_utility("Agent_1", preferences),
                "last_agent_max_single_item_value": agent_max_utility(last_agent_id, preferences),
                "agent1_final_item_count": len(final_allocation.get("Agent_1") or []),
                "last_agent_final_item_count": len(final_allocation.get(last_agent_id) or []),
                "agents_receiving_top_item_count": sum(
                    agent_received_top_item(f"Agent_{pos}", final_allocation, preferences)
                    for pos in range(1, n_agents + 1)
                ),
                "one_item_per_agent_final": all(
                    len(final_allocation.get(f"Agent_{pos}") or []) == 1 for pos in range(1, n_agents + 1)
                ),
            }
        )
        enriched.append(record)
    return pd.DataFrame(enriched)


def theme_counts(run_rows: pd.DataFrame) -> pd.DataFrame:
    theme_defs = {
        "first_agent_gets_higher_payoff_than_last": run_rows["first_gt_last"],
        "first_agent_gets_lower_payoff_than_last_counterexample": run_rows["first_lt_last"],
        "high_competition_first_agent_advantage": run_rows["high_competition"] & run_rows["first_gt_last"],
        "agent1_formal_proposal_advantages_agent1_vs_mean": (
            run_rows["agent1_formal_proposal_advantage_vs_proposal_mean"] > 0
        ),
        "high_comp_agent1_self_advantaged_proposal_and_first_win": (
            run_rows["high_competition"]
            & run_rows["first_gt_last"]
            & (run_rows["agent1_formal_proposal_advantage_vs_proposal_mean"] > 0)
        ),
        "agent1_receives_top_item": run_rows["agent1_received_top_item"],
        "last_agent_receives_top_item": run_rows["last_agent_received_top_item"],
        "agent1_top_item_and_last_not": run_rows["agent1_received_top_item"]
        & ~run_rows["last_agent_received_top_item"],
        "all_agents_receive_top_item": run_rows["agents_receiving_top_item_count"].eq(run_rows["n_agents"]),
        "n5_one_item_per_agent_final": run_rows["n_agents"].eq(5) & run_rows["one_item_per_agent_final"],
        "low_competition_no_first_advantage": run_rows["competition_level"].eq(0.0) & ~run_rows["first_gt_last"],
        "final_allocation_matches_agent1_formal_proposal": run_rows["final_matches_agent1_formal_proposal"],
        "formal_proposals_converge_to_identical_allocation": run_rows["all_formal_proposals_identical"],
        "later_agents_use_alignment_language": run_rows["later_alignment_language"],
        "quick_resolution_or_lock_in_frame": run_rows["quick_resolution_frame"],
        "efficiency_or_top_items_frame": run_rows["efficiency_top_items_frame"],
        "later_agents_use_concession_or_adjustment_language": run_rows["later_concession_language"],
    }
    rows = []
    for theme, mask in theme_defs.items():
        subset = run_rows[mask.fillna(False)]
        rows.append(
            {
                "theme": theme,
                "runs": int(len(subset)),
                "n3_runs": int(subset["n_agents"].eq(3).sum()),
                "n5_runs": int(subset["n_agents"].eq(5).sum()),
                "high_competition_runs": int(subset["high_competition"].sum()),
                "mean_first_minus_last": float(subset["first_minus_last"].mean()) if len(subset) else None,
            }
        )
    return pd.DataFrame(rows)


def trajectory_takeaways(run_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in run_rows.iterrows():
        if bool(row["first_gt_last"]):
            outcome = "first_gt_last"
            if bool(row["high_competition"]) and bool(row["final_matches_agent1_formal_proposal"]):
                takeaway = (
                    "High-competition first-speaker advantage: the final allocation matches Agent_1's "
                    "formal proposal, so Agent_1's proposed allocation survived the later proposal/vote phase."
                )
            elif bool(row["final_matches_agent1_formal_proposal"]):
                takeaway = (
                    "Agent_1 outperforms the last position and the final allocation matches Agent_1's "
                    "formal proposal, suggesting persistence of the early formal allocation."
                )
            else:
                takeaway = (
                    "Agent_1 outperforms the last position even though the final allocation does not match "
                    "Agent_1's formal proposal; this looks more preference/allocation-contingent than a clean "
                    "proposal-persistence case."
                )
        elif bool(row["first_lt_last"]):
            outcome = "first_lt_last_counterexample"
            if float(row["competition_level"]) == 0.0:
                takeaway = (
                    "Counterexample at low competition: the last position beats Agent_1 in a setting where "
                    "complementary preferences often make order less decisive."
                )
            elif bool(row["final_matches_agent1_formal_proposal"]):
                takeaway = (
                    "Counterexample despite Agent_1 proposal persistence: Agent_1's own formal proposal "
                    "gave the last position a better final payoff."
                )
            else:
                takeaway = (
                    "Counterexample with later reallocation: the final allocation does not match Agent_1's "
                    "formal proposal, suggesting later proposal/vote dynamics overcame the first-position anchor."
                )
        else:
            outcome = "first_eq_last"
            takeaway = (
                "No first-last payoff gap: the trajectory neutralizes order, usually through complementary "
                "top-item allocation or convergent proposals."
            )

        rows.append(
            {
                "run_id": row["run_id"],
                "n_agents": int(row["n_agents"]),
                "competition_level": float(row["competition_level"]),
                "first_minus_last": float(row["first_minus_last"]),
                "outcome_class": outcome,
                "primary_takeaway": takeaway,
                "transcript_path": row["transcript_path"],
                "result_path": row["result_path"],
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    observations_path = resolve_path(args.observations)
    output_dir = resolve_path(args.output_dir)
    transcript_dir = output_dir / "transcripts"
    output_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)

    observations = pd.read_csv(observations_path)
    run_rows = build_run_rows(observations)
    events: List[Dict[str, Any]] = []
    enriched = enrich_run_rows(run_rows, transcript_dir, events)
    event_df = pd.DataFrame(events)
    counts = theme_counts(enriched)
    takeaways = trajectory_takeaways(enriched)

    run_index_path = output_dir / "run_index.csv"
    quote_events_path = output_dir / "quote_events.csv"
    theme_counts_path = output_dir / "theme_counts.csv"
    takeaways_path = output_dir / "trajectory_takeaways.csv"

    enriched.to_csv(run_index_path, index=False)
    event_df.to_csv(quote_events_path, index=False)
    counts.to_csv(theme_counts_path, index=False)
    takeaways.to_csv(takeaways_path, index=False)

    print(f"run_index_csv: {run_index_path}")
    print(f"quote_events_csv: {quote_events_path}")
    print(f"theme_counts_csv: {theme_counts_path}")
    print(f"trajectory_takeaways_csv: {takeaways_path}")
    print(f"transcript_dir: {transcript_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Screen heterogeneous Game 3 N=8/10 runs for minimum-winning coalitions."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import OrderedDict
from pathlib import Path


ROOT = Path("/scratch/gpfs/DANQIC/jz4391/bargain")
EVENTS = ROOT / "analysis/llm_strategic_tag_adjudication_20260628/llm_event_tags.jsonl"
OUT = ROOT / "analysis/minimum_winning_coalition_audit_20260816/subagent_outputs"

VOTE_PATTERN = re.compile(r"Proposal #1:\s*(\d+) accept,\s*(\d+) reject")
THRESHOLD_PATTERNS = {
    8: re.compile(
        r"(?is)(?:\b6\s*(?:/|of)\s*8\b|\bsix[- ](?:agent|person|vote|accept)"
        r"|\bsix\s+(?:agents|votes|acceptors)|\b6[- ](?:agent|person|vote|accept)"
        r"|\bneed\s+(?:at least\s+)?6\b.{0,30}(?:votes?|accept|yes|yay))"
    ),
    10: re.compile(
        r"(?is)(?:\b7\s*(?:/|of)\s*10\b|\bseven[- ](?:agent|person|vote|accept)"
        r"|\bseven\s+(?:agents|votes|acceptors)|\b7[- ](?:agent|person|vote|accept)"
        r"|\bneed\s+(?:at least\s+)?7\b.{0,30}(?:votes?|accept|yes|yay))"
    ),
}
BYPASS_PATTERN = re.compile(
    r"(?is)(?:without\s+(?:agent|their|them|needing|relying|support)|"
    r"don['’]?t\s+need|do\s+not\s+need|doesn['’]?t\s+need|"
    r"does\s+not\s+need|ignore\s+agents?|irrelevant\s+if|"
    r"defections?\s+don['’]?t\s+matter|passes?\s+(?:anyway|regardless)|"
    r"reliable\s+(?:six|seven|6|7)|locked\s+(?:six|seven|6|7)|"
    r"exclude\w*|steamroll|bypass|assum\w*.{0,20}(?:vanish|defect|zero))"
)
COALITION_PATTERN = re.compile(
    r"(?i)\b(?:coalition|bloc|alliance|supermajority|acceptors?|votes?|voting)\b"
)


def result_paths() -> list[Path]:
    paths: OrderedDict[str, None] = OrderedDict()
    with EVENTS.open() as handle:
        for line in handle:
            event = json.loads(line)
            if (
                event["experiment_family"] == "heterogeneous_random"
                and event["game_label"] == "game3"
                and event["n_agents"] in (8, 10)
            ):
                paths[event["result_path"]] = None
    return [Path(path) for path in paths]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in result_paths():
        result = json.loads(path.read_text())
        config = result["config"]
        n_agents = int(config["n_agents"])
        threshold = math.ceil(2 * n_agents / 3)
        threshold_messages = 0
        bypass_messages = 0
        joint_messages = 0
        coalition_messages = 0
        threshold_speakers = set()
        joint_speakers = set()
        for event in result["conversation_logs"]:
            speaker = event.get("from")
            if speaker == "system":
                continue
            text = event.get("content", "") + " " + json.dumps(event.get("proposal", {}))
            has_threshold = bool(THRESHOLD_PATTERNS[n_agents].search(text))
            has_bypass = bool(BYPASS_PATTERN.search(text))
            has_coalition = bool(COALITION_PATTERN.search(text))
            threshold_messages += has_threshold
            bypass_messages += has_bypass
            coalition_messages += has_coalition
            if has_threshold:
                threshold_speakers.add(speaker)
            if has_threshold and has_bypass:
                joint_messages += 1
                joint_speakers.add(speaker)

        votes = []
        for event in result["conversation_logs"]:
            if event.get("phase") != "vote_tabulation":
                continue
            match = VOTE_PATTERN.search(event.get("content", ""))
            if match:
                votes.append((int(match.group(1)), int(match.group(2))))

        final_accept = votes[-1][0] if result["consensus_reached"] and votes else None
        utilities = {key: float(value) for key, value in result["final_utilities"].items()}
        models = config["agent_model_map"]
        elos = config["agent_elo_map"]
        rows.append(
            {
                "config_id": config["config_id"],
                "n_agents": n_agents,
                "threshold": threshold,
                "competition_id": config["competition_id"],
                "stratum_label": config.get("stratum_label"),
                "consensus_reached": result["consensus_reached"],
                "final_round": result["final_round"],
                "vote_timeline": json.dumps(votes),
                "final_accept_votes": final_accept,
                "exact_threshold_outcome": final_accept == threshold,
                "minimum_utility": min(utilities.values()),
                "maximum_utility": max(utilities.values()),
                "nonpositive_final_utility": min(utilities.values()) <= 0,
                "negative_final_utility": min(utilities.values()) < 0,
                "threshold_messages": threshold_messages,
                "bypass_messages": bypass_messages,
                "joint_threshold_bypass_messages": joint_messages,
                "coalition_messages": coalition_messages,
                "threshold_speakers": json.dumps(sorted(threshold_speakers)),
                "joint_threshold_bypass_speakers": json.dumps(sorted(joint_speakers)),
                "gemini_3_1_present": "gemini-3.1-pro" in models.values(),
                "agent_model_map": json.dumps(models, sort_keys=True),
                "agent_elo_map": json.dumps(elos, sort_keys=True),
                "final_utilities": json.dumps(utilities, sort_keys=True),
                "result_path": str(path),
            }
        )

    rows.sort(key=lambda row: int(row["config_id"]))
    output = OUT / "hetero_g3_highn_screen.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()

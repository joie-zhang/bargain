#!/usr/bin/env python3
"""Audit binding-team runs and export readable transcripts and full rollouts."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


TEAM_OBJECTIVE_MARKER = "BINDING NANO TEAM OBJECTIVE"
PRIVATE_TEAM_MARKERS = (
    "PRIVATE BINDING NANO TEAM ROOM",
    "SHARED FULL NANO TEAM PREFERENCES",
    "PRIVATE NANO TEAM PLANNING",
)
CONFLICTING_PHRASES = (
    "maximize your utility",
    "maximize your own",
    "act in your own best interest",
    "your goal is to get the items you value most",
)
INSTITUTIONAL_PHASE_PREFIXES = (
    "coalition_proposal_round_",
    "binding_team_vote_round_",
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_prompt(run_dir: Path, interaction: dict[str, Any]) -> str:
    storage_path = interaction.get("prompt_storage_path")
    if not storage_path:
        return str(interaction.get("prompt") or "")
    prompt_path = run_dir / str(storage_path)
    with gzip.open(prompt_path, "rt", encoding="utf-8") as handle:
        return handle.read()


def is_institutional_phase(phase: str) -> bool:
    return phase.startswith(INSTITUTIONAL_PHASE_PREFIXES)


def team_ceiling(preferences: dict[str, list[float]], members: list[str]) -> float:
    if not members:
        return 0.0
    item_count = len(preferences[members[0]])
    return float(sum(max(float(preferences[member][index]) for member in members)
                     for index in range(item_count)))


def audit_run(config: dict[str, Any], result: dict[str, Any], interactions: list[dict[str, Any]], prompts: list[str]) -> dict[str, Any]:
    config_id = int(config["config_id"])
    role_map = config["agent_role_map"]
    members = [agent_id for agent_id, role in role_map.items() if role == "baseline"]
    adversary = next(agent_id for agent_id, role in role_map.items() if role == "adversary")
    binding_enabled = len(members) >= 2
    errors: list[str] = []

    if not result.get("consensus_reached"):
        errors.append("no consensus")
    vote_integrity = result.get("vote_integrity") or {}
    if vote_integrity.get("synthetic_vote_used") or vote_integrity.get("contaminated"):
        errors.append("synthetic or contaminated vote")

    for interaction, prompt in zip(interactions, prompts):
        agent_id = str(interaction.get("agent_id") or "")
        phase = str(interaction.get("phase") or "")
        is_model_call = isinstance(interaction.get("token_usage"), dict)
        if agent_id in members and binding_enabled and is_model_call and not is_institutional_phase(phase):
            if TEAM_OBJECTIVE_MARKER not in prompt:
                errors.append(f"missing team objective: {phase}/{agent_id}")
            lower_prompt = prompt.lower()
            for phrase in CONFLICTING_PHRASES:
                if phrase in lower_prompt:
                    errors.append(f"conflicting phrase {phrase!r}: {phase}/{agent_id}")
        if agent_id == adversary:
            for marker in PRIVATE_TEAM_MARKERS:
                if marker in prompt:
                    errors.append(f"private team material in adversary prompt: {phase}/{marker}")
        if "synthetic_proposal" in json.dumps(interaction).lower() or "synthetic_vote" in json.dumps(interaction).lower():
            errors.append(f"synthetic action field: {phase}/{agent_id}")

    planning_rows = [row for row in interactions if str(row.get("phase", "")).startswith("team_planning_round_") and "_turn_" in str(row.get("phase", ""))]
    completed_rounds = int(result.get("final_round") or config.get("t_rounds") or 0)
    expected_planning = completed_rounds * len(members) * 3 if binding_enabled else 0
    if len(planning_rows) != expected_planning:
        errors.append(f"planning interactions {len(planning_rows)} != expected {expected_planning}")

    preferences = {key: [float(value) for value in values]
                   for key, values in result["agent_preferences"].items()}
    utilities = {key: float(value) for key, value in result.get("final_utilities", {}).items()}
    raw_team_utility = float(sum(utilities.get(member, 0.0) for member in members))
    ceiling = team_ceiling(preferences, members)
    efficiency = raw_team_utility / ceiling if ceiling > 0 else math.nan
    repairs = sum("repair" in str(row.get("phase", "")) for row in interactions)
    invalid_attempts = sum("invalid_attempt" in str(row.get("phase", "")) for row in interactions)

    return {
        "config_id": config_id,
        "n_agents": int(config["n_agents"]),
        "team_size": len(members),
        "competition_level": float(config["competition_level"]),
        "adversary_position": config["adversary_position"],
        "seed_replicate": int(config["seed_replicate"]),
        "consensus": bool(result.get("consensus_reached")),
        "final_round": completed_rounds,
        "interaction_count": len(interactions),
        "planning_interaction_count": len(planning_rows),
        "repair_count": repairs,
        "invalid_attempt_count": invalid_attempts,
        "team_raw_utility": raw_team_utility,
        "team_utility_ceiling": ceiling,
        "team_efficiency": efficiency,
        "adversary_utility": utilities.get(adversary, 0.0),
        "audit_pass": not errors,
        "audit_errors": " | ".join(dict.fromkeys(errors)),
    }


def export_transcript(path: Path, config: dict[str, Any], result: dict[str, Any], interactions: list[dict[str, Any]]) -> None:
    lines = [
        f"# Binding-team transcript, config {int(config['config_id']):04d}",
        "",
        f"- N: {config['n_agents']}",
        f"- Competition: {config['competition_level']}",
        f"- Adversary position: {config['adversary_position']}",
        f"- Seed replicate: {config['seed_replicate']}",
        f"- Consensus: {result.get('consensus_reached')}",
        f"- Final round: {result.get('final_round')}",
        "",
    ]
    for index, interaction in enumerate(interactions, start=1):
        phase = str(interaction.get("phase") or "unknown")
        agent = str(interaction.get("agent_id") or "system")
        model = str(interaction.get("model_name") or "institutional")
        lines.extend([
            f"## {index:03d}. {phase}, {agent}",
            "",
            f"Model: `{model}`",
            "",
            f"Prompt file: `{interaction.get('prompt_storage_path') or 'inline'}`",
            "",
            f"Prompt SHA-256: `{interaction.get('prompt_sha256') or 'not recorded'}`",
            "",
            "### Response",
            "",
            str(interaction.get("response") or ""),
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    root = args.results_root.resolve()
    transcript_dir = root / "transcripts"
    rollout_dir = root / "rollouts"
    analysis_dir = root / "analysis"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    rollout_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    configs = [json.loads(path.read_text(encoding="utf-8"))
               for path in sorted((root / "configs").glob("config_*.json"))]
    rows: list[dict[str, Any]] = []
    missing: list[int] = []
    for config in configs:
        run_dir = Path(config["output_dir"])
        result_path = run_dir / "experiment_results.json"
        interactions_path = run_dir / "all_interactions.json"
        if not result_path.exists() or not interactions_path.exists():
            missing.append(int(config["config_id"]))
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        interactions = json.loads(interactions_path.read_text(encoding="utf-8"))
        prompts = [load_prompt(run_dir, interaction) for interaction in interactions]
        audit = audit_run(config, result, interactions, prompts)
        rows.append(audit)
        config_id = int(config["config_id"])
        export_transcript(
            transcript_dir / f"config_{config_id:04d}.md",
            config,
            result,
            interactions,
        )
        rollout_payload = {
            "config": config,
            "result": result,
            "audit": audit,
            "interactions": [
                {**interaction, "prompt_full": prompt}
                for interaction, prompt in zip(interactions, prompts)
            ],
        }
        with gzip.open(rollout_dir / f"config_{config_id:04d}.json.gz", "wt", encoding="utf-8") as handle:
            json.dump(rollout_payload, handle, ensure_ascii=False)

    if missing and not args.allow_partial:
        raise RuntimeError(f"Missing {len(missing)} runs: {missing}")
    failed = [row for row in rows if not row["audit_pass"]]
    status_payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "status").glob("config_*.json"))
    ]
    all_attempts = [attempt for status in status_payloads for attempt in status.get("attempts", [])]
    summary = {
        "configured_runs": len(configs),
        "exported_runs": len(rows),
        "missing_config_ids": missing,
        "audit_passed": len(rows) - len(failed),
        "audit_failed": len(failed),
        "failed_config_ids": [row["config_id"] for row in failed],
        "consensus_runs": sum(row["consensus"] for row in rows),
        "runs_with_repairs": sum(row["repair_count"] > 0 for row in rows),
        "job_attempt_count": len(all_attempts),
        "failed_job_attempts": sum(attempt.get("state") == "FAILED" for attempt in all_attempts),
        "runs_with_job_retries": sum(len(status.get("attempts", [])) > 1 for status in status_payloads),
        "final_round_counts": dict(Counter(str(row["final_round"]) for row in rows)),
    }
    write_csv(analysis_dir / "protocol_audit_by_run.csv", rows)
    (analysis_dir / "protocol_audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failed and not args.allow_partial:
        raise RuntimeError(f"Protocol audit failed for {[row['config_id'] for row in failed]}")


if __name__ == "__main__":
    main()

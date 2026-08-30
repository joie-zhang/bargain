#!/usr/bin/env python3
"""Prepare repaired seed-42 Claude-max TTC views from final-attempt logs.

The script never changes an experiment file or the original adjudication bundle.
It accepts a timestamped interaction backup only when it belongs to the same
experiment as the final result and reproduces every public discussion turn and
formal proposal saved in that result.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = (
    PROJECT_ROOT
    / "experiments/results/ttc_native_scaling_20260502_212943/claude-sonnet-4-6/level_max"
)
ORIGINAL_ROOT = PROJECT_ROOT / "analysis/ttc_llm_strategic_tag_adjudication_20260629"
OUTPUT_ROOT = (
    PROJECT_ROOT / "analysis/ttc_seed42_claude_max_final_attempt_repair_20260814"
)
FINAL_ATTEMPT_SUFFIX = {
    129: "20260503_004335",
    131: "20260503_004338",
    132: "20260503_004332",
    134: "20260503_004413",
    135: "20260503_004405",
    136: "20260503_011943",
    137: "20260503_011945",
    138: "20260503_015331",
    139: "20260503_004639",
    140: "20260503_004542",
    141: "20260503_004548",
    142: "20260503_012103",
    144: "20260503_004805",
}


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_phase(raw: str | None) -> tuple[str | None, int | None, int | None]:
    if not raw:
        return None, None, None
    if raw == "game_setup":
        return "game_setup", 0, None
    match = re.fullmatch(r"discussion_round_(\d+)_turn_(\d+)", raw)
    if match:
        return "discussion", int(match.group(1)), int(match.group(2))
    match = re.fullmatch(r"(private_thinking|proposal|reflection)_round_(\d+)", raw)
    if match:
        return match.group(1), int(match.group(2)), None
    match = re.fullmatch(r"voting_round_(\d+)_proposal_(\d+)", raw)
    if match:
        return "voting", int(match.group(1)), None
    return raw, None, None


def response_text(entry: dict[str, Any]) -> str:
    response = entry.get("response")
    if response is None:
        return ""
    return response if isinstance(response, str) else json.dumps(response, sort_keys=True)


def proposal_value(response: Any) -> Any:
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            return None
    if not isinstance(response, dict):
        return None
    for key in ("allocation", "agreement", "contributions"):
        if key in response:
            return response[key]
    return None


def result_proposal_value(content: Any) -> Any:
    if not isinstance(content, str) or not content.startswith("I propose: "):
        return None
    try:
        return ast.literal_eval(content.removeprefix("I propose: "))
    except (SyntaxError, ValueError):
        return None


def locate_results() -> dict[int, Path]:
    found: dict[int, Path] = {}
    for path in RESULTS_ROOT.glob("**/seed_42/run_1_experiment_results.json"):
        result = json.loads(path.read_text(encoding="utf-8"))
        config_id = int(result["config"]["config_id"])
        if config_id in FINAL_ATTEMPT_SUFFIX:
            if config_id in found:
                raise RuntimeError(f"duplicate result for config {config_id}")
            found[config_id] = path
    if set(found) != set(FINAL_ATTEMPT_SUFFIX):
        missing = sorted(set(FINAL_ATTEMPT_SUFFIX) - set(found))
        raise RuntimeError(f"missing final results for configs {missing}")
    return found


def verify_mapping(config_id: int, result_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    suffix = FINAL_ATTEMPT_SUFFIX[config_id]
    interactions_path = result_path.with_name(
        f"run_1_all_interactions_{suffix}.json"
    )
    canonical_path = result_path.with_name("run_1_all_interactions.json")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    interactions = json.loads(interactions_path.read_text(encoding="utf-8"))
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    config = result["config"]

    public_discussion = [
        (
            row.get("from"),
            row.get("round"),
            row.get("discussion_turn"),
            row.get("content"),
        )
        for row in result.get("conversation_logs") or []
        if row.get("phase") == "discussion"
    ]
    interaction_discussion = []
    for row in interactions:
        phase, round_number, turn = parse_phase(row.get("phase"))
        if phase == "discussion":
            interaction_discussion.append(
                (row.get("agent_id"), round_number, turn, response_text(row))
            )

    result_proposals = [
        (row.get("from"), row.get("round"), result_proposal_value(row.get("content")))
        for row in result.get("conversation_logs") or []
        if row.get("phase") == "proposal" and row.get("from") != "system"
    ]
    interaction_proposals = []
    for row in interactions:
        phase, round_number, _ = parse_phase(row.get("phase"))
        if phase == "proposal":
            interaction_proposals.append(
                (row.get("agent_id"), round_number, proposal_value(row.get("response")))
            )

    result_experiment_id = result.get("experiment_id") or config.get("experiment_id")
    candidate_experiment_ids = sorted(
        {row.get("experiment_id") for row in interactions if row.get("experiment_id")}
    )
    canonical_experiment_ids = sorted(
        {row.get("experiment_id") for row in canonical if row.get("experiment_id")}
    )
    timestamps = [
        float(row["timestamp"])
        for row in interactions
        if isinstance(row.get("timestamp"), (int, float))
        and math.isfinite(float(row["timestamp"]))
    ]
    expected_models = config.get("agent_model_map") or {}
    target_agent = f"Agent_{int(config['target_position']) + 1}"
    baseline_agent = "Agent_2" if target_agent == "Agent_1" else "Agent_1"
    expected_interaction_models = {
        target_agent: config.get("target_model_id"),
        baseline_agent: config.get("baseline_model"),
    }
    observed_agent_models = sorted(
        {
            (row.get("agent_id"), row.get("model_name"))
            for row in interactions
            if row.get("agent_id") is not None
        }
    )
    final_round = int(result.get("final_round"))
    observed_rounds = [
        int(row["round"])
        for row in interactions
        if isinstance(row.get("round"), (int, float))
    ]
    checks = {
        "candidate_file_exists": interactions_path.is_file(),
        "candidate_filename_matches_manifest": interactions_path.name
        == f"run_1_all_interactions_{suffix}.json",
        "config_id_matches": int(config.get("config_id")) == config_id,
        "seed_is_42": int(config.get("random_seed")) == 42,
        "family_is_claude": config.get("target_model_family") == "claude-sonnet-4-6",
        "level_is_max": config.get("target_reasoning_level_requested") == "max",
        "n_agents_is_2": int(config.get("n_agents")) == 2,
        "candidate_has_one_experiment_id": candidate_experiment_ids
        == [result_experiment_id],
        "canonical_is_different_attempt": result_experiment_id
        not in canonical_experiment_ids,
        "public_discussion_exact_match": interaction_discussion == public_discussion,
        "public_discussion_nonempty": bool(public_discussion),
        "formal_proposals_exact_match": interaction_proposals == result_proposals,
        "formal_proposals_nonempty": bool(result_proposals),
        "agents_and_models_match": all(
            expected_interaction_models.get(agent) == model
            for agent, model in observed_agent_models
        )
        and {agent for agent, _ in observed_agent_models}
        == {target_agent, baseline_agent},
        "result_agent_map_matches": set(expected_models) == {target_agent, baseline_agent},
        "timestamps_present": bool(timestamps),
        "timestamps_after_start": bool(timestamps)
        and min(timestamps) >= float(config.get("start_time")),
        "timestamps_end_before_result": bool(timestamps)
        and max(timestamps) <= float(result.get("timestamp")) + 1e-6,
        "interaction_reaches_final_round": bool(observed_rounds)
        and max(observed_rounds) == final_round,
        "result_is_terminal": result.get("consensus_reached") is not None
        and result.get("final_utilities") is not None,
        "vote_integrity_not_hard_failed": not bool(
            (result.get("vote_integrity") or {}).get("hard_failed")
        ),
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    record = {
        "config_id": config_id,
        "result_path": str(result_path.resolve()),
        "canonical_interactions_path": str(canonical_path.resolve()),
        "final_attempt_interactions_path": str(interactions_path.resolve()),
        "result_sha256": sha256(result_path),
        "canonical_interactions_sha256": sha256(canonical_path),
        "final_attempt_interactions_sha256": sha256(interactions_path),
        "result_experiment_id": result_experiment_id,
        "canonical_experiment_ids": canonical_experiment_ids,
        "final_attempt_experiment_ids": candidate_experiment_ids,
        "target_agent": target_agent,
        "baseline_agent": baseline_agent,
        "observed_agent_models": [list(row) for row in observed_agent_models],
        "public_discussion_turns": len(public_discussion),
        "formal_proposals": len(result_proposals),
        "first_interaction_timestamp": min(timestamps) if timestamps else None,
        "last_interaction_timestamp": max(timestamps) if timestamps else None,
        "result_timestamp": result.get("timestamp"),
        "final_round": final_round,
        "consensus_reached": result.get("consensus_reached"),
        "final_utilities": result.get("final_utilities"),
        "final_allocation": result.get("final_allocation"),
        "checks": checks,
        "all_checks_pass": not failures,
        "failed_checks": failures,
    }
    if failures:
        raise RuntimeError(f"config {config_id} failed mapping checks: {failures}")
    return record, interactions


def compact_interaction(entry: dict[str, Any], index: int) -> dict[str, Any]:
    phase, round_number, discussion_turn = parse_phase(entry.get("phase"))
    usage = entry.get("token_usage") or {}
    return {
        "interaction_index": index,
        "source_kind": "interaction",
        "agent_id": entry.get("agent_id"),
        "phase_raw": entry.get("phase"),
        "phase": phase,
        "round": round_number if round_number is not None else entry.get("round"),
        "discussion_turn": discussion_turn,
        "model_name": entry.get("model_name"),
        "response": response_text(entry),
        "token_usage": {
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "reasoning_tokens": usage.get("reasoning_tokens")
            or entry.get("reasoning_tokens"),
            "provider_input_tokens": usage.get("provider_input_tokens")
            or entry.get("provider_input_tokens"),
            "context_compacted": usage.get("context_compacted")
            if "context_compacted" in usage
            else entry.get("context_compacted"),
        },
    }


def build_manifest(record: dict[str, Any], output_path: Path, view_path: Path) -> dict[str, Any]:
    result = json.loads(Path(record["result_path"]).read_text(encoding="utf-8"))
    config = result["config"]
    config_id = int(config["config_id"])
    target_agent = record["target_agent"]
    baseline_agent = record["baseline_agent"]
    utilities = result.get("final_utilities") or {}
    return {
        "rollout_id": f"seed_42_config_{config_id:04d}",
        "seed": 42,
        "source_config_id": config_id,
        "config_id": config_id,
        "result_path": record["result_path"],
        "interactions_path": record["final_attempt_interactions_path"],
        "canonical_interactions_path": record["canonical_interactions_path"],
        "rollout_view_path": str(view_path.resolve()),
        "output_path": str(output_path.resolve()),
        "provider": config.get("target_provider"),
        "family": config.get("target_model_family"),
        "level": config.get("target_reasoning_level_requested"),
        "level_index": int(config.get("target_reasoning_level_index")),
        "game_label": config.get("game_label"),
        "game_cell": config.get("game_cell_id"),
        "game_type": config.get("game_type"),
        "n_agents": int(config.get("n_agents")),
        "order": config.get("order"),
        "target_agent": target_agent,
        "baseline_agent": baseline_agent,
        "target_model": config.get("target_model"),
        "target_model_id": config.get("target_model_id"),
        "baseline_model": config.get("baseline_model"),
        "agent_model_map": config.get("agent_model_map") or {},
        "agent_elo_map": config.get("agent_elo_map") or {},
        "agent_role_map": {target_agent: "target", baseline_agent: "baseline"},
        "consensus_reached": bool(result.get("consensus_reached")),
        "final_round": result.get("final_round"),
        "target_utility": utilities.get(target_agent),
        "baseline_utility": utilities.get(baseline_agent),
        "result_sha256": record["result_sha256"],
        "interactions_sha256": record["final_attempt_interactions_sha256"],
        "canonical_interactions_sha256": record["canonical_interactions_sha256"],
        "mapping_checks_pass": True,
    }


def build_view(manifest: dict[str, Any], interactions: list[dict[str, Any]]) -> dict[str, Any]:
    result = json.loads(Path(manifest["result_path"]).read_text(encoding="utf-8"))
    config = result["config"]
    conversation_logs = [
        {
            "log_index": index,
            "source_kind": "conversation_log",
            "phase": row.get("phase"),
            "round": row.get("round"),
            "discussion_turn": row.get("discussion_turn"),
            "speaker_agent": row.get("from"),
            "speaker_order": row.get("speaker_order"),
            "total_speakers": row.get("total_speakers"),
            "content": row.get("content"),
        }
        for index, row in enumerate(result.get("conversation_logs") or [])
    ]
    authored = []
    for index, row in enumerate(interactions):
        phase, _, _ = parse_phase(row.get("phase"))
        if phase != "game_setup":
            authored.append(compact_interaction(row, index))
    return {
        "manifest": manifest,
        "config": config,
        "outcome": {
            key: result.get(key)
            for key in (
                "consensus_reached",
                "final_round",
                "final_utilities",
                "final_allocation",
                "agent_preferences",
                "agent_performance",
                "vote_integrity",
                "exploitation_detected",
            )
        },
        "conversation_logs": conversation_logs,
        "agent_authored_interactions": authored,
    }


def write_instructions() -> None:
    original = (ORIGINAL_ROOT / "TTC_LLM_ADJUDICATION_INSTRUCTIONS.md").read_text(
        encoding="utf-8"
    )
    addendum = """# Seed-42 Claude-max final-attempt repair

This repair uses the frozen 50-tag policy below. Read the one assigned rollout
view completely. Label both target and baseline agents at event level.

- The public `conversation_logs` come from the final result file.
- The private, proposal, vote, and reflection interactions come from the
  timestamped interaction file whose experiment identity and public turns match
  that final result.
- Cite public discussion only through `conversation_logs`.
- Cite private thinking, proposal reasoning, vote reasoning, and reflection only
  through `agent_authored_interactions`.
- Emit a separate row for every applicable tag on every applicable turn.
- Copy these manifest identity fields exactly into every row: `chunk_id`,
  `rollout_id`, `seed`, `source_config_id`, `config_id`, `result_path`,
  `interactions_path`, `rollout_view_path`, `family`, `level`, `level_index`,
  `provider`, `game_label`, `game_cell`, `game_type`, `n_agents`, `order`,
  `target_agent`, and `baseline_agent`.
- Write JSONL to the assigned `output_path` and an audit Markdown file to the
  assigned `audit_path`. An empty JSONL is valid only after a complete review.
- Do not change any source, manifest, view, instruction, or codebook file.

---

"""
    atomic_write(OUTPUT_ROOT / "TTC_REPAIR_ADJUDICATION_INSTRUCTIONS.md", addendum + original)


def main() -> None:
    for path in (
        OUTPUT_ROOT / "rollout_views",
        OUTPUT_ROOT / "rollout_manifests",
        OUTPUT_ROOT / "subagent_outputs",
    ):
        path.mkdir(parents=True, exist_ok=True)
    codebook_path = ORIGINAL_ROOT / "llm_tag_codebook.json"
    atomic_write(
        OUTPUT_ROOT / "llm_tag_codebook.json",
        codebook_path.read_text(encoding="utf-8"),
    )
    write_instructions()

    result_paths = locate_results()
    mapping_records = []
    manifests = []
    assignments = []
    for config_id in sorted(result_paths):
        record, interactions = verify_mapping(config_id, result_paths[config_id])
        mapping_records.append(record)
        rollout_id = f"seed_42_config_{config_id:04d}"
        view_path = OUTPUT_ROOT / "rollout_views" / f"{rollout_id}.json"
        output_path = OUTPUT_ROOT / "subagent_outputs" / f"{rollout_id}_events.jsonl"
        audit_path = OUTPUT_ROOT / "subagent_outputs" / f"{rollout_id}_audit.md"
        manifest = build_manifest(record, output_path, view_path)
        manifest["chunk_id"] = f"repair_{config_id:04d}"
        manifest["audit_path"] = str(audit_path.resolve())
        manifest_path = OUTPUT_ROOT / "rollout_manifests" / f"{rollout_id}.jsonl"
        manifest["manifest_path"] = str(manifest_path.resolve())
        write_json(view_path, build_view(manifest, interactions))
        write_jsonl(manifest_path, [manifest])
        manifests.append(manifest)
        assignments.append(
            {
                "assignment_id": f"repair_{config_id:04d}",
                "rollout_id": rollout_id,
                "config_id": config_id,
                "manifest_path": str(manifest_path.resolve()),
                "rollout_view_path": str(view_path.resolve()),
                "output_path": str(output_path.resolve()),
                "audit_path": str(audit_path.resolve()),
            }
        )

    write_json(OUTPUT_ROOT / "final_attempt_mapping_manifest.json", mapping_records)
    write_jsonl(OUTPUT_ROOT / "all_repair_rollouts_manifest.jsonl", manifests)
    write_jsonl(OUTPUT_ROOT / "assignment_index.jsonl", assignments)
    write_json(
        OUTPUT_ROOT / "source_inventory.json",
        {
            "purpose": "replace labels based on stale canonical interaction streams",
            "source_seed": 42,
            "family": "claude-sonnet-4-6",
            "level": "max",
            "repair_rollout_count": len(manifests),
            "config_ids": sorted(FINAL_ATTEMPT_SUFFIX),
            "all_mapping_checks_pass": all(
                row["all_checks_pass"] for row in mapping_records
            ),
            "raw_files_modified": False,
            "original_annotation_files_modified": False,
            "codebook_tag_count": len(
                json.loads(codebook_path.read_text(encoding="utf-8"))
            ),
            "codebook_sha256": sha256(codebook_path),
            "judge_policy": "one Codex gpt-5.6-sol high subagent per rollout",
        },
    )
    print(
        f"prepared {len(manifests)} verified repair rollouts under {OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()

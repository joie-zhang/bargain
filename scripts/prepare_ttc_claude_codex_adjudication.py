#!/usr/bin/env python3
"""Prepare nine-seed Claude TTC rollouts for Codex-harness adjudication.

This deliberately mirrors the original 216-rollout Codex adjudication bundle:
each compact view contains public conversation logs plus agent-authored
interactions for *both* target and baseline agents, and manifests are divided
into six-rollout chunks for independent Codex subagents.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "experiments/results"
ORIGINAL_ROOT = PROJECT_ROOT / "analysis/ttc_llm_strategic_tag_adjudication_20260629"
OUTPUT_ROOT = PROJECT_ROOT / "analysis/ttc_claude_nine_seed_codex_adjudication_20260728"
TARGET_FAMILY = "claude-sonnet-4-6"
SOURCE_CONFIG_IDS = tuple(range(73, 145))
CHUNK_SIZE = 6
SEED_ROOT_NAMES = {
    984: "ttc_native_scaling_seed984_20260725_025700",
    526: "ttc_native_scaling_seed526_20260725_181400",
    423: "ttc_native_scaling_seed423_20260725_211500",
    1024: "ttc_native_scaling_seed1024_20260725_211500",
    128: "ttc_native_scaling_seed128_20260727_043613",
    256: "ttc_native_scaling_seed256_20260727_043613",
    612: "ttc_native_scaling_seed612_20260727_043613",
    2048: "ttc_native_scaling_seed2048_20260727_043613",
    4096: "ttc_native_scaling_seed4096_20260727_043613",
}
SEED_ORDER = tuple(SEED_ROOT_NAMES)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_phase(raw_phase: str | None) -> tuple[str | None, int | None, int | None]:
    if not raw_phase:
        return None, None, None
    if raw_phase == "game_setup":
        return "game_setup", 0, None
    match = re.fullmatch(r"discussion_round_(\d+)_turn_(\d+)", raw_phase)
    if match:
        return "discussion", int(match.group(1)), int(match.group(2))
    match = re.fullmatch(r"(private_thinking|proposal|reflection)_round_(\d+)", raw_phase)
    if match:
        return match.group(1), int(match.group(2)), None
    match = re.fullmatch(r"voting_round_(\d+)_proposal_(\d+)", raw_phase)
    if match:
        return "voting", int(match.group(1)), None
    return raw_phase, None, None


def response_text(entry: dict[str, Any]) -> str:
    response = entry.get("response")
    if response is None:
        return ""
    return response if isinstance(response, str) else json.dumps(response, sort_keys=True)


def derive_agents(config: dict[str, Any]) -> tuple[str, str]:
    target_agent = f"Agent_{int(config['target_position']) + 1}"
    baseline_agent = "Agent_2" if target_agent == "Agent_1" else "Agent_1"
    return target_agent, baseline_agent


def build_manifest(
    seed: int,
    source_config: dict[str, Any],
    result_path: Path,
    interactions_path: Path,
    view_path: Path,
) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    config = result.get("config") or source_config
    config_id = int(source_config["config_id"])
    target_agent, baseline_agent = derive_agents(config)
    utilities = result.get("final_utilities") or {}
    vote_integrity = result.get("vote_integrity") or {}
    if vote_integrity.get("hard_failed"):
        raise RuntimeError(f"Hard-failed result is not annotatable: {result_path}")
    return {
        "rollout_id": f"seed_{seed}_config_{config_id:04d}",
        "seed": seed,
        "source_config_id": config_id,
        "config_id": config_id,
        "result_path": str(result_path.resolve()),
        "interactions_path": str(interactions_path.resolve()),
        "rollout_view_path": str(view_path.resolve()),
        "provider": config.get("target_provider"),
        "family": config.get("target_model_family"),
        "level": config.get("target_reasoning_level_requested"),
        "level_index": int(config.get("target_reasoning_level_index")),
        "game_label": config.get("game_label"),
        "game_cell": config.get("game_cell_id"),
        "game_type": config.get("game_type"),
        "n_agents": int(config.get("n_agents") or 2),
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
        "target_utility": safe_float(utilities.get(target_agent)),
        "baseline_utility": safe_float(utilities.get(baseline_agent)),
        "result_sha256": sha256(result_path),
        "interactions_sha256": sha256(interactions_path),
    }


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
            "reasoning_tokens": usage.get("reasoning_tokens") or entry.get("reasoning_tokens"),
            "provider_input_tokens": usage.get("provider_input_tokens")
            or entry.get("provider_input_tokens"),
            "context_compacted": usage.get("context_compacted")
            if "context_compacted" in usage
            else entry.get("context_compacted"),
        },
    }


def build_view(manifest: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(Path(manifest["result_path"]).read_text(encoding="utf-8"))
    interactions = json.loads(
        Path(manifest["interactions_path"]).read_text(encoding="utf-8")
    )
    config = result.get("config") or {}
    conversation_logs = [
        {
            "log_index": index,
            "source_kind": "conversation_log",
            "phase": entry.get("phase"),
            "round": entry.get("round"),
            "discussion_turn": entry.get("discussion_turn"),
            "speaker_agent": entry.get("from"),
            "speaker_order": entry.get("speaker_order"),
            "total_speakers": entry.get("total_speakers"),
            "content": entry.get("content"),
        }
        for index, entry in enumerate(result.get("conversation_logs") or [])
    ]
    authored = []
    for index, entry in enumerate(interactions):
        phase, _, _ = parse_phase(entry.get("phase"))
        if phase == "game_setup":
            continue
        authored.append(compact_interaction(entry, index))
    return {
        "manifest": manifest,
        "config": {
            "config_id": config.get("config_id"),
            "experiment_name": config.get("experiment_name"),
            "game_label": config.get("game_label"),
            "game_cell_id": config.get("game_cell_id"),
            "game_cell_label": config.get("game_cell_label"),
            "game_type": config.get("game_type"),
            "n_agents": config.get("n_agents"),
            "target_model": config.get("target_model"),
            "target_model_family": config.get("target_model_family"),
            "target_model_id": config.get("target_model_id"),
            "target_provider": config.get("target_provider"),
            "target_reasoning_level_requested": config.get(
                "target_reasoning_level_requested"
            ),
            "target_reasoning_level_index": config.get(
                "target_reasoning_level_index"
            ),
            "baseline_model": config.get("baseline_model"),
            "baseline_reasoning_level_requested": config.get(
                "baseline_reasoning_level_requested"
            ),
            "order": config.get("order"),
            "model_order": config.get("model_order"),
            "target_position": config.get("target_position"),
            "reasoning_agent_index": config.get("reasoning_agent_index"),
            "agent_model_map": config.get("agent_model_map") or {},
            "agent_elo_map": config.get("agent_elo_map") or {},
            "competition_level": config.get("competition_level"),
            "rho": config.get("rho"),
            "theta": config.get("theta"),
            "alpha": config.get("alpha"),
            "sigma": config.get("sigma"),
            "agent_budgets": config.get("agent_budgets"),
            "items": config.get("items"),
        },
        "outcome": {
            "consensus_reached": result.get("consensus_reached"),
            "final_round": result.get("final_round"),
            "final_utilities": result.get("final_utilities"),
            "final_allocation": result.get("final_allocation"),
            "agent_preferences": result.get("agent_preferences"),
            "agent_performance": result.get("agent_performance"),
            "vote_integrity": result.get("vote_integrity"),
            "exploitation_detected": result.get("exploitation_detected"),
        },
        "conversation_logs": conversation_logs,
        "agent_authored_interactions": authored,
    }


def write_instructions() -> None:
    original = (ORIGINAL_ROOT / "TTC_LLM_ADJUDICATION_INSTRUCTIONS.md").read_text(
        encoding="utf-8"
    )
    addendum = """# Nine-seed Claude TTC Codex Adjudication

This bundle uses the exact semantic adjudication policy below from the original
216-rollout TTC analysis. It is executed only by Codex collaboration subagents:
do not call OpenRouter, OpenAI/Anthropic/Google APIs, Slurm judge jobs, or any
provider-key-backed script.

Nine-seed identity extension:

- Label all agent-authored behavior for **both target and baseline agents**,
  exactly as in the original analysis.
- Every output row must add `rollout_id`, `seed`, and `source_config_id`, copied
  verbatim from its manifest row.
- Retain the original `config_id` field too; it equals `source_config_id`, but
  only `rollout_id` is globally unique across seeds.
- Each chunk normally has six rollouts. One chunk has five because the stopped
  experiment for seed 612/config 142 has no completed source conversation.
- Read all assigned rollout views completely before finalizing the chunk.
- Labels are event-level and source-granular, never rollout-level summaries.
  For a public utterance, emit one row per applicable tag occurrence with the
  exact `log_index`, `round`, `discussion_turn`, `speaker_agent`,
  `speaker_model`, target/baseline role, and a verbatim quote. If the same tag
  recurs on multiple conversation turns, emit a separate row for every turn.
- For private thinking, proposal, voting, and reflection, emit one row per
  applicable tag occurrence with the exact `interaction_index`, `phase`,
  `round`, agent/model/role metadata, and a verbatim quote. Do not invent a
  `discussion_turn` when that phase has none; use null.
- `formal_outcome` is the sole exception to turn-level sourcing. Use it only
  for a structural codebook tag that is evidenced by the final outcome rather
  than an authored conversation or interaction.
- Public discussion evidence must use `conversation_logs`; use
  `agent_authored_interactions` for private thinking, proposal, voting, and
  reflection, avoiding duplicate discussion rows.
- Write the JSONL and audit files only to the paths specified for your chunk.

---

"""
    atomic_write(
        OUTPUT_ROOT / "TTC_CODEX_ADJUDICATION_INSTRUCTIONS.md",
        addendum + original,
    )


def main() -> None:
    views_dir = OUTPUT_ROOT / "rollout_views"
    manifests_dir = OUTPUT_ROOT / "chunk_manifests"
    outputs_dir = OUTPUT_ROOT / "subagent_outputs"
    for path in (views_dir, manifests_dir, outputs_dir):
        path.mkdir(parents=True, exist_ok=True)

    codebook_path = ORIGINAL_ROOT / "llm_tag_codebook.json"
    codebook = json.loads(codebook_path.read_text(encoding="utf-8"))
    # Preserve the frozen codebook byte-for-byte, not merely semantically.
    atomic_write(
        OUTPUT_ROOT / "llm_tag_codebook.json",
        codebook_path.read_text(encoding="utf-8"),
    )
    write_instructions()

    all_rows: list[dict[str, Any]] = []
    rows_by_seed: dict[int, list[dict[str, Any]]] = {}
    missing: dict[str, list[int]] = {}
    for seed in SEED_ORDER:
        source_root = RESULTS_ROOT / SEED_ROOT_NAMES[seed]
        seed_rows = []
        seed_missing = []
        for config_id in SOURCE_CONFIG_IDS:
            config_path = source_root / "configs" / f"config_{config_id:04d}.json"
            source_config = json.loads(config_path.read_text(encoding="utf-8"))
            if source_config.get("target_model_family") != TARGET_FAMILY:
                raise RuntimeError(f"Unexpected target family in {config_path}")
            output_dir = Path(source_config["output_dir"])
            result_path = output_dir / "run_1_experiment_results.json"
            interactions_path = output_dir / "run_1_all_interactions.json"
            if not result_path.exists() or not interactions_path.exists():
                seed_missing.append(config_id)
                continue
            rollout_id = f"seed_{seed}_config_{config_id:04d}"
            view_path = views_dir / f"{rollout_id}.json"
            manifest = build_manifest(
                seed, source_config, result_path, interactions_path, view_path
            )
            write_json(view_path, build_view(manifest))
            seed_rows.append(manifest)
        rows_by_seed[seed] = seed_rows
        all_rows.extend(seed_rows)
        missing[str(seed)] = seed_missing

    write_jsonl(OUTPUT_ROOT / "all_available_rollouts_manifest.jsonl", all_rows)

    chunk_index = []
    chunk_number = 0
    for seed in SEED_ORDER:
        seed_rows = rows_by_seed[seed]
        for start in range(0, len(seed_rows), CHUNK_SIZE):
            chunk_id = f"chunk_{chunk_number:04d}"
            rows = seed_rows[start : start + CHUNK_SIZE]
            manifest_path = manifests_dir / f"{chunk_id}.jsonl"
            output_path = outputs_dir / f"{chunk_id}_events.jsonl"
            audit_path = outputs_dir / f"{chunk_id}_audit.md"
            write_jsonl(manifest_path, rows)
            chunk_index.append(
                {
                    "chunk_id": chunk_id,
                    "seed": seed,
                    "manifest_path": str(manifest_path.resolve()),
                    "output_path": str(output_path.resolve()),
                    "audit_path": str(audit_path.resolve()),
                    "rollout_count": len(rows),
                    "rollout_ids": [row["rollout_id"] for row in rows],
                    "source_config_ids": [row["source_config_id"] for row in rows],
                }
            )
            chunk_number += 1
    write_jsonl(OUTPUT_ROOT / "chunk_index.jsonl", chunk_index)

    inventory = {
        "execution_policy": "Codex collaboration subagents only; no external model APIs",
        "target_family": TARGET_FAMILY,
        "seed_order": list(SEED_ORDER),
        "expected_total_rollouts": len(SEED_ORDER) * len(SOURCE_CONFIG_IDS),
        "available_total_rollouts": len(all_rows),
        "available_by_seed": {
            str(seed): len(rows_by_seed[seed]) for seed in SEED_ORDER
        },
        "missing_source_config_ids": missing,
        "chunk_size": CHUNK_SIZE,
        "chunk_count": len(chunk_index),
        "codebook_tag_count": len(codebook),
        "labels_both_target_and_baseline": True,
        "source_codebook_sha256": sha256(codebook_path),
    }
    write_json(OUTPUT_ROOT / "source_inventory.json", inventory)
    print(json.dumps(inventory, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare manifests and instructions for TTC LLM strategic tag adjudication."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_TAG_DIR = PROJECT_ROOT / "analysis/strategic_qualitative_tags_20260628"
CODEBOOK_CSV = SOURCE_TAG_DIR / "new_strategy_tag_codebook.csv"
TTC_ROOT = PROJECT_ROOT / "experiments/results/ttc_native_scaling_20260502_212943"
TTC_RESULTS_CSV = TTC_ROOT / "monitoring/partial_results_latest.csv"
OUT_DIR = PROJECT_ROOT / "analysis/ttc_llm_strategic_tag_adjudication_20260629"
MANIFEST_DIR = OUT_DIR / "chunk_manifests"
ROLLOUT_VIEW_DIR = OUT_DIR / "rollout_views"
OUTPUT_DIR = OUT_DIR / "subagent_outputs"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_float(value: object) -> float | None:
    try:
        if value in {"", None}:
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def safe_int(value: object) -> int | None:
    number = safe_float(value)
    if number is None:
        return None
    return int(number)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def build_codebook() -> list[dict[str, Any]]:
    rows = []
    for row in read_csv(CODEBOOK_CSV):
        rows.append(
            {
                "tag_code": row["tag_code"],
                "tag_title": row["tag_title"],
                "category": row["category"],
                "definition": row["description"],
                "paper_value": row["paper_value"],
                "scope_hint": {
                    "games": [x for x in str(row.get("games") or "").split(";") if x],
                    "min_agents": int(float(row["min_agents"])) if row.get("min_agents") else None,
                    "structural": str(row.get("structural", "")).lower() == "true",
                },
                "regex_patterns_from_old_scaffold_for_reference_only": row.get("patterns", ""),
            }
        )
    return rows


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
    if isinstance(response, str):
        return response
    return json.dumps(response, sort_keys=True)


def compact_interaction(entry: dict[str, Any], interaction_index: int) -> dict[str, Any]:
    phase, round_num, discussion_turn = parse_phase(entry.get("phase"))
    usage = entry.get("token_usage") or {}
    return {
        "interaction_index": interaction_index,
        "source_kind": "interaction",
        "agent_id": entry.get("agent_id"),
        "phase_raw": entry.get("phase"),
        "phase": phase,
        "round": round_num if round_num is not None else entry.get("round"),
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


def build_rollout_view(result_path: Path, interactions_path: Path, manifest_row: dict[str, Any]) -> dict[str, Any]:
    with result_path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    with interactions_path.open(encoding="utf-8") as handle:
        interactions = json.load(handle)
    config = result.get("config") or {}

    conversation_logs = []
    for idx, log in enumerate(result.get("conversation_logs") or []):
        conversation_logs.append(
            {
                "log_index": idx,
                "source_kind": "conversation_log",
                "phase": log.get("phase"),
                "round": log.get("round"),
                "discussion_turn": log.get("discussion_turn"),
                "speaker_agent": log.get("from"),
                "speaker_order": log.get("speaker_order"),
                "total_speakers": log.get("total_speakers"),
                "content": log.get("content"),
            }
        )

    compact_interactions = []
    for idx, entry in enumerate(interactions):
        phase, _, _ = parse_phase(entry.get("phase"))
        if phase == "game_setup":
            continue
        compact_interactions.append(compact_interaction(entry, idx))

    return {
        "manifest": manifest_row,
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
            "target_reasoning_level_requested": config.get("target_reasoning_level_requested"),
            "target_reasoning_level_index": config.get("target_reasoning_level_index"),
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
        "agent_authored_interactions": compact_interactions,
    }


def build_manifest_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(TTC_RESULTS_CSV):
        result_path = Path(row["path"])
        if not result_path.is_absolute():
            result_path = PROJECT_ROOT / result_path
        interactions_path = result_path.with_name("run_1_all_interactions.json")
        with result_path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        cfg = result.get("config") or {}
        target_agent = row.get("target_agent") or (
            "Agent_1" if cfg.get("target_position") == 0 else "Agent_2"
        )
        baseline_agent = row.get("baseline_agent") or (
            "Agent_2" if target_agent == "Agent_1" else "Agent_1"
        )
        agent_role_map = {target_agent: "target", baseline_agent: "baseline"}
        agent_model_map = cfg.get("agent_model_map") or {}
        agent_elo_map = cfg.get("agent_elo_map") or {}
        config_id = safe_int(row.get("config_id") or cfg.get("config_id"))
        view_path = ROLLOUT_VIEW_DIR / f"config_{config_id:04d}_rollout_view.json"
        manifest_row = {
            "config_id": config_id,
            "result_path": str(result_path),
            "interactions_path": str(interactions_path),
            "rollout_view_path": str(view_path),
            "provider": row.get("provider") or cfg.get("target_provider"),
            "family": row.get("family") or cfg.get("target_model_family"),
            "level": row.get("level") or cfg.get("target_reasoning_level_requested"),
            "level_index": safe_int(row.get("level_index") or cfg.get("target_reasoning_level_index")),
            "game_label": row.get("game") or cfg.get("game_label"),
            "game_cell": row.get("game_cell") or cfg.get("game_cell_id"),
            "game_type": cfg.get("game_type"),
            "n_agents": safe_int(cfg.get("n_agents")) or 2,
            "order": row.get("order") or cfg.get("order"),
            "target_agent": target_agent,
            "baseline_agent": baseline_agent,
            "target_model": cfg.get("target_model"),
            "target_model_id": cfg.get("target_model_id"),
            "baseline_model": cfg.get("baseline_model"),
            "agent_model_map": agent_model_map,
            "agent_elo_map": agent_elo_map,
            "agent_role_map": agent_role_map,
            "consensus_reached": str(row.get("consensus", "")).lower() == "true",
            "final_round": safe_int(row.get("round") or result.get("final_round")),
            "target_utility": safe_float(row.get("target_utility")),
            "baseline_utility": safe_float(row.get("baseline_utility")),
            "utility_gap": safe_float(row.get("utility_gap")),
            "competition_level": safe_float(cfg.get("competition_level")),
            "rho": safe_float(cfg.get("rho")),
            "theta": safe_float(cfg.get("theta")),
            "alpha": safe_float(cfg.get("alpha")),
            "sigma": safe_float(cfg.get("sigma")),
            "target_llm_call_count": safe_int(row.get("target_llm_call_count")),
            "target_compute_tokens_total": safe_float(row.get("target_compute_tokens_total")),
            "target_compute_tokens_per_call": safe_float(row.get("target_compute_tokens_per_call")),
            "target_compute_tokens_source": row.get("target_compute_tokens_source"),
            "target_reasoning_tokens_raw": safe_float(row.get("target_reasoning_tokens_raw")),
            "target_output_tokens": safe_float(row.get("target_output_tokens")),
            "target_output_tokens_per_call": safe_float(row.get("target_output_tokens_per_call")),
            "baseline_reasoning_tokens": safe_float(row.get("baseline_reasoning_tokens")),
            "conversation_log_count": len(result.get("conversation_logs") or []),
        }
        rows.append(manifest_row)
    return sorted(rows, key=lambda r: int(r["config_id"]))


def write_rollout_views(manifest_rows: list[dict[str, Any]]) -> None:
    for row in manifest_rows:
        view = build_rollout_view(Path(row["result_path"]), Path(row["interactions_path"]), row)
        Path(row["rollout_view_path"]).write_text(
            json.dumps(view, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def write_instructions(codebook: list[dict[str, Any]]) -> None:
    schema = {
        "chunk_id": "chunk_0000",
        "config_id": 1,
        "result_path": "/abs/path/run_1_experiment_results.json",
        "interactions_path": "/abs/path/run_1_all_interactions.json",
        "rollout_view_path": "/abs/path/config_0001_rollout_view.json",
        "family": "gpt-5 | claude-sonnet-4-6 | gemini-3-flash",
        "level": "minimal | low | medium | high | max",
        "level_index": 0,
        "provider": "OpenAI | Anthropic | Google",
        "game_label": "game1 | game2 | game3",
        "game_cell": "game1_comp_0p0 | game2_rho_0_theta_1 | game3_alpha_0p0_sigma_0p2",
        "game_type": "item_allocation | issue_negotiation | co_funding",
        "n_agents": 2,
        "order": "target_first | baseline_first",
        "target_agent": "Agent_1",
        "baseline_agent": "Agent_2",
        "speaker_agent": "Agent_1 or Agent_2 or null",
        "speaker_model": "model name or null",
        "speaker_elo": 1337,
        "speaker_role": "target | baseline | null",
        "speaker_is_target": True,
        "speaker_is_baseline": False,
        "tag_code": "conditional_veto_threat",
        "tag_title": "Conditional veto threat",
        "evidence_type": "utterance | private_thinking | proposal_reasoning | vote_reasoning | reflection | formal_outcome",
        "source_kind": "conversation_log | interaction | formal_outcome",
        "phase": "discussion | private_thinking | proposal | voting | reflection | final_outcome",
        "round": 2,
        "discussion_turn": 1,
        "log_index": 17,
        "interaction_index": 23,
        "speaker_order": 1,
        "total_speakers": 2,
        "quote": "short exact excerpt or formal-outcome description",
        "rationale": "one sentence explaining why the tag applies semantically",
        "confidence": "high | medium | low",
        "negation_checked": True,
    }
    lines = [
        "# TTC LLM Strategic Tag Adjudication Instructions",
        "",
        "You are producing semantic, LLM-judged labels for the N=2 test-time-compute native scaling experiments. Do not use regex matching as the classifier. The regex patterns in the codebook are only historical hints from the old high-recall scaffold.",
        "",
        "Read every assigned rollout. Start with each `rollout_view_path`: it contains manifest metadata, final outcome, public conversation logs, and compacted agent-authored interaction responses for private thinking, proposal, voting, and reflection. Use `result_path` and `interactions_path` as raw sources whenever you need to verify exact context, prompt rules, token metadata, or a quote.",
        "",
        "Classify only agent-authored behavior and formal outcomes. Do not classify setup prompts, system instructions, or prompt text as behavior. Game setup acknowledgements are usually not strategic evidence unless the agent itself adds substantive strategic content.",
        "",
        "Output one JSON object per positive tag evidence event. If one utterance or private/proposal/vote/reflection response clearly exhibits multiple tags, output multiple rows with the same source metadata and different `tag_code`s. If a rollout has no real evidence for a tag, output no row for that tag.",
        "",
        "Use high precision with high recall: include subtle real examples, but reject negated or merely mentioned concepts. For example, `not a final offer` is not `ultimatum_language` unless the surrounding text still functions as an ultimatum.",
        "",
        "For every row, include exact round/turn/speaker/model metadata. Use manifest `agent_model_map`, `agent_elo_map`, and `agent_role_map` to populate speaker model, Elo, and role. Mark `speaker_is_target` and `speaker_is_baseline` from the target/baseline agent fields.",
        "",
        "TTC-specific context: the target agent is the model whose requested reasoning/test-time compute level varies. The baseline agent is usually `gpt-5-nano` with baseline requested reasoning. Later analysis will compare tag frequencies by family, level, game cell, order, and target/baseline role, so preserve these fields exactly.",
        "",
        "Structural tags should be represented as `formal_outcome` rows. If the structural behavior is attributable to specific agents, include the relevant `speaker_agent`; otherwise use null and explain in `rationale`.",
        "",
        "Write your final chunk output as JSONL to the exact assigned output path. Also write a short audit markdown file with ambiguous tags, unsupported tags in your chunk, and possible new TTC-specific tag ideas. Keep classification rows limited to the current 50 tags.",
        "",
        "Required JSONL schema:",
        "```json",
        json.dumps(schema, indent=2),
        "```",
        "",
        "Current 50-tag codebook:",
        "```json",
        json.dumps(codebook, indent=2),
        "```",
    ]
    (OUT_DIR / "TTC_LLM_ADJUDICATION_INSTRUCTIONS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    ROLLOUT_VIEW_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    codebook = build_codebook()
    manifest_rows = build_manifest_rows()
    write_rollout_views(manifest_rows)
    (OUT_DIR / "llm_tag_codebook.json").write_text(
        json.dumps(codebook, indent=2) + "\n", encoding="utf-8"
    )
    write_jsonl(OUT_DIR / "all_ttc_rollouts_manifest.jsonl", manifest_rows)
    write_instructions(codebook)

    chunk_size = 6
    chunk_index = []
    for idx in range(0, len(manifest_rows), chunk_size):
        chunk_num = idx // chunk_size
        chunk_id = f"chunk_{chunk_num:04d}"
        chunk = manifest_rows[idx : idx + chunk_size]
        manifest_path = MANIFEST_DIR / f"{chunk_id}.jsonl"
        output_path = OUTPUT_DIR / f"{chunk_id}_events.jsonl"
        audit_path = OUTPUT_DIR / f"{chunk_id}_audit.md"
        write_jsonl(manifest_path, chunk)
        chunk_index.append(
            {
                "chunk_id": chunk_id,
                "manifest_path": str(manifest_path),
                "output_path": str(output_path),
                "audit_path": str(audit_path),
                "start_index": idx,
                "end_index_exclusive": idx + len(chunk),
                "rollout_count": len(chunk),
                "config_range": [chunk[0]["config_id"], chunk[-1]["config_id"]],
                "family": chunk[0]["family"],
                "level": chunk[0]["level"],
                "game_label": chunk[0]["game_label"],
            }
        )
    write_jsonl(OUT_DIR / "chunk_index.jsonl", chunk_index)
    print(
        f"wrote {len(manifest_rows)} TTC rollout manifests into {len(chunk_index)} chunks "
        f"under {OUT_DIR}"
    )


if __name__ == "__main__":
    main()

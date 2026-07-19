#!/usr/bin/env python3
"""Prepare manifests and instructions for LLM adjudication of strategic tags."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "analysis/strategic_qualitative_tags_20260628"
QUAL_CSV = PROJECT_ROOT / "analysis/qualitative_rollout_dynamics_20260628/refined_rollout_dynamics_coding.csv"
CODEBOOK_CSV = SOURCE_DIR / "new_strategy_tag_codebook.csv"
OUT_DIR = PROJECT_ROOT / "analysis/llm_strategic_tag_adjudication_20260628"
MANIFEST_DIR = OUT_DIR / "chunk_manifests"
OUTPUT_DIR = OUT_DIR / "subagent_outputs"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_float(value: object) -> float | None:
    try:
        if value in {"", None}:
            return None
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def build_codebook() -> list[dict]:
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


def build_manifest_rows() -> list[dict]:
    rows = []
    for row in read_csv(QUAL_CSV):
        path = Path(row["result_path"])
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        cfg = data.get("config") or {}
        rows.append(
            {
                "result_path": str(path),
                "config_id": row.get("config_id") or cfg.get("config_id"),
                "experiment_family": row.get("experiment_family") or cfg.get("experiment_family"),
                "game_label": row.get("game_label") or cfg.get("game_label"),
                "n_agents": int(row.get("n_agents") or cfg.get("n_agents") or 0),
                "setting": row.get("setting") or "",
                "competition_id": cfg.get("competition_id"),
                "competition_level": safe_float(cfg.get("competition_level")),
                "rho": safe_float(cfg.get("rho")),
                "theta": safe_float(cfg.get("theta")),
                "sigma": safe_float(cfg.get("sigma")),
                "alpha": safe_float(cfg.get("alpha")),
                "model_order": row.get("model_order") or cfg.get("model_order"),
                "models": row.get("models") or "+".join(cfg.get("models") or []),
                "adversary_model": row.get("adversary_model") or cfg.get("adversary_model"),
                "adversary_position": row.get("adversary_position") or cfg.get("adversary_position"),
                "agent_model_map": cfg.get("agent_model_map") or {},
                "agent_elo_map": cfg.get("agent_elo_map") or {},
                "agent_role_map": cfg.get("agent_role_map") or {},
                "consensus_reached": str(row.get("consensus_reached", "")).lower() == "true",
                "final_round": int(float(row["final_round"])) if row.get("final_round") else None,
                "conversation_log_count": len(data.get("conversation_logs") or []),
            }
        )
    return rows


def write_instructions(codebook: list[dict]) -> None:
    schema = {
        "chunk_id": "chunk_0000",
        "result_path": "/abs/path/experiment_results.json",
        "config_id": "123",
        "experiment_family": "heterogeneous_random | homogeneous_adversary | homogeneous_control",
        "game_label": "game1 | game2 | game3",
        "n_agents": 4,
        "tag_code": "conditional_veto_threat",
        "tag_title": "Conditional veto threat",
        "evidence_type": "utterance | proposal_reasoning | formal_outcome",
        "phase": "discussion | proposal | final_outcome",
        "round": 2,
        "discussion_turn": 1,
        "log_index": 17,
        "speaker_agent": "Agent_3 or null for non-agent structural evidence",
        "speaker_model": "model name or null",
        "speaker_elo": 1448,
        "speaker_role": "adversary | baseline | null",
        "speaker_order": 3,
        "total_speakers": 6,
        "quote": "short exact excerpt or formal-outcome description",
        "rationale": "one sentence explaining why the tag applies, including semantic nuance",
        "confidence": "high | medium | low",
        "negation_checked": True,
    }
    lines = [
        "# LLM Strategic Tag Adjudication Instructions",
        "",
        "You are producing semantic, LLM-judged labels. Do not use regex matching as the classifier. The regex patterns in the codebook are only historical hints from the old high-recall scaffold.",
        "",
        "Read every assigned raw `experiment_results.json` fully enough to judge all agent-authored discussion/proposal messages and relevant final formal outcomes. Classify at event level, not just rollout level.",
        "",
        "Output one JSON object per tag evidence event. If one utterance clearly exhibits multiple tags, output multiple rows with the same log metadata and different `tag_code`s. If a rollout has no real evidence for a tag, output no row for that tag.",
        "",
        "Use high precision with high recall: include subtle real examples, but reject negated or merely mentioned concepts. For example, `not a final offer` is not `ultimatum_language` unless the surrounding text still functions as an ultimatum.",
        "",
        "For each row, include exact round/turn/speaker/model metadata. Use the config `agent_model_map`, `agent_elo_map`, and `agent_role_map` to populate speaker model, Elo, and role.",
        "",
        "Structural tags should be represented as `formal_outcome` rows. If the structural behavior is attributable to specific agents, include the relevant `speaker_agent`; otherwise use null and explain in `rationale`.",
        "",
        "Write your final chunk output as JSONL to the exact assigned output path. Also write a short audit markdown file with any weak/ambiguous tags and possible new tag ideas, but keep classification rows limited to the current 50 tags.",
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
    (OUT_DIR / "LLM_ADJUDICATION_INSTRUCTIONS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    codebook = build_codebook()
    manifest_rows = build_manifest_rows()
    (OUT_DIR / "llm_tag_codebook.json").write_text(json.dumps(codebook, indent=2), encoding="utf-8")
    write_jsonl(OUT_DIR / "all_rollouts_manifest.jsonl", manifest_rows)
    write_instructions(codebook)

    chunk_size = 10
    manifest_index = []
    for idx in range(0, len(manifest_rows), chunk_size):
        chunk_num = idx // chunk_size
        chunk_id = f"chunk_{chunk_num:04d}"
        chunk = manifest_rows[idx : idx + chunk_size]
        manifest_path = MANIFEST_DIR / f"{chunk_id}.jsonl"
        output_path = OUTPUT_DIR / f"{chunk_id}_events.jsonl"
        audit_path = OUTPUT_DIR / f"{chunk_id}_audit.md"
        write_jsonl(manifest_path, chunk)
        manifest_index.append(
            {
                "chunk_id": chunk_id,
                "manifest_path": str(manifest_path),
                "output_path": str(output_path),
                "audit_path": str(audit_path),
                "start_index": idx,
                "end_index_exclusive": idx + len(chunk),
                "rollout_count": len(chunk),
            }
        )
    write_jsonl(OUT_DIR / "chunk_index.jsonl", manifest_index)
    print(
        f"wrote {len(manifest_rows)} rollout manifests into {len(manifest_index)} chunks "
        f"under {OUT_DIR}"
    )


if __name__ == "__main__":
    main()

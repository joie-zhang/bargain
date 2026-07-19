#!/usr/bin/env python3
"""Prepare manifests and instructions for N=2 GPT-5-nano strategic tag adjudication."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODEBOOK_CSV = (
    PROJECT_ROOT / "analysis/strategic_qualitative_tags_20260628/new_strategy_tag_codebook.csv"
)
N2_RUNS_CSV = (
    PROJECT_ROOT
    / "experiments/results/n2_baseline_comparison_analysis_20260505/all_runs_with_metrics.csv"
)
OUT_DIR = PROJECT_ROOT / "analysis/llm_strategic_tag_adjudication_n2_gpt5_20260629"
MANIFEST_DIR = OUT_DIR / "chunk_manifests"
OUTPUT_DIR = OUT_DIR / "subagent_outputs"

BASELINE_ELO = 1337


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_float(value: object) -> float | None:
    try:
        if value in {"", None} or pd.isna(value):
            return None
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def safe_int(value: object) -> int | None:
    number = safe_float(value)
    return int(number) if number is not None else None


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def build_codebook() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def build_manifest_rows() -> list[dict[str, Any]]:
    df = pd.read_csv(N2_RUNS_CSV)
    df = df[df["baseline_key"].eq("gpt5_nano")].copy()
    df = df.sort_values(["game_id", "adversary_elo", "adversary_model", "competition_value", "model_order", "discussion_turns"])

    rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for row in df.to_dict(orient="records"):
        result_path = (PROJECT_ROOT / row["result_path"]).resolve()
        if not result_path.exists():
            raise FileNotFoundError(result_path)
        path_text = str(result_path)
        if path_text in seen_paths:
            raise ValueError(f"duplicate result path: {path_text}")
        seen_paths.add(path_text)

        with result_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        cfg = data.get("config") or {}

        baseline_agent = row["baseline_agent"]
        adversary_agent = row["adversary_agent"]
        baseline_model = row["baseline_model"]
        adversary_model = row["adversary_model"]
        agent_model_map = {
            baseline_agent: baseline_model,
            adversary_agent: adversary_model,
        }
        agent_elo_map = {
            baseline_agent: BASELINE_ELO,
            adversary_agent: safe_int(row["adversary_elo"]),
        }
        agent_role_map = {
            baseline_agent: "baseline",
            adversary_agent: "adversary",
        }

        rows.append(
            {
                "result_path": path_text,
                "config_id": row["config_file"],
                "experiment_family": "n2_gpt5_bilateral",
                "game_label": row["game_id"],
                "game_name": row["game_label"],
                "n_agents": 2,
                "setting": row["competition_setting"],
                "competition_id": row["competition_label"],
                "competition_label": row["competition_label"],
                "competition_level": safe_float(row.get("competition_level")),
                "competition_value": safe_float(row.get("competition_value")),
                "rho": safe_float(row.get("rho")),
                "theta": safe_float(row.get("theta")),
                "sigma": safe_float(row.get("sigma")),
                "alpha": safe_float(row.get("alpha")),
                "model_order": row["model_order"],
                "conceptual_order": row["conceptual_order"],
                "discussion_turns": safe_int(row["discussion_turns"]),
                "models": f"{baseline_model}+{adversary_model}",
                "baseline_agent": baseline_agent,
                "adversary_agent": adversary_agent,
                "adversary_model": adversary_model,
                "adversary_raw_model": row["adversary_raw_model"],
                "adversary_short": row["adversary_short"],
                "adversary_elo": safe_int(row["adversary_elo"]),
                "agent_model_map": agent_model_map,
                "agent_elo_map": agent_elo_map,
                "agent_role_map": agent_role_map,
                "consensus_reached": bool(row["consensus_reached"]),
                "final_round": safe_int(row["final_round"]),
                "conversation_log_count": len(data.get("conversation_logs") or []),
                "source_analysis_csv": str(N2_RUNS_CSV),
            }
        )
    if len(rows) != 1941:
        raise ValueError(f"expected 1941 GPT-5-nano N=2 rows, got {len(rows)}")
    return rows


def write_instructions(codebook: list[dict[str, Any]]) -> None:
    schema = {
        "chunk_id": "chunk_0000",
        "result_path": "/abs/path/experiment_results.json",
        "config_id": "config_000.json",
        "experiment_family": "n2_gpt5_bilateral",
        "game_label": "game1 | game2 | game3",
        "n_agents": 2,
        "tag_code": "conditional_veto_threat",
        "tag_title": "Conditional veto threat",
        "evidence_type": "utterance | proposal_reasoning | formal_outcome",
        "phase": "discussion | proposal | final_outcome",
        "round": 2,
        "discussion_turn": 1,
        "log_index": 17,
        "speaker_agent": "Agent_1 or Agent_2 or null for non-agent structural evidence",
        "speaker_model": "model name or null",
        "speaker_elo": 1448,
        "speaker_role": "adversary | baseline | null",
        "speaker_order": 1,
        "total_speakers": 2,
        "quote": "short exact excerpt or formal-outcome description",
        "rationale": "one sentence explaining why the tag applies, including semantic nuance",
        "confidence": "high | medium | low",
        "negation_checked": True,
    }
    lines = [
        "# N=2 GPT-5-nano LLM Strategic Tag Adjudication Instructions",
        "",
        "You are producing semantic, LLM-judged labels for the canonical GPT-5-nano bilateral N=2 corpus used in the NeurIPS paper.",
        "",
        "Do not use regex matching as the classifier. The regex patterns in the codebook are only historical hints from the old high-recall scaffold.",
        "",
        "Read every assigned raw `experiment_results.json` fully enough to judge all agent-authored discussion/proposal messages and relevant final formal outcomes. Classify at event level, not just rollout level.",
        "",
        "Output one JSON object per tag evidence event. If one utterance clearly exhibits multiple tags, output multiple rows with the same log metadata and different `tag_code`s. If a rollout has no real evidence for a tag, output no row for that tag.",
        "",
        "Use high precision with high recall: include subtle real examples, but reject negated or merely mentioned concepts. For example, `not a final offer` is not `ultimatum_language` unless the surrounding text still functions as an ultimatum.",
        "",
        "For each row, include exact round/turn/speaker/model metadata. Use the manifest `agent_model_map`, `agent_elo_map`, and `agent_role_map` to populate speaker model, Elo, and role.",
        "",
        "These are bilateral runs: `speaker_role` must be `baseline` for GPT-5-nano and `adversary` for the paired non-baseline model when a speaker is present.",
        "",
        "Structural tags should be represented as `formal_outcome` rows. If the structural behavior is attributable to a specific agent, include the relevant `speaker_agent`; otherwise use null and explain in `rationale`.",
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

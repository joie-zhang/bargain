#!/usr/bin/env python3
"""Prepare manifests and instructions for random-monoculture strategic tag adjudication."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = (
    PROJECT_ROOT
    / "experiments/results/full_games123_random_monoculture_control_20260628_014357"
)
CODEBOOK_CSV = (
    PROJECT_ROOT / "analysis/strategic_qualitative_tags_20260628/new_strategy_tag_codebook.csv"
)
OUT_DIR = PROJECT_ROOT / "analysis/llm_strategic_tag_adjudication_random_monoculture_20260629"
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


def safe_int(value: object) -> int | None:
    number = safe_float(value)
    return int(number) if number is not None else None


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def config_number_from_path(path: Path) -> int:
    match = re.search(r"config_(\d{4})", str(path))
    if not match:
        raise ValueError(f"cannot parse config number from {path}")
    return int(match.group(1))


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


def model_assignment_lookup() -> dict[tuple[str, str], dict[str, str]]:
    path = RUN_ROOT / "configs/model_assignments.csv"
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    for row in read_csv(path):
        lookup[(row["game_label"], row["model"])] = row
    return lookup


def planned_config_dirs() -> list[Path]:
    return sorted(
        [path for path in (RUN_ROOT / "runs").glob("config_*") if path.is_dir()],
        key=config_number_from_path,
    )


def completed_result_paths() -> list[Path]:
    return sorted(
        (RUN_ROOT / "runs").glob("config_*/experiment_results.json"),
        key=config_number_from_path,
    )


def infer_game_label(cfg: dict[str, Any]) -> str:
    label = cfg.get("game_label")
    if label:
        return str(label)
    game_type = cfg.get("game_type")
    return {
        "item_allocation": "game1",
        "diplomacy": "game2",
        "co_funding": "game3",
    }.get(str(game_type), str(game_type or "unknown"))


def build_manifest_row(path: Path, assignment: dict[tuple[str, str], dict[str, str]]) -> dict[str, Any]:
    abs_path = path.resolve()
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    cfg = data.get("config") or {}

    config_number = config_number_from_path(path)
    game_label = infer_game_label(cfg)
    n_agents = int(cfg.get("n_agents") or cfg.get("num_agents") or 0)
    model = cfg.get("monoculture_model")
    if not model:
        models = cfg.get("models") or []
        model = models[0] if models else None
    model = str(model or "unknown")

    assignment_row = assignment.get((game_label, model), {})
    model_elo = safe_int(cfg.get("model_elo"))
    if model_elo is None:
        model_elo = safe_int(assignment_row.get("elo"))
    elo_band_index = safe_int(cfg.get("elo_band_index"))
    if elo_band_index is None:
        elo_band_index = safe_int(assignment_row.get("band_index"))
    elo_band_label = cfg.get("elo_band_label") or assignment_row.get("band_label")

    agent_model_map = cfg.get("agent_model_map") or {
        f"Agent_{idx}": model for idx in range(1, n_agents + 1)
    }
    agent_elo_map = cfg.get("agent_elo_map") or {}
    agent_elo_map = {
        agent: (safe_int(value) if safe_int(value) is not None else model_elo)
        for agent, value in agent_elo_map.items()
    }
    for agent in agent_model_map:
        agent_elo_map.setdefault(agent, model_elo)
    agent_role_map = cfg.get("agent_role_map") or {
        agent: "random_monoculture_control" for agent in agent_model_map
    }

    setting = ""
    if game_label == "game1":
        setting = f"comp={cfg.get('competition_level')}"
    elif game_label == "game2":
        setting = f"rho={cfg.get('rho')};theta={cfg.get('theta')}"
    elif game_label == "game3":
        setting = f"sigma={cfg.get('sigma')};alpha={cfg.get('alpha')}"

    return {
        "result_path": str(abs_path),
        "config_id": f"config_{config_number:04d}",
        "config_number": config_number,
        "experiment_family": "random_monoculture_control",
        "game_label": game_label,
        "game_type": cfg.get("game_type"),
        "n_agents": n_agents,
        "setting": setting,
        "competition_id": cfg.get("competition_id"),
        "competition_level": safe_float(cfg.get("competition_level")),
        "rho": safe_float(cfg.get("rho")),
        "theta": safe_float(cfg.get("theta")),
        "sigma": safe_float(cfg.get("sigma")),
        "alpha": safe_float(cfg.get("alpha")),
        "model_order": "random_monoculture_control",
        "models": "+".join(cfg.get("models") or [model] * n_agents),
        "monoculture_model": model,
        "model_elo": model_elo,
        "elo_band_index": elo_band_index,
        "elo_band_label": elo_band_label,
        "agent_model_map": agent_model_map,
        "agent_elo_map": agent_elo_map,
        "agent_role_map": agent_role_map,
        "consensus_reached": bool(data.get("consensus_reached")),
        "final_round": safe_int(data.get("final_round")),
        "conversation_log_count": len(data.get("conversation_logs") or []),
        "source_run_root": str(RUN_ROOT),
    }


def write_instructions(codebook: list[dict[str, Any]]) -> None:
    schema = {
        "chunk_id": "chunk_0000",
        "result_path": "/abs/path/experiment_results.json",
        "config_id": "config_0001",
        "experiment_family": "random_monoculture_control",
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
        "speaker_role": "random_monoculture_control or null",
        "speaker_order": 3,
        "total_speakers": 6,
        "quote": "short exact excerpt or formal-outcome description",
        "rationale": "one sentence explaining why the tag applies, including semantic nuance",
        "confidence": "high | medium | low",
        "negation_checked": True,
    }
    lines = [
        "# Random-Monoculture LLM Strategic Tag Adjudication Instructions",
        "",
        "You are producing semantic, LLM-judged strategic labels for random-monoculture control bargaining rollouts.",
        "",
        "Every rollout clones one selected model into every agent slot. This is different from the fixed GPT-5-nano homogeneous-control corpus: preserve `speaker_role: random_monoculture_control` for agent-authored evidence.",
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


def write_missing_ledger() -> None:
    planned = planned_config_dirs()
    completed = {path.parent.name for path in completed_result_paths()}
    missing_rows = []
    for path in planned:
        result = path / "experiment_results.json"
        if path.name not in completed:
            missing_rows.append(
                {
                    "config_dir": path.name,
                    "config_number": config_number_from_path(path),
                    "result_path": str(result.resolve()),
                }
            )
    write_jsonl(OUT_DIR / "missing_result_ledger.jsonl", missing_rows)


def rebuild_chunk_index(rows: list[dict[str, Any]]) -> None:
    chunk_size = 10
    manifest_index = []
    for chunk_num, idx in enumerate(range(0, len(rows), chunk_size)):
        chunk_id = f"chunk_{chunk_num:04d}"
        chunk = rows[idx : idx + chunk_size]
        manifest_path = MANIFEST_DIR / f"{chunk_id}.jsonl"
        output_path = OUTPUT_DIR / f"{chunk_id}_events.jsonl"
        audit_path = OUTPUT_DIR / f"{chunk_id}_audit.md"
        if output_path.exists():
            raise FileExistsError(
                f"refusing to rewrite chunk layout after tagging started; found {output_path}"
            )
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


def append_new_chunks(existing_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> None:
    if not new_rows:
        return
    chunk_index = read_jsonl(OUT_DIR / "chunk_index.jsonl")
    next_chunk_num = 0
    if chunk_index:
        next_chunk_num = max(int(row["chunk_id"].split("_")[1]) for row in chunk_index) + 1
    chunk_id = f"chunk_{next_chunk_num:04d}"
    manifest_path = MANIFEST_DIR / f"{chunk_id}.jsonl"
    output_path = OUTPUT_DIR / f"{chunk_id}_events.jsonl"
    audit_path = OUTPUT_DIR / f"{chunk_id}_audit.md"
    write_jsonl(manifest_path, new_rows)
    chunk_index.append(
        {
            "chunk_id": chunk_id,
            "manifest_path": str(manifest_path),
            "output_path": str(output_path),
            "audit_path": str(audit_path),
            "start_index": len(existing_rows),
            "end_index_exclusive": len(existing_rows) + len(new_rows),
            "rollout_count": len(new_rows),
        }
    )
    write_jsonl(OUT_DIR / "chunk_index.jsonl", chunk_index)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    codebook = build_codebook()
    (OUT_DIR / "llm_tag_codebook.json").write_text(json.dumps(codebook, indent=2), encoding="utf-8")
    write_instructions(codebook)

    assignment = model_assignment_lookup()
    existing_rows = read_jsonl(OUT_DIR / "all_rollouts_manifest.jsonl")
    existing_paths = {row["result_path"] for row in existing_rows}
    current_rows = [build_manifest_row(path, assignment) for path in completed_result_paths()]
    new_rows = [row for row in current_rows if row["result_path"] not in existing_paths]

    if existing_rows:
        all_rows = existing_rows + new_rows
        append_new_chunks(existing_rows, new_rows)
    else:
        all_rows = current_rows
        rebuild_chunk_index(all_rows)

    all_rows = sorted(all_rows, key=lambda row: int(row["config_number"]))
    write_jsonl(OUT_DIR / "all_rollouts_manifest.jsonl", all_rows)
    write_missing_ledger()

    print(
        f"manifest_rows={len(all_rows)} new_rows={len(new_rows)} "
        f"chunks={len(read_jsonl(OUT_DIR / 'chunk_index.jsonl'))} out_dir={OUT_DIR}"
    )


if __name__ == "__main__":
    main()

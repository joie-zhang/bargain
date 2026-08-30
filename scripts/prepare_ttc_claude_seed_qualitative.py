#!/usr/bin/env python3
"""Prepare stable rollout views for nine-seed Claude TTC adjudication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "experiments/results"
ORIGINAL_ADJUDICATION_ROOT = (
    PROJECT_ROOT / "analysis/ttc_llm_strategic_tag_adjudication_20260629"
)
OUTPUT_ROOT = (
    PROJECT_ROOT / "analysis/ttc_claude_seed_qualitative_adjudication_20260728"
)
TARGET_FAMILY = "claude-sonnet-4-6"
SOURCE_CONFIG_IDS = tuple(range(73, 145))
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
EXPECTED_ROLLOUTS_PER_SEED = 72
EXPECTED_TOTAL_ROLLOUTS = len(SEED_ORDER) * EXPECTED_ROLLOUTS_PER_SEED


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write_text(
        path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )


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


def derive_agents(config: dict[str, Any]) -> tuple[str, str]:
    target_position = int(config["target_position"])
    target_agent = f"Agent_{target_position + 1}"
    baseline_agent = "Agent_2" if target_agent == "Agent_1" else "Agent_1"
    return target_agent, baseline_agent


def build_manifest_row(
    *,
    seed: int,
    config_path: Path,
    result_path: Path,
    interactions_path: Path,
    rollout_view_path: Path,
) -> dict[str, Any]:
    source_config = json.loads(config_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    config = result.get("config") or source_config
    source_config_id = int(source_config["config_id"])
    target_agent, baseline_agent = derive_agents(config)
    utilities = result.get("final_utilities") or {}
    vote_integrity = result.get("vote_integrity") or config.get("vote_integrity") or {}
    if vote_integrity.get("hard_failed"):
        raise RuntimeError(f"Hard-failed result is not annotatable: {result_path}")

    return {
        "rollout_id": f"seed_{seed}_config_{source_config_id:04d}",
        "seed": seed,
        "source_config_id": source_config_id,
        "result_path": str(result_path.resolve()),
        "interactions_path": str(interactions_path.resolve()),
        "rollout_view_path": str(rollout_view_path.resolve()),
        "family": config["target_model_family"],
        "level": config["target_reasoning_level_requested"],
        "level_index": int(config["target_reasoning_level_index"]),
        "provider": config["target_provider"],
        "game_label": config["game_label"],
        "game_cell": config["game_cell_id"],
        "game_type": config["game_type"],
        "n_agents": int(config.get("n_agents") or 2),
        "order": config["order"],
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
        "result_sha256": sha256(result_path),
        "interactions_sha256": sha256(interactions_path),
    }


def build_rollout_view(manifest: dict[str, Any]) -> dict[str, Any]:
    result_path = Path(manifest["result_path"])
    interactions_path = Path(manifest["interactions_path"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    interactions = json.loads(interactions_path.read_text(encoding="utf-8"))
    config = result.get("config") or {}
    target_agent = manifest["target_agent"]

    conversation_logs = []
    for index, entry in enumerate(result.get("conversation_logs") or []):
        conversation_logs.append(
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
        )

    target_private_interactions = []
    for index, entry in enumerate(interactions):
        phase, round_number, discussion_turn = parse_phase(entry.get("phase"))
        if entry.get("agent_id") != target_agent:
            continue
        if phase in {"game_setup", "discussion"}:
            continue
        target_private_interactions.append(
            {
                "interaction_index": index,
                "source_kind": "interaction",
                "speaker_agent": target_agent,
                "phase_raw": entry.get("phase"),
                "phase": phase,
                "round": round_number if round_number is not None else entry.get("round"),
                "discussion_turn": discussion_turn,
                "model_name": entry.get("model_name"),
                "response": response_text(entry),
            }
        )

    return {
        "manifest": manifest,
        "config": {
            "game_label": config.get("game_label"),
            "game_cell_id": config.get("game_cell_id"),
            "game_type": config.get("game_type"),
            "target_reasoning_level_requested": config.get(
                "target_reasoning_level_requested"
            ),
            "target_reasoning_level_index": config.get(
                "target_reasoning_level_index"
            ),
            "order": config.get("order"),
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
            "vote_integrity": result.get("vote_integrity"),
        },
        "conversation_logs": conversation_logs,
        "target_private_interactions": target_private_interactions,
    }


def paper_tag_subset(codebook: list[dict[str, Any]]) -> list[dict[str, Any]]:
    review = json.loads((PROJECT_ROOT / "strategic_tag_review_final.json").read_text())
    hot_codes = {
        row["tag_code"]
        for row in review["responses"]
        if row.get("decision") == "hot"
    }
    subset = [
        row
        for row in codebook
        if row["tag_code"] in hot_codes and row.get("category") != "coalition"
    ]
    if len(subset) != 23:
        raise RuntimeError(f"Expected the paper subset to contain 23 tags, found {len(subset)}")
    return subset


def prepare(*, require_complete: bool) -> dict[str, Any]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rollout_view_dir = OUTPUT_ROOT / "rollout_views"
    rollout_view_dir.mkdir(parents=True, exist_ok=True)

    source_codebook_path = ORIGINAL_ADJUDICATION_ROOT / "llm_tag_codebook.json"
    codebook = json.loads(source_codebook_path.read_text(encoding="utf-8"))
    write_json(OUTPUT_ROOT / "llm_tag_codebook_full50.json", codebook)
    write_json(OUTPUT_ROOT / "paper_tag_subset_23.json", paper_tag_subset(codebook))

    manifest_rows: list[dict[str, Any]] = []
    missing: dict[str, list[int]] = {}
    configured_counts: dict[str, int] = {}
    available_counts: dict[str, int] = {}

    for seed in SEED_ORDER:
        root = RESULTS_ROOT / SEED_ROOT_NAMES[seed]
        configured = 0
        available = 0
        seed_missing = []
        for source_config_id in SOURCE_CONFIG_IDS:
            config_path = root / "configs" / f"config_{source_config_id:04d}.json"
            if not config_path.exists():
                raise RuntimeError(f"Missing source config: {config_path}")
            source_config = json.loads(config_path.read_text(encoding="utf-8"))
            if source_config.get("target_model_family") != TARGET_FAMILY:
                raise RuntimeError(
                    f"Config {config_path} is not a {TARGET_FAMILY} target"
                )
            configured += 1
            output_dir = Path(source_config["output_dir"])
            result_path = output_dir / "run_1_experiment_results.json"
            interactions_path = output_dir / "run_1_all_interactions.json"
            if not result_path.exists() or not interactions_path.exists():
                seed_missing.append(source_config_id)
                continue
            available += 1
            rollout_id = f"seed_{seed}_config_{source_config_id:04d}"
            rollout_view_path = rollout_view_dir / f"{rollout_id}.json"
            manifest = build_manifest_row(
                seed=seed,
                config_path=config_path,
                result_path=result_path,
                interactions_path=interactions_path,
                rollout_view_path=rollout_view_path,
            )
            if manifest["family"] != TARGET_FAMILY:
                raise RuntimeError(f"Unexpected result family in {result_path}")
            write_json(rollout_view_path, build_rollout_view(manifest))
            manifest_rows.append(manifest)
        configured_counts[str(seed)] = configured
        available_counts[str(seed)] = available
        missing[str(seed)] = seed_missing

    manifest_rows.sort(key=lambda row: (SEED_ORDER.index(row["seed"]), row["source_config_id"]))
    rollout_ids = [row["rollout_id"] for row in manifest_rows]
    if len(rollout_ids) != len(set(rollout_ids)):
        raise RuntimeError("Duplicate rollout IDs in prepared manifest")
    write_jsonl(OUTPUT_ROOT / "all_available_rollouts_manifest.jsonl", manifest_rows)

    inventory = {
        "target_family": TARGET_FAMILY,
        "seed_order": list(SEED_ORDER),
        "source_config_id_range": [SOURCE_CONFIG_IDS[0], SOURCE_CONFIG_IDS[-1]],
        "expected_rollouts_per_seed": EXPECTED_ROLLOUTS_PER_SEED,
        "expected_total_rollouts": EXPECTED_TOTAL_ROLLOUTS,
        "available_total_rollouts": len(manifest_rows),
        "configured_counts": configured_counts,
        "available_counts": available_counts,
        "missing_source_config_ids": missing,
        "source_roots": {
            str(seed): str((RESULTS_ROOT / SEED_ROOT_NAMES[seed]).resolve())
            for seed in SEED_ORDER
        },
        "source_codebook": str(source_codebook_path.resolve()),
        "source_codebook_sha256": sha256(source_codebook_path),
        "full_codebook_tag_count": len(codebook),
        "paper_subset_tag_count": 23,
        "annotation_scope": (
            "Target-authored Claude behavior plus target-attributable formal outcomes; "
            "baseline utterances remain in public context but are not labeled."
        ),
    }
    write_json(OUTPUT_ROOT / "source_inventory.json", inventory)
    if require_complete and len(manifest_rows) != EXPECTED_TOTAL_ROLLOUTS:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_ROLLOUTS} source rollouts, "
            f"found {len(manifest_rows)}; see source_inventory.json"
        )
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    inventory = prepare(require_complete=args.require_complete)
    print(json.dumps(inventory, indent=2))


if __name__ == "__main__":
    main()

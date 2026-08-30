#!/usr/bin/env python3
"""Generate and audit the 25-run GPT-4o-mini replacement for Claude 3 Haiku."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = (
    PROJECT_ROOT
    / "experiments/results/full_games123_random_monoculture_control_20260628_014357"
)
DEFAULT_RESULTS_ROOT = (
    PROJECT_ROOT
    / "experiments/results/full_games123_random_monoculture_gpt4omini_game1_replacement_20260816"
)
SOURCE_MODEL = "claude-3-haiku-20240307"
TARGET_MODEL = "gpt-4o-mini-2024-07-18"
TARGET_INTERACTION_MODEL = TARGET_MODEL
TARGET_ELO = 1317
TARGET_POOL_INDEX = 19
TARGET_OUTPUT_SLUG = "gpt_4o_mini_2024_07_18"
TARGET_EXPERIMENT_SLUG = "gpt4omini"
TARGET_BATCH_TYPE = "random_monoculture_gpt4omini_game1_replacement"
TARGET_DISPLAY_NAME = "GPT-4o-mini"
TARGET_MODEL_CONFIG_OVERRIDES: dict[str, Any] = {}
SOURCE_CONFIG_NUMBERS = tuple(range(1, 26))
REPLACE_SOURCE_MODEL_IN_SELECTED_ROSTER = True
EXPECTED_COMPETITION_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
EXPECTED_N_VALUES = (2, 4, 6, 8, 10)
TOKEN_LIMIT = 16384
SOURCE_FILES = (
    "run_strong_models_experiment.py",
    "game_environments/item_allocation.py",
    "negotiation/preferences.py",
    "negotiation/multi_agent_vector_generator.py",
    "negotiation/llm_agents.py",
    "strong_models_experiment/configs.py",
    "strong_models_experiment/experiment.py",
    "strong_models_experiment/phases/phase_handlers.py",
    "scripts/full_games123_multiagent_batch.py",
    "scripts/random_monoculture_control_batch.py",
    "scripts/generate_gpt4omini_game1_replacement.py",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_result_path(config_id: str) -> Path:
    matches = sorted((SOURCE_ROOT / "runs").glob(f"{config_id}_*/experiment_results.json"))
    if len(matches) != 1:
        raise ValueError(f"Expected one source result for {config_id}, found {len(matches)}")
    return matches[0].resolve()


def git_metadata() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    hashes = {
        relative: file_hash(PROJECT_ROOT / relative)
        for relative in SOURCE_FILES
        if (PROJECT_ROOT / relative).exists()
    }
    return {
        "git_head": head,
        "git_worktree_dirty": bool(status.strip()),
        "git_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "source_file_sha256": hashes,
    }


def generate(results_root: Path, force: bool) -> None:
    if results_root.exists() and any(results_root.iterdir()) and not force:
        raise FileExistsError(f"Refusing to overwrite non-empty result root: {results_root}")
    for name in ("configs", "runs", "status", "logs", "monitoring", "selections", "submissions", "slurm"):
        (results_root / name).mkdir(parents=True, exist_ok=True)

    configs: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    for number, source_number in enumerate(SOURCE_CONFIG_NUMBERS, start=1):
        config_id = f"config_{number:04d}"
        source_config_id = f"config_{source_number:04d}"
        source_config_path = SOURCE_ROOT / "configs" / f"{source_config_id}.json"
        source_config = read_json(source_config_path)
        source_result = source_result_path(source_config_id)
        source_payload = read_json(source_result)
        preferences = source_payload.get("agent_preferences")
        if not isinstance(preferences, dict) or len(preferences) != int(source_config["n_agents"]):
            raise ValueError(f"Invalid source preferences for {config_id}")

        config = dict(source_config)
        n_agents = int(config["n_agents"])
        models = [TARGET_MODEL] * n_agents
        if REPLACE_SOURCE_MODEL_IN_SELECTED_ROSTER:
            selected_models = [
                TARGET_MODEL if model == SOURCE_MODEL else model
                for model in config["selected_models"]
            ]
            expected_target_count = config["selected_models"].count(TARGET_MODEL) + 1
            if (
                len(selected_models) != 15
                or SOURCE_MODEL in selected_models
                or selected_models.count(TARGET_MODEL) != expected_target_count
            ):
                raise ValueError(f"Replacement selected-model roster is invalid for {config_id}")
        else:
            selected_models = list(config["selected_models"])
            if len(selected_models) != 15:
                raise ValueError(f"Source selected-model roster is invalid for {config_id}")

        output_dir = (
            results_root
            / "runs"
            / f"{config_id}_game1_n{n_agents}_{config['competition_id']}_{TARGET_OUTPUT_SLUG}"
        )
        config.update(
            {
                "config_id": config_id,
                "experiment_id": f"{TARGET_EXPERIMENT_SLUG}_game1_replacement_{config_id}",
                "batch_type": TARGET_BATCH_TYPE,
                "experiment_family": "random_monoculture_control_replacement",
                "experiment_type": "random_monoculture_control_replacement",
                "models": models,
                "baseline_model": TARGET_MODEL,
                "monoculture_model": TARGET_MODEL,
                "agent_model_map": {
                    f"Agent_{index}": TARGET_MODEL for index in range(1, n_agents + 1)
                },
                "agent_elo_map": {
                    f"Agent_{index}": TARGET_ELO for index in range(1, n_agents + 1)
                },
                "model_elo": TARGET_ELO,
                "model_pool_index": TARGET_POOL_INDEX,
                "selected_models": selected_models,
                "fixed_agent_preferences": preferences,
                "preference_source": "fixed_agent_preferences_from_source_result",
                "source_model": SOURCE_MODEL,
                "replacement_model": TARGET_MODEL,
                "replacement_source_config_id": source_config_id,
                "replacement_source_config_path": str(source_config_path.resolve()),
                "replacement_source_result_path": str(source_result),
                "replacement_source_preferences_sha256": canonical_hash(preferences),
                "post_hoc_sensitivity": True,
                "random_seed": int(source_config["random_seed"]),
                "seed": int(source_config["seed"]),
                "output_dir": str(output_dir.resolve()),
                "notes": (
                    f"Seed- and preference-matched {TARGET_DISPLAY_NAME} Game 1 replacement for the "
                    "original Claude 3 Haiku random-monoculture run."
                ),
                "max_tokens_discussion": TOKEN_LIMIT,
                "max_tokens_proposal": TOKEN_LIMIT,
                "max_tokens_voting": TOKEN_LIMIT,
                "max_tokens_reflection": TOKEN_LIMIT,
                "max_tokens_thinking": TOKEN_LIMIT,
                "max_tokens_default": TOKEN_LIMIT,
                "max_tokens_per_phase": TOKEN_LIMIT,
            }
        )
        if TARGET_MODEL_CONFIG_OVERRIDES:
            config["model_config_overrides"] = {
                TARGET_MODEL: dict(TARGET_MODEL_CONFIG_OVERRIDES)
            }
        write_json(results_root / "configs" / f"{config_id}.json", config)
        configs.append(config)
        mapping.append(
            {
                "config_id": config_id,
                "n_agents": n_agents,
                "competition_level": float(config["competition_level"]),
                "num_items": int(config["num_items"]),
                "random_seed": int(config["random_seed"]),
                "source_result_path": str(source_result),
                "preferences_sha256": canonical_hash(preferences),
                "output_dir": str(output_dir.resolve()),
            }
        )

    provenance = git_metadata()
    manifest = {
        "batch_type": TARGET_BATCH_TYPE,
        "results_root": str(results_root.resolve()),
        "source_root": str(SOURCE_ROOT.resolve()),
        "source_model": SOURCE_MODEL,
        "replacement_model": TARGET_MODEL,
        "source_model_elo": 1260,
        "replacement_model_elo": TARGET_ELO,
        "replacement_model_pool_index": TARGET_POOL_INDEX,
        "replacement_model_config_overrides": {
            TARGET_MODEL: dict(TARGET_MODEL_CONFIG_OVERRIDES)
        },
        "seed_policy": "reuse_source_config_seed",
        "preference_policy": "copy_source_final_artifact_preferences",
        "expected_total_configs": 25,
        "expected_game_counts": {"game1": 25},
        "n_values": list(EXPECTED_N_VALUES),
        "competition_levels": list(EXPECTED_COMPETITION_LEVELS),
        "slurm_time": "08:00:00",
        "slurm_max_concurrent": 5,
        "post_hoc_sensitivity": True,
        **provenance,
    }
    write_json(results_root / "manifest.json", manifest)
    write_json(results_root / "configs" / "source_mapping.json", mapping)
    all_ids = [config["config_id"] for config in configs]
    (results_root / "configs" / "all_configs.txt").write_text(
        "\n".join(all_ids) + "\n", encoding="utf-8"
    )
    (results_root / "selections" / "pilot_config_ids.txt").write_text(
        "config_0001\nconfig_0025\n", encoding="utf-8"
    )
    (results_root / "selections" / "remaining_config_ids.txt").write_text(
        "\n".join(all_ids[1:24]) + "\n", encoding="utf-8"
    )
    validate_configs(results_root)
    print(f"Generated 25 configurations under {results_root}")


def load_configs(results_root: Path) -> list[dict[str, Any]]:
    return [read_json(path) for path in sorted((results_root / "configs").glob("config_*.json"))]


def validate_configs(results_root: Path) -> None:
    configs = load_configs(results_root)
    errors: list[str] = []
    if len(configs) != 25:
        errors.append(f"expected 25 configs, found {len(configs)}")
    cells = Counter()
    seeds = set()
    for config in configs:
        config_id = str(config.get("config_id"))
        n_agents = int(config.get("n_agents", -1))
        competition = float(config.get("competition_level", math.nan))
        cells[(n_agents, competition)] += 1
        seeds.add(config.get("random_seed"))
        if config.get("game_label") != "game1" or config.get("game_type") != "item_allocation":
            errors.append(f"{config_id}: not a Game 1 item-allocation config")
        if config.get("models") != [TARGET_MODEL] * n_agents:
            errors.append(f"{config_id}: model roster mismatch")
        if config.get("monoculture_model") != TARGET_MODEL or config.get("model_elo") != TARGET_ELO:
            errors.append(f"{config_id}: replacement model metadata mismatch")
        if int(config.get("num_items", -1)) != int(2.5 * n_agents):
            errors.append(f"{config_id}: item count mismatch")
        if config.get("seed") != config.get("random_seed"):
            errors.append(f"{config_id}: seed fields differ")
        fixed = config.get("fixed_agent_preferences")
        if canonical_hash(fixed) != config.get("replacement_source_preferences_sha256"):
            errors.append(f"{config_id}: fixed preference hash mismatch")
        if any(config.get(key) != TOKEN_LIMIT for key in (
            "max_tokens_discussion", "max_tokens_proposal", "max_tokens_voting",
            "max_tokens_reflection", "max_tokens_thinking", "max_tokens_default",
            "max_tokens_per_phase",
        )):
            errors.append(f"{config_id}: token limit mismatch")
    expected_cells = {(n, c) for n in EXPECTED_N_VALUES for c in EXPECTED_COMPETITION_LEVELS}
    if set(cells) != expected_cells or any(count != 1 for count in cells.values()):
        errors.append("the 5 x 5 Game 1 grid is incomplete or duplicated")
    if len(seeds) != 25:
        errors.append(f"expected 25 distinct seeds, found {len(seeds)}")
    if errors:
        raise ValueError("Configuration validation failed:\n- " + "\n- ".join(errors))
    print(f"Configuration validation passed for {len(configs)} configs")


def result_path_for(config: dict[str, Any]) -> Path:
    path = Path(config["output_dir"]) / "experiment_results.json"
    return path.resolve()


def realized_preferences(payload: dict[str, Any]) -> dict[str, list[float]]:
    preferences = payload.get("agent_preferences") or {}
    if preferences:
        return preferences
    if payload.get("consensus_reached"):
        raise ValueError("consensus result is missing agent_preferences")
    # The experiment serializer leaves this top-level field empty after a
    # no-consensus result. The immutable locked values remain in the saved
    # resolved config and were used throughout the run.
    return (payload.get("config") or {}).get("fixed_agent_preferences") or {}


def recompute_game1_utilities(payload: dict[str, Any]) -> dict[str, float]:
    preferences = realized_preferences(payload)
    if not payload.get("consensus_reached"):
        return {agent_id: 0.0 for agent_id in preferences}
    allocation = payload["final_allocation"]
    discount = float(payload["config"]["gamma_discount"]) ** (int(payload["final_round"]) - 1)
    return {
        agent_id: sum(float(preferences[agent_id][int(item)]) for item in allocation[agent_id]) * discount
        for agent_id in preferences
    }


def audit_results(results_root: Path, require_all: bool) -> None:
    validate_configs(results_root)
    configs = load_configs(results_root)
    errors: list[str] = []
    completed = 0
    for config in configs:
        config_id = config["config_id"]
        result_path = result_path_for(config)
        if not result_path.exists():
            if require_all:
                errors.append(f"{config_id}: missing result")
            continue
        completed += 1
        payload = read_json(result_path)
        result_config = payload.get("config") or {}
        if result_config.get("models") != config["models"]:
            errors.append(f"{config_id}: saved model roster mismatch")
        if int(result_config.get("random_seed", -1)) != int(config["random_seed"]):
            errors.append(f"{config_id}: saved seed mismatch")
        if realized_preferences(payload) != config["fixed_agent_preferences"]:
            errors.append(f"{config_id}: realized preferences differ from locked preferences")
        vote_integrity = payload.get("vote_integrity") or {}
        for key in ("synthetic_vote_used", "contaminated", "hard_failed"):
            if vote_integrity.get(key):
                errors.append(f"{config_id}: vote_integrity.{key} is true")
        expected_utilities = recompute_game1_utilities(payload)
        actual_utilities = {key: float(value) for key, value in payload["final_utilities"].items()}
        for agent_id, expected in expected_utilities.items():
            if not math.isclose(expected, actual_utilities[agent_id], rel_tol=1e-9, abs_tol=1e-8):
                errors.append(
                    f"{config_id}: utility mismatch for {agent_id}: "
                    f"expected {expected}, observed {actual_utilities[agent_id]}"
                )
        interactions_path = Path(config["output_dir"]) / "all_interactions.json"
        if not interactions_path.exists():
            errors.append(f"{config_id}: missing all_interactions.json")
        else:
            interactions = read_json(interactions_path)
            model_names = {str(row.get("model_name")) for row in interactions if row.get("model_name")}
            if model_names != {TARGET_INTERACTION_MODEL}:
                errors.append(f"{config_id}: interaction model names are {sorted(model_names)}")
    if errors:
        raise ValueError("Result audit failed:\n- " + "\n- ".join(errors))
    print(f"Result audit passed for {completed}/25 completed configs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("generate", "validate-configs", "audit-results")
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--require-all", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    results_root = args.results_root.resolve()
    if args.command == "generate":
        generate(results_root, args.force)
    elif args.command == "validate-configs":
        validate_configs(results_root)
    else:
        audit_results(results_root, args.require_all)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate and audit the four-run matched GPT-5.4 Game 1 coalition pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = (
    PROJECT_ROOT
    / "experiments/results/full_games123_random_monoculture_control_20260628_014357"
)
DEFAULT_RESULTS_ROOT = (
    PROJECT_ROOT
    / "experiments/results/game1_gpt54_high_matched_coalition_pilot_20260829"
)
SOURCE_CASES = (
    PROJECT_ROOT
    / "docs/analysis/assets/minimum_winning_coalition_20260817"
    / "game1_coalition_proposer_elo_cases.csv"
)
TARGET_MODEL = "gpt-5.4-high"
TARGET_MODEL_ID = "openai/gpt-5.4"
TARGET_ELO = 1484
TOKEN_LIMIT = 65_536
SOURCE_CONFIG_IDS = (118, 119, 124, 125)
EXPECTED_CELLS = ((8, 0.5), (8, 0.75), (10, 0.75), (10, 1.0))
LIST_PRICE_INPUT_PER_MILLION = 2.50
LIST_PRICE_OUTPUT_PER_MILLION = 15.00
SOURCE_FILES = (
    "run_strong_models_experiment.py",
    "game_environments/item_allocation.py",
    "negotiation/preferences.py",
    "negotiation/multi_agent_vector_generator.py",
    "negotiation/llm_agents.py",
    "negotiation/openrouter_client.py",
    "strong_models_experiment/configs.py",
    "strong_models_experiment/experiment.py",
    "strong_models_experiment/phases/phase_handlers.py",
    "scripts/full_games123_multiagent_batch.py",
    "scripts/random_monoculture_control_batch.py",
    "scripts/generate_gpt54_game1_coalition_pilot.py",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonical_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_result_path(source_id: int) -> Path:
    matches = sorted(
        (SOURCE_ROOT / "runs").glob(
            f"config_{source_id:04d}_*/experiment_results.json"
        )
    )
    if len(matches) != 1:
        raise ValueError(
            f"Expected one source result for config_{source_id:04d}, found {len(matches)}"
        )
    return matches[0].resolve()


def positive_source_ids() -> set[int]:
    found: set[int] = set()
    with SOURCE_CASES.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row["run_family"] == "random_monoculture"
                and row["primary_organizer_model"] == "gemini-3.1-pro"
            ):
                found.add(int(row["config_id"]))
    return found


def git_metadata() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "git_head": head,
        "git_worktree_dirty": bool(status.strip()),
        "git_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "source_file_sha256": {
            relative: file_hash(PROJECT_ROOT / relative)
            for relative in SOURCE_FILES
            if (PROJECT_ROOT / relative).exists()
        },
    }


def generate(results_root: Path, provider_route: str) -> None:
    if results_root.exists() and any(results_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty root: {results_root}")
    for name in (
        "configs",
        "runs",
        "status",
        "logs",
        "monitoring",
        "selections",
        "submissions",
        "slurm",
        "analysis",
    ):
        (results_root / name).mkdir(parents=True, exist_ok=True)

    positive_ids = positive_source_ids()
    missing = set(SOURCE_CONFIG_IDS) - positive_ids
    if missing:
        raise ValueError(f"Source configs are not adjudicated Gemini-positive: {sorted(missing)}")

    mappings: list[dict[str, Any]] = []
    for pilot_id, source_id in enumerate(SOURCE_CONFIG_IDS, start=1):
        source_config_path = SOURCE_ROOT / "configs" / f"config_{source_id:04d}.json"
        source_config = read_json(source_config_path)
        source_result = source_result_path(source_id)
        source_payload = read_json(source_result)
        preferences = source_payload.get("agent_preferences")
        n_agents = int(source_config["n_agents"])
        if not isinstance(preferences, dict) or len(preferences) != n_agents:
            raise ValueError(f"Invalid source preferences for config_{source_id:04d}")

        config_id = f"config_{pilot_id:04d}"
        competition_id = str(source_config["competition_id"])
        output_dir = (
            results_root
            / "runs"
            / f"{config_id}_game1_n{n_agents}_{competition_id}_gpt_5p4_high"
        )
        models = [TARGET_MODEL] * n_agents
        config = dict(source_config)
        config.update(
            {
                "config_id": config_id,
                "experiment_id": f"gpt54_high_matched_coalition_pilot_{config_id}",
                "batch_type": "game1_gpt54_high_matched_coalition_pilot",
                "experiment_family": "random_monoculture_control_matched_pilot",
                "experiment_type": "random_monoculture_control_matched_pilot",
                "models": models,
                "baseline_model": TARGET_MODEL,
                "monoculture_model": TARGET_MODEL,
                "agent_model_map": {
                    f"Agent_{index}": TARGET_MODEL
                    for index in range(1, n_agents + 1)
                },
                "agent_elo_map": {
                    f"Agent_{index}": TARGET_ELO
                    for index in range(1, n_agents + 1)
                },
                "agent_role_map": {
                    f"Agent_{index}": "random_monoculture_control"
                    for index in range(1, n_agents + 1)
                },
                "model_elo": TARGET_ELO,
                "elo_snapshot_date": "2026-03-31",
                "provider_route": provider_route,
                "requested_provider_model_id": (
                    "gpt-5.4" if provider_route == "direct-openai" else TARGET_MODEL_ID
                ),
                "requested_reasoning_effort": "high",
                "fixed_agent_preferences": preferences,
                "preference_source": "fixed_agent_preferences_from_source_result",
                "source_model": "gemini-3.1-pro",
                "source_config_id": f"config_{source_id:04d}",
                "source_config_path": str(source_config_path.resolve()),
                "source_result_path": str(source_result),
                "source_preferences_sha256": canonical_hash(preferences),
                "source_harmful_coalition_adjudicated": True,
                "pilot_selection_basis": "post_hoc_Gemini_positive_challenge_set",
                "primary_outcome": "any_harmful_exclusionary_coalition_proposal",
                "random_seed": int(source_config["random_seed"]),
                "seed": int(source_config["seed"]),
                "output_dir": str(output_dir.resolve()),
                "max_tokens_discussion": TOKEN_LIMIT,
                "max_tokens_proposal": TOKEN_LIMIT,
                "max_tokens_voting": TOKEN_LIMIT,
                "max_tokens_reflection": TOKEN_LIMIT,
                "max_tokens_thinking": TOKEN_LIMIT,
                "max_tokens_default": TOKEN_LIMIT,
                "max_tokens_per_phase": TOKEN_LIMIT,
                "notes": (
                    "Post-hoc detection pilot using GPT-5.4 High in every seat and the exact "
                    "saved preferences from one Gemini 3.1 Pro coalition-positive environment."
                ),
            }
        )
        if provider_route == "direct-openai":
            config["model_config_overrides"] = {
                TARGET_MODEL: {
                    "api_type": "openai",
                    "provider": "OpenAI",
                    "model_id": "gpt-5.4",
                    "reasoning_effort": "high",
                    "custom_parameters": {
                        "phase_token_cap_policy": "prefer_model_cap_when_experiment_default"
                    },
                }
            }
        write_json(results_root / "configs" / f"{config_id}.json", config)
        mappings.append(
            {
                "config_id": config_id,
                "source_config_id": f"config_{source_id:04d}",
                "n_agents": n_agents,
                "competition_level": float(config["competition_level"]),
                "random_seed": int(config["random_seed"]),
                "preferences_sha256": canonical_hash(preferences),
                "source_result_path": str(source_result),
                "output_dir": str(output_dir.resolve()),
            }
        )

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "batch_type": "game1_gpt54_high_matched_coalition_pilot",
        "results_root": str(results_root.resolve()),
        "source_root": str(SOURCE_ROOT.resolve()),
        "source_case_audit": str(SOURCE_CASES.resolve()),
        "target_model": TARGET_MODEL,
        "target_model_id": "gpt-5.4" if provider_route == "direct-openai" else TARGET_MODEL_ID,
        "provider_route": provider_route,
        "target_elo": TARGET_ELO,
        "target_reasoning_effort": "high",
        "expected_total_configs": 4,
        "source_config_ids": [f"config_{value:04d}" for value in SOURCE_CONFIG_IDS],
        "selection_policy": "four post-hoc Gemini-positive high-n environments",
        "preference_policy": "copy exact source result preferences and verify SHA-256",
        "slurm_time": "12:00:00",
        "slurm_max_concurrent": 3,
        "api_transport": "file_proxy",
        "api_proxy_poll_dir": "/home/jz4391/openrouter_proxy",
        "list_price_input_per_million_usd": LIST_PRICE_INPUT_PER_MILLION,
        "list_price_output_per_million_usd": LIST_PRICE_OUTPUT_PER_MILLION,
        "planning_cost_cap_usd": 70,
        **git_metadata(),
    }
    write_json(results_root / "manifest.json", manifest)
    write_json(results_root / "configs" / "source_mapping.json", mappings)
    (results_root / "selections" / "access_check_config_ids.txt").write_text(
        "config_0001\n", encoding="utf-8"
    )
    (results_root / "selections" / "remaining_config_ids.txt").write_text(
        "config_0002\nconfig_0003\nconfig_0004\n", encoding="utf-8"
    )
    (results_root / "selections" / "all_config_ids.txt").write_text(
        "config_0001\nconfig_0002\nconfig_0003\nconfig_0004\n", encoding="utf-8"
    )
    validate_configs(results_root)
    print(f"Generated four pilot configs under {results_root}")


def load_configs(results_root: Path) -> list[dict[str, Any]]:
    return [read_json(path) for path in sorted((results_root / "configs").glob("config_*.json"))]


def validate_configs(results_root: Path) -> None:
    configs = load_configs(results_root)
    errors: list[str] = []
    if len(configs) != 4:
        errors.append(f"expected four configs, found {len(configs)}")
    observed_cells: list[tuple[int, float]] = []
    for index, config in enumerate(configs, start=1):
        config_id = f"config_{index:04d}"
        n_agents = int(config.get("n_agents", -1))
        observed_cells.append((n_agents, float(config.get("competition_level", math.nan))))
        if config.get("config_id") != config_id:
            errors.append(f"{config_id}: config ID mismatch")
        if config.get("game_label") != "game1" or config.get("game_type") != "item_allocation":
            errors.append(f"{config_id}: not Game 1")
        if config.get("models") != [TARGET_MODEL] * n_agents:
            errors.append(f"{config_id}: model roster mismatch")
        provider_route = config.get("provider_route", "openrouter")
        if provider_route not in {"openrouter", "direct-openai"}:
            errors.append(f"{config_id}: invalid provider route {provider_route!r}")
        if provider_route == "direct-openai":
            override = (config.get("model_config_overrides") or {}).get(TARGET_MODEL) or {}
            if (
                override.get("api_type") != "openai"
                or override.get("model_id") != "gpt-5.4"
                or override.get("reasoning_effort") != "high"
            ):
                errors.append(f"{config_id}: direct OpenAI override mismatch")
        if config.get("model_order") != "random_monoculture_control":
            errors.append(f"{config_id}: model order mismatch")
        if int(config.get("max_rounds", -1)) != 10 or int(config.get("discussion_turns", -1)) != 2:
            errors.append(f"{config_id}: protocol mismatch")
        if not math.isclose(float(config.get("gamma_discount", -1)), 0.9):
            errors.append(f"{config_id}: discount mismatch")
        if config.get("parallel_phases") is not True:
            errors.append(f"{config_id}: parallel phases disabled")
        fixed = config.get("fixed_agent_preferences")
        if not isinstance(fixed, dict) or len(fixed) != n_agents:
            errors.append(f"{config_id}: invalid fixed preferences")
        elif canonical_hash(fixed) != config.get("source_preferences_sha256"):
            errors.append(f"{config_id}: fixed preference hash mismatch")
        if any(
            config.get(key) != TOKEN_LIMIT
            for key in (
                "max_tokens_discussion",
                "max_tokens_proposal",
                "max_tokens_voting",
                "max_tokens_reflection",
                "max_tokens_thinking",
                "max_tokens_default",
                "max_tokens_per_phase",
            )
        ):
            errors.append(f"{config_id}: token limits do not match GPT-5.4 High")
    if tuple(observed_cells) != EXPECTED_CELLS:
        errors.append(f"cell mismatch: {observed_cells}")
    if errors:
        raise ValueError("Config validation failed:\n- " + "\n- ".join(errors))
    print("Configuration validation passed for four configs")


def realized_preferences(payload: dict[str, Any]) -> dict[str, list[float]]:
    preferences = payload.get("agent_preferences") or {}
    if preferences:
        return preferences
    return (payload.get("config") or {}).get("fixed_agent_preferences") or {}


def audit_results(results_root: Path, require_all: bool) -> None:
    validate_configs(results_root)
    errors: list[str] = []
    summaries: list[dict[str, Any]] = []
    for config in load_configs(results_root):
        config_id = config["config_id"]
        output_dir = Path(config["output_dir"])
        result_path = output_dir / "experiment_results.json"
        if not result_path.exists():
            if require_all:
                errors.append(f"{config_id}: missing result")
            continue
        payload = read_json(result_path)
        result_config = payload.get("config") or {}
        if result_config.get("models") != config["models"]:
            errors.append(f"{config_id}: saved roster mismatch")
        if realized_preferences(payload) != config["fixed_agent_preferences"]:
            errors.append(f"{config_id}: realized preferences mismatch")
        vote_integrity = payload.get("vote_integrity") or {}
        for key in ("synthetic_vote_used", "contaminated", "hard_failed"):
            if vote_integrity.get(key):
                errors.append(f"{config_id}: vote_integrity.{key} is true")

        input_tokens = 0
        output_tokens = 0
        model_names: set[str] = set()
        for interaction_path in output_dir.glob("agent_*_interactions.json"):
            interaction_payload = read_json(interaction_path)
            for row in interaction_payload.get("interactions", []):
                if row.get("model_name"):
                    model_names.add(str(row["model_name"]))
                usage = row.get("token_usage") or {}
                input_tokens += int(usage.get("input_tokens", 0) or 0)
                output_tokens += int(usage.get("output_tokens", 0) or 0)
        expected_model_name = (
            "gpt-5.4"
            if config.get("provider_route") == "direct-openai"
            else TARGET_MODEL
        )
        if model_names != {expected_model_name}:
            errors.append(f"{config_id}: interaction model names are {sorted(model_names)}")
        estimated_cost = (
            input_tokens * LIST_PRICE_INPUT_PER_MILLION
            + output_tokens * LIST_PRICE_OUTPUT_PER_MILLION
        ) / 1_000_000
        summaries.append(
            {
                "config_id": config_id,
                "source_config_id": config["source_config_id"],
                "n_agents": config["n_agents"],
                "competition_level": config["competition_level"],
                "consensus_reached": bool(payload.get("consensus_reached")),
                "final_round": payload.get("final_round"),
                "final_utilities": payload.get("final_utilities"),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "estimated_list_cost_usd": round(estimated_cost, 6),
                "result_path": str(result_path.resolve()),
            }
        )
    if errors:
        raise ValueError("Result audit failed:\n- " + "\n- ".join(errors))
    summary = {
        "completed": len(summaries),
        "expected": 4,
        "input_tokens": sum(row["input_tokens"] for row in summaries),
        "output_tokens": sum(row["output_tokens"] for row in summaries),
        "estimated_list_cost_usd": round(
            sum(row["estimated_list_cost_usd"] for row in summaries), 6
        ),
        "runs": summaries,
    }
    write_json(results_root / "analysis" / "result_audit_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "validate-configs", "audit-results"))
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--provider-route",
        choices=("openrouter", "direct-openai"),
        default="openrouter",
    )
    parser.add_argument("--require-all", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    results_root = args.results_root.resolve()
    if args.command == "generate":
        generate(results_root, args.provider_route)
    elif args.command == "validate-configs":
        validate_configs(results_root)
    else:
        audit_results(results_root, args.require_all)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

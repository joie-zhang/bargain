#!/usr/bin/env python3
"""Add the remaining coalition-eligible Gemini-matched Game 1 cells."""

from __future__ import annotations

import json
from pathlib import Path

import generate_gpt54_game1_coalition_pilot as pilot


RESULTS_ROOT = (
    pilot.PROJECT_ROOT
    / "experiments/results/game1_gpt54_high_matched_coalition_pilot_direct_openai_20260829"
)
SOURCE_IDS = (
    106, 107, 108, 109, 110,
    111, 112, 113, 114, 115,
    116, 117, 120,
    121, 122, 123,
)
REPLACEMENT_SOURCE_ID = 125


def build_config(config_number: int, source_id: int, replacement: bool = False) -> dict:
    source_config_path = pilot.SOURCE_ROOT / "configs" / f"config_{source_id:04d}.json"
    source_config = pilot.read_json(source_config_path)
    source_result = pilot.source_result_path(source_id)
    source_payload = pilot.read_json(source_result)
    preferences = source_payload.get("agent_preferences")
    n_agents = int(source_config["n_agents"])
    if not isinstance(preferences, dict) or len(preferences) != n_agents:
        raise ValueError(f"Invalid source preferences for config_{source_id:04d}")

    config_id = f"config_{config_number:04d}"
    competition_id = str(source_config["competition_id"])
    suffix = "_replacement" if replacement else ""
    output_dir = (
        RESULTS_ROOT
        / "runs"
        / f"{config_id}_game1_n{n_agents}_{competition_id}_gpt_5p4_high{suffix}"
    )
    models = [pilot.TARGET_MODEL] * n_agents
    config = dict(source_config)
    config.update(
        {
            "config_id": config_id,
            "experiment_id": f"gpt54_high_matched_full_grid_{config_id}",
            "batch_type": "game1_gpt54_high_matched_full_grid",
            "experiment_family": "random_monoculture_control_matched",
            "experiment_type": "random_monoculture_control_matched",
            "models": models,
            "baseline_model": pilot.TARGET_MODEL,
            "monoculture_model": pilot.TARGET_MODEL,
            "agent_model_map": {
                f"Agent_{index}": pilot.TARGET_MODEL for index in range(1, n_agents + 1)
            },
            "agent_elo_map": {
                f"Agent_{index}": pilot.TARGET_ELO for index in range(1, n_agents + 1)
            },
            "agent_role_map": {
                f"Agent_{index}": "random_monoculture_control"
                for index in range(1, n_agents + 1)
            },
            "model_elo": pilot.TARGET_ELO,
            "elo_snapshot_date": "2026-03-31",
            "provider_route": "direct-openai",
            "requested_provider_model_id": "gpt-5.4",
            "requested_reasoning_effort": "high",
            "fixed_agent_preferences": preferences,
            "preference_source": "fixed_agent_preferences_from_source_result",
            "source_model": "gemini-3.1-pro",
            "source_config_id": f"config_{source_id:04d}",
            "source_config_path": str(source_config_path.resolve()),
            "source_result_path": str(source_result),
            "source_preferences_sha256": pilot.canonical_hash(preferences),
            "source_harmful_coalition_adjudicated": source_id in pilot.positive_source_ids(),
            "pilot_selection_basis": "full_Gemini_Game1_coalition_eligible_grid",
            "primary_outcome": "any_harmful_exclusionary_coalition_proposal",
            "random_seed": int(source_config["random_seed"]),
            "seed": int(source_config["seed"]),
            "output_dir": str(output_dir.resolve()),
            "max_tokens_discussion": pilot.TOKEN_LIMIT,
            "max_tokens_proposal": pilot.TOKEN_LIMIT,
            "max_tokens_voting": pilot.TOKEN_LIMIT,
            "max_tokens_reflection": pilot.TOKEN_LIMIT,
            "max_tokens_thinking": pilot.TOKEN_LIMIT,
            "max_tokens_default": pilot.TOKEN_LIMIT,
            "max_tokens_per_phase": pilot.TOKEN_LIMIT,
            "model_config_overrides": {
                pilot.TARGET_MODEL: {
                    "api_type": "openai",
                    "provider": "OpenAI",
                    "model_id": "gpt-5.4",
                    "reasoning_effort": "high",
                    "custom_parameters": {
                        "phase_token_cap_policy": "prefer_model_cap_when_experiment_default"
                    },
                }
            },
            "replacement_for_failed_config": "config_0004" if replacement else None,
            "notes": (
                "Native-OpenAI-only matched Game 1 run using the exact saved preferences "
                "from the corresponding Gemini 3.1 Pro environment."
            ),
        }
    )
    return config


def main() -> int:
    mappings_path = RESULTS_ROOT / "configs" / "source_mapping.json"
    mappings = pilot.read_json(mappings_path)
    existing_ids = {row["config_id"] for row in mappings}

    additions = [(index, source_id, False) for index, source_id in enumerate(SOURCE_IDS, 5)]
    additions.append((21, REPLACEMENT_SOURCE_ID, True))
    for config_number, source_id, replacement in additions:
        config = build_config(config_number, source_id, replacement)
        config_id = config["config_id"]
        config_path = RESULTS_ROOT / "configs" / f"{config_id}.json"
        if config_path.exists():
            observed = pilot.read_json(config_path)
            if observed != config:
                raise FileExistsError(f"Existing config differs: {config_path}")
        else:
            pilot.write_json(config_path, config)
        if config_id not in existing_ids:
            mappings.append(
                {
                    "config_id": config_id,
                    "source_config_id": config["source_config_id"],
                    "n_agents": config["n_agents"],
                    "competition_level": config["competition_level"],
                    "random_seed": config["random_seed"],
                    "preferences_sha256": config["source_preferences_sha256"],
                    "source_result_path": config["source_result_path"],
                    "output_dir": config["output_dir"],
                    "replacement_for_failed_config": config["replacement_for_failed_config"],
                }
            )
            existing_ids.add(config_id)

    pilot.write_json(mappings_path, mappings)
    selection = ["config_0021", *(f"config_{index:04d}" for index in range(5, 21))]
    selection_path = RESULTS_ROOT / "selections" / "budget_109p50_direct_openai_config_ids.txt"
    selection_path.write_text("\n".join(selection) + "\n", encoding="utf-8")

    manifest_path = RESULTS_ROOT / "manifest.json"
    manifest = pilot.read_json(manifest_path)
    manifest["full_eligible_grid_target"] = 20
    manifest["direct_openai_only_continuation_budget_usd"] = 109.50
    manifest["openrouter_provider_fallback_allowed"] = False
    manifest["extended_config_count"] = 21
    manifest["replacement_config_id"] = "config_0021"
    pilot.write_json(manifest_path, manifest)

    print(json.dumps({
        "results_root": str(RESULTS_ROOT),
        "added_or_verified": len(additions),
        "selection": str(selection_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

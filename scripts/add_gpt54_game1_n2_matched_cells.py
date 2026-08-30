#!/usr/bin/env python3
"""Add the five missing N=2 cells to the matched GPT-5.4 Game 1 grid."""

from __future__ import annotations

import json

import extend_gpt54_game1_matched_grid as grid
import generate_gpt54_game1_coalition_pilot as pilot


SOURCE_IDS = (101, 102, 103, 104, 105)
CONFIG_NUMBERS = (22, 23, 24, 25, 26)
SELECTION_NAME = "n2_full_grid_completion_direct_openai_config_ids.txt"


def main() -> int:
    results_root = grid.RESULTS_ROOT
    mappings_path = results_root / "configs" / "source_mapping.json"
    mappings = pilot.read_json(mappings_path)
    mappings_by_config = {row["config_id"]: row for row in mappings}

    for config_number, source_id in zip(CONFIG_NUMBERS, SOURCE_IDS, strict=True):
        config = grid.build_config(config_number, source_id)
        config["pilot_selection_basis"] = "full_Gemini_Game1_grid_completion"
        config["primary_outcome"] = "matched_Game1_outcome"
        config["notes"] = (
            "Native-OpenAI-only matched Game 1 N=2 run using the exact saved "
            "preferences from the corresponding Gemini 3.1 Pro environment."
        )
        config_id = config["config_id"]
        config_path = results_root / "configs" / f"{config_id}.json"
        if config_path.exists():
            observed = pilot.read_json(config_path)
            if observed != config:
                raise FileExistsError(f"Existing config differs: {config_path}")
        else:
            pilot.write_json(config_path, config)

        mapping = {
            "config_id": config_id,
            "source_config_id": config["source_config_id"],
            "n_agents": config["n_agents"],
            "competition_level": config["competition_level"],
            "random_seed": config["random_seed"],
            "preferences_sha256": config["source_preferences_sha256"],
            "source_result_path": config["source_result_path"],
            "output_dir": config["output_dir"],
            "replacement_for_failed_config": None,
        }
        existing = mappings_by_config.get(config_id)
        if existing is not None and existing != mapping:
            raise ValueError(f"Existing source mapping differs for {config_id}")
        if existing is None:
            mappings.append(mapping)
            mappings_by_config[config_id] = mapping

    pilot.write_json(mappings_path, mappings)
    selection = [f"config_{value:04d}" for value in CONFIG_NUMBERS]
    selection_path = results_root / "selections" / SELECTION_NAME
    selection_path.write_text("\n".join(selection) + "\n", encoding="utf-8")

    manifest_path = results_root / "manifest.json"
    manifest = pilot.read_json(manifest_path)
    manifest["full_game1_grid_target"] = 25
    manifest["coalition_eligible_grid_target"] = 20
    manifest["configured_source_cell_count"] = 25
    manifest["configured_run_count_including_replacement"] = 26
    manifest["n2_completion_config_ids"] = selection
    manifest["n2_completion_source_config_ids"] = [
        f"config_{value:04d}" for value in SOURCE_IDS
    ]
    manifest["n2_completion_budget_cap_usd"] = 109.50
    manifest["n2_completion_prior_recorded_spend_usd"] = 81.3564775
    manifest["openrouter_provider_fallback_allowed"] = False
    pilot.write_json(manifest_path, manifest)

    print(
        json.dumps(
            {
                "results_root": str(results_root),
                "added_or_verified": len(selection),
                "selection": str(selection_path),
                "source_config_ids": [f"config_{value:04d}" for value in SOURCE_IDS],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

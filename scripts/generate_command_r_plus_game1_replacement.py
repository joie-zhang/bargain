#!/usr/bin/env python3
"""Generate and audit the 25-run Command R Plus replacement for Claude 3 Haiku."""

from __future__ import annotations

from pathlib import Path

import generate_gpt4omini_game1_replacement as replacement


PROJECT_ROOT = Path(__file__).resolve().parents[1]

replacement.DEFAULT_RESULTS_ROOT = (
    PROJECT_ROOT
    / "experiments/results/full_games123_random_monoculture_command_r_plus_game1_replacement_20260816"
)
replacement.TARGET_MODEL = "command-r-plus-08-2024"
replacement.TARGET_INTERACTION_MODEL = "cohere/command-r-plus-08-2024"
replacement.TARGET_ELO = 1276
replacement.TARGET_POOL_INDEX = 21
replacement.TARGET_OUTPUT_SLUG = "command_r_plus_08_2024"
replacement.TARGET_EXPERIMENT_SLUG = "command_r_plus"
replacement.TARGET_BATCH_TYPE = "random_monoculture_command_r_plus_game1_replacement"
replacement.TARGET_DISPLAY_NAME = "Command R Plus 08-2024"
replacement.TARGET_MODEL_CONFIG_OVERRIDES = {
    "max_tokens_discussion": 4000,
    "max_tokens_thinking": 4000,
    "max_tokens_proposal": 4000,
    "max_tokens_voting": 4000,
    "max_tokens_reflection": 4000,
    "max_tokens_default": 4000,
}
replacement.SOURCE_FILES = tuple(
    dict.fromkeys(
        replacement.SOURCE_FILES
        + ("scripts/generate_command_r_plus_game1_replacement.py",)
    )
)


if __name__ == "__main__":
    raise SystemExit(replacement.main())

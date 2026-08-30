#!/usr/bin/env python3
"""Generate and audit an exact-cell GPT-5 Nano High direct-OpenAI rerun."""

from __future__ import annotations

from pathlib import Path

import generate_gpt4omini_game1_replacement as replacement


PROJECT_ROOT = Path(__file__).resolve().parents[1]

replacement.DEFAULT_RESULTS_ROOT = (
    PROJECT_ROOT
    / "experiments/results/full_games123_random_monoculture_gpt5nano_high_direct_rerun_20260816"
)
replacement.SOURCE_MODEL = "gpt-5-nano-high"
replacement.TARGET_MODEL = "gpt-5-nano-high"
replacement.TARGET_INTERACTION_MODEL = "gpt-5-nano"
replacement.TARGET_ELO = 1337
replacement.TARGET_POOL_INDEX = 16
replacement.TARGET_OUTPUT_SLUG = "gpt_5_nano_high_direct"
replacement.TARGET_EXPERIMENT_SLUG = "gpt5nano_high_direct_rerun"
replacement.TARGET_BATCH_TYPE = "random_monoculture_gpt5nano_high_direct_rerun"
replacement.TARGET_DISPLAY_NAME = "GPT-5 Nano High via direct OpenAI"
replacement.SOURCE_CONFIG_NUMBERS = tuple(range(26, 51))
replacement.REPLACE_SOURCE_MODEL_IN_SELECTED_ROSTER = False
replacement.TOKEN_LIMIT = 32768
replacement.TARGET_MODEL_CONFIG_OVERRIDES = {
    "model_id": "gpt-5-nano",
    "provider": "OpenAI",
    "api_type": "openai",
    "reasoning_effort": "high",
    "custom_parameters": {},
}
replacement.SOURCE_FILES = tuple(
    dict.fromkeys(
        replacement.SOURCE_FILES
        + ("scripts/generate_gpt5nano_high_direct_rerun.py",)
    )
)


if __name__ == "__main__":
    raise SystemExit(replacement.main())

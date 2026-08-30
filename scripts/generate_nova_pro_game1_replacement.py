#!/usr/bin/env python3
"""Generate and audit the 25-run Nova Pro replacement for Claude 3 Haiku."""

from __future__ import annotations

from pathlib import Path

import generate_gpt4omini_game1_replacement as replacement


PROJECT_ROOT = Path(__file__).resolve().parents[1]

replacement.DEFAULT_RESULTS_ROOT = (
    PROJECT_ROOT
    / "experiments/results/full_games123_random_monoculture_nova_pro_game1_replacement_20260816"
)
replacement.TARGET_MODEL = "amazon-nova-pro-v1.0"
replacement.TARGET_INTERACTION_MODEL = "amazon/nova-pro-v1"
replacement.TARGET_ELO = 1290
replacement.TARGET_POOL_INDEX = 20
replacement.TARGET_OUTPUT_SLUG = "amazon_nova_pro_v1p0"
replacement.TARGET_EXPERIMENT_SLUG = "nova_pro"
replacement.TARGET_BATCH_TYPE = "random_monoculture_nova_pro_game1_replacement"
replacement.TARGET_DISPLAY_NAME = "Amazon Nova Pro v1.0"
replacement.TARGET_MODEL_CONFIG_OVERRIDES = {
    "max_tokens_discussion": 5120,
    "max_tokens_thinking": 5120,
    "max_tokens_proposal": 5120,
    "max_tokens_voting": 5120,
    "max_tokens_reflection": 5120,
    "max_tokens_default": 5120,
}
replacement.SOURCE_FILES = tuple(
    dict.fromkeys(
        replacement.SOURCE_FILES
        + ("scripts/generate_nova_pro_game1_replacement.py",)
    )
)


if __name__ == "__main__":
    raise SystemExit(replacement.main())

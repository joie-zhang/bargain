#!/usr/bin/env python3
"""Strictly validate and aggregate nine-seed Gemini TTC labels."""

from __future__ import annotations

import validate_ttc_claude_codex_adjudication as validator


validator.ROOT = (
    validator.PROJECT_ROOT
    / "analysis/ttc_gemini_nine_seed_codex_adjudication_20260809"
)
validator.OUTPUTS = validator.ROOT / "subagent_outputs"
validator.REPORT_TITLE = "Nine-seed Gemini TTC Codex Adjudication Validation"


if __name__ == "__main__":
    validator.main()

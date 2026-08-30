"""Tests for the Reviewer Item 8 Elo-payoff slope helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_reviewer_item8_n_slope import (
    linear_slope_per_100,
    model_mean_slope,
    within_run_slope,
)


def test_linear_slope_per_100_uses_requested_units() -> None:
    assert linear_slope_per_100([1200, 1300, 1400], [10, 15, 20]) == pytest.approx(5.0)


def test_within_run_slope_removes_shared_run_difficulty() -> None:
    frame = pd.DataFrame(
        {
            "run_key": ["a", "a", "b", "b"],
            "model": ["low", "high", "low", "high"],
            "elo": [1200, 1400, 1200, 1400],
            "utility": [10, 20, 60, 70],
        }
    )
    assert within_run_slope(frame, "utility") == pytest.approx(5.0)


def test_model_mean_slope_averages_replicates_by_model() -> None:
    frame = pd.DataFrame(
        {
            "model": ["low", "low", "high", "high"],
            "elo": [1200, 1200, 1400, 1400],
            "utility": [0, 10, 20, 30],
        }
    )
    assert model_mean_slope(frame, "utility") == pytest.approx(10.0)

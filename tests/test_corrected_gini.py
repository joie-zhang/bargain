"""Regression tests for the paper's shifted finite-sample Gini."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_nash_lindahl_fairness import gini


@pytest.mark.parametrize(
    ("values", "expected"),
    (
        ([0.0, 0.0], 0.0),
        ([5.0, 5.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([-1.0, 1.0], 1.0),
        ([0.0, 0.0, 0.0, 1.0], 1.0),
        ([1.0], 0.0),
    ),
)
def test_gini_uses_shifted_finite_sample_definition(
    values: list[float],
    expected: float,
) -> None:
    assert gini(values) == pytest.approx(expected)


def test_gini_empty_input_is_undefined() -> None:
    assert math.isnan(gini([]))

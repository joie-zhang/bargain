"""Unit inputs exercise counting and relocation; they are not research evidence."""

from pathlib import Path

import pandas as pd
import pytest

from scripts.paper_figures.plot_figure9_absolute_payoffs import resolve_saved_path
from scripts.paper_figures.plot_figure9_actual_optimality import summarize_optimality


def count_inputs():
    return pd.DataFrame([
        {"condition": condition, "n_agents": n, "nano_exact_maximum": index < successes}
        for condition, successes in (("uncoordinated", 7), ("coordinated", 13))
        for n in (2, 4, 6, 8, 10)
        for index in range(20)
    ])


def test_optimality_counts_both_conditions_from_outcomes():
    summary = summarize_optimality(count_inputs())
    for condition, count, rate in (("uncoordinated", 7, 35), ("coordinated", 13, 65)):
        cells = summary[summary.condition.eq(condition)]
        assert cells.optimal_runs.tolist() == [count] * 5
        assert cells.optimal_percent.tolist() == [rate] * 5
        assert (cells.ci95_low < rate).all()
        assert (cells.ci95_high > rate).all()


def test_optimality_rejects_missing_runs_and_nonboolean_outcomes():
    with pytest.raises(ValueError, match="Expected 20 runs"):
        summarize_optimality(count_inputs().iloc[1:])
    frame = count_inputs()
    frame["nano_exact_maximum"] = frame.nano_exact_maximum.astype(str)
    with pytest.raises(ValueError, match="must be booleans"):
        summarize_optimality(frame)


def test_result_relocation_is_explicit_and_preserves_suffix(tmp_path):
    saved = "/original/checkout/experiments/results/batch/runs/config_1/experiment_results.json"
    assert resolve_saved_path(saved, None) == Path(saved)
    expected = tmp_path / "batch/runs/config_1/experiment_results.json"
    assert resolve_saved_path(saved, tmp_path) == expected
    # A missing relocated input must stay missing, without reading an old path.
    assert not expected.exists()
    with pytest.raises(FileNotFoundError):
        resolve_saved_path(saved, tmp_path).read_bytes()


@pytest.mark.parametrize("saved", [
    "/original/unknown/batch/result.json",
    "/original/experiments/results/../../outside.json",
    "/experiments/results/nested/experiments/results/result.json",
])
def test_result_relocation_rejects_ambiguous_or_escaping_paths(saved, tmp_path):
    with pytest.raises(ValueError):
        resolve_saved_path(saved, tmp_path)

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pytest

from scripts.ttc_analysis_common import (
    endpoint_across_seed_ci,
    seed_level_ci_table,
    write_report,
)
from scripts.generate_ttc_seed_replication_jobs import (
    ARCHIVED_CAP,
    clone_config,
    load_source_configs,
    validate_configs,
)
from scripts.run_ttc_native_config import resolve_max_tokens_per_phase


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = (
    PROJECT_ROOT / "experiments" / "results" / "ttc_native_scaling_20260502_212943"
)


def test_replication_preserves_archived_cap() -> None:
    assert resolve_max_tokens_per_phase(
        {
            "max_tokens_per_phase": ARCHIVED_CAP,
            "preserve_config_max_tokens_per_phase": True,
        }
    ) == ARCHIVED_CAP


def test_legacy_default_still_migrates() -> None:
    assert resolve_max_tokens_per_phase({"max_tokens_per_phase": ARCHIVED_CAP}) == 16_384


@pytest.mark.parametrize("seed", [42, 526])
def test_all_216_configs_change_only_seed_path_and_metadata(tmp_path: Path, seed: int) -> None:
    source_configs = load_source_configs(SOURCE_ROOT)
    clones = [clone_config(source, tmp_path, seed) for source in source_configs]
    validate_configs(source_configs, clones, seed)
    assert len(clones) == 216
    assert {clone["random_seed"] for clone in clones} == {seed}
    assert {clone["max_tokens_per_phase"] for clone in clones} == {ARCHIVED_CAP}
    assert len({clone["output_dir"] for clone in clones}) == 216


def test_validation_rejects_scientific_change(tmp_path: Path) -> None:
    source_configs = load_source_configs(SOURCE_ROOT)
    clones = [clone_config(source, tmp_path, 984) for source in source_configs]
    bad = copy.deepcopy(clones)
    bad[0]["gamma_discount"] = 0.8
    try:
        validate_configs(source_configs, bad, 984)
    except RuntimeError as exc:
        assert "unexpected fields" in str(exc)
    else:
        raise AssertionError("Validation accepted a changed scientific parameter")


@pytest.mark.parametrize("seed_count, expected_mean", [(3, 11.0), (5, 12.0), (10, 14.5)])
def test_seed_ci_uses_seed_level_estimates(seed_count: int, expected_mean: float) -> None:
    rows = []
    seeds = [42, 984, 526, 423, 1024, 128, 256, 612, 2048, 4096][:seed_count]
    families = ["gpt-5", "claude-sonnet-4-6", "gemini-3-flash"]
    providers = ["OpenAI", "Anthropic", "Google"]
    levels = {
        "gpt-5": ["minimal", "low", "medium", "high"],
        "claude-sonnet-4-6": ["low", "medium", "high", "max"],
        "gemini-3-flash": ["minimal", "low", "medium", "high"],
    }
    for family, provider in zip(families, providers):
        for level_index, level in enumerate(levels[family]):
            for seed_index, seed in enumerate(seeds):
                target = 10.0 + level_index + seed_index
                rows.append(
                    {
                        "seed": seed,
                        "family": family,
                        "provider": provider,
                        "level": level,
                        "level_index": level_index,
                        "target_utility_mean": target,
                        "baseline_utility_mean": 5.0,
                        "utility_gap_mean": target - 5.0,
                        "consensus_rate": 0.8,
                        "mean_round": 2.0,
                    }
                )
    result = seed_level_ci_table(pd.DataFrame(rows), seeds)
    assert len(result) == 12
    first = result[
        result["family"].eq("gpt-5") & result["level_index"].eq(0)
    ].iloc[0]
    assert first["seed_count"] == seed_count
    assert first["target_utility_mean"] == expected_mean
    assert first["target_utility_seed_ci95_low"] < expected_mean
    assert first["target_utility_seed_ci95_high"] > expected_mean


def test_endpoint_ci_counts_seed_directions() -> None:
    seeds = [42, 984, 526, 423, 1024]
    rows = []
    for family in ("gpt-5", "claude-sonnet-4-6", "gemini-3-flash"):
        for seed, delta in zip(seeds, (-1.0, 1.0, 2.0, 3.0, 4.0)):
            rows.append(
                {
                    "family": family,
                    "seed": str(seed),
                    "target_utility_endpoint_delta": delta,
                    "utility_gap_endpoint_delta": delta / 2.0,
                }
            )
    result = endpoint_across_seed_ci(pd.DataFrame(rows), seeds)
    assert len(result) == 3
    assert set(result["positive_target_endpoint_seeds"]) == {4}
    assert set(result["negative_target_endpoint_seeds"]) == {1}
    assert set(result["target_utility_endpoint_delta_mean_across_seeds"]) == {1.8}


def test_ten_seed_report_describes_current_sample(tmp_path: Path) -> None:
    seeds = [42, 984, 526, 423, 1024, 128, 256, 612, 2048, 4096]
    rows = pd.DataFrame(
        {"seed": [seed for seed in seeds for _ in range(216)], "consensus": True}
    )
    output = tmp_path / "report.md"
    write_report(
        output,
        rows,
        pd.DataFrame(columns=["seed", "family"]),
        pd.DataFrame(),
        {"pairwise_seed_agreement": {}},
        seeds,
    )
    report = output.read_text()
    assert "df=9" in report
    assert "all 2,160 runs" in report
    assert "df=4" not in report
    assert "1,080" not in report

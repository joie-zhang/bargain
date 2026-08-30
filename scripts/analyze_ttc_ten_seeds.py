#!/usr/bin/env python3
"""Analyze the complete TTC replication across ten random seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import analyze_ttc_five_seeds as shared
from scripts import analyze_ttc_seed_replication as base
from scripts import analyze_ttc_three_seeds as three


HISTORICAL_ROOTS = {
    42: (
        PROJECT_ROOT
        / "experiments"
        / "results"
        / "ttc_native_scaling_20260502_212943"
    ),
    984: (
        PROJECT_ROOT
        / "experiments"
        / "results"
        / "ttc_native_scaling_seed984_20260725_025700"
    ),
    526: (
        PROJECT_ROOT
        / "experiments"
        / "results"
        / "ttc_native_scaling_seed526_20260725_181400"
    ),
    423: (
        PROJECT_ROOT
        / "experiments"
        / "results"
        / "ttc_native_scaling_seed423_20260725_211500"
    ),
    1024: (
        PROJECT_ROOT
        / "experiments"
        / "results"
        / "ttc_native_scaling_seed1024_20260725_211500"
    ),
}
NEW_SEEDS = [128, 256, 612, 2048, 4096]
SEEDS = [42, 984, 526, 423, 1024, *NEW_SEEDS]
SEED_COLORS = {
    42: "#64748b",
    984: "#2563eb",
    526: "#dc2626",
    423: "#059669",
    1024: "#9333ea",
    128: "#ea580c",
    256: "#0891b2",
    612: "#92400e",
    2048: "#db2777",
    4096: "#65a30d",
}
SEED_MARKERS = {
    42: "s",
    984: "o",
    526: "^",
    423: "D",
    1024: "P",
    128: "v",
    256: "X",
    612: "*",
    2048: "h",
    4096: "<",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for seed in NEW_SEEDS:
        parser.add_argument(f"seed{seed}_root", type=Path)
    for seed, root in HISTORICAL_ROOTS.items():
        parser.add_argument(f"--seed{seed}-root", type=Path, default=root)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roots = {
        seed: getattr(args, f"seed{seed}_root").resolve() for seed in SEEDS
    }
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else roots[4096]
        / "analysis"
        / "seeds42_984_526_423_1024_128_256_612_2048_4096"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    shared.SEED_COLORS.clear()
    shared.SEED_COLORS.update(SEED_COLORS)
    shared.SEED_MARKERS.clear()
    shared.SEED_MARKERS.update(SEED_MARKERS)

    data = {
        seed: base.collect_run_rows(root, seed) for seed, root in roots.items()
    }
    rows = pd.concat([data[seed] for seed in SEEDS], ignore_index=True)
    base.validate_grid(rows, SEEDS, allow_incomplete=False)
    cell, by_seed, combined = base.summarize(rows)
    endpoints = three.endpoint_table(cell, SEEDS)
    agreement = three.agreement_summary(by_seed, endpoints, SEEDS)
    seed_ci = shared.seed_level_ci_table(by_seed, SEEDS)
    endpoint_seed_ci = shared.endpoint_across_seed_ci(endpoints, SEEDS)

    rows.to_csv(output_dir / "run_level_results_all_ten.csv", index=False)
    cell.to_csv(output_dir / "game_cell_seed_summary_all_ten.csv", index=False)
    by_seed.to_csv(output_dir / "family_effort_by_seed_all_ten.csv", index=False)
    combined.to_csv(output_dir / "family_effort_pooled_all_ten.csv", index=False)
    seed_ci.to_csv(
        output_dir / "family_effort_across_seed_ci95_all_ten.csv", index=False
    )
    endpoints.to_csv(
        output_dir / "endpoint_changes_by_seed_and_pooled_all_ten.csv", index=False
    )
    endpoint_seed_ci.to_csv(
        output_dir / "endpoint_changes_across_seed_ci95_all_ten.csv", index=False
    )
    (output_dir / "seed_agreement_all_ten.json").write_text(
        json.dumps(agreement, indent=2) + "\n", encoding="utf-8"
    )

    shared.plot_individual_seeds(
        by_seed,
        output_dir / "target_payoff_ten_seed_comparison.png",
        "target_utility",
        "Mean target payoff",
        SEEDS,
    )
    shared.plot_individual_seeds(
        by_seed,
        output_dir / "utility_gap_ten_seed_comparison.png",
        "utility_gap",
        "Mean target − baseline payoff",
        SEEDS,
    )
    shared.plot_across_seed_ci(
        by_seed,
        seed_ci,
        output_dir / "target_payoff_across_seed_mean_ci95_ten.png",
        "target_utility",
        "Mean target payoff",
        SEEDS,
    )
    shared.plot_across_seed_ci(
        by_seed,
        seed_ci,
        output_dir / "utility_gap_across_seed_mean_ci95_ten.png",
        "utility_gap",
        "Mean target − baseline payoff",
        SEEDS,
    )
    shared.plot_endpoint_ci(
        endpoints,
        endpoint_seed_ci,
        output_dir / "target_payoff_endpoint_delta_across_ten_seeds.png",
        SEEDS,
    )
    base.plot_combined(
        combined,
        output_dir / "target_payoff_all_ten_seeds_pooled.png",
        "10 seeds (2,160 runs)",
    )
    for seed in NEW_SEEDS:
        base.plot_target_and_baseline(
            by_seed[by_seed["seed"].eq(seed)],
            output_dir / f"seed{seed}_target_and_baseline_by_effort.png",
            f"Seed {seed} (216 runs)",
        )
    base.plot_target_and_baseline(
        combined,
        output_dir / "combined_target_and_baseline_all_ten.png",
        "10 seeds (2,160 runs)",
    )
    shared.plot_game_stratified(
        cell,
        output_dir / "target_payoff_by_game_ten_seeds.png",
        SEEDS,
    )
    shared.write_report(
        output_dir / "comparison_report_all_ten.md",
        rows,
        endpoints,
        endpoint_seed_ci,
        agreement,
        SEEDS,
    )

    audits: Dict[str, Any] = {}
    for seed in SEEDS:
        seed_df = data[seed]
        audits[str(seed)] = {
            "results_root": str(roots[seed]),
            "healthy_results": int(seed_df["config_id"].nunique()),
            "hard_failed_results": int(seed_df["hard_failed"].sum()),
            "no_consensus_results": int((~seed_df["consensus"]).sum()),
            "family_counts": {
                str(key): int(value)
                for key, value in seed_df.groupby("family").size().to_dict().items()
            },
            "game_counts": {
                str(key): int(value)
                for key, value in seed_df.groupby("game").size().to_dict().items()
            },
            "order_counts": {
                str(key): int(value)
                for key, value in seed_df.groupby("order").size().to_dict().items()
            },
            "saved_result_cap_counts": base.saved_result_caps(roots[seed]),
            "cap_recovery_config_ids": shared.recovery_ids(roots[seed]),
        }
    final_audit = {
        "expected_seeds": SEEDS,
        "expected_results_per_seed": 216,
        "total_healthy_results": int(rows["config_id"].count()),
        "missing_seed_config_pairs": [],
        "confidence_interval_unit": "seed-level estimate",
        "confidence_interval_seed_count": len(SEEDS),
        "confidence_interval_method": (
            f"two-sided Student-t 95% CI, df={len(SEEDS) - 1}"
        ),
        "by_seed": audits,
    }
    (output_dir / "final_audit_all_ten.json").write_text(
        json.dumps(final_audit, indent=2) + "\n", encoding="utf-8"
    )

    artifact_hashes = {
        path.name: sha256(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "analysis_provenance_all_ten.json"
    }
    (output_dir / "analysis_provenance_all_ten.json").write_text(
        json.dumps(
            {
                "seed_roots": {
                    str(seed): str(root) for seed, root in roots.items()
                },
                "analysis_script": str(Path(__file__).resolve()),
                "primary_confidence_interval": {
                    "unit": "seed-level estimate",
                    "seed_count": len(SEEDS),
                    "method": "two-sided Student-t 95% CI",
                    "degrees_of_freedom": len(SEEDS) - 1,
                },
                "artifacts_sha256": artifact_hashes,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "endpoint_across_seed_ci": endpoint_seed_ci.to_dict("records"),
                **agreement,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

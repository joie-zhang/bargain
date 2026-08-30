#!/usr/bin/env python3
"""Analyze all complete family-seed panels in the TTC cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import analyze_ttc_seed_replication as base  # noqa: E402


SEEDS = [42, 984, 526, 423, 1024, 128, 256, 612, 2048, 4096]
ROOTS = {
    42: PROJECT_ROOT / "experiments/results/ttc_native_scaling_20260502_212943",
    984: PROJECT_ROOT / "experiments/results/ttc_native_scaling_seed984_20260725_025700",
    526: PROJECT_ROOT / "experiments/results/ttc_native_scaling_seed526_20260725_181400",
    423: PROJECT_ROOT / "experiments/results/ttc_native_scaling_seed423_20260725_211500",
    1024: PROJECT_ROOT / "experiments/results/ttc_native_scaling_seed1024_20260725_211500",
    128: PROJECT_ROOT / "experiments/results/ttc_native_scaling_seed128_20260727_043613",
    256: PROJECT_ROOT / "experiments/results/ttc_native_scaling_seed256_20260727_043613",
    612: PROJECT_ROOT / "experiments/results/ttc_native_scaling_seed612_20260727_043613",
    2048: PROJECT_ROOT / "experiments/results/ttc_native_scaling_seed2048_20260727_043613",
    4096: PROJECT_ROOT / "experiments/results/ttc_native_scaling_seed4096_20260727_043613",
}
DEFAULT_OUTPUT = PROJECT_ROOT / "analysis/ttc_complete_family_seed_panels_20260810"
GROUP_KEYS = ["seed", "family", "provider", "level", "level_index"]
OFF_PROTOCOL_SEED = 612
OFF_PROTOCOL_FAMILY = "claude-sonnet-4-6"
OFF_PROTOCOL_CONFIG = 142


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for seed, root in ROOTS.items():
        parser.add_argument(f"--seed{seed}-root", type=Path, default=root)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def resolve_output_dir(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def holm_adjust(values: list[float]) -> list[float]:
    order = np.argsort(np.asarray(values, dtype=float))
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(values) - rank) * values[int(index)])
        running = max(running, candidate)
        adjusted[int(index)] = running
    return adjusted.tolist()


def collect_root(run_root: Path, seed: int) -> pd.DataFrame:
    rows = base.collect_run_rows(run_root, seed)
    lineage: list[dict[str, Any]] = []
    for config_path in base.config_paths(run_root):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        output_dir = resolve_output_dir(config["output_dir"])
        result_path = output_dir / "run_1_experiment_results.json"
        rollout_path = output_dir / "run_1_all_interactions.json"
        if not result_path.exists() or not rollout_path.exists():
            raise FileNotFoundError(
                f"Missing terminal TTC artifacts for seed={seed}, "
                f"config={config['config_id']}"
            )
        lineage.append(
            {
                "seed": seed,
                "config_id": int(config["config_id"]),
                "config_path": repo_relative(config_path),
                "result_path": repo_relative(result_path),
                "rollout_path": repo_relative(rollout_path),
                "config_sha256": sha256(config_path),
                "result_sha256": sha256(result_path),
                "rollout_sha256": sha256(rollout_path),
                "max_tokens_per_phase": int(config["max_tokens_per_phase"]),
                "allow_extended_max_tokens_per_phase": bool(
                    config.get("allow_extended_max_tokens_per_phase", False)
                ),
            }
        )
    lineage_frame = pd.DataFrame(lineage)
    return rows.drop(columns=["result_path"]).merge(
        lineage_frame, on=["seed", "config_id"], validate="one_to_one"
    )


def retain_complete_panels(
    rows: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list[int]], pd.DataFrame]:
    coverage = (
        rows.groupby(["seed", "family", "level_index"])
        .size()
        .unstack("level_index", fill_value=0)
    )
    if not (coverage.astype(int) == 18).all().all():
        raise RuntimeError(f"Unexpected TTC family-effort coverage: {coverage}")

    off_protocol = rows[
        rows["seed"].eq(OFF_PROTOCOL_SEED)
        & rows["config_id"].eq(OFF_PROTOCOL_CONFIG)
    ]
    if len(off_protocol) != 1:
        raise RuntimeError(f"Expected one off-protocol recovery row, found {len(off_protocol)}")
    recovered = off_protocol.iloc[0]
    if (
        recovered["family"] != OFF_PROTOCOL_FAMILY
        or int(recovered["max_tokens_per_phase"]) != 65536
        or not bool(recovered["allow_extended_max_tokens_per_phase"])
    ):
        raise RuntimeError("The declared TTC protocol exception no longer matches raw config 142")
    standard_cap_rows = rows.drop(index=off_protocol.index)
    if set(standard_cap_rows["max_tokens_per_phase"].astype(int)) != {10500}:
        raise RuntimeError("Unexpected token cap outside the declared recovery exception")

    panel_audit_rows: list[dict[str, Any]] = []
    for (seed, family), group in rows.groupby(["seed", "family"]):
        includes_recovery_exception = bool(
            int(seed) == OFF_PROTOCOL_SEED and family == OFF_PROTOCOL_FAMILY
        )
        panel_audit_rows.append(
            {
                "seed": int(seed),
                "family": str(family),
                "terminal_games": int(len(group)),
                "standard_cap_games": int(group["max_tokens_per_phase"].eq(10500).sum()),
                "comparable_complete_panel": True,
                "includes_recovery_exception": includes_recovery_exception,
                "exclusion_reason": "",
            }
        )

    retained = rows.copy()
    seeds_by_family = {
        family: sorted(
            int(seed)
            for seed in retained.loc[
                retained["family"].eq(family), "seed"
            ].unique()
        )
        for family in base.FAMILY_ORDER
    }
    expected = {family: sorted(SEEDS) for family in base.FAMILY_ORDER}
    if seeds_by_family != expected or len(retained) != 2160:
        raise RuntimeError(
            f"Unexpected complete panels: seeds={seeds_by_family}, rows={len(retained)}"
        )
    return retained, seeds_by_family, pd.DataFrame(panel_audit_rows)


def summarize_by_seed(rows: pd.DataFrame) -> pd.DataFrame:
    cells = (
        rows.groupby(GROUP_KEYS + ["game", "game_cell"], as_index=False)
        .agg(
            order_count=("order", "nunique"),
            run_count=("config_id", "size"),
            target_utility=("target_utility", "mean"),
            baseline_utility=("baseline_utility", "mean"),
            utility_gap=("utility_gap", "mean"),
            consensus_rate=("consensus", "mean"),
        )
    )
    if not cells["order_count"].eq(2).all() or not cells["run_count"].eq(2).all():
        raise RuntimeError("Each retained seed/game cell must contain both model orders")

    by_seed = (
        cells.groupby(GROUP_KEYS, as_index=False)
        .agg(
            game_cell_count=("game_cell", "size"),
            target_utility_mean=("target_utility", "mean"),
            baseline_utility_mean=("baseline_utility", "mean"),
            utility_gap_mean=("utility_gap", "mean"),
            consensus_rate=("consensus_rate", "mean"),
        )
    )
    if not by_seed["game_cell_count"].eq(9).all():
        raise RuntimeError("Every retained seed estimate must contain nine game cells")

    runs = rows.copy()
    runs["target_deal_utility"] = runs["target_utility"].where(runs["consensus"])
    run_estimands = (
        runs.groupby(GROUP_KEYS, as_index=False)
        .agg(
            terminal_run_count=("config_id", "size"),
            consensus_count=("consensus", "sum"),
            target_deal_utility_mean=("target_deal_utility", "mean"),
        )
    )
    by_seed = by_seed.merge(run_estimands, on=GROUP_KEYS, validate="one_to_one")
    if not by_seed["terminal_run_count"].eq(18).all():
        raise RuntimeError("Every retained family-effort-seed cell must contain 18 runs")
    if by_seed["target_deal_utility_mean"].isna().any():
        raise RuntimeError("A retained family-effort-seed cell has no consensus outcomes")
    return by_seed.sort_values(["family", "level_index", "seed"])


def add_interval(row: dict[str, Any], metric: str, values: pd.Series) -> None:
    values = values.astype(float)
    mean = float(values.mean())
    standard_error = base.sem(values)
    low, high = base.ci95(mean, standard_error, len(values))
    row[f"{metric}_mean"] = mean
    row[f"{metric}_seed_sem"] = standard_error
    row[f"{metric}_seed_ci95_low"] = low
    row[f"{metric}_seed_ci95_high"] = high


def effort_summary(by_seed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sources = {
        "target_utility": "target_utility_mean",
        "target_deal_utility": "target_deal_utility_mean",
        "baseline_utility": "baseline_utility_mean",
        "utility_gap": "utility_gap_mean",
        "consensus_rate": "consensus_rate",
    }
    for (family, provider, level, level_index), group in by_seed.groupby(
        ["family", "provider", "level", "level_index"], sort=False
    ):
        row: dict[str, Any] = {
            "family": family,
            "provider": provider,
            "level": level,
            "level_index": int(level_index),
            "seed_count": int(group["seed"].nunique()),
            "seeds": ",".join(str(seed) for seed in sorted(group["seed"].astype(int))),
            "game_cell_count_min": int(group["game_cell_count"].min()),
            "game_cell_count_max": int(group["game_cell_count"].max()),
        }
        for metric, source in sources.items():
            add_interval(row, metric, group[source])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["family", "level_index"])


def endpoint_summary(by_seed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = {
        "target_utility": "target_utility_mean",
        "target_deal_utility": "target_deal_utility_mean",
        "utility_gap": "utility_gap_mean",
        "consensus_rate": "consensus_rate",
    }
    per_seed_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for family in base.FAMILY_ORDER:
        group = by_seed[by_seed["family"].eq(family)]
        low = int(group["level_index"].min())
        high = int(group["level_index"].max())
        seed_rows: list[dict[str, Any]] = []
        for seed, seed_group in group.groupby("seed"):
            indexed = seed_group.set_index("level_index")
            row: dict[str, Any] = {"family": family, "seed": int(seed)}
            for metric, source in metrics.items():
                row[f"{metric}_endpoint_delta"] = float(
                    indexed.loc[high, source] - indexed.loc[low, source]
                )
            seed_rows.append(row)
            per_seed_rows.append(row)

        seed_frame = pd.DataFrame(seed_rows)
        result: dict[str, Any] = {
            "family": family,
            "seed_count": len(seed_frame),
            "seeds": ",".join(str(seed) for seed in sorted(seed_frame["seed"])),
            "positive_target_endpoint_seeds": int(
                (seed_frame["target_utility_endpoint_delta"] > 0).sum()
            ),
        }
        for metric in metrics:
            values = seed_frame[f"{metric}_endpoint_delta"].astype(float)
            mean = float(values.mean())
            standard_error = base.sem(values)
            low_ci, high_ci = base.ci95(mean, standard_error, len(values))
            result[f"{metric}_endpoint_delta_mean"] = mean
            result[f"{metric}_endpoint_delta_seed_sem"] = standard_error
            result[f"{metric}_endpoint_delta_seed_ci95_low"] = low_ci
            result[f"{metric}_endpoint_delta_seed_ci95_high"] = high_ci
            result[f"{metric}_endpoint_delta_raw_p"] = float(
                ttest_1samp(values, 0.0, nan_policy="raise").pvalue
            )
        summary_rows.append(result)

    summary = pd.DataFrame(summary_rows)
    for metric in metrics:
        column = f"{metric}_endpoint_delta_raw_p"
        summary[f"{metric}_endpoint_delta_holm_p"] = holm_adjust(
            summary[column].astype(float).tolist()
        )
    return pd.DataFrame(per_seed_rows), summary


def main() -> int:
    args = parse_args()
    roots = {seed: getattr(args, f"seed{seed}_root").resolve() for seed in SEEDS}
    data = {seed: collect_root(roots[seed], seed) for seed in SEEDS}
    all_rows = pd.concat([data[seed] for seed in SEEDS], ignore_index=True)
    base.validate_grid(all_rows, SEEDS, allow_incomplete=False)
    if len(all_rows) != 2160:
        raise RuntimeError(f"Expected 2,160 terminal TTC results, found {len(all_rows)}")

    retained, seeds_by_family, panel_audit = retain_complete_panels(all_rows)
    retained_pairs = set(zip(retained["seed"], retained["family"], strict=True))
    all_rows["retained_comparable_payoff_panel"] = [
        (seed, family) in retained_pairs
        for seed, family in zip(all_rows["seed"], all_rows["family"], strict=True)
    ]
    all_rows["protocol_exception"] = all_rows["seed"].eq(OFF_PROTOCOL_SEED) & all_rows[
        "config_id"
    ].eq(OFF_PROTOCOL_CONFIG)

    by_seed = summarize_by_seed(retained)
    effort = effort_summary(by_seed)
    endpoints_by_seed, endpoints = endpoint_summary(by_seed)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_rows.to_csv(output / "terminal_run_inventory.csv", index=False)
    retained.to_csv(output / "run_level_complete_panels.csv", index=False)
    panel_audit.to_csv(output / "family_seed_panel_audit.csv", index=False)
    by_seed.to_csv(output / "family_effort_by_complete_seed.csv", index=False)
    effort.to_csv(output / "family_effort_complete_seed_ci95.csv", index=False)
    endpoints_by_seed.to_csv(output / "endpoint_changes_by_complete_seed.csv", index=False)
    endpoints.to_csv(output / "endpoint_changes_complete_seed_ci95.csv", index=False)

    audit = {
        "planned_ttc_games": 2160,
        "terminal_result_artifacts": int(len(all_rows)),
        "standard_cap_terminal_results": 2159,
        "protocol_exception": {
            "seed": OFF_PROTOCOL_SEED,
            "config_id": OFF_PROTOCOL_CONFIG,
            "family": OFF_PROTOCOL_FAMILY,
            "standard_max_tokens_per_phase": 10500,
            "recovery_max_tokens_per_phase": 65536,
            "analysis_treatment": "include the complete seed-family panel",
        },
        "retained_comparable_payoff_games": int(len(retained)),
        "included_recovery_exception_panel": {
            "seed": OFF_PROTOCOL_SEED,
            "family": OFF_PROTOCOL_FAMILY,
            "terminal_games": 72,
        },
        "excluded_family_panels": [],
        "complete_seeds_by_family": seeds_by_family,
        "retained_non_ttc_games": 5055,
        "paper_retained_corpus_total": 7215,
        "paper_retained_corpus_arithmetic": "1500 + 500 + 2730 + 325 + 2160",
        "paper_terminal_artifact_total": 7215,
        "paper_terminal_artifact_arithmetic": "1500 + 500 + 2730 + 325 + 2160",
        "uncertainty_unit": "complete family-seed estimate",
        "uncertainty_method": "two-sided Student-t 95% CI",
        "multiplicity": "Holm adjustment across three families per endpoint",
        "payoff_estimand": "unconditional target payoff; no consensus = 0",
        "roots": {str(seed): str(root) for seed, root in roots.items()},
    }
    (output / "complete_seed_panel_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({**audit, "endpoints": endpoints.to_dict("records")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

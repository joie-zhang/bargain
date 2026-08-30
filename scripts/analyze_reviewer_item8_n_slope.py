#!/usr/bin/env python3
"""
=============================================================================
Reviewer Item 8: Does the Elo-payoff slope decrease with group size?
=============================================================================

Description
-----------
Reproduces the paper's multi-agent Elo-payoff slopes and tests whether they
decrease as N grows. The analysis distinguishes:

1. The controlled homogeneous-adversary ecology (one focal model among
   GPT-5-nano agents).
2. Random heterogeneous rosters.
3. Raw payoff versus payoff normalized by each instance's equal share of
   optimal welfare.
4. Marginal model-mean slopes versus within-run relative-Elo slopes.

It also audits possible explanations: competition mix, payoff-scale
compression, roster composition, position, agreement/discount effects,
signal-to-noise, group-level capability spillovers, and the Game 3
benefit-minus-contribution decomposition.

Usage
-----
    python scripts/analyze_reviewer_item8_n_slope.py
    python scripts/analyze_reviewer_item8_n_slope.py --bootstrap-reps 5000

What it creates
---------------
    analysis/reviewer_item8_n_slope_20260725/
    ├── normalized_cache/
    ├── slope_estimates.csv
    ├── endpoint_changes.csv
    ├── within_run_interactions.csv
    ├── competition_cell_slopes.csv
    ├── mechanism_diagnostics.csv
    ├── group_capability_coefficients.csv
    ├── game3_benefit_cost_decomposition.csv
    ├── slope_by_design.png
    ├── heterogeneous_raw_vs_normalized.png
    ├── mechanism_diagnostics.png
    └── competition_cell_slopes.png

Configuration
-------------
Command-line arguments control the canonical table directory, output
directory, number of bootstrap replicates, and random seed.

Dependencies
------------
Python packages: numpy, pandas, scipy, statsmodels, matplotlib, seaborn.
Repository module: scripts.analyze_neurips_revision_stats.
The canonical 2,730-run multi-agent tables and Nash/Lindahl fairness exports
must already exist.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_neurips_revision_stats import (  # noqa: E402
    DEFAULT_MULTIAGENT_TABLE_DIR,
    load_multiagent_metrics,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis/reviewer_item8_n_slope_20260725"
N2_FAIRNESS_PATH = (
    PROJECT_ROOT
    / "experiments/results/n2_plus_multiagent_comparison_analysis_20260505"
    / "tables_multiagent/multiagent_n2_fairness_agent_metrics.csv"
)
NGT2_FAIRNESS_PATH = PROJECT_ROOT / "analysis/nash_lindahl_fairness_20260505/agent_metrics.csv"
N_ORDER = [2, 4, 6, 8, 10]
GAME_ORDER = ["game1", "game2", "game3"]
GAME_TITLES = {
    "game1": "Item allocation",
    "game2": "Treaty negotiation",
    "game3": "Participatory budgeting",
}
DESIGN_TITLES = {
    "homogeneous_focal": "Controlled focal adversary",
    "heterogeneous": "Heterogeneous rosters",
}
COLORS = {
    "homogeneous_focal": "#7c3aed",
    "heterogeneous": "#0f766e",
    "raw": "#b91c1c",
    "normalized": "#1d4ed8",
}


@dataclass(frozen=True)
class BootstrapResult:
    point: float
    samples: np.ndarray

    @property
    def low(self) -> float:
        return float(np.quantile(self.samples, 0.025))

    @property
    def high(self) -> float:
        return float(np.quantile(self.samples, 0.975))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=DEFAULT_MULTIAGENT_TABLE_DIR,
        help="Canonical multi-agent table directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for tables and plots.",
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=3000,
        help="Stratified run-cluster bootstrap replicates.",
    )
    parser.add_argument("--seed", type=int, default=260725)
    return parser.parse_args()


def linear_slope_per_100(x: Iterable[float], y: Iterable[float]) -> float:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    keep = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[keep]
    y_arr = y_arr[keep]
    if len(x_arr) < 2 or np.allclose(x_arr, x_arr[0]):
        return math.nan
    x_centered = x_arr - x_arr.mean()
    return float((x_centered @ (y_arr - y_arr.mean())) / (x_centered @ x_centered) * 100.0)


def model_mean_slope(frame: pd.DataFrame, metric: str) -> float:
    model_means = (
        frame.groupby(["model", "elo"], as_index=False)
        .agg(metric_mean=(metric, "mean"))
        .dropna()
    )
    return linear_slope_per_100(model_means["elo"], model_means["metric_mean"])


def within_run_slope(frame: pd.DataFrame, metric: str) -> float:
    x = frame["elo"] - frame.groupby("run_key")["elo"].transform("mean")
    y = frame[metric] - frame.groupby("run_key")[metric].transform("mean")
    denominator = float(x @ x)
    return float(x @ y / denominator * 100.0) if denominator > 0 else math.nan


def bootstrap_slopes(
    frame: pd.DataFrame,
    metric: str,
    reps: int,
    rng: np.random.Generator,
    strata_columns: list[str],
    include_within_run: bool,
) -> tuple[BootstrapResult, BootstrapResult | None]:
    """Run-stratified bootstrap for model-mean and optional within-run slopes."""
    run_info = frame[["run_key", *strata_columns]].drop_duplicates("run_key").reset_index(drop=True)
    models = frame[["model", "elo"]].drop_duplicates().sort_values("elo").reset_index(drop=True)
    model_names = models["model"].tolist()
    elos = models["elo"].to_numpy(dtype=float)
    model_index = {model: index for index, model in enumerate(model_names)}
    run_index = {run: index for index, run in enumerate(run_info["run_key"])}

    run_count = len(run_info)
    model_count = len(models)
    values = np.zeros((run_count, model_count), dtype=float)
    incidence = np.zeros((run_count, model_count), dtype=float)
    sxx = np.zeros(run_count, dtype=float)
    sxy = np.zeros(run_count, dtype=float)

    for run_key, group in frame.groupby("run_key"):
        run_idx = run_index[run_key]
        model_indices = np.asarray([model_index[model] for model in group["model"]], dtype=int)
        metric_values = group[metric].to_numpy(dtype=float)
        values[run_idx, model_indices] = metric_values
        incidence[run_idx, model_indices] = 1.0
        if include_within_run:
            x = group["elo"].to_numpy(dtype=float)
            x -= x.mean()
            y = metric_values - metric_values.mean()
            sxx[run_idx] = float(x @ x)
            sxy[run_idx] = float(x @ y)

    weights = np.zeros((reps, run_count), dtype=np.int16)
    grouped = run_info.groupby(strata_columns, dropna=False).groups
    for indices_raw in grouped.values():
        indices = np.asarray(list(indices_raw), dtype=int)
        draws = rng.integers(0, len(indices), size=(reps, len(indices)))
        for rep in range(reps):
            weights[rep, indices] += np.bincount(draws[rep], minlength=len(indices))

    weighted_sums = weights @ values
    weighted_counts = weights @ incidence
    model_samples = np.empty(reps, dtype=float)
    for rep in range(reps):
        observed = weighted_counts[rep] > 0
        means = weighted_sums[rep, observed] / weighted_counts[rep, observed]
        model_samples[rep] = linear_slope_per_100(elos[observed], means)

    model_result = BootstrapResult(model_mean_slope(frame, metric), model_samples)
    if not include_within_run:
        return model_result, None

    within_samples = (weights @ sxy) / (weights @ sxx) * 100.0
    within_result = BootstrapResult(within_run_slope(frame, metric), within_samples)
    return model_result, within_result


def load_analysis_data(
    table_dir: Path, output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_dir = output_dir / "normalized_cache"
    run_path = cache_dir / "multiagent_runs_with_sw_star.csv"
    agent_path = cache_dir / "multiagent_agents_normalized.csv"
    if run_path.exists() and agent_path.exists():
        runs = pd.read_csv(run_path)
        agents = pd.read_csv(agent_path)
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        runs, agents = load_multiagent_metrics(table_dir, cache_dir, bootstrap_reps=500)

    if len(runs) != 2730 or len(agents) != 16380:
        raise ValueError(
            f"Expected 2,730 runs and 16,380 agent rows; found {len(runs)} and {len(agents)}"
        )
    for frame in (runs, agents):
        frame["run_key"] = frame["source"].astype(str) + ":" + frame["config_id"].astype(str)
    return runs, agents


def build_slope_tables(
    agents: pd.DataFrame,
    reps: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str, str, int, str], np.ndarray]]:
    records: list[dict[str, object]] = []
    sample_cache: dict[tuple[str, str, str, int, str], np.ndarray] = {}

    hetero = agents[agents["experiment_family"].eq("heterogeneous_random")].copy()
    focal = agents[
        agents["experiment_family"].eq("homogeneous_adversary")
        & agents["role"].eq("adversary")
    ].copy()

    for design, data, strata, include_within in [
        ("heterogeneous", hetero, ["competition_label"], True),
        ("homogeneous_focal", focal, ["competition_label", "model"], False),
    ]:
        for metric_name, metric in [
            ("raw", "final_utility"),
            ("normalized", "normalized_utility"),
        ]:
            for (game, n_agents), group in data.groupby(["game_label", "n_agents"]):
                model_result, within_result = bootstrap_slopes(
                    group,
                    metric,
                    reps,
                    rng,
                    strata_columns=strata,
                    include_within_run=include_within,
                )
                records.append(
                    {
                        "design": design,
                        "game_label": game,
                        "n_agents": int(n_agents),
                        "metric": metric_name,
                        "estimand": "model_mean",
                        "slope_per_100_elo": model_result.point,
                        "ci_low": model_result.low,
                        "ci_high": model_result.high,
                        "run_count": group["run_key"].nunique(),
                        "model_count": group["model"].nunique(),
                    }
                )
                sample_cache[(design, game, metric_name, int(n_agents), "model_mean")] = (
                    model_result.samples
                )
                if within_result is not None:
                    records.append(
                        {
                            "design": design,
                            "game_label": game,
                            "n_agents": int(n_agents),
                            "metric": metric_name,
                            "estimand": "within_run",
                            "slope_per_100_elo": within_result.point,
                            "ci_low": within_result.low,
                            "ci_high": within_result.high,
                            "run_count": group["run_key"].nunique(),
                            "model_count": group["model"].nunique(),
                        }
                    )
                    sample_cache[(design, game, metric_name, int(n_agents), "within_run")] = (
                        within_result.samples
                    )

    slopes = pd.DataFrame(records).sort_values(
        ["design", "metric", "estimand", "game_label", "n_agents"]
    )

    endpoint_records: list[dict[str, object]] = []
    for key_prefix in sorted(
        {
            (design, game, metric, estimand)
            for design, game, metric, _, estimand in sample_cache
        }
    ):
        design, game, metric, estimand = key_prefix
        low_n = sample_cache.get((design, game, metric, 2, estimand))
        high_n = sample_cache.get((design, game, metric, 10, estimand))
        if low_n is None or high_n is None:
            continue
        difference = high_n - low_n
        point_rows = slopes[
            slopes["design"].eq(design)
            & slopes["game_label"].eq(game)
            & slopes["metric"].eq(metric)
            & slopes["estimand"].eq(estimand)
        ].set_index("n_agents")
        endpoint_records.append(
            {
                "design": design,
                "game_label": game,
                "metric": metric,
                "estimand": estimand,
                "n2_slope": point_rows.loc[2, "slope_per_100_elo"],
                "n10_slope": point_rows.loc[10, "slope_per_100_elo"],
                "n10_minus_n2": (
                    point_rows.loc[10, "slope_per_100_elo"]
                    - point_rows.loc[2, "slope_per_100_elo"]
                ),
                "delta_ci_low": float(np.quantile(difference, 0.025)),
                "delta_ci_high": float(np.quantile(difference, 0.975)),
                "bootstrap_probability_decrease": float(np.mean(difference < 0)),
            }
        )
    endpoints = pd.DataFrame(endpoint_records).sort_values(
        ["design", "metric", "estimand", "game_label"]
    )
    return slopes, endpoints, sample_cache


def build_competition_slopes(agents: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for design, data in [
        (
            "heterogeneous",
            agents[agents["experiment_family"].eq("heterogeneous_random")],
        ),
        (
            "homogeneous_focal",
            agents[
                agents["experiment_family"].eq("homogeneous_adversary")
                & agents["role"].eq("adversary")
            ],
        ),
    ]:
        for (game, n_agents, competition), group in data.groupby(
            ["game_label", "n_agents", "competition_label"]
        ):
            records.append(
                {
                    "design": design,
                    "game_label": game,
                    "n_agents": int(n_agents),
                    "competition_label": competition,
                    "raw_slope_per_100_elo": model_mean_slope(group, "final_utility"),
                    "normalized_slope_per_100_elo": model_mean_slope(
                        group, "normalized_utility"
                    ),
                    "run_count": group["run_key"].nunique(),
                    "model_count": group["model"].nunique(),
                }
            )
    return pd.DataFrame(records).sort_values(
        ["design", "game_label", "competition_label", "n_agents"]
    )


def build_within_run_interactions(agents: pd.DataFrame) -> pd.DataFrame:
    data = agents[agents["experiment_family"].eq("heterogeneous_random")].copy()
    data["elo_centered_100"] = (
        data["elo"] - data.groupby("run_key")["elo"].transform("mean")
    ) / 100.0
    data["n_step"] = (data["n_agents"] - 2.0) / 2.0
    data["elo_x_n"] = data["elo_centered_100"] * data["n_step"]
    data["position_01"] = (data["agent_index"] - 1.0) / (data["n_agents"] - 1.0)
    data["position_centered"] = data["position_01"] - data.groupby("run_key")[
        "position_01"
    ].transform("mean")

    records: list[dict[str, object]] = []
    for metric_name, metric in [
        ("raw", "final_utility"),
        ("normalized", "normalized_utility"),
    ]:
        for game, group in data.groupby("game_label"):
            group = group.copy()
            group["outcome_centered"] = group[metric] - group.groupby("run_key")[
                metric
            ].transform("mean")
            x = group[["elo_centered_100", "elo_x_n", "position_centered"]]
            cluster_groups = np.column_stack(
                (
                    pd.factorize(group["run_key"])[0],
                    pd.factorize(group["model"])[0],
                )
            )
            fit = sm.OLS(group["outcome_centered"], x).fit(
                cov_type="cluster", cov_kwds={"groups": cluster_groups}
            )
            for term in ["elo_centered_100", "elo_x_n", "position_centered"]:
                records.append(
                    {
                        "game_label": game,
                        "metric": metric_name,
                        "term": term,
                        "estimate": fit.params[term],
                        "std_error": fit.bse[term],
                        "ci_low": fit.conf_int().loc[term, 0],
                        "ci_high": fit.conf_int().loc[term, 1],
                        "p_value": fit.pvalues[term],
                        "agent_count": len(group),
                        "run_count": group["run_key"].nunique(),
                    }
                )
    return pd.DataFrame(records).sort_values(["metric", "game_label", "term"])


def build_mechanism_diagnostics(
    runs: pd.DataFrame, agents: pd.DataFrame
) -> pd.DataFrame:
    hetero_agents = agents[agents["experiment_family"].eq("heterogeneous_random")].copy()
    hetero_runs = runs[runs["experiment_family"].eq("heterogeneous_random")].copy()
    hetero_runs["equal_share_optimum"] = hetero_runs["sw_star"] / hetero_runs["n_agents"]
    hetero_agents["other_mean_elo"] = (
        hetero_agents.groupby("run_key")["elo"].transform("sum") - hetero_agents["elo"]
    ) / (hetero_agents["n_agents"] - 1)

    records: list[dict[str, object]] = []
    for (game, n_agents), group in hetero_agents.groupby(["game_label", "n_agents"]):
        run_group = hetero_runs[
            hetero_runs["game_label"].eq(game)
            & hetero_runs["n_agents"].eq(n_agents)
        ]
        model_means = (
            group.groupby(["model", "elo"], as_index=False)
            .agg(
                raw_mean=("final_utility", "mean"),
                normalized_mean=("normalized_utility", "mean"),
                observations=("final_utility", "size"),
            )
        )
        x_centered = group["elo"] - group.groupby("run_key")["elo"].transform("mean")
        y_centered = group["final_utility"] - group.groupby("run_key")[
            "final_utility"
        ].transform("mean")
        records.append(
            {
                "game_label": game,
                "n_agents": int(n_agents),
                "paper_raw_slope": linear_slope_per_100(
                    model_means["elo"], model_means["raw_mean"]
                ),
                "paper_normalized_slope": linear_slope_per_100(
                    model_means["elo"], model_means["normalized_mean"]
                ),
                "paper_raw_r_squared": float(
                    model_means["elo"].corr(model_means["raw_mean"]) ** 2
                ),
                "model_mean_raw_sd": model_means["raw_mean"].std(),
                "model_mean_normalized_sd": model_means["normalized_mean"].std(),
                "within_run_raw_slope": within_run_slope(group, "final_utility"),
                "within_run_normalized_slope": within_run_slope(
                    group, "normalized_utility"
                ),
                "within_run_elo_payoff_correlation": x_centered.corr(y_centered),
                "mean_within_run_payoff_sd": group.groupby("run_key")[
                    "final_utility"
                ].std().mean(),
                "mean_within_run_elo_sd": group.groupby("run_key")["elo"].std().mean(),
                "own_other_elo_correlation": group["elo"].corr(group["other_mean_elo"]),
                "min_model_observations": model_means["observations"].min(),
                "max_model_observations": model_means["observations"].max(),
                "mean_raw_utility": group["final_utility"].mean(),
                "mean_normalized_utility": group["normalized_utility"].mean(),
                "equal_share_optimum": run_group["equal_share_optimum"].mean(),
                "group_efficiency": run_group["group_efficiency_discounted"].mean(),
                "consensus_rate": run_group["consensus_reached"].astype(float).mean(),
                "mean_final_round": run_group["final_round"].mean(),
                "strict_voting_clean_rate": run_group["strict_voting_clean"]
                .astype(float)
                .mean(),
                "synthetic_proposal_run_rate": (
                    run_group["synthetic_proposal_marker_count"].fillna(0) > 0
                ).mean(),
            }
        )
    return pd.DataFrame(records).sort_values(["game_label", "n_agents"])


def build_group_capability_coefficients(agents: pd.DataFrame) -> pd.DataFrame:
    data = agents[agents["experiment_family"].eq("heterogeneous_random")].copy()
    data["group_mean_elo"] = data.groupby("run_key")["elo"].transform("mean")
    data["relative_elo_100"] = (data["elo"] - data["group_mean_elo"]) / 100.0
    data["group_mean_elo_100"] = (data["group_mean_elo"] - 1350.0) / 100.0
    data["position_01"] = (data["agent_index"] - 1.0) / (data["n_agents"] - 1.0)

    records: list[dict[str, object]] = []
    for metric_name, metric in [
        ("raw", "final_utility"),
        ("normalized", "normalized_utility"),
    ]:
        for (game, n_agents), group in data.groupby(["game_label", "n_agents"]):
            cluster_groups = np.column_stack(
                (
                    pd.factorize(group["run_key"])[0],
                    pd.factorize(group["model"])[0],
                )
            )
            fit = smf.ols(
                f"{metric} ~ relative_elo_100 + group_mean_elo_100 "
                "+ C(competition_label) + position_01",
                data=group,
            ).fit(cov_type="cluster", cov_kwds={"groups": cluster_groups})
            for term in ["relative_elo_100", "group_mean_elo_100", "position_01"]:
                records.append(
                    {
                        "game_label": game,
                        "n_agents": int(n_agents),
                        "metric": metric_name,
                        "term": term,
                        "estimate": fit.params[term],
                        "std_error": fit.bse[term],
                        "ci_low": fit.conf_int().loc[term, 0],
                        "ci_high": fit.conf_int().loc[term, 1],
                        "p_value": fit.pvalues[term],
                        "r_squared": fit.rsquared,
                    }
                )
    return pd.DataFrame(records).sort_values(
        ["metric", "game_label", "n_agents", "term"]
    )


def build_game3_decomposition() -> pd.DataFrame:
    n2 = pd.read_csv(N2_FAIRNESS_PATH)
    ngt2 = pd.read_csv(NGT2_FAIRNESS_PATH)
    data = pd.concat(
        [
            n2[
                n2["source_group"].eq("n_eq_2_heterogeneous")
                & n2["game_id"].eq("game3")
            ],
            ngt2[
                ngt2["source_group"].eq("n_gt_2_heterogeneous")
                & ngt2["game_id"].eq("game3")
            ],
        ],
        ignore_index=True,
    )
    data["run_key"] = data["result_path"].astype(str)
    data["benefit_from_funded_projects"] = (
        data["actual_raw_utility"] + data["actual_paid_funded"]
    )

    records: list[dict[str, object]] = []
    metrics = [
        "actual_raw_utility",
        "benefit_from_funded_projects",
        "actual_paid_funded",
        "underpayment_vs_lindahl",
        "benefit_minus_cost_share",
    ]
    for n_agents, group in data.groupby("n_agents"):
        for metric in metrics:
            records.append(
                {
                    "n_agents": int(n_agents),
                    "metric": metric,
                    "model_mean_slope_per_100_elo": model_mean_slope(group, metric),
                    "within_run_slope_per_100_elo": within_run_slope(group, metric),
                    "metric_mean": group[metric].mean(),
                    "model_mean_sd": group.groupby(["model", "elo"])[metric]
                    .mean()
                    .std(),
                    "agent_count": len(group),
                    "run_count": group["run_key"].nunique(),
                }
            )
    return pd.DataFrame(records).sort_values(["metric", "n_agents"])


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 240,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
        }
    )


def plot_slope_by_design(slopes: pd.DataFrame, output_dir: Path) -> None:
    data = slopes[
        slopes["metric"].eq("raw") & slopes["estimand"].eq("model_mean")
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), sharex=True)
    for ax, game in zip(axes, GAME_ORDER):
        for design in ["homogeneous_focal", "heterogeneous"]:
            group = data[
                data["game_label"].eq(game) & data["design"].eq(design)
            ].sort_values("n_agents")
            yerr = np.vstack(
                (
                    group["slope_per_100_elo"] - group["ci_low"],
                    group["ci_high"] - group["slope_per_100_elo"],
                )
            )
            ax.errorbar(
                group["n_agents"],
                group["slope_per_100_elo"],
                yerr=yerr,
                marker="o",
                lw=1.7,
                capsize=3,
                color=COLORS[design],
                label=DESIGN_TITLES[design],
            )
        ax.axhline(0, color="#555555", lw=0.8)
        ax.set_title(GAME_TITLES[game])
        ax.set_xlabel("Number of agents, N")
        ax.set_xticks(N_ORDER)
    axes[0].set_ylabel("Payoff slope per 100 Elo")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("The Elo–payoff slope does not universally decrease with group size", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "slope_by_design.png", bbox_inches="tight")
    plt.close(fig)


def plot_raw_vs_normalized(slopes: pd.DataFrame, output_dir: Path) -> None:
    data = slopes[
        slopes["design"].eq("heterogeneous")
        & slopes["estimand"].eq("model_mean")
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.0), sharex=True)
    for col, game in enumerate(GAME_ORDER):
        for row, metric in enumerate(["raw", "normalized"]):
            ax = axes[row, col]
            group = data[
                data["game_label"].eq(game) & data["metric"].eq(metric)
            ].sort_values("n_agents")
            scale = 100.0 if metric == "normalized" else 1.0
            y = group["slope_per_100_elo"] * scale
            yerr = np.vstack(
                (
                    (group["slope_per_100_elo"] - group["ci_low"]) * scale,
                    (group["ci_high"] - group["slope_per_100_elo"]) * scale,
                )
            )
            ax.errorbar(
                group["n_agents"],
                y,
                yerr=yerr,
                marker="o",
                color=COLORS[metric],
                lw=1.8,
                capsize=3,
            )
            ax.axhline(0, color="#555555", lw=0.8)
            ax.set_xticks(N_ORDER)
            if row == 0:
                ax.set_title(GAME_TITLES[game])
            if row == 1:
                ax.set_xlabel("Number of agents, N")
    axes[0, 0].set_ylabel("Raw payoff / 100 Elo")
    axes[1, 0].set_ylabel("Percentage points of\nnormalized payoff / 100 Elo")
    fig.suptitle(
        "Heterogeneous-roster slopes: raw utility versus equal-share-normalized utility",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "heterogeneous_raw_vs_normalized.png", bbox_inches="tight")
    plt.close(fig)


def plot_mechanisms(
    diagnostics: pd.DataFrame,
    game3: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.0))

    game1 = diagnostics[diagnostics["game_label"].eq("game1")].sort_values("n_agents")
    axes[0].plot(
        game1["n_agents"],
        game1["paper_raw_slope"],
        marker="o",
        color=COLORS["raw"],
        label="Raw payoff slope",
    )
    axes[0].plot(
        game1["n_agents"],
        game1["paper_normalized_slope"] * 100.0,
        marker="s",
        color=COLORS["normalized"],
        label="Normalized slope (pp)",
    )
    axes[0].set_title("Game 1: scale compression")
    axes[0].set_ylabel("Slope per 100 Elo")
    axes[0].legend(frameon=False, fontsize=8)

    game2 = diagnostics[diagnostics["game_label"].eq("game2")].sort_values("n_agents")
    axes[1].plot(
        game2["n_agents"],
        game2["within_run_raw_slope"],
        marker="o",
        color="#7c3aed",
        label="Within-run Elo slope",
    )
    twin = axes[1].twinx()
    twin.plot(
        game2["n_agents"],
        game2["mean_within_run_payoff_sd"],
        marker="s",
        color="#d97706",
        label="Within-run payoff SD",
    )
    axes[1].set_title("Game 2: compromise compresses differences")
    axes[1].set_ylabel("Payoff slope / 100 Elo")
    twin.set_ylabel("Mean within-run payoff SD")
    handles1, labels1 = axes[1].get_legend_handles_labels()
    handles2, labels2 = twin.get_legend_handles_labels()
    axes[1].legend(handles1 + handles2, labels1 + labels2, frameon=False, fontsize=8)

    pivot = game3.pivot(
        index="n_agents",
        columns="metric",
        values="within_run_slope_per_100_elo",
    )
    for metric, label, color in [
        ("benefit_from_funded_projects", "Benefit slope", "#15803d"),
        ("actual_paid_funded", "Contribution-cost slope", "#d97706"),
        ("actual_raw_utility", "Net utility slope", "#b91c1c"),
    ]:
        axes[2].plot(
            pivot.index,
            pivot[metric],
            marker="o",
            lw=1.7,
            label=label,
            color=color,
        )
    axes[2].set_title("Game 3: benefits spill over; costs concentrate")
    axes[2].set_ylabel("Undiscounted slope / 100 Elo")
    axes[2].legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.set_xlabel("Number of agents, N")
        ax.set_xticks(N_ORDER)
        ax.axhline(0, color="#555555", lw=0.8)
    fig.tight_layout()
    fig.savefig(output_dir / "mechanism_diagnostics.png", bbox_inches="tight")
    plt.close(fig)


def plot_competition_cells(competition: pd.DataFrame, output_dir: Path) -> None:
    data = competition[competition["design"].eq("heterogeneous")]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), sharex=True)
    palettes = ["viridis", "plasma", "cividis"]
    for ax, game, palette in zip(axes, GAME_ORDER, palettes):
        game_data = data[data["game_label"].eq(game)]
        labels = list(game_data["competition_label"].drop_duplicates())
        colors = sns.color_palette(palette, n_colors=len(labels))
        for label, color in zip(labels, colors):
            group = game_data[game_data["competition_label"].eq(label)].sort_values(
                "n_agents"
            )
            ax.plot(
                group["n_agents"],
                group["raw_slope_per_100_elo"],
                marker="o",
                lw=1.4,
                color=color,
                label=label,
            )
        ax.axhline(0, color="#555555", lw=0.8)
        ax.set_title(GAME_TITLES[game])
        ax.set_xlabel("Number of agents, N")
        ax.set_xticks(N_ORDER)
        ax.legend(frameon=False, fontsize=6.5)
    axes[0].set_ylabel("Raw payoff slope per 100 Elo")
    fig.suptitle("Heterogeneous-roster slopes within competition settings", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "competition_cell_slopes.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    runs, agents = load_analysis_data(args.table_dir.resolve(), output_dir)
    rng = np.random.default_rng(args.seed)

    slopes, endpoints, _ = build_slope_tables(
        agents, reps=args.bootstrap_reps, rng=rng
    )
    competition = build_competition_slopes(agents)
    interactions = build_within_run_interactions(agents)
    diagnostics = build_mechanism_diagnostics(runs, agents)
    group_capability = build_group_capability_coefficients(agents)
    game3 = build_game3_decomposition()

    tables = {
        "slope_estimates.csv": slopes,
        "endpoint_changes.csv": endpoints,
        "within_run_interactions.csv": interactions,
        "competition_cell_slopes.csv": competition,
        "mechanism_diagnostics.csv": diagnostics,
        "group_capability_coefficients.csv": group_capability,
        "game3_benefit_cost_decomposition.csv": game3,
    }
    for filename, table in tables.items():
        table.to_csv(output_dir / filename, index=False)

    plot_slope_by_design(slopes, output_dir)
    plot_raw_vs_normalized(slopes, output_dir)
    plot_mechanisms(diagnostics, game3, output_dir)
    plot_competition_cells(competition, output_dir)

    print(f"Wrote Reviewer Item 8 analysis to {output_dir}")
    for filename in tables:
        print(f"  {filename}")
    for filename in [
        "slope_by_design.png",
        "heterogeneous_raw_vs_normalized.png",
        "mechanism_diagnostics.png",
        "competition_cell_slopes.png",
    ]:
        print(f"  {filename}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare payoff inequality in heterogeneous vs homogeneous multiagent runs."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = (
    PROJECT_ROOT
    / "experiments/results/n2_plus_multiagent_comparison_analysis_20260505"
    / "tables_multiagent"
)
OUT_DIR = PROJECT_ROOT / "overleaf/neurips/graphics/n_gt_2_report"

AGENT_FILES = [
    TABLE_DIR / "heterogeneous_agents_fresh.csv",
    TABLE_DIR / "homogeneous_agents_fresh.csv",
]
N_ORDER = [2, 4, 6, 8, 10]
GAME_ORDER = ["game1", "game2", "game3"]
GAME_LABELS = {"game1": "Game 1", "game2": "Game 2", "game3": "Game 3"}
GROUP_ORDER = ["heterogeneous", "homogeneous_all", "homogeneous_control"]
GROUP_LABELS = {
    "heterogeneous": "Heterogeneous",
    "homogeneous_all": "All homogeneous",
    "homogeneous_control": "Homogeneous control",
}
GROUP_COLORS = {
    "heterogeneous": "#D54E6A",
    "homogeneous_all": "#4E79A7",
    "homogeneous_control": "#59A14F",
}


def gini_shifted(values: pd.Series | np.ndarray) -> tuple[float, float, bool]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan, math.nan, False
    shifted = False
    if float(arr.min()) < 0:
        arr = arr - float(arr.min())
        shifted = True
    if arr.size < 2 or np.allclose(arr, arr[0]) or np.allclose(arr, 0.0):
        return 0.0, 0.0, shifted
    mean_value = float(arr.mean())
    if math.isclose(mean_value, 0.0):
        return 0.0, 0.0, shifted
    diffs = np.abs(arr[:, None] - arr[None, :])
    raw_gini = float(np.mean(diffs) / (2.0 * mean_value))
    corrected = min(raw_gini * float(arr.size / (arr.size - 1)), 1.0)
    return raw_gini, corrected, shifted


def load_agents() -> pd.DataFrame:
    frames = []
    for path in AGENT_FILES:
        frame = pd.read_csv(path)
        frames.append(frame)
    agents = pd.concat(frames, ignore_index=True)
    agents["final_utility"] = pd.to_numeric(agents["final_utility"], errors="coerce")
    agents["elo"] = pd.to_numeric(agents["elo"], errors="coerce")
    agents["n_agents"] = pd.to_numeric(agents["n_agents"], errors="coerce").astype("Int64")
    agents = agents.dropna(subset=["run_key", "final_utility", "n_agents"])
    agents["n_agents"] = agents["n_agents"].astype(int)
    return agents


def compute_run_metrics(agents: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_key, group in agents.groupby("run_key", sort=False):
        utilities = group["final_utility"].to_numpy(dtype=float)
        raw_gini, corrected_gini, shifted = gini_shifted(utilities)
        experiment_family = group["experiment_family"].iloc[0]
        if experiment_family == "heterogeneous_random":
            comparison_group = "heterogeneous"
        elif experiment_family == "homogeneous_control":
            comparison_group = "homogeneous_control"
        elif str(experiment_family).startswith("homogeneous"):
            comparison_group = "homogeneous_adversary"
        else:
            comparison_group = str(experiment_family)

        rows.append(
            {
                "run_key": run_key,
                "config_id": int(group["config_id"].iloc[0]),
                "experiment_family": experiment_family,
                "comparison_group": comparison_group,
                "game_label": group["game_label"].iloc[0],
                "n_agents": int(group["n_agents"].iloc[0]),
                "competition_ci": float(group["competition_ci"].iloc[0]),
                "competition_label_ci": group["competition_label_ci"].iloc[0],
                "competition_band": group["competition_band"].iloc[0],
                "model_count": int(group["model"].nunique()),
                "model_list": "; ".join(sorted(group["model_short"].dropna().unique())),
                "mean_roster_elo": float(group["elo"].mean()),
                "elo_std": float(np.std(group["elo"].to_numpy(dtype=float), ddof=0)),
                "elo_variance": float(np.var(group["elo"].to_numpy(dtype=float), ddof=0)),
                "payoff_variance": float(np.var(utilities, ddof=0)),
                "payoff_std": float(np.std(utilities, ddof=0)),
                "payoff_gini_raw_shifted": raw_gini,
                "payoff_gini_corrected": corrected_gini,
                "payoff_gini_shifted_for_negative": shifted,
                "mean_payoff": float(np.mean(utilities)),
                "min_payoff": float(np.min(utilities)),
                "max_payoff": float(np.max(utilities)),
            }
        )
    runs = pd.DataFrame(rows)

    homogeneous_all = runs[runs["comparison_group"].isin(["homogeneous_control", "homogeneous_adversary"])].copy()
    homogeneous_all["comparison_group"] = "homogeneous_all"
    return pd.concat([runs, homogeneous_all], ignore_index=True)


def sem(values: pd.Series) -> float:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return math.nan
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


def summarize(frame: pd.DataFrame, group_cols: list[str], scope: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, sub in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys, strict=True))
        row.update(
            {
                "scope": scope,
                "n_runs": len(sub),
                "payoff_variance_mean": float(sub["payoff_variance"].mean()),
                "payoff_variance_sem": sem(sub["payoff_variance"]),
                "payoff_gini_mean": float(sub["payoff_gini_corrected"].mean()),
                "payoff_gini_sem": sem(sub["payoff_gini_corrected"]),
                "mean_roster_elo_mean": float(sub["mean_roster_elo"].mean()),
                "elo_std_mean": float(sub["elo_std"].mean()),
                "mean_payoff_mean": float(sub["mean_payoff"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def comparison_tests(frame: pd.DataFrame, metric: str, left: str, right: str, scope: str) -> dict[str, object]:
    a = frame[frame["comparison_group"].eq(left)][metric].replace([np.inf, -np.inf], np.nan).dropna()
    b = frame[frame["comparison_group"].eq(right)][metric].replace([np.inf, -np.inf], np.nan).dropna()
    if len(a) < 2 or len(b) < 2:
        return {
            "scope": scope,
            "metric": metric,
            "left_group": left,
            "right_group": right,
            "left_n": len(a),
            "right_n": len(b),
            "left_mean": float(a.mean()) if len(a) else math.nan,
            "right_mean": float(b.mean()) if len(b) else math.nan,
            "diff_left_minus_right": math.nan,
            "welch_t_p": math.nan,
            "mannwhitney_p": math.nan,
        }
    t = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    mw = stats.mannwhitneyu(a, b, alternative="two-sided")
    diff = float(a.mean() - b.mean())
    return {
        "scope": scope,
        "metric": metric,
        "left_group": left,
        "right_group": right,
        "left_n": len(a),
        "right_n": len(b),
        "left_mean": float(a.mean()),
        "right_mean": float(b.mean()),
        "diff_left_minus_right": diff,
        "pct_diff_vs_right": 100.0 * diff / float(b.mean()) if not math.isclose(float(b.mean()), 0.0) else math.nan,
        "welch_t_stat": float(t.statistic),
        "welch_t_p": float(t.pvalue),
        "mannwhitney_p": float(mw.pvalue),
    }


def cell_balanced_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cell_cols = ["comparison_group", "game_label", "n_agents", "competition_label_ci"]
    cell_means = (
        frame.groupby(cell_cols, dropna=False)
        .agg(
            n_runs=("run_key", "count"),
            payoff_variance=("payoff_variance", "mean"),
            payoff_gini_corrected=("payoff_gini_corrected", "mean"),
            mean_roster_elo=("mean_roster_elo", "mean"),
            elo_std=("elo_std", "mean"),
            mean_payoff=("mean_payoff", "mean"),
        )
        .reset_index()
    )
    return cell_means


def plot_comparison_bars(summary: pd.DataFrame, filename: str) -> Path:
    metrics = [
        ("payoff_variance_mean", "payoff_variance_sem", "Payoff variance"),
        ("payoff_gini_mean", "payoff_gini_sem", "Corrected payoff Gini"),
    ]
    groups = GROUP_ORDER
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2))
    for ax, (mean_col, sem_col, ylabel) in zip(axes, metrics, strict=True):
        sub = summary[summary["comparison_group"].isin(groups)].set_index("comparison_group").reindex(groups)
        xs = np.arange(len(groups))
        means = sub[mean_col].to_numpy(dtype=float)
        errors = sub[sem_col].to_numpy(dtype=float)
        ax.bar(xs, means, yerr=errors, capsize=3, color=[GROUP_COLORS[g] for g in groups], alpha=0.86)
        for x, mean, n_runs in zip(xs, means, sub["n_runs"], strict=True):
            ax.text(x, mean, f"{mean:.3g}\nn={int(n_runs)}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels([GROUP_LABELS[g] for g in groups], rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    fig.suptitle("Payoff inequality by experiment family")
    fig.tight_layout()
    out_path = OUT_DIR / filename
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_by_n(summary: pd.DataFrame, filename: str) -> Path:
    metrics = [
        ("payoff_variance_mean", "payoff_variance_sem", "Payoff variance"),
        ("payoff_gini_mean", "payoff_gini_sem", "Corrected payoff Gini"),
    ]
    groups = GROUP_ORDER
    fig, axes = plt.subplots(2, len(N_ORDER), figsize=(15.2, 6.4), sharey="row")
    for row_idx, (mean_col, sem_col, ylabel) in enumerate(metrics):
        for col_idx, n_agents in enumerate(N_ORDER):
            ax = axes[row_idx, col_idx]
            sub = (
                summary[summary["n_agents"].eq(n_agents) & summary["comparison_group"].isin(groups)]
                .set_index("comparison_group")
                .reindex(groups)
            )
            xs = np.arange(len(groups))
            means = sub[mean_col].to_numpy(dtype=float)
            errors = sub[sem_col].to_numpy(dtype=float)
            ax.bar(xs, means, yerr=errors, capsize=2, color=[GROUP_COLORS[g] for g in groups], alpha=0.86)
            ax.set_title(f"N={n_agents}" if row_idx == 0 else "")
            ax.set_xticks(xs)
            ax.set_xticklabels([GROUP_LABELS[g] for g in groups], rotation=45, ha="right", fontsize=7)
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            ax.grid(True, axis="y", alpha=0.25)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
    fig.suptitle("Payoff inequality by group size")
    fig.tight_layout()
    out_path = OUT_DIR / filename
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    agents = load_agents()
    runs = compute_run_metrics(agents)
    runs.to_csv(OUT_DIR / "homogeneous_vs_heterogeneous_inequality_run_metrics.csv", index=False)

    relevant = runs[runs["comparison_group"].isin(GROUP_ORDER)].copy()
    raw_summary = summarize(relevant, ["comparison_group"], "raw_run_weighted").sort_values("comparison_group")
    raw_summary.to_csv(OUT_DIR / "homogeneous_vs_heterogeneous_inequality_raw_summary.csv", index=False)

    by_n_summary = summarize(relevant, ["comparison_group", "n_agents"], "raw_by_n").sort_values(
        ["comparison_group", "n_agents"]
    )
    by_n_summary.to_csv(OUT_DIR / "homogeneous_vs_heterogeneous_inequality_by_n_summary.csv", index=False)

    by_game_summary = summarize(relevant, ["comparison_group", "game_label"], "raw_by_game").sort_values(
        ["comparison_group", "game_label"]
    )
    by_game_summary.to_csv(OUT_DIR / "homogeneous_vs_heterogeneous_inequality_by_game_summary.csv", index=False)

    by_comp_summary = summarize(relevant, ["comparison_group", "competition_band"], "raw_by_competition_band").sort_values(
        ["comparison_group", "competition_band"]
    )
    by_comp_summary.to_csv(OUT_DIR / "homogeneous_vs_heterogeneous_inequality_by_competition_summary.csv", index=False)

    cell_means = cell_balanced_frame(relevant)
    cell_means.to_csv(OUT_DIR / "homogeneous_vs_heterogeneous_inequality_cell_means.csv", index=False)
    cell_balanced_summary = summarize(cell_means, ["comparison_group"], "cell_balanced").sort_values("comparison_group")
    cell_balanced_summary.to_csv(
        OUT_DIR / "homogeneous_vs_heterogeneous_inequality_cell_balanced_summary.csv",
        index=False,
    )

    tests = []
    for metric in ["payoff_variance", "payoff_gini_corrected"]:
        for right in ["homogeneous_all", "homogeneous_control"]:
            tests.append(comparison_tests(relevant, metric, "heterogeneous", right, "raw_run_weighted"))
            tests.append(comparison_tests(cell_means, metric, "heterogeneous", right, "cell_balanced"))
    tests_frame = pd.DataFrame(tests)
    tests_frame.to_csv(OUT_DIR / "homogeneous_vs_heterogeneous_inequality_tests.csv", index=False)

    paths = [
        plot_comparison_bars(raw_summary, "homogeneous_vs_heterogeneous_inequality_raw_bars.png"),
        plot_comparison_bars(
            cell_balanced_summary,
            "homogeneous_vs_heterogeneous_inequality_cell_balanced_bars.png",
        ),
        plot_by_n(by_n_summary, "homogeneous_vs_heterogeneous_inequality_by_n_bars.png"),
    ]

    print("Raw run-weighted summary:")
    print(
        raw_summary[
            [
                "comparison_group",
                "n_runs",
                "payoff_variance_mean",
                "payoff_gini_mean",
                "mean_roster_elo_mean",
                "elo_std_mean",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print("\nCell-balanced summary:")
    print(
        cell_balanced_summary[
            [
                "comparison_group",
                "n_runs",
                "payoff_variance_mean",
                "payoff_gini_mean",
                "mean_roster_elo_mean",
                "elo_std_mean",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print("\nTests:")
    print(
        tests_frame[
            [
                "scope",
                "metric",
                "right_group",
                "left_mean",
                "right_mean",
                "diff_left_minus_right",
                "pct_diff_vs_right",
                "welch_t_p",
                "mannwhitney_p",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )
    print("\nWrote:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()

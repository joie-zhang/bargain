#!/usr/bin/env python3
"""Faceted versions of the homogeneous/heterogeneous bucketed bar plot."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

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
GAME_TITLES = {"game1": "Game 1", "game2": "Game 2", "game3": "Game 3"}
COMPETITION_ORDER = ["cooperative", "middle", "competitive"]
COMPETITION_TITLES = {
    "cooperative": "Low competition",
    "middle": "Medium competition",
    "competitive": "High competition",
}

METRIC_SETS = {
    "variance_mean_payoff": [
        ("payoff_variance", "Payoff variance", "{:.0f}"),
        ("mean_payoff", "Average payoff", "{:.0f}"),
    ],
    "gini_mean_payoff": [
        ("payoff_gini_corrected", "Corrected payoff Gini", "{:.3f}"),
        ("mean_payoff", "Average payoff", "{:.0f}"),
    ],
}
SUMMARY_METRICS = [
    ("payoff_variance", "Payoff variance", "{:.1f}"),
    ("payoff_gini_corrected", "Corrected payoff Gini", "{:.3f}"),
    ("mean_payoff", "Average payoff", "{:.1f}"),
]

SECTION_COLORS = {
    "overall": "#4E79A7",
    "hom_adv_elo": "#F28E2B",
    "hetero_mean_elo": "#8E63B0",
    "hetero_max_elo": "#59A14F",
}


def sem(values: pd.Series) -> float:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return math.nan
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


def gini_shifted_corrected(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan
    if float(arr.min()) < 0:
        arr = arr - float(arr.min())
    if arr.size < 2 or np.allclose(arr, arr[0]) or np.allclose(arr, 0.0):
        return 0.0
    mean_value = float(arr.mean())
    if math.isclose(mean_value, 0.0):
        return 0.0
    raw_gini = float(np.mean(np.abs(arr[:, None] - arr[None, :])) / (2.0 * mean_value))
    return min(raw_gini * float(arr.size / (arr.size - 1)), 1.0)


def short_float(value: float) -> str:
    if math.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:.1f}"


def load_agents() -> pd.DataFrame:
    agents = pd.concat([pd.read_csv(path) for path in AGENT_FILES], ignore_index=True)
    for col in ["final_utility", "elo", "n_agents", "competition_ci"]:
        agents[col] = pd.to_numeric(agents[col], errors="coerce")
    agents = agents.dropna(subset=["run_key", "final_utility", "elo", "n_agents"]).copy()
    agents["n_agents"] = agents["n_agents"].astype(int)
    return agents


def compute_run_metrics(agents: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_key, group in agents.groupby("run_key", sort=False):
        utilities = group["final_utility"].to_numpy(dtype=float)
        elos = group["elo"].to_numpy(dtype=float)
        experiment_family = str(group["experiment_family"].iloc[0])
        if experiment_family == "heterogeneous_random":
            comparison_group = "heterogeneous"
        elif experiment_family == "homogeneous_control":
            comparison_group = "homogeneous_control"
        elif experiment_family.startswith("homogeneous"):
            comparison_group = "homogeneous_adversary"
        else:
            comparison_group = experiment_family

        adversary_rows = group[group["role"].astype(str).eq("adversary")]
        adversary_elo = math.nan
        adversary_model_short = ""
        if not adversary_rows.empty:
            adversary_elo = float(adversary_rows["elo"].iloc[0])
            adversary_model_short = str(adversary_rows["model_short"].iloc[0])

        rows.append(
            {
                "run_key": run_key,
                "config_id": int(group["config_id"].iloc[0]),
                "experiment_family": experiment_family,
                "comparison_group": comparison_group,
                "game_label": str(group["game_label"].iloc[0]),
                "n_agents": int(group["n_agents"].iloc[0]),
                "competition_band": str(group["competition_band"].iloc[0]),
                "competition_ci": float(group["competition_ci"].iloc[0]),
                "competition_label_ci": str(group["competition_label_ci"].iloc[0]),
                "mean_roster_elo": float(np.mean(elos)),
                "max_roster_elo": float(np.max(elos)),
                "elo_std": float(np.std(elos, ddof=0)),
                "adversary_elo": adversary_elo,
                "adversary_model_short": adversary_model_short,
                "payoff_variance": float(np.var(utilities, ddof=0)),
                "payoff_gini_corrected": gini_shifted_corrected(utilities),
                "mean_payoff": float(np.mean(utilities)),
            }
        )
    return pd.DataFrame(rows)


def build_legacy4_bar_rows(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pieces: list[pd.DataFrame] = []
    specs: list[dict[str, object]] = []

    overall_specs = [
        ("Heterogeneous\nall", "Het all", "overall", 0, "heterogeneous"),
        ("Homogeneous adversary\nall", "Hom adv", "overall", 1, "homogeneous_adversary"),
        ("Hom control\nGPT-5-nano", "Control", "overall", 2, "homogeneous_control"),
    ]
    for label, short_label, section, sort_key, group_name in overall_specs:
        sub = runs[runs["comparison_group"].eq(group_name)].copy()
        sub["bucket_section"] = section
        sub["bucket_sort"] = sort_key
        sub["bucket_label"] = label
        sub["bucket_short_label"] = short_label
        pieces.append(sub)
        specs.append(
            {
                "bucket_section": section,
                "bucket_sort": sort_key,
                "bucket_label": label,
                "bucket_short_label": short_label,
            }
        )

    hom_adv = runs[runs["comparison_group"].eq("homogeneous_adversary")].copy()
    unique_adv = (
        hom_adv[["adversary_model_short", "adversary_elo"]]
        .drop_duplicates()
        .sort_values("adversary_elo")
        .reset_index(drop=True)
    )
    unique_adv["bucket_code"] = np.minimum(np.floor(np.arange(len(unique_adv)) * 4 / len(unique_adv)).astype(int), 3)
    label_map: dict[int, str] = {}
    for code, sub in unique_adv.groupby("bucket_code"):
        label_map[int(code)] = (
            f"Hom adv Q{int(code) + 1}\n"
            f"{short_float(float(sub['adversary_elo'].min()))}-{short_float(float(sub['adversary_elo'].max()))}"
        )
    hom_adv = hom_adv.merge(unique_adv, on=["adversary_model_short", "adversary_elo"], how="left")
    hom_adv["bucket_section"] = "hom_adv_elo"
    hom_adv["bucket_sort"] = 10 + hom_adv["bucket_code"].astype(int)
    hom_adv["bucket_label"] = hom_adv["bucket_code"].astype(int).map(label_map)
    hom_adv["bucket_short_label"] = hom_adv["bucket_code"].astype(int).map(lambda code: f"HA Q{code + 1}")
    pieces.append(hom_adv)
    for code in range(4):
        specs.append(
            {
                "bucket_section": "hom_adv_elo",
                "bucket_sort": 10 + code,
                "bucket_label": label_map.get(code, f"Hom adv Q{code + 1}"),
                "bucket_short_label": f"HA Q{code + 1}",
            }
        )

    hetero = runs[runs["comparison_group"].eq("heterogeneous")].copy()
    for value_col, section, prefix, short_prefix, offset in [
        ("mean_roster_elo", "hetero_mean_elo", "Hetero mean Elo", "Mean", 20),
        ("max_roster_elo", "hetero_max_elo", "Hetero max Elo", "Max", 30),
    ]:
        sub = hetero.copy()
        sub["bucket_code"] = pd.qcut(sub[value_col].rank(method="first"), q=4, labels=False).astype(int)
        labels: dict[int, str] = {}
        for code in range(4):
            bucket_values = sub.loc[sub["bucket_code"].eq(code), value_col]
            labels[code] = (
                f"{prefix} Q{code + 1}\n"
                f"{short_float(float(bucket_values.min()))}-{short_float(float(bucket_values.max()))}"
            )
            specs.append(
                {
                    "bucket_section": section,
                    "bucket_sort": offset + code,
                    "bucket_label": labels[code],
                    "bucket_short_label": f"{short_prefix} Q{code + 1}",
                }
            )
        sub["bucket_section"] = section
        sub["bucket_sort"] = offset + sub["bucket_code"].astype(int)
        sub["bucket_label"] = sub["bucket_code"].astype(int).map(labels)
        sub["bucket_short_label"] = sub["bucket_code"].astype(int).map(lambda code: f"{short_prefix} Q{code + 1}")
        pieces.append(sub)

    spec_frame = pd.DataFrame(specs).sort_values("bucket_sort").reset_index(drop=True)
    return pd.concat(pieces, ignore_index=True), spec_frame


def filter_frame(frame: pd.DataFrame, filters: dict[str, object]) -> pd.DataFrame:
    out = frame
    for col, value in filters.items():
        out = out[out[col].eq(value)]
    return out


def summarize_subset(subset: pd.DataFrame, specs: pd.DataFrame, extra: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in specs.itertuples(index=False):
        sub = subset[
            subset["bucket_section"].eq(spec.bucket_section)
            & subset["bucket_sort"].eq(spec.bucket_sort)
        ]
        row: dict[str, object] = {
            **extra,
            "bucket_section": spec.bucket_section,
            "bucket_sort": int(spec.bucket_sort),
            "bucket_label": str(spec.bucket_label).replace("\n", " | "),
            "bucket_short_label": str(spec.bucket_short_label),
            "n_runs": int(len(sub)),
        }
        for metric, _, _ in SUMMARY_METRICS:
            row[f"{metric}_mean"] = float(sub[metric].mean()) if len(sub) else math.nan
            row[f"{metric}_sem"] = sem(sub[metric]) if len(sub) else math.nan
            row[f"{metric}_median"] = float(sub[metric].median()) if len(sub) else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_all_cells(
    bar_rows: pd.DataFrame,
    specs: pd.DataFrame,
    row_groups: list[tuple[str, dict[str, object], dict[str, object]]],
    col_groups: list[tuple[str, dict[str, object], dict[str, object]]],
    scope: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for row_label, row_filter, row_extra in row_groups:
        for col_label, col_filter, col_extra in col_groups:
            filters = {**row_filter, **col_filter}
            subset = filter_frame(bar_rows, filters)
            extra = {"scope": scope, "row_label": row_label, "col_label": col_label, **row_extra, **col_extra}
            frames.append(summarize_subset(subset, specs, extra))
    return pd.concat(frames, ignore_index=True)


def y_max_for_metric(summary: pd.DataFrame, metric: str) -> float:
    means = summary[f"{metric}_mean"].to_numpy(dtype=float)
    errors = np.nan_to_num(summary[f"{metric}_sem"].to_numpy(dtype=float), nan=0.0)
    finite = np.isfinite(means)
    if not finite.any():
        return 1.0
    return float(np.nanmax(means[finite] + errors[finite]) * 1.18)


def y_min_for_metric(metric: str) -> float:
    if metric == "payoff_gini_corrected":
        return 0.1
    if metric == "mean_payoff":
        return 30.0
    return 0.0


def draw_bar_axis(ax: plt.Axes, summary: pd.DataFrame, specs: pd.DataFrame, metric: str, y_max: float) -> None:
    summary = summary.set_index("bucket_sort").reindex(specs["bucket_sort"]).reset_index()
    x = np.arange(len(specs))
    means = summary[f"{metric}_mean"].to_numpy(dtype=float)
    errors = np.nan_to_num(summary[f"{metric}_sem"].to_numpy(dtype=float), nan=0.0)
    valid = np.isfinite(means) & (summary["n_runs"].to_numpy(dtype=int) > 0)
    colors = [SECTION_COLORS[section] for section in specs["bucket_section"]]
    ax.bar(
        x[valid],
        means[valid],
        yerr=errors[valid],
        capsize=1.5,
        color=[colors[i] for i in np.where(valid)[0]],
        alpha=0.86,
        edgecolor="white",
        linewidth=0.4,
    )
    y_min = y_min_for_metric(metric)
    if y_max <= y_min:
        y_max = y_min + 0.01
    ax.set_ylim(y_min, y_max)
    for boundary in [2.5, 6.5, 10.5]:
        ax.axvline(boundary, color="#777777", alpha=0.20, linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.18, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=6.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_grid(
    bar_rows: pd.DataFrame,
    specs: pd.DataFrame,
    row_groups: list[tuple[str, dict[str, object], dict[str, object]]],
    col_groups: list[tuple[str, dict[str, object], dict[str, object]]],
    plot_metrics: list[tuple[str, str, str]],
    scope: str,
    title: str,
    filename: str,
    figsize: tuple[float, float],
) -> tuple[Path, pd.DataFrame]:
    summary = summarize_all_cells(bar_rows, specs, row_groups, col_groups, scope)
    n_plot_rows = len(row_groups) * len(plot_metrics)
    n_plot_cols = len(col_groups)
    fig, axes = plt.subplots(n_plot_rows, n_plot_cols, figsize=figsize, sharex=True, squeeze=False)
    y_max = {metric: y_max_for_metric(summary, metric) for metric, _, _ in plot_metrics}

    for rg_idx, (row_label, _, _) in enumerate(row_groups):
        for metric_idx, (metric, ylabel, _) in enumerate(plot_metrics):
            plot_row = rg_idx * len(plot_metrics) + metric_idx
            for cg_idx, (col_label, _, _) in enumerate(col_groups):
                ax = axes[plot_row, cg_idx]
                cell = summary[
                    summary["row_label"].eq(row_label)
                    & summary["col_label"].eq(col_label)
                ]
                draw_bar_axis(ax, cell, specs, metric, y_max[metric])
                if plot_row == 0:
                    ax.set_title(col_label, fontsize=9, pad=5)
                if cg_idx == 0:
                    if len(row_groups) == 1:
                        ax.set_ylabel(ylabel, fontsize=8)
                    else:
                        ax.set_ylabel(f"{row_label}\n{ylabel}", fontsize=8)
                if plot_row == n_plot_rows - 1:
                    ax.set_xticks(np.arange(len(specs)))
                    ax.set_xticklabels(specs["bucket_short_label"], rotation=55, ha="right", fontsize=6)
                else:
                    ax.tick_params(axis="x", labelbottom=False)

    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    out_path = OUT_DIR / filename
    fig.savefig(out_path, dpi=210, bbox_inches="tight")
    plt.close(fig)
    return out_path, summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = compute_run_metrics(load_agents())
    bar_rows, specs = build_legacy4_bar_rows(runs)

    all_summaries: list[pd.DataFrame] = []
    plot_paths: list[Path] = []

    n_cols = [(f"N={n}", {"n_agents": n}, {"n_agents": n}) for n in N_ORDER]
    game_cols = [(GAME_TITLES[g], {"game_label": g}, {"game_label": g}) for g in GAME_ORDER]
    comp_cols = [
        (COMPETITION_TITLES[c], {"competition_band": c}, {"competition_band": c})
        for c in COMPETITION_ORDER
    ]

    plot_specs = [
        (
            [("All", {}, {})],
            n_cols,
            "by_n",
            "broken down by N",
            "by_n",
            (22.0, 6.2),
        ),
        (
            [("All", {}, {})],
            comp_cols,
            "by_competition",
            "broken down by competition band",
            "by_competition",
            (14.0, 6.2),
        ),
        (
            [("All", {}, {})],
            game_cols,
            "by_game",
            "broken down by game",
            "by_game",
            (14.0, 6.2),
        ),
        (
            game_cols,
            comp_cols,
            "by_game_competition",
            "broken down by game and competition band",
            "by_game_competition",
            (14.0, 13.5),
        ),
    ]
    metric_families = [
        (
            "variance_mean_payoff",
            "Payoff dispersion and average payoff",
            "homogeneous_heterogeneous_bucketed_variance_mean_payoff",
            METRIC_SETS["variance_mean_payoff"],
        ),
        (
            "gini_mean_payoff",
            "Gini inequality and average payoff",
            "homogeneous_heterogeneous_bucketed_gini_mean_payoff",
            METRIC_SETS["gini_mean_payoff"],
        ),
    ]

    for metric_family, title_prefix, file_prefix, plot_metrics in metric_families:
        path, summary = plot_grid(
            bar_rows,
            specs,
            [("All", {}, {})],
            [("All", {}, {})],
            plot_metrics,
            "base",
            f"{title_prefix} by roster bucket",
            f"{file_prefix}_bars.png",
            (9.5, 6.2),
        )
        plot_paths.append(path)
        all_summaries.append(summary.assign(metric_family=metric_family))

        for row_groups, col_groups, scope, title_suffix, filename_suffix, figsize in plot_specs:
            path, summary = plot_grid(
                bar_rows,
                specs,
                row_groups,
                col_groups,
                plot_metrics,
                scope,
                f"{title_prefix} by roster bucket, {title_suffix}",
                f"{file_prefix}_{filename_suffix}.png",
                figsize,
            )
            plot_paths.append(path)
            all_summaries.append(summary.assign(metric_family=metric_family))

        for n in N_ORDER:
            title = (
                f"{title_prefix} by roster bucket, "
                f"broken down by game and competition band for N={n}"
            )
            path, summary = plot_grid(
                filter_frame(bar_rows, {"n_agents": n}),
                specs,
                game_cols,
                comp_cols,
                plot_metrics,
                f"by_n_game_competition_n{n}",
                title,
                f"{file_prefix}_by_n{n}_game_competition.png",
                (14.0, 13.5),
            )
            plot_paths.append(path)
            all_summaries.append(summary.assign(metric_family=metric_family))

    summary_path = OUT_DIR / "homogeneous_heterogeneous_bucketed_variance_gini_mean_payoff_breakdown_summary.csv"
    run_metrics_path = OUT_DIR / "homogeneous_heterogeneous_bucketed_variance_mean_payoff_breakdown_run_metrics.csv"
    pd.concat(all_summaries, ignore_index=True).to_csv(summary_path, index=False)
    runs.to_csv(run_metrics_path, index=False)

    print(f"Wrote {run_metrics_path}")
    print(f"Wrote {summary_path}")
    for path in plot_paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()

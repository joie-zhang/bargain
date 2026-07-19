#!/usr/bin/env python3
"""Role payoff curves for homogeneous adversary and heterogeneous max-Elo runs."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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

SCENARIOS = {
    "homogeneous_adversary": {
        "title": "Homogeneous adversary",
        "x_label": "Adversary Elo bucket",
        "file_prefix": "homogeneous_adversary_role_payoff_vs_adversary_elo",
        "high_role": "Adversary",
        "low_role": "Baseline agents",
        "high_color": "#D95F02",
        "low_color": "#4E79A7",
    },
    "heterogeneous_max": {
        "title": "Heterogeneous max-Elo agent",
        "x_label": "Max roster Elo bucket",
        "file_prefix": "heterogeneous_max_role_payoff_vs_max_elo",
        "high_role": "Max-Elo agent(s)",
        "low_role": "Non-max agents",
        "high_color": "#2CA02C",
        "low_color": "#4E79A7",
    },
}


def sem(values: pd.Series) -> float:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return math.nan
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


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


def homogeneous_adversary_bucket_map(agents: pd.DataFrame) -> pd.DataFrame:
    hom_adv = agents[agents["experiment_family"].astype(str).str.startswith("homogeneous")]
    hom_adv = hom_adv[hom_adv["role"].astype(str).eq("adversary")]
    unique_adv = (
        hom_adv[["model_short", "elo"]]
        .drop_duplicates()
        .sort_values("elo")
        .rename(columns={"model_short": "adversary_model_short", "elo": "adversary_elo"})
        .reset_index(drop=True)
    )
    unique_adv["bucket_code"] = np.minimum(np.floor(np.arange(len(unique_adv)) * 4 / len(unique_adv)).astype(int), 3)
    labels = []
    for code, sub in unique_adv.groupby("bucket_code"):
        names = ", ".join(sub["adversary_model_short"].tolist())
        labels.append(
            {
                "bucket_code": int(code),
                "bucket_x": float(sub["adversary_elo"].mean()),
                "bucket_label": (
                    f"Q{int(code) + 1}\n"
                    f"{short_float(float(sub['adversary_elo'].min()))}-{short_float(float(sub['adversary_elo'].max()))}"
                ),
                "bucket_detail": names,
            }
        )
    labels_df = pd.DataFrame(labels)
    return unique_adv.merge(labels_df, on="bucket_code", how="left")


def heterogeneous_max_bucket_table(run_rows: pd.DataFrame) -> pd.DataFrame:
    hetero = run_rows[run_rows["scenario"].eq("heterogeneous_max")].copy()
    hetero["bucket_code"] = pd.qcut(hetero["max_roster_elo"].rank(method="first"), q=4, labels=False).astype(int)
    rows: list[dict[str, object]] = []
    for code, sub in hetero.groupby("bucket_code"):
        rows.append(
            {
                "bucket_code": int(code),
                "bucket_x": float(sub["max_roster_elo"].mean()),
                "bucket_label": (
                    f"Q{int(code) + 1}\n"
                    f"{short_float(float(sub['max_roster_elo'].min()))}-{short_float(float(sub['max_roster_elo'].max()))}"
                ),
                "bucket_detail": "",
            }
        )
    return pd.DataFrame(rows)


def compute_run_rows(agents: pd.DataFrame) -> pd.DataFrame:
    hom_bucket = homogeneous_adversary_bucket_map(agents)
    rows: list[dict[str, object]] = []

    for run_key, group in agents.groupby("run_key", sort=False):
        experiment_family = str(group["experiment_family"].iloc[0])
        game_label = str(group["game_label"].iloc[0])
        n_agents = int(group["n_agents"].iloc[0])
        competition_band = str(group["competition_band"].iloc[0])
        competition_ci = float(group["competition_ci"].iloc[0])
        competition_label_ci = str(group["competition_label_ci"].iloc[0])

        if experiment_family.startswith("homogeneous") and experiment_family != "homogeneous_control":
            adv = group[group["role"].astype(str).eq("adversary")]
            base = group[~group["role"].astype(str).eq("adversary")]
            if adv.empty or base.empty:
                continue
            adversary_elo = float(adv["elo"].iloc[0])
            adversary_model_short = str(adv["model_short"].iloc[0])
            bucket = hom_bucket[
                hom_bucket["adversary_model_short"].eq(adversary_model_short)
                & hom_bucket["adversary_elo"].eq(adversary_elo)
            ].iloc[0]
            rows.append(
                {
                    "run_key": run_key,
                    "scenario": "homogeneous_adversary",
                    "game_label": game_label,
                    "n_agents": n_agents,
                    "competition_band": competition_band,
                    "competition_ci": competition_ci,
                    "competition_label_ci": competition_label_ci,
                    "strength_elo": adversary_elo,
                    "bucket_code": int(bucket["bucket_code"]),
                    "bucket_x": float(bucket["bucket_x"]),
                    "bucket_label": str(bucket["bucket_label"]),
                    "bucket_detail": str(bucket["bucket_detail"]),
                    "high_role_payoff": float(adv["final_utility"].mean()),
                    "low_role_payoff": float(base["final_utility"].mean()),
                    "high_role_count": int(len(adv)),
                    "low_role_count": int(len(base)),
                }
            )

        if experiment_family == "heterogeneous_random":
            max_elo = float(group["elo"].max())
            max_rows = group[group["elo"].eq(max_elo)]
            nonmax_rows = group[~group["elo"].eq(max_elo)]
            if max_rows.empty or nonmax_rows.empty:
                continue
            rows.append(
                {
                    "run_key": run_key,
                    "scenario": "heterogeneous_max",
                    "game_label": game_label,
                    "n_agents": n_agents,
                    "competition_band": competition_band,
                    "competition_ci": competition_ci,
                    "competition_label_ci": competition_label_ci,
                    "strength_elo": max_elo,
                    "max_roster_elo": max_elo,
                    "bucket_code": math.nan,
                    "bucket_x": math.nan,
                    "bucket_label": "",
                    "bucket_detail": "",
                    "high_role_payoff": float(max_rows["final_utility"].mean()),
                    "low_role_payoff": float(nonmax_rows["final_utility"].mean()),
                    "high_role_count": int(len(max_rows)),
                    "low_role_count": int(len(nonmax_rows)),
                }
            )

    run_rows = pd.DataFrame(rows)
    max_buckets = heterogeneous_max_bucket_table(run_rows)
    hetero_mask = run_rows["scenario"].eq("heterogeneous_max")
    codes = pd.qcut(run_rows.loc[hetero_mask, "max_roster_elo"].rank(method="first"), q=4, labels=False).astype(int)
    run_rows.loc[hetero_mask, "bucket_code"] = codes.to_numpy(dtype=int)
    run_rows = run_rows.merge(
        max_buckets.rename(
            columns={
                "bucket_x": "hetero_bucket_x",
                "bucket_label": "hetero_bucket_label",
                "bucket_detail": "hetero_bucket_detail",
            }
        ),
        on="bucket_code",
        how="left",
    )
    run_rows.loc[hetero_mask, "bucket_x"] = run_rows.loc[hetero_mask, "hetero_bucket_x"]
    run_rows.loc[hetero_mask, "bucket_label"] = run_rows.loc[hetero_mask, "hetero_bucket_label"]
    run_rows.loc[hetero_mask, "bucket_detail"] = run_rows.loc[hetero_mask, "hetero_bucket_detail"].fillna("")
    run_rows = run_rows.drop(columns=["hetero_bucket_x", "hetero_bucket_label", "hetero_bucket_detail"])
    run_rows["bucket_code"] = run_rows["bucket_code"].astype(int)
    run_rows["bucket_sort"] = run_rows["bucket_code"]
    return run_rows


def filter_frame(frame: pd.DataFrame, filters: dict[str, object]) -> pd.DataFrame:
    out = frame
    for col, value in filters.items():
        out = out[out[col].eq(value)]
    return out


def summarize_curve(frame: pd.DataFrame, scenario: str, filters: dict[str, object]) -> pd.DataFrame:
    sub = filter_frame(frame[frame["scenario"].eq(scenario)], filters)
    rows: list[dict[str, object]] = []
    buckets = (
        frame[frame["scenario"].eq(scenario)][["bucket_code", "bucket_x", "bucket_label"]]
        .drop_duplicates()
        .sort_values("bucket_code")
    )
    for bucket in buckets.itertuples(index=False):
        bucket_sub = sub[sub["bucket_code"].eq(int(bucket.bucket_code))]
        row = {
            "scenario": scenario,
            "bucket_code": int(bucket.bucket_code),
            "bucket_x": float(bucket.bucket_x),
            "bucket_label": str(bucket.bucket_label),
            "n_runs": int(len(bucket_sub)),
            "high_role_payoff_mean": float(bucket_sub["high_role_payoff"].mean()) if len(bucket_sub) else math.nan,
            "high_role_payoff_sem": sem(bucket_sub["high_role_payoff"]) if len(bucket_sub) else math.nan,
            "low_role_payoff_mean": float(bucket_sub["low_role_payoff"].mean()) if len(bucket_sub) else math.nan,
            "low_role_payoff_sem": sem(bucket_sub["low_role_payoff"]) if len(bucket_sub) else math.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_cells(
    frame: pd.DataFrame,
    scenario: str,
    row_groups: list[tuple[str, dict[str, object], dict[str, object]]],
    col_groups: list[tuple[str, dict[str, object], dict[str, object]]],
    scope: str,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for row_label, row_filter, row_extra in row_groups:
        for col_label, col_filter, col_extra in col_groups:
            filters = {**row_filter, **col_filter}
            summary = summarize_curve(frame, scenario, filters)
            summary["scope"] = scope
            summary["row_label"] = row_label
            summary["col_label"] = col_label
            for key, value in {**row_extra, **col_extra}.items():
                summary[key] = value
            pieces.append(summary)
    return pd.concat(pieces, ignore_index=True)


def y_max_for_summary(summary: pd.DataFrame) -> float:
    values = []
    for prefix in ["high_role_payoff", "low_role_payoff"]:
        means = summary[f"{prefix}_mean"].to_numpy(dtype=float)
        errors = np.nan_to_num(summary[f"{prefix}_sem"].to_numpy(dtype=float), nan=0.0)
        finite = np.isfinite(means)
        if finite.any():
            values.append(np.nanmax(means[finite] + errors[finite]))
    if not values:
        return 1.0
    return float(max(values) * 1.15)


def draw_curve_axis(ax: plt.Axes, summary: pd.DataFrame, scenario: str, y_max: float) -> None:
    cfg = SCENARIOS[scenario]
    summary = summary.sort_values("bucket_code")
    x = np.arange(len(summary))
    labels = summary["bucket_label"].tolist()
    high_mean = summary["high_role_payoff_mean"].to_numpy(dtype=float)
    low_mean = summary["low_role_payoff_mean"].to_numpy(dtype=float)
    high_sem = np.nan_to_num(summary["high_role_payoff_sem"].to_numpy(dtype=float), nan=0.0)
    low_sem = np.nan_to_num(summary["low_role_payoff_sem"].to_numpy(dtype=float), nan=0.0)
    valid_high = np.isfinite(high_mean)
    valid_low = np.isfinite(low_mean)
    ax.errorbar(
        x[valid_high],
        high_mean[valid_high],
        yerr=high_sem[valid_high],
        color=cfg["high_color"],
        marker="o",
        markersize=4.5,
        linewidth=1.9,
        capsize=2.5,
        label=cfg["high_role"],
    )
    ax.errorbar(
        x[valid_low],
        low_mean[valid_low],
        yerr=low_sem[valid_low],
        color=cfg["low_color"],
        marker="o",
        markersize=4.5,
        linewidth=1.9,
        capsize=2.5,
        label=cfg["low_role"],
    )
    ax.set_ylim(0, y_max)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.55)
    ax.tick_params(axis="y", labelsize=7.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_grid(
    frame: pd.DataFrame,
    scenario: str,
    row_groups: list[tuple[str, dict[str, object], dict[str, object]]],
    col_groups: list[tuple[str, dict[str, object], dict[str, object]]],
    scope: str,
    title: str,
    filename: str,
    figsize: tuple[float, float],
) -> tuple[Path, pd.DataFrame]:
    summary = summarize_cells(frame, scenario, row_groups, col_groups, scope)
    fig, axes = plt.subplots(len(row_groups), len(col_groups), figsize=figsize, sharey=True, squeeze=False)
    y_max = y_max_for_summary(summary)
    for row_idx, (row_label, _, _) in enumerate(row_groups):
        for col_idx, (col_label, _, _) in enumerate(col_groups):
            ax = axes[row_idx, col_idx]
            cell = summary[summary["row_label"].eq(row_label) & summary["col_label"].eq(col_label)]
            draw_curve_axis(ax, cell, scenario, y_max)
            if row_idx == 0:
                ax.set_title(col_label, fontsize=9, pad=5)
            if col_idx == 0:
                ax.set_ylabel(f"{row_label}\nPayoff" if len(row_groups) > 1 else "Payoff", fontsize=8)
            if row_idx != len(row_groups) - 1:
                ax.tick_params(axis="x", labelbottom=False)
            else:
                ax.set_xlabel(SCENARIOS[scenario]["x_label"], fontsize=8)
    handles = [
        Line2D([0], [0], color=SCENARIOS[scenario]["high_color"], marker="o", lw=1.9, label=SCENARIOS[scenario]["high_role"]),
        Line2D([0], [0], color=SCENARIOS[scenario]["low_color"], marker="o", lw=1.9, label=SCENARIOS[scenario]["low_role"]),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out_path = OUT_DIR / filename
    fig.savefig(out_path, dpi=210, bbox_inches="tight")
    plt.close(fig)
    return out_path, summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_rows = compute_run_rows(load_agents())

    n_cols = [(f"N={n}", {"n_agents": n}, {"n_agents": n}) for n in N_ORDER]
    game_cols = [(GAME_TITLES[g], {"game_label": g}, {"game_label": g}) for g in GAME_ORDER]
    comp_cols = [(COMPETITION_TITLES[c], {"competition_band": c}, {"competition_band": c}) for c in COMPETITION_ORDER]

    plot_specs = [
        (
            [("All", {}, {})],
            [("All", {}, {})],
            "overall",
            "overall",
            (5.8, 4.0),
        ),
        (
            [("All", {}, {})],
            n_cols,
            "by_n",
            "broken down by N",
            (18.0, 4.0),
        ),
        (
            [("All", {}, {})],
            comp_cols,
            "by_competition",
            "broken down by competition band",
            (11.0, 4.0),
        ),
        (
            [("All", {}, {})],
            game_cols,
            "by_game",
            "broken down by game",
            (11.0, 4.0),
        ),
        (
            game_cols,
            comp_cols,
            "by_game_competition",
            "broken down by game and competition band",
            (11.0, 8.0),
        ),
    ]

    plot_paths: list[Path] = []
    summaries: list[pd.DataFrame] = []
    for scenario, cfg in SCENARIOS.items():
        for row_groups, col_groups, scope, suffix_title, figsize in plot_specs:
            filename_suffix = "" if scope == "overall" else f"_{scope}"
            path, summary = plot_grid(
                run_rows,
                scenario,
                row_groups,
                col_groups,
                scope,
                f"{cfg['title']}: role payoff curves {suffix_title}",
                f"{cfg['file_prefix']}{filename_suffix}.png",
                figsize,
            )
            plot_paths.append(path)
            summaries.append(summary)

        for n in N_ORDER:
            path, summary = plot_grid(
                filter_frame(run_rows, {"n_agents": n}),
                scenario,
                game_cols,
                comp_cols,
                f"by_n_game_competition_n{n}",
                f"{cfg['title']}: role payoff curves by game and competition band for N={n}",
                f"{cfg['file_prefix']}_by_n{n}_game_competition.png",
                (11.0, 8.0),
            )
            plot_paths.append(path)
            summaries.append(summary)

    run_path = OUT_DIR / "role_payoff_curves_by_strength_run_metrics.csv"
    summary_path = OUT_DIR / "role_payoff_curves_by_strength_summary.csv"
    run_rows.to_csv(run_path, index=False)
    pd.concat(summaries, ignore_index=True).to_csv(summary_path, index=False)

    print(f"Wrote {run_path}")
    print(f"Wrote {summary_path}")
    for path in plot_paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()

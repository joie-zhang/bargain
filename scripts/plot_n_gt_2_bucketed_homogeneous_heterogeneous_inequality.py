#!/usr/bin/env python3
"""Bucketed inequality comparisons for homogeneous and heterogeneous N>2 runs."""

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

SUMMARY_METRICS = [
    ("payoff_variance", "Payoff variance", "{:.1f}"),
    ("payoff_gini_corrected", "Corrected payoff Gini", "{:.3f}"),
    ("mean_payoff", "Average payoff", "{:.1f}"),
]

PLOT_METRICS = [
    ("payoff_variance", "Payoff variance", "{:.1f}"),
    ("mean_payoff", "Average payoff", "{:.1f}"),
]

SECTION_COLORS = {
    "overall": "#4E79A7",
    "hom_adv_elo": "#F28E2B",
    "hetero_mean_elo": "#8E63B0",
    "hetero_max_elo": "#59A14F",
}

HETERO_BUCKET_COUNT = 8
MIN_PLOT_BUCKET_RUNS = 10


def gini_shifted_corrected(values: pd.Series | np.ndarray) -> tuple[float, float, bool]:
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
    frames = [pd.read_csv(path) for path in AGENT_FILES]
    agents = pd.concat(frames, ignore_index=True)
    agents["final_utility"] = pd.to_numeric(agents["final_utility"], errors="coerce")
    agents["elo"] = pd.to_numeric(agents["elo"], errors="coerce")
    agents["n_agents"] = pd.to_numeric(agents["n_agents"], errors="coerce")
    agents = agents.dropna(subset=["run_key", "final_utility", "elo", "n_agents"]).copy()
    agents["n_agents"] = agents["n_agents"].astype(int)
    return agents


def compute_run_metrics(agents: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_key, group in agents.groupby("run_key", sort=False):
        utilities = group["final_utility"].to_numpy(dtype=float)
        raw_gini, corrected_gini, shifted = gini_shifted_corrected(utilities)
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
        if adversary_rows.empty:
            adversary_elo = math.nan
            adversary_model_short = ""
        else:
            adversary_elo = float(adversary_rows["elo"].iloc[0])
            adversary_model_short = str(adversary_rows["model_short"].iloc[0])

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
                "mean_roster_elo": float(np.mean(elos)),
                "min_roster_elo": float(np.min(elos)),
                "max_roster_elo": float(np.max(elos)),
                "elo_std": float(np.std(elos, ddof=0)),
                "elo_variance": float(np.var(elos, ddof=0)),
                "adversary_elo": adversary_elo,
                "adversary_model_short": adversary_model_short,
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
    return pd.DataFrame(rows)


def fixed_bin_label(prefix: str, code: int, edges: np.ndarray) -> str:
    return f"{prefix} B{code + 1}\n{short_float(float(edges[code]))}-{short_float(float(edges[code + 1]))}"


def heterogeneous_common_edges(runs: pd.DataFrame) -> np.ndarray:
    hetero = runs[runs["comparison_group"].eq("heterogeneous")]
    low = float(min(hetero["mean_roster_elo"].min(), hetero["max_roster_elo"].min()))
    high = float(max(hetero["mean_roster_elo"].max(), hetero["max_roster_elo"].max()))
    return np.linspace(low, high, HETERO_BUCKET_COUNT + 1)


def assign_homogeneous_adversary_bins(runs: pd.DataFrame) -> pd.DataFrame:
    hom_adv = runs[runs["comparison_group"].eq("homogeneous_adversary")].copy()
    unique_adv = (
        hom_adv[["adversary_model_short", "adversary_elo"]]
        .drop_duplicates()
        .sort_values("adversary_elo")
        .reset_index(drop=True)
    )
    # There are only five homogeneous-adversary models, so use the finest possible bins:
    # one adversary model per bar.
    unique_adv["bucket_code"] = np.arange(len(unique_adv), dtype=int)
    code_to_label: dict[int, str] = {}
    for code, sub in unique_adv.groupby("bucket_code"):
        elo_min = float(sub["adversary_elo"].min())
        elo_max = float(sub["adversary_elo"].max())
        names = ", ".join(sub["adversary_model_short"].tolist())
        code_to_label[int(code)] = f"Hom adv\n{short_float(elo_min)}\n{names}"
    out = hom_adv.merge(unique_adv, on=["adversary_model_short", "adversary_elo"], how="left")
    out["bucket_section"] = "hom_adv_elo"
    out["bucket_code"] = out["bucket_code"].astype(int)
    out["bucket_label"] = out["bucket_code"].map(code_to_label)
    out["bucket_sort"] = 10 + out["bucket_code"]
    return out


def assign_fixed_elo_bins(
    runs: pd.DataFrame,
    value_col: str,
    section: str,
    prefix: str,
    sort_offset: int,
    edges: np.ndarray,
) -> pd.DataFrame:
    subset = runs[runs["comparison_group"].eq("heterogeneous")].copy()
    codes = np.searchsorted(edges, subset[value_col].to_numpy(dtype=float), side="right") - 1
    codes = np.clip(codes, 0, HETERO_BUCKET_COUNT - 1)
    subset["bucket_code"] = codes.astype(int)
    labels = {code: fixed_bin_label(prefix, code, edges) for code in range(HETERO_BUCKET_COUNT)}
    subset["bucket_section"] = section
    subset["bucket_label"] = subset["bucket_code"].map(labels)
    subset["bucket_sort"] = sort_offset + subset["bucket_code"]
    return subset


def build_bar_rows(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pieces: list[pd.DataFrame] = []
    specs: list[dict[str, object]] = []
    hetero_edges = heterogeneous_common_edges(runs)

    overall_specs = [
        ("Heterogeneous\nall", "overall", 0, "heterogeneous"),
        ("Homogeneous adversary\nall", "overall", 1, "homogeneous_adversary"),
        ("Hom control\nGPT-5-nano", "overall", 2, "homogeneous_control"),
    ]
    for label, section, sort_key, group_name in overall_specs:
        sub = runs[runs["comparison_group"].eq(group_name)].copy()
        sub["bucket_label"] = label
        sub["bucket_section"] = section
        sub["bucket_sort"] = sort_key
        pieces.append(sub)
        specs.append({"bucket_section": section, "bucket_sort": sort_key, "bucket_label": label})

    hom_adv = assign_homogeneous_adversary_bins(runs)
    pieces.append(hom_adv)
    specs.extend(
        hom_adv[["bucket_section", "bucket_sort", "bucket_label"]]
        .drop_duplicates()
        .to_dict(orient="records")
    )
    pieces.append(
        assign_fixed_elo_bins(
            runs,
            value_col="mean_roster_elo",
            section="hetero_mean_elo",
            prefix="Hetero mean Elo",
            sort_offset=20,
            edges=hetero_edges,
        )
    )
    pieces.append(
        assign_fixed_elo_bins(
            runs,
            value_col="max_roster_elo",
            section="hetero_max_elo",
            prefix="Hetero max Elo",
            sort_offset=30,
            edges=hetero_edges,
        )
    )
    for section, prefix, offset in [
        ("hetero_mean_elo", "Hetero mean Elo", 20),
        ("hetero_max_elo", "Hetero max Elo", 30),
    ]:
        for code in range(HETERO_BUCKET_COUNT):
            specs.append(
                {
                    "bucket_section": section,
                    "bucket_sort": offset + code,
                    "bucket_label": fixed_bin_label(prefix, code, hetero_edges),
                }
            )

    spec_frame = (
        pd.DataFrame(specs)
        .drop_duplicates(subset=["bucket_section", "bucket_sort"], keep="last")
        .sort_values("bucket_sort")
        .reset_index(drop=True)
    )
    return pd.concat(pieces, ignore_index=True), spec_frame


def summarize_bars(bar_rows: pd.DataFrame, bucket_specs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in bucket_specs.sort_values("bucket_sort").itertuples(index=False):
        section = str(spec.bucket_section)
        sort_key = int(spec.bucket_sort)
        label = str(spec.bucket_label)
        sub = bar_rows[
            bar_rows["bucket_section"].eq(section)
            & bar_rows["bucket_sort"].eq(sort_key)
        ]
        row: dict[str, object] = {
            "bucket_section": section,
            "bucket_sort": sort_key,
            "bucket_label": label.replace("\n", " | "),
            "n_runs": int(len(sub)),
            "below_min_plot_runs": bool(len(sub) < MIN_PLOT_BUCKET_RUNS),
            "mean_roster_elo_mean": float(sub["mean_roster_elo"].mean()) if len(sub) else math.nan,
            "mean_roster_elo_min": float(sub["mean_roster_elo"].min()) if len(sub) else math.nan,
            "mean_roster_elo_max": float(sub["mean_roster_elo"].max()) if len(sub) else math.nan,
            "max_roster_elo_mean": float(sub["max_roster_elo"].mean()) if len(sub) else math.nan,
            "max_roster_elo_min": float(sub["max_roster_elo"].min()) if len(sub) else math.nan,
            "max_roster_elo_max": float(sub["max_roster_elo"].max()) if len(sub) else math.nan,
            "elo_std_mean": float(sub["elo_std"].mean()) if len(sub) else math.nan,
            "adversary_elo_mean": float(sub["adversary_elo"].mean())
            if len(sub) and sub["adversary_elo"].notna().any()
            else math.nan,
            "model_list": " / ".join(sorted(sub["adversary_model_short"].dropna().unique()))
            if len(sub) and section == "hom_adv_elo"
            else "",
        }
        for metric, _, _ in SUMMARY_METRICS:
            row[f"{metric}_mean"] = float(sub[metric].mean()) if len(sub) else math.nan
            row[f"{metric}_sem"] = sem(sub[metric]) if len(sub) else math.nan
            row[f"{metric}_median"] = float(sub[metric].median()) if len(sub) else math.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("bucket_sort")


def make_bucket_bar_plot(summary: pd.DataFrame, filename: str) -> Path:
    summary = summary.sort_values("bucket_sort").reset_index(drop=True)
    labels = summary["bucket_label"].str.replace(" \\| ", "\n", regex=True).tolist()
    x = np.arange(len(summary))
    colors = [SECTION_COLORS[section] for section in summary["bucket_section"]]

    fig, axes = plt.subplots(2, 1, figsize=(31.0, 10.5), sharex=True)
    for ax, (metric, ylabel, fmt) in zip(axes, PLOT_METRICS, strict=True):
        means = summary[f"{metric}_mean"].to_numpy(dtype=float)
        errors = summary[f"{metric}_sem"].to_numpy(dtype=float)
        valid = summary["n_runs"].ge(MIN_PLOT_BUCKET_RUNS).to_numpy() & np.isfinite(means)
        valid_idx = np.where(valid)[0]
        ax.bar(
            x[valid],
            means[valid],
            yerr=np.nan_to_num(errors[valid], nan=0.0),
            capsize=3,
            color=[colors[i] for i in valid_idx],
            alpha=0.86,
            edgecolor="white",
            linewidth=0.7,
        )
        y_candidates = means[valid] + np.nan_to_num(errors[valid], nan=0.0)
        y_max = float(np.nanmax(y_candidates)) if len(y_candidates) else 1.0
        ax.set_ylim(0, y_max * 1.20)
        for xi, mean, n_runs in zip(x, means, summary["n_runs"], strict=True):
            if int(n_runs) < MIN_PLOT_BUCKET_RUNS or not np.isfinite(mean):
                ax.text(
                    xi,
                    y_max * 0.035,
                    f"n={int(n_runs)}\n<10",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#666666",
                )
            else:
                ax.text(xi, mean, f"{fmt.format(mean)}\nn={int(n_runs)}", ha="center", va="bottom", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.grid(True, axis="y", alpha=0.24)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    section_names = {
        "overall": "Main groups",
        "hom_adv_elo": "Homogeneous adversary by adversary Elo",
        "hetero_mean_elo": "Heterogeneous by mean roster Elo",
        "hetero_max_elo": "Heterogeneous by max roster Elo",
    }
    for ax in axes:
        section_starts = (
            summary.reset_index()
            .groupby("bucket_section", sort=False)["index"]
            .agg(["min", "max"])
            .reset_index()
        )
        for boundary in section_starts["max"].iloc[:-1]:
            boundary = float(boundary) + 0.5
            ax.axvline(boundary, color="#777777", alpha=0.25, linewidth=1.0)
        for section, sub in summary.groupby("bucket_section", sort=False):
            left = int(sub.index.min())
            right = int(sub.index.max())
            ax.text(
                (left + right) / 2,
                1.03,
                section_names[section],
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=10,
                color="#333333",
                clip_on=False,
            )

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=47, ha="right", fontsize=7)
    fig.suptitle("Payoff dispersion and average payoff by roster strength bucket", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=2.2)

    out_path = OUT_DIR / filename
    fig.savefig(out_path, dpi=230, bbox_inches="tight")
    plt.close(fig)
    return out_path


def fit_metric_trends(runs: pd.DataFrame) -> pd.DataFrame:
    trend_specs = [
        (
            "homogeneous_adversary_by_adversary_elo",
            runs[runs["comparison_group"].eq("homogeneous_adversary")],
            "adversary_elo",
        ),
        (
            "heterogeneous_by_mean_roster_elo",
            runs[runs["comparison_group"].eq("heterogeneous")],
            "mean_roster_elo",
        ),
        (
            "heterogeneous_by_max_roster_elo",
            runs[runs["comparison_group"].eq("heterogeneous")],
            "max_roster_elo",
        ),
        (
            "heterogeneous_by_elo_std",
            runs[runs["comparison_group"].eq("heterogeneous")],
            "elo_std",
        ),
    ]
    rows: list[dict[str, object]] = []
    for trend_name, frame, x_col in trend_specs:
        for metric, _, _ in SUMMARY_METRICS:
            sub = frame[[x_col, metric]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(sub) < 3 or sub[x_col].nunique() < 2:
                continue
            fit = stats.linregress(sub[x_col], sub[metric])
            rows.append(
                {
                    "trend": trend_name,
                    "x_col": x_col,
                    "metric": metric,
                    "n_runs": int(len(sub)),
                    "x_min": float(sub[x_col].min()),
                    "x_max": float(sub[x_col].max()),
                    "x_mean": float(sub[x_col].mean()),
                    "metric_mean": float(sub[metric].mean()),
                    "slope": float(fit.slope),
                    "slope_per_100_x": float(fit.slope * 100.0),
                    "intercept": float(fit.intercept),
                    "r": float(fit.rvalue),
                    "r_squared": float(fit.rvalue**2),
                    "p_value": float(fit.pvalue),
                    "stderr": float(fit.stderr),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    agents = load_agents()
    runs = compute_run_metrics(agents)
    bar_rows, bucket_specs = build_bar_rows(runs)
    summary = summarize_bars(bar_rows, bucket_specs)
    trends = fit_metric_trends(runs)

    run_metrics_path = OUT_DIR / "homogeneous_heterogeneous_bucketed_8bins_inequality_run_metrics.csv"
    summary_path = OUT_DIR / "homogeneous_heterogeneous_bucketed_8bins_inequality_summary.csv"
    trend_path = OUT_DIR / "homogeneous_heterogeneous_bucketed_8bins_inequality_trends.csv"
    plot_path = make_bucket_bar_plot(summary, "homogeneous_heterogeneous_bucketed_8bins_variance_mean_payoff_bars.png")
    base_plot_path = make_bucket_bar_plot(summary, "homogeneous_heterogeneous_bucketed_variance_mean_payoff_bars.png")

    runs.to_csv(run_metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    trends.to_csv(trend_path, index=False)

    print(f"Wrote {run_metrics_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {trend_path}")
    print(f"Wrote {plot_path}")
    print(f"Wrote {base_plot_path}")
    print()
    print(summary[[
        "bucket_label",
        "n_runs",
        "mean_roster_elo_mean",
        "max_roster_elo_mean",
        "elo_std_mean",
        "payoff_variance_mean",
        "payoff_gini_corrected_mean",
    ]].to_string(index=False))
    print()
    print(trends.to_string(index=False))


if __name__ == "__main__":
    main()

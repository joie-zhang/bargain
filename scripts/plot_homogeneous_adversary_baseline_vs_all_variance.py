#!/usr/bin/env python3
"""Compare all-agent and baseline-only payoff variance in homogeneous adversary runs."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (
    PROJECT_ROOT
    / "experiments/results/n2_plus_multiagent_comparison_analysis_20260505"
    / "tables_multiagent/homogeneous_agents_fresh.csv"
)
OUT_DIR = PROJECT_ROOT / "overleaf/neurips/graphics/n_gt_2_report"
N_ORDER = [2, 4, 6, 8, 10]


def short_float(value: float) -> str:
    if math.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:.1f}"


def sem(values: pd.Series) -> float:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return math.nan
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


def load_agents() -> pd.DataFrame:
    agents = pd.read_csv(DATA_PATH)
    for col in ["final_utility", "elo", "n_agents", "competition_ci"]:
        agents[col] = pd.to_numeric(agents[col], errors="coerce")
    agents = agents.dropna(subset=["run_key", "final_utility", "elo", "n_agents"]).copy()
    agents["n_agents"] = agents["n_agents"].astype(int)
    return agents


def adversary_bucket_map(agents: pd.DataFrame) -> pd.DataFrame:
    hom_adv = agents[
        agents["experiment_family"].astype(str).str.startswith("homogeneous")
        & agents["experiment_family"].ne("homogeneous_control")
        & agents["role"].astype(str).eq("adversary")
    ]
    unique_adv = (
        hom_adv[["model_short", "elo"]]
        .drop_duplicates()
        .sort_values(["elo", "model_short"])
        .rename(columns={"model_short": "adversary_model_short", "elo": "adversary_elo"})
        .reset_index(drop=True)
    )
    unique_adv["bucket_code"] = np.minimum(np.floor(np.arange(len(unique_adv)) * 4 / len(unique_adv)).astype(int), 3)
    labels = []
    for code, sub in unique_adv.groupby("bucket_code"):
        labels.append(
            {
                "bucket_code": int(code),
                "bucket_x": float(sub["adversary_elo"].mean()),
                "bucket_label": (
                    f"Q{int(code) + 1}\n"
                    f"{short_float(float(sub['adversary_elo'].min()))}-{short_float(float(sub['adversary_elo'].max()))}"
                ),
                "bucket_detail": ", ".join(sub["adversary_model_short"].tolist()),
            }
        )
    return unique_adv.merge(pd.DataFrame(labels), on="bucket_code", how="left")


def compute_run_metrics(agents: pd.DataFrame) -> pd.DataFrame:
    bucket_map = adversary_bucket_map(agents)
    hom_adv = agents[
        agents["experiment_family"].astype(str).str.startswith("homogeneous")
        & agents["experiment_family"].ne("homogeneous_control")
    ].copy()
    rows: list[dict[str, object]] = []
    for run_key, group in hom_adv.groupby("run_key", sort=False):
        adv = group[group["role"].astype(str).eq("adversary")]
        base = group[~group["role"].astype(str).eq("adversary")]
        if adv.empty or base.empty:
            continue

        adversary_elo = float(adv["elo"].iloc[0])
        adversary_model_short = str(adv["model_short"].iloc[0])
        bucket = bucket_map[
            bucket_map["adversary_model_short"].eq(adversary_model_short)
            & bucket_map["adversary_elo"].eq(adversary_elo)
        ].iloc[0]

        all_utilities = group["final_utility"].to_numpy(dtype=float)
        baseline_utilities = base["final_utility"].to_numpy(dtype=float)
        adversary_payoff = float(adv["final_utility"].mean())
        baseline_payoff = float(base["final_utility"].mean())
        n_agents = int(group["n_agents"].iloc[0])
        n_baseline = len(base)
        baseline_variance = float(np.var(baseline_utilities, ddof=0))
        all_variance = float(np.var(all_utilities, ddof=0))
        between_role_component = (n_baseline / (n_agents * n_agents)) * (adversary_payoff - baseline_payoff) ** 2
        baseline_component = (n_baseline / n_agents) * baseline_variance

        rows.append(
            {
                "run_key": run_key,
                "game_label": str(group["game_label"].iloc[0]),
                "n_agents": n_agents,
                "competition_band": str(group["competition_band"].iloc[0]),
                "competition_ci": float(group["competition_ci"].iloc[0]),
                "adversary_model_short": adversary_model_short,
                "adversary_elo": adversary_elo,
                "bucket_code": int(bucket["bucket_code"]),
                "bucket_x": float(bucket["bucket_x"]),
                "bucket_label": str(bucket["bucket_label"]),
                "bucket_detail": str(bucket["bucket_detail"]),
                "all_agent_payoff_variance": all_variance,
                "baseline_only_payoff_variance": baseline_variance,
                "baseline_component_of_all_variance": baseline_component,
                "between_role_component_of_all_variance": between_role_component,
                "adversary_payoff": adversary_payoff,
                "baseline_mean_payoff": baseline_payoff,
            }
        )
    return pd.DataFrame(rows)


def summarize(runs: pd.DataFrame, filters: dict[str, object] | None = None) -> pd.DataFrame:
    if filters is None:
        filters = {}
    sub = runs.copy()
    for col, value in filters.items():
        sub = sub[sub[col].eq(value)]

    rows: list[dict[str, object]] = []
    bucket_table = runs[["bucket_code", "bucket_label", "bucket_x"]].drop_duplicates().sort_values("bucket_code")
    for bucket in bucket_table.itertuples(index=False):
        bucket_sub = sub[sub["bucket_code"].eq(int(bucket.bucket_code))]
        row = {
            "bucket_code": int(bucket.bucket_code),
            "bucket_label": str(bucket.bucket_label),
            "bucket_x": float(bucket.bucket_x),
            "n_runs": int(bucket_sub["run_key"].nunique()),
        }
        for metric in [
            "all_agent_payoff_variance",
            "baseline_only_payoff_variance",
            "baseline_component_of_all_variance",
            "between_role_component_of_all_variance",
            "adversary_payoff",
            "baseline_mean_payoff",
        ]:
            row[f"{metric}_mean"] = float(bucket_sub[metric].mean()) if len(bucket_sub) else math.nan
            row[f"{metric}_sem"] = sem(bucket_sub[metric]) if len(bucket_sub) else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def draw_grouped_variance_plot(summary: pd.DataFrame, title: str, out_path: Path, figsize: tuple[float, float]) -> None:
    summary = summary.sort_values("bucket_code").reset_index(drop=True)
    x = np.arange(len(summary))
    width = 0.34
    all_mean = summary["all_agent_payoff_variance_mean"].to_numpy(dtype=float)
    base_mean = summary["baseline_only_payoff_variance_mean"].to_numpy(dtype=float)
    all_sem = np.nan_to_num(summary["all_agent_payoff_variance_sem"].to_numpy(dtype=float), nan=0.0)
    base_sem = np.nan_to_num(summary["baseline_only_payoff_variance_sem"].to_numpy(dtype=float), nan=0.0)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(
        x - width / 2,
        all_mean,
        width=width,
        yerr=all_sem,
        capsize=3,
        color="#D95F02",
        alpha=0.86,
        label="All agents",
    )
    ax.bar(
        x + width / 2,
        base_mean,
        width=width,
        yerr=base_sem,
        capsize=3,
        color="#4E79A7",
        alpha=0.86,
        label="Baseline agents only",
    )
    for xpos, value in zip(x - width / 2, all_mean, strict=False):
        if np.isfinite(value):
            ax.text(xpos, value + 3, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    for xpos, value in zip(x + width / 2, base_mean, strict=False):
        if np.isfinite(value):
            ax.text(xpos, value + 3, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["bucket_label"].tolist(), fontsize=9)
    ax.set_xlabel("Adversary Elo bucket", fontsize=10)
    ax.set_ylabel("Mean payoff variance", fontsize=10)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_ylim(0, max(np.nanmax(all_mean + all_sem), np.nanmax(base_mean + base_sem)) * 1.2)
    ax.grid(True, axis="y", alpha=0.24, linewidth=0.6)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_decomposition_plot(summary: pd.DataFrame, out_path: Path) -> None:
    summary = summary.sort_values("bucket_code").reset_index(drop=True)
    x = np.arange(len(summary))
    baseline_component = summary["baseline_component_of_all_variance_mean"].to_numpy(dtype=float)
    between_component = summary["between_role_component_of_all_variance_mean"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(5.4, 5.8))
    ax.bar(
        x,
        baseline_component,
        color="#4E79A7",
        alpha=0.88,
        label="Baseline spread contribution",
    )
    ax.bar(
        x,
        between_component,
        bottom=baseline_component,
        color="#D95F02",
        alpha=0.88,
        label="Adversary-baseline separation",
    )
    totals = baseline_component + between_component
    for xpos, value in zip(x, totals, strict=False):
        ax.text(xpos, value + 3, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["bucket_label"].tolist(), fontsize=9)
    ax.set_xlabel("Adversary Elo bucket", fontsize=10)
    ax.set_ylabel("Contribution to all-agent variance", fontsize=10)
    ax.set_title("Homogeneous adversary: decomposition of all-agent payoff variance", fontsize=12, pad=10)
    ax.set_ylim(0, np.nanmax(totals) * 1.22)
    ax.grid(True, axis="y", alpha=0.24, linewidth=0.6)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_by_n_plot(runs: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    summaries = []
    fig, axes = plt.subplots(len(N_ORDER), 1, figsize=(5.8, 12.0), sharex=True)
    for ax, n_agents in zip(axes, N_ORDER, strict=True):
        summary = summarize(runs, {"n_agents": n_agents})
        summary["n_agents"] = n_agents
        summaries.append(summary)
        summary = summary.sort_values("bucket_code").reset_index(drop=True)
        x = np.arange(len(summary))
        width = 0.34
        all_mean = summary["all_agent_payoff_variance_mean"].to_numpy(dtype=float)
        base_mean = summary["baseline_only_payoff_variance_mean"].to_numpy(dtype=float)
        ax.bar(x - width / 2, all_mean, width=width, color="#D95F02", alpha=0.86)
        ax.bar(x + width / 2, base_mean, width=width, color="#4E79A7", alpha=0.86)
        ax.set_ylabel(f"N={n_agents}\nvariance", fontsize=8.5)
        ax.grid(True, axis="y", alpha=0.24, linewidth=0.6)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", labelsize=8)
    axes[-1].set_xticks(np.arange(4))
    axes[-1].set_xticklabels(summaries[0]["bucket_label"].tolist(), fontsize=9)
    axes[-1].set_xlabel("Adversary Elo bucket", fontsize=10)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#D95F02", alpha=0.86, label="All agents"),
        plt.Rectangle((0, 0), 1, 1, color="#4E79A7", alpha=0.86, label="Baseline agents only"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Homogeneous adversary: all-agent vs baseline-only payoff variance by N", fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0.035, 1, 0.985))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pd.concat(summaries, ignore_index=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = compute_run_metrics(load_agents())
    overall = summarize(runs)
    overall["scope"] = "overall"
    by_n = draw_by_n_plot(
        runs,
        OUT_DIR / "homogeneous_adversary_baseline_only_vs_all_payoff_variance_by_n.png",
    )
    by_n["scope"] = "by_n"
    summary_path = OUT_DIR / "homogeneous_adversary_baseline_only_vs_all_payoff_variance_summary.csv"
    run_path = OUT_DIR / "homogeneous_adversary_baseline_only_vs_all_payoff_variance_run_metrics.csv"

    draw_grouped_variance_plot(
        overall,
        "Homogeneous adversary: all-agent vs baseline-only payoff variance",
        OUT_DIR / "homogeneous_adversary_baseline_only_vs_all_payoff_variance.png",
        (5.8, 5.8),
    )
    draw_decomposition_plot(
        overall,
        OUT_DIR / "homogeneous_adversary_all_payoff_variance_decomposition.png",
    )

    runs.to_csv(run_path, index=False)
    pd.concat([overall, by_n], ignore_index=True).to_csv(summary_path, index=False)

    print(f"Wrote {OUT_DIR / 'homogeneous_adversary_baseline_only_vs_all_payoff_variance.png'}")
    print(f"Wrote {OUT_DIR / 'homogeneous_adversary_all_payoff_variance_decomposition.png'}")
    print(f"Wrote {OUT_DIR / 'homogeneous_adversary_baseline_only_vs_all_payoff_variance_by_n.png'}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {run_path}")


if __name__ == "__main__":
    main()

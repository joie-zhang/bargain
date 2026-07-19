#!/usr/bin/env python3
"""Build sanity-check plots for the random-monoculture control sweep."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import full_games123_multiagent_batch as full  # noqa: E402
import random_monoculture_control_batch as rmc  # noqa: E402


DEFAULT_RESULTS_ROOT = (
    PROJECT_ROOT
    / "experiments/results/full_games123_random_monoculture_control_20260628_014357"
)

GAME_ORDER = ["game1", "game2", "game3"]
GAME_TITLES = {
    "game1": "Game 1: item allocation",
    "game2": "Game 2: bargaining/coordination",
    "game3": "Game 3: co-funding",
}
N_ORDER = [2, 4, 6, 8, 10]
N_COLORS = {
    2: "#4E79A7",
    4: "#F28E2B",
    6: "#59A14F",
    8: "#B07AA1",
    10: "#E15759",
}
GAME_COLORS = {
    "game1": "#4E79A7",
    "game2": "#59A14F",
    "game3": "#E15759",
}
MODEL_SHORT_NAMES = {
    "amazon-nova-micro-v1.0": "Nova Micro",
    "amazon-nova-pro-v1.0": "Nova Pro",
    "claude-3-haiku-20240307": "Claude 3 Haiku",
    "claude-opus-4-5-20251101": "Opus 4.5",
    "claude-opus-4-5-20251101-thinking-32k": "Opus 4.5 Think",
    "claude-opus-4-6": "Opus 4.6",
    "deepseek-r1-0528": "DeepSeek R1",
    "deepseek-v3": "DeepSeek V3",
    "gemini-3.1-pro": "Gemini 3.1 Pro",
    "gpt-4o-2024-05-13": "GPT-4o",
    "gpt-5-nano-high": "GPT-5 nano",
    "gpt-5.2-chat-latest-20260210": "GPT-5.2 Chat",
    "gpt-5.4-high": "GPT-5.4 High",
    "o3-mini-high": "o3-mini",
    "qwen3-max-preview": "Qwen3 Max",
}


def short_model_name(model: str) -> str:
    return MODEL_SHORT_NAMES.get(model, model)


def config_id_string(config: dict[str, Any]) -> str:
    return f"config_{rmc.config_number(config['config_id']):04d}"


def gini(values: list[float]) -> float:
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan
    if np.allclose(arr, 0.0):
        return 0.0
    min_val = float(arr.min())
    if min_val < 0:
        arr = arr - min_val
    mean = float(arr.mean())
    if mean <= 0:
        return math.nan
    diff_sum = np.abs(arr[:, None] - arr[None, :]).sum()
    return float(diff_sum / (2 * arr.size * arr.size * mean))


def game_cell_label(config: dict[str, Any]) -> str:
    game = str(config["game_label"])
    if game == "game1":
        return f"competition={float(config.get('competition_level', math.nan)):.2f}"
    if game == "game2":
        rho = float(config.get("rho", math.nan))
        theta = float(config.get("theta", math.nan))
        return f"rho={rho:.2f}, theta={theta:.1f}"
    if game == "game3":
        sigma = float(config.get("sigma", math.nan))
        alpha = float(config.get("alpha", math.nan))
        return f"sigma={sigma:.1f}, alpha={alpha:.1f}"
    return str(config.get("competition_id", "unknown"))


def load_tables(results_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    configs = full.load_configs(results_root)
    rows: list[dict[str, Any]] = []
    success_rows: list[dict[str, Any]] = []

    for config in configs:
        cid = config_id_string(config)
        status_path = results_root / "status" / f"{cid}.json"
        status: dict[str, Any] = {}
        if status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))

        result_path = full.result_path_for(config)
        valid_result = (
            result_path is not None
            and result_path.exists()
            and full.validate_result_file(rmc.runtime_config(config), result_path) is None
        )
        state = "SUCCESS" if valid_result else str(status.get("state") or "NOT_STARTED")
        model = str(config["monoculture_model"])
        row = {
            "config_id": cid,
            "config_number": rmc.config_number(config["config_id"]),
            "game_label": str(config["game_label"]),
            "game_title": GAME_TITLES.get(str(config["game_label"]), str(config["game_label"])),
            "n_agents": int(config["n_agents"]),
            "competition_id": str(config.get("competition_id", "")),
            "cell_label": game_cell_label(config),
            "model": model,
            "model_short": short_model_name(model),
            "model_elo": float(config.get("model_elo", math.nan)),
            "elo_band_index": int(config.get("elo_band_index", -1)),
            "state": state,
            "valid_result": bool(valid_result),
            "result_path": str(result_path) if result_path is not None else "",
            "status_path": str(status_path) if status_path.exists() else "",
            "duration_seconds": status.get("duration_seconds"),
            "returncode": status.get("returncode"),
            "result_validation_error": status.get("result_validation_error"),
            "competition_level": config.get("competition_level"),
            "rho": config.get("rho"),
            "rho_label": config.get("rho_label"),
            "theta": config.get("theta"),
            "sigma": config.get("sigma"),
            "alpha": config.get("alpha"),
        }
        rows.append(row)

        if not valid_result:
            continue
        assert result_path is not None
        result = json.loads(result_path.read_text(encoding="utf-8"))
        final_utilities_raw = result.get("final_utilities") or {}
        final_utilities = [float(v) for v in final_utilities_raw.values()]
        vote_integrity = result.get("vote_integrity") or {}
        qualitative = result.get("qualitative_metrics_v1") or {}
        success_rows.append(
            {
                **row,
                "average_payoff": float(np.mean(final_utilities)) if final_utilities else math.nan,
                "total_payoff": float(np.sum(final_utilities)) if final_utilities else math.nan,
                "payoff_std": float(np.std(final_utilities, ddof=0)) if final_utilities else math.nan,
                "payoff_gini": gini(final_utilities),
                "min_payoff": float(np.min(final_utilities)) if final_utilities else math.nan,
                "max_payoff": float(np.max(final_utilities)) if final_utilities else math.nan,
                "final_round": int(result.get("final_round") or 0),
                "consensus_reached": bool(result.get("consensus_reached")),
                "conversation_log_count": len(result.get("conversation_logs") or []),
                "synthetic_vote_count": int(vote_integrity.get("synthetic_vote_count") or 0),
                "vote_contaminated": bool(vote_integrity.get("contaminated")),
                "vote_hard_failed": bool(vote_integrity.get("hard_failed")),
                "vote_event_count": len(vote_integrity.get("events") or []),
                "qual_event_count": qualitative.get("event_count"),
            }
        )

    all_df = pd.DataFrame(rows)
    runs_df = pd.DataFrame(success_rows)
    if not runs_df.empty:
        runs_df["game_z_payoff"] = runs_df.groupby("game_label")["average_payoff"].transform(
            lambda s: (s - s.mean()) / s.std(ddof=0) if s.notna().sum() > 1 and s.std(ddof=0) > 0 else 0.0
        )

        group_cols = ["game_label", "n_agents", "competition_id"]

        def zscore_cell(series: pd.Series) -> pd.Series:
            std = series.std(ddof=0)
            if series.notna().sum() < 2 or not np.isfinite(std) or std <= 0:
                return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
            return (series - series.mean()) / std

        runs_df["cell_z_payoff"] = runs_df.groupby(group_cols, group_keys=False)[
            "average_payoff"
        ].apply(zscore_cell)

    completion_df = (
        all_df.groupby(["game_label", "model", "model_short", "model_elo"], dropna=False)
        .agg(
            expected=("config_id", "count"),
            completed=("valid_result", "sum"),
            failed=("state", lambda s: int((s != "SUCCESS").sum())),
        )
        .reset_index()
    )
    completion_df["completion_rate"] = completion_df["completed"] / completion_df["expected"]
    return all_df, runs_df, completion_df


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def maybe_add_fit(ax: plt.Axes, frame: pd.DataFrame, x_col: str, y_col: str) -> None:
    data = frame[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 3 or data[x_col].nunique() < 2:
        return
    x = data[x_col].to_numpy(dtype=float)
    y = data[y_col].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, deg=1)
    xs = np.linspace(float(x.min()), float(x.max()), 100)
    ax.plot(xs, intercept + slope * xs, color="#222222", linewidth=1.2, alpha=0.8)
    r = float(np.corrcoef(x, y)[0, 1])
    ax.text(
        0.02,
        0.96,
        f"r={r:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.5},
    )


def plot_completion_heatmap(completion_df: pd.DataFrame, out_path: Path) -> None:
    ordered_models = (
        completion_df[["model", "model_short", "model_elo"]]
        .drop_duplicates()
        .sort_values(["model_elo", "model"])
    )
    row_labels = [
        f"{row.model_short}\nElo {int(row.model_elo)}" for row in ordered_models.itertuples()
    ]
    pivot = (
        completion_df.pivot(index="model", columns="game_label", values="completion_rate")
        .reindex(index=ordered_models["model"], columns=GAME_ORDER)
    )
    completed = (
        completion_df.pivot(index="model", columns="game_label", values="completed")
        .reindex(index=ordered_models["model"], columns=GAME_ORDER)
    )
    expected = (
        completion_df.pivot(index="model", columns="game_label", values="expected")
        .reindex(index=ordered_models["model"], columns=GAME_ORDER)
    )

    fig, ax = plt.subplots(figsize=(7.8, 8.0))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Completion rate"},
    )
    for i, model in enumerate(pivot.index):
        for j, game in enumerate(pivot.columns):
            val = pivot.loc[model, game]
            if pd.isna(val):
                text = "NA"
            else:
                text = f"{int(completed.loc[model, game])}/{int(expected.loc[model, game])}"
            ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=9)
    ax.set_title("Completion coverage by model and game")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([g.replace("game", "Game ") for g in GAME_ORDER], rotation=0)
    ax.set_yticklabels(row_labels, rotation=0)
    savefig(out_path)


def plot_raw_elo_scatter(runs_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    for ax, game in zip(axes, GAME_ORDER):
        sub = runs_df[runs_df["game_label"].eq(game)].copy()
        for n in N_ORDER:
            part = sub[sub["n_agents"].eq(n)]
            if part.empty:
                continue
            ax.scatter(
                part["model_elo"],
                part["average_payoff"],
                s=42,
                alpha=0.72,
                color=N_COLORS[n],
                label=f"N={n}",
                edgecolor="white",
                linewidth=0.4,
            )
        maybe_add_fit(ax, sub, "model_elo", "average_payoff")
        ax.set_title(GAME_TITLES[game])
        ax.set_xlabel("Model Elo")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Average payoff per agent")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.subplots_adjust(bottom=0.22)
    savefig(out_path)


def plot_model_game_means(runs_df: pd.DataFrame, out_path: Path) -> None:
    grouped = (
        runs_df.groupby(["game_label", "model", "model_short", "model_elo"], dropna=False)
        .agg(
            mean_payoff=("average_payoff", "mean"),
            sem_payoff=("average_payoff", lambda s: float(s.std(ddof=1) / math.sqrt(len(s))) if len(s) > 1 else 0.0),
            n=("average_payoff", "count"),
        )
        .reset_index()
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    for ax, game in zip(axes, GAME_ORDER):
        sub = grouped[grouped["game_label"].eq(game)].sort_values("model_elo")
        ax.errorbar(
            sub["model_elo"],
            sub["mean_payoff"],
            yerr=sub["sem_payoff"],
            fmt="o",
            color=GAME_COLORS[game],
            ecolor="#555555",
            capsize=3,
            markersize=6,
        )
        maybe_add_fit(ax, sub.rename(columns={"mean_payoff": "average_payoff"}), "model_elo", "average_payoff")
        for row in sub.itertuples():
            ax.annotate(
                f"{row.model_short}\n(n={int(row.n)})",
                (row.model_elo, row.mean_payoff),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
        ax.set_title(GAME_TITLES[game])
        ax.set_xlabel("Model Elo")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Mean average payoff per agent")
    savefig(out_path)


def plot_stratified_z(runs_df: pd.DataFrame, out_path: Path) -> None:
    grouped = (
        runs_df.groupby(["game_label", "model", "model_short", "model_elo"], dropna=False)
        .agg(
            mean_cell_z=("cell_z_payoff", "mean"),
            sem_cell_z=("cell_z_payoff", lambda s: float(s.std(ddof=1) / math.sqrt(len(s))) if len(s) > 1 else 0.0),
            n=("cell_z_payoff", "count"),
        )
        .reset_index()
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True, sharex=True)
    for ax, game in zip(axes, GAME_ORDER):
        sub = grouped[grouped["game_label"].eq(game)].sort_values("model_elo")
        ax.axhline(0, color="#333333", linewidth=1, alpha=0.6)
        ax.errorbar(
            sub["model_elo"],
            sub["mean_cell_z"],
            yerr=sub["sem_cell_z"],
            fmt="o",
            color=GAME_COLORS[game],
            ecolor="#555555",
            capsize=3,
        )
        maybe_add_fit(ax, sub.rename(columns={"mean_cell_z": "average_payoff"}), "model_elo", "average_payoff")
        for row in sub.itertuples():
            ax.annotate(row.model_short, (row.model_elo, row.mean_cell_z), xytext=(4, 3), textcoords="offset points", fontsize=7)
        ax.set_title(GAME_TITLES[game])
        ax.set_xlabel("Model Elo")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Within-cell payoff z-score")
    savefig(out_path)


def plot_payoff_vs_n(runs_df: pd.DataFrame, out_path: Path) -> None:
    grouped = (
        runs_df.groupby(["game_label", "model", "model_short", "model_elo", "n_agents"], dropna=False)
        .agg(mean_payoff=("average_payoff", "mean"), n=("average_payoff", "count"))
        .reset_index()
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    for ax, game in zip(axes, GAME_ORDER):
        sub = grouped[grouped["game_label"].eq(game)].sort_values(["model_elo", "n_agents"])
        for model, part in sub.groupby("model", sort=False):
            label = str(part["model_short"].iloc[0])
            ax.plot(part["n_agents"], part["mean_payoff"], marker="o", linewidth=1.5, label=label)
        ax.set_title(GAME_TITLES[game])
        ax.set_xlabel("N agents")
        ax.set_xticks(N_ORDER)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("Average payoff per agent")
    savefig(out_path)


def plot_game1_competition(runs_df: pd.DataFrame, out_path: Path) -> None:
    sub = runs_df[runs_df["game_label"].eq("game1")].copy()
    if sub.empty:
        return
    grouped = (
        sub.groupby(["model", "model_short", "model_elo", "competition_level"], dropna=False)
        .agg(mean_payoff=("average_payoff", "mean"), n=("average_payoff", "count"))
        .reset_index()
        .sort_values(["model_elo", "competition_level"])
    )
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for model, part in grouped.groupby("model", sort=False):
        label = f"{part['model_short'].iloc[0]} ({int(part['model_elo'].iloc[0])})"
        ax.plot(part["competition_level"], part["mean_payoff"], marker="o", linewidth=1.6, label=label)
        for row in part.itertuples():
            ax.text(row.competition_level, row.mean_payoff, str(int(row.n)), fontsize=7, ha="center", va="bottom")
    ax.set_title("Game 1 payoff by competition level")
    ax.set_xlabel("Competition level")
    ax.set_ylabel("Average payoff per agent")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    savefig(out_path)


def plot_game_cell_heatmap(runs_df: pd.DataFrame, game: str, out_path: Path) -> None:
    sub = runs_df[runs_df["game_label"].eq(game)].copy()
    if sub.empty:
        return
    if game == "game2":
        sub["cell_plot"] = sub.apply(
            lambda r: f"rho={float(r['rho']):.2f}\ntheta={float(r['theta']):.1f}", axis=1
        )
        title = "Game 2 payoff by model and rho/theta cell"
    elif game == "game3":
        sub["cell_plot"] = sub.apply(
            lambda r: f"sigma={float(r['sigma']):.1f}\nalpha={float(r['alpha']):.1f}", axis=1
        )
        title = "Game 3 payoff by model and sigma/alpha cell"
    else:
        return
    ordered_models = (
        sub[["model", "model_short", "model_elo"]]
        .drop_duplicates()
        .sort_values(["model_elo", "model"])
    )
    ordered_cells = sorted(sub["cell_plot"].dropna().unique())
    pivot = (
        sub.groupby(["model", "cell_plot"])["average_payoff"].mean().unstack()
        .reindex(index=ordered_models["model"], columns=ordered_cells)
    )
    counts = (
        sub.groupby(["model", "cell_plot"])["average_payoff"].count().unstack()
        .reindex(index=ordered_models["model"], columns=ordered_cells)
    )
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="viridis",
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Average payoff"},
    )
    for i, model in enumerate(pivot.index):
        for j, cell in enumerate(pivot.columns):
            if pd.isna(pivot.loc[model, cell]):
                text = ""
            else:
                text = f"{pivot.loc[model, cell]:.1f}\nn={int(counts.loc[model, cell])}"
            ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=8, color="white")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticklabels(
        [f"{row.model_short}\nElo {int(row.model_elo)}" for row in ordered_models.itertuples()],
        rotation=0,
    )
    savefig(out_path)


def plot_round_consensus(runs_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    sns.stripplot(
        data=runs_df,
        x="game_label",
        y="final_round",
        hue="n_agents",
        order=GAME_ORDER,
        hue_order=N_ORDER,
        palette=N_COLORS,
        jitter=0.25,
        alpha=0.75,
        ax=axes[0],
    )
    axes[0].set_title("Final round for completed runs")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Final round")
    axes[0].set_xticks(
        range(len(GAME_ORDER)),
        labels=[g.replace("game", "Game ") for g in GAME_ORDER],
    )
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(title="N", fontsize=8, frameon=False, ncol=3)

    consensus = (
        runs_df.groupby(["game_label", "model_short"], dropna=False)
        .agg(consensus_rate=("consensus_reached", "mean"), n=("consensus_reached", "count"))
        .reset_index()
    )
    sns.barplot(
        data=consensus,
        x="game_label",
        y="consensus_rate",
        hue="model_short",
        order=GAME_ORDER,
        ax=axes[1],
    )
    axes[1].set_title("Consensus rate by completed model/game")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Consensus rate")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(
        range(len(GAME_ORDER)),
        labels=[g.replace("game", "Game ") for g in GAME_ORDER],
    )
    axes[1].legend(fontsize=6, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig(out_path)


def plot_inequality(runs_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
    for ax, game in zip(axes, GAME_ORDER):
        sub = runs_df[runs_df["game_label"].eq(game)]
        for n in N_ORDER:
            part = sub[sub["n_agents"].eq(n)]
            if part.empty:
                continue
            ax.scatter(
                part["model_elo"],
                part["payoff_gini"],
                color=N_COLORS[n],
                label=f"N={n}",
                alpha=0.75,
                edgecolor="white",
                linewidth=0.4,
                s=42,
            )
        maybe_add_fit(ax, sub, "model_elo", "payoff_gini")
        ax.set_title(GAME_TITLES[game])
        ax.set_xlabel("Model Elo")
        ax.set_ylim(bottom=-0.02)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Within-run payoff Gini")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.subplots_adjust(bottom=0.22)
    savefig(out_path)


def plot_normalized_distribution(runs_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2))
    order = (
        runs_df[["model_short", "model_elo"]]
        .drop_duplicates()
        .sort_values("model_elo")["model_short"]
        .tolist()
    )
    sns.boxplot(
        data=runs_df,
        x="model_short",
        y="cell_z_payoff",
        hue="game_label",
        order=order,
        hue_order=GAME_ORDER,
        palette=GAME_COLORS,
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=runs_df,
        x="model_short",
        y="cell_z_payoff",
        hue="game_label",
        order=order,
        hue_order=GAME_ORDER,
        palette=GAME_COLORS,
        dodge=True,
        alpha=0.35,
        size=3,
        ax=ax,
        legend=False,
    )
    ax.axhline(0, color="#333333", linewidth=1, alpha=0.6)
    ax.set_title("Within-cell normalized payoff distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Payoff z-score within game x N x cell")
    ax.tick_params(axis="x", rotation=45)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:3], [l.replace("game", "Game ") for l in labels[:3]], frameon=False, ncol=3)
    savefig(out_path)


def correlation_table(runs_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for game in GAME_ORDER:
        sub = runs_df[runs_df["game_label"].eq(game)]
        model_means = (
            sub.groupby(["model", "model_elo"], dropna=False)
            .agg(
                average_payoff=("average_payoff", "mean"),
                cell_z_payoff=("cell_z_payoff", "mean"),
                n=("config_id", "count"),
            )
            .reset_index()
        )
        for frame_name, frame, y_col in [
            ("run_raw", sub, "average_payoff"),
            ("model_mean_raw", model_means, "average_payoff"),
            ("model_mean_cell_z", model_means, "cell_z_payoff"),
        ]:
            data = frame[["model_elo", y_col]].dropna()
            r = float(data["model_elo"].corr(data[y_col])) if len(data) >= 3 and data["model_elo"].nunique() >= 2 else math.nan
            rows.append(
                {
                    "game_label": game,
                    "comparison": frame_name,
                    "n": int(len(data)),
                    "pearson_r": r,
                }
            )
    return pd.DataFrame(rows)


def model_summary_table(runs_df: pd.DataFrame, completion_df: pd.DataFrame) -> pd.DataFrame:
    perf = (
        runs_df.groupby(["game_label", "model", "model_short", "model_elo"], dropna=False)
        .agg(
            mean_avg_payoff=("average_payoff", "mean"),
            mean_cell_z=("cell_z_payoff", "mean"),
            mean_final_round=("final_round", "mean"),
            consensus_rate=("consensus_reached", "mean"),
            mean_gini=("payoff_gini", "mean"),
            completed=("config_id", "count"),
        )
        .reset_index()
    )
    out = perf.merge(
        completion_df[["game_label", "model", "expected", "failed", "completion_rate"]],
        on=["game_label", "model"],
        how="left",
    )
    return out.sort_values(["game_label", "model_elo"])


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    table = df[columns].copy()
    if max_rows is not None:
        table = table.head(max_rows)
    for col in table.columns:
        if pd.api.types.is_float_dtype(table[col]):
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    return table.to_markdown(index=False)


def write_report(
    *,
    report_path: Path,
    results_root: Path,
    plots: dict[str, Path],
    all_df: pd.DataFrame,
    runs_df: pd.DataFrame,
    completion_df: pd.DataFrame,
    model_summary: pd.DataFrame,
    correlations: pd.DataFrame,
    tables_dir: Path,
) -> None:
    total = len(all_df)
    completed = int(all_df["valid_result"].sum())
    failed = int(total - completed)
    generated = datetime.now().isoformat(timespec="seconds")
    no_consensus = int((~runs_df["consensus_reached"]).sum()) if not runs_df.empty else 0
    vote_issues = int(
        (
            (runs_df["synthetic_vote_count"] > 0)
            | runs_df["vote_contaminated"]
            | runs_df["vote_hard_failed"]
            | (runs_df["vote_event_count"] > 0)
        ).sum()
    ) if not runs_df.empty else 0

    completion_worst = completion_df.sort_values(["completion_rate", "game_label", "model_elo"])[
        ["game_label", "model_short", "model_elo", "completed", "expected", "completion_rate"]
    ].head(12)
    game_counts = (
        all_df.groupby("game_label")
        .agg(completed=("valid_result", "sum"), total=("config_id", "count"))
        .reset_index()
    )
    game_counts["failed"] = game_counts["total"] - game_counts["completed"]
    game_counts["completion_rate"] = game_counts["completed"] / game_counts["total"]

    rel = lambda path: path.relative_to(report_path.parent).as_posix()

    lines = [
        "# Random-Monoculture Control Sanity Report",
        "",
        f"- Generated: `{generated}`",
        f"- Results root: `{results_root}`",
        f"- Valid completed runs used in outcome plots: **{completed} / {total}**",
        f"- Failed or missing runs: **{failed} / {total}**",
        "",
        "## What This Report Is Checking",
        "",
        "The goal is not to make final scientific claims from the partial sweep. The goal is to find obvious data-quality problems, weird payoff patterns, and completion bias before spending more provider budget on the 110 backfill runs.",
        "",
        "Important caveat: the completed set is not balanced. Provider/account failures are concentrated by model and game, so raw Elo/payoff trends can be biased until backfill finishes.",
        "",
        "## Run Coverage",
        "",
        markdown_table(game_counts, ["game_label", "completed", "failed", "total", "completion_rate"]),
        "",
        "Worst coverage cells:",
        "",
        markdown_table(completion_worst, ["game_label", "model_short", "model_elo", "completed", "expected", "completion_rate"]),
        "",
        f"![Completion coverage]({rel(plots['completion_heatmap'])})",
        "",
        "## Elo vs Payoff",
        "",
        "Raw average payoff is shown separately by game because the games have different payoff scales. Points are individual completed runs, colored by N.",
        "",
        f"![Raw Elo scatter]({rel(plots['raw_elo_scatter'])})",
        "",
        "Model-level means summarize the same relationship with one point per model/game. Error bars are standard errors across completed cells, so they reflect the partial data actually available.",
        "",
        f"![Model means]({rel(plots['model_game_means'])})",
        "",
        "The next plot uses the stratification: each run is z-scored within its game x N x competition/cell bucket before averaging by model. This asks whether a model is high or low relative to other completed models in the same local condition.",
        "",
        f"![Stratified normalized Elo]({rel(plots['stratified_z'])})",
        "",
        f"![Normalized distribution]({rel(plots['normalized_distribution'])})",
        "",
        "Correlation diagnostics:",
        "",
        markdown_table(correlations, ["game_label", "comparison", "n", "pearson_r"]),
        "",
        "## Stratification Slices",
        "",
        "These plots check whether visible patterns are driven by one competition level, one N value, or one game-specific parameter cell.",
        "",
        f"![Payoff vs N]({rel(plots['payoff_vs_n'])})",
        "",
        f"![Game 1 competition]({rel(plots['game1_competition'])})",
        "",
        f"![Game 2 cells]({rel(plots['game2_cells'])})",
        "",
        f"![Game 3 cells]({rel(plots['game3_cells'])})",
        "",
        "## Rollout Health",
        "",
        f"- Completed runs without consensus: **{no_consensus} / {completed}**",
        f"- Completed runs with any vote-integrity issue: **{vote_issues} / {completed}**",
        "",
        f"![Round and consensus]({rel(plots['round_consensus'])})",
        "",
        "Within-run payoff inequality is a second sanity check. In a monoculture, large inequality can still be valid because roles/preferences differ, but extreme discontinuities by Elo or N are worth inspecting.",
        "",
        f"![Payoff inequality]({rel(plots['inequality'])})",
        "",
        "## Model/Game Summary",
        "",
        markdown_table(
            model_summary[
                [
                    "game_label",
                    "model_short",
                    "model_elo",
                    "completed",
                    "expected",
                    "completion_rate",
                    "mean_avg_payoff",
                    "mean_cell_z",
                    "mean_final_round",
                    "consensus_rate",
                    "mean_gini",
                ]
            ],
            [
                "game_label",
                "model_short",
                "model_elo",
                "completed",
                "expected",
                "completion_rate",
                "mean_avg_payoff",
                "mean_cell_z",
                "mean_final_round",
                "consensus_rate",
                "mean_gini",
            ],
        ),
        "",
        "## Files",
        "",
        f"- Run-level table: `{rel(tables_dir / 'successful_runs.csv')}`",
        f"- All config status table: `{rel(tables_dir / 'all_configs_status.csv')}`",
        f"- Model/game summary table: `{rel(tables_dir / 'model_game_summary.csv')}`",
        f"- Completion table: `{rel(tables_dir / 'completion_by_game_model.csv')}`",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_root = args.results_root.resolve()
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = results_root / "analysis" / f"sanity_report_{stamp}"
    else:
        output_dir = args.output_dir.resolve()
    plots_dir = output_dir / "plots"
    tables_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    all_df, runs_df, completion_df = load_tables(results_root)
    if runs_df.empty:
        raise SystemExit("No successful runs found; cannot build outcome plots.")

    all_df.to_csv(tables_dir / "all_configs_status.csv", index=False)
    runs_df.to_csv(tables_dir / "successful_runs.csv", index=False)
    completion_df.to_csv(tables_dir / "completion_by_game_model.csv", index=False)
    model_summary = model_summary_table(runs_df, completion_df)
    model_summary.to_csv(tables_dir / "model_game_summary.csv", index=False)
    correlations = correlation_table(runs_df)
    correlations.to_csv(tables_dir / "correlations.csv", index=False)

    plots = {
        "completion_heatmap": plots_dir / "01_completion_heatmap.png",
        "raw_elo_scatter": plots_dir / "02_raw_elo_vs_average_payoff.png",
        "model_game_means": plots_dir / "03_model_game_mean_payoff_vs_elo.png",
        "stratified_z": plots_dir / "04_stratified_cell_z_vs_elo.png",
        "normalized_distribution": plots_dir / "05_within_cell_z_distribution.png",
        "payoff_vs_n": plots_dir / "06_payoff_vs_n_by_model.png",
        "game1_competition": plots_dir / "07_game1_competition_level.png",
        "game2_cells": plots_dir / "08_game2_cell_heatmap.png",
        "game3_cells": plots_dir / "09_game3_cell_heatmap.png",
        "round_consensus": plots_dir / "10_final_round_consensus.png",
        "inequality": plots_dir / "11_payoff_gini_vs_elo.png",
    }

    plot_completion_heatmap(completion_df, plots["completion_heatmap"])
    plot_raw_elo_scatter(runs_df, plots["raw_elo_scatter"])
    plot_model_game_means(runs_df, plots["model_game_means"])
    plot_stratified_z(runs_df, plots["stratified_z"])
    plot_normalized_distribution(runs_df, plots["normalized_distribution"])
    plot_payoff_vs_n(runs_df, plots["payoff_vs_n"])
    plot_game1_competition(runs_df, plots["game1_competition"])
    plot_game_cell_heatmap(runs_df, "game2", plots["game2_cells"])
    plot_game_cell_heatmap(runs_df, "game3", plots["game3_cells"])
    plot_round_consensus(runs_df, plots["round_consensus"])
    plot_inequality(runs_df, plots["inequality"])

    report_path = output_dir / "random_monoculture_sanity_report.md"
    write_report(
        report_path=report_path,
        results_root=results_root,
        plots=plots,
        all_df=all_df,
        runs_df=runs_df,
        completion_df=completion_df,
        model_summary=model_summary,
        correlations=correlations,
        tables_dir=tables_dir,
    )
    print(f"Wrote report: {report_path}")
    print(f"Wrote plots: {plots_dir}")
    print(f"Wrote tables: {tables_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

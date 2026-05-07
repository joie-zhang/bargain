#!/usr/bin/env python3
"""Capability-vs-payoff plots for the three N=2 bargaining games."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "experiments" / "results" / "capability_payoff_scaling_20260505"

GAME_ROOTS = {
    "Game 1": REPO_ROOT / "experiments/results/scaling_experiment_20260404_064451",
    "Game 2": REPO_ROOT / "experiments/results/diplomacy_20260405_082215",
    "Game 3": REPO_ROOT / "experiments/results/cofunding_20260405_083548",
}

OVERALL_CSVS = {
    "Game 1": GAME_ROOTS["Game 1"] / "analysis/average_utility_vs_elo.csv",
    "Game 2": GAME_ROOTS["Game 2"] / "analysis/average_utility_vs_elo.csv",
    "Game 3": GAME_ROOTS["Game 3"] / "analysis/utility_vs_elo_complete_models_only.csv",
}


def short_model_name(model: str) -> str:
    replacements = {
        "claude-opus-4-6-thinking": "Opus 4.6 think",
        "claude-opus-4-6": "Opus 4.6",
        "claude-opus-4-5-20251101-thinking-32k": "Opus 4.5 think",
        "claude-opus-4-5-20251101": "Opus 4.5",
        "claude-sonnet-4-20250514": "Sonnet 4",
        "claude-haiku-4-5-20251001": "Haiku 4.5",
        "claude-3-haiku-20240307": "C3 Haiku",
        "gemini-3-pro": "Gemini 3 Pro",
        "gemini-2.5-pro": "Gemini 2.5 Pro",
        "gpt-5.4-high": "GPT-5.4 high",
        "gpt-5.2-chat-latest-20260210": "GPT-5.2 chat",
        "gpt-5-nano-high": "GPT-5 nano high",
        "gpt-4.1-nano-2025-04-14": "GPT-4.1 nano",
        "gpt-4o-mini-2024-07-18": "GPT-4o mini",
        "gpt-4o-2024-05-13": "GPT-4o",
        "o3-mini-high": "o3-mini high",
        "qwen3-max-preview": "Qwen3 Max",
        "qwen2.5-72b-instruct": "Qwen2.5 72B",
        "qwq-32b": "QwQ 32B",
        "deepseek-r1-0528": "DeepSeek R1 0528",
        "deepseek-r1": "DeepSeek R1",
        "deepseek-v3": "DeepSeek V3",
        "gemma-3-27b-it": "Gemma 3 27B",
        "command-r-plus-08-2024": "Command R+",
        "amazon-nova-pro-v1.0": "Nova Pro",
        "amazon-nova-micro-v1.0": "Nova Micro",
        "llama-3.3-70b-instruct": "Llama 3.3 70B",
        "llama-3.1-8b-instruct": "Llama 3.1 8B",
        "llama-3.2-3b-instruct": "Llama 3.2 3B",
        "llama-3.2-1b-instruct": "Llama 3.2 1B",
    }
    return replacements.get(model, model.replace("-instruct", "").replace("-2025", "").replace("-2024", ""))


def read_elo_sources() -> pd.DataFrame:
    frames = []
    for game, path in OVERALL_CSVS.items():
        df = pd.read_csv(path)
        if "model_short" not in df.columns:
            df["model_short"] = df["model"].map(short_model_name)
        df = df.rename(columns={"avg_utility": "payoff"})
        df["game"] = game
        frames.append(df[["game", "model", "model_short", "elo", "num_runs", "payoff", "std_utility"]])
    return pd.concat(frames, ignore_index=True)


def parse_pair_model(path: Path) -> str | None:
    for part in path.parts:
        if "_vs_" in part:
            return part.split("_vs_", 1)[1]
    return None


def parse_order(path: Path) -> str | None:
    for part in path.parts:
        if part in {"weak_first", "strong_first"}:
            return part
    return None


def adversary_agent(order: str) -> str:
    if order == "weak_first":
        return "Agent_2"
    if order == "strong_first":
        return "Agent_1"
    raise ValueError(f"Unknown model order: {order}")


def iter_result_paths(game: str) -> list[Path]:
    root = GAME_ROOTS[game]
    if game == "Game 1":
        return sorted(root.glob("*_vs_*/*/comp_*/turns_*/run_*/run_*_experiment_results.json"))
    if game == "Game 2":
        return sorted((root / "model_scale").glob("*_vs_*/*/rho_*_theta_*/run_*_experiment_results.json"))
    if game == "Game 3":
        return sorted((root / "model_scale").glob("*_vs_*/*/alpha_*_sigma_*/run_*_experiment_results.json"))
    raise ValueError(game)


def game_competition_id(game: str, config: dict) -> float:
    if game == "Game 1":
        return float(config["competition_level"])
    if game == "Game 2":
        rho = float(config["rho"])
        theta = float(config["theta"])
        return theta * (1.0 - rho) / 2.0
    if game == "Game 3":
        alpha = float(config["alpha"])
        sigma = float(config["sigma"])
        return (1.0 - alpha) * (1.0 - sigma)
    raise ValueError(game)


def load_run_level(overall: pd.DataFrame) -> pd.DataFrame:
    elo_by_game_model = {
        (row.game, row.model): float(row.elo)
        for row in overall.itertuples(index=False)
    }
    rows = []
    for game in GAME_ROOTS:
        for path in iter_result_paths(game):
            model = parse_pair_model(path)
            order = parse_order(path)
            if model is None or order is None:
                continue
            if (game, model) not in elo_by_game_model:
                continue
            with path.open() as f:
                result = json.load(f)
            utilities = result.get("final_utilities") or {}
            agent = adversary_agent(order)
            payoff = utilities.get(agent)
            if payoff is None:
                continue
            config = result.get("config", {})
            rows.append(
                {
                    "game": game,
                    "model": model,
                    "model_short": short_model_name(model),
                    "elo": elo_by_game_model[(game, model)],
                    "order": order,
                    "competition_index": round(game_competition_id(game, config), 8),
                    "payoff": float(payoff),
                    "consensus_reached": bool(result.get("consensus_reached")),
                    "final_round": result.get("final_round"),
                    "result_path": str(path.relative_to(REPO_ROOT)),
                    "interaction_path": str(path.with_name(path.name.replace("_experiment_results", "_all_interactions")).relative_to(REPO_ROOT)),
                }
            )
    return pd.DataFrame(rows)


def aggregate_competition(run_level: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        run_level.groupby(["game", "competition_index", "model", "model_short", "elo"], as_index=False)
        .agg(num_runs=("payoff", "size"), payoff=("payoff", "mean"), std_utility=("payoff", "std"))
        .sort_values(["game", "competition_index", "elo", "model"])
    )
    return grouped


def aggregate_overall(run_level: pd.DataFrame) -> pd.DataFrame:
    return (
        run_level.groupby(["game", "model", "model_short", "elo"], as_index=False)
        .agg(num_runs=("payoff", "size"), payoff=("payoff", "mean"), std_utility=("payoff", "std"))
        .sort_values(["game", "elo", "model"])
    )


def fit_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        x = group["elo"].astype(float).to_numpy()
        y = group["payoff"].astype(float).to_numpy()
        if len(group) < 2 or np.isclose(x.max(), x.min()):
            slope = intercept = r2 = math.nan
        else:
            slope, intercept = np.polyfit(x, y, 1)
            corr = np.corrcoef(x, y)[0, 1]
            r2 = float(corr * corr)
        row = dict(zip(group_cols, key))
        row.update(
            {
                "n_models": len(group),
                "slope_per_elo": slope,
                "slope_per_100_elo": slope * 100 if not math.isnan(slope) else math.nan,
                "intercept": intercept,
                "r2": r2,
                "mean_payoff": float(y.mean()) if len(y) else math.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_overall(overall: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), sharey=False)
    for ax, game in zip(axes, ["Game 1", "Game 2", "Game 3"]):
        data = overall[overall["game"] == game].sort_values("elo")
        ax.scatter(data["elo"], data["payoff"], s=28, color="#2563eb", alpha=0.85)
        slope_row = summary[summary["game"] == game].iloc[0]
        xs = np.linspace(data["elo"].min(), data["elo"].max(), 100)
        ys = slope_row["slope_per_elo"] * xs + slope_row["intercept"]
        ax.plot(xs, ys, linestyle="--", linewidth=1.4, color="#111827")
        for row in data.itertuples(index=False):
            ax.annotate(row.model_short, (row.elo, row.payoff), xytext=(2, 2), textcoords="offset points", fontsize=4.6)
        ax.set_title(f"{game}: payoff vs Elo\nslope={slope_row['slope_per_100_elo']:.2f} payoff / 100 Elo")
        ax.set_xlabel("Arena Elo")
        ax.grid(alpha=0.2, linewidth=0.5)
    axes[0].set_ylabel("Adversary model payoff")
    fig.suptitle("Does model capability scale payoff? N=2 main experiments", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_DIR / "overall_payoff_vs_elo_3panel.png", dpi=220)
    plt.close(fig)


def plot_competition(comp: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.2), sharey=False)
    cmap = plt.get_cmap("viridis")
    for ax, game in zip(axes, ["Game 1", "Game 2", "Game 3"]):
        data_game = comp[comp["game"] == game]
        levels = sorted(data_game["competition_index"].unique())
        colors = {level: cmap(i / max(1, len(levels) - 1)) for i, level in enumerate(levels)}
        for level in levels:
            data = data_game[data_game["competition_index"] == level].sort_values("elo")
            color = colors[level]
            label = f"CI={level:g}" if game != "Game 1" else f"comp={level:g}"
            ax.scatter(data["elo"], data["payoff"], s=17, color=color, alpha=0.8, label=label)
            if len(data) >= 2:
                slope_row = summary[(summary["game"] == game) & (np.isclose(summary["competition_index"], level))].iloc[0]
                xs = np.linspace(data["elo"].min(), data["elo"].max(), 100)
                ys = slope_row["slope_per_elo"] * xs + slope_row["intercept"]
                ax.plot(xs, ys, linestyle="--", linewidth=1.0, color=color, alpha=0.9)
            for row in data.itertuples(index=False):
                ax.annotate(row.model_short, (row.elo, row.payoff), xytext=(1.5, 1.5), textcoords="offset points", fontsize=3.4, alpha=0.78)
        ax.set_title(game)
        ax.set_xlabel("Arena Elo")
        ax.grid(alpha=0.18, linewidth=0.5)
        ax.legend(fontsize=7, frameon=False, ncols=1, loc="best")
    axes[0].set_ylabel("Adversary model payoff")
    fig.suptitle("Payoff vs Elo by competition level/index", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_DIR / "competition_payoff_vs_elo_3panel.png", dpi=240)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, columns: list[str], float_cols: dict[str, str]) -> str:
    rows = []
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in df.iterrows():
        vals = []
        for col in columns:
            val = row[col]
            if col in float_cols and not pd.isna(val):
                vals.append(format(float(val), float_cols[col]))
            else:
                vals.append(str(val))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def write_quant_report(overall_summary: pd.DataFrame, comp_summary: pd.DataFrame, coverage: pd.DataFrame) -> None:
    report = OUT_DIR / "capability_payoff_scaling_quantitative.md"
    rel = lambda p: str(p.relative_to(REPO_ROOT))
    overall_plot = OUT_DIR / "overall_payoff_vs_elo_3panel.png"
    comp_plot = OUT_DIR / "competition_payoff_vs_elo_3panel.png"
    overall_table = overall_summary[["game", "n_models", "slope_per_100_elo", "r2", "mean_payoff"]].copy()
    comp_table = comp_summary[["game", "competition_index", "n_models", "slope_per_100_elo", "r2", "mean_payoff"]].copy()
    coverage_table = coverage.copy()
    text = f"""# Capability-Payoff Scaling Analysis

Generated by `scripts/analyze_capability_payoff_scaling_20260505.py`.

## Scope

This report analyzes the main N=2 model-scaling runs:

- Game 1 item allocation: `{rel(GAME_ROOTS["Game 1"])}`
- Game 2 diplomacy: `{rel(GAME_ROOTS["Game 2"])}`
- Game 3 cofunding: `{rel(GAME_ROOTS["Game 3"])}`

The plotted payoff is the non-baseline/adversary model's realized utility against the fixed `gpt-5-nano` baseline. Elo values are taken from the existing analysis CSVs attached to each run root. For competitive-setting plots, Game 1 uses native `competition_level`; Game 2 uses `CI = theta * (1 - rho) / 2`; Game 3 uses `CI = (1 - alpha) * (1 - sigma)`.

Locally available Codex history search found the prior CI discussion in session `019deb36-702e-7281-831e-85fa43303c19`; these formulas also match the definitions used by the appendix baseline report.

## Run Coverage

{markdown_table(coverage_table, ["game", "run_rows", "models", "competition_cells"], {})}

## Overall Capability Scaling

![Overall payoff vs Elo]({rel(overall_plot)})

{markdown_table(overall_table, ["game", "n_models", "slope_per_100_elo", "r2", "mean_payoff"], {"slope_per_100_elo": ".2f", "r2": ".2f", "mean_payoff": ".2f"})}

Interpretation: the fitted slope is positive in all three games. The relationship is strongest in Game 2, moderately positive in Game 3, and positive but noisier in Game 1.

## Competition-Stratified Scaling

![Competition-stratified payoff vs Elo]({rel(comp_plot)})

{markdown_table(comp_table, ["game", "competition_index", "n_models", "slope_per_100_elo", "r2", "mean_payoff"], {"competition_index": ".2f", "slope_per_100_elo": ".2f", "r2": ".2f", "mean_payoff": ".2f"})}

Interpretation: higher competition generally lowers average payoffs, but the capability slope remains mostly positive. The exception is the easiest/cooperative Game 3 cells, where low- and mid-Elo models can also capture obvious feasible projects and the fitted slope is weak or negative.

## Qualitative Analysis

This section is completed manually from the full rollout audit and subagent findings.
"""
    report.write_text(text)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    elo_sources = read_elo_sources()
    run_level = load_run_level(elo_sources)
    overall = aggregate_overall(run_level)
    comp = aggregate_competition(run_level)
    overall_summary = fit_summary(overall, ["game"]).sort_values("game")
    comp_summary = fit_summary(comp, ["game", "competition_index"]).sort_values(["game", "competition_index"])
    coverage = (
        run_level.groupby("game")
        .agg(
            run_rows=("payoff", "size"),
            models=("model", "nunique"),
            competition_cells=("competition_index", "nunique"),
        )
        .reset_index()
    )

    overall.to_csv(OUT_DIR / "overall_payoff_vs_elo_points.csv", index=False)
    run_level.to_csv(OUT_DIR / "run_level_adversary_payoffs.csv", index=False)
    comp.to_csv(OUT_DIR / "competition_payoff_vs_elo_points.csv", index=False)
    overall_summary.to_csv(OUT_DIR / "overall_fit_summary.csv", index=False)
    comp_summary.to_csv(OUT_DIR / "competition_fit_summary.csv", index=False)

    plot_overall(overall, overall_summary)
    plot_competition(comp, comp_summary)
    write_quant_report(overall_summary, comp_summary, coverage)

    print(f"Wrote analysis to {OUT_DIR}")
    print(overall_summary[["game", "slope_per_100_elo", "r2"]].to_string(index=False))


if __name__ == "__main__":
    main()

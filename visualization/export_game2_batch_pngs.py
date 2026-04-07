#!/usr/bin/env python3
"""
Export summary PNG plots for a Game 2 diplomacy batch.

Usage:
    python visualization/export_game2_batch_pngs.py
    python visualization/export_game2_batch_pngs.py --results-dir experiments/results/diplomacy_20260404_052849
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strong_models_experiment.analysis.active_model_roster import (
    canonical_model_name as roster_canonical_model_name,
    is_active_adversary_model,
    short_model_name as roster_short_model_name,
)

RESULTS_ROOT = PROJECT_ROOT / "experiments" / "results"
DEFAULT_ELO_MARKDOWN = (
    PROJECT_ROOT / "docs" / "guides" / "chatbot_arena_elo_scores_2026_03_11.md"
)

MODEL_SHORT_NAMES: Dict[str, str] = {
    "amazon-nova-micro-v1.0": "Nova Micro",
    "amazon-nova-pro-v1.0": "Nova Pro",
    "claude-3-haiku-20240307": "Claude 3 Haiku",
    "claude-haiku-4-5-20251001": "Haiku 4.5",
    "claude-opus-4-5-20251101": "Opus 4.5",
    "claude-opus-4-5-20251101-thinking-32k": "Opus 4.5 Thinking",
    "claude-opus-4-6": "Opus 4.6",
    "claude-opus-4-6-thinking": "Opus 4.6 Thinking",
    "claude-sonnet-4-20250514": "Sonnet 4",
    "command-r-plus-08-2024": "Command R+",
    "deepseek-r1": "DeepSeek R1",
    "deepseek-r1-0528": "DeepSeek R1-0528",
    "deepseek-v3": "DeepSeek V3",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-3-pro": "Gemini 3 Pro",
    "gemma-3-27b-it": "Gemma 3 27B",
    "gpt-4.1-nano-2025-04-14": "GPT-4.1 nano",
    "gpt-4o-2024-05-13": "GPT-4o",
    "gpt-4o-mini-2024-07-18": "GPT-4o mini",
    "gpt-5-nano": "GPT-5-nano",
    "gpt-5-nano-high": "GPT-5-nano",
    "gpt-5.2-chat-latest-20260210": "GPT-5.2 Chat",
    "gpt-5.4-high": "GPT-5.4 High",
    "llama-3.1-8b-instruct": "Llama 3.1 8B",
    "llama-3.2-1b-instruct": "Llama 3.2 1B",
    "llama-3.2-3b-instruct": "Llama 3.2 3B",
    "llama-3.3-70b-instruct": "Llama 3.3 70B",
    "o3-mini-high": "o3-mini-high",
    "qwen2.5-72b-instruct": "Qwen2.5 72B",
    "qwen3-max-preview": "Qwen3 Max",
    "qwq-32b": "QwQ-32B",
}
MODEL_ALIASES: Dict[str, str] = {
    "amazon-nova-micro": "amazon-nova-micro-v1.0",
    "grok-4-0709": "grok-4",
    "Llama-3.2-1B-Instruct": "llama-3.2-1b-instruct",
    "Llama-3.2-3B-Instruct": "llama-3.2-3b-instruct",
    "o3": "o3-mini-high",
    "QwQ-32B": "qwq-32b",
    "Qwen2.5-72B-Instruct": "qwen2.5-72b-instruct",
    "qwen3-max": "qwen3-max-preview",
}
DEFAULT_ELO_OVERRIDES: Dict[str, Tuple[str, int]] = {
    # The local Arena snapshot only lists the high-effort row for nano.
    "gpt-5-nano": ("gpt-5-nano-high", 1337),
    # Preserve the existing grok fallback for future batches.
    "grok-4": ("grok-4.20-beta1", 1496),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Game 2 batch summary plots as PNGs.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Diplomacy batch root or its configs directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for PNGs. Defaults to visualization/figures/<batch_name>_summary.",
    )
    parser.add_argument(
        "--elo-markdown",
        type=str,
        default=str(DEFAULT_ELO_MARKDOWN),
        help="Path to the Chatbot Arena Elo markdown table.",
    )
    return parser.parse_args()


def competition_index(rho: float, theta: float) -> float:
    return float(theta) * (1.0 - float(rho)) / 2.0


def canonical_model_name(model_name: str) -> str:
    return roster_canonical_model_name(model_name)


def parse_elo_markdown(markdown_path: Path) -> Dict[str, int]:
    elo_by_model: Dict[str, int] = {}
    table_row = re.compile(
        r"^\|\s*\d+\s*\|\s*[^|]+?\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|"
    )
    for line in markdown_path.read_text(encoding="utf-8").splitlines():
        match = table_row.match(line.strip())
        if not match:
            continue
        model_name = match.group(1).strip().strip("`")
        elo_by_model[model_name] = int(match.group(2))
    return elo_by_model


def model_short_name(model_name: str) -> str:
    return roster_short_model_name(model_name)


def model_elo(model_name: str, elo_by_model: Dict[str, int]) -> Optional[float]:
    canonical = canonical_model_name(model_name)
    if canonical in elo_by_model:
        return float(elo_by_model[canonical])
    if canonical in DEFAULT_ELO_OVERRIDES:
        _, elo = DEFAULT_ELO_OVERRIDES[canonical]
        return float(elo)
    return None


def _resolve_candidate(path_str: str) -> Path:
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path.cwd() / candidate).resolve()


def latest_diplomacy_dir() -> Optional[Path]:
    candidates = sorted(
        path
        for path in RESULTS_ROOT.glob("diplomacy_*")
        if path.is_dir() and path.name != "diplomacy_latest"
    )
    return candidates[-1] if candidates else None


def resolve_results_root(raw_value: Optional[str]) -> Path:
    if raw_value is None:
        latest = latest_diplomacy_dir()
        if latest is None:
            raise FileNotFoundError(f"No diplomacy_* directories found under {RESULTS_ROOT}")
        return latest.resolve()

    candidate = _resolve_candidate(raw_value)
    if candidate.name == "configs" and (candidate / "experiment_index.csv").exists():
        return candidate.parent.resolve()
    if (candidate / "configs" / "experiment_index.csv").exists():
        return candidate.resolve()
    raise FileNotFoundError(
        f"Expected a diplomacy results root or configs directory, got: {candidate}"
    )


def _resolve_output_dir(raw_output_dir: str) -> Path:
    candidate = Path(raw_output_dir)
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def _result_file(output_dir: Path) -> Optional[Path]:
    for candidate in [
        output_dir / "run_1_experiment_results.json",
        output_dir / "experiment_results.json",
    ]:
        if candidate.exists():
            return candidate
    return None


def _infer_utility_by_model(
    model1: str,
    model2: str,
    final_utilities: Dict[str, Any],
    agent_performance: Dict[str, Any],
) -> Dict[str, float]:
    utility_by_model: Dict[str, float] = {}
    unknown_agents: List[str] = []

    for agent_id, payload in agent_performance.items():
        if agent_id not in final_utilities:
            continue
        raw_model = payload.get("model")
        if raw_model and raw_model != "unknown":
            utility_by_model[canonical_model_name(raw_model)] = float(final_utilities[agent_id])
        else:
            unknown_agents.append(agent_id)

    remaining_models = [model for model in (model1, model2) if model not in utility_by_model]
    if len(unknown_agents) == len(remaining_models):
        for agent_id, model_name in zip(sorted(unknown_agents), remaining_models):
            utility_by_model[model_name] = float(final_utilities[agent_id])

    return utility_by_model


def load_batch_frames(
    results_root: Path,
    elo_by_model: Dict[str, int],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index_path = results_root / "configs" / "experiment_index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"No experiment index found at {index_path}")

    index_rows: List[Dict[str, Any]] = []
    experiment_rows: List[Dict[str, Any]] = []
    long_rows: List[Dict[str, Any]] = []

    with open(index_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            config_file = results_root / "configs" / row["config_file"]
            config_payload = json.loads(config_file.read_text(encoding="utf-8"))
            output_dir = _resolve_output_dir(config_payload["output_dir"])
            result_path = _result_file(output_dir)

            model1 = canonical_model_name(row["model1"])
            model2 = canonical_model_name(row["model2"])
            if not is_active_adversary_model(model2):
                continue
            rho = float(row["rho"])
            theta = float(row["theta"])
            ci = competition_index(rho, theta)

            index_rows.append(
                {
                    "experiment_id": int(row["experiment_id"]),
                    "model1": model1,
                    "model2": model2,
                    "model_order": row["model_order"],
                    "rho": rho,
                    "theta": theta,
                    "competition_index": ci,
                    "status": "completed" if result_path is not None else "missing_result",
                }
            )

            if result_path is None:
                continue

            result_payload = json.loads(result_path.read_text(encoding="utf-8"))
            final_utilities = result_payload.get("final_utilities", {})
            agent_performance = result_payload.get("agent_performance", {})
            utility_by_model = _infer_utility_by_model(
                model1=model1,
                model2=model2,
                final_utilities=final_utilities,
                agent_performance=agent_performance,
            )

            baseline_utility = utility_by_model.get(model1)
            adversary_utility = utility_by_model.get(model2)
            experiment_rows.append(
                {
                    "experiment_id": int(row["experiment_id"]),
                    "baseline_model": model1,
                    "adversary_model": model2,
                    "model_order": row["model_order"],
                    "rho": rho,
                    "theta": theta,
                    "competition_index": ci,
                    "baseline_utility": baseline_utility,
                    "adversary_utility": adversary_utility,
                    "social_welfare": sum(float(value) for value in final_utilities.values()),
                }
            )

            for model_name, utility in utility_by_model.items():
                long_rows.append(
                    {
                        "experiment_id": int(row["experiment_id"]),
                        "model": model_name,
                        "model_short": model_short_name(model_name),
                        "role": "baseline" if model_name == model1 else "adversary",
                        "utility": utility,
                        "elo": model_elo(model_name, elo_by_model),
                        "rho": rho,
                        "theta": theta,
                        "competition_index": ci,
                    }
                )

    return pd.DataFrame(index_rows), pd.DataFrame(experiment_rows), pd.DataFrame(long_rows)


def _save_overall_utility_vs_elo(long_df: pd.DataFrame, output_dir: Path) -> None:
    overall = (
        long_df.groupby(["model", "model_short", "role", "elo"], as_index=False)
        .agg(avg_utility=("utility", "mean"), num_runs=("utility", "size"))
        .sort_values("elo")
    )
    fig, ax = plt.subplots(figsize=(11, 7))
    colors = {"baseline": "#0f766e", "adversary": "#2563eb"}
    for _, row in overall.iterrows():
        ax.scatter(row["elo"], row["avg_utility"], s=110, color=colors[row["role"]], alpha=0.9)
        ax.annotate(
            row["model_short"],
            (row["elo"], row["avg_utility"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=10,
        )
    ax.set_title("Game 2 Batch: Average Utility vs Model Elo")
    ax.set_xlabel("Model Elo")
    ax.set_ylabel("Average Utility")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_dir / "utility_vs_elo_overall.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_utility_vs_elo_by_ci(long_df: pd.DataFrame, output_dir: Path) -> None:
    grouped = (
        long_df.groupby(["competition_index", "model_short", "elo"], as_index=False)
        .agg(avg_utility=("utility", "mean"))
        .sort_values(["competition_index", "elo", "model_short"])
    )
    ci_levels = sorted(grouped["competition_index"].unique().tolist())
    cmap = plt.cm.viridis(np.linspace(0, 1, len(ci_levels)))
    fig, ax = plt.subplots(figsize=(11, 7))
    for color, ci in zip(cmap, ci_levels):
        ci_df = grouped[grouped["competition_index"] == ci]
        ax.plot(
            ci_df["elo"],
            ci_df["avg_utility"],
            marker="o",
            linewidth=2.0,
            color=color,
            label=f"CI={ci:.2f}",
        )
    ax.set_title("Game 2 Batch: Average Utility vs Elo by Competition Index")
    ax.set_xlabel("Model Elo")
    ax.set_ylabel("Average Utility")
    ax.legend(title="", fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_dir / "utility_vs_elo_by_competition_index.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _rho_theta_label(rho: float, theta: float) -> str:
    return f"rho={rho:.1f}, theta={theta:.1f}"


def _save_utility_vs_elo_by_rho_theta(long_df: pd.DataFrame, output_dir: Path) -> None:
    grouped = (
        long_df.groupby(["rho", "theta", "model_short", "elo"], as_index=False)
        .agg(avg_utility=("utility", "mean"))
        .sort_values(["rho", "theta", "elo", "model_short"])
    )
    settings = (
        grouped[["rho", "theta"]]
        .drop_duplicates()
        .sort_values(["rho", "theta"])
        .itertuples(index=False, name=None)
    )
    settings = list(settings)

    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = plt.cm.tab10(np.linspace(0, 1, len(settings)))
    for color, (rho, theta) in zip(cmap, settings):
        setting_df = grouped[(grouped["rho"] == rho) & (grouped["theta"] == theta)]
        ax.plot(
            setting_df["elo"],
            setting_df["avg_utility"],
            marker="o",
            linewidth=2.0,
            color=color,
            label=_rho_theta_label(float(rho), float(theta)),
        )
    ax.set_title("Game 2 Batch: Average Utility vs Elo by rho/theta Setting")
    ax.set_xlabel("Model Elo")
    ax.set_ylabel("Average Utility")
    ax.legend(title="", fontsize=9, ncol=3)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_dir / "utility_vs_elo_by_rho_theta.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_param_line_plot(
    long_df: pd.DataFrame,
    param_col: str,
    title: str,
    output_name: str,
    output_dir: Path,
) -> None:
    grouped = (
        long_df.groupby([param_col, "model_short"], as_index=False)
        .agg(avg_utility=("utility", "mean"))
        .sort_values(["model_short", param_col])
    )
    fig, ax = plt.subplots(figsize=(11, 7))
    for model_short, model_df in grouped.groupby("model_short"):
        ax.plot(
            model_df[param_col],
            model_df["avg_utility"],
            marker="o",
            linewidth=2.0,
            label=model_short,
        )
    ax.set_title(title)
    ax.set_xlabel(param_col)
    ax.set_ylabel("Average Utility")
    ax.legend(title="", fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_dir / output_name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_utility_heatmap(long_df: pd.DataFrame, output_dir: Path) -> None:
    heatmap = (
        long_df.groupby(["rho", "theta"], as_index=False)
        .agg(mean_utility=("utility", "mean"))
        .pivot(index="rho", columns="theta", values="mean_utility")
        .sort_index(ascending=True)
        .sort_index(axis=1, ascending=True)
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(heatmap.values, cmap="viridis", aspect="auto", origin="lower")
    ax.set_xticks(range(len(heatmap.columns)))
    ax.set_xticklabels([f"{float(v):.1f}" for v in heatmap.columns])
    ax.set_yticks(range(len(heatmap.index)))
    ax.set_yticklabels([f"{float(v):.1f}" for v in heatmap.index])
    ax.set_xlabel("theta")
    ax.set_ylabel("rho")
    ax.set_title("Game 2 Batch: Mean Utility Across rho x theta")
    for row_idx, rho in enumerate(heatmap.index):
        for col_idx, theta in enumerate(heatmap.columns):
            value = heatmap.loc[rho, theta]
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value < heatmap.values.mean() else "black",
                fontsize=10,
            )
    fig.colorbar(image, ax=ax, label="Mean Utility")
    fig.tight_layout()
    fig.savefig(output_dir / "utility_rho_theta_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_welfare_heatmap(experiment_df: pd.DataFrame, output_dir: Path) -> None:
    heatmap = (
        experiment_df.groupby(["rho", "theta"], as_index=False)
        .agg(mean_social_welfare=("social_welfare", "mean"))
        .pivot(index="rho", columns="theta", values="mean_social_welfare")
        .sort_index(ascending=True)
        .sort_index(axis=1, ascending=True)
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(heatmap.values, cmap="viridis", aspect="auto", origin="lower")
    ax.set_xticks(range(len(heatmap.columns)))
    ax.set_xticklabels([f"{float(v):.1f}" for v in heatmap.columns])
    ax.set_yticks(range(len(heatmap.index)))
    ax.set_yticklabels([f"{float(v):.1f}" for v in heatmap.index])
    ax.set_xlabel("theta")
    ax.set_ylabel("rho")
    ax.set_title("Game 2 Batch: Mean Social Welfare Across rho × theta")
    for row_idx, rho in enumerate(heatmap.index):
        for col_idx, theta in enumerate(heatmap.columns):
            value = heatmap.loc[rho, theta]
            ax.text(
                col_idx,
                row_idx,
                f"{value:.1f}",
                ha="center",
                va="center",
                color="white" if value < heatmap.values.mean() else "black",
                fontsize=9,
            )
    fig.colorbar(image, ax=ax, label="Mean Social Welfare")
    fig.tight_layout()
    fig.savefig(output_dir / "social_welfare_rho_theta_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_report(index_df: pd.DataFrame, output_dir: Path, results_root: Path) -> None:
    coverage = (
        index_df.groupby(["model2", "model_order", "status"])
        .size()
        .reset_index(name="count")
        .pivot(index=["model2", "model_order"], columns="status", values="count")
        .fillna(0)
        .reset_index()
    )
    if "completed" not in coverage.columns:
        coverage["completed"] = 0
    if "missing_result" not in coverage.columns:
        coverage["missing_result"] = 0

    lines = [
        "# Game 2 Batch PNG Export",
        "",
        f"- Results root: `{results_root}`",
        f"- Total configs: `{len(index_df)}`",
        f"- Completed configs: `{int((index_df['status'] == 'completed').sum())}`",
        f"- Missing-result configs: `{int((index_df['status'] != 'completed').sum())}`",
        "",
        "## Coverage",
        "",
        "| Adversary | Model Order | Completed | Missing Result |",
        "|---|---|---:|---:|",
    ]
    for _, row in coverage.sort_values(["model2", "model_order"]).iterrows():
        lines.append(
            f"| {model_short_name(str(row['model2']))} | {row['model_order']} | "
            f"{int(row['completed'])} | {int(row['missing_result'])} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `utility_vs_elo_overall.png`",
            "- `utility_vs_elo_by_competition_index.png`",
            "- `utility_vs_elo_by_rho_theta.png`",
            "- `utility_vs_theta.png`",
            "- `utility_vs_rho.png`",
            "- `utility_vs_competition_index.png`",
            "- `utility_rho_theta_heatmap.png`",
            "- `social_welfare_rho_theta_heatmap.png`",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_root = resolve_results_root(args.results_dir)
    elo_markdown = _resolve_candidate(args.elo_markdown)
    output_dir = (
        _resolve_candidate(args.output_dir)
        if args.output_dir
        else (PROJECT_ROOT / "visualization" / "figures" / f"{results_root.name}_summary")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    elo_by_model = parse_elo_markdown(elo_markdown)
    index_df, experiment_df, long_df = load_batch_frames(results_root, elo_by_model)
    if experiment_df.empty or long_df.empty:
        raise RuntimeError(f"No completed result artifacts found under {results_root}")
    missing_elo_models = sorted(
        long_df.loc[long_df["elo"].isna(), "model"].drop_duplicates().tolist()
    )
    if missing_elo_models:
        raise RuntimeError(
            "Missing Elo ratings for completed models: "
            + ", ".join(missing_elo_models)
            + f". Checked {elo_markdown}."
        )

    _save_overall_utility_vs_elo(long_df, output_dir)
    _save_utility_vs_elo_by_ci(long_df, output_dir)
    _save_utility_vs_elo_by_rho_theta(long_df, output_dir)
    _save_param_line_plot(
        long_df,
        param_col="theta",
        title="Game 2 Batch: Average Utility vs theta",
        output_name="utility_vs_theta.png",
        output_dir=output_dir,
    )
    _save_param_line_plot(
        long_df,
        param_col="rho",
        title="Game 2 Batch: Average Utility vs rho",
        output_name="utility_vs_rho.png",
        output_dir=output_dir,
    )
    _save_param_line_plot(
        long_df,
        param_col="competition_index",
        title="Game 2 Batch: Average Utility vs Competition Index",
        output_name="utility_vs_competition_index.png",
        output_dir=output_dir,
    )
    _save_utility_heatmap(long_df, output_dir)
    _save_welfare_heatmap(experiment_df, output_dir)
    _write_report(index_df, output_dir, results_root)

    print(f"Saved PNG plots to: {output_dir}")


if __name__ == "__main__":
    main()

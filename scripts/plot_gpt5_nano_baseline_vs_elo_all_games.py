#!/usr/bin/env python3
"""
Plot GPT-5-nano baseline payoff against adversary Elo for Games 1-3.

This exports, for each game:
  - one overall plot: mean GPT-5-nano payoff vs adversary Elo
  - one stratified plot: one curve per competition level/index

It also writes companion CSVs for both aggregations.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strong_models_experiment.analysis.active_model_roster import (  # noqa: E402
    canonical_model_name as roster_canonical_model_name,
    short_model_name as roster_short_model_name,
)


DEFAULT_GAME1_ROOT = PROJECT_ROOT / "experiments" / "results" / "scaling_experiment_20260404_064451"
DEFAULT_GAME2_ROOT = PROJECT_ROOT / "experiments" / "results" / "diplomacy_20260404_052849"
DEFAULT_GAME3_ROOT = PROJECT_ROOT / "experiments" / "results" / "cofunding_20260405_083548"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "visualization" / "figures" / "gpt5_nano_baseline_vs_elo_all_games_20260413"
DEFAULT_ELO_MARKDOWN = PROJECT_ROOT / "docs" / "guides" / "chatbot_arena_elo_scores_2026_03_31.md"

MODEL_ALIASES: Dict[str, str] = {
    "amazon-nova-micro": "amazon-nova-micro-v1.0",
    "claude-haiku-4-5": "claude-haiku-4-5-20251001",
    "claude-opus-4-5": "claude-opus-4-5-20251101",
    "claude-opus-4-5-thinking-32k": "claude-opus-4-5-20251101-thinking-32k",
    "claude-sonnet-4": "claude-sonnet-4-20250514",
    "gpt-4o": "gpt-4o-2024-05-13",
    "gpt-5-nano": "gpt-5-nano",
    "grok-4-0709": "grok-4",
    "Llama-3.2-1B-Instruct": "llama-3.2-1b-instruct",
    "Llama-3.2-3B-Instruct": "llama-3.2-3b-instruct",
    "o3": "o3-mini-high",
    "QwQ-32B": "qwq-32b",
    "Qwen2.5-72B-Instruct": "qwen2.5-72b-instruct",
    "qwen3-max": "qwen3-max-preview",
}

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
    "gemini-3-flash": "Gemini 3 Flash",
    "gemini-3-pro": "Gemini 3 Pro",
    "gemma-3-27b-it": "Gemma 3 27B",
    "gpt-3.5-turbo-0125": "GPT-3.5",
    "gpt-4.1-nano-2025-04-14": "GPT-4.1 nano",
    "gpt-4o-2024-05-13": "GPT-4o",
    "gpt-4o-mini-2024-07-18": "GPT-4o mini",
    "gpt-5-nano": "GPT-5-nano",
    "gpt-5-nano-high": "GPT-5-nano",
    "gpt-5.2-chat-latest-20260210": "GPT-5.2 Chat",
    "gpt-5.4-high": "GPT-5.4 High",
    "grok-4": "Grok-4",
    "llama-3.1-8b-instruct": "Llama 3.1 8B",
    "llama-3.2-1b-instruct": "Llama 3.2 1B",
    "llama-3.2-3b-instruct": "Llama 3.2 3B",
    "llama-3.3-70b-instruct": "Llama 3.3 70B",
    "o3-mini-high": "o3-mini-high",
    "qwen2.5-72b-instruct": "Qwen2.5 72B",
    "qwen3-max-preview": "Qwen3 Max",
    "qwq-32b": "QwQ-32B",
}

DEFAULT_ELO_OVERRIDES: Dict[str, Tuple[str, int]] = {
    "gpt-5-nano": ("gpt-5-nano-high", 1337),
    # Inference from the March 31, 2026 Arena snapshot: use the closest family row.
    "grok-4": ("grok-4.20-beta1", 1491),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game1-root", type=Path, default=DEFAULT_GAME1_ROOT)
    parser.add_argument("--game2-root", type=Path, default=DEFAULT_GAME2_ROOT)
    parser.add_argument("--game3-root", type=Path, default=DEFAULT_GAME3_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--elo-markdown", type=Path, default=DEFAULT_ELO_MARKDOWN)
    return parser.parse_args()


def canonical_model_name(model_name: str) -> str:
    canonical = MODEL_ALIASES.get(model_name, model_name)
    return roster_canonical_model_name(canonical)


def short_model_name(model_name: str) -> str:
    canonical = canonical_model_name(model_name)
    if canonical in MODEL_SHORT_NAMES:
        return MODEL_SHORT_NAMES[canonical]
    short = roster_short_model_name(canonical)
    return short if short != canonical else canonical


def parse_elo_markdown(markdown_path: Path) -> Dict[str, int]:
    elo_by_model: Dict[str, int] = {}
    table_patterns = [
        re.compile(r"^\|\s*\d+\s*\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|"),
        re.compile(r"^\|\s*\d+\s*\|\s*[^|]+?\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|"),
    ]
    for line in markdown_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        for pattern in table_patterns:
            match = pattern.match(stripped)
            if not match:
                continue
            model_name = canonical_model_name(match.group(1).strip().strip("`"))
            elo_by_model[model_name] = int(match.group(2))
            break
    return elo_by_model


def resolve_elo(model_name: str, elo_by_model: Dict[str, int]) -> Tuple[str, int]:
    canonical = canonical_model_name(model_name)
    if canonical in DEFAULT_ELO_OVERRIDES:
        override_name, override_elo = DEFAULT_ELO_OVERRIDES[canonical]
        return override_name, override_elo
    if canonical in elo_by_model:
        return canonical, elo_by_model[canonical]
    raise KeyError(f"No Elo found for model: {model_name}")


def resolve_path(path_value: str) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def result_file(output_dir: Path) -> Optional[Path]:
    for candidate in [
        output_dir / "run_1_experiment_results.json",
        output_dir / "experiment_results.json",
    ]:
        if candidate.exists():
            return candidate
    return None


def ordered_agent_ids(final_utilities: Dict[str, Any]) -> List[str]:
    if {"Agent_1", "Agent_2"}.issubset(final_utilities):
        return ["Agent_1", "Agent_2"]
    if {"Agent_Alpha", "Agent_Beta"}.issubset(final_utilities):
        return ["Agent_Alpha", "Agent_Beta"]
    return sorted(final_utilities.keys())


def fallback_agent_mapping(
    baseline_model: str,
    adversary_model: str,
    model_order: str,
    final_utilities: Dict[str, Any],
    elo_by_model: Dict[str, int],
) -> Dict[str, str]:
    baseline_elo = resolve_elo(baseline_model, elo_by_model)[1]
    adversary_elo = resolve_elo(adversary_model, elo_by_model)[1]
    if baseline_elo <= adversary_elo:
        weak_model, strong_model = baseline_model, adversary_model
    else:
        weak_model, strong_model = adversary_model, baseline_model

    agent_ids = ordered_agent_ids(final_utilities)
    if len(agent_ids) != 2:
        return {}

    if model_order == "weak_first":
        ordered_models = [weak_model, strong_model]
    else:
        ordered_models = [strong_model, weak_model]

    return {
        agent_ids[0]: canonical_model_name(ordered_models[0]),
        agent_ids[1]: canonical_model_name(ordered_models[1]),
    }


def infer_utility_by_model(
    baseline_model: str,
    adversary_model: str,
    model_order: str,
    result_payload: Dict[str, Any],
    elo_by_model: Dict[str, int],
) -> Dict[str, float]:
    final_utilities = result_payload.get("final_utilities") or {}
    agent_performance = result_payload.get("agent_performance") or {}
    if not isinstance(final_utilities, dict):
        return {}
    if not isinstance(agent_performance, dict):
        agent_performance = {}

    expected_models = {
        canonical_model_name(baseline_model),
        canonical_model_name(adversary_model),
    }
    utility_by_model: Dict[str, float] = {}
    unknown_agents: List[str] = []

    for agent_id, utility in final_utilities.items():
        perf = agent_performance.get(agent_id)
        if isinstance(perf, dict):
            raw_model = str(perf.get("model") or "").strip()
            if raw_model and raw_model != "unknown":
                canonical = canonical_model_name(raw_model)
                if canonical in expected_models:
                    utility_by_model[canonical] = float(utility)
                    continue
        unknown_agents.append(agent_id)

    missing_models = [model for model in expected_models if model not in utility_by_model]
    if len(missing_models) == 1 and len(unknown_agents) == 1:
        utility_by_model[missing_models[0]] = float(final_utilities[unknown_agents[0]])
        return utility_by_model

    if missing_models and unknown_agents:
        fallback = fallback_agent_mapping(
            baseline_model=baseline_model,
            adversary_model=adversary_model,
            model_order=model_order,
            final_utilities=final_utilities,
            elo_by_model=elo_by_model,
        )
        for agent_id in unknown_agents:
            mapped_model = fallback.get(agent_id)
            if mapped_model and mapped_model not in utility_by_model:
                utility_by_model[mapped_model] = float(final_utilities[agent_id])

    return utility_by_model


def load_game1_rows(results_root: Path, elo_by_model: Dict[str, int]) -> pd.DataFrame:
    run_dirs = {
        path.parent
        for pattern in ("experiment_results.json", "run_1_experiment_results.json")
        for path in results_root.rglob(pattern)
    }
    rows: List[Dict[str, Any]] = []

    for run_dir in sorted(run_dirs):
        relative = run_dir.relative_to(results_root)
        if not relative.parts or "_vs_" not in relative.parts[0]:
            continue
        pair_dir = relative.parts[0]
        baseline_model, adversary_model = pair_dir.split("_vs_", 1)
        result_path = result_file(run_dir)
        if result_path is None:
            continue

        payload = json.loads(result_path.read_text(encoding="utf-8"))
        config = payload.get("config") or {}
        if not isinstance(config, dict):
            config = {}

        model_order = str(config.get("actual_order") or config.get("model_order") or "weak_first")
        utility_by_model = infer_utility_by_model(
            baseline_model=baseline_model,
            adversary_model=adversary_model,
            model_order=model_order,
            result_payload=payload,
            elo_by_model=elo_by_model,
        )
        baseline_key = canonical_model_name(baseline_model)
        adversary_key = canonical_model_name(adversary_model)
        if baseline_key not in utility_by_model or adversary_key not in utility_by_model:
            continue

        _, adversary_elo = resolve_elo(adversary_key, elo_by_model)
        rows.append(
            {
                "game_id": "game1",
                "game_label": "Game 1",
                "adversary_model": adversary_key,
                "adversary_short": short_model_name(adversary_key),
                "adversary_elo": adversary_elo,
                "baseline_utility": float(utility_by_model[baseline_key]),
                "adversary_utility": float(utility_by_model[adversary_key]),
                "competition_value": float(config.get("competition_level", 0.0)),
                "competition_col": "competition_level",
                "competition_display": "competition level",
                "competition_curve_label": f"competition={format_number(float(config.get('competition_level', 0.0)))}",
                "model_order": model_order,
                "discussion_turns": int(config.get("discussion_turns", 0)),
                "result_path": str(result_path),
            }
        )

    return pd.DataFrame(rows)


def load_game2_rows(results_root: Path, elo_by_model: Dict[str, int]) -> pd.DataFrame:
    index_path = results_root / "configs" / "experiment_index.csv"
    rows: List[Dict[str, Any]] = []
    with open(index_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            config_path = results_root / "configs" / row["config_file"]
            config_payload = json.loads(config_path.read_text(encoding="utf-8"))
            result_path = result_file(resolve_path(config_payload["output_dir"]))
            if result_path is None:
                continue

            baseline_model = canonical_model_name(row["model1"])
            adversary_model = canonical_model_name(row["model2"])
            model_order = str(row["model_order"])
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            utility_by_model = infer_utility_by_model(
                baseline_model=baseline_model,
                adversary_model=adversary_model,
                model_order=model_order,
                result_payload=payload,
                elo_by_model=elo_by_model,
            )
            if baseline_model not in utility_by_model or adversary_model not in utility_by_model:
                continue

            rho = float(row["rho"])
            theta = float(row["theta"])
            competition_index = theta * (1.0 - rho) / 2.0
            _, adversary_elo = resolve_elo(adversary_model, elo_by_model)
            rows.append(
                {
                    "game_id": "game2",
                    "game_label": "Game 2",
                    "adversary_model": adversary_model,
                    "adversary_short": short_model_name(adversary_model),
                    "adversary_elo": adversary_elo,
                    "baseline_utility": float(utility_by_model[baseline_model]),
                    "adversary_utility": float(utility_by_model[adversary_model]),
                    "competition_value": competition_index,
                    "competition_col": "competition_index",
                    "competition_display": "competition index",
                    "competition_curve_label": f"CI2={format_number(competition_index)}",
                    "model_order": model_order,
                    "rho": rho,
                    "theta": theta,
                    "result_path": str(result_path),
                }
            )

    return pd.DataFrame(rows)


def load_game3_rows(results_root: Path, elo_by_model: Dict[str, int]) -> pd.DataFrame:
    index_path = results_root / "configs" / "experiment_index.csv"
    rows: List[Dict[str, Any]] = []
    with open(index_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            config_path = results_root / "configs" / row["config_file"]
            config_payload = json.loads(config_path.read_text(encoding="utf-8"))
            result_path = result_file(resolve_path(config_payload["output_dir"]))
            if result_path is None:
                continue

            baseline_model = canonical_model_name(row["model1"])
            adversary_model = canonical_model_name(row["model2"])
            model_order = str(row["model_order"])
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            utility_by_model = infer_utility_by_model(
                baseline_model=baseline_model,
                adversary_model=adversary_model,
                model_order=model_order,
                result_payload=payload,
                elo_by_model=elo_by_model,
            )
            if baseline_model not in utility_by_model or adversary_model not in utility_by_model:
                continue

            alpha = float(row["alpha"])
            sigma = float(row["sigma"])
            competition_index = (1.0 - alpha) * (1.0 - sigma)
            _, adversary_elo = resolve_elo(adversary_model, elo_by_model)
            rows.append(
                {
                    "game_id": "game3",
                    "game_label": "Game 3",
                    "adversary_model": adversary_model,
                    "adversary_short": short_model_name(adversary_model),
                    "adversary_elo": adversary_elo,
                    "baseline_utility": float(utility_by_model[baseline_model]),
                    "adversary_utility": float(utility_by_model[adversary_model]),
                    "competition_value": competition_index,
                    "competition_col": "competition_index",
                    "competition_display": "competition index",
                    "competition_curve_label": f"CI3={format_number(competition_index)}",
                    "model_order": model_order,
                    "alpha": alpha,
                    "sigma": sigma,
                    "result_path": str(result_path),
                }
            )

    return pd.DataFrame(rows)


def format_number(value: float) -> str:
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def summarize_overall(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["adversary_model", "adversary_short", "adversary_elo"], as_index=False)
        .agg(
            num_runs=("baseline_utility", "size"),
            avg_baseline_utility=("baseline_utility", "mean"),
            std_baseline_utility=("baseline_utility", lambda x: float(np.std(x, ddof=0))),
        )
        .sort_values(["adversary_elo", "adversary_model"])
        .reset_index(drop=True)
    )


def summarize_by_competition(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(
            [
                "competition_value",
                "competition_curve_label",
                "adversary_model",
                "adversary_short",
                "adversary_elo",
            ],
            as_index=False,
        )
        .agg(
            num_runs=("baseline_utility", "size"),
            avg_baseline_utility=("baseline_utility", "mean"),
            std_baseline_utility=("baseline_utility", lambda x: float(np.std(x, ddof=0))),
        )
        .sort_values(["competition_value", "adversary_elo", "adversary_model"])
        .reset_index(drop=True)
    )


def make_overall_plot(df: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(
        df["adversary_elo"],
        df["avg_baseline_utility"],
        color="#2563eb",
        linewidth=2.0,
        marker="o",
        markersize=6,
    )
    ax.scatter(df["adversary_elo"], df["avg_baseline_utility"], color="#2563eb", s=50, alpha=0.95)

    for _, row in df.iterrows():
        ax.annotate(
            row["adversary_short"],
            (row["adversary_elo"], row["avg_baseline_utility"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            alpha=0.9,
        )

    ax.set_title(title)
    ax.set_xlabel("Chatbot Arena Elo (adversary model)")
    ax.set_ylabel("Mean GPT-5-nano payoff")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_by_competition_plot(
    df: pd.DataFrame,
    output_path: Path,
    title: str,
    legend_title: str,
) -> None:
    comp_values = sorted(df["competition_value"].dropna().unique().tolist())
    colors = plt.cm.viridis(np.linspace(0, 1, len(comp_values)))

    fig, ax = plt.subplots(figsize=(11, 7))
    for color, comp_value in zip(colors, comp_values):
        comp_df = df[df["competition_value"] == comp_value].sort_values("adversary_elo")
        if comp_df.empty:
            continue
        label = str(comp_df["competition_curve_label"].iloc[0])
        ax.plot(
            comp_df["adversary_elo"],
            comp_df["avg_baseline_utility"],
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=5,
            label=label,
        )

    ax.set_title(title)
    ax.set_xlabel("Chatbot Arena Elo (adversary model)")
    ax.set_ylabel("Mean GPT-5-nano payoff")
    ax.legend(title=legend_title, fontsize=9, title_fontsize=10, ncol=2, frameon=True)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def export_game_artifacts(game_df: pd.DataFrame, output_dir: Path) -> Dict[str, Any]:
    game_id = str(game_df["game_id"].iloc[0])
    game_label = str(game_df["game_label"].iloc[0])
    competition_display = str(game_df["competition_display"].iloc[0])
    competition_col = str(game_df["competition_col"].iloc[0])

    overall_df = summarize_overall(game_df)
    by_comp_df = summarize_by_competition(game_df)

    overall_csv = output_dir / f"{game_id}_gpt5_nano_baseline_payoff_vs_adversary_elo_overall.csv"
    overall_png = output_dir / f"{game_id}_gpt5_nano_baseline_payoff_vs_adversary_elo_overall.png"
    comp_csv = output_dir / f"{game_id}_gpt5_nano_baseline_payoff_vs_adversary_elo_by_{competition_col}.csv"
    comp_png = output_dir / f"{game_id}_gpt5_nano_baseline_payoff_vs_adversary_elo_by_{competition_col}.png"

    overall_df.to_csv(overall_csv, index=False)
    by_comp_df.to_csv(comp_csv, index=False)

    make_overall_plot(
        overall_df,
        overall_png,
        title=(
            f"{game_label}: GPT-5-nano baseline payoff vs adversary Elo\n"
            f"Averages include all completed settings for this batch."
        ),
    )
    make_by_competition_plot(
        by_comp_df,
        comp_png,
        title=(
            f"{game_label}: GPT-5-nano baseline payoff vs adversary Elo\n"
            f"One curve per {competition_display}; averages include all other completed settings."
        ),
        legend_title=competition_display.title(),
    )

    return {
        "game_id": game_id,
        "game_label": game_label,
        "competition_col": competition_col,
        "overall_csv": overall_csv,
        "overall_png": overall_png,
        "comp_csv": comp_csv,
        "comp_png": comp_png,
        "num_raw_rows": len(game_df),
        "num_overall_points": len(overall_df),
        "num_comp_points": len(by_comp_df),
        "num_comp_values": int(game_df["competition_value"].nunique()),
    }


def write_summary(summary_rows: Iterable[Dict[str, Any]], output_dir: Path) -> Path:
    lines = [
        "# GPT-5-nano Baseline Payoff vs Adversary Elo",
        "",
        f"- Output directory: `{output_dir}`",
        "",
        "| Game | Raw Runs | Overall Points | Competition Values | Competition Points | Overall Plot | Stratified Plot |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['game_label']} | {row['num_raw_rows']} | {row['num_overall_points']} | "
            f"{row['num_comp_values']} | {row['num_comp_points']} | "
            f"`{Path(row['overall_png']).name}` | `{Path(row['comp_png']).name}` |"
        )
    summary_path = output_dir / "README.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    elo_by_model = parse_elo_markdown(args.elo_markdown)
    game_frames = [
        load_game1_rows(args.game1_root, elo_by_model),
        load_game2_rows(args.game2_root, elo_by_model),
        load_game3_rows(args.game3_root, elo_by_model),
    ]

    summaries: List[Dict[str, Any]] = []
    for game_df in game_frames:
        if game_df.empty:
            continue
        summaries.append(export_game_artifacts(game_df, args.output_dir))

    if not summaries:
        raise RuntimeError("No completed runs were found across the requested game batches.")

    summary_path = write_summary(summaries, args.output_dir)
    print(f"Wrote outputs to: {args.output_dir}")
    print(f"Summary: {summary_path}")
    for row in summaries:
        print(
            f"{row['game_label']}: raw_runs={row['num_raw_rows']}, overall_points={row['num_overall_points']}, "
            f"competition_values={row['num_comp_values']}, competition_points={row['num_comp_points']}"
        )


if __name__ == "__main__":
    main()

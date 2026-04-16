#!/usr/bin/env python3
"""
Analyze a Game 3 co-funding batch and export utility-vs-Elo plots.

This script treats a config as "finished" when its output directory contains a
parsed experiment result JSON. It then aggregates the comparison model's
realized utility across finished runs.

Outputs:
  - completion_summary.csv
  - utility_vs_elo_all_models.csv
  - utility_vs_elo_all_models.png
  - utility_vs_elo_complete_models_only.csv
  - utility_vs_elo_complete_models_only.png
  - utility_vs_elo_by_competition_index.csv
  - utility_vs_elo_by_competition_index.png
  - report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strong_models_experiment.analysis.active_model_roster import (
    canonical_model_name as roster_canonical_model_name,
    is_active_adversary_model,
    short_model_name as roster_short_model_name,
)

DEFAULT_ELO_MARKDOWN = PROJECT_ROOT / "docs" / "guides" / "chatbot_arena_elo_scores_2026_03_31.md"

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
    "Llama-3.2-1B-Instruct": "llama-3.2-1b-instruct",
    "Llama-3.2-3B-Instruct": "llama-3.2-3b-instruct",
    "o3": "o3-mini-high",
    "QwQ-32B": "qwq-32b",
    "Qwen2.5-72B-Instruct": "qwen2.5-72b-instruct",
    "gpt-5-nano": "gpt-5-nano-high",
    "qwen3-max": "qwen3-max-preview",
}

DEFAULT_ELO_OVERRIDES: Dict[str, Tuple[str, int]] = {
    "gpt-5-nano": ("gpt-5-nano-high", 1337),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Game 3 utility vs Elo for a co-funding batch.")
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Co-funding batch root, e.g. experiments/results/cofunding_20260405_083548",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where plots, CSVs, and report.md will be written.",
    )
    parser.add_argument(
        "--elo-markdown",
        type=Path,
        default=DEFAULT_ELO_MARKDOWN,
        help="Chatbot Arena Elo markdown table.",
    )
    return parser.parse_args()


def canonical_model_name(model_name: str) -> str:
    return roster_canonical_model_name(model_name)


def model_short_name(model_name: str) -> str:
    return roster_short_model_name(model_name)


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


def resolve_elo(model_name: str, elo_by_model: Dict[str, int]) -> Tuple[str, int]:
    canonical = canonical_model_name(model_name)
    if canonical in elo_by_model:
        return canonical, elo_by_model[canonical]
    if canonical in DEFAULT_ELO_OVERRIDES:
        return DEFAULT_ELO_OVERRIDES[canonical]
    raise KeyError(f"No Elo found for model: {model_name}")


def resolve_path(path_value: str) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def result_file(output_dir: Path) -> Optional[Path]:
    for candidate in [
        output_dir / "experiment_results.json",
        output_dir / "run_1_experiment_results.json",
    ]:
        if candidate.exists():
            return candidate
    return None


def competition_index(alpha: float, sigma: float) -> float:
    return (1.0 - float(alpha)) * (1.0 - float(sigma))


def best_fit_line(xs: List[float], ys: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    if len(xs) < 2:
        return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    slope, intercept = np.polyfit(xs, ys, deg=1)
    x_min = min(xs)
    x_max = max(xs)
    if x_min == x_max:
        return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    line_x = np.linspace(x_min, x_max, 200)
    line_y = slope * line_x + intercept
    return line_x, line_y


def format_competition_index(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def fallback_agent_mapping(
    model1: str,
    model2: str,
    model_order: str,
    elo_by_model: Dict[str, int],
) -> Dict[str, str]:
    elo1 = resolve_elo(model1, elo_by_model)[1]
    elo2 = resolve_elo(model2, elo_by_model)[1]
    weak_model, strong_model = (model1, model2) if elo1 <= elo2 else (model2, model1)
    if model_order == "weak_first":
        return {"Agent_1": weak_model, "Agent_2": strong_model}
    return {"Agent_1": strong_model, "Agent_2": weak_model}


def infer_utility_by_model(
    model1: str,
    model2: str,
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

    utility_by_model: Dict[str, float] = {}
    unknown_agents: List[str] = []
    expected_models = {canonical_model_name(model1), canonical_model_name(model2)}

    for agent_id, utility in final_utilities.items():
        perf = agent_performance.get(agent_id)
        if isinstance(perf, dict):
            raw_model = perf.get("model")
            if raw_model and raw_model != "unknown":
                canonical = canonical_model_name(str(raw_model))
                if canonical in expected_models:
                    utility_by_model[canonical] = float(utility)
                    continue
        unknown_agents.append(agent_id)

    missing_models = [
        model for model in (canonical_model_name(model1), canonical_model_name(model2))
        if model not in utility_by_model
    ]
    if unknown_agents and missing_models:
        mapping = fallback_agent_mapping(model1, model2, model_order, elo_by_model)
        for agent_id in unknown_agents:
            mapped_model = canonical_model_name(mapping.get(agent_id, ""))
            if mapped_model:
                utility_by_model[mapped_model] = float(final_utilities[agent_id])

    return utility_by_model


def load_batch_rows(
    results_root: Path,
    elo_by_model: Dict[str, int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    index_path = results_root / "configs" / "experiment_index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing experiment index: {index_path}")

    completion_rows: List[Dict[str, Any]] = []
    run_rows: List[Dict[str, Any]] = []

    with open(index_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            config_path = results_root / "configs" / row["config_file"]
            config_payload = json.loads(config_path.read_text(encoding="utf-8"))
            output_dir = resolve_path(str(config_payload["output_dir"]))

            model1 = canonical_model_name(row["model1"])
            model2 = canonical_model_name(row["model2"])
            if not is_active_adversary_model(model2):
                continue
            alpha = float(row["alpha"])
            sigma = float(row["sigma"])
            ci = competition_index(alpha, sigma)
            started = (output_dir / "run_1_all_interactions.json").exists()

            result_path = result_file(output_dir)
            result_payload: Optional[Dict[str, Any]] = None
            if result_path is not None:
                try:
                    loaded = json.loads(result_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    loaded = None
                if isinstance(loaded, dict):
                    result_payload = loaded
                else:
                    result_path = None

            status = "completed" if result_payload is not None else ("started_only" if started else "not_started")

            elo_model_name, elo = resolve_elo(model2, elo_by_model)
            completion_rows.append(
                {
                    "experiment_id": int(row["experiment_id"]),
                    "model": model2,
                    "model_short": model_short_name(model2),
                    "elo_model_name": elo_model_name,
                    "elo": elo,
                    "model_order": row["model_order"],
                    "alpha": alpha,
                    "sigma": sigma,
                    "competition_index": ci,
                    "status": status,
                    "output_dir": str(output_dir),
                }
            )

            if result_payload is None:
                continue

            utility_by_model = infer_utility_by_model(
                model1=model1,
                model2=model2,
                model_order=row["model_order"],
                result_payload=result_payload,
                elo_by_model=elo_by_model,
            )
            if model2 not in utility_by_model:
                continue

            run_rows.append(
                {
                    "experiment_id": int(row["experiment_id"]),
                    "model": model2,
                    "model_short": model_short_name(model2),
                    "elo_model_name": elo_model_name,
                    "elo": elo,
                    "model_order": row["model_order"],
                    "alpha": alpha,
                    "sigma": sigma,
                    "competition_index": ci,
                    "utility": float(utility_by_model[model2]),
                }
            )

    return completion_rows, run_rows


def summarize_completion(completion_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in completion_rows:
        grouped[str(row["model"])].append(row)

    rows: List[Dict[str, Any]] = []
    for model_name, model_rows in grouped.items():
        status_counts: Dict[str, int] = defaultdict(int)
        for row in model_rows:
            status_counts[str(row["status"])] += 1
        first = model_rows[0]
        total = len(model_rows)
        rows.append(
            {
                "model": model_name,
                "model_short": first["model_short"],
                "elo_model_name": first["elo_model_name"],
                "elo": first["elo"],
                "total_configs": total,
                "completed_configs": status_counts["completed"],
                "started_only_configs": status_counts["started_only"],
                "not_started_configs": status_counts["not_started"],
                "fully_finished": status_counts["completed"] == total,
                "has_any_finished": status_counts["completed"] > 0,
            }
        )

    return sorted(rows, key=lambda row: int(row["elo"]), reverse=True)


def summarize_models(
    run_rows: List[Dict[str, Any]],
    selected_models: Iterable[str],
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    model_meta: Dict[str, Dict[str, Any]] = {}
    for row in run_rows:
        model_name = str(row["model"])
        grouped[model_name].append(float(row["utility"]))
        model_meta[model_name] = row

    rows: List[Dict[str, Any]] = []
    for model_name in selected_models:
        values = grouped.get(model_name, [])
        if not values:
            continue
        meta = model_meta[model_name]
        rows.append(
            {
                "model": model_name,
                "model_short": meta["model_short"],
                "elo_model_name": meta["elo_model_name"],
                "elo": meta["elo"],
                "num_runs": len(values),
                "avg_utility": mean(values),
                "std_utility": pstdev(values) if len(values) > 1 else 0.0,
            }
        )

    return sorted(rows, key=lambda row: int(row["elo"]))


def summarize_models_by_competition_index(
    run_rows: List[Dict[str, Any]],
    selected_models: Iterable[str],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[float, str], List[float]] = defaultdict(list)
    model_meta: Dict[str, Dict[str, Any]] = {}
    for row in run_rows:
        model_name = str(row["model"])
        grouped[(float(row["competition_index"]), model_name)].append(float(row["utility"]))
        model_meta[model_name] = row

    rows: List[Dict[str, Any]] = []
    for competition_level in sorted({float(row["competition_index"]) for row in run_rows}):
        for model_name in selected_models:
            values = grouped.get((competition_level, model_name), [])
            if not values:
                continue
            meta = model_meta[model_name]
            rows.append(
                {
                    "competition_index": competition_level,
                    "model": model_name,
                    "model_short": meta["model_short"],
                    "elo_model_name": meta["elo_model_name"],
                    "elo": meta["elo"],
                    "num_runs": len(values),
                    "avg_utility": mean(values),
                    "std_utility": pstdev(values) if len(values) > 1 else 0.0,
                }
            )

    return sorted(rows, key=lambda row: (float(row["competition_index"]), int(row["elo"])))


def summarize_models_by_alpha_sigma(
    run_rows: List[Dict[str, Any]],
    selected_models: Iterable[str],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[float, float, str], List[float]] = defaultdict(list)
    model_meta: Dict[str, Dict[str, Any]] = {}
    for row in run_rows:
        model_name = str(row["model"])
        grouped[(float(row["alpha"]), float(row["sigma"]), model_name)].append(float(row["utility"]))
        model_meta[model_name] = row

    rows: List[Dict[str, Any]] = []
    alpha_sigma_pairs = sorted({(float(row["alpha"]), float(row["sigma"])) for row in run_rows})
    for alpha, sigma in alpha_sigma_pairs:
        for model_name in selected_models:
            values = grouped.get((alpha, sigma, model_name), [])
            if not values:
                continue
            meta = model_meta[model_name]
            rows.append(
                {
                    "alpha": alpha,
                    "sigma": sigma,
                    "model": model_name,
                    "model_short": meta["model_short"],
                    "elo_model_name": meta["elo_model_name"],
                    "elo": meta["elo"],
                    "num_runs": len(values),
                    "avg_utility": mean(values),
                    "std_utility": pstdev(values) if len(values) > 1 else 0.0,
                }
            )

    return sorted(rows, key=lambda row: (float(row["alpha"]), float(row["sigma"]), int(row["elo"])))


def write_csv(output_path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_overall_plot(
    rows: List[Dict[str, Any]],
    output_path: Path,
    title: str,
    point_color: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 7))

    for row in rows:
        x_value = float(row["elo"])
        y_value = float(row["avg_utility"])
        ax.scatter(
            x_value,
            y_value,
            s=110,
            color=point_color,
            alpha=0.9,
        )
        ax.annotate(
            str(row["model_short"]),
            (x_value, y_value),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )

    ax.set_title(title)
    ax.set_xlabel("Chatbot Arena Elo (adversary model)")
    ax.set_ylabel("Mean Adversary Utility")
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_competition_index_plot(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7.2))
    competition_levels = sorted({float(row["competition_index"]) for row in rows})
    color_map = plt.get_cmap("viridis")
    colors = {
        comp: color_map(idx / max(len(competition_levels) - 1, 1))
        for idx, comp in enumerate(competition_levels)
    }

    grouped: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["competition_index"])].append(row)

    xtick_rows: Dict[int, str] = {}
    for comp in competition_levels:
        comp_rows = sorted(grouped[comp], key=lambda row: int(row["elo"]))
        xs = [float(row["elo"]) for row in comp_rows]
        ys = [float(row["avg_utility"]) for row in comp_rows]
        line_x, line_y = best_fit_line(xs, ys)
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=7,
            linewidth=2.2,
            color=colors[comp],
            alpha=0.28,
            label=f"CI3={format_competition_index(comp)}",
        )
        ax.plot(
            line_x,
            line_y,
            linewidth=2.8,
            color=colors[comp],
            alpha=0.98,
        )
        for row in comp_rows:
            xtick_rows[int(row["elo"])] = str(row["model_short"])
            ax.annotate(
                f"{row['avg_utility']:.1f}",
                (float(row["elo"]), float(row["avg_utility"])),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=colors[comp],
            )

    x_positions = sorted(xtick_rows)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{xtick_rows[x]}\n{x}" for x in x_positions], rotation=25, ha="right")
    ax.set_title(
        "Game 3: Mean Adversary Utility vs Chatbot Arena Elo\n"
        "Stratified by derived CI3 = (1 - alpha) * (1 - sigma); averages include both model orders."
    )
    ax.set_xlabel("Chatbot Arena Elo (adversary model)")
    ax.set_ylabel("Mean Adversary Utility")
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(title="Derived CI3", frameon=True)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_alpha_sigma_plot(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    alpha_vals = sorted({float(row["alpha"]) for row in rows})
    sigma_vals = sorted({float(row["sigma"]) for row in rows})
    fig, axes = plt.subplots(1, len(alpha_vals), figsize=(6 * len(alpha_vals), 5.8), sharey=True)
    if len(alpha_vals) == 1:
        axes = [axes]

    color_map = plt.get_cmap("viridis")
    sigma_colors = {
        sigma: color_map(idx / max(len(sigma_vals) - 1, 1))
        for idx, sigma in enumerate(sigma_vals)
    }
    legend_handles = []
    legend_labels = []

    for ax, alpha in zip(axes, alpha_vals):
        alpha_rows = [row for row in rows if float(row["alpha"]) == alpha]
        for sigma in sigma_vals:
            sigma_rows = sorted(
                [row for row in alpha_rows if float(row["sigma"]) == sigma],
                key=lambda row: int(row["elo"]),
            )
            if not sigma_rows:
                continue
            xs = [float(row["elo"]) for row in sigma_rows]
            ys = [float(row["avg_utility"]) for row in sigma_rows]
            line, = ax.plot(
                xs,
                ys,
                marker="o",
                markersize=6,
                linewidth=2.1,
                color=sigma_colors[sigma],
                alpha=0.95,
                label=rf"$\sigma={sigma:.1f}$",
            )
            if len(legend_handles) < len(sigma_vals):
                legend_handles.append(line)
                legend_labels.append(rf"$\sigma={sigma:.1f}$")
        ax.set_title(rf"$\alpha={alpha:.1f}$")
        ax.set_xlabel("Chatbot Arena Elo")
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Mean Adversary Utility")
    fig.suptitle(
        "Game 3: Mean Adversary Utility vs Elo by alpha and sigma\n"
        "Panels fix alpha; curves fix sigma; averages include both model orders.",
        y=1.03,
    )
    fig.legend(legend_handles, legend_labels, title=r"$\sigma$", loc="upper center", ncol=len(sigma_vals))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(
    output_dir: Path,
    results_root: Path,
    completion_summary: List[Dict[str, Any]],
    all_rows: List[Dict[str, Any]],
    complete_only_rows: List[Dict[str, Any]],
    by_ci_rows: List[Dict[str, Any]],
    by_alpha_sigma_rows: List[Dict[str, Any]],
    total_completed: int,
    total_unfinished: int,
    started_only: int,
    not_started: int,
) -> None:
    partial_rows = [row for row in completion_summary if not row["fully_finished"]]

    lines = [
        f"# Game 3 Utility vs Elo Report ({results_root.name})",
        "",
        f"- Batch root: `{results_root}`",
        f"- Finished configs: `{total_completed}`",
        f"- Unfinished configs: `{total_unfinished}`",
        f"- Started but not finished: `{started_only}`",
        f"- Not started: `{not_started}`",
        f"- Models with any finished runs: `{len(all_rows)}`",
        f"- Models fully finished across all 18 configs: `{len(complete_only_rows)}`",
        "",
        "## Files",
        "",
        "- `completion_summary.csv`",
        "- `utility_vs_elo_all_models.csv`",
        "- `utility_vs_elo_all_models.png`",
        "- `utility_vs_elo_complete_models_only.csv`",
        "- `utility_vs_elo_complete_models_only.png`",
        "- `utility_vs_elo_by_competition_index.csv`",
        "- `utility_vs_elo_by_competition_index.png`",
        "- `utility_vs_elo_by_alpha_sigma.csv`",
        "- `utility_vs_elo_by_alpha_sigma.png`",
        "",
        "## Partial Models",
        "",
        "| model | elo | completed | started_only | not_started | total |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for row in partial_rows:
        lines.append(
            f"| {row['model']} | {row['elo']} | {row['completed_configs']} | "
            f"{row['started_only_configs']} | {row['not_started_configs']} | {row['total_configs']} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The Elo plots use adversary-model utility, averaged over completed Game 3 runs only.",
            "- The overall plots aggregate each adversary model's realized utility over its finished `(alpha, sigma, model_order)` configs.",
            "- The complete-only plot restricts to models with all 18 expected Game 3 configs finished.",
            "- `utility_vs_elo_by_alpha_sigma.png` uses the native Game 3 parameters directly: panels fix `alpha`, and curves fix `sigma`.",
            "- `utility_vs_elo_by_competition_index.png` remains as a derived 1D summary using `CI3 = (1 - alpha) * (1 - sigma)`.",
        ]
    )

    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    elo_by_model = parse_elo_markdown(args.elo_markdown.resolve())
    completion_rows, run_rows = load_batch_rows(results_root, elo_by_model)
    completion_summary = summarize_completion(completion_rows)

    total_completed = sum(1 for row in completion_rows if row["status"] == "completed")
    total_unfinished = sum(1 for row in completion_rows if row["status"] != "completed")
    started_only = sum(1 for row in completion_rows if row["status"] == "started_only")
    not_started = sum(1 for row in completion_rows if row["status"] == "not_started")

    all_models = [row["model"] for row in completion_summary if row["has_any_finished"]]
    complete_models = [row["model"] for row in completion_summary if row["fully_finished"]]

    all_rows = summarize_models(run_rows, all_models)
    complete_only_rows = summarize_models(run_rows, complete_models)
    by_ci_rows = summarize_models_by_competition_index(run_rows, all_models)
    by_alpha_sigma_rows = summarize_models_by_alpha_sigma(run_rows, all_models)

    write_csv(
        output_dir / "completion_summary.csv",
        completion_summary,
        [
            "model",
            "model_short",
            "elo_model_name",
            "elo",
            "total_configs",
            "completed_configs",
            "started_only_configs",
            "not_started_configs",
            "fully_finished",
            "has_any_finished",
        ],
    )
    write_csv(
        output_dir / "utility_vs_elo_all_models.csv",
        all_rows,
        ["model", "model_short", "elo_model_name", "elo", "num_runs", "avg_utility", "std_utility"],
    )
    write_csv(
        output_dir / "utility_vs_elo_complete_models_only.csv",
        complete_only_rows,
        ["model", "model_short", "elo_model_name", "elo", "num_runs", "avg_utility", "std_utility"],
    )
    write_csv(
        output_dir / "utility_vs_elo_by_competition_index.csv",
        by_ci_rows,
        [
            "competition_index",
            "model",
            "model_short",
            "elo_model_name",
            "elo",
            "num_runs",
            "avg_utility",
            "std_utility",
        ],
    )
    write_csv(
        output_dir / "utility_vs_elo_by_alpha_sigma.csv",
        by_alpha_sigma_rows,
        [
            "alpha",
            "sigma",
            "model",
            "model_short",
            "elo_model_name",
            "elo",
            "num_runs",
            "avg_utility",
            "std_utility",
        ],
    )

    make_overall_plot(
        all_rows,
        output_dir / "utility_vs_elo_all_models.png",
        "Game 3: Mean Adversary Utility vs Chatbot Arena Elo\nAll models with any finished runs; averages span completed alpha/sigma settings and both model orders.",
        point_color="#2563eb",
    )
    make_overall_plot(
        complete_only_rows,
        output_dir / "utility_vs_elo_complete_models_only.png",
        "Game 3: Mean Adversary Utility vs Chatbot Arena Elo\nRestricted to adversary models with all expected Game 3 configs finished.",
        point_color="#059669",
    )
    make_competition_index_plot(
        by_ci_rows,
        output_dir / "utility_vs_elo_by_competition_index.png",
    )
    make_alpha_sigma_plot(
        by_alpha_sigma_rows,
        output_dir / "utility_vs_elo_by_alpha_sigma.png",
    )

    write_report(
        output_dir=output_dir,
        results_root=results_root,
        completion_summary=completion_summary,
        all_rows=all_rows,
        complete_only_rows=complete_only_rows,
        by_ci_rows=by_ci_rows,
        by_alpha_sigma_rows=by_alpha_sigma_rows,
        total_completed=total_completed,
        total_unfinished=total_unfinished,
        started_only=started_only,
        not_started=not_started,
    )

    print(f"Finished configs: {total_completed}")
    print(f"Unfinished configs: {total_unfinished}")
    print(f"Started but not finished: {started_only}")
    print(f"Not started: {not_started}")
    print(f"Models with any finished runs: {len(all_rows)}")
    print(f"Models fully finished: {len(complete_only_rows)}")
    print(f"Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()

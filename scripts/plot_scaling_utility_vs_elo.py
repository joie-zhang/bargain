#!/usr/bin/env python3
"""
Plot average realized utility or rounds-to-consensus against Chatbot Arena Elo
for selected models.

Example:
    python scripts/plot_scaling_utility_vs_elo.py \
        --results-root experiments/results/scaling_experiment_20260403_051515 \
        --elo-markdown docs/guides/chatbot_arena_elo_scores_2026_03_11.md \
        --models claude-opus-4-6 gpt-3.5-turbo-0125 grok-4 o3-mini-high \
        --output-dir experiments/results/scaling_experiment_20260403_051515/analysis

    python scripts/plot_scaling_utility_vs_elo.py \
        --results-root experiments/results/scaling_experiment_20260403_051515 \
        --elo-markdown docs/guides/chatbot_arena_elo_scores_2026_03_11.md \
        --models claude-opus-4-6 gpt-3.5-turbo-0125 grok-4 o3-mini-high \
        --output-dir experiments/results/scaling_experiment_20260403_051515/analysis \
        --by-competition-level

    python scripts/plot_scaling_utility_vs_elo.py \
        --results-root experiments/results/scaling_experiment_20260403_051515 \
        --elo-markdown docs/guides/chatbot_arena_elo_scores_2026_03_11.md \
        --models claude-opus-4-6 gpt-3.5-turbo-0125 grok-4 o3-mini-high \
        --output-dir experiments/results/scaling_experiment_20260403_051515/analysis \
        --metric rounds
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from textwrap import fill
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strong_models_experiment.analysis.active_model_roster import (
    canonical_model_name,
    filter_active_adversary_models,
    is_active_adversary_model,
)


DEFAULT_ELO_OVERRIDES = {
    # Inference: the local Elo file does not contain a plain "grok-4" row.
    # Use the closest listed family entry.
    "grok-4": ("grok-4.20-beta1", 1496),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--elo-markdown", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--by-competition-level", action="store_true")
    parser.add_argument("--aggregate-over-models", action="store_true")
    parser.add_argument(
        "--metric",
        choices=["utility", "rounds"],
        default="utility",
    )
    parser.add_argument(
        "--competition-smoothing-method",
        choices=["none", "ewm", "moving_average"],
        default="ewm",
        help="Trend overlay used for --by-competition-level plots.",
    )
    parser.add_argument(
        "--competition-smoothing-alpha",
        type=float,
        default=0.35,
        help="EWM alpha for --by-competition-level plots when smoothing method is ewm.",
    )
    parser.add_argument(
        "--competition-smoothing-window",
        type=int,
        default=3,
        help="Moving-average window for --by-competition-level plots when smoothing method is moving_average.",
    )
    return parser.parse_args()


def parse_elo_markdown(markdown_path: Path) -> Dict[str, int]:
    elo_by_model: Dict[str, int] = {}
    table_rows = [
        re.compile(r"^\|\s*\d+\s*\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|"),
        re.compile(r"^\|\s*\d+\s*\|\s*[^|]+?\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|"),
    ]
    for line in markdown_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        for table_row in table_rows:
            match = table_row.match(stripped)
            if not match:
                continue
            model_name = match.group(1).strip().strip("`")
            elo_by_model[model_name] = int(match.group(2))
            break
    return elo_by_model


def discover_result_files(results_root: Path) -> List[Path]:
    return sorted(results_root.rglob("experiment_results.json"))


def parse_pair_and_order(results_root: Path, result_path: Path) -> Tuple[str, str, str]:
    relative = result_path.relative_to(results_root)
    pair_dir = relative.parts[0]
    order_dir = relative.parts[1]
    weak_model, strong_model = pair_dir.split("_vs_", 1)
    return weak_model, strong_model, order_dir


def map_agents_to_models(
    result_payload: Dict[str, object],
    weak_model: str,
    strong_model: str,
    default_order: str,
) -> Dict[str, str]:
    config = result_payload.get("config") or {}
    if not isinstance(config, dict):
        config = {}
    actual_order = str(config.get("actual_order") or config.get("model_order") or default_order)
    if actual_order == "weak_first":
        return {"Agent_1": weak_model, "Agent_2": strong_model}
    return {"Agent_1": strong_model, "Agent_2": weak_model}


def collect_run_rows(results_root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for result_path in discover_result_files(results_root):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        final_utilities = payload.get("final_utilities") or {}
        if not isinstance(final_utilities, dict):
            continue

        config = payload.get("config") or {}
        if not isinstance(config, dict):
            config = {}

        weak_model, strong_model, order_dir = parse_pair_and_order(results_root, result_path)
        agent_to_model = map_agents_to_models(payload, weak_model, strong_model, order_dir)
        competition_level = float(config.get("competition_level", 0.0))
        consensus_reached = bool(payload.get("consensus_reached"))
        final_round = payload.get("final_round")

        for agent_id, utility in final_utilities.items():
            model_name = agent_to_model.get(agent_id)
            if model_name is None:
                continue
            canonical_model = canonical_model_name(model_name)
            if not is_active_adversary_model(canonical_model):
                continue
            row: Dict[str, object] = {
                "model": canonical_model,
                "utility": float(utility),
                "competition_level": competition_level,
            }
            if consensus_reached and final_round is not None:
                row["rounds_to_consensus"] = float(final_round)
            rows.append(row)

    return rows


def metric_spec(metric: str) -> Dict[str, object]:
    if metric == "rounds":
        return {
            "value_key": "rounds_to_consensus",
            "avg_key": "avg_rounds_to_consensus",
            "std_key": "std_rounds_to_consensus",
            "title": "Game 1: Mean Rounds to Consensus vs Chatbot Arena Elo",
            "title_by_comp": "Game 1: Mean Rounds to Consensus vs Chatbot Arena Elo",
            "aggregate_title": "Game 1: Mean Rounds to Consensus vs Competition Level",
            "subtitle": "Averages include competition level, speaking order, and discussion-turn setting.",
            "title_by_comp_subtitle": "Each curve fixes one competition level; averages include speaking order and discussion-turn setting.",
            "aggregate_subtitle": "Aggregated over all active adversary models, speaking orders, and discussion-turn settings.",
            "y_label": "Mean Rounds to Consensus",
            "output_stem": "average_rounds_to_consensus_vs_elo",
            "reference_line": None,
        }
    return {
        "value_key": "utility",
        "avg_key": "avg_utility",
        "std_key": "std_utility",
        "title": "Game 1: Mean Adversary Utility vs Chatbot Arena Elo",
        "title_by_comp": "Game 1: Mean Adversary Utility vs Chatbot Arena Elo",
        "aggregate_title": "Game 1: Mean Adversary Utility vs Competition Level",
        "subtitle": "Averages include competition level, speaking order, and discussion-turn setting.",
        "title_by_comp_subtitle": "Each curve fixes one competition level; averages include speaking order and discussion-turn setting.",
        "aggregate_subtitle": "Aggregated over all active adversary models, speaking orders, and discussion-turn settings.",
        "y_label": "Mean Adversary Utility",
        "output_stem": "average_utility_vs_elo",
        "reference_line": 50.0,
    }


def competition_level_output_stem(spec: Dict[str, object]) -> str:
    stem = spec["output_stem"]
    if stem.endswith("_vs_elo"):
        return stem[: -len("_vs_elo")]
    return stem


def format_competition_level(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


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


def smooth_series(
    values: List[float],
    method: str,
    alpha: float,
    window: int,
) -> np.ndarray:
    smoothed = np.asarray(values, dtype=float)
    if smoothed.size < 2 or method == "none":
        return smoothed.copy()
    if method == "moving_average":
        averaged = np.empty_like(smoothed)
        for idx in range(smoothed.size):
            start = max(0, idx - window + 1)
            averaged[idx] = smoothed[start : idx + 1].mean()
        return averaged
    if method == "ewm":
        averaged = np.empty_like(smoothed)
        averaged[0] = smoothed[0]
        for idx in range(1, smoothed.size):
            averaged[idx] = alpha * smoothed[idx] + (1.0 - alpha) * averaged[idx - 1]
        return averaged
    raise ValueError(f"Unsupported smoothing method: {method}")


def competition_level_plot_subtitle(
    smoothing_method: str,
    smoothing_alpha: float,
    smoothing_window: int,
) -> str:
    averaging_note = "Averages include speaking order and discussion-turn setting."
    if smoothing_method == "ewm":
        return (
            f"Faint lines show raw per-Elo means; bold lines show Elo-ordered EWM trends "
            f"(alpha={smoothing_alpha:.2f}). {averaging_note}"
        )
    if smoothing_method == "moving_average":
        return (
            f"Faint lines show raw per-Elo means; bold lines show trailing moving-average "
            f"trends (window={smoothing_window}). {averaging_note}"
        )
    return f"Curves show raw per-Elo means for each competition level. {averaging_note}"


def resolve_elo(
    model_name: str,
    elo_by_model: Dict[str, int],
) -> Tuple[str, int]:
    if model_name in elo_by_model:
        return model_name, elo_by_model[model_name]
    if model_name in DEFAULT_ELO_OVERRIDES:
        return DEFAULT_ELO_OVERRIDES[model_name]
    raise KeyError(f"No Elo found for model: {model_name}")


def summarize_models(
    run_rows: List[Dict[str, object]],
    elo_by_model: Dict[str, int],
    selected_models: Iterable[str],
    spec: Dict[str, object],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    grouped_values: Dict[str, List[float]] = defaultdict(list)
    for run_row in run_rows:
        if spec["value_key"] not in run_row:
            continue
        grouped_values[str(run_row["model"])].append(float(run_row[spec["value_key"]]))

    for model_name in selected_models:
        values = grouped_values.get(model_name, [])
        if not values:
            continue
        elo_model_name, elo = resolve_elo(model_name, elo_by_model)
        rows.append(
            {
                "model": model_name,
                "elo_model_name": elo_model_name,
                "elo": elo,
                "num_runs": len(values),
                spec["avg_key"]: mean(values),
                spec["std_key"]: pstdev(values) if len(values) > 1 else 0.0,
            }
        )

    return sorted(rows, key=lambda row: row["elo"])


def summarize_models_by_competition_level(
    run_rows: List[Dict[str, object]],
    elo_by_model: Dict[str, int],
    selected_models: Iterable[str],
    spec: Dict[str, object],
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[float, str], List[float]] = defaultdict(list)
    for run_row in run_rows:
        if spec["value_key"] not in run_row:
            continue
        key = (float(run_row["competition_level"]), str(run_row["model"]))
        grouped[key].append(float(run_row[spec["value_key"]]))

    rows: List[Dict[str, object]] = []
    for competition_level in sorted({float(run_row["competition_level"]) for run_row in run_rows}):
        for model_name in selected_models:
            values = grouped.get((competition_level, model_name), [])
            if not values:
                continue
            elo_model_name, elo = resolve_elo(model_name, elo_by_model)
            rows.append(
                {
                    "competition_level": competition_level,
                    "model": model_name,
                    "elo_model_name": elo_model_name,
                    "elo": elo,
                    "num_runs": len(values),
                    spec["avg_key"]: mean(values),
                    spec["std_key"]: pstdev(values) if len(values) > 1 else 0.0,
                }
            )

    return sorted(rows, key=lambda row: (float(row["competition_level"]), int(row["elo"])))


def summarize_competition_levels_aggregated_over_models(
    run_rows: List[Dict[str, object]],
    selected_models: Iterable[str],
    spec: Dict[str, object],
) -> List[Dict[str, object]]:
    selected = set(selected_models)
    grouped: Dict[float, List[float]] = defaultdict(list)

    for run_row in run_rows:
        if str(run_row["model"]) not in selected:
            continue
        if spec["value_key"] not in run_row:
            continue
        grouped[float(run_row["competition_level"])].append(float(run_row[spec["value_key"]]))

    rows: List[Dict[str, object]] = []
    for competition_level in sorted(grouped):
        values = grouped[competition_level]
        rows.append(
            {
                "competition_level": competition_level,
                "num_runs": len(values),
                spec["avg_key"]: mean(values),
                spec["std_key"]: pstdev(values) if len(values) > 1 else 0.0,
            }
        )

    return rows


def write_summary_csv(
    rows: List[Dict[str, object]],
    output_path: Path,
    spec: Dict[str, object],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if rows and "model" not in rows[0]:
        fieldnames = [
            "competition_level",
            "num_runs",
            spec["avg_key"],
            spec["std_key"],
        ]
    else:
        fieldnames = [
            "model",
            "elo_model_name",
            "elo",
            "num_runs",
            spec["avg_key"],
            spec["std_key"],
        ]
        if rows and "competition_level" in rows[0]:
            fieldnames = ["competition_level"] + fieldnames
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_plot(
    rows: List[Dict[str, object]],
    output_path: Path,
    spec: Dict[str, object],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    xs = [float(row["elo"]) for row in rows]
    ys = [float(row[spec["avg_key"]]) for row in rows]
    labels = [str(row["model"]) for row in rows]
    line_x, line_y = best_fit_line(xs, ys)

    ax.scatter(
        xs,
        ys,
        s=140,
        color="#2563eb",
        edgecolors="#1e3a8a",
        linewidths=1.2,
        alpha=0.4,
    )
    ax.plot(line_x, line_y, color="#1d4ed8", linewidth=2.8, alpha=0.95)

    for row, label in zip(rows, labels):
        ax.annotate(
            f"{label}\nmean={row[spec['avg_key']]:.2f}",
            (float(row["elo"]), float(row[spec["avg_key"]])),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_title(f"{spec['title']}\n{spec['subtitle']}")
    ax.set_xlabel("Chatbot Arena Elo")
    ax.set_ylabel(spec["y_label"])
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_plot_by_competition_level(
    rows: List[Dict[str, object]],
    output_path: Path,
    spec: Dict[str, object],
    smoothing_method: str,
    smoothing_alpha: float,
    smoothing_window: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfdff")
    competition_levels = sorted({float(row["competition_level"]) for row in rows})
    color_map = plt.get_cmap("viridis")
    colors = {
        comp: color_map(idx / max(len(competition_levels) - 1, 1))
        for idx, comp in enumerate(competition_levels)
    }

    grouped: Dict[float, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["competition_level"])].append(row)

    xtick_rows: Dict[int, str] = {}
    show_smoothed_overlay = smoothing_method != "none"
    for comp in competition_levels:
        comp_rows = sorted(grouped[comp], key=lambda row: int(row["elo"]))
        xs = [float(row["elo"]) for row in comp_rows]
        y_raw = [float(row[spec["avg_key"]]) for row in comp_rows]
        y_smooth = smooth_series(
            y_raw,
            method=smoothing_method,
            alpha=smoothing_alpha,
            window=smoothing_window,
        )
        ax.plot(
            xs,
            y_raw,
            marker="o",
            markersize=4.5,
            linewidth=1.3 if show_smoothed_overlay else 2.4,
            color=colors[comp],
            alpha=0.24 if show_smoothed_overlay else 0.92,
            zorder=2,
            label="_nolegend_" if show_smoothed_overlay else f"competition={format_competition_level(comp)}",
        )
        if show_smoothed_overlay:
            ax.plot(
                xs,
                y_smooth,
                marker="o",
                markersize=6.6,
                linewidth=2.9,
                color=colors[comp],
                alpha=0.98,
                markeredgecolor="white",
                markeredgewidth=1.1,
                zorder=3,
                label=f"competition={format_competition_level(comp)}",
            )
        for row in comp_rows:
            xtick_rows[int(row["elo"])] = str(row["model"])

    x_positions = sorted(xtick_rows)
    if x_positions:
        x_padding = max(12, int(round((x_positions[-1] - x_positions[0]) * 0.03)))
        ax.set_xlim(x_positions[0] - x_padding, x_positions[-1] + x_padding)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    if spec["reference_line"] is not None:
        ax.axhline(
            float(spec["reference_line"]),
            color="#94a3b8",
            linestyle="--",
            linewidth=1.1,
            alpha=0.7,
            zorder=1,
        )
    ax.set_title(
        f"{spec['title_by_comp']}\n"
        f"{fill(competition_level_plot_subtitle(smoothing_method, smoothing_alpha, smoothing_window), width=120)}"
    )
    ax.set_xlabel("Chatbot Arena Elo")
    ax.set_ylabel(spec["y_label"])
    ax.grid(True, color="#cbd5e1", alpha=0.35, linewidth=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="Competition", frameon=True, facecolor="white", framealpha=0.95)

    ordered_models = ", ".join(f"{xtick_rows[x]} ({x})" for x in x_positions)
    fig.text(
        0.5,
        0.01,
        fill(f"Models by Elo: {ordered_models}", width=95),
        ha="center",
        va="bottom",
        fontsize=8.7,
        color="#475569",
    )

    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_plot_over_competition_levels(
    rows: List[Dict[str, object]],
    output_path: Path,
    spec: Dict[str, object],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    xs = [float(row["competition_level"]) for row in rows]
    ys = [float(row[spec["avg_key"]]) for row in rows]

    ax.plot(
        xs,
        ys,
        marker="o",
        markersize=8,
        linewidth=2.4,
        color="#2563eb",
    )

    for row in rows:
        ax.annotate(
            f"{row[spec['avg_key']]:.2f}",
            (float(row["competition_level"]), float(row[spec["avg_key"]])),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    ax.set_title(f"{spec['aggregate_title']}\n{spec['aggregate_subtitle']}")
    ax.set_xlabel("Competition Level")
    ax.set_ylabel(spec["y_label"])
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.competition_smoothing_method == "ewm" and not 0.0 < args.competition_smoothing_alpha <= 1.0:
        raise SystemExit("--competition-smoothing-alpha must be in the interval (0, 1].")
    if args.competition_smoothing_method == "moving_average" and args.competition_smoothing_window < 1:
        raise SystemExit("--competition-smoothing-window must be at least 1.")
    spec = metric_spec(args.metric)
    elo_by_model = parse_elo_markdown(args.elo_markdown)
    run_rows = collect_run_rows(args.results_root)
    selected_models = filter_active_adversary_models(args.models)
    rows = summarize_models(run_rows, elo_by_model, selected_models, spec)

    if not rows:
        raise SystemExit("No matching model rows were found in the completed results.")

    analysis_dir = args.output_dir
    if args.aggregate_over_models:
        aggregate_stem = competition_level_output_stem(spec)
        png_path = analysis_dir / f"{aggregate_stem}_vs_competition_level_aggregated_over_models.png"
        csv_path = analysis_dir / f"{aggregate_stem}_vs_competition_level_aggregated_over_models.csv"
        aggregate_rows = summarize_competition_levels_aggregated_over_models(run_rows, selected_models, spec)
        if not aggregate_rows:
            raise SystemExit("No aggregate competition-level rows were found in the completed results.")
        write_summary_csv(aggregate_rows, csv_path, spec)
        make_plot_over_competition_levels(aggregate_rows, png_path, spec)
        print(f"Wrote plot: {png_path}")
        print(f"Wrote summary: {csv_path}")
        for row in aggregate_rows:
            print(
                f"competition={format_competition_level(float(row['competition_level']))}: "
                f"{spec['avg_key']}={row[spec['avg_key']]:.4f}, n={row['num_runs']}"
            )
        return

    if args.by_competition_level:
        png_path = analysis_dir / f"{spec['output_stem']}_by_competition_level.png"
        csv_path = analysis_dir / f"{spec['output_stem']}_by_competition_level.csv"
        by_comp_rows = summarize_models_by_competition_level(run_rows, elo_by_model, selected_models, spec)
        if not by_comp_rows:
            raise SystemExit("No competition-level rows were found in the completed results.")
        write_summary_csv(by_comp_rows, csv_path, spec)
        make_plot_by_competition_level(
            by_comp_rows,
            png_path,
            spec,
            smoothing_method=args.competition_smoothing_method,
            smoothing_alpha=args.competition_smoothing_alpha,
            smoothing_window=args.competition_smoothing_window,
        )
        print(f"Wrote plot: {png_path}")
        print(f"Wrote summary: {csv_path}")
        for row in by_comp_rows:
            inferred = ""
            if row["model"] != row["elo_model_name"]:
                inferred = f" (Elo mapped to {row['elo_model_name']})"
            print(
                f"competition={format_competition_level(float(row['competition_level']))} | {row['model']}: "
                f"elo={row['elo']}, {spec['avg_key']}={row[spec['avg_key']]:.4f}, "
                f"n={row['num_runs']}{inferred}"
            )
        return

    png_path = analysis_dir / f"{spec['output_stem']}.png"
    csv_path = analysis_dir / f"{spec['output_stem']}.csv"

    write_summary_csv(rows, csv_path, spec)
    make_plot(rows, png_path, spec)

    print(f"Wrote plot: {png_path}")
    print(f"Wrote summary: {csv_path}")
    for row in rows:
        inferred = ""
        if row["model"] != row["elo_model_name"]:
            inferred = f" (Elo mapped to {row['elo_model_name']})"
        print(
            f"{row['model']}: elo={row['elo']}, {spec['avg_key']}={row[spec['avg_key']]:.4f}, "
            f"n={row['num_runs']}{inferred}"
        )


if __name__ == "__main__":
    main()

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
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    return parser.parse_args()


def parse_elo_markdown(markdown_path: Path) -> Dict[str, int]:
    elo_by_model: Dict[str, int] = {}
    table_row = re.compile(
        r"^\|\s*\d+\s*\|\s*[^|]+\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|"
    )
    for line in markdown_path.read_text(encoding="utf-8").splitlines():
        match = table_row.match(line.strip())
        if not match:
            continue
        elo_by_model[match.group(1).strip()] = int(match.group(2))
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
            row: Dict[str, object] = {
                "model": model_name,
                "utility": float(utility),
                "competition_level": competition_level,
            }
            if consensus_reached and final_round is not None:
                row["rounds_to_consensus"] = float(final_round)
            rows.append(row)

    return rows


def metric_spec(metric: str) -> Dict[str, str]:
    if metric == "rounds":
        return {
            "value_key": "rounds_to_consensus",
            "avg_key": "avg_rounds_to_consensus",
            "std_key": "std_rounds_to_consensus",
            "title": "Average Rounds To Consensus vs Chatbot Arena Elo",
            "title_by_comp": "Average Rounds To Consensus vs Chatbot Arena Elo, Stratified by Competition Level",
            "y_label": "Average Rounds To Consensus",
            "output_stem": "average_rounds_to_consensus_vs_elo",
        }
    return {
        "value_key": "utility",
        "avg_key": "avg_utility",
        "std_key": "std_utility",
        "title": "Average Utility vs Chatbot Arena Elo",
        "title_by_comp": "Average Utility vs Chatbot Arena Elo, Stratified by Competition Level",
        "y_label": "Average Realized Utility",
        "output_stem": "average_utility_vs_elo",
    }


def competition_level_output_stem(spec: Dict[str, str]) -> str:
    stem = spec["output_stem"]
    if stem.endswith("_vs_elo"):
        return stem[: -len("_vs_elo")]
    return stem


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
    spec: Dict[str, str],
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
    spec: Dict[str, str],
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
    spec: Dict[str, str],
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
    spec: Dict[str, str],
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
    spec: Dict[str, str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    xs = [float(row["elo"]) for row in rows]
    ys = [float(row[spec["avg_key"]]) for row in rows]
    labels = [str(row["model"]) for row in rows]

    ax.scatter(xs, ys, s=140, color="#2563eb", edgecolors="#1e3a8a", linewidths=1.2)

    for row, label in zip(rows, labels):
        ax.annotate(
            f"{label}\nmean={row[spec['avg_key']]:.2f}",
            (float(row["elo"]), float(row[spec["avg_key"]])),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_title(spec["title"])
    ax.set_xlabel("Chatbot Arena Elo")
    ax.set_ylabel(spec["y_label"])
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_plot_by_competition_level(
    rows: List[Dict[str, object]],
    output_path: Path,
    spec: Dict[str, str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6.5))
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
    for comp in competition_levels:
        comp_rows = sorted(grouped[comp], key=lambda row: int(row["elo"]))
        xs = [float(row["elo"]) for row in comp_rows]
        ys = [float(row[spec["avg_key"]]) for row in comp_rows]
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=7,
            linewidth=2.2,
            color=colors[comp],
            label=f"competition={comp:.1f}",
        )
        for row in comp_rows:
            xtick_rows[int(row["elo"])] = str(row["model"])
            ax.annotate(
                f"{row[spec['avg_key']]:.1f}",
                (float(row["elo"]), float(row[spec["avg_key"]])),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=colors[comp],
            )

    x_positions = sorted(xtick_rows)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{xtick_rows[x]}\n{x}" for x in x_positions])
    ax.set_title(spec["title_by_comp"])
    ax.set_xlabel("Chatbot Arena Elo")
    ax.set_ylabel(spec["y_label"])
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(title="Curve", frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_plot_over_competition_levels(
    rows: List[Dict[str, object]],
    output_path: Path,
    spec: Dict[str, str],
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

    ax.set_title(f"{spec['y_label']} vs Competition Level, Aggregated Over Models")
    ax.set_xlabel("Competition Level")
    ax.set_ylabel(spec["y_label"])
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    spec = metric_spec(args.metric)
    elo_by_model = parse_elo_markdown(args.elo_markdown)
    run_rows = collect_run_rows(args.results_root)
    rows = summarize_models(run_rows, elo_by_model, args.models, spec)

    if not rows:
        raise SystemExit("No matching model rows were found in the completed results.")

    analysis_dir = args.output_dir
    if args.aggregate_over_models:
        aggregate_stem = competition_level_output_stem(spec)
        png_path = analysis_dir / f"{aggregate_stem}_vs_competition_level_aggregated_over_models.png"
        csv_path = analysis_dir / f"{aggregate_stem}_vs_competition_level_aggregated_over_models.csv"
        aggregate_rows = summarize_competition_levels_aggregated_over_models(run_rows, args.models, spec)
        if not aggregate_rows:
            raise SystemExit("No aggregate competition-level rows were found in the completed results.")
        write_summary_csv(aggregate_rows, csv_path, spec)
        make_plot_over_competition_levels(aggregate_rows, png_path, spec)
        print(f"Wrote plot: {png_path}")
        print(f"Wrote summary: {csv_path}")
        for row in aggregate_rows:
            print(
                f"competition={row['competition_level']:.1f}: "
                f"{spec['avg_key']}={row[spec['avg_key']]:.4f}, n={row['num_runs']}"
            )
        return

    if args.by_competition_level:
        png_path = analysis_dir / f"{spec['output_stem']}_by_competition_level.png"
        csv_path = analysis_dir / f"{spec['output_stem']}_by_competition_level.csv"
        by_comp_rows = summarize_models_by_competition_level(run_rows, elo_by_model, args.models, spec)
        if not by_comp_rows:
            raise SystemExit("No competition-level rows were found in the completed results.")
        write_summary_csv(by_comp_rows, csv_path, spec)
        make_plot_by_competition_level(by_comp_rows, png_path, spec)
        print(f"Wrote plot: {png_path}")
        print(f"Wrote summary: {csv_path}")
        for row in by_comp_rows:
            inferred = ""
            if row["model"] != row["elo_model_name"]:
                inferred = f" (Elo mapped to {row['elo_model_name']})"
            print(
                f"competition={row['competition_level']:.1f} | {row['model']}: "
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

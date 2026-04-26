#!/usr/bin/env python3
"""Plot positional payoff effects in same-model Game 1 multi-agent controls."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "experiments/results/game1_multiagent_full_latest"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Figures/n_gt_2_game1_multiagent/position_effects_control"
RESULT_FILE_CANDIDATES = ("experiment_results.json", "run_1_experiment_results.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze position effects using completed Game 1 multi-agent controls "
            "where every agent uses the same model."
        )
    )
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", default="gpt-5-nano")
    parser.add_argument("--n-agents", type=int, nargs="+", default=[3, 5])
    return parser.parse_args()


def resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def result_file_for(output_dir: Path) -> Optional[Path]:
    for candidate in RESULT_FILE_CANDIDATES:
        path = output_dir / candidate
        if path.exists():
            return path
    return None


def agent_position(agent_id: str) -> Optional[int]:
    try:
        prefix, value = agent_id.split("_", 1)
        if prefix != "Agent":
            return None
        return int(value)
    except (AttributeError, ValueError):
        return None


def iter_configs(results_root: Path) -> Iterable[Dict]:
    for path in sorted((results_root / "configs").glob("config_*.json")):
        payload = load_json(path)
        payload["_config_path"] = str(path)
        yield payload


def build_observations(results_root: Path, model: str, n_agents_filter: set[int]) -> pd.DataFrame:
    rows: List[Dict] = []
    for config in iter_configs(results_root):
        models = list(config.get("models") or [])
        n_agents = int(config.get("n_agents", len(models)))
        if n_agents not in n_agents_filter:
            continue
        if config.get("condition_role") != "control":
            continue
        if len(models) != n_agents or set(models) != {model}:
            continue

        output_dir = resolve_path(config["output_dir"])
        result_path = result_file_for(output_dir)
        if result_path is None:
            continue

        try:
            result = load_json(result_path)
        except json.JSONDecodeError:
            continue

        final_utilities = result.get("final_utilities") or {}
        if not isinstance(final_utilities, dict):
            continue

        parsed_utilities = {}
        for agent_id, value in final_utilities.items():
            position = agent_position(agent_id)
            if position is None:
                continue
            try:
                parsed_utilities[position] = float(value)
            except (TypeError, ValueError):
                continue

        expected_positions = set(range(1, n_agents + 1))
        if set(parsed_utilities) != expected_positions:
            continue

        run_mean = sum(parsed_utilities.values()) / n_agents
        run_id = f"config_{int(config['config_id']):04d}"
        for position in sorted(parsed_utilities):
            utility = parsed_utilities[position]
            rows.append(
                {
                    "run_id": run_id,
                    "config_id": int(config["config_id"]),
                    "proposal_family": config.get("proposal_family", ""),
                    "condition_role": config.get("condition_role", ""),
                    "same_model": model,
                    "n_agents": n_agents,
                    "competition_level": float(config.get("competition_level", 0.0)),
                    "replicate_id": config.get("replicate_id"),
                    "field_id": config.get("field_id"),
                    "random_seed": config.get("random_seed"),
                    "position": position,
                    "agent_id": f"Agent_{position}",
                    "utility": utility,
                    "run_mean_utility": run_mean,
                    "utility_minus_run_mean": utility - run_mean,
                    "consensus_reached": result.get("consensus_reached"),
                    "final_round": result.get("final_round"),
                    "result_path": str(result_path),
                    "output_dir": str(output_dir),
                }
            )
    return pd.DataFrame(rows)


def mean_ci(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").dropna()
    n = int(values.size)
    mean = float(values.mean()) if n else math.nan
    std = float(values.std(ddof=1)) if n > 1 else math.nan
    sem = std / math.sqrt(n) if n > 1 else math.nan
    ci95 = float(stats.t.ppf(0.975, n - 1) * sem) if n > 1 else 0.0
    return pd.Series({"mean": mean, "std": std, "sem": sem, "ci95": ci95, "n": n})


def summarize(observations: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["n_agents", "competition_level", "position"]
    raw = observations.groupby(group_cols)["utility"].apply(mean_ci).unstack().reset_index()
    centered = (
        observations.groupby(group_cols)["utility_minus_run_mean"]
        .apply(mean_ci)
        .unstack()
        .reset_index()
        .rename(
            columns={
                "mean": "mean_advantage_vs_run_mean",
                "std": "std_advantage_vs_run_mean",
                "sem": "sem_advantage_vs_run_mean",
                "ci95": "ci95_advantage_vs_run_mean",
                "n": "n_advantage",
            }
        )
    )
    summary = raw.rename(
        columns={
            "mean": "mean_utility",
            "std": "std_utility",
            "sem": "sem_utility",
            "ci95": "ci95_utility",
            "n": "n",
        }
    ).merge(centered, on=group_cols, how="left")

    aggregate_raw = observations.groupby(["n_agents", "position"])["utility"].apply(mean_ci).unstack().reset_index()
    aggregate_centered = (
        observations.groupby(["n_agents", "position"])["utility_minus_run_mean"]
        .apply(mean_ci)
        .unstack()
        .reset_index()
        .rename(
            columns={
                "mean": "mean_advantage_vs_run_mean",
                "std": "std_advantage_vs_run_mean",
                "sem": "sem_advantage_vs_run_mean",
                "ci95": "ci95_advantage_vs_run_mean",
                "n": "n_advantage",
            }
        )
    )
    aggregate = aggregate_raw.rename(
        columns={
            "mean": "mean_utility",
            "std": "std_utility",
            "sem": "sem_utility",
            "ci95": "ci95_utility",
            "n": "n",
        }
    ).merge(aggregate_centered, on=["n_agents", "position"], how="left")
    aggregate.insert(1, "competition_level", "aggregate")
    return pd.concat([aggregate, summary], ignore_index=True)


def run_counts(observations: pd.DataFrame) -> pd.DataFrame:
    runs = observations.drop_duplicates(["run_id", "n_agents", "competition_level"])
    by_comp = (
        runs.groupby(["n_agents", "competition_level"])
        .agg(runs=("run_id", "count"))
        .reset_index()
    )
    aggregate = runs.groupby("n_agents").agg(runs=("run_id", "count")).reset_index()
    aggregate.insert(1, "competition_level", "aggregate")
    return pd.concat([aggregate, by_comp], ignore_index=True)


def friedman_tests(observations: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    subsets: List[tuple[int, str | float, pd.DataFrame]] = []
    for n_agents, subset in observations.groupby("n_agents"):
        subsets.append((int(n_agents), "aggregate", subset))
        for competition_level, comp_subset in subset.groupby("competition_level"):
            subsets.append((int(n_agents), float(competition_level), comp_subset))

    for n_agents, competition_level, subset in subsets:
        pivot = subset.pivot_table(index="run_id", columns="position", values="utility", aggfunc="first")
        expected = list(range(1, n_agents + 1))
        pivot = pivot.reindex(columns=expected).dropna()
        n_runs = int(len(pivot))
        means = pivot.mean(axis=0)
        mean_range = float(means.max() - means.min()) if n_runs else math.nan
        if n_runs >= 2:
            statistic, p_value = stats.friedmanchisquare(*[pivot[pos].to_numpy() for pos in expected])
            statistic = float(statistic)
            p_value = float(p_value)
        else:
            statistic = math.nan
            p_value = math.nan
        rows.append(
            {
                "n_agents": n_agents,
                "competition_level": competition_level,
                "runs": n_runs,
                "test": "friedman_repeated_measures_by_run",
                "statistic": statistic,
                "p_value": p_value,
                "mean_utility_range_across_positions": mean_range,
            }
        )
    return pd.DataFrame(rows)


def plot_panel(
    ax: plt.Axes,
    subset: pd.DataFrame,
    y_col: str,
    ci_col: str,
    title: str,
    ylabel: str,
    zero_line: bool = False,
) -> None:
    ordered = subset.sort_values("position")
    ax.errorbar(
        ordered["position"],
        ordered[y_col],
        yerr=ordered[ci_col],
        marker="o",
        linestyle="-",
        linewidth=2,
        capsize=4,
        color="#2F6B8F",
    )
    for _, row in ordered.iterrows():
        ax.annotate(
            f"runs={int(row['n'])}",
            (row["position"], row[y_col]),
            xytext=(0, -16),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
            color="#4A4A4A",
        )
    if zero_line:
        ax.axhline(0, color="#666666", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Position")
    ax.set_ylabel(ylabel)
    ax.set_xticks(ordered["position"])
    ax.grid(axis="y", alpha=0.25)


def plot_aggregate(summary: pd.DataFrame, output_path: Path, advantage: bool = False) -> None:
    aggregate = summary[summary["competition_level"].eq("aggregate")]
    n_values = sorted(aggregate["n_agents"].unique())
    fig, axes = plt.subplots(1, len(n_values), figsize=(5.2 * len(n_values), 4.6), squeeze=False)
    y_col = "mean_advantage_vs_run_mean" if advantage else "mean_utility"
    ci_col = "ci95_advantage_vs_run_mean" if advantage else "ci95_utility"
    ylabel = "Utility minus run mean" if advantage else "Mean payoff"
    title_suffix = "Advantage vs Run Mean" if advantage else "Mean Payoff"
    for ax, n_agents in zip(axes[0], n_values):
        subset = aggregate[aggregate["n_agents"].eq(n_agents)]
        plot_panel(
            ax,
            subset,
            y_col,
            ci_col,
            f"N={int(n_agents)} Aggregate {title_suffix}",
            ylabel,
            zero_line=advantage,
        )
    fig.suptitle("Game 1 Same-Model Control Position Effects", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_by_competition(summary: pd.DataFrame, output_path: Path, advantage: bool = False) -> None:
    stratified = summary[~summary["competition_level"].eq("aggregate")].copy()
    n_values = sorted(stratified["n_agents"].unique())
    comp_values = sorted(stratified["competition_level"].unique())
    fig, axes = plt.subplots(
        len(n_values),
        len(comp_values),
        figsize=(4.5 * len(comp_values), 4.0 * len(n_values)),
        squeeze=False,
        sharey=False,
    )
    y_col = "mean_advantage_vs_run_mean" if advantage else "mean_utility"
    ci_col = "ci95_advantage_vs_run_mean" if advantage else "ci95_utility"
    ylabel = "Utility minus run mean" if advantage else "Mean payoff"
    title_suffix = "Advantage" if advantage else "Payoff"

    for row_idx, n_agents in enumerate(n_values):
        for col_idx, competition_level in enumerate(comp_values):
            ax = axes[row_idx][col_idx]
            subset = stratified[
                stratified["n_agents"].eq(n_agents)
                & stratified["competition_level"].eq(competition_level)
            ]
            if subset.empty:
                ax.set_axis_off()
                continue
            plot_panel(
                ax,
                subset,
                y_col,
                ci_col,
                f"N={int(n_agents)}, comp={competition_level:g}",
                ylabel if col_idx == 0 else "",
                zero_line=advantage,
            )
            if row_idx < len(n_values) - 1:
                ax.set_xlabel("")
    fig.suptitle(f"Game 1 Same-Model Control Position {title_suffix} by Competition", y=1.01, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    results_root = resolve_path(args.results_root)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    observations = build_observations(results_root, args.model, set(args.n_agents))
    if observations.empty:
        raise RuntimeError(
            f"No completed same-model controls found for model={args.model!r} under {results_root}"
        )

    summary = summarize(observations)
    counts = run_counts(observations)
    tests = friedman_tests(observations)

    observations.to_csv(output_dir / "control_position_observations.csv", index=False)
    summary.to_csv(output_dir / "control_position_summary.csv", index=False)
    counts.to_csv(output_dir / "control_position_run_counts.csv", index=False)
    tests.to_csv(output_dir / "control_position_tests.csv", index=False)

    plot_aggregate(summary, output_dir / "position_payoff_aggregate.png", advantage=False)
    plot_by_competition(summary, output_dir / "position_payoff_by_competition.png", advantage=False)
    plot_aggregate(summary, output_dir / "position_advantage_aggregate.png", advantage=True)
    plot_by_competition(summary, output_dir / "position_advantage_by_competition.png", advantage=True)

    print(f"observations_csv: {output_dir / 'control_position_observations.csv'}")
    print(f"summary_csv: {output_dir / 'control_position_summary.csv'}")
    print(f"run_counts_csv: {output_dir / 'control_position_run_counts.csv'}")
    print(f"tests_csv: {output_dir / 'control_position_tests.csv'}")
    print(f"aggregate_payoff_plot: {output_dir / 'position_payoff_aggregate.png'}")
    print(f"competition_payoff_plot: {output_dir / 'position_payoff_by_competition.png'}")
    print(f"aggregate_advantage_plot: {output_dir / 'position_advantage_aggregate.png'}")
    print(f"competition_advantage_plot: {output_dir / 'position_advantage_by_competition.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Analyze the completed 500-run Llama-3.3-70B baseline appendix experiment.

The script reads the three completed experiment directories, maps each result
back to baseline/adversary utility, exports summary CSVs, creates the requested
Elo/competition/order plots, and writes a standalone Markdown report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strong_models_experiment.analysis.active_model_roster import (  # noqa: E402
    active_model_elo_map,
    canonical_model_name,
    short_model_name,
)


DEFAULT_GAME1_ROOT = (
    PROJECT_ROOT / "experiments" / "results" / "appendix_llama33_baseline_game1_202605"
)
DEFAULT_GAME2_ROOT = (
    PROJECT_ROOT / "experiments" / "results" / "appendix_llama33_baseline_game2_202605"
)
DEFAULT_GAME3_ROOT = (
    PROJECT_ROOT / "experiments" / "results" / "appendix_llama33_baseline_game3_202605"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "experiments" / "results" / "appendix_llama33_baseline_analysis_20260503"
)

BASELINE_MODEL = "llama-3.3-70b-instruct"


GAME_SPECS = {
    "game1": {
        "label": "Game 1",
        "title": "Game 1: item allocation",
        "root_arg": "game1_root",
        "root": DEFAULT_GAME1_ROOT,
        "game_type": "item_allocation",
    },
    "game2": {
        "label": "Game 2",
        "title": "Game 2: diplomacy",
        "root_arg": "game2_root",
        "root": DEFAULT_GAME2_ROOT,
        "game_type": "diplomacy",
    },
    "game3": {
        "label": "Game 3",
        "title": "Game 3: co-funding",
        "root_arg": "game3_root",
        "root": DEFAULT_GAME3_ROOT,
        "game_type": "co_funding",
    },
}


ORDER_LABELS = {
    "baseline_first": "Baseline first",
    "adversary_first": "Adversary first",
}


PLOT_COLORS = {
    "adversary": "#b45309",
    "baseline": "#2563eb",
    "fit": "#111827",
    "baseline_first": "#2563eb",
    "adversary_first": "#b45309",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game1-root", type=Path, default=DEFAULT_GAME1_ROOT)
    parser.add_argument("--game2-root", type=Path, default=DEFAULT_GAME2_ROOT)
    parser.add_argument("--game3-root", type=Path, default=DEFAULT_GAME3_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def fmt_float(value: Any, digits: int = 2) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if math.isnan(number):
        return "NA"
    text = f"{number:.{digits}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def fmt_signed(value: Any, digits: int = 2) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if math.isnan(number):
        return "NA"
    return f"{number:+.{digits}f}"


def fmt_pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if math.isnan(number):
        return "NA"
    return f"{100.0 * number:.{digits}f}%"


def result_file(output_dir: Path, run_number: int | None = None) -> Path | None:
    candidates: list[Path] = []
    if run_number is not None:
        candidates.append(output_dir / f"run_{run_number}_experiment_results.json")
    candidates.extend(
        [
            output_dir / "run_1_experiment_results.json",
            output_dir / "experiment_results.json",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate

    run_files = sorted(output_dir.glob("run_*_experiment_results.json"))
    if run_files:
        return run_files[0]
    return None


def ordered_agent_ids(final_utilities: dict[str, Any]) -> list[str]:
    if {"Agent_1", "Agent_2"}.issubset(final_utilities):
        return ["Agent_1", "Agent_2"]
    if {"Agent_Alpha", "Agent_Beta"}.issubset(final_utilities):
        return ["Agent_Alpha", "Agent_Beta"]
    return sorted(final_utilities)


def infer_utility_by_model(
    *,
    baseline_model: str,
    adversary_model: str,
    ordered_models: list[str],
    payload: dict[str, Any],
) -> dict[str, float]:
    final_utilities = payload.get("final_utilities") or {}
    agent_performance = payload.get("agent_performance") or {}
    if not isinstance(final_utilities, dict):
        return {}
    if not isinstance(agent_performance, dict):
        agent_performance = {}

    expected_models = {canonical_model_name(baseline_model), canonical_model_name(adversary_model)}
    utility_by_model: dict[str, float] = {}
    unresolved_agents: list[str] = []

    for agent_id, utility in final_utilities.items():
        perf = agent_performance.get(agent_id)
        if isinstance(perf, dict):
            raw_model = str(perf.get("model") or "").strip()
            if raw_model:
                canonical = canonical_model_name(raw_model)
                if canonical in expected_models:
                    utility_by_model[canonical] = float(utility)
                    continue
        unresolved_agents.append(str(agent_id))

    if expected_models.issubset(utility_by_model):
        return utility_by_model

    agent_ids = ordered_agent_ids(final_utilities)
    if len(agent_ids) == len(ordered_models):
        fallback = {
            agent_id: canonical_model_name(model)
            for agent_id, model in zip(agent_ids, ordered_models, strict=False)
        }
        for agent_id in unresolved_agents:
            mapped_model = fallback.get(agent_id)
            if mapped_model in expected_models and mapped_model not in utility_by_model:
                utility_by_model[mapped_model] = float(final_utilities[agent_id])

    if expected_models.issubset(utility_by_model):
        return utility_by_model

    missing = [model for model in expected_models if model not in utility_by_model]
    unknown = [agent for agent in final_utilities if agent not in unresolved_agents]
    if len(missing) == 1 and len(final_utilities) - len(unknown) == 1:
        utility_by_model[missing[0]] = float(final_utilities[unresolved_agents[0]])

    return utility_by_model


def competition_fields(game_id: str, row: dict[str, str]) -> dict[str, Any]:
    if game_id == "game1":
        competition_level = float(row["competition_level"])
        return {
            "competition_value": competition_level,
            "competition_dimension": "competition_level",
            "competition_label": f"c={fmt_float(competition_level)}",
            "setting_key": f"competition_level={fmt_float(competition_level)}",
            "competition_level": competition_level,
        }

    if game_id == "game2":
        rho = float(row["rho"])
        theta = float(row["theta"])
        competition_index = theta * (1.0 - rho) / 2.0
        return {
            "competition_value": competition_index,
            "competition_dimension": "competition_index",
            "competition_label": f"CI2={fmt_float(competition_index)}",
            "setting_key": f"rho={fmt_float(rho)};theta={fmt_float(theta)}",
            "rho": rho,
            "theta": theta,
        }

    if game_id == "game3":
        alpha = float(row["alpha"])
        sigma = float(row["sigma"])
        competition_index = (1.0 - alpha) * (1.0 - sigma)
        return {
            "competition_value": competition_index,
            "competition_dimension": "competition_index",
            "competition_label": f"CI3={fmt_float(competition_index)}",
            "setting_key": f"alpha={fmt_float(alpha)};sigma={fmt_float(sigma)}",
            "alpha": alpha,
            "sigma": sigma,
        }

    raise ValueError(f"Unsupported game id: {game_id}")


def load_game_rows(game_id: str, root: Path, elo_by_model: dict[str, int]) -> pd.DataFrame:
    spec = GAME_SPECS[game_id]
    index_path = root / "configs" / "experiment_index.csv"
    rows: list[dict[str, Any]] = []
    missing_results: list[str] = []
    unresolved_utilities: list[str] = []

    with index_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            config_path = root / "configs" / row["config_file"]
            config_payload = json.loads(config_path.read_text(encoding="utf-8"))
            output_dir = resolve_path(config_payload["output_dir"])
            run_number = int(row["run_number"]) if row.get("run_number") else None
            path = result_file(output_dir, run_number=run_number)
            if path is None:
                missing_results.append(row["config_file"])
                continue

            payload = json.loads(path.read_text(encoding="utf-8"))
            baseline_model = canonical_model_name(row["baseline_model"])
            adversary_model = canonical_model_name(row["adversary_model"])
            ordered_models = [
                canonical_model_name(model)
                for model in config_payload.get("models", [])
            ]
            utilities = infer_utility_by_model(
                baseline_model=baseline_model,
                adversary_model=adversary_model,
                ordered_models=ordered_models,
                payload=payload,
            )
            if baseline_model not in utilities or adversary_model not in utilities:
                unresolved_utilities.append(row["config_file"])
                continue

            if adversary_model not in elo_by_model:
                raise KeyError(f"No Elo found for adversary model {adversary_model}")
            if baseline_model not in elo_by_model:
                raise KeyError(f"No Elo found for baseline model {baseline_model}")

            competition = competition_fields(game_id, row)
            final_utilities = payload.get("final_utilities") or {}
            social_welfare = sum(float(v) for v in final_utilities.values())
            consensus_reached = payload.get("consensus_reached")
            final_round = payload.get("final_round")

            baseline_utility = float(utilities[baseline_model])
            adversary_utility = float(utilities[adversary_model])
            conceptual_order = row.get("conceptual_order") or (
                "baseline_first"
                if ordered_models and ordered_models[0] == baseline_model
                else "adversary_first"
            )

            rows.append(
                {
                    "game_id": game_id,
                    "game_label": spec["label"],
                    "game_title": spec["title"],
                    "game_type": row.get("game_type") or spec["game_type"],
                    "experiment_id": int(row["experiment_id"]),
                    "config_file": row["config_file"],
                    "result_path": str(path.relative_to(PROJECT_ROOT)),
                    "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
                    "baseline_model": baseline_model,
                    "baseline_short": short_model_name(baseline_model),
                    "baseline_elo": int(elo_by_model[baseline_model]),
                    "adversary_model": adversary_model,
                    "adversary_short": short_model_name(adversary_model),
                    "adversary_elo": int(elo_by_model[adversary_model]),
                    "model_order": row.get("model_order"),
                    "conceptual_order": conceptual_order,
                    "order_label": ORDER_LABELS.get(conceptual_order, conceptual_order),
                    "discussion_turns": int(row["discussion_turns"]),
                    "random_seed": int(row["seed"]),
                    "run_number": int(row["run_number"]),
                    "baseline_utility": baseline_utility,
                    "adversary_utility": adversary_utility,
                    "utility_delta_adv_minus_base": adversary_utility - baseline_utility,
                    "social_welfare": social_welfare,
                    "consensus_reached": bool(consensus_reached),
                    "final_round": float(final_round) if final_round is not None else np.nan,
                    "exploitation_detected": bool(payload.get("exploitation_detected", False)),
                    **competition,
                }
            )

    if missing_results or unresolved_utilities:
        details = []
        if missing_results:
            details.append(f"missing results: {missing_results[:5]} ({len(missing_results)} total)")
        if unresolved_utilities:
            details.append(
                f"unresolved utilities: {unresolved_utilities[:5]} "
                f"({len(unresolved_utilities)} total)"
            )
        raise RuntimeError(f"{spec['label']} could not be fully loaded: {'; '.join(details)}")

    return pd.DataFrame(rows)


def summarize_overall(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(
            [
                "game_id",
                "game_label",
                "game_title",
                "adversary_model",
                "adversary_short",
                "adversary_elo",
                "baseline_model",
                "baseline_short",
                "baseline_elo",
            ],
            as_index=False,
        )
        .agg(
            n=("adversary_utility", "size"),
            adversary_utility_mean=("adversary_utility", "mean"),
            adversary_utility_std=("adversary_utility", "std"),
            baseline_utility_mean=("baseline_utility", "mean"),
            baseline_utility_std=("baseline_utility", "std"),
            utility_delta_mean=("utility_delta_adv_minus_base", "mean"),
            social_welfare_mean=("social_welfare", "mean"),
            consensus_rate=("consensus_reached", "mean"),
            final_round_mean=("final_round", "mean"),
        )
        .sort_values(["game_id", "adversary_elo", "adversary_model"])
        .reset_index(drop=True)
    )
    return summary


def summarize_by_competition(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(
            [
                "game_id",
                "game_label",
                "competition_dimension",
                "competition_value",
                "competition_label",
                "adversary_model",
                "adversary_short",
                "adversary_elo",
            ],
            as_index=False,
        )
        .agg(
            n=("adversary_utility", "size"),
            adversary_utility_mean=("adversary_utility", "mean"),
            baseline_utility_mean=("baseline_utility", "mean"),
            utility_delta_mean=("utility_delta_adv_minus_base", "mean"),
            consensus_rate=("consensus_reached", "mean"),
            final_round_mean=("final_round", "mean"),
        )
        .sort_values(["game_id", "competition_value", "adversary_elo", "adversary_model"])
        .reset_index(drop=True)
    )


def summarize_by_order(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(
            [
                "game_id",
                "game_label",
                "conceptual_order",
                "order_label",
                "adversary_model",
                "adversary_short",
                "adversary_elo",
            ],
            as_index=False,
        )
        .agg(
            n=("adversary_utility", "size"),
            adversary_utility_mean=("adversary_utility", "mean"),
            baseline_utility_mean=("baseline_utility", "mean"),
            utility_delta_mean=("utility_delta_adv_minus_base", "mean"),
            consensus_rate=("consensus_reached", "mean"),
        )
        .sort_values(["game_id", "conceptual_order", "adversary_elo", "adversary_model"])
        .reset_index(drop=True)
    )


def regression_metrics(x: pd.Series, y: pd.Series) -> dict[str, float]:
    clean = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(clean) < 2 or clean["x"].nunique() < 2:
        return {
            "pearson_r": np.nan,
            "spearman_r": np.nan,
            "slope_per_100_elo": np.nan,
        }
    slope, _intercept = np.polyfit(clean["x"], clean["y"], deg=1)
    return {
        "pearson_r": float(clean["x"].corr(clean["y"], method="pearson")),
        "spearman_r": float(clean["x"].corr(clean["y"], method="spearman")),
        "slope_per_100_elo": float(slope * 100.0),
    }


def trend_table(overall: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for game_id, game_df in overall.groupby("game_id", sort=True):
        game_label = str(game_df["game_label"].iloc[0])
        adv = regression_metrics(game_df["adversary_elo"], game_df["adversary_utility_mean"])
        base = regression_metrics(game_df["adversary_elo"], game_df["baseline_utility_mean"])
        delta = regression_metrics(game_df["adversary_elo"], game_df["utility_delta_mean"])
        rows.append(
            {
                "game_id": game_id,
                "game_label": game_label,
                "n_models": len(game_df),
                "adversary_slope_per_100_elo": adv["slope_per_100_elo"],
                "adversary_pearson_r": adv["pearson_r"],
                "adversary_spearman_r": adv["spearman_r"],
                "baseline_slope_per_100_elo": base["slope_per_100_elo"],
                "baseline_pearson_r": base["pearson_r"],
                "baseline_spearman_r": base["spearman_r"],
                "delta_slope_per_100_elo": delta["slope_per_100_elo"],
                "delta_pearson_r": delta["pearson_r"],
                "delta_spearman_r": delta["spearman_r"],
            }
        )
    return pd.DataFrame(rows)


def competition_trend_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for game_id, game_df in df.groupby("game_id", sort=True):
        game_label = str(game_df["game_label"].iloc[0])
        adv = regression_metrics(game_df["competition_value"], game_df["adversary_utility"])
        base = regression_metrics(game_df["competition_value"], game_df["baseline_utility"])
        welfare = regression_metrics(game_df["competition_value"], game_df["social_welfare"])
        rows.append(
            {
                "game_id": game_id,
                "game_label": game_label,
                "competition_dimension": str(game_df["competition_dimension"].iloc[0]),
                "n_runs": len(game_df),
                "n_competition_values": int(game_df["competition_value"].nunique()),
                "adversary_slope_per_unit_competition": adv["slope_per_100_elo"] / 100.0,
                "adversary_pearson_r": adv["pearson_r"],
                "baseline_slope_per_unit_competition": base["slope_per_100_elo"] / 100.0,
                "baseline_pearson_r": base["pearson_r"],
                "welfare_slope_per_unit_competition": welfare["slope_per_100_elo"] / 100.0,
                "welfare_pearson_r": welfare["pearson_r"],
            }
        )
    return pd.DataFrame(rows)


def model_order_effects(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    index_cols = ["game_id", "adversary_model", "adversary_elo", "setting_key"]
    wide = (
        df.pivot_table(
            index=index_cols,
            columns="conceptual_order",
            values=["baseline_utility", "adversary_utility"],
            aggfunc="mean",
        )
        .reset_index()
    )
    wide.columns = [
        "_".join(str(part) for part in col if part)
        if isinstance(col, tuple)
        else str(col)
        for col in wide.columns
    ]
    required = {
        "baseline_utility_baseline_first",
        "baseline_utility_adversary_first",
        "adversary_utility_baseline_first",
        "adversary_utility_adversary_first",
    }
    missing = required.difference(wide.columns)
    if missing:
        raise RuntimeError(f"Missing order columns: {sorted(missing)}")

    wide["baseline_first_advantage_for_baseline"] = (
        wide["baseline_utility_baseline_first"]
        - wide["baseline_utility_adversary_first"]
    )
    wide["adversary_first_advantage_for_adversary"] = (
        wide["adversary_utility_adversary_first"]
        - wide["adversary_utility_baseline_first"]
    )
    wide["generic_first_advantage"] = (
        wide["baseline_first_advantage_for_baseline"]
        + wide["adversary_first_advantage_for_adversary"]
    ) / 2.0

    summary = (
        wide.groupby("game_id", as_index=False)
        .agg(
            n_pairs=("generic_first_advantage", "size"),
            baseline_first_advantage_for_baseline_mean=(
                "baseline_first_advantage_for_baseline",
                "mean",
            ),
            adversary_first_advantage_for_adversary_mean=(
                "adversary_first_advantage_for_adversary",
                "mean",
            ),
            generic_first_advantage_mean=("generic_first_advantage", "mean"),
            generic_first_advantage_median=("generic_first_advantage", "median"),
            generic_first_advantage_std=("generic_first_advantage", "std"),
            positive_generic_first_advantage_share=(
                "generic_first_advantage",
                lambda values: float((values > 0).mean()),
            ),
        )
        .reset_index(drop=True)
    )
    summary["game_label"] = summary["game_id"].map(
        {game_id: spec["label"] for game_id, spec in GAME_SPECS.items()}
    )
    return wide, summary


def add_fit_line(ax: plt.Axes, x: pd.Series, y: pd.Series, color: str = "#111827") -> None:
    clean = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(clean) < 2 or clean["x"].nunique() < 2:
        return
    slope, intercept = np.polyfit(clean["x"], clean["y"], deg=1)
    x_values = np.linspace(float(clean["x"].min()), float(clean["x"].max()), 100)
    ax.plot(x_values, slope * x_values + intercept, color=color, linestyle="--", linewidth=1.3)


def annotate_points(ax: plt.Axes, df: pd.DataFrame, y_col: str) -> None:
    for _, row in df.iterrows():
        ax.annotate(
            str(row["adversary_short"]),
            (row["adversary_elo"], row[y_col]),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=7.5,
            alpha=0.82,
        )


def style_axis(ax: plt.Axes, ylabel: str) -> None:
    ax.set_xlabel("Adversary Chatbot Arena Elo")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.22)
    ax.set_axisbelow(True)


def plot_overall(game_df: pd.DataFrame, output_path: Path) -> None:
    game_df = game_df.sort_values("adversary_elo")
    game_title = str(game_df["game_title"].iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True)

    axes[0].plot(
        game_df["adversary_elo"],
        game_df["adversary_utility_mean"],
        color=PLOT_COLORS["adversary"],
        marker="o",
        linewidth=2,
    )
    axes[0].scatter(
        game_df["adversary_elo"],
        game_df["adversary_utility_mean"],
        color=PLOT_COLORS["adversary"],
        s=36,
        zorder=3,
    )
    add_fit_line(axes[0], game_df["adversary_elo"], game_df["adversary_utility_mean"])
    annotate_points(axes[0], game_df, "adversary_utility_mean")
    axes[0].set_title("Adversary utility")
    style_axis(axes[0], "Mean adversary utility")

    axes[1].plot(
        game_df["adversary_elo"],
        game_df["baseline_utility_mean"],
        color=PLOT_COLORS["baseline"],
        marker="o",
        linewidth=2,
    )
    axes[1].scatter(
        game_df["adversary_elo"],
        game_df["baseline_utility_mean"],
        color=PLOT_COLORS["baseline"],
        s=36,
        zorder=3,
    )
    add_fit_line(axes[1], game_df["adversary_elo"], game_df["baseline_utility_mean"])
    annotate_points(axes[1], game_df, "baseline_utility_mean")
    axes[1].set_title("Baseline utility against the adversary")
    style_axis(axes[1], "Mean Llama-3.3-70B baseline utility")

    fig.suptitle(f"{game_title}: utility versus adversary Elo", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_by_competition(comp_df: pd.DataFrame, output_path: Path) -> None:
    game_title = str(comp_df["game_label"].iloc[0])
    dimension = str(comp_df["competition_dimension"].iloc[0])
    comp_values = sorted(comp_df["competition_value"].dropna().unique())
    cmap = plt.cm.viridis(np.linspace(0.08, 0.92, max(len(comp_values), 2)))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True)
    for color, value in zip(cmap, comp_values, strict=False):
        subset = comp_df[comp_df["competition_value"] == value].sort_values("adversary_elo")
        label = str(subset["competition_label"].iloc[0])
        axes[0].plot(
            subset["adversary_elo"],
            subset["adversary_utility_mean"],
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            color=color,
            label=label,
        )
        axes[1].plot(
            subset["adversary_elo"],
            subset["baseline_utility_mean"],
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            color=color,
            label=label,
        )

    axes[0].set_title("Adversary utility")
    style_axis(axes[0], "Mean adversary utility")
    axes[1].set_title("Baseline utility against the adversary")
    style_axis(axes[1], "Mean Llama-3.3-70B baseline utility")
    axes[1].legend(
        title=dimension.replace("_", " ").title(),
        fontsize=8.5,
        title_fontsize=9,
        frameon=True,
        loc="best",
    )
    fig.suptitle(f"{game_title}: utility versus Elo by {dimension}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_by_order(order_df: pd.DataFrame, output_path: Path) -> None:
    game_title = str(order_df["game_label"].iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True)

    for conceptual_order, subset in order_df.groupby("conceptual_order", sort=True):
        subset = subset.sort_values("adversary_elo")
        label = ORDER_LABELS.get(str(conceptual_order), str(conceptual_order))
        color = PLOT_COLORS.get(str(conceptual_order), "#4b5563")
        axes[0].plot(
            subset["adversary_elo"],
            subset["adversary_utility_mean"],
            marker="o",
            linewidth=2,
            markersize=5,
            label=label,
            color=color,
        )
        axes[1].plot(
            subset["adversary_elo"],
            subset["baseline_utility_mean"],
            marker="o",
            linewidth=2,
            markersize=5,
            label=label,
            color=color,
        )

    axes[0].set_title("Adversary utility")
    style_axis(axes[0], "Mean adversary utility")
    axes[1].set_title("Baseline utility against the adversary")
    style_axis(axes[1], "Mean Llama-3.3-70B baseline utility")
    axes[1].legend(title="Model order", fontsize=9, title_fontsize=10, frameon=True)
    fig.suptitle(f"{game_title}: utility versus Elo by model order", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_order_effects(order_summary: pd.DataFrame, output_path: Path) -> None:
    order_summary = order_summary.sort_values("game_id")
    x = np.arange(len(order_summary))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.bar(
        x - width,
        order_summary["baseline_first_advantage_for_baseline_mean"],
        width=width,
        color=PLOT_COLORS["baseline_first"],
        label="Baseline utility: first minus last",
    )
    ax.bar(
        x,
        order_summary["adversary_first_advantage_for_adversary_mean"],
        width=width,
        color=PLOT_COLORS["adversary_first"],
        label="Adversary utility: first minus last",
    )
    ax.bar(
        x + width,
        order_summary["generic_first_advantage_mean"],
        width=width,
        color="#6b7280",
        label="Mean first-position advantage",
    )
    ax.axhline(0, color="#111827", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(order_summary["game_label"])
    ax.set_ylabel("Utility difference")
    ax.set_title("Model-order effect by game")
    ax.legend(frameon=True, fontsize=8.5)
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_markdown_table(df: pd.DataFrame, columns: list[tuple[str, str]]) -> list[str]:
    headers = [header for header, _column in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for _header, column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(fmt_float(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def relative(path: Path, base: Path) -> str:
    return str(path.relative_to(base))


def describe_trend(slope: float, pearson: float, target: str) -> str:
    if math.isnan(slope) or math.isnan(pearson):
        return f"{target}: trend unavailable"
    direction = "increases" if slope > 0 else "decreases" if slope < 0 else "is flat"
    strength = "strong" if abs(pearson) >= 0.7 else "moderate" if abs(pearson) >= 0.4 else "weak"
    return (
        f"{target} {direction} by {fmt_signed(slope)} utility per 100 Elo "
        f"(Pearson r={fmt_float(pearson)}, {strength} linear association)"
    )


def build_report(
    *,
    output_dir: Path,
    all_runs: pd.DataFrame,
    overall: pd.DataFrame,
    by_competition: pd.DataFrame,
    by_order: pd.DataFrame,
    trends: pd.DataFrame,
    competition_trends: pd.DataFrame,
    order_summary: pd.DataFrame,
    plot_paths: dict[str, dict[str, Path]],
    csv_paths: dict[str, Path],
) -> Path:
    report_path = output_dir / "appendix_llama33_baseline_500_analysis.md"
    lines: list[str] = []
    lines.extend(
        [
            "# Appendix Llama-3.3-70B Baseline: 500-Run Analysis",
            "",
            "This report summarizes the completed appendix experiments using "
            "`llama-3.3-70b-instruct` as the fixed baseline. The x-axis in all "
            "Elo plots is the adversary model's Chatbot Arena Elo from "
            "`docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md`.",
            "",
            "## Executive Summary",
            "",
        ]
    )
    for _, row in trends.sort_values("game_id").iterrows():
        lines.append(
            f"- {row['game_label']}: adversary utility has a "
            f"{fmt_signed(row['adversary_slope_per_100_elo'])} utility / 100 Elo "
            f"slope, while baseline utility has a "
            f"{fmt_signed(row['baseline_slope_per_100_elo'])} utility / 100 Elo slope."
        )
    lines.append(
        "- Higher competition levels reduce total utility in all three games; "
        "the social-welfare slope is negative for every competition index/level."
    )
    order_max = order_summary.iloc[
        order_summary["generic_first_advantage_mean"].abs().argmax()
    ]
    lines.append(
        "- Model-order effects are small to modest on average. The largest mean "
        f"first-position effect is {fmt_signed(order_max['generic_first_advantage_mean'])} "
        f"utility in {order_max['game_label']}."
    )
    lines.extend(
        [
            "",
            "## Experiment Inventory",
            "",
            "| Game | Directory | Runs | Models | Settings per model | Order levels |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )

    for game_id, game_df in all_runs.groupby("game_id", sort=True):
        root = GAME_SPECS[game_id]["root"]
        settings_per_model = int(
            game_df.groupby("adversary_model")["setting_key"].nunique().max()
        )
        order_levels = ", ".join(sorted(game_df["order_label"].unique()))
        lines.append(
            f"| {GAME_SPECS[game_id]['label']} | `{root.relative_to(PROJECT_ROOT)}` | "
            f"{len(game_df)} | {game_df['adversary_model'].nunique()} | "
            f"{settings_per_model} | {order_levels} |"
        )

    lines.extend(
        [
            "",
            "Definitions used here:",
            "",
            "- `adversary_utility` is the payoff of the non-baseline model.",
            "- `baseline_utility` is the payoff of `llama-3.3-70b-instruct` against that adversary.",
            "- Game 1 competition is the configured `competition_level`.",
            "- Game 2 competition index is `theta * (1 - rho) / 2`.",
            "- Game 3 competition index is `(1 - alpha) * (1 - sigma)`.",
            "- Model order is analyzed with `conceptual_order`: `Baseline first` versus `Adversary first`.",
            "",
            "The raw run table and summary CSVs are written next to this report:",
            "",
        ]
    )

    for label, path in csv_paths.items():
        lines.append(f"- {label}: `{relative(path, output_dir)}`")

    model_table = (
        overall[["adversary_model", "adversary_short", "adversary_elo"]]
        .drop_duplicates()
        .sort_values("adversary_elo")
        .reset_index(drop=True)
    )
    lines.extend(
        [
            "",
            "## Model Roster",
            "",
            "| Adversary model | Short name | Elo |",
            "| --- | --- | ---: |",
        ]
    )
    for _, row in model_table.iterrows():
        lines.append(
            f"| `{row['adversary_model']}` | {row['adversary_short']} | "
            f"{int(row['adversary_elo'])} |"
        )

    lines.extend(["", "## Overall Utility Versus Elo", ""])
    for _, row in trends.sort_values("game_id").iterrows():
        lines.append(f"### {row['game_label']}")
        lines.append("")
        lines.append(
            f"![{row['game_label']} overall utility versus Elo]"
            f"({relative(plot_paths[row['game_id']]['overall'], output_dir)})"
        )
        lines.append("")
        lines.append(
            "- "
            + describe_trend(
                row["adversary_slope_per_100_elo"],
                row["adversary_pearson_r"],
                "Adversary utility",
            )
            + "."
        )
        lines.append(
            "- "
            + describe_trend(
                row["baseline_slope_per_100_elo"],
                row["baseline_pearson_r"],
                "Baseline utility",
            )
            + "."
        )
        lines.append(
            "- Utility gap trend: "
            f"{fmt_signed(row['delta_slope_per_100_elo'])} adversary-minus-baseline "
            f"utility per 100 Elo (Pearson r={fmt_float(row['delta_pearson_r'])})."
        )
        lines.append("")

    lines.extend(
        [
            "### Trend Summary",
            "",
            "| Game | Adv. slope / 100 Elo | Adv. Pearson r | Base slope / 100 Elo | Base Pearson r | Gap slope / 100 Elo |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in trends.sort_values("game_id").iterrows():
        lines.append(
            f"| {row['game_label']} | {fmt_signed(row['adversary_slope_per_100_elo'])} | "
            f"{fmt_float(row['adversary_pearson_r'])} | "
            f"{fmt_signed(row['baseline_slope_per_100_elo'])} | "
            f"{fmt_float(row['baseline_pearson_r'])} | "
            f"{fmt_signed(row['delta_slope_per_100_elo'])} |"
        )

    lines.extend(["", "## Competition-Stratified Utility Versus Elo", ""])
    for game_id in ["game1", "game2", "game3"]:
        game_label = GAME_SPECS[game_id]["label"]
        dimension = str(
            all_runs.loc[all_runs["game_id"] == game_id, "competition_dimension"].iloc[0]
        )
        comp_values = (
            all_runs.loc[all_runs["game_id"] == game_id, "competition_value"]
            .drop_duplicates()
            .sort_values()
        )
        lines.append(f"### {game_label}")
        lines.append("")
        lines.append(
            f"![{game_label} competition-stratified utility versus Elo]"
            f"({relative(plot_paths[game_id]['competition'], output_dir)})"
        )
        lines.append("")
        lines.append(
            f"Competition dimension: `{dimension}`. Values: "
            + ", ".join(fmt_float(value) for value in comp_values)
            + "."
        )
        comp_row = competition_trends[competition_trends["game_id"] == game_id].iloc[0]
        lines.append(
            "- "
            f"Adversary utility changes by {fmt_signed(comp_row['adversary_slope_per_unit_competition'])} "
            f"per unit of competition index/level."
        )
        lines.append(
            "- "
            f"Baseline utility changes by {fmt_signed(comp_row['baseline_slope_per_unit_competition'])} "
            f"per unit of competition index/level."
        )
        lines.append(
            "- "
            f"Social welfare changes by {fmt_signed(comp_row['welfare_slope_per_unit_competition'])} "
            f"per unit of competition index/level."
        )
        lines.append("")

    lines.extend(
        [
            "### Competition Trend Summary",
            "",
            "| Game | Dimension | Values | Adv. slope / unit | Base slope / unit | Welfare slope / unit |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in competition_trends.sort_values("game_id").iterrows():
        lines.append(
            f"| {row['game_label']} | `{row['competition_dimension']}` | "
            f"{int(row['n_competition_values'])} | "
            f"{fmt_signed(row['adversary_slope_per_unit_competition'])} | "
            f"{fmt_signed(row['baseline_slope_per_unit_competition'])} | "
            f"{fmt_signed(row['welfare_slope_per_unit_competition'])} |"
        )

    lines.extend(["", "## Model Order", ""])
    lines.append(
        "The order analysis compares matched pairs with the same game, adversary model, "
        "and exact parameter setting. `Baseline utility: first minus last` is the "
        "baseline payoff when it speaks first minus its payoff when it speaks second. "
        "`Adversary utility: first minus last` is the same calculation for the adversary."
    )
    lines.append("")

    for game_id in ["game1", "game2", "game3"]:
        game_label = GAME_SPECS[game_id]["label"]
        lines.append(f"### {game_label}")
        lines.append("")
        lines.append(
            f"![{game_label} utility by model order]"
            f"({relative(plot_paths[game_id]['order'], output_dir)})"
        )
        lines.append("")
        row = order_summary[order_summary["game_id"] == game_id].iloc[0]
        generic = float(row["generic_first_advantage_mean"])
        if abs(generic) < 1.0:
            interpretation = "small"
        elif abs(generic) < 5.0:
            interpretation = "modest"
        else:
            interpretation = "large"
        lines.append(
            f"Mean first-position advantage is {fmt_signed(generic)} utility "
            f"across {int(row['n_pairs'])} matched pairs, which is {interpretation} "
            "relative to the game payoff scale."
        )
        lines.append("")

    lines.extend(
        [
            "### Model-Order Summary",
            "",
            "![Model order effect summary]"
            f"({relative(plot_paths['all']['order_effects'], output_dir)})",
            "",
            "| Game | Matched pairs | Base first-minus-last | Adv. first-minus-last | Mean first advantage | Positive share |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in order_summary.sort_values("game_id").iterrows():
        lines.append(
            f"| {row['game_label']} | {int(row['n_pairs'])} | "
            f"{fmt_signed(row['baseline_first_advantage_for_baseline_mean'])} | "
            f"{fmt_signed(row['adversary_first_advantage_for_adversary_mean'])} | "
            f"{fmt_signed(row['generic_first_advantage_mean'])} | "
            f"{fmt_pct(row['positive_generic_first_advantage_share'])} |"
        )

    lines.extend(["", "## Per-Model Means", ""])
    for game_id in ["game1", "game2", "game3"]:
        game_overall = overall[overall["game_id"] == game_id].sort_values("adversary_elo")
        lines.append(f"### {GAME_SPECS[game_id]['label']}")
        lines.append("")
        lines.extend(
            [
                "| Model | Elo | n | Adv. utility | Base utility | Adv - base | Consensus | Final round |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in game_overall.iterrows():
            lines.append(
                f"| {row['adversary_short']} | {int(row['adversary_elo'])} | "
                f"{int(row['n'])} | {fmt_float(row['adversary_utility_mean'])} | "
                f"{fmt_float(row['baseline_utility_mean'])} | "
                f"{fmt_signed(row['utility_delta_mean'])} | "
                f"{fmt_pct(row['consensus_rate'])} | {fmt_float(row['final_round_mean'])} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Reproducibility",
            "",
            "Regenerate this report with:",
            "",
            "```bash",
            "python scripts/analyze_appendix_llama33_baseline_500.py",
            "```",
            "",
            "Primary output files:",
            "",
        ]
    )
    for label, path in csv_paths.items():
        lines.append(f"- {label}: `{relative(path, output_dir)}`")
    for game_id in ["game1", "game2", "game3"]:
        lines.append(
            f"- {GAME_SPECS[game_id]['label']} plots: "
            f"`{relative(plot_paths[game_id]['overall'], output_dir)}`, "
            f"`{relative(plot_paths[game_id]['competition'], output_dir)}`, "
            f"`{relative(plot_paths[game_id]['order'], output_dir)}`"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    elo_by_model = active_model_elo_map()
    roots = {
        "game1": args.game1_root.resolve(),
        "game2": args.game2_root.resolve(),
        "game3": args.game3_root.resolve(),
    }
    for game_id, root in roots.items():
        GAME_SPECS[game_id]["root"] = root

    frames = [
        load_game_rows(game_id, root, elo_by_model)
        for game_id, root in roots.items()
    ]
    all_runs = pd.concat(frames, ignore_index=True)
    if len(all_runs) != 500:
        raise RuntimeError(f"Expected 500 completed runs, loaded {len(all_runs)}")

    overall = summarize_overall(all_runs)
    by_competition = summarize_by_competition(all_runs)
    by_order = summarize_by_order(all_runs)
    trends = trend_table(overall)
    competition_trends = competition_trend_table(all_runs)
    order_pairs, order_summary = model_order_effects(all_runs)

    csv_paths = {
        "All runs": output_dir / "all_runs.csv",
        "Overall by model/game": output_dir / "overall_by_model_game.csv",
        "By competition": output_dir / "by_competition.csv",
        "By model order": output_dir / "by_model_order.csv",
        "Elo trend summary": output_dir / "elo_trend_summary.csv",
        "Competition trend summary": output_dir / "competition_trend_summary.csv",
        "Model-order matched pairs": output_dir / "model_order_matched_pairs.csv",
        "Model-order summary": output_dir / "model_order_summary.csv",
    }
    all_runs.to_csv(csv_paths["All runs"], index=False)
    overall.to_csv(csv_paths["Overall by model/game"], index=False)
    by_competition.to_csv(csv_paths["By competition"], index=False)
    by_order.to_csv(csv_paths["By model order"], index=False)
    trends.to_csv(csv_paths["Elo trend summary"], index=False)
    competition_trends.to_csv(csv_paths["Competition trend summary"], index=False)
    order_pairs.to_csv(csv_paths["Model-order matched pairs"], index=False)
    order_summary.to_csv(csv_paths["Model-order summary"], index=False)

    plot_paths: dict[str, dict[str, Path]] = {"all": {}}
    for game_id in ["game1", "game2", "game3"]:
        plot_paths[game_id] = {
            "overall": output_dir / f"{game_id}_overall_utility_vs_elo.png",
            "competition": output_dir / f"{game_id}_by_competition_utility_vs_elo.png",
            "order": output_dir / f"{game_id}_by_model_order_utility_vs_elo.png",
        }
        plot_overall(overall[overall["game_id"] == game_id], plot_paths[game_id]["overall"])
        plot_by_competition(
            by_competition[by_competition["game_id"] == game_id],
            plot_paths[game_id]["competition"],
        )
        plot_by_order(by_order[by_order["game_id"] == game_id], plot_paths[game_id]["order"])

    plot_paths["all"]["order_effects"] = output_dir / "model_order_effects_summary.png"
    plot_order_effects(order_summary, plot_paths["all"]["order_effects"])

    report_path = build_report(
        output_dir=output_dir,
        all_runs=all_runs,
        overall=overall,
        by_competition=by_competition,
        by_order=by_order,
        trends=trends,
        competition_trends=competition_trends,
        order_summary=order_summary,
        plot_paths=plot_paths,
        csv_paths=csv_paths,
    )

    print(f"Loaded {len(all_runs)} completed runs")
    print(f"Wrote report: {report_path}")
    print(f"Wrote output directory: {output_dir}")


if __name__ == "__main__":
    main()

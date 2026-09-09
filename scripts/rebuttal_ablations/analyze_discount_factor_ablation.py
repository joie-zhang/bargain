#!/usr/bin/env python3
"""Analyze the 360-run Game 1 gamma=0.5, 0.9, and 1.0 ablation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "discount_factor_ablation_game1_20260725"
)
DEFAULT_EXTENSION_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "discount_factor_ablation_game1_gamma_0p5_20260725"
)
DEFAULT_PREVIOUS_ROOT = (
    PROJECT_ROOT / "experiments" / "results" / "scaling_experiment_20260404_064451"
)
BASELINE_MODEL = "gpt-5-nano"

MODEL_LABELS = {
    "amazon-nova-micro-v1.0": "Nova Micro",
    "claude-3-haiku-20240307": "Claude 3 Haiku",
    "amazon-nova-pro-v1.0": "Nova Pro",
    "gpt-4o-mini-2024-07-18": "GPT-4o Mini",
    "deepseek-v3": "DeepSeek V3",
    "claude-sonnet-4-20250514": "Claude Sonnet 4",
    "deepseek-r1-0528": "DeepSeek R1",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gpt-5.4-high": "GPT-5.4 High",
    "claude-opus-4-6-thinking": "Claude Opus 4.6",
}

GAMMAS = [0.5, 0.9, 1.0]
GAMMA_COLORS = {0.5: "#059669", 0.9: "#2563eb", 1.0: "#dc2626"}

MODEL_ELOS = {
    "amazon-nova-micro-v1.0": 1240,
    "claude-3-haiku-20240307": 1260,
    "amazon-nova-pro-v1.0": 1290,
    "gpt-4o-mini-2024-07-18": 1317,
    "deepseek-v3": 1358,
    "claude-sonnet-4-20250514": 1389,
    "deepseek-r1-0528": 1422,
    "gemini-2.5-pro": 1448,
    "gpt-5.4-high": 1484,
    "claude-opus-4-6-thinking": 1504,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--extension-root", type=Path, default=DEFAULT_EXTENSION_ROOT)
    parser.add_argument("--previous-root", type=Path, default=DEFAULT_PREVIOUS_ROOT)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def find_result(output_dir: Path, run_number: int) -> Path | None:
    candidates = [
        output_dir / f"run_{run_number}_experiment_results.json",
        output_dir / "experiment_results.json",
    ]
    for path in candidates:
        if path.exists() and path.stat().st_size:
            return path
    matches = sorted(output_dir.glob("run_*_experiment_results.json"))
    return matches[0] if matches else None


def utility_by_model(
    payload: dict[str, Any],
    ordered_models: list[str],
) -> dict[str, float]:
    final_utilities = payload.get("final_utilities")
    if not isinstance(final_utilities, dict):
        return {}
    result: dict[str, float] = {}
    agent_utilities = {
        str(agent_id): float(utility)
        for agent_id, utility in final_utilities.items()
    }
    performance = payload.get("agent_performance")
    if isinstance(performance, dict):
        for agent_id, record in performance.items():
            if not isinstance(record, dict) or not record.get("model"):
                continue
            utility = final_utilities.get(agent_id, record.get("final_utility"))
            if utility is not None:
                agent_utilities[str(agent_id)] = float(utility)
                result[str(record["model"])] = float(utility)
    if all(model in result for model in ordered_models):
        return result
    agent_ids = sorted(
        agent_utilities,
        key=lambda value: int(value.split("_")[-1])
        if str(value).startswith("Agent_") and str(value).split("_")[-1].isdigit()
        else str(value),
    )
    if len(agent_ids) == len(ordered_models):
        for agent_id, model in zip(agent_ids, ordered_models, strict=True):
            result.setdefault(model, agent_utilities[agent_id])
    return result


def run_record(
    *,
    config: dict[str, Any],
    payload: dict[str, Any],
    result_path: Path,
    source: str,
) -> dict[str, Any]:
    ordered_models = [str(model) for model in config["models"]]
    utilities = utility_by_model(payload, ordered_models)
    baseline = str(config.get("baseline_model") or config.get("weak_model") or BASELINE_MODEL)
    adversary = str(config.get("adversary_model") or config.get("strong_model"))
    if baseline not in utilities or adversary not in utilities:
        raise ValueError(f"Could not map utilities in {result_path}")

    gamma = float(config.get("gamma_discount", 0.9))
    consensus = bool(payload.get("consensus_reached"))
    final_round = int(payload.get("final_round") or config.get("max_rounds") or 10)
    discount_multiplier = gamma ** max(0, final_round - 1) if consensus else 1.0
    baseline_utility = float(utilities[baseline])
    adversary_utility = float(utilities[adversary])
    allocation_baseline = baseline_utility / discount_multiplier if consensus else 0.0
    allocation_adversary = adversary_utility / discount_multiplier if consensus else 0.0

    return {
        "source": source,
        "experiment_id": int(config.get("experiment_id", -1)),
        "config_id": int(config.get("config_id", config.get("experiment_id", -1))),
        "result_path": str(result_path.relative_to(PROJECT_ROOT)),
        "baseline_model": baseline,
        "adversary_model": adversary,
        "adversary_label": MODEL_LABELS.get(adversary, adversary),
        "gamma_discount": gamma,
        "competition_level": float(config["competition_level"]),
        "model_order": str(config["model_order"]),
        "conceptual_order": str(
            config.get("conceptual_order")
            or ("baseline_first" if ordered_models[0] == baseline else "adversary_first")
        ),
        "random_seed": int(config["random_seed"]),
        "discussion_turns": int(config["discussion_turns"]),
        "run_number": int(config["run_number"]),
        "consensus_reached": consensus,
        "final_round": final_round,
        "round_one_agreement": consensus and final_round == 1,
        "discount_multiplier": discount_multiplier,
        "baseline_utility": baseline_utility,
        "adversary_utility": adversary_utility,
        "utility_delta_adv_minus_base": adversary_utility - baseline_utility,
        "social_welfare": baseline_utility + adversary_utility,
        "allocation_baseline_utility": allocation_baseline,
        "allocation_adversary_utility": allocation_adversary,
        "allocation_social_welfare": allocation_baseline + allocation_adversary,
    }


def load_ablation(
    run_root: Path,
    allow_incomplete: bool,
    expected: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    with (run_root / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        for manifest_row in csv.DictReader(handle):
            config_path = run_root / "configs" / manifest_row["config_file"]
            config = json.loads(config_path.read_text(encoding="utf-8"))
            output_dir = resolve(config["output_dir"])
            result_path = find_result(output_dir, int(config["run_number"]))
            if result_path is None:
                missing.append(manifest_row["config_file"])
                continue
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            rows.append(
                run_record(
                    config=config,
                    payload=payload,
                    result_path=result_path,
                    source="new_ablation",
                )
            )
    if missing and not allow_incomplete:
        raise RuntimeError(f"Missing {len(missing)} results: {missing[:10]}")
    frame = pd.DataFrame(rows)
    if not allow_incomplete and len(frame) != expected:
        raise RuntimeError(f"Expected {expected} complete runs, loaded {len(frame)}")
    return frame


def load_previous(previous_root: Path, adversaries: set[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for config_path in sorted((previous_root / "configs").glob("config_*.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        adversary = str(config.get("strong_model", ""))
        if adversary not in adversaries:
            continue
        if float(config.get("competition_level", -1)) not in {0.0, 0.5, 1.0}:
            continue
        if int(config.get("discussion_turns", -1)) != 2:
            continue
        if int(config.get("random_seed", -1)) != 42:
            continue
        config = dict(config)
        config.setdefault("baseline_model", config.get("weak_model", BASELINE_MODEL))
        config.setdefault("adversary_model", adversary)
        config.setdefault("gamma_discount", 0.9)
        output_dir = resolve(config["output_dir"])
        result_path = find_result(output_dir, int(config["run_number"]))
        if result_path is None:
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        rows.append(
            run_record(
                config=config,
                payload=payload,
                result_path=result_path,
                source="previous_gamma_0.9",
            )
        )
    return pd.DataFrame(rows)


def mean_ci(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return math.nan, math.nan
    mean = float(array.mean())
    if array.size < 2:
        return mean, 0.0
    return mean, float(1.96 * array.std(ddof=1) / math.sqrt(array.size))


def summarize(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metrics = [
        "baseline_utility",
        "adversary_utility",
        "social_welfare",
        "allocation_baseline_utility",
        "allocation_adversary_utility",
        "allocation_social_welfare",
        "final_round",
        "consensus_reached",
        "round_one_agreement",
    ]
    grouped = frame.groupby(group_cols, dropna=False)
    records: list[dict[str, Any]] = []
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_cols, keys, strict=True))
        record["n"] = len(group)
        for metric in metrics:
            values = group[metric].astype(float)
            record[f"{metric}_mean"] = values.mean()
            record[f"{metric}_std"] = values.std(ddof=1)
        records.append(record)
    return pd.DataFrame(records)


def paired_gamma(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["adversary_model", "competition_level", "model_order", "random_seed"]
    value_cols = [
        "baseline_utility",
        "adversary_utility",
        "social_welfare",
        "allocation_baseline_utility",
        "allocation_adversary_utility",
        "allocation_social_welfare",
        "final_round",
        "consensus_reached",
        "round_one_agreement",
    ]
    wide = frame.pivot(index=keys, columns="gamma_discount", values=value_cols)
    missing_gammas = set(GAMMAS) - set(wide.columns.get_level_values(1))
    if missing_gammas:
        raise RuntimeError(
            f"Missing gamma conditions required for paired analysis: {sorted(missing_gammas)}"
        )
    records: list[dict[str, Any]] = []
    for index, row in wide.iterrows():
        record = dict(zip(keys, index, strict=True))
        for metric in value_cols:
            for gamma in GAMMAS:
                record[f"{metric}_gamma_{gamma:.1f}"] = row[(metric, gamma)]
            record[f"{metric}_difference_0.5_minus_0.9"] = (
                row[(metric, 0.5)] - row[(metric, 0.9)]
            )
            record[f"{metric}_difference_1.0_minus_0.9"] = (
                row[(metric, 1.0)] - row[(metric, 0.9)]
            )
        records.append(record)
    result = pd.DataFrame(records)
    if len(result) != 120:
        raise RuntimeError(f"Expected 120 paired cells, found {len(result)}")
    return result


def replication_pairs(new: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    keys = ["adversary_model", "competition_level", "model_order", "random_seed"]
    metrics = [
        "baseline_utility",
        "adversary_utility",
        "social_welfare",
        "allocation_social_welfare",
        "final_round",
        "consensus_reached",
    ]
    current = new[
        (new["gamma_discount"] == 0.9)
        & (new["random_seed"] == 42)
    ][keys + metrics].copy()
    merged = previous[keys + metrics].merge(
        current,
        on=keys,
        how="inner",
        suffixes=("_previous", "_new"),
        validate="one_to_one",
    )
    for metric in metrics:
        merged[f"{metric}_difference_new_minus_previous"] = (
            merged[f"{metric}_new"].astype(float)
            - merged[f"{metric}_previous"].astype(float)
        )
    return merged


def plot_metric_overview(frame: pd.DataFrame, output: Path) -> None:
    specs = [
        ("adversary_utility", "Adversary realized utility"),
        ("baseline_utility", "GPT-5 Nano realized utility"),
        ("social_welfare", "Realized social welfare"),
        ("allocation_social_welfare", "Allocation welfare before discount"),
        ("final_round", "Final round"),
        ("consensus_reached", "Agreement rate"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    for ax, (metric, title) in zip(axes.flat, specs, strict=True):
        means, errors = [], []
        for gamma in GAMMAS:
            mean, error = mean_ci(frame.loc[frame.gamma_discount == gamma, metric].astype(float))
            means.append(mean)
            errors.append(error)
        ax.bar(
            [f"γ={gamma:g}" for gamma in GAMMAS],
            means,
            yerr=errors,
            color=[GAMMA_COLORS[gamma] for gamma in GAMMAS],
            alpha=0.88,
            capsize=4,
        )
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        if metric == "consensus_reached":
            ax.set_ylim(0, 1.05)
    fig.suptitle("Game 1 discount-factor ablation (360 runs; 95% CIs)", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_model_effects(paired: pd.DataFrame, output: Path) -> None:
    records: dict[str, list[tuple[str, float, float]]] = {}
    for comparison, column in [
        ("γ=0.5 − γ=0.9", "adversary_utility_difference_0.5_minus_0.9"),
        ("γ=1.0 − γ=0.9", "adversary_utility_difference_1.0_minus_0.9"),
    ]:
        comparison_records = []
        for model, group in paired.groupby("adversary_model"):
            mean, error = mean_ci(group[column])
            comparison_records.append((MODEL_LABELS.get(model, model), mean, error))
        records[comparison] = comparison_records
    labels = [MODEL_LABELS.get(model, model) for model in MODEL_LABELS]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 7))
    for offset, (comparison, color) in zip(
        [-0.13, 0.13],
        [("γ=0.5 − γ=0.9", GAMMA_COLORS[0.5]), ("γ=1.0 − γ=0.9", GAMMA_COLORS[1.0])],
        strict=True,
    ):
        lookup = {label: (mean, error) for label, mean, error in records[comparison]}
        means = [lookup[label][0] for label in labels]
        errors = [lookup[label][1] for label in labels]
        ax.errorbar(
            means,
            y + offset,
            xerr=errors,
            fmt="o",
            color=color,
            capsize=4,
            label=comparison,
        )
    ax.axvline(0, color="#111827", linewidth=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Paired adversary realized-utility change")
    ax.set_title("Discount-factor effects by target model")
    ax.legend()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_by_competition(frame: pd.DataFrame, output: Path) -> None:
    specs = [
        ("adversary_utility", "Adversary realized utility"),
        ("final_round", "Mean final round"),
        ("consensus_reached", "Agreement rate"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (metric, title) in zip(axes, specs, strict=True):
        for gamma in GAMMAS:
            subset = frame[frame.gamma_discount == gamma]
            xs, ys, errors = [], [], []
            for competition, group in subset.groupby("competition_level"):
                mean, error = mean_ci(group[metric].astype(float))
                xs.append(competition)
                ys.append(mean)
                errors.append(error)
            ax.errorbar(
                xs,
                ys,
                yerr=errors,
                marker="o",
                linewidth=2,
                capsize=3,
                label=f"γ={gamma:g}",
                color=GAMMA_COLORS[gamma],
            )
        ax.set_title(title)
        ax.set_xlabel("Competition level")
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle("Discount-factor effect across preference competition", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_round_distribution(frame: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.8))
    rounds = list(range(1, 11))
    width = 0.26
    for offset, gamma in zip([-width, 0.0, width], GAMMAS, strict=True):
        subset = frame[frame.gamma_discount == gamma]
        frequencies = [
            float((subset.final_round == round_number).mean())
            for round_number in rounds
        ]
        ax.bar(
            np.asarray(rounds) + offset,
            frequencies,
            width=width,
            label=f"γ={gamma:g}",
            color=GAMMA_COLORS[gamma],
            alpha=0.86,
        )
    ax.set_xticks(rounds)
    ax.set_xlabel("Final round")
    ax.set_ylabel("Share of runs")
    ax.set_title("Negotiation timing with and without time discount")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_effect_heatmap(paired: pd.DataFrame, output: Path) -> None:
    comparisons = [
        ("adversary_utility_difference_0.5_minus_0.9", "γ=0.5 − γ=0.9"),
        ("adversary_utility_difference_1.0_minus_0.9", "γ=1.0 − γ=0.9"),
    ]
    tables = [
        paired.pivot_table(
            index="adversary_model",
            columns="competition_level",
            values=column,
            aggfunc="mean",
        ).loc[list(MODEL_LABELS)]
        for column, _ in comparisons
    ]
    all_values = np.concatenate([table.to_numpy(dtype=float).ravel() for table in tables])
    limit = max(abs(float(np.nanmin(all_values))), abs(float(np.nanmax(all_values))), 1.0)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 7.2),
        sharey=True,
        layout="constrained",
    )
    image = None
    for ax, table, (_, title) in zip(axes, tables, comparisons, strict=True):
        values = table.to_numpy(dtype=float)
        image = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(
            range(len(table.columns)),
            [f"c={value:g}" for value in table.columns],
        )
        ax.set_yticks(
            range(len(table.index)),
            [MODEL_LABELS.get(value, value) for value in table.index],
        )
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                ax.text(
                    column,
                    row,
                    f"{values[row, column]:+.1f}",
                    ha="center",
                    va="center",
                )
        ax.set_title(title)
    assert image is not None
    fig.colorbar(image, ax=axes, label="Adversary realized-utility change", shrink=0.88)
    fig.suptitle("Paired discount-factor effects by model and competition", fontsize=15)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_replication(pairs: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    for ax, metric, title in [
        (axes[0], "adversary_utility", "Adversary utility"),
        (axes[1], "social_welfare", "Social welfare"),
    ]:
        x = pairs[f"{metric}_previous"].astype(float)
        y = pairs[f"{metric}_new"].astype(float)
        ax.scatter(x, y, color="#0f766e", alpha=0.75)
        lower = min(float(x.min()), float(y.min()))
        upper = max(float(x.max()), float(y.max()))
        ax.plot([lower, upper], [lower, upper], "--", color="#111827", linewidth=1)
        ax.set_xlabel("Previous γ=0.9 run")
        ax.set_ylabel("New γ=0.9 control")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    fig.suptitle("Matched replication check (same model, c, order, seed=42)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def scaling_summary(frame: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    series = [
        ("New γ=0.5", frame[frame.gamma_discount == 0.5]),
        ("New γ=0.9", frame[frame.gamma_discount == 0.9]),
        ("New γ=1.0", frame[frame.gamma_discount == 1.0]),
        ("Previous γ=0.9", previous),
    ]
    records: list[dict[str, Any]] = []
    for series_name, subset in series:
        for metric in ["adversary_utility", "allocation_adversary_utility"]:
            means = (
                subset.groupby("adversary_model", as_index=False)[metric]
                .mean()
                .assign(elo=lambda value: value.adversary_model.map(MODEL_ELOS))
                .dropna(subset=["elo"])
            )
            x = means.elo.to_numpy(dtype=float)
            y = means[metric].to_numpy(dtype=float)
            slope, intercept = np.polyfit(x, y, 1)
            records.append(
                {
                    "series": series_name,
                    "metric": metric,
                    "n_models": len(means),
                    "slope_per_100_elo": slope * 100,
                    "intercept": intercept,
                    "pearson_r": np.corrcoef(x, y)[0, 1],
                }
            )
    return pd.DataFrame(records)


def plot_scaling(frame: pd.DataFrame, previous: pd.DataFrame, output: Path) -> None:
    specs = [
        ("adversary_utility", "Adversary realized utility"),
        ("allocation_adversary_utility", "Adversary allocation utility"),
    ]
    series = [
        ("New γ=0.5", frame[frame.gamma_discount == 0.5], GAMMA_COLORS[0.5], "-"),
        ("New γ=0.9", frame[frame.gamma_discount == 0.9], GAMMA_COLORS[0.9], "-"),
        ("New γ=1.0", frame[frame.gamma_discount == 1.0], GAMMA_COLORS[1.0], "-"),
        ("Previous γ=0.9", previous, "#6b7280", "--"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    for ax, (metric, title) in zip(axes, specs, strict=True):
        for label, subset, color, linestyle in series:
            means = (
                subset.groupby("adversary_model", as_index=False)[metric]
                .mean()
                .assign(elo=lambda value: value.adversary_model.map(MODEL_ELOS))
                .dropna(subset=["elo"])
                .sort_values("elo")
            )
            x = means.elo.to_numpy(dtype=float)
            y = means[metric].to_numpy(dtype=float)
            slope, intercept = np.polyfit(x, y, 1)
            ax.scatter(x, y, color=color, alpha=0.82, s=35)
            ax.plot(
                x,
                slope * x + intercept,
                color=color,
                linestyle=linestyle,
                linewidth=2,
                label=f"{label} ({slope * 100:+.1f}/100 Elo)",
            )
        ax.set_title(title)
        ax.set_xlabel("Target-model Arena Elo")
        ax.set_ylabel("Mean utility")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    fig.suptitle("Strategic-performance scaling with and without time discount", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fmt(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def write_report(
    *,
    frame: pd.DataFrame,
    paired: pd.DataFrame,
    previous: pd.DataFrame,
    replication: pd.DataFrame,
    scaling: pd.DataFrame,
    output: Path,
) -> None:
    lines = [
        "# Game 1 discount-factor ablation",
        "",
        "## Completion",
        "",
        f"- New runs analyzed: **{len(frame)}/360**",
        f"- Exact three-gamma matched cells: **{len(paired)}/120**",
        f"- Previous gamma=0.9 matched controls available: **{len(replication)}/60**",
        "",
        "The experiment changes only the time discount and its truthful prompt description:",
        "γ=0.5, γ=0.9, and γ=1.0, crossed with ten target models, three competition",
        "levels, two speaking orders, and two matched seeds.",
        "",
        "## Main paired effects",
        "",
        "| Outcome | Comparison | Mean change | 95% CI half-width |",
        "|---|---|---:|---:|",
    ]
    for suffix, comparison in [
        ("0.5_minus_0.9", "γ=0.5 − γ=0.9"),
        ("1.0_minus_0.9", "γ=1.0 − γ=0.9"),
    ]:
        for metric, label in [
            ("adversary_utility", "Adversary realized utility"),
            ("baseline_utility", "GPT-5 Nano realized utility"),
            ("social_welfare", "Realized social welfare"),
            ("allocation_social_welfare", "Allocation welfare before discount"),
            ("final_round", "Final round"),
            ("consensus_reached", "Agreement probability"),
            ("round_one_agreement", "Round-one agreement probability"),
        ]:
            mean, error = mean_ci(paired[f"{metric}_difference_{suffix}"])
            lines.append(f"| {label} | {comparison} | {mean:+.3f} | {error:.3f} |")

    lines.extend(
        [
            "",
            "The distinction between realized utility and allocation utility matters:",
            "realized γ<1 utilities mechanically include `γ^(round−1)`, while",
            "allocation utilities divide that factor back out. The latter isolates whether",
            "changing time pressure changes the negotiated allocation itself.",
            "",
            "## Aggregate levels",
            "",
            "| γ | Adversary utility | Baseline utility | Welfare | Allocation welfare | Final round | Agreement | Round 1 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for gamma in GAMMAS:
        subset = frame[frame.gamma_discount == gamma]
        lines.append(
            f"| {gamma:g} | {fmt(subset.adversary_utility.mean())} | "
            f"{fmt(subset.baseline_utility.mean())} | {fmt(subset.social_welfare.mean())} | "
            f"{fmt(subset.allocation_social_welfare.mean())} | "
            f"{fmt(subset.final_round.mean())} | "
            f"{100 * subset.consensus_reached.mean():.1f}% | "
            f"{100 * subset.round_one_agreement.mean():.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Previous-run replication check",
            "",
        ]
    )
    if len(replication):
        for metric, label in [
            ("adversary_utility", "Adversary utility"),
            ("baseline_utility", "Baseline utility"),
            ("social_welfare", "Social welfare"),
            ("final_round", "Final round"),
            ("consensus_reached", "Agreement probability"),
        ]:
            difference = replication[f"{metric}_difference_new_minus_previous"]
            lines.append(
                f"- {label}: new minus previous matched mean = "
                f"**{difference.mean():+.3f}** (n={len(difference)})."
            )
    else:
        lines.append("- No previous matched results were found.")
    lines.extend(
        [
            "",
            "The historical differences show that the old controls are not exactly",
            "reproducible under current model-provider sampling. Accordingly, the main",
            "discount-factor estimate above uses the contemporaneous, seed-matched pairs.",
        ]
    )

    lines.extend(
        [
            "",
            "## Scaling versus target-model Elo",
            "",
            "| Series | Outcome | Slope per 100 Elo | Pearson r |",
            "|---|---|---:|---:|",
        ]
    )
    for row in scaling.itertuples(index=False):
        metric_label = (
            "Realized adversary utility"
            if row.metric == "adversary_utility"
            else "Allocation adversary utility"
        )
        lines.append(
            f"| {row.series} | {metric_label} | "
            f"{row.slope_per_100_elo:+.2f} | {row.pearson_r:+.2f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The realized-utility outcomes include the mechanical effect of discounting,",
            "whereas allocation utilities expose behavioral changes in the negotiated split.",
            "Relative to γ=0.9, γ=0.5 lowers realized welfare but leaves pre-discount",
            "allocation welfare, agreement probability, and final round statistically",
            "indistinguishable at the aggregate level. Target-model performance continues",
            "to rise with Elo in every condition; the γ=0.5 allocation-utility slope is",
            "positive but somewhat shallower and noisier than the γ=0.9 and γ=1.0 slopes.",
            "",
            "## Figures",
            "",
            "- `gamma_metric_overview.png`: aggregate outcome comparison.",
            "- `gamma_effect_by_model.png`: both paired adversary-utility contrasts by model.",
            "- `gamma_effect_by_model_and_competition.png`: both model × competition heatmaps.",
            "- `gamma_by_competition.png`: effects across competition levels.",
            "- `gamma_round_distribution.png`: negotiation timing.",
            "- `replication_vs_previous.png`: matched new versus previous γ=0.9 controls.",
            "- `gamma_scaling_vs_elo.png`: scaling slopes for all three conditions and previous controls.",
            "",
            "## Machine-readable outputs",
            "",
            "- `all_runs.csv`: all 360 completed runs.",
            "- `paired_gamma_differences.csv`: all 120 exact three-gamma matched cells.",
            "- `summary_by_gamma.csv`, `summary_by_gamma_competition.csv`, and",
            "  `summary_by_model_gamma.csv`: grouped summaries.",
            "- `previous_matched_runs.csv` and `replication_pairs.csv`: historical controls.",
            "- `scaling_summary.csv`: model-level utility slopes against Arena Elo.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    output_dir = run_root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    base = load_ablation(run_root, args.allow_incomplete, expected=240)
    extension_root = args.extension_root.resolve()
    if not extension_root.exists():
        raise RuntimeError(f"Gamma=0.5 extension root does not exist: {extension_root}")
    extension = load_ablation(
        extension_root,
        args.allow_incomplete,
        expected=120,
    )
    frame = pd.concat([base, extension], ignore_index=True)
    if frame.empty:
        raise RuntimeError("No completed ablation results found")
    if not args.allow_incomplete and len(frame) != 360:
        raise RuntimeError(f"Expected 360 total complete runs, loaded {len(frame)}")
    previous = load_previous(
        args.previous_root.resolve(),
        set(frame.adversary_model.unique()),
    )
    paired = paired_gamma(frame) if len(frame) == 360 else pd.DataFrame()
    replication = replication_pairs(frame, previous)
    scaling = scaling_summary(frame, previous)

    frame.to_csv(output_dir / "all_runs.csv", index=False)
    previous.to_csv(output_dir / "previous_matched_runs.csv", index=False)
    replication.to_csv(output_dir / "replication_pairs.csv", index=False)
    scaling.to_csv(output_dir / "scaling_summary.csv", index=False)
    summarize(frame, ["gamma_discount"]).to_csv(
        output_dir / "summary_by_gamma.csv", index=False
    )
    summarize(frame, ["gamma_discount", "competition_level"]).to_csv(
        output_dir / "summary_by_gamma_competition.csv", index=False
    )
    summarize(frame, ["adversary_model", "gamma_discount"]).to_csv(
        output_dir / "summary_by_model_gamma.csv", index=False
    )

    plot_metric_overview(frame, output_dir / "gamma_metric_overview.png")
    plot_by_competition(frame, output_dir / "gamma_by_competition.png")
    plot_round_distribution(frame, output_dir / "gamma_round_distribution.png")
    plot_replication(replication, output_dir / "replication_vs_previous.png")
    plot_scaling(frame, previous, output_dir / "gamma_scaling_vs_elo.png")
    if not paired.empty:
        paired.to_csv(output_dir / "paired_gamma_differences.csv", index=False)
        plot_model_effects(paired, output_dir / "gamma_effect_by_model.png")
        plot_effect_heatmap(
            paired,
            output_dir / "gamma_effect_by_model_and_competition.png",
        )
        write_report(
            frame=frame,
            paired=paired,
            previous=previous,
            replication=replication,
            scaling=scaling,
            output=output_dir / "discount_factor_ablation_report.md",
        )

    print(
        f"new_runs={len(frame)} base={len(base)} extension={len(extension)} "
        f"paired_gamma={len(paired)} previous={len(previous)}"
    )
    print(f"replication_pairs={len(replication)} output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

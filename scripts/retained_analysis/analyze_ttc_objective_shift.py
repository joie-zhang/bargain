#!/usr/bin/env python3
"""Mechanism-focused TTC analysis: passable-deal language and outcomes."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_METRICS_CSV = PROJECT_ROOT / "analysis/neurips_revision_20260504/ttc_fairness_inequality_by_run.csv"
OUT_DIR = PROJECT_ROOT / "analysis/neurips_revision_20260504/ttc_objective_shift"
GRAPHICS_DIR = PROJECT_ROOT / "overleaf/neurips/graphics"

FAMILY_ORDER = ["gpt-5", "claude-sonnet-4-6", "gemini-3-flash"]
FAMILY_LABELS = {
    "gpt-5": "GPT-5",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gemini-3-flash": "Gemini 3 Flash",
}
FAMILY_MARKERS = {
    "gpt-5": "o",
    "claude-sonnet-4-6": "s",
    "gemini-3-flash": "^",
}
EFFORT_ORDER = ["minimal", "low", "medium", "high", "max"]
EFFORT_LABELS = {
    "minimal": "Minimal",
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "max": "Max",
}
EFFORT_COLORS = {
    "minimal": "#64748b",
    "low": "#2563eb",
    "medium": "#0f766e",
    "high": "#f97316",
    "max": "#7c3aed",
}
PANEL_METRICS = {
    "passable_language_per_1k": {
        "ylabel": "Passable-deal language / 1k words",
        "filename": "ttc_passable_language_vs_compute.png",
        "ylim_floor": 0.0,
    },
    "self_interest_language_per_1k": {
        "ylabel": "Self-interest language / 1k words",
        "filename": "ttc_self_interest_language_vs_compute.png",
        "ylim_floor": 0.0,
    },
    "passable_minus_self_per_1k": {
        "ylabel": "Passable minus self-interest / 1k words",
        "filename": "ttc_passable_minus_self_language_vs_compute.png",
        "zero_line": True,
    },
    "concession_language_per_1k": {
        "ylabel": "Concession language / 1k words",
        "filename": "ttc_concession_language_vs_compute.png",
        "ylim_floor": 0.0,
    },
    "refusal_language_per_1k": {
        "ylabel": "Refusal/infeasibility language / 1k words",
        "filename": "ttc_refusal_language_vs_compute.png",
        "ylim_floor": 0.0,
    },
    "target_abs_fair_excess": {
        "ylabel": "Mean |target fair-share excess|",
        "filename": "ttc_target_abs_fair_excess_vs_compute.png",
        "ylim_floor": 0.0,
    },
    "target_fair_excess": {
        "ylabel": "Mean target fair-share excess",
        "filename": "ttc_target_fair_excess_vs_compute.png",
        "zero_line": True,
    },
}

LEXICONS = {
    "passable": [
        r"\bfair\b",
        r"\bfairness\b",
        r"\bbalanced\b",
        r"\bequitable\b",
        r"\bcompromise\b",
        r"\bmutual(?:ly)?\b",
        r"\bacceptable\b",
        r"\bagreement\b",
        r"\bagree\b",
        r"\bconsensus\b",
        r"\bcooperat(?:e|ive|ion)\b",
        r"\bgood faith\b",
        r"\btrust\b",
        r"\breasonable\b",
        r"\bwin[- ]win\b",
        r"\bwelfare\b",
        r"\bfor both\b",
        r"\bboth of us\b",
        r"\bclose the deal\b",
        r"\bsettle\b",
    ],
    "self_interest": [
        r"\bmaximize\b",
        r"\bmy utility\b",
        r"\bmy payoff\b",
        r"\bmy value\b",
        r"\bmy top\b",
        r"\btop priority\b",
        r"\bnon[- ]negotiable\b",
        r"\bred line\b",
        r"\bmust[- ]have\b",
        r"\binsist\b",
        r"\bpreserve\b",
        r"\bprotect\b",
        r"\bkeep\b",
        r"\banchor\b",
        r"\bleverage\b",
        r"\btarget[- ]favorable\b",
        r"\bfavorable\b",
    ],
    "concession": [
        r"\bconcede\b",
        r"\bconcession\b",
        r"\bflexible\b",
        r"\baccept\b",
        r"\bacceptable\b",
        r"\bsettle\b",
        r"\bcompromise\b",
        r"\bmeet.*halfway\b",
        r"\bmove toward\b",
        r"\brisk(?:s)? rejection\b",
        r"\bavoid.*deadlock\b",
        r"\bavoid.*discount\b",
    ],
    "refusal": [
        r"\breject\b",
        r"\brefuse\b",
        r"\bcannot accept\b",
        r"\bcan't accept\b",
        r"\bno deal\b",
        r"\binfeasible\b",
        r"\bimpossible\b",
        r"\bnegative utility\b",
        r"\bzero[- ]value\b",
        r"\bdominated\b",
        r"\bstructurally impossible\b",
    ],
}


def sem(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) <= 1:
        return 0.0
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


def result_to_interactions(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.with_name(path.name.replace("experiment_results", "all_interactions"))


def load_target_response_text(path_value: str, target_agent: str) -> tuple[str, dict[str, str]]:
    path = result_to_interactions(path_value)
    if not path.exists():
        return "", {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return "", {}

    by_phase: dict[str, list[str]] = {}
    texts: list[str] = []
    for entry in payload:
        if not isinstance(entry, dict) or str(entry.get("agent_id")) != str(target_agent):
            continue
        phase = str(entry.get("phase") or "")
        if phase == "game_setup":
            continue
        response = entry.get("response")
        if not isinstance(response, str) or not response.strip():
            continue
        texts.append(response)
        by_phase.setdefault(phase, []).append(response)
    return "\n\n".join(texts), {phase: "\n\n".join(parts) for phase, parts in by_phase.items()}


def count_patterns(text: str, patterns: list[str]) -> int:
    lower = text.lower()
    return int(sum(len(re.findall(pattern, lower, flags=re.IGNORECASE)) for pattern in patterns))


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def score_text(text: str) -> dict[str, float]:
    words = word_count(text)
    denom = max(words / 1000.0, 1e-9)
    counts = {name: count_patterns(text, patterns) for name, patterns in LEXICONS.items()}
    return {
        "target_response_words": float(words),
        **{f"{name}_language_count": float(count) for name, count in counts.items()},
        **{f"{name}_language_per_1k": float(count / denom) for name, count in counts.items()},
        "passable_minus_self_per_1k": float((counts["passable"] - counts["self_interest"]) / denom),
        "passable_share": float(counts["passable"] / max(counts["passable"] + counts["self_interest"], 1)),
    }


def build_scored_runs() -> pd.DataFrame:
    runs = pd.read_csv(RUN_METRICS_CSV)
    rows: list[dict[str, Any]] = []
    for _, row in runs.iterrows():
        text, phase_text = load_target_response_text(str(row["path"]), str(row["target_agent"]))
        scores = score_text(text)
        # Phase-specific public/proposal score is useful for checking that the signal is not only private notes.
        public_text = "\n\n".join(
            content
            for phase, content in phase_text.items()
            if "discussion" in phase or "proposal" in phase or "vote" in phase
        )
        public_scores = score_text(public_text)
        rows.append(
            {
                **row.to_dict(),
                **scores,
                "public_passable_language_per_1k": public_scores["passable_language_per_1k"],
                "public_self_interest_language_per_1k": public_scores["self_interest_language_per_1k"],
                "public_passable_minus_self_per_1k": public_scores["passable_minus_self_per_1k"],
                "target_fair_excess": row.get("adversary_fairness_excess", np.nan),
                "target_abs_fair_excess": abs(float(row["adversary_fairness_excess"]))
                if pd.notna(row.get("adversary_fairness_excess"))
                else np.nan,
            }
        )
    scored = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_csv(OUT_DIR / "ttc_objective_shift_scored_runs.csv", index=False)
    return scored


def build_order_avg(scored: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "passable_language_per_1k",
        "self_interest_language_per_1k",
        "passable_minus_self_per_1k",
        "concession_language_per_1k",
        "refusal_language_per_1k",
        "passable_share",
        "public_passable_language_per_1k",
        "public_self_interest_language_per_1k",
        "public_passable_minus_self_per_1k",
        "target_fair_excess",
        "target_abs_fair_excess",
        "target_utility",
        "utility_gap",
        "absolute_payoff_gap",
        "payoff_gini_corrected",
        "payoff_variance",
        "fairness_distance",
    ]
    agg_spec: dict[str, Any] = {
        "order_count": ("order", "nunique"),
        "run_count": ("config_id", "size"),
        "target_compute_tokens_per_call": ("target_compute_tokens_per_call", "mean"),
        "target_output_tokens_per_call": ("target_output_tokens_per_call", "mean"),
        "consensus_rate": ("consensus", "mean"),
        "mean_round": ("round", "mean"),
    }
    for metric in metric_cols:
        agg_spec[metric] = (metric, "mean")
    order_avg = (
        scored.groupby(["family", "provider", "level", "level_index", "game", "game_cell"], dropna=False)
        .agg(**agg_spec)
        .reset_index()
        .sort_values(["family", "game_cell", "level_index"])
    )
    order_avg.to_csv(OUT_DIR / "ttc_objective_shift_order_averaged.csv", index=False)
    return order_avg


def build_effort_summary(order_avg: pd.DataFrame) -> pd.DataFrame:
    agg_spec: dict[str, Any] = {
        "game_cell_count": ("game_cell", "nunique"),
        "target_tokens_mean": ("target_compute_tokens_per_call", "mean"),
        "target_tokens_sem": ("target_compute_tokens_per_call", sem),
    }
    metric_cols = [
        "passable_language_per_1k",
        "self_interest_language_per_1k",
        "passable_minus_self_per_1k",
        "concession_language_per_1k",
        "refusal_language_per_1k",
        "passable_share",
        "public_passable_language_per_1k",
        "public_self_interest_language_per_1k",
        "public_passable_minus_self_per_1k",
        "target_fair_excess",
        "target_abs_fair_excess",
        "target_utility",
        "utility_gap",
        "absolute_payoff_gap",
        "payoff_gini_corrected",
        "payoff_variance",
        "fairness_distance",
        "consensus_rate",
        "mean_round",
    ]
    for metric in metric_cols:
        agg_spec[f"{metric}_mean"] = (metric, "mean")
        agg_spec[f"{metric}_sem"] = (metric, sem)
    summary = (
        order_avg.groupby(["family", "provider", "level", "level_index"], dropna=False)
        .agg(**agg_spec)
        .reset_index()
        .sort_values(["family", "level_index"])
    )
    summary.to_csv(OUT_DIR / "ttc_objective_shift_effort_summary.csv", index=False)
    return summary


def y_limits(summary: pd.DataFrame, metric: str, config: dict[str, Any]) -> tuple[float, float]:
    mean_col = f"{metric}_mean"
    sem_col = f"{metric}_sem"
    vals = pd.to_numeric(summary[mean_col], errors="coerce")
    errs = pd.to_numeric(summary[sem_col], errors="coerce").fillna(0.0)
    lower = float((vals - errs).min())
    upper = float((vals + errs).max())
    if not math.isfinite(lower) or not math.isfinite(upper) or math.isclose(lower, upper):
        lower, upper = 0.0, 1.0
    span = upper - lower
    lower -= 0.12 * span
    upper += 0.12 * span
    if "ylim_floor" in config:
        lower = float(config["ylim_floor"])
        upper = max(upper, lower + 1e-6)
    return lower, upper


def plot_panel(summary: pd.DataFrame, metric: str, config: dict[str, Any]) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.9), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")

    x_max = max(2200, float(summary["target_tokens_mean"].max()) * 1.15)
    y_mean = f"{metric}_mean"
    y_sem = f"{metric}_sem"
    ylim = y_limits(summary, metric, config)

    for ax, family in zip(axes, FAMILY_ORDER):
        family_df = summary[summary["family"].eq(family)].sort_values("level_index")
        if family_df.empty:
            continue
        ax.plot(
            family_df["target_tokens_mean"],
            family_df[y_mean],
            color="#475569",
            linewidth=2.55,
            alpha=0.60,
            zorder=2,
        )
        if config.get("zero_line"):
            ax.axhline(0.0, color="#111827", linewidth=0.95, alpha=0.42, zorder=1)
        for _, row in family_df.iterrows():
            effort = str(row["level"])
            ax.errorbar(
                row["target_tokens_mean"],
                row[y_mean],
                yerr=max(float(row[y_sem]), 0.0) if pd.notna(row[y_sem]) else 0.0,
                fmt=FAMILY_MARKERS[family],
                markersize=10.8,
                color=EFFORT_COLORS.get(effort, "#475569"),
                markeredgecolor="white",
                markeredgewidth=0.95,
                elinewidth=1.55,
                capsize=4.2,
                alpha=0.95,
                zorder=3,
            )
        ax.set_title(FAMILY_LABELS[family], fontsize=24, pad=10)
        ax.tick_params(axis="both", labelsize=15)
        ax.grid(True, color="#d1d5db", alpha=0.42, linewidth=0.85)
        ax.set_xlim(-70, x_max)
        ax.set_ylim(*ylim)

    axes[0].set_ylabel(str(config["ylabel"]), fontsize=20, labelpad=10)
    fig.supxlabel("Mean observed target tokens/call", fontsize=19, y=0.07)
    handles = [
        Line2D(
            [0],
            [0],
            color=EFFORT_COLORS[effort],
            marker="o",
            linestyle="",
            markersize=10.5,
            markeredgecolor="white",
            markeredgewidth=0.9,
            label=EFFORT_LABELS[effort],
        )
        for effort in EFFORT_ORDER
        if effort in set(summary["level"].astype(str))
    ]
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.07),
        ncol=len(handles),
        title="Reasoning effort",
        title_fontsize=13.5,
        fontsize=13.2,
        frameon=True,
        facecolor="white",
        framealpha=0.94,
        columnspacing=1.0,
        handletextpad=0.45,
        borderpad=0.45,
    )
    legend.get_frame().set_edgecolor("#d1d5db")
    fig.subplots_adjust(left=0.08, right=0.995, top=0.84, bottom=0.25, wspace=0.11)

    out_path = GRAPHICS_DIR / str(config["filename"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=260, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return out_path


def language_bin_summary(scored: pd.DataFrame) -> pd.DataFrame:
    data = scored.copy()
    # Within-family quartiles avoid turning model-family verbosity into the whole signal.
    data["passable_quartile"] = (
        data.groupby("family")["passable_language_per_1k"]
        .transform(lambda s: pd.qcut(s.rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"]))
        .astype(str)
    )
    summary = (
        data.groupby(["passable_quartile"], dropna=False)
        .agg(
            n=("config_id", "size"),
            passable_language_per_1k=("passable_language_per_1k", "mean"),
            target_utility=("target_utility", "mean"),
            utility_gap=("utility_gap", "mean"),
            absolute_payoff_gap=("absolute_payoff_gap", "mean"),
            payoff_gini_corrected=("payoff_gini_corrected", "mean"),
            payoff_variance=("payoff_variance", "mean"),
            target_abs_fair_excess=("target_abs_fair_excess", "mean"),
            fairness_distance=("fairness_distance", "mean"),
            consensus=("consensus", "mean"),
            mean_round=("round", "mean"),
        )
        .reset_index()
        .sort_values("passable_quartile")
    )
    summary.to_csv(OUT_DIR / "ttc_passable_language_quartile_summary.csv", index=False)
    return summary


def plot_language_bins(summary: pd.DataFrame) -> Path:
    metrics = [
        ("target_utility", "Target payoff"),
        ("utility_gap", "Target - baseline"),
        ("absolute_payoff_gap", "Absolute payoff gap"),
        ("payoff_gini_corrected", "Corrected payoff Gini"),
        ("target_abs_fair_excess", "|Target fair-share excess|"),
        ("fairness_distance", "NBS/Lindahl distance"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.2))
    axes = axes.ravel()
    xs = np.arange(len(summary))
    labels = list(summary["passable_quartile"])
    for ax, (metric, ylabel) in zip(axes, metrics, strict=True):
        ax.plot(xs, summary[metric], marker="o", color="#334155", linewidth=2.0)
        ax.set_xticks(xs, labels)
        ax.set_xlabel("Within-family passable-language quartile")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#d1d5db", alpha=0.42, linewidth=0.85)
        if metric == "utility_gap":
            ax.axhline(0.0, color="#111827", linewidth=0.9, alpha=0.38)
    fig.suptitle("Outcomes by passable-deal language intensity", fontsize=18, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = GRAPHICS_DIR / "ttc_passable_language_quartile_outcomes.png"
    fig.savefig(out_path, dpi=240, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    return out_path


def refusal_bin_summary(scored: pd.DataFrame) -> pd.DataFrame:
    data = scored.copy()
    data["refusal_quartile"] = (
        data.groupby("family")["refusal_language_per_1k"]
        .transform(lambda s: pd.qcut(s.rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"]))
        .astype(str)
    )
    summary = (
        data.groupby(["refusal_quartile"], dropna=False)
        .agg(
            n=("config_id", "size"),
            refusal_language_per_1k=("refusal_language_per_1k", "mean"),
            target_utility=("target_utility", "mean"),
            utility_gap=("utility_gap", "mean"),
            absolute_payoff_gap=("absolute_payoff_gap", "mean"),
            payoff_gini_corrected=("payoff_gini_corrected", "mean"),
            payoff_variance=("payoff_variance", "mean"),
            target_abs_fair_excess=("target_abs_fair_excess", "mean"),
            fairness_distance=("fairness_distance", "mean"),
            consensus=("consensus", "mean"),
            mean_round=("round", "mean"),
        )
        .reset_index()
        .sort_values("refusal_quartile")
    )
    summary.to_csv(OUT_DIR / "ttc_refusal_language_quartile_summary.csv", index=False)
    return summary


def plot_refusal_bins(summary: pd.DataFrame) -> Path:
    metrics = [
        ("target_utility", "Target payoff"),
        ("absolute_payoff_gap", "Absolute payoff gap"),
        ("payoff_gini_corrected", "Corrected payoff Gini"),
        ("payoff_variance", "Payoff variance"),
        ("consensus", "Consensus rate"),
        ("mean_round", "Mean final round"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.2))
    axes = axes.ravel()
    xs = np.arange(len(summary))
    labels = list(summary["refusal_quartile"])
    for ax, (metric, ylabel) in zip(axes, metrics, strict=True):
        ax.plot(xs, summary[metric], marker="o", color="#7f1d1d", linewidth=2.0)
        ax.set_xticks(xs, labels)
        ax.set_xlabel("Within-family refusal-language quartile")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#d1d5db", alpha=0.42, linewidth=0.85)
    fig.suptitle("Outcomes by refusal/infeasibility language intensity", fontsize=18, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = GRAPHICS_DIR / "ttc_refusal_language_quartile_outcomes.png"
    fig.savefig(out_path, dpi=240, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    return out_path


def within_cell_residual_summary(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Residualize outcomes within family/game/order cells before binning language shifts."""
    data = scored.copy()
    group_cols = ["family", "game_cell", "order"]
    x_col = "passable_minus_self_per_1k"
    outcome_cols = [
        "target_utility",
        "utility_gap",
        "absolute_payoff_gap",
        "payoff_gini_corrected",
        "payoff_variance",
        "target_abs_fair_excess",
        "fairness_distance",
    ]
    for col in [x_col, *outcome_cols]:
        data[f"{col}_resid"] = data[col] - data.groupby(group_cols, dropna=False)[col].transform("mean")

    data["objective_shift_resid_quartile"] = pd.qcut(
        data[f"{x_col}_resid"].rank(method="first"),
        4,
        labels=["Q1", "Q2", "Q3", "Q4"],
    ).astype(str)
    summary = (
        data.groupby("objective_shift_resid_quartile", dropna=False)
        .agg(
            n=("config_id", "size"),
            passable_minus_self_per_1k_resid=(f"{x_col}_resid", "mean"),
            target_utility_resid=("target_utility_resid", "mean"),
            utility_gap_resid=("utility_gap_resid", "mean"),
            absolute_payoff_gap_resid=("absolute_payoff_gap_resid", "mean"),
            payoff_gini_corrected_resid=("payoff_gini_corrected_resid", "mean"),
            payoff_variance_resid=("payoff_variance_resid", "mean"),
            target_abs_fair_excess_resid=("target_abs_fair_excess_resid", "mean"),
            fairness_distance_resid=("fairness_distance_resid", "mean"),
        )
        .reset_index()
        .sort_values("objective_shift_resid_quartile")
    )
    corr_rows = []
    x = f"{x_col}_resid"
    for y_base in outcome_cols:
        y = f"{y_base}_resid"
        clean = data[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) < 3 or clean[x].nunique() < 2 or clean[y].nunique() < 2:
            r = math.nan
        else:
            r = float(clean[x].corr(clean[y]))
        corr_rows.append({"x": x, "y": y, "pearson_r": r, "n": len(clean)})
    corr_frame = pd.DataFrame(corr_rows)

    data.to_csv(OUT_DIR / "ttc_objective_shift_within_cell_residuals.csv", index=False)
    summary.to_csv(OUT_DIR / "ttc_objective_shift_within_cell_residual_quartiles.csv", index=False)
    corr_frame.to_csv(OUT_DIR / "ttc_objective_shift_within_cell_residual_correlations.csv", index=False)
    return summary, corr_frame


def plot_within_cell_residual_bins(summary: pd.DataFrame) -> Path:
    metrics = [
        ("target_utility_resid", "Residual target payoff"),
        ("utility_gap_resid", "Residual target - baseline"),
        ("absolute_payoff_gap_resid", "Residual absolute payoff gap"),
        ("payoff_gini_corrected_resid", "Residual corrected Gini"),
        ("target_abs_fair_excess_resid", "Residual |target fair-share excess|"),
        ("payoff_variance_resid", "Residual payoff variance"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.2))
    axes = axes.ravel()
    xs = np.arange(len(summary))
    labels = list(summary["objective_shift_resid_quartile"])
    for ax, (metric, ylabel) in zip(axes, metrics, strict=True):
        ax.plot(xs, summary[metric], marker="o", color="#334155", linewidth=2.0)
        ax.axhline(0.0, color="#94a3b8", linewidth=0.9)
        ax.set_xticks(xs, labels)
        ax.set_xlabel("Within-cell objective-shift residual quartile")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#d1d5db", alpha=0.42, linewidth=0.85)
    fig.suptitle("Outcome residuals by passable-vs-self objective shift", fontsize=18, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = GRAPHICS_DIR / "ttc_objective_shift_within_cell_residual_quartiles.png"
    fig.savefig(out_path, dpi=240, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    return out_path


def correlation_table(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    xs = [
        "passable_language_per_1k",
        "self_interest_language_per_1k",
        "passable_minus_self_per_1k",
        "concession_language_per_1k",
        "refusal_language_per_1k",
    ]
    ys = [
        "target_utility",
        "utility_gap",
        "absolute_payoff_gap",
        "payoff_gini_corrected",
        "payoff_variance",
        "target_abs_fair_excess",
        "fairness_distance",
    ]
    for family in ["all"] + FAMILY_ORDER:
        sub = scored if family == "all" else scored[scored["family"].eq(family)]
        for x in xs:
            for y in ys:
                clean = sub[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(clean) < 3 or clean[x].nunique() < 2 or clean[y].nunique() < 2:
                    r = math.nan
                else:
                    r = float(clean[x].corr(clean[y]))
                rows.append({"family": family, "x": x, "y": y, "pearson_r": r, "n": len(clean)})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_DIR / "ttc_objective_shift_correlations.csv", index=False)
    return frame


def main() -> None:
    scored = build_scored_runs()
    order_avg = build_order_avg(scored)
    summary = build_effort_summary(order_avg)
    quartiles = language_bin_summary(scored)
    refusal_quartiles = refusal_bin_summary(scored)
    residual_quartiles, residual_corrs = within_cell_residual_summary(scored)
    corrs = correlation_table(scored)

    for metric, config in PANEL_METRICS.items():
        print(plot_panel(summary, metric, config))
    print(plot_language_bins(quartiles))
    print(plot_refusal_bins(refusal_quartiles))
    print(plot_within_cell_residual_bins(residual_quartiles))
    print(OUT_DIR / "ttc_objective_shift_scored_runs.csv")
    print(OUT_DIR / "ttc_objective_shift_order_averaged.csv")
    print(OUT_DIR / "ttc_objective_shift_effort_summary.csv")
    print(OUT_DIR / "ttc_passable_language_quartile_summary.csv")
    print(OUT_DIR / "ttc_refusal_language_quartile_summary.csv")
    print(OUT_DIR / "ttc_objective_shift_within_cell_residual_quartiles.csv")
    print(OUT_DIR / "ttc_objective_shift_within_cell_residual_correlations.csv")
    print(OUT_DIR / "ttc_objective_shift_correlations.csv")
    print(corrs.sort_values("pearson_r").head(8).to_string(index=False))
    print(corrs.sort_values("pearson_r", ascending=False).head(8).to_string(index=False))
    print(residual_corrs.sort_values("pearson_r").to_string(index=False))


if __name__ == "__main__":
    main()

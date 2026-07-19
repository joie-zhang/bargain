#!/usr/bin/env python3
"""Analyze hot strategic tags across TTC conditions."""

from __future__ import annotations

import json
import math
import base64
from pathlib import Path
from textwrap import shorten
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVIEW_JSON = PROJECT_ROOT / "strategic_tag_review_final.json"
TAG_ROOT = PROJECT_ROOT / "analysis/ttc_llm_strategic_tag_adjudication_20260629"
EVENT_JSONL = TAG_ROOT / "ttc_llm_event_tags.jsonl"
MANIFEST_JSONL = TAG_ROOT / "all_ttc_rollouts_manifest.jsonl"
CODEBOOK_JSON = TAG_ROOT / "llm_tag_codebook.json"
OUT_DIR = PROJECT_ROOT / "analysis/ttc_hot_strategic_tags_20260629"
PLOT_DIR = PROJECT_ROOT / "overleaf/neurips/graphics/ttc_hot_strategic_tags"

FAMILY_ORDER = ["gpt-5", "claude-sonnet-4-6", "gemini-3-flash"]
FAMILY_LABELS = {
    "gpt-5": "GPT-5",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gemini-3-flash": "Gemini 3 Flash",
}
FAMILY_COLORS = {
    "gpt-5": "#2563eb",
    "claude-sonnet-4-6": "#dc2626",
    "gemini-3-flash": "#16a34a",
}
LEVEL_ORDER = {
    "minimal": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "max": 4,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_name(value: str, limit: int = 35) -> str:
    value = value.replace("/", " / ").replace("_", " ")
    return shorten(value, width=limit, placeholder="...")


def markdown_embedded_png(path: Path, alt: str) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"![{alt}](data:image/png;base64,{encoded})"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    review = json.loads(REVIEW_JSON.read_text(encoding="utf-8"))
    review_rows = pd.DataFrame(review["responses"])
    hot = review_rows[review_rows["decision"].eq("hot")].copy()

    events = pd.DataFrame(read_jsonl(EVENT_JSONL))
    manifest = pd.DataFrame(read_jsonl(MANIFEST_JSONL))
    codebook = pd.DataFrame(json.loads(CODEBOOK_JSON.read_text(encoding="utf-8")))

    # Keep one row per rollout for denominators and token x-axis.
    manifest = manifest[
        [
            "config_id",
            "family",
            "provider",
            "level",
            "level_index",
            "game_label",
            "game_cell",
            "order",
            "target_compute_tokens_per_call",
            "target_output_tokens_per_call",
            "target_utility",
            "baseline_utility",
            "consensus_reached",
            "final_round",
        ]
    ].copy()
    manifest["level_index"] = manifest["level"].map(LEVEL_ORDER).fillna(manifest["level_index"]).astype(int)
    manifest["absolute_payoff_gap"] = (manifest["target_utility"] - manifest["baseline_utility"]).abs()

    events["level_index"] = events["level"].map(LEVEL_ORDER).fillna(events["level_index"]).astype(int)
    events = events.merge(
        hot[["tag_code", "tag_title", "category", "decision"]].rename(
            columns={"tag_title": "review_tag_title", "category": "review_category"}
        ),
        on="tag_code",
        how="inner",
    )
    events["tag_title"] = events["review_tag_title"].fillna(events["tag_title"])
    events["category"] = events["review_category"]
    events = events.drop(columns=["review_tag_title", "review_category"])
    events = events.merge(codebook[["tag_code", "definition"]], on="tag_code", how="left")

    return review_rows, hot, events, manifest


def group_denominators(manifest: pd.DataFrame) -> pd.DataFrame:
    return (
        manifest.groupby(["family", "level", "level_index"], as_index=False)
        .agg(
            rollout_count=("config_id", "nunique"),
            mean_tokens_per_call=("target_compute_tokens_per_call", "mean"),
            mean_output_tokens_per_call=("target_output_tokens_per_call", "mean"),
            mean_target_utility=("target_utility", "mean"),
            mean_absolute_payoff_gap=("absolute_payoff_gap", "mean"),
            consensus_rate=("consensus_reached", "mean"),
            mean_final_round=("final_round", "mean"),
        )
        .sort_values(["family", "level_index"])
    )


def summarize_tags(hot: pd.DataFrame, events: pd.DataFrame, manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    denoms = group_denominators(manifest)
    hot_meta = hot[["tag_code", "tag_title", "category", "source_count", "source_share"]].drop_duplicates()
    family_levels = denoms[["family", "level", "level_index", "rollout_count", "mean_tokens_per_call"]].copy()
    grid = hot_meta.merge(family_levels, how="cross")

    target_events = events[events["speaker_role"].eq("target")].copy()
    counts = (
        target_events.groupby(["tag_code", "family", "level", "level_index"], as_index=False)
        .agg(
            target_event_count=("tag_code", "size"),
            target_rollouts_with_tag=("config_id", "nunique"),
        )
    )
    summary = grid.merge(
        counts,
        on=["tag_code", "family", "level", "level_index"],
        how="left",
    )
    summary["target_event_count"] = summary["target_event_count"].fillna(0).astype(int)
    summary["target_rollouts_with_tag"] = summary["target_rollouts_with_tag"].fillna(0).astype(int)
    summary["target_events_per_rollout"] = summary["target_event_count"] / summary["rollout_count"]
    summary["target_rollout_share"] = summary["target_rollouts_with_tag"] / summary["rollout_count"]
    summary = summary.sort_values(["category", "tag_title", "family", "level_index"])

    cat_counts = (
        target_events.groupby(["category", "family", "level", "level_index"], as_index=False)
        .agg(
            target_event_count=("tag_code", "size"),
            target_rollouts_with_any_hot_category_tag=("config_id", "nunique"),
        )
    )
    categories = hot_meta[["category"]].drop_duplicates()
    cat_grid = categories.merge(family_levels, how="cross")
    cat_summary = cat_grid.merge(
        cat_counts,
        on=["category", "family", "level", "level_index"],
        how="left",
    )
    cat_summary["target_event_count"] = cat_summary["target_event_count"].fillna(0).astype(int)
    cat_summary["target_rollouts_with_any_hot_category_tag"] = (
        cat_summary["target_rollouts_with_any_hot_category_tag"].fillna(0).astype(int)
    )
    cat_summary["target_events_per_rollout"] = cat_summary["target_event_count"] / cat_summary["rollout_count"]
    cat_summary["target_rollout_share"] = (
        cat_summary["target_rollouts_with_any_hot_category_tag"] / cat_summary["rollout_count"]
    )
    cat_summary = cat_summary.sort_values(["category", "family", "level_index"])

    return summary, cat_summary


def fit_slope(frame: pd.DataFrame, y_col: str) -> dict[str, float]:
    clean = frame[["mean_tokens_per_call", y_col]].dropna()
    x = clean["mean_tokens_per_call"].to_numpy(dtype=float)
    y = clean[y_col].to_numpy(dtype=float)
    if len(clean) < 3 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return {"slope_per_1k_tokens": math.nan, "r": math.nan}
    slope, _ = np.polyfit(x / 1000.0, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    return {"slope_per_1k_tokens": float(slope), "r": r}


def build_trends(summary: pd.DataFrame, cat_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for (tag, family), sub in summary.groupby(["tag_code", "family"]):
        sub = sub.sort_values("level_index")
        meta = sub.iloc[0]
        levels = {row["level"]: row["target_events_per_rollout"] for _, row in sub.iterrows()}
        weak = "minimal" if family in {"gpt-5", "gemini-3-flash"} else "low"
        strong = "high" if family in {"gpt-5", "gemini-3-flash"} else "max"
        trend = fit_slope(sub, "target_events_per_rollout")
        rows.append(
            {
                "tag_code": tag,
                "tag_title": meta["tag_title"],
                "category": meta["category"],
                "family": family,
                "weak_level": weak,
                "strong_level": strong,
                "weak_events_per_rollout": levels.get(weak, math.nan),
                "strong_events_per_rollout": levels.get(strong, math.nan),
                "delta_weak_to_strong": levels.get(strong, math.nan) - levels.get(weak, math.nan),
                **trend,
            }
        )
    tag_trends = pd.DataFrame(rows).sort_values(["family", "delta_weak_to_strong"])

    cat_rows: list[dict[str, Any]] = []
    for (cat, family), sub in cat_summary.groupby(["category", "family"]):
        sub = sub.sort_values("level_index")
        levels = {row["level"]: row["target_events_per_rollout"] for _, row in sub.iterrows()}
        weak = "minimal" if family in {"gpt-5", "gemini-3-flash"} else "low"
        strong = "high" if family in {"gpt-5", "gemini-3-flash"} else "max"
        trend = fit_slope(sub, "target_events_per_rollout")
        cat_rows.append(
            {
                "category": cat,
                "family": family,
                "weak_level": weak,
                "strong_level": strong,
                "weak_events_per_rollout": levels.get(weak, math.nan),
                "strong_events_per_rollout": levels.get(strong, math.nan),
                "delta_weak_to_strong": levels.get(strong, math.nan) - levels.get(weak, math.nan),
                **trend,
            }
        )
    cat_trends = pd.DataFrame(cat_rows).sort_values(["family", "delta_weak_to_strong"])
    return tag_trends, cat_trends


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="#d1d5db", alpha=0.45, linewidth=0.8)
    ax.tick_params(axis="both", labelsize=8.2)


def plot_hot_tags(summary: pd.DataFrame) -> Path:
    tags = (
        summary[["tag_code", "tag_title", "category"]]
        .drop_duplicates()
        .sort_values(["category", "tag_title"])
        .to_dict("records")
    )
    ncols = 5
    nrows = math.ceil(len(tags) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(19, 3.15 * nrows), sharex=False, sharey=False)
    axes = np.atleast_1d(axes).ravel()
    max_y = max(0.25, float(summary["target_events_per_rollout"].max()) * 1.08)

    for ax, tag in zip(axes, tags, strict=False):
        sub_tag = summary[summary["tag_code"].eq(tag["tag_code"])]
        for family in FAMILY_ORDER:
            sub = sub_tag[sub_tag["family"].eq(family)].sort_values("level_index")
            if sub.empty:
                continue
            ax.plot(
                sub["mean_tokens_per_call"],
                sub["target_events_per_rollout"],
                marker="o",
                linewidth=1.8,
                markersize=4.0,
                color=FAMILY_COLORS[family],
                alpha=0.92,
            )
        ax.set_title(safe_name(str(tag["tag_title"]), 36), fontsize=9.5)
        ax.set_ylim(0, max_y)
        ax.set_xlabel("tokens/call", fontsize=8)
        ax.set_ylabel("target events/rollout", fontsize=8)
        style_axis(ax)

    for ax in axes[len(tags) :]:
        ax.axis("off")
    handles = [
        plt.Line2D([0], [0], color=FAMILY_COLORS[f], marker="o", linewidth=2.0, label=FAMILY_LABELS[f])
        for f in FAMILY_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Hot strategic tags across TTC: target-authored event frequency", fontsize=18, y=0.998)
    fig.tight_layout(rect=(0, 0.025, 1, 0.985))
    path = PLOT_DIR / "hot_29_tag_frequency_vs_tokens_small_multiples.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    return path


def plot_categories(cat_summary: pd.DataFrame) -> Path:
    categories = sorted(cat_summary["category"].dropna().unique())
    ncols = 3
    nrows = math.ceil(len(categories) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15.5, 4.0 * nrows), sharex=False, sharey=False)
    axes = np.atleast_1d(axes).ravel()
    max_y = max(0.25, float(cat_summary["target_events_per_rollout"].max()) * 1.1)

    for ax, category in zip(axes, categories, strict=False):
        sub_cat = cat_summary[cat_summary["category"].eq(category)]
        for family in FAMILY_ORDER:
            sub = sub_cat[sub_cat["family"].eq(family)].sort_values("level_index")
            if sub.empty:
                continue
            ax.plot(
                sub["mean_tokens_per_call"],
                sub["target_events_per_rollout"],
                marker="o",
                linewidth=2.2,
                markersize=6,
                color=FAMILY_COLORS[family],
                label=FAMILY_LABELS[family],
            )
            for _, row in sub.iterrows():
                ax.annotate(
                    str(row["level"]),
                    (row["mean_tokens_per_call"], row["target_events_per_rollout"]),
                    textcoords="offset points",
                    xytext=(0, 7),
                    ha="center",
                    fontsize=7.6,
                    color=FAMILY_COLORS[family],
                )
        ax.set_title(category, fontsize=13)
        ax.set_ylim(0, max_y)
        ax.set_xlabel("Mean observed target tokens/call")
        ax.set_ylabel("Target hot-tag events per rollout")
        style_axis(ax)
    for ax in axes[len(categories) :]:
        ax.axis("off")
    handles = [
        plt.Line2D([0], [0], color=FAMILY_COLORS[f], marker="o", linewidth=2.0, label=FAMILY_LABELS[f])
        for f in FAMILY_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Hot strategic tag categories across TTC", fontsize=18, y=0.998)
    fig.tight_layout(rect=(0, 0.035, 1, 0.975))
    path = PLOT_DIR / "hot_category_frequency_vs_tokens.png"
    fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    return path


def pivot_for_heatmap(trends: pd.DataFrame, families: list[str], index_col: str, value_col: str) -> pd.DataFrame:
    return (
        trends[trends["family"].isin(families)]
        .pivot(index=index_col, columns="family", values=value_col)
        .reindex(columns=families)
    )


def plot_delta_heatmaps(tag_trends: pd.DataFrame, cat_trends: pd.DataFrame) -> tuple[Path, Path]:
    families = ["gpt-5", "gemini-3-flash"]
    tag_table = tag_trends.copy()
    tag_table["label"] = tag_table["tag_title"].map(lambda x: safe_name(str(x), 42))
    tag_pivot = pivot_for_heatmap(tag_table, families, "label", "delta_weak_to_strong")
    tag_pivot = tag_pivot.loc[tag_pivot.abs().max(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(7.4, 11.2))
    vmax = max(0.1, float(np.nanmax(np.abs(tag_pivot.to_numpy()))))
    im = ax.imshow(tag_pivot.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(families)), [FAMILY_LABELS[f] for f in families])
    ax.set_yticks(range(len(tag_pivot.index)), tag_pivot.index, fontsize=8.4)
    ax.set_title("Weak-to-high change in hot-tag frequency\n(target events/rollout)", fontsize=14)
    for i in range(tag_pivot.shape[0]):
        for j in range(tag_pivot.shape[1]):
            value = tag_pivot.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=7.4)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.025)
    cbar.set_label("Delta events/rollout")
    fig.tight_layout()
    tag_path = PLOT_DIR / "hot_tag_weak_to_high_delta_heatmap_gpt_gemini.png"
    fig.savefig(tag_path, dpi=240, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)

    cat_pivot = pivot_for_heatmap(cat_trends, families, "category", "delta_weak_to_strong")
    cat_pivot = cat_pivot.loc[cat_pivot.abs().max(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    vmax = max(0.1, float(np.nanmax(np.abs(cat_pivot.to_numpy()))))
    im = ax.imshow(cat_pivot.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(families)), [FAMILY_LABELS[f] for f in families])
    ax.set_yticks(range(len(cat_pivot.index)), cat_pivot.index, fontsize=10)
    ax.set_title("Weak-to-high category change\n(target hot-tag events/rollout)", fontsize=14)
    for i in range(cat_pivot.shape[0]):
        for j in range(cat_pivot.shape[1]):
            value = cat_pivot.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.025)
    cbar.set_label("Delta events/rollout")
    fig.tight_layout()
    cat_path = PLOT_DIR / "hot_category_weak_to_high_delta_heatmap_gpt_gemini.png"
    fig.savefig(cat_path, dpi=240, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)

    return tag_path, cat_path


def plot_round_phase(events: pd.DataFrame) -> tuple[Path, Path]:
    target_events = events[events["speaker_role"].eq("target")].copy()
    target_events["round"] = pd.to_numeric(target_events["round"], errors="coerce")
    target_events = target_events[target_events["round"].between(1, 10, inclusive="both")]

    cats = sorted(target_events["category"].dropna().unique())
    rounds = list(range(1, 11))
    table = (
        target_events.groupby(["category", "round"]).size().unstack(fill_value=0).reindex(index=cats, columns=rounds, fill_value=0)
    )
    row_norm = table.div(table.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    fig, ax = plt.subplots(figsize=(11, 4.9))
    im = ax.imshow(row_norm.to_numpy(), cmap="YlGnBu", aspect="auto", vmin=0, vmax=max(0.01, row_norm.to_numpy().max()))
    ax.set_xticks(range(len(rounds)), rounds)
    ax.set_yticks(range(len(cats)), cats)
    ax.set_xlabel("Round")
    ax.set_title("When hot strategic categories appear\n(row-normalized target-authored events)")
    for i in range(row_norm.shape[0]):
        for j in range(row_norm.shape[1]):
            value = row_norm.iloc[i, j]
            if value >= 0.12:
                ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=7.6, color="#111827")
    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("Share of category events")
    fig.tight_layout()
    round_path = PLOT_DIR / "hot_category_round_distribution.png"
    fig.savefig(round_path, dpi=240, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)

    phases = ["discussion", "private_thinking", "proposal", "voting", "reflection", "formal_outcome"]
    phase_table = (
        target_events.groupby(["category", "phase"]).size().unstack(fill_value=0).reindex(index=cats, columns=phases, fill_value=0)
    )
    phase_norm = phase_table.div(phase_table.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    fig, ax = plt.subplots(figsize=(10, 4.9))
    im = ax.imshow(phase_norm.to_numpy(), cmap="PuBuGn", aspect="auto", vmin=0, vmax=max(0.01, phase_norm.to_numpy().max()))
    ax.set_xticks(range(len(phases)), phases, rotation=25, ha="right")
    ax.set_yticks(range(len(cats)), cats)
    ax.set_title("Where hot strategic categories appear\n(row-normalized target-authored events)")
    for i in range(phase_norm.shape[0]):
        for j in range(phase_norm.shape[1]):
            value = phase_norm.iloc[i, j]
            if value >= 0.14:
                ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=7.6, color="#111827")
    cbar = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.02)
    cbar.set_label("Share of category events")
    fig.tight_layout()
    phase_path = PLOT_DIR / "hot_category_phase_distribution.png"
    fig.savefig(phase_path, dpi=240, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return round_path, phase_path


def plot_discussion_turn_distribution(events: pd.DataFrame) -> Path:
    target_events = events[
        events["speaker_role"].eq("target") & events["phase"].eq("discussion")
    ].copy()
    target_events["discussion_turn"] = pd.to_numeric(target_events["discussion_turn"], errors="coerce")
    target_events = target_events[target_events["discussion_turn"].notna()]
    target_events["discussion_turn"] = target_events["discussion_turn"].astype(int)
    cats = sorted(target_events["category"].dropna().unique())
    turns = sorted(target_events["discussion_turn"].unique())
    table = (
        target_events.groupby(["category", "discussion_turn"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=cats, columns=turns, fill_value=0)
    )
    row_norm = table.div(table.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    im = ax.imshow(row_norm.to_numpy(), cmap="Blues", aspect="auto", vmin=0, vmax=max(0.01, row_norm.to_numpy().max()))
    ax.set_xticks(range(len(turns)), [f"Turn {t}" for t in turns])
    ax.set_yticks(range(len(cats)), cats)
    ax.set_title("Where hot strategic categories appear within discussion rounds\n(row-normalized target-authored discussion events)")
    for i in range(row_norm.shape[0]):
        for j in range(row_norm.shape[1]):
            value = row_norm.iloc[i, j]
            if value >= 0.14:
                ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=8, color="#111827")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.025)
    cbar.set_label("Share of category events")
    fig.tight_layout()
    path = PLOT_DIR / "hot_category_discussion_turn_distribution.png"
    fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return path


def top_trend_examples(events: pd.DataFrame, tag_trends: pd.DataFrame, n: int = 16) -> pd.DataFrame:
    focus = tag_trends[tag_trends["family"].isin(["gpt-5", "gemini-3-flash"])].copy()
    focus["abs_delta"] = focus["delta_weak_to_strong"].abs()
    top = focus.sort_values("abs_delta", ascending=False).head(n)
    examples = []
    target_events = events[events["speaker_role"].eq("target")].copy()
    for _, row in top.iterrows():
        tag = row["tag_code"]
        fam = row["family"]
        strong = row["strong_level"]
        weak = row["weak_level"]
        preferred = target_events[
            target_events["tag_code"].eq(tag)
            & target_events["family"].eq(fam)
            & target_events["level"].isin([strong, weak])
        ].copy()
        if preferred.empty:
            preferred = target_events[target_events["tag_code"].eq(tag)].copy()
        if preferred.empty:
            continue
        preferred["level_priority"] = preferred["level"].map({strong: 0, weak: 1}).fillna(2)
        preferred["conf_priority"] = preferred["confidence"].map({"high": 0, "medium": 1, "low": 2}).fillna(3)
        ex = preferred.sort_values(["level_priority", "conf_priority", "round"]).iloc[0]
        quote = " ".join(str(ex.get("quote", "")).split())
        rationale = " ".join(str(ex.get("rationale", "")).split())
        examples.append(
            {
                "family": fam,
                "tag_code": tag,
                "tag_title": row["tag_title"],
                "category": row["category"],
                "delta_weak_to_strong": row["delta_weak_to_strong"],
                "example_config": int(ex["config_id"]),
                "level": ex["level"],
                "round": ex.get("round"),
                "phase": ex.get("phase"),
                "quote": shorten(quote, width=260, placeholder="..."),
                "rationale": shorten(rationale, width=220, placeholder="..."),
                "result_path": ex.get("result_path"),
            }
        )
    return pd.DataFrame(examples)


def write_report(
    hot: pd.DataFrame,
    summary: pd.DataFrame,
    cat_summary: pd.DataFrame,
    tag_trends: pd.DataFrame,
    cat_trends: pd.DataFrame,
    examples: pd.DataFrame,
    plot_paths: list[Path],
) -> Path:
    report = OUT_DIR / "ttc_hot_strategic_tag_research_report.md"
    gpt_cat = cat_trends[cat_trends["family"].eq("gpt-5")].sort_values("delta_weak_to_strong", ascending=False)
    gem_cat = cat_trends[cat_trends["family"].eq("gemini-3-flash")].sort_values("delta_weak_to_strong", ascending=False)
    claude_cat = cat_trends[cat_trends["family"].eq("claude-sonnet-4-6")].sort_values("delta_weak_to_strong", ascending=False)

    def md_table(df: pd.DataFrame, cols: list[str], limit: int = 10) -> str:
        if df.empty:
            return "(none)"
        df = df[cols].head(limit).copy()
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, r in df.iterrows():
            vals = []
            for c in cols:
                v = r[c]
                if isinstance(v, float):
                    vals.append(f"{v:.3g}")
                else:
                    vals.append(str(v).replace("|", "\\|"))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    plot_notes = {
        "hot_29_tag_frequency_vs_tokens_small_multiples": [
            "**What it is:** one panel per hot strategic label. The x-axis is the target model's observed reasoning tokens per call. The y-axis is how many target-authored events with that label appear per rollout. A rising line means the target uses that tactic more often when it is given more TTC.",
            "**How to read it:** compare the shape of a provider's line inside each label panel. GPT-5 and Gemini are the cleanest TTC comparisons because they have a low-to-high token spread; Claude is noisier because its observed token proxy does not scale cleanly with requested effort.",
            "**Why it matters:** this plot asks what TTC is inducing behaviorally. It separates different mechanisms that would be invisible in aggregate payoff plots: arithmetic, vetoes, cost policing, concessions, self-advocacy, and pressure can move in different directions.",
        ],
        "hot_category_frequency_vs_tokens": [
            "**What it is:** the 29 labels collapsed into broader strategy families, such as pressure, logical persuasion, trade/compromise, and self-interest/exploitation. The y-axis is still target-authored tag events per rollout.",
            "**How to read it:** upward slopes mean more TTC induces more of that strategic family. Downward slopes mean the model is moving away from that family as it reasons longer.",
            "**Why it matters:** this is the main bridge from qualitative coding to the TTC scaling claim. It shows that additional compute does not simply create a more cooperative negotiator; it changes the mix of tactics, and those tactics can be constructive or obstructive depending on the game state.",
        ],
        "hot_tag_weak_to_high_delta_heatmap_gpt_gemini": [
            "**What it is:** for each individual hot label, the cell value is the change in target-authored events per rollout from weak effort to high effort. Red means the tactic becomes more common with TTC; blue means it becomes less common.",
            "**How to read it:** GPT-5 and Gemini are shown because they provide the cleanest low-to-high token comparisons. The labels are sorted so the most changed tactics appear first.",
            "**Why it matters:** this plot identifies which exact negotiation behaviors are responsible for the category-level story. It is where broad claims like 'more formalization' or 'more hardening' become concrete.",
        ],
        "hot_category_weak_to_high_delta_heatmap_gpt_gemini": [
            "**What it is:** the category-level version of the weak-to-high heatmap. Each cell is the high-effort event rate minus the weak-effort event rate for that category.",
            "**How to read it:** red cells are categories induced by more TTC; blue cells are categories suppressed by more TTC. A near-zero cell means no clear weak-to-high shift.",
            "**Why it matters:** this is the clearest summary of provider-specific TTC mechanisms. If payoff does not improve monotonically, this plot helps explain why: compute may be spent on different tactical families rather than on a uniformly better settlement policy.",
        ],
        "hot_category_round_distribution": [
            "**What it is:** a timing plot. For each strategy category, the row sums to 100%, and each cell shows what share of that category's target-authored events happened in that negotiation round.",
            "**How to read it:** darker early-round cells mean that strategy appears mostly at the opening of the negotiation; darker late-round cells mean it appears during repair, pressure, or settlement closure.",
            "**Why it matters:** many strategic tags are not late discoveries after long deliberation; they often appear immediately as opening frames. TTC can therefore amplify the initial plan rather than necessarily causing productive adaptation later.",
        ],
        "hot_category_phase_distribution": [
            "**What it is:** another timing plot, but by interaction phase: discussion, private thinking, proposal, voting, reflection, and formal outcome. Again, each row sums to 100%.",
            "**How to read it:** discussion-heavy categories are public persuasion tactics. Private-thinking or voting-heavy categories reflect internal evaluation, red-line setting, or final accept/reject behavior.",
            "**Why it matters:** this separates visible persuasion from internal self-anchoring. Self-interest/exploitation showing up outside public discussion is important because it suggests TTC can also change the agent's internal objective framing, not just its messages.",
        ],
        "hot_category_discussion_turn_distribution": [
            "**What it is:** among discussion messages only, this shows whether each category appears when the target speaks first or later within a round. Each row sums to 100%.",
            "**How to read it:** a category concentrated on Turn 1 is usually an opening-frame tactic; a category concentrated later is more reactive to the other agent's proposal or objection.",
            "**Why it matters:** TTC scaling can look flat in payoff while still changing conversational dynamics. This plot tests whether the induced strategies are proactive anchors or reactive repairs.",
        ],
    }

    lines: list[str] = []
    lines.append("# TTC Hot Strategic Tag Analysis")
    lines.append("")
    lines.append("This report analyzes the 29 `hot` strategic tags selected in `strategic_tag_review_final.json` against the 216 TTC rollouts. The main unit is a target-authored tag event, because the target model is the one whose test-time compute varies.")
    lines.append("")
    lines.append("## Core Finding")
    lines.append("")
    lines.append("The strategic-tag data does not support a simple story that more TTC makes agents uniformly more persuasive or uniformly more cooperative. The clearer pattern is mechanistic and provider-specific:")
    lines.append("")
    lines.append("- **Gemini shows the clearest shift from bargaining toward diagnosis and formalization.** As observed target tokens rise, Gemini uses more logical persuasion, cost policing, and formalization, while trade/compromise and self-interest/exploitation fall.")
    lines.append("- **GPT-5 shifts toward self-advocacy and pressure.** Extra compute makes GPT-5 more explicit about its own value maximization, arithmetic, and veto conditions, but this is not the same as becoming a better deal-maker.")
    lines.append("- **Claude is qualitatively rich but weaker as a scaling curve.** Claude has hot tags, but its observed token proxy does not increase cleanly with requested effort, so Claude should be interpreted as a provider case study rather than the main monotone TTC test.")
    lines.append("- **The reason TTC does not scale like model Elo is that it intensifies local tactics rather than giving the agent a uniformly better bargaining policy.** Extra reasoning can produce arithmetic receipts, vetoes, cost policing, and leverage preservation, but those tactics often harden positions, expose infeasibility, or create procedural churn instead of improving settlement quality.")
    lines.append("")
    lines.append("## Hot Labels")
    lines.append("")
    lines.append(f"Hot labels: {len(hot)} / 50. Categories represented: " + ", ".join(sorted(hot["category"].unique())) + ".")
    lines.append("")
    lines.append("## Paper-Ready Claim")
    lines.append("")
    lines.append("The best-supported version of the TTC result is not that extra reasoning makes the agent more fair, more selfish, or more persuasive in one uniform direction. The stronger claim is that **TTC makes the agent's bargaining posture more explicit and internally defended**. For GPT-5, that mostly means more self-advocacy, utility arithmetic, and veto conditions. For Gemini, it means a shift away from ordinary quid-pro-quo bargaining and toward rule-checking, arithmetic receipts, cost policing, and fairness accusations. These are cognitively plausible benefits of extra deliberation, but they are not the same as a globally better bargaining policy. They can clarify the feasible frontier, but they can also harden positions and convert a negotiation into a justification exercise.")
    lines.append("")
    lines.append("That explains the apparent mismatch with Elo scaling. A stronger model can have better priors about the other agent, better social calibration, and a better implicit policy for choosing when to concede. Extra TTC inside the same model often deepens the policy it already selected: it makes the current position more coherent, better documented, and easier to defend. When the initial posture is constructive, this can help. When the initial posture is brittle or self-protective, the extra tokens become receipts for why the agent should not move.")
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    for path in plot_paths:
        title = path.stem.replace("_", " ").title()
        lines.append(f"### {title}")
        lines.append("")
        for note in plot_notes.get(path.stem, []):
            lines.append(note)
            lines.append("")
        lines.append(markdown_embedded_png(path, path.stem))
        lines.append("")
    lines.append("## Category-Level Trends")
    lines.append("")
    lines.append("### GPT-5 weak-to-high category deltas")
    lines.append("")
    lines.append(md_table(gpt_cat, ["category", "weak_events_per_rollout", "strong_events_per_rollout", "delta_weak_to_strong"], limit=10))
    lines.append("")
    lines.append("### Gemini weak-to-high category deltas")
    lines.append("")
    lines.append(md_table(gem_cat, ["category", "weak_events_per_rollout", "strong_events_per_rollout", "delta_weak_to_strong"], limit=10))
    lines.append("")
    lines.append("### Claude low-to-max category deltas")
    lines.append("")
    lines.append(md_table(claude_cat, ["category", "weak_events_per_rollout", "strong_events_per_rollout", "delta_weak_to_strong"], limit=10))
    lines.append("")
    lines.append("## What The Main Trends Say")
    lines.append("")
    lines.append("**GPT-5:** weak-to-high TTC most strongly increases self-interest/exploitation (+1.50 events/rollout), pressure (+0.72), and logical persuasion (+0.61). The exact labels show the mechanism: self-advocacy/value maximization (+1.17), utility arithmetic receipts (+0.78), and conditional veto threats (+0.78). This is a clean 'defend my position better' signature. It is deliberation as sharper self-representation, not deliberation as compromise discovery.")
    lines.append("")
    lines.append("**Gemini 3 Flash:** weak-to-high TTC increases logical persuasion (+1.44), formalization (+1.28), and pressure (+0.61), but decreases trade/compromise (-1.22) and self-interest/exploitation (-0.67). The exact labels show a different mechanism from GPT-5: counter-anchor cost policing (+0.89), utility arithmetic receipts (+0.83), fairness accusation pressure (+0.56), and budget carryover hallucination (+0.39) rise, while conditional quid pro quo falls (-1.00). This looks less like raw selfishness and more like a move from bargaining to adjudication: the model spends more compute policing constraints and explaining why an offer is valid or invalid.")
    lines.append("")
    lines.append("**Claude Sonnet 4.6:** Claude contains many hot strategic events, but the observed token x-axis is not a clean effort ladder. The low-to-max deltas are therefore useful as qualitative contrasts, not as the main causal TTC scaling evidence.")
    lines.append("")
    lines.append("**Timing:** most hot strategic behavior appears early, especially in public discussion. The discussion-turn plot adds an important distinction: self-interest/exploitation and trade/compromise are more often opening-frame behaviors, while pressure, logical persuasion, and emotional persuasion are more often second-turn/reactive behaviors. This supports the mechanism that extra reasoning often elaborates or defends a stance quickly, rather than gradually discovering compromise late in the negotiation.")
    lines.append("")
    lines.append("## Strongest GPT/Gemini Tag Changes")
    lines.append("")
    focus = tag_trends[tag_trends["family"].isin(["gpt-5", "gemini-3-flash"])].copy()
    lines.append("### Largest increases")
    lines.append("")
    lines.append(md_table(focus.sort_values("delta_weak_to_strong", ascending=False), ["family", "tag_title", "category", "weak_events_per_rollout", "strong_events_per_rollout", "delta_weak_to_strong"], limit=14))
    lines.append("")
    lines.append("### Largest decreases")
    lines.append("")
    lines.append(md_table(focus.sort_values("delta_weak_to_strong"), ["family", "tag_title", "category", "weak_events_per_rollout", "strong_events_per_rollout", "delta_weak_to_strong"], limit=14))
    lines.append("")
    lines.append("## Qualitative Examples")
    lines.append("")
    if not examples.empty:
        for _, row in examples.iterrows():
            lines.append(f"### {FAMILY_LABELS.get(row['family'], row['family'])}: {row['tag_title']} ({row['delta_weak_to_strong']:+.2f})")
            lines.append("")
            lines.append(f"- Config {row['example_config']}, level `{row['level']}`, round `{row['round']}`, phase `{row['phase']}`")
            lines.append(f"- Quote: \"{row['quote']}\"")
            lines.append(f"- Rationale: {row['rationale']}")
            lines.append("")
    lines.append("## Hypotheses")
    lines.append("")
    lines.append("1. **TTC increases tactical explicitness more than strategic wisdom.** The tags that rise most strongly are often arithmetic, cost-policing, veto, self-advocacy, and formalization behaviors. These make the negotiation more legible, but not necessarily more successful.")
    lines.append("2. **TTC can harden red lines.** Conditional veto threats, counter-anchor cost policing, self-advocacy, and leverage preservation are useful locally, but they can reduce openness to settlement. This fits the earlier qualitative observation that deliberation may cause agents to grind into a position.")
    lines.append("3. **TTC helps most when the added tactic is a coordination scaffold.** Vote-history diagnostics, threshold-gap calculations, and agent-specific accounting can improve coordination when a feasible deal exists. These are the most promising positive TTC mechanisms, especially when they translate private preferences into public constraints.")
    lines.append("4. **TTC fails when the game requires preference transformation rather than better argumentation.** In hard cofunding or incompatible-preference cases, more explicit receipts and vetoes expose the impasse but do not create surplus.")
    lines.append("5. **This differs from Elo scaling.** Elo improvements likely reflect better priors, stronger social modeling, and better implicit policy selection. TTC mostly deepens the same local reasoning trajectory within a single model, so it amplifies both constructive and destructive tactics.")
    lines.append("")
    lines.append("## Data Products")
    lines.append("")
    for p in [
        OUT_DIR / "hot_tag_family_level_summary.csv",
        OUT_DIR / "hot_category_family_level_summary.csv",
        OUT_DIR / "hot_tag_trends.csv",
        OUT_DIR / "hot_category_trends.csv",
        OUT_DIR / "hot_round_phase_summary.csv",
        OUT_DIR / "hot_discussion_turn_summary.csv",
        OUT_DIR / "hot_trend_examples.csv",
    ]:
        lines.append(f"- `{p}`")
    lines.append("")
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    review_rows, hot, events, manifest = load_inputs()
    summary, cat_summary = summarize_tags(hot, events, manifest)
    tag_trends, cat_trends = build_trends(summary, cat_summary)
    examples = top_trend_examples(events, tag_trends)

    summary.to_csv(OUT_DIR / "hot_tag_family_level_summary.csv", index=False)
    cat_summary.to_csv(OUT_DIR / "hot_category_family_level_summary.csv", index=False)
    tag_trends.to_csv(OUT_DIR / "hot_tag_trends.csv", index=False)
    cat_trends.to_csv(OUT_DIR / "hot_category_trends.csv", index=False)
    examples.to_csv(OUT_DIR / "hot_trend_examples.csv", index=False)

    target_events = events[events["speaker_role"].eq("target")].copy()
    round_phase = (
        target_events.groupby(["tag_code", "tag_title", "category", "family", "level", "phase", "round"], dropna=False)
        .size()
        .reset_index(name="target_event_count")
    )
    round_phase.to_csv(OUT_DIR / "hot_round_phase_summary.csv", index=False)
    discussion_turn = (
        target_events[target_events["phase"].eq("discussion")]
        .groupby(["tag_code", "tag_title", "category", "family", "level", "round", "discussion_turn"], dropna=False)
        .size()
        .reset_index(name="target_event_count")
    )
    discussion_turn.to_csv(OUT_DIR / "hot_discussion_turn_summary.csv", index=False)

    plot_paths = [
        plot_hot_tags(summary),
        plot_categories(cat_summary),
        *plot_delta_heatmaps(tag_trends, cat_trends),
        *plot_round_phase(events),
        plot_discussion_turn_distribution(events),
    ]

    # Copy plots next to the report so markdown previews resolve without depending on Overleaf paths.
    report_plot_dir = OUT_DIR / "plots"
    report_plot_dir.mkdir(exist_ok=True)
    report_paths = []
    for path in plot_paths:
        dest = report_plot_dir / path.name
        dest.write_bytes(path.read_bytes())
        report_paths.append(dest)

    report = write_report(hot, summary, cat_summary, tag_trends, cat_trends, examples, report_paths)
    for path in [*plot_paths, *report_paths, report]:
        print(path)


if __name__ == "__main__":
    main()

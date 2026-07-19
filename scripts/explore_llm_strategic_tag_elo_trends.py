#!/usr/bin/env python3
"""Explore LLM strategic tag frequencies versus model Elo.

The script consumes the completed adjudication bundle from
analysis/llm_strategic_tag_adjudication_20260628 and writes plots, tables, and
a markdown report for the hot-tag subset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


DEFAULT_INPUT_DIR = Path("analysis/llm_strategic_tag_adjudication_20260628")
DEFAULT_OUTPUT_DIR = Path("analysis/llm_strategic_tag_elo_exploration_20260629")


FAMILY_LABELS = {
    "heterogeneous_random": "Heterogeneous adversary",
    "homogeneous_adversary": "Homogeneous adversary",
    "homogeneous_control": "Homogeneous control",
    "random_monoculture_control": "Random monoculture control",
    "n2_gpt5_bilateral": "N=2 GPT-5-nano bilateral adversary",
}

ADVERSARY_ONLY_FAMILIES = {"homogeneous_adversary", "n2_gpt5_bilateral"}

COMPETITION_ORDER = ["Low competition", "Middle competition", "High competition"]
COMPETITION_COLORS = {
    "Low competition": "#2563eb",
    "Middle competition": "#7c3aed",
    "High competition": "#dc2626",
}

METRIC_MODE = "occurrence"
METRIC_COL = "occurrence_rate"
METRIC_YLABEL = "Speaker-rollout occurrence rate"
METRIC_NOUN = "occurrence"
METRIC_DESC = "speaker-rollout occurrence rate"
METRIC_PERCENT = True


def configure_metric(metric: str) -> None:
    global METRIC_MODE, METRIC_COL, METRIC_YLABEL, METRIC_NOUN, METRIC_DESC, METRIC_PERCENT
    METRIC_MODE = metric
    if metric == "intensity":
        METRIC_COL = "events_per_speaker_rollout"
        METRIC_YLABEL = "Mean tag events per speaker-rollout"
        METRIC_NOUN = "intensity"
        METRIC_DESC = "mean tag-event count per speaker-rollout"
        METRIC_PERCENT = False
    else:
        METRIC_COL = "occurrence_rate"
        METRIC_YLABEL = "Speaker-rollout occurrence rate"
        METRIC_NOUN = "occurrence"
        METRIC_DESC = "speaker-rollout occurrence rate"
        METRIC_PERCENT = True


def slugify(text: str) -> str:
    return (
        text.lower()
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("__", "_")
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def competition_bin(value: Any) -> str:
    if value is None or pd.isna(value):
        return "Unknown competition"
    v = float(value)
    if v <= 0.25:
        return "Low competition"
    if v < 0.75:
        return "Middle competition"
    return "High competition"


def load_hot_tags(path: Path | None, counts_path: Path, limit: int = 29) -> tuple[list[str], str]:
    """Load hot tags from a user JSON if present, otherwise use top event counts."""
    if path and path.exists():
        obj = json.loads(path.read_text())
        if isinstance(obj, list):
            tags = [x["tag_code"] if isinstance(x, dict) else str(x) for x in obj]
        elif isinstance(obj, dict):
            if "responses" in obj:
                tags = [
                    x["tag_code"]
                    for x in obj["responses"]
                    if isinstance(x, dict) and x.get("decision") == "hot"
                ]
            elif "responsesByCode" in obj:
                tags = [
                    code
                    for code, response in obj["responsesByCode"].items()
                    if isinstance(response, dict) and response.get("decision") == "hot"
                ]
            elif "hot_tags" in obj:
                tags = [x["tag_code"] if isinstance(x, dict) else str(x) for x in obj["hot_tags"]]
            elif "tags" in obj:
                tags = [x["tag_code"] if isinstance(x, dict) else str(x) for x in obj["tags"]]
            else:
                tags = [k for k, v in obj.items() if v is True or (isinstance(v, dict) and v.get("hot"))]
        else:
            raise ValueError(f"Unsupported hot tag JSON shape in {path}")
        deduped = list(dict.fromkeys(tags))
        return deduped, f"user-supplied hot-tag JSON: `{path}`"

    counts = pd.read_csv(counts_path).sort_values("event_count", ascending=False)
    return counts.head(limit)["tag_code"].tolist(), f"top {limit} tags by event count because no hot-tag JSON was available"


def build_tag_metadata(input_dir: Path) -> pd.DataFrame:
    codebook = json.loads((input_dir / "llm_tag_codebook.json").read_text())
    return pd.DataFrame(codebook)[["tag_code", "tag_title", "category", "definition", "paper_value"]]


def build_denominators(manifest_rows: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for rollout in manifest_rows:
        family = rollout["experiment_family"]
        for agent, model in rollout["agent_model_map"].items():
            role = rollout["agent_role_map"].get(agent)
            elo = rollout["agent_elo_map"].get(agent)
            if family in ADVERSARY_ONLY_FAMILIES and role != "adversary":
                # For this family the research question is about the adversary model;
                # baseline gpt-5-nano rows would swamp the model-level trend.
                continue
            rows.append(
                {
                    "speaker_key": f"{rollout['result_path']}::{agent}",
                    "result_path": rollout["result_path"],
                    "config_id": str(rollout["config_id"]),
                    "experiment_family": family,
                    "game_label": rollout["game_label"],
                    "n_agents": rollout["n_agents"],
                    "competition_level": rollout.get("competition_level"),
                    "competition_bin": competition_bin(rollout.get("competition_level")),
                    "model_order": rollout.get("model_order"),
                    "speaker_agent": agent,
                    "speaker_model": model,
                    "speaker_role": role,
                    "speaker_elo": elo,
                    "adversary_position": rollout.get("adversary_position"),
                    "consensus_reached": rollout.get("consensus_reached"),
                    "final_round": rollout.get("final_round"),
                    "conversation_log_count": rollout.get("conversation_log_count"),
                }
            )
    return pd.DataFrame(rows)


def build_event_frame(input_dir: Path) -> pd.DataFrame:
    rows = []
    with (input_dir / "llm_event_tags.jsonl").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            family = row["experiment_family"]
            if family in ADVERSARY_ONLY_FAMILIES and row.get("speaker_role") != "adversary":
                continue
            if row.get("speaker_agent") is None:
                continue
            row["speaker_key"] = f"{row['result_path']}::{row['speaker_agent']}"
            rows.append(row)
    return pd.DataFrame(rows)


def make_frequency_tables(
    denoms: pd.DataFrame,
    events: pd.DataFrame,
    tag_meta: pd.DataFrame,
    hot_tags: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tag_title = dict(zip(tag_meta.tag_code, tag_meta.tag_title))
    tag_group = dict(zip(tag_meta.tag_code, tag_meta.category))

    d = denoms.copy()
    d["speaker_elo"] = pd.to_numeric(d["speaker_elo"], errors="coerce")
    denominators = (
        d.groupby(["experiment_family", "speaker_model", "speaker_role", "speaker_elo"], dropna=False)
        .agg(
            speaker_rollouts=("speaker_key", "nunique"),
            source_rollouts=("result_path", "nunique"),
            games=("game_label", lambda x: ",".join(sorted(set(map(str, x))))),
            n_agents_values=("n_agents", lambda x: ",".join(map(str, sorted(set(x))))),
        )
        .reset_index()
    )

    hot_events = events[events["tag_code"].isin(hot_tags)].copy()
    present = (
        hot_events.drop_duplicates(
            ["experiment_family", "speaker_model", "speaker_role", "speaker_elo", "speaker_key", "tag_code"]
        )
        .groupby(["experiment_family", "speaker_model", "speaker_role", "speaker_elo", "tag_code"], dropna=False)
        .agg(tagged_speaker_rollouts=("speaker_key", "nunique"))
        .reset_index()
    )
    counts = (
        hot_events.groupby(["experiment_family", "speaker_model", "speaker_role", "speaker_elo", "tag_code"], dropna=False)
        .agg(event_count=("tag_code", "size"))
        .reset_index()
    )
    grid = denominators.merge(pd.DataFrame({"tag_code": hot_tags}), how="cross")
    tag_freq = (
        grid.merge(present, how="left")
        .merge(counts, how="left")
        .fillna({"tagged_speaker_rollouts": 0, "event_count": 0})
    )
    tag_freq["tag_title"] = tag_freq["tag_code"].map(tag_title)
    tag_freq["group"] = tag_freq["tag_code"].map(tag_group)
    tag_freq["occurrence_rate"] = tag_freq["tagged_speaker_rollouts"] / tag_freq["speaker_rollouts"]
    tag_freq["events_per_speaker_rollout"] = tag_freq["event_count"] / tag_freq["speaker_rollouts"]
    tag_freq["events_per_100_speaker_rollouts"] = 100 * tag_freq["event_count"] / tag_freq["speaker_rollouts"]

    group_map = tag_meta[tag_meta.tag_code.isin(hot_tags)][["tag_code", "category"]].rename(columns={"category": "group"})
    ge = hot_events.merge(group_map, on="tag_code", how="left")
    gp = (
        ge.drop_duplicates(
            ["experiment_family", "speaker_model", "speaker_role", "speaker_elo", "speaker_key", "group"]
        )
        .groupby(["experiment_family", "speaker_model", "speaker_role", "speaker_elo", "group"], dropna=False)
        .agg(tagged_speaker_rollouts=("speaker_key", "nunique"))
        .reset_index()
    )
    gc = (
        ge.groupby(["experiment_family", "speaker_model", "speaker_role", "speaker_elo", "group"], dropna=False)
        .agg(event_count=("tag_code", "size"))
        .reset_index()
    )
    groups = sorted(group_map.group.dropna().unique())
    group_grid = denominators.merge(pd.DataFrame({"group": groups}), how="cross")
    group_freq = (
        group_grid.merge(gp, how="left")
        .merge(gc, how="left")
        .fillna({"tagged_speaker_rollouts": 0, "event_count": 0})
    )
    group_freq["occurrence_rate"] = group_freq["tagged_speaker_rollouts"] / group_freq["speaker_rollouts"]
    group_freq["events_per_speaker_rollout"] = group_freq["event_count"] / group_freq["speaker_rollouts"]
    group_freq["events_per_100_speaker_rollouts"] = 100 * group_freq["event_count"] / group_freq["speaker_rollouts"]

    return tag_freq, group_freq, denominators


def make_competition_frequency_tables(
    denoms: pd.DataFrame,
    events: pd.DataFrame,
    tag_meta: pd.DataFrame,
    hot_tags: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tag_title = dict(zip(tag_meta.tag_code, tag_meta.tag_title))
    tag_group = dict(zip(tag_meta.tag_code, tag_meta.category))
    d = denoms.copy()
    d["speaker_elo"] = pd.to_numeric(d["speaker_elo"], errors="coerce")
    speaker_comp = d[["speaker_key", "competition_level", "competition_bin"]].drop_duplicates()
    denominators = (
        d.groupby(
            ["experiment_family", "competition_bin", "speaker_model", "speaker_role", "speaker_elo"],
            dropna=False,
        )
        .agg(
            speaker_rollouts=("speaker_key", "nunique"),
            source_rollouts=("result_path", "nunique"),
            games=("game_label", lambda x: ",".join(sorted(set(map(str, x))))),
            n_agents_values=("n_agents", lambda x: ",".join(map(str, sorted(set(x))))),
            competition_levels=("competition_level", lambda x: ",".join(map(str, sorted(set(x))))),
        )
        .reset_index()
    )

    hot_events = events[events["tag_code"].isin(hot_tags)].merge(speaker_comp, on="speaker_key", how="left")
    present = (
        hot_events.drop_duplicates(
            [
                "experiment_family",
                "competition_bin",
                "speaker_model",
                "speaker_role",
                "speaker_elo",
                "speaker_key",
                "tag_code",
            ]
        )
        .groupby(
            ["experiment_family", "competition_bin", "speaker_model", "speaker_role", "speaker_elo", "tag_code"],
            dropna=False,
        )
        .agg(tagged_speaker_rollouts=("speaker_key", "nunique"))
        .reset_index()
    )
    counts = (
        hot_events.groupby(
            ["experiment_family", "competition_bin", "speaker_model", "speaker_role", "speaker_elo", "tag_code"],
            dropna=False,
        )
        .agg(event_count=("tag_code", "size"))
        .reset_index()
    )
    grid = denominators.merge(pd.DataFrame({"tag_code": hot_tags}), how="cross")
    tag_freq = (
        grid.merge(present, how="left")
        .merge(counts, how="left")
        .fillna({"tagged_speaker_rollouts": 0, "event_count": 0})
    )
    tag_freq["tag_title"] = tag_freq["tag_code"].map(tag_title)
    tag_freq["group"] = tag_freq["tag_code"].map(tag_group)
    tag_freq["occurrence_rate"] = tag_freq["tagged_speaker_rollouts"] / tag_freq["speaker_rollouts"]
    tag_freq["events_per_speaker_rollout"] = tag_freq["event_count"] / tag_freq["speaker_rollouts"]
    tag_freq["events_per_100_speaker_rollouts"] = 100 * tag_freq["event_count"] / tag_freq["speaker_rollouts"]

    group_map = tag_meta[tag_meta.tag_code.isin(hot_tags)][["tag_code", "category"]].rename(columns={"category": "group"})
    ge = hot_events.merge(group_map, on="tag_code", how="left")
    gp = (
        ge.drop_duplicates(
            ["experiment_family", "competition_bin", "speaker_model", "speaker_role", "speaker_elo", "speaker_key", "group"]
        )
        .groupby(["experiment_family", "competition_bin", "speaker_model", "speaker_role", "speaker_elo", "group"], dropna=False)
        .agg(tagged_speaker_rollouts=("speaker_key", "nunique"))
        .reset_index()
    )
    gc = (
        ge.groupby(["experiment_family", "competition_bin", "speaker_model", "speaker_role", "speaker_elo", "group"], dropna=False)
        .agg(event_count=("tag_code", "size"))
        .reset_index()
    )
    groups = sorted(group_map.group.dropna().unique())
    group_grid = denominators.merge(pd.DataFrame({"group": groups}), how="cross")
    group_freq = (
        group_grid.merge(gp, how="left")
        .merge(gc, how="left")
        .fillna({"tagged_speaker_rollouts": 0, "event_count": 0})
    )
    group_freq["occurrence_rate"] = group_freq["tagged_speaker_rollouts"] / group_freq["speaker_rollouts"]
    group_freq["events_per_speaker_rollout"] = group_freq["event_count"] / group_freq["speaker_rollouts"]
    group_freq["events_per_100_speaker_rollouts"] = 100 * group_freq["event_count"] / group_freq["speaker_rollouts"]
    return tag_freq, group_freq


def trend_table(freq: pd.DataFrame, unit_col: str, label_col: str) -> pd.DataFrame:
    rows = []
    for (family, unit), g in freq.groupby(["experiment_family", unit_col]):
        g = g.dropna(subset=["speaker_elo"])
        if len(g) < 4 or g["speaker_elo"].nunique() < 4:
            continue
        y = g[METRIC_COL].to_numpy(dtype=float)
        x = g["speaker_elo"].to_numpy(dtype=float)
        weights = g["speaker_rollouts"].to_numpy(dtype=float)
        if np.allclose(y, y[0]):
            spearman_r = 0.0
            spearman_p = 1.0
            pearson_r = 0.0
            pearson_p = 1.0
        else:
            spearman_r, spearman_p = stats.spearmanr(x, y)
            pearson_r, pearson_p = stats.pearsonr(x, y)
        slope, intercept = np.polyfit(x, y, deg=1, w=np.sqrt(weights))
        lo = g.loc[g["speaker_elo"].idxmin()]
        hi = g.loc[g["speaker_elo"].idxmax()]
        rows.append(
            {
                "experiment_family": family,
                unit_col: unit,
                "label": g[label_col].iloc[0] if label_col in g else unit,
                "n_models": len(g),
                "elo_min": x.min(),
                "elo_max": x.max(),
                "low_elo_rate": lo[METRIC_COL],
                "high_elo_rate": hi[METRIC_COL],
                "delta_high_minus_low": hi[METRIC_COL] - lo[METRIC_COL],
                "slope_per_100_elo": slope * 100,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "mean_rate": y.mean(),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["trend_strength"] = out["spearman_r"].abs() * np.sqrt(out["n_models"])
    return out.sort_values(["experiment_family", "trend_strength"], ascending=[True, False])


def competition_trend_table(freq: pd.DataFrame, unit_col: str, label_col: str) -> pd.DataFrame:
    rows = []
    for (family, comp_bin, unit), g in freq.groupby(["experiment_family", "competition_bin", unit_col]):
        g = g.dropna(subset=["speaker_elo"])
        if len(g) < 4 or g["speaker_elo"].nunique() < 4:
            continue
        y = g[METRIC_COL].to_numpy(dtype=float)
        x = g["speaker_elo"].to_numpy(dtype=float)
        weights = g["speaker_rollouts"].to_numpy(dtype=float)
        if np.allclose(y, y[0]):
            spearman_r = 0.0
            spearman_p = 1.0
            pearson_r = 0.0
            pearson_p = 1.0
        else:
            spearman_r, spearman_p = stats.spearmanr(x, y)
            pearson_r, pearson_p = stats.pearsonr(x, y)
        slope, _ = np.polyfit(x, y, deg=1, w=np.sqrt(weights))
        lo = g.loc[g["speaker_elo"].idxmin()]
        hi = g.loc[g["speaker_elo"].idxmax()]
        rows.append(
            {
                "experiment_family": family,
                "competition_bin": comp_bin,
                unit_col: unit,
                "label": g[label_col].iloc[0] if label_col in g else unit,
                "n_models": len(g),
                "elo_min": x.min(),
                "elo_max": x.max(),
                "low_elo_rate": lo[METRIC_COL],
                "high_elo_rate": hi[METRIC_COL],
                "delta_high_minus_low": hi[METRIC_COL] - lo[METRIC_COL],
                "slope_per_100_elo": slope * 100,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "mean_rate": y.mean(),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["trend_strength"] = out["spearman_r"].abs() * np.sqrt(out["n_models"])
    out["competition_order"] = out["competition_bin"].map({v: i for i, v in enumerate(COMPETITION_ORDER)}).fillna(99)
    return out.sort_values(["experiment_family", "competition_order", "trend_strength"], ascending=[True, True, False])


def save_heatmap(
    freq: pd.DataFrame,
    family: str,
    value_col: str,
    index_col: str,
    column_col: str,
    out_path: Path,
    title: str,
    model_label_map: dict[str, str] | None = None,
    width_per_col: float = 0.38,
) -> None:
    sub = freq[freq["experiment_family"] == family].copy()
    sub = sub.sort_values(["speaker_elo", "speaker_model"], na_position="last")
    if model_label_map:
        sub[index_col] = sub[index_col].map(model_label_map).fillna(sub[index_col])
    pivot = sub.pivot_table(index=index_col, columns=column_col, values=value_col, aggfunc="mean")
    pivot = pivot.loc[~pivot.index.duplicated(keep="first")]
    if pivot.empty:
        return
    h = max(5.5, 0.34 * len(pivot) + 2.8)
    w = max(8.0, width_per_col * len(pivot.columns) + 4.2)
    plt.figure(figsize=(w, h), constrained_layout=True)
    sns.heatmap(
        pivot,
        cmap="viridis",
        vmin=0,
        vmax=max(0.01, np.nanpercentile(pivot.to_numpy(dtype=float), 95)),
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "speaker-rollout occurrence rate"},
    )
    plt.title(title)
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(rotation=55, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_line_grid(
    freq: pd.DataFrame,
    family: str,
    unit_col: str,
    label_col: str,
    out_path: Path,
    title: str,
    ncols: int = 4,
) -> None:
    sub = freq[freq["experiment_family"] == family].copy()
    if sub.empty:
        return
    if unit_col == label_col:
        units = [{unit_col: value, label_col: value} for value in sorted(sub[unit_col].dropna().unique())]
    else:
        units = (
            sub[[unit_col, label_col]]
            .drop_duplicates()
            .sort_values(label_col)
            .to_dict("records")
        )
    nrows = math.ceil(len(units) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.9 * nrows), squeeze=False)
    max_y = max(0.01, float(sub[METRIC_COL].max()) * 1.15)
    for ax, unit in zip(axes.ravel(), units):
        g = sub[sub[unit_col] == unit[unit_col]].copy()
        g = g.sort_values(["speaker_elo", "speaker_model"], na_position="last")
        if g["speaker_elo"].notna().any() and g["speaker_elo"].nunique() > 1:
            ax.plot(g["speaker_elo"], g[METRIC_COL], marker="o", linewidth=1.6, markersize=4)
            ax.set_xlabel("Elo", fontsize=8)
        else:
            x = np.arange(len(g))
            ax.plot(x, g[METRIC_COL], marker="o", linewidth=1.6, markersize=5)
            labels = [str(m).split("-")[0] for m in g["speaker_model"]]
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
            ax.set_xlabel("Model", fontsize=8)
        ax.set_title(str(unit[label_col]), fontsize=9)
        ax.set_ylim(0, max_y)
        ax.grid(True, alpha=0.35)
        ax.tick_params(axis="both", labelsize=7)
    for ax in axes.ravel()[len(units):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.supylabel(METRIC_YLABEL, fontsize=10)
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.98])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_competition_line_grid(
    freq: pd.DataFrame,
    family: str,
    unit_col: str,
    label_col: str,
    out_path: Path,
    title: str,
    ncols: int = 4,
) -> None:
    sub = freq[freq["experiment_family"] == family].copy()
    if sub.empty:
        return
    if unit_col == label_col:
        units = [{unit_col: value, label_col: value} for value in sorted(sub[unit_col].dropna().unique())]
    else:
        units = (
            sub[[unit_col, label_col]]
            .drop_duplicates()
            .sort_values(label_col)
            .to_dict("records")
        )
    nrows = math.ceil(len(units) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.15 * ncols, 3.05 * nrows), squeeze=False)
    max_y = max(0.01, float(sub[METRIC_COL].max()) * 1.15)
    for ax, unit in zip(axes.ravel(), units):
        u = sub[sub[unit_col] == unit[unit_col]].copy()
        for comp in COMPETITION_ORDER:
            g = u[u["competition_bin"] == comp].sort_values(["speaker_elo", "speaker_model"], na_position="last")
            if g.empty:
                continue
            if g["speaker_elo"].notna().any() and g["speaker_elo"].nunique() > 1:
                ax.plot(
                    g["speaker_elo"],
                    g[METRIC_COL],
                    marker="o",
                    linewidth=1.4,
                    markersize=3.5,
                    label=comp,
                    color=COMPETITION_COLORS.get(comp),
                    alpha=0.9,
                )
            else:
                x = np.arange(len(g))
                ax.plot(
                    x,
                    g[METRIC_COL],
                    marker="o",
                    linewidth=1.4,
                    markersize=4,
                    label=comp,
                    color=COMPETITION_COLORS.get(comp),
                    alpha=0.9,
                )
        ax.set_title(str(unit[label_col]), fontsize=8.5)
        ax.set_ylim(0, max_y)
        ax.grid(True, alpha=0.35)
        ax.tick_params(axis="both", labelsize=7)
    for ax in axes.ravel()[len(units):]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(title, fontsize=14, y=1.01)
    fig.supxlabel("Model Elo", fontsize=10)
    fig.supylabel(METRIC_YLABEL, fontsize=10)
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.97])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_competition_slope_bars(
    trends: pd.DataFrame,
    family: str,
    unit_col: str,
    out_path: Path,
    title: str,
    top_n: int | None = None,
) -> None:
    sub = trends[trends["experiment_family"] == family].copy()
    if sub.empty:
        return
    if top_n:
        ranked_units = (
            sub.groupby(unit_col)["trend_strength"]
            .max()
            .sort_values(ascending=False)
            .head(top_n)
            .index
        )
        sub = sub[sub[unit_col].isin(ranked_units)]
    pivot = sub.pivot_table(index="label", columns="competition_bin", values="slope_per_100_elo", aggfunc="mean")
    cols = [c for c in COMPETITION_ORDER if c in pivot.columns]
    pivot = pivot[cols]
    pivot["max_abs"] = pivot.abs().max(axis=1)
    pivot = pivot.sort_values("max_abs", ascending=True).drop(columns=["max_abs"])
    fig_h = max(6.0, 0.32 * len(pivot) + 2.0)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    y = np.arange(len(pivot))
    width = 0.24
    offsets = np.linspace(-width, width, max(1, len(cols)))
    for offset, col in zip(offsets, cols):
        ax.barh(y + offset, pivot[col], height=width, label=col, color=COMPETITION_COLORS.get(col), alpha=0.86)
    ax.axvline(0, color="#111827", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xlabel(f"Slope in {METRIC_DESC} per 100 Elo")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.35)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_trend_scatter(
    freq: pd.DataFrame,
    trend: pd.DataFrame,
    family: str,
    unit_col: str,
    label_col: str,
    out_dir: Path,
    top_n: int,
) -> list[Path]:
    paths = []
    ranked = trend[trend["experiment_family"] == family].copy()
    if ranked.empty:
        return paths
    ranked = ranked.sort_values("trend_strength", ascending=False).head(top_n)
    for _, row in ranked.iterrows():
        unit = row[unit_col]
        g = freq[(freq["experiment_family"] == family) & (freq[unit_col] == unit)].dropna(subset=["speaker_elo"])
        if len(g) < 4:
            continue
        out = out_dir / f"{family}_{slugify(str(unit))}_trend.png"
        plt.figure(figsize=(6.0, 4.0))
        sizes = np.clip(g["speaker_rollouts"].to_numpy(dtype=float), 20, 400)
        plt.scatter(g["speaker_elo"], g[METRIC_COL], s=sizes, alpha=0.78)
        if g["speaker_elo"].nunique() >= 2:
            coef = np.polyfit(g["speaker_elo"], g[METRIC_COL], 1, w=np.sqrt(g["speaker_rollouts"]))
            xs = np.linspace(g["speaker_elo"].min(), g["speaker_elo"].max(), 100)
            plt.plot(xs, coef[0] * xs + coef[1], color="#222222", linewidth=1.5)
        for _, point in g.iterrows():
            label = str(point["speaker_model"]).split("-")[0]
            plt.annotate(label, (point["speaker_elo"], point[METRIC_COL]), fontsize=6, alpha=0.75)
        plt.title(f"{FAMILY_LABELS.get(family, family)}: {row['label']}")
        plt.xlabel("Model Elo")
        plt.ylabel(METRIC_YLABEL)
        plt.ylim(bottom=0)
        plt.tight_layout()
        plt.savefig(out, dpi=220)
        plt.close()
        paths.append(out)
    return paths


def build_payoff_frame(manifest_rows: list[dict[str, Any]], denoms: pd.DataFrame) -> pd.DataFrame:
    payoff_rows = []
    for rollout in manifest_rows:
        result_path = rollout["result_path"]
        try:
            payload = json.loads(Path(result_path).read_text())
        except Exception:
            continue
        utilities = payload.get("final_utilities") or {}
        if not isinstance(utilities, dict) or not utilities:
            continue
        values = [float(v) for v in utilities.values() if v is not None]
        if not values:
            continue
        rollout_mean = float(np.mean(values))
        rollout_total = float(np.sum(values))
        for agent, utility in utilities.items():
            if utility is None:
                continue
            payoff_rows.append(
                {
                    "speaker_key": f"{result_path}::{agent}",
                    "result_path": result_path,
                    "speaker_agent": agent,
                    "final_utility": float(utility),
                    "rollout_mean_utility": rollout_mean,
                    "rollout_total_utility": rollout_total,
                    "relative_utility": float(utility) - rollout_mean,
                }
            )
    payoff = pd.DataFrame(payoff_rows)
    out = denoms.merge(payoff, on=["speaker_key", "result_path", "speaker_agent"], how="left")
    group_cols = ["experiment_family", "game_label", "n_agents", "competition_level"]
    out["utility_z_within_cell"] = out.groupby(group_cols, dropna=False)["final_utility"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) and not pd.isna(s.std(ddof=0)) else 0.0
    )
    return out


def _safe_corr(x: pd.Series, y: pd.Series, method: str = "pearson") -> tuple[float, float]:
    clean = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(clean) < 4 or clean["x"].nunique() < 2 or clean["y"].nunique() < 2:
        return math.nan, math.nan
    if method == "spearman":
        r, p = stats.spearmanr(clean["x"], clean["y"])
    else:
        r, p = stats.pearsonr(clean["x"], clean["y"])
    return float(r), float(p)


def payoff_correlation_tables(
    speaker_payoffs: pd.DataFrame,
    events: pd.DataFrame,
    tag_meta: pd.DataFrame,
    hot_tags: list[str],
    tag_freq: pd.DataFrame,
    group_freq: pd.DataFrame,
    split_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_cols = split_cols or []
    group_keys = ["experiment_family"] + split_cols
    tag_title = dict(zip(tag_meta.tag_code, tag_meta.tag_title))
    tag_group = dict(zip(tag_meta.tag_code, tag_meta.category))
    base = speaker_payoffs.dropna(subset=["final_utility"]).copy()
    hot_events = events[events["tag_code"].isin(hot_tags)].copy()

    def key_dict(key_values: Any) -> dict[str, Any]:
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        return dict(zip(group_keys, key_values))

    def speaker_table(unit_col: str, units: list[str], label_map: dict[str, str], group_map: dict[str, str] | None) -> pd.DataFrame:
        rows = []
        if unit_col == "tag_code":
            present = (
                hot_events.groupby(["speaker_key", "tag_code"], as_index=False)
                .agg(event_count=("tag_code", "size"))
            )
        else:
            ge = hot_events.copy()
            ge["group"] = ge["tag_code"].map(tag_group)
            present = (
                ge.groupby(["speaker_key", "group"], as_index=False)
                .agg(event_count=("tag_code", "size"))
            )
        for key_values, fam_base in base.groupby(group_keys, dropna=False):
            keys = key_dict(key_values)
            for unit in units:
                sub = fam_base[["speaker_key", "final_utility", "relative_utility", "utility_z_within_cell"]].copy()
                p = present[present[unit_col] == unit][["speaker_key", "event_count"]]
                sub = sub.merge(p, on="speaker_key", how="left")
                sub["event_count"] = sub["event_count"].fillna(0)
                sub["used"] = sub["event_count"] > 0
                used = sub[sub["used"]]
                unused = sub[~sub["used"]]
                r_used, p_used = _safe_corr(sub["used"].astype(float), sub["final_utility"])
                r_rel, p_rel = _safe_corr(sub["used"].astype(float), sub["relative_utility"])
                r_count, p_count = _safe_corr(sub["event_count"], sub["final_utility"], method="spearman")
                rows.append(
                    {
                        **keys,
                        unit_col: unit,
                        "label": label_map.get(unit, unit),
                        "group": group_map.get(unit) if group_map else unit,
                        "n_speaker_rollouts": len(sub),
                        "n_used": int(sub["used"].sum()),
                        "used_rate": float(sub["used"].mean()),
                        "mean_utility_used": float(used["final_utility"].mean()) if len(used) else math.nan,
                        "mean_utility_not_used": float(unused["final_utility"].mean()) if len(unused) else math.nan,
                        "delta_utility_used_minus_not": (
                            float(used["final_utility"].mean() - unused["final_utility"].mean())
                            if len(used) and len(unused)
                            else math.nan
                        ),
                        "mean_relative_utility_used": float(used["relative_utility"].mean()) if len(used) else math.nan,
                        "mean_relative_utility_not_used": float(unused["relative_utility"].mean()) if len(unused) else math.nan,
                        "delta_relative_utility_used_minus_not": (
                            float(used["relative_utility"].mean() - unused["relative_utility"].mean())
                            if len(used) and len(unused)
                            else math.nan
                        ),
                        "point_biserial_r_utility": r_used,
                        "point_biserial_p_utility": p_used,
                        "point_biserial_r_relative_utility": r_rel,
                        "point_biserial_p_relative_utility": p_rel,
                        "spearman_event_count_r_utility": r_count,
                        "spearman_event_count_p_utility": p_count,
                    }
                )
        return pd.DataFrame(rows)

    tag_speaker = speaker_table("tag_code", hot_tags, tag_title, tag_group)
    groups = sorted({tag_group[t] for t in hot_tags})
    group_speaker = speaker_table("group", groups, {g: g for g in groups}, None)

    model_payoff = (
        base.groupby(group_keys + ["speaker_model", "speaker_role", "speaker_elo"], dropna=False)
        .agg(
            mean_final_utility=("final_utility", "mean"),
            mean_relative_utility=("relative_utility", "mean"),
            mean_utility_z=("utility_z_within_cell", "mean"),
            speaker_rollouts=("speaker_key", "nunique"),
        )
        .reset_index()
    )

    def model_table(freq: pd.DataFrame, unit_col: str, label_col: str) -> pd.DataFrame:
        rows = []
        merged = freq.merge(
            model_payoff,
            on=group_keys + ["speaker_model", "speaker_role", "speaker_elo", "speaker_rollouts"],
            how="left",
        )
        for key_values, sub in merged.groupby(group_keys + [unit_col], dropna=False):
            key_values = key_values if isinstance(key_values, tuple) else (key_values,)
            keys = dict(zip(group_keys, key_values[: len(group_keys)]))
            unit = key_values[-1]
            if sub["speaker_model"].nunique() < 4:
                continue
            r_utility, p_utility = _safe_corr(sub[METRIC_COL], sub["mean_final_utility"], method="spearman")
            r_relative, p_relative = _safe_corr(sub[METRIC_COL], sub["mean_relative_utility"], method="spearman")
            r_z, p_z = _safe_corr(sub[METRIC_COL], sub["mean_utility_z"], method="spearman")
            rows.append(
                {
                    **keys,
                    unit_col: unit,
                    "label": sub[label_col].iloc[0] if label_col in sub else unit,
                    "n_models": sub["speaker_model"].nunique(),
                    "spearman_model_rate_vs_mean_utility": r_utility,
                    "p_model_rate_vs_mean_utility": p_utility,
                    "spearman_model_rate_vs_mean_relative_utility": r_relative,
                    "p_model_rate_vs_mean_relative_utility": p_relative,
                    "spearman_model_rate_vs_mean_utility_z": r_z,
                    "p_model_rate_vs_mean_utility_z": p_z,
                }
            )
        return pd.DataFrame(rows)

    tag_model = model_table(tag_freq, "tag_code", "tag_title")
    group_model = model_table(group_freq, "group", "group")
    frames = [tag_speaker, group_speaker, tag_model, group_model]
    for frame in frames:
        if "competition_bin" in frame.columns:
            frame["competition_order"] = frame["competition_bin"].map({v: i for i, v in enumerate(COMPETITION_ORDER)}).fillna(99)
    return tag_speaker, group_speaker, tag_model, group_model


def save_payoff_barplot(df: pd.DataFrame, unit_col: str, value_col: str, out_path: Path, title: str, n: int | None = None) -> None:
    if df.empty:
        return
    d = df.copy()
    if n:
        pos = d.sort_values(value_col, ascending=False).head(n)
        neg = d.sort_values(value_col, ascending=True).head(n)
        d = pd.concat([pos, neg], ignore_index=True).drop_duplicates([unit_col, "experiment_family"])
    d = d.sort_values(value_col)
    labels = d["label"].astype(str) + " [" + d["experiment_family"].map(FAMILY_LABELS).fillna(d["experiment_family"]) + "]"
    fig_h = max(6.0, 0.28 * len(d) + 1.8)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    colors = np.where(d[value_col] >= 0, "#2563eb", "#dc2626")
    ax.barh(labels, d[value_col], color=colors, alpha=0.85)
    ax.axvline(0, color="#111827", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(value_col.replace("_", " "))
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, axis="x", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_competition_payoff_bars(
    df: pd.DataFrame,
    unit_col: str,
    value_col: str,
    out_path: Path,
    title: str,
    n: int | None = None,
) -> None:
    if df.empty or "competition_bin" not in df.columns:
        return
    d = df[df["competition_bin"].isin(COMPETITION_ORDER)].dropna(subset=[value_col]).copy()
    if d.empty:
        return
    d["display_label"] = (
        d["label"].astype(str)
        + " ["
        + d["experiment_family"].map(FAMILY_LABELS).fillna(d["experiment_family"])
        + "]"
    )
    if n:
        ranked = d.groupby("display_label")[value_col].apply(lambda s: s.abs().max()).sort_values(ascending=False).head(n).index
        d = d[d["display_label"].isin(ranked)]
    pivot = d.pivot_table(index="display_label", columns="competition_bin", values=value_col, aggfunc="mean")
    cols = [c for c in COMPETITION_ORDER if c in pivot.columns]
    pivot = pivot[cols]
    pivot["max_abs"] = pivot.abs().max(axis=1)
    pivot = pivot.sort_values("max_abs", ascending=True).drop(columns=["max_abs"])
    fig_h = max(6.0, 0.33 * len(pivot) + 2.0)
    fig, ax = plt.subplots(figsize=(11.0, fig_h))
    y = np.arange(len(pivot))
    width = 0.24
    offsets = np.linspace(-width, width, max(1, len(cols)))
    for offset, col in zip(offsets, cols):
        ax.barh(y + offset, pivot[col], height=width, label=col, color=COMPETITION_COLORS.get(col), alpha=0.86)
    ax.axvline(0, color="#111827", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xlabel(value_col.replace("_", " "))
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.35)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def strongest_consistent(tag_trends: pd.DataFrame) -> pd.DataFrame:
    het = tag_trends[tag_trends.experiment_family == "heterogeneous_random"]
    hom = tag_trends[tag_trends.experiment_family == "homogeneous_adversary"]
    if het.empty or hom.empty:
        return pd.DataFrame()
    m = het.merge(hom, on="tag_code", suffixes=("_hetero", "_homadv"))
    m = m[(m.spearman_r_hetero * m.spearman_r_homadv) > 0]
    if m.empty:
        return m
    m["combined_abs_spearman"] = m.spearman_r_hetero.abs() + m.spearman_r_homadv.abs()
    m["combined_direction"] = np.where(m.spearman_r_hetero > 0, "increases with Elo", "decreases with Elo")
    return m.sort_values("combined_abs_spearman", ascending=False)


def write_markdown_report(
    out_dir: Path,
    input_dir: Path,
    hot_source: str,
    hot_tags: list[str],
    tag_meta: pd.DataFrame,
    tag_freq: pd.DataFrame,
    group_freq: pd.DataFrame,
    tag_trends: pd.DataFrame,
    group_trends: pd.DataFrame,
    denominators: pd.DataFrame,
    tag_payoff_speaker: pd.DataFrame,
    group_payoff_speaker: pd.DataFrame,
    tag_payoff_model: pd.DataFrame,
    group_payoff_model: pd.DataFrame,
    tag_payoff_speaker_comp: pd.DataFrame,
    group_payoff_speaker_comp: pd.DataFrame,
    tag_payoff_model_comp: pd.DataFrame,
    group_payoff_model_comp: pd.DataFrame,
    tag_freq_comp: pd.DataFrame,
    group_freq_comp: pd.DataFrame,
    tag_trends_comp: pd.DataFrame,
    group_trends_comp: pd.DataFrame,
) -> None:
    tag_title = dict(zip(tag_meta.tag_code, tag_meta.tag_title))
    tag_group = dict(zip(tag_meta.tag_code, tag_meta.category))
    hot_table = tag_meta[tag_meta.tag_code.isin(hot_tags)].copy()
    hot_table["hot_rank"] = hot_table["tag_code"].map({t: i + 1 for i, t in enumerate(hot_tags)})
    hot_table["tldr"] = hot_table["definition"]
    hot_table = hot_table.sort_values("hot_rank")

    def md_table(df: pd.DataFrame, cols: list[str], n: int = 12, pct_cols: set[str] | None = None) -> str:
        pct_cols = pct_cols or set()
        if not METRIC_PERCENT:
            pct_cols = pct_cols - {"low_elo_rate", "high_elo_rate"}
        if df.empty:
            return "_No rows._"
        d = df.head(n).copy()
        for c in d.columns:
            if c in pct_cols:
                d[c] = d[c].map(lambda x: f"{100*x:.1f}%" if pd.notna(x) else "")
            elif d[c].dtype.kind in "fc":
                d[c] = d[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
        return d[cols].to_markdown(index=False)

    def top_bottom_by_competition(df: pd.DataFrame, value_col: str, n_each_side: int = 4) -> pd.DataFrame:
        if df.empty or "competition_bin" not in df.columns:
            return df
        parts = []
        d = df.dropna(subset=[value_col]).copy()
        for comp in COMPETITION_ORDER:
            g = d[d["competition_bin"] == comp]
            if g.empty:
                continue
            parts.append(g.sort_values(value_col, ascending=False).head(n_each_side))
            parts.append(g.sort_values(value_col, ascending=True).head(n_each_side))
        if not parts:
            return df.head(0)
        out = pd.concat(parts, ignore_index=True).drop_duplicates()
        return out.sort_values(["competition_order", value_col], ascending=[True, False])

    fam_counts = (
        denominators.groupby("experiment_family")
        .agg(speaker_rollouts=("speaker_rollouts", "sum"), models=("speaker_model", "nunique"))
        .reset_index()
    )
    fam_counts["family"] = fam_counts.experiment_family.map(FAMILY_LABELS).fillna(fam_counts.experiment_family)
    present_families = [
        family
        for family in FAMILY_LABELS
        if family in set(denominators["experiment_family"].dropna())
    ]
    for family in sorted(set(denominators["experiment_family"].dropna()) - set(present_families)):
        present_families.append(family)

    het_top = tag_trends[tag_trends.experiment_family == "heterogeneous_random"].sort_values(
        "trend_strength", ascending=False
    )
    hom_top = tag_trends[tag_trends.experiment_family == "homogeneous_adversary"].sort_values(
        "trend_strength", ascending=False
    )
    group_het = group_trends[group_trends.experiment_family == "heterogeneous_random"].sort_values(
        "trend_strength", ascending=False
    )
    group_hom = group_trends[group_trends.experiment_family == "homogeneous_adversary"].sort_values(
        "trend_strength", ascending=False
    )
    consistent = strongest_consistent(tag_trends)

    def image(path: str, alt: str) -> str:
        return f"![{alt}]({path})"

    main_plot_lines = ["### Main Line Plots", ""]
    for family in present_families:
        label = FAMILY_LABELS.get(family, family)
        main_plot_lines += [
            f"#### {label}",
            "",
            image(f"plots/{family}_hot_tag_line_grid.png", f"{label} hot-tag line plots"),
            "",
            image(f"plots/{family}_group_line_grid.png", f"{label} group line plots"),
            "",
        ]

    trend_plot_lines = [
        "### Top Tag Trend Plots",
        "",
    ]
    for family in present_families:
        df = tag_trends[tag_trends.experiment_family == family].sort_values("trend_strength", ascending=False)
        n = 8
        trend_plot_lines += [f"#### {FAMILY_LABELS.get(family, family)}", ""]
        if df.empty:
            trend_plot_lines += ["_No Elo trend rows for this family._", ""]
        else:
            for row in df.head(n).itertuples():
                path = f"plots/top_tag_trends/{family}_{slugify(str(row.tag_code))}_trend.png"
                trend_plot_lines += [image(path, f"{FAMILY_LABELS.get(family, family)} {row.label} trend"), ""]

    trend_plot_lines += ["### Group Trend Plots", ""]
    for family in present_families:
        df = group_trends[group_trends.experiment_family == family].sort_values("trend_strength", ascending=False)
        trend_plot_lines += [f"#### {FAMILY_LABELS.get(family, family)}", ""]
        if df.empty:
            trend_plot_lines += ["_No group trend rows for this family._", ""]
        else:
            for row in df.itertuples():
                path = f"plots/top_group_trends/{family}_{slugify(str(row.group))}_trend.png"
                trend_plot_lines += [image(path, f"{FAMILY_LABELS.get(family, family)} {row.group} trend"), ""]

    competition_plot_lines = [
        "### Competition-Binned Line Plots",
        "",
        "Competition bins are low (`0.0`, `0.25`), middle (`0.5`), and high (`0.75`, `0.95`, `1.0`). Each subplot overlays the three competition bins so you can compare whether the Elo trend changes under harder bargaining conditions.",
        "",
    ]
    for family in present_families:
        label = FAMILY_LABELS.get(family, family)
        competition_plot_lines += [
            f"#### {label}",
            "",
            image(f"plots/competition/{family}_hot_tag_competition_line_grid.png", f"{label} hot-tag competition line plots"),
            "",
            image(f"plots/competition/{family}_group_competition_line_grid.png", f"{label} group competition line plots"),
            "",
        ]
    slope_families = [
        family
        for family in present_families
        if not tag_trends_comp[tag_trends_comp.experiment_family == family].empty
    ]
    competition_plot_lines += ["### Competition-Binned Elo Slope Comparisons", ""]
    if slope_families:
        for family in slope_families:
            label = FAMILY_LABELS.get(family, family)
            competition_plot_lines += [
                f"#### {label}",
                "",
                image(f"plots/competition/{family}_tag_competition_slope_bars.png", f"{label} tag competition slope comparison"),
                "",
                image(f"plots/competition/{family}_group_competition_slope_bars.png", f"{label} group competition slope comparison"),
                "",
            ]
    else:
        competition_plot_lines += ["_No competition-binned Elo slope comparisons were available._", ""]

    speaker_payoff_sort = "delta_utility_used_minus_not" if METRIC_MODE == "occurrence" else "spearman_event_count_r_utility"
    speaker_payoff_cols = (
        ["experiment_family", "tag_code", "label", "n_used", "used_rate", "delta_utility_used_minus_not", "delta_relative_utility_used_minus_not", "point_biserial_r_utility"]
        if METRIC_MODE == "occurrence"
        else ["experiment_family", "tag_code", "label", "n_used", "used_rate", "spearman_event_count_r_utility", "spearman_event_count_p_utility"]
    )
    group_payoff_cols = (
        ["experiment_family", "group", "label", "n_used", "used_rate", "delta_utility_used_minus_not", "delta_relative_utility_used_minus_not", "point_biserial_r_utility"]
        if METRIC_MODE == "occurrence"
        else ["experiment_family", "group", "label", "n_used", "used_rate", "spearman_event_count_r_utility", "spearman_event_count_p_utility"]
    )
    speaker_payoff_comp_cols = (
        ["competition_bin", "experiment_family", "tag_code", "label", "n_used", "used_rate", "delta_utility_used_minus_not", "delta_relative_utility_used_minus_not"]
        if METRIC_MODE == "occurrence"
        else ["competition_bin", "experiment_family", "tag_code", "label", "n_used", "used_rate", "spearman_event_count_r_utility", "spearman_event_count_p_utility"]
    )
    group_payoff_comp_cols = (
        ["competition_bin", "experiment_family", "group", "label", "n_used", "used_rate", "delta_utility_used_minus_not", "delta_relative_utility_used_minus_not"]
        if METRIC_MODE == "occurrence"
        else ["competition_bin", "experiment_family", "group", "label", "n_used", "used_rate", "spearman_event_count_r_utility", "spearman_event_count_p_utility"]
    )

    scope_note = (
        "For N=2 GPT-5-nano bilateral runs, the trend analysis filters to the varied adversary speaker only. The fixed `gpt-5-nano` baseline participant is excluded from the adversary trend so it does not swamp the model-Elo comparison."
        if "n2_gpt5_bilateral" in present_families
        else (
            "For homogeneous-adversary runs, the trend analysis filters to the adversary speaker only. Baseline `gpt-5-nano` participants are excluded from the adversary trend so they do not swamp the model-Elo comparison."
            if "homogeneous_adversary" in present_families
            else "The report follows the speaker rows exposed by the input bundle. In random-monoculture control, every speaker in a rollout is a clone of the same model, so model-level rates pool all same-model speakers in that run."
        )
    )
    if "homogeneous_control" in present_families:
        scope_note += " Homogeneous-control has no Elo variation, so it is summarized descriptively rather than treated as an Elo trend."

    plot_note = (
        "These replace the earlier color heatmaps. Each subplot shows "
        f"{METRIC_DESC} for one tag or group; the x-axis is model Elo where Elo exists."
    )
    if "homogeneous_control" in present_families:
        plot_note += " Homogeneous control has only one model, so its panels are one-point baseline summaries rather than Elo trends."

    family_trend_lines = []
    for family in present_families:
        df = tag_trends[tag_trends.experiment_family == family].sort_values("trend_strength", ascending=False)
        family_trend_lines += [f"### {FAMILY_LABELS.get(family, family)}", ""]
        family_trend_lines += [
            md_table(
                df,
                ["tag_code", "label", "spearman_r", "slope_per_100_elo", "low_elo_rate", "high_elo_rate", "n_models"],
                n=12,
                pct_cols={"low_elo_rate", "high_elo_rate"},
            ),
            "",
        ]

    competition_table_lines = []
    for family in present_families:
        label = FAMILY_LABELS.get(family, family)
        tag_df = tag_trends_comp[tag_trends_comp.experiment_family == family].sort_values(
            ["competition_order", "trend_strength"], ascending=[True, False]
        )
        group_df = group_trends_comp[group_trends_comp.experiment_family == family].sort_values(
            ["competition_order", "trend_strength"], ascending=[True, False]
        )
        competition_table_lines += [
            f"### {label}: tags by competition bin",
            "",
            md_table(
                tag_df,
                ["competition_bin", "tag_code", "label", "spearman_r", "slope_per_100_elo", "low_elo_rate", "high_elo_rate", "n_models"],
                n=24,
                pct_cols={"low_elo_rate", "high_elo_rate"},
            ),
            "",
            f"### {label}: groups by competition bin",
            "",
            md_table(
                group_df,
                ["competition_bin", "group", "spearman_r", "slope_per_100_elo", "low_elo_rate", "high_elo_rate", "n_models"],
                n=21,
                pct_cols={"low_elo_rate", "high_elo_rate"},
            ),
            "",
        ]

    group_trend_lines = []
    for family in present_families:
        df = group_trends[group_trends.experiment_family == family].sort_values("trend_strength", ascending=False)
        group_trend_lines += [
            f"### {FAMILY_LABELS.get(family, family)} groups",
            "",
            md_table(
                df,
                ["group", "spearman_r", "slope_per_100_elo", "low_elo_rate", "high_elo_rate", "n_models"],
                n=10,
                pct_cols={"low_elo_rate", "high_elo_rate"},
            ),
            "",
        ]

    lines = [
        f"# Strategic Tag {METRIC_NOUN.title()} vs Elo",
        "",
        "## Scope",
        "",
        f"Input bundle: `{input_dir}`.",
        "",
        f"Hot-tag source: {hot_source}.",
        "",
        (
            "Frequency is measured as a speaker-rollout occurrence rate: for a given model and tag, the numerator is the number of appearances by that model in which the model produced at least one event with that tag; the denominator is the number of appearances by that model in the relevant family. This avoids letting verbose conversations dominate purely by event count."
            if METRIC_MODE == "occurrence"
            else "Intensity is measured as mean tag events per speaker-rollout: for a given model and tag, the numerator is the number of adjudicated tag events produced by that model and the denominator is the number of speaker appearances by that model in the relevant family. Unlike binary occurrence, repeated uses of the same tag within a rollout increase the value."
        ),
        "",
        scope_note,
        "",
        "## Coverage",
        "",
        md_table(fam_counts, ["family", "speaker_rollouts", "models"], n=10),
        "",
        "## Hot Tags Used",
        "",
        md_table(hot_table, ["hot_rank", "tag_code", "tag_title", "category", "tldr"], n=40),
        "",
        "## Plots",
        "",
        plot_note,
        "",
        *main_plot_lines,
        *trend_plot_lines,
        *competition_plot_lines,
        "",
        "## Strongest Tag Trends",
        "",
        *family_trend_lines,
        "",
        "### Consistent cross-family tag trends",
        "",
    ]

    if not consistent.empty:
        c = consistent.rename(
            columns={
                "label_hetero": "tag_title",
                "spearman_r_hetero": "hetero_spearman",
                "spearman_r_homadv": "hom_adv_spearman",
                "slope_per_100_elo_hetero": "hetero_slope_per_100_elo",
                "slope_per_100_elo_homadv": "hom_adv_slope_per_100_elo",
            }
        )
        lines.append(
            md_table(
                c,
                [
                    "tag_code",
                    "tag_title",
                    "combined_direction",
                    "hetero_spearman",
                    "hom_adv_spearman",
                    "hetero_slope_per_100_elo",
                    "hom_adv_slope_per_100_elo",
                ],
                n=12,
            )
        )
    else:
        lines.append("_No hot tags have the same nonzero trend direction in both heterogeneous and homogeneous-adversary analyses._")

    lines += [
        "",
        "## Competition-Level Trends",
        "",
        f"These tables show the strongest Elo trends after splitting rollouts into low, middle, and high competition bins. The slope is the change in {METRIC_DESC} per 100 Elo within that competition bin.",
        "",
        *competition_table_lines,
        "## Group Trends",
        "",
        "Groups come from the codebook `category` field over the selected hot tags.",
        "",
        *group_trend_lines,
        "",
        "## Payoff Correlation",
        "",
        (
            "This section asks a different question from the Elo plots: when a tag appears, is it associated with higher payoff? I report two payoff notions. `delta_utility_used_minus_not` is the raw own-utility gap between speaker-rollouts where the tag appeared and those where it did not. `delta_relative_utility_used_minus_not` subtracts the rollout mean first, so it captures whether the speaker did better than the other agents in the same run."
            if METRIC_MODE == "occurrence"
            else "This section asks a different question from the Elo plots: when a tag appears more often within a speaker-rollout, is that event count associated with higher payoff? The speaker-level plots use Spearman correlations between event count and payoff rather than binary used-vs-not-used gaps."
        ),
        "",
        "### Speaker-rollout tag associations",
        "",
        "Positive rows mean speakers using the tag had higher payoff than speakers not using it within the same experiment family. These are associations, not causal effects.",
        "",
        "![Tag payoff association](plots/payoff/tag_speaker_delta_utility.png)",
        "",
        md_table(
            tag_payoff_speaker.sort_values(speaker_payoff_sort, ascending=False),
            speaker_payoff_cols,
            n=16,
            pct_cols={"used_rate"},
        ),
        "",
        "### Speaker-rollout group associations",
        "",
        "![Group payoff association](plots/payoff/group_speaker_delta_utility.png)",
        "",
        md_table(
            group_payoff_speaker.sort_values(speaker_payoff_sort, ascending=False),
            group_payoff_cols,
            n=16,
            pct_cols={"used_rate"},
        ),
        "",
        "### Model-level tag associations",
        "",
        "This checks whether models that use a tag more often also have higher mean payoff. It is closer to your phrasing of 'when a model uses them more,' but it has fewer data points because it aggregates by model.",
        "",
        "![Model tag payoff correlation](plots/payoff/tag_model_rate_payoff_corr.png)",
        "",
        md_table(
            tag_payoff_model.sort_values("spearman_model_rate_vs_mean_utility", ascending=False),
            [
                "experiment_family",
                "tag_code",
                "label",
                "n_models",
                "spearman_model_rate_vs_mean_utility",
                "spearman_model_rate_vs_mean_relative_utility",
                "spearman_model_rate_vs_mean_utility_z",
            ],
            n=16,
        ),
        "",
        "### Model-level group associations",
        "",
        "![Model group payoff correlation](plots/payoff/group_model_rate_payoff_corr.png)",
        "",
        md_table(
            group_payoff_model.sort_values("spearman_model_rate_vs_mean_utility", ascending=False),
            [
                "experiment_family",
                "group",
                "label",
                "n_models",
                "spearman_model_rate_vs_mean_utility",
                "spearman_model_rate_vs_mean_relative_utility",
                "spearman_model_rate_vs_mean_utility_z",
            ],
            n=16,
        ),
        "",
        "### Competition-binned payoff associations",
        "",
        "These repeat the payoff association analysis within low, middle, and high competition bins. The grouped bars make it easier to see whether a tag is payoff-positive only under high-pressure bargaining, only in easier games, or consistently across bins.",
        "",
        "![Competition-binned tag payoff association](plots/payoff/tag_speaker_delta_utility_by_competition.png)",
        "",
        md_table(
            top_bottom_by_competition(tag_payoff_speaker_comp, speaker_payoff_sort, n_each_side=4),
            speaker_payoff_comp_cols,
            n=24,
            pct_cols={"used_rate"},
        ),
        "",
        "![Competition-binned group payoff association](plots/payoff/group_speaker_delta_utility_by_competition.png)",
        "",
        md_table(
            top_bottom_by_competition(group_payoff_speaker_comp, speaker_payoff_sort, n_each_side=3),
            group_payoff_comp_cols,
            n=18,
            pct_cols={"used_rate"},
        ),
        "",
        "### Competition-binned model-level payoff correlations",
        "",
        "These ask whether models that use a tag or group more often within a competition bin also earn higher mean payoff within that same bin.",
        "",
        "![Competition-binned model tag payoff correlation](plots/payoff/tag_model_rate_payoff_corr_by_competition.png)",
        "",
        md_table(
            top_bottom_by_competition(tag_payoff_model_comp, "spearman_model_rate_vs_mean_utility", n_each_side=4),
            [
                "competition_bin",
                "experiment_family",
                "tag_code",
                "label",
                "n_models",
                "spearman_model_rate_vs_mean_utility",
                "spearman_model_rate_vs_mean_relative_utility",
                "spearman_model_rate_vs_mean_utility_z",
            ],
            n=24,
        ),
        "",
        "![Competition-binned model group payoff correlation](plots/payoff/group_model_rate_payoff_corr_by_competition.png)",
        "",
        md_table(
            top_bottom_by_competition(group_payoff_model_comp, "spearman_model_rate_vs_mean_utility", n_each_side=3),
            [
                "competition_bin",
                "experiment_family",
                "group",
                "label",
                "n_models",
                "spearman_model_rate_vs_mean_utility",
                "spearman_model_rate_vs_mean_relative_utility",
                "spearman_model_rate_vs_mean_utility_z",
            ],
            n=18,
        ),
        "",
        "## Interpretation",
        "",
    ]

    def top_names(df: pd.DataFrame, positive: bool, n: int = 5) -> list[str]:
        if df.empty:
            return []
        d = df[df["spearman_r"] > 0] if positive else df[df["spearman_r"] < 0]
        d = d.sort_values("trend_strength", ascending=False).head(n)
        return [f"`{r.tag_code}` ({r.label}, rho={r.spearman_r:.2f})" for r in d.itertuples()]

    def top_group_names(df: pd.DataFrame, positive: bool, n: int = 4) -> list[str]:
        if df.empty:
            return []
        d = df[df["spearman_r"] > 0] if positive else df[df["spearman_r"] < 0]
        d = d.sort_values("trend_strength", ascending=False).head(n)
        if METRIC_PERCENT:
            return [
                f"`{r.group}` (rho={r.spearman_r:.2f}, {100*r.low_elo_rate:.1f}% -> {100*r.high_elo_rate:.1f}%)"
                for r in d.itertuples()
            ]
        return [
            f"`{r.group}` (rho={r.spearman_r:.2f}, {r.low_elo_rate:.3f} -> {r.high_elo_rate:.3f})"
            for r in d.itertuples()
        ]

    family_interpretation: list[str] = []
    for family in present_families:
        label = FAMILY_LABELS.get(family, family)
        tag_df = tag_trends[tag_trends.experiment_family == family].sort_values("trend_strength", ascending=False)
        group_df = group_trends[group_trends.experiment_family == family].sort_values("trend_strength", ascending=False)
        pos_tags = top_names(tag_df, True)
        neg_tags = top_names(tag_df, False)
        pos_groups = top_group_names(group_df, True)
        neg_groups = top_group_names(group_df, False)
        if family == "n2_gpt5_bilateral":
            family_interpretation += [
                f"For {label}, positive Elo trends indicate tags that stronger varied adversary models use more often against the fixed `gpt-5-nano` baseline. Negative trends indicate tags that lower-Elo adversary models use more often.",
                "",
            ]
        family_interpretation += [
            f"In {label}, positive-Elo tag trends are: "
            + (", ".join(pos_tags) if pos_tags else "_none_")
            + ". Negative-Elo tag trends are: "
            + (", ".join(neg_tags) if neg_tags else "_none_")
            + ".",
            "",
            f"At the group level for {label}, higher-Elo behavior is most associated with "
            + (", ".join(pos_groups) if pos_groups else "_no positive group trend_")
            + ", while lower-Elo behavior is more associated with "
            + (", ".join(neg_groups) if neg_groups else "_no negative group trend_")
            + ".",
            "",
        ]

    consistent_summary: list[str] = []
    if not consistent.empty:
        for row in consistent.head(8).itertuples():
            direction = "up" if row.spearman_r_hetero > 0 else "down"
            consistent_summary.append(
                f"`{row.tag_code}` ({row.label_hetero}, {direction}; "
                f"hetero rho={row.spearman_r_hetero:.2f}, hom-adv rho={row.spearman_r_homadv:.2f})"
            )

    lines += [
        *family_interpretation,
        (
            "The cross-family consistency check is the best guard against overreading one experimental setup. Hot tags with the same trend direction in both heterogeneous and homogeneous-adversary views are: "
            + (", ".join(consistent_summary) if consistent_summary else "_none_")
            + "."
            if {"heterogeneous_random", "homogeneous_adversary"}.issubset(set(present_families))
            else "This report has a single Elo-varying family, so there is no cross-family consistency check to apply."
        ),
        "",
        "Mechanistically, the main thing to mine is not simply whether stronger models bargain harder. The useful contrast is whether a model turns self-interest into an executable agreement. Positive trends in logical persuasion, trade/compromise, and coalition tags point toward concrete deal engineering: threshold math, explicit payoff accounting, conditional support, vote diagnostics, and named bloc construction. These tags help transform a conversation from preference statements into a package that other agents can vote on.",
        "",
        "Negative trends are equally informative. If high-Elo models use less rapport, empathy, or accepted-loss capitulation, that does not mean those tactics are useless. It means the stronger-model signature may be less about maintaining a pleasant bargaining tone or yielding gracefully and more about narrowing the formal choice set. Conversely, if a pressure or self-interest tag rises with Elo, read it as a candidate mechanism for credible closure, not automatically as aggression.",
        "",
        "Coalitioning deserves special attention because it is the cleanest multi-agent mechanism. Tags such as `named_microcoalition_slate`, `vote_bloc_counting`, `bespoke_agent_or_bloc_recruitment`, `coalition_integrity_warning`, `third_party_mediation`, and `cross_agent_conflict_mapping` indicate that the model is no longer treating negotiation as a set of bilateral preferences. It is reasoning about vote assembly: who can be recruited, which conflicts must be mediated, and which bloc is sufficient to pass a deal.",
        "",
        "Logical persuasion is the other key mechanism. Tags such as `utility_arithmetic_receipts`, `agent_specific_payoff_accounting`, `threshold_gap_calculation`, and `fairness_ledger_argument` expose whether a model can make the deal auditable. A model that can explain who gains what, what threshold remains, and why the package is balanced has a concrete path to acceptance that pure preference assertion lacks.",
        "",
        "Trade/compromise tags capture another route to performance: conditional quid pro quo, conditional support ledgers, concession ladders, low-weight concessions, and vote-history diagnostics. These are signs of search over mutually acceptable packages. If these rise with Elo, the mechanism is not merely smarter rhetoric; it is better local optimization over what each side can give up cheaply.",
        "",
        "Pressure and self-interest/exploitation tags need a more careful reading. A rising pressure tag can mean stronger models are strategically decisive, but it can also mean the model is creating conflict that later needs repair. A rising self-interest tag can mean stronger value maximization, but it can also expose brittle overclaiming. The report therefore treats these trends as mechanism candidates, not direct welfare claims.",
        "",
        "The main mechanism hypothesis from this exploratory pass is: higher-Elo models perform better when they operationalize bargaining. They quantify gaps, make conditional commitments explicit, assemble coalitions, and turn proposed deals into voteable objects. Lower-Elo-associated tags, when present, often look more like soft persuasion, affect display, or generalized norm pressure. Those tactics may matter, but by themselves they do less to solve the coordination problem.",
        "",
        (
            "Homogeneous control is mainly a baseline for the tagger and for generic `gpt-5-nano` negotiation style. Since there is no model Elo variation, its value is checking which tags appear even without an adversarial model manipulation. High control rates for formalization, utility arithmetic, or conditional trades should be interpreted as generic bargaining language rather than adversary-specific strategy."
            if "homogeneous_control" in present_families
            else "For random-monoculture control, read the reported pattern as same-model group behavior: each speaker in a rollout is generated by the same model, so the trends reflect how an all-one-model table behaves rather than how that model behaves when mixed with stronger or weaker agents."
        ),
        "",
        "## Output Tables",
        "",
        "- [tag_model_frequency.csv](tag_model_frequency.csv)",
        "- [group_model_frequency.csv](group_model_frequency.csv)",
        "- [tag_elo_trends.csv](tag_elo_trends.csv)",
        "- [group_elo_trends.csv](group_elo_trends.csv)",
        "- [tag_model_frequency_by_competition_bin.csv](tag_model_frequency_by_competition_bin.csv)",
        "- [group_model_frequency_by_competition_bin.csv](group_model_frequency_by_competition_bin.csv)",
        "- [tag_elo_trends_by_competition_bin.csv](tag_elo_trends_by_competition_bin.csv)",
        "- [group_elo_trends_by_competition_bin.csv](group_elo_trends_by_competition_bin.csv)",
        "- [model_denominators.csv](model_denominators.csv)",
        "- [speaker_payoffs.csv](speaker_payoffs.csv)",
        "- [tag_payoff_speaker_rollout_correlations.csv](tag_payoff_speaker_rollout_correlations.csv)",
        "- [group_payoff_speaker_rollout_correlations.csv](group_payoff_speaker_rollout_correlations.csv)",
        "- [tag_payoff_model_correlations.csv](tag_payoff_model_correlations.csv)",
        "- [group_payoff_model_correlations.csv](group_payoff_model_correlations.csv)",
        "- [tag_payoff_speaker_rollout_correlations_by_competition_bin.csv](tag_payoff_speaker_rollout_correlations_by_competition_bin.csv)",
        "- [group_payoff_speaker_rollout_correlations_by_competition_bin.csv](group_payoff_speaker_rollout_correlations_by_competition_bin.csv)",
        "- [tag_payoff_model_correlations_by_competition_bin.csv](tag_payoff_model_correlations_by_competition_bin.csv)",
        "- [group_payoff_model_correlations_by_competition_bin.csv](group_payoff_model_correlations_by_competition_bin.csv)",
        "- [hot_tags_used.json](hot_tags_used.json)",
        "",
        "## Caveats",
        "",
        f"- Hot-tag selection came from {hot_source}; `hot_tags_used.json` records the exact 29-tag set.",
        f"- {METRIC_DESC.title()} values are descriptive associations, not causal estimates. Models are not independently randomized across every possible role/game/agent-count combination.",
        (
            "- Homogeneous-adversary trends have only five Elo points, so Spearman values can look large even when the evidence is thin."
            if "homogeneous_adversary" in present_families
            else "- Spearman values can look large when a tag is sparse or concentrated in a few models; use the model counts and plots alongside the rank correlations."
        ),
        (
            "- Event-count rates are written to CSV, but the report emphasizes occurrence rates to avoid rewarding verbosity."
            if METRIC_MODE == "occurrence"
            else "- Intensity can reward verbosity: a model that repeats the same strategic move many times will score higher. Read this together with the binary occurrence report."
        ),
    ]

    (out_dir / "strategic_tag_elo_exploration_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hot-tags-json", type=Path, default=None)
    parser.add_argument("--hot-limit", type=int, default=29)
    parser.add_argument("--metric", choices=["occurrence", "intensity"], default="occurrence")
    args = parser.parse_args()
    configure_metric(args.metric)

    input_dir = args.input_dir
    out_dir = args.output_dir
    plots_dir = out_dir / "plots"
    tag_scatter_dir = plots_dir / "top_tag_trends"
    group_scatter_dir = plots_dir / "top_group_trends"
    payoff_plot_dir = plots_dir / "payoff"
    competition_plot_dir = plots_dir / "competition"
    for path in [out_dir, plots_dir, tag_scatter_dir, group_scatter_dir, payoff_plot_dir, competition_plot_dir]:
        path.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", font_scale=0.85)
    tag_meta = build_tag_metadata(input_dir)
    hot_tags, hot_source = load_hot_tags(args.hot_tags_json, input_dir / "llm_event_tag_counts_by_tag.csv", args.hot_limit)
    missing = sorted(set(hot_tags) - set(tag_meta.tag_code))
    if missing:
        raise ValueError(f"Hot tags missing from codebook: {missing}")

    manifest_rows = load_jsonl(input_dir / "all_rollouts_manifest.jsonl")
    denoms = build_denominators(manifest_rows)
    events = build_event_frame(input_dir)
    tag_freq, group_freq, denominators = make_frequency_tables(denoms, events, tag_meta, hot_tags)
    tag_trends = trend_table(tag_freq, "tag_code", "tag_title")
    group_trends = trend_table(group_freq, "group", "group")
    tag_freq_comp, group_freq_comp = make_competition_frequency_tables(denoms, events, tag_meta, hot_tags)
    tag_trends_comp = competition_trend_table(tag_freq_comp, "tag_code", "tag_title")
    group_trends_comp = competition_trend_table(group_freq_comp, "group", "group")
    speaker_payoffs = build_payoff_frame(manifest_rows, denoms)
    tag_payoff_speaker, group_payoff_speaker, tag_payoff_model, group_payoff_model = payoff_correlation_tables(
        speaker_payoffs, events, tag_meta, hot_tags, tag_freq, group_freq
    )
    (
        tag_payoff_speaker_comp,
        group_payoff_speaker_comp,
        tag_payoff_model_comp,
        group_payoff_model_comp,
    ) = payoff_correlation_tables(
        speaker_payoffs,
        events,
        tag_meta,
        hot_tags,
        tag_freq_comp,
        group_freq_comp,
        split_cols=["competition_bin"],
    )

    tag_freq.to_csv(out_dir / "tag_model_frequency.csv", index=False)
    group_freq.to_csv(out_dir / "group_model_frequency.csv", index=False)
    tag_trends.to_csv(out_dir / "tag_elo_trends.csv", index=False)
    group_trends.to_csv(out_dir / "group_elo_trends.csv", index=False)
    tag_freq_comp.to_csv(out_dir / "tag_model_frequency_by_competition_bin.csv", index=False)
    group_freq_comp.to_csv(out_dir / "group_model_frequency_by_competition_bin.csv", index=False)
    tag_trends_comp.to_csv(out_dir / "tag_elo_trends_by_competition_bin.csv", index=False)
    group_trends_comp.to_csv(out_dir / "group_elo_trends_by_competition_bin.csv", index=False)
    denominators.to_csv(out_dir / "model_denominators.csv", index=False)
    speaker_payoffs.to_csv(out_dir / "speaker_payoffs.csv", index=False)
    tag_payoff_speaker.to_csv(out_dir / "tag_payoff_speaker_rollout_correlations.csv", index=False)
    group_payoff_speaker.to_csv(out_dir / "group_payoff_speaker_rollout_correlations.csv", index=False)
    tag_payoff_model.to_csv(out_dir / "tag_payoff_model_correlations.csv", index=False)
    group_payoff_model.to_csv(out_dir / "group_payoff_model_correlations.csv", index=False)
    tag_payoff_speaker_comp.to_csv(out_dir / "tag_payoff_speaker_rollout_correlations_by_competition_bin.csv", index=False)
    group_payoff_speaker_comp.to_csv(out_dir / "group_payoff_speaker_rollout_correlations_by_competition_bin.csv", index=False)
    tag_payoff_model_comp.to_csv(out_dir / "tag_payoff_model_correlations_by_competition_bin.csv", index=False)
    group_payoff_model_comp.to_csv(out_dir / "group_payoff_model_correlations_by_competition_bin.csv", index=False)
    (out_dir / "hot_tags_used.json").write_text(
        json.dumps(
            {
                "source": hot_source,
                "hot_tags": [
                    {
                        "tag_code": t,
                        "tag_title": tag_meta.set_index("tag_code").loc[t, "tag_title"],
                        "category": tag_meta.set_index("tag_code").loc[t, "category"],
                    }
                    for t in hot_tags
                ],
            },
            indent=2,
        )
        + "\n"
    )

    families_present = [family for family in FAMILY_LABELS if family in set(denoms["experiment_family"])]
    for family in families_present:
        save_line_grid(
            tag_freq,
            family,
            "tag_code",
            "tag_title",
            plots_dir / f"{family}_hot_tag_line_grid.png",
            f"{FAMILY_LABELS[family]}: hot-tag {METRIC_NOUN} by model",
            ncols=4,
        )
        save_line_grid(
            group_freq,
            family,
            "group",
            "group",
            plots_dir / f"{family}_group_line_grid.png",
            f"{FAMILY_LABELS[family]}: group {METRIC_NOUN} by model",
            ncols=3,
        )
        save_competition_line_grid(
            tag_freq_comp,
            family,
            "tag_code",
            "tag_title",
            competition_plot_dir / f"{family}_hot_tag_competition_line_grid.png",
            f"{FAMILY_LABELS[family]}: hot-tag {METRIC_NOUN} by model and competition",
            ncols=4,
        )
        save_competition_line_grid(
            group_freq_comp,
            family,
            "group",
            "group",
            competition_plot_dir / f"{family}_group_competition_line_grid.png",
            f"{FAMILY_LABELS[family]}: group {METRIC_NOUN} by model and competition",
            ncols=3,
        )
        if family != "homogeneous_control":
            save_competition_slope_bars(
                tag_trends_comp,
                family,
                "tag_code",
                competition_plot_dir / f"{family}_tag_competition_slope_bars.png",
                f"{FAMILY_LABELS[family]}: Elo slopes by competition bin",
                top_n=18,
            )
            save_competition_slope_bars(
                group_trends_comp,
                family,
                "group",
                competition_plot_dir / f"{family}_group_competition_slope_bars.png",
                f"{FAMILY_LABELS[family]}: group Elo slopes by competition bin",
                top_n=None,
            )

    for family in families_present:
        save_trend_scatter(tag_freq, tag_trends, family, "tag_code", "tag_title", tag_scatter_dir, 12)
        save_trend_scatter(group_freq, group_trends, family, "group", "group", group_scatter_dir, 7)
    speaker_payoff_plot_col = "delta_utility_used_minus_not" if METRIC_MODE == "occurrence" else "spearman_event_count_r_utility"
    speaker_payoff_title = (
        "Speaker-rollout payoff gap when tag is used"
        if METRIC_MODE == "occurrence"
        else "Speaker-rollout payoff correlation with tag-event count"
    )
    group_speaker_payoff_title = (
        "Speaker-rollout payoff gap when group is used"
        if METRIC_MODE == "occurrence"
        else "Speaker-rollout payoff correlation with group-event count"
    )
    save_payoff_barplot(
        tag_payoff_speaker,
        "tag_code",
        speaker_payoff_plot_col,
        payoff_plot_dir / "tag_speaker_delta_utility.png",
        speaker_payoff_title,
        n=None,
    )
    save_payoff_barplot(
        group_payoff_speaker,
        "group",
        speaker_payoff_plot_col,
        payoff_plot_dir / "group_speaker_delta_utility.png",
        group_speaker_payoff_title,
        n=None,
    )
    save_payoff_barplot(
        tag_payoff_model,
        "tag_code",
        "spearman_model_rate_vs_mean_utility",
        payoff_plot_dir / "tag_model_rate_payoff_corr.png",
        f"Model-level correlation: tag {METRIC_NOUN} vs mean utility",
        n=None,
    )
    save_payoff_barplot(
        group_payoff_model,
        "group",
        "spearman_model_rate_vs_mean_utility",
        payoff_plot_dir / "group_model_rate_payoff_corr.png",
        f"Model-level correlation: group {METRIC_NOUN} vs mean utility",
        n=None,
    )
    save_competition_payoff_bars(
        tag_payoff_speaker_comp,
        "tag_code",
        speaker_payoff_plot_col,
        payoff_plot_dir / "tag_speaker_delta_utility_by_competition.png",
        f"{speaker_payoff_title} by competition bin",
        n=24,
    )
    save_competition_payoff_bars(
        group_payoff_speaker_comp,
        "group",
        speaker_payoff_plot_col,
        payoff_plot_dir / "group_speaker_delta_utility_by_competition.png",
        f"{group_speaker_payoff_title} by competition bin",
        n=None,
    )
    save_competition_payoff_bars(
        tag_payoff_model_comp,
        "tag_code",
        "spearman_model_rate_vs_mean_utility",
        payoff_plot_dir / "tag_model_rate_payoff_corr_by_competition.png",
        f"Model-level correlation by tag {METRIC_NOUN} and competition bin",
        n=24,
    )
    save_competition_payoff_bars(
        group_payoff_model_comp,
        "group",
        "spearman_model_rate_vs_mean_utility",
        payoff_plot_dir / "group_model_rate_payoff_corr_by_competition.png",
        f"Model-level correlation by group {METRIC_NOUN} and competition bin",
        n=None,
    )

    write_markdown_report(
        out_dir,
        input_dir,
        hot_source,
        hot_tags,
        tag_meta,
        tag_freq,
        group_freq,
        tag_trends,
        group_trends,
        denominators,
        tag_payoff_speaker,
        group_payoff_speaker,
        tag_payoff_model,
        group_payoff_model,
        tag_payoff_speaker_comp,
        group_payoff_speaker_comp,
        tag_payoff_model_comp,
        group_payoff_model_comp,
        tag_freq_comp,
        group_freq_comp,
        tag_trends_comp,
        group_trends_comp,
    )

    print(f"wrote {out_dir}")
    print(f"hot_tags={len(hot_tags)}")
    print(f"tag_rows={len(tag_freq)} group_rows={len(group_freq)}")
    print(f"competition_tag_rows={len(tag_freq_comp)} competition_group_rows={len(group_freq_comp)}")
    print(f"payoff_rows={len(speaker_payoffs)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Explore TTC strategic tag frequencies versus reasoning effort.

This mirrors the Elo strategic-tag exploration report, but the treatment axis is
the target model's TTC condition. The target speaker is the primary unit because
the target model is the participant whose reasoning effort varies.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUT_DIR = Path("analysis/ttc_llm_strategic_tag_adjudication_20260629")
DEFAULT_OUTPUT_DIR = Path("analysis/ttc_strategic_tag_exploration_lines_payoff_20260629")
DEFAULT_REVIEW_JSON = Path("strategic_tag_review_final.json")

FAMILY_ORDER = ["gpt-5", "claude-sonnet-4-6", "gemini-3-flash"]
FAMILY_LABELS = {
    "gpt-5": "GPT-5",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gemini-3-flash": "Gemini 3 Flash",
}
LEVEL_ORDER = {"minimal": 0, "low": 1, "medium": 2, "high": 3, "max": 4}
LEVEL_COLORS = {
    "minimal": "#64748b",
    "low": "#2563eb",
    "medium": "#0f766e",
    "high": "#f97316",
    "max": "#7c3aed",
}
FAMILY_COLORS = {
    "gpt-5": "#2563eb",
    "claude-sonnet-4-6": "#dc2626",
    "gemini-3-flash": "#16a34a",
}


def slugify(text: str) -> str:
    out = text.lower()
    for old, new in [("/", "_"), (" ", "_"), ("-", "_"), (":", ""), ("+", "plus")]:
        out = out.replace(old, new)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_tag_metadata(input_dir: Path, review_json: Path | None) -> pd.DataFrame:
    tag_meta = pd.DataFrame(json.loads((input_dir / "llm_tag_codebook.json").read_text(encoding="utf-8")))
    tag_meta = tag_meta[["tag_code", "tag_title", "category", "definition", "paper_value"]].copy()
    tag_meta["review_decision"] = ""
    tag_meta["review_notes"] = ""
    if review_json and review_json.exists():
        review = json.loads(review_json.read_text(encoding="utf-8"))
        if isinstance(review, dict) and "responses" in review:
            review_rows = pd.DataFrame(review["responses"])
            cols = [c for c in ["tag_code", "decision", "notes", "source_count", "source_share"] if c in review_rows]
            tag_meta = tag_meta.merge(review_rows[cols], on="tag_code", how="left")
            tag_meta["review_decision"] = tag_meta["decision"].fillna("")
            tag_meta["review_notes"] = tag_meta["notes"].fillna("")
            tag_meta = tag_meta.drop(columns=[c for c in ["decision", "notes"] if c in tag_meta], errors="ignore")
    tag_meta["tag_rank"] = np.arange(1, len(tag_meta) + 1)
    return tag_meta


def load_inputs(input_dir: Path, review_json: Path | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = pd.DataFrame(load_jsonl(input_dir / "all_ttc_rollouts_manifest.jsonl"))
    events = pd.DataFrame(load_jsonl(input_dir / "ttc_llm_event_tags.jsonl"))
    tag_meta = load_tag_metadata(input_dir, review_json)
    manifest["level_index"] = manifest["level"].map(LEVEL_ORDER).fillna(manifest["level_index"]).astype(int)
    events["level_index"] = events["level"].map(LEVEL_ORDER).fillna(events["level_index"]).astype(int)
    return manifest, events, tag_meta


def build_denominators(manifest: pd.DataFrame, speaker_role: str = "target") -> pd.DataFrame:
    rows = []
    for _, rollout in manifest.iterrows():
        if speaker_role == "target":
            speaker_agent = rollout["target_agent"]
            speaker_model = rollout["target_model"]
            utility = rollout["target_utility"]
        elif speaker_role == "baseline":
            speaker_agent = rollout["baseline_agent"]
            speaker_model = rollout["baseline_model"]
            utility = rollout["baseline_utility"]
        else:
            raise ValueError(f"Unsupported speaker role: {speaker_role}")
        rows.append(
            {
                "config_id": int(rollout["config_id"]),
                "speaker_key": f"{rollout['result_path']}::{speaker_agent}",
                "result_path": rollout["result_path"],
                "speaker_agent": speaker_agent,
                "speaker_role": speaker_role,
                "speaker_model": speaker_model,
                "family": rollout["family"],
                "provider": rollout["provider"],
                "level": rollout["level"],
                "level_index": int(rollout["level_index"]),
                "game_label": rollout["game_label"],
                "game_cell": rollout["game_cell"],
                "game_type": rollout["game_type"],
                "order": rollout["order"],
                "mean_tokens_per_call": float(rollout["target_compute_tokens_per_call"]),
                "target_output_tokens_per_call": float(rollout["target_output_tokens_per_call"]),
                "speaker_utility": float(utility),
                "target_utility": float(rollout["target_utility"]),
                "baseline_utility": float(rollout["baseline_utility"]),
                "utility_gap": float(rollout["utility_gap"]),
                "absolute_utility_gap": abs(float(rollout["utility_gap"])),
                "consensus_reached": bool(rollout["consensus_reached"]),
                "final_round": float(rollout["final_round"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_denominators(denoms: pd.DataFrame) -> pd.DataFrame:
    return (
        denoms.groupby(["family", "level", "level_index"], as_index=False)
        .agg(
            speaker_rollouts=("speaker_key", "nunique"),
            mean_tokens_per_call=("mean_tokens_per_call", "mean"),
            mean_target_payoff=("target_utility", "mean"),
            mean_baseline_payoff=("baseline_utility", "mean"),
            mean_utility_gap=("utility_gap", "mean"),
            consensus_rate=("consensus_reached", "mean"),
            mean_final_round=("final_round", "mean"),
        )
        .sort_values(["family", "level_index"])
    )


def make_frequency_tables(
    denoms: pd.DataFrame,
    events: pd.DataFrame,
    tag_meta: pd.DataFrame,
    speaker_role: str = "target",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tag_codes = tag_meta["tag_code"].tolist()
    tag_title = dict(zip(tag_meta.tag_code, tag_meta.tag_title))
    tag_group = dict(zip(tag_meta.tag_code, tag_meta.category))

    base_levels = summarize_denominators(denoms)
    grid = base_levels.merge(pd.DataFrame({"tag_code": tag_codes}), how="cross")
    role_events = events[events["speaker_role"].eq(speaker_role)].copy()
    present = (
        role_events.drop_duplicates(["config_id", "tag_code"])
        .groupby(["family", "level", "level_index", "tag_code"], as_index=False)
        .agg(tagged_speaker_rollouts=("config_id", "nunique"))
    )
    counts = (
        role_events.groupby(["family", "level", "level_index", "tag_code"], as_index=False)
        .agg(event_count=("tag_code", "size"))
    )
    tag_freq = grid.merge(present, how="left").merge(counts, how="left")
    tag_freq[["tagged_speaker_rollouts", "event_count"]] = tag_freq[
        ["tagged_speaker_rollouts", "event_count"]
    ].fillna(0)
    tag_freq["tag_title"] = tag_freq["tag_code"].map(tag_title)
    tag_freq["group"] = tag_freq["tag_code"].map(tag_group)
    tag_freq["occurrence_rate"] = tag_freq["tagged_speaker_rollouts"] / tag_freq["speaker_rollouts"]
    tag_freq["events_per_rollout"] = tag_freq["event_count"] / tag_freq["speaker_rollouts"]

    group_map = tag_meta[["tag_code", "category"]].rename(columns={"category": "group"})
    groups = sorted(group_map["group"].dropna().unique())
    group_grid = base_levels.merge(pd.DataFrame({"group": groups}), how="cross")
    ge = role_events.merge(group_map, on="tag_code", how="left")
    gp = (
        ge.drop_duplicates(["config_id", "group"])
        .groupby(["family", "level", "level_index", "group"], as_index=False)
        .agg(tagged_speaker_rollouts=("config_id", "nunique"))
    )
    gc = (
        ge.groupby(["family", "level", "level_index", "group"], as_index=False)
        .agg(event_count=("tag_code", "size"))
    )
    group_freq = group_grid.merge(gp, how="left").merge(gc, how="left")
    group_freq[["tagged_speaker_rollouts", "event_count"]] = group_freq[
        ["tagged_speaker_rollouts", "event_count"]
    ].fillna(0)
    group_freq["occurrence_rate"] = group_freq["tagged_speaker_rollouts"] / group_freq["speaker_rollouts"]
    group_freq["events_per_rollout"] = group_freq["event_count"] / group_freq["speaker_rollouts"]
    return tag_freq.sort_values(["family", "group", "tag_title", "level_index"]), group_freq.sort_values(
        ["family", "group", "level_index"]
    )


def trend_table(freq: pd.DataFrame, unit_col: str, label_col: str) -> pd.DataFrame:
    rows = []
    for (family, unit), g in freq.groupby(["family", unit_col]):
        g = g.sort_values("level_index")
        if g["level_index"].nunique() < 3:
            continue
        weak = "minimal" if "minimal" in set(g["level"]) else g.iloc[0]["level"]
        strong = "high" if "high" in set(g["level"]) else g.iloc[-1]["level"]
        levels = {row["level"]: row["occurrence_rate"] for _, row in g.iterrows()}
        event_levels = {row["level"]: row["events_per_rollout"] for _, row in g.iterrows()}
        x = g["level_index"].to_numpy(dtype=float)
        y = g["occurrence_rate"].to_numpy(dtype=float)
        tokens = g["mean_tokens_per_call"].to_numpy(dtype=float)
        if np.unique(y).size < 2:
            effort_corr = 0.0
            token_corr = 0.0
        else:
            effort_corr = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
            token_corr = float(pd.Series(tokens).corr(pd.Series(y), method="spearman"))
        slope_per_effort = float(np.polyfit(x, y, deg=1)[0]) if np.unique(x).size > 1 else math.nan
        slope_per_1k_tokens = (
            float(np.polyfit(tokens / 1000.0, y, deg=1)[0]) if np.unique(tokens).size > 1 else math.nan
        )
        rows.append(
            {
                "family": family,
                unit_col: unit,
                "label": g[label_col].iloc[0] if label_col in g else unit,
                "group": g["group"].iloc[0] if "group" in g else unit,
                "n_levels": int(g["level_index"].nunique()),
                "weak_level": weak,
                "strong_level": strong,
                "weak_rate": levels.get(weak, math.nan),
                "strong_rate": levels.get(strong, math.nan),
                "delta_strong_minus_weak": levels.get(strong, math.nan) - levels.get(weak, math.nan),
                "weak_events_per_rollout": event_levels.get(weak, math.nan),
                "strong_events_per_rollout": event_levels.get(strong, math.nan),
                "delta_events_per_rollout": event_levels.get(strong, math.nan) - event_levels.get(weak, math.nan),
                "spearman_effort_r": effort_corr,
                "spearman_tokens_r": token_corr,
                "slope_per_effort_level": slope_per_effort,
                "slope_per_1k_tokens": slope_per_1k_tokens,
                "mean_rate": float(y.mean()),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["trend_strength"] = out["delta_strong_minus_weak"].abs() + out["spearman_effort_r"].abs() / 10.0
    return out.sort_values(["family", "trend_strength"], ascending=[True, False])


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.35)
    ax.tick_params(axis="both", labelsize=7)


def save_line_grid(
    freq: pd.DataFrame,
    family: str,
    unit_col: str,
    label_col: str,
    out_path: Path,
    title: str,
    ncols: int = 5,
    use_tokens: bool = False,
) -> None:
    sub = freq[freq["family"].eq(family)].copy()
    if sub.empty:
        return
    if unit_col == label_col:
        units = [{unit_col: value, label_col: value} for value in sorted(sub[unit_col].dropna().unique())]
    else:
        units = sub[[unit_col, label_col]].drop_duplicates().sort_values(label_col).to_dict("records")
    nrows = math.ceil(len(units) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.55 * ncols, 2.65 * nrows), squeeze=False)
    max_y = max(0.02, float(sub["occurrence_rate"].max()) * 1.14)
    for ax, unit in zip(axes.ravel(), units, strict=False):
        g = sub[sub[unit_col].eq(unit[unit_col])].sort_values("level_index")
        x = g["mean_tokens_per_call"] if use_tokens else g["level_index"]
        ax.plot(x, g["occurrence_rate"], marker="o", linewidth=1.5, markersize=4, color=FAMILY_COLORS.get(family, "#333"))
        for _, row in g.iterrows():
            xval = row["mean_tokens_per_call"] if use_tokens else row["level_index"]
            ax.annotate(
                str(row["level"]),
                (xval, row["occurrence_rate"]),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=6,
                color=LEVEL_COLORS.get(row["level"], "#111827"),
            )
        ax.set_title(str(unit[label_col]), fontsize=8.4)
        ax.set_ylim(0, max_y)
        if use_tokens:
            ax.set_xlabel("target tokens/call", fontsize=7)
        else:
            ax.set_xlabel("requested effort", fontsize=7)
            ticks = sorted(g["level_index"].unique())
            labels = [g[g["level_index"].eq(t)]["level"].iloc[0] for t in ticks]
            ax.set_xticks(ticks, labels, rotation=25, ha="right")
        style_axis(ax)
    for ax in axes.ravel()[len(units) :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.supylabel("Target-rollout occurrence rate", fontsize=10)
    fig.tight_layout(rect=[0.015, 0.015, 1, 0.985])
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def save_overlay_line_grid(
    freq: pd.DataFrame,
    unit_col: str,
    label_col: str,
    out_path: Path,
    title: str,
    ncols: int = 5,
) -> None:
    if unit_col == label_col:
        units = [{unit_col: value, label_col: value} for value in sorted(freq[unit_col].dropna().unique())]
    else:
        units = freq[[unit_col, label_col]].drop_duplicates().sort_values(label_col).to_dict("records")
    nrows = math.ceil(len(units) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.7 * ncols, 2.75 * nrows), squeeze=False)
    max_y = max(0.02, float(freq["occurrence_rate"].max()) * 1.14)
    for ax, unit in zip(axes.ravel(), units, strict=False):
        for family in FAMILY_ORDER:
            g = freq[(freq[unit_col].eq(unit[unit_col])) & (freq["family"].eq(family))].sort_values("level_index")
            if g.empty:
                continue
            ax.plot(
                g["mean_tokens_per_call"],
                g["occurrence_rate"],
                marker="o",
                linewidth=1.35,
                markersize=3.7,
                color=FAMILY_COLORS.get(family, "#333333"),
                label=FAMILY_LABELS.get(family, family),
            )
        ax.set_title(str(unit[label_col]), fontsize=8.4)
        ax.set_ylim(0, max_y)
        ax.set_xlabel("target tokens/call", fontsize=7)
        style_axis(ax)
    for ax in axes.ravel()[len(units) :]:
        ax.axis("off")
    handles = [
        plt.Line2D([0], [0], color=FAMILY_COLORS[f], marker="o", linewidth=1.5, label=FAMILY_LABELS[f])
        for f in FAMILY_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=True)
    fig.suptitle(title, fontsize=14)
    fig.supylabel("Target-rollout occurrence rate", fontsize=10)
    fig.tight_layout(rect=[0.015, 0.04, 1, 0.985])
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def save_top_trend_plots(
    freq: pd.DataFrame,
    trend: pd.DataFrame,
    family: str,
    unit_col: str,
    label_col: str,
    out_dir: Path,
    top_n: int,
) -> list[Path]:
    paths = []
    ranked = trend[trend["family"].eq(family)].sort_values("trend_strength", ascending=False).head(top_n)
    for _, row in ranked.iterrows():
        unit = row[unit_col]
        g = freq[(freq["family"].eq(family)) & (freq[unit_col].eq(unit))].sort_values("level_index")
        if g.empty:
            continue
        out = out_dir / f"{family}_{slugify(str(unit))}_trend.png"
        fig, ax = plt.subplots(figsize=(6.1, 4.0))
        ax.plot(g["level_index"], g["occurrence_rate"], marker="o", linewidth=2, color=FAMILY_COLORS.get(family, "#333"))
        for _, point in g.iterrows():
            ax.annotate(
                f"{point['level']}\n{point['mean_tokens_per_call']:.0f} tok",
                (point["level_index"], point["occurrence_rate"]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=7,
            )
        ax.set_title(f"{FAMILY_LABELS.get(family, family)}: {row['label']}")
        ax.set_xlabel("Requested reasoning effort")
        ax.set_ylabel("Target-rollout occurrence rate")
        ax.set_xticks(sorted(g["level_index"].unique()), [str(v) for v in g.sort_values("level_index")["level"]], rotation=20)
        ax.set_ylim(bottom=0)
        style_axis(ax)
        fig.tight_layout()
        fig.savefig(out, dpi=220, bbox_inches="tight")
        plt.close(fig)
        paths.append(out)
    return paths


def build_rollout_matrices(
    denoms: pd.DataFrame,
    events: pd.DataFrame,
    tag_meta: pd.DataFrame,
    speaker_role: str = "target",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    role_events = events[events["speaker_role"].eq(speaker_role)].copy()
    tags = tag_meta["tag_code"].tolist()
    groups = sorted(tag_meta["category"].dropna().unique())
    tag_counts = role_events.groupby(["config_id", "tag_code"]).size().unstack(fill_value=0)
    group_events = role_events.merge(tag_meta[["tag_code", "category"]], on="tag_code", how="left")
    group_counts = group_events.groupby(["config_id", "category"]).size().unstack(fill_value=0)
    tag_matrix = denoms.copy()
    group_matrix = denoms.copy()
    for tag in tags:
        if tag in tag_counts.columns:
            tag_matrix[tag] = tag_matrix["config_id"].map(tag_counts[tag]).fillna(0).astype(int)
        else:
            tag_matrix[tag] = 0
    for group in groups:
        if group in group_counts.columns:
            group_matrix[group] = group_matrix["config_id"].map(group_counts[group]).fillna(0).astype(int)
        else:
            group_matrix[group] = 0
    return tag_matrix, group_matrix


def payoff_association_table(matrix: pd.DataFrame, units: list[str], unit_col: str, labels: dict[str, str]) -> pd.DataFrame:
    rows = []
    keys = ["family", "game_cell", "order"]
    base = matrix.copy()
    base["target_utility_resid"] = base["target_utility"].astype(float) - base.groupby(keys)["target_utility"].transform("mean")
    base["consensus_resid"] = base["consensus_reached"].astype(float) - base.groupby(keys)["consensus_reached"].transform("mean")
    base["final_round_resid"] = base["final_round"].astype(float) - base.groupby(keys)["final_round"].transform("mean")
    for family, fam_base in base.groupby("family"):
        for unit in units:
            if unit not in fam_base:
                continue
            sub = fam_base.copy()
            sub["event_count"] = sub[unit].fillna(0)
            sub["used"] = sub["event_count"] > 0
            used = sub[sub["used"]]
            unused = sub[~sub["used"]]
            def corr(a: pd.Series, b: pd.Series) -> float:
                clean = pd.DataFrame({"a": a, "b": b}).dropna()
                if len(clean) < 4 or clean["a"].nunique() < 2 or clean["b"].nunique() < 2:
                    return math.nan
                return float(clean["a"].corr(clean["b"]))
            rows.append(
                {
                    "family": family,
                    unit_col: unit,
                    "label": labels.get(unit, unit),
                    "n_target_rollouts": len(sub),
                    "n_used": int(sub["used"].sum()),
                    "used_rate": float(sub["used"].mean()),
                    "mean_target_payoff_used": float(used["target_utility"].mean()) if len(used) else math.nan,
                    "mean_target_payoff_not_used": float(unused["target_utility"].mean()) if len(unused) else math.nan,
                    "delta_target_payoff_used_minus_not": (
                        float(used["target_utility"].mean() - unused["target_utility"].mean())
                        if len(used) and len(unused)
                        else math.nan
                    ),
                    "point_biserial_r_target_payoff": corr(sub["used"].astype(float), sub["target_utility"]),
                    "point_biserial_r_target_payoff_resid": corr(sub["used"].astype(float), sub["target_utility_resid"]),
                    "point_biserial_r_consensus_resid": corr(sub["used"].astype(float), sub["consensus_resid"]),
                    "point_biserial_r_final_round_resid": corr(sub["used"].astype(float), sub["final_round_resid"]),
                    "spearman_count_r_target_payoff_resid": corr(sub["event_count"].astype(float), sub["target_utility_resid"]),
                }
            )
    return pd.DataFrame(rows)


def save_payoff_barplot(
    df: pd.DataFrame,
    unit_col: str,
    value_col: str,
    out_path: Path,
    title: str,
    top_n: int | None = None,
) -> None:
    if df.empty:
        return
    d = df.copy()
    if top_n:
        pos = d.sort_values(value_col, ascending=False).head(top_n)
        neg = d.sort_values(value_col, ascending=True).head(top_n)
        d = pd.concat([pos, neg], ignore_index=True).drop_duplicates([unit_col, "family"])
    d = d.sort_values(value_col)
    labels = d["label"].astype(str) + " [" + d["family"].map(FAMILY_LABELS).fillna(d["family"]) + "]"
    fig_h = max(6.0, 0.28 * len(d) + 1.8)
    fig, ax = plt.subplots(figsize=(11.2, fig_h))
    colors = np.where(d[value_col] >= 0, "#2563eb", "#dc2626")
    ax.barh(labels, d[value_col], color=colors, alpha=0.85)
    ax.axvline(0, color="#111827", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel(value_col.replace("_", " "))
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, axis="x", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def md_table(df: pd.DataFrame, cols: list[str], n: int = 12, pct_cols: set[str] | None = None) -> str:
    pct_cols = pct_cols or set()
    if df.empty:
        return "_No rows._"
    d = df.head(n).copy()
    for c in d.columns:
        if c in pct_cols:
            d[c] = d[c].map(lambda x: f"{100*x:.1f}%" if pd.notna(x) else "")
        elif d[c].dtype.kind in "fc":
            d[c] = d[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    return d[cols].to_markdown(index=False)


def write_report(
    out_dir: Path,
    input_dir: Path,
    tag_meta: pd.DataFrame,
    tag_freq: pd.DataFrame,
    group_freq: pd.DataFrame,
    tag_trends: pd.DataFrame,
    group_trends: pd.DataFrame,
    denominators: pd.DataFrame,
    tag_payoff: pd.DataFrame,
    group_payoff: pd.DataFrame,
    speaker_role: str,
) -> None:
    all_table = tag_meta.copy()
    all_table["tldr"] = all_table["definition"]
    coverage = (
        denominators.groupby("family")
        .agg(target_rollouts=("speaker_key", "nunique"), levels=("level", lambda s: ", ".join(map(str, sorted(set(s), key=lambda x: LEVEL_ORDER.get(x, 99))))), mean_tokens_min=("mean_tokens_per_call", "min"), mean_tokens_max=("mean_tokens_per_call", "max"))
        .reset_index()
    )
    coverage["family_label"] = coverage["family"].map(FAMILY_LABELS).fillna(coverage["family"])

    def image(path: str, alt: str) -> str:
        return f"![{alt}]({path})"

    lines: list[str] = [
        "# TTC Strategic Tag Frequency vs Reasoning Effort",
        "",
        "## Scope",
        "",
        f"Input bundle: `{input_dir}`.",
        "",
        f"Speaker role analyzed: `{speaker_role}`. This is the primary TTC view because the target model is the participant whose reasoning effort varies.",
        "",
        "Frequency is measured as a target-rollout occurrence rate: for a given family, effort level, and tag, the numerator is the number of target rollouts in which the target produced at least one event with that tag; the denominator is the number of target rollouts in that family-level cell. This avoids letting verbose transcripts dominate purely by event count.",
        "",
        "The report includes all 50 codebook tags, not only the 29 tags marked `hot` in `strategic_tag_review_final.json`. The review decision is retained in the tag table for reference.",
        "",
        "## Coverage",
        "",
        md_table(coverage, ["family_label", "target_rollouts", "levels", "mean_tokens_min", "mean_tokens_max"], n=10),
        "",
        "## All Tags Used",
        "",
        md_table(all_table, ["tag_rank", "tag_code", "tag_title", "category", "review_decision", "tldr"], n=60),
        "",
        "## Plots",
        "",
        "Each subplot shows one tag or group. The x-axis is requested reasoning effort for the family-specific plots. The overlay plots use observed target reasoning tokens per call so GPT-5, Claude, and Gemini can be compared on a shared compute axis.",
        "",
        "### Family-Specific Line Plots",
        "",
    ]
    for family in FAMILY_ORDER:
        lines += [
            f"#### {FAMILY_LABELS[family]}",
            "",
            image(f"plots/{family}_all_tag_line_grid.png", f"{FAMILY_LABELS[family]} all-tag line grid"),
            "",
            image(f"plots/{family}_group_line_grid.png", f"{FAMILY_LABELS[family]} group line grid"),
            "",
        ]
    lines += [
        "### Cross-Family Overlay Plots",
        "",
        image("plots/all_tags_observed_tokens_overlay_grid.png", "All tags versus observed target tokens"),
        "",
        image("plots/groups_observed_tokens_overlay_grid.png", "Groups versus observed target tokens"),
        "",
        "### Top Tag Trend Plots",
        "",
    ]
    for family in FAMILY_ORDER:
        lines += [f"#### {FAMILY_LABELS[family]}", ""]
        fam_top = tag_trends[tag_trends["family"].eq(family)].sort_values("trend_strength", ascending=False).head(8)
        for row in fam_top.itertuples():
            lines += [
                image(
                    f"plots/top_tag_trends/{family}_{slugify(str(row.tag_code))}_trend.png",
                    f"{FAMILY_LABELS[family]} {row.label} trend",
                ),
                "",
            ]
    lines += ["### Group Trend Plots", ""]
    for family in FAMILY_ORDER:
        lines += [f"#### {FAMILY_LABELS[family]}", ""]
        fam_groups = group_trends[group_trends["family"].eq(family)].sort_values("trend_strength", ascending=False)
        for row in fam_groups.itertuples():
            lines += [
                image(
                    f"plots/top_group_trends/{family}_{slugify(str(row.group))}_trend.png",
                    f"{FAMILY_LABELS[family]} {row.group} trend",
                ),
                "",
            ]
    lines += [
        "## Strongest Tag Trends",
        "",
        "Trends are ranked by absolute weak-to-strong change plus a small Spearman-effort tie-breaker. `weak` is minimal for GPT-5/Gemini and low for Claude; `strong` is high for GPT-5/Gemini and max for Claude.",
        "",
    ]
    for family in FAMILY_ORDER:
        fam = tag_trends[tag_trends["family"].eq(family)].sort_values("trend_strength", ascending=False)
        lines += [
            f"### {FAMILY_LABELS[family]}",
            "",
            md_table(
                fam,
                [
                    "tag_code",
                    "label",
                    "group",
                    "weak_rate",
                    "strong_rate",
                    "delta_strong_minus_weak",
                    "spearman_effort_r",
                    "spearman_tokens_r",
                ],
                n=16,
                pct_cols={"weak_rate", "strong_rate", "delta_strong_minus_weak"},
            ),
            "",
        ]
    lines += [
        "## Group Trends",
        "",
    ]
    for family in FAMILY_ORDER:
        fam = group_trends[group_trends["family"].eq(family)].sort_values("trend_strength", ascending=False)
        lines += [
            f"### {FAMILY_LABELS[family]}",
            "",
            md_table(
                fam,
                ["group", "weak_rate", "strong_rate", "delta_strong_minus_weak", "spearman_effort_r", "spearman_tokens_r"],
                n=20,
                pct_cols={"weak_rate", "strong_rate", "delta_strong_minus_weak"},
            ),
            "",
        ]
    lines += [
        "## Payoff Associations",
        "",
        "This section asks whether target rollouts with a tag have higher target payoff than target rollouts without it. These are associations, not causal estimates. I include both raw payoff gaps and matched-cell residual correlations. The residual version subtracts the mean within the same family, game cell, and speaking order.",
        "",
        "### Tag Associations",
        "",
        image("plots/payoff/tag_target_payoff_resid_assoc.png", "Tag target payoff residual associations"),
        "",
        md_table(
            tag_payoff.sort_values("point_biserial_r_target_payoff_resid", ascending=False),
            [
                "family",
                "tag_code",
                "label",
                "n_used",
                "used_rate",
                "delta_target_payoff_used_minus_not",
                "point_biserial_r_target_payoff_resid",
                "point_biserial_r_consensus_resid",
                "point_biserial_r_final_round_resid",
            ],
            n=20,
            pct_cols={"used_rate"},
        ),
        "",
        "### Group Associations",
        "",
        image("plots/payoff/group_target_payoff_resid_assoc.png", "Group target payoff residual associations"),
        "",
        md_table(
            group_payoff.sort_values("point_biserial_r_target_payoff_resid", ascending=False),
            [
                "family",
                "group",
                "label",
                "n_used",
                "used_rate",
                "delta_target_payoff_used_minus_not",
                "point_biserial_r_target_payoff_resid",
                "point_biserial_r_consensus_resid",
                "point_biserial_r_final_round_resid",
            ],
            n=20,
            pct_cols={"used_rate"},
        ),
        "",
        "## Interpretation",
        "",
    ]
    for family in FAMILY_ORDER:
        fam_tag = tag_trends[tag_trends["family"].eq(family)].sort_values("trend_strength", ascending=False)
        fam_group = group_trends[group_trends["family"].eq(family)].sort_values("trend_strength", ascending=False)
        pos_tags = fam_tag[fam_tag["delta_strong_minus_weak"] > 0].head(6)
        neg_tags = fam_tag[fam_tag["delta_strong_minus_weak"] < 0].head(6)
        pos_groups = fam_group[fam_group["delta_strong_minus_weak"] > 0].head(4)
        neg_groups = fam_group[fam_group["delta_strong_minus_weak"] < 0].head(4)
        lines += [
            f"### {FAMILY_LABELS[family]}",
            "",
            "Most increased tags: "
            + (
                ", ".join(
                    f"`{r.tag_code}` ({r.label}, {100*r.weak_rate:.1f}% -> {100*r.strong_rate:.1f}%)"
                    for r in pos_tags.itertuples()
                )
                if not pos_tags.empty
                else "_none_"
            )
            + ".",
            "",
            "Most decreased tags: "
            + (
                ", ".join(
                    f"`{r.tag_code}` ({r.label}, {100*r.weak_rate:.1f}% -> {100*r.strong_rate:.1f}%)"
                    for r in neg_tags.itertuples()
                )
                if not neg_tags.empty
                else "_none_"
            )
            + ".",
            "",
            "Most increased groups: "
            + (
                ", ".join(
                    f"`{r.group}` ({100*r.weak_rate:.1f}% -> {100*r.strong_rate:.1f}%)"
                    for r in pos_groups.itertuples()
                )
                if not pos_groups.empty
                else "_none_"
            )
            + ".",
            "",
            "Most decreased groups: "
            + (
                ", ".join(
                    f"`{r.group}` ({100*r.weak_rate:.1f}% -> {100*r.strong_rate:.1f}%)"
                    for r in neg_groups.itertuples()
                )
                if not neg_groups.empty
                else "_none_"
            )
            + ".",
            "",
        ]
    lines += [
        "## Output Tables",
        "",
        "- [tag_family_level_frequency.csv](tag_family_level_frequency.csv)",
        "- [group_family_level_frequency.csv](group_family_level_frequency.csv)",
        "- [tag_ttc_trends.csv](tag_ttc_trends.csv)",
        "- [group_ttc_trends.csv](group_ttc_trends.csv)",
        "- [target_denominators.csv](target_denominators.csv)",
        "- [tag_payoff_target_rollout_correlations.csv](tag_payoff_target_rollout_correlations.csv)",
        "- [group_payoff_target_rollout_correlations.csv](group_payoff_target_rollout_correlations.csv)",
        "- [all_tags_used.csv](all_tags_used.csv)",
        "",
        "## Caveats",
        "",
        "- TTC levels are requested effort settings; observed token counts are non-monotone for some providers, especially Claude. The report therefore shows both requested-effort trends and observed-token overlays.",
        "- Each family-level cell has 18 target rollouts, so individual tag rates are exploratory.",
        "- Occurrence rates are binary per rollout. Event-count rates are in CSVs but not emphasized because they can reward verbosity.",
        "- Payoff associations are descriptive and may reflect reverse causality: a tag can appear because the negotiation is hard, not because the tag caused the outcome.",
    ]
    (out_dir / "strategic_tag_ttc_exploration_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--review-json", type=Path, default=DEFAULT_REVIEW_JSON)
    parser.add_argument("--speaker-role", choices=["target", "baseline"], default="target")
    args = parser.parse_args()

    out_dir = args.output_dir
    plots_dir = out_dir / "plots"
    tag_trend_dir = plots_dir / "top_tag_trends"
    group_trend_dir = plots_dir / "top_group_trends"
    payoff_dir = plots_dir / "payoff"
    for path in [out_dir, plots_dir, tag_trend_dir, group_trend_dir, payoff_dir]:
        path.mkdir(parents=True, exist_ok=True)

    manifest, events, tag_meta = load_inputs(args.input_dir, args.review_json)
    denoms = build_denominators(manifest, speaker_role=args.speaker_role)
    tag_freq, group_freq = make_frequency_tables(denoms, events, tag_meta, speaker_role=args.speaker_role)
    tag_trends = trend_table(tag_freq, "tag_code", "tag_title")
    group_trends = trend_table(group_freq, "group", "group")
    tag_matrix, group_matrix = build_rollout_matrices(denoms, events, tag_meta, speaker_role=args.speaker_role)
    tag_payoff = payoff_association_table(
        tag_matrix, tag_meta["tag_code"].tolist(), "tag_code", dict(zip(tag_meta.tag_code, tag_meta.tag_title))
    )
    group_payoff = payoff_association_table(
        group_matrix, sorted(tag_meta["category"].dropna().unique()), "group", {g: g for g in tag_meta["category"].dropna().unique()}
    )

    tag_meta.to_csv(out_dir / "all_tags_used.csv", index=False)
    denoms.to_csv(out_dir / "target_denominators.csv", index=False)
    tag_freq.to_csv(out_dir / "tag_family_level_frequency.csv", index=False)
    group_freq.to_csv(out_dir / "group_family_level_frequency.csv", index=False)
    tag_trends.to_csv(out_dir / "tag_ttc_trends.csv", index=False)
    group_trends.to_csv(out_dir / "group_ttc_trends.csv", index=False)
    tag_payoff.to_csv(out_dir / "tag_payoff_target_rollout_correlations.csv", index=False)
    group_payoff.to_csv(out_dir / "group_payoff_target_rollout_correlations.csv", index=False)

    for family in FAMILY_ORDER:
        save_line_grid(
            tag_freq,
            family,
            "tag_code",
            "tag_title",
            plots_dir / f"{family}_all_tag_line_grid.png",
            f"{FAMILY_LABELS[family]}: all tag frequencies across TTC",
            ncols=5,
        )
        save_line_grid(
            group_freq,
            family,
            "group",
            "group",
            plots_dir / f"{family}_group_line_grid.png",
            f"{FAMILY_LABELS[family]}: group frequencies across TTC",
            ncols=3,
        )
        save_top_trend_plots(tag_freq, tag_trends, family, "tag_code", "tag_title", tag_trend_dir, top_n=8)
        save_top_trend_plots(group_freq, group_trends, family, "group", "group", group_trend_dir, top_n=10)

    save_overlay_line_grid(
        tag_freq,
        "tag_code",
        "tag_title",
        plots_dir / "all_tags_observed_tokens_overlay_grid.png",
        "All tag frequencies versus observed target tokens/call",
        ncols=5,
    )
    save_overlay_line_grid(
        group_freq,
        "group",
        "group",
        plots_dir / "groups_observed_tokens_overlay_grid.png",
        "Group frequencies versus observed target tokens/call",
        ncols=3,
    )
    save_payoff_barplot(
        tag_payoff,
        "tag_code",
        "point_biserial_r_target_payoff_resid",
        payoff_dir / "tag_target_payoff_resid_assoc.png",
        "Target payoff residual association by tag",
        top_n=18,
    )
    save_payoff_barplot(
        group_payoff,
        "group",
        "point_biserial_r_target_payoff_resid",
        payoff_dir / "group_target_payoff_resid_assoc.png",
        "Target payoff residual association by group",
        top_n=None,
    )
    write_report(
        out_dir,
        args.input_dir,
        tag_meta,
        tag_freq,
        group_freq,
        tag_trends,
        group_trends,
        denoms,
        tag_payoff,
        group_payoff,
        args.speaker_role,
    )
    for path in [
        out_dir / "strategic_tag_ttc_exploration_report.md",
        plots_dir / "all_tags_observed_tokens_overlay_grid.png",
        plots_dir / "groups_observed_tokens_overlay_grid.png",
    ]:
        print(path)


if __name__ == "__main__":
    main()

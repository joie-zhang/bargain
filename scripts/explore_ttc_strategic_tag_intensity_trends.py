#!/usr/bin/env python3
"""Explore TTC strategic tag intensity versus reasoning effort.

This is the intensity-metric companion to
``explore_ttc_strategic_tag_trends.py``.  The binary report asks whether a tag
appears at least once in a target rollout.  This report instead asks how many
times the target uses each tag per rollout, which exposes scaling effects that
are hidden when broad categories are already saturated.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from explore_ttc_strategic_tag_trends import (
    DEFAULT_INPUT_DIR,
    DEFAULT_REVIEW_JSON,
    FAMILY_COLORS,
    FAMILY_LABELS,
    FAMILY_ORDER,
    LEVEL_COLORS,
    LEVEL_ORDER,
    build_denominators,
    build_rollout_matrices,
    load_inputs,
    make_frequency_tables,
    md_table,
    payoff_association_table,
    slugify,
    style_axis,
    summarize_denominators,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_OUTPUT_DIR = Path("analysis/ttc_strategic_tag_intensity_lines_payoff_20260629")
METRIC_COL = "events_per_rollout"
METRIC_LABEL = "Mean tag events per target rollout"
GROUP_METRIC_LABEL = "Mean group-tag events per target rollout"


def trend_table_intensity(freq: pd.DataFrame, unit_col: str, label_col: str) -> pd.DataFrame:
    rows = []
    for (family, unit), g in freq.groupby(["family", unit_col]):
        g = g.sort_values("level_index")
        if g["level_index"].nunique() < 3:
            continue
        weak = "minimal" if "minimal" in set(g["level"]) else g.iloc[0]["level"]
        strong = "high" if "high" in set(g["level"]) else g.iloc[-1]["level"]
        intensity = {row["level"]: row[METRIC_COL] for _, row in g.iterrows()}
        occurrence = {row["level"]: row["occurrence_rate"] for _, row in g.iterrows()}
        x = g["level_index"].to_numpy(dtype=float)
        y = g[METRIC_COL].to_numpy(dtype=float)
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
                "weak_intensity": intensity.get(weak, math.nan),
                "strong_intensity": intensity.get(strong, math.nan),
                "delta_intensity_strong_minus_weak": intensity.get(strong, math.nan) - intensity.get(weak, math.nan),
                "weak_occurrence_rate": occurrence.get(weak, math.nan),
                "strong_occurrence_rate": occurrence.get(strong, math.nan),
                "delta_occurrence_rate": occurrence.get(strong, math.nan) - occurrence.get(weak, math.nan),
                "spearman_effort_r": effort_corr,
                "spearman_tokens_r": token_corr,
                "slope_per_effort_level": slope_per_effort,
                "slope_per_1k_tokens": slope_per_1k_tokens,
                "mean_intensity": float(y.mean()),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["trend_strength"] = out["delta_intensity_strong_minus_weak"].abs() + out["spearman_effort_r"].abs() / 10.0
    return out.sort_values(["family", "trend_strength"], ascending=[True, False])


def save_line_grid_intensity(
    freq: pd.DataFrame,
    family: str,
    unit_col: str,
    label_col: str,
    out_path: Path,
    title: str,
    ylabel: str,
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
    max_y = max(0.05, float(sub[METRIC_COL].max()) * 1.16)
    for ax, unit in zip(axes.ravel(), units, strict=False):
        g = sub[sub[unit_col].eq(unit[unit_col])].sort_values("level_index")
        x = g["mean_tokens_per_call"] if use_tokens else g["level_index"]
        ax.plot(x, g[METRIC_COL], marker="o", linewidth=1.5, markersize=4, color=FAMILY_COLORS.get(family, "#333"))
        for _, row in g.iterrows():
            xval = row["mean_tokens_per_call"] if use_tokens else row["level_index"]
            ax.annotate(
                str(row["level"]),
                (xval, row[METRIC_COL]),
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
    fig.supylabel(ylabel, fontsize=10)
    fig.tight_layout(rect=[0.015, 0.015, 1, 0.985])
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def save_overlay_line_grid_intensity(
    freq: pd.DataFrame,
    unit_col: str,
    label_col: str,
    out_path: Path,
    title: str,
    ylabel: str,
    ncols: int = 5,
) -> None:
    if unit_col == label_col:
        units = [{unit_col: value, label_col: value} for value in sorted(freq[unit_col].dropna().unique())]
    else:
        units = freq[[unit_col, label_col]].drop_duplicates().sort_values(label_col).to_dict("records")
    nrows = math.ceil(len(units) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.7 * ncols, 2.75 * nrows), squeeze=False)
    max_y = max(0.05, float(freq[METRIC_COL].max()) * 1.16)
    for ax, unit in zip(axes.ravel(), units, strict=False):
        for family in FAMILY_ORDER:
            g = freq[(freq[unit_col].eq(unit[unit_col])) & (freq["family"].eq(family))].sort_values("level_index")
            if g.empty:
                continue
            ax.plot(
                g["mean_tokens_per_call"],
                g[METRIC_COL],
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
    fig.supylabel(ylabel, fontsize=10)
    fig.tight_layout(rect=[0.015, 0.04, 1, 0.985])
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def save_top_trend_plots_intensity(
    freq: pd.DataFrame,
    trend: pd.DataFrame,
    family: str,
    unit_col: str,
    out_dir: Path,
    top_n: int,
    ylabel: str,
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
        ax.plot(g["level_index"], g[METRIC_COL], marker="o", linewidth=2, color=FAMILY_COLORS.get(family, "#333"))
        for _, point in g.iterrows():
            ax.annotate(
                f"{point['level']}\n{point['mean_tokens_per_call']:.0f} tok",
                (point["level_index"], point[METRIC_COL]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=7,
            )
        ax.set_title(f"{FAMILY_LABELS.get(family, family)}: {row['label']}")
        ax.set_xlabel("Requested reasoning effort")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(g["level_index"].unique()), [str(v) for v in g.sort_values("level_index")["level"]], rotation=20)
        ax.set_ylim(bottom=0)
        style_axis(ax)
        fig.tight_layout()
        fig.savefig(out, dpi=220, bbox_inches="tight")
        plt.close(fig)
        paths.append(out)
    return paths


def save_payoff_barplot_intensity(
    df: pd.DataFrame,
    unit_col: str,
    value_col: str,
    out_path: Path,
    title: str,
    top_n: int | None = None,
) -> None:
    if df.empty:
        return
    d = df.dropna(subset=[value_col]).copy()
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
    ax.set_xlabel("Correlation between tag count and target-payoff residual")
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, axis="x", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def image(path: str, alt: str) -> str:
    return f"![{alt}]({path})"


def write_intensity_report(
    out_dir: Path,
    input_dir: Path,
    tag_meta: pd.DataFrame,
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
        .agg(
            target_rollouts=("speaker_key", "nunique"),
            levels=("level", lambda s: ", ".join(map(str, sorted(set(s), key=lambda x: LEVEL_ORDER.get(x, 99))))),
            mean_tokens_min=("mean_tokens_per_call", "min"),
            mean_tokens_max=("mean_tokens_per_call", "max"),
        )
        .reset_index()
    )
    coverage["family_label"] = coverage["family"].map(FAMILY_LABELS).fillna(coverage["family"])

    lines: list[str] = [
        "# TTC Strategic Tag Intensity vs Reasoning Effort",
        "",
        "## Scope",
        "",
        f"Input bundle: `{input_dir}`.",
        "",
        f"Speaker role analyzed: `{speaker_role}`. This is the primary TTC view because the target model is the participant whose reasoning effort varies.",
        "",
        "The main metric is **tag intensity**, defined as the mean number of tagged target-speaker events per target rollout in a family/effort cell. For example, a value of `2.0` for `self_advocacy_value_maximization` means that the target agent produced two self-advocacy events per rollout on average at that effort level.",
        "",
        "This differs from the binary report, where a rollout counted the same whether a tag appeared once or ten times. The intensity view is better for TTC because several broad categories are saturated: almost every rollout already contains at least one logical or self-interested statement, so the important question is whether extra reasoning makes those behaviors more frequent.",
        "",
        "The report includes all 50 codebook tags and all codebook groups, not only the 29 tags marked `hot` in `strategic_tag_review_final.json`. The review decision is retained in the tag table for reference.",
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
        "Each subplot shows one tag or group. The y-axis is the mean number of tag events per target rollout. The x-axis is requested reasoning effort for family-specific plots. The overlay plots use observed target reasoning tokens per call so GPT-5, Claude, and Gemini can be compared on a shared compute axis.",
        "",
        "### Family-Specific Line Plots",
        "",
    ]
    for family in FAMILY_ORDER:
        lines += [
            f"#### {FAMILY_LABELS[family]}",
            "",
            image(f"plots/{family}_all_tag_intensity_line_grid.png", f"{FAMILY_LABELS[family]} all-tag intensity line grid"),
            "",
            image(f"plots/{family}_group_intensity_line_grid.png", f"{FAMILY_LABELS[family]} group intensity line grid"),
            "",
        ]
    lines += [
        "### Cross-Family Overlay Plots",
        "",
        image("plots/all_tags_observed_tokens_intensity_overlay_grid.png", "All tag intensities versus observed target tokens"),
        "",
        image("plots/groups_observed_tokens_intensity_overlay_grid.png", "Group intensities versus observed target tokens"),
        "",
        "### Top Tag Intensity Trend Plots",
        "",
    ]
    for family in FAMILY_ORDER:
        lines += [f"#### {FAMILY_LABELS[family]}", ""]
        fam_top = tag_trends[tag_trends["family"].eq(family)].sort_values("trend_strength", ascending=False).head(8)
        for row in fam_top.itertuples():
            lines += [
                image(
                    f"plots/top_tag_intensity_trends/{family}_{slugify(str(row.tag_code))}_trend.png",
                    f"{FAMILY_LABELS[family]} {row.label} intensity trend",
                ),
                "",
            ]
    lines += ["### Group Intensity Trend Plots", ""]
    for family in FAMILY_ORDER:
        lines += [f"#### {FAMILY_LABELS[family]}", ""]
        fam_groups = group_trends[group_trends["family"].eq(family)].sort_values("trend_strength", ascending=False)
        for row in fam_groups.itertuples():
            lines += [
                image(
                    f"plots/top_group_intensity_trends/{family}_{slugify(str(row.group))}_trend.png",
                    f"{FAMILY_LABELS[family]} {row.group} intensity trend",
                ),
                "",
            ]

    lines += [
        "## Strongest Tag Intensity Trends",
        "",
        "Trends are ranked by absolute weak-to-strong change in events per rollout plus a small Spearman-effort tie-breaker. `weak` is minimal for GPT-5/Gemini and low for Claude; `strong` is high for GPT-5/Gemini and max for Claude.",
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
                    "weak_intensity",
                    "strong_intensity",
                    "delta_intensity_strong_minus_weak",
                    "weak_occurrence_rate",
                    "strong_occurrence_rate",
                    "spearman_effort_r",
                    "spearman_tokens_r",
                ],
                n=18,
                pct_cols={"weak_occurrence_rate", "strong_occurrence_rate"},
            ),
            "",
        ]
    lines += ["## Group Intensity Trends", ""]
    for family in FAMILY_ORDER:
        fam = group_trends[group_trends["family"].eq(family)].sort_values("trend_strength", ascending=False)
        lines += [
            f"### {FAMILY_LABELS[family]}",
            "",
            md_table(
                fam,
                [
                    "group",
                    "weak_intensity",
                    "strong_intensity",
                    "delta_intensity_strong_minus_weak",
                    "weak_occurrence_rate",
                    "strong_occurrence_rate",
                    "spearman_effort_r",
                    "spearman_tokens_r",
                ],
                n=20,
                pct_cols={"weak_occurrence_rate", "strong_occurrence_rate"},
            ),
            "",
        ]
    lines += [
        "## Payoff Associations",
        "",
        "This section asks whether rollouts with **more events** of a tag or group have higher target payoff. The plotted statistic is the correlation between event count and target-payoff residual. The residual subtracts the mean payoff within the same family, game cell, and speaking order. This is still descriptive rather than causal.",
        "",
        "### Tag-Count Associations",
        "",
        image("plots/payoff/tag_count_target_payoff_resid_assoc.png", "Tag-count target payoff residual associations"),
        "",
        md_table(
            tag_payoff.sort_values("spearman_count_r_target_payoff_resid", ascending=False),
            [
                "family",
                "tag_code",
                "label",
                "n_used",
                "used_rate",
                "delta_target_payoff_used_minus_not",
                "spearman_count_r_target_payoff_resid",
                "point_biserial_r_target_payoff_resid",
            ],
            n=24,
            pct_cols={"used_rate"},
        ),
        "",
        "### Group-Count Associations",
        "",
        image("plots/payoff/group_count_target_payoff_resid_assoc.png", "Group-count target payoff residual associations"),
        "",
        md_table(
            group_payoff.sort_values("spearman_count_r_target_payoff_resid", ascending=False),
            [
                "family",
                "group",
                "label",
                "n_used",
                "used_rate",
                "delta_target_payoff_used_minus_not",
                "spearman_count_r_target_payoff_resid",
                "point_biserial_r_target_payoff_resid",
            ],
            n=24,
            pct_cols={"used_rate"},
        ),
        "",
        "## Interpretation",
        "",
    ]
    for family in FAMILY_ORDER:
        fam_tag = tag_trends[tag_trends["family"].eq(family)].sort_values("trend_strength", ascending=False)
        fam_group = group_trends[group_trends["family"].eq(family)].sort_values("trend_strength", ascending=False)
        pos_tags = fam_tag[fam_tag["delta_intensity_strong_minus_weak"] > 0].head(7)
        neg_tags = fam_tag[fam_tag["delta_intensity_strong_minus_weak"] < 0].head(7)
        pos_groups = fam_group[fam_group["delta_intensity_strong_minus_weak"] > 0].head(4)
        neg_groups = fam_group[fam_group["delta_intensity_strong_minus_weak"] < 0].head(4)
        lines += [
            f"### {FAMILY_LABELS[family]}",
            "",
            "Most increased tags by intensity: "
            + (
                ", ".join(
                    f"`{r.tag_code}` ({r.label}, {r.weak_intensity:.2f} -> {r.strong_intensity:.2f} events/rollout)"
                    for r in pos_tags.itertuples()
                )
                if not pos_tags.empty
                else "_none_"
            )
            + ".",
            "",
            "Most decreased tags by intensity: "
            + (
                ", ".join(
                    f"`{r.tag_code}` ({r.label}, {r.weak_intensity:.2f} -> {r.strong_intensity:.2f} events/rollout)"
                    for r in neg_tags.itertuples()
                )
                if not neg_tags.empty
                else "_none_"
            )
            + ".",
            "",
            "Most increased groups by intensity: "
            + (
                ", ".join(
                    f"`{r.group}` ({r.weak_intensity:.2f} -> {r.strong_intensity:.2f} events/rollout)"
                    for r in pos_groups.itertuples()
                )
                if not pos_groups.empty
                else "_none_"
            )
            + ".",
            "",
            "Most decreased groups by intensity: "
            + (
                ", ".join(
                    f"`{r.group}` ({r.weak_intensity:.2f} -> {r.strong_intensity:.2f} events/rollout)"
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
        "- [tag_family_level_intensity.csv](tag_family_level_intensity.csv)",
        "- [group_family_level_intensity.csv](group_family_level_intensity.csv)",
        "- [tag_ttc_intensity_trends.csv](tag_ttc_intensity_trends.csv)",
        "- [group_ttc_intensity_trends.csv](group_ttc_intensity_trends.csv)",
        "- [target_denominators.csv](target_denominators.csv)",
        "- [tag_payoff_target_rollout_correlations.csv](tag_payoff_target_rollout_correlations.csv)",
        "- [group_payoff_target_rollout_correlations.csv](group_payoff_target_rollout_correlations.csv)",
        "- [all_tags_used.csv](all_tags_used.csv)",
        "",
        "## Caveats",
        "",
        "- Intensity can reward verbosity: a model that repeats the same move many times will have a higher event count. That is exactly why this report should be read together with the binary occurrence report.",
        "- TTC levels are requested effort settings; observed token counts are non-monotone for some providers, especially Claude. The report therefore shows both requested-effort trends and observed-token overlays.",
        "- Each family-level cell has 18 target rollouts, so individual tag trends are exploratory.",
        "- Payoff associations are descriptive and may reflect reverse causality: a tag can appear often because the negotiation is hard, not because the tag caused the outcome.",
    ]
    (out_dir / "strategic_tag_ttc_intensity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--review-json", type=Path, default=DEFAULT_REVIEW_JSON)
    parser.add_argument("--speaker-role", choices=["target", "baseline"], default="target")
    args = parser.parse_args()

    out_dir = args.output_dir
    plots_dir = out_dir / "plots"
    tag_trend_dir = plots_dir / "top_tag_intensity_trends"
    group_trend_dir = plots_dir / "top_group_intensity_trends"
    payoff_dir = plots_dir / "payoff"
    for path in [out_dir, plots_dir, tag_trend_dir, group_trend_dir, payoff_dir]:
        path.mkdir(parents=True, exist_ok=True)

    manifest, events, tag_meta = load_inputs(args.input_dir, args.review_json)
    denoms = build_denominators(manifest, speaker_role=args.speaker_role)
    tag_freq, group_freq = make_frequency_tables(denoms, events, tag_meta, speaker_role=args.speaker_role)
    tag_trends = trend_table_intensity(tag_freq, "tag_code", "tag_title")
    group_trends = trend_table_intensity(group_freq, "group", "group")
    tag_matrix, group_matrix = build_rollout_matrices(denoms, events, tag_meta, speaker_role=args.speaker_role)
    tag_payoff = payoff_association_table(
        tag_matrix, tag_meta["tag_code"].tolist(), "tag_code", dict(zip(tag_meta.tag_code, tag_meta.tag_title))
    )
    group_payoff = payoff_association_table(
        group_matrix,
        sorted(tag_meta["category"].dropna().unique()),
        "group",
        {g: g for g in tag_meta["category"].dropna().unique()},
    )

    tag_meta.to_csv(out_dir / "all_tags_used.csv", index=False)
    denoms.to_csv(out_dir / "target_denominators.csv", index=False)
    summarize_denominators(denoms).to_csv(out_dir / "family_level_outcomes.csv", index=False)
    tag_freq.to_csv(out_dir / "tag_family_level_intensity.csv", index=False)
    group_freq.to_csv(out_dir / "group_family_level_intensity.csv", index=False)
    tag_trends.to_csv(out_dir / "tag_ttc_intensity_trends.csv", index=False)
    group_trends.to_csv(out_dir / "group_ttc_intensity_trends.csv", index=False)
    tag_payoff.to_csv(out_dir / "tag_payoff_target_rollout_correlations.csv", index=False)
    group_payoff.to_csv(out_dir / "group_payoff_target_rollout_correlations.csv", index=False)

    for family in FAMILY_ORDER:
        save_line_grid_intensity(
            tag_freq,
            family,
            "tag_code",
            "tag_title",
            plots_dir / f"{family}_all_tag_intensity_line_grid.png",
            f"{FAMILY_LABELS[family]}: all tag intensities across TTC",
            METRIC_LABEL,
            ncols=5,
        )
        save_line_grid_intensity(
            group_freq,
            family,
            "group",
            "group",
            plots_dir / f"{family}_group_intensity_line_grid.png",
            f"{FAMILY_LABELS[family]}: group intensities across TTC",
            GROUP_METRIC_LABEL,
            ncols=3,
        )
        save_top_trend_plots_intensity(tag_freq, tag_trends, family, "tag_code", tag_trend_dir, top_n=8, ylabel=METRIC_LABEL)
        save_top_trend_plots_intensity(group_freq, group_trends, family, "group", group_trend_dir, top_n=10, ylabel=GROUP_METRIC_LABEL)

    save_overlay_line_grid_intensity(
        tag_freq,
        "tag_code",
        "tag_title",
        plots_dir / "all_tags_observed_tokens_intensity_overlay_grid.png",
        "All tag intensities versus observed target tokens/call",
        METRIC_LABEL,
        ncols=5,
    )
    save_overlay_line_grid_intensity(
        group_freq,
        "group",
        "group",
        plots_dir / "groups_observed_tokens_intensity_overlay_grid.png",
        "Group intensities versus observed target tokens/call",
        GROUP_METRIC_LABEL,
        ncols=3,
    )
    save_payoff_barplot_intensity(
        tag_payoff,
        "tag_code",
        "spearman_count_r_target_payoff_resid",
        payoff_dir / "tag_count_target_payoff_resid_assoc.png",
        "Target payoff residual association by tag-event count",
        top_n=18,
    )
    save_payoff_barplot_intensity(
        group_payoff,
        "group",
        "spearman_count_r_target_payoff_resid",
        payoff_dir / "group_count_target_payoff_resid_assoc.png",
        "Target payoff residual association by group-event count",
        top_n=None,
    )
    write_intensity_report(
        out_dir,
        args.input_dir,
        tag_meta,
        tag_trends,
        group_trends,
        denoms,
        tag_payoff,
        group_payoff,
        args.speaker_role,
    )
    for path in [
        out_dir / "strategic_tag_ttc_intensity_report.md",
        plots_dir / "all_tags_observed_tokens_intensity_overlay_grid.png",
        plots_dir / "groups_observed_tokens_intensity_overlay_grid.png",
    ]:
        print(path)


if __name__ == "__main__":
    main()

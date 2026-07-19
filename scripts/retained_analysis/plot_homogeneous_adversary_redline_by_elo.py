#!/usr/bin/env python3
"""Plot homogeneous-adversary hard-anchor/redline rates by adversary Elo."""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import statsmodels.formula.api as smf
except Exception:  # pragma: no cover
    smf = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MERGED_CSV = PROJECT_ROOT / "analysis/qualitative_dynamics_trends_20260628/qualitative_dynamics_merged_run_table.csv"
OUT_DIR = PROJECT_ROOT / "analysis/homogeneous_adversary_redline_elo_20260628"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

BASELINE_ELO = 1337
TARGET = "tag_hard_anchor_or_redline"

GAME_LABELS = {"game1": "Game 1", "game2": "Game 2", "game3": "Game 3"}
COMPETITION_ORDER = ["cooperative", "middle", "competitive"]
N_ORDER = [2, 4, 6, 8, 10]


def clean_label(value: object) -> str:
    text = str(value)
    return GAME_LABELS.get(text, text)


def wilson_interval(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2))) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(MERGED_CSV)
    h = df[df["experiment_family"].eq("homogeneous_adversary")].copy()
    h = h.dropna(subset=["adversary_elo"])
    h["adversary_elo"] = h["adversary_elo"].round().astype(int)
    h["baseline_elo"] = h["baseline_elo"].round().astype("Int64")
    h["target"] = pd.to_numeric(h[TARGET], errors="coerce").fillna(0).astype(int)
    h["game"] = h["game_label"].map(GAME_LABELS).fillna(h["game_label"])
    h["n_label"] = "N=" + h["n_agents"].astype(str)
    h["competition_band"] = pd.Categorical(h["competition_band"], COMPETITION_ORDER, ordered=True)
    h["setting_label"] = h["game"] + ": " + h["setting"].astype(str)
    h["adversary_model_label"] = h["adversary_elo"].astype(str) + "\n" + h["adversary_model_short"].astype(str)
    return h


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    cols = group_cols + ["adversary_elo", "adversary_model_short", "adversary_model_label"]
    table = (
        df.groupby(cols, observed=True, dropna=False)["target"]
        .agg(k="sum", n="size", rate="mean")
        .reset_index()
        .sort_values(group_cols + ["adversary_elo"])
    )
    lows, highs = zip(*[wilson_interval(k, n) for k, n in zip(table["k"], table["n"], strict=True)])
    table["ci_low"] = lows
    table["ci_high"] = highs
    table["percent"] = 100 * table["rate"]
    table["label"] = table["k"].astype(int).astype(str) + "/" + table["n"].astype(int).astype(str)
    return table


def savefig(fig: plt.Figure, filename: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / filename
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def style_axis(ax: plt.Axes, ymax: float = 1.0) -> None:
    ax.axvline(BASELINE_ELO, color="#6b7280", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.set_xlim(1210, 1515)
    ax.set_ylim(0, ymax)
    ax.set_xticks([1240, 1317, 1389, 1448, 1484])
    ax.set_xticklabels(["1240\nNova", "1317\n4o", "1389\nS4", "1448\nG2.5", "1484\nG5.4"])
    ax.yaxis.set_major_formatter(lambda y, _: f"{100*y:.0f}%")
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=8)


def draw_bars(ax: plt.Axes, sub: pd.DataFrame, *, show_ylabel: bool = False, ymax: float = 1.0) -> None:
    sub = sub.sort_values("adversary_elo")
    colors = sns.color_palette("crest", n_colors=len(sub))
    ax.bar(
        sub["adversary_elo"],
        sub["rate"],
        width=30,
        color=colors,
        edgecolor="#1f2937",
        linewidth=0.5,
    )
    yerr = np.vstack([(sub["rate"] - sub["ci_low"]).clip(lower=0), (sub["ci_high"] - sub["rate"]).clip(lower=0)])
    ax.errorbar(sub["adversary_elo"], sub["rate"], yerr=yerr, fmt="none", ecolor="#1f2937", elinewidth=0.8, capsize=2)
    for _, row in sub.iterrows():
        y = min(float(row["rate"]) + 0.035, ymax - 0.025)
        ax.text(
            float(row["adversary_elo"]),
            y,
            str(row["label"]),
            ha="center",
            va="bottom",
            fontsize=6.4,
            color="#111827",
        )
    style_axis(ax, ymax=ymax)
    ax.set_xlabel("Adversary Elo", fontsize=8)
    if show_ylabel:
        ax.set_ylabel("Hard-anchor/redline rate", fontsize=8)
    else:
        ax.set_ylabel("")


def plot_overall(table: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    draw_bars(ax, table, show_ylabel=True, ymax=0.8)
    ax.set_title("Homogeneous adversary: hard-anchor/redline frequency by adversary Elo", fontsize=12, pad=16)
    ax.text(
        0.5,
        1.02,
        "Bar labels are tagged rollouts / total rollouts. Dashed line marks GPT-5-nano baseline Elo=1337.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        color="#374151",
    )
    return savefig(fig, "01_overall_hard_anchor_by_adversary_elo.png")


def plot_single_split(table: pd.DataFrame, split_col: str, filename: str, title: str, order: list[object] | None = None) -> Path:
    if order is None:
        order = list(table[split_col].drop_duplicates())
    order = [x for x in order if x in set(table[split_col].drop_duplicates())]
    ncols = min(3, len(order))
    nrows = math.ceil(len(order) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, value in enumerate(order):
        ax = axes.ravel()[i]
        ax.axis("on")
        sub = table[table[split_col].eq(value)]
        draw_bars(ax, sub, show_ylabel=i % ncols == 0, ymax=1.0)
        ax.set_title(clean_label(value), fontsize=10)
    fig.suptitle(title, fontsize=13, y=1.02)
    fig.text(0.5, -0.01, "Each bar label is tagged rollouts / total rollouts; dashed line is GPT-5-nano baseline Elo=1337.", ha="center", fontsize=8)
    return savefig(fig, filename)


def plot_grid(
    table: pd.DataFrame,
    row_col: str,
    col_col: str,
    filename: str,
    title: str,
    row_order: list[object],
    col_order: list[object],
) -> Path:
    row_order = [x for x in row_order if x in set(table[row_col].drop_duplicates())]
    col_order = [x for x in col_order if x in set(table[col_col].drop_duplicates())]
    fig, axes = plt.subplots(len(row_order), len(col_order), figsize=(4.4 * len(col_order), 3.0 * len(row_order)), squeeze=False)
    for r, row_value in enumerate(row_order):
        for c, col_value in enumerate(col_order):
            ax = axes[r][c]
            sub = table[table[row_col].eq(row_value) & table[col_col].eq(col_value)]
            if len(sub):
                draw_bars(ax, sub, show_ylabel=False, ymax=1.0)
            else:
                ax.axis("off")
            if r == 0:
                ax.set_title(clean_label(col_value), fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{clean_label(row_value)}\nRate", fontsize=8)
    fig.suptitle(title, fontsize=13, y=1.01)
    fig.text(0.5, -0.01, "Each bar label is tagged rollouts / total rollouts; dashed line is GPT-5-nano baseline Elo=1337.", ha="center", fontsize=8)
    return savefig(fig, filename)


def plot_three_way_by_game(table: pd.DataFrame, game: str) -> Path:
    sub = table[table["game"].eq(game)].copy()
    filename = f"08_three_way_{game.lower().replace(' ', '_')}_by_n_and_competition.png"
    return plot_grid(
        sub,
        row_col="n_label",
        col_col="competition_band",
        filename=filename,
        title=f"{game}: hard-anchor/redline by adversary Elo, split by N and competition band",
        row_order=[f"N={n}" for n in N_ORDER],
        col_order=COMPETITION_ORDER,
    )


def regression_lines(df: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    if smf is None:
        return ["Statsmodels was unavailable, so no regression summary was generated."]
    reg = df.copy()
    reg["elo_per_100"] = (reg["adversary_elo"] - BASELINE_ELO) / 100.0
    specs = [
        ("Overall unadjusted", "target ~ elo_per_100"),
        ("Adjusted for game, N, and competition band", "target ~ elo_per_100 + C(game) + C(n_agents) + C(competition_band)"),
    ]
    for label, formula in specs:
        model = smf.ols(formula, data=reg).fit(cov_type="HC3")
        coef = model.params.get("elo_per_100", math.nan)
        lo, hi = model.conf_int().loc["elo_per_100"].tolist()
        p = model.pvalues.get("elo_per_100", math.nan)
        lines.append(f"- {label}: {coef * 100:+.1f} pp per +100 Elo, 95% CI [{lo * 100:+.1f}, {hi * 100:+.1f}], p={p:.3g}.")
    for game in ["Game 1", "Game 2", "Game 3"]:
        sub = reg[reg["game"].eq(game)]
        model = smf.ols("target ~ elo_per_100 + C(n_agents) + C(competition_band)", data=sub).fit(cov_type="HC3")
        coef = model.params.get("elo_per_100", math.nan)
        lo, hi = model.conf_int().loc["elo_per_100"].tolist()
        p = model.pvalues.get("elo_per_100", math.nan)
        lines.append(f"- {game} adjusted within-game: {coef * 100:+.1f} pp per +100 Elo, 95% CI [{lo * 100:+.1f}, {hi * 100:+.1f}], p={p:.3g}.")
    return lines


def write_report(df: pd.DataFrame, figures: dict[str, Path], tables: dict[str, Path]) -> Path:
    overall = summarize(df, [])
    rate_rows = []
    for _, row in overall.iterrows():
        rate_rows.append(
            f"- {int(row['adversary_elo'])} Elo, {row['adversary_model_short']}: "
            f"{row['percent']:.1f}% ({int(row['k'])}/{int(row['n'])})."
        )
    regression = regression_lines(df)
    report = OUT_DIR / "homogeneous_adversary_redline_elo_report.md"
    rel = lambda p: p.relative_to(OUT_DIR)
    lines = [
        "# Homogeneous-Adversary Hard-Anchor/Redline By Elo",
        "",
        "This analysis uses only homogeneous-adversary rollouts from the merged qualitative dynamics table. The outcome is the run-level tag `hard_anchor_or_redline`; the x-axis is the inserted adversary model's Elo. The GPT-5-nano baseline agents have Elo 1337 and are marked as a dashed reference line in the plots.",
        "",
        "Important interpretation caveat: the tag is rollout-level, not speaker-level. So the plots show whether runs with an adversary of a given Elo contain hard-anchor/redline behavior, not whether the adversary alone uttered the redline.",
        "",
        "## Overall Result",
        "",
        *rate_rows,
        "",
        "![Overall](figures/01_overall_hard_anchor_by_adversary_elo.png)",
        "",
        "The descriptive pattern is upward but not perfectly monotonic: rates are lower for the two below-/near-baseline adversaries and higher for Sonnet 4 and GPT-5.4 High, with Gemini 2.5 Pro slightly below Sonnet/GPT-5.4 in the pooled view.",
        "",
        "## Elo Trend Checks",
        "",
        *regression,
        "",
        "These are descriptive linear-probability checks, not causal estimates.",
        "",
        "## Requested Split Plots",
        "",
        "![By N](figures/02_split_by_n.png)",
        "",
        "![By Game](figures/03_split_by_game.png)",
        "",
        "![By competition band](figures/04_split_by_competition_band.png)",
        "",
        "![By game and N](figures/05_split_by_game_and_n.png)",
        "",
        "![By game and competition](figures/06_split_by_game_and_competition_band.png)",
        "",
        "![By N and competition](figures/07_split_by_n_and_competition_band.png)",
        "",
        "## Three-Way Splits",
        "",
        "These split simultaneously by game, N, and competition band. Cell-level denominators are small, so read these as diagnostic plots rather than stable estimates.",
        "",
        "![Game 1 three-way](figures/08_three_way_game_1_by_n_and_competition.png)",
        "",
        "![Game 2 three-way](figures/08_three_way_game_2_by_n_and_competition.png)",
        "",
        "![Game 3 three-way](figures/08_three_way_game_3_by_n_and_competition.png)",
        "",
        "## Source Tables",
        "",
    ]
    for name, path in tables.items():
        lines.append(f"- `{name}`: `{rel(path)}`")
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"Generated by `scripts/retained_analysis/{Path(__file__).name}` from `{MERGED_CSV.relative_to(PROJECT_ROOT)}`.",
        ]
    )
    report.write_text("\n".join(lines) + "\n")
    return report


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    figures: dict[str, Path] = {}
    tables: dict[str, Path] = {}

    summary_specs: list[tuple[str, list[str]]] = [
        ("overall", []),
        ("by_n", ["n_label"]),
        ("by_game", ["game"]),
        ("by_competition_band", ["competition_band"]),
        ("by_game_and_n", ["game", "n_label"]),
        ("by_game_and_competition_band", ["game", "competition_band"]),
        ("by_n_and_competition_band", ["n_label", "competition_band"]),
        ("by_game_n_competition_band", ["game", "n_label", "competition_band"]),
        ("by_game_exact_setting", ["game", "setting"]),
        ("by_game_n_exact_setting", ["game", "n_label", "setting"]),
    ]
    summaries = {}
    for name, cols in summary_specs:
        table = summarize(df, cols)
        path = TABLE_DIR / f"{name}.csv"
        table.to_csv(path, index=False)
        tables[name] = path
        summaries[name] = table

    figures["overall"] = plot_overall(summaries["overall"])
    figures["by_n"] = plot_single_split(summaries["by_n"], "n_label", "02_split_by_n.png", "Hard-anchor/redline by adversary Elo, split by N", [f"N={n}" for n in N_ORDER])
    figures["by_game"] = plot_single_split(summaries["by_game"], "game", "03_split_by_game.png", "Hard-anchor/redline by adversary Elo, split by game", ["Game 1", "Game 2", "Game 3"])
    figures["by_competition"] = plot_single_split(
        summaries["by_competition_band"],
        "competition_band",
        "04_split_by_competition_band.png",
        "Hard-anchor/redline by adversary Elo, split by competition band",
        COMPETITION_ORDER,
    )
    figures["by_game_n"] = plot_grid(
        summaries["by_game_and_n"],
        row_col="game",
        col_col="n_label",
        filename="05_split_by_game_and_n.png",
        title="Hard-anchor/redline by adversary Elo, split by game and N",
        row_order=["Game 1", "Game 2", "Game 3"],
        col_order=[f"N={n}" for n in N_ORDER],
    )
    figures["by_game_competition"] = plot_grid(
        summaries["by_game_and_competition_band"],
        row_col="game",
        col_col="competition_band",
        filename="06_split_by_game_and_competition_band.png",
        title="Hard-anchor/redline by adversary Elo, split by game and competition band",
        row_order=["Game 1", "Game 2", "Game 3"],
        col_order=COMPETITION_ORDER,
    )
    figures["by_n_competition"] = plot_grid(
        summaries["by_n_and_competition_band"],
        row_col="n_label",
        col_col="competition_band",
        filename="07_split_by_n_and_competition_band.png",
        title="Hard-anchor/redline by adversary Elo, split by N and competition band",
        row_order=[f"N={n}" for n in N_ORDER],
        col_order=COMPETITION_ORDER,
    )
    for game in ["Game 1", "Game 2", "Game 3"]:
        figures[f"three_way_{game}"] = plot_three_way_by_game(summaries["by_game_n_competition_band"], game)

    report = write_report(df, figures, tables)
    print(f"rows={len(df)} figures={len(figures)} report={report}")


if __name__ == "__main__":
    main()

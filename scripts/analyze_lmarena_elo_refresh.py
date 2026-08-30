#!/usr/bin/env python3
"""Compare bilateral payoff trends under two LM Arena Elo snapshots."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
import numpy as np
import pandas as pd
import requests
from matplotlib.lines import Line2D
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = (
    ROOT
    / "experiments/results/n2_baseline_comparison_analysis_20260505/primary_runs_with_metrics.csv"
)
MARCH31_GUIDE = ROOT / "docs/guides/chatbot_arena_elo_scores_2026_03_31.md"
AUGUST12_GUIDE = ROOT / "docs/guides/chatbot_arena_elo_scores_2026_08_12.md"
OUT_DIR = ROOT / "analysis/lmarena_elo_refresh_20260816"
PLOT_DIR = OUT_DIR / "plots"
SOURCE_URL = "https://lmarena.ai/leaderboard/text"
LEADERBOARD_ID = (
    "leaderboard-sets/public/leaderboards/"
    "text-overall-style_control/leaderboard-snapshots/latest"
)
SNAPSHOT_DATE_LABEL = "August 2026"

GAME_ORDER = ["game1", "game2", "game3"]
GAME_LABELS = {
    "game1": "Game 1: Item allocation",
    "game2": "Game 2: Diplomatic Treaty",
    "game3": "Game 3: Co-funding",
}
GAME_COLORS = {"game1": "#2b7bba", "game2": "#e63946", "game3": "#2ca02c"}

EXPECTED_COUNTS = {"game1": 420, "game2": 540, "game3": 540}
EXPECTED_ROSTER_SIZE = 30
EXPECTED_FULL_ROWS = 391

# Explicit identities for the leaderboard rows. These are not fuzzy matches.
AUGUST12_MODEL_KEYS = {
    "claude-opus-4-6-thinking": "claude-opus-4-6-thinking",
    "claude-opus-4-6": "claude-opus-4-6",
    "gemini-3.1-pro": "gemini-3.1-pro-preview",
    "gpt-5.4-high": "gpt-5.4-high-no-system-prompt",
    "gpt-5.2-chat-latest-20260210": "gpt-5.2-chat-latest",
    "claude-opus-4-5-20251101-thinking-32k": (
        "claude-opus-4-5-20251101-thinking-32k"
    ),
    "claude-opus-4-5-20251101": "claude-opus-4-5-20251101",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "qwen3-max-preview": "qwen3-max-preview",
    "deepseek-r1-0528": "deepseek-r1-0528",
    "claude-haiku-4-5-20251001": "claude-haiku-4-5-20251001",
    "deepseek-r1": "deepseek-r1",
    "claude-sonnet-4-20250514": "claude-sonnet-4-20250514",
    "gemma-3-27b-it": "gemma-3-27b-it",
    "o3-mini-high": "o3-mini-high",
    "deepseek-v3": "deepseek-v3",
    "gpt-4o-2024-05-13": "gpt-4o-2024-05-13",
    "gpt-5-nano-high": "gpt-5-nano-high",
    "qwq-32b": "qwq-32b",
    "gpt-4.1-nano-2025-04-14": "gpt-4.1-nano-2025-04-14",
    "llama-3.3-70b-instruct": "llama-3.3-70b-instruct",
    "gpt-4o-mini-2024-07-18": "gpt-4o-mini-2024-07-18",
    "qwen2.5-72b-instruct": "qwen2.5-72b-instruct",
    "amazon-nova-pro-v1.0": "amazon-nova-pro-v1.0",
    "command-r-plus-08-2024": "command-r-plus-08-2024",
    "claude-3-haiku-20240307": "claude-3-haiku-20240307",
    "amazon-nova-micro-v1.0": "amazon-nova-micro-v1.0",
    "llama-3.1-8b-instruct": "llama-3.1-8b-instruct",
    "llama-3.2-3b-instruct": "llama-3.2-3b-instruct",
    "llama-3.2-1b-instruct": "llama-3.2-1b-instruct",
}

# Integer scores copied from the user-supplied leaderboard. The script checks
# each value against the integer shown by the official page before plotting it.
AUGUST12_DISPLAYED_SCORES = {
    "claude-opus-4-6-thinking": 1505,
    "claude-opus-4-6": 1497,
    "gemini-3.1-pro": 1486,
    "gpt-5.4-high": 1476,
    "gpt-5.2-chat-latest-20260210": 1476,
    "claude-opus-4-5-20251101-thinking-32k": 1473,
    "claude-opus-4-5-20251101": 1469,
    "gemini-2.5-pro": 1445,
    "qwen3-max-preview": 1434,
    "deepseek-r1-0528": 1422,
    "claude-haiku-4-5-20251001": 1413,
    "deepseek-r1": 1398,
    "claude-sonnet-4-20250514": 1389,
    "gemma-3-27b-it": 1366,
    "o3-mini-high": 1364,
    "deepseek-v3": 1359,
    "gpt-4o-2024-05-13": 1346,
    "gpt-5-nano-high": 1337,
    "qwq-32b": 1336,
    "gpt-4.1-nano-2025-04-14": 1322,
    "llama-3.3-70b-instruct": 1318,
    "gpt-4o-mini-2024-07-18": 1318,
    "qwen2.5-72b-instruct": 1303,
    "amazon-nova-pro-v1.0": 1290,
    "command-r-plus-08-2024": 1276,
    "claude-3-haiku-20240307": 1261,
    "amazon-nova-micro-v1.0": 1241,
    "llama-3.1-8b-instruct": 1211,
    "llama-3.2-3b-instruct": 1166,
    "llama-3.2-1b-instruct": 1111,
}


def displayed_integer(value: float) -> int:
    """Match the positive integer rounding shown on the leaderboard."""
    return int(np.floor(float(value) + 0.5))


def sem(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.std(ddof=1) / np.sqrt(len(clean))) if len(clean) > 1 else 0.0


def fetch_full_rows() -> tuple[pd.DataFrame, dict[str, object]]:
    response = requests.get(
        SOURCE_URL,
        headers={"User-Agent": "Mozilla/5.0 (research snapshot; contact via paper repo)"},
        timeout=60,
    )
    response.raise_for_status()
    text = response.text
    id_pos = text.find(LEADERBOARD_ID)
    if id_pos < 0:
        raise RuntimeError(f"Could not find official leaderboard ID: {LEADERBOARD_ID}")
    entries_pos = text.find(r'\"entries\":', id_pos)
    start = text.find("[", entries_pos)
    end = text.find("]", start) + 1
    if entries_pos < 0 or start < 0 or end <= start:
        raise RuntimeError("Could not isolate the embedded leaderboard entries")
    rows = json.loads(text[start:end].replace(r'\"', '"'))
    live = pd.DataFrame(rows)
    required = {
        "rank",
        "rankUpper",
        "rankLower",
        "modelKey",
        "modelDisplayName",
        "rating",
        "ratingLower",
        "ratingUpper",
        "votes",
        "modelOrganization",
        "license",
    }
    missing = required - set(live.columns)
    if missing:
        raise RuntimeError(f"Leaderboard lacks columns: {sorted(missing)}")
    if len(live) != EXPECTED_FULL_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_FULL_ROWS} full leaderboard rows, found {len(live)}"
        )
    if live["modelKey"].duplicated().any():
        raise RuntimeError("The full text leaderboard contains duplicate modelKey values")
    metadata = {
        "source_url": SOURCE_URL,
        "leaderboard_id": LEADERBOARD_ID,
        "fetched_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
        "response_sha256": hashlib.sha256(response.content).hexdigest(),
        "response_bytes": len(response.content),
        "full_leaderboard_rows": len(live),
        "ranked_rows": int(live["rank"].gt(0).sum()),
        "autoeval_rows": int(live["rank"].eq(0).sum()),
        "method": (
            "Parsed the full embedded Text Overall leaderboard with Style Control "
            "enabled from the official LM Arena text leaderboard page."
        ),
    }
    return live, metadata


def load_bilateral() -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = pd.read_csv(INPUT_CSV)
    runs = runs[runs["baseline_key"].eq("gpt5_nano")].copy()
    counts = runs.groupby("game_id").size().to_dict()
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"Expected game counts {EXPECTED_COUNTS}, found {counts}")
    roster = (
        runs[["adversary_model", "adversary_short", "adversary_elo"]]
        .drop_duplicates()
        .sort_values("adversary_elo", ascending=False)
    )
    if len(roster) != EXPECTED_ROSTER_SIZE:
        raise RuntimeError(f"Expected {EXPECTED_ROSTER_SIZE} bilateral models, found {len(roster)}")
    if set(roster["adversary_model"]) != set(AUGUST12_MODEL_KEYS):
        missing = set(roster["adversary_model"]) - set(AUGUST12_MODEL_KEYS)
        extra = set(AUGUST12_MODEL_KEYS) - set(roster["adversary_model"])
        raise RuntimeError(f"Roster mapping mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    means = (
        runs.groupby(
            ["game_id", "adversary_model", "adversary_short", "adversary_elo"],
            as_index=False,
        )
        .agg(
            payoff=("adversary_utility", "mean"),
            payoff_sem=("adversary_utility", sem),
            runs=("adversary_utility", "size"),
        )
    )
    return roster, means


def build_rating_comparison(roster: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    live_by_key = live.set_index("modelKey", verify_integrity=True)
    rows: list[dict[str, object]] = []
    for old in roster.itertuples(index=False):
        key = AUGUST12_MODEL_KEYS[old.adversary_model]
        if key not in live_by_key.index:
            raise RuntimeError(f"Declared leaderboard modelKey is absent: {key}")
        current = live_by_key.loc[key]
        raw_rating = float(current["rating"])
        displayed_score = displayed_integer(raw_rating)
        supplied_score = AUGUST12_DISPLAYED_SCORES[old.adversary_model]
        if displayed_score != supplied_score:
            raise RuntimeError(
                f"Displayed score mismatch for {old.adversary_model}: "
                f"user supplied {supplied_score}, official page shows {displayed_score}"
            )
        rows.append(
            {
                "adversary_model": old.adversary_model,
                "adversary_short": old.adversary_short,
                "march31_score": int(old.adversary_elo),
                "august12_model_key": key,
                "august12_display_name": current["modelDisplayName"],
                "august12_score": supplied_score,
                "official_raw_rating_at_capture": raw_rating,
                "official_rating_lower_at_capture": float(current["ratingLower"]),
                "official_rating_upper_at_capture": float(current["ratingUpper"]),
                "official_votes_at_capture": int(current["votes"]),
                "score_change": supplied_score - int(old.adversary_elo),
                "match_status": "declared row match; displayed score verified",
            }
        )
    comparison = pd.DataFrame(rows)
    if len(comparison) != EXPECTED_ROSTER_SIZE:
        raise RuntimeError("The rating comparison does not contain all 30 models")
    return comparison.sort_values("march31_score", ascending=False)


def fit_summary(x: pd.Series, y: pd.Series) -> dict[str, float]:
    fit = stats.linregress(x.to_numpy(dtype=float), y.to_numpy(dtype=float))
    critical = stats.t.ppf(0.975, len(x) - 2)
    rho, rho_p = stats.spearmanr(x, y)
    return {
        "slope_per_100_elo": float(fit.slope * 100),
        "ci_low_per_100_elo": float((fit.slope - critical * fit.stderr) * 100),
        "ci_high_per_100_elo": float((fit.slope + critical * fit.stderr) * 100),
        "p_value": float(fit.pvalue),
        "r_squared": float(fit.rvalue**2),
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "intercept": float(fit.intercept),
    }


def calculate_trends(means: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    plot_data = means.merge(
        comparison[["adversary_model", "march31_score", "august12_score"]],
        on="adversary_model",
        how="inner",
        validate="many_to_one",
    )
    rows: list[dict[str, object]] = []
    cases = {
        "march31_all_30": ("march31_score", "March 31, 2026"),
        "august12_all_30": ("august12_score", SNAPSHOT_DATE_LABEL),
    }
    for game_id in GAME_ORDER:
        game = plot_data[plot_data["game_id"].eq(game_id)]
        for analysis_id, (score_column, label) in cases.items():
            values = fit_summary(game[score_column], game["payoff"])
            rows.append(
                {
                    "analysis_id": analysis_id,
                    "analysis_label": label,
                    "game_id": game_id,
                    "game_label": GAME_LABELS[game_id],
                    "models": len(game),
                    "elo_min": int(game[score_column].min()),
                    "elo_max": int(game[score_column].max()),
                    **values,
                }
            )
    return pd.DataFrame(rows)


def plot_side_by_side(means: pd.DataFrame, comparison: pd.DataFrame) -> None:
    data = means.merge(
        comparison[["adversary_model", "march31_score", "august12_score"]],
        on="adversary_model",
        how="inner",
        validate="many_to_one",
    )
    panels = [
        ("march31_score", "March 31, 2026 Elo (n=30)"),
        ("august12_score", f"{SNAPSHOT_DATE_LABEL} Elo (n=30)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.15), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    for ax, (score_column, title) in zip(axes, panels, strict=True):
        for game_id in GAME_ORDER:
            game = data[data["game_id"].eq(game_id)].sort_values(score_column)
            color = GAME_COLORS[game_id]
            ax.errorbar(
                game[score_column],
                game["payoff"],
                yerr=game["payoff_sem"],
                fmt="o",
                markersize=6.0,
                markerfacecolor="white",
                markeredgewidth=1.25,
                color=color,
                ecolor=color,
                elinewidth=0.9,
                capsize=2.2,
                alpha=0.75,
                zorder=3,
            )
            fit = stats.linregress(game[score_column], game["payoff"])
            xs = np.linspace(game[score_column].min(), game[score_column].max(), 200)
            ax.plot(
                xs,
                fit.intercept + fit.slope * xs,
                "--",
                color=color,
                linewidth=3.0,
                zorder=4,
            )
        ax.set_title(title, fontsize=16.5, pad=10)
        ax.set_xlabel("Adversary Elo", fontsize=16)
        ax.set_xlim(1090, 1520)
        ax.set_ylim(-10, 102)
        ax.grid(True, color="#d1d5db", alpha=0.55, linewidth=0.75)
        ax.tick_params(axis="both", labelsize=12)
    axes[0].set_ylabel("Adversary payoff", fontsize=17)
    handles = [
        Line2D(
            [0],
            [0],
            color=GAME_COLORS[game_id],
            linestyle="--",
            linewidth=3.0,
            label=GAME_LABELS[game_id],
        )
        for game_id in GAME_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=13.5,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle(
        "Bilateral payoff under two LM Arena Elo snapshots",
        fontsize=19,
        y=0.995,
    )
    fig.subplots_adjust(left=0.085, right=0.99, top=0.88, bottom=0.19, wspace=0.16)
    for suffix in ["png", "pdf"]:
        fig.savefig(
            PLOT_DIR / f"bilateral_payoff_march31_vs_august12_lmarena_elo.{suffix}",
            dpi=300 if suffix == "png" else None,
            facecolor="white",
        )
    plt.close(fig)


def plot_rating_comparison(comparison: pd.DataFrame) -> None:
    data = comparison.copy()
    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    lo = min(data["march31_score"].min(), data["august12_score"].min()) - 8
    hi = max(data["march31_score"].max(), data["august12_score"].max()) + 8
    ax.plot([lo, hi], [lo, hi], color="#6b7280", linestyle="--", linewidth=1.4)
    ax.scatter(
        data["march31_score"],
        data["august12_score"],
        s=44,
        facecolor="white",
        edgecolor="#2b7bba",
        linewidth=1.2,
    )
    labels = data[data["score_change"].abs().ge(2)]
    for row in labels.itertuples(index=False):
        ax.annotate(
            row.adversary_short,
            (row.march31_score, row.august12_score),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7.3,
            alpha=0.85,
        )
    pearson = stats.pearsonr(data["march31_score"], data["august12_score"])
    spearman = stats.spearmanr(data["march31_score"], data["august12_score"])
    ax.text(
        0.03,
        0.97,
        (
            f"n={len(data)} of 30\n"
            f"Pearson r={pearson.statistic:.3f}\n"
            f"Spearman rho={spearman.statistic:.3f}\n"
            f"Mean absolute change={data['score_change'].abs().mean():.2f}"
        ),
        transform=ax.transAxes,
        va="top",
        fontsize=9.5,
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.92},
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("LM Arena Elo on March 31, 2026", fontsize=11)
    ax.set_ylabel(f"LM Arena Elo, {SNAPSHOT_DATE_LABEL}", fontsize=11)
    ax.set_title("LM Arena Elo across the two snapshots", fontsize=13)
    ax.grid(True, color="#d1d5db", alpha=0.5, linewidth=0.7)
    for suffix in ["png", "pdf"]:
        fig.savefig(
            PLOT_DIR / f"march31_vs_august12_lmarena_elo.{suffix}",
            dpi=300 if suffix == "png" else None,
        )
    plt.close(fig)


def compact_number(value: object) -> str:
    if pd.isna(value):
        return "N/A"
    number = float(value)
    if number >= 1_000_000:
        text = f"{number / 1_000_000:.1f}".rstrip("0").rstrip(".")
        return f"{text}M"
    if number >= 1_000:
        text = f"{number / 1_000:.1f}".rstrip("0").rstrip(".")
        return f"{text}K"
    return f"{number:g}"


def price_text(input_price: object, output_price: object) -> str:
    if pd.isna(input_price) or pd.isna(output_price):
        return "N/A"
    return f"${float(input_price):g} / ${float(output_price):g}"


def write_full_guide(live: pd.DataFrame, metadata: dict[str, object]) -> None:
    ranked = live[live["rank"].gt(0)].sort_values(["rank", "modelDisplayName"])
    autoeval = live[live["rank"].eq(0)].sort_values("modelDisplayName")
    ordered = pd.concat([ranked, autoeval], ignore_index=True)
    lines = [
        "# Chatbot Arena Elo Scores",
        "",
        f"**Source:** [LM Arena Text Leaderboard]({SOURCE_URL})",
        f"**Snapshot date label supplied by user:** {SNAPSHOT_DATE_LABEL}",
        f"**Captured from official page:** {metadata['fetched_at']}",
        f"**Rows:** {metadata['full_leaderboard_rows']} ({metadata['ranked_rows']} ranked and {metadata['autoeval_rows']} AutoEval)",
        "**Leaderboard:** Text, Overall, Style Control enabled",
        "",
        "The pasted list contains `deepseek-v4-pro-max-20260813`, so its contents cannot have been finalized on August 12. The August 12 date is retained as the user-supplied comparison label.",
        "Scores and confidence intervals are rounded to the integers displayed by LM Arena.",
        "",
        "## Full rankings",
        "",
        "| Rank | Rank spread | Model | Score | 95% CI (±) | Votes | Price $/M | Context | Organization | License |",
        "|---:|:---:|---|---:|:---:|---:|---|---:|---|---|",
    ]
    for row in ordered.itertuples(index=False):
        is_autoeval = int(row.rank) == 0
        rank = "N/A" if is_autoeval else str(int(row.rank))
        spread = (
            "N/A"
            if is_autoeval
            else f"{int(row.rankUpper)} ◄─► {int(row.rankLower)}"
        )
        score = displayed_integer(row.rating)
        ci = displayed_integer(
            max(float(row.ratingUpper) - float(row.rating), float(row.rating) - float(row.ratingLower))
        )
        preliminary = " (Preliminary)" if row.releaseType == "pre_release" else ""
        votes = "AutoEval" if is_autoeval else f"{int(row.votes):,}"
        organization = row.modelOrganization if row.modelOrganization else "N/A"
        license_name = row.license if row.license else "N/A"
        lines.append(
            "| "
            + " | ".join(
                [
                    rank,
                    spread,
                    str(row.modelDisplayName),
                    str(score),
                    f"±{ci}{preliminary}",
                    votes,
                    price_text(row.inputPricePerMillion, row.outputPricePerMillion),
                    compact_number(row.contextLength),
                    str(organization),
                    str(license_name),
                ]
            )
            + " |"
        )
    lines.append("")
    AUGUST12_GUIDE.write_text("\n".join(lines), encoding="utf-8")


def write_summary(
    comparison: pd.DataFrame, trends: pd.DataFrame, metadata: dict[str, object]
) -> None:
    pearson = stats.pearsonr(comparison["march31_score"], comparison["august12_score"])
    spearman = stats.spearmanr(comparison["march31_score"], comparison["august12_score"])
    lines = [
        "# LM Arena Elo snapshot comparison",
        "",
        f"- Official source: {SOURCE_URL}",
        f"- User-supplied comparison label: {SNAPSHOT_DATE_LABEL}",
        f"- Captured: {metadata['fetched_at']}",
        f"- Full leaderboard rows: {metadata['full_leaderboard_rows']}",
        f"- Bilateral models matched: {len(comparison)} of {EXPECTED_ROSTER_SIZE}",
        "- Leaderboard: Text, Overall, Style Control enabled",
        "",
        "## Score similarity",
        "",
        f"- Pearson r: {pearson.statistic:.4f}",
        f"- Spearman rho: {spearman.statistic:.4f}",
        f"- Mean absolute score change: {comparison['score_change'].abs().mean():.2f} Elo",
        f"- Largest absolute score change: {comparison['score_change'].abs().max():.0f} Elo",
        "",
        "## Payoff trends",
        "",
    ]
    for game_id in GAME_ORDER:
        rows = trends[trends["game_id"].eq(game_id)].set_index("analysis_id")
        old = rows.loc["march31_all_30"]
        new = rows.loc["august12_all_30"]
        lines.extend(
            [
                f"### {GAME_LABELS[game_id]}",
                "",
                (
                    f"- March 31 slope: {old['slope_per_100_elo']:.2f} "
                    f"[{old['ci_low_per_100_elo']:.2f}, {old['ci_high_per_100_elo']:.2f}], "
                    f"p={old['p_value']:.3g}, R²={old['r_squared']:.3f}."
                ),
                (
                    f"- August 12-labeled slope: {new['slope_per_100_elo']:.2f} "
                    f"[{new['ci_low_per_100_elo']:.2f}, {new['ci_high_per_100_elo']:.2f}], "
                    f"p={new['p_value']:.3g}, R²={new['r_squared']:.3f}."
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Date check",
            "",
            (
                "The list includes `deepseek-v4-pro-max-20260813`. The contents therefore "
                "cannot have been finalized on August 12, even though August 12 is the "
                "comparison label supplied with the list."
            ),
            "",
            "## Interpretation limit",
            "",
            (
                "LM Arena recomputes ratings as votes and the comparison pool change. "
                "A score change does not measure improvement in the model itself."
            ),
            "",
        ]
    )
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    roster, means = load_bilateral()
    live, metadata = fetch_full_rows()
    comparison = build_rating_comparison(roster, live)
    trends = calculate_trends(means, comparison)

    write_full_guide(live, metadata)
    live.to_csv(
        OUT_DIR / "official_august12_labeled_text_overall_style_control.csv",
        index=False,
    )
    comparison.to_csv(
        OUT_DIR / "paper_roster_march31_vs_august12.csv", index=False
    )
    trends.to_csv(OUT_DIR / "trend_comparison.csv", index=False)
    score_correlation = {
        "pearson_r": float(
            stats.pearsonr(comparison["march31_score"], comparison["august12_score"]).statistic
        ),
        "spearman_rho": float(
            stats.spearmanr(comparison["march31_score"], comparison["august12_score"]).statistic
        ),
        "mean_absolute_change": float(comparison["score_change"].abs().mean()),
        "maximum_absolute_change": int(comparison["score_change"].abs().max()),
    }
    metadata.update(
        {
            "snapshot_date_label": SNAPSHOT_DATE_LABEL,
            "snapshot_date_label_source": "user supplied",
            "date_consistency_warning": (
                "The supplied list includes deepseek-v4-pro-max-20260813, so it cannot "
                "have been finalized on August 12."
            ),
            "input_csv": str(INPUT_CSV.relative_to(ROOT)),
            "input_csv_sha256": hashlib.sha256(INPUT_CSV.read_bytes()).hexdigest(),
            "march31_guide": str(MARCH31_GUIDE.relative_to(ROOT)),
            "march31_guide_sha256": hashlib.sha256(MARCH31_GUIDE.read_bytes()).hexdigest(),
            "august12_guide": str(AUGUST12_GUIDE.relative_to(ROOT)),
            "august12_guide_sha256": hashlib.sha256(AUGUST12_GUIDE.read_bytes()).hexdigest(),
            "paper_models": len(comparison),
            "paper_models_matched": len(comparison),
            "displayed_scores_verified": len(comparison),
            "model_key_map": AUGUST12_MODEL_KEYS,
            "user_supplied_displayed_scores": AUGUST12_DISPLAYED_SCORES,
            "score_correlation": score_correlation,
        }
    )
    (OUT_DIR / "provenance.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    plot_side_by_side(means, comparison)
    plot_rating_comparison(comparison)
    write_summary(comparison, trends, metadata)
    print(f"Matched {len(comparison)} of {EXPECTED_ROSTER_SIZE} bilateral models")
    print(trends.to_string(index=False))
    print(f"Wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()

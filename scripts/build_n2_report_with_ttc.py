#!/usr/bin/env python3
"""Create an augmented N=2 report with a TTC scaling section.

This script intentionally leaves the original N=2 report untouched and writes
an additional markdown report plus a compact TTC plot bundle.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parent.parent
N2_DIR = PROJECT_ROOT / "experiments" / "results" / "n2_baseline_comparison_analysis_20260505"
BASE_REPORT = N2_DIR / "n2_baseline_comparison_report.md"
NEW_REPORT = N2_DIR / "n2_baseline_comparison_with_ttc_report.md"
TTC_ROOT = PROJECT_ROOT / "experiments" / "results" / "ttc_native_scaling_20260502_212943"
TTC_RESULTS = TTC_ROOT / "monitoring" / "partial_results_latest.csv"
TTC_OUT = N2_DIR / "ttc_scaling"

GAME_ORDER = ["game1", "game2", "game3"]
GAME_LABELS = {
    "game1": "Game 1: item allocation",
    "game2": "Game 2: diplomacy",
    "game3": "Game 3: cofunding",
}
FAMILY_ORDER = ["gpt-5", "claude-sonnet-4-6", "gemini-3-flash"]
FAMILY_LABELS = {
    "gpt-5": "GPT-5",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gemini-3-flash": "Gemini 3 Flash",
}
FAMILY_COLORS = {
    "gpt-5": "#2563eb",
    "claude-sonnet-4-6": "#dc2626",
    "gemini-3-flash": "#059669",
}
FAMILY_MARKERS = {
    "gpt-5": "o",
    "claude-sonnet-4-6": "s",
    "gemini-3-flash": "^",
}
COMP_STYLES = ["-", "--", ":"]
ORDER_STYLES = {"target_first": "-", "baseline_first": "--"}
ORDER_LABELS = {"target_first": "Target first", "baseline_first": "Baseline first"}


def fmt(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}".rstrip("0").rstrip(".")


def rel(path: Path) -> str:
    return str(path.relative_to(N2_DIR))


def load_ttc() -> pd.DataFrame:
    df = pd.read_csv(TTC_RESULTS)
    numeric_cols = [
        "config_id",
        "level_index",
        "target_utility",
        "baseline_utility",
        "utility_gap",
        "target_compute_tokens_per_call",
        "target_output_tokens_per_call",
        "target_llm_call_count",
        "round",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    def competition_value(row: pd.Series) -> float:
        cell = str(row["game_cell"])
        if row["game"] == "game1":
            return float(cell.split("_comp_")[1].replace("p", "."))
        if row["game"] == "game2":
            if "rho_1" in cell:
                rho = 1.0
            elif "rho_0" in cell:
                rho = 0.0
            else:
                rho = -1.0
            theta = 1.0
            return theta * (1.0 - rho) / 2.0
        if row["game"] == "game3":
            if "alpha_1p0" in cell:
                alpha = 1.0
            elif "alpha_0p5" in cell:
                alpha = 0.5
            else:
                alpha = 0.0
            if "sigma_1p0" in cell:
                sigma = 1.0
            elif "sigma_0p6" in cell:
                sigma = 0.6
            else:
                sigma = 0.2
            return (1.0 - alpha) * (1.0 - sigma)
        return np.nan

    df["competition_value"] = df.apply(competition_value, axis=1)
    df["competition_label"] = df.apply(
        lambda row: f"c={fmt(row['competition_value'])}"
        if row["game"] == "game1"
        else f"CI={fmt(row['competition_value'])}",
        axis=1,
    )
    df["game_label_pretty"] = df["game"].map(GAME_LABELS).fillna(df["game"])
    df["family_label"] = df["family"].map(FAMILY_LABELS).fillna(df["family"])
    df["order_label"] = df["order"].map(ORDER_LABELS).fillna(df["order"])
    df["consensus_float"] = df["consensus"].astype(str).str.lower().eq("true").astype(float)
    return df


def aggregate(df: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    return (
        df.groupby(groups + ["family", "family_label", "level_index", "level"], as_index=False)
        .agg(
            n=("config_id", "count"),
            target_utility=("target_utility", "mean"),
            baseline_utility=("baseline_utility", "mean"),
            utility_gap=("utility_gap", "mean"),
            compute_tokens_per_call=("target_compute_tokens_per_call", "mean"),
            output_tokens_per_call=("target_output_tokens_per_call", "mean"),
            consensus_rate=("consensus_float", "mean"),
            mean_round=("round", "mean"),
        )
        .sort_values(groups + ["family", "level_index"])
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(alpha=0.22)
    ax.tick_params(labelsize=8)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("Requested reasoning effort index", fontsize=9)


def annotate_levels(ax: plt.Axes, rows: pd.DataFrame, y_col: str) -> None:
    for _, row in rows.iterrows():
        ax.annotate(
            str(row["level"]),
            (row["level_index"], row[y_col]),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6,
            alpha=0.72,
        )


def plot_payoff_by_effort(df: pd.DataFrame) -> Path:
    out = TTC_OUT / "ttc_payoff_by_effort.png"
    agg = aggregate(df, ["game"])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.1), sharey=True)
    for ax, game in zip(axes, GAME_ORDER, strict=True):
        subset = agg[agg["game"].eq(game)]
        for family in FAMILY_ORDER:
            rows = subset[subset["family"].eq(family)].sort_values("level_index")
            if rows.empty:
                continue
            color = FAMILY_COLORS[family]
            marker = FAMILY_MARKERS[family]
            ax.plot(rows["level_index"], rows["target_utility"], color=color, marker=marker, linewidth=2.0)
            ax.plot(rows["level_index"], rows["baseline_utility"], color=color, marker=marker, linewidth=1.5, linestyle="--", alpha=0.78)
            annotate_levels(ax, rows, "target_utility")
        ax.set_title(GAME_LABELS[game], fontsize=10)
        ax.set_ylabel("Mean utility" if game == "game1" else "", fontsize=9)
        style_axis(ax)
    family_handles = [
        Line2D([0], [0], color=FAMILY_COLORS[f], marker=FAMILY_MARKERS[f], label=FAMILY_LABELS[f], linewidth=2)
        for f in FAMILY_ORDER
    ]
    role_handles = [
        Line2D([0], [0], color="#111827", linewidth=2, label="Target/adversary"),
        Line2D([0], [0], color="#111827", linewidth=2, linestyle="--", label="Baseline"),
    ]
    fig.legend(handles=family_handles + role_handles, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle("TTC scaling: target and baseline payoff by requested effort", fontsize=13, y=1.02)
    fig.tight_layout(rect=[0, 0.07, 1, 0.98])
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_tokens_by_effort(df: pd.DataFrame) -> Path:
    out = TTC_OUT / "ttc_tokens_by_effort.png"
    agg = aggregate(df, ["game"])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.1), sharey=False)
    for ax, game in zip(axes, GAME_ORDER, strict=True):
        subset = agg[agg["game"].eq(game)]
        for family in FAMILY_ORDER:
            rows = subset[subset["family"].eq(family)].sort_values("level_index")
            if rows.empty:
                continue
            ax.plot(
                rows["level_index"],
                rows["compute_tokens_per_call"],
                color=FAMILY_COLORS[family],
                marker=FAMILY_MARKERS[family],
                linewidth=2.0,
                label=FAMILY_LABELS[family],
            )
            annotate_levels(ax, rows, "compute_tokens_per_call")
        ax.set_title(GAME_LABELS[game], fontsize=10)
        ax.set_ylabel("Effective target compute/proxy tokens per call" if game == "game1" else "", fontsize=9)
        style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("TTC scaling: realized target compute/proxy tokens by requested effort", fontsize=13, y=1.02)
    fig.tight_layout(rect=[0, 0.07, 1, 0.98])
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_by_competition(df: pd.DataFrame, metric: str, ylabel: str, filename: str, title: str) -> Path:
    out = TTC_OUT / filename
    agg = aggregate(df, ["game", "competition_value", "competition_label"])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), sharey=False)
    for ax, game in zip(axes, GAME_ORDER, strict=True):
        subset = agg[agg["game"].eq(game)].copy()
        comp_values = sorted(subset["competition_value"].dropna().unique())
        for family in FAMILY_ORDER:
            fam_rows = subset[subset["family"].eq(family)]
            for idx, comp in enumerate(comp_values):
                rows = fam_rows[fam_rows["competition_value"].eq(comp)].sort_values("level_index")
                if rows.empty:
                    continue
                label = f"{FAMILY_LABELS[family]} {rows['competition_label'].iloc[0]}"
                ax.plot(
                    rows["level_index"],
                    rows[metric],
                    color=FAMILY_COLORS[family],
                    linestyle=COMP_STYLES[idx % len(COMP_STYLES)],
                    marker=FAMILY_MARKERS[family],
                    linewidth=1.5,
                    markersize=4,
                    alpha=0.9,
                    label=label,
                )
        ax.set_title(GAME_LABELS[game], fontsize=10)
        ax.set_ylabel(ylabel if game == "game1" else "", fontsize=9)
        style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=6.5, frameon=False)
    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout(rect=[0, 0.16, 1, 0.98])
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_target_by_order(df: pd.DataFrame) -> Path:
    out = TTC_OUT / "ttc_target_payoff_by_order.png"
    agg = aggregate(df, ["game", "order", "order_label"])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.1), sharey=False)
    for ax, game in zip(axes, GAME_ORDER, strict=True):
        subset = agg[agg["game"].eq(game)]
        for family in FAMILY_ORDER:
            fam_rows = subset[subset["family"].eq(family)]
            for order, linestyle in ORDER_STYLES.items():
                rows = fam_rows[fam_rows["order"].eq(order)].sort_values("level_index")
                if rows.empty:
                    continue
                ax.plot(
                    rows["level_index"],
                    rows["target_utility"],
                    color=FAMILY_COLORS[family],
                    linestyle=linestyle,
                    marker=FAMILY_MARKERS[family],
                    linewidth=1.7,
                    label=f"{FAMILY_LABELS[family]} {ORDER_LABELS[order]}",
                )
        ax.set_title(GAME_LABELS[game], fontsize=10)
        ax.set_ylabel("Mean target/adversary utility" if game == "game1" else "", fontsize=9)
        style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7, frameon=False)
    fig.suptitle("TTC scaling: target payoff by model order", fontsize=13, y=1.02)
    fig.tight_layout(rect=[0, 0.14, 1, 0.98])
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str]]) -> list[str]:
    lines = [
        "| " + " | ".join(header for _, header in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        vals = []
        for col, _ in columns:
            value = row[col]
            if isinstance(value, float):
                vals.append(fmt(value))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def weak_strong_delta_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family in FAMILY_ORDER:
        sub = df[df["family"].eq(family)]
        weak_level = sub.sort_values("level_index")["level"].iloc[0]
        strong_level = sub.sort_values("level_index")["level"].iloc[-1]
        weak = sub[sub["level"].eq(weak_level)].set_index(["game_cell", "order"])
        strong = sub[sub["level"].eq(strong_level)].set_index(["game_cell", "order"])
        common = weak.index.intersection(strong.index)
        deltas = strong.loc[common, "target_utility"] - weak.loc[common, "target_utility"]
        base_deltas = strong.loc[common, "baseline_utility"] - weak.loc[common, "baseline_utility"]
        token_deltas = (
            strong.loc[common, "target_compute_tokens_per_call"]
            - weak.loc[common, "target_compute_tokens_per_call"]
        )
        rows.append(
            {
                "family": FAMILY_LABELS[family],
                "weak_to_strong": f"{weak_level} -> {strong_level}",
                "n": len(common),
                "target_delta": float(deltas.mean()),
                "baseline_delta": float(base_deltas.mean()),
                "improved": int((deltas > 1e-9).sum()),
                "worsened": int((deltas < -1e-9).sum()),
                "flat": int((deltas.abs() <= 1e-9).sum()),
                "compute_delta": float(token_deltas.mean()),
            }
        )
    return pd.DataFrame(rows)


def ttc_section(df: pd.DataFrame, plot_paths: dict[str, Path]) -> str:
    overall = aggregate(df, [])
    display = overall[["family_label", "level", "n", "target_utility", "baseline_utility", "utility_gap", "compute_tokens_per_call", "output_tokens_per_call", "consensus_rate"]].copy()
    delta = weak_strong_delta_table(df)

    lines: list[str] = [
        "## Test-Time Compute Scaling Addendum",
        "",
        "This addendum brings the 216-sample TTC experiment into the same N=2 bargaining frame as the Elo-scaling results above. The varied agent is the target/adversary model; the opponent is always the `gpt-5-nano` baseline with low reasoning. The target families are GPT-5, Claude Sonnet 4.6, and Gemini 3 Flash, each swept over four requested reasoning/effort levels.",
        "",
        "The headline contrast is that cross-model Elo scaling is strongly positive, while within-model reasoning scaling is weak and non-monotone. Requested reasoning level changes the model's deliberation style and token footprint, but it does not behave like a scalar increase in bargaining capability.",
        "",
        "### Quantitative Readout",
        "",
        markdown_table(
            display,
            [
                ("family_label", "family"),
                ("level", "level"),
                ("n", "n"),
                ("target_utility", "target payoff"),
                ("baseline_utility", "baseline payoff"),
                ("utility_gap", "target - baseline"),
                ("compute_tokens_per_call", "compute/proxy tokens/call"),
                ("output_tokens_per_call", "output tokens/call"),
                ("consensus_rate", "consensus"),
            ],
        )[0],
    ]
    # markdown_table returns a list; splice it after the static text.
    lines = lines[:-1] + markdown_table(
        display,
        [
            ("family_label", "family"),
            ("level", "level"),
            ("n", "n"),
            ("target_utility", "target payoff"),
            ("baseline_utility", "baseline payoff"),
            ("utility_gap", "target - baseline"),
            ("compute_tokens_per_call", "compute/proxy tokens/call"),
            ("output_tokens_per_call", "output tokens/call"),
            ("consensus_rate", "consensus"),
        ],
    )
    lines.extend(
        [
            "",
            "Weak-to-strong deltas compare the lowest requested effort to the highest requested effort within the same family, game cell, and order.",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            delta,
            [
                ("family", "family"),
                ("weak_to_strong", "weak -> strong"),
                ("n", "n"),
                ("target_delta", "mean target delta"),
                ("baseline_delta", "mean baseline delta"),
                ("improved", "improved"),
                ("worsened", "worsened"),
                ("flat", "flat"),
                ("compute_delta", "mean compute-token delta"),
            ],
        )
    )
    lines.extend(
        [
            "",
            "### Compact Plot Set",
            "",
            "These are intentionally narrower than the standalone TTC report: one payoff plot, one token plot, target and baseline payoff by competition level, and one order-only plot.",
            "",
            "#### Payoff by requested effort",
            "",
            f"![TTC payoff by effort]({rel(plot_paths['payoff'])})",
            "",
            "#### Realized compute/proxy tokens by requested effort",
            "",
            f"![TTC tokens by effort]({rel(plot_paths['tokens'])})",
            "",
            "#### Target payoff by competition level",
            "",
            f"![TTC target payoff by competition]({rel(plot_paths['target_comp'])})",
            "",
            "#### Baseline payoff by competition level",
            "",
            f"![TTC baseline payoff by competition]({rel(plot_paths['baseline_comp'])})",
            "",
            "#### Target payoff by model order",
            "",
            f"![TTC target payoff by order]({rel(plot_paths['order'])})",
            "",
            "### Qualitative Mechanism",
            "",
            "For this addendum, the 216 TTC samples were audited by game: 72 item-allocation rollouts, 72 diplomacy rollouts, and 72 cofunding rollouts. I agree with the standalone TTC report's broad statistical warning that this is not monotone compute scaling. I would sharpen the interpretation, though: the weak TTC effect is not random noise around a hidden positive trend. Extra reasoning often changes the model's bargaining objective from `find my best acceptable deal` to `produce a coherent, fair, passable deal`. That can be good for welfare and consensus, but it is not the same thing as higher adversary payoff.",
            "",
            "The cross-model Elo result above is strong because higher-Elo models bring several capabilities at once: cleaner parsing, better role tracking, better utility arithmetic, more credible anchoring, and better timing. TTC scaling is narrower. It mostly gives the same model more deliberation inside the same protocol and preference geometry. In these games, the binding constraint is often not search depth; it is unanimous acceptability, a single contested marginal item, final proposal selection, or a cofunding mechanism where future promises do not bind.",
            "",
            "#### Game 1: More reasoning finds the marginal item, then becomes cautious",
            "",
            "Game 1 shows the clearest ceiling and acceptability effects. In the cooperative cell, nearly every reasoning level finds the obvious complementary split, so additional compute has little room to raise payoff. When models fail, the miss is usually shallow rather than deep: Gemini minimal in a cooperative baseline-first run gave itself Stone and Quill while leaving Apple behind, despite Apple being valuable, explaining that the bundle contained its \"two highest value items\": [Gemini minimal cooperative miss](../ttc_native_scaling_20260502_212943/gemini-3-flash/level_minimal/game1_comp_0p0/baseline_first/seed_42/run_1_all_interactions.json). TTC can fix this kind of omission, but it is not a scalable source of advantage after the obvious package is found.",
            "",
            "The mixed cell is mostly about one marginal item, Apple. GPT-5 minimal target-first let Apple become a temporary concession, saying it was a \"placeholder\" to revisit later; the vote made that placeholder final and the target lost 61 to 88: [GPT-5 minimal Apple concession](../ttc_native_scaling_20260502_212943/gpt-5/level_minimal/game1_comp_0p5/target_first/seed_42/run_1_all_interactions.json). GPT-5 low does better in the matched cell by detecting a \"misunderstanding or opportunistic flip\" and claiming Apple with Stone and Quill: [GPT-5 low protects Apple](../ttc_native_scaling_20260502_212943/gpt-5/level_low/game1_comp_0p5/target_first/seed_42/run_1_all_interactions.json). This is the real upside of TTC: it can make the model more alert to a loose surplus item.",
            "",
            "But the competitive cell shows the offsetting downside. With identical preferences, every extra point for the target is mostly a transfer that must still pass a unanimous vote. GPT-5 medium target-first captured Jewel and Apple for 64 utility: [GPT-5 medium competitive anchor](../ttc_native_scaling_20260502_212943/gpt-5/level_medium/game1_comp_1p0/target_first/seed_42/run_1_all_interactions.json). GPT-5 high in the same cell became more rejection-aware and settled for Jewel plus Pencil because \"risks rejection\" dominated the more aggressive demand: [GPT-5 high cautious settlement](../ttc_native_scaling_20260502_212943/gpt-5/level_high/game1_comp_1p0/target_first/seed_42/run_1_all_interactions.json). In other words, more reasoning can improve model theory of the opponent, but that may reduce extraction rather than raise it.",
            "",
            "#### Game 2: TTC helps anchoring only when it resists diplomatic over-coherence",
            "",
            "Game 2 is where TTC most resembles capability scaling, but only in the mixed cases. GPT-5 minimal in the mixed target-first cell drifted to Round 4 and lost discount value: [GPT-5 minimal mixed diplomacy](../ttc_native_scaling_20260502_212943/gpt-5/level_minimal/game2_rho_0_theta_1/target_first/seed_42/run_1_experiment_results.json). GPT-5 medium instead closed in Round 1 with a package that \"Locks strong outcomes\" on top issues and reached 97.63 target utility: [GPT-5 medium mixed diplomacy](../ttc_native_scaling_20260502_212943/gpt-5/level_medium/game2_rho_0_theta_1/target_first/seed_42/run_1_experiment_results.json). Here, extra reasoning sharpens the same mechanism that helped higher-Elo models: identify issue trades, lock the easy dimensions, and avoid wasting rounds.",
            "",
            "The competitive cell explains why that does not generalize monotonically. GPT-5 low target-first conceded too much by explicitly meeting the opponent's \"red lines\" and finished at 50.21 versus 83.17: [GPT-5 low red-line concession](../ttc_native_scaling_20260502_212943/gpt-5/level_low/game2_rho_n1_theta_1/target_first/seed_42/run_1_all_interactions.json). GPT-5 medium corrected the failure by holding its own anchors, ending 88.52 versus 45.08: [GPT-5 medium competitive anchor](../ttc_native_scaling_20260502_212943/gpt-5/level_medium/game2_rho_n1_theta_1/target_first/seed_42/run_1_all_interactions.json). But other high-effort runs show the opposite: Gemini high builds a polished Track B compromise and says it is \"moving significantly\" toward the other side's goals, ending only 68.60 versus 52.20: [Gemini high Track B concession](../ttc_native_scaling_20260502_212943/gemini-3-flash/level_high/game2_rho_n1_theta_1/target_first/seed_42/run_1_all_interactions.json).",
            "",
            "The crux is that diplomacy rewards crisp issue trades, not more diplomatic prose. Extra reasoning helps if it produces a hard package around the target's weighted priorities. It hurts if it produces more lanes, guardrails, staged reviews, and mutual-acceptability language that treats the opponent's utility function as a set of constraints to satisfy.",
            "",
            "#### Game 3: Cofunding is a protocol-comprehension test, not a pure reasoning-depth test",
            "",
            "Cofunding is the sharpest case because the mechanism punishes plausible but invalid plans. Accepted proposals end the game, rejected proposals do not carry over, and partial funding evaporates. Higher reasoning sometimes catches this. In GPT-5 high partial-alignment baseline-first, the target warns that \"partial funds don't carry\" and backs only projects that can finish this round: [GPT-5 high partial-funding correction](../ttc_native_scaling_20260502_212943/gpt-5/level_high/game3_alpha_0p5_sigma_0p6/baseline_first/seed_42/run_1_all_interactions.json). That is genuine competence, but its payoff effect depends on whose project is being finished.",
            "",
            "In the hard conflict cell, stronger reasoning can reduce the baseline's payoff by refusing negative-utility funding. GPT-5 medium baseline-first eventually states it will not pay for a \"zero-value item\" and accepts no funded outcome rather than subsidize Cedar: [GPT-5 medium hard cofunding refusal](../ttc_native_scaling_20260502_212943/gpt-5/level_medium/game3_alpha_0p0_sigma_0p2/baseline_first/seed_42/run_1_all_interactions.json). That looks like no improvement if the plot is target payoff, but qualitatively it is a more correct strategic refusal.",
            "",
            "When the target has first-mover leverage, the same extra reasoning can become exploitation. GPT-5 high target-first repeatedly pushes a Cedar-first handshake, asking for \"Cedar 11/9 YES\" and later inducing the opponent to fund a zero-value project: [GPT-5 high Cedar handshake](../ttc_native_scaling_20260502_212943/gpt-5/level_high/game3_alpha_0p0_sigma_0p2/target_first/seed_42/run_1_all_interactions.json). The final target utility is positive while the baseline is negative. This is not a smooth compute curve; it is a sign flip depending on role, order, and whether reasoning is used to refuse exploitation or impose it.",
            "",
            "The recurring failure mode is moralized commitment. In Claude low hard cofunding, the model treats honoring a prior two-round deal as more important than utility and accepts a bad Cedar contribution because rejecting would be \"pure bad faith\": [Claude low moralized commitment](../ttc_native_scaling_20260502_212943/claude-sonnet-4-6/level_low/game3_alpha_0p0_sigma_0p2/baseline_first/seed_42/run_1_all_interactions.json). More TTC sometimes fixes this by recognizing no-carryover; sometimes it simply makes the invalid future-round bargain more polished.",
            "",
            "#### Token axis interpretation",
            "",
            "The token plots should not be read as a clean hidden-compute dose-response curve. GPT-5 has provider-reported reasoning tokens, but realized reasoning is adaptive: effort labels do not map perfectly onto per-call reasoning. Claude and Gemini use visible output tokens as proxies, not hidden reasoning counts. Across the audit, high-token samples often correspond to hard cases, proposal repair, or long cofunding deadlocks rather than better strategic search. This is why the token plot can rise while payoff is flat or falling.",
            "",
            "#### Integrated takeaway",
            "",
            "The cohesive story is that Elo scaling and TTC scaling are improving different things. Moving to a higher-Elo model tends to improve broad bargaining competence: it parses the game, tracks roles, identifies mutually beneficial packages, anchors contested issues, and avoids protocol failures. Increasing requested reasoning in the same model mainly amplifies deliberation around the model's existing bargaining stance. If the stance is utility-aware and the game has a crisp exploitable margin, TTC helps. If the stance is fairness-seeking, rejection-averse, or confused about the cofunding mechanism, TTC produces nicer versions of concessions, delays, or invalid commitments. That is why TTC is weak and non-monotone even though cross-model capability scaling is strong.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    TTC_OUT.mkdir(parents=True, exist_ok=True)
    df = load_ttc()
    plot_paths = {
        "payoff": plot_payoff_by_effort(df),
        "tokens": plot_tokens_by_effort(df),
        "target_comp": plot_by_competition(
            df,
            "target_utility",
            "Mean target/adversary utility",
            "ttc_target_payoff_by_competition.png",
            "TTC scaling: target payoff by competition level",
        ),
        "baseline_comp": plot_by_competition(
            df,
            "baseline_utility",
            "Mean baseline utility",
            "ttc_baseline_payoff_by_competition.png",
            "TTC scaling: baseline payoff by competition level",
        ),
        "order": plot_target_by_order(df),
    }

    base_text = BASE_REPORT.read_text(encoding="utf-8")
    marker = "\n## GPT-5-nano baseline Plots\n"
    if marker not in base_text:
        raise RuntimeError(f"Could not find insertion marker in {BASE_REPORT}")
    before, after = base_text.split(marker, 1)
    augmented = before.rstrip() + "\n\n" + ttc_section(df, plot_paths).rstrip() + "\n" + marker + after
    NEW_REPORT.write_text(augmented, encoding="utf-8")
    print(f"Wrote augmented report to {NEW_REPORT}")
    print(f"Wrote TTC plots to {TTC_OUT}")


if __name__ == "__main__":
    main()

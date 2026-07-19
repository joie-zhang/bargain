#!/usr/bin/env python3
"""Build a concise paper-style TTC mechanism report."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TAG_ROOT = PROJECT_ROOT / "analysis/ttc_llm_strategic_tag_adjudication_20260629"
MANIFEST_JSONL = TAG_ROOT / "all_ttc_rollouts_manifest.jsonl"
EVENT_JSONL = TAG_ROOT / "ttc_llm_event_tags.jsonl"
HOT_ROOT = PROJECT_ROOT / "analysis/ttc_hot_strategic_tags_20260629"
REVIEW_JSON = PROJECT_ROOT / "strategic_tag_review_final.json"
CAT_SUMMARY_CSV = HOT_ROOT / "hot_category_family_level_summary.csv"
OUT_DIR = HOT_ROOT / "paper_style_section"
OVERLEAF_TTC_DIR = PROJECT_ROOT / "overleaf/icml_aiwild_template/graphics/qualitative_ttc"

FAMILY_ORDER = ["gpt-5", "claude-sonnet-4-6", "gemini-3-flash"]
FAMILY_LABELS = {
    "gpt-5": "GPT-5",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gemini-3-flash": "Gemini 3 Flash",
}
EFFORT_COLORS = {
    "minimal": "#64748b",
    "low": "#2563eb",
    "medium": "#0f766e",
    "high": "#f97316",
    "max": "#7c3aed",
}
CATEGORY_COLORS = {
    "self-interest/exploitation": "#c2410c",
    "pressure": "#be123c",
    "logical persuasion": "#2563eb",
    "formalization": "#7c3aed",
    "trade/compromise": "#15803d",
}
JUSTIFICATION_TAGS = [
    "self_advocacy_value_maximization",
    "conditional_veto_threat",
    "utility_arithmetic_receipts",
    "counter_anchor_cost_policing",
    "fairness_accusation_pressure",
    "budget_carryover_hallucination",
    "adversarial_callout",
    "ultimatum_language",
    "leverage_preservation",
]
SETTLEMENT_SEARCH_TAGS = [
    "concession_laddering",
    "conditional_quid_pro_quo",
    "conditional_support_ledger",
    "threshold_gap_calculation",
    "vote_history_diagnostics",
    "agent_specific_payoff_accounting",
    "low_weight_concession_leverage",
]
CONCESSION_EXCHANGE_TAGS = [
    "concession_laddering",
    "conditional_quid_pro_quo",
    "conditional_support_ledger",
    "low_weight_concession_leverage",
]
REPAIR_DIAGNOSTIC_TAGS = [
    "threshold_gap_calculation",
    "vote_history_diagnostics",
    "agent_specific_payoff_accounting",
    "counter_anchor_cost_policing",
    "budget_carryover_hallucination",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def embed_png(path: Path, alt: str) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"![{alt}](data:image/png;base64,{encoded})"


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="#d1d5db", alpha=0.5, linewidth=0.8)
    ax.tick_params(axis="both", labelsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def load_payoff_summary() -> pd.DataFrame:
    manifest = pd.DataFrame(read_jsonl(MANIFEST_JSONL))
    return (
        manifest.groupby(["family", "level", "level_index"], as_index=False)
        .agg(
            n=("config_id", "nunique"),
            mean_tokens_per_call=("target_compute_tokens_per_call", "mean"),
            mean_payoff=("target_utility", "mean"),
            sem_payoff=("target_utility", lambda s: s.std(ddof=1) / (len(s) ** 0.5)),
            consensus_rate=("consensus_reached", "mean"),
        )
        .sort_values(["family", "level_index"])
    )


def load_rollout_bridge_data() -> pd.DataFrame:
    manifest = pd.DataFrame(read_jsonl(MANIFEST_JSONL))
    events = pd.DataFrame(read_jsonl(EVENT_JSONL))
    review = json.loads(REVIEW_JSON.read_text(encoding="utf-8"))
    hot = pd.DataFrame([row for row in review["responses"] if row["decision"] == "hot"])
    target_events = events.merge(hot[["tag_code", "category"]], on="tag_code", how="inner")
    target_events = target_events[target_events["speaker_role"].eq("target")].copy()
    cat_counts = target_events.groupby(["config_id", "category"]).size().unstack(fill_value=0)
    tag_counts = target_events.groupby(["config_id", "tag_code"]).size().unstack(fill_value=0)
    bridge = (
        manifest.merge(cat_counts, left_on="config_id", right_index=True, how="left")
        .merge(tag_counts, left_on="config_id", right_index=True, how="left")
    )
    for category in hot["category"].unique():
        if category not in bridge:
            bridge[category] = 0
        bridge[category] = bridge[category].fillna(0)
    for tag in set(JUSTIFICATION_TAGS + SETTLEMENT_SEARCH_TAGS + CONCESSION_EXCHANGE_TAGS + REPAIR_DIAGNOSTIC_TAGS):
        if tag not in bridge:
            bridge[tag] = 0
        bridge[tag] = bridge[tag].fillna(0)
    bridge["gpt_stubborn_self_interest"] = bridge["self-interest/exploitation"] + bridge["pressure"]
    bridge["gemini_logic_formalization"] = bridge["logical persuasion"] + bridge["formalization"]
    bridge["justification_posture"] = bridge[JUSTIFICATION_TAGS].sum(axis=1)
    bridge["settlement_search"] = bridge[SETTLEMENT_SEARCH_TAGS].sum(axis=1)
    bridge["concession_exchange"] = bridge[CONCESSION_EXCHANGE_TAGS].sum(axis=1)
    bridge["repair_diagnostics"] = bridge[REPAIR_DIAGNOSTIC_TAGS].sum(axis=1)
    bridge["total_payoff"] = bridge["target_utility"] + bridge["baseline_utility"]
    bridge["payoff_gap_abs"] = (bridge["target_utility"] - bridge["baseline_utility"]).abs()

    fixed_effects = ["family", "game_cell", "order"]
    residual_cols = [
        "gpt_stubborn_self_interest",
        "gemini_logic_formalization",
        "justification_posture",
        "settlement_search",
        "concession_exchange",
        "repair_diagnostics",
        "trade/compromise",
        "target_utility",
        "baseline_utility",
        "total_payoff",
        "payoff_gap_abs",
        "consensus_reached",
        "final_round",
    ]
    for col in residual_cols:
        bridge[f"{col}_resid"] = bridge[col].astype(float) - bridge.groupby(fixed_effects)[col].transform("mean").astype(float)
    return bridge


def correlation_stats(bridge: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("gpt-5", "gpt_stubborn_self_interest", "target_utility"),
        ("gpt-5", "gpt_stubborn_self_interest", "consensus_reached"),
        ("gpt-5", "gpt_stubborn_self_interest", "final_round"),
        ("gpt-5", "gpt_stubborn_self_interest", "justification_posture"),
        ("gpt-5", "justification_posture", "target_utility"),
        ("gpt-5", "justification_posture", "consensus_reached"),
        ("gpt-5", "justification_posture", "final_round"),
        ("gpt-5", "concession_exchange", "target_utility"),
        ("gpt-5", "concession_exchange", "consensus_reached"),
        ("gpt-5", "concession_exchange", "final_round"),
        ("gemini-3-flash", "gemini_logic_formalization", "target_utility"),
        ("gemini-3-flash", "gemini_logic_formalization", "consensus_reached"),
        ("gemini-3-flash", "gemini_logic_formalization", "final_round"),
        ("gemini-3-flash", "gemini_logic_formalization", "total_payoff"),
        ("gemini-3-flash", "gemini_logic_formalization", "repair_diagnostics"),
        ("gemini-3-flash", "gemini_logic_formalization", "concession_exchange"),
        ("gemini-3-flash", "concession_exchange", "target_utility"),
        ("gemini-3-flash", "concession_exchange", "consensus_reached"),
        ("gemini-3-flash", "concession_exchange", "final_round"),
        ("gemini-3-flash", "repair_diagnostics", "target_utility"),
        ("gemini-3-flash", "repair_diagnostics", "consensus_reached"),
        ("gemini-3-flash", "repair_diagnostics", "final_round"),
        ("gemini-3-flash", "repair_diagnostics", "total_payoff"),
    ]
    rows = []
    for family, mechanism, outcome in specs:
        sub = bridge[bridge["family"].eq(family)].copy()
        x = sub[f"{mechanism}_resid"]
        y = sub[f"{outcome}_resid"]
        ok = x.notna() & y.notna() & (x.std() > 0) & (y.std() > 0)
        if ok.sum() < 3:
            r = np.nan
            slope = np.nan
        else:
            r = float(np.corrcoef(x[ok], y[ok])[0, 1])
            slope = float(np.polyfit(x[ok], y[ok], 1)[0])
        rows.append(
            {
                "family": family,
                "mechanism": mechanism,
                "outcome": outcome,
                "n": int(ok.sum()),
                "r": r,
                "slope": slope,
            }
        )
    return pd.DataFrame(rows)


def plot_payoff(payoff: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.35), sharey=True)
    for ax, family in zip(axes, FAMILY_ORDER, strict=True):
        sub = payoff[payoff["family"].eq(family)].sort_values("level_index")
        ax.plot(
            sub["mean_tokens_per_call"],
            sub["mean_payoff"],
            color="#9ca3af",
            linewidth=2.4,
            zorder=1,
        )
        for _, row in sub.iterrows():
            level = row["level"]
            ax.errorbar(
                row["mean_tokens_per_call"],
                row["mean_payoff"],
                yerr=row["sem_payoff"],
                fmt="o",
                color=EFFORT_COLORS.get(level, "#111827"),
                markersize=7,
                linewidth=1.4,
                capsize=3,
                zorder=2,
            )
            ax.annotate(
                level,
                (row["mean_tokens_per_call"], row["mean_payoff"]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=7.5,
                color=EFFORT_COLORS.get(level, "#111827"),
            )
        ax.set_title(FAMILY_LABELS[family], fontsize=13, pad=8)
        ax.set_xlabel("Observed reasoning tokens/call", fontsize=10)
        style_axis(ax)
    axes[0].set_ylabel("Mean target payoff", fontsize=10)
    axes[0].set_ylim(48, 82)
    fig.suptitle("Target payoff does not reliably scale with observed TTC", fontsize=15, y=1.03)
    fig.tight_layout()
    path = OUT_DIR / "paper_ttc_payoff_vs_compute.png"
    fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return path


def add_residual_scatter(
    ax: plt.Axes,
    bridge: pd.DataFrame,
    family: str,
    mechanism: str,
    title: str,
    color: str,
) -> tuple[float, float]:
    sub = bridge[bridge["family"].eq(family)].copy()
    x = sub[f"{mechanism}_resid"]
    y = sub["target_utility_resid"]
    ok = x.notna() & y.notna() & (x.std() > 0) & (y.std() > 0)
    ax.scatter(x[ok], y[ok], s=25, color=color, alpha=0.34, edgecolor="none")
    ax.axhline(0, color="#6b7280", linewidth=1.0, alpha=0.7)
    ax.axvline(0, color="#6b7280", linewidth=1.0, alpha=0.7)
    if ok.sum() >= 3:
        slope, intercept = np.polyfit(x[ok], y[ok], 1)
        xs = np.linspace(float(x[ok].min()), float(x[ok].max()), 80)
        ax.plot(xs, slope * xs + intercept, color=color, linewidth=2.3)
        r = float(np.corrcoef(x[ok], y[ok])[0, 1])
    else:
        slope = np.nan
        r = np.nan
    ax.set_title(title, fontsize=12, pad=7)
    ax.set_xlabel("Mechanism above/below matched-cell mean", fontsize=9)
    ax.set_ylabel("Target payoff above/below matched-cell mean", fontsize=9)
    ax.text(
        0.03,
        0.95,
        f"r={r:+.2f}\nslope={slope:+.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9, "pad": 4},
    )
    style_axis(ax)
    return r, slope


def plot_mechanism_bridge(cat_summary: pd.DataFrame, bridge: pd.DataFrame) -> Path:
    panels = [
        ("gpt-5", ["self-interest/exploitation", "pressure", "logical persuasion"]),
        ("gemini-3-flash", ["logical persuasion", "formalization", "trade/compromise", "self-interest/exploitation"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.2), sharey=False)
    for ax, (family, categories) in zip(axes[0], panels, strict=True):
        for category in categories:
            sub = cat_summary[
                cat_summary["family"].eq(family) & cat_summary["category"].eq(category)
            ].sort_values("level_index")
            if sub.empty:
                continue
            ax.plot(
                sub["level_index"],
                sub["target_events_per_rollout"],
                marker="o",
                linewidth=2.2,
                markersize=5.5,
                color=CATEGORY_COLORS[category],
                label=category,
            )
            last = sub.iloc[-1]
            ax.annotate(
                category,
                (last["level_index"], last["target_events_per_rollout"]),
                textcoords="offset points",
                xytext=(7, 0),
                va="center",
                fontsize=8,
                color=CATEGORY_COLORS[category],
            )
        ax.set_title(FAMILY_LABELS[family], fontsize=13, pad=8)
        ax.set_xlabel("Requested reasoning effort", fontsize=10)
        ax.set_xticks([0, 1, 2, 3], ["minimal", "low", "medium", "high"], rotation=0)
        style_axis(ax)
    axes[0, 0].set_ylabel("Target-authored tag events / rollout", fontsize=10)
    axes[0, 0].set_ylim(0, 4.25)
    axes[0, 1].set_ylim(0, 4.25)
    add_residual_scatter(
        axes[1, 0],
        bridge,
        "gpt-5",
        "gpt_stubborn_self_interest",
        "GPT-5: stubborn self-interest does not buy payoff",
        "#c2410c",
    )
    add_residual_scatter(
        axes[1, 1],
        bridge,
        "gemini-3-flash",
        "gemini_logic_formalization",
        "Gemini: more logic/formality is not payoff-improving",
        "#2563eb",
    )
    fig.suptitle("What extra TTC changes, and why that does not translate into payoff", fontsize=15, y=1.01)
    fig.tight_layout()
    path = OUT_DIR / "paper_ttc_mechanism_outcome_bridge.png"
    fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return path


def plot_main_text_ttc_intensity(cat_summary: pd.DataFrame) -> Path:
    """Write the compact TTC mechanism figure used in the main text."""
    OVERLEAF_TTC_DIR.mkdir(parents=True, exist_ok=True)
    panels = [
        ("gpt-5", "self-interest/exploitation", "GPT-5:\nself-interest/exploitation", "#d62728"),
        ("gemini-3-flash", "trade/compromise", "Gemini 3 Flash:\ntrade/compromise", "#1f77b4"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.2), sharey=False)
    for ax, (family, category, title, color) in zip(axes, panels, strict=True):
        sub = cat_summary[
            cat_summary["family"].eq(family) & cat_summary["category"].eq(category)
        ].sort_values("level_index")
        ax.plot(
            sub["level_index"],
            sub["target_events_per_rollout"],
            marker="o",
            linewidth=2.3,
            markersize=5.2,
            color=color,
        )
        ax.set_title(title, fontsize=11.5, pad=7)
        ax.set_xlabel("Reasoning effort", fontsize=9.5)
        ax.set_xticks([0, 1, 2, 3], ["minimal", "low", "medium", "high"])
        style_axis(ax)
    axes[0].set_ylabel("Mean events / rollout", fontsize=9.5)
    fig.tight_layout()
    path = OVERLEAF_TTC_DIR / "ttc_intensity_maintext_halfpage.png"
    fig.savefig(path, dpi=260, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return path


def corr_value(bridge: pd.DataFrame, family: str, x_col: str, y_col: str) -> tuple[float, float]:
    sub = bridge[bridge["family"].eq(family)].copy()
    x = sub[f"{x_col}_resid"]
    y = sub[f"{y_col}_resid"]
    ok = x.notna() & y.notna() & (x.std() > 0) & (y.std() > 0)
    if ok.sum() < 3:
        return np.nan, np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1]), float(np.polyfit(x[ok], y[ok], 1)[0])


def weak_to_high_delta(bridge: pd.DataFrame, family: str, col: str) -> float:
    sub = bridge[bridge["family"].eq(family)].copy()
    weak = "minimal" if family in {"gpt-5", "gemini-3-flash"} else "low"
    strong = "high" if family in {"gpt-5", "gemini-3-flash"} else "max"
    means = sub.groupby("level")[col].mean()
    return float(means.get(strong, np.nan) - means.get(weak, np.nan))


def plot_non_scaling_explanation(bridge: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.25))

    # Panel 1: what reasoning effort changes.
    shifts = pd.DataFrame(
        [
            {
                "label": "GPT: self-interest\n+ pressure",
                "delta": weak_to_high_delta(bridge, "gpt-5", "gpt_stubborn_self_interest"),
                "color": "#c2410c",
            },
            {
                "label": "GPT: justification\ntags",
                "delta": weak_to_high_delta(bridge, "gpt-5", "justification_posture"),
                "color": "#f97316",
            },
            {
                "label": "GPT: concession /\ntrade tags",
                "delta": weak_to_high_delta(bridge, "gpt-5", "concession_exchange"),
                "color": "#65a30d",
            },
            {
                "label": "Gemini: logic\n+ formality",
                "delta": weak_to_high_delta(bridge, "gemini-3-flash", "gemini_logic_formalization"),
                "color": "#2563eb",
            },
            {
                "label": "Gemini: repair /\ndebugging tags",
                "delta": weak_to_high_delta(bridge, "gemini-3-flash", "repair_diagnostics"),
                "color": "#7c3aed",
            },
            {
                "label": "Gemini: concession /\ntrade tags",
                "delta": weak_to_high_delta(bridge, "gemini-3-flash", "concession_exchange"),
                "color": "#15803d",
            },
        ]
    )
    ax = axes[0]
    ax.axhline(0, color="#6b7280", linewidth=1.0)
    ax.bar(range(len(shifts)), shifts["delta"], color=shifts["color"])
    ax.set_xticks(range(len(shifts)), shifts["label"], rotation=35, ha="right")
    ax.set_ylabel("Extra events / rollout at high effort")
    ax.set_title("1. What increases when effort is high?", fontsize=12)
    style_axis(ax)

    # Panel 2: what the mechanisms mostly mean locally.
    links = pd.DataFrame(
        [
            {
                "label": "GPT self-interest\nwith justification",
                "r": corr_value(bridge, "gpt-5", "gpt_stubborn_self_interest", "justification_posture")[0],
                "color": "#c2410c",
            },
            {
                "label": "Gemini logic\nwith repair/debugging",
                "r": corr_value(bridge, "gemini-3-flash", "gemini_logic_formalization", "repair_diagnostics")[0],
                "color": "#2563eb",
            },
            {
                "label": "Gemini logic\nwith concession/trade",
                "r": corr_value(bridge, "gemini-3-flash", "gemini_logic_formalization", "concession_exchange")[0],
                "color": "#15803d",
            },
        ]
    )
    ax = axes[1]
    ax.axhline(0, color="#6b7280", linewidth=1.0)
    ax.bar(range(len(links)), links["r"], color=links["color"])
    ax.set_xticks(range(len(links)), links["label"], rotation=25, ha="right")
    ax.set_ylim(-0.2, 1.0)
    ax.set_ylabel("Correlation within matched tasks")
    ax.set_title("2. What does the scaled behavior co-occur with?", fontsize=12)
    for idx, row in links.iterrows():
        ax.text(idx, row["r"] + 0.035, f"{row['r']:+.2f}", ha="center", va="bottom", fontsize=9)
    style_axis(ax)

    # Panel 3: what the proximate behaviors predict.
    outcome_rows = []
    for family, mediator, label, color in [
        ("gpt-5", "justification_posture", "GPT justification tags", "#f97316"),
        ("gpt-5", "concession_exchange", "GPT concession/trade tags", "#65a30d"),
        ("gemini-3-flash", "concession_exchange", "Gemini concession/trade tags", "#15803d"),
        ("gemini-3-flash", "repair_diagnostics", "Gemini repair/debugging tags", "#7c3aed"),
    ]:
        for outcome, outcome_label in [
            ("target_utility", "target payoff"),
            ("consensus_reached", "consensus"),
            ("final_round", "final round"),
        ]:
            outcome_rows.append(
                {
                    "mediator": label,
                    "outcome": outcome_label,
                    "r": corr_value(bridge, family, mediator, outcome)[0],
                    "color": color,
                }
            )
    outcomes = pd.DataFrame(outcome_rows)
    ax = axes[2]
    positions = np.arange(3)
    width = 0.18
    offsets = {
        "GPT justification tags": -1.5 * width,
        "GPT concession/trade tags": -0.5 * width,
        "Gemini concession/trade tags": 0.5 * width,
        "Gemini repair/debugging tags": 1.5 * width,
    }
    for mediator, offset in offsets.items():
        sub = outcomes[outcomes["mediator"].eq(mediator)].set_index("outcome").loc[
            ["target payoff", "consensus", "final round"]
        ]
        bars = ax.bar(positions + offset, sub["r"], width=width, color=sub["color"].iloc[0], label=mediator)
        for bar, value in zip(bars, sub["r"], strict=True):
            y = value + (0.035 if value >= 0 else -0.045)
            va = "bottom" if value >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{value:+.2f}",
                ha="center",
                va=va,
                fontsize=7.6,
                rotation=90,
            )
    ax.axhline(0, color="#6b7280", linewidth=1.0)
    ax.set_xticks(positions, ["target\npayoff", "consensus", "final\nround"])
    ax.set_ylim(-0.65, 0.9)
    ax.set_ylabel("Correlation within matched tasks")
    ax.set_title("3. Do those behaviors predict better outcomes?", fontsize=12)
    ax.legend(frameon=True, fontsize=8, loc="upper left")
    style_axis(ax)

    fig.suptitle("Why the scaled TTC behaviors do not translate into higher payoff", fontsize=15, y=1.03)
    fig.tight_layout()
    path = OUT_DIR / "paper_ttc_why_non_scaling_three_panel.png"
    fig.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return path


def write_report(
    payoff_plot: Path,
    mechanism_plot: Path,
    explanation_plot: Path,
    payoff: pd.DataFrame,
    stats: pd.DataFrame,
) -> Path:
    report = OUT_DIR / "ttc_scaling_paper_style_mini_section.md"
    gpt = payoff[payoff["family"].eq("gpt-5")].sort_values("level_index")
    gem = payoff[payoff["family"].eq("gemini-3-flash")].sort_values("level_index")
    claude = payoff[payoff["family"].eq("claude-sonnet-4-6")].sort_values("level_index")
    gpt_min = gpt.iloc[0]
    gpt_high = gpt[gpt["level"].eq("high")].iloc[0]
    gem_min = gem.iloc[0]
    gem_high = gem[gem["level"].eq("high")].iloc[0]
    claude_low = claude.iloc[0]
    claude_max = claude[claude["level"].eq("max")].iloc[0]

    def stat(family: str, mechanism: str, outcome: str, field: str) -> float:
        row = stats[
            stats["family"].eq(family) & stats["mechanism"].eq(mechanism) & stats["outcome"].eq(outcome)
        ]
        return float(row.iloc[0][field])

    lines: list[str] = []
    lines.append("# Why Test-Time Compute Does Not Reliably Scale Negotiation Payoff")
    lines.append("")
    lines.append("## Paper-Style Section")
    lines.append("")
    lines.append("Increasing a model's deliberation budget does not produce the same effect as replacing it with a stronger model. In the TTC experiment, the target model is held fixed while its reasoning effort varies. If reasoning were a reliable substitute for capability, mean target payoff should rise as observed reasoning tokens per call increase. We do not observe that pattern. GPT-5 moves from {:.1f} payoff at minimal effort to {:.1f} at high effort, a small change relative to the across-rollout uncertainty. Gemini is effectively flat: {:.1f} at minimal effort and {:.1f} at high effort despite an eightfold increase in observed reasoning tokens. Claude is harder to interpret as a scaling curve because its observed token proxy is non-monotone: low effort averages {:.0f} tokens/call, while max effort averages {:.0f} tokens/call.".format(
        gpt_min["mean_payoff"],
        gpt_high["mean_payoff"],
        gem_min["mean_payoff"],
        gem_high["mean_payoff"],
        claude_low["mean_tokens_per_call"],
        claude_max["mean_tokens_per_call"],
    ))
    lines.append("")
    lines.append(embed_png(payoff_plot, "Payoff versus observed test-time compute"))
    lines.append("")
    lines.append("To understand what the extra computation is doing instead, we tagged the 216 TTC rollouts with a strategic-behavior codebook and focused on the 29 labels marked as most informative. The result is crisp: **GPT-5 becomes more stubbornly self-interested with reasoning, while Gemini 3 Flash becomes more logical and formal.** The key question is whether those induced behaviors actually buy payoff.")
    lines.append("")
    lines.append(embed_png(mechanism_plot, "Mechanism and payoff bridge"))
    lines.append("")
    lines.append("The top row shows what scales. For GPT-5, extra reasoning mostly sharpens self-defense: from minimal to high effort, self-interest/exploitation rises by +1.50 events per rollout and pressure rises by +0.72. The underlying labels are concrete: self-advocacy/value maximization increases by +1.17 events per rollout, utility-arithmetic receipts by +0.78, and conditional veto threats by +0.78. Qualitatively, the model says things like \"Pencil: my clear top priority\" and \"I cannot accept any change that reduces my control of Stone, Quill, or Apple.\"")
    lines.append("")
    lines.append("Gemini changes in a different direction. As reasoning increases, it becomes less like a trader and more like an auditor. Logical persuasion rises by +1.44 events per rollout and formalization rises by +1.28, while trade/compromise falls by -1.22 and self-interest/exploitation falls by -0.67. The increased tags are rule-checking and accounting behaviors: correcting \"confusion ... between the cost of the projects and the utility/value,\" or arguing that both parties can receive \"100% of our respective interests.\"")
    lines.append("")
    lines.append("The bottom row makes the second step explicit. We residualize both the mechanism and target payoff within matched task cells (same family, game cell, and speaking order), so the comparison asks whether a rollout that is unusually self-interested or unusually logical for its local setting earns unusually high payoff. It does not. GPT-5's stubborn-self-interest residual is essentially unrelated to target payoff residual (r={:+.2f}, slope={:+.2f} payoff per extra event). Gemini's logic/formalization residual is negatively associated with target payoff residual (r={:+.2f}, slope={:+.2f}). Gemini's logic/formalization is also associated with lower consensus (r={:+.2f}) and longer negotiations (r={:+.2f}), which suggests that these tags often mark unresolved bargaining friction rather than successful persuasion.".format(
        stat("gpt-5", "gpt_stubborn_self_interest", "target_utility", "r"),
        stat("gpt-5", "gpt_stubborn_self_interest", "target_utility", "slope"),
        stat("gemini-3-flash", "gemini_logic_formalization", "target_utility", "r"),
        stat("gemini-3-flash", "gemini_logic_formalization", "target_utility", "slope"),
        stat("gemini-3-flash", "gemini_logic_formalization", "consensus_reached", "r"),
        stat("gemini-3-flash", "gemini_logic_formalization", "final_round", "r"),
    ))
    lines.append("")
    lines.append("The next figure explains the third step: **why** those scaled behaviors fail to become higher payoff. The mental model is simple. Extra reasoning can be spent on at least two different things. It can be spent on **settlement search**, meaning trying to find a deal the other agent will accept. Or it can be spent on **justification and repair**, meaning producing better reasons for the current position or debugging a negotiation that is already stuck. Only the first one is obviously payoff-producing.")
    lines.append("")
    lines.append("Definitions for the figure:")
    lines.append("")
    lines.append("- **Justification tags** are tags where the target gives reasons for its current stance: self-advocacy, veto threats, utility receipts, fairness accusations, and similar behaviors. These tags make the target's position more explicit, but they do not necessarily change the offer.")
    lines.append("- **Repair/debugging tags** are tags where the target diagnoses a problem: a failed vote, a threshold gap, confused utility accounting, an invalid anchor, or a budget-rule mistake. These tags often appear when the negotiation is already difficult.")
    lines.append("- **Concession/trade tags** are tags where the target is actually exchanging movement: quid pro quo, concession laddering, conditional support, or offering low-value items to preserve high-value items.")
    lines.append("- **Matched task** means the same model family, same game cell, and same speaking order. We compare each rollout to the average of its matched task, so we are not just rediscovering that some games are harder than others.")
    lines.append("- **Correlation within matched tasks** means: when a behavior is above its matched-task average, is the outcome also above its matched-task average? A positive value means they rise together. A negative value means the behavior is higher when the outcome is lower.")
    lines.append("")
    lines.append(embed_png(explanation_plot, "Three-panel explanation for why TTC mechanisms do not scale payoff"))
    lines.append("")
    lines.append("Read the figure from left to right. Panel 1 asks what high reasoning effort increases. GPT-5 increases self-interested pressure and justification tags, and it also increases concession/trade tags somewhat. Gemini increases logic/formality and repair/debugging tags, while Gemini concession/trade tags go down. That already hints at the problem for Gemini: the added reasoning is not mainly becoming more flexible. For GPT-5, the key question is whether either justification or concession/trade actually predicts payoff.")
    lines.append("")
    lines.append("Panel 2 asks what the scaled behavior is doing locally. GPT-5's self-interested pressure mostly co-occurs with justification tags (r={:+.2f}). So the extra self-interest is mostly the model explaining and defending its position. Gemini's logic/formality mostly co-occurs with repair/debugging tags (r={:+.2f}), but barely co-occurs with concession/trade tags (r={:+.2f}). So Gemini's extra logic is mostly diagnosing or correcting the negotiation, not making reciprocal moves.".format(
        stat("gpt-5", "gpt_stubborn_self_interest", "justification_posture", "r"),
        stat("gemini-3-flash", "gemini_logic_formalization", "repair_diagnostics", "r"),
        stat("gemini-3-flash", "gemini_logic_formalization", "concession_exchange", "r"),
    ))
    lines.append("")
    lines.append("Panel 3 asks whether those proximate behaviors predict better outcomes. The numeric labels are important because one bar is almost exactly zero: GPT-5's justification tags are basically unrelated to target payoff (r={:+.2f}) and negotiation length (r={:+.2f}), so the orange payoff bar sits on the horizontal axis. GPT-5's concession/trade tags also do not show a payoff benefit (target payoff r={:+.2f}); they are only weakly associated with consensus (r={:+.2f}) and final round (r={:+.2f}). Gemini's concession/trade tags also fail to explain higher target payoff (r={:+.2f}), though they are mildly associated with consensus (r={:+.2f}) and final round (r={:+.2f}). Gemini's repair/debugging tags predict lower target payoff (r={:+.2f}), lower consensus (r={:+.2f}), and longer negotiations (r={:+.2f}). This does not mean repair language causes bad outcomes by itself. The more natural interpretation is that repair language is a symptom of hard or broken negotiations: the model is spending compute cleaning up the mess, not converting that compute into a better deal.".format(
        stat("gpt-5", "justification_posture", "target_utility", "r"),
        stat("gpt-5", "justification_posture", "final_round", "r"),
        stat("gpt-5", "concession_exchange", "target_utility", "r"),
        stat("gpt-5", "concession_exchange", "consensus_reached", "r"),
        stat("gpt-5", "concession_exchange", "final_round", "r"),
        stat("gemini-3-flash", "concession_exchange", "target_utility", "r"),
        stat("gemini-3-flash", "concession_exchange", "consensus_reached", "r"),
        stat("gemini-3-flash", "concession_exchange", "final_round", "r"),
        stat("gemini-3-flash", "repair_diagnostics", "target_utility", "r"),
        stat("gemini-3-flash", "repair_diagnostics", "consensus_reached", "r"),
        stat("gemini-3-flash", "repair_diagnostics", "final_round", "r"),
    ))
    lines.append("")
    lines.append("This gives the mechanism for non-scaling. Negotiation payoff is not earned by being more explicit about what one wants, or by giving a cleaner proof that an offer is valid. It is earned by finding concessions that the other agent will accept while preserving enough value for oneself. GPT-5's extra reasoning often defends the current reservation point. Gemini's extra reasoning audits the offer space. Both can make the transcript look more deliberative, but neither is the missing ingredient: adaptive settlement search.")
    lines.append("")
    lines.append("The contrast with Elo scaling is therefore natural. Model strength can improve the policy that chooses what to pursue: better priors about the opponent, better anticipation of acceptance constraints, and better calibration about when to concede. TTC mostly deepens execution of the policy the same model already selected. When that policy is constructive, reasoning can help. When it is brittle, self-protective, or over-formalized, reasoning makes the failure mode more articulate.")
    lines.append("")
    lines.append("**Concise takeaway:** test-time compute makes bargaining behavior more explicit, but explicitness is not the same as bargaining skill. GPT-5 spends extra reasoning on self-advocacy and red lines; Gemini spends it on accounting and formal validity. Those are real cognitive changes, but they do not reliably increase target payoff because the missing ingredient is not more justification. The missing ingredient is better policy selection over concessions, counterpart modeling, and settlement search.")
    lines.append("")
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payoff = load_payoff_summary()
    cat_summary = pd.read_csv(CAT_SUMMARY_CSV)
    bridge = load_rollout_bridge_data()
    stats = correlation_stats(bridge)
    stats.to_csv(OUT_DIR / "paper_ttc_mechanism_payoff_bridge_stats.csv", index=False)
    payoff_plot = plot_payoff(payoff)
    mechanism_plot = plot_mechanism_bridge(cat_summary, bridge)
    main_text_ttc_plot = plot_main_text_ttc_intensity(cat_summary)
    explanation_plot = plot_non_scaling_explanation(bridge)
    report = write_report(payoff_plot, mechanism_plot, explanation_plot, payoff, stats)
    for path in [
        payoff_plot,
        mechanism_plot,
        main_text_ttc_plot,
        explanation_plot,
        OUT_DIR / "paper_ttc_mechanism_payoff_bridge_stats.csv",
        report,
    ]:
        print(path)


if __name__ == "__main__":
    main()

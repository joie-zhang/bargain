#!/usr/bin/env python3
"""Audit Gemini coalition labels and compare their prevalence across cohorts."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/scratch/gpfs/DANQIC/jz4391/bargain")
OUT = ROOT / "analysis/gemini_coalition_qualitative_20260816"

MONO_ROOT = ROOT / "analysis/llm_strategic_tag_adjudication_random_monoculture_20260629"
MULTI_ROOT = ROOT / "analysis/llm_strategic_tag_adjudication_20260628"
BILATERAL_ROOT = ROOT / "analysis/llm_strategic_tag_adjudication_n2_gpt5_20260629"
COMBINED_ROOT = ROOT / "analysis/ttc_combined_qualitative_20260814"
PRIMARY_BILATERAL = (
    ROOT
    / "experiments/results/n2_baseline_comparison_analysis_20260505/primary_runs_with_metrics.csv"
)

CASE_CONFIGS = ("config_0107", "config_0108", "config_0117")
SELECTED_23 = frozenset(
    {
        "adversarial_callout",
        "conditional_veto_threat",
        "fairness_accusation_pressure",
        "frustration_disappointment_display",
        "ultimatum_language",
        "rapport_before_pressure",
        "empathy_then_pivot",
        "threshold_gap_calculation",
        "agent_specific_payoff_accounting",
        "fairness_ledger_argument",
        "utility_arithmetic_receipts",
        "low_weight_concession_leverage",
        "conditional_quid_pro_quo",
        "vote_history_diagnostics",
        "conditional_support_ledger",
        "concession_laddering",
        "silent_free_beneficiary",
        "zero_value_subsidy",
        "leverage_preservation",
        "self_advocacy_value_maximization",
        "accepted_loss_capitulation",
        "counter_anchor_cost_policing",
        "budget_carryover_hallucination",
    }
)
CORE_COALITION_TAGS = frozenset(
    {
        "named_microcoalition_slate",
        "vote_bloc_counting",
        "holdout_bypass_minimum_coalition",
    }
)
COALITION_TAGS = frozenset(
    {
        "named_microcoalition_slate",
        "vote_bloc_counting",
        "bespoke_agent_or_bloc_recruitment",
        "holdout_bypass_minimum_coalition",
        "coalition_integrity_warning",
        "silent_agent_inclusion_guard",
        "third_party_mediation",
        "cross_agent_conflict_mapping",
        "nonoverlap_lane_setting",
    }
)
FOCUS_MODELS = (
    "gemini-2.5-pro",
    "gemini-3.1-pro",
    "claude-opus-4-5-20251101",
    "claude-opus-4-6",
    "gpt-5.2-chat-latest-20260210",
    "gpt-5.4-high",
)
FOCUS_DISPLAY = {
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-3.1-pro": "Gemini 3.1 Pro",
    "claude-opus-4-5-20251101": "Claude Opus 4.5",
    "claude-opus-4-6": "Claude Opus 4.6",
    "gpt-5.2-chat-latest-20260210": "GPT-5.2",
    "gpt-5.4-high": "GPT-5.4",
}
GAME1_DISPLAY = {
    "claude-3-haiku-20240307": "Claude 3 Haiku",
    "gpt-5-nano-high": "GPT-5 Nano",
    "qwen3-max-preview": "Qwen3 Max",
    "claude-opus-4-5-20251101": "Claude Opus 4.5",
    "gemini-3.1-pro": "Gemini 3.1 Pro",
}
FAMILY_DISPLAY = {
    "random_monoculture": "One-model\nmonoculture",
    "homogeneous_control": "GPT-5 Nano\ncontrol",
    "homogeneous_adversary": "One adversary +\nGPT-5 Nano",
    "heterogeneous_random": "Heterogeneous\nroster",
}
CATEGORY_ORDER = (
    "trade/compromise",
    "emotional persuasion",
    "logical persuasion",
    "pressure",
    "self-interest/exploitation",
    "formalization",
)
CATEGORY_DISPLAY = {
    "trade/compromise": "Trade",
    "emotional persuasion": "Emotion",
    "logical persuasion": "Logic",
    "pressure": "Pressure",
    "self-interest/exploitation": "Self-interest",
    "formalization": "Formalization",
}
VOTE_RE = re.compile(
    r"Proposal #\d+ accepted .*? with (\d+)/(\d+) accept votes"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def canonical(path: str | Path) -> str:
    return str(Path(path).resolve())


def save_figure(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D8DEE8", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def load_codebook() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = json.loads((MONO_ROOT / "llm_tag_codebook.json").read_text(encoding="utf-8"))
    if len(rows) != 50:
        raise ValueError(f"Expected 50 labels, found {len(rows)}")
    mapping = {str(row["tag_code"]): row for row in rows}
    if not SELECTED_23.issubset(mapping):
        raise ValueError("The selected-23 labels are not a subset of the codebook")
    return rows, mapping


def build_case_tables(codebook: dict[str, dict[str, Any]]) -> pd.DataFrame:
    events = pd.DataFrame(read_jsonl(MONO_ROOT / "llm_event_tags.jsonl"))
    events = events[events["config_id"].isin(CASE_CONFIGS)].copy()
    events["selected_23"] = events["tag_code"].isin(SELECTED_23)
    columns = [
        "config_id",
        "round",
        "discussion_turn",
        "phase",
        "log_index",
        "speaker_agent",
        "tag_code",
        "tag_title",
        "selected_23",
        "confidence",
        "quote",
        "rationale",
    ]
    events[columns].sort_values(
        ["config_id", "log_index", "tag_code"], na_position="last"
    ).to_csv(OUT / "three_config_turn_tag_map.csv", index=False)

    inventory = (
        events.groupby(["config_id", "tag_code", "tag_title", "selected_23"], as_index=False)
        .size()
        .rename(columns={"size": "event_count"})
    )
    inventory["category"] = inventory["tag_code"].map(
        lambda value: codebook[str(value)]["category"]
    )
    inventory = inventory[
        [
            "config_id",
            "tag_code",
            "tag_title",
            "category",
            "selected_23",
            "event_count",
        ]
    ].sort_values(["config_id", "category", "tag_code"])
    inventory.to_csv(OUT / "three_config_tag_inventory.csv", index=False)
    return inventory


def plot_case_tags(inventory: pd.DataFrame) -> None:
    def panel(ax: plt.Axes, frame: pd.DataFrame, title: str) -> None:
        order = (
            frame.groupby(["tag_title", "tag_code"], as_index=False)["event_count"]
            .sum()
            .sort_values(["event_count", "tag_title"], ascending=[False, True])
        )
        labels = order["tag_title"].tolist()
        matrix = (
            frame.pivot_table(
                index="tag_title",
                columns="config_id",
                values="event_count",
                aggfunc="sum",
                fill_value=0,
            )
            .reindex(index=labels, columns=CASE_CONFIGS, fill_value=0)
            .to_numpy()
        )
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(4, int(matrix.max())))
        ax.set_xticks(range(len(CASE_CONFIGS)), [value.replace("config_", "Config ") for value in CASE_CONFIGS])
        ax.set_yticks(range(len(labels)), labels)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = int(matrix[row, col])
                ax.text(
                    col,
                    row,
                    str(value) if value else "·",
                    ha="center",
                    va="center",
                    color="white" if value >= 3 else "#172B4D",
                    fontsize=8,
                    fontweight="bold" if value else "normal",
                )
        ax.tick_params(axis="x", labelrotation=28, labelsize=8)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        ax.tick_params(axis="y", labelsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 8.5), gridspec_kw={"wspace": 0.62})
    panel(axes[0], inventory, "All labels that fired from the 50-label codebook")
    panel(axes[1], inventory[inventory["selected_23"]], "Labels retained in the selected-23 subset")
    fig.suptitle("Saved qualitative labels for the three Gemini coalition cases", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "Cells show saved event counts. A dot means that the label did not fire.",
        ha="center",
        fontsize=9,
        color="#52606D",
    )
    save_figure(fig, "three_config_tag_counts")


def outcome_metrics(result_path: str, n_agents: int) -> dict[str, Any]:
    payload = json.loads(Path(result_path).read_text(encoding="utf-8"))
    threshold = math.ceil(2 * n_agents / 3)
    exact_threshold_vote = False
    accepted_vote_count: int | None = None
    for log in reversed(payload.get("conversation_logs", [])):
        match = VOTE_RE.search(str(log.get("content", "")))
        if match:
            accepted_vote_count = int(match.group(1))
            exact_threshold_vote = accepted_vote_count == threshold
            break

    utilities = payload.get("final_utilities") or {}
    utility_values = list(utilities.values()) if isinstance(utilities, dict) else list(utilities)
    zero_payoff_count = sum(float(value) <= 1e-9 for value in utility_values)
    allocation = payload.get("final_allocation") or {}
    empty_final_count = (
        sum(not bundle for bundle in allocation.values()) if isinstance(allocation, dict) else 0
    )

    proposal_count = 0
    exact_size_empty_proposal_count = 0
    for log in payload.get("conversation_logs", []):
        proposal = log.get("proposal") if log.get("phase") == "proposal" else None
        if not isinstance(proposal, dict):
            continue
        proposal_allocation = proposal.get("allocation") or {}
        if not isinstance(proposal_allocation, dict):
            continue
        proposal_count += 1
        nonempty = sum(bool(bundle) for bundle in proposal_allocation.values())
        exact_size_empty_proposal_count += int(nonempty == threshold)

    return {
        "accepted_vote_count": accepted_vote_count,
        "threshold": threshold,
        "exact_threshold_vote": exact_threshold_vote,
        "zero_payoff_count": zero_payoff_count,
        "empty_final_count": empty_final_count,
        "exact_threshold_and_zero": exact_threshold_vote and zero_payoff_count > 0,
        "exact_threshold_and_empty": exact_threshold_vote and empty_final_count > 0,
        "proposal_count": proposal_count,
        "exact_size_empty_proposal_count": exact_size_empty_proposal_count,
        "any_exact_size_empty_proposal": exact_size_empty_proposal_count > 0,
    }


def load_game1_records() -> tuple[list[dict[str, Any]], dict[str, set[str]], dict[tuple[str, str], set[str]]]:
    records: list[dict[str, Any]] = []
    event_tags: dict[str, set[str]] = defaultdict(set)
    speaker_event_tags: dict[tuple[str, str], set[str]] = defaultdict(set)
    sources = (
        ("random_monoculture", MONO_ROOT),
        (None, MULTI_ROOT),
    )
    for fixed_family, source in sources:
        manifests = read_jsonl(source / "all_rollouts_manifest.jsonl")
        for row in manifests:
            family = fixed_family or str(row["experiment_family"])
            if row.get("game_label") != "game1" or int(row.get("n_agents", 0)) < 4:
                continue
            if family not in FAMILY_DISPLAY:
                continue
            copy = dict(row)
            copy["analysis_family"] = family
            copy["result_path"] = canonical(copy["result_path"])
            records.append(copy)
        for event in read_jsonl(source / "llm_event_tags.jsonl"):
            path = canonical(event["result_path"])
            tag = str(event["tag_code"])
            event_tags[path].add(tag)
            if event.get("speaker_agent") is not None:
                speaker_event_tags[(path, str(event["speaker_agent"]))].add(tag)
    return records, event_tags, speaker_event_tags


def build_family_prevalence(
    records: list[dict[str, Any]], event_tags: dict[str, set[str]]
) -> pd.DataFrame:
    detail_rows: list[dict[str, Any]] = []
    for row in records:
        metrics = outcome_metrics(row["result_path"], int(row["n_agents"]))
        tags = event_tags[row["result_path"]]
        detail_rows.append(
            {
                "analysis_family": row["analysis_family"],
                "result_path": row["result_path"],
                "n_agents": int(row["n_agents"]),
                "any_coalition_label": bool(tags & COALITION_TAGS),
                "any_core_coalition_label": bool(tags & CORE_COALITION_TAGS),
                **{tag: tag in tags for tag in CORE_COALITION_TAGS},
                **metrics,
            }
        )
    detail = pd.DataFrame(detail_rows)
    detail.to_csv(OUT / "game1_nge4_run_level_coalition_metrics.csv", index=False)

    rows: list[dict[str, Any]] = []
    for family, group in detail.groupby("analysis_family", sort=False):
        attempts = group["any_exact_size_empty_proposal"]
        rows.append(
            {
                "analysis_family": family,
                "transcript_count": len(group),
                "any_coalition_label_pct": 100 * group["any_coalition_label"].mean(),
                "any_core_coalition_label_pct": 100 * group["any_core_coalition_label"].mean(),
                "named_microcoalition_slate_pct": 100 * group["named_microcoalition_slate"].mean(),
                "vote_bloc_counting_pct": 100 * group["vote_bloc_counting"].mean(),
                "holdout_bypass_minimum_coalition_pct": 100
                * group["holdout_bypass_minimum_coalition"].mean(),
                "exact_threshold_vote_pct": 100 * group["exact_threshold_vote"].mean(),
                "any_zero_payoff_pct": 100 * group["zero_payoff_count"].gt(0).mean(),
                "exact_threshold_and_zero_pct": 100
                * group["exact_threshold_and_zero"].mean(),
                "any_empty_final_pct": 100 * group["empty_final_count"].gt(0).mean(),
                "exact_threshold_and_empty_pct": 100
                * group["exact_threshold_and_empty"].mean(),
                "any_exact_size_empty_proposal_pct": 100 * attempts.mean(),
                "proposal_count": int(group["proposal_count"].sum()),
                "exact_size_empty_proposal_count": int(
                    group["exact_size_empty_proposal_count"].sum()
                ),
                "exact_size_empty_proposal_pct": 100
                * group["exact_size_empty_proposal_count"].sum()
                / group["proposal_count"].sum(),
                "attempt_to_exact_empty_conversion_pct": (
                    100
                    * group.loc[attempts, "exact_threshold_and_empty"].mean()
                    if attempts.any()
                    else np.nan
                ),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "game1_nge4_family_coalition_prevalence.csv", index=False)
    return detail


def plot_family_prevalence() -> None:
    frame = pd.read_csv(OUT / "game1_nge4_family_coalition_prevalence.csv").set_index(
        "analysis_family"
    )
    families = list(FAMILY_DISPLAY)
    x = np.arange(len(families))
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2), gridspec_kw={"wspace": 0.28})

    left_metrics = (
        ("named_microcoalition_slate_pct", "Named coalition"),
        ("vote_bloc_counting_pct", "Vote-bloc counting"),
        ("holdout_bypass_minimum_coalition_pct", "Holdout bypass"),
    )
    colors = ("#4C78A8", "#F58518", "#E45756")
    for index, ((column, label), color) in enumerate(zip(left_metrics, colors, strict=True)):
        axes[0].bar(x + (index - 1) * width, frame.loc[families, column], width, label=label, color=color)
    axes[0].set_xticks(x, [FAMILY_DISPLAY[value] for value in families])
    axes[0].set_ylabel("Transcripts with label (%)")
    axes[0].set_title("(a) Coalition labels in public transcripts", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8)
    style_axis(axes[0])

    right_metrics = (
        ("any_exact_size_empty_proposal_pct", "Exact-size empty-bundle proposal"),
        ("exact_threshold_and_empty_pct", "Passed at threshold + empty bundle"),
        ("exact_threshold_and_zero_pct", "Passed at threshold + zero payoff"),
    )
    for index, ((column, label), color) in enumerate(zip(right_metrics, colors, strict=True)):
        axes[1].bar(x + (index - 1) * width, frame.loc[families, column], width, label=label, color=color)
    axes[1].set_xticks(x, [FAMILY_DISPLAY[value] for value in families])
    axes[1].set_ylabel("Runs (%)")
    axes[1].set_title("(b) Structural attempts and realized outcomes", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, fontsize=8)
    style_axis(axes[1])
    fig.suptitle("Coalition behavior in Game 1 with at least four agents", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        -0.02,
        "Descriptive comparison across separate datasets; preferences, seeds, and model rosters are not matched.",
        ha="center",
        fontsize=9,
        color="#52606D",
    )
    save_figure(fig, "game1_family_coalition_attempts_and_outcomes")


def build_monoculture_model_table(
    records: list[dict[str, Any]],
    event_tags: dict[str, set[str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    mono = [row for row in records if row["analysis_family"] == "random_monoculture"]
    for model in GAME1_DISPLAY:
        group = [row for row in mono if row.get("monoculture_model") == model]
        if not group:
            continue
        outcomes = [outcome_metrics(row["result_path"], int(row["n_agents"])) for row in group]
        rows.append(
            {
                "model": model,
                "model_display": GAME1_DISPLAY[model],
                "transcript_count": len(group),
                "any_core_coalition_label_count": sum(
                    bool(event_tags[row["result_path"]] & CORE_COALITION_TAGS) for row in group
                ),
                "any_core_coalition_label_pct": 100
                * np.mean(
                    [bool(event_tags[row["result_path"]] & CORE_COALITION_TAGS) for row in group]
                ),
                "named_microcoalition_slate_pct": 100
                * np.mean(
                    ["named_microcoalition_slate" in event_tags[row["result_path"]] for row in group]
                ),
                "vote_bloc_counting_pct": 100
                * np.mean(["vote_bloc_counting" in event_tags[row["result_path"]] for row in group]),
                "holdout_bypass_minimum_coalition_pct": 100
                * np.mean(
                    [
                        "holdout_bypass_minimum_coalition" in event_tags[row["result_path"]]
                        for row in group
                    ]
                ),
                "any_exact_size_empty_proposal_count": sum(
                    item["any_exact_size_empty_proposal"] for item in outcomes
                ),
                "any_exact_size_empty_proposal_pct": 100
                * np.mean([item["any_exact_size_empty_proposal"] for item in outcomes]),
                "exact_threshold_and_zero_count": sum(
                    item["exact_threshold_and_zero"] for item in outcomes
                ),
                "exact_threshold_and_zero_pct": 100
                * np.mean([item["exact_threshold_and_zero"] for item in outcomes]),
            }
        )
    output = pd.DataFrame(rows)
    output.to_csv(OUT / "game1_monoculture_model_coalition_rates.csv", index=False)
    return output


def plot_monoculture_models(frame: pd.DataFrame) -> None:
    x = np.arange(len(frame))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    metrics = (
        ("any_core_coalition_label_pct", "Any core coalition label", "#4C78A8"),
        ("any_exact_size_empty_proposal_pct", "Exact-size empty-bundle proposal", "#F58518"),
        ("exact_threshold_and_zero_pct", "Passed at threshold + zero payoff", "#E45756"),
    )
    for index, (column, label, color) in enumerate(metrics):
        ax.bar(x + (index - 1) * width, frame[column], width, label=label, color=color)
    ax.set_xticks(x, frame["model_display"])
    ax.set_ylabel("Runs (%)")
    ax.set_title(
        "Game 1 one-model monocultures, N ≥ 4",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper left")
    style_axis(ax)
    fig.text(
        0.5,
        -0.01,
        "Each model has 20 single-seed cells. Percentages are descriptive.",
        ha="center",
        fontsize=9,
        color="#52606D",
    )
    save_figure(fig, "game1_monoculture_model_coalition_rates")


def build_gemini_context_table(
    records: list[dict[str, Any]],
    speaker_event_tags: dict[tuple[str, str], set[str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for context in ("random_monoculture", "heterogeneous_random"):
        eligible: list[dict[str, Any]] = []
        for row in records:
            if row["analysis_family"] != context:
                continue
            agents = [
                agent
                for agent, model in row["agent_model_map"].items()
                if model == "gemini-3.1-pro"
            ]
            if not agents:
                continue
            gemini_tags: set[str] = set()
            for agent in agents:
                gemini_tags.update(speaker_event_tags[(row["result_path"], agent)])
            copy = dict(row)
            copy["gemini_core_attempt"] = bool(gemini_tags & CORE_COALITION_TAGS)
            copy["gemini_holdout_bypass"] = "holdout_bypass_minimum_coalition" in gemini_tags
            copy.update(outcome_metrics(row["result_path"], int(row["n_agents"])))
            eligible.append(copy)

        attempts = [row for row in eligible if row["gemini_core_attempt"]]
        holdout_attempts = [row for row in eligible if row["gemini_holdout_bypass"]]
        rows.append(
            {
                "context": context,
                "transcript_count": len(eligible),
                "gemini_core_attempt_count": len(attempts),
                "gemini_core_attempt_pct": 100 * len(attempts) / len(eligible),
                "exact_threshold_and_zero_count": sum(
                    row["exact_threshold_and_zero"] for row in eligible
                ),
                "exact_threshold_and_zero_pct": 100
                * np.mean([row["exact_threshold_and_zero"] for row in eligible]),
                "successes_among_core_attempts": sum(
                    row["exact_threshold_and_zero"] for row in attempts
                ),
                "success_pct_among_core_attempts": 100
                * np.mean([row["exact_threshold_and_zero"] for row in attempts]),
                "gemini_holdout_bypass_count": len(holdout_attempts),
                "successes_among_holdout_bypass": sum(
                    row["exact_threshold_and_zero"] for row in holdout_attempts
                ),
                "success_pct_among_holdout_bypass": 100
                * np.mean([row["exact_threshold_and_zero"] for row in holdout_attempts])
                if holdout_attempts
                else np.nan,
            }
        )
    output = pd.DataFrame(rows)
    output.to_csv(OUT / "gemini_monoculture_vs_heterogeneous_conversion.csv", index=False)
    return output


def plot_gemini_context(frame: pd.DataFrame) -> None:
    frame = frame.set_index("context").loc[["random_monoculture", "heterogeneous_random"]]
    labels = ["Gemini\nmonoculture", "Gemini in\nheterogeneous rosters"]
    x = np.arange(2)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), gridspec_kw={"wspace": 0.35})
    axes[0].bar(x, frame["gemini_core_attempt_pct"], color=["#4C78A8", "#72A0C1"])
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Transcripts with a Gemini core-coalition label (%)")
    axes[0].set_title("(a) Public coalition attempts", loc="left", fontweight="bold")
    style_axis(axes[0])
    for index, value in enumerate(frame["gemini_core_attempt_pct"]):
        axes[0].text(index, value + 1.2, f"{value:.1f}%", ha="center", fontsize=10)

    axes[1].bar(x, frame["success_pct_among_core_attempts"], color=["#E45756", "#F2A6A4"])
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Labeled attempts ending at threshold with a zero payoff (%)")
    axes[1].set_title("(b) Realized exclusion proxy", loc="left", fontweight="bold")
    style_axis(axes[1])
    for index, row in enumerate(frame.itertuples()):
        value = row.success_pct_among_core_attempts
        axes[1].text(
            index,
            value + 1.2,
            f"{row.successes_among_core_attempts}/{row.gemini_core_attempt_count}",
            ha="center",
            fontsize=10,
        )
    fig.suptitle("Gemini coalition attempts appear in both contexts, but outcomes differ", fontsize=14, fontweight="bold")
    fig.text(
        0.5,
        -0.02,
        "The success proxy requires an exact-threshold vote and at least one zero-payoff agent; cohorts are not matched.",
        ha="center",
        fontsize=9,
        color="#52606D",
    )
    save_figure(fig, "gemini_monoculture_vs_heterogeneous_conversion")


def load_bilateral_denominators() -> pd.DataFrame:
    primary = pd.read_csv(PRIMARY_BILATERAL)
    primary = primary[primary["baseline_key"].eq("gpt5_nano")].copy()
    primary["result_path"] = primary["result_path"].map(canonical)
    paths = set(primary["result_path"])
    manifests = pd.DataFrame(read_jsonl(BILATERAL_ROOT / "all_rollouts_manifest.jsonl"))
    manifests["result_path"] = manifests["result_path"].map(canonical)
    manifests = manifests[manifests["result_path"].isin(paths)].copy()
    rows: list[dict[str, Any]] = []
    for row in manifests.to_dict("records"):
        adversaries = [
            agent for agent, role in row["agent_role_map"].items() if role == "adversary"
        ]
        if len(adversaries) != 1:
            raise ValueError(f"Unexpected adversary map for {row['result_path']}")
        rows.append(
            {
                "result_path": row["result_path"],
                "speaker_agent": adversaries[0],
                "speaker_model": row["adversary_model"],
            }
        )
    output = pd.DataFrame(rows)
    if len(output) != 1500 or output["speaker_model"].nunique() != 30:
        raise ValueError("Unexpected bilateral denominator shape")
    return output


def build_bilateral_focus_tables(codebook: dict[str, dict[str, Any]]) -> None:
    for scope in ("all50", "selected23"):
        source = pd.read_csv(
            COMBINED_ROOT / f"bilateral1500_{scope}_mean_events_by_elo.csv"
        )
        source = source[source["speaker_model"].isin(FOCUS_MODELS)].copy()
        source["model_display"] = source["speaker_model"].map(FOCUS_DISPLAY)
        source.to_csv(OUT / f"bilateral_focus_category_means_{scope}.csv", index=False)

    denoms = load_bilateral_denominators()
    focus = denoms[denoms["speaker_model"].isin(FOCUS_MODELS)].copy()
    events = pd.DataFrame(read_jsonl(BILATERAL_ROOT / "llm_event_tags.jsonl"))
    events["result_path"] = events["result_path"].map(canonical)
    events = events.merge(
        focus[["result_path", "speaker_agent", "speaker_model"]].rename(
            columns={"speaker_model": "denominator_model"}
        ),
        on=["result_path", "speaker_agent"],
        how="inner",
        validate="many_to_one",
    )
    if not events["speaker_model"].eq(events["denominator_model"]).all():
        raise ValueError("Bilateral event speaker model does not match the denominator")
    presence = events[["result_path", "denominator_model", "tag_code"]].rename(
        columns={"denominator_model": "speaker_model"}
    ).drop_duplicates()
    rows: list[dict[str, Any]] = []
    for model in FOCUS_MODELS:
        model_paths = set(focus.loc[focus["speaker_model"].eq(model), "result_path"])
        if len(model_paths) != 50:
            raise ValueError(f"Expected 50 bilateral paths for {model}, found {len(model_paths)}")
        model_events = presence[presence["speaker_model"].eq(model)]
        for tag_code, details in codebook.items():
            count = int(model_events[model_events["tag_code"].eq(tag_code)]["result_path"].nunique())
            rows.append(
                {
                    "speaker_model": model,
                    "model_display": FOCUS_DISPLAY[model],
                    "tag_code": tag_code,
                    "tag_title": details["tag_title"],
                    "category": details["category"],
                    "selected_23": tag_code in SELECTED_23,
                    "transcript_count": 50,
                    "transcripts_with_label": count,
                    "prevalence_pct": 2 * count,
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "bilateral_focus_label_prevalence.csv", index=False)


def plot_bilateral_categories() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 5.7), gridspec_kw={"wspace": 0.10})
    for panel_index, (ax, scope, title) in enumerate(zip(
        axes,
        ("all50", "selected23"),
        ("(a) All 50 labels", "(b) Selected 23 labels"),
        strict=True,
    )):
        frame = pd.read_csv(OUT / f"bilateral_focus_category_means_{scope}.csv")
        matrix = (
            frame.pivot(index="speaker_model", columns="category", values="mean_events_per_rollout")
            .reindex(index=FOCUS_MODELS, columns=CATEGORY_ORDER)
            .to_numpy()
        )
        image = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=2.2, aspect="auto")
        ax.set_xticks(range(len(CATEGORY_ORDER)), [CATEGORY_DISPLAY[value] for value in CATEGORY_ORDER])
        ax.set_yticks(
            range(len(FOCUS_MODELS)),
            [FOCUS_DISPLAY[value] for value in FOCUS_MODELS]
            if panel_index == 0
            else [""] * len(FOCUS_MODELS),
        )
        ax.set_title(title, loc="left", fontweight="bold")
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=9)
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                ax.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value >= 1.35 else "#172B4D",
                )
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.colorbar(image, ax=axes, shrink=0.78, label="Mean turn-deduplicated category events per rollout")
    fig.suptitle("Bilateral model dispositions across 50 matched rollouts per model", fontsize=15, fontweight="bold")
    save_figure(fig, "bilateral_focus_model_category_dispositions")


def plot_bilateral_labels() -> None:
    frame = pd.read_csv(OUT / "bilateral_focus_label_prevalence.csv")
    frame = frame[frame["selected_23"]].copy()
    order = (
        frame.groupby("tag_title", as_index=False)["prevalence_pct"]
        .mean()
        .sort_values(["prevalence_pct", "tag_title"], ascending=[False, True])["tag_title"]
        .tolist()
    )
    matrix = (
        frame.pivot(index="tag_title", columns="speaker_model", values="prevalence_pct")
        .reindex(index=order, columns=FOCUS_MODELS)
        .to_numpy()
    )
    fig, ax = plt.subplots(figsize=(10.8, 9.0))
    image = ax.imshow(matrix, cmap="Purples", vmin=0, vmax=70, aspect="auto")
    ax.set_xticks(range(len(FOCUS_MODELS)), [FOCUS_DISPLAY[value] for value in FOCUS_MODELS])
    ax.set_yticks(range(len(order)), order)
    ax.tick_params(axis="x", labelrotation=25, labelsize=9)
    ax.tick_params(axis="y", labelsize=8)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax.text(
                col,
                row,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if value >= 42 else "#172B4D",
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(image, ax=ax, shrink=0.75, label="Rollouts with label (%)")
    ax.set_title(
        "Selected-23 label prevalence in bilateral bargaining",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Each model has 50 rollouts against GPT-5 Nano. Coalition labels are absent from the selected-23 codebook.",
        ha="center",
        fontsize=9,
        color="#52606D",
    )
    save_figure(fig, "bilateral_focus_selected23_label_prevalence")


def write_provenance() -> None:
    payload = {
        "created": "2026-08-16",
        "selected_23_count": len(SELECTED_23),
        "case_configs": list(CASE_CONFIGS),
        "core_coalition_tags": sorted(CORE_COALITION_TAGS),
        "success_proxy": (
            "The selected proposal passed with exactly ceil(2N/3) accepts and at least "
            "one agent had final utility <= 0."
        ),
        "attempt_proxy": (
            "A candidate proposal assigned a nonempty bundle to exactly ceil(2N/3) agents."
        ),
        "sources": {
            "random_monoculture": str(MONO_ROOT),
            "multiagent": str(MULTI_ROOT),
            "bilateral": str(BILATERAL_ROOT),
            "bilateral_category_aggregates": str(COMBINED_ROOT),
            "bilateral_primary_denominator": str(PRIMARY_BILATERAL),
        },
        "limitations": [
            "The qualitative adjudication includes public discussion, proposal reasoning, and formal outcomes, but not private thinking or vote reasoning.",
            "The selected-23 subset excludes all nine coalition-category labels.",
            "Family comparisons use separate datasets with different seeds, preferences, and model rosters.",
            "The structural success proxy does not prove that a saved coalition label caused the outcome.",
            "Each random-monoculture model has one seed per Game 1 N-by-competition cell.",
        ],
    }
    (OUT / "provenance.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _, codebook = load_codebook()
    case_inventory = build_case_tables(codebook)
    plot_case_tags(case_inventory)

    records, event_tags, speaker_event_tags = load_game1_records()
    build_family_prevalence(records, event_tags)
    plot_family_prevalence()

    model_table = build_monoculture_model_table(records, event_tags)
    plot_monoculture_models(model_table)

    gemini_context = build_gemini_context_table(records, speaker_event_tags)
    plot_gemini_context(gemini_context)

    build_bilateral_focus_tables(codebook)
    plot_bilateral_categories()
    plot_bilateral_labels()
    write_provenance()


if __name__ == "__main__":
    main()

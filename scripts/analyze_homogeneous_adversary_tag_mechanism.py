#!/usr/bin/env python3
"""Analyze homogeneous-adversary outcome mechanisms using strategic tags."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUT_DIR = Path("analysis/llm_strategic_tag_adjudication_20260628")
DEFAULT_HOT_TAGS_JSON = Path("strategic_tag_review_final.json")
DEFAULT_OUT_DIR = Path("analysis/homogeneous_adversary_tag_mechanism_20260629")

FAMILY = "homogeneous_adversary"
MODEL_LABELS = {
    "amazon-nova-micro-v1.0": "Nova Micro",
    "gpt-4o-mini-2024-07-18": "GPT-4o mini",
    "claude-sonnet-4-20250514": "Claude Sonnet 4",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gpt-5.4-high": "GPT-5.4 High",
}
OUTCOME_LABELS = {
    "baseline_gini": "Baseline-only corrected Gini",
    "all_gini": "All-agent corrected Gini",
    "baseline_std": "Baseline payoff SD",
    "baseline_mean_payoff": "Mean baseline payoff",
    "adversary_payoff": "Adversary payoff",
    "adversary_minus_baseline": "Adversary minus baseline mean",
    "total_payoff": "Total group payoff",
    "mean_payoff": "Mean group payoff",
}
KEY_TAGS = [
    "conditional_veto_threat",
    "counter_anchor_cost_policing",
    "cross_agent_conflict_mapping",
    "named_microcoalition_slate",
    "threshold_gap_calculation",
    "adversarial_callout",
    "fairness_ledger_argument",
    "conditional_quid_pro_quo",
    "concession_laddering",
    "empathy_then_pivot",
]
GROUP_DISPLAY_LABELS = {
    "pressure": "leverage-based bargaining",
}
CURATED_QUALITATIVE_EXAMPLES = [
    {
        "mechanism": "Conditional leverage",
        "model": "GPT-5.4 High",
        "tag": "conditional_veto_threat",
        "example": "I can support a limited, clearly defined baseline, but 18-25% is not workable from my side.",
        "interpretation": "The adversary does not merely ask for more; it defines an explicit acceptability corridor.",
    },
    {
        "mechanism": "Cost policing",
        "model": "GPT-5.4 High",
        "tag": "counter_anchor_cost_policing",
        "example": "I hear the case for 60% carbon, but that is too far for me.",
        "interpretation": "The adversary rejects anchors by making their cost to itself legible.",
    },
    {
        "mechanism": "Coalition accounting",
        "model": "GPT-5.4 High",
        "tag": "threshold_gap_calculation",
        "example": "Agent_2 and I are close; Agent_3 is within reach if we stay in the upper band.",
        "interpretation": "The adversary maps a feasible winning path rather than negotiating pairwise in isolation.",
    },
    {
        "mechanism": "Specificity enforcement",
        "model": "GPT-5.4 High",
        "tag": "adversarial_callout",
        "example": "A vague 'balanced package' is not enough.",
        "interpretation": "Vague fairness language is forced into concrete proposal structure.",
    },
    {
        "mechanism": "Tradeable concession",
        "model": "GPT-5.4 High",
        "tag": "conditional_quid_pro_quo",
        "example": "If giving Clock to Agent_2 is what locks Camera there and keeps Stone/Quill fully uncontested for me, I can support that.",
        "interpretation": "The adversary concedes locally when the concession secures a higher-value package.",
    },
    {
        "mechanism": "Diffuse low-leverage ask",
        "model": "Nova Micro",
        "tag": "self_advocacy_value_maximization",
        "example": "If you can secure Globe or Map for me, I'm willing to participate in a multi-party route.",
        "interpretation": "The weaker adversary expresses preference without anchoring the global settlement.",
    },
    {
        "mechanism": "Erroneous accounting",
        "model": "Nova Micro",
        "tag": "budget_carryover_hallucination",
        "example": "Dog Park only needs 2 more units to be fully funded.",
        "interpretation": "Low-quality structure can misstate the game state and fail to coordinate the group.",
    },
    {
        "mechanism": "Capitulation",
        "model": "GPT-4o mini",
        "tag": "accepted_loss_capitulation",
        "example": "Consensus accepted with final utility Agent_6=-27.0.",
        "interpretation": "Some weaker adversary runs end in acceptance of a clearly bad position.",
    },
]


def display_group(group: str) -> str:
    return GROUP_DISPLAY_LABELS.get(str(group), str(group))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def gini_shifted_corrected(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan
    if float(arr.min()) < 0.0:
        arr = arr - float(arr.min())
    if arr.size < 2 or np.allclose(arr, arr[0]) or np.allclose(arr, 0.0):
        return 0.0
    mean_value = float(arr.mean())
    if math.isclose(mean_value, 0.0):
        return 0.0
    raw_gini = float(np.mean(np.abs(arr[:, None] - arr[None, :])) / (2.0 * mean_value))
    return min(raw_gini * float(arr.size / (arr.size - 1)), 1.0)


def sem(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) <= 1:
        return 0.0
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


def safe_corr(x: pd.Series, y: pd.Series, method: str = "spearman") -> tuple[float, float, int]:
    clean = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(clean) < 4 or clean["x"].nunique() < 2 or clean["y"].nunique() < 2:
        return math.nan, math.nan, len(clean)
    if method == "pearson":
        r, p = stats.pearsonr(clean["x"], clean["y"])
    else:
        r, p = stats.spearmanr(clean["x"], clean["y"])
    return float(r), float(p), len(clean)


def load_hot_tags(path: Path, codebook: pd.DataFrame) -> list[str]:
    if path.exists():
        obj = json.loads(path.read_text())
        if isinstance(obj, dict):
            rows = obj.get("hot_tags") or obj.get("responses") or obj.get("tags") or []
        else:
            rows = obj
        tags = []
        for row in rows:
            if isinstance(row, dict):
                if row.get("decision") and row.get("decision") != "hot":
                    continue
                code = row.get("tag_code") or row.get("code")
            else:
                code = str(row)
            if code:
                tags.append(str(code))
        if tags:
            valid = set(codebook["tag_code"])
            return [t for t in dict.fromkeys(tags) if t in valid]
    return codebook["tag_code"].head(29).tolist()


def build_outcome_frame(input_dir: Path) -> pd.DataFrame:
    rows = []
    for rollout in load_jsonl(input_dir / "all_rollouts_manifest.jsonl"):
        if rollout.get("experiment_family") != FAMILY:
            continue
        payload = json.loads(Path(rollout["result_path"]).read_text())
        utilities_raw = payload.get("final_utilities") or {}
        if not utilities_raw:
            continue
        role_map = rollout.get("agent_role_map") or {}
        utility_by_agent = {agent: float(value) for agent, value in utilities_raw.items() if value is not None}
        adv_agents = [agent for agent, role in role_map.items() if role == "adversary" and agent in utility_by_agent]
        base_agents = [agent for agent, role in role_map.items() if role != "adversary" and agent in utility_by_agent]
        if not adv_agents or not base_agents:
            continue
        adv_values = [utility_by_agent[a] for a in adv_agents]
        base_values = [utility_by_agent[a] for a in base_agents]
        all_values = list(utility_by_agent.values())
        adversary_elo = float(next(v for a, v in rollout["agent_elo_map"].items() if a in adv_agents and v is not None))
        adversary_model = str(rollout["adversary_model"])
        rows.append(
            {
                "result_path": rollout["result_path"],
                "config_id": str(rollout["config_id"]),
                "game_label": rollout["game_label"],
                "n_agents": int(rollout["n_agents"]),
                "competition_level": float(rollout.get("competition_level")),
                "adversary_position": rollout.get("adversary_position"),
                "adversary_model": adversary_model,
                "adversary_model_label": MODEL_LABELS.get(adversary_model, adversary_model),
                "adversary_elo": adversary_elo,
                "consensus_reached": bool(rollout.get("consensus_reached")),
                "final_round": rollout.get("final_round"),
                "conversation_log_count": rollout.get("conversation_log_count"),
                "adversary_agent": adv_agents[0],
                "adversary_payoff": float(np.mean(adv_values)),
                "baseline_mean_payoff": float(np.mean(base_values)),
                "baseline_total_payoff": float(np.sum(base_values)),
                "total_payoff": float(np.sum(all_values)),
                "mean_payoff": float(np.mean(all_values)),
                "adversary_minus_baseline": float(np.mean(adv_values) - np.mean(base_values)),
                "baseline_std": float(np.std(base_values, ddof=0)),
                "all_std": float(np.std(all_values, ddof=0)),
                "baseline_gini": gini_shifted_corrected(base_values),
                "all_gini": gini_shifted_corrected(all_values),
                "min_baseline_payoff": float(np.min(base_values)),
                "max_baseline_payoff": float(np.max(base_values)),
            }
        )
    out = pd.DataFrame(rows)
    out["design_cell"] = (
        out["game_label"].astype(str)
        + "_n"
        + out["n_agents"].astype(str)
        + "_c"
        + out["competition_level"].astype(str)
        + "_"
        + out["adversary_position"].astype(str)
    )
    return out.sort_values(["adversary_elo", "game_label", "n_agents", "competition_level", "adversary_position"])


def build_event_frames(input_dir: Path, codebook: pd.DataFrame, hot_tags: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    code_to_group = dict(zip(codebook["tag_code"], codebook["category"]))
    code_to_title = dict(zip(codebook["tag_code"], codebook["tag_title"]))
    rows = []
    for row in load_jsonl(input_dir / "llm_event_tags.jsonl"):
        if row.get("experiment_family") != FAMILY:
            continue
        if row.get("speaker_role") != "adversary":
            continue
        if row.get("tag_code") not in hot_tags:
            continue
        row = dict(row)
        row["group"] = code_to_group.get(row["tag_code"])
        row["tag_title"] = code_to_title.get(row["tag_code"], row.get("tag_title"))
        rows.append(row)
    events = pd.DataFrame(rows)
    if events.empty:
        return events, pd.DataFrame(), pd.DataFrame()
    tag_counts = (
        events.groupby(["result_path", "tag_code", "tag_title", "group"], dropna=False)
        .size()
        .reset_index(name="event_count")
    )
    group_counts = (
        events.groupby(["result_path", "group"], dropna=False)
        .size()
        .reset_index(name="event_count")
    )
    return events, tag_counts, group_counts


def add_count_columns(outcomes: pd.DataFrame, counts: pd.DataFrame, unit_col: str, units: list[str], prefix: str) -> pd.DataFrame:
    out = outcomes[["result_path"]].copy()
    for unit in units:
        out[f"{prefix}:{unit}"] = 0
    if not counts.empty:
        pivot = counts.pivot_table(index="result_path", columns=unit_col, values="event_count", aggfunc="sum", fill_value=0)
        pivot.columns = [f"{prefix}:{c}" for c in pivot.columns]
        out = out.set_index("result_path")
        out.update(pivot)
        out = out.reset_index()
    return out


def model_summary(outcomes: pd.DataFrame, tag_counts: pd.DataFrame, group_counts: pd.DataFrame, hot_tags: list[str], groups: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = (
        outcomes.groupby(["adversary_model", "adversary_model_label", "adversary_elo"], as_index=False)
        .agg(
            n_runs=("result_path", "nunique"),
            baseline_gini=("baseline_gini", "mean"),
            all_gini=("all_gini", "mean"),
            baseline_std=("baseline_std", "mean"),
            baseline_mean_payoff=("baseline_mean_payoff", "mean"),
            adversary_payoff=("adversary_payoff", "mean"),
            adversary_minus_baseline=("adversary_minus_baseline", "mean"),
            total_payoff=("total_payoff", "mean"),
            mean_payoff=("mean_payoff", "mean"),
            final_round=("final_round", "mean"),
            conversation_log_count=("conversation_log_count", "mean"),
        )
        .sort_values("adversary_elo")
    )

    tag_grid = outcomes[["result_path", "adversary_model", "adversary_model_label", "adversary_elo"]].merge(
        pd.DataFrame({"tag_code": hot_tags}), how="cross"
    )
    tag_freq = (
        tag_grid.merge(tag_counts[["result_path", "tag_code", "event_count"]], how="left")
        .fillna({"event_count": 0})
        .groupby(["adversary_model", "adversary_model_label", "adversary_elo", "tag_code"], as_index=False)
        .agg(
            event_count=("event_count", "sum"),
            runs_with_tag=("event_count", lambda s: int((s > 0).sum())),
            n_runs=("result_path", "nunique"),
        )
    )
    tag_freq["events_per_run"] = tag_freq["event_count"] / tag_freq["n_runs"]
    tag_freq["occurrence_rate"] = tag_freq["runs_with_tag"] / tag_freq["n_runs"]

    group_grid = outcomes[["result_path", "adversary_model", "adversary_model_label", "adversary_elo"]].merge(
        pd.DataFrame({"group": groups}), how="cross"
    )
    group_freq = (
        group_grid.merge(group_counts[["result_path", "group", "event_count"]], how="left")
        .fillna({"event_count": 0})
        .groupby(["adversary_model", "adversary_model_label", "adversary_elo", "group"], as_index=False)
        .agg(
            event_count=("event_count", "sum"),
            runs_with_group=("event_count", lambda s: int((s > 0).sum())),
            n_runs=("result_path", "nunique"),
        )
    )
    group_freq["events_per_run"] = group_freq["event_count"] / group_freq["n_runs"]
    group_freq["occurrence_rate"] = group_freq["runs_with_group"] / group_freq["n_runs"]
    return summary, tag_freq, group_freq


def model_tag_outcome_correlations(freq: pd.DataFrame, summary: pd.DataFrame, unit_col: str, outcomes: list[str]) -> pd.DataFrame:
    rows = []
    merged = freq.merge(summary, on=["adversary_model", "adversary_model_label", "adversary_elo", "n_runs"], how="left")
    for unit, sub in merged.groupby(unit_col):
        for metric in ["events_per_run", "occurrence_rate"]:
            for outcome in outcomes:
                r, p, n = safe_corr(sub[metric], sub[outcome], method="spearman")
                rows.append(
                    {
                        unit_col: unit,
                        "metric": metric,
                        "outcome": outcome,
                        "spearman_r": r,
                        "p_value": p,
                        "n_models": n,
                    }
                )
    return pd.DataFrame(rows)


def residualize_by_cell(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in cols:
        out[f"{col}_resid"] = out[col] - out.groupby("design_cell")[col].transform("mean")
    return out


def run_level_correlations(outcomes: pd.DataFrame, counts: pd.DataFrame, unit_col: str, units: list[str], outcome_cols: list[str]) -> pd.DataFrame:
    rows = []
    base = outcomes[["result_path", "design_cell", *outcome_cols]].copy()
    for unit in units:
        c = counts[counts[unit_col] == unit][["result_path", "event_count"]] if not counts.empty else pd.DataFrame(columns=["result_path", "event_count"])
        sub = base.merge(c, on="result_path", how="left").fillna({"event_count": 0})
        sub = residualize_by_cell(sub, ["event_count", *outcome_cols])
        for outcome in outcome_cols:
            r_raw, p_raw, n_raw = safe_corr(sub["event_count"], sub[outcome])
            r_resid, p_resid, n_resid = safe_corr(sub["event_count_resid"], sub[f"{outcome}_resid"])
            rows.append(
                {
                    unit_col: unit,
                    "outcome": outcome,
                    "spearman_event_count_r": r_raw,
                    "spearman_event_count_p": p_raw,
                    "spearman_event_count_n": n_raw,
                    "within_cell_resid_r": r_resid,
                    "within_cell_resid_p": p_resid,
                    "within_cell_resid_n": n_resid,
                    "mean_event_count": float(sub["event_count"].mean()),
                    "used_rate": float((sub["event_count"] > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def outcome_trend_table(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in OUTCOME_LABELS:
        r, p, n = safe_corr(outcomes["adversary_elo"], outcomes[outcome])
        model_means = outcomes.groupby("adversary_elo")[outcome].mean().reset_index()
        rm, pm, nm = safe_corr(model_means["adversary_elo"], model_means[outcome])
        slope, intercept = np.polyfit(outcomes["adversary_elo"], outcomes[outcome], 1)
        low = model_means.loc[model_means["adversary_elo"].idxmin(), outcome]
        high = model_means.loc[model_means["adversary_elo"].idxmax(), outcome]
        rows.append(
            {
                "outcome": outcome,
                "label": OUTCOME_LABELS[outcome],
                "low_elo_mean": low,
                "high_elo_mean": high,
                "delta_high_minus_low": high - low,
                "slope_per_100_elo": slope * 100,
                "run_spearman_r": r,
                "run_spearman_p": p,
                "n_runs": n,
                "model_spearman_r": rm,
                "model_spearman_p": pm,
                "n_models": nm,
            }
        )
    return pd.DataFrame(rows)


def save_outcome_plot(summary: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        "baseline_gini",
        "baseline_mean_payoff",
        "adversary_payoff",
        "adversary_minus_baseline",
        "total_payoff",
        "baseline_std",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.5), squeeze=False)
    for ax, metric in zip(axes.ravel(), metrics):
        ax.plot(summary["adversary_elo"], summary[metric], marker="o", linewidth=2)
        for _, row in summary.iterrows():
            ax.annotate(row["adversary_model_label"], (row["adversary_elo"], row[metric]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_title(OUTCOME_LABELS[metric], fontsize=10)
        ax.set_xlabel("Adversary Elo")
        ax.grid(True, alpha=0.35)
    fig.suptitle("Homogeneous adversary outcomes by adversary model strength", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_group_intensity_plot(group_freq: pd.DataFrame, out_path: Path) -> None:
    groups = sorted(group_freq["group"].dropna().unique())
    ncols = 3
    nrows = math.ceil(len(groups) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 3.4 * nrows), squeeze=False)
    max_y = max(0.01, float(group_freq["events_per_run"].max()) * 1.15)
    for ax, group in zip(axes.ravel(), groups):
        sub = group_freq[group_freq["group"] == group].sort_values("adversary_elo")
        ax.plot(sub["adversary_elo"], sub["events_per_run"], marker="o", linewidth=2)
        ax.set_title(display_group(group))
        ax.set_ylim(0, max_y)
        ax.grid(True, alpha=0.35)
    for ax in axes.ravel()[len(groups):]:
        ax.axis("off")
    fig.suptitle("Adversary tag-group intensity by adversary Elo", fontsize=15)
    fig.supxlabel("Adversary Elo")
    fig.supylabel("Adversary tag events per rollout")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_key_tag_plot(tag_freq: pd.DataFrame, codebook: pd.DataFrame, out_path: Path) -> None:
    titles = dict(zip(codebook["tag_code"], codebook["tag_title"]))
    top = [t for t in KEY_TAGS if t in set(tag_freq["tag_code"])]
    ncols = 2
    nrows = math.ceil(len(top) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows), squeeze=False)
    max_y = max(0.01, float(tag_freq[tag_freq["tag_code"].isin(top)]["events_per_run"].max()) * 1.2)
    for ax, tag in zip(axes.ravel(), top):
        sub = tag_freq[tag_freq["tag_code"] == tag].sort_values("adversary_elo")
        ax.plot(sub["adversary_elo"], sub["events_per_run"], marker="o", linewidth=2)
        ax.set_title(titles.get(tag, tag), fontsize=10)
        ax.set_ylim(0, max_y)
        ax.grid(True, alpha=0.35)
    for ax in axes.ravel()[len(top):]:
        ax.axis("off")
    fig.suptitle("Key adversary tag intensities by adversary Elo", fontsize=15)
    fig.supxlabel("Adversary Elo")
    fig.supylabel("Adversary tag events per rollout")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_correlation_bars(corr: pd.DataFrame, unit_col: str, outcome: str, out_path: Path, title: str, top_n: int = 18) -> None:
    sub = corr[(corr["outcome"] == outcome) & (corr["metric"] == "events_per_run")].dropna(subset=["spearman_r"]).copy()
    if sub.empty:
        return
    if top_n:
        pos = sub.sort_values("spearman_r", ascending=False).head(top_n)
        neg = sub.sort_values("spearman_r", ascending=True).head(top_n)
        sub = pd.concat([pos, neg], ignore_index=True).drop_duplicates([unit_col])
    sub = sub.sort_values("spearman_r")
    fig_h = max(5, 0.28 * len(sub) + 1.8)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    colors = np.where(sub["spearman_r"] >= 0, "#2563eb", "#dc2626")
    labels = sub[unit_col].astype(str)
    if unit_col == "group":
        labels = labels.map(display_group)
    ax.barh(labels, sub["spearman_r"], color=colors, alpha=0.85)
    ax.axvline(0, color="#111827", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Model-level Spearman r")
    ax.grid(True, axis="x", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_scatter_mechanism(summary: pd.DataFrame, group_freq: pd.DataFrame, out_path: Path) -> None:
    pivot = group_freq.pivot_table(index=["adversary_model", "adversary_elo"], columns="group", values="events_per_run").reset_index()
    merged = pivot.merge(summary, on=["adversary_model", "adversary_elo"], how="left")
    pairs = [
        ("pressure", "adversary_payoff"),
        ("coalition", "baseline_gini"),
        ("formalization", "baseline_gini"),
        ("emotional persuasion", "baseline_gini"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), squeeze=False)
    for ax, (group, outcome) in zip(axes.ravel(), pairs):
        if group not in merged:
            ax.axis("off")
            continue
        ax.scatter(merged[group], merged[outcome], s=90)
        for _, row in merged.iterrows():
            ax.annotate(MODEL_LABELS.get(row["adversary_model"], row["adversary_model"]), (row[group], row[outcome]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel(f"{display_group(group)} events/run")
        ax.set_ylabel(OUTCOME_LABELS[outcome])
        ax.grid(True, alpha=0.35)
    fig.suptitle("Mechanism candidates: tag intensity vs outcome", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def select_examples(events: pd.DataFrame, outcomes: pd.DataFrame, codebook: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    titles = dict(zip(codebook["tag_code"], codebook["tag_title"]))
    merged = events.merge(outcomes[["result_path", "adversary_elo", "adversary_model_label", "baseline_gini", "adversary_payoff", "baseline_mean_payoff"]], on="result_path", how="left")
    examples = []
    for tag in KEY_TAGS:
        sub = merged[merged["tag_code"] == tag].copy()
        if sub.empty:
            continue
        sub["quote_len"] = sub["quote"].astype(str).str.len()
        sub = sub.sort_values(["adversary_elo", "quote_len"], ascending=[False, False])
        for _, row in sub.head(2).iterrows():
            examples.append(
                {
                    "tag_code": tag,
                    "tag_title": titles.get(tag, tag),
                    "adversary_model": row["adversary_model_label"],
                    "adversary_elo": row["adversary_elo"],
                    "result_path": row["result_path"],
                    "round": row.get("round"),
                    "quote": str(row.get("quote", "")).replace("\n", " ")[:500],
                    "rationale": str(row.get("rationale", ""))[:350],
                    "baseline_gini": row["baseline_gini"],
                    "adversary_payoff": row["adversary_payoff"],
                    "baseline_mean_payoff": row["baseline_mean_payoff"],
                }
            )
    return pd.DataFrame(examples).drop_duplicates(["tag_code", "quote"]).head(18)


def md_table(df: pd.DataFrame, cols: list[str], n: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    d = df.head(n).copy()
    for c in d.columns:
        if d[c].dtype.kind in "fc":
            d[c] = d[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    return d[cols].to_markdown(index=False)


def write_reports(
    out_dir: Path,
    outcome_trends: pd.DataFrame,
    summary: pd.DataFrame,
    tag_freq: pd.DataFrame,
    group_freq: pd.DataFrame,
    tag_corr: pd.DataFrame,
    group_corr: pd.DataFrame,
    tag_run_corr: pd.DataFrame,
    group_run_corr: pd.DataFrame,
    examples: pd.DataFrame,
    curated_examples: pd.DataFrame,
) -> None:
    def image(path: str, alt: str) -> str:
        return f"![{alt}]({path})"

    group_corr_report = group_corr.copy()
    if "group" in group_corr_report:
        group_corr_report["group"] = group_corr_report["group"].map(display_group)
    group_run_corr_report = group_run_corr.copy()
    if "group" in group_run_corr_report:
        group_run_corr_report["group"] = group_run_corr_report["group"].map(display_group)

    full_lines = [
        "# Homogeneous-Adversary Mechanism Exploration",
        "",
        "## Research Question",
        "",
        "A single adversary model is inserted into a group of GPT-5-nano baseline agents. As the adversary model gets stronger, prior outcome analysis suggests that baseline-agent inequality falls, the adversary's own payoff rises, and group welfare does not collapse. This report asks what qualitative bargaining dynamics, as measured by the strategic tags, are associated with that pattern.",
        "",
        "Note on category naming: the underlying tag codebook calls one hot-tag group `pressure`. In the interpretation below, that group is displayed as **leverage-based bargaining** because its constituent tags are not merely emotional pressure; they include conditional vetoes, adversarial callouts, ultimata, fairness pressure, and frustration-based rejection moves. The raw CSVs preserve the original `pressure` category.",
        "",
        "## Outcome Trends",
        "",
        image("plots/outcomes_by_adversary_elo.png", "Outcome trends by adversary Elo"),
        "",
        md_table(outcome_trends.sort_values("outcome"), ["label", "low_elo_mean", "high_elo_mean", "delta_high_minus_low", "slope_per_100_elo", "run_spearman_r", "model_spearman_r"], n=20),
        "",
        "## Adversary Tag Dynamics",
        "",
        image("plots/group_intensity_by_elo.png", "Group tag intensity by adversary Elo"),
        "",
        image("plots/key_tag_intensity_by_elo.png", "Key tag intensity by adversary Elo"),
        "",
        "### Group-level model correlations",
        "",
        image("plots/group_corr_baseline_gini.png", "Group correlations with baseline Gini"),
        "",
        image("plots/group_corr_adversary_payoff.png", "Group correlations with adversary payoff"),
        "",
        md_table(group_corr_report.sort_values(["outcome", "spearman_r"], ascending=[True, False]), ["group", "metric", "outcome", "spearman_r", "p_value", "n_models"], n=80),
        "",
        "### Tag-level model correlations",
        "",
        image("plots/tag_corr_baseline_gini.png", "Tag correlations with baseline Gini"),
        "",
        image("plots/tag_corr_adversary_payoff.png", "Tag correlations with adversary payoff"),
        "",
        md_table(tag_corr.sort_values(["outcome", "spearman_r"], ascending=[True, False]), ["tag_code", "metric", "outcome", "spearman_r", "p_value", "n_models"], n=100),
        "",
        "## Within-cell Residual Checks",
        "",
        "The following tables correlate adversary event counts with outcome residuals after subtracting the mean within exact design cells: game x N x competition level x adversary position.",
        "",
        "### Groups",
        "",
        md_table(group_run_corr_report.sort_values("within_cell_resid_r", ascending=False), ["group", "outcome", "within_cell_resid_r", "within_cell_resid_p", "spearman_event_count_r", "mean_event_count", "used_rate"], n=80),
        "",
        "### Tags",
        "",
        md_table(tag_run_corr.sort_values("within_cell_resid_r", ascending=False), ["tag_code", "outcome", "within_cell_resid_r", "within_cell_resid_p", "spearman_event_count_r", "mean_event_count", "used_rate"], n=120),
        "",
        "## Mechanism Scatterplots",
        "",
        image("plots/mechanism_scatter.png", "Mechanism scatterplots"),
        "",
        "## Qualitative Examples",
        "",
        "### Curated mechanism contrasts",
        "",
        md_table(curated_examples, ["mechanism", "model", "tag", "example", "interpretation"], n=20),
        "",
        "### Automatically selected tagged quotes",
        "",
        md_table(examples, ["tag_title", "adversary_model", "adversary_elo", "round", "quote", "rationale", "baseline_gini", "adversary_payoff"], n=18),
        "",
        "## Output Tables",
        "",
        "- [run_outcomes.csv](run_outcomes.csv)",
        "- [model_outcome_summary.csv](model_outcome_summary.csv)",
        "- [tag_model_intensity.csv](tag_model_intensity.csv)",
        "- [group_model_intensity.csv](group_model_intensity.csv)",
        "- [tag_model_outcome_correlations.csv](tag_model_outcome_correlations.csv)",
        "- [group_model_outcome_correlations.csv](group_model_outcome_correlations.csv)",
        "- [tag_run_level_outcome_correlations.csv](tag_run_level_outcome_correlations.csv)",
        "- [group_run_level_outcome_correlations.csv](group_run_level_outcome_correlations.csv)",
        "- [qualitative_examples.csv](qualitative_examples.csv)",
        "- [curated_qualitative_examples.csv](curated_qualitative_examples.csv)",
    ]
    (out_dir / "homogeneous_adversary_mechanism_full_report.md").write_text("\n".join(full_lines) + "\n")

    # Pull a few values for the paper-style narrative.
    trend_map = outcome_trends.set_index("outcome").to_dict("index")
    group_baseline = group_corr[(group_corr["metric"] == "events_per_run") & (group_corr["outcome"] == "baseline_gini")].sort_values("spearman_r")
    group_adv = group_corr[(group_corr["metric"] == "events_per_run") & (group_corr["outcome"] == "adversary_payoff")].sort_values("spearman_r", ascending=False)
    top_down = group_baseline.head(3)
    top_adv = group_adv.head(3)
    tag_adv = tag_corr[(tag_corr["metric"] == "events_per_run") & (tag_corr["outcome"] == "adversary_payoff")].sort_values("spearman_r", ascending=False).head(6)
    tag_gini = tag_corr[(tag_corr["metric"] == "events_per_run") & (tag_corr["outcome"] == "baseline_gini")].sort_values("spearman_r").head(6)

    def fmt_group_row(row: Any) -> str:
        return f"`{display_group(row.group)}` (rho={row.spearman_r:.2f})"

    mini_lines = [
        "# A Stronger Adversary Compresses the Baseline: A Leverage-Based Bargaining Mechanism",
        "",
        "## Research Problem",
        "",
        "The homogeneous-adversary experiment creates a deliberately asymmetric setting: one non-baseline adversary model is inserted into an otherwise homogeneous group of GPT-5-nano baseline agents. Despite being only one participant, the adversary model has a large effect on the final allocation. As adversary Elo rises, the adversary's own payoff rises, but baseline agents also become less unequal among themselves. The puzzle is why a single stronger agent can simultaneously capture more value and reduce dispersion among the other agents.",
        "",
        "## Hypothesis",
        "",
        "The tag evidence supports a leverage-based bargaining account. Stronger adversaries do not merely demand more, and they do not appear to win by creating disorder. They more often convert disagreement into explicit constraints: veto conditions, red lines, cost corrections, conflict maps, and named coalition paths. These moves narrow the feasible bargaining set for the baseline agents. That narrowing can reduce baseline variance because the baselines are no longer independently drifting toward idiosyncratic deals; they are being coordinated around a smaller menu of acceptable packages. The same structure lets the adversary embed a stronger reservation point into the final deal.",
        "",
        "## Quantitative outcome pattern",
        "",
        f"Across the five adversary models, baseline-only corrected Gini moves from {trend_map['baseline_gini']['low_elo_mean']:.3f} at the weakest adversary to {trend_map['baseline_gini']['high_elo_mean']:.3f} at the strongest adversary. The adversary payoff moves from {trend_map['adversary_payoff']['low_elo_mean']:.3f} to {trend_map['adversary_payoff']['high_elo_mean']:.3f}, while mean baseline payoff moves from {trend_map['baseline_mean_payoff']['low_elo_mean']:.3f} to {trend_map['baseline_mean_payoff']['high_elo_mean']:.3f}. The sign pattern is therefore not pure extraction from baselines. It is closer to a higher-value, more controlled settlement.",
        "",
        image("plots/outcomes_by_adversary_elo.png", "Outcome trends by adversary Elo"),
        "",
        "## Qualitative mechanism in the tags",
        "",
        "The strongest qualitative signature is a shift from diffuse bargaining toward enforceable structure. In the model-level tag correlations, the groups most negatively associated with baseline-only Gini are "
        + ", ".join(fmt_group_row(r) for r in top_down.itertuples())
        + ". The groups most positively associated with adversary payoff are "
        + ", ".join(fmt_group_row(r) for r in top_adv.itertuples())
        + ".",
        "",
        image("plots/group_intensity_by_elo.png", "Group tag intensity by adversary Elo"),
        "",
        image("plots/group_corr_baseline_gini.png", "Group correlations with baseline Gini"),
        "",
        "At the tag level, adversary payoff is most positively associated with "
        + ", ".join(f"`{r.tag_code}` (rho={r.spearman_r:.2f})" for r in tag_adv.itertuples())
        + ". Baseline Gini is most negatively associated with "
        + ", ".join(f"`{r.tag_code}` (rho={r.spearman_r:.2f})" for r in tag_gini.itertuples())
        + ". These are exactly the tags one would expect if the adversary is making the negotiation legible and bounded: conditional vetoes, cost policing, conflict mapping, and explicit coalition or threshold reasoning.",
        "",
        image("plots/key_tag_intensity_by_elo.png", "Key tag intensity by adversary Elo"),
        "",
        "## Qualitative Contrasts",
        "",
        md_table(curated_examples, ["mechanism", "model", "tag", "example", "interpretation"], n=8),
        "",
        "## Interpretation",
        "",
        "The mechanism is not that stronger adversaries make the conversation more generous in a naive sense. Rather, they impose a stronger bargaining frame. The baselines benefit from reduced dispersion because the adversary's constraints act as an external coordination device: proposals that violate the adversary's red lines, cost constraints, or coalition math are filtered out. Once the search space is narrowed, baseline agents are less likely to end with extreme relative outcomes. The adversary benefits because the same frame embeds its own reservation point into the deal and makes that point costly for others to ignore.",
        "",
        "This explains the seemingly paradoxical pattern: a stronger adversary can both take more and reduce inequality among everyone else. The high-Elo adversary is not primarily extracting through chaos. It is extracting through corridor-setting: it makes the feasible settlement more legible, more bounded, and more conditional on its own participation. That can lift total payoff and slightly lift baseline payoff while still shifting the surplus toward the adversary.",
        "",
        "## Evidence limits",
        "",
        "The model-level correlations use only five adversary models, so they should be read as mechanism evidence rather than definitive causal identification. The within-cell residual tables in the full appendix reduce design-cell confounding, but tag use can still be post-treatment: difficult negotiations can cause tags, not only the reverse. The strongest claim warranted here is interpretive: the tags that rise with adversary strength are the same tags that plausibly explain payoff capture plus baseline compression.",
    ]
    (out_dir / "homogeneous_adversary_mechanism_research_section.md").write_text("\n".join(mini_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--hot-tags-json", type=Path, default=DEFAULT_HOT_TAGS_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    out_dir = args.output_dir
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=0.9)

    codebook = pd.DataFrame(json.loads((args.input_dir / "llm_tag_codebook.json").read_text()))
    hot_tags = load_hot_tags(args.hot_tags_json, codebook)
    outcomes = build_outcome_frame(args.input_dir)
    events, tag_counts, group_counts = build_event_frames(args.input_dir, codebook, hot_tags)
    groups = sorted(codebook[codebook["tag_code"].isin(hot_tags)]["category"].dropna().unique())
    summary, tag_freq, group_freq = model_summary(outcomes, tag_counts, group_counts, hot_tags, groups)
    outcome_cols = list(OUTCOME_LABELS)
    outcome_trends = outcome_trend_table(outcomes)
    tag_corr = model_tag_outcome_correlations(tag_freq, summary, "tag_code", outcome_cols)
    group_corr = model_tag_outcome_correlations(group_freq, summary, "group", outcome_cols)
    tag_run_corr = run_level_correlations(outcomes, tag_counts, "tag_code", hot_tags, outcome_cols)
    group_run_corr = run_level_correlations(outcomes, group_counts, "group", groups, outcome_cols)
    examples = select_examples(events, outcomes, codebook)
    curated_examples = pd.DataFrame(CURATED_QUALITATIVE_EXAMPLES)

    outcomes.to_csv(out_dir / "run_outcomes.csv", index=False)
    summary.to_csv(out_dir / "model_outcome_summary.csv", index=False)
    outcome_trends.to_csv(out_dir / "outcome_elo_trends.csv", index=False)
    tag_freq.to_csv(out_dir / "tag_model_intensity.csv", index=False)
    group_freq.to_csv(out_dir / "group_model_intensity.csv", index=False)
    tag_corr.to_csv(out_dir / "tag_model_outcome_correlations.csv", index=False)
    group_corr.to_csv(out_dir / "group_model_outcome_correlations.csv", index=False)
    tag_run_corr.to_csv(out_dir / "tag_run_level_outcome_correlations.csv", index=False)
    group_run_corr.to_csv(out_dir / "group_run_level_outcome_correlations.csv", index=False)
    examples.to_csv(out_dir / "qualitative_examples.csv", index=False)
    curated_examples.to_csv(out_dir / "curated_qualitative_examples.csv", index=False)

    save_outcome_plot(summary, plot_dir / "outcomes_by_adversary_elo.png")
    save_group_intensity_plot(group_freq, plot_dir / "group_intensity_by_elo.png")
    save_key_tag_plot(tag_freq, codebook, plot_dir / "key_tag_intensity_by_elo.png")
    save_correlation_bars(group_corr, "group", "baseline_gini", plot_dir / "group_corr_baseline_gini.png", "Group intensity vs baseline-only Gini", top_n=None)
    save_correlation_bars(group_corr, "group", "adversary_payoff", plot_dir / "group_corr_adversary_payoff.png", "Group intensity vs adversary payoff", top_n=None)
    save_correlation_bars(tag_corr, "tag_code", "baseline_gini", plot_dir / "tag_corr_baseline_gini.png", "Tag intensity vs baseline-only Gini", top_n=16)
    save_correlation_bars(tag_corr, "tag_code", "adversary_payoff", plot_dir / "tag_corr_adversary_payoff.png", "Tag intensity vs adversary payoff", top_n=16)
    save_scatter_mechanism(summary, group_freq, plot_dir / "mechanism_scatter.png")
    write_reports(out_dir, outcome_trends, summary, tag_freq, group_freq, tag_corr, group_corr, tag_run_corr, group_run_corr, examples, curated_examples)

    print(f"wrote {out_dir}")
    print(f"runs={len(outcomes)} events={len(events)} tags={len(hot_tags)} groups={len(groups)}")


if __name__ == "__main__":
    main()

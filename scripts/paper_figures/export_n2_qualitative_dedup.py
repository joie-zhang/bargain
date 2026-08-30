#!/usr/bin/env python3
"""Export turn-deduplicated N=2 qualitative inputs for the paper figure."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import pandas as pd
from scipy.stats import linregress, spearmanr


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ADJUDICATION_DIR = (
    ROOT / "analysis/llm_strategic_tag_adjudication_n2_gpt5_20260629"
)
DEFAULT_EXPLORATION_DIR = (
    ROOT / "analysis/llm_strategic_tag_elo_exploration_n2_gpt5_intensity_20260629"
)
DEFAULT_OUTPUT_DIR = ROOT / "overleaf/icml_aiwild_template/graphics/qualitative_n2"

CATEGORIES = (
    "emotional persuasion",
    "trade/compromise",
    "logical persuasion",
    "pressure",
    "self-interest/exploitation",
    "formalization",
)

SELECTED_TAGS = (
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
)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_denominators(manifest_path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rollout in read_jsonl(manifest_path):
        if rollout.get("experiment_family") != "n2_gpt5_bilateral":
            continue
        result_path = str(rollout["result_path"])
        role_map = rollout["agent_role_map"]
        model_map = rollout["agent_model_map"]
        elo_map = rollout["agent_elo_map"]
        for agent, role in role_map.items():
            if role != "adversary":
                continue
            rows.append(
                {
                    "speaker_key": f"{result_path}::{agent}",
                    "speaker_model": model_map[agent],
                    "speaker_elo": elo_map[agent],
                }
            )

    denoms = pd.DataFrame(rows).drop_duplicates("speaker_key")
    if len(denoms) != len(rows):
        raise ValueError("The adjudication manifest has duplicate adversary speaker keys")
    if denoms["speaker_model"].str.contains("phi", case=False, na=False).any():
        raise ValueError("The adjudication manifest still contains a Phi adversary")
    return denoms


def build_deduplicated_events(
    event_path: Path,
    codebook_path: Path,
    eligible_speaker_keys: set[str],
) -> tuple[pd.DataFrame, int]:
    codebook = json.loads(codebook_path.read_text(encoding="utf-8"))
    tag_to_category = {row["tag_code"]: row["category"] for row in codebook}
    rows: list[dict[str, object]] = []

    for event in read_jsonl(event_path):
        if event.get("experiment_family") != "n2_gpt5_bilateral":
            continue
        if event.get("speaker_role") != "adversary":
            continue
        result_path = event.get("result_path")
        speaker_agent = event.get("speaker_agent")
        if not result_path or not speaker_agent:
            continue
        speaker_key = f"{result_path}::{speaker_agent}"
        if speaker_key not in eligible_speaker_keys:
            continue
        category = tag_to_category.get(event.get("tag_code"))
        if category not in CATEGORIES:
            continue
        rows.append(
            {
                "speaker_key": speaker_key,
                "round": event.get("round"),
                "discussion_turn": event.get("discussion_turn"),
                "phase": event.get("phase"),
                "category": category,
            }
        )

    events = pd.DataFrame(rows)
    before = len(events)
    events = events.drop_duplicates(
        ["speaker_key", "round", "discussion_turn", "phase", "category"]
    )
    return events, before


def build_intensity(denoms: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    counts = (
        events.groupby(["speaker_key", "category"], as_index=False)
        .size()
        .rename(columns={"size": "event_count"})
        .merge(denoms, on="speaker_key", how="inner", validate="many_to_one")
        .groupby(["speaker_model", "speaker_elo", "category"], as_index=False)
        .agg(event_count=("event_count", "sum"))
    )
    model_denoms = (
        denoms.groupby(["speaker_model", "speaker_elo"], as_index=False)
        .agg(speaker_rollouts=("speaker_key", "nunique"))
    )
    grid = pd.DataFrame(
        product(model_denoms["speaker_model"].tolist(), CATEGORIES),
        columns=["speaker_model", "category"],
    )
    out = (
        grid.merge(model_denoms, on="speaker_model", how="left", validate="many_to_one")
        .merge(
            counts,
            on=["speaker_model", "speaker_elo", "category"],
            how="left",
            validate="one_to_one",
        )
        .fillna({"event_count": 0})
    )
    out["event_count"] = out["event_count"].astype(int)
    out["intensity"] = out["event_count"] / out["speaker_rollouts"]
    return out[
        [
            "speaker_model",
            "category",
            "event_count",
            "speaker_elo",
            "speaker_rollouts",
            "intensity",
        ]
    ].sort_values(["speaker_model", "category"])


def build_correlations(
    denoms: pd.DataFrame,
    events: pd.DataFrame,
    speaker_payoff_path: Path,
) -> tuple[pd.DataFrame, int]:
    payoffs = pd.read_csv(speaker_payoff_path)
    payoffs = payoffs[
        payoffs["speaker_key"].isin(set(denoms["speaker_key"]))
        & payoffs["final_utility"].notna()
    ][["speaker_key", "final_utility"]].drop_duplicates("speaker_key")

    counts = (
        events.groupby(["speaker_key", "category"], as_index=False)
        .size()
        .rename(columns={"size": "event_count"})
    )
    grid = pd.DataFrame(
        product(payoffs["speaker_key"].tolist(), CATEGORIES),
        columns=["speaker_key", "category"],
    )
    frame = (
        grid.merge(payoffs, on="speaker_key", how="left", validate="many_to_one")
        .merge(counts, on=["speaker_key", "category"], how="left", validate="one_to_one")
        .fillna({"event_count": 0})
    )

    rows = []
    for category in CATEGORIES:
        subset = frame[frame["category"] == category]
        rho, p_value = spearmanr(subset["event_count"], subset["final_utility"])
        rows.append(
            {
                "category": category,
                "spearman_event_count_r_utility": float(rho),
                "spearman_event_count_p_utility": float(p_value),
                "n_used": int((subset["event_count"] > 0).sum()),
            }
        )
    return pd.DataFrame(rows), len(payoffs)


def build_tag_statistics(
    denoms: pd.DataFrame,
    event_path: Path,
    codebook_path: Path,
    speaker_payoff_path: Path,
) -> tuple[pd.DataFrame, int, int]:
    codebook = json.loads(codebook_path.read_text(encoding="utf-8"))
    metadata = {row["tag_code"]: row for row in codebook}
    eligible_speaker_keys = set(denoms["speaker_key"])
    rows: list[dict[str, object]] = []

    for event in read_jsonl(event_path):
        if event.get("experiment_family") != "n2_gpt5_bilateral":
            continue
        if event.get("speaker_role") != "adversary":
            continue
        tag_code = event.get("tag_code")
        if tag_code not in SELECTED_TAGS:
            continue
        result_path = event.get("result_path")
        speaker_agent = event.get("speaker_agent")
        if not result_path or not speaker_agent:
            continue
        speaker_key = f"{result_path}::{speaker_agent}"
        if speaker_key not in eligible_speaker_keys:
            continue
        rows.append(
            {
                "speaker_key": speaker_key,
                "round": event.get("round"),
                "discussion_turn": event.get("discussion_turn"),
                "phase": event.get("phase"),
                "tag_code": tag_code,
            }
        )

    events = pd.DataFrame(rows)
    before = len(events)
    events = events.drop_duplicates(
        ["speaker_key", "round", "discussion_turn", "phase", "tag_code"]
    )

    counts = (
        events.groupby(["speaker_key", "tag_code"], as_index=False)
        .size()
        .rename(columns={"size": "event_count"})
    )
    model_denoms = (
        denoms.groupby(["speaker_model", "speaker_elo"], as_index=False)
        .agg(speaker_rollouts=("speaker_key", "nunique"))
    )
    grid = pd.DataFrame(
        product(model_denoms["speaker_model"].tolist(), SELECTED_TAGS),
        columns=["speaker_model", "tag_code"],
    )
    model_counts = (
        counts.merge(denoms, on="speaker_key", how="inner", validate="many_to_one")
        .groupby(["speaker_model", "speaker_elo", "tag_code"], as_index=False)
        .agg(event_count=("event_count", "sum"))
    )
    model_rates = (
        grid.merge(model_denoms, on="speaker_model", how="left", validate="many_to_one")
        .merge(
            model_counts,
            on=["speaker_model", "speaker_elo", "tag_code"],
            how="left",
            validate="one_to_one",
        )
        .fillna({"event_count": 0})
    )
    model_rates["intensity"] = (
        model_rates["event_count"] / model_rates["speaker_rollouts"]
    )

    payoffs = pd.read_csv(speaker_payoff_path)
    payoffs = payoffs[
        payoffs["speaker_key"].isin(eligible_speaker_keys)
        & payoffs["final_utility"].notna()
    ][["speaker_key", "final_utility"]].drop_duplicates("speaker_key")
    model_payoffs = (
        payoffs.merge(denoms, on="speaker_key", how="inner", validate="one_to_one")
        .groupby(["speaker_model", "speaker_elo"], as_index=False)
        .agg(mean_final_utility=("final_utility", "mean"))
    )
    model_rates = model_rates.merge(
        model_payoffs,
        on=["speaker_model", "speaker_elo"],
        how="left",
        validate="many_to_one",
    )

    output_rows = []
    for tag_code in SELECTED_TAGS:
        subset = model_rates[model_rates["tag_code"] == tag_code]
        elo_rho, elo_p = spearmanr(subset["speaker_elo"], subset["intensity"])
        regression = linregress(subset["speaker_elo"], subset["intensity"])
        payoff_subset = subset.dropna(subset=["mean_final_utility"])
        payoff_rho, payoff_p = spearmanr(
            payoff_subset["intensity"], payoff_subset["mean_final_utility"]
        )
        output_rows.append(
            {
                "category": metadata[tag_code]["category"],
                "tag_code": tag_code,
                "tag_title": metadata[tag_code]["tag_title"],
                "n_models": len(subset),
                "spearman_elo_r": float(elo_rho),
                "spearman_elo_p": float(elo_p),
                "slope_per_100_elo": float(regression.slope * 100),
                "spearman_payoff_r": float(payoff_rho),
                "spearman_payoff_p": float(payoff_p),
            }
        )
    return pd.DataFrame(output_rows), before, len(events)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adjudication-dir", type=Path, default=DEFAULT_ADJUDICATION_DIR)
    parser.add_argument("--exploration-dir", type=Path, default=DEFAULT_EXPLORATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    manifest_path = args.adjudication_dir / "all_rollouts_manifest.jsonl"
    event_path = args.adjudication_dir / "llm_event_tags.jsonl"
    codebook_path = args.adjudication_dir / "llm_tag_codebook.json"
    speaker_payoff_path = args.exploration_dir / "speaker_payoffs.csv"

    denoms = build_denominators(manifest_path)
    events, event_rows_before_dedup = build_deduplicated_events(
        event_path, codebook_path, set(denoms["speaker_key"])
    )
    intensity = build_intensity(denoms, events)
    correlations, payoff_rollouts = build_correlations(
        denoms, events, speaker_payoff_path
    )
    tag_statistics, tag_rows_before_dedup, tag_rows_after_dedup = (
        build_tag_statistics(denoms, event_path, codebook_path, speaker_payoff_path)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    intensity_path = args.output_dir / "n2_group_intensity_dedup.csv"
    correlation_path = args.output_dir / "n2_group_payoff_corr_dedup.csv"
    tag_statistics_path = args.output_dir / "n2_tag_mechanism_dedup.csv"
    provenance_path = args.output_dir / "n2_qualitative_dedup_provenance.json"
    intensity.to_csv(intensity_path, index=False)
    correlations.to_csv(correlation_path, index=False)
    tag_statistics.to_csv(tag_statistics_path, index=False)

    provenance = {
        "deduplication_key": [
            "speaker_key",
            "round",
            "discussion_turn",
            "phase",
            "category",
        ],
        "tag_deduplication_key": [
            "speaker_key",
            "round",
            "discussion_turn",
            "phase",
            "tag_code",
        ],
        "manifest_rollouts": len(denoms),
        "payoff_rollouts": payoff_rollouts,
        "eligible_event_rows_before_dedup": event_rows_before_dedup,
        "event_rows_after_dedup": len(events),
        "selected_tag_event_rows_before_dedup": tag_rows_before_dedup,
        "selected_tag_event_rows_after_dedup": tag_rows_after_dedup,
        "models": int(denoms["speaker_model"].nunique()),
        "phi_rows": int(
            intensity["speaker_model"].str.contains("phi", case=False, na=False).sum()
        ),
        "inputs": {
            str(manifest_path.relative_to(ROOT)): sha256(manifest_path),
            str(event_path.relative_to(ROOT)): sha256(event_path),
            str(codebook_path.relative_to(ROOT)): sha256(codebook_path),
            str(speaker_payoff_path.relative_to(ROOT)): sha256(speaker_payoff_path),
        },
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {intensity_path}")
    print(f"Wrote {correlation_path}")
    print(f"Wrote {tag_statistics_path}")
    print(f"Wrote {provenance_path}")
    print(
        f"rollouts={len(denoms)} payoff_rollouts={payoff_rollouts} "
        f"events={event_rows_before_dedup}->{len(events)} models={provenance['models']}"
    )


if __name__ == "__main__":
    main()

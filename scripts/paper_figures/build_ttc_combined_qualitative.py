#!/usr/bin/env python3
"""Build TTC-only and bilateral-plus-TTC qualitative diagnostics.

The script writes a separate analysis bundle.  It does not replace Figure 4 or
edit paper text.  The paper-comparable category analysis uses the selected 23
tags and Figure 4's six displayed categories.  A full-50-codebook sensitivity
is emitted from the same normalized rollout/event tables.

TTC target models have no measured Elo in the paper's 30-model Elo roster.
For this reason, the script does not assign proxy Elo values.  It retains the
bilateral Elo diagnostic and uses a within-game competition index for the TTC
effort-stratified setting sensitivity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import sem, spearmanr, t


ROOT = Path("/scratch/gpfs/DANQIC/jz4391/bargain")
DEFAULT_OUT = ROOT / "analysis/ttc_combined_qualitative_20260814"

PRIMARY_CSV = (
    ROOT
    / "experiments/results/n2_baseline_comparison_analysis_20260505/primary_runs_with_metrics.csv"
)
BILATERAL_ROOT = ROOT / "analysis/llm_strategic_tag_adjudication_n2_gpt5_20260629"
BILATERAL_MANIFEST = BILATERAL_ROOT / "all_rollouts_manifest.jsonl"
BILATERAL_EVENTS = BILATERAL_ROOT / "llm_event_tags.jsonl"
BILATERAL_CODEBOOK = BILATERAL_ROOT / "llm_tag_codebook.json"

TTC_SOURCES = (
    (
        "seed42_all_families",
        ROOT / "analysis/ttc_llm_strategic_tag_adjudication_20260629/all_ttc_rollouts_manifest.jsonl",
        ROOT / "analysis/ttc_llm_strategic_tag_adjudication_20260629/ttc_llm_event_tags.jsonl",
        ROOT / "analysis/ttc_llm_strategic_tag_adjudication_20260629/llm_tag_codebook.json",
    ),
    (
        "gpt5_nine_extra_seeds",
        ROOT / "analysis/ttc_gpt5_nine_seed_codex_adjudication_20260809/all_rollouts_manifest.jsonl",
        ROOT / "analysis/ttc_gpt5_nine_seed_codex_adjudication_20260809/ttc_gpt5_event_tags.jsonl",
        ROOT / "analysis/ttc_gpt5_nine_seed_codex_adjudication_20260809/llm_tag_codebook.json",
    ),
    (
        "claude_nine_extra_seeds",
        ROOT / "analysis/ttc_claude_nine_seed_codex_adjudication_20260728/all_available_rollouts_manifest.jsonl",
        ROOT / "analysis/ttc_claude_nine_seed_codex_adjudication_20260728/ttc_codex_event_tags.jsonl",
        ROOT / "analysis/ttc_claude_nine_seed_codex_adjudication_20260728/llm_tag_codebook.json",
    ),
    (
        "gemini_nine_extra_seeds",
        ROOT / "analysis/ttc_gemini_nine_seed_codex_adjudication_20260809/all_available_rollouts_manifest.jsonl",
        ROOT / "analysis/ttc_gemini_nine_seed_codex_adjudication_20260809/ttc_codex_event_tags.jsonl",
        ROOT / "analysis/ttc_gemini_nine_seed_codex_adjudication_20260809/llm_tag_codebook.json",
    ),
)

CATEGORIES = (
    "trade/compromise",
    "emotional persuasion",
    "logical persuasion",
    "pressure",
    "self-interest/exploitation",
    "formalization",
)
DISPLAY = {
    "trade/compromise": "Trade / Compromise",
    "emotional persuasion": "Emotional persuasion",
    "logical persuasion": "Logical persuasion",
    "pressure": "Pressure",
    "self-interest/exploitation": "Self-interest / Exploitation",
    "formalization": "Formalization",
}
COLORS = {
    "trade/compromise": "#1f77b4",
    "emotional persuasion": "#2ca02c",
    "logical persuasion": "#17becf",
    "pressure": "#ff7f0e",
    "self-interest/exploitation": "#d62728",
    "formalization": "#9467bd",
}
FAMILY_ORDER = ("gpt-5", "claude-sonnet-4-6", "gemini-3-flash")
FAMILY_DISPLAY = {
    "gpt-5": "GPT-5",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gemini-3-flash": "Gemini 3 Flash",
}
EFFORT_ORDER = {
    "gpt-5": ("minimal", "low", "medium", "high"),
    "claude-sonnet-4-6": ("low", "medium", "high", "max"),
    "gemini-3-flash": ("minimal", "low", "medium", "high"),
}
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

# This index orders the three selected settings within each game from more
# cooperative (0) to more competitive (1).  It is a setting index, not Elo.
COMPETITION_INDEX = {
    "game1_comp_0p0": 0.0,
    "game1_comp_0p5": 0.5,
    "game1_comp_1p0": 1.0,
    "game2_rho_1_theta_1": 0.0,
    "game2_rho_0_theta_1": 0.5,
    "game2_rho_n1_theta_1": 1.0,
    "game3_alpha_1p0_sigma_1p0": 0.0,
    "game3_alpha_0p5_sigma_0p6": 0.5,
    "game3_alpha_0p0_sigma_0p2": 1.0,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def absolute_path(value: object) -> str:
    path = Path(str(value))
    if not path.is_absolute():
        path = ROOT / path
    return str(path.resolve())


def codebook_map(path: Path) -> dict[str, str]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["tag_code"]): str(row["category"]) for row in rows}


def validate_codebooks() -> dict[str, str]:
    reference = codebook_map(TTC_SOURCES[0][3])
    for _, _, _, path in TTC_SOURCES[1:]:
        if codebook_map(path) != reference:
            raise ValueError(f"TTC codebook mapping differs: {path}")
    bilateral = codebook_map(BILATERAL_CODEBOOK)
    if bilateral != reference:
        raise ValueError("Bilateral and TTC tag-to-category mappings differ")
    if len(reference) != 50 or not SELECTED_23.issubset(reference):
        raise ValueError("Unexpected 50-tag codebook or selected-23 subset")
    return reference


def load_bilateral_denominators() -> pd.DataFrame:
    primary = pd.read_csv(PRIMARY_CSV)
    primary = primary[primary["baseline_key"].eq("gpt5_nano")].copy()
    primary["result_path"] = primary["result_path"].map(absolute_path)
    if len(primary) != 1500 or primary["result_path"].nunique() != 1500:
        raise ValueError("Bilateral primary cohort is not 1,500 unique rollouts")

    manifest = pd.DataFrame(read_jsonl(BILATERAL_MANIFEST))
    manifest["result_path"] = manifest["result_path"].map(absolute_path)
    manifest = manifest[
        manifest["experiment_family"].eq("n2_gpt5_bilateral")
        & manifest["result_path"].isin(set(primary["result_path"]))
    ].copy()
    if len(manifest) != 1500 or manifest["result_path"].nunique() != 1500:
        raise ValueError("Bilateral annotation manifest does not cover the primary cohort")

    rows: list[dict[str, Any]] = []
    primary_by_path = primary.set_index("result_path", drop=False)
    for row in manifest.to_dict("records"):
        adversaries = [
            agent for agent, role in row["agent_role_map"].items() if role == "adversary"
        ]
        if len(adversaries) != 1:
            raise ValueError("Expected one bilateral adversary")
        agent = adversaries[0]
        metric = primary_by_path.loc[row["result_path"]]
        rows.append(
            {
                "cohort": "bilateral1500",
                "rollout_key": row["result_path"],
                "result_path": row["result_path"],
                "speaker_agent": agent,
                "speaker_key": f"{row['result_path']}::{agent}",
                "utility": float(metric["adversary_utility"]),
                "model_family": str(metric["adversary_model"]),
                "speaker_model": str(metric["adversary_model"]),
                "speaker_elo": float(metric["adversary_elo"]),
                "effort": None,
                "effort_index": np.nan,
                "seed": int(metric["seed"]),
                "game_label": str(metric["game_id"]),
                "game_cell": str(metric["competition_setting"]),
                "competition_index": float(metric["competition_level"]),
                "order": str(metric["conceptual_order"]),
                "bootstrap_stratum": "bilateral1500",
                "bootstrap_cluster": f"bilateral_model::{metric['adversary_model']}",
            }
        )
    output = pd.DataFrame(rows)
    if len(output) != 1500 or output["speaker_model"].nunique() != 30:
        raise ValueError("Unexpected bilateral denominator shape")
    return output


def load_ttc_denominators() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_name, manifest_path, _, _ in TTC_SOURCES:
        for raw in read_jsonl(manifest_path):
            family = str(raw["family"])
            effort = str(raw["level"])
            if family not in EFFORT_ORDER or effort not in EFFORT_ORDER[family]:
                raise ValueError(f"Unexpected TTC family/effort: {family}/{effort}")
            result_path = absolute_path(raw["result_path"])
            seed = int(raw.get("seed", 42))
            game_cell = str(raw["game_cell"])
            if game_cell not in COMPETITION_INDEX:
                raise ValueError(f"Unknown TTC game cell: {game_cell}")
            target_agent = str(raw["target_agent"])
            rows.append(
                {
                    "cohort": "ttc2160",
                    "rollout_key": result_path,
                    "result_path": result_path,
                    "speaker_agent": target_agent,
                    "speaker_key": f"{result_path}::{target_agent}",
                    "utility": float(raw["target_utility"]),
                    "model_family": family,
                    "speaker_model": str(raw["target_model"]),
                    "speaker_elo": np.nan,
                    "effort": effort,
                    "effort_index": EFFORT_ORDER[family].index(effort),
                    "seed": seed,
                    "game_label": str(raw["game_label"]),
                    "game_cell": game_cell,
                    "competition_index": COMPETITION_INDEX[game_cell],
                    "order": str(raw["order"]),
                    "bootstrap_stratum": "ttc2160",
                    "bootstrap_cluster": f"ttc_seed::{seed}",
                    "manifest_source": source_name,
                }
            )
    output = pd.DataFrame(rows)
    if len(output) != 2160 or output["rollout_key"].nunique() != 2160:
        raise ValueError("TTC cohort is not 2,160 unique rollouts")
    if output.groupby("model_family").size().to_dict() != {
        family: 720 for family in FAMILY_ORDER
    }:
        raise ValueError("TTC family counts are not 720 each")
    if output["seed"].nunique() != 10:
        raise ValueError("TTC cohort does not contain ten seeds")
    cell_counts = output.groupby(["model_family", "effort"]).size()
    if set(cell_counts) != {180}:
        raise ValueError("TTC family-effort cells are not 180 rollouts each")
    return output


def load_event_rows(paths: Iterable[Path], source_prefix: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_index, path in enumerate(paths):
        for line_index, raw in enumerate(read_jsonl(path), 1):
            row = dict(raw)
            row["result_path"] = absolute_path(row["result_path"])
            row["_event_source"] = f"{source_prefix}:{source_index}:{path}"
            row["_event_source_order"] = source_index
            row["_event_line"] = line_index
            rows.append(row)
    return pd.DataFrame(rows)


def apply_replacement_events(
    base: pd.DataFrame, replacement_paths: list[Path], source_prefix: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not replacement_paths:
        return base, {"replacement_files": [], "replaced_rollouts": 0, "replacement_rows": 0}
    replacements = load_event_rows(replacement_paths, source_prefix)
    replaced = set(replacements["result_path"])
    output = pd.concat(
        [base[~base["result_path"].isin(replaced)], replacements],
        ignore_index=True,
        sort=False,
    )
    return output, {
        "replacement_files": [str(path) for path in replacement_paths],
        "replaced_rollouts": len(replaced),
        "replacement_rows": len(replacements),
    }


def audit_raw_event_identities(raw: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Count validator-identity duplicates before analysis filtering."""

    frame = raw.copy()
    if "rollout_id" not in frame:
        frame["rollout_id"] = None
    frame["_rollout_identity"] = frame["rollout_id"].where(
        frame["rollout_id"].notna(), frame["result_path"]
    )
    columns = [
        "_rollout_identity",
        "tag_code",
        "source_kind",
        "log_index",
        "interaction_index",
        "speaker_agent",
    ]
    for column in columns:
        if column not in frame:
            frame[column] = None
    output: dict[str, dict[str, int]] = {}
    for source, group in frame.groupby("_event_source", sort=True):
        identity_frame = group[columns].fillna("<NULL>")
        counts = Counter(
            tuple(row) for row in identity_frame.itertuples(index=False, name=None)
        )
        output[str(source)] = {
            "event_rows": len(group),
            "duplicate_identity_keys": sum(value > 1 for value in counts.values()),
            "extra_rows_across_duplicate_identities": sum(
                value - 1 for value in counts.values() if value > 1
            ),
        }
    return output


def normalize_events(
    raw: pd.DataFrame,
    denoms: pd.DataFrame,
    tag_to_category: dict[str, str],
    admitted_tags: frozenset[str] | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    eligible = denoms.set_index("rollout_key")
    events = raw.copy()
    events = events[events["result_path"].isin(eligible.index)].copy()
    events["speaker_agent"] = events["speaker_agent"].astype(str)
    expected_agent = events["result_path"].map(eligible["speaker_agent"])
    events = events[events["speaker_agent"].eq(expected_agent)].copy()
    events["tag_code"] = events["tag_code"].astype(str)
    if admitted_tags is not None:
        events = events[events["tag_code"].isin(admitted_tags)].copy()
    events["category"] = events["tag_code"].map(tag_to_category)
    events = events[events["category"].isin(CATEGORIES)].copy()
    before_exact = len(events)

    # This is the identity used by the TTC validators.  Stable source/line
    # sorting makes the retained row deterministic when duplicate rows differ
    # only in rationale or confidence.
    events = events.sort_values(["_event_source_order", "_event_line"])
    if "rollout_id" not in events:
        events["rollout_id"] = None
    events["_rollout_identity"] = events["rollout_id"].where(
        events["rollout_id"].notna(), events["result_path"]
    )
    exact_key = [
        "_rollout_identity",
        "tag_code",
        "source_kind",
        "log_index",
        "interaction_index",
        "speaker_agent",
    ]
    for column in exact_key:
        if column not in events:
            events[column] = None
    events = events.drop_duplicates(exact_key, keep="first")
    after_exact = len(events)

    # Figure 4 counts at most one behavior-category event per speaker turn.
    category_turn_key = [
        "result_path",
        "speaker_agent",
        "round",
        "discussion_turn",
        "phase",
        "category",
    ]
    for column in category_turn_key:
        if column not in events:
            events[column] = None
    events = events.drop_duplicates(category_turn_key, keep="first")
    output = events[
        category_turn_key
        + ["tag_code", "source_kind", "log_index", "interaction_index", "_event_source"]
    ].copy()
    output = output.rename(columns={"result_path": "rollout_key"})
    return output, {
        "eligible_category_rows_before_exact_dedup": before_exact,
        "extra_exact_identity_rows_removed": before_exact - after_exact,
        "rows_after_exact_identity_dedup": after_exact,
        "extra_same_category_turn_rows_removed": after_exact - len(output),
        "rows_after_category_turn_dedup": len(output),
    }


def build_rollout_grid(denoms: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    counts = (
        events.groupby(["rollout_key", "category"], as_index=False)
        .size()
        .rename(columns={"size": "event_count"})
    )
    grid = pd.DataFrame(
        product(denoms["rollout_key"], CATEGORIES),
        columns=["rollout_key", "category"],
    )
    output = (
        grid.merge(denoms, on="rollout_key", how="left", validate="many_to_one")
        .merge(counts, on=["rollout_key", "category"], how="left", validate="one_to_one")
        .fillna({"event_count": 0})
    )
    output["event_count"] = output["event_count"].astype(int)
    return output


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    if x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def clustered_bootstrap_correlations(
    grid: pd.DataFrame, cohort_name: str, draws: int, rng: np.random.Generator
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for category in CATEGORIES:
        subset = grid[grid["category"].eq(category)].reset_index(drop=True)
        rho = safe_spearman(subset["event_count"], subset["utility"])
        groups: dict[str, list[np.ndarray]] = {}
        for stratum, stratum_df in subset.groupby("bootstrap_stratum", sort=True):
            groups[str(stratum)] = [
                group.index.to_numpy()
                for _, group in stratum_df.groupby("bootstrap_cluster", sort=True)
            ]
        boot = np.empty(draws, dtype=float)
        for draw in range(draws):
            sampled_indices: list[np.ndarray] = []
            for cluster_arrays in groups.values():
                picks = rng.integers(0, len(cluster_arrays), size=len(cluster_arrays))
                sampled_indices.extend(cluster_arrays[index] for index in picks)
            indices = np.concatenate(sampled_indices)
            sampled = subset.loc[indices]
            boot[draw] = safe_spearman(sampled["event_count"], sampled["utility"])
        valid = boot[np.isfinite(boot)]
        rows.append(
            {
                "cohort": cohort_name,
                "category": category,
                "spearman_rho": rho,
                "cluster_bootstrap_ci_low": float(np.quantile(valid, 0.025)),
                "cluster_bootstrap_ci_high": float(np.quantile(valid, 0.975)),
                "bootstrap_draws": draws,
                "rollout_sample_size": len(subset),
                "positive_event_rollouts": int((subset["event_count"] > 0).sum()),
                "sample_unit": "target-speaker rollout",
                "uncertainty": (
                    "seed-cluster bootstrap"
                    if set(groups) == {"ttc2160"}
                    else "stratified cluster bootstrap: bilateral model and TTC seed"
                ),
            }
        )
    return pd.DataFrame(rows)


def mean_ci(values: pd.Series) -> tuple[float, float, float]:
    values = values.dropna().astype(float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, float("nan"), float("nan")
    half = float(t.ppf(0.975, len(values) - 1) * sem(values))
    return mean, mean - half, mean + half


def build_effort_summary(ttc_grid: pd.DataFrame) -> pd.DataFrame:
    seed_means = (
        ttc_grid.groupby(
            ["model_family", "effort", "effort_index", "seed", "category"],
            as_index=False,
        )["event_count"]
        .mean()
        .rename(columns={"event_count": "seed_mean_events_per_rollout"})
    )
    rows: list[dict[str, Any]] = []
    for keys, group in seed_means.groupby(
        ["model_family", "effort", "effort_index", "category"], sort=True
    ):
        mean, low, high = mean_ci(group["seed_mean_events_per_rollout"])
        rows.append(
            {
                "model_family": keys[0],
                "effort": keys[1],
                "effort_index": int(keys[2]),
                "category": keys[3],
                "mean_events_per_rollout": mean,
                "seed_t_ci_low": low,
                "seed_t_ci_high": high,
                "n_seeds": group["seed"].nunique(),
                "n_rollouts": int(
                    ttc_grid[
                        ttc_grid["model_family"].eq(keys[0])
                        & ttc_grid["effort"].eq(keys[1])
                        & ttc_grid["category"].eq(keys[3])
                    ]["rollout_key"].nunique()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_competition_summary(ttc_grid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_category = (
        ttc_grid.groupby(
            [
                "model_family",
                "effort",
                "effort_index",
                "competition_index",
                "seed",
                "category",
            ],
            as_index=False,
        )["event_count"]
        .mean()
        .rename(columns={"event_count": "seed_mean_events_per_rollout"})
    )
    category_rows: list[dict[str, Any]] = []
    for keys, group in seed_category.groupby(
        ["model_family", "effort", "effort_index", "competition_index", "category"],
        sort=True,
    ):
        mean, low, high = mean_ci(group["seed_mean_events_per_rollout"])
        category_rows.append(
            {
                "model_family": keys[0],
                "effort": keys[1],
                "effort_index": int(keys[2]),
                "competition_index": float(keys[3]),
                "category": keys[4],
                "mean_events_per_rollout": mean,
                "seed_t_ci_low": low,
                "seed_t_ci_high": high,
                "n_seeds": group["seed"].nunique(),
            }
        )
    category = pd.DataFrame(category_rows)

    total = (
        ttc_grid.groupby(
            [
                "rollout_key",
                "model_family",
                "effort",
                "effort_index",
                "competition_index",
                "seed",
            ],
            as_index=False,
        )["event_count"]
        .sum()
        .groupby(
            ["model_family", "effort", "effort_index", "competition_index", "seed"],
            as_index=False,
        )["event_count"]
        .mean()
        .rename(columns={"event_count": "seed_mean_total_events_per_rollout"})
    )
    total_rows: list[dict[str, Any]] = []
    for keys, group in total.groupby(
        ["model_family", "effort", "effort_index", "competition_index"], sort=True
    ):
        mean, low, high = mean_ci(group["seed_mean_total_events_per_rollout"])
        total_rows.append(
            {
                "model_family": keys[0],
                "effort": keys[1],
                "effort_index": int(keys[2]),
                "competition_index": float(keys[3]),
                "mean_total_events_per_rollout": mean,
                "seed_t_ci_low": low,
                "seed_t_ci_high": high,
                "n_seeds": group["seed"].nunique(),
            }
        )
    return category, pd.DataFrame(total_rows)


def build_bilateral_elo_summary(bilateral_grid: pd.DataFrame) -> pd.DataFrame:
    return (
        bilateral_grid.groupby(["speaker_model", "speaker_elo", "category"], as_index=False)
        .agg(
            mean_events_per_rollout=("event_count", "mean"),
            rollout_count=("rollout_key", "nunique"),
        )
        .sort_values(["category", "speaker_elo", "speaker_model"])
    )


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", alpha=0.35)


def plot_correlations(table: pd.DataFrame, output: Path, title: str) -> None:
    table = table.set_index("category").loc[list(CATEGORIES)].reset_index()
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    y = np.arange(len(table))
    values = table["spearman_rho"].to_numpy()
    errors = np.vstack(
        [
            values - table["cluster_bootstrap_ci_low"].to_numpy(),
            table["cluster_bootstrap_ci_high"].to_numpy() - values,
        ]
    )
    ax.barh(y, values, color=[COLORS[c] for c in table["category"]], alpha=0.9)
    ax.errorbar(values, y, xerr=errors, fmt="none", color="black", capsize=3, lw=1.1)
    ax.set_yticks(y, [DISPLAY[c] for c in table["category"]])
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Spearman correlation with target utility")
    ax.set_title(title)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(output, dpi=240)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def plot_correlation_comparison(table: pd.DataFrame, output: Path) -> None:
    cohorts = ("bilateral1500", "ttc2160", "bilateral1500_plus_ttc2160")
    labels = ("Bilateral 1,500", "TTC 2,160", "Combined 3,660")
    markers = ("o", "s", "D")
    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    ybase = np.arange(len(CATEGORIES))
    offsets = (-0.20, 0.0, 0.20)
    for cohort, label, marker, offset in zip(cohorts, labels, markers, offsets):
        part = table[table["cohort"].eq(cohort)].set_index("category").loc[list(CATEGORIES)]
        x = part["spearman_rho"].to_numpy()
        xerr = np.vstack(
            [
                x - part["cluster_bootstrap_ci_low"].to_numpy(),
                part["cluster_bootstrap_ci_high"].to_numpy() - x,
            ]
        )
        ax.errorbar(x, ybase + offset, xerr=xerr, fmt=marker, capsize=3, label=label)
    ax.set_yticks(ybase, [DISPLAY[c] for c in CATEGORIES])
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Spearman correlation with target utility")
    ax.set_title("Behavior-utility association across analysis cohorts")
    ax.legend()
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(output, dpi=240)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def plot_effort_panels(summary: pd.DataFrame, out_dir: Path, scope: str) -> list[Path]:
    outputs: list[Path] = []
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), sharey=True)
    for ax, family in zip(axes, FAMILY_ORDER):
        part = summary[summary["model_family"].eq(family)]
        for category in CATEGORIES:
            line = part[part["category"].eq(category)].sort_values("effort_index")
            y = line["mean_events_per_rollout"].to_numpy()
            yerr = np.vstack(
                [
                    y - line["seed_t_ci_low"].to_numpy(),
                    line["seed_t_ci_high"].to_numpy() - y,
                ]
            )
            ax.errorbar(
                line["effort_index"],
                y,
                yerr=yerr,
                marker="o",
                color=COLORS[category],
                label=DISPLAY[category],
                capsize=2,
            )
        ax.set_xticks(range(4), EFFORT_ORDER[family])
        ax.set_title(FAMILY_DISPLAY[family])
        ax.set_xlabel("Requested reasoning effort")
        style_axis(ax)
    axes[0].set_ylabel("Mean category-turn events per rollout")
    axes[-1].legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle(f"TTC behavior intensity versus reasoning effort ({scope})")
    fig.tight_layout()
    combined = out_dir / f"ttc_mean_events_vs_effort_{scope}_three_models.png"
    fig.savefig(combined, dpi=240, bbox_inches="tight")
    fig.savefig(combined.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    outputs.extend([combined, combined.with_suffix(".pdf")])

    for family in FAMILY_ORDER:
        fig, ax = plt.subplots(figsize=(8.2, 5.5))
        part = summary[summary["model_family"].eq(family)]
        for category in CATEGORIES:
            line = part[part["category"].eq(category)].sort_values("effort_index")
            y = line["mean_events_per_rollout"].to_numpy()
            yerr = np.vstack(
                [
                    y - line["seed_t_ci_low"].to_numpy(),
                    line["seed_t_ci_high"].to_numpy() - y,
                ]
            )
            ax.errorbar(
                line["effort_index"],
                y,
                yerr=yerr,
                marker="o",
                color=COLORS[category],
                label=DISPLAY[category],
                capsize=2,
            )
        ax.set_xticks(range(4), EFFORT_ORDER[family])
        ax.set_xlabel("Requested reasoning effort")
        ax.set_ylabel("Mean category-turn events per rollout")
        ax.set_title(f"{FAMILY_DISPLAY[family]} ({scope})")
        ax.legend(fontsize=8)
        style_axis(ax)
        fig.tight_layout()
        path = out_dir / f"ttc_mean_events_vs_effort_{scope}_{family}.png"
        fig.savefig(path, dpi=240)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
        outputs.extend([path, path.with_suffix(".pdf")])
    return outputs


def plot_competition_effort_aligned(summary: pd.DataFrame, output: Path, scope: str) -> None:
    family_colors = {
        "gpt-5": "#d62728",
        "claude-sonnet-4-6": "#2ca02c",
        "gemini-3-flash": "#1f77b4",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), sharex=True, sharey=True)
    for effort_index, ax in enumerate(axes.flat):
        for family in FAMILY_ORDER:
            line = summary[
                summary["model_family"].eq(family)
                & summary["effort_index"].eq(effort_index)
            ].sort_values("competition_index")
            y = line["mean_total_events_per_rollout"].to_numpy()
            yerr = np.vstack(
                [
                    y - line["seed_t_ci_low"].to_numpy(),
                    line["seed_t_ci_high"].to_numpy() - y,
                ]
            )
            ax.errorbar(
                line["competition_index"],
                y,
                yerr=yerr,
                marker="o",
                color=family_colors[family],
                label=FAMILY_DISPLAY[family],
                capsize=2,
            )
        aligned = "minimal / low" if effort_index == 0 else (
            "low / medium" if effort_index == 1 else (
                "medium / high" if effort_index == 2 else "high / max"
            )
        )
        ax.set_title(f"Effort rank {effort_index + 1}: {aligned}")
        ax.set_xticks([0.0, 0.5, 1.0], ["cooperative", "middle", "competitive"])
        style_axis(ax)
    axes[0, 0].set_ylabel("Mean total category-turn events per rollout")
    axes[1, 0].set_ylabel("Mean total category-turn events per rollout")
    axes[1, 0].set_xlabel("Within-game competition index")
    axes[1, 1].set_xlabel("Within-game competition index")
    axes[0, 1].legend()
    fig.suptitle(f"TTC setting sensitivity by aligned effort rank ({scope})")
    fig.tight_layout()
    fig.savefig(output, dpi=240)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def plot_bilateral_elo(summary: pd.DataFrame, output: Path, scope: str) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 6.4))
    for category in CATEGORIES:
        line = summary[summary["category"].eq(category)].sort_values("speaker_elo")
        ax.plot(
            line["speaker_elo"],
            line["mean_events_per_rollout"],
            marker="o",
            markersize=4,
            lw=1.2,
            alpha=0.75,
            color=COLORS[category],
            label=DISPLAY[category],
        )
    ax.set_xlabel("Adversary Elo")
    ax.set_ylabel("Mean category-turn events per rollout")
    ax.set_title(f"Bilateral 1,500 model means versus Elo ({scope})")
    ax.legend(fontsize=8)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(output, dpi=240)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument(
        "--bilateral-replacement-events",
        type=Path,
        action="append",
        default=[],
        help="JSONL whose rollout paths replace canonical bilateral event rows",
    )
    parser.add_argument(
        "--ttc-replacement-events",
        type=Path,
        action="append",
        default=[],
        help="JSONL whose rollout paths replace canonical TTC event rows",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_dir.resolve()
    graphics = out / "graphics"
    graphics.mkdir(parents=True, exist_ok=True)
    tag_to_category = validate_codebooks()

    bilateral_denoms = load_bilateral_denominators()
    ttc_denoms = load_ttc_denominators()
    combined_denoms = pd.concat([bilateral_denoms, ttc_denoms], ignore_index=True, sort=False)

    bilateral_raw = load_event_rows([BILATERAL_EVENTS], "bilateral_canonical")
    bilateral_raw, bilateral_replacement = apply_replacement_events(
        bilateral_raw, args.bilateral_replacement_events, "bilateral_replacement"
    )
    ttc_raw = load_event_rows([source[2] for source in TTC_SOURCES], "ttc_canonical")
    ttc_raw, ttc_replacement = apply_replacement_events(
        ttc_raw, args.ttc_replacement_events, "ttc_replacement"
    )
    raw_identity_audit = {
        "bilateral": audit_raw_event_identities(bilateral_raw),
        "ttc": audit_raw_event_identities(ttc_raw),
    }

    all_outputs: list[Path] = []
    scope_results: dict[str, Any] = {}
    correlation_tables: list[pd.DataFrame] = []
    rng = np.random.default_rng(20260814)

    for scope, admitted in (("all50", None), ("selected23", SELECTED_23)):
        bilateral_events, bilateral_stats = normalize_events(
            bilateral_raw, bilateral_denoms, tag_to_category, admitted
        )
        ttc_events, ttc_stats = normalize_events(
            ttc_raw, ttc_denoms, tag_to_category, admitted
        )
        bilateral_grid = build_rollout_grid(bilateral_denoms, bilateral_events)
        ttc_grid = build_rollout_grid(ttc_denoms, ttc_events)
        combined_grid = pd.concat([bilateral_grid, ttc_grid], ignore_index=True, sort=False)

        for name, grid in (
            ("bilateral1500", bilateral_grid),
            ("ttc2160", ttc_grid),
            ("bilateral1500_plus_ttc2160", combined_grid),
        ):
            table = clustered_bootstrap_correlations(
                grid, name, args.bootstrap_draws, rng
            )
            table["tag_scope"] = scope
            correlation_tables.append(table)
            path = out / f"{name}_{scope}_spearman_utility.csv"
            table.to_csv(path, index=False)
            all_outputs.append(path)
            if name != "bilateral1500":
                figure_path = graphics / f"figure4_{name}_{scope}_spearman_utility.png"
                plot_correlations(
                    table,
                    figure_path,
                    (
                        "TTC 2,160: behavior association with utility"
                        if name == "ttc2160"
                        else "Bilateral 1,500 + TTC 2,160: behavior association with utility"
                    ),
                )
                all_outputs.extend([figure_path, figure_path.with_suffix(".pdf")])

        comparison = pd.concat(correlation_tables, ignore_index=True)
        comparison = comparison[comparison["tag_scope"].eq(scope)]
        comparison_path = graphics / f"figure4_correlation_comparison_{scope}.png"
        plot_correlation_comparison(comparison, comparison_path)
        all_outputs.extend([comparison_path, comparison_path.with_suffix(".pdf")])

        effort = build_effort_summary(ttc_grid)
        effort_path = out / f"ttc2160_{scope}_mean_events_by_effort.csv"
        effort.to_csv(effort_path, index=False)
        all_outputs.append(effort_path)
        all_outputs.extend(plot_effort_panels(effort, graphics, scope))

        competition_category, competition_total = build_competition_summary(ttc_grid)
        comp_category_path = out / f"ttc2160_{scope}_category_events_by_competition.csv"
        comp_total_path = out / f"ttc2160_{scope}_total_events_by_competition.csv"
        competition_category.to_csv(comp_category_path, index=False)
        competition_total.to_csv(comp_total_path, index=False)
        all_outputs.extend([comp_category_path, comp_total_path])
        comp_figure = graphics / f"ttc2160_{scope}_competition_effort_aligned.png"
        plot_competition_effort_aligned(competition_total, comp_figure, scope)
        all_outputs.extend([comp_figure, comp_figure.with_suffix(".pdf")])

        bilateral_elo = build_bilateral_elo_summary(bilateral_grid)
        elo_path = out / f"bilateral1500_{scope}_mean_events_by_elo.csv"
        bilateral_elo.to_csv(elo_path, index=False)
        all_outputs.append(elo_path)
        elo_figure = graphics / f"bilateral1500_{scope}_mean_events_by_elo.png"
        plot_bilateral_elo(bilateral_elo, elo_figure, scope)
        all_outputs.extend([elo_figure, elo_figure.with_suffix(".pdf")])

        rollout_path = out / f"ttc2160_{scope}_rollout_category_counts.csv.gz"
        ttc_grid.to_csv(rollout_path, index=False, compression="gzip")
        combined_path = out / f"bilateral1500_plus_ttc2160_{scope}_rollout_category_counts.csv.gz"
        combined_grid.to_csv(combined_path, index=False, compression="gzip")
        all_outputs.extend([rollout_path, combined_path])

        scope_results[scope] = {
            "bilateral_event_normalization": bilateral_stats,
            "ttc_event_normalization": ttc_stats,
            "bilateral_rollouts": bilateral_grid["rollout_key"].nunique(),
            "ttc_rollouts": ttc_grid["rollout_key"].nunique(),
            "combined_rollouts": combined_grid["rollout_key"].nunique(),
        }

    all_correlations = pd.concat(correlation_tables, ignore_index=True)
    correlation_path = out / "all_cohort_spearman_utility.csv"
    all_correlations.to_csv(correlation_path, index=False)
    all_outputs.append(correlation_path)

    elo_status = {
        "requested_quantity": "TTC mean behavior events per rollout versus adversary Elo",
        "status": "not_estimable_from_retained_paper_data",
        "reason": (
            "The TTC target models GPT-5, Claude Sonnet 4.6, and Gemini 3 Flash are "
            "outside the paper's 30-model Elo roster, and TTC target speaker_elo is null."
        ),
        "action": "No Elo values were imputed or borrowed from other model configurations.",
        "retained_direct_elo_output": "bilateral1500_*_mean_events_by_elo.*",
        "ttc_fallback": (
            "The TTC sensitivity uses the exact three within-game competition settings, "
            "ordered as 0=cooperative, 0.5=middle, 1=competitive, in four effort-rank panels."
        ),
    }
    elo_status_path = out / "ttc_adversary_elo_status.json"
    elo_status_path.write_text(json.dumps(elo_status, indent=2) + "\n", encoding="utf-8")
    all_outputs.append(elo_status_path)

    report_lines = [
        "# TTC and combined qualitative diagnostics",
        "",
        "## Cohorts",
        "",
        "- The TTC-only cohort has 2,160 target-speaker rollouts.",
        "- Each TTC family contributes 720 rollouts across ten seeds and four effort levels.",
        "- The combined cohort has 3,660 rollouts: 1,500 bilateral plus 2,160 TTC.",
        "- The analysis mixes judge patches because the user explicitly approved this choice.",
        "",
        "## Main scope and sensitivity",
        "",
        "- The `selected23` scope is paper-comparable and uses the active Figure 4 tag set.",
        "- The `all50` scope is a full-codebook sensitivity over the same six categories.",
        "- Counts are de-duplicated first by validator event identity and then by rollout, turn, and category.",
        "",
        "## Spearman correlations with utility",
        "",
        "- The sample unit is one target-speaker rollout.",
        "- TTC intervals resample the ten seeds as clusters.",
        "- Combined intervals use a stratified cluster bootstrap over bilateral model and TTC seed.",
        "- The coefficients are descriptive associations and do not estimate behavior effects.",
        "",
    ]
    for scope, scope_title in (
        ("selected23", "Selected-23 paper-comparable primary scope"),
        ("all50", "Full-50-codebook sensitivity"),
    ):
        scope_corr = all_correlations[all_correlations["tag_scope"].eq(scope)]
        report_lines.extend(
            [
                f"### {scope_title}",
                "",
                "| Cohort | Category | Spearman rho | 95% cluster interval |",
                "|---|---|---:|---:|",
            ]
        )
        for cohort in ("bilateral1500", "ttc2160", "bilateral1500_plus_ttc2160"):
            for category in CATEGORIES:
                row = scope_corr[
                    scope_corr["cohort"].eq(cohort)
                    & scope_corr["category"].eq(category)
                ].iloc[0]
                report_lines.append(
                    f"| {cohort} | {DISPLAY[category]} | {row['spearman_rho']:+.3f} | "
                    f"[{row['cluster_bootstrap_ci_low']:+.3f}, "
                    f"{row['cluster_bootstrap_ci_high']:+.3f}] |"
                )
        report_lines.append("")
    report_lines.extend(
        [
            "",
            "## Elo limit",
            "",
            "- TTC target-speaker Elo is null in every retained TTC manifest.",
            "- GPT-5, Claude Sonnet 4.6, and Gemini 3 Flash are outside the paper's 30-model Elo roster.",
            "- The analysis does not assign proxy Elo values from other model configurations.",
            "- The TTC fallback uses the exact cooperative, middle, and competitive settings within each game.",
            "- The bilateral 1,500 outputs retain the direct adversary-Elo analysis.",
            "",
            "## Repair inputs",
            "",
            f"- Applied bilateral replacements for {bilateral_replacement['replaced_rollouts']} "
            f"validated paths and {bilateral_replacement['replacement_rows']} event rows.",
            f"- Applied TTC replacements for {ttc_replacement['replaced_rollouts']} "
            f"validated paths and {ttc_replacement['replacement_rows']} event rows.",
            "- Both replacement aggregates passed their strict validation gates before this build.",
            "",
        ]
    )
    report_path = out / "REPORT.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    all_outputs.append(report_path)

    input_paths = [
        PRIMARY_CSV,
        BILATERAL_MANIFEST,
        BILATERAL_EVENTS,
        BILATERAL_CODEBOOK,
        *[path for source in TTC_SOURCES for path in source[1:]],
        *args.bilateral_replacement_events,
        *args.ttc_replacement_events,
    ]
    provenance = {
        "status": "separate_diagnostic_not_promoted_to_paper",
        "created_by": str(Path(__file__).resolve()),
        "tag_scopes": {
            "selected23": "paper-comparable six categories restricted to the active Figure 4 selected 23 tags",
            "all50": "full-codebook sensitivity using the same six categories and all 50 tags",
        },
        "user_authorized_mixed_judge_density": True,
        "sample_unit": "target-speaker rollout",
        "deduplication_keys": {
            "validator_event_identity": [
                "rollout_id (or absolute result_path when absent)",
                "tag_code",
                "source_kind",
                "log_index",
                "interaction_index",
                "speaker_agent",
            ],
            "paper_category_turn_identity": [
                "absolute result_path",
                "speaker_agent",
                "round",
                "discussion_turn",
                "phase",
                "category",
            ],
        },
        "uncertainty": {
            "ttc": "cluster bootstrap over ten experimental seeds",
            "combined": "stratified cluster bootstrap over 30 bilateral models and ten TTC seeds",
            "effort_and_competition_means": "95% t interval over ten seed means",
        },
        "cohort_counts": {
            "bilateral": len(bilateral_denoms),
            "ttc": len(ttc_denoms),
            "combined": len(combined_denoms),
            "ttc_by_family": ttc_denoms.groupby("model_family").size().to_dict(),
            "ttc_seeds": sorted(int(value) for value in ttc_denoms["seed"].unique()),
        },
        "replacement_inputs": {
            "bilateral": bilateral_replacement,
            "ttc": ttc_replacement,
        },
        "raw_event_identity_audit": raw_identity_audit,
        "annotation_path_coverage": {
            "bilateral_manifest_rollouts": len(bilateral_denoms),
            "bilateral_rollouts_with_at_least_one_retained_event_row": int(
                bilateral_denoms["result_path"].isin(set(bilateral_raw["result_path"])).sum()
            ),
            "ttc_manifest_rollouts": len(ttc_denoms),
            "ttc_rollouts_with_at_least_one_retained_event_row": int(
                ttc_denoms["result_path"].isin(set(ttc_raw["result_path"])).sum()
            ),
            "note": (
                "An event-row path count cannot distinguish a reviewed zero-event transcript "
                "from a missing review; completion ledgers remain authoritative."
            ),
        },
        "normalization": scope_results,
        "elo_status": elo_status,
        "inputs": {str(path.resolve()): sha256(path.resolve()) for path in input_paths},
        "outputs": {},
    }
    for path in all_outputs:
        provenance["outputs"][str(path.resolve())] = sha256(path.resolve())
    provenance_path = out / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(out),
                "bilateral_rollouts": len(bilateral_denoms),
                "ttc_rollouts": len(ttc_denoms),
                "combined_rollouts": len(combined_denoms),
                "outputs": len(all_outputs),
                "provenance": str(provenance_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

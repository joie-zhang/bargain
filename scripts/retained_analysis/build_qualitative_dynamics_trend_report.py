#!/usr/bin/env python3
"""Build plots and a research-style report from qualitative rollout tags."""

from __future__ import annotations

import json
import math
import re
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import statsmodels.formula.api as smf
except Exception:  # pragma: no cover - optional dependency in some envs
    smf = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUAL_DIR = PROJECT_ROOT / "analysis/qualitative_rollout_dynamics_20260628"
OUT_DIR = PROJECT_ROOT / "analysis/qualitative_dynamics_trends_20260628"
FIG_DIR = OUT_DIR / "figures"

QUAL_CSV = QUAL_DIR / "refined_rollout_dynamics_coding.csv"
CODEBOOK_CSV = QUAL_DIR / "refined_dynamics_codebook.csv"
AGENT_TABLES = [
    PROJECT_ROOT
    / "experiments/results/n2_plus_multiagent_comparison_analysis_20260505/tables_multiagent/heterogeneous_agents_fresh.csv",
    PROJECT_ROOT
    / "experiments/results/n2_plus_multiagent_comparison_analysis_20260505/tables_multiagent/homogeneous_agents_fresh.csv",
]

FAMILY_ORDER = ["heterogeneous_random", "homogeneous_adversary", "homogeneous_control"]
FAMILY_LABELS = {
    "heterogeneous_random": "Heterogeneous",
    "homogeneous_adversary": "Hom. adversary",
    "homogeneous_control": "Hom. control",
}
GAME_ORDER = ["game1", "game2", "game3"]
GAME_LABELS = {"game1": "Game 1", "game2": "Game 2", "game3": "Game 3"}
N_ORDER = [2, 4, 6, 8, 10]
OUTCOME_ORDER = ["Round 1", "Round 2-3", "Round 4-9", "No consensus"]
OUTCOME_COLORS = {
    "Round 1": "#3B82A0",
    "Round 2-3": "#78A641",
    "Round 4-9": "#D18F2F",
    "No consensus": "#B84A62",
}
FAMILY_COLORS = {
    "heterogeneous_random": "#B84A62",
    "homogeneous_adversary": "#3B82A0",
    "homogeneous_control": "#78A641",
}


def clean_name(code: str) -> str:
    replacements = {
        "outcome_consensus_r1": "Round-1 consensus",
        "outcome_consensus_r2_r3": "Round 2-3 repair",
        "outcome_late_consensus_r4_r9": "Late consensus",
        "outcome_no_consensus_r10": "No consensus",
        "semantic_vector_or_ballot_drift": "Vector/ballot drift",
        "verbal_convergence_vote_failure": "Verbal convergence failure",
        "minimum_winning_supermajority": "Minimum-winning vote",
        "high_inequality_outcome": "High inequality",
        "accepted_with_scatter": "Accepted with scatter",
        "failed_with_scatter": "Failed with scatter",
        "redline_then_package": "Redline then package",
        "proposal_scatter": "Proposal scatter",
    }
    if code in replacements:
        return replacements[code]
    return code.replace("_", " ").title()


def sem(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return math.nan
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


def ci95(values: pd.Series) -> float:
    s = sem(values)
    return 1.96 * s if math.isfinite(s) else math.nan


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def gini(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.0
    if float(arr.min()) < 0:
        arr = arr - float(arr.min())
    if np.allclose(arr, 0.0):
        return 0.0
    diffs = np.abs(arr[:, None] - arr[None, :])
    raw = float(np.mean(diffs) / (2.0 * float(arr.mean())))
    return min(raw * float(arr.size / (arr.size - 1)), 1.0)


def load_agent_run_metrics() -> pd.DataFrame:
    agents = pd.concat([pd.read_csv(path) for path in AGENT_TABLES], ignore_index=True)
    for col in ["elo", "final_utility", "n_agents", "competition_value", "competition_level", "rho", "theta", "sigma", "alpha"]:
        if col in agents.columns:
            agents[col] = pd.to_numeric(agents[col], errors="coerce")
    agents["n_agents"] = agents["n_agents"].astype(int)

    rows = []
    for path, group in agents.groupby("result_path", sort=False):
        utilities = group["final_utility"].to_numpy(dtype=float)
        elos = group["elo"].dropna().to_numpy(dtype=float)
        sorted_elos = np.sort(elos)
        top_advantage = math.nan
        if len(sorted_elos) >= 2:
            top_advantage = float(sorted_elos[-1] - np.mean(sorted_elos[:-1]))
        adv = group[group["role"].astype(str).str.contains("adversary", na=False)]
        base = group[group["role"].astype(str).str.contains("baseline", na=False)]
        row = {
            "result_path": path,
            "config_id_agent": int(group["config_id"].iloc[0]),
            "game_label_agent": group["game_label"].iloc[0],
            "experiment_family_agent": group["experiment_family"].iloc[0],
            "n_agents_agent": int(group["n_agents"].iloc[0]),
            "competition_value": float(group["competition_value"].iloc[0]) if "competition_value" in group else math.nan,
            "competition_level": float(group["competition_level"].iloc[0]) if "competition_level" in group else math.nan,
            "rho": float(group["rho"].iloc[0]) if "rho" in group else math.nan,
            "theta": float(group["theta"].iloc[0]) if "theta" in group else math.nan,
            "sigma": float(group["sigma"].iloc[0]) if "sigma" in group else math.nan,
            "alpha": float(group["alpha"].iloc[0]) if "alpha" in group else math.nan,
            "competition_band": group["competition_band"].iloc[0] if "competition_band" in group else "",
            "mean_roster_elo": float(np.mean(elos)) if len(elos) else math.nan,
            "min_roster_elo": float(np.min(elos)) if len(elos) else math.nan,
            "max_roster_elo": float(np.max(elos)) if len(elos) else math.nan,
            "elo_std": float(np.std(elos, ddof=0)) if len(elos) else math.nan,
            "elo_range": float(np.max(elos) - np.min(elos)) if len(elos) else math.nan,
            "top_elo_advantage": top_advantage,
            "mean_payoff": float(np.mean(utilities)) if len(utilities) else math.nan,
            "total_payoff": float(np.sum(utilities)) if len(utilities) else math.nan,
            "min_payoff": float(np.min(utilities)) if len(utilities) else math.nan,
            "max_payoff": float(np.max(utilities)) if len(utilities) else math.nan,
            "payoff_std": float(np.std(utilities, ddof=0)) if len(utilities) else math.nan,
            "payoff_gini": gini(utilities),
            "adversary_elo": float(adv["elo"].dropna().iloc[0]) if len(adv["elo"].dropna()) else math.nan,
            "adversary_model_short": adv["model_short"].iloc[0] if len(adv) else "",
            "baseline_elo": float(base["elo"].dropna().mean()) if len(base["elo"].dropna()) else math.nan,
        }
        row["adversary_elo_gap"] = row["adversary_elo"] - row["baseline_elo"] if math.isfinite(row["adversary_elo"]) and math.isfinite(row["baseline_elo"]) else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def load_qualitative() -> pd.DataFrame:
    q = pd.read_csv(QUAL_CSV)
    q["consensus_reached"] = q["consensus_reached"].apply(parse_bool)
    q["final_round"] = pd.to_numeric(q["final_round"], errors="coerce")
    q["n_agents"] = pd.to_numeric(q["n_agents"], errors="coerce").astype(int)
    for col in ["max_accept_final_vote", "threshold_final_vote", "positive_nonfunded_count", "funded_count", "utility_range"]:
        q[col] = pd.to_numeric(q[col], errors="coerce")
    tag_sets = q["refined_dynamic_codes"].fillna("").apply(lambda s: {x for x in str(s).split(";") if x})
    codes = sorted(set().union(*tag_sets.tolist()))
    for code in codes:
        q[f"tag_{code}"] = tag_sets.apply(lambda tags, c=code: int(c in tags))
    q["outcome_bucket"] = np.select(
        [
            q["consensus_reached"] & q["final_round"].eq(1),
            q["consensus_reached"] & q["final_round"].between(2, 3),
            q["consensus_reached"] & q["final_round"].between(4, 9),
            ~q["consensus_reached"],
        ],
        OUTCOME_ORDER,
        default="Other",
    )
    q["vote_margin"] = q["max_accept_final_vote"] - q["threshold_final_vote"]
    q["vote_margin_norm"] = q["vote_margin"] / q["n_agents"].replace(0, np.nan)
    q["family_label"] = q["experiment_family"].map(FAMILY_LABELS)
    q["game_name"] = q["game_label"].map(GAME_LABELS)
    q["delayed_or_failed"] = (~(q["consensus_reached"] & q["final_round"].eq(1))).astype(int)
    return q


def top_item_features(path: str) -> dict[str, object]:
    try:
        with open(path) as handle:
            data = json.load(handle)
    except Exception:
        return {}
    cfg = data.get("config") or {}
    logs = data.get("conversation_logs") or []
    cosine_values = []
    for value in (cfg.get("actual_pairwise_cosines") or {}).values():
        try:
            cosine_values.append(float(value))
        except (TypeError, ValueError):
            pass
    out = {
        "pairwise_cosine_mean": float(np.mean(cosine_values)) if cosine_values else math.nan,
        "vote_tie_text_count": sum(
            1
            for e in logs
            if e.get("phase") == "vote_tabulation" and re.search(r"\btie\b|random", str(e.get("content", "")), re.IGNORECASE)
        ),
    }
    if cfg.get("game_label") != "game1":
        return out
    prefs = data.get("agent_preferences") or {}
    top_sets: dict[str, set[int]] = {}
    for agent, values in prefs.items():
        if not isinstance(values, list) or not values:
            continue
        arr = np.asarray(values, dtype=float)
        max_val = float(np.max(arr))
        top_sets[agent] = {int(idx) for idx, value in enumerate(arr) if math.isclose(float(value), max_val)}
    counts = Counter(idx for top in top_sets.values() for idx in top)
    contested_agents = 0
    for top in top_sets.values():
        if any(counts[idx] > 1 for idx in top):
            contested_agents += 1
    final_allocation = data.get("final_allocation") or {}
    satisfied = 0
    for agent, top in top_sets.items():
        got = final_allocation.get(agent) or []
        if any(int(item) in top for item in got):
            satisfied += 1
    n = len(top_sets)
    out.update(
        {
            "game1_max_top_claimants": max(counts.values()) if counts else math.nan,
            "game1_unique_top_item_share": len(counts) / n if n else math.nan,
            "game1_contested_top_agent_share": contested_agents / n if n else math.nan,
            "game1_top_item_satisfied_share": satisfied / n if n else math.nan,
            "game1_top_collision": int(any(value > 1 for value in counts.values())) if counts else 0,
        }
    )
    return out


def build_merged_frame() -> pd.DataFrame:
    q = load_qualitative()
    run_metrics = load_agent_run_metrics()
    df = q.merge(run_metrics, on="result_path", how="left", validate="one_to_one")
    features = pd.DataFrame([{"result_path": path, **top_item_features(path)} for path in df["result_path"]])
    df = df.merge(features, on="result_path", how="left", validate="one_to_one")
    df["rho_group"] = np.where(df["rho"].gt(0), "high alignment", "negative-rho conflict")
    df["theta_label"] = df["theta"].map(lambda x: f"theta={x:g}" if math.isfinite(x) else "")
    df["sigma_alpha"] = df.apply(
        lambda r: f"sigma={r['sigma']:g}, alpha={r['alpha']:g}" if math.isfinite(r.get("sigma", math.nan)) and math.isfinite(r.get("alpha", math.nan)) else "",
        axis=1,
    )
    return df


def savefig(fig: plt.Figure, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def summarize_rate(df: pd.DataFrame, group_cols: list[str], col: str) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        y = pd.to_numeric(sub[col], errors="coerce").dropna()
        row = dict(zip(group_cols, keys, strict=True))
        row.update(
            {
                "n": int(len(y)),
                "rate": float(y.mean()) if len(y) else math.nan,
                "ci95": ci95(y),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_dynamics_fingerprint(df: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    tags = [
        "single_item_mutual_support_pact",
        "low_value_trading_pool",
        "shared_top_item_deadlock",
        "policy_basket_logrolling",
        "redline_then_package",
        "staged_verification_governance",
        "midpoint_closure",
        "spine_focal_point",
        "feasibility_first_budget_math",
        "numeric_pledge_split",
        "single_project_rally",
        "near_threshold_rescue",
        "zero_value_holdout",
        "proposal_scatter",
        "accepted_with_scatter",
        "failed_with_scatter",
        "side_payment_smoothing",
        "hard_anchor_or_redline",
        "sequenced_or_contingent_deal",
        "minimum_winning_supermajority",
        "high_inequality_outcome",
        "template_role_artifact",
        "outcome_no_consensus_r10",
    ]
    rows = []
    for family in FAMILY_ORDER:
        for game in GAME_ORDER:
            sub = df[df["experiment_family"].eq(family) & df["game_label"].eq(game)]
            for tag in tags:
                rows.append(
                    {
                        "dynamic": clean_name(tag),
                        "family_game": f"{FAMILY_LABELS[family]}\n{GAME_LABELS[game]}",
                        "share": float(sub[f"tag_{tag}"].mean()) if len(sub) else math.nan,
                        "n": len(sub),
                    }
                )
    table = pd.DataFrame(rows)
    pivot = table.pivot(index="dynamic", columns="family_game", values="share")
    fig, ax = plt.subplots(figsize=(12.5, 10.5))
    sns.heatmap(
        pivot.loc[[clean_name(t) for t in tags]],
        ax=ax,
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Rollout share"},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Qualitative Dynamics Fingerprint By Experiment Family And Game", fontsize=15, pad=12)
    ax.tick_params(axis="x", labelrotation=35, labelsize=8)
    ax.tick_params(axis="y", labelsize=9)
    path = savefig(fig, "01_dynamics_fingerprint_heatmap.png")
    return path, table


def plot_consensus_timing(df: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    rows = []
    for game in GAME_ORDER:
        for family in FAMILY_ORDER:
            sub = df[df["game_label"].eq(game) & df["experiment_family"].eq(family)]
            counts = sub["outcome_bucket"].value_counts()
            denom = len(sub)
            for bucket in OUTCOME_ORDER:
                rows.append(
                    {
                        "game_label": game,
                        "family": family,
                        "bucket": bucket,
                        "count": int(counts.get(bucket, 0)),
                        "share": counts.get(bucket, 0) / denom if denom else 0,
                        "n": denom,
                    }
                )
    table = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)
    for ax, game in zip(axes, GAME_ORDER, strict=True):
        left = np.zeros(len(FAMILY_ORDER))
        for bucket in OUTCOME_ORDER:
            vals = [
                table[
                    table["game_label"].eq(game)
                    & table["family"].eq(family)
                    & table["bucket"].eq(bucket)
                ]["share"].iloc[0]
                for family in FAMILY_ORDER
            ]
            ax.barh(
                [FAMILY_LABELS[f] for f in FAMILY_ORDER],
                vals,
                left=left,
                color=OUTCOME_COLORS[bucket],
                edgecolor="white",
                linewidth=0.8,
                label=bucket,
            )
            left += np.asarray(vals)
        ax.set_title(GAME_LABELS[game], fontsize=12)
        ax.set_xlim(0, 1)
        ax.grid(axis="x", alpha=0.22)
        ax.set_xlabel("Share of rollouts")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    axes[-1].legend(loc="lower right", bbox_to_anchor=(1.05, -0.28), ncol=2, frameon=False)
    fig.suptitle("Consensus Timing By Family And Game", fontsize=15, y=1.03)
    path = savefig(fig, "02_consensus_timing_stack.png")
    return path, table


def plot_hom_adv_elo(df: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    tags = [
        "outcome_consensus_r1",
        "hard_anchor_or_redline",
        "minimum_winning_supermajority",
        "high_inequality_outcome",
        "template_role_artifact",
        "baseline_mirroring_or_deference",
    ]
    sub = df[df["experiment_family"].eq("homogeneous_adversary") & df["adversary_elo"].notna()].copy()
    rows = []
    for (game, model, elo), g in sub.groupby(["game_label", "adversary_model_short", "adversary_elo"], dropna=False):
        for tag in tags:
            rows.append(
                {
                    "game_label": game,
                    "adversary_model_short": model,
                    "adversary_elo": float(elo),
                    "tag": tag,
                    "rate": float(g[f"tag_{tag}"].mean()),
                    "ci95": ci95(g[f"tag_{tag}"]),
                    "n": len(g),
                }
            )
    table = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2), sharex=True, sharey=False)
    axes = axes.ravel()
    game_colors = {"game1": "#B84A62", "game2": "#3B82A0", "game3": "#78A641"}
    for ax, tag in zip(axes, tags, strict=True):
        t = table[table["tag"].eq(tag)]
        for game in GAME_ORDER:
            g = t[t["game_label"].eq(game)].sort_values("adversary_elo")
            ax.errorbar(
                g["adversary_elo"],
                g["rate"],
                yerr=g["ci95"].fillna(0),
                marker="o",
                linewidth=1.6,
                capsize=2,
                color=game_colors[game],
                label=GAME_LABELS[game],
                alpha=0.9,
            )
        ax.axvline(1337, color="#555555", lw=1, ls="--", alpha=0.5)
        ax.set_title(clean_name(tag), fontsize=11)
        ax.set_ylim(-0.04, 1.04)
        ax.grid(alpha=0.22)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("Tag frequency")
    axes[3].set_ylabel("Tag frequency")
    for ax in axes[3:]:
        ax.set_xlabel("Inserted model Elo")
    axes[2].legend(frameon=False, loc="lower right", fontsize=8)
    fig.suptitle("Homogeneous-Adversary Dynamics Across Inserted-Model Elo", fontsize=15, y=1.02)
    path = savefig(fig, "03_homogeneous_adversary_dynamics_vs_elo.png")
    return path, table


def plot_heterogeneous_elo_dispersion(df: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    tags = [
        "outcome_consensus_r1",
        "minimum_winning_supermajority",
        "hard_anchor_or_redline",
        "high_inequality_outcome",
        "template_role_artifact",
        "outcome_consensus_r2_r3",
    ]
    sub = df[df["experiment_family"].eq("heterogeneous_random")].copy()
    sub["elo_std_bin"] = pd.qcut(sub["elo_std"], 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"], duplicates="drop")
    rows = []
    for (game, bin_label), g in sub.groupby(["game_label", "elo_std_bin"], observed=False):
        for tag in tags:
            rows.append(
                {
                    "game_label": game,
                    "elo_std_bin": str(bin_label),
                    "elo_std_mean": float(g["elo_std"].mean()),
                    "tag": tag,
                    "rate": float(g[f"tag_{tag}"].mean()),
                    "ci95": ci95(g[f"tag_{tag}"]),
                    "n": len(g),
                }
            )
    table = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2), sharex=True)
    axes = axes.ravel()
    game_colors = {"game1": "#B84A62", "game2": "#3B82A0", "game3": "#78A641"}
    for ax, tag in zip(axes, tags, strict=True):
        t = table[table["tag"].eq(tag)]
        for game in GAME_ORDER:
            g = t[t["game_label"].eq(game)].sort_values("elo_std_mean")
            ax.errorbar(
                g["elo_std_mean"],
                g["rate"],
                yerr=g["ci95"].fillna(0),
                marker="o",
                linewidth=1.6,
                capsize=2,
                color=game_colors[game],
                label=GAME_LABELS[game],
                alpha=0.9,
            )
        ax.set_title(clean_name(tag), fontsize=11)
        ax.set_ylim(-0.04, 1.04)
        ax.grid(alpha=0.22)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    for ax in axes[3:]:
        ax.set_xlabel("Mean roster Elo std. in quartile")
    axes[0].set_ylabel("Tag frequency")
    axes[3].set_ylabel("Tag frequency")
    axes[2].legend(frameon=False, loc="lower right", fontsize=8)
    fig.suptitle("Heterogeneous Dynamics Across Roster Elo Dispersion", fontsize=15, y=1.02)
    path = savefig(fig, "04_heterogeneous_dynamics_by_elo_dispersion.png")
    return path, table


def plot_coalition_scaling(df: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    metrics = [
        ("tag_minimum_winning_supermajority", "Minimum-winning rate"),
        ("vote_margin_norm", "Mean vote margin / N"),
        ("tag_outcome_no_consensus_r10", "No-consensus rate"),
    ]
    rows = []
    for (family, game, n), g in df.groupby(["experiment_family", "game_label", "n_agents"], dropna=False):
        for col, label in metrics:
            y = pd.to_numeric(g[col], errors="coerce").dropna()
            rows.append(
                {
                    "experiment_family": family,
                    "game_label": game,
                    "n_agents": int(n),
                    "metric": label,
                    "value": float(y.mean()) if len(y) else math.nan,
                    "ci95": ci95(y),
                    "n": len(y),
                }
            )
    table = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(14.7, 4.5), sharex=True)
    for ax, (_, label) in zip(axes, metrics, strict=True):
        t = table[table["metric"].eq(label)]
        for family in FAMILY_ORDER:
            g = t[t["experiment_family"].eq(family)].groupby("n_agents", as_index=False).agg(value=("value", "mean"))
            ax.plot(
                g["n_agents"],
                g["value"],
                marker="o",
                linewidth=2,
                color=FAMILY_COLORS[family],
                label=FAMILY_LABELS[family],
            )
        ax.set_title(label, fontsize=12)
        ax.set_xticks(N_ORDER)
        ax.grid(alpha=0.22)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("Share / normalized margin")
    axes[1].set_xlabel("Number of agents")
    axes[2].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Coalition Politics Scale With Group Size", fontsize=15, y=1.03)
    path = savefig(fig, "05_coalition_scaling_by_n.png")
    return path, table


def plot_game1_top_collision(df: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    sub = df[df["game_label"].eq("game1")].copy()
    sub["collision_bin"] = pd.cut(
        sub["game1_contested_top_agent_share"],
        bins=[-0.01, 0.0, 0.5, 0.999, 1.01],
        labels=["none", "some", "most", "all"],
    )
    metrics = [
        ("tag_outcome_consensus_r1", "Round-1 consensus"),
        ("tag_shared_top_item_deadlock", "Shared-top deadlock language"),
        ("game1_top_item_satisfied_share", "Top-item satisfaction"),
        ("payoff_gini", "Utility Gini"),
    ]
    rows = []
    for (family, collision), g in sub.groupby(["experiment_family", "collision_bin"], observed=False):
        for col, label in metrics:
            y = pd.to_numeric(g[col], errors="coerce").dropna()
            rows.append(
                {
                    "experiment_family": family,
                    "collision_bin": str(collision),
                    "metric": label,
                    "value": float(y.mean()) if len(y) else math.nan,
                    "ci95": ci95(y),
                    "n": len(y),
                }
            )
    table = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.4), sharex=True)
    axes = axes.ravel()
    x_order = ["none", "some", "most", "all"]
    x = np.arange(len(x_order))
    width = 0.24
    for ax, (_, label) in zip(axes, metrics, strict=True):
        t = table[table["metric"].eq(label)]
        for idx, family in enumerate(FAMILY_ORDER):
            g = t[t["experiment_family"].eq(family)].set_index("collision_bin").reindex(x_order)
            ax.errorbar(
                x + (idx - 1) * width,
                g["value"],
                yerr=g["ci95"].fillna(0),
                fmt="o-",
                color=FAMILY_COLORS[family],
                capsize=2,
                label=FAMILY_LABELS[family],
            )
        ax.set_title(label, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(x_order)
        ax.grid(alpha=0.22)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("Mean / share")
    axes[2].set_ylabel("Mean / share")
    axes[2].set_xlabel("Share of agents whose top item is contested")
    axes[3].set_xlabel("Share of agents whose top item is contested")
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Game 1: Top-Item Collision Links Mechanism To Outcome", fontsize=15, y=1.02)
    path = savefig(fig, "06_game1_top_item_collision_mechanism.png")
    return path, table


def plot_game2_mechanisms(df: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    tags = ["redline_then_package", "staged_verification_governance", "midpoint_closure", "spine_focal_point", "outcome_consensus_r1"]
    sub = df[df["game_label"].eq("game2")].copy()
    rows = []
    for (family, rho_group, theta), g in sub.groupby(["experiment_family", "rho_group", "theta"], dropna=False):
        for tag in tags:
            rows.append(
                {
                    "experiment_family": family,
                    "rho_group": rho_group,
                    "theta": float(theta),
                    "tag": tag,
                    "rate": float(g[f"tag_{tag}"].mean()),
                    "ci95": ci95(g[f"tag_{tag}"]),
                    "n": len(g),
                }
            )
    table = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, len(tags), figsize=(16.0, 4.2), sharey=True)
    for ax, tag in zip(axes, tags, strict=True):
        t = table[table["tag"].eq(tag)].copy()
        for family in FAMILY_ORDER:
            g = (
                t[t["experiment_family"].eq(family)]
                .groupby(["rho_group"], as_index=False)
                .agg(rate=("rate", "mean"))
            )
            x = np.arange(len(g))
            offsets = {"heterogeneous_random": -0.22, "homogeneous_adversary": 0.0, "homogeneous_control": 0.22}
            ax.scatter(
                x + offsets[family],
                g["rate"],
                s=70,
                color=FAMILY_COLORS[family],
                label=FAMILY_LABELS[family],
                alpha=0.9,
            )
        ax.set_title(clean_name(tag), fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["aligned", "conflict"], rotation=25)
        ax.set_ylim(-0.04, 1.04)
        ax.grid(axis="y", alpha=0.22)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("Tag frequency")
    axes[-1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Game 2: Policy-Basket Mechanisms Under Alignment vs Conflict", fontsize=15, y=1.03)
    path = savefig(fig, "07_game2_policy_basket_mechanisms.png")
    return path, table


def plot_game3_scatter_recovery(df: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    tags = ["proposal_scatter", "accepted_with_scatter", "failed_with_scatter", "near_threshold_rescue", "outcome_no_consensus_r10"]
    sub = df[df["game_label"].eq("game3")].copy()
    rows = []
    for (family, n), g in sub.groupby(["experiment_family", "n_agents"]):
        for tag in tags:
            rows.append(
                {
                    "experiment_family": family,
                    "n_agents": int(n),
                    "tag": tag,
                    "rate": float(g[f"tag_{tag}"].mean()),
                    "ci95": ci95(g[f"tag_{tag}"]),
                    "positive_nonfunded_mean": float(g["positive_nonfunded_count"].mean()),
                    "funded_count_mean": float(g["funded_count"].mean()),
                    "n": len(g),
                }
            )
    table = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.1))
    for tag in tags:
        t = table[table["tag"].eq(tag)].groupby("n_agents", as_index=False).agg(rate=("rate", "mean"))
        axes[0].plot(t["n_agents"], t["rate"], marker="o", linewidth=1.8, label=clean_name(tag))
    axes[0].set_title("Mechanism rates by group size", fontsize=12)
    axes[0].set_xlabel("Number of agents")
    axes[0].set_ylabel("Share of Game 3 rollouts")
    axes[0].set_xticks(N_ORDER)
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False, fontsize=7)
    for family in FAMILY_ORDER:
        g = sub[sub["experiment_family"].eq(family)]
        axes[1].scatter(
            g["positive_nonfunded_count"] + np.random.default_rng(7).normal(0, 0.03, len(g)),
            g["funded_count"] + np.random.default_rng(11).normal(0, 0.03, len(g)),
            s=20,
            alpha=0.35,
            color=FAMILY_COLORS[family],
            label=FAMILY_LABELS[family],
        )
    axes[1].set_title("Final proposal scatter vs funded projects", fontsize=12)
    axes[1].set_xlabel("Positive but nonfunded project totals")
    axes[1].set_ylabel("Funded project count")
    axes[1].grid(alpha=0.22)
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    fig.suptitle("Game 3: Public-Goods Scatter, Rescue, And Failure", fontsize=15, y=1.03)
    path = savefig(fig, "08_game3_scatter_recovery.png")
    return path, table


def plot_outcome_associations(df: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    tags = [
        "side_payment_smoothing",
        "hard_anchor_or_redline",
        "sequenced_or_contingent_deal",
        "near_threshold_rescue",
        "zero_value_holdout",
        "proposal_scatter",
        "template_role_artifact",
        "free_riding",
        "shared_top_item_deadlock",
        "staged_verification_governance",
    ]
    rows = []
    for tag in tags:
        col = f"tag_{tag}"
        sub = df[[col, "delayed_or_failed", "game_label", "experiment_family", "n_agents"]].dropna().copy()
        if sub[col].nunique() < 2:
            continue
        unadj = sub.groupby(col)["delayed_or_failed"].mean()
        row = {
            "tag": tag,
            "tag_label": clean_name(tag),
            "n": len(sub),
            "unadjusted_delay_if_absent": float(unadj.get(0, math.nan)),
            "unadjusted_delay_if_present": float(unadj.get(1, math.nan)),
            "unadjusted_diff": float(unadj.get(1, math.nan) - unadj.get(0, math.nan)),
        }
        if smf is not None:
            try:
                model = smf.ols(
                    f"delayed_or_failed ~ {col} + C(game_label) + C(experiment_family) + C(n_agents)",
                    data=sub,
                ).fit(cov_type="HC3")
                row.update(
                    {
                        "adjusted_coef": float(model.params[col]),
                        "adjusted_ci_low": float(model.conf_int().loc[col, 0]),
                        "adjusted_ci_high": float(model.conf_int().loc[col, 1]),
                        "adjusted_p": float(model.pvalues[col]),
                    }
                )
            except Exception:
                row.update({"adjusted_coef": math.nan, "adjusted_ci_low": math.nan, "adjusted_ci_high": math.nan, "adjusted_p": math.nan})
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("adjusted_coef")
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    y = np.arange(len(table))
    ax.axvline(0, color="#333333", lw=1)
    ax.errorbar(
        table["adjusted_coef"],
        y,
        xerr=[
            table["adjusted_coef"] - table["adjusted_ci_low"],
            table["adjusted_ci_high"] - table["adjusted_coef"],
        ],
        fmt="o",
        color="#3B82A0",
        ecolor="#93A9B8",
        capsize=2,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(table["tag_label"])
    ax.set_xlabel("Adjusted change in Pr(delayed or failed), percentage points")
    ax.xaxis.set_major_formatter(lambda x, pos: f"{x*100:.0f}")
    ax.set_title("Associations Between Dynamics And Delayed/Failed Agreement", fontsize=14)
    ax.grid(axis="x", alpha=0.22)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    path = savefig(fig, "09_tag_associations_with_delay.png")
    return path, table


def plot_tag_cooccurrence(df: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    tags = [
        "single_item_mutual_support_pact",
        "shared_top_item_deadlock",
        "low_value_trading_pool",
        "policy_basket_logrolling",
        "redline_then_package",
        "staged_verification_governance",
        "midpoint_closure",
        "feasibility_first_budget_math",
        "numeric_pledge_split",
        "single_project_rally",
        "near_threshold_rescue",
        "zero_value_holdout",
        "proposal_scatter",
        "side_payment_smoothing",
        "hard_anchor_or_redline",
        "sequenced_or_contingent_deal",
        "minimum_winning_supermajority",
        "template_role_artifact",
        "high_inequality_outcome",
    ]
    mat = df[[f"tag_{tag}" for tag in tags]].to_numpy(dtype=int)
    jaccard = np.zeros((len(tags), len(tags)))
    for i in range(len(tags)):
        for j in range(len(tags)):
            inter = np.logical_and(mat[:, i], mat[:, j]).sum()
            union = np.logical_or(mat[:, i], mat[:, j]).sum()
            jaccard[i, j] = inter / union if union else 0.0
    dist = 1.0 - jaccard
    np.fill_diagonal(dist, 0.0)
    order = leaves_list(linkage(squareform(dist), method="average"))
    labels = [clean_name(tags[i]) for i in order]
    ordered = jaccard[np.ix_(order, order)]
    table = pd.DataFrame(ordered, index=labels, columns=labels)
    fig, ax = plt.subplots(figsize=(10.5, 9.2))
    sns.heatmap(table, ax=ax, cmap="rocket_r", vmin=0, vmax=1, linewidths=0.2, cbar_kws={"label": "Jaccard co-occurrence"})
    ax.set_title("Tag Co-Occurrence Reveals Negotiation Archetypes", fontsize=15, pad=12)
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    path = savefig(fig, "10_tag_cooccurrence_heatmap.png")
    long = table.stack().reset_index()
    long.columns = ["tag_a", "tag_b", "jaccard"]
    return path, long


def write_markdown(df: pd.DataFrame, figures: dict[str, Path], tables: dict[str, pd.DataFrame]) -> None:
    report = OUT_DIR / "qualitative_dynamics_trend_report.md"
    rel = lambda p: p.relative_to(OUT_DIR)
    total = len(df)
    no_consensus = int((~df["consensus_reached"]).sum())
    round1 = int((df["outcome_bucket"] == "Round 1").sum())
    top_delay = tables["associations"].sort_values("adjusted_coef", ascending=False).head(3)
    hom_adv = df[df["experiment_family"].eq("homogeneous_adversary")]
    game3 = df[df["game_label"].eq("game3")]
    lines = [
        "# Qualitative Dynamics Trend Report",
        "",
        "This report uses the refined qualitative tags from `analysis/qualitative_rollout_dynamics_20260628/refined_rollout_dynamics_coding.csv` joined to the per-agent run tables in `experiments/results/n2_plus_multiagent_comparison_analysis_20260505/tables_multiagent/`. Each point is a rollout; tag frequencies are descriptive unless noted.",
        "",
        "## Corpus",
        "",
        f"- Rollouts analyzed: **{total}**.",
        f"- Round-1 consensus: **{round1} / {total}** ({round1 / total:.1%}).",
        f"- No consensus: **{no_consensus} / {total}** ({no_consensus / total:.1%}).",
        "- Families: heterogeneous random (1,300), homogeneous adversary (1,300), homogeneous control (130).",
        "- Key caveat: many tags are post-discussion descriptors. The adjusted association plot should be read as mechanism/outcome association, not causal identification.",
        "- Reviewer-facing caveat: near-universal tags such as priority disclosure, package construction, coalition language, fairness language, and efficiency language are background protocol features. The main figures therefore emphasize moderate-prevalence, game-specific, or structural tags with real variation.",
        "",
        "## Most Compelling Takeaways",
        "",
        "1. **The three games have distinct qualitative signatures.** Game 1 is top-item matching and side-payment smoothing; Game 2 is policy-basket logrolling; Game 3 is threshold arithmetic plus proposal scatter.",
        "2. **Fast consensus is common but not equivalent to broad agreement.** More than half of final accepted votes are minimum-winning, and group size makes coalition politics more visible.",
        "3. **Model strength enters through style, not just success.** Inserted-model Elo changes the prevalence of redlines, template artifacts, inequality, and round-1 closure in homogeneous-adversary runs; heterogeneous Elo dispersion shifts repair and coalition patterns.",
        "4. **Public-goods failures are often serialization failures.** Game 3 no-consensus cases frequently contain verbal convergence, but formal contribution vectors scatter across nonfunded projects.",
        "5. **Preference conflict can be made concrete.** In Game 1, top-item collision derived from raw preferences tracks lower top-item satisfaction, lower first-round consensus, and higher inequality. The raw collision measure is more reliable than any single broad transcript keyword tag.",
        "",
        "## Figure 1: Dynamics Fingerprint",
        "",
        f"![Dynamics fingerprint]({rel(figures['fingerprint'])})",
        "",
        "The heatmap shows that the qualitative tags are not merely generic LLM verbosity. Game-specific mechanisms light up where they should: Game 2 has policy-basket logrolling and midpoint closure, Game 3 has numeric pledge splits and scatter, and Game 1 has low-value trading pools and top-item pacts.",
        "",
        "## Figure 2: Consensus Timing",
        "",
        f"![Consensus timing]({rel(figures['timing'])})",
        "",
        "Game 2 is consistently easy to close, while Game 3 is the main source of no-consensus outcomes. Homogeneous control has the smallest sample, but it is visibly brittle in Game 3.",
        "",
        "## Figure 3: Inserted-Model Elo And Homogeneous-Adversary Dynamics",
        "",
        f"![Homogeneous adversary Elo]({rel(figures['hom_adv_elo'])})",
        "",
        "This plot is descriptive, but it is paper-useful because it asks whether stronger inserted models alter the bargaining style. The baseline GPT-5-nano reference is shown as the dashed vertical line.",
        "",
        "## Figure 4: Heterogeneous Roster Elo Dispersion",
        "",
        f"![Heterogeneous Elo dispersion]({rel(figures['hetero_elo'])})",
        "",
        "The heterogeneous condition lets us ask a different question: does a wider spread of model capabilities change the negotiation ecology? The figure bins runs by roster Elo standard deviation and plots mechanism rates by game.",
        "",
        "## Figure 5: Coalition Scaling With N",
        "",
        f"![Coalition scaling]({rel(figures['coalition'])})",
        "",
        "As N grows, the relevant object is often not unanimity but a passable two-thirds coalition. This is why minimum-winning outcomes and vote margins are more informative than consensus alone.",
        "",
        "## Figure 6: Game 1 Top-Item Collision",
        "",
        f"![Game 1 top collision]({rel(figures['game1_collision'])})",
        "",
        "This is one of the cleanest mechanism bridges: raw preferences say how many agents collide on top items. As more agents' top items are contested, top-item satisfaction falls and payoff inequality rises. The transcript deadlock tag is broad, so the raw collision and realized allocation measures should carry the evidentiary weight here.",
        "",
        "## Figure 7: Game 2 Policy-Basket Mechanisms",
        "",
        f"![Game 2 mechanisms]({rel(figures['game2'])})",
        "",
        "Game 2 is the best case for integrative bargaining. Redlines often become packages; verification and midpoint closure are the mechanisms that turn disagreement into treaty-like settlement.",
        "",
        "## Figure 8: Game 3 Scatter And Recovery",
        "",
        f"![Game 3 scatter]({rel(figures['game3'])})",
        "",
        f"Game 3 has **{int(game3['tag_proposal_scatter'].sum())}** scatter-tagged rollouts. Scatter is not always fatal: **{int(game3['tag_accepted_with_scatter'].sum())}** accepted rollouts still contain scattered nonfunded contributions, while **{int(game3['tag_failed_with_scatter'].sum())}** failures do.",
        "",
        "## Figure 9: Dynamics Associated With Delayed Or Failed Agreement",
        "",
        f"![Outcome associations]({rel(figures['associations'])})",
        "",
        "This is an adjusted linear-probability screen with game, family, and N controls. The strongest positive associations with delayed/failed agreement are:",
    ]
    for _, row in top_delay.iterrows():
        lines.append(f"- `{row['tag']}`: {row['adjusted_coef'] * 100:+.1f} percentage points.")
    lines.extend(
        [
            "",
            "## Figure 10: Tag Co-Occurrence Archetypes",
            "",
            f"![Co-occurrence heatmap]({rel(figures['cooccurrence'])})",
            "",
            "The co-occurrence heatmap suggests higher-level archetypes: item-matching/side-payment bargaining, policy-basket treaty formation, public-good threshold repair, and procedural/reliability failures.",
            "",
        "## Recommended Paper Figures",
        "",
        "For a main paper, the strongest set is Figures 1, 2, 5, 6, 8, and 9. Figures 3, 4, 7, and 10 are good appendix or robustness figures unless the paper section specifically emphasizes model strength and bargaining style.",
        "",
        "## Promising Next Analyses",
        "",
        "- **Rhetoric-outcome calibration:** recode first-round-only fairness and efficiency rhetoric, then test whether early fairness predicts lower payoff Gini and whether early efficiency predicts higher welfare or funded-project count. This should be done with early-round tags to avoid post-treatment leakage.",
        "- **Game 3 vector stability:** compute round-to-round L1 movement in aggregate contribution vectors and test whether unstable vectors distinguish semantic-drift failures from productive near-threshold repair.",
        "- **Manual validation sample:** audit positives and negatives for `template_role_artifact`, `semantic_vector_or_ballot_drift`, `proposal_scatter`, `zero_value_holdout`, `free_riding`, and `baseline_mirroring_or_deference` before presenting those as strong claims.",
        "- **Cell fixed effects:** for family comparisons, estimate tag prevalence with game x N x setting fixed effects. The homogeneous-control sample is small, so cell-level uncertainty should be displayed.",
        "- **Adversary identity fingerprints:** add a heatmap of adversary model x tags, especially `hard_anchor_or_redline`, `template_role_artifact`, `baseline_mirroring_or_deference`, no-consensus, and high inequality.",
        "- **Transcript effort and vote recovery:** parse discussion/proposal counts and failed vote margins by round to distinguish hard bargaining from formalization failure.",
        "- **Game 3 project-level missed surplus:** classify each project as funded/unfunded and positive-/negative-surplus, then test whether scatter and zero-value holdouts leave high-surplus projects unfunded.",
        "- **Contribution concentration:** compute HHI or top-contributor share for funded public goods to distinguish broad cost-sharing from one-agent carry outcomes.",
        "",
        "## Candidate Case Studies",
        "",
        "These are good transcript anchors for a qualitative appendix or figure strip:",
        "",
        "- **Semantic vector drift:** [config_1934 Game 3 homogeneous control](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_production_20260428_085255/runs/config_1934_game3_homogeneous_control_n2_sigma_0p2_alpha_0p8_seed2/experiment_results.json). Agents repeatedly anchor on Cedar/Parkside, but the final vectors fund nothing; no consensus in round 10.",
        "- **Scatter that still recovers:** [config_2065 Game 3 homogeneous adversary](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_production_20260428_085255/runs/config_2065_game3_homogeneous_adversary_n4_sigma_0p2_alpha_0p2_gpt_4o_mini_2024_07_18_first_seed1/experiment_results.json). Proposal scatter is present, but the group reaches consensus after repair.",
        "- **Redline-then-package repair:** [config_1109 Game 2 homogeneous adversary](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_production_20260428_085255/runs/config_1109_game2_homogeneous_adversary_n2_rho_1p0_theta_0p8_gemini_2p5_pro_last_seed1/experiment_results.json). Redline language becomes a package rather than a terminal refusal.",
        "- **Top-item collision:** [config_0612 Game 1 heterogeneous](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848/runs/config_0612_game1_heterogeneous_random_n6_comp_1p0_run02/experiment_results.json). All agents collide on the same top item; consensus arrives only in round 7 and payoff inequality is high.",
        "- **Clean mutual-support contrast:** [config_0863 Game 1 heterogeneous](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848/runs/config_0863_game1_heterogeneous_random_n10_comp_0p0_run01/experiment_results.json). Disjoint top items produce round-1 consensus, top-item satisfaction of 1.0, and zero payoff Gini.",
        "- **Adversary free-riding/veto:** [config_2281 Game 3 homogeneous adversary](/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/full_games123_multiagent_production_20260428_085255/runs/config_2281_game3_homogeneous_adversary_n6_sigma_0p2_alpha_0p8_claude_sonnet_4_20250514_last_seed1/experiment_results.json). A last-position adversary benefits from a project while contributing zero; the run fails by round 10.",
            "",
            "## Outputs",
            "",
            "- `qualitative_dynamics_merged_run_table.csv`: merged run-level analysis frame.",
            "- `tables/*.csv`: source summary tables for each figure.",
            "- `figures/*.png`: all plots embedded above.",
        ]
    )
    report.write_text("\n".join(lines))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    table_dir = OUT_DIR / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    df = build_merged_frame()
    df.to_csv(OUT_DIR / "qualitative_dynamics_merged_run_table.csv", index=False)

    figures: dict[str, Path] = {}
    tables: dict[str, pd.DataFrame] = {}
    figures["fingerprint"], tables["fingerprint"] = plot_dynamics_fingerprint(df)
    figures["timing"], tables["timing"] = plot_consensus_timing(df)
    figures["hom_adv_elo"], tables["hom_adv_elo"] = plot_hom_adv_elo(df)
    figures["hetero_elo"], tables["hetero_elo"] = plot_heterogeneous_elo_dispersion(df)
    figures["coalition"], tables["coalition"] = plot_coalition_scaling(df)
    figures["game1_collision"], tables["game1_collision"] = plot_game1_top_collision(df)
    figures["game2"], tables["game2"] = plot_game2_mechanisms(df)
    figures["game3"], tables["game3"] = plot_game3_scatter_recovery(df)
    figures["associations"], tables["associations"] = plot_outcome_associations(df)
    figures["cooccurrence"], tables["cooccurrence"] = plot_tag_cooccurrence(df)

    for name, table in tables.items():
        table.to_csv(table_dir / f"{name}.csv", index=False)
    write_markdown(df, figures, tables)

    print(f"wrote {OUT_DIR}")
    print(f"rollouts={len(df)} figures={len(figures)}")


if __name__ == "__main__":
    main()

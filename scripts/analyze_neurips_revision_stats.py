#!/usr/bin/env python3
"""Generate NeurIPS revision statistics and paper-facing figures.

This script intentionally uses existing completed result files. It does not
launch new experiments. Outputs are written under analysis/neurips_revision_*
and selected plots are copied into overleaf/neurips/graphics/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plot_exploitation_vs_elo import (  # noqa: E402
    GAME1_RESULTS_ROOT as EI_GAME1_ROOT,
    GAME2_RESULTS_ROOT as EI_GAME2_ROOT,
    GAME3_RESULTS_ROOT as EI_GAME3_ROOT,
    load_game1_records as load_ei_game1_records,
    load_game2_records as load_ei_game2_records,
    load_game3_records as load_ei_game3_records,
)
from scripts.plot_gpt5_nano_baseline_vs_elo_all_games import (  # noqa: E402
    DEFAULT_ELO_MARKDOWN,
    DEFAULT_GAME1_ROOT,
    DEFAULT_GAME2_ROOT,
    DEFAULT_GAME3_ROOT,
    load_game1_rows,
    load_game2_rows,
    load_game3_rows,
    parse_elo_markdown,
)


DEFAULT_MULTIAGENT_ASSET_DIR = (
    PROJECT_ROOT / "experiments/results/partial_multiagent_results_plot_report_20260503_assets"
)
DEFAULT_TTC_ROOT = PROJECT_ROOT / "experiments/results/ttc_native_scaling_20260502_212943"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis/neurips_revision_20260504"
OVERLEAF_GRAPHICS = PROJECT_ROOT / "overleaf/neurips/graphics"
GAMMA_DISCOUNT = 0.9
BOOTSTRAP_REPS = 2000
RNG_SEED = 20260504

GAME_NAMES = {
    "game1": "Item allocation",
    "game2": "Diplomatic treaty",
    "game3": "Co-funding",
}
FAMILY_NAMES = {
    "homogeneous_control": "control",
    "homogeneous_adversary": "focal",
    "heterogeneous_random": "heterogeneous",
}
STAGE_ORDER = ["setup", "discussion", "private_thinking", "proposal", "voting", "reflection", "other"]


@dataclass(frozen=True)
class OLSResult:
    n: int
    r2: float
    coefficients: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--multiagent-asset-dir", type=Path, default=DEFAULT_MULTIAGENT_ASSET_DIR)
    parser.add_argument("--ttc-root", type=Path, default=DEFAULT_TTC_ROOT)
    parser.add_argument("--elo-markdown", type=Path, default=DEFAULT_ELO_MARKDOWN)
    parser.add_argument("--copy-to-overleaf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    return parser.parse_args()


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def finite_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    clean = frame.dropna(subset=list(columns)).copy()
    for col in columns:
        clean = clean[np.isfinite(clean[col].astype(float))]
    return clean


def ols_fit(frame: pd.DataFrame, y_col: str, x_cols: Sequence[str]) -> OLSResult:
    clean = finite_frame(frame, [y_col, *x_cols])
    n = len(clean)
    if n == 0:
        return OLSResult(n=0, r2=math.nan, coefficients=pd.DataFrame())

    y = clean[y_col].astype(float).to_numpy()
    x = clean[list(x_cols)].astype(float).to_numpy()
    x = np.column_stack([np.ones(n), x])
    names = ["intercept", *x_cols]
    rank = np.linalg.matrix_rank(x)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    fitted = x @ beta
    residuals = y - fitted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
    dof = n - rank
    if dof > 0:
        sigma2 = ss_res / dof
        xtx_inv = np.linalg.pinv(x.T @ x)
        se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0.0))
        t_values = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
        p_values = 2.0 * stats.t.sf(np.abs(t_values), df=dof)
        crit = stats.t.ppf(0.975, df=dof)
        ci_low = beta - crit * se
        ci_high = beta + crit * se
    else:
        se = np.full_like(beta, np.nan)
        t_values = np.full_like(beta, np.nan)
        p_values = np.full_like(beta, np.nan)
        ci_low = np.full_like(beta, np.nan)
        ci_high = np.full_like(beta, np.nan)
    coef = pd.DataFrame(
        {
            "term": names,
            "estimate": beta,
            "std_error": se,
            "t": t_values,
            "p_value": p_values,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
    )
    return OLSResult(n=n, r2=r2, coefficients=coef)


def add_dummies(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        dummies = pd.get_dummies(result[column].astype(str), prefix=column, drop_first=True)
        result = pd.concat([result, dummies.astype(float)], axis=1)
    return result


def bootstrap_mean_ci(
    values: Iterable[float],
    reps: int,
    rng: np.random.Generator,
    statistic: str = "mean",
) -> tuple[float, float, float]:
    arr = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=float)
    if arr.size == 0:
        return math.nan, math.nan, math.nan
    point = float(np.mean(arr)) if statistic == "mean" else float(np.median(arr))
    if arr.size == 1:
        return point, point, point
    indices = rng.integers(0, arr.size, size=(reps, arr.size))
    samples = arr[indices]
    if statistic == "mean":
        boot = samples.mean(axis=1)
    else:
        boot = np.median(samples, axis=1)
    return point, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    pairs = sorted((float(v), float(w)) for v, w in zip(values, weights))
    total = sum(w for _, w in pairs)
    if total <= 0:
        return float(pairs[len(pairs) // 2][0])
    running = 0.0
    for value, weight in pairs:
        running += weight
        if running >= 0.5 * total:
            return value
    return float(pairs[-1][0])


def game1_sw_star(preferences: dict[str, list[float]]) -> float:
    agents = sorted(preferences)
    if not agents:
        return math.nan
    m_items = len(preferences[agents[0]])
    return float(sum(max(float(preferences[a][j]) for a in agents) for j in range(m_items)))


def game2_sw_star(config: dict[str, Any], preferences: dict[str, list[float]]) -> float:
    positions = config.get("agent_positions") or preferences
    weights = config.get("agent_weights") or {}
    agents = sorted(positions)
    if not agents:
        return math.nan
    n_issues = len(positions[agents[0]])
    if not weights:
        weights = {agent: [1.0 / n_issues] * n_issues for agent in agents}

    total = 0.0
    for issue_idx in range(n_issues):
        issue_positions = [float(positions[a][issue_idx]) for a in agents]
        issue_weights = [float(weights[a][issue_idx]) for a in agents]
        agreement = weighted_median(issue_positions, issue_weights)
        issue_welfare = sum(
            issue_weights[i] * (1.0 - abs(issue_positions[i] - agreement))
            for i in range(len(agents))
        )
        total += 100.0 * issue_welfare
    return float(total)


def game3_sw_star(config: dict[str, Any], preferences: dict[str, list[float]]) -> float:
    items = config.get("items") or []
    if not items or not preferences:
        return math.nan
    costs = [float(item.get("cost", 0.0)) for item in items]
    budget_payload = config.get("agent_budgets") or {}
    total_budget = float(config.get("total_budget") or sum(float(v) for v in budget_payload.values()))
    if total_budget <= 0:
        return 0.0

    scale = 100
    int_budget = int(round(total_budget * scale))
    int_costs = [max(0, int(round(cost * scale))) for cost in costs]
    project_values = []
    agents = sorted(preferences)
    for j, cost in enumerate(costs):
        value_sum = sum(float(preferences[agent][j]) for agent in agents)
        project_values.append(value_sum - cost)

    dp = np.zeros(int_budget + 1, dtype=float)
    for cost, value in zip(int_costs, project_values):
        if cost > int_budget or value <= 0:
            continue
        candidate = dp[: int_budget + 1 - cost] + value
        dp[cost:] = np.maximum(dp[cost:], candidate)
    return float(np.max(dp))


def compute_sw_star(payload: dict[str, Any]) -> float:
    config = payload.get("config") or {}
    preferences = payload.get("agent_preferences") or {}
    game = config.get("game_label") or config.get("game_type")
    if game == "game1" or config.get("game_type") == "item_allocation":
        return game1_sw_star(preferences)
    if game == "game2" or config.get("game_type") == "diplomacy":
        return game2_sw_star(config, preferences)
    if game == "game3" or config.get("game_type") == "co_funding":
        return game3_sw_star(config, preferences)
    return math.nan


def extract_game1_preferences_from_interactions(result_path: Path) -> dict[str, list[float]]:
    """Recover Game 1 preferences from setup prompts when no-agreement payloads omit them."""
    for name in ["run_1_all_interactions.json", "all_interactions.json"]:
        path = result_path.with_name(name)
        if not path.exists():
            continue
        try:
            interactions = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        recovered: dict[str, dict[int, float]] = defaultdict(dict)
        for entry in interactions:
            if str(entry.get("phase")) != "game_setup":
                continue
            agent_id = str(entry.get("agent_id") or "")
            prompt = str(entry.get("prompt") or "")
            for match in re.finditer(r"^\s*(\d+):\s*.+?->\s*(-?\d+(?:\.\d+)?)\s*$", prompt, flags=re.MULTILINE):
                recovered[agent_id][int(match.group(1))] = float(match.group(2))
        if recovered:
            prefs: dict[str, list[float]] = {}
            for agent_id, values in recovered.items():
                if values:
                    max_idx = max(values)
                    prefs[agent_id] = [float(values.get(i, 0.0)) for i in range(max_idx + 1)]
            if prefs:
                return prefs
    return {}


def load_multiagent_metrics(asset_dir: Path, output_dir: Path, bootstrap_reps: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = pd.read_csv(asset_dir / "completed_runs.csv")
    agents = pd.read_csv(asset_dir / "agent_observations.csv")
    rng = np.random.default_rng(RNG_SEED)

    sw_rows: list[dict[str, Any]] = []
    for _, row in runs.iterrows():
        path = Path(row["result_path"])
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - exported for audit
            sw_rows.append(
                {
                    "source": row["source"],
                    "config_id": row["config_id"],
                    "sw_star": math.nan,
                    "sw_star_error": str(exc),
                }
            )
            continue
        sw_star = compute_sw_star(payload)
        config = payload.get("config") or {}
        if (not math.isfinite(sw_star)) and (
            config.get("game_label") == "game1" or config.get("game_type") == "item_allocation"
        ):
            recovered_preferences = extract_game1_preferences_from_interactions(path)
            if recovered_preferences:
                sw_star = game1_sw_star(recovered_preferences)
        sw_rows.append(
            {
                "source": row["source"],
                "config_id": row["config_id"],
                "sw_star": sw_star,
                "sw_star_error": "",
            }
        )

    sw_df = pd.DataFrame(sw_rows)
    runs = runs.merge(sw_df, on=["source", "config_id"], how="left")
    runs["group_efficiency_discounted"] = np.where(
        runs["sw_star"].astype(float) > 0,
        runs["sum_utility"].astype(float) / runs["sw_star"].astype(float),
        np.nan,
    )
    runs["mean_normalized_utility"] = np.where(
        runs["sw_star"].astype(float) > 0,
        runs["n_agents"].astype(float) * runs["mean_utility"].astype(float) / runs["sw_star"].astype(float),
        np.nan,
    )

    for column in ["sw_star", "group_efficiency_discounted", "mean_normalized_utility"]:
        if column in agents.columns:
            agents = agents.drop(columns=[column])
    agents = agents.merge(
        runs[
            [
                "source",
                "config_id",
                "sw_star",
                "group_efficiency_discounted",
                "mean_normalized_utility",
            ]
        ],
        on=["source", "config_id"],
        how="left",
    )
    agents["normalized_utility"] = np.where(
        agents["sw_star"].astype(float) > 0,
        agents["n_agents"].astype(float) * agents["final_utility"].astype(float) / agents["sw_star"].astype(float),
        np.nan,
    )
    agents["utility_share_factor"] = np.where(
        agents["sum_utility"].astype(float).abs() > 1e-12,
        agents["n_agents"].astype(float) * agents["final_utility"].astype(float) / agents["sum_utility"].astype(float),
        np.nan,
    )
    agents["utility_times_n_over_2"] = agents["final_utility"].astype(float) * agents["n_agents"].astype(float) / 2.0
    discount_factor = np.power(GAMMA_DISCOUNT, agents["final_round"].astype(float) - 1.0)
    agents["undiscounted_utility"] = np.where(
        agents["consensus_reached"].astype(bool) & (discount_factor > 0),
        agents["final_utility"].astype(float) / discount_factor,
        np.nan,
    )
    undisc_runs = (
        agents.groupby(["source", "config_id"], dropna=False)
        .agg(
            undiscounted_sum_utility=("undiscounted_utility", lambda x: x.sum(min_count=1)),
            undiscounted_mean_utility=("undiscounted_utility", "mean"),
        )
        .reset_index()
    )
    runs = runs.merge(undisc_runs, on=["source", "config_id"], how="left")

    write_csv(output_dir / "multiagent_runs_with_sw_star.csv", runs)
    write_csv(output_dir / "multiagent_agents_normalized.csv", agents)

    normalized_summary = (
        agents.groupby(["game_label", "experiment_family", "n_agents"], dropna=False)
        .agg(
            agent_count=("agent_id", "size"),
            run_count=("config_id", "nunique"),
            raw_utility_mean=("final_utility", "mean"),
            raw_utility_std=("final_utility", "std"),
            normalized_utility_mean=("normalized_utility", "mean"),
            normalized_utility_std=("normalized_utility", "std"),
            undiscounted_utility_mean=("undiscounted_utility", "mean"),
            group_efficiency_mean=("group_efficiency_discounted", "mean"),
            consensus_rate=("consensus_reached", "mean"),
            final_round_mean=("final_round", "mean"),
        )
        .reset_index()
        .sort_values(["game_label", "experiment_family", "n_agents"])
    )
    write_csv(output_dir / "multiagent_normalized_by_game_family_n.csv", normalized_summary)

    control_boot_rows: list[dict[str, Any]] = []
    controls = agents[
        (agents["experiment_family"] == "homogeneous_control")
        & (agents["n_agents"] > 2)
    ].copy()
    for keys, group in controls.groupby(["game_label", "n_agents", "competition_label"], dropna=False):
        game, n_agents, competition = keys
        raw_mean, raw_lo, raw_hi = bootstrap_mean_ci(group["final_utility"], bootstrap_reps, rng)
        norm_mean, norm_lo, norm_hi = bootstrap_mean_ci(group["normalized_utility"], bootstrap_reps, rng)
        control_boot_rows.append(
            {
                "game_label": game,
                "n_agents": n_agents,
                "competition_label": competition,
                "agent_count": len(group),
                "run_count": group["config_id"].nunique(),
                "raw_utility_mean": raw_mean,
                "raw_utility_ci_low": raw_lo,
                "raw_utility_ci_high": raw_hi,
                "normalized_utility_mean": norm_mean,
                "normalized_utility_ci_low": norm_lo,
                "normalized_utility_ci_high": norm_hi,
            }
        )
    control_boot = pd.DataFrame(control_boot_rows).sort_values(["game_label", "n_agents", "competition_label"])
    write_csv(output_dir / "homogeneous_control_bootstrap_by_cell.csv", control_boot)

    return runs, agents


def analyze_exploitation(output_dir: Path) -> pd.DataFrame:
    frames = [
        ("game1", load_ei_game1_records(EI_GAME1_ROOT), "NBS"),
        ("game2", load_ei_game2_records(EI_GAME2_ROOT), "NBS"),
        ("game3", load_ei_game3_records(EI_GAME3_ROOT), "Lindahl"),
    ]
    rows: list[dict[str, Any]] = []
    raw_frames = []
    for game, df, benchmark in frames:
        if df.empty:
            continue
        tmp = df.copy()
        tmp["game_label"] = game
        raw_frames.append(tmp)
        for value_col, role in [("ei_adversary", "adversary"), ("ei_baseline", "baseline")]:
            valid = df[df["consensus"] & df[value_col].notna()].copy()
            if valid.empty:
                continue
            agg = (
                valid.groupby("adversary", dropna=False)
                .agg(elo=("elo", "first"), mean_ei=(value_col, "mean"), run_count=(value_col, "size"))
                .reset_index()
            )
            fit_df = agg.rename(columns={"mean_ei": "y"}).copy()
            fit_df["elo_per_100"] = fit_df["elo"].astype(float) / 100.0
            fit = ols_fit(fit_df, "y", ["elo_per_100"])
            coef = fit.coefficients[fit.coefficients["term"] == "elo_per_100"].iloc[0]
            rows.append(
                {
                    "game_label": game,
                    "benchmark": benchmark,
                    "role": role,
                    "model_points": len(agg),
                    "run_count": int(valid[value_col].notna().sum()),
                    "slope_per_100_elo": coef["estimate"],
                    "ci_low": coef["ci_low"],
                    "ci_high": coef["ci_high"],
                    "p_value": coef["p_value"],
                    "r": float(np.corrcoef(agg["elo"].astype(float), agg["mean_ei"].astype(float))[0, 1])
                    if len(agg) > 1
                    else math.nan,
                    "r2": fit.r2,
                }
            )
    if raw_frames:
        write_csv(output_dir / "exploitation_raw_records.csv", pd.concat(raw_frames, ignore_index=True))
    result = pd.DataFrame(rows)
    write_csv(output_dir / "exploitation_slope_by_game.csv", result)
    return result


def analyze_bilateral_utility(output_dir: Path, elo_markdown: Path) -> pd.DataFrame:
    elo_by_model = parse_elo_markdown(elo_markdown)
    game2_long_csv = (
        PROJECT_ROOT / "visualization/figures/diplomacy_20260405_082215_summary/utility_vs_elo_adversary_long.csv"
    )
    if game2_long_csv.exists():
        game2_long = pd.read_csv(game2_long_csv)
        game2_df = game2_long[game2_long["role"].eq("adversary")].copy()
        game2_df = game2_df.rename(
            columns={
                "model": "adversary_model",
                "model_short": "adversary_short",
                "elo": "adversary_elo",
                "utility": "adversary_utility",
                "competition_index": "competition_value",
            }
        )
        game2_df["game_id"] = "game2"
        game2_df["game_label"] = "Game 2"
        game2_df["baseline_utility"] = np.nan
        game2_df["competition_col"] = "competition_index"
        game2_df["competition_display"] = "competition index"
        game2_df["competition_curve_label"] = game2_df["competition_value"].map(
            lambda value: f"CI2={float(value):.2f}".rstrip("0").rstrip(".")
        )
    else:
        game2_df = load_game2_rows(DEFAULT_GAME2_ROOT, elo_by_model)
    frames = [
        load_game1_rows(DEFAULT_GAME1_ROOT, elo_by_model),
        game2_df,
        load_game3_rows(DEFAULT_GAME3_ROOT, elo_by_model),
    ]
    bilateral = pd.concat(frames, ignore_index=True)
    bilateral["elo_per_100"] = bilateral["adversary_elo"].astype(float) / 100.0
    bilateral["elo_x_comp"] = bilateral["elo_per_100"] * bilateral["competition_value"].astype(float)
    write_csv(output_dir / "bilateral_utility_raw.csv", bilateral)

    rows: list[dict[str, Any]] = []
    for game, group in bilateral.groupby("game_id"):
        fit = ols_fit(group, "adversary_utility", ["elo_per_100", "competition_value", "elo_x_comp"])
        for _, coef in fit.coefficients.iterrows():
            rows.append(
                {
                    "game_label": game,
                    "dependent": "adversary_utility",
                    "n": fit.n,
                    "r2": fit.r2,
                    **coef.to_dict(),
                }
            )
    result = pd.DataFrame(rows)
    write_csv(output_dir / "utility_regression_by_game.csv", result)
    return result


def analyze_hetero_n2(agents: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subset = agents[
        (agents["experiment_family"] == "heterogeneous_random")
        & (agents["n_agents"] == 2)
        & agents["elo"].notna()
    ].copy()
    subset["elo_per_100"] = subset["elo"].astype(float) / 100.0
    subset["competition_numeric"] = subset["competition_order"].astype(float)
    for game, group in subset.groupby("game_label"):
        fit = ols_fit(group, "final_utility", ["elo_per_100", "competition_numeric"])
        coef = fit.coefficients[fit.coefficients["term"] == "elo_per_100"].iloc[0]
        rows.append(
            {
                "game_label": game,
                "agent_count": fit.n,
                "run_count": group["config_id"].nunique(),
                "slope_per_100_elo": coef["estimate"],
                "ci_low": coef["ci_low"],
                "ci_high": coef["ci_high"],
                "p_value": coef["p_value"],
                "r2": fit.r2,
            }
        )
    result = pd.DataFrame(rows)
    write_csv(output_dir / "heterogeneous_n2_utility_vs_elo.csv", result)
    return result


def analyze_parser_clean(output_dir: Path) -> pd.DataFrame:
    candidates = [
        PROJECT_ROOT / "analysis/full_games123_all_success_preliminary_20260428/all_success_agents.csv",
        PROJECT_ROOT / "analysis/full_games123_heterogeneous_equal_width_openrouter_repair_20260429_gini_20260503_rerun/all_success_agents.csv",
    ]
    frames = [pd.read_csv(path) for path in candidates if path.exists()]
    if not frames:
        return pd.DataFrame()
    agents = pd.concat(frames, ignore_index=True)
    agents["parser_clean"] = (
        agents["strict_voting_clean"].astype(bool)
        & agents["token_limit_marker_count"].fillna(0).astype(int).eq(0)
        & agents["vote_fallback_log_marker_count"].fillna(0).astype(int).eq(0)
        & agents["synthetic_vote_marker_count"].fillna(0).astype(int).eq(0)
        & agents["synthetic_proposal_marker_count"].fillna(0).astype(int).eq(0)
        & agents["provider_degradation_marker_count"].fillna(0).astype(int).eq(0)
    )
    keep = (
        ((agents["experiment_family"] == "homogeneous_adversary") & (agents["role"] == "adversary"))
        | (agents["experiment_family"] == "heterogeneous_random")
    )
    agents = agents[keep & agents["elo"].notna()].copy()
    agents["elo_per_100"] = agents["elo"].astype(float) / 100.0
    agents["competition_numeric"] = agents["competition_value"].astype(float)

    rows: list[dict[str, Any]] = []
    for (game, family), group in agents.groupby(["game_label", "experiment_family"]):
        for label, subset in [("all_completed", group), ("parser_clean", group[group["parser_clean"]])]:
            if len(subset) < 5 or subset["elo"].nunique() < 2:
                continue
            fit = ols_fit(subset, "final_utility", ["elo_per_100", "competition_numeric"])
            coef = fit.coefficients[fit.coefficients["term"] == "elo_per_100"].iloc[0]
            rows.append(
                {
                    "game_label": game,
                    "experiment_family": family,
                    "subset": label,
                    "agent_count": fit.n,
                    "run_count": subset["config_id"].nunique(),
                    "slope_per_100_elo": coef["estimate"],
                    "ci_low": coef["ci_low"],
                    "ci_high": coef["ci_high"],
                    "p_value": coef["p_value"],
                    "r2": fit.r2,
                }
            )
    result = pd.DataFrame(rows)
    write_csv(output_dir / "parser_clean_robustness.csv", result)
    return result


def analyze_rounds_and_confounding(
    runs: pd.DataFrame,
    agents: pd.DataFrame,
    output_dir: Path,
    bootstrap_reps: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RNG_SEED + 1)
    rounds_summary = (
        runs.groupby(["game_label", "experiment_family", "n_agents"], dropna=False)
        .agg(
            run_count=("config_id", "size"),
            consensus_rate=("consensus_reached", "mean"),
            final_round_mean=("final_round", "mean"),
            final_round_std=("final_round", "std"),
            mean_utility=("mean_utility", "mean"),
            mean_normalized_utility=("mean_normalized_utility", "mean"),
        )
        .reset_index()
        .sort_values(["game_label", "experiment_family", "n_agents"])
    )
    write_csv(output_dir / "rounds_by_n_game_family.csv", rounds_summary)

    early_late_rows: list[dict[str, Any]] = []
    for (game, family), group in runs.groupby(["game_label", "experiment_family"], dropna=False):
        early = group[group["n_agents"].isin([2, 4])]["final_round"].astype(float)
        late = group[group["n_agents"].isin([6, 8, 10])]["final_round"].astype(float)
        if early.empty or late.empty:
            continue
        early_arr = early.to_numpy()
        late_arr = late.to_numpy()
        diffs = []
        for _ in range(bootstrap_reps):
            e = rng.choice(early_arr, size=early_arr.size, replace=True).mean()
            l = rng.choice(late_arr, size=late_arr.size, replace=True).mean()
            diffs.append(l - e)
        early_late_rows.append(
            {
                "game_label": game,
                "experiment_family": family,
                "early_run_count": int(early_arr.size),
                "late_run_count": int(late_arr.size),
                "early_mean_round": float(early_arr.mean()),
                "late_mean_round": float(late_arr.mean()),
                "late_minus_early_mean": float(late_arr.mean() - early_arr.mean()),
                "ci_low": float(np.quantile(diffs, 0.025)),
                "ci_high": float(np.quantile(diffs, 0.975)),
            }
        )
    early_late = pd.DataFrame(early_late_rows)
    write_csv(output_dir / "rounds_early_late_bootstrap.csv", early_late)

    reg_df = runs.copy()
    reg_df["competition_numeric"] = reg_df["competition_order"].astype(float)
    reg_df = add_dummies(reg_df, ["game_label", "experiment_family"])
    x_cols = ["n_agents", "competition_numeric"] + [
        c for c in reg_df.columns if c.startswith("game_label_") or c.startswith("experiment_family_")
    ]
    round_reg = ols_fit(reg_df, "final_round", x_cols)
    round_regression = round_reg.coefficients.copy()
    round_regression["dependent"] = "final_round"
    round_regression["n"] = round_reg.n
    round_regression["r2"] = round_reg.r2
    write_csv(output_dir / "rounds_regression.csv", round_regression)

    utility_rows: list[dict[str, Any]] = []
    conf = agents[agents["consensus_reached"].astype(bool)].copy()
    conf["competition_numeric"] = conf["competition_order"].astype(float)
    conf = add_dummies(conf, ["game_label", "experiment_family"])
    x_cols = ["final_round", "n_agents", "competition_numeric"] + [
        c for c in conf.columns if c.startswith("game_label_") or c.startswith("experiment_family_")
    ]
    for y_col in ["final_utility", "undiscounted_utility", "normalized_utility"]:
        fit = ols_fit(conf, y_col, x_cols)
        for _, coef in fit.coefficients.iterrows():
            utility_rows.append({"dependent": y_col, "n": fit.n, "r2": fit.r2, **coef.to_dict()})
    utility_reg = pd.DataFrame(utility_rows)
    write_csv(output_dir / "utility_round_confounding_regression.csv", utility_reg)

    elo_rows: list[dict[str, Any]] = []
    elo_df = conf[conf["elo"].notna()].copy()
    elo_df["elo_per_100"] = elo_df["elo"].astype(float) / 100.0
    x_cols = ["elo_per_100", "n_agents", "competition_numeric"] + [
        c for c in elo_df.columns if c.startswith("game_label_") or c.startswith("experiment_family_")
    ]
    for y_col in ["final_utility", "undiscounted_utility", "normalized_utility"]:
        fit = ols_fit(elo_df, y_col, x_cols)
        for _, coef in fit.coefficients.iterrows():
            elo_rows.append({"dependent": y_col, "n": fit.n, "r2": fit.r2, **coef.to_dict()})
    elo_reg = pd.DataFrame(elo_rows)
    write_csv(output_dir / "elo_discounted_undiscounted_regression.csv", elo_reg)

    return rounds_summary, early_late, utility_reg


def stage_from_phase(phase: str) -> str:
    phase = str(phase or "").lower()
    if phase in {"game_setup", "preference_assignment"}:
        return "setup"
    if phase.startswith("discussion"):
        return "discussion"
    if phase.startswith("private_thinking"):
        return "private_thinking"
    if phase.startswith("proposal"):
        return "proposal"
    if phase.startswith("voting") or "vote" in phase:
        return "voting"
    if phase.startswith("reflection"):
        return "reflection"
    return "other"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ttc_target_agent(config: dict[str, Any]) -> str:
    return "Agent_1" if int(config.get("target_position", 0)) == 0 else "Agent_2"


def analyze_ttc(ttc_root: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    partial_path = ttc_root / "monitoring/partial_results_latest.csv"
    ttc = pd.read_csv(partial_path)
    order_avg = (
        ttc.groupby(["family", "provider", "level", "level_index", "game", "game_cell"], dropna=False)
        .agg(
            order_count=("order", "nunique"),
            run_count=("config_id", "size"),
            target_utility=("target_utility", "mean"),
            baseline_utility=("baseline_utility", "mean"),
            utility_gap=("utility_gap", "mean"),
            target_compute_tokens_per_call=("target_compute_tokens_per_call", "mean"),
            target_output_tokens_per_call=("target_output_tokens_per_call", "mean"),
            target_reasoning_tokens_raw_per_call=("target_reasoning_tokens_raw_per_call", "mean"),
            consensus_rate=("consensus", "mean"),
            mean_round=("round", "mean"),
        )
        .reset_index()
        .sort_values(["family", "game_cell", "level_index"])
    )
    write_csv(output_dir / "ttc_order_averaged.csv", order_avg)

    config_rows = []
    for path in sorted((ttc_root / "configs").glob("config_*.json")):
        cfg = load_json(path)
        config_rows.append(cfg)
    config_df = pd.DataFrame(config_rows)

    stage_rows: list[dict[str, Any]] = []
    run_stage_totals: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for _, cfg in config_df.iterrows():
        config_id = int(cfg["config_id"])
        run_dir = PROJECT_ROOT / str(cfg["output_dir"])
        interaction_path = run_dir / "run_1_all_interactions.json"
        if not interaction_path.exists():
            continue
        target_agent = ttc_target_agent(cfg.to_dict())
        interactions = load_json(interaction_path)
        for interaction in interactions:
            if interaction.get("agent_id") != target_agent:
                continue
            usage = interaction.get("token_usage") or {}
            if not usage:
                continue
            stage = stage_from_phase(str(interaction.get("phase", "")))
            input_tokens = float(usage.get("input_tokens") or 0.0)
            output_tokens = float(usage.get("output_tokens") or 0.0)
            reasoning_tokens = float(usage.get("reasoning_tokens") or usage.get("thinking_tokens") or 0.0)
            total_tokens = input_tokens + output_tokens + reasoning_tokens
            run_stage_totals[config_id][f"{stage}_total_tokens"] += total_tokens
            run_stage_totals[config_id][f"{stage}_output_tokens"] += output_tokens
            run_stage_totals[config_id][f"{stage}_reasoning_tokens"] += reasoning_tokens
            run_stage_totals[config_id][f"{stage}_call_count"] += 1.0
            stage_rows.append(
                {
                    "config_id": config_id,
                    "family": cfg["target_model_family"],
                    "level": cfg["target_reasoning_level_requested"],
                    "level_index": cfg["target_reasoning_level_index"],
                    "game": cfg["game_label"],
                    "game_cell": cfg["game_cell_id"],
                    "order": cfg["order"],
                    "stage": stage,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "total_tokens": total_tokens,
                }
            )
    stage_raw = pd.DataFrame(stage_rows)
    write_csv(output_dir / "ttc_stage_token_calls.csv", stage_raw)

    if stage_raw.empty:
        return order_avg, pd.DataFrame()

    stage_summary = (
        stage_raw.groupby(["family", "level", "level_index", "stage"], dropna=False)
        .agg(
            call_count=("total_tokens", "size"),
            total_tokens=("total_tokens", "sum"),
            mean_tokens_per_call=("total_tokens", "mean"),
            input_tokens_total=("input_tokens", "sum"),
            output_tokens_total=("output_tokens", "sum"),
            reasoning_tokens_total=("reasoning_tokens", "sum"),
        )
        .reset_index()
    )
    stage_summary["stage"] = pd.Categorical(stage_summary["stage"], STAGE_ORDER, ordered=True)
    stage_summary = stage_summary.sort_values(["family", "level_index", "stage"])
    write_csv(output_dir / "ttc_stage_tokens_by_stage.csv", stage_summary)

    run_stage = pd.DataFrame(
        [{"config_id": config_id, **values} for config_id, values in run_stage_totals.items()]
    ).fillna(0.0)
    ttc_run = ttc.merge(run_stage, on="config_id", how="left").fillna(0.0)
    corr_rows: list[dict[str, Any]] = []
    for stage in STAGE_ORDER:
        for token_col in [f"{stage}_total_tokens", f"{stage}_reasoning_tokens", f"{stage}_output_tokens"]:
            if token_col not in ttc_run.columns or np.isclose(ttc_run[token_col].var(), 0.0):
                continue
            for y_col in ["target_utility", "utility_gap"]:
                corr = ttc_run[[token_col, y_col]].corr().iloc[0, 1]
                corr_rows.append(
                    {
                        "stage": stage,
                        "token_column": token_col,
                        "outcome": y_col,
                        "pearson_r": corr,
                        "n": int(ttc_run[[token_col, y_col]].dropna().shape[0]),
                    }
                )
    corr_df = pd.DataFrame(corr_rows)
    write_csv(output_dir / "ttc_stage_token_correlations.csv", corr_df)
    write_csv(output_dir / "ttc_run_stage_tokens.csv", ttc_run)
    return order_avg, stage_summary


def plot_normalized_multiagent(summary: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.1), sharey=False)
    families = ["homogeneous_control", "homogeneous_adversary", "heterogeneous_random"]
    colors = {
        "homogeneous_control": "#2563eb",
        "homogeneous_adversary": "#dc2626",
        "heterogeneous_random": "#16a34a",
    }
    for ax, game in zip(axes, ["game1", "game2", "game3"]):
        game_df = summary[summary["game_label"] == game]
        for family in families:
            group = game_df[game_df["experiment_family"] == family].sort_values("n_agents")
            if group.empty:
                continue
            ax.plot(
                group["n_agents"],
                group["raw_utility_mean"],
                marker="o",
                linestyle="--",
                linewidth=1.0,
                color=colors[family],
                alpha=0.35,
            )
            ax.plot(
                group["n_agents"],
                group["normalized_utility_mean"],
                marker="o",
                linewidth=1.8,
                color=colors[family],
                label=FAMILY_NAMES[family],
            )
        ax.axhline(1.0, color="#444444", linewidth=0.7, alpha=0.6)
        ax.set_title(GAME_NAMES[game])
        ax.set_xlabel("Number of agents")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("Mean normalized utility (solid); raw utility (faint dashed)")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = output_dir / "multiagent_normalized_utility_by_n.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_rounds(rounds_summary: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.1), sharey=True)
    colors = {
        "homogeneous_control": "#2563eb",
        "homogeneous_adversary": "#dc2626",
        "heterogeneous_random": "#16a34a",
    }
    for ax, game in zip(axes, ["game1", "game2", "game3"]):
        game_df = rounds_summary[rounds_summary["game_label"] == game]
        for family, group in game_df.groupby("experiment_family"):
            group = group.sort_values("n_agents")
            err = group["final_round_std"].fillna(0.0) / np.sqrt(group["run_count"].clip(lower=1))
            ax.errorbar(
                group["n_agents"],
                group["final_round_mean"],
                yerr=1.96 * err,
                marker="o",
                linewidth=1.7,
                capsize=3,
                color=colors.get(family, "#555555"),
                label=FAMILY_NAMES.get(family, family),
            )
        ax.set_title(GAME_NAMES[game])
        ax.set_xlabel("Number of agents")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("Mean final round")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = output_dir / "rounds_to_consensus_by_n_game_family.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_utility_by_round(agents: pd.DataFrame, output_dir: Path) -> Path:
    plot_df = agents[(agents["n_agents"] > 2) & agents["consensus_reached"].astype(bool)].copy()
    if plot_df.empty:
        return output_dir / "discounted_undiscounted_utility_by_round.png"
    grouped = (
        plot_df.groupby(["game_label", "final_round"], dropna=False)
        .agg(
            discounted=("final_utility", "mean"),
            undiscounted=("undiscounted_utility", "mean"),
            n=("agent_id", "size"),
        )
        .reset_index()
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.1), sharey=False)
    for ax, game in zip(axes, ["game1", "game2", "game3"]):
        group = grouped[grouped["game_label"] == game].sort_values("final_round")
        if group.empty:
            continue
        ax.plot(group["final_round"], group["discounted"], marker="o", label="discounted", color="#2563eb")
        ax.plot(group["final_round"], group["undiscounted"], marker="s", label="undiscounted", color="#dc2626")
        ax.set_title(GAME_NAMES[game])
        ax.set_xlabel("Final round")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("Mean agent utility")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = output_dir / "discounted_undiscounted_utility_by_round.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ttc_order_avg(order_avg: pd.DataFrame, output_dir: Path) -> Path:
    colors = {"gpt-5": "#2563eb", "claude-sonnet-4-6": "#dc2626", "gemini-3-flash": "#16a34a"}
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.1), sharey=True)
    for ax, game in zip(axes, ["game1", "game2", "game3"]):
        game_df = order_avg[order_avg["game"] == game]
        for family, group in game_df.groupby("family"):
            ax.scatter(
                group["target_compute_tokens_per_call"],
                group["target_utility"],
                c=group["level_index"],
                cmap="viridis",
                s=56,
                marker="o" if family == "gpt-5" else ("s" if family == "claude-sonnet-4-6" else "^"),
                edgecolor=colors.get(family, "#333333"),
                linewidth=0.7,
                alpha=0.88,
                label=family,
            )
        ax.set_title(GAME_NAMES[game])
        ax.set_xlabel("Target compute tokens per call")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("Order-averaged target utility")
    handles, labels = axes[-1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[-1].legend(by_label.values(), by_label.keys(), frameon=False, fontsize=8)
    fig.tight_layout()
    path = output_dir / "ttc_order_averaged_target_payoff_vs_compute.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ttc_stage(stage_summary: pd.DataFrame, output_dir: Path) -> Path:
    if stage_summary.empty:
        return output_dir / "ttc_stage_tokens_by_phase.png"
    overall = (
        stage_summary.groupby("stage", dropna=False)
        .agg(total_tokens=("total_tokens", "sum"), mean_tokens_per_call=("mean_tokens_per_call", "mean"))
        .reset_index()
    )
    overall["stage"] = pd.Categorical(overall["stage"], STAGE_ORDER, ordered=True)
    overall = overall.sort_values("stage")
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(overall["stage"].astype(str), overall["total_tokens"], color="#2563eb", alpha=0.82)
    ax.set_ylabel("Target total tokens")
    ax.set_xlabel("Phase")
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    path = output_dir / "ttc_stage_tokens_by_phase.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def copy_for_paper(paths: Sequence[Path], output_dir: Path) -> None:
    OVERLEAF_GRAPHICS.mkdir(parents=True, exist_ok=True)
    for src in paths:
        if src.exists():
            shutil.copy2(src, OVERLEAF_GRAPHICS / src.name)

    requested = [
        PROJECT_ROOT
        / "analysis/full_games123_all_success_preliminary_20260428/all_success_game1_hom_adversary_payoff_vs_elo_mean_faceted_by_competition.png",
        PROJECT_ROOT
        / "analysis/full_games123_all_success_preliminary_20260428/all_success_game1_hom_baseline_payoff_vs_adversary_elo_mean_faceted_by_competition.png",
    ]
    for src in requested:
        if src.exists():
            shutil.copy2(src, OVERLEAF_GRAPHICS / src.name)


def fmt_num(value: Any, digits: int = 2) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"


def write_latex_snippets(
    output_dir: Path,
    exploitation: pd.DataFrame,
    utility_reg: pd.DataFrame,
    normalized_summary: pd.DataFrame,
    early_late: pd.DataFrame,
    utility_round_reg: pd.DataFrame,
    hetero_n2: pd.DataFrame,
) -> None:
    lines: list[str] = [
        "% Auto-generated by scripts/analyze_neurips_revision_stats.py",
    ]
    adv_ei = exploitation[(exploitation["role"] == "adversary")].copy()
    for _, row in adv_ei.iterrows():
        key = row["game_label"]
        lines.append(
            f"\\newcommand{{\\{key}AdvExploitationSlope}}{{{fmt_num(row['slope_per_100_elo'], 3)}}}"
        )
        lines.append(f"\\newcommand{{\\{key}AdvExploitationP}}{{{fmt_num(row['p_value'], 3)}}}")
    for game in ["game1", "game2", "game3"]:
        term = utility_reg[
            (utility_reg["game_label"] == game) & (utility_reg["term"] == "elo_per_100")
        ]
        if not term.empty:
            row = term.iloc[0]
            lines.append(f"\\newcommand{{\\{game}UtilityEloSlope}}{{{fmt_num(row['estimate'], 2)}}}")
            lines.append(f"\\newcommand{{\\{game}UtilityEloP}}{{{fmt_num(row['p_value'], 3)}}}")
    controls = normalized_summary[normalized_summary["experiment_family"] == "homogeneous_control"]
    for game in ["game1", "game2", "game3"]:
        g2 = controls[(controls["game_label"] == game) & (controls["n_agents"] == 2)]
        g10 = controls[(controls["game_label"] == game) & (controls["n_agents"] == 10)]
        if not g2.empty and not g10.empty:
            lines.append(f"\\newcommand{{\\{game}ControlRawNtwo}}{{{fmt_num(g2.iloc[0]['raw_utility_mean'], 1)}}}")
            lines.append(f"\\newcommand{{\\{game}ControlRawNten}}{{{fmt_num(g10.iloc[0]['raw_utility_mean'], 1)}}}")
            lines.append(f"\\newcommand{{\\{game}ControlNormNtwo}}{{{fmt_num(g2.iloc[0]['normalized_utility_mean'], 2)}}}")
            lines.append(f"\\newcommand{{\\{game}ControlNormNten}}{{{fmt_num(g10.iloc[0]['normalized_utility_mean'], 2)}}}")
    round_term = utility_round_reg[
        (utility_round_reg["dependent"] == "final_utility") & (utility_round_reg["term"] == "final_round")
    ]
    if not round_term.empty:
        row = round_term.iloc[0]
        lines.append(f"\\newcommand{{\\UtilityRoundDiscountedCoef}}{{{fmt_num(row['estimate'], 2)}}}")
        lines.append(f"\\newcommand{{\\UtilityRoundDiscountedP}}{{{fmt_num(row['p_value'], 3)}}}")
    for _, row in hetero_n2.iterrows():
        key = row["game_label"]
        lines.append(f"\\newcommand{{\\{key}HeteroNtwoSlope}}{{{fmt_num(row['slope_per_100_elo'], 2)}}}")
    output_path = output_dir / "latex_snippets.tex"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown_summary(output_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    lines = ["# NeurIPS Revision Analysis Summary", ""]
    for name, table in tables.items():
        lines.append(f"## {name}")
        if table.empty:
            lines.append("_No rows._")
        else:
            lines.append(table.head(12).to_markdown(index=False))
        lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing outputs to {output_dir}")
    exploitation = analyze_exploitation(output_dir)
    utility_reg = analyze_bilateral_utility(output_dir, args.elo_markdown)
    runs, agents = load_multiagent_metrics(args.multiagent_asset_dir, output_dir, args.bootstrap_reps)
    normalized_summary = pd.read_csv(output_dir / "multiagent_normalized_by_game_family_n.csv")
    hetero_n2 = analyze_hetero_n2(agents, output_dir)
    parser_clean = analyze_parser_clean(output_dir)
    rounds_summary, early_late, utility_round_reg = analyze_rounds_and_confounding(
        runs, agents, output_dir, args.bootstrap_reps
    )
    order_avg, stage_summary = analyze_ttc(args.ttc_root, output_dir)

    plot_paths = [
        plot_normalized_multiagent(normalized_summary, output_dir),
        plot_rounds(rounds_summary, output_dir),
        plot_utility_by_round(agents, output_dir),
        plot_ttc_order_avg(order_avg, output_dir),
        plot_ttc_stage(stage_summary, output_dir),
    ]
    if args.copy_to_overleaf:
        copy_for_paper(plot_paths, output_dir)

    write_latex_snippets(
        output_dir,
        exploitation,
        utility_reg,
        normalized_summary,
        early_late,
        utility_round_reg,
        hetero_n2,
    )
    write_markdown_summary(
        output_dir,
        {
            "Exploitation slopes": exploitation,
            "Utility regressions": utility_reg[utility_reg["term"].isin(["elo_per_100", "competition_value", "elo_x_comp"])],
            "Normalized multi-agent summary": normalized_summary,
            "Rounds early-late": early_late,
            "Utility-round regression": utility_round_reg[utility_round_reg["term"] == "final_round"],
            "Heterogeneous N=2 Elo slopes": hetero_n2,
            "Parser-clean robustness": parser_clean,
        },
    )
    print("Done.")


if __name__ == "__main__":
    main()

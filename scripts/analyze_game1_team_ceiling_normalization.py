#!/usr/bin/env python3
"""Normalize Game 1 team-coordination outcomes by exact attainable welfare.

The script reads the 100 locked treatment configurations and both members of
each matched control/treatment pair.  It recomputes utilities directly from the
preference vectors and final allocations, calculates the exact utilitarian
welfare ceiling, solves the exact discrete maximin allocation problem, and
separates structural scarcity from allocation/timing performance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import Bounds, LinearConstraint, milp


NS = (2, 4, 6, 8, 10)
COMPETITIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
CONDITIONS = ("control", "team")
CONDITION_COLORS = {"control": "#6B7280", "team": "#2563EB"}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def agent_sort_key(agent_id: str) -> int:
    return int(agent_id.rsplit("_", 1)[-1])


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    data = np.asarray(list(values), dtype=float)
    n = int(data.size)
    mean = float(np.mean(data)) if n else math.nan
    if n < 2:
        return {
            "n": n,
            "mean": mean,
            "sd": math.nan,
            "sem": math.nan,
            "ci95_low": math.nan,
            "ci95_high": math.nan,
        }
    sd = float(np.std(data, ddof=1))
    sem = sd / math.sqrt(n)
    half_width = float(stats.t.ppf(0.975, n - 1)) * sem
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "sem": sem,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def exact_utilitarian_ceiling(preferences: dict[str, list[float]]) -> float:
    """Maximum total raw utility: assign each item to its highest valuer."""

    agents = sorted(preferences, key=agent_sort_key)
    item_count = len(preferences[agents[0]])
    return float(
        sum(max(float(preferences[agent][item]) for agent in agents) for item in range(item_count))
    )


def exact_maximin_floor(preferences: dict[str, list[float]]) -> tuple[float, float]:
    """Largest utility t that every agent can simultaneously receive.

    This is a binary linear program because every indivisible item must be
    assigned to exactly one agent.  Preference entries in this experiment are
    integer-valued, and HiGHS certifies optimality through scipy.optimize.milp.
    """

    agents = sorted(preferences, key=agent_sort_key)
    values = np.asarray([preferences[agent] for agent in agents], dtype=float)
    n_agents, n_items = values.shape
    variable_count = n_agents * n_items + 1
    t_index = variable_count - 1
    objective = np.zeros(variable_count)
    objective[t_index] = -1.0

    constraint_rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []
    for item in range(n_items):
        row = np.zeros(variable_count)
        for agent_index in range(n_agents):
            row[agent_index * n_items + item] = 1.0
        constraint_rows.append(row)
        lower.append(1.0)
        upper.append(1.0)
    for agent_index in range(n_agents):
        row = np.zeros(variable_count)
        start = agent_index * n_items
        row[start:start + n_items] = -values[agent_index]
        row[t_index] = 1.0
        constraint_rows.append(row)
        lower.append(-np.inf)
        upper.append(0.0)

    result = milp(
        objective,
        integrality=np.r_[np.ones(n_agents * n_items), 0],
        bounds=Bounds(np.zeros(variable_count), np.r_[np.ones(n_agents * n_items), 100.0]),
        constraints=LinearConstraint(
            np.asarray(constraint_rows), np.asarray(lower), np.asarray(upper)
        ),
        options={"time_limit": 60.0, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Maximin MILP failed: {result.message}")
    return float(result.x[t_index]), float(getattr(result, "mip_gap", 0.0) or 0.0)


def utilities_from_allocation(
    preferences: dict[str, list[float]], allocation: dict[str, list[int]]
) -> dict[str, float]:
    utilities: dict[str, float] = {}
    for agent, values in preferences.items():
        utilities[agent] = float(
            sum(float(values[item]) for item in allocation.get(agent, []))
        )
    return utilities


def result_metrics(
    result_path: Path,
    preferences: dict[str, list[float]],
    role_map: dict[str, str],
    gamma: float,
    welfare_ceiling: float,
    per_capita_ceiling: float,
) -> tuple[dict[str, float | int | bool], list[dict[str, Any]]]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    allocation = result.get("final_allocation") or {}
    final_round = int(result.get("final_round") or 0)
    consensus = bool(result.get("consensus_reached"))
    raw = utilities_from_allocation(preferences, allocation) if consensus else {
        agent: 0.0 for agent in preferences
    }
    discount_factor = gamma ** (final_round - 1) if final_round > 0 else 0.0
    recorded = {agent: float(value) for agent, value in (result.get("final_utilities") or {}).items()}
    if set(recorded) != set(preferences):
        raise ValueError(f"Final-utility roster mismatch: {result_path}")
    max_delta = max(
        abs(recorded[agent] - raw[agent] * discount_factor) for agent in preferences
    )
    if max_delta > 1e-7:
        raise ValueError(f"Utility recomputation mismatch {max_delta}: {result_path}")

    adversaries = [agent for agent, role in role_map.items() if role == "adversary"]
    baselines = [agent for agent, role in role_map.items() if role == "baseline"]
    if len(adversaries) != 1 or len(baselines) != len(preferences) - 1:
        raise ValueError(f"Invalid role map: {role_map}")
    adversary = adversaries[0]
    raw_adversary = raw[adversary]
    raw_baseline_mean = float(np.mean([raw[agent] for agent in baselines]))
    discounted_adversary = recorded[adversary]
    discounted_baseline_mean = float(np.mean([recorded[agent] for agent in baselines]))
    raw_total = float(sum(raw.values()))
    discounted_total = float(sum(recorded.values()))

    agent_rows = []
    for agent in sorted(preferences, key=agent_sort_key):
        agent_rows.append({
            "agent_id": agent,
            "role": role_map[agent],
            "personal_ideal_ceiling": float(sum(preferences[agent])),
            "joint_per_capita_ceiling": per_capita_ceiling,
            "raw_utility": raw[agent],
            "discounted_utility": recorded[agent],
            "personal_ideal_attainment": raw[agent] / float(sum(preferences[agent])),
            "per_capita_ceiling_multiple": raw[agent] / per_capita_ceiling,
        })

    return {
        "consensus": consensus,
        "final_round": final_round,
        "discount_factor": discount_factor,
        "raw_total_welfare": raw_total,
        "discounted_total_welfare": discounted_total,
        "raw_mean_agent_payoff": raw_total / len(preferences),
        "discounted_mean_agent_payoff": discounted_total / len(preferences),
        "raw_welfare_efficiency": raw_total / welfare_ceiling,
        "discounted_welfare_efficiency": discounted_total / welfare_ceiling,
        "raw_adversary_payoff": raw_adversary,
        "raw_baseline_mean_payoff": raw_baseline_mean,
        "raw_per_agent_gap": raw_baseline_mean - raw_adversary,
        "raw_adversary_minus_nano_gap": raw_adversary - raw_baseline_mean,
        "raw_adversary_ceiling_multiple": raw_adversary / per_capita_ceiling,
        "raw_baseline_mean_ceiling_multiple": raw_baseline_mean / per_capita_ceiling,
        "discounted_adversary_payoff": discounted_adversary,
        "discounted_baseline_mean_payoff": discounted_baseline_mean,
        "discounted_per_agent_gap": discounted_baseline_mean - discounted_adversary,
        "discounted_adversary_minus_nano_gap": (
            discounted_adversary - discounted_baseline_mean
        ),
        "discounted_adversary_ceiling_multiple": discounted_adversary / per_capita_ceiling,
        "discounted_baseline_mean_ceiling_multiple": discounted_baseline_mean / per_capita_ceiling,
        "raw_ceiling_normalized_gap": (raw_baseline_mean - raw_adversary) / per_capita_ceiling,
        "raw_adversary_minus_nano_ceiling_normalized_gap": (
            raw_adversary - raw_baseline_mean
        ) / per_capita_ceiling,
        "discounted_ceiling_normalized_gap": (
            discounted_baseline_mean - discounted_adversary
        ) / per_capita_ceiling,
        "discounted_adversary_minus_nano_ceiling_normalized_gap": (
            discounted_adversary - discounted_baseline_mean
        ) / per_capita_ceiling,
        "utility_recompute_max_abs_delta": max_delta,
    }, agent_rows


def load_all(results_root: Path) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    ceiling_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []
    agent_rows: list[dict[str, Any]] = []
    validations = {
        "configs": 0,
        "conditions": 0,
        "agents": 0,
        "personal_ceiling_mismatches": [],
        "item_scale_mismatches": [],
        "utility_recompute_max_abs_delta": 0.0,
        "maximin_max_mip_gap": 0.0,
        "pairwise_cosine_abs_error_sum": 0.0,
        "pairwise_cosine_pairs": 0,
        "pairwise_cosine_max_abs_error": 0.0,
    }
    for config_path in sorted((results_root / "configs").glob("config_*.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config_id = int(config["config_id"])
        n_agents = int(config["n_agents"])
        competition = float(config["competition_level"])
        preferences = {
            agent: [float(value) for value in values]
            for agent, values in config["fixed_agent_preferences"].items()
        }
        agents = sorted(preferences, key=agent_sort_key)
        item_count = len(preferences[agents[0]])
        if item_count != int(2.5 * n_agents):
            validations["item_scale_mismatches"].append({
                "config_id": config_id,
                "n_agents": n_agents,
                "n_items": item_count,
            })
        personal_sums = [float(sum(preferences[agent])) for agent in agents]
        for agent, total in zip(agents, personal_sums):
            if abs(total - 100.0) > 1e-9:
                validations["personal_ceiling_mismatches"].append({
                    "config_id": config_id, "agent_id": agent, "sum": total
                })
        welfare_ceiling = exact_utilitarian_ceiling(preferences)
        per_capita_ceiling = welfare_ceiling / n_agents
        maximin_floor, mip_gap = exact_maximin_floor(preferences)
        pairwise_cosines = []
        for left, right in combinations(agents, 2):
            vector_left = np.asarray(preferences[left], dtype=float)
            vector_right = np.asarray(preferences[right], dtype=float)
            pairwise_cosines.append(float(
                np.dot(vector_left, vector_right)
                / (np.linalg.norm(vector_left) * np.linalg.norm(vector_right))
            ))
        cosine_errors = [abs(value - competition) for value in pairwise_cosines]
        validations["pairwise_cosine_abs_error_sum"] += sum(cosine_errors)
        validations["pairwise_cosine_pairs"] += len(cosine_errors)
        validations["pairwise_cosine_max_abs_error"] = max(
            validations["pairwise_cosine_max_abs_error"], max(cosine_errors, default=0.0)
        )
        validations["maximin_max_mip_gap"] = max(
            validations["maximin_max_mip_gap"], mip_gap
        )
        common = {
            "config_id": config_id,
            "control_config_id": int(config["control_config_id"]),
            "n_agents": n_agents,
            "team_size": n_agents - 1,
            "n_items": item_count,
            "items_per_agent": item_count / n_agents,
            "competition_level": competition,
            "adversary_position": config["adversary_position"],
            "seed_replicate": int(config["seed_replicate"]),
            "preference_hash": config["fixed_agent_preferences_sha256"],
        }
        ceiling_rows.append({
            **common,
            "personal_ideal_ceiling_min": min(personal_sums),
            "personal_ideal_ceiling_max": max(personal_sums),
            "utilitarian_total_ceiling": welfare_ceiling,
            "utilitarian_per_capita_ceiling": per_capita_ceiling,
            "maximin_common_floor": maximin_floor,
            "maximin_fraction_of_personal_ideal": maximin_floor / 100.0,
            "maximin_mip_gap": mip_gap,
            "realized_pairwise_cosine_min": min(pairwise_cosines),
            "realized_pairwise_cosine_mean": float(np.mean(pairwise_cosines)),
            "realized_pairwise_cosine_max": max(pairwise_cosines),
            "pairwise_cosine_mean_abs_error": float(np.mean(cosine_errors)),
            "pairwise_cosine_max_abs_error": max(cosine_errors),
        })

        paths = {
            "control": Path(config["control_result_path"]),
            "team": Path(config["output_dir"]) / "experiment_results.json",
        }
        condition_metrics: dict[str, dict[str, Any]] = {}
        for condition, result_path in paths.items():
            metrics, per_agent = result_metrics(
                result_path,
                preferences,
                config["agent_role_map"],
                float(config["gamma_discount"]),
                welfare_ceiling,
                per_capita_ceiling,
            )
            condition_metrics[condition] = metrics
            outcome_rows.append({
                **common,
                "condition": condition,
                "result_path": str(result_path),
                "utilitarian_total_ceiling": welfare_ceiling,
                "utilitarian_per_capita_ceiling": per_capita_ceiling,
                "maximin_common_floor": maximin_floor,
                **metrics,
            })
            for row in per_agent:
                agent_rows.append({
                    **common,
                    "condition": condition,
                    "result_path": str(result_path),
                    "maximin_common_floor": maximin_floor,
                    **row,
                })
            validations["utility_recompute_max_abs_delta"] = max(
                validations["utility_recompute_max_abs_delta"],
                float(metrics["utility_recompute_max_abs_delta"]),
            )

        control = condition_metrics["control"]
        team = condition_metrics["team"]
        outcome_rows.append({
            **common,
            "condition": "paired_effect",
            "result_path": "",
            "utilitarian_total_ceiling": welfare_ceiling,
            "utilitarian_per_capita_ceiling": per_capita_ceiling,
            "maximin_common_floor": maximin_floor,
            "raw_gap_change": float(team["raw_per_agent_gap"]) - float(control["raw_per_agent_gap"]),
            "discounted_gap_change": float(team["discounted_per_agent_gap"]) - float(control["discounted_per_agent_gap"]),
            "raw_ceiling_normalized_gap_change": float(team["raw_ceiling_normalized_gap"]) - float(control["raw_ceiling_normalized_gap"]),
            "discounted_ceiling_normalized_gap_change": float(team["discounted_ceiling_normalized_gap"]) - float(control["discounted_ceiling_normalized_gap"]),
            "raw_adversary_minus_nano_gap_change": float(team["raw_adversary_minus_nano_gap"]) - float(control["raw_adversary_minus_nano_gap"]),
            "discounted_adversary_minus_nano_gap_change": float(team["discounted_adversary_minus_nano_gap"]) - float(control["discounted_adversary_minus_nano_gap"]),
            "raw_adversary_minus_nano_ceiling_normalized_gap_change": float(team["raw_adversary_minus_nano_ceiling_normalized_gap"]) - float(control["raw_adversary_minus_nano_ceiling_normalized_gap"]),
            "discounted_adversary_minus_nano_ceiling_normalized_gap_change": float(team["discounted_adversary_minus_nano_ceiling_normalized_gap"]) - float(control["discounted_adversary_minus_nano_ceiling_normalized_gap"]),
            "raw_welfare_efficiency_change": float(team["raw_welfare_efficiency"]) - float(control["raw_welfare_efficiency"]),
        })
        validations["configs"] += 1

    validations["conditions"] = sum(row["condition"] in CONDITIONS for row in outcome_rows)
    validations["agents"] = len(agent_rows)
    validations["pairwise_cosine_mean_abs_error"] = (
        validations["pairwise_cosine_abs_error_sum"]
        / validations["pairwise_cosine_pairs"]
    )
    del validations["pairwise_cosine_abs_error_sum"]
    validations["valid"] = (
        validations["configs"] == 100
        and validations["conditions"] == 200
        and validations["agents"] == 1200
        and not validations["personal_ceiling_mismatches"]
        and not validations["item_scale_mismatches"]
        and validations["utility_recompute_max_abs_delta"] <= 1e-7
        and validations["maximin_max_mip_gap"] <= 1e-9
    )
    return ceiling_rows, outcome_rows, agent_rows, validations


def build_summaries(
    ceiling_rows: list[dict[str, Any]], outcome_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ceiling_summary: list[dict[str, Any]] = []
    for competition in COMPETITIONS:
        for n_agents in NS:
            selected = [
                row for row in ceiling_rows
                if row["n_agents"] == n_agents and row["competition_level"] == competition
            ]
            for metric in (
                "utilitarian_total_ceiling",
                "utilitarian_per_capita_ceiling",
                "maximin_common_floor",
            ):
                ceiling_summary.append({
                    "competition_level": competition,
                    "n_agents": n_agents,
                    "team_size": n_agents - 1,
                    "metric": metric,
                    **summarize(float(row[metric]) for row in selected),
                })

    outcome_summary: list[dict[str, Any]] = []
    metrics_by_condition = {
        "control": (
            "raw_mean_agent_payoff",
            "raw_welfare_efficiency",
            "raw_adversary_minus_nano_ceiling_normalized_gap",
            "raw_adversary_ceiling_multiple",
            "raw_baseline_mean_ceiling_multiple",
            "discounted_welfare_efficiency",
        ),
        "team": (
            "raw_mean_agent_payoff",
            "raw_welfare_efficiency",
            "raw_adversary_minus_nano_ceiling_normalized_gap",
            "raw_adversary_ceiling_multiple",
            "raw_baseline_mean_ceiling_multiple",
            "discounted_welfare_efficiency",
        ),
        "paired_effect": (
            "raw_adversary_minus_nano_ceiling_normalized_gap_change",
            "raw_welfare_efficiency_change",
        ),
    }
    for condition, metrics in metrics_by_condition.items():
        for competition in COMPETITIONS:
            for n_agents in NS:
                selected = [
                    row for row in outcome_rows
                    if row["condition"] == condition
                    and row["n_agents"] == n_agents
                    and row["competition_level"] == competition
                ]
                for metric in metrics:
                    outcome_summary.append({
                        "condition": condition,
                        "competition_level": competition,
                        "n_agents": n_agents,
                        "team_size": n_agents - 1,
                        "metric": metric,
                        **summarize(float(row[metric]) for row in selected),
                    })
    return ceiling_summary, outcome_summary


def build_by_n_summary(outcome_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate normalized metrics over the five equally represented competitions."""

    rows: list[dict[str, Any]] = []
    metrics_by_condition = {
        "control": (
            "raw_welfare_efficiency",
            "raw_adversary_minus_nano_ceiling_normalized_gap",
        ),
        "team": (
            "raw_welfare_efficiency",
            "raw_adversary_minus_nano_ceiling_normalized_gap",
            "raw_adversary_ceiling_multiple",
            "raw_baseline_mean_ceiling_multiple",
        ),
        "paired_effect": ("raw_adversary_minus_nano_ceiling_normalized_gap_change",),
    }
    for condition, metrics in metrics_by_condition.items():
        for n_agents in NS:
            selected = [
                row for row in outcome_rows
                if row["condition"] == condition and row["n_agents"] == n_agents
            ]
            for metric in metrics:
                values = np.asarray([float(row[metric]) for row in selected], dtype=float)
                estimate = summarize(values)
                t_result = stats.ttest_1samp(values, 0.0) if values.size >= 2 else None
                rows.append({
                    "condition": condition,
                    "n_agents": n_agents,
                    "team_size": n_agents - 1,
                    "metric": metric,
                    **estimate,
                    "one_sample_t": float(t_result.statistic) if t_result is not None else math.nan,
                    "one_sample_p": float(t_result.pvalue) if t_result is not None else math.nan,
                })
    return rows


def bootstrap_percentage_summaries(
    outcome_rows: list[dict[str, Any]],
    *,
    bootstrap_reps: int = 20_000,
    random_seed: int = 20260809,
) -> list[dict[str, Any]]:
    """Estimate Nano payoff as a percentage of adversary payoff.

    Ratios are ratios of cell means, not means of run-level ratios.  A cell
    remains undefined when its adversary mean is zero.  Paired effects use the
    same resampled config indices in both conditions.
    """

    rng = np.random.default_rng(random_seed)
    rows_by_key = {
        (row["condition"], int(row["n_agents"]), float(row["competition_level"]), int(row["config_id"])): row
        for row in outcome_rows
        if row["condition"] in CONDITIONS
    }
    cell_points: dict[tuple[str, int, float], float] = {}
    cell_samples: dict[tuple[str, int, float], np.ndarray] = {}
    paired_points: dict[tuple[int, float], float] = {}
    paired_samples: dict[tuple[int, float], np.ndarray] = {}
    summary_rows: list[dict[str, Any]] = []

    def finite_interval(samples: np.ndarray) -> tuple[float, float, int]:
        finite = samples[np.isfinite(samples)]
        if not finite.size:
            return math.nan, math.nan, 0
        return (
            float(np.quantile(finite, 0.025)),
            float(np.quantile(finite, 0.975)),
            int(finite.size),
        )

    def percentage_ratio(nano_mean: float, adversary_mean: float) -> float:
        if adversary_mean == 0.0:
            return math.nan
        return 100.0 * nano_mean / adversary_mean

    for n_agents in NS:
        for competition in COMPETITIONS:
            config_ids = sorted({
                key[3] for key in rows_by_key
                if key[1] == n_agents and key[2] == competition
            })
            if len(config_ids) != 4:
                raise ValueError(
                    f"Expected four matched runs for N={n_agents}, competition={competition}; "
                    f"found {len(config_ids)}"
                )
            sample_indices = rng.integers(
                0, len(config_ids), size=(bootstrap_reps, len(config_ids))
            )
            condition_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for condition in CONDITIONS:
                adversary = np.asarray([
                    float(rows_by_key[(condition, n_agents, competition, config_id)]["raw_adversary_payoff"])
                    for config_id in config_ids
                ])
                baseline = np.asarray([
                    float(rows_by_key[(condition, n_agents, competition, config_id)]["raw_baseline_mean_payoff"])
                    for config_id in config_ids
                ])
                condition_arrays[condition] = (adversary, baseline)
                point = percentage_ratio(
                    float(np.mean(baseline)), float(np.mean(adversary))
                )
                sampled_adversary = adversary[sample_indices].mean(axis=1)
                sampled_baseline = baseline[sample_indices].mean(axis=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    samples = 100.0 * sampled_baseline / sampled_adversary
                low, high, finite_reps = finite_interval(samples)
                cell_points[(condition, n_agents, competition)] = point
                cell_samples[(condition, n_agents, competition)] = samples
                summary_rows.append({
                    "aggregation": "n_by_competition",
                    "condition": condition,
                    "n_agents": n_agents,
                    "team_size": n_agents - 1,
                    "competition_level": competition,
                    "metric": "nano_payoff_percent_of_adversary",
                    "n": len(config_ids),
                    "estimate": point,
                    "ci95_low": low,
                    "ci95_high": high,
                    "bootstrap_reps": bootstrap_reps,
                    "finite_bootstrap_reps": finite_reps,
                    "adversary_mean_payoff": float(np.mean(adversary)),
                    "nano_mean_payoff": float(np.mean(baseline)),
                })

            control_adversary, control_baseline = condition_arrays["control"]
            team_adversary, team_baseline = condition_arrays["team"]
            point = percentage_ratio(
                float(np.mean(team_baseline)), float(np.mean(team_adversary))
            ) - percentage_ratio(
                float(np.mean(control_baseline)), float(np.mean(control_adversary))
            )
            control_sampled_adversary = control_adversary[sample_indices].mean(axis=1)
            control_sampled_baseline = control_baseline[sample_indices].mean(axis=1)
            team_sampled_adversary = team_adversary[sample_indices].mean(axis=1)
            team_sampled_baseline = team_baseline[sample_indices].mean(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                samples = (
                    100.0 * team_sampled_baseline / team_sampled_adversary
                    - 100.0 * control_sampled_baseline / control_sampled_adversary
                )
            low, high, finite_reps = finite_interval(samples)
            paired_points[(n_agents, competition)] = point
            paired_samples[(n_agents, competition)] = samples
            summary_rows.append({
                "aggregation": "n_by_competition",
                "condition": "paired_effect",
                "n_agents": n_agents,
                "team_size": n_agents - 1,
                "competition_level": competition,
                "metric": "percentage_point_change_team_vs_control",
                "n": len(config_ids),
                "estimate": point,
                "ci95_low": low,
                "ci95_high": high,
                "bootstrap_reps": bootstrap_reps,
                "finite_bootstrap_reps": finite_reps,
                "adversary_mean_payoff": "",
                "nano_mean_payoff": "",
            })

    for n_agents in NS:
        for condition in CONDITIONS:
            points = np.asarray([
                cell_points[(condition, n_agents, competition)]
                for competition in COMPETITIONS
            ])
            samples = np.vstack([
                cell_samples[(condition, n_agents, competition)]
                for competition in COMPETITIONS
            ])
            with np.errstate(invalid="ignore"):
                aggregate_samples = np.nanmean(samples, axis=0)
            low, high, finite_reps = finite_interval(aggregate_samples)
            summary_rows.append({
                "aggregation": "n_equal_competition_weight",
                "condition": condition,
                "n_agents": n_agents,
                "team_size": n_agents - 1,
                "competition_level": "all_equal_weight",
                "metric": "nano_payoff_percent_of_adversary",
                "n": 20,
                "estimate": float(np.mean(points)),
                "ci95_low": low,
                "ci95_high": high,
                "bootstrap_reps": bootstrap_reps,
                "finite_bootstrap_reps": finite_reps,
                "adversary_mean_payoff": "",
                "nano_mean_payoff": "",
            })
        points = np.asarray([
            paired_points[(n_agents, competition)] for competition in COMPETITIONS
        ])
        samples = np.vstack([
            paired_samples[(n_agents, competition)] for competition in COMPETITIONS
        ])
        with np.errstate(invalid="ignore"):
            aggregate_samples = np.nanmean(samples, axis=0)
        low, high, finite_reps = finite_interval(aggregate_samples)
        summary_rows.append({
            "aggregation": "n_equal_competition_weight",
            "condition": "paired_effect",
            "n_agents": n_agents,
            "team_size": n_agents - 1,
            "competition_level": "all_equal_weight",
            "metric": "percentage_point_change_team_vs_control",
            "n": 20,
            "estimate": float(np.mean(points)),
            "ci95_low": low,
            "ci95_high": high,
            "bootstrap_reps": bootstrap_reps,
            "finite_bootstrap_reps": finite_reps,
            "adversary_mean_payoff": "",
            "nano_mean_payoff": "",
        })
    return summary_rows


def endpoint_decomposition(
    ceiling_rows: list[dict[str, Any]], outcome_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for competition in COMPETITIONS:
            def cell_mean(source: list[dict[str, Any]], n_agents: int, metric: str) -> float:
                selected = [
                    float(row[metric]) for row in source
                    if row["n_agents"] == n_agents
                    and row["competition_level"] == competition
                    and ("condition" not in row or row["condition"] == condition)
                ]
                return float(np.mean(selected))

            ceiling_n2 = cell_mean(ceiling_rows, 2, "utilitarian_per_capita_ceiling")
            ceiling_n10 = cell_mean(ceiling_rows, 10, "utilitarian_per_capita_ceiling")
            efficiency_n2 = cell_mean(outcome_rows, 2, "raw_welfare_efficiency")
            efficiency_n10 = cell_mean(outcome_rows, 10, "raw_welfare_efficiency")
            payoff_n2 = cell_mean(outcome_rows, 2, "raw_mean_agent_payoff")
            payoff_n10 = cell_mean(outcome_rows, 10, "raw_mean_agent_payoff")
            observed_drop = payoff_n2 - payoff_n10
            structural_component = (
                0.5 * (efficiency_n2 + efficiency_n10) * (ceiling_n2 - ceiling_n10)
            )
            efficiency_component = (
                0.5 * (ceiling_n2 + ceiling_n10) * (efficiency_n2 - efficiency_n10)
            )
            rows.append({
                "condition": condition,
                "competition_level": competition,
                "n2_per_capita_ceiling": ceiling_n2,
                "n10_per_capita_ceiling": ceiling_n10,
                "n2_raw_efficiency": efficiency_n2,
                "n10_raw_efficiency": efficiency_n10,
                "n2_raw_mean_payoff": payoff_n2,
                "n10_raw_mean_payoff": payoff_n10,
                "observed_payoff_drop_n2_to_n10": observed_drop,
                "structural_ceiling_component": structural_component,
                "negotiation_efficiency_component": efficiency_component,
                "structural_share_of_drop": (
                    structural_component / observed_drop if abs(observed_drop) > 1e-12 else math.nan
                ),
            })
    return rows


def estimate_lookup(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {
        (
            row.get("condition"),
            float(row["competition_level"]),
            int(row["n_agents"]),
            row["metric"],
        ): row
        for row in rows
    }


def add_errorbar(
    ax: plt.Axes,
    x: list[int],
    estimates: list[dict[str, Any]],
    *,
    label: str,
    color: Any,
    marker: str = "o",
    linestyle: str = "-",
) -> None:
    means = np.asarray([float(row["mean"]) for row in estimates])
    low = means - np.asarray([float(row["ci95_low"]) for row in estimates])
    high = np.asarray([float(row["ci95_high"]) for row in estimates]) - means
    ax.errorbar(
        x, means, yerr=np.vstack([low, high]), label=label, color=color,
        marker=marker, linestyle=linestyle, linewidth=2, markersize=5, capsize=3,
    )


def plot_ceiling_structure(
    ceiling_summary: list[dict[str, Any]], output_dir: Path
) -> Path:
    lookup = estimate_lookup(ceiling_summary)
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(COMPETITIONS)))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    metrics = (
        ("utilitarian_total_ceiling", "Maximum total points", "Total points"),
        ("utilitarian_per_capita_ceiling", "Maximum average points per agent", "Points per agent"),
        ("maximin_common_floor", "Largest score attainable by every agent", "Points per agent"),
    )
    for competition, color in zip(COMPETITIONS, colors):
        for ax, (metric, _, _) in zip(axes, metrics):
            estimates = [lookup[(None, competition, n_agents, metric)] for n_agents in NS]
            add_errorbar(
                ax, list(NS), estimates,
                label=f"Competition {competition:g}", color=color,
            )
    for ax, (_, title, ylabel) in zip(axes, metrics):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Total agents N (items = 2.5N)")
        ax.set_xticks(NS)
        ax.grid(alpha=0.22)
    axes[2].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Exact structural payoff ceilings from the locked Game 1 preference vectors\n"
        "Each agent's personal ideal is always 100; points shown are jointly feasible",
        fontsize=15,
    )
    path = output_dir / "attainable_ceiling_by_n_competition.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def plot_payoff_against_ceiling(
    ceiling_summary: list[dict[str, Any]],
    outcome_summary: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    ceiling_lookup = estimate_lookup(ceiling_summary)
    outcome_lookup = estimate_lookup(outcome_summary)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True, constrained_layout=True)
    axes_flat = axes.ravel()
    for index, competition in enumerate(COMPETITIONS):
        ax = axes_flat[index]
        ceiling_estimates = [
            ceiling_lookup[(None, competition, n_agents, "utilitarian_per_capita_ceiling")]
            for n_agents in NS
        ]
        add_errorbar(
            ax, list(NS), ceiling_estimates, label="Attainable average ceiling",
            color="black", marker="D", linestyle="--",
        )
        for condition, marker in (("control", "o"), ("team", "s")):
            estimates = [
                outcome_lookup[(condition, competition, n_agents, "raw_mean_agent_payoff")]
                for n_agents in NS
            ]
            add_errorbar(
                ax, list(NS), estimates,
                label="Independent control" if condition == "control" else "Coordinated team",
                color=CONDITION_COLORS[condition], marker=marker,
            )
        ax.set_title(f"Competition = {competition:g}")
        ax.set_xticks(NS)
        ax.grid(alpha=0.22)
        ax.set_xlabel("N")
        ax.set_ylabel("Raw mean payoff per agent")
    axes_flat[0].legend(frameon=False, fontsize=9)
    axes_flat[-1].axis("off")
    fig.suptitle(
        "Observed payoff versus the exact per-agent feasible ceiling\n"
        "Raw utilities remove round discounting; four matched preference tables per point",
        fontsize=15,
    )
    path = output_dir / "payoff_vs_attainable_ceiling_by_competition.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def plot_normalized_performance(
    outcome_summary: list[dict[str, Any]], output_dir: Path
) -> Path:
    lookup = estimate_lookup(outcome_summary)
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(COMPETITIONS)))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for competition, color in zip(COMPETITIONS, colors):
        efficiency = [
            lookup[("team", competition, n_agents, "raw_welfare_efficiency")]
            for n_agents in NS
        ]
        gap = [
            lookup[("team", competition, n_agents, "raw_adversary_minus_nano_ceiling_normalized_gap")]
            for n_agents in NS
        ]
        effect = [
            lookup[("paired_effect", competition, n_agents, "raw_adversary_minus_nano_ceiling_normalized_gap_change")]
            for n_agents in NS
        ]
        add_errorbar(
            axes[0], list(NS), efficiency,
            label=f"Competition {competition:g}", color=color,
        )
        add_errorbar(axes[1], list(NS), gap, label="", color=color)
        add_errorbar(axes[2], list(NS), effect, label="", color=color)
    axes[0].set_title("Fraction of feasible welfare realized")
    axes[0].set_ylabel("Raw total payoff / exact maximum")
    axes[1].set_title("Adversary gap after ceiling normalization")
    axes[1].set_ylabel("(GPT-5.4 − Nano mean) / per-capita ceiling")
    axes[2].set_title("Coordination effect after normalization")
    axes[2].set_ylabel("Change in normalized GPT-5.4 − Nano gap")
    for ax in axes:
        ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.65)
        ax.set_xlabel("Total agents N")
        ax.set_xticks(NS)
        ax.grid(alpha=0.22)
    axes[0].set_ylim(bottom=-0.03)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Ceiling-normalized Game 1 performance by competition level\n"
        "Raw allocation utilities remove the separate effect of round discounting",
        fontsize=15,
    )
    path = output_dir / "ceiling_normalized_performance.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def plot_recommended_primary(
    ceiling_summary: list[dict[str, Any]],
    by_n_summary: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Publication-facing replacement for the raw-payoff scaling figures."""

    ceiling_lookup = estimate_lookup(ceiling_summary)
    by_n_lookup = {
        (row["condition"], int(row["n_agents"]), row["metric"]): row
        for row in by_n_summary
    }
    competition_colors = plt.get_cmap("viridis")(
        np.linspace(0.05, 0.95, len(COMPETITIONS))
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10), constrained_layout=True)

    for competition, color in zip(COMPETITIONS, competition_colors):
        estimates = [
            ceiling_lookup[(None, competition, n_agents, "utilitarian_per_capita_ceiling")]
            for n_agents in NS
        ]
        add_errorbar(
            axes[0, 0], list(NS), estimates,
            label=f"Competition {competition:g}", color=color,
        )
    axes[0, 0].set_title("A. Structural payoff scale")
    axes[0, 0].set_ylabel("Maximum feasible average payoff")
    axes[0, 0].legend(frameon=False, fontsize=8.5, ncol=2)

    for condition, marker in (("control", "o"), ("team", "s")):
        estimates = [
            by_n_lookup[(condition, n_agents, "raw_welfare_efficiency")]
            for n_agents in NS
        ]
        add_errorbar(
            axes[0, 1], list(NS), estimates,
            label="Independent control" if condition == "control" else "Coordinated team",
            color=CONDITION_COLORS[condition], marker=marker,
        )
    axes[0, 1].axhline(1, color="black", linewidth=1, linestyle="--", alpha=0.65)
    axes[0, 1].set_title("B. Fraction of feasible welfare realized")
    axes[0, 1].set_ylabel("Raw total payoff / exact maximum")
    axes[0, 1].set_ylim(0.72, 1.04)
    axes[0, 1].legend(frameon=False, fontsize=9)

    for metric, label, color, marker in (
        (
            "raw_adversary_ceiling_multiple", "GPT-5.4 adversary",
            "#C43C39", "o",
        ),
        (
            "raw_baseline_mean_ceiling_multiple", "Average GPT-5 Nano teammate",
            "#2563EB", "s",
        ),
    ):
        estimates = [by_n_lookup[("team", n_agents, metric)] for n_agents in NS]
        add_errorbar(
            axes[1, 0], list(NS), estimates,
            label=label, color=color, marker=marker,
        )
    axes[1, 0].axhline(1, color="black", linewidth=1, linestyle="--", alpha=0.65)
    axes[1, 0].set_title("C. Team versus adversary on a comparable scale")
    axes[1, 0].set_ylabel("Raw payoff / per-capita feasible ceiling")
    axes[1, 0].legend(frameon=False, fontsize=9)

    effect = [
        by_n_lookup[("paired_effect", n_agents, "raw_adversary_minus_nano_ceiling_normalized_gap_change")]
        for n_agents in NS
    ]
    add_errorbar(
        axes[1, 1], list(NS), effect,
        label="Coordinated − independent", color="#7C3AED", marker="D",
    )
    axes[1, 1].axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.65)
    axes[1, 1].set_title("D. Paired coordination contrast on the normalized scale")
    axes[1, 1].set_ylabel("Change in normalized adversary − Nano gap")
    axes[1, 1].legend(frameon=False, fontsize=9)

    for ax in axes.ravel():
        ax.set_xlabel("Total agents N (Nano team size = N−1)")
        ax.set_xticks(NS, [f"{n}\n({n-1})" for n in NS])
        ax.grid(alpha=0.22)
    fig.suptitle(
        "Recommended Game 1 scaling analysis: ceiling-normalized outcomes\n"
        "Panels B–D remove round discounting; estimates average equally represented competition cells",
        fontsize=15,
    )
    path = output_dir / "recommended_primary_ceiling_normalized_results.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def plot_recommended_by_competition(
    outcome_summary: list[dict[str, Any]], output_dir: Path
) -> Path:
    """Competition-stratified normalized role outcomes and paired effects."""

    lookup = estimate_lookup(outcome_summary)
    fig, axes = plt.subplots(
        len(COMPETITIONS), 2, figsize=(13, 19), sharex=True, sharey="col",
        constrained_layout=True,
    )
    for row_index, competition in enumerate(COMPETITIONS):
        for metric, label, color, marker in (
            (
                "raw_adversary_ceiling_multiple", "GPT-5.4 adversary",
                "#C43C39", "o",
            ),
            (
                "raw_baseline_mean_ceiling_multiple", "Average GPT-5 Nano teammate",
                "#2563EB", "s",
            ),
        ):
            estimates = [
                lookup[("team", competition, n_agents, metric)] for n_agents in NS
            ]
            add_errorbar(
                axes[row_index, 0], list(NS), estimates,
                label=label, color=color, marker=marker,
            )
        effect = [
            lookup[(
                "paired_effect", competition, n_agents,
                "raw_adversary_minus_nano_ceiling_normalized_gap_change",
            )]
            for n_agents in NS
        ]
        add_errorbar(
            axes[row_index, 1], list(NS), effect,
            label="Coordinated − independent", color="#7C3AED", marker="D",
        )
        axes[row_index, 0].axhline(
            1, color="black", linewidth=1, linestyle="--", alpha=0.65
        )
        axes[row_index, 1].axhline(
            0, color="black", linewidth=1, linestyle="--", alpha=0.65
        )
        axes[row_index, 0].set_ylabel(
            f"Competition = {competition:g}\nPayoff / per-capita ceiling"
        )
        axes[row_index, 1].set_ylabel("Change in normalized gap")
        for ax in axes[row_index]:
            ax.set_xticks(NS, [f"{n}\n({n-1})" for n in NS])
            ax.grid(alpha=0.22)
            if row_index == len(COMPETITIONS) - 1:
                ax.set_xlabel("Total agents N (Nano team size = N−1)")
    axes[0, 0].set_title("Coordinated team: Nano versus GPT-5.4")
    axes[0, 1].set_title("Coordination effect versus matched control")
    axes[0, 0].legend(frameon=False, fontsize=9)
    axes[0, 1].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Recommended competition-stratified Game 1 analysis\n"
        "Ceiling-normalized raw utilities; 95% t intervals across four matched runs per point",
        fontsize=15,
    )
    path = output_dir / "recommended_by_competition_ceiling_normalized_results.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def add_bootstrap_interval(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    *,
    label: str,
    color: str,
    marker: str,
) -> None:
    means = np.asarray([float(row["estimate"]) for row in rows])
    low = means - np.asarray([float(row["ci95_low"]) for row in rows])
    high = np.asarray([float(row["ci95_high"]) for row in rows]) - means
    ax.errorbar(
        list(NS), means, yerr=np.vstack([low, high]), label=label, color=color,
        marker=marker, linewidth=2.2, markersize=6, capsize=3,
    )


def plot_percentage_primary(
    percentage_rows: list[dict[str, Any]], output_dir: Path
) -> Path:
    selected = [
        row for row in percentage_rows
        if row["aggregation"] == "n_equal_competition_weight"
    ]
    lookup = {
        (row["condition"], int(row["n_agents"]), row["metric"]): row
        for row in selected
    }
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    for condition, marker in (("control", "o"), ("team", "s")):
        estimates = [
            lookup[(condition, n_agents, "nano_payoff_percent_of_adversary")]
            for n_agents in NS
        ]
        add_bootstrap_interval(
            axes[0], estimates,
            label="Independent control" if condition == "control" else "Coordinated team",
            color=CONDITION_COLORS[condition], marker=marker,
        )
    effect = [
        lookup[("paired_effect", n_agents, "percentage_point_change_team_vs_control")]
        for n_agents in NS
    ]
    add_bootstrap_interval(
        axes[1], effect, label="Coordinated − independent",
        color="#7C3AED", marker="D",
    )
    axes[0].axhline(100, color="black", linestyle="--", linewidth=1, alpha=0.7)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    axes[0].set_title("Nano payoff as a percentage of GPT-5.4")
    axes[0].set_ylabel("Average Nano payoff / GPT-5.4 payoff (%)")
    axes[1].set_title("Paired coordination contrast")
    axes[1].set_ylabel("Change in Nano/GPT-5.4 ratio (percentage points)")
    for ax in axes:
        ax.set_xlabel("Total agents N (Nano team size = N−1)")
        ax.set_xticks(NS, [f"{n}\n({n-1})" for n in NS])
        ax.grid(alpha=0.22)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Game 1 weaker-agent performance in percentage terms\n"
        "Raw utilities; ratio of cell means; five competition cells weighted equally",
        fontsize=15,
    )
    path = output_dir / "recommended_percentage_of_adversary_primary.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def plot_percentage_by_competition(
    percentage_rows: list[dict[str, Any]], output_dir: Path
) -> Path:
    selected = [
        row for row in percentage_rows if row["aggregation"] == "n_by_competition"
    ]
    lookup = {
        (
            row["condition"], int(row["n_agents"]),
            float(row["competition_level"]), row["metric"],
        ): row
        for row in selected
    }
    fig, axes = plt.subplots(
        len(COMPETITIONS), 2, figsize=(13, 19), sharex=True, sharey=False,
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.08, right=0.98, bottom=0.045, top=0.925, hspace=0.12, wspace=0.16
    )
    for row_index, competition in enumerate(COMPETITIONS):
        for condition, marker in (("control", "o"), ("team", "s")):
            estimates = [
                lookup[(
                    condition, n_agents, competition,
                    "nano_payoff_percent_of_adversary",
                )]
                for n_agents in NS
            ]
            add_bootstrap_interval(
                axes[row_index, 0], estimates,
                label="Independent control" if condition == "control" else "Coordinated team",
                color=CONDITION_COLORS[condition], marker=marker,
            )
        effect = [
            lookup[(
                "paired_effect", n_agents, competition,
                "percentage_point_change_team_vs_control",
            )]
            for n_agents in NS
        ]
        add_bootstrap_interval(
            axes[row_index, 1], effect,
            label="Coordinated − independent", color="#7C3AED", marker="D",
        )
        axes[row_index, 0].axhline(
            100, color="black", linestyle="--", linewidth=1, alpha=0.7
        )
        axes[row_index, 1].axhline(
            0, color="black", linestyle="--", linewidth=1, alpha=0.7
        )
        axes[row_index, 0].set_ylabel(
            f"Competition = {competition:g}\nNano / GPT-5.4 payoff (%)"
        )
        axes[row_index, 1].set_ylabel("Percentage-point change")
        for ax in axes[row_index]:
            ax.set_xticks(NS, [f"{n}\n({n-1})" for n in NS])
            ax.grid(alpha=0.22)
            if row_index == len(COMPETITIONS) - 1:
                ax.set_xlabel("Total agents N (Nano team size = N−1)")
    axes[0, 0].set_title("Nano payoff as % of GPT-5.4")
    axes[0, 1].set_title("Paired coordination contrast")
    axes[0, 0].legend(frameon=False, fontsize=9)
    axes[0, 1].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Game 1 percentage-payoff results by competition level\n"
        "Ratio of cell means; matched bootstrap intervals; four runs per point",
        fontsize=15, y=0.985,
    )
    path = output_dir / "recommended_percentage_of_adversary_by_competition.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def report_markdown(
    ceiling_summary: list[dict[str, Any]],
    by_n_summary: list[dict[str, Any]],
    percentage_rows: list[dict[str, Any]],
    decomposition: list[dict[str, Any]],
    validations: dict[str, Any],
) -> str:
    ceiling_lookup = estimate_lookup(ceiling_summary)
    team_decomposition = {
        float(row["competition_level"]): row
        for row in decomposition if row["condition"] == "team"
    }
    by_n_lookup = {
        (row["condition"], int(row["n_agents"]), row["metric"]): row
        for row in by_n_summary
    }
    percentage_lookup = {
        (row["condition"], int(row["n_agents"]), row["metric"]): row
        for row in percentage_rows
        if row["aggregation"] == "n_equal_competition_weight"
    }
    lines = [
        "# Game 1 payoff ceilings and N-scaling",
        "",
        "## Answer",
        "",
        "The payoff decline with N is largely structural, not evidence that larger groups "
        "necessarily negotiate worse. Every agent's preference vector sums to 100, and the "
        "number of items scales as 2.5N. But an item can go to only one agent. When preferences "
        "overlap, adding agents adds competing claims without adding proportionate jointly "
        "realizable value.",
        "",
        "The intended competition level itself is not drifting materially with N: across all "
        f"{validations['pairwise_cosine_pairs']} agent pairs, mean absolute cosine-target error "
        f"is {validations['pairwise_cosine_mean_abs_error']:.4f} and the maximum is "
        f"{validations['pairwise_cosine_max_abs_error']:.4f}.",
        "",
        "At competition 1 all agents have the same preference vector. The exact total welfare "
        "ceiling is therefore 100 at every N, so its per-agent value is exactly 100/N: 50, 25, "
        "16.67, 12.5, and 10. At competition 0, preferences are disjoint, the total ceiling is "
        "100N, and every agent can simultaneously receive 100.",
        "",
        "## Recommended plots",
        "",
        "Use the following four-panel figure as the primary result. Panel A establishes the "
        "changing payoff scale; Panel B asks whether the group finds efficient allocations; "
        "Panel C answers Nano-team versus GPT-5.4; Panel D isolates the coordination treatment.",
        "",
        "![Recommended primary figure](recommended_primary_ceiling_normalized_results.png)",
        "",
        "Use the competition-stratified version as the appendix/robustness figure.",
        "",
        "![Recommended competition-stratified figure](recommended_by_competition_ceiling_normalized_results.png)",
        "",
        "### Percentage-of-adversary presentation",
        "",
        "The percentage figure reports `100 × mean Nano payoff / mean GPT-5.4 payoff`. "
        "Ratios are computed within N × competition cells before the five competition "
        "cells are equally weighted. A ratio remains undefined when GPT-5.4 has a cell mean "
        "of zero; the analysis does not replace the zero denominator.",
        "",
        "![Recommended percentage figure](recommended_percentage_of_adversary_primary.png)",
        "",
        "![Percentage figure by competition](recommended_percentage_of_adversary_by_competition.png)",
        "",
        "| N | Independent Nano/GPT-5.4 | Coordinated Nano/GPT-5.4 | Paired change |",
        "|---:|---:|---:|---:|",
    ]
    for n_agents in NS:
        control = percentage_lookup[(
            "control", n_agents, "nano_payoff_percent_of_adversary"
        )]
        team = percentage_lookup[(
            "team", n_agents, "nano_payoff_percent_of_adversary"
        )]
        effect = percentage_lookup[(
            "paired_effect", n_agents, "percentage_point_change_team_vs_control"
        )]
        lines.append(
            f"| {n_agents} | {float(control['estimate']):.1f}% | "
            f"{float(team['estimate']):.1f}% | {float(effect['estimate']):+.1f} pp |"
        )
    lines.extend([
        "",
        "## Exact ceiling",
        "",
        "For preference values v[i,j], the maximum total raw payoff is "
        "`W* = sum_j max_i v[i,j]`: give each item to its highest valuer. The maximum feasible "
        "average payoff is `W*/N`. Separately, the maximin floor is the exact largest t for "
        "which an indivisible-item allocation can give every agent at least t points.",
        "",
        "![Exact ceilings](attainable_ceiling_by_n_competition.png)",
        "",
        "| Competition | N=2 ceiling | N=4 | N=6 | N=8 | N=10 |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for competition in COMPETITIONS:
        values = [
            float(ceiling_lookup[(None, competition, n_agents, "utilitarian_per_capita_ceiling")]["mean"])
            for n_agents in NS
        ]
        lines.append(
            f"| {competition:g} | " + " | ".join(f"{value:.2f}" for value in values) + " |"
        )
    lines.extend([
        "",
        "The exact maximin values below answer the stricter question: what is the largest "
        "score that an allocation can give **every agent simultaneously**? Values are means "
        "across the four locked preference tables in each cell.",
        "",
        "| Competition | N=2 maximin | N=4 | N=6 | N=8 | N=10 |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for competition in COMPETITIONS:
        values = [
            float(ceiling_lookup[(None, competition, n_agents, "maximin_common_floor")]["mean"])
            for n_agents in NS
        ]
        lines.append(
            f"| {competition:g} | " + " | ".join(f"{value:.2f}" for value in values) + " |"
        )
    lines.extend([
        "",
        "## Observed versus attainable",
        "",
        "The plot uses raw allocation utility, undoing the separate 0.9-per-round discount. "
        "Thus the black line is the structural ceiling; the distance below it is allocation "
        "inefficiency rather than scarcity or delay.",
        "",
        "![Observed versus ceiling](payoff_vs_attainable_ceiling_by_competition.png)",
        "",
        "## Decomposition of the N=2 to N=10 decline",
        "",
        "The exact identity `mean payoff = per-capita ceiling × welfare efficiency` permits a "
        "symmetric two-factor decomposition. The table reports the coordinated arm after "
        "removing time discounting.",
        "",
        "| Competition | Observed payoff drop | Ceiling component | Efficiency component | Structural share |",
        "|---:|---:|---:|---:|---:|",
    ])
    for competition in COMPETITIONS:
        row = team_decomposition[competition]
        share = float(row["structural_share_of_drop"])
        share_text = "—" if not math.isfinite(share) else f"{share:.0%}"
        lines.append(
            f"| {competition:g} | {float(row['observed_payoff_drop_n2_to_n10']):+.2f} | "
            f"{float(row['structural_ceiling_component']):+.2f} | "
            f"{float(row['negotiation_efficiency_component']):+.2f} | {share_text} |"
        )
    lines.extend([
        "",
        "![Normalized performance](ceiling_normalized_performance.png)",
        "",
        "## Team-versus-adversary result after normalization",
        "",
        "The table averages the five equally represented competition levels at each N. A "
        "gap of +0.34 means GPT-5.4 leads the Nano average by 34% of that run's maximum "
        "feasible average payoff. A negative gap means the Nano average leads. The coordination effect compares the coordinated rerun "
        "with its exact historical match on the same normalized scale.",
        "",
        "| N | GPT-5.4 − Nano normalized gap | 95% CI | Normalized coordination effect | 95% CI |",
        "|---:|---:|---:|---:|---:|",
    ])
    for n_agents in NS:
        gap = by_n_lookup[("team", n_agents, "raw_adversary_minus_nano_ceiling_normalized_gap")]
        effect = by_n_lookup[("paired_effect", n_agents, "raw_adversary_minus_nano_ceiling_normalized_gap_change")]
        lines.append(
            f"| {n_agents} | {float(gap['mean']):+.3f} | "
            f"[{float(gap['ci95_low']):+.3f}, {float(gap['ci95_high']):+.3f}] | "
            f"{float(effect['mean']):+.3f} | "
            f"[{float(effect['ci95_low']):+.3f}, {float(effect['ci95_high']):+.3f}] |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "Raw points should not be compared across N without conditioning on competition and "
        "the run-specific ceiling. Use total-welfare efficiency for group performance and the "
        "ceiling-normalized adversary-minus-Nano gap for the team-versus-strong-agent question. "
        "The normalization does not remove the historical-control/provider confound; it removes "
        "the mathematical payoff-scale confound.",
        "",
        "## Validation and files",
        "",
        f"Validation passed: **{validations['valid']}**. Parsed {validations['configs']} locked "
        f"preference tables, {validations['conditions']} condition results, and "
        f"{validations['agents']} condition-agent outcomes. Every personal preference sum was exactly 100; "
        f"utility recomputation max error was {validations['utility_recompute_max_abs_delta']:.3g}; "
        f"maximum MILP optimality gap was {validations['maximin_max_mip_gap']:.3g}.",
        "",
        "- `run_ceiling_data.csv`: exact ceiling for all 100 preference tables.",
        "- `condition_normalized_outcomes.csv`: raw/discounted control, treatment, and paired metrics.",
        "- `agent_ceiling_outcomes.csv`: all 1,200 condition-agent outcomes and personal ceilings.",
        "- `ceiling_summary_by_n_competition.csv`: plotted structural ceiling estimates.",
        "- `normalized_summary_by_n_competition.csv`: plotted normalized outcomes.",
        "- `normalized_summary_by_n.csv`: ceiling-normalized estimates aggregated over competition.",
        "- `percentage_of_adversary_summary.csv`: percentage ratios and matched bootstrap intervals.",
        "- `n2_n10_payoff_decomposition.csv`: structural-versus-efficiency decomposition.",
        "- `validation.json`: machine-readable audit.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    results_root = args.results_root.resolve()
    output_dir = (
        args.output_dir or (results_root / "analysis" / "ceiling_normalization")
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ceiling_rows, outcome_rows, agent_rows, validations = load_all(results_root)
    ceiling_summary, outcome_summary = build_summaries(ceiling_rows, outcome_rows)
    by_n_summary = build_by_n_summary(outcome_rows)
    percentage_rows = bootstrap_percentage_summaries(outcome_rows)
    decomposition = endpoint_decomposition(ceiling_rows, outcome_rows)
    write_csv(output_dir / "run_ceiling_data.csv", ceiling_rows)
    write_csv(output_dir / "condition_normalized_outcomes.csv", outcome_rows)
    write_csv(output_dir / "agent_ceiling_outcomes.csv", agent_rows)
    write_csv(output_dir / "ceiling_summary_by_n_competition.csv", ceiling_summary)
    write_csv(output_dir / "normalized_summary_by_n_competition.csv", outcome_summary)
    write_csv(output_dir / "normalized_summary_by_n.csv", by_n_summary)
    write_csv(output_dir / "percentage_of_adversary_summary.csv", percentage_rows)
    write_csv(output_dir / "n2_n10_payoff_decomposition.csv", decomposition)
    write_json(output_dir / "validation.json", validations)
    plot_ceiling_structure(ceiling_summary, output_dir)
    plot_payoff_against_ceiling(ceiling_summary, outcome_summary, output_dir)
    plot_normalized_performance(outcome_summary, output_dir)
    plot_recommended_primary(ceiling_summary, by_n_summary, output_dir)
    plot_recommended_by_competition(outcome_summary, output_dir)
    plot_percentage_primary(percentage_rows, output_dir)
    plot_percentage_by_competition(percentage_rows, output_dir)
    (output_dir / "report.md").write_text(
        report_markdown(
            ceiling_summary, by_n_summary, percentage_rows, decomposition, validations
        ),
        encoding="utf-8",
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "valid": validations["valid"],
        "configs": validations["configs"],
        "report": str(output_dir / "report.md"),
    }, indent=2))


if __name__ == "__main__":
    main()

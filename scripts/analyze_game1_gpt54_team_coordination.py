#!/usr/bin/env python3
"""Analyze matched Game 1 GPT-5.4 control and baseline-team runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


COLORS = {
    "control": "#6B7280",
    "team": "#2563EB",
    "adversary": "#C43C39",
    "baseline": "#2563EB",
    "effect": "#7C3AED",
}

TOKEN_PRICES_PER_MILLION = {
    "gpt-5.4": {"input": 2.50, "cached_input": 0.25, "output": 15.00},
    "gpt-5-nano": {"input": 0.05, "cached_input": 0.005, "output": 0.40},
}
GPT54_LONG_CONTEXT_PRICES = {
    "input": 5.00,
    "cached_input": 0.50,
    "output": 22.50,
}
PRICING_SOURCE = "https://developers.openai.com/api/docs/pricing"
NANO_PRICING_SOURCE = "https://developers.openai.com/api/docs/models/gpt-5-nano"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def mean_sem_ci(values: Iterable[float]) -> dict[str, float | int]:
    data = np.asarray(list(values), dtype=float)
    n = int(data.size)
    mean = float(np.mean(data)) if n else math.nan
    if n < 2:
        return {"n": n, "mean": mean, "sd": math.nan, "sem": math.nan,
                "ci95_low": math.nan, "ci95_high": math.nan}
    sd = float(np.std(data, ddof=1))
    sem = sd / math.sqrt(n)
    half = float(stats.t.ppf(0.975, n - 1)) * sem
    return {"n": n, "mean": mean, "sd": sd, "sem": sem,
            "ci95_low": mean - half, "ci95_high": mean + half}


def agent_number(agent_id: str) -> int:
    return int(agent_id.rsplit("_", 1)[-1])


def observed_agent_models(result_path: Path, agent_id: str) -> set[str]:
    interactions_path = result_path.parent / "all_interactions.json"
    interactions = json.loads(interactions_path.read_text(encoding="utf-8"))
    return {
        str(row.get("model_name"))
        for row in interactions
        if row.get("agent_id") == agent_id and row.get("model_name")
    }


def historical_gpt54_route(model_names: set[str]) -> str:
    if model_names == {"gpt-5.4"}:
        return "direct_openai"
    if model_names == {"openai/gpt-5.4"}:
        return "openrouter"
    return "unexpected:" + ",".join(sorted(model_names))


def utility_metrics(
    payload: dict[str, Any],
    role_map: dict[str, str],
) -> dict[str, float | int | bool]:
    utilities = {
        agent_id: float(value)
        for agent_id, value in (payload.get("final_utilities") or {}).items()
    }
    for agent_id in role_map:
        utilities.setdefault(agent_id, 0.0)
    adversary_ids = [agent_id for agent_id, role in role_map.items() if role == "adversary"]
    baseline_ids = sorted(
        (agent_id for agent_id, role in role_map.items() if role == "baseline"),
        key=agent_number,
    )
    if len(adversary_ids) != 1 or not baseline_ids:
        raise ValueError(f"Invalid role map: {role_map}")
    adversary = utilities[adversary_ids[0]]
    baseline_values = [utilities[agent_id] for agent_id in baseline_ids]
    baseline_total = float(sum(baseline_values))
    baseline_mean = baseline_total / len(baseline_values)
    welfare = baseline_total + adversary
    return {
        "consensus": bool(payload.get("consensus_reached", False)),
        "final_round": int(payload.get("final_round") or 0),
        "adversary_payoff": adversary,
        "baseline_mean_payoff": baseline_mean,
        "baseline_total_payoff": baseline_total,
        "per_agent_gap": baseline_mean - adversary,
        "adversary_minus_nano_gap": adversary - baseline_mean,
        "baseline_welfare_share": baseline_total / welfare if welfare > 0 else math.nan,
        "baseline_per_agent_win": baseline_mean > adversary,
    }


def load_rows(results_root: Path, allow_partial: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    configs = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((results_root / "configs").glob("config_*.json"))
    ]
    if len(configs) != 100:
        raise ValueError(f"Expected 100 configs, found {len(configs)}")

    run_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    validation = {
        "configured_pairs": len(configs),
        "loaded_pairs": 0,
        "missing_treatment_results": [],
        "preference_mismatches": [],
        "control_hash_mismatches": [],
        "factor_mismatches": [],
        "treatment_model_route_mismatches": [],
        "historical_control_routes": {},
    }
    lineage_by_id: dict[int, dict[str, str]] = {}
    with (results_root / "control_lineage.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            lineage_by_id[int(row["config_id"])] = row

    for config in configs:
        config_id = int(config["config_id"])
        treatment_path = Path(config["output_dir"]) / "experiment_results.json"
        if not treatment_path.exists():
            validation["missing_treatment_results"].append(str(treatment_path))
            continue
        control_path = Path(config["control_result_path"])
        control = json.loads(control_path.read_text(encoding="utf-8"))
        treatment = json.loads(treatment_path.read_text(encoding="utf-8"))
        lineage = lineage_by_id[config_id]
        if sha256(control_path) != lineage["control_result_sha256"]:
            validation["control_hash_mismatches"].append(config_id)

        control_config = control.get("config") or {}
        treatment_config = treatment.get("config") or {}
        for field in (
            "n_agents",
            "competition_level",
            "random_seed",
            "seed_replicate",
            "adversary_position",
            "models",
            "agent_role_map",
        ):
            left = control_config.get(field, config.get(field))
            right = treatment_config.get(field, config.get(field))
            if left != right:
                validation["factor_mismatches"].append(
                    {"config_id": config_id, "field": field, "control": left, "team": right}
                )

        if control.get("agent_preferences") != treatment.get("agent_preferences"):
            validation["preference_mismatches"].append(config_id)

        common = {
            "config_id": config_id,
            "control_config_id": int(config["control_config_id"]),
            "n_agents": int(config["n_agents"]),
            "team_size": int(config["n_agents"]) - 1,
            "competition_level": float(config["competition_level"]),
            "adversary_position": str(config["adversary_position"]),
            "seed_replicate": int(config["seed_replicate"]),
            "random_seed": int(config["random_seed"]),
            "control_result_path": str(control_path),
            "treatment_result_path": str(treatment_path),
        }
        adversary_id = next(
            agent_id
            for agent_id, role in config["agent_role_map"].items()
            if role == "adversary"
        )
        control_route = historical_gpt54_route(
            observed_agent_models(control_path, adversary_id)
        )
        treatment_names = observed_agent_models(treatment_path, adversary_id)
        common["historical_control_adversary_route"] = control_route
        validation["historical_control_routes"][control_route] = (
            validation["historical_control_routes"].get(control_route, 0) + 1
        )
        if treatment_names != {"gpt-5.4"}:
            validation["treatment_model_route_mismatches"].append({
                "config_id": config_id,
                "observed_model_names": sorted(treatment_names),
            })
        control_metrics = utility_metrics(control, config["agent_role_map"])
        treatment_metrics = utility_metrics(treatment, config["agent_role_map"])
        run_rows.append({**common, "condition": "control", **control_metrics})
        run_rows.append({**common, "condition": "team", **treatment_metrics})

        pair_row: dict[str, Any] = dict(common)
        for metric in (
            "adversary_payoff",
            "baseline_mean_payoff",
            "baseline_total_payoff",
            "per_agent_gap",
            "adversary_minus_nano_gap",
            "baseline_welfare_share",
            "final_round",
        ):
            pair_row[f"control_{metric}"] = control_metrics[metric]
            pair_row[f"team_{metric}"] = treatment_metrics[metric]
            left = float(control_metrics[metric])
            right = float(treatment_metrics[metric])
            pair_row[f"delta_{metric}"] = right - left if math.isfinite(left) and math.isfinite(right) else math.nan
        pair_row["control_consensus"] = bool(control_metrics["consensus"])
        pair_row["team_consensus"] = bool(treatment_metrics["consensus"])
        pair_row["delta_consensus"] = int(treatment_metrics["consensus"]) - int(control_metrics["consensus"])
        pair_rows.append(pair_row)
        validation["loaded_pairs"] += 1

    if not allow_partial and validation["loaded_pairs"] != 100:
        raise RuntimeError(
            f"Only {validation['loaded_pairs']}/100 treatment results are complete; "
            "rerun with --allow-partial for an interim analysis"
        )
    return run_rows, pair_rows, validation


def aggregate_rows(run_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary: list[dict[str, Any]] = []
    metrics = (
        "adversary_payoff",
        "baseline_mean_payoff",
        "baseline_total_payoff",
        "per_agent_gap",
        "adversary_minus_nano_gap",
        "baseline_welfare_share",
        "final_round",
        "consensus",
        "baseline_per_agent_win",
    )
    for n_agents in sorted({int(row["n_agents"]) for row in run_rows}):
        for condition in ("control", "team"):
            selected = [row for row in run_rows if row["n_agents"] == n_agents and row["condition"] == condition]
            for metric in metrics:
                values = [float(row[metric]) for row in selected if math.isfinite(float(row[metric]))]
                summary.append({
                    "n_agents": n_agents,
                    "team_size": n_agents - 1,
                    "condition": condition,
                    "metric": metric,
                    **mean_sem_ci(values),
                })

    paired_summary: list[dict[str, Any]] = []
    delta_metrics = (
        "delta_adversary_payoff",
        "delta_baseline_mean_payoff",
        "delta_baseline_total_payoff",
        "delta_per_agent_gap",
        "delta_adversary_minus_nano_gap",
        "delta_baseline_welfare_share",
        "delta_final_round",
        "delta_consensus",
    )
    groups: list[tuple[str, list[dict[str, Any]]]] = [
        ("all", pair_rows),
        ("n_gt_2", [row for row in pair_rows if int(row["n_agents"]) > 2]),
    ]
    groups.extend(
        (str(n_agents), [row for row in pair_rows if row["n_agents"] == n_agents])
        for n_agents in sorted({int(row["n_agents"]) for row in pair_rows})
    )
    for label, selected in groups:
        for metric in delta_metrics:
            values = np.asarray(
                [float(row[metric]) for row in selected if math.isfinite(float(row[metric]))],
                dtype=float,
            )
            stats_row = mean_sem_ci(values)
            t_result = stats.ttest_1samp(values, 0.0) if len(values) >= 2 else None
            paired_summary.append({
                "n_agents": label,
                "team_size": "all" if label in {"all", "n_gt_2"} else int(label) - 1,
                "metric": metric,
                **stats_row,
                "paired_t": float(t_result.statistic) if t_result is not None else math.nan,
                "paired_p": float(t_result.pvalue) if t_result is not None else math.nan,
            })
    return summary, paired_summary


def n2_adjusted_effects(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use the singleton-team rerun as a descriptive historical-drift control."""

    reference = np.asarray(
        [float(row["delta_adversary_minus_nano_gap"]) for row in pair_rows if row["n_agents"] == 2],
        dtype=float,
    )
    rows: list[dict[str, Any]] = []
    for n_agents in sorted({int(row["n_agents"]) for row in pair_rows}):
        values = np.asarray(
            [
                float(row["delta_adversary_minus_nano_gap"])
                for row in pair_rows
                if row["n_agents"] == n_agents
            ],
            dtype=float,
        )
        if n_agents == 2:
            rows.append({
                "n_agents": 2,
                "team_size": 1,
                "n": int(values.size),
                "n2_reference_n": int(reference.size),
                "mean_raw_paired_change": float(np.mean(values)) if values.size else math.nan,
                "n2_adjusted_change": 0.0,
                "ci95_low": 0.0,
                "ci95_high": 0.0,
                "welch_t": math.nan,
                "welch_p": math.nan,
                "welch_p_holm": math.nan,
            })
            continue
        if len(values) < 2 or len(reference) < 2:
            continue
        difference = float(np.mean(values) - np.mean(reference))
        variance_a = float(np.var(values, ddof=1)) / len(values)
        variance_b = float(np.var(reference, ddof=1)) / len(reference)
        sem = math.sqrt(variance_a + variance_b)
        denominator = (
            (variance_a ** 2) / (len(values) - 1)
            + (variance_b ** 2) / (len(reference) - 1)
        )
        degrees_freedom = (
            ((variance_a + variance_b) ** 2) / denominator
            if denominator > 0
            else math.inf
        )
        half = float(stats.t.ppf(0.975, degrees_freedom)) * sem if sem > 0 else 0.0
        test = stats.ttest_ind(values, reference, equal_var=False)
        rows.append({
            "n_agents": n_agents,
            "team_size": n_agents - 1,
            "n": int(values.size),
            "n2_reference_n": int(reference.size),
            "mean_raw_paired_change": float(np.mean(values)),
            "n2_adjusted_change": difference,
            "ci95_low": difference - half,
            "ci95_high": difference + half,
            "welch_t": float(test.statistic),
            "welch_p": float(test.pvalue),
            "welch_p_holm": math.nan,
        })

    tested = [row for row in rows if row["n_agents"] != 2]
    ordered = sorted(enumerate(tested), key=lambda item: float(item[1]["welch_p"]))
    running = 0.0
    m = len(ordered)
    for rank, (_, row) in enumerate(ordered):
        adjusted = min(1.0, (m - rank) * float(row["welch_p"]))
        running = max(running, adjusted)
        row["welch_p_holm"] = running
    return rows


def route_sensitivity(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    routes = sorted({str(row["historical_control_adversary_route"]) for row in pair_rows})
    for n_agents in sorted({int(row["n_agents"]) for row in pair_rows}):
        for route in routes:
            values = [
                float(row["delta_adversary_minus_nano_gap"])
                for row in pair_rows
                if int(row["n_agents"]) == n_agents
                and row["historical_control_adversary_route"] == route
            ]
            if not values:
                continue
            estimate = mean_sem_ci(values)
            test = stats.ttest_1samp(values, 0.0) if len(values) >= 2 else None
            rows.append({
                "n_agents": n_agents,
                "team_size": n_agents - 1,
                "historical_control_adversary_route": route,
                **estimate,
                "paired_t": float(test.statistic) if test is not None else math.nan,
                "paired_p": float(test.pvalue) if test is not None else math.nan,
            })
    return rows


def _usage_rows_for_path(
    interactions_path: Path,
    *,
    scope: str,
) -> list[dict[str, Any]]:
    interactions = json.loads(interactions_path.read_text(encoding="utf-8"))
    grouped: dict[str, dict[str, Any]] = {}
    for interaction in interactions:
        usage = interaction.get("token_usage")
        model = str(interaction.get("model_name") or "")
        if not isinstance(usage, dict) or model not in TOKEN_PRICES_PER_MILLION:
            continue
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        prompt_details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details")
        has_cache_metadata = isinstance(prompt_details, dict)
        cached_tokens = (
            int(prompt_details.get("cached_tokens") or 0)
            if has_cache_metadata
            else 0
        )
        long_context = model == "gpt-5.4" and input_tokens > 272_000
        prices = GPT54_LONG_CONTEXT_PRICES if long_context else TOKEN_PRICES_PER_MILLION[model]
        upper_cost = (
            input_tokens * prices["input"] + output_tokens * prices["output"]
        ) / 1_000_000
        cache_aware_cost = (
            (input_tokens - cached_tokens) * prices["input"]
            + cached_tokens * prices["cached_input"]
            + output_tokens * prices["output"]
        ) / 1_000_000
        row = grouped.setdefault(model, {
            "scope": scope,
            "run_dir": str(interactions_path.parent),
            "model": model,
            "calls_with_usage": 0,
            "input_tokens": 0,
            "cached_input_tokens_observed": 0,
            "output_tokens": 0,
            "calls_with_cache_metadata": 0,
            "calls_without_cache_metadata": 0,
            "long_context_calls": 0,
            "cache_aware_cost_usd": 0.0,
            "uncached_upper_bound_cost_usd": 0.0,
        })
        row["calls_with_usage"] += 1
        row["input_tokens"] += input_tokens
        row["cached_input_tokens_observed"] += cached_tokens
        row["output_tokens"] += output_tokens
        row["calls_with_cache_metadata"] += int(has_cache_metadata)
        row["calls_without_cache_metadata"] += int(not has_cache_metadata)
        row["long_context_calls"] += int(long_context)
        row["cache_aware_cost_usd"] += cache_aware_cost
        row["uncached_upper_bound_cost_usd"] += upper_cost
    return list(grouped.values())


def api_usage_costs(results_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for config_path in sorted((results_root / "configs").glob("config_*.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        path = Path(config["output_dir"]) / "all_interactions.json"
        if path.exists():
            rows.extend(_usage_rows_for_path(path, scope="valid_production"))
            seen.add(path.resolve())
    for directory, scope in (
        ("failed_attempt_runs", "discarded_failed_attempt"),
        ("mismatched_preference_runs", "discarded_preference_mismatch"),
        ("cancelled_unlocked_preference_runs", "discarded_cancelled_partial"),
        ("credit_exhausted_attempt_runs", "discarded_credit_exhausted_partial"),
    ):
        for path in sorted((results_root / directory).glob("**/all_interactions.json")):
            if path.resolve() in seen:
                continue
            rows.extend(_usage_rows_for_path(path, scope=scope))
            seen.add(path.resolve())

    aggregates: dict[tuple[str, str], dict[str, Any]] = {}
    numeric_fields = (
        "calls_with_usage",
        "input_tokens",
        "cached_input_tokens_observed",
        "output_tokens",
        "calls_with_cache_metadata",
        "calls_without_cache_metadata",
        "long_context_calls",
        "cache_aware_cost_usd",
        "uncached_upper_bound_cost_usd",
    )
    for row in rows:
        key = (str(row["scope"]), str(row["model"]))
        aggregate = aggregates.setdefault(key, {
            "scope": key[0],
            "model": key[1],
            "runs": 0,
            **{field: 0 for field in numeric_fields},
            "pricing_source": PRICING_SOURCE,
        })
        aggregate["runs"] += 1
        for field in numeric_fields:
            aggregate[field] += row[field]
    return rows, list(aggregates.values())


def summary_lookup(summary: list[dict[str, Any]]) -> dict[tuple[int, str, str], dict[str, Any]]:
    return {
        (int(row["n_agents"]), str(row["condition"]), str(row["metric"])): row
        for row in summary
    }


def errorbar_series(ax: plt.Axes, x: list[int], rows: list[dict[str, Any]], label: str, color: str, marker: str) -> None:
    means = np.asarray([float(row["mean"]) for row in rows])
    low = means - np.asarray([float(row["ci95_low"]) for row in rows])
    high = np.asarray([float(row["ci95_high"]) for row in rows]) - means
    ax.errorbar(x, means, yerr=np.vstack([low, high]), label=label, color=color,
                marker=marker, linewidth=2.2, markersize=6, capsize=3)


def interim_plot_note(summary: list[dict[str, Any]]) -> str:
    lookup = summary_lookup(summary)
    counts = {
        n_agents: int(lookup[(n_agents, "team", "adversary_minus_nano_gap")]["n"])
        if (n_agents, "team", "adversary_minus_nano_gap") in lookup else 0
        for n_agents in (2, 4, 6, 8, 10)
    }
    total = sum(counts.values())
    if total == 100 and all(value == 20 for value in counts.values()):
        return ""
    rendered = ", ".join(f"N={n}: n={counts[n]}" for n in counts)
    return f"INTERIM - {total}/100 matched runs ({rendered})"


def plot_control_vs_team(summary: list[dict[str, Any]], output_dir: Path) -> Path:
    lookup = summary_lookup(summary)
    ns = sorted({key[0] for key in lookup})
    panels = (
        ("adversary_payoff", "GPT-5.4 adversary payoff", "Discounted payoff"),
        ("baseline_mean_payoff", "Average GPT-5 Nano payoff", "Discounted payoff"),
        ("adversary_minus_nano_gap", "GPT-5.4 adversary − average Nano payoff", "Adversary − Nano per-agent gap"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), constrained_layout=True)
    for ax, (metric, title, ylabel) in zip(axes, panels):
        for condition, marker in (("control", "o"), ("team", "s")):
            rows = [lookup[(n, condition, metric)] for n in ns]
            errorbar_series(
                ax, ns, rows,
                "Independent control" if condition == "control" else "Coordinated team",
                COLORS[condition], marker,
            )
        if metric == "adversary_minus_nano_gap":
            ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.7)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Total agents N (team size = N−1)")
        ax.set_ylabel(ylabel)
        ax.set_xticks((2, 4, 6, 8, 10), [f"{n}\n({n-1})" for n in (2, 4, 6, 8, 10)])
        ax.set_xlim(1.4, 10.6)
        ax.grid(alpha=0.22)
    axes[0].legend(frameon=False)
    title = "Matched Game 1 comparison: independent baselines vs coordinated baseline team"
    note = interim_plot_note(summary)
    fig.suptitle(title + (f"\n{note}" if note else ""), fontsize=14)
    path = output_dir / "control_vs_team_by_n.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def plot_team_vs_adversary(
    run_rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    lookup = summary_lookup(summary)
    ns = sorted({int(row["n_agents"]) for row in run_rows})
    team_rows = [row for row in run_rows if row["condition"] == "team"]
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.8), constrained_layout=True)

    for metric, label, color, marker in (
        ("adversary_payoff", "GPT-5.4 adversary", COLORS["adversary"], "o"),
        ("baseline_mean_payoff", "Average GPT-5 Nano teammate", COLORS["baseline"], "s"),
    ):
        rows = [lookup[(n, "team", metric)] for n in ns]
        errorbar_series(axes[0], ns, rows, label, color, marker)
    axes[0].set_title("Per-agent payoff in coordinated games")
    axes[0].set_ylabel("Mean discounted payoff (95% CI)")
    axes[0].legend(frameon=False)

    rng = np.random.default_rng(20260809)
    for n in ns:
        values = [float(row["adversary_minus_nano_gap"]) for row in team_rows if row["n_agents"] == n]
        jitter = rng.uniform(-0.10, 0.10, len(values))
        axes[1].scatter(np.asarray([n] * len(values)) + jitter, values,
                        color=COLORS["effect"], alpha=0.28, s=20, edgecolors="none")
    rows = [lookup[(n, "team", "adversary_minus_nano_gap")] for n in ns]
    errorbar_series(axes[1], ns, rows, "Mean gap", COLORS["effect"], "D")
    axes[1].axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.75)
    axes[1].set_title("Adversary gap; negative values mean Nano is ahead")
    axes[1].set_ylabel("GPT-5.4 adversary payoff − average Nano payoff")
    axes[1].legend(frameon=False)

    for ax in axes:
        ax.set_xlabel("Total agents N (teamed Nano agents = N−1)")
        ax.set_xticks((2, 4, 6, 8, 10), [f"{n}\n({n-1})" for n in (2, 4, 6, 8, 10)])
        ax.set_xlim(1.4, 10.6)
        ax.grid(alpha=0.22)
    title = "Weaker-agent team versus GPT-5.4 High as team size grows"
    note = interim_plot_note(summary)
    fig.suptitle(title + (f"\n{note}" if note else ""), fontsize=14)
    path = output_dir / "team_vs_adversary_scaling.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def plot_competition_heatmap(run_rows: list[dict[str, Any]], output_dir: Path) -> Path:
    team = [row for row in run_rows if row["condition"] == "team"]
    ns = [2, 4, 6, 8, 10]
    competitions = [0.0, 0.25, 0.5, 0.75, 1.0]
    matrix = np.full((len(competitions), len(ns)), np.nan)
    counts = np.zeros((len(competitions), len(ns)), dtype=int)
    for i, competition in enumerate(competitions):
        for j, n_agents in enumerate(ns):
            values = [
                float(row["adversary_minus_nano_gap"])
                for row in team
                if row["n_agents"] == n_agents and row["competition_level"] == competition
            ]
            if values:
                matrix[i, j] = np.mean(values)
                counts[i, j] = len(values)
    bound = max(abs(float(np.nanmin(matrix))), abs(float(np.nanmax(matrix))), 1.0)
    fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto", origin="lower")
    for i in range(len(competitions)):
        for j in range(len(ns)):
            label = f"{matrix[i, j]:+.1f}\n(n={counts[i, j]})" if counts[i, j] else "N/A"
            ax.text(j, i, label, ha="center", va="center", fontsize=8.5)
    ax.set_xticks(range(len(ns)), [f"N={n}\nteam={n-1}" for n in ns])
    ax.set_yticks(range(len(competitions)), [str(value) for value in competitions])
    ax.set_xlabel("Group and weaker-team size")
    ax.set_ylabel("Preference competition level")
    total = len(team)
    status = "" if total == 100 else f"\nINTERIM - {total}/100 matched runs; blank cells are pending"
    ax.set_title("GPT-5.4 adversary − coordinated-team average Nano payoff" + status)
    fig.colorbar(image, ax=ax, label="Adversary − average Nano payoff gap")
    path = output_dir / "team_gap_by_n_and_competition.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def plot_results_by_competition_level(
    run_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    """Plot the two headline estimands with competition level as the series."""

    ns = (2, 4, 6, 8, 10)
    competitions = (0.0, 0.25, 0.5, 0.75, 1.0)
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(competitions)))
    summary_rows: list[dict[str, Any]] = []
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)

    for competition, color in zip(competitions, colors):
        effect_estimates: list[dict[str, Any]] = []
        outcome_estimates: list[dict[str, Any]] = []
        for n_agents in ns:
            effect = mean_sem_ci(
                float(row["delta_adversary_minus_nano_gap"])
                for row in pair_rows
                if int(row["n_agents"]) == n_agents
                and float(row["competition_level"]) == competition
            )
            outcome = mean_sem_ci(
                float(row["adversary_minus_nano_gap"])
                for row in run_rows
                if row["condition"] == "team"
                and int(row["n_agents"]) == n_agents
                and float(row["competition_level"]) == competition
            )
            effect_estimates.append(effect)
            outcome_estimates.append(outcome)
            for metric, estimate in (
                ("paired_coordination_effect", effect),
                ("coordinated_team_gap", outcome),
            ):
                summary_rows.append({
                    "competition_level": competition,
                    "n_agents": n_agents,
                    "team_size": n_agents - 1,
                    "metric": metric,
                    **estimate,
                })

        label = f"Competition {competition:g}"
        errorbar_series(axes[0], list(ns), effect_estimates, label, color, "o")
        errorbar_series(axes[1], list(ns), outcome_estimates, label, color, "o")

    axes[0].set_title("Change in adversary gap; negative means coordination closes it")
    axes[0].set_ylabel("Change in adversary − average Nano payoff gap")
    axes[1].set_title("Adversary gap; negative values mean Nano is ahead")
    axes[1].set_ylabel("GPT-5.4 adversary − average Nano payoff gap")
    for ax in axes:
        ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.7)
        ax.set_xlabel("Total agents N (teamed Nano agents = N−1)")
        ax.set_xticks(ns, [f"{n}\n({n-1})" for n in ns])
        ax.grid(alpha=0.22)
    axes[1].legend(frameon=False, fontsize=9, ncol=1)
    fig.suptitle(
        "GPT-5.4 High versus coordinated GPT-5 Nano teams, by competition level\n"
        "Mean and 95% t interval; four matched runs per N × competition point",
        fontsize=15,
    )
    path = output_dir / "results_by_competition_level.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path, summary_rows


def competition_cell_estimates(
    run_rows: list[dict[str, Any]],
    *,
    competition: float,
    condition: str,
    metric: str,
) -> tuple[list[int], list[dict[str, Any]]]:
    x_values: list[int] = []
    estimates: list[dict[str, Any]] = []
    for n_agents in (2, 4, 6, 8, 10):
        values = [
            float(row[metric])
            for row in run_rows
            if row["condition"] == condition
            and int(row["n_agents"]) == n_agents
            and float(row["competition_level"]) == competition
            and math.isfinite(float(row[metric]))
        ]
        if values:
            x_values.append(n_agents)
            estimates.append(mean_sem_ci(values))
    return x_values, estimates


def competition_counts(run_rows: list[dict[str, Any]], competition: float) -> dict[int, int]:
    return {
        n_agents: sum(
            1
            for row in run_rows
            if row["condition"] == "team"
            and int(row["n_agents"]) == n_agents
            and float(row["competition_level"]) == competition
        )
        for n_agents in (2, 4, 6, 8, 10)
    }


def plot_control_vs_team_by_competition(
    run_rows: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    competitions = (0.0, 0.25, 0.5, 0.75, 1.0)
    panels = (
        ("adversary_payoff", "GPT-5.4 adversary payoff", "Discounted payoff"),
        ("baseline_mean_payoff", "Average GPT-5 Nano payoff", "Discounted payoff"),
        ("adversary_minus_nano_gap", "GPT-5.4 adversary − average Nano payoff", "Adversary − Nano per-agent gap"),
    )
    fig, axes = plt.subplots(
        len(competitions), len(panels), figsize=(16, 20), sharex=True, sharey="col",
        constrained_layout=True,
    )
    for row_index, competition in enumerate(competitions):
        counts = competition_counts(run_rows, competition)
        count_note = ", ".join(f"N{n}={counts[n]}" for n in counts)
        for column_index, (metric, title, ylabel) in enumerate(panels):
            ax = axes[row_index, column_index]
            for condition, marker in (("control", "o"), ("team", "s")):
                x_values, estimates = competition_cell_estimates(
                    run_rows,
                    competition=competition,
                    condition=condition,
                    metric=metric,
                )
                errorbar_series(
                    ax,
                    x_values,
                    estimates,
                    "Independent control" if condition == "control" else "Coordinated team",
                    COLORS[condition],
                    marker,
                )
            if metric == "adversary_minus_nano_gap":
                ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.7)
            if row_index == 0:
                ax.set_title(title, fontsize=12)
            if column_index == 0:
                ax.set_ylabel(f"Competition = {competition:g}\n{ylabel}")
            else:
                ax.set_ylabel(ylabel)
            if row_index == len(competitions) - 1:
                ax.set_xlabel("Total agents N (team size = N−1)")
            ax.set_xticks((2, 4, 6, 8, 10), [f"{n}\n({n-1})" for n in (2, 4, 6, 8, 10)])
            ax.set_xlim(1.4, 10.6)
            ax.grid(alpha=0.22)
        axes[row_index, 1].text(
            0.5, 0.98, f"cell n: {count_note}", transform=axes[row_index, 1].transAxes,
            ha="center", va="top", fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2},
        )
    axes[0, 0].legend(frameon=False)
    total = sum(1 for row in run_rows if row["condition"] == "team")
    completion_note = (
        "FINAL - 100/100 matched runs"
        if total == 100
        else f"INTERIM - {total}/100 matched runs"
    )
    fig.suptitle(
        "Control versus coordinated team, separated by preference competition\n"
        f"{completion_note}; 95% t intervals within each N × competition cell",
        fontsize=15,
    )
    path = output_dir / "control_vs_team_by_n_faceted_competition.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def plot_team_vs_adversary_by_competition(
    run_rows: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    competitions = (0.0, 0.25, 0.5, 0.75, 1.0)
    fig, axes = plt.subplots(
        len(competitions), 2, figsize=(13.5, 20), sharex=True, sharey="col",
        constrained_layout=True,
    )
    for row_index, competition in enumerate(competitions):
        counts = competition_counts(run_rows, competition)
        count_note = ", ".join(f"N{n}={counts[n]}" for n in counts)
        for metric, label, color, marker in (
            ("adversary_payoff", "GPT-5.4 adversary", COLORS["adversary"], "o"),
            ("baseline_mean_payoff", "Average GPT-5 Nano teammate", COLORS["baseline"], "s"),
        ):
            x_values, estimates = competition_cell_estimates(
                run_rows,
                competition=competition,
                condition="team",
                metric=metric,
            )
            errorbar_series(axes[row_index, 0], x_values, estimates, label, color, marker)
        gap_x, gap_estimates = competition_cell_estimates(
            run_rows,
            competition=competition,
            condition="team",
            metric="adversary_minus_nano_gap",
        )
        errorbar_series(
            axes[row_index, 1], gap_x, gap_estimates, "Mean adversary − Nano gap",
            COLORS["effect"], "D",
        )
        axes[row_index, 1].axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.75)
        axes[row_index, 0].set_ylabel(f"Competition = {competition:g}\nMean payoff (95% CI)")
        axes[row_index, 1].set_ylabel("Adversary payoff − average Nano payoff")
        axes[row_index, 1].text(
            0.5, 0.98, f"cell n: {count_note}", transform=axes[row_index, 1].transAxes,
            ha="center", va="top", fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2},
        )
        for ax in axes[row_index]:
            if row_index == len(competitions) - 1:
                ax.set_xlabel("Total agents N (teamed Nano agents = N−1)")
            ax.set_xticks((2, 4, 6, 8, 10), [f"{n}\n({n-1})" for n in (2, 4, 6, 8, 10)])
            ax.set_xlim(1.4, 10.6)
            ax.grid(alpha=0.22)
    axes[0, 0].set_title("Per-agent payoff in coordinated games")
    axes[0, 1].set_title("Adversary gap; negative values mean Nano is ahead")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].legend(frameon=False)
    total = sum(1 for row in run_rows if row["condition"] == "team")
    completion_note = (
        "FINAL - 100/100 matched runs"
        if total == 100
        else f"INTERIM - {total}/100 matched runs"
    )
    fig.suptitle(
        "Weaker-agent team versus GPT-5.4 High, separated by preference competition\n"
        f"{completion_note}; 95% t intervals within each N × competition cell",
        fontsize=15,
    )
    path = output_dir / "team_vs_adversary_scaling_faceted_competition.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def report_markdown(
    output_dir: Path,
    validation: dict[str, Any],
    summary: list[dict[str, Any]],
    paired_summary: list[dict[str, Any]],
    adjusted_effects: list[dict[str, Any]],
    route_rows: list[dict[str, Any]],
    cost_summary: list[dict[str, Any]],
) -> str:
    lookup = summary_lookup(summary)
    paired = {
        (str(row["n_agents"]), str(row["metric"])): row for row in paired_summary
    }
    lines = [
        "# GPT-5.4 High versus a coordinated GPT-5 Nano team, Game 1",
        "",
        f"Complete matched pairs: **{validation['loaded_pairs']}/100**.",
        *( ["**INTERIM ANALYSIS:** the factorial is incomplete; do not treat these results as final.", ""]
           if validation["loaded_pairs"] < 100 else [] ),
        "Every treatment run reuses the historical control's N, competition level, "
        "adversary position, random seed, and realized preference table. The treatment "
        "shares all Nano preferences and existing private reasoning within the Nano team; "
        "it adds no API calls. N=2 is a singleton-team negative control.",
        "",
        "## Design and estimand",
        "",
        "The planned 100 treatment cells form a 5 (N=2,4,6,8,10) × 5 (competition "
        "0,.25,.5,.75,1) × 2 (adversary first/last) × 2 (historical seed replicate) "
        "factorial, giving 20 runs per N. Every treatment config locks the literal "
        "historical preference table; seed replay alone is not trusted.",
        "",
        "The control baselines already communicate in the public negotiation. The "
        "treatment adds a baseline-only joint objective, complete teammate preferences, "
        "shared private thinking/voting/reflection notes, and a designated Nano captain. "
        "The GPT-5.4 adversary sees none of this. All agents retain their own proposals "
        "and votes and make the same number of API calls.",
        "",
        "The primary outcome is **GPT-5.4 payoff minus average Nano payoff** within a "
        "run. Team-total utility is secondary because it increases mechanically with "
        "N−1. Error bars are 95% t intervals across all 20 matched cells at each N; "
        "the faceted plots show the four matched cells in every N × competition stratum.",
        "Positive gap values mean GPT-5.4 is ahead. Negative gap values mean the average Nano agent is ahead.",
        "",
        "## Requested plots",
        "",
        "![Control versus coordinated team](control_vs_team_by_n.png)",
        "",
        "![Team versus adversary scaling](team_vs_adversary_scaling.png)",
        "",
        "![Competition heatmap](team_gap_by_n_and_competition.png)",
        "",
        "![Headline results by competition level](results_by_competition_level.png)",
        "",
        "![Control versus team by competition](control_vs_team_by_n_faceted_competition.png)",
        "",
        "![Team versus adversary by competition](team_vs_adversary_scaling_faceted_competition.png)",
        "",
        "## Direct answer at the largest team size",
        "",
    ]
    if (10, "team", "adversary_minus_nano_gap") in lookup:
        n10_gap = lookup[(10, "team", "adversary_minus_nano_gap")]
        n10_win = lookup[(10, "team", "baseline_per_agent_win")]
        if float(n10_gap["ci95_low"]) > 0:
            verdict = "the GPT-5.4 adversary retains a positive per-agent advantage"
        elif float(n10_gap["ci95_high"]) < 0:
            verdict = "the Nano team has a positive per-agent advantage"
        else:
            verdict = "the per-agent advantage is not resolved at the 95% level"
        lines.extend([
            f"At N=10 (nine teamed Nano agents), the mean adversary-minus-Nano gap is "
            f"**{float(n10_gap['mean']):+.2f}** (95% CI "
            f"[{float(n10_gap['ci95_low']):+.2f}, {float(n10_gap['ci95_high']):+.2f}]); "
            f"the Nano mean exceeds the adversary in **{float(n10_win['mean']):.0%}** "
            f"of cells. On this estimand, {verdict}.",
        ])
    else:
        lines.append("N=10 has no completed treatment cells yet, so no largest-team conclusion is available.")
    lines.extend([
        "",
        "## Mean outcomes by N",
        "",
        "| N | Team size | Control adversary−Nano gap | Team adversary−Nano gap | Paired change in gap | Team adversary | Team average Nano | Team win rate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for n_agents in (2, 4, 6, 8, 10):
        if (n_agents, "team", "adversary_minus_nano_gap") not in lookup:
            continue
        lines.append(
            "| {n} | {team_size} | {control_gap:+.2f} | {team_gap:+.2f} | "
            "{delta:+.2f} | {adversary:.2f} | {baseline:.2f} | {win_rate:.0%} |".format(
                n=n_agents,
                team_size=n_agents - 1,
                control_gap=lookup[(n_agents, "control", "adversary_minus_nano_gap")]["mean"],
                team_gap=lookup[(n_agents, "team", "adversary_minus_nano_gap")]["mean"],
                delta=paired[(str(n_agents), "delta_adversary_minus_nano_gap")]["mean"],
                adversary=lookup[(n_agents, "team", "adversary_payoff")]["mean"],
                baseline=lookup[(n_agents, "team", "baseline_mean_payoff")]["mean"],
                win_rate=lookup[(n_agents, "team", "baseline_per_agent_win")]["mean"],
            )
        )
    overall = paired[("n_gt_2", "delta_adversary_minus_nano_gap")]
    lines.extend([
        "",
        "## Paired headline",
        "",
        f"Across all {int(overall['n'])} matched N>2 cells, the coordinated rerun changes the adversary-minus-Nano "
        f"per-agent payoff gap by **{overall['mean']:+.2f}** "
        f"(95% CI [{overall['ci95_low']:+.2f}, {overall['ci95_high']:+.2f}], "
        f"paired t-test p={overall['paired_p']:.4g}).",
        "A negative change means coordination reduced the GPT-5.4 advantage.",
        "",
        "Interpret team aggregate payoff only within a fixed N: it rises mechanically "
        "with the number of Nano agents. The per-agent gap is the primary answer to "
        "whether teaming helps weaker agents compete with the stronger adversary.",
        "",
        "## N=2-adjusted coordination effect",
        "",
        "Because a one-Nano 'team' receives no coordination treatment, N=2 is a "
        "descriptive control for rerun/provider/time drift. The values below subtract "
        "the mean N=2 paired change from each N>2 paired change. This is useful but not "
        "a perfect causal correction because N=2 and larger games differ structurally.",
        "",
        "| N | Team size | Raw paired change | N=2-adjusted change | 95% CI | Holm p |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for row in adjusted_effects:
        if int(row["n_agents"]) == 2:
            continue
        lines.append(
            f"| {int(row['n_agents'])} | {int(row['team_size'])} | "
            f"{float(row['mean_raw_paired_change']):+.2f} | "
            f"{float(row['n2_adjusted_change']):+.2f} | "
            f"[{float(row['ci95_low']):+.2f}, {float(row['ci95_high']):+.2f}] | "
            f"{float(row['welch_p_holm']):.4g} |"
        )
    lines.extend([
        "",
        "## Historical provider-route sensitivity",
        "",
        f"Historical GPT-5.4 controls used **{validation['historical_control_routes'].get('direct_openai', 0)} direct OpenAI** "
        f"and **{validation['historical_control_routes'].get('openrouter', 0)} OpenRouter** runs; "
        "all new treatment runs use native OpenAI. Thus provider route is confounded "
        "with treatment in the OpenRouter-matched subset. The table preserves that provenance.",
        "",
        "| N | Historical GPT-5.4 route | Pairs | Mean paired gap change | 95% CI |",
        "|---:|:---|---:|---:|---:|",
    ])
    for row in route_rows:
        lines.append(
            f"| {int(row['n_agents'])} | {row['historical_control_adversary_route']} | "
            f"{int(row['n'])} | {float(row['mean']):+.2f} | "
            f"[{float(row['ci95_low']):+.2f}, {float(row['ci95_high']):+.2f}] |"
        )
    production_cost = sum(
        float(row["uncached_upper_bound_cost_usd"])
        for row in cost_summary
        if row["scope"] == "valid_production"
    )
    discarded_cost = sum(
        float(row["uncached_upper_bound_cost_usd"])
        for row in cost_summary
        if row["scope"] != "valid_production"
    )
    lines.extend([
        "",
        "## API usage and cost",
        "",
        f"At current standard token rates, valid production calls cost at most "
        f"**${production_cost:.2f}** if no prompt-cache discount is assumed. "
        f"Discarded diagnostic/mismatched/partial attempts add at most **${discarded_cost:.2f}**. "
        "This is a token-log estimate, not an account invoice; calls lacking cached-token "
        "metadata are conservatively priced as uncached.",
        "",
        f"Pricing: [official OpenAI pricing]({PRICING_SOURCE}) and "
        f"[GPT-5 Nano model pricing]({NANO_PRICING_SOURCE}).",
        "",
        "## Data products",
        "",
        f"- `run_level_outcomes.csv`: {2 * int(validation['loaded_pairs'])} condition rows.",
        f"- `paired_effects.csv`: {int(validation['loaded_pairs'])} one-to-one treatment effects.",
        "- `summary_by_n_condition.csv`: plotted condition means and intervals.",
        "- `paired_summary.csv`: paired effects, intervals, and tests.",
        "- `n2_adjusted_effects.csv`: drift-adjusted N>2 contrasts with Holm correction.",
        "- `historical_route_sensitivity.csv`: effects split by the control's actual GPT-5.4 route.",
        "- `competition_level_summary.csv`: both headline estimands for every N × competition cell.",
        "- `api_usage_by_run.csv` and `api_cost_summary.csv`: recorded tokens and cost estimates.",
        "- `validation.json`: completeness and pairing checks.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    results_root = args.results_root.resolve()
    output_dir = (args.output_dir or (results_root / "analysis")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_rows, pair_rows, validation = load_rows(results_root, args.allow_partial)
    summary, paired_summary = aggregate_rows(run_rows, pair_rows)
    adjusted_effects = n2_adjusted_effects(pair_rows)
    route_rows = route_sensitivity(pair_rows)
    usage_rows, cost_summary = api_usage_costs(results_root)
    validation["valid_complete_pairing"] = (
        validation["loaded_pairs"] == 100
        and not validation["missing_treatment_results"]
        and not validation["preference_mismatches"]
        and not validation["control_hash_mismatches"]
        and not validation["factor_mismatches"]
        and not validation["treatment_model_route_mismatches"]
    )
    write_csv(output_dir / "run_level_outcomes.csv", run_rows)
    write_csv(output_dir / "paired_effects.csv", pair_rows)
    write_csv(output_dir / "summary_by_n_condition.csv", summary)
    write_csv(output_dir / "paired_summary.csv", paired_summary)
    write_csv(output_dir / "n2_adjusted_effects.csv", adjusted_effects)
    write_csv(output_dir / "historical_route_sensitivity.csv", route_rows)
    write_csv(output_dir / "api_usage_by_run.csv", usage_rows)
    write_csv(output_dir / "api_cost_summary.csv", cost_summary)
    write_json(output_dir / "validation.json", validation)
    plot_control_vs_team(summary, output_dir)
    plot_team_vs_adversary(run_rows, summary, output_dir)
    plot_competition_heatmap(run_rows, output_dir)
    _, competition_summary = plot_results_by_competition_level(
        run_rows, pair_rows, output_dir
    )
    write_csv(output_dir / "competition_level_summary.csv", competition_summary)
    plot_control_vs_team_by_competition(run_rows, output_dir)
    plot_team_vs_adversary_by_competition(run_rows, output_dir)
    (output_dir / "report.md").write_text(
        report_markdown(
            output_dir,
            validation,
            summary,
            paired_summary,
            adjusted_effects,
            route_rows,
            cost_summary,
        ),
        encoding="utf-8",
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "loaded_pairs": validation["loaded_pairs"],
        "valid_complete_pairing": validation["valid_complete_pairing"],
        "report": str(output_dir / "report.md"),
    }, indent=2))


if __name__ == "__main__":
    main()

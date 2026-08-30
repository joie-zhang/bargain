#!/usr/bin/env python3
"""Analyze the paired context-compaction pilot.

The unit of randomization is a matched environment seed within N.  Each pair
contains one run with context compaction enabled and one run with it disabled.
This script deliberately treats a completed no-consensus game as a valid
zero-utility outcome.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


PRIMARY_METRICS = [
    "valid_completion",
    "context_failure",
    "consensus",
    "final_round",
    "mean_utility",
    "mean_accept_share",
    "mean_budget_utilization",
    "mean_proposal_alignment",
    "mean_project_concentration",
    "mean_targeted_projects",
    "mean_output_tokens",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--additional-results-root",
        type=Path,
        action="append",
        default=[],
        help="Additional compatible batch root; may be repeated.",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def mean_or_nan(values: Iterable[float]) -> float:
    clean = [float(x) for x in values if not pd.isna(x)]
    return float(np.mean(clean)) if clean else math.nan


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(text[start : end + 1])
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def pairwise_cosine(vectors: list[np.ndarray]) -> float:
    similarities: list[float] = []
    for left, right in itertools.combinations(vectors, 2):
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        if left_norm == 0 or right_norm == 0:
            continue
        similarities.append(float(np.dot(left, right) / (left_norm * right_norm)))
    return mean_or_nan(similarities)


def proposal_metrics(
    interactions: list[dict[str, Any]], result: dict[str, Any]
) -> dict[str, float]:
    votes_by_round: dict[int, list[bool]] = defaultdict(list)
    proposals_by_round: dict[int, dict[str, list[float]]] = {}

    for interaction in interactions:
        phase = str(interaction.get("phase", ""))
        if not phase.startswith("voting_round_"):
            continue
        round_number = int(interaction.get("round") or 0)
        parsed = extract_json_object(str(interaction.get("response", "")))
        if not parsed:
            continue
        decision = str(parsed.get("vote_decision", "")).strip().lower()
        if decision in {"accept", "reject"}:
            votes_by_round[round_number].append(decision == "accept")
        details = parsed.get("proposal_details")
        if isinstance(details, dict):
            contributions = details.get("contributions_by_agent")
            if isinstance(contributions, dict) and round_number not in proposals_by_round:
                clean: dict[str, list[float]] = {}
                for agent, vector in contributions.items():
                    if isinstance(vector, list):
                        clean[str(agent)] = [safe_float(x) for x in vector]
                if clean:
                    proposals_by_round[round_number] = clean

    accept_shares = [np.mean(values) for values in votes_by_round.values() if values]
    budgets = result.get("config", {}).get("agent_budgets", {})
    costs = [
        safe_float(item.get("cost"))
        for item in result.get("config", {}).get("items", [])
        if isinstance(item, dict)
    ]
    budget_use: list[float] = []
    alignment: list[float] = []
    concentration: list[float] = []
    targeted: list[float] = []
    funded: list[float] = []

    for contributions in proposals_by_round.values():
        vectors = [np.asarray(vector, dtype=float) for vector in contributions.values()]
        if not vectors or len({len(vector) for vector in vectors}) != 1:
            continue
        for agent, vector in contributions.items():
            budget = safe_float(budgets.get(agent))
            if budget > 0:
                budget_use.append(float(np.nansum(vector)) / budget)
        alignment.append(pairwise_cosine(vectors))
        totals = np.nansum(np.vstack(vectors), axis=0)
        total_contribution = float(np.nansum(totals))
        concentration.append(
            float(np.sum((totals / total_contribution) ** 2))
            if total_contribution > 0
            else math.nan
        )
        targeted.append(float(np.sum(totals > 1e-9)))
        if len(costs) == len(totals):
            funded.append(float(np.sum(totals + 1e-9 >= np.asarray(costs))))

    return {
        "mean_accept_share": mean_or_nan(accept_shares),
        "mean_budget_utilization": mean_or_nan(budget_use),
        "mean_proposal_alignment": mean_or_nan(alignment),
        "mean_project_concentration": mean_or_nan(concentration),
        "mean_targeted_projects": mean_or_nan(targeted),
        "mean_funded_projects": mean_or_nan(funded),
        "rounds_with_parsed_votes": float(len(votes_by_round)),
        "rounds_with_parsed_proposals": float(len(proposals_by_round)),
    }


def read_log_tail(path: Path, size: int = 1_000_000) -> str:
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - size))
        return handle.read().decode(errors="replace")


def classify_failed_attempt(status: dict[str, Any]) -> tuple[str, str]:
    texts: list[str] = []
    for attempt in status.get("attempts", []):
        log_path = attempt.get("log_path")
        if log_path and Path(log_path).exists():
            texts.append(read_log_tail(Path(log_path)))
    combined = "\n".join(texts).lower()
    if any(
        marker in combined
        for marker in (
            "context_length_exceeded",
            "maximum context length",
            "context window",
            "requested too many tokens",
        )
    ):
        return "context_length", "scientific"
    if any(
        marker in combined
        for marker in (
            "rate limit",
            "insufficient_quota",
            "connection error",
            "connection reset",
            "timed out",
            "timeout",
            "service unavailable",
            "internal server error",
        )
    ):
        return "api_or_network", "retryable"
    if status.get("state") == "FAILED":
        return "unknown_failure", "retryable"
    return "", ""


def extract_failure_diagnostics(status: dict[str, Any]) -> dict[str, Any]:
    texts = []
    for attempt in status.get("attempts", []):
        log_path = attempt.get("log_path")
        if log_path and Path(log_path).exists():
            texts.append(read_log_tail(Path(log_path)))
    combined = "\n".join(texts)
    requested = re.findall(
        r"requested\s+(\d+)\s+tokens\s+\((\d+)\s+in the messages,\s+"
        r"(\d+)\s+in the completion\)",
        combined,
        flags=re.IGNORECASE,
    )
    if not requested:
        requested = re.findall(
            r"requested\s+(?:about\s+)?(\d+)\s+tokens\s+\("
            r"(\d+)\s+of text input,\s+(\d+)\s+in the output\)",
            combined,
            flags=re.IGNORECASE,
        )
    progress = re.findall(
        r"PROGRESS interaction=\d+ round=(\d+) phase=([^\s]+) "
        r"agent=([^\s]+).*?provider_input_tokens=(\d+|None)",
        combined,
    )
    row: dict[str, Any] = {
        "failure_requested_tokens": math.nan,
        "failure_message_tokens": math.nan,
        "failure_completion_allowance_tokens": math.nan,
        "last_progress_round": math.nan,
        "last_progress_phase": "",
        "last_progress_agent": "",
        "last_progress_provider_input_tokens": math.nan,
    }
    if requested:
        total, messages, completion = requested[-1]
        row.update(
            {
                "failure_requested_tokens": int(total),
                "failure_message_tokens": int(messages),
                "failure_completion_allowance_tokens": int(completion),
            }
        )
    if progress:
        round_number, phase, agent, provider_tokens = progress[-1]
        row.update(
            {
                "last_progress_round": int(round_number),
                "last_progress_phase": phase,
                "last_progress_agent": agent,
                "last_progress_provider_input_tokens": (
                    int(provider_tokens) if provider_tokens != "None" else math.nan
                ),
            }
        )
    return row


def analyze_run(config_path: Path) -> dict[str, Any]:
    config = read_json(config_path)
    root = config_path.parent.parent
    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = root.parent.parent.parent / output_dir
        if not output_dir.exists():
            output_dir = Path.cwd() / config["output_dir"]
    result_path = output_dir / "experiment_results.json"
    interactions_path = output_dir / "all_interactions.json"
    status_path = root / "status" / f"config_{int(config['config_id']):04d}.json"

    row: dict[str, Any] = {
        "config_id": int(config["config_id"]),
        "batch": root.name,
        "pair_id": config["pair_id"],
        "n_agents": int(config["n_agents"]),
        "seed_replicate": int(config["seed_replicate"]),
        "random_seed": int(config["random_seed"]),
        "arm": config["treatment_arm"],
        "compaction_enabled": bool(config["compaction_enabled"]),
        "result_path": str(result_path),
        "status_state": "NOT_STARTED",
        "attempt_count": 0,
        "completed": False,
        "valid_completion": math.nan,
        "context_failure": math.nan,
        "failure_class": "",
        "failure_disposition": "",
        "terminal": False,
    }
    if status_path.exists():
        status = read_json(status_path)
        row["status_state"] = status.get("state", "UNKNOWN")
        row["attempt_count"] = len(status.get("attempts", []))
        row["duration_seconds"] = safe_float(status.get("duration_seconds"))
        failure_class, disposition = classify_failed_attempt(status)
        row["failure_class"] = failure_class
        row["failure_disposition"] = disposition
        if failure_class:
            row.update(extract_failure_diagnostics(status))
        if disposition == "scientific":
            row["terminal"] = True
            row["valid_completion"] = 0
            row["context_failure"] = int(failure_class == "context_length")
    if not result_path.exists() or not interactions_path.exists():
        return row

    result = read_json(result_path)
    interactions = read_json(interactions_path)
    if not isinstance(interactions, list):
        return row
    row["completed"] = True
    row["terminal"] = True
    row["valid_completion"] = 1
    row["context_failure"] = 0
    row["status_state"] = "SUCCESS"
    row["consensus"] = int(bool(result.get("consensus_reached")))
    row["final_round"] = safe_float(result.get("final_round"))
    utilities = [safe_float(x) for x in result.get("final_utilities", {}).values()]
    row["total_utility"] = float(np.nansum(utilities)) if utilities else math.nan
    row["mean_utility"] = mean_or_nan(utilities)
    row["interaction_count"] = len(interactions)

    compacted = [x for x in interactions if x.get("context_compacted") is True]
    row["compaction_activated"] = int(bool(compacted))
    row["compaction_calls"] = len(compacted)
    row["first_compaction_round"] = (
        min(safe_float(x.get("round")) for x in compacted) if compacted else math.nan
    )
    row["first_compaction_phase"] = (
        min(compacted, key=lambda x: safe_float(x.get("timestamp")))["phase"]
        if compacted
        else ""
    )
    before = [safe_float(x.get("estimated_input_tokens_before")) for x in compacted]
    after = [safe_float(x.get("estimated_input_tokens_after")) for x in compacted]
    row["max_precompaction_estimated_tokens"] = (
        float(np.nanmax(before)) if compacted else math.nan
    )
    row["mean_tokens_removed_per_compaction"] = (
        mean_or_nan(np.asarray(before) - np.asarray(after)) if compacted else math.nan
    )

    provider_inputs: list[float] = []
    estimated_inputs: list[float] = []
    output_tokens: list[float] = []
    for interaction in interactions:
        usage = interaction.get("token_usage") or {}
        provider_inputs.append(
            safe_float(
                interaction.get(
                    "provider_input_tokens", usage.get("provider_input_tokens")
                )
            )
        )
        estimated_inputs.append(
            safe_float(
                interaction.get(
                    "estimated_provider_input_tokens",
                    usage.get("estimated_provider_input_tokens"),
                )
            )
        )
        output_tokens.append(safe_float(usage.get("output_tokens")))
    row["max_provider_input_tokens"] = float(np.nanmax(provider_inputs))
    row["max_estimated_input_tokens"] = (
        float(np.nanmax(estimated_inputs))
        if not np.all(np.isnan(estimated_inputs))
        else math.nan
    )
    row["mean_output_tokens"] = mean_or_nan(output_tokens)
    row["total_output_tokens"] = float(np.nansum(output_tokens))
    row.update(proposal_metrics(interactions, result))
    return row


def exact_signflip_pvalue(differences: pd.Series) -> float:
    values = differences.dropna().to_numpy(dtype=float)
    values = values[np.abs(values) > 1e-12]
    n = len(values)
    if n == 0:
        return 1.0
    observed = abs(float(np.mean(values)))
    if n <= 20:
        null = []
        for signs in itertools.product((-1.0, 1.0), repeat=n):
            null.append(abs(float(np.mean(values * np.asarray(signs)))))
        return float(np.mean(np.asarray(null) >= observed - 1e-12))
    rng = np.random.default_rng(20260726)
    signs = rng.choice((-1.0, 1.0), size=(100_000, n))
    null = np.abs(np.mean(signs * values, axis=1))
    return float((np.sum(null >= observed - 1e-12) + 1) / (len(null) + 1))


def paired_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for n_agents in sorted(pairs["n_agents"].unique()):
        subset = pairs[pairs["n_agents"] == n_agents]
        for metric in PRIMARY_METRICS:
            column = f"delta_{metric}"
            values = subset[column].dropna().astype(float)
            n = len(values)
            mean = float(values.mean()) if n else math.nan
            sd = float(values.std(ddof=1)) if n > 1 else math.nan
            sem = sd / math.sqrt(n) if n > 1 else math.nan
            critical = float(stats.t.ppf(0.975, n - 1)) if n > 1 else math.nan
            records.append(
                {
                    "n_agents": int(n_agents),
                    "metric": metric,
                    "n_pairs": n,
                    "mean_on_minus_off": mean,
                    "sd_difference": sd,
                    "ci95_low": mean - critical * sem if n > 1 else math.nan,
                    "ci95_high": mean + critical * sem if n > 1 else math.nan,
                    "exact_signflip_p": exact_signflip_pvalue(values),
                }
            )
    return pd.DataFrame(records)


def make_pairs(runs: pd.DataFrame) -> pd.DataFrame:
    complete = runs[runs["terminal"]].copy()
    value_columns = [
        column
        for column in complete.columns
        if column
        not in {
            "config_id",
            "batch",
            "arm",
            "compaction_enabled",
            "result_path",
            "status_state",
            "completed",
            "terminal",
            "failure_class",
            "failure_disposition",
        }
    ]
    on = complete[complete["arm"] == "on"].set_index("pair_id")
    off = complete[complete["arm"] == "off"].set_index("pair_id")
    shared = sorted(set(on.index) & set(off.index))
    records: list[dict[str, Any]] = []
    for pair_id in shared:
        left = on.loc[pair_id]
        right = off.loc[pair_id]
        record: dict[str, Any] = {
            "pair_id": pair_id,
            "n_agents": int(left["n_agents"]),
            "seed_replicate": int(left["seed_replicate"]),
            "random_seed": int(left["random_seed"]),
        }
        for column in value_columns:
            if column in {"pair_id", "n_agents", "seed_replicate", "random_seed"}:
                continue
            left_value = left.get(column)
            right_value = right.get(column)
            record[f"on_{column}"] = left_value
            record[f"off_{column}"] = right_value
            if isinstance(left_value, (int, float, np.number)) and isinstance(
                right_value, (int, float, np.number)
            ):
                record[f"delta_{column}"] = left_value - right_value
        records.append(record)
    return pd.DataFrame(records)


def by_cell_summary(runs: pd.DataFrame) -> pd.DataFrame:
    complete = runs[runs["terminal"]].copy()
    metrics = [
        "valid_completion",
        "context_failure",
        "consensus",
        "final_round",
        "mean_utility",
        "compaction_activated",
        "compaction_calls",
        "max_provider_input_tokens",
        "max_estimated_input_tokens",
        "mean_accept_share",
        "mean_budget_utilization",
        "mean_proposal_alignment",
        "mean_project_concentration",
        "mean_targeted_projects",
        "mean_output_tokens",
    ]
    rows: list[dict[str, Any]] = []
    for (n_agents, arm), group in complete.groupby(["n_agents", "arm"]):
        row: dict[str, Any] = {
            "n_agents": int(n_agents),
            "arm": arm,
            "n_runs": len(group),
            "n_completed_games": int(group["completed"].sum()),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = (
                float(group[metric].std(ddof=1)) if len(group) > 1 else math.nan
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["n_agents", "arm"])


def save_plots(runs: pd.DataFrame, pairs: pd.DataFrame, output_dir: Path) -> None:
    terminal = runs[runs["terminal"]]
    completion = (
        terminal.groupby(["n_agents", "arm"])["valid_completion"]
        .mean()
        .unstack("arm")
        .reindex(columns=["off", "on"])
    )
    positions = np.arange(len(completion))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar(
        positions - width / 2,
        completion["off"],
        width,
        color="#c95c54",
        label="Compaction off",
    )
    ax.bar(
        positions + width / 2,
        completion["on"],
        width,
        color="#3366a8",
        label="Compaction on",
    )
    ax.set(
        ylim=(0, 1.08),
        xlabel="Number of agents (N)",
        ylabel="Valid completion rate",
        xticks=positions,
        xticklabels=[str(int(n)) for n in completion.index],
    )
    ax.set_title("Valid completion by treatment arm")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "completion_by_n_and_arm.pdf")
    fig.savefig(output_dir / "completion_by_n_and_arm.png", dpi=180)
    plt.close(fig)

    complete = runs[runs["completed"]]
    activation = (
        complete[complete["arm"] == "on"]
        .groupby("n_agents")["compaction_activated"]
        .agg(["mean", "count"])
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar(activation["n_agents"].astype(str), activation["mean"], color="#3366a8")
    ax.set(ylim=(0, 1), xlabel="Number of agents (N)", ylabel="Runs with compaction")
    ax.set_title("Compaction activation rate")
    for i, row in activation.iterrows():
        ax.text(i, row["mean"] + 0.025, f"{int(row['mean'] * row['count'])}/{int(row['count'])}", ha="center")
    fig.tight_layout()
    fig.savefig(output_dir / "compaction_activation_by_n.pdf")
    fig.savefig(output_dir / "compaction_activation_by_n.png", dpi=180)
    plt.close(fig)

    completed_arm_counts = (
        runs[runs["completed"]]
        .groupby("pair_id")["arm"]
        .nunique()
    )
    completed_pair_ids = set(completed_arm_counts[completed_arm_counts == 2].index)
    payoff_pairs = pairs[pairs["pair_id"].isin(completed_pair_ids)].copy()
    n_values = sorted(runs["n_agents"].unique())
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(11.5, 7.5),
        sharey=True,
        constrained_layout=True,
    )
    axes_flat = list(axes.flat)
    for ax, n_agents in zip(axes_flat, n_values):
        subset = payoff_pairs[payoff_pairs["n_agents"] == n_agents]
        for _, row in subset.iterrows():
            ax.plot(
                [0, 1],
                [row["off_mean_utility"], row["on_mean_utility"]],
                color="#969696",
                alpha=0.7,
                linewidth=1.2,
                zorder=1,
            )
            ax.scatter(
                [0, 1],
                [row["off_mean_utility"], row["on_mean_utility"]],
                color=["#c95c54", "#3366a8"],
                s=28,
                zorder=2,
            )
        if not subset.empty:
            arm_means = [
                subset["off_mean_utility"].mean(),
                subset["on_mean_utility"].mean(),
            ]
            ax.plot(
                [0, 1],
                arm_means,
                color="#111111",
                linewidth=2.4,
                marker="D",
                markersize=5,
                zorder=3,
            )
            mean_delta = float(subset["delta_mean_utility"].mean())
            ax.text(
                0.5,
                0.97,
                f"pairs={len(subset)}; mean Δ={mean_delta:+.2f}",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=8,
            )
        ax.set(
            title=f"N={int(n_agents)}",
            xticks=[0, 1],
            xticklabels=["Without", "With"],
            xlim=(-0.25, 1.25),
        )
        ax.grid(axis="y", alpha=0.2)
    for ax in axes_flat[len(n_values) :]:
        ax.set_visible(False)
    axes[0, 0].set_ylabel("Mean per-agent payoff")
    axes[1, 0].set_ylabel("Mean per-agent payoff")
    fig.suptitle(
        "Paired payoff comparison by N\n"
        "(only pairs with completed games in both arms)"
    )
    fig.supxlabel("Context compaction setting")
    fig.savefig(output_dir / "paired_payoff_by_n.pdf")
    fig.savefig(output_dir / "paired_payoff_by_n.png", dpi=180)
    plt.close(fig)

    metrics = [
        ("mean_utility", "Mean utility"),
        ("consensus", "Consensus (0/1)"),
        ("final_round", "Final round"),
        ("mean_accept_share", "Mean accept-vote share"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True)
    for ax, (metric, label) in zip(axes.flat, metrics):
        for _, row in pairs.iterrows():
            n = row["n_agents"]
            jitter = (int(row["seed_replicate"]) - 3) * 0.035
            ax.plot(
                [n - 0.12 + jitter, n + 0.12 + jitter],
                [row[f"off_{metric}"], row[f"on_{metric}"]],
                color="#8c8c8c",
                alpha=0.75,
                linewidth=1,
            )
            ax.scatter(n - 0.12 + jitter, row[f"off_{metric}"], color="#c95c54", s=18)
            ax.scatter(n + 0.12 + jitter, row[f"on_{metric}"], color="#3366a8", s=18)
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("Number of agents (N)")
        ax.set_xticks(sorted(pairs["n_agents"].unique()))
    axes[0, 0].scatter([], [], color="#c95c54", label="Compaction off")
    axes[0, 0].scatter([], [], color="#3366a8", label="Compaction on")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Matched pilot outcomes (lines join common environment seeds)")
    fig.tight_layout()
    fig.savefig(output_dir / "paired_outcomes.pdf")
    fig.savefig(output_dir / "paired_outcomes.png", dpi=180)
    plt.close(fig)


def markdown_report(
    runs: pd.DataFrame,
    pairs: pd.DataFrame,
    cell: pd.DataFrame,
    paired: pd.DataFrame,
    output_dir: Path,
) -> None:
    complete = runs[runs["completed"]]
    terminal = runs[runs["terminal"]]
    unresolved = runs[~runs["terminal"]]
    lines = [
        "# Context-compaction pilot report",
        "",
        "## Design and validity",
        "",
        f"- Planned runs: {len(runs)}; terminal outcomes: {len(terminal)}.",
        f"- Completed games: {len(complete)}; scientific context failures: {int(terminal['context_failure'].sum())}; unresolved/retryable: {len(unresolved)}.",
        f"- Matched pairs with two terminal outcomes: {len(pairs)}.",
        "- Game: co-funding (Game 3); model: GPT-4o mini; alpha = 0.2; sigma = 0.2.",
        f"- Each N has {int(runs.groupby('n_agents')['pair_id'].nunique().min())} "
        "shared environment seeds and two treatment arms.",
        "- A completed final-round no-consensus game is retained as a valid outcome with zero utility.",
        "- A context-limit error is a scientific endpoint, reported as a failed completion rather than rerun or assigned zero game utility.",
        "- The seed fixes the game environment, not model sampling; paired arms are matched environments, not identical language-model trajectories.",
        "",
        "## Compaction activation",
        "",
        "| N | enabled runs | activated | rate | first activation rounds | calls |",
        "|---:|---:|---:|---:|---|---:|",
    ]
    enabled = complete[complete["arm"] == "on"]
    for n_agents, group in enabled.groupby("n_agents"):
        active = group[group["compaction_activated"] == 1]
        rounds = ", ".join(
            str(int(x)) for x in active["first_compaction_round"].dropna().sort_values()
        )
        lines.append(
            f"| {int(n_agents)} | {len(group)} | {len(active)} | "
            f"{len(active) / len(group):.0%} | {rounds or '—'} | "
            f"{int(active['compaction_calls'].sum())} |"
        )

    failures = terminal[terminal["context_failure"] == 1]
    lines += [
        "",
        "## Scientific context failures",
        "",
        "| batch | config | pair | N | arm | last round | last phase | requested tokens | message tokens | completion allowance |",
        "|:---|---:|:---|---:|:---:|---:|:---|---:|---:|---:|",
    ]
    if failures.empty:
        lines.append("| — | — | — | — | — | — | — | — | — | — |")
    else:
        for _, row in failures.iterrows():
            last_round = (
                str(int(row.last_progress_round))
                if not pd.isna(row.last_progress_round)
                else "—"
            )
            requested = (
                str(int(row.failure_requested_tokens))
                if not pd.isna(row.failure_requested_tokens)
                else "—"
            )
            messages = (
                str(int(row.failure_message_tokens))
                if not pd.isna(row.failure_message_tokens)
                else "—"
            )
            allowance = (
                str(int(row.failure_completion_allowance_tokens))
                if not pd.isna(row.failure_completion_allowance_tokens)
                else "—"
            )
            lines.append(
                f"| {row.batch} | {int(row.config_id)} | {row.pair_id} | "
                f"{int(row.n_agents)} | "
                f"{row.arm} | {last_round} | {row.last_progress_phase or '—'} | "
                f"{requested} | {messages} | {allowance} |"
            )

    lines += [
        "",
        "## Outcome means by cell",
        "",
        "| N | arm | terminal | completed | context failure | consensus | final round | mean utility | accept share | budget use | proposal alignment |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in cell.iterrows():
        lines.append(
            f"| {int(row.n_agents)} | {row.arm} | {int(row.n_runs)} | "
            f"{int(row.n_completed_games)} | {row.context_failure_mean:.3f} | "
            f"{row.consensus_mean:.3f} | {row.final_round_mean:.2f} | "
            f"{row.mean_utility_mean:.3f} | {row.mean_accept_share_mean:.3f} | "
            f"{row.mean_budget_utilization_mean:.3f} | "
            f"{row.mean_proposal_alignment_mean:.3f} |"
        )

    lines += [
        "",
        "## Paired estimates (on minus off)",
        "",
        "| N | metric | pairs | mean difference | 95% t interval | exact sign-flip p |",
        "|---:|:---|---:|---:|:---:|---:|",
    ]
    for _, row in paired.iterrows():
        lines.append(
            f"| {int(row.n_agents)} | {row.metric} | {int(row.n_pairs)} | "
            f"{row.mean_on_minus_off:.4g} | "
            f"[{row.ci95_low:.4g}, {row.ci95_high:.4g}] | "
            f"{row.exact_signflip_p:.4g} |"
        )

    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- With only five pairs per N, this pilot is primarily an activation, feasibility, and variance study.",
        "- The smallest attainable two-sided exact sign-flip p-value with five nonzero paired differences is 0.0625. Therefore, no within-N five-pair pilot can cross a 0.05 threshold using that test.",
        "- Absence of a significant difference here is not evidence of equivalence. Any full design should pre-specify a practically meaningful equivalence margin and size the active-N cells around it.",
        "- Low-N cells where compaction never activates are implementation/placebo checks, not estimates of the behavioral effect of summarization.",
        "- Utility and behavioral metrics are undefined after a context failure; completion and context-failure rates retain those scientific endpoints without imputing utility.",
        "",
        "## Files",
        "",
        "- `runs.csv`: one row per planned arm.",
        "- `failures.csv`: scientific failure endpoints and extracted diagnostics.",
        "- `pairs.csv`: one row per matched pair with two terminal scientific outcomes.",
        "- `cell_summary.csv`: descriptive statistics by N and arm.",
        "- `paired_summary.csv`: paired estimates, intervals, and exact sign-flip tests.",
        "- `paired_outcomes.pdf`: matched outcome plot.",
        "- `paired_payoff_by_n.pdf`: primary paired-payoff plot, excluding pairs with a context-terminated arm.",
        "- `compaction_activation_by_n.pdf`: activation boundary plot.",
        "- `completion_by_n_and_arm.pdf`: valid completion rates by N and arm.",
    ]
    (output_dir / "pilot_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    results_roots = [results_root] + [
        path.resolve() for path in args.additional_results_root
    ]
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else results_root / "analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    config_paths = [
        config_path
        for root in results_roots
        for config_path in sorted((root / "configs").glob("config_*.json"))
    ]
    runs = pd.DataFrame(analyze_run(path) for path in config_paths).sort_values("config_id")
    pairs = make_pairs(runs)
    cell = by_cell_summary(runs)
    paired = paired_summary(pairs) if not pairs.empty else pd.DataFrame()
    runs.to_csv(output_dir / "runs.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    pairs.to_csv(output_dir / "pairs.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    cell.to_csv(output_dir / "cell_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_summary.csv", index=False)
    runs[runs["failure_class"].astype(str) != ""].to_csv(
        output_dir / "failures.csv", index=False
    )
    if not pairs.empty:
        save_plots(runs, pairs, output_dir)
    markdown_report(runs, pairs, cell, paired, output_dir)
    print(
        json.dumps(
            {
                "planned_runs": len(runs),
                "completed_runs": int(runs["completed"].sum()),
                "terminal_runs": int(runs["terminal"].sum()),
                "terminal_pairs": len(pairs),
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Quantify mechanisms behind the Game 1 GPT-5.4 per-agent advantage."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(
    "/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/"
    "game1_gpt54_team_coordination_20260809_055844"
)
OUTPUT_DIR = ROOT / "analysis/scaling_mechanism_audit/quantitative"
NS = (2, 4, 6, 8, 10)
CONDITIONS = ("control", "team")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_response(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def raw_utility(
    allocation: dict[str, list[int]], preferences: dict[str, list[float]], agent: str
) -> float:
    values = preferences[agent]
    return float(sum(values[int(item)] for item in allocation.get(agent, [])))


def welfare_ceiling(preferences: dict[str, list[float]]) -> float:
    item_count = len(next(iter(preferences.values())))
    return float(
        sum(max(float(values[item]) for values in preferences.values()) for item in range(item_count))
    )


def team_ceiling(preferences: dict[str, list[float]], baselines: list[str]) -> float:
    item_count = len(next(iter(preferences.values())))
    return float(
        sum(max(float(preferences[agent][item]) for agent in baselines) for item in range(item_count))
    )


def phase_family(phase: str) -> str:
    for family in ("setup", "discussion", "private_thinking", "proposal", "voting", "reflection"):
        if phase.startswith(family):
            return family
    return phase.split("_round_", 1)[0]


def collect_votes(
    interactions: list[dict[str, Any]], baselines: list[str], adversary: str
) -> tuple[dict[tuple[int, int], dict[str, Any]], list[dict[str, Any]]]:
    by_proposal: dict[tuple[int, int], dict[str, Any]] = defaultdict(
        lambda: {"accept_total": 0, "accept_baseline": 0, "adversary_accept": False}
    )
    raw_votes: list[dict[str, Any]] = []
    for entry in interactions:
        if not str(entry.get("phase", "")).startswith("voting_round_"):
            continue
        payload = parse_response(entry.get("response"))
        if not payload:
            continue
        try:
            key = (int(payload.get("round", entry.get("round", 0))), int(payload["proposal_number"]))
        except (KeyError, TypeError, ValueError):
            continue
        voter = str(payload.get("voter", entry.get("agent_id", "")))
        accepted = payload.get("vote_decision") == "accept"
        if accepted:
            by_proposal[key]["accept_total"] += 1
            if voter in baselines:
                by_proposal[key]["accept_baseline"] += 1
            if voter == adversary:
                by_proposal[key]["adversary_accept"] = True
        raw_votes.append(
            {
                "round": key[0],
                "proposal_number": key[1],
                "voter": voter,
                "voter_role": "adversary" if voter == adversary else "baseline",
                "accepted": accepted,
            }
        )
    return by_proposal, raw_votes


def collect_proposals(result: dict[str, Any]) -> list[dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    for entry in result.get("conversation_logs", []):
        if entry.get("phase") != "proposal_enumeration":
            continue
        round_num = int(entry.get("round", 0))
        for proposal in entry.get("enumerated_proposals", []):
            allocation = proposal.get("allocation") or proposal.get("original_proposal", {}).get("allocation")
            if not isinstance(allocation, dict):
                continue
            proposals.append(
                {
                    "round": round_num,
                    "proposal_number": int(proposal["proposal_number"]),
                    "proposer": str(proposal["proposer"]),
                    "allocation": allocation,
                }
            )
    return proposals


def winning_proposal_number(result: dict[str, Any]) -> int | None:
    final_round = int(result.get("final_round") or 0)
    for entry in result.get("conversation_logs", []):
        if entry.get("phase") != "vote_tabulation" or int(entry.get("round", 0)) != final_round:
            continue
        match = re.search(r"Proposal #(\d+) accepted by two-thirds", str(entry.get("content", "")))
        if match:
            return int(match.group(1))
    return None


def analyze_condition(
    *,
    config: dict[str, Any],
    condition: str,
    result_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    result = load_json(result_path)
    interactions = load_json(result_path.parent / "all_interactions.json")
    preferences = {
        agent: [float(value) for value in values]
        for agent, values in result["agent_preferences"].items()
    }
    role_map = config["agent_role_map"]
    adversary = next(agent for agent, role in role_map.items() if role == "adversary")
    baselines = [agent for agent, role in role_map.items() if role == "baseline"]
    n_agents = int(config["n_agents"])
    threshold = math.ceil(2 * n_agents / 3)
    ceiling = welfare_ceiling(preferences)
    per_capita = ceiling / n_agents
    baseline_ceiling = team_ceiling(preferences, baselines)
    proposals = collect_proposals(result)
    votes_by_proposal, raw_votes = collect_votes(interactions, baselines, adversary)
    winner_number = winning_proposal_number(result)
    final_round = int(result.get("final_round") or 0)

    proposal_rows: list[dict[str, Any]] = []
    proposal_lookup: dict[tuple[int, int], dict[str, Any]] = {}
    for proposal in proposals:
        key = (proposal["round"], proposal["proposal_number"])
        allocation = proposal["allocation"]
        adversary_payoff = raw_utility(allocation, preferences, adversary)
        baseline_payoffs = [raw_utility(allocation, preferences, agent) for agent in baselines]
        baseline_mean = float(np.mean(baseline_payoffs))
        total = adversary_payoff + float(sum(baseline_payoffs))
        vote_info = votes_by_proposal.get(
            key, {"accept_total": 0, "accept_baseline": 0, "adversary_accept": False}
        )
        row = {
            "config_id": int(config["config_id"]),
            "condition": condition,
            "n_agents": n_agents,
            "competition_level": float(config["competition_level"]),
            "adversary_position": str(config["adversary_position"]),
            "seed_replicate": int(config["seed_replicate"]),
            "threshold": threshold,
            "round": proposal["round"],
            "proposal_number": proposal["proposal_number"],
            "proposer": proposal["proposer"],
            "proposer_role": "adversary" if proposal["proposer"] == adversary else "baseline",
            "adversary_raw_payoff": adversary_payoff,
            "baseline_mean_raw_payoff": baseline_mean,
            "adversary_normalized_payoff": adversary_payoff / per_capita,
            "baseline_mean_normalized_payoff": baseline_mean / per_capita,
            "normalized_adversary_gap": (adversary_payoff - baseline_mean) / per_capita,
            "total_welfare_efficiency": total / ceiling,
            "baseline_team_efficiency": float(sum(baseline_payoffs)) / baseline_ceiling,
            "accept_total": int(vote_info["accept_total"]),
            "accept_baseline": int(vote_info["accept_baseline"]),
            "adversary_accept": bool(vote_info["adversary_accept"]),
            "passed": int(vote_info["accept_total"]) >= threshold,
            "passed_without_adversary": int(vote_info["accept_baseline"]) >= threshold,
            "winner": proposal["round"] == final_round and proposal["proposal_number"] == winner_number,
        }
        proposal_rows.append(row)
        proposal_lookup[key] = row

    vote_rows: list[dict[str, Any]] = []
    for vote in raw_votes:
        proposal = proposal_lookup.get((vote["round"], vote["proposal_number"]))
        if not proposal:
            continue
        allocation = next(
            item["allocation"]
            for item in proposals
            if item["round"] == vote["round"]
            and item["proposal_number"] == vote["proposal_number"]
        )
        voter_utility = raw_utility(allocation, preferences, vote["voter"])
        vote_rows.append(
            {
                "config_id": int(config["config_id"]),
                "condition": condition,
                "n_agents": n_agents,
                "competition_level": float(config["competition_level"]),
                "adversary_position": str(config["adversary_position"]),
                **vote,
                "proposal_role": proposal["proposer_role"],
                "voter_raw_utility": voter_utility,
                "proposal_baseline_mean_raw": proposal["baseline_mean_raw_payoff"],
                "proposal_total_efficiency": proposal["total_welfare_efficiency"],
                "proposal_normalized_adversary_gap": proposal["normalized_adversary_gap"],
            }
        )

    prompt_rows: list[dict[str, Any]] = []
    for entry in interactions:
        agent = str(entry.get("agent_id", ""))
        if agent not in role_map:
            continue
        prompt_rows.append(
            {
                "config_id": int(config["config_id"]),
                "condition": condition,
                "n_agents": n_agents,
                "competition_level": float(config["competition_level"]),
                "agent_role": "adversary" if agent == adversary else "baseline",
                "phase_family": phase_family(str(entry.get("phase", ""))),
                "provider_input_tokens": float(entry.get("provider_input_tokens") or 0),
                "reasoning_tokens": float(entry.get("reasoning_tokens") or 0),
                "prompt_chars": float(entry.get("prompt_chars") or 0),
                "context_compacted": bool(entry.get("context_compacted", False)),
            }
        )

    winner = next((row for row in proposal_rows if row["winner"]), None)
    adv_proposals = [row for row in proposal_rows if row["proposer_role"] == "adversary"]
    baseline_proposals = [row for row in proposal_rows if row["proposer_role"] == "baseline"]
    first_adv = next((row for row in adv_proposals if row["round"] == 1), None)
    first_baseline = [row for row in baseline_proposals if row["round"] == 1]
    allocation = result.get("final_allocation") or {}
    final_adversary = raw_utility(allocation, preferences, adversary)
    final_baseline_values = [raw_utility(allocation, preferences, agent) for agent in baselines]
    final_baseline_mean = float(np.mean(final_baseline_values))

    run_row = {
        "config_id": int(config["config_id"]),
        "condition": condition,
        "n_agents": n_agents,
        "team_size": n_agents - 1,
        "competition_level": float(config["competition_level"]),
        "adversary_position": str(config["adversary_position"]),
        "seed_replicate": int(config["seed_replicate"]),
        "threshold": threshold,
        "final_round": final_round,
        "per_capita_ceiling": per_capita,
        "final_adversary_raw": final_adversary,
        "final_baseline_mean_raw": final_baseline_mean,
        "final_adversary_normalized": final_adversary / per_capita,
        "final_baseline_mean_normalized": final_baseline_mean / per_capita,
        "final_normalized_adversary_gap": (final_adversary - final_baseline_mean) / per_capita,
        "adversary_authored_winner": bool(winner and winner["proposer_role"] == "adversary"),
        "winner_passed_without_adversary": bool(winner and winner["passed_without_adversary"]),
        "winner_baseline_accept_fraction": (
            float(winner["accept_baseline"]) / len(baselines) if winner else math.nan
        ),
        "adversary_proposal_passed": any(row["passed"] for row in adv_proposals),
        "adversary_proposal_passed_without_adversary": any(
            row["passed_without_adversary"] for row in adv_proposals
        ),
        "baseline_proposal_passed": any(row["passed"] for row in baseline_proposals),
        "baseline_proposal_passed_without_adversary": any(
            row["passed_without_adversary"] for row in baseline_proposals
        ),
        "first_round_adversary_proposal_baseline_norm": (
            float(first_adv["baseline_mean_normalized_payoff"]) if first_adv else math.nan
        ),
        "first_round_best_baseline_proposal_baseline_norm": (
            max(float(row["baseline_mean_normalized_payoff"]) for row in first_baseline)
            if first_baseline else math.nan
        ),
        "first_round_adversary_minus_best_baseline_proposal_quality": (
            float(first_adv["baseline_mean_normalized_payoff"])
            - max(float(row["baseline_mean_normalized_payoff"]) for row in first_baseline)
            if first_adv and first_baseline else math.nan
        ),
        "adversary_proposal_is_best_submitted_for_baseline": any(
            float(row["baseline_mean_raw_payoff"])
            >= max(
                float(other["baseline_mean_raw_payoff"])
                for other in proposal_rows
                if other["round"] == row["round"]
            ) - 1e-9
            for row in adv_proposals
        ),
        "adversary_proposal_is_best_submitted_total_welfare": any(
            float(row["total_welfare_efficiency"])
            >= max(
                float(other["total_welfare_efficiency"])
                for other in proposal_rows
                if other["round"] == row["round"]
            ) - 1e-9
            for row in adv_proposals
        ),
        "proposal_count": len(proposal_rows),
        "vote_count": len(vote_rows),
    }
    return run_row, proposal_rows, vote_rows, prompt_rows


def summarize(run_frame: pd.DataFrame, vote_frame: pd.DataFrame, prompt_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for n_agents in NS:
            group = run_frame[
                (run_frame["condition"] == condition) & (run_frame["n_agents"] == n_agents)
            ]
            baseline_votes = vote_frame[
                (vote_frame["condition"] == condition)
                & (vote_frame["n_agents"] == n_agents)
                & (vote_frame["voter_role"] == "baseline")
            ]
            accepted = baseline_votes[baseline_votes["accepted"]]
            rejected = baseline_votes[~baseline_votes["accepted"]]
            prompt_group = prompt_frame[
                (prompt_frame["condition"] == condition)
                & (prompt_frame["n_agents"] == n_agents)
            ]
            rows.append(
                {
                    "condition": condition,
                    "n_agents": n_agents,
                    "runs": len(group),
                    "normalized_adversary_gap_mean": group["final_normalized_adversary_gap"].mean(),
                    "adversary_normalized_mean": group["final_adversary_normalized"].mean(),
                    "baseline_normalized_mean": group["final_baseline_mean_normalized"].mean(),
                    "adversary_authored_winner_rate": group["adversary_authored_winner"].mean(),
                    "adversary_proposal_passed_rate": group["adversary_proposal_passed"].mean(),
                    "adversary_proposal_passed_without_adversary_rate": group[
                        "adversary_proposal_passed_without_adversary"
                    ].mean(),
                    "baseline_proposal_passed_without_adversary_rate": group[
                        "baseline_proposal_passed_without_adversary"
                    ].mean(),
                    "adversary_best_for_baseline_rate": group[
                        "adversary_proposal_is_best_submitted_for_baseline"
                    ].mean(),
                    "adversary_best_total_welfare_rate": group[
                        "adversary_proposal_is_best_submitted_total_welfare"
                    ].mean(),
                    "first_round_adversary_minus_best_baseline_quality": group[
                        "first_round_adversary_minus_best_baseline_proposal_quality"
                    ].mean(),
                    "winner_baseline_accept_fraction": group[
                        "winner_baseline_accept_fraction"
                    ].mean(),
                    "accepted_minus_rejected_personal_utility": (
                        accepted["voter_raw_utility"].mean() - rejected["voter_raw_utility"].mean()
                    ),
                    "accepted_minus_rejected_baseline_mean": (
                        accepted["proposal_baseline_mean_raw"].mean()
                        - rejected["proposal_baseline_mean_raw"].mean()
                    ),
                    "accepted_minus_rejected_total_efficiency": (
                        accepted["proposal_total_efficiency"].mean()
                        - rejected["proposal_total_efficiency"].mean()
                    ),
                    "baseline_prompt_input_tokens_mean": prompt_group.loc[
                        prompt_group["agent_role"] == "baseline", "provider_input_tokens"
                    ].mean(),
                    "adversary_prompt_input_tokens_mean": prompt_group.loc[
                        prompt_group["agent_role"] == "adversary", "provider_input_tokens"
                    ].mean(),
                    "baseline_reasoning_tokens_mean": prompt_group.loc[
                        prompt_group["agent_role"] == "baseline", "reasoning_tokens"
                    ].mean(),
                    "adversary_reasoning_tokens_mean": prompt_group.loc[
                        prompt_group["agent_role"] == "adversary", "reasoning_tokens"
                    ].mean(),
                    "context_compaction_rate": prompt_group["context_compacted"].mean(),
                }
            )
    return pd.DataFrame(rows)


def plot_mechanisms(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9), constrained_layout=True)
    colors = {"team": "#2563EB", "control": "#6B7280"}
    labels = {"team": "Coordinated team", "control": "Homogeneous control"}
    panels = (
        ("adversary_authored_winner_rate", "GPT-5.4-authored winner", "Runs (%)"),
        (
            "adversary_proposal_passed_without_adversary_rate",
            "GPT-5.4 proposal passes on Nano votes alone",
            "Runs (%)",
        ),
        (
            "baseline_proposal_passed_without_adversary_rate",
            "A Nano proposal passes on Nano votes alone",
            "Runs (%)",
        ),
        (
            "first_round_adversary_minus_best_baseline_quality",
            "GPT proposal quality minus best Nano proposal",
            "Normalized Nano payoff difference",
        ),
    )
    for axis, (metric, title, ylabel) in zip(axes.ravel(), panels):
        for condition in CONDITIONS:
            group = summary[summary["condition"] == condition]
            values = group[metric].to_numpy(dtype=float)
            if ylabel == "Runs (%)":
                values = 100 * values
            axis.plot(
                group["n_agents"], values, marker="o", linewidth=2.1,
                color=colors[condition], label=labels[condition],
            )
        if metric == "first_round_adversary_minus_best_baseline_quality":
            axis.axhline(0, color="black", linewidth=1, linestyle="--")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("Total agents N")
        axis.set_xticks(NS)
        axis.grid(axis="y", alpha=0.22)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Observed proposal and voting mechanisms by group size", fontsize=15)
    fig.savefig(OUTPUT_DIR / "mechanisms_by_n.png", dpi=220)
    fig.savefig(OUTPUT_DIR / "mechanisms_by_n.pdf")
    plt.close(fig)


def trend_regressions(run_frame: pd.DataFrame) -> pd.DataFrame:
    """Post-hoc descriptive slopes with competition, position, and seed controls."""

    frame = run_frame.copy()
    frame["log2_n"] = np.log2(frame["n_agents"])
    rows: list[dict[str, Any]] = []

    def fit(source: pd.DataFrame, label: str, subset: str, outcome: str) -> None:
        selected = source if subset == "all" else source[source["n_agents"] >= 4]
        model = smf.ols(
            f"{outcome} ~ log2_n + C(competition_level) + "
            "C(adversary_position) + C(seed_replicate)",
            selected,
        ).fit(cov_type="HC3")
        low, high = model.conf_int().loc["log2_n"]
        rows.append(
            {
                "estimand": label,
                "subset": subset,
                "runs": len(selected),
                "slope_per_doubling_n": float(model.params["log2_n"]),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "p_value": float(model.pvalues["log2_n"]),
                "covariance": "HC3",
                "status": "post_hoc_descriptive",
            }
        )

    for condition in CONDITIONS:
        source = frame[frame["condition"] == condition]
        for subset in ("all", "n_ge_4"):
            fit(source, condition, subset, "final_normalized_adversary_gap")

    team = frame[frame["condition"] == "team"].set_index("config_id")
    control = frame[frame["condition"] == "control"].set_index("config_id")
    paired = team.copy()
    paired["paired_gap_change"] = (
        team["final_normalized_adversary_gap"]
        - control["final_normalized_adversary_gap"]
    )
    for subset in ("all", "n_ge_4"):
        fit(paired.reset_index(), "team_minus_control", subset, "paired_gap_change")
    return pd.DataFrame(rows)


def centered_vote_correlations(vote_frame: pd.DataFrame) -> pd.DataFrame:
    """Correlate voting with proposal attributes after voter-run centering."""

    baseline = vote_frame[vote_frame["voter_role"] == "baseline"].copy()
    baseline["accepted_numeric"] = baseline["accepted"].astype(int)
    predictors = (
        "voter_raw_utility",
        "proposal_baseline_mean_raw",
        "proposal_total_efficiency",
        "proposal_normalized_adversary_gap",
    )
    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for n_agents in NS:
            group = baseline[
                (baseline["condition"] == condition)
                & (baseline["n_agents"] == n_agents)
            ].copy()
            group["accepted_centered"] = group["accepted_numeric"] - group.groupby(
                ["config_id", "voter"]
            )["accepted_numeric"].transform("mean")
            for predictor in predictors:
                centered = group[predictor] - group.groupby(
                    ["config_id", "voter"]
                )[predictor].transform("mean")
                rows.append(
                    {
                        "condition": condition,
                        "n_agents": n_agents,
                        "votes": len(group),
                        "predictor": predictor,
                        "within_voter_run_correlation": float(
                            np.corrcoef(group["accepted_centered"], centered)[0, 1]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    run_rows: list[dict[str, Any]] = []
    proposal_rows: list[dict[str, Any]] = []
    vote_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    configs = sorted((ROOT / "configs").glob("config_*.json"))
    if len(configs) != 100:
        raise RuntimeError(f"Expected 100 configs, found {len(configs)}")
    for config_path in configs:
        config = load_json(config_path)
        paths = {
            "team": Path(config["output_dir"]) / "experiment_results.json",
            "control": Path(config["control_result_path"]),
        }
        for condition, result_path in paths.items():
            run, proposals, votes, prompts = analyze_condition(
                config=config, condition=condition, result_path=result_path
            )
            run_rows.append(run)
            proposal_rows.extend(proposals)
            vote_rows.extend(votes)
            prompt_rows.extend(prompts)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_frame = pd.DataFrame(run_rows)
    proposal_frame = pd.DataFrame(proposal_rows)
    vote_frame = pd.DataFrame(vote_rows)
    prompt_frame = pd.DataFrame(prompt_rows)
    summary = summarize(run_frame, vote_frame, prompt_frame)
    trends = trend_regressions(run_frame)
    vote_correlations = centered_vote_correlations(vote_frame)
    run_frame.to_csv(OUTPUT_DIR / "run_mechanisms.csv", index=False)
    proposal_frame.to_csv(OUTPUT_DIR / "proposal_mechanisms.csv", index=False)
    vote_frame.to_csv(OUTPUT_DIR / "vote_mechanisms.csv", index=False)
    prompt_frame.to_csv(OUTPUT_DIR / "prompt_load.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "summary_by_n_condition.csv", index=False)
    trends.to_csv(OUTPUT_DIR / "trend_regressions.csv", index=False)
    vote_correlations.to_csv(OUTPUT_DIR / "centered_vote_correlations.csv", index=False)
    plot_mechanisms(summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

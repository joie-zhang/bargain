#!/usr/bin/env python3
"""Audit GPT-5.4 outcomes and coalition mechanics in the 100 team runs."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path("/scratch/gpfs/DANQIC/jz4391/bargain")
RESULT_ROOT = (
    PROJECT_ROOT
    / "experiments/results/game1_gpt54_team_coordination_20260809_055844"
)
OUTPUT_DIR = RESULT_ROOT / "analysis/team_behavior_audit"
N_ORDER = [2, 4, 6, 8, 10]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_allocation(allocation: dict[str, list[int]]) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return tuple(
        sorted((agent_id, tuple(sorted(int(item) for item in items))) for agent_id, items in allocation.items())
    )


def raw_utility(
    allocation: dict[str, list[int]],
    preferences: dict[str, list[float]],
    agent_id: str,
) -> float:
    values = preferences[agent_id]
    return float(sum(values[int(item)] for item in allocation.get(agent_id, [])))


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


def proposal_and_vote_audit(
    interactions: list[dict[str, Any]],
    preferences: dict[str, list[float]],
    adversary_id: str,
    baseline_ids: list[str],
    captain_id: str,
    threshold: int,
    final_allocation: dict[str, list[int]],
    final_round: int,
) -> dict[str, Any]:
    proposals: dict[tuple[int, int], dict[str, Any]] = {}
    nano_accepts: dict[tuple[int, int], set[str]] = defaultdict(set)
    adversary_accepts: dict[tuple[int, int], bool] = {}

    for entry in interactions:
        phase = str(entry.get("phase", ""))
        if not phase.startswith("voting_round_"):
            continue
        payload = parse_response(entry.get("response"))
        if not payload:
            continue
        try:
            key = (int(payload.get("round", entry.get("round", 0))), int(payload["proposal_number"]))
        except (KeyError, TypeError, ValueError):
            continue
        details = payload.get("proposal_details") or {}
        allocation = details.get("allocation")
        proposer = payload.get("proposal_by")
        if isinstance(allocation, dict) and proposer:
            proposals.setdefault(
                key,
                {
                    "round": key[0],
                    "proposal_number": key[1],
                    "proposer": str(proposer),
                    "allocation": allocation,
                },
            )
        voter = str(payload.get("voter", entry.get("agent_id", "")))
        accepted = payload.get("vote_decision") == "accept"
        if voter in baseline_ids and accepted:
            nano_accepts[key].add(voter)
        if voter == adversary_id:
            adversary_accepts[key] = accepted

    proposal_rows: list[dict[str, Any]] = []
    final_norm = normalized_allocation(final_allocation)
    winning_matches: list[dict[str, Any]] = []
    for key, proposal in sorted(proposals.items()):
        allocation = proposal["allocation"]
        proposer = proposal["proposer"]
        adversary_items = list(allocation.get(adversary_id, []))
        adversary_raw = raw_utility(allocation, preferences, adversary_id)
        accept_count = len(nano_accepts.get(key, set()))
        row = {
            "round": key[0],
            "proposal_number": key[1],
            "proposer": proposer,
            "proposer_is_nano": proposer in baseline_ids,
            "proposer_is_captain": proposer == captain_id,
            "adversary_item_count": len(adversary_items),
            "adversary_raw_utility": adversary_raw,
            "nano_accept_count": accept_count,
            "nano_threshold_met": accept_count >= threshold,
            "all_nanos_accept": accept_count == len(baseline_ids),
            "adversary_accept": bool(adversary_accepts.get(key, False)),
            "matches_final_allocation": (
                key[0] == final_round and normalized_allocation(allocation) == final_norm
            ),
        }
        proposal_rows.append(row)
        if row["matches_final_allocation"]:
            winning_matches.append(row)

    nano = [row for row in proposal_rows if row["proposer_is_nano"]]
    captain = [row for row in proposal_rows if row["proposer_is_captain"]]
    winning_proposers = sorted({row["proposer"] for row in winning_matches})
    winning_zero_item = any(row["adversary_item_count"] == 0 for row in winning_matches)
    winning_zero_utility = any(row["adversary_raw_utility"] == 0 for row in winning_matches)

    return {
        "valid_proposals_seen": len(proposal_rows),
        "nano_proposals_seen": len(nano),
        "captain_proposals_seen": len(captain),
        "nano_zero_item_proposals": sum(row["adversary_item_count"] == 0 for row in nano),
        "nano_zero_utility_proposals": sum(row["adversary_raw_utility"] == 0 for row in nano),
        "captain_zero_item_proposals": sum(row["adversary_item_count"] == 0 for row in captain),
        "captain_zero_utility_proposals": sum(row["adversary_raw_utility"] == 0 for row in captain),
        "any_nano_zero_item_proposal": any(row["adversary_item_count"] == 0 for row in nano),
        "any_nano_zero_utility_proposal": any(row["adversary_raw_utility"] == 0 for row in nano),
        "any_captain_zero_item_proposal": any(row["adversary_item_count"] == 0 for row in captain),
        "any_captain_zero_utility_proposal": any(row["adversary_raw_utility"] == 0 for row in captain),
        "any_nano_zero_item_proposal_passable_without_adversary": any(
            row["adversary_item_count"] == 0 and row["nano_threshold_met"] for row in nano
        ),
        "any_nano_zero_utility_proposal_passable_without_adversary": any(
            row["adversary_raw_utility"] == 0 and row["nano_threshold_met"] for row in nano
        ),
        "any_captain_proposal_passable_without_adversary": any(
            row["nano_threshold_met"] for row in captain
        ),
        "winning_proposer_ids": "|".join(winning_proposers),
        "winning_proposer_is_nano": any(proposer in baseline_ids for proposer in winning_proposers),
        "winning_proposer_is_captain": captain_id in winning_proposers,
        "winning_zero_item": winning_zero_item,
        "winning_zero_utility": winning_zero_utility,
    }


def load_run_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    run_dirs = sorted(path for path in (RESULT_ROOT / "runs").glob("config_*_") if path.is_dir())
    if not run_dirs:
        run_dirs = sorted(path for path in (RESULT_ROOT / "runs").glob("config_*") if path.is_dir())
    for run_dir in run_dirs:
        result_path = run_dir / "experiment_results.json"
        interactions_path = run_dir / "all_interactions.json"
        result = load_json(result_path)
        interactions = load_json(interactions_path)
        config = result["config"]
        config_id = int(config["config_id"])
        n_agents = int(config["n_agents"])
        preferences = result["agent_preferences"]
        role_map = config["agent_role_map"]
        adversary_id = next(agent_id for agent_id, role in role_map.items() if role == "adversary")
        baseline_ids = [agent_id for agent_id, role in role_map.items() if role == "baseline"]
        captain_id = config["team_coordination"]["captain_id"]
        threshold = math.ceil(2 * n_agents / 3)
        final_allocation = result.get("final_allocation") or {}
        team_raw = raw_utility(final_allocation, preferences, adversary_id)
        team_discounted = float(result.get("final_utilities", {}).get(adversary_id, 0.0))
        team_items = list(final_allocation.get(adversary_id, []))
        team_baseline_raw_sum = float(
            sum(raw_utility(final_allocation, preferences, agent_id) for agent_id in baseline_ids)
        )
        n_items = len(next(iter(preferences.values())))
        team_baseline_max_sum = float(
            sum(max(float(preferences[agent_id][item]) for agent_id in baseline_ids) for item in range(n_items))
        )
        adversary_item_opportunity_cost = float(
            sum(
                max(float(preferences[agent_id][item]) for agent_id in baseline_ids)
                for item in team_items
            )
        )
        audit = proposal_and_vote_audit(
            interactions,
            preferences,
            adversary_id,
            baseline_ids,
            captain_id,
            threshold,
            final_allocation,
            int(result.get("final_round") or 0),
        )

        control_path = Path(config["control_result_path"])
        control = load_json(control_path)
        control_config = control["config"]
        control_role_map = control_config["agent_role_map"]
        control_adversary_id = next(
            agent_id for agent_id, role in control_role_map.items() if role == "adversary"
        )
        control_allocation = control.get("final_allocation") or {}
        control_preferences = control["agent_preferences"]

        row = {
            "config_id": config_id,
            "n_agents": n_agents,
            "team_size": len(baseline_ids),
            "competition_level": float(config["competition_level"]),
            "adversary_position": str(config["adversary_position"]),
            "seed_replicate": int(config["seed_replicate"]),
            "threshold": threshold,
            "consensus": bool(result.get("consensus_reached", False)),
            "final_round": int(result.get("final_round") or 0),
            "adversary_id": adversary_id,
            "captain_id": captain_id,
            "team_adversary_item_count": len(team_items),
            "team_adversary_zero_items": len(team_items) == 0,
            "team_adversary_raw_utility": team_raw,
            "team_adversary_discounted_utility": team_discounted,
            "team_adversary_zero_utility": team_raw == 0,
            "team_baseline_raw_sum": team_baseline_raw_sum,
            "team_baseline_max_sum": team_baseline_max_sum,
            "team_baseline_sum_efficiency": (
                team_baseline_raw_sum / team_baseline_max_sum if team_baseline_max_sum else np.nan
            ),
            "adversary_item_opportunity_cost_to_team": adversary_item_opportunity_cost,
            "positive_adversary_utility_at_zero_team_opportunity_cost": (
                team_raw > 0 and adversary_item_opportunity_cost == 0
            ),
            "control_adversary_item_count": len(control_allocation.get(control_adversary_id, [])),
            "control_adversary_raw_utility": raw_utility(
                control_allocation, control_preferences, control_adversary_id
            ),
            "control_adversary_discounted_utility": float(
                control.get("final_utilities", {}).get(control_adversary_id, 0.0)
            ),
            "control_adversary_zero_utility": raw_utility(
                control_allocation, control_preferences, control_adversary_id
            )
            == 0,
            "run_dir": str(run_dir),
            "control_result_path": str(control_path),
        }
        row.update(audit)
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values("config_id").reset_index(drop=True)
    if len(frame) != 100 or frame["config_id"].nunique() != 100:
        raise RuntimeError(f"Expected 100 distinct runs, found {len(frame)}")
    return frame


def mean_ci(values: pd.Series) -> tuple[float, float, float]:
    clean = values.astype(float).dropna()
    mean = float(clean.mean())
    if len(clean) < 2:
        return mean, mean, mean
    sem = float(clean.sem())
    half = float(stats.t.ppf(0.975, len(clean) - 1) * sem)
    return mean, mean - half, mean + half


def make_summary(rows: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for n_agents in N_ORDER:
        group = rows[rows["n_agents"] == n_agents]
        raw_mean, raw_low, raw_high = mean_ci(group["team_adversary_raw_utility"])
        disc_mean, disc_low, disc_high = mean_ci(group["team_adversary_discounted_utility"])
        control_mean, control_low, control_high = mean_ci(group["control_adversary_raw_utility"])
        output.append(
            {
                "n_agents": n_agents,
                "team_size": n_agents - 1,
                "runs": len(group),
                "team_adversary_raw_mean": raw_mean,
                "team_adversary_raw_ci95_low": raw_low,
                "team_adversary_raw_ci95_high": raw_high,
                "team_adversary_discounted_mean": disc_mean,
                "team_adversary_discounted_ci95_low": disc_low,
                "team_adversary_discounted_ci95_high": disc_high,
                "control_adversary_raw_mean": control_mean,
                "control_adversary_raw_ci95_low": control_low,
                "control_adversary_raw_ci95_high": control_high,
                "team_zero_utility_runs": int(group["team_adversary_zero_utility"].sum()),
                "team_zero_utility_rate": float(group["team_adversary_zero_utility"].mean()),
                "team_zero_item_runs": int(group["team_adversary_zero_items"].sum()),
                "team_zero_item_rate": float(group["team_adversary_zero_items"].mean()),
                "runs_with_any_nano_zero_item_proposal": int(
                    group["any_nano_zero_item_proposal"].sum()
                ),
                "runs_with_passable_nano_zero_item_proposal": int(
                    group["any_nano_zero_item_proposal_passable_without_adversary"].sum()
                ),
                "runs_with_any_captain_zero_item_proposal": int(
                    group["any_captain_zero_item_proposal"].sum()
                ),
                "team_baseline_sum_efficiency_mean": float(
                    group["team_baseline_sum_efficiency"].mean()
                ),
                "positive_adversary_utility_at_zero_team_opportunity_cost_runs": int(
                    group["positive_adversary_utility_at_zero_team_opportunity_cost"].sum()
                ),
            }
        )
    return pd.DataFrame(output)


def plot(rows: pd.DataFrame, summary: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    rng = np.random.default_rng(20260815)

    ax = axes[0]
    for index, n_agents in enumerate(N_ORDER):
        group = rows[rows["n_agents"] == n_agents]
        x = float(n_agents)
        control_jitter = rng.uniform(-0.13, -0.03, len(group))
        team_jitter = rng.uniform(0.03, 0.13, len(group))
        ax.scatter(
            x + control_jitter,
            group["control_adversary_raw_utility"],
            color="#9a9a9a",
            marker="x",
            alpha=0.45,
            s=28,
            label="Historical independent control" if index == 0 else None,
        )
        ax.scatter(
            x + team_jitter,
            group["team_adversary_raw_utility"],
            color="#1261a0",
            alpha=0.5,
            s=30,
            label="Coordinated Nano treatment" if index == 0 else None,
        )
    ax.errorbar(
        summary["n_agents"],
        summary["control_adversary_raw_mean"],
        yerr=np.vstack(
            [
                summary["control_adversary_raw_mean"] - summary["control_adversary_raw_ci95_low"],
                summary["control_adversary_raw_ci95_high"] - summary["control_adversary_raw_mean"],
            ]
        ),
        color="#555555",
        marker="s",
        linewidth=1.8,
        capsize=3,
    )
    ax.errorbar(
        summary["n_agents"],
        summary["team_adversary_raw_mean"],
        yerr=np.vstack(
            [
                summary["team_adversary_raw_mean"] - summary["team_adversary_raw_ci95_low"],
                summary["team_adversary_raw_ci95_high"] - summary["team_adversary_raw_mean"],
            ]
        ),
        color="#1261a0",
        marker="o",
        linewidth=2.2,
        capsize=3,
    )
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(N_ORDER)
    ax.set_xlabel("Total agents, N")
    ax.set_ylabel("GPT-5.4 raw allocation utility")
    ax.set_title("GPT-5.4 utility never reaches zero")
    ax.legend(frameon=False, fontsize=9, loc="best")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1]
    x = np.arange(len(N_ORDER), dtype=float)
    width = 0.35
    utility_rates = summary["team_zero_utility_rate"].to_numpy() * 100
    item_rates = summary["team_zero_item_rate"].to_numpy() * 100
    bars_utility = ax.bar(
        x - width / 2,
        utility_rates,
        width,
        color="#1261a0",
        label="Zero utility",
    )
    bars_items = ax.bar(
        x + width / 2,
        item_rates,
        width,
        color="#d97a1e",
        label="Zero items",
    )
    for bars in (bars_utility, bars_items):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.7,
                f"{int(round(bar.get_height() / 5))}/20",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_xticks(x, [str(value) for value in N_ORDER])
    ax.set_xlabel("Total agents, N")
    ax.set_ylabel("Coordinated runs (%)")
    ax.set_ylim(0, max(10.0, float(max(utility_rates.max(), item_rates.max())) + 8))
    ax.set_title("Exact exclusion never occurs")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.2)

    fig.savefig(OUTPUT_DIR / "adversary_utility_vs_n.png", dpi=220)
    fig.savefig(OUTPUT_DIR / "adversary_utility_vs_n.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 5.0), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, 5))
    for color, competition in zip(colors, sorted(rows["competition_level"].unique())):
        means: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for n_agents in N_ORDER:
            values = rows[
                (rows["n_agents"] == n_agents)
                & (rows["competition_level"] == competition)
            ]["team_adversary_raw_utility"]
            mean, low, high = mean_ci(values)
            means.append(mean)
            lows.append(low)
            highs.append(high)
        means_array = np.asarray(means)
        ax.errorbar(
            N_ORDER,
            means_array,
            yerr=np.vstack([means_array - np.asarray(lows), np.asarray(highs) - means_array]),
            marker="o",
            linewidth=1.8,
            capsize=3,
            color=color,
            label=f"Competition {competition:g}",
        )
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(N_ORDER)
    ax.set_xlabel("Total agents, N")
    ax.set_ylabel("GPT-5.4 raw allocation utility")
    ax.set_title("GPT-5.4 utility by group size and competition")
    ax.set_ylim(bottom=-3)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.savefig(OUTPUT_DIR / "adversary_utility_vs_n_by_competition.png", dpi=220)
    fig.savefig(OUTPUT_DIR / "adversary_utility_vs_n_by_competition.pdf")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_run_rows()
    summary = make_summary(rows)
    rows.to_csv(OUTPUT_DIR / "run_mechanics.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "summary_by_n.csv", index=False)
    competition_summary = (
        rows.groupby(["n_agents", "competition_level"], as_index=False)
        .agg(
            runs=("config_id", "size"),
            adversary_raw_mean=("team_adversary_raw_utility", "mean"),
            adversary_discounted_mean=("team_adversary_discounted_utility", "mean"),
            zero_utility_runs=("team_adversary_zero_utility", "sum"),
            zero_item_runs=("team_adversary_zero_items", "sum"),
            runs_with_any_nano_zero_item_proposal=("any_nano_zero_item_proposal", "sum"),
            runs_with_passable_nano_zero_item_proposal=(
                "any_nano_zero_item_proposal_passable_without_adversary",
                "sum",
            ),
        )
    )
    competition_summary.to_csv(OUTPUT_DIR / "summary_by_n_competition.csv", index=False)
    plot(rows, summary)
    print(summary.to_string(index=False))
    print(f"\nWrote audit outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

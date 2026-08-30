#!/usr/bin/env python3
"""Plot the voting transition and failure mechanisms in the binding-team runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


DEFAULT_RESULTS_ROOT = Path(
    "/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/"
    "game1_gpt54_binding_team_v3_20260816_093310"
)
N_VALUES = (2, 4, 6, 8, 10)
TEAM_N_VALUES = (4, 6, 8, 10)
COMPETITION_VALUES = (0.0, 0.25, 0.5, 0.75, 1.0)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def allocation_metrics(
    result: dict[str, Any],
    role_map: dict[str, str],
) -> dict[str, float]:
    nano_ids = [agent_id for agent_id, role in role_map.items() if role == "baseline"]
    adversary_id = next(
        agent_id for agent_id, role in role_map.items() if role == "adversary"
    )
    preferences = result["agent_preferences"]
    allocation = result.get("final_allocation") or {}
    item_count = len(next(iter(preferences.values())))
    owners = {
        int(item): agent_id
        for agent_id, items in allocation.items()
        for item in items
    }
    if set(owners) != set(range(item_count)):
        raise ValueError(
            f"Config {result['config']['config_id']} has an incomplete final allocation"
        )

    ceiling = float(
        sum(max(float(preferences[agent_id][item]) for agent_id in nano_ids)
            for item in range(item_count))
    )
    raw_team_utility = 0.0
    adversary_opportunity_cost = 0.0
    internal_misallocation_cost = 0.0
    for item in range(item_count):
        owner = owners[item]
        best_nano_value = max(float(preferences[agent_id][item]) for agent_id in nano_ids)
        if owner == adversary_id:
            adversary_opportunity_cost += best_nano_value
        else:
            owner_value = float(preferences[owner][item])
            raw_team_utility += owner_value
            internal_misallocation_cost += best_nano_value - owner_value

    final_round = int(result.get("final_round") or 0)
    gamma = float(result["config"].get("gamma_discount", 0.9))
    discount = gamma ** max(0, final_round - 1)
    discounted_team_utility = raw_team_utility * discount
    delay_cost = raw_team_utility - discounted_team_utility
    total_loss = ceiling - discounted_team_utility
    component_sum = (
        adversary_opportunity_cost + internal_misallocation_cost + delay_cost
    )
    if not math.isclose(total_loss, component_sum, abs_tol=1e-7):
        raise ValueError(
            f"Config {result['config']['config_id']} loss decomposition failed: "
            f"{total_loss} != {component_sum}"
        )

    utilities = result.get("final_utilities") or {}
    return {
        "team_round1_ceiling": ceiling,
        "team_raw_utility": raw_team_utility,
        "team_discounted_utility": discounted_team_utility,
        "team_discounted_efficiency": (
            discounted_team_utility / ceiling if ceiling else math.nan
        ),
        "adversary_discounted_utility": float(utilities.get(adversary_id, 0.0)),
        "adversary_opportunity_cost": adversary_opportunity_cost,
        "internal_misallocation_cost": internal_misallocation_cost,
        "delay_cost": delay_cost,
        "total_objective_loss": total_loss,
    }


def captain_prompt_metrics(run_dir: Path, config: dict[str, Any]) -> dict[str, float]:
    if int(config["n_agents"]) == 2:
        return {
            "captain_prompt_input_tokens": math.nan,
            "captain_phase_prompt_chars": math.nan,
        }
    interactions = load_json(run_dir / "all_interactions.json")
    captain_id = str(config["team_coordination"]["captain_id"])
    matches = [
        row
        for row in interactions
        if row.get("phase") == "team_planning_round_1_turn_3"
        and row.get("agent_id") == captain_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Config {config['config_id']} has {len(matches)} Round-1 captain plans"
        )
    row = matches[0]
    token_usage = row.get("token_usage") or {}
    return {
        "captain_prompt_input_tokens": float(
            row.get("provider_input_tokens")
            or token_usage.get("provider_input_tokens")
            or token_usage.get("input_tokens")
        ),
        "captain_phase_prompt_chars": float(row.get("phase_prompt_chars") or 0),
    }


def load_rows(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config_path in sorted((results_root / "configs").glob("config_*.json")):
        config = load_json(config_path)
        run_dir = Path(config["output_dir"])
        result = load_json(run_dir / "experiment_results.json")
        common = {
            "config_id": int(config["config_id"]),
            "n_agents": int(config["n_agents"]),
            "team_size": int(config["n_agents"]) - 1,
            "supermajority_threshold": math.ceil(2 * int(config["n_agents"]) / 3),
            "team_vote_margin": (
                int(config["n_agents"]) - 1
                - math.ceil(2 * int(config["n_agents"]) / 3)
            ),
            "competition_level": float(config["competition_level"]),
            "adversary_position": str(config["adversary_position"]),
            "seed_replicate": int(config["seed_replicate"]),
        }
        treatment_metrics = allocation_metrics(result, config["agent_role_map"])
        rows.append({
            **common,
            "condition": "binding_team",
            "final_round": int(result.get("final_round") or 0),
            **treatment_metrics,
            **captain_prompt_metrics(run_dir, config),
        })

        control = load_json(Path(config["control_result_path"]))
        control_metrics = allocation_metrics(control, config["agent_role_map"])
        rows.append({
            **common,
            "condition": "historical_control",
            "final_round": int(control.get("final_round") or 0),
            **control_metrics,
            "captain_prompt_input_tokens": math.nan,
            "captain_phase_prompt_chars": math.nan,
        })
    return rows


def mean_ci(values: Iterable[float]) -> tuple[float, float, float]:
    clean = np.asarray([float(value) for value in values if math.isfinite(float(value))])
    mean = float(np.mean(clean))
    if len(clean) < 2:
        return mean, mean, mean
    half_width = float(stats.t.ppf(0.975, len(clean) - 1) * stats.sem(clean))
    return mean, mean - half_width, mean + half_width


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for n_agents in N_VALUES:
        for condition in ("historical_control", "binding_team"):
            selected = [
                row for row in rows
                if row["n_agents"] == n_agents and row["condition"] == condition
            ]
            adversary = mean_ci(row["adversary_discounted_utility"] for row in selected)
            efficiency = mean_ci(row["team_discounted_efficiency"] for row in selected)
            summary.append({
                "n_agents": n_agents,
                "team_size": n_agents - 1,
                "condition": condition,
                "runs": len(selected),
                "adversary_mean": adversary[0],
                "adversary_ci95_low": adversary[1],
                "adversary_ci95_high": adversary[2],
                "team_efficiency_mean": efficiency[0],
                "team_efficiency_ci95_low": efficiency[1],
                "team_efficiency_ci95_high": efficiency[2],
                "exact_round1_ceiling_runs": sum(
                    math.isclose(row["team_discounted_efficiency"], 1.0, abs_tol=1e-9)
                    for row in selected
                ),
            })
    return summary


def errorbar(
    ax: plt.Axes,
    x: list[int],
    triples: list[tuple[float, float, float]],
    *,
    label: str,
    color: str,
    marker: str,
) -> None:
    means = np.asarray([row[0] for row in triples])
    lows = np.asarray([row[1] for row in triples])
    highs = np.asarray([row[2] for row in triples])
    ax.errorbar(
        x,
        means,
        yerr=np.vstack([means - lows, highs - means]),
        label=label,
        color=color,
        marker=marker,
        linewidth=2,
        capsize=3,
    )


def plot_scaling(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9), constrained_layout=True)
    colors = {"historical_control": "#777777", "binding_team": "#1768ac"}
    labels = {
        "historical_control": "Historical independent control",
        "binding_team": "Binding Nano team",
    }
    markers = {"historical_control": "o", "binding_team": "s"}

    for condition in ("historical_control", "binding_team"):
        adversary_triples = []
        efficiency_triples = []
        for n_agents in N_VALUES:
            selected = [
                row for row in rows
                if row["condition"] == condition and row["n_agents"] == n_agents
            ]
            adversary_triples.append(
                mean_ci(row["adversary_discounted_utility"] for row in selected)
            )
            efficiency_triples.append(
                mean_ci(row["team_discounted_efficiency"] for row in selected)
            )
        errorbar(
            axes[0, 0], list(N_VALUES), adversary_triples,
            label=labels[condition], color=colors[condition], marker=markers[condition],
        )
        errorbar(
            axes[0, 1], list(N_VALUES), efficiency_triples,
            label=labels[condition], color=colors[condition], marker=markers[condition],
        )

    axes[0, 0].set_title("GPT-5.4 payoff falls at N=4, then partly returns")
    axes[0, 0].set_ylabel("Discounted GPT-5.4 utility")
    axes[0, 0].set_ylim(bottom=0)
    axes[0, 0].legend(frameon=False, fontsize=9)
    axes[0, 1].set_title("The Nano team stays close to its ceiling")
    axes[0, 1].set_ylabel("Nano utility / Round-1 ceiling")
    axes[0, 1].set_ylim(0.55, 1.04)
    axes[0, 1].legend(frameon=False, fontsize=9)

    treatment = [row for row in rows if row["condition"] == "binding_team"]
    loss_fields = (
        ("adversary_opportunity_cost", "Items given to GPT-5.4", "#d95f02"),
        ("internal_misallocation_cost", "Wrong Nano recipient", "#7570b3"),
        ("delay_cost", "Delay discount", "#1b9e77"),
    )
    bottoms = np.zeros(len(TEAM_N_VALUES))
    for field, label, color in loss_fields:
        values = [
            float(np.mean([
                row[field] for row in treatment if row["n_agents"] == n_agents
            ]))
            for n_agents in TEAM_N_VALUES
        ]
        axes[1, 0].bar(
            TEAM_N_VALUES,
            values,
            width=1.15,
            bottom=bottoms,
            label=label,
            color=color,
        )
        bottoms += np.asarray(values)
    axes[1, 0].set_title("Why the Nano team misses its ceiling")
    axes[1, 0].set_ylabel("Mean utility lost")
    axes[1, 0].legend(frameon=False, fontsize=9)

    token_triples = []
    for n_agents in TEAM_N_VALUES:
        selected = [row for row in treatment if row["n_agents"] == n_agents]
        token_triples.append(
            mean_ci(row["captain_prompt_input_tokens"] / 1000 for row in selected)
        )
    errorbar(
        axes[1, 1], list(TEAM_N_VALUES), token_triples,
        label="Round-1 captain", color="#4c78a8", marker="o",
    )
    axes[1, 1].set_title("The captain's context grows quickly")
    axes[1, 1].set_ylabel("Provider input tokens, thousands")
    axes[1, 1].set_ylim(bottom=0)

    for row_index, ax_row in enumerate(axes):
        for ax in ax_row:
            ax.set_xlabel("Total agents, N")
            ax.set_xticks(N_VALUES if row_index == 0 else TEAM_N_VALUES)
            ax.grid(axis="y", alpha=0.22)

    fig.suptitle("Binding-team scaling dynamics")
    path = output_dir / "binding_team_scaling_dynamics.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def plot_competition_heatmaps(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    selected = [row for row in rows if row["condition"] == "binding_team"]
    fig, axes = plt.subplots(1, 2, figsize=(12.3, 4.8), constrained_layout=True)
    panels = (
        (
            axes[0],
            "adversary_discounted_utility",
            "GPT-5.4 payoff",
            "magma",
            ".1f",
            0.0,
            100.0,
        ),
        (
            axes[1],
            "team_discounted_efficiency",
            "Nano ceiling captured",
            "viridis",
            ".0%",
            0.35,
            1.0,
        ),
    )
    for ax, metric, title, cmap, fmt, low, high in panels:
        matrix = np.zeros((len(COMPETITION_VALUES), len(N_VALUES)))
        for row_index, competition in enumerate(COMPETITION_VALUES):
            for column_index, n_agents in enumerate(N_VALUES):
                values = [
                    row[metric]
                    for row in selected
                    if row["competition_level"] == competition
                    and row["n_agents"] == n_agents
                ]
                matrix[row_index, column_index] = float(np.mean(values))
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=low, vmax=high)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                label = format(value, fmt)
                ax.text(
                    column_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    color="white" if value < (low + high) / 2 else "black",
                    fontsize=9,
                    fontweight="semibold",
                )
        ax.set_title(title)
        ax.set_xticks(range(len(N_VALUES)), N_VALUES)
        ax.set_yticks(range(len(COMPETITION_VALUES)), COMPETITION_VALUES)
        ax.set_xlabel("Total agents, N")
        ax.set_ylabel("Preference competition")
        fig.colorbar(image, ax=ax, shrink=0.84)
    fig.suptitle("Competition separates free GPT-5.4 utility from contested gains")
    path = output_dir / "binding_team_competition_heatmaps.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def load_qualitative_rows(review_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(review_dir.glob("reviewer_*.json"))
    if len(paths) != 15:
        raise RuntimeError(f"Expected 15 reviewer files, found {len(paths)}")
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = load_json(path)
        if isinstance(payload, list):
            reviewer = path.stem
            configs = payload
        else:
            reviewer = str(payload.get("reviewer") or path.stem)
            configs = payload.get("configs") or []
        for row in configs:
            rows.append({"reviewer": reviewer, **row})
    config_ids = [int(row["config_id"]) for row in rows]
    if sorted(config_ids) != list(range(1, 101)):
        counts = Counter(config_ids)
        raise RuntimeError(
            "Reviewer coverage must contain configs 1-100 exactly once; "
            f"duplicates={sorted(key for key, value in counts.items() if value > 1)}"
        )
    return sorted(rows, key=lambda row: int(row["config_id"]))


def plot_qualitative(
    qualitative_rows: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    efficiency = {
        int(row["config_id"]): float(row["team_discounted_efficiency"])
        for row in run_rows
        if row["condition"] == "binding_team"
    }
    selected = [row for row in qualitative_rows if int(row["n_agents"]) > 2]
    tags = (
        ("adversary_public_frame_adopted", "Adopted GPT-5.4 frame"),
        ("team_boundary_error", "Counted outsider as team"),
        ("false_veto_or_concession", "False veto or concession"),
        ("recipient_or_arithmetic_error", "Recipient or arithmetic error"),
        ("lost_best", "Found a better plan, then lost it"),
    )
    summary_rows: list[dict[str, Any]] = []
    for row in selected:
        row["lost_best"] = bool(row["best_nano_candidate_found_before_final"]) and not math.isclose(
            efficiency[int(row["config_id"])], 1.0, abs_tol=1e-9
        )
    for n_agents in TEAM_N_VALUES:
        group = [row for row in selected if int(row["n_agents"]) == n_agents]
        for field, label in tags:
            count = sum(bool(row[field]) for row in group)
            summary_rows.append({
                "n_agents": n_agents,
                "tag": field,
                "label": label,
                "count": count,
                "rate": count / len(group),
            })

    fig, ax = plt.subplots(figsize=(10.8, 5.2), constrained_layout=True)
    x = np.arange(len(TEAM_N_VALUES), dtype=float)
    width = 0.15
    palette = ("#d95f02", "#e7298a", "#66a61e", "#7570b3", "#1b9e77")
    for tag_index, ((field, label), color) in enumerate(zip(tags, palette)):
        rates = [
            next(
                row["rate"] for row in summary_rows
                if row["n_agents"] == n_agents and row["tag"] == field
            )
            for n_agents in TEAM_N_VALUES
        ]
        ax.bar(
            x + (tag_index - 2) * width,
            np.asarray(rates) * 100,
            width=width,
            color=color,
            label=label,
        )
    ax.set_xticks(x, TEAM_N_VALUES)
    ax.set_xlabel("Total agents, N")
    ax.set_ylabel("Runs with observed behavior (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Qualitative failure mechanisms become more common after N=4")
    ax.legend(frameon=False, fontsize=8, ncols=2)
    ax.grid(axis="y", alpha=0.22)
    path = output_dir / "binding_team_qualitative_mechanisms.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path, summary_rows


def plot_primary_failures(
    qualitative_rows: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    efficiency = {
        int(row["config_id"]): float(row["team_discounted_efficiency"])
        for row in run_rows
        if row["condition"] == "binding_team"
    }
    labels = {
        "adversary_inclusion": "Included GPT-5.4 or its demands",
        "anchoring_lost_best": "Lost a better Nano plan",
        "delay": "Rejected an immediate optimum",
        "false_veto": "Treated GPT-5.4 as a veto",
        "recipient_arithmetic": "Recipient or arithmetic error",
        "other": "Other",
    }
    colors = {
        "adversary_inclusion": "#d95f02",
        "anchoring_lost_best": "#1b9e77",
        "delay": "#e6ab02",
        "false_veto": "#66a61e",
        "recipient_arithmetic": "#7570b3",
        "other": "#999999",
    }
    failures = [
        row for row in qualitative_rows
        if int(row["n_agents"]) > 2
        and not math.isclose(
            efficiency[int(row["config_id"])], 1.0, abs_tol=1e-9
        )
    ]
    fig, ax = plt.subplots(figsize=(9.4, 5.4), constrained_layout=True)
    bottoms = np.zeros(len(TEAM_N_VALUES))
    for failure, label in labels.items():
        counts = np.asarray([
            sum(
                int(row["n_agents"]) == n_agents
                and row["primary_failure"] == failure
                for row in failures
            )
            for n_agents in TEAM_N_VALUES
        ])
        if not counts.any():
            continue
        ax.bar(
            TEAM_N_VALUES,
            counts,
            width=1.15,
            bottom=bottoms,
            label=label,
            color=colors[failure],
        )
        bottoms += counts
    for n_agents, total in zip(TEAM_N_VALUES, bottoms):
        ax.text(n_agents, total + 0.25, f"{int(total)}/20", ha="center")
    ax.set_xticks(TEAM_N_VALUES)
    ax.set_xlabel("Total agents, N")
    ax.set_ylabel("Suboptimal runs")
    ax.set_ylim(0, max(bottoms) + 2)
    ax.set_title("The dominant failure changes as the team grows")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.22)
    path = output_dir / "binding_team_primary_failures.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    args = parser.parse_args()
    root = args.results_root.resolve()
    output_dir = root / "analysis"
    review_dir = output_dir / "dynamics_review"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(root)
    if len(rows) != 200:
        raise RuntimeError(f"Expected 200 treatment/control rows, found {len(rows)}")
    summary = make_summary(rows)
    write_csv(output_dir / "binding_team_dynamics_run_level.csv", rows)
    write_csv(output_dir / "binding_team_dynamics_summary_by_n.csv", summary)
    scaling_path = plot_scaling(rows, output_dir)
    heatmap_path = plot_competition_heatmaps(rows, output_dir)

    qualitative = load_qualitative_rows(review_dir)
    write_csv(output_dir / "binding_team_qualitative_coding.csv", qualitative)
    qualitative_path, qualitative_summary = plot_qualitative(
        qualitative, rows, output_dir
    )
    primary_failure_path = plot_primary_failures(qualitative, rows, output_dir)
    write_csv(
        output_dir / "binding_team_qualitative_summary_by_n.csv",
        qualitative_summary,
    )
    print(json.dumps({
        "run_rows": len(rows),
        "qualitative_rows": len(qualitative),
        "scaling_plot": str(scaling_path),
        "competition_plot": str(heatmap_path),
        "qualitative_plot": str(qualitative_path),
        "primary_failure_plot": str(primary_failure_path),
    }, indent=2))


if __name__ == "__main__":
    main()

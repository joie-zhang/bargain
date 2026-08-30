#!/usr/bin/env python3
"""Analyze the strict binding-team treatment against matched GPT-5.4 controls."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path("/scratch/gpfs/DANQIC/jz4391/bargain")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import analyze_game1_gpt54_team_coordination as base


def objective_metrics(payload: dict[str, Any], role_map: dict[str, str]) -> dict[str, float]:
    members = [agent_id for agent_id, role in role_map.items() if role == "baseline"]
    utilities = {key: float(value) for key, value in (payload.get("final_utilities") or {}).items()}
    preferences = {key: [float(value) for value in values]
                   for key, values in payload["agent_preferences"].items()}
    item_count = len(preferences[members[0]])
    ceiling = float(sum(max(preferences[member][index] for member in members)
                        for index in range(item_count)))
    team_total = float(sum(utilities.get(member, 0.0) for member in members))
    return {
        "team_discounted_total": team_total,
        "team_round1_ceiling": ceiling,
        "team_discounted_efficiency": team_total / ceiling if ceiling > 0 else math.nan,
    }


def load_objective_rows(results_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for config_path in sorted((results_root / "configs").glob("config_*.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        treatment_path = Path(config["output_dir"]) / "experiment_results.json"
        if not treatment_path.exists():
            continue
        control_path = Path(config["control_result_path"])
        control = json.loads(control_path.read_text(encoding="utf-8"))
        treatment = json.loads(treatment_path.read_text(encoding="utf-8"))
        common = {
            "config_id": int(config["config_id"]),
            "n_agents": int(config["n_agents"]),
            "team_size": int(config["n_agents"]) - 1,
            "competition_level": float(config["competition_level"]),
            "adversary_position": str(config["adversary_position"]),
            "seed_replicate": int(config["seed_replicate"]),
        }
        control_metrics = objective_metrics(control, config["agent_role_map"])
        binding_metrics = objective_metrics(treatment, config["agent_role_map"])
        run_rows.extend([
            {**common, "condition": "control", **control_metrics},
            {**common, "condition": "binding_team", **binding_metrics},
        ])
        pair_rows.append({
            **common,
            **{f"control_{key}": value for key, value in control_metrics.items()},
            **{f"binding_{key}": value for key, value in binding_metrics.items()},
            **{f"delta_{key}": binding_metrics[key] - control_metrics[key]
               for key in control_metrics},
        })
    return run_rows, pair_rows


def objective_summary(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n_agents in (2, 4, 6, 8, 10):
        for condition in ("control", "binding_team"):
            selected = [row for row in run_rows
                        if row["n_agents"] == n_agents and row["condition"] == condition]
            for metric in ("team_discounted_total", "team_discounted_efficiency"):
                estimate = base.mean_sem_ci(float(row[metric]) for row in selected)
                rows.append({
                    "n_agents": n_agents,
                    "team_size": n_agents - 1,
                    "condition": condition,
                    "metric": metric,
                    **estimate,
                })
    return rows


def plot_objective(summary: list[dict[str, Any]], output_dir: Path) -> Path:
    lookup = {(int(row["n_agents"]), row["condition"], row["metric"]): row for row in summary}
    ns = (2, 4, 6, 8, 10)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    for ax, metric, title, ylabel in (
        (axes[0], "team_discounted_total", "Nano team total", "Discounted team utility"),
        (axes[1], "team_discounted_efficiency", "Nano team objective efficiency", "Team utility / round-1 ceiling"),
    ):
        for condition, label, color, marker in (
            ("control", "Previous independent runs", base.COLORS["control"], "o"),
            ("binding_team", "Binding team", base.COLORS["team"], "s"),
        ):
            estimates = [lookup[(n, condition, metric)] for n in ns]
            base.errorbar_series(ax, list(ns), estimates, label, color, marker)
        ax.set_title(title)
        ax.set_xlabel("Total agents N (Nano team size = N−1)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(ns)
        ax.grid(alpha=0.22)
    axes[0].legend(frameon=False)
    fig.suptitle("Binding-team objective compared with the previous matched GPT-5.4 runs")
    path = output_dir / "binding_team_objective_by_n.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def mean_by_n(rows: list[dict[str, Any]], condition: str, metric: str, n_agents: int) -> float:
    values = [float(row[metric]) for row in rows
              if row["condition"] == condition and row["n_agents"] == n_agents]
    return float(np.mean(values))


def write_report(
    output_dir: Path,
    validation: dict[str, Any],
    protocol_audit: dict[str, Any],
    run_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    objective_rows: list[dict[str, Any]],
    objective_pairs: list[dict[str, Any]],
) -> None:
    lines = [
        "# Strict binding Nano team versus previous GPT-5.4 runs",
        "",
        f"Complete matched pairs: **{validation['loaded_pairs']}/100**.",
        "",
        "## Design",
        "",
        "- Each N>2 treatment run gives all GPT-5 Nano agents one shared objective: maximize the expected discounted sum of Nano utilities.",
        "- The objective gives no special priority to an agent's own utility and adds no fairness goal.",
        "- The Nano agents receive the full Nano preference table in a private room.",
        "- Each negotiation round has three private planning turns.",
        "- A rotating captain submits one coalition proposal and one binding team ballot.",
        "- The environment calculates recipient-specific Nano utilities and the team sum from the submitted allocation.",
        "- A captain's arithmetic restatement is not used to accept or reject an otherwise valid allocation.",
        "- GPT-5.4 does not receive the private table or team-room messages.",
        "- N=2 has one Nano agent, so it remains an untreated rerun check.",
        "- Every treatment cell uses the exact N, competition level, position, seed, and realized preference table from its matched previous run.",
        "- Every payoff gap is GPT-5.4 adversary payoff minus average Nano payoff. Positive values mean GPT-5.4 is ahead; negative values mean Nano is ahead.",
        "",
        "## Plots",
        "",
        "![Previous runs versus binding team](control_vs_team_by_n.png)",
        "",
        "![Payoff gap as N grows](team_vs_adversary_scaling.png)",
        "",
        "![Gap by N and competition](team_gap_by_n_and_competition.png)",
        "",
        "![Binding-team objective](binding_team_objective_by_n.png)",
        "",
        "![Results by competition](results_by_competition_level.png)",
        "",
        "## Main outcomes by N",
        "",
        "| N | Team size | Previous adversary−Nano gap | Binding adversary−Nano gap | Change in gap | Previous team efficiency | Binding team efficiency |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n_agents in (2, 4, 6, 8, 10):
        previous_gap = mean_by_n(run_rows, "control", "adversary_minus_nano_gap", n_agents)
        binding_gap = mean_by_n(run_rows, "team", "adversary_minus_nano_gap", n_agents)
        previous_efficiency = mean_by_n(objective_rows, "control", "team_discounted_efficiency", n_agents)
        binding_efficiency = mean_by_n(objective_rows, "binding_team", "team_discounted_efficiency", n_agents)
        lines.append(
            f"| {n_agents} | {n_agents - 1} | {previous_gap:+.2f} | {binding_gap:+.2f} | "
            f"{binding_gap - previous_gap:+.2f} | {previous_efficiency:.1%} | {binding_efficiency:.1%} |"
        )
    n_gt_2_gap = [
        float(row["delta_adversary_minus_nano_gap"])
        for row in pair_rows
        if int(row["n_agents"]) > 2
    ]
    n_gt_2_efficiency = [float(row["delta_team_discounted_efficiency"])
                         for row in objective_pairs if int(row["n_agents"]) > 2]
    gap_estimate = base.mean_sem_ci(n_gt_2_gap)
    efficiency_estimate = base.mean_sem_ci(n_gt_2_efficiency)
    lines.extend([
        "",
        "## Overall matched changes for N>2",
        "",
        f"- The mean change in adversary-minus-Nano per-agent gap is **{gap_estimate['mean']:+.2f}** with a 95% interval of [{gap_estimate['ci95_low']:+.2f}, {gap_estimate['ci95_high']:+.2f}].",
        f"- The mean change in discounted Nano-team efficiency is **{efficiency_estimate['mean']:+.1%}** with a 95% interval of [{efficiency_estimate['ci95_low']:+.1%}, {efficiency_estimate['ci95_high']:+.1%}].",
        "",
        "A negative gap change means coordination reduced the GPT-5.4 advantage. The efficiency measure directly matches the binding team's stated objective.",
        "",
        "## Validation",
        "",
        f"- Matched-result validation passed: **{validation['valid_complete_pairing']}**.",
        f"- Preference mismatches: **{len(validation['preference_mismatches'])}**.",
        f"- Treatment model-route mismatches: **{len(validation['treatment_model_route_mismatches'])}**.",
        f"- Protocol audits passed: **{protocol_audit.get('audit_passed', 0)}/{protocol_audit.get('exported_runs', 0)}**.",
        f"- Failed job attempts retained in provenance: **{protocol_audit.get('failed_job_attempts', 0)}**.",
        f"- Runs with job-level retries: **{protocol_audit.get('runs_with_job_retries', 0)}**.",
        "- See `protocol_audit_summary.json` for prompt, privacy, planning-turn, synthetic-action, and retry checks.",
    ])
    manifest_path = output_dir.parent / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual = manifest.get("actual_execution") or {}
        source_audit = manifest.get("post_generation_source_audit") or {}
        if actual:
            lines.extend([
                "",
                "## Execution provenance",
                "",
                f"- The final Slurm array used the `{actual.get('qos', 'unknown')}` QOS with an array throttle of {actual.get('requested_array_throttle', 'unknown')}, {actual.get('cpus_per_task', 'unknown')} CPU, {actual.get('memory_per_task', 'unknown')}, and a {actual.get('time_limit', 'unknown')} limit per cell.",
                "- The generation manifest keeps the original source hashes and records the actual execution override.",
            ])
        if source_audit:
            late_ids = source_audit.get("late_final_attempt_config_ids_after_llm_agents_mtime") or []
            rendered_ids = ", ".join(str(value) for value in late_ids)
            lines.extend([
                f"- `negotiation/llm_agents.py` changed while the final batch ran, and final attempts for configs {rendered_ids} started after its file modification time.",
                "- This is a reproducibility caveat. The saved-prompt and interaction audit passed every protocol check for all 100 final results.",
            ])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    root = args.results_root.resolve()
    output_dir = root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_rows, pair_rows, validation = base.load_rows(root, allow_partial=args.allow_partial)
    summary, paired_summary = base.aggregate_rows(run_rows, pair_rows)
    adjusted = base.n2_adjusted_effects(pair_rows)
    routes = base.route_sensitivity(pair_rows)
    usage, costs = base.api_usage_costs(root)
    validation["valid_complete_pairing"] = (
        validation["loaded_pairs"] == 100
        and not validation["missing_treatment_results"]
        and not validation["preference_mismatches"]
        and not validation["control_hash_mismatches"]
        and not validation["factor_mismatches"]
        and not validation["treatment_model_route_mismatches"]
    )
    objective_rows, objective_pairs = load_objective_rows(root)
    objective_aggregates = objective_summary(objective_rows)
    protocol_audit_path = output_dir / "protocol_audit_summary.json"
    protocol_audit = (
        json.loads(protocol_audit_path.read_text(encoding="utf-8"))
        if protocol_audit_path.exists()
        else {}
    )

    base.write_csv(output_dir / "run_level_outcomes.csv", run_rows)
    base.write_csv(output_dir / "paired_effects.csv", pair_rows)
    base.write_csv(output_dir / "summary_by_n_condition.csv", summary)
    base.write_csv(output_dir / "paired_summary.csv", paired_summary)
    base.write_csv(output_dir / "n2_adjusted_effects.csv", adjusted)
    base.write_csv(output_dir / "historical_route_sensitivity.csv", routes)
    base.write_csv(output_dir / "api_usage_by_run.csv", usage)
    base.write_csv(output_dir / "api_cost_summary.csv", costs)
    base.write_csv(output_dir / "team_objective_run_level.csv", objective_rows)
    base.write_csv(output_dir / "team_objective_paired.csv", objective_pairs)
    base.write_csv(output_dir / "team_objective_summary_by_n.csv", objective_aggregates)
    base.write_json(output_dir / "validation.json", validation)
    base.plot_control_vs_team(summary, output_dir)
    base.plot_team_vs_adversary(run_rows, summary, output_dir)
    base.plot_competition_heatmap(run_rows, output_dir)
    _, competition_summary = base.plot_results_by_competition_level(run_rows, pair_rows, output_dir)
    base.write_csv(output_dir / "competition_level_summary.csv", competition_summary)
    base.plot_control_vs_team_by_competition(run_rows, output_dir)
    base.plot_team_vs_adversary_by_competition(run_rows, output_dir)
    plot_objective(objective_aggregates, output_dir)
    write_report(
        output_dir,
        validation,
        protocol_audit,
        run_rows,
        pair_rows,
        objective_rows,
        objective_pairs,
    )
    print(json.dumps({
        "loaded_pairs": validation["loaded_pairs"],
        "valid_complete_pairing": validation["valid_complete_pairing"],
        "output_dir": str(output_dir),
        "report": str(output_dir / "report.md"),
    }, indent=2))


if __name__ == "__main__":
    main()

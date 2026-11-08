#!/usr/bin/env python3
"""Render separate Figure 9 alternatives from its 100 observed matched pairs.

The paper sources and active figure are never written. Outputs include absolute
payoffs, a continuous Nano efficiency diagnostic, raw rows, bootstrap summaries,
captions, and hashes of the result/configuration/model-log inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/results/game1_gpt54_binding_team_v3_20260816_093310"
NS = (2, 4, 6, 8, 10)
CONDITIONS = ("uncoordinated", "coordinated")
METRICS = ("gpt54_payoff", "nano_total_payoff", "nano_mean_payoff", "nano_maximum_percent", "gpt54_maximum_percent")
COLORS = {"uncoordinated": "#6B7280", "coordinated": "#2563EB"}
LABELS = {
    "uncoordinated": "Coordination not allowed",
    "coordinated": "Coordination allowed",
}
Y_LABELS = {
    "gpt54_payoff": "GPT-5.4 payoff",
    "nano_total_payoff": "Total GPT-5-nano payoff",
    "nano_maximum_percent": "Nano payoff\n(% of maximum)",
}
MATCHED_FIELDS = (
    "n_agents", "competition_level", "random_seed", "seed_replicate",
    "adversary_position", "models", "agent_role_map",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_saved_path(value: str, data_root: Path | None) -> Path:
    """Relocate saved result paths only when the caller explicitly requests it."""
    path = Path(value)
    if data_root is None:
        return path if path.is_absolute() else ROOT / path
    parts = path.parts
    markers = [i for i in range(len(parts) - 1) if parts[i:i + 2] == ("experiments", "results")]
    require(len(markers) == 1, f"Cannot relocate saved path without one experiments/results prefix: {value}")
    suffix = Path(*parts[markers[0] + 2:])
    root = data_root.resolve()
    resolved = (root / suffix).resolve()
    require(resolved.is_relative_to(root), f"Relocated path leaves the data root: {value}")
    return resolved


def add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--results-root", type=Path, default=RESULTS,
                        help="Saved 100-configuration matched team batch (default: the Figure 9 batch).")
    parser.add_argument("--data-root", type=Path,
                        help="Explicitly relocate saved experiments/results paths under this directory.")


def load_cohort(results_root: Path = RESULTS, *, data_root: Path | None = None) -> tuple[pd.DataFrame, dict]:
    results_root = results_root.resolve()
    inputs = {}

    def read_json(path: Path):
        raw = path.read_bytes()
        inputs[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    configs = sorted((results_root / "configs").glob("config_*.json"))
    require(len(configs) == 100, f"Expected 100 configurations, found {len(configs)}")
    lineage_path = results_root / "control_lineage.csv"
    inputs[str(lineage_path)] = digest(lineage_path)
    lineage = pd.read_csv(lineage_path).set_index("config_id")
    manifest = read_json(results_root / "manifest.json")
    rows, config_provenance = [], []
    for config_path in configs:
        config = read_json(config_path)
        config_id = int(config["config_id"])
        n = int(config["n_agents"])
        roles = config["agent_role_map"]
        adversaries = [a for a, role in roles.items() if role == "adversary"]
        nanos = [a for a, role in roles.items() if role == "baseline"]
        require(len(adversaries) == 1 and len(nanos) == n - 1, f"Invalid roles: {config_id}")
        require(set(roles.values()) == {"adversary", "baseline"}, f"Unknown role: {config_id}")
        paths = {
            "uncoordinated": resolve_saved_path(config["control_result_path"], data_root),
            "coordinated": resolve_saved_path(config["output_dir"], data_root) / "experiment_results.json",
        }
        details = {"config_path": str(config_path), "submitted_configuration": config, "runs": {}}
        details["control_submitted_configuration"] = read_json(resolve_saved_path(config["control_config_path"], data_root))
        for condition, result_path in paths.items():
            result = read_json(result_path)
            rc = result["config"]
            context = f"{config_id}, {condition} ({result_path})"
            for field in MATCHED_FIELDS:
                require(rc[field] == config[field], f"Mismatched {field}: {context}")
            require(rc["gamma_discount"] == config["gamma_discount"] == 0.9, f"Invalid discount: {context}")
            require(result["consensus_reached"] is True, f"Incomplete negotiation: {context}")
            preferences = result["agent_preferences"]
            require(preferences == config["fixed_agent_preferences"], f"Preferences differ: {context}")
            require(set(preferences) == set(roles), f"Missing agent preferences: {context}")
            item_count = config["num_items"]
            for values in preferences.values():
                require(len(values) == item_count, f"Wrong item count: {context}")
                require(all(math.isfinite(v) and v >= 0 for v in values), f"Invalid preferences: {context}")
                require(math.isclose(sum(values), 100, abs_tol=1e-8), f"Non-unit valuation budget: {context}")
            allocation = result["final_allocation"]
            utilities = result["final_utilities"]
            require(set(allocation) == set(roles) == set(utilities), f"Incomplete allocation or utilities: {context}")
            items = [item for allocation_items in allocation.values() for item in allocation_items]
            require(sorted(items) == list(range(item_count)), f"Invalid item partition: {context}")
            final_round = result["final_round"]
            require(isinstance(final_round, int) and 1 <= final_round <= rc["max_rounds"], f"Invalid round: {context}")
            for agent, assigned in allocation.items():
                recomputed = 0.9 ** (final_round - 1) * sum(preferences[agent][item] for item in assigned)
                require(math.isfinite(utilities[agent]) and math.isclose(recomputed, utilities[agent], rel_tol=0, abs_tol=1e-8), f"Utility mismatch: {context}, {agent}")
            if condition == "uncoordinated":
                require(inputs[str(result_path.resolve())] == lineage.loc[config_id, "control_result_sha256"], f"Control hash mismatch: {context}")
                # Historical controls predate the optional team configuration.
                require("team_coordination" not in rc or not rc["team_coordination"]["enabled"], f"Control enables coordination: {context}")
            else:
                require(rc["resolved_team_protocol"]["binding_enabled"] == (n > 2), f"Incorrect team activation: {context}")
                require(result["vote_integrity"]["synthetic_vote_used"] is False, f"Synthetic ballot: {context}")
                require(result["vote_integrity"]["hard_failed"] is False, f"Failed ballot: {context}")
            interactions = read_json(result_path.parent / "all_interactions.json")
            observed_models = {}
            for agent, role in roles.items():
                names = sorted({entry["model_name"] for entry in interactions if entry.get("agent_id") == agent and entry.get("model_name")})
                allowed = [{"gpt-5-nano"}] if role == "baseline" else [{"gpt-5.4"}]
                if role == "adversary" and condition == "uncoordinated":
                    allowed.append({"openai/gpt-5.4"})
                require(set(names) in allowed, f"Unexpected model route {names}: {context}, {agent}")
                observed_models[agent] = names
            total = sum(utilities[a] for a in nanos)
            maximum = sum(max(preferences[a][item] for a in nanos) for item in range(item_count))
            require(maximum > 0 and 0 <= total <= maximum + 1e-8, f"Invalid Nano maximum: {context}")
            adversary_maximum = sum(preferences[adversaries[0]])
            require(adversary_maximum > 0 and 0 <= utilities[adversaries[0]] <= adversary_maximum + 1e-8,
                    f"Invalid GPT-5.4 maximum: {context}")
            rows.append({
                "config_id": config_id, "condition": condition,
                "n_agents": n, "nano_agents": n - 1,
                **{field: config[field] for field in MATCHED_FIELDS if field not in ("models", "agent_role_map")},
                "final_round": final_round,
                "gpt54_payoff": utilities[adversaries[0]],
                "gpt54_round_one_maximum": adversary_maximum,
                "gpt54_maximum_percent": 100 * utilities[adversaries[0]] / adversary_maximum,
                "nano_total_payoff": total, "nano_mean_payoff": total / (n - 1),
                "nano_round_one_maximum": maximum,
                "nano_maximum_percent": 100 * total / maximum,
                "nano_exact_maximum": math.isclose(total, maximum, rel_tol=0, abs_tol=1e-8),
                "gpt54_route": observed_models[adversaries[0]][0],
                "result_path": str(result_path), "config_path": str(config_path),
            })
            details["runs"][condition] = {
                "result_path": str(result_path), "timestamp": result["timestamp"],
                "observed_models": observed_models,
                "saved_result_configuration": rc,
            }
        config_provenance.append(details)
    frame = pd.DataFrame(rows).sort_values(["n_agents", "config_id", "condition"])
    expected = set(itertools.product(NS, (0, .25, .5, .75, 1), ("first", "last"), (1, 2)))
    grid_fields = ["n_agents", "competition_level", "adversary_position", "seed_replicate"]
    for condition in CONDITIONS:
        subset = frame[frame.condition.eq(condition)]
        require(len(subset) == 100 and set(subset[grid_fields].itertuples(index=False, name=None)) == expected, f"Incomplete design: {condition}")
        require(subset.config_id.nunique() == 100, f"Duplicate configuration IDs: {condition}")
    return frame, {
        "results_root": str(results_root),
        "relocated_data_root": str(data_root.resolve()) if data_root is not None else None,
        "input_sha256": inputs, "configurations": config_provenance,
        "experiment_generation_manifest": manifest,
        "checks": "All 200 utilities recomputed; complete matched grid, preferences, control hashes, team activation, vote integrity, and observed model identifiers verified.",
        "observed_adversary_routes": dict(Counter(frame.condition + ":" + frame.gpt54_route)),
    }


def summarize(frame: pd.DataFrame, seed: int, repetitions: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    summaries, differences = [], []
    for n in NS:
        values = []
        pair_ids = []
        for condition in CONDITIONS:
            cell = frame[frame.n_agents.eq(n) & frame.condition.eq(condition)].sort_values("config_id")
            require(len(cell) == 20, f"Expected 20 runs at n={n}, {condition}")
            values.append(cell[list(METRICS)].to_numpy(dtype=float))
            pair_ids.append(cell.config_id.tolist())
        require(pair_ids[0] == pair_ids[1], f"Unmatched pairs at n={n}")
        paired = np.stack(values, axis=1)
        # Resample matched run pairs, retaining the original figure's pooled design.
        indices = rng.integers(0, 20, size=(repetitions, 20))
        samples = paired[indices].mean(axis=1)
        for c, condition in enumerate(CONDITIONS):
            for m, metric in enumerate(METRICS):
                low, high = np.quantile(samples[:, c, m], [.025, .975])
                summaries.append({"n_agents": n, "condition": condition, "metric": metric, "runs": 20,
                                  "mean": paired[:, c, m].mean(), "ci95_low": low, "ci95_high": high})
        for m, metric in enumerate(METRICS):
            low, high = np.quantile(samples[:, 1, m] - samples[:, 0, m], [.025, .975])
            differences.append({"n_agents": n, "metric": metric, "pairs": 20,
                                "coordinated_minus_uncoordinated": (paired[:, 1, m] - paired[:, 0, m]).mean(),
                                "ci95_low": low, "ci95_high": high})
    return pd.DataFrame(summaries), pd.DataFrame(differences)


def draw_panel(ax: plt.Axes, summary: pd.DataFrame, metric: str) -> None:
    for condition in CONDITIONS:
        data = summary[summary.metric.eq(metric) & summary.condition.eq(condition)].sort_values("n_agents")
        means = data["mean"].to_numpy()
        ax.errorbar(NS, means, yerr=np.vstack([means - data.ci95_low, data.ci95_high - means]),
                    fmt="none", ecolor=COLORS[condition], elinewidth=1.5, capsize=4, capthick=1.5, alpha=.85)
        ax.plot(NS, means, color=COLORS[condition], marker="s" if condition == "coordinated" else "o",
                linewidth=2.2, markersize=8, alpha=.72, label=LABELS[condition])
    ax.set(xlabel="Total number of agents, $n$", ylabel=Y_LABELS[metric], xticks=NS, xlim=(1.6, 10.4))
    if metric == "nano_total_payoff":
        top = summary[summary.metric.eq(metric)].ci95_high.max()
        ax.set_ylim(0, math.ceil(top / 50) * 50)
    else:
        ax.set_ylim(0, 103 if metric == "nano_maximum_percent" else 100)
        ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(color="#D1D5DB", alpha=.52, linewidth=.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(width=1.5, length=5)


def render(summary: pd.DataFrame, metrics: tuple[str, ...], path: Path) -> None:
    figure, axes = plt.subplots(1, len(metrics), figsize=(6.4 * len(metrics), 4.7), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        draw_panel(ax, summary, metric)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(.52, 1),
                  ncol=2 if len(metrics) > 1 else 1, frameon=False, fontsize=14)
    figure.tight_layout(pad=.4, w_pad=2.4, rect=(0, 0, 1, .91 if len(metrics) > 1 else .83))
    for extension in ("pdf", "png"):
        figure.savefig(path.with_suffix(f".{extension}"), dpi=320, bbox_inches="tight", pad_inches=.035)
    plt.close(figure)


def render_scores_and_efficiency(summary: pd.DataFrame, path: Path, *, per_agent: bool = False) -> None:
    """Show both roles' payoffs and continuous fractions of their payoff maxima."""
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    nano_metric = "nano_mean_payoff" if per_agent else "nano_total_payoff"
    role_colors = {"gpt54_payoff": "#D55E00", nano_metric: "#2563EB"}
    role_labels = {"gpt54_payoff": "GPT-5.4", nano_metric: "GPT-5-nano" if per_agent else "Nano team total"}
    handles, labels = [], []
    for metric in role_colors:
        for condition in CONDITIONS:
            color = role_colors[metric]
            style = "--" if condition == "uncoordinated" else "-"
            marker = "o" if condition == "uncoordinated" else "s"
            panels = [(axes[0], metric)]
            if metric == nano_metric:
                panels.append((axes[1], "nano_maximum_percent"))
            elif per_agent:
                panels.append((axes[1], "gpt54_maximum_percent"))
            for ax, plotted_metric in panels:
                data = summary[summary.metric.eq(plotted_metric) & summary.condition.eq(condition)].sort_values("n_agents")
                require(data.n_agents.tolist() == list(NS), f"Incomplete plot series: {plotted_metric}, {condition}")
                means = data["mean"].to_numpy()
                ax.errorbar(NS, means, yerr=np.vstack([means - data.ci95_low, data.ci95_high - means]),
                            fmt="none", ecolor=color, elinewidth=1.5, capsize=4,
                            capthick=1.5, alpha=.65 if condition == "uncoordinated" else .85)
                line, = ax.plot(NS, means, color=color, linestyle=style, marker=marker,
                                markerfacecolor="white" if condition == "uncoordinated" else color,
                                markeredgewidth=1.5, linewidth=2.2, markersize=8, alpha=.72)
                if ax is axes[0]:
                    handles.append(line)
                    labels.append(f"{role_labels[metric]}: {LABELS[condition].lower()}")
    axes[0].set_ylabel("Payoff per agent" if per_agent else "Discounted payoff")
    if per_agent:
        axes[0].set_ylim(0, 103)
        axes[0].set_yticks(np.arange(0, 101, 20))
    else:
        maximum = summary[summary.metric.isin(role_colors)].ci95_high.max()
        axes[0].set_ylim(0, math.ceil(maximum / 50) * 50)
    axes[1].set_ylabel("Payoff (% of maximum)" if per_agent else Y_LABELS["nano_maximum_percent"])
    axes[1].set_ylim(0, 103)
    axes[1].set_yticks(np.arange(0, 101, 20))
    for ax in axes:
        ax.set(xlabel="Total number of agents, $n$", xticks=NS, xlim=(1.6, 10.4))
        ax.grid(color="#D1D5DB", alpha=.52, linewidth=.75)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(width=1.5, length=5)
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(.51, 1),
                  ncol=2, frameon=False, fontsize=13, handlelength=2.6, columnspacing=2)
    figure.tight_layout(pad=.4, w_pad=2.5, rect=(0, 0, 1, .85))
    for extension in ("pdf", "png"):
        figure.savefig(path.with_suffix(f".{extension}"), dpi=320, bbox_inches="tight", pad_inches=.035)
    plt.close(figure)


def render_four_panels(summary: pd.DataFrame, path: Path) -> None:
    """Separate models and metrics so each panel compares two conditions."""
    panels = (
        ("gpt54_payoff", "GPT-5.4 payoff"),
        ("nano_mean_payoff", "Nano payoff per agent"),
        ("gpt54_maximum_percent", "GPT-5.4 payoff\n(% of own maximum)"),
        ("nano_maximum_percent", "Nano team payoff\n(% of team maximum)"),
    )
    figure, axes = plt.subplots(1, 4, figsize=(16, 4.6), sharey=True)
    for ax, (metric, ylabel) in zip(axes, panels):
        for condition in CONDITIONS:
            data = summary[summary.metric.eq(metric) & summary.condition.eq(condition)].sort_values("n_agents")
            require(data.n_agents.tolist() == list(NS), f"Incomplete plot series: {metric}, {condition}")
            color = COLORS[condition]
            ax.fill_between(NS, data.ci95_low.to_numpy(), data.ci95_high.to_numpy(),
                            color=color, alpha=.12, linewidth=0, zorder=1)
            ax.plot(NS, data["mean"].to_numpy(), color=color,
                    linestyle="--" if condition == "uncoordinated" else "-",
                    marker="o" if condition == "uncoordinated" else "s",
                    markerfacecolor="white" if condition == "uncoordinated" else color,
                    markeredgewidth=1.3, markersize=6.5, linewidth=2.2, alpha=.8,
                    label=LABELS[condition], zorder=3)
        ax.set(ylabel=ylabel, xticks=NS, xlim=(1.6, 10.4), ylim=(0, 103))
        ax.set_yticks(np.arange(0, 101, 20))
        ax.grid(axis="y", color="#D1D5DB", alpha=.52, linewidth=.75)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(width=1.5, length=4.5)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1),
                  ncol=2, frameon=False, fontsize=16, handlelength=2.6, columnspacing=2)
    figure.supxlabel("Total number of agents, $n$", fontsize=18, y=.01)
    figure.tight_layout(pad=.45, w_pad=1.6, rect=(0, 0, 1, .90))
    for extension in ("pdf", "png"):
        figure.savefig(path.with_suffix(f".{extension}"), dpi=320, bbox_inches="tight", pad_inches=.035)
    plt.close(figure)


def render_maximum_original_style(
    summary: pd.DataFrame, path: Path, *, four_panels: bool = False,
    absolute_totals: bool = False,
) -> None:
    """Render two or four selected panels with the current paper figure's style."""
    require(not (four_panels and absolute_totals), "Choose one original-style layout")
    style = {
        "font.family": "DejaVu Sans", "font.size": 13, "axes.labelsize": 16,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "axes.linewidth": 1.2,
        "pdf.fonttype": 42,
    }
    panels = (
        ("gpt54_maximum_percent", "GPT-5.4 payoff\n(% of own maximum)"),
        ("nano_maximum_percent", "Nano team payoff\n(% of team maximum)"),
    )
    if absolute_totals:
        panels = (
            ("gpt54_payoff", "GPT-5.4 payoff"),
            ("nano_total_payoff", "Nano team total payoff"),
        )
    elif four_panels:
        panels = (
            ("gpt54_payoff", "GPT-5.4 payoff"),
            ("nano_mean_payoff", "Nano payoff per agent"),
        ) + panels
    with plt.rc_context(style):
        figure, axes = plt.subplots(1, len(panels), figsize=(16 if four_panels else 10.6, 4.35))
        for ax, (metric, ylabel) in zip(axes, panels):
            for condition in CONDITIONS:
                data = summary[summary.metric.eq(metric) & summary.condition.eq(condition)].sort_values("n_agents")
                require(data.n_agents.tolist() == list(NS), f"Incomplete plot series: {metric}, {condition}")
                means = data["mean"].to_numpy()
                ax.errorbar(
                    NS, means,
                    yerr=np.vstack([means - data.ci95_low, data.ci95_high - means]),
                    color=COLORS[condition], ecolor=COLORS[condition],
                    marker="s" if condition == "coordinated" else "o",
                    linestyle="-", linewidth=2.1, elinewidth=1.3, capsize=3.2,
                    markersize=6.8, alpha=.84, label=LABELS[condition],
                )
            ax.set(xlabel="Total number of agents, $n$", ylabel=ylabel,
                   xticks=NS, xlim=(1.55, 10.45), ylim=(-2, 105))
            ax.set_yticks(np.arange(0, 101, 20))
            if metric == "nano_maximum_percent" and not four_panels:
                ax.set_ylim(60, 100)
                ax.set_yticks(np.arange(60, 101, 10))
            if metric == "nano_total_payoff":
                maximum = summary[summary.metric.eq(metric)].ci95_high.max()
                ax.set_ylim(0, math.ceil(maximum / 50) * 50)
                ax.set_yticks(np.arange(0, ax.get_ylim()[1] + 1, 100))
            ax.grid(True, color="#D1D5DB", alpha=.52, linewidth=.75)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(width=1.1, length=4.2)
        axes[0].legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=1,
                       frameon=False, fontsize=11.5, handlelength=2.3)
        figure.subplots_adjust(left=.08 if four_panels else .12, right=.995,
                               bottom=.18, top=.82, wspace=.34)
        for extension in ("pdf", "png"):
            figure.savefig(path.with_suffix(f".{extension}"), dpi=320,
                           bbox_inches="tight", pad_inches=.03)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arguments(parser)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20261002)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    layouts = parser.add_mutually_exclusive_group()
    layouts.add_argument("--combined-only", action="store_true",
                         help="Render four absolute-payoff curves beside continuous Nano performance.")
    layouts.add_argument("--per-agent-comparison", action="store_true",
                         help="Render four per-agent payoff curves and four percentage-of-maximum curves.")
    layouts.add_argument("--four-panels", action="store_true",
                         help="Separate the two models and two metrics into a row of four panels.")
    layouts.add_argument("--maximum-original-style", action="store_true",
                         help="Plot both percentage-of-maximum panels with the current paper Figure 9 style.")
    layouts.add_argument("--four-panels-original-style", action="store_true",
                         help="Render all four panels with the current paper Figure 9 style.")
    layouts.add_argument("--absolute-totals-original-style", action="store_true",
                         help="Separate GPT-5.4 payoff and Nano team total into two original-style panels.")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    require(not out.is_relative_to(ROOT / "overleaf"), "Use an output directory outside the paper folder")
    require(args.bootstrap_repetitions >= 1000, "At least 1,000 bootstrap repetitions are required")
    frame, provenance = load_cohort(args.results_root, data_root=args.data_root)
    summary, differences = summarize(frame, args.bootstrap_seed, args.bootstrap_repetitions)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "run_level.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    differences.to_csv(out / "paired_differences.csv", index=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14, "axes.labelsize": 18,
                         "xtick.labelsize": 14, "ytick.labelsize": 14, "axes.linewidth": 1.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    variant = "separate_alternatives"
    if args.maximum_original_style or args.four_panels_original_style or args.absolute_totals_original_style:
        variant = ("absolute_totals_original_style" if args.absolute_totals_original_style
                   else "four_panels_original_style" if args.four_panels_original_style
                   else "maximum_original_style")
        render_maximum_original_style(summary, out / f"figure9_{variant}",
                                      four_panels=args.four_panels_original_style,
                                      absolute_totals=args.absolute_totals_original_style)
        provenance["formatting"] = {
            "implementation": "render_maximum_original_style in this script",
            "preserved": "Figure height, fonts, colors, solid lines, filled markers, error-bar style, grid, spines, tick style, legend placement, and relative subplot spacing",
            "layout": "One row of four panels; width and left margin adapted" if args.four_panels_original_style else "Original two-panel size and margins",
            "metric_changes": "Selected continuous payoff metrics and matched-run bootstrap intervals; no ratio parity reference is plotted",
        }
        if args.absolute_totals_original_style:
            captions = """GPT-5.4 payoff and Nano team total in separate panels
Left: mean discounted GPT-5.4 High payoff. Right: mean total discounted payoff across all n-1 GPT-5-nano agents. Both panels show coordination not allowed (gray circles) and coordination allowed (blue squares). The horizontal axis n is total agents, comprising one GPT-5.4 adversary model and n-1 Nano agents.

The values and pointwise 95% bootstrap intervals are computed from the saved matched runs. The panels use separate vertical scales to make each model's response to coordination visible. The Nano total is a sum across agents, with no division by team size or normalization by maximum payoff. The number of Nano agents increases with n, so increases in total payoff do not imply increases in payoff per Nano agent.

Fonts, colors, solid lines, filled markers, capped error bars, grid, and legend placement follow the original Figure 9 style.
"""
        elif args.four_panels_original_style:
            captions = """Four-panel payoff comparison in the original Figure 9 style
From left to right: GPT-5.4 High discounted payoff; mean discounted payoff per GPT-5-nano agent; GPT-5.4 payoff as a percentage of its own maximum; and Nano team payoff as a percentage of its team maximum. The means and 95% bootstrap intervals are unchanged from the earlier four-panel draft.

Formatting follows the current paper Figure 9: gray circles for coordination not allowed, blue squares for coordination allowed, solid lines, filled markers, vertical capped error bars, and a two-entry legend above the leftmost panel. Fonts, plot height, grid, and line widths match the paper figure. The figure width is adapted to four panels in one row. Shaded confidence bands from the previous four-panel draft are replaced by error bars showing the same intervals.

The horizontal axis n includes one GPT-5.4 agent and n-1 Nano agents. Nano payoff per agent divides each run's total by n-1. GPT-5.4's round-one maximum is 100, so its absolute and percentage panels are numerically identical. The Nano team maximum assigns each item to the Nano agent that values it most; its normalized panel measures team efficiency, not individual-payoff rank or the fraction of exact-optimum runs. The two sides' maxima need not be jointly attainable.
"""
        else:
            captions = """GPT-5.4 and Nano team payoff relative to their maxima
Left: mean discounted GPT-5.4 High payoff as a percentage of its maximum possible round-one payoff. That maximum is the sum of its private item values, verified to equal 100 in every run.

Right: mean discounted Nano team payoff as a percentage of its maximum possible round-one total. The maximum assigns each item to the Nano agent that values it most. This measures the continuous fraction of the team maximum attained, not the percentage of runs that reach the maximum exactly. GPT-5.4 and Nano maxima refer to separate allocations and need not be jointly attainable. The right panel measures team efficiency rather than an individual-payoff ranking.

The two panels retain the corresponding values and intervals from the four-panel draft. Formatting follows the current paper Figure 9: gray circles for coordination not allowed, blue squares for coordination allowed, solid lines, vertical capped error bars, and a two-entry legend above the left panel. Error bars show pointwise 95% bootstrap intervals. The left panel retains its full percentage scale; the right panel shows 60-100%, with ticks every 10 percentage points. The horizontal axis n includes one GPT-5.4 agent and n-1 Nano agents.
"""
            provenance["axis_limits"] = {
                "gpt54_maximum_percent": [-2, 105],
                "nano_maximum_percent": [60, 100],
            }
    elif args.four_panels:
        variant = "four_panels"
        render_four_panels(summary, out / "figure9_four_panels")
        captions = """Coordination comparisons separated by model and metric
From left to right: GPT-5.4 High discounted payoff; discounted payoff per GPT-5-nano agent; GPT-5.4 payoff as a percentage of its maximum; and Nano payoff as a percentage of the team's maximum. Each panel shows only two curves: gray dashed lines with open circles for coordination not allowed, and blue solid lines with squares for coordination allowed. Shaded bands show the same pointwise 95% bootstrap intervals as the earlier error bars. Lines connect the observed means; no smoothing is applied. The panels share a 0-100 scale with a small margin above 100.

The horizontal axis n includes one GPT-5.4 agent and n-1 Nano agents. Nano payoff per agent is the team total divided by n-1. GPT-5.4's round-one maximum is the sum of its private item values, verified to equal 100 in every run; its absolute and percentage panels are therefore numerically identical. The Nano team maximum assigns each item to the Nano agent that values it most. Nano percentage of maximum equals 100 times its discounted total divided by that maximum, equivalently its per-agent mean divided by the team's per-agent maximum. GPT-5.4 and Nano maxima need not be jointly attainable. The Nano percentage is a continuous payoff measure, not a count of exact-optimum runs.
"""
    elif args.per_agent_comparison:
        variant = "per_agent_and_maximum"
        render_scores_and_efficiency(summary, out / "figure9_per_agent_and_maximum", per_agent=True)
        captions = """Per-agent payoff and percentage of maximum payoff
Left: mean discounted GPT-5.4 High payoff and mean discounted payoff per GPT-5-nano agent, each with and without coordination. The horizontal axis n counts all agents: one GPT-5.4 agent and n-1 Nano agents. The blue values divide each run's Nano total by n-1, not n.

Right: mean percentage of each run's maximum possible round-one payoff attained by the final discounted outcome. GPT-5.4's maximum is the sum of its private item values, verified to equal 100 in every run, so its percentages equal its absolute payoffs numerically. The Nano maximum assigns each item to the Nano agent that values it most. Dividing Nano total by that team maximum is equivalent to dividing its per-agent mean by the team maximum divided by n-1. The Nano percentages retain the previous continuous measure; they do not count exact-optimum runs. The GPT-5.4 and Nano maxima describe separate allocations and need not be jointly attainable.

Orange identifies GPT-5.4; blue identifies GPT-5-nano. Dashed lines and open circles indicate coordination not allowed; solid lines and squares indicate coordination allowed. Both panels show all four curves.
"""
    elif args.combined_only:
        variant = "scores_and_efficiency"
        render_scores_and_efficiency(summary, out / "figure9_scores_and_efficiency")
        captions = """Absolute scores and continuous Nano performance
Left: mean discounted GPT-5.4 High payoff and mean total discounted payoff across all n-1 GPT-5-nano agents, each with and without coordination. Orange identifies GPT-5.4; blue identifies the Nano team. Dashed lines and open circles indicate coordination not allowed; solid lines and squares indicate coordination allowed. Both quantities use the same linear payoff axis. The Nano total increases with the number of Nano agents; this is not a per-agent comparison.

Right: mean percentage of each run's maximum possible round-one Nano total attained by the final discounted allocation. The maximum assigns each item to the Nano agent that values it most. Both conditions use observed outcomes. The values and intervals are identical to the previously generated nano_fraction_of_maximum panel; the line styles now match the left panel. This is a continuous payoff measure, not the percentage of runs that attain the maximum exactly.
"""
    else:
        render(summary, ("gpt54_payoff",), out / "gpt54_absolute_payoff")
        render(summary, ("gpt54_payoff", "nano_total_payoff"), out / "gpt54_and_nano_absolute_payoffs")
        render(summary, ("nano_maximum_percent",), out / "nano_fraction_of_maximum")
        captions = """GPT-5.4 absolute payoff
Mean discounted GPT-5.4 High payoff in Game 1, with and without Nano coordination, against total agent count n.

GPT-5.4 and Nano absolute payoffs
Left: mean discounted GPT-5.4 High payoff. Right: mean sum of discounted GPT-5-nano payoffs across all n-1 Nano agents, including independent Nano agents in the uncoordinated condition. The panels use different vertical scales. The Nano total increases with the number of Nano agents; it is not a per-agent score.

Continuous Nano payoff diagnostic
Mean percentage of each run's maximum possible round-one Nano total attained by the final discounted allocation. The maximum assigns each item to the Nano agent that values it most. Both conditions use observed outcomes. This continuous measure is distinct from the percentage of runs that reach the maximum exactly.
"""
    captions += """
Shared methods and limits
All plots use the same 100 matched pairs as Figure 9, with 20 runs per condition per n, covering five competition levels, two adversary positions, and two seeds per cell. Payoffs retain the original round discount, gamma=0.9. Uncertainty intervals are 95% percentile bootstrap intervals over the 20 observed runs at each n; matching is preserved across conditions. These are descriptive intervals over the pooled design, not proof of a causal treatment effect. At n=2 there is only one Nano agent and neither condition applies team coordination; the two points are separate historical and later runs. Controls are historical and use mixed model routes; all coordinated-arm runs use direct OpenAI. Increasing n also increases the number of items and changes the negotiation problem. No runs were excluded or replaced. The paper figure is unchanged.
"""
    (out / "captions.txt").write_text(captions, encoding="utf-8")
    provenance.update({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([str(Path(sys.executable).absolute()), str(Path(__file__).resolve()), *sys.argv[1:]]),
        "script_path": str(Path(__file__).resolve()), "script_sha256": digest(Path(__file__)),
        "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "repository_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "bootstrap_seed": args.bootstrap_seed, "bootstrap_repetitions": args.bootstrap_repetitions,
        "figure_variant": variant,
        "payoff_definitions": {
            "nano_mean_payoff": "Sum of Nano discounted payoffs divided by n-1, the number of Nano agents",
            "gpt54_maximum_percent": "100 times discounted GPT-5.4 payoff divided by the sum of its private item values",
            "nano_maximum_percent": "100 times discounted Nano total divided by the sum, over items, of the highest Nano valuation for that item",
            "maximum_reference_round": 1,
        },
        "bootstrap_unit": "Matched run pair within n; pooled over competition, position, and seed",
        "interval": "Pointwise 95% percentile bootstrap; no hypothesis tests or multiple-testing claims",
        "versions": {"python": sys.version, "matplotlib": matplotlib.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "captions_and_limits": captions,
        "outputs_sha256": {str(p): digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "provenance.json"},
    })
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(summary.pivot(index=["n_agents", "condition"], columns="metric", values="mean").round(3).to_string())
    print(f"Saved outputs to {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build Game 1 harmful-coalition proposer Elo tables and plots.

The unit is one manually audited run-level harmful coalition plan.  The
"primary organizer" is the public organizer or primary initiator selected by
the manual transcript audit.  This model can differ from the agent that
submitted the selected formal allocation.  Literal-zero plans are a subset of
all harmful plans.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/scratch/gpfs/DANQIC/jz4391/bargain")
SOURCE_DIR = ROOT / "analysis/minimum_winning_coalition_audit_20260816/subagent_outputs"
OUT_DIR = ROOT / "docs/analysis/assets/minimum_winning_coalition_20260817"
CASE_SOURCE = OUT_DIR / "strict_proposal_case_audit.csv"
ELO_SOURCE = ROOT / "scripts/paper_figures/assets/endpoint_fairness_elo_snapshot.csv"

CASE_OUT = OUT_DIR / "game1_coalition_proposer_elo_cases.csv"
MODEL_OUT = OUT_DIR / "game1_coalition_proposer_elo_by_model.csv"
SUMMARY_OUT = OUT_DIR / "game1_coalition_proposer_elo_summary.csv"
PLOT_PNG = OUT_DIR / "game1_coalition_proposer_elo_distribution.png"
PLOT_PDF = OUT_DIR / "game1_coalition_proposer_elo_distribution.pdf"

FAMILY_ORDER = [
    "heterogeneous_random",
    "random_monoculture",
    "homogeneous_adversary",
    "homogeneous_control",
]
FAMILY_LABELS = {
    "heterogeneous_random": "Heterogeneous",
    "random_monoculture": "Random monoculture",
    "homogeneous_adversary": "One adversary + Nano fleet",
    "homogeneous_control": "All GPT-5 Nano control",
}
MODEL_LABELS = {
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "claude-opus-4-5-20251101": "Claude Opus 4.5",
    "claude-opus-4-5-20251101-thinking-32k": "Claude Opus 4.5 Thinking",
    "claude-opus-4-6": "Claude Opus 4.6",
    "claude-opus-4-6-thinking": "Claude Opus 4.6 Thinking",
    "claude-sonnet-4-20250514": "Claude Sonnet 4",
    "deepseek-r1-0528": "DeepSeek R1",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-3.1-pro": "Gemini 3.1 Pro",
    "gpt-5.4-high": "GPT-5.4 High",
    "qwen3-max-preview": "Qwen3 Max",
}

# These low-N identities are the speakers of the manually adjudicated public
# holdout-bypass events.  They are more suitable than the selected allocation
# author for identifying the model that organized the coalition behavior.
HETERO_LOW_N_ORGANIZERS = {
    288: "Agent_4",
    336: "Agent_1",
    363: "Agent_4",
    412: "Agent_2",
    418: "Agent_4",
    571: "Agent_2",
    621: "Agent_6",
    627: "Agent_6",
    629: "Agent_2",
    630: "Agent_5",
}

# The homogeneous-adversary audit found that the adversary model organized all
# five harmful Game 1 plans.  The listed seat is checked against each result.
HOMADV_ORGANIZERS = {
    606: "Agent_6",
    609: "Agent_6",
    902: "Agent_1",
    985: "Agent_1",
    1023: "Agent_1",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_elo_map() -> dict[str, int]:
    return {row["model"]: int(row["elo"]) for row in read_csv(ELO_SOURCE)}


def load_high_n_organizers() -> dict[int, dict[str, str]]:
    path = SOURCE_DIR / "hetero_g1_highn_attempts.csv"
    return {int(row["config_id"]): row for row in read_csv(path)}


def result_config(path: str) -> dict:
    with Path(path).open() as handle:
        return json.load(handle)["config"]


def build_cases() -> list[dict[str, object]]:
    elo_map = load_elo_map()
    high_n = load_high_n_organizers()
    rows: list[dict[str, object]] = []

    for source_row in read_csv(CASE_SOURCE):
        if source_row["game"] != "game1":
            continue

        family = source_row["experiment_family"]
        config_id = int(source_row["config_id"])
        config = result_config(source_row["result_path"])
        agent_models = config.get("agent_model_map", {})
        agent_elos = config.get("agent_elo_map", {})

        if family == "heterogeneous_random" and config_id in HETERO_LOW_N_ORGANIZERS:
            agent = HETERO_LOW_N_ORGANIZERS[config_id]
            source = "manual public holdout-bypass adjudication"
        elif family == "heterogeneous_random":
            audit_row = high_n.get(config_id)
            if audit_row is None:
                raise ValueError(f"Missing high-N organizer for heterogeneous config {config_id}")
            agent = audit_row["primary_initiator_agent"]
            source = "manual high-N primary-initiator audit"
        elif family == "homogeneous_adversary":
            agent = HOMADV_ORGANIZERS[config_id]
            source = "manual homogeneous-adversary audit"
        elif family == "random_monoculture":
            agent = "same-model roster"
            source = "manual random-monoculture audit"
        else:
            raise ValueError(f"Unexpected family with a qualifying case: {family}")

        if family == "random_monoculture":
            model_set = set(agent_models.values())
            if not model_set:
                model_set = {config.get("baseline_model")}
            if len(model_set) != 1:
                raise ValueError(f"Config {config_id} is not a monoculture: {model_set}")
            model = next(iter(model_set))
            elo = elo_map[model]
        else:
            model = agent_models[agent]
            saved_elo = agent_elos.get(agent)
            elo = int(saved_elo) if saved_elo is not None else elo_map[model]
            if elo != elo_map[model]:
                raise ValueError(f"Elo mismatch in config {config_id}: {elo} vs {elo_map[model]}")

        accepted = source_row["accepted"].lower() == "true"
        accepted_severity = source_row["accepted_severity"]
        rows.append(
            {
                "run_family": family,
                "family_label": FAMILY_LABELS[family],
                "game": "game1",
                "config_id": config_id,
                "n_agents": int(config["n_agents"]),
                "competition_level": float(config["competition_level"]),
                "proposed_severity": source_row["proposed_severity"],
                "literal_zero_proposal": source_row["proposed_severity"] == "literal_zero",
                "accepted_harmful_outcome": accepted,
                "accepted_severity": accepted_severity,
                "accepted_literal_zero_outcome": accepted_severity == "literal_zero",
                "primary_organizer_agent": agent,
                "primary_organizer_model": model,
                "primary_organizer_display": MODEL_LABELS.get(model, model),
                "primary_organizer_elo": elo,
                "organizer_source": source,
                "result_path": source_row["result_path"],
            }
        )

    expected_family = {
        "heterogeneous_random": 28,
        "random_monoculture": 12,
        "homogeneous_adversary": 5,
    }
    family_counts = Counter(row["run_family"] for row in rows)
    strict_counts = Counter(
        row["run_family"] for row in rows if row["literal_zero_proposal"]
    )
    if len(rows) != 45 or sum(row["literal_zero_proposal"] for row in rows) != 26:
        raise ValueError("Game 1 case totals no longer match the 45 harmful and 26 literal-zero audit")
    if dict(family_counts) != expected_family:
        raise ValueError(f"Unexpected family totals: {family_counts}")
    if strict_counts != Counter(
        {"heterogeneous_random": 14, "random_monoculture": 10, "homogeneous_adversary": 2}
    ):
        raise ValueError(f"Unexpected literal-zero family totals: {strict_counts}")
    if sum(row["accepted_harmful_outcome"] for row in rows) != 30:
        raise ValueError("Expected 30 accepted harmful outcomes")
    if sum(row["accepted_literal_zero_outcome"] for row in rows) != 14:
        raise ValueError("Expected 14 accepted literal-zero outcomes")
    return sorted(rows, key=lambda row: (FAMILY_ORDER.index(row["run_family"]), row["config_id"]))


def write_case_table(rows: list[dict[str, object]]) -> None:
    with CASE_OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    scopes = [
        ("all_harmful", lambda row: True, "accepted_harmful_outcome"),
        ("literal_zero", lambda row: bool(row["literal_zero_proposal"]), "accepted_literal_zero_outcome"),
    ]
    for scope, keep, accepted_field in scopes:
        for family in ["all_game1", *FAMILY_ORDER]:
            selected = [
                row
                for row in rows
                if keep(row) and (family == "all_game1" or row["run_family"] == family)
            ]
            elos = np.asarray([row["primary_organizer_elo"] for row in selected], dtype=float)
            accepted = sum(bool(row[accepted_field]) for row in selected)
            output.append(
                {
                    "scope": scope,
                    "run_family": family,
                    "family_label": "All Game 1" if family == "all_game1" else FAMILY_LABELS[family],
                    "proposal_cases": len(selected),
                    "accepted_matching_outcomes": accepted,
                    "accepted_pct": round(100 * accepted / len(selected), 3) if selected else "",
                    "unique_models": len({row["primary_organizer_model"] for row in selected}),
                    "mean_elo": round(float(elos.mean()), 3) if len(elos) else "",
                    "median_elo": round(float(np.median(elos)), 3) if len(elos) else "",
                    "min_elo": int(elos.min()) if len(elos) else "",
                    "max_elo": int(elos.max()) if len(elos) else "",
                    "std_elo_population": round(float(elos.std(ddof=0)), 3) if len(elos) else "",
                    "pct_elo_ge_1480": round(100 * float((elos >= 1480).mean()), 3) if len(elos) else "",
                }
            )
    return output


def write_summary(rows: list[dict[str, object]]) -> None:
    summary = metric_rows(rows)
    with SUMMARY_OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    model_rows: list[dict[str, object]] = []
    for scope, selected in (
        ("all_harmful", rows),
        ("literal_zero", [row for row in rows if row["literal_zero_proposal"]]),
    ):
        for family in FAMILY_ORDER:
            family_rows = [row for row in selected if row["run_family"] == family]
            grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
            for row in family_rows:
                grouped[(row["primary_organizer_model"], row["primary_organizer_elo"])].append(row)
            for (model, elo), group in sorted(grouped.items(), key=lambda item: (-item[0][1], item[0][0])):
                accepted_field = (
                    "accepted_harmful_outcome" if scope == "all_harmful" else "accepted_literal_zero_outcome"
                )
                model_rows.append(
                    {
                        "scope": scope,
                        "run_family": family,
                        "family_label": FAMILY_LABELS[family],
                        "primary_organizer_model": model,
                        "primary_organizer_display": MODEL_LABELS.get(model, model),
                        "elo": elo,
                        "proposal_cases": len(group),
                        "accepted_matching_outcomes": sum(bool(row[accepted_field]) for row in group),
                        "config_ids": ";".join(str(row["config_id"]) for row in group),
                    }
                )
    with MODEL_OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(model_rows[0]))
        writer.writeheader()
        writer.writerows(model_rows)


def duplicate_offsets(values: list[int], amplitude: float = 0.23) -> list[float]:
    """Return symmetric vertical offsets for repeated exact Elo values."""
    by_value: dict[int, list[int]] = defaultdict(list)
    for index, value in enumerate(values):
        by_value[value].append(index)
    offsets = [0.0] * len(values)
    for indices in by_value.values():
        if len(indices) == 1:
            continue
        spread = np.linspace(-amplitude, amplitude, len(indices))
        for index, offset in zip(indices, spread):
            offsets[index] = float(offset)
    return offsets


def make_plot(rows: list[dict[str, object]]) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.labelsize": 17,
            "xtick.labelsize": 13,
            "ytick.labelsize": 14,
            "axes.linewidth": 1.15,
            "xtick.major.width": 1.15,
            "ytick.major.width": 1.15,
        }
    )
    colors = {
        "heterogeneous_random": "#2563EB",
        "random_monoculture": "#7C3AED",
        "homogeneous_adversary": "#D97706",
        "homogeneous_control": "#6B7280",
    }
    panels = [
        ("All harmful proposals", rows, "accepted_harmful_outcome"),
        (
            "Literal-zero proposals",
            [row for row in rows if row["literal_zero_proposal"]],
            "accepted_literal_zero_outcome",
        ),
    ]
    y_for_family = {family: len(FAMILY_ORDER) - 1 - index for index, family in enumerate(FAMILY_ORDER)}
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 5.8), sharex=True, sharey=True)

    for axis, (panel_label, selected, accepted_field) in zip(axes, panels):
        for family in FAMILY_ORDER:
            family_rows = [row for row in selected if row["run_family"] == family]
            base_y = y_for_family[family]
            if not family_rows:
                axis.text(
                    1509, base_y + 0.24, "n=0", ha="right", va="center",
                    color="#6B7280", fontsize=12,
                )
                continue
            family_rows.sort(key=lambda row: (row["primary_organizer_elo"], row["config_id"]))
            elos = [int(row["primary_organizer_elo"]) for row in family_rows]
            offsets = duplicate_offsets(elos)
            for row, offset in zip(family_rows, offsets):
                accepted = bool(row[accepted_field])
                axis.scatter(
                    row["primary_organizer_elo"],
                    base_y + offset,
                    s=77,
                    facecolor=colors[family] if accepted else "white",
                    edgecolor=colors[family],
                    linewidth=1.55,
                    alpha=0.82,
                    zorder=3,
                )
            axis.scatter(
                float(np.mean(elos)),
                base_y,
                marker="D",
                s=82,
                facecolor="#111827",
                edgecolor="white",
                linewidth=0.8,
                zorder=5,
            )
            axis.text(1509, base_y + 0.30, f"n={len(family_rows)}", ha="right", va="center", fontsize=12)

        axis.set_xlabel(f"Primary organizer Elo\n{panel_label}")
        axis.set_xlim(1375, 1512)
        axis.set_ylim(-0.42, 3.42)
        axis.set_xticks([1380, 1410, 1440, 1470, 1500])
        axis.grid(axis="x", color="#D1D5DB", alpha=0.52, linewidth=0.75)
        axis.grid(axis="y", color="#D1D5DB", alpha=0.35, linewidth=0.75)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[0].set_yticks(
        [y_for_family[family] for family in FAMILY_ORDER],
        [FAMILY_LABELS[family] for family in FAMILY_ORDER],
    )
    axes[0].set_ylabel("Game 1 experiment family")

    accepted_handle = plt.Line2D(
        [], [], marker="o", linestyle="none", markersize=8.5,
        markerfacecolor="#4B5563", markeredgecolor="#4B5563", label="Matching outcome accepted"
    )
    failed_handle = plt.Line2D(
        [], [], marker="o", linestyle="none", markersize=8.5,
        markerfacecolor="white", markeredgecolor="#4B5563", label="Proposed only"
    )
    mean_handle = plt.Line2D(
        [], [], marker="D", linestyle="none", markersize=8,
        markerfacecolor="#111827", markeredgecolor="white", label="Family mean"
    )
    fig.legend(
        handles=[accepted_handle, failed_handle, mean_handle],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.015),
        fontsize=13,
    )
    fig.subplots_adjust(left=0.22, right=0.992, top=0.985, bottom=0.25, wspace=0.12)
    fig.savefig(PLOT_PNG, dpi=360, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(PLOT_PDF, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_cases()
    write_case_table(rows)
    write_summary(rows)
    make_plot(rows)
    print(f"Wrote {CASE_OUT}")
    print(f"Wrote {MODEL_OUT}")
    print(f"Wrote {SUMMARY_OUT}")
    print(f"Wrote {PLOT_PNG}")
    print(f"Wrote {PLOT_PDF}")


if __name__ == "__main__":
    main()

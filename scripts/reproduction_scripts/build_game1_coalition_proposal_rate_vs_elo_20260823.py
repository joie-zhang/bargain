#!/usr/bin/env python3
"""Plot harmful Game 1 coalition-proposal rates against model Elo.

The numerator is the number of manually audited harmful coalition plans for
which a model was the primary organizer.  The denominator is the number of
eligible Game 1 runs with N >= 4 that contained that model.  Each run counts
once for a model even when a monoculture or Nano fleet contains several copies.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/scratch/gpfs/DANQIC/jz4391/bargain")
ASSET_DIR = ROOT / "docs/analysis/assets/minimum_winning_coalition_20260817"
CASE_PATH = ASSET_DIR / "game1_coalition_proposer_elo_cases.csv"
ELO_PATH = ROOT / "scripts/paper_figures/assets/endpoint_fairness_elo_snapshot.csv"
MANIFEST_PATHS = [
    ROOT / "analysis/llm_strategic_tag_adjudication_20260628/all_rollouts_manifest.jsonl",
    ROOT
    / "analysis/llm_strategic_tag_adjudication_random_monoculture_20260629/all_rollouts_manifest.jsonl",
]

MODEL_CSV = ASSET_DIR / "game1_coalition_proposal_rate_by_model.csv"
BIN_CSV = ASSET_DIR / "game1_coalition_proposal_rate_by_elo_bin.csv"
MODEL_PNG = ASSET_DIR / "game1_coalition_proposal_rate_vs_elo.png"
MODEL_PDF = ASSET_DIR / "game1_coalition_proposal_rate_vs_elo.pdf"
BIN_PNG = ASSET_DIR / "game1_coalition_proposal_rate_by_elo_bin.png"
BIN_PDF = ASSET_DIR / "game1_coalition_proposal_rate_by_elo_bin.pdf"
BIN_100_CSV = ASSET_DIR / "game1_coalition_proposal_rate_by_elo_bin_100.csv"
BIN_100_PNG = ASSET_DIR / "game1_coalition_proposal_rate_by_elo_bin_100.png"
BIN_100_PDF = ASSET_DIR / "game1_coalition_proposal_rate_by_elo_bin_100.pdf"
BIN_50_CSV = ASSET_DIR / "game1_coalition_proposal_rate_by_elo_bin_50.csv"
BIN_50_PNG = ASSET_DIR / "game1_coalition_proposal_rate_by_elo_bin_50.png"
BIN_50_PDF = ASSET_DIR / "game1_coalition_proposal_rate_by_elo_bin_50.pdf"

MODEL_LABELS = {
    "amazon-nova-micro-v1.0": "Nova Micro",
    "claude-3-haiku-20240307": "Claude 3 Haiku",
    "command-r-plus-08-2024": "Command R+",
    "amazon-nova-pro-v1.0": "Nova Pro",
    "gpt-4o-mini-2024-07-18": "GPT-4o Mini",
    "llama-3.3-70b-instruct": "Llama 3.3 70B",
    "gpt-4.1-nano-2025-04-14": "GPT-4.1 Nano",
    "gpt-5-nano-high": "GPT-5 Nano High",
    "gpt-4o-2024-05-13": "GPT-4o",
    "deepseek-v3": "DeepSeek V3",
    "o3-mini-high": "o3-mini High",
    "gemma-3-27b-it": "Gemma 3 27B",
    "claude-sonnet-4-20250514": "Claude Sonnet 4",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "deepseek-r1-0528": "DeepSeek R1",
    "qwen3-max-preview": "Qwen3 Max",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "claude-opus-4-5-20251101": "Claude Opus 4.5",
    "claude-opus-4-5-20251101-thinking-32k": "Opus 4.5 Thinking",
    "gpt-5.2-chat-latest-20260210": "GPT-5.2 Chat",
    "gpt-5.4-high": "GPT-5.4 High",
    "gemini-3.1-pro": "Gemini 3.1 Pro",
    "claude-opus-4-6": "Claude Opus 4.6",
    "claude-opus-4-6-thinking": "Opus 4.6 Thinking",
    "gpt-5-nano": "GPT-5 Nano",
}

FAMILY_NAMES = {
    "random_monoculture_control": "random_monoculture",
    "random_monoculture": "random_monoculture",
    "heterogeneous_random": "heterogeneous_random",
    "homogeneous_adversary": "homogeneous_adversary",
    "homogeneous_control": "homogeneous_control",
}

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return math.nan, math.nan
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials))
        / denominator
    )
    return max(0.0, 100 * (center - half_width)), min(100.0, 100 * (center + half_width))


def load_exposures() -> tuple[Counter[str], dict[str, Counter[str]]]:
    exposures: Counter[str] = Counter()
    family_exposures: dict[str, Counter[str]] = defaultdict(Counter)
    eligible_runs = 0
    for path in MANIFEST_PATHS:
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row["game_label"] != "game1" or int(row["n_agents"]) < 4:
                    continue
                eligible_runs += 1
                family = FAMILY_NAMES[row["experiment_family"]]
                for model in set(row["agent_model_map"].values()):
                    exposures[model] += 1
                    family_exposures[model][family] += 1
    if eligible_runs != 940:
        raise ValueError(f"Expected 940 eligible Game 1 runs, found {eligible_runs}")
    return exposures, family_exposures


def build_model_rows() -> list[dict[str, object]]:
    cases = read_csv(CASE_PATH)
    exposures, family_exposures = load_exposures()
    elo_map = {row["model"]: int(row["elo"]) for row in read_csv(ELO_PATH)}
    harmful = Counter(row["primary_organizer_model"] for row in cases)
    literal_zero = Counter(
        row["primary_organizer_model"] for row in cases if row["literal_zero_proposal"] == "True"
    )
    harmful_by_family = Counter(
        (row["primary_organizer_model"], row["run_family"]) for row in cases
    )
    zero_by_family = Counter(
        (row["primary_organizer_model"], row["run_family"])
        for row in cases
        if row["literal_zero_proposal"] == "True"
    )
    accepted_harmful = Counter(
        row["primary_organizer_model"]
        for row in cases
        if row["accepted_harmful_outcome"] == "True"
    )
    accepted_zero = Counter(
        row["primary_organizer_model"]
        for row in cases
        if row["accepted_literal_zero_outcome"] == "True"
    )

    if sum(harmful.values()) != 45 or sum(literal_zero.values()) != 26:
        raise ValueError("Case input no longer has 45 harmful and 26 literal-zero proposals")
    if set(harmful) - set(exposures):
        raise ValueError(f"Proposer models missing denominator: {set(harmful) - set(exposures)}")

    rows: list[dict[str, object]] = []
    for model in exposures:
        denominator = exposures[model]
        harmful_low, harmful_high = wilson_interval(harmful[model], denominator)
        zero_low, zero_high = wilson_interval(literal_zero[model], denominator)
        row = {
            "model": model,
            "model_display": MODEL_LABELS.get(model, model),
            "elo": elo_map.get(model, ""),
            "eligible_game1_runs_containing_model": denominator,
            "heterogeneous_eligible_runs": family_exposures[model]["heterogeneous_random"],
            "random_monoculture_eligible_runs": family_exposures[model]["random_monoculture"],
            "homogeneous_adversary_eligible_runs": family_exposures[model]["homogeneous_adversary"],
            "homogeneous_control_eligible_runs": family_exposures[model]["homogeneous_control"],
            "harmful_proposals": harmful[model],
            "harmful_proposal_pct": round(100 * harmful[model] / denominator, 4),
            "harmful_wilson_low_pct": round(harmful_low, 4),
            "harmful_wilson_high_pct": round(harmful_high, 4),
            "accepted_harmful_outcomes": accepted_harmful[model],
            "literal_zero_proposals": literal_zero[model],
            "literal_zero_proposal_pct": round(100 * literal_zero[model] / denominator, 4),
            "literal_zero_wilson_low_pct": round(zero_low, 4),
            "literal_zero_wilson_high_pct": round(zero_high, 4),
            "accepted_literal_zero_outcomes": accepted_zero[model],
        }
        for family in (
            "heterogeneous_random",
            "random_monoculture",
            "homogeneous_adversary",
            "homogeneous_control",
        ):
            denominator_family = family_exposures[model][family]
            family_prefix = {
                "heterogeneous_random": "heterogeneous",
                "random_monoculture": "random_monoculture",
                "homogeneous_adversary": "homogeneous_adversary",
                "homogeneous_control": "homogeneous_control",
            }[family]
            family_harmful = harmful_by_family[(model, family)]
            family_zero = zero_by_family[(model, family)]
            row[f"{family_prefix}_harmful_proposals"] = family_harmful
            row[f"{family_prefix}_harmful_proposal_pct"] = (
                round(100 * family_harmful / denominator_family, 4) if denominator_family else ""
            )
            row[f"{family_prefix}_literal_zero_proposals"] = family_zero
            row[f"{family_prefix}_literal_zero_proposal_pct"] = (
                round(100 * family_zero / denominator_family, 4) if denominator_family else ""
            )
        rows.append(row)
    return sorted(rows, key=lambda row: (row["elo"] == "", row["elo"] or 10_000, row["model"]))


def build_bin_rows(model_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    bins = [
        ("<1400", lambda elo: elo < 1400),
        ("1400–1449", lambda elo: 1400 <= elo < 1450),
        ("1450–1479", lambda elo: 1450 <= elo < 1480),
        ("≥1480", lambda elo: elo >= 1480),
    ]
    output: list[dict[str, object]] = []
    for label, contains in bins:
        selected = [row for row in model_rows if row["elo"] != "" and contains(int(row["elo"]))]
        denominator = sum(int(row["eligible_game1_runs_containing_model"]) for row in selected)
        harmful = sum(int(row["harmful_proposals"]) for row in selected)
        zero = sum(int(row["literal_zero_proposals"]) for row in selected)
        harmful_low, harmful_high = wilson_interval(harmful, denominator)
        zero_low, zero_high = wilson_interval(zero, denominator)
        output.append(
            {
                "elo_bin": label,
                "models": len(selected),
                "eligible_model_run_appearances": denominator,
                "harmful_proposals": harmful,
                "harmful_proposal_pct": round(100 * harmful / denominator, 4),
                "harmful_wilson_low_pct": round(harmful_low, 4),
                "harmful_wilson_high_pct": round(harmful_high, 4),
                "literal_zero_proposals": zero,
                "literal_zero_proposal_pct": round(100 * zero / denominator, 4),
                "literal_zero_wilson_low_pct": round(zero_low, 4),
                "literal_zero_wilson_high_pct": round(zero_high, 4),
            }
        )
    return output


def build_equal_width_bin_rows(
    model_rows: list[dict[str, object]], width: int
) -> list[dict[str, object]]:
    available = [row for row in model_rows if row["elo"] != ""]
    minimum = min(int(row["elo"]) for row in available)
    maximum = max(int(row["elo"]) for row in available)
    start = (minimum // width) * width
    stop = ((maximum // width) + 1) * width
    output: list[dict[str, object]] = []
    for lower in range(start, stop, width):
        upper_exclusive = lower + width
        selected = [
            row for row in available if lower <= int(row["elo"]) < upper_exclusive
        ]
        denominator = sum(int(row["eligible_game1_runs_containing_model"]) for row in selected)
        harmful = sum(int(row["harmful_proposals"]) for row in selected)
        zero = sum(int(row["literal_zero_proposals"]) for row in selected)
        harmful_low, harmful_high = wilson_interval(harmful, denominator)
        zero_low, zero_high = wilson_interval(zero, denominator)
        output.append(
            {
                "elo_bin": f"{lower}–{upper_exclusive - 1}",
                "elo_bin_lower": lower,
                "elo_bin_upper_inclusive": upper_exclusive - 1,
                "models": len(selected),
                "eligible_model_run_appearances": denominator,
                "harmful_proposals": harmful,
                "harmful_proposal_pct": round(100 * harmful / denominator, 4),
                "harmful_wilson_low_pct": round(harmful_low, 4),
                "harmful_wilson_high_pct": round(harmful_high, 4),
                "literal_zero_proposals": zero,
                "literal_zero_proposal_pct": round(100 * zero / denominator, 4),
                "literal_zero_wilson_low_pct": round(zero_low, 4),
                "literal_zero_wilson_high_pct": round(zero_high, 4),
            }
        )
    return output


def build_elo_50_rows_with_merged_terminal_bin(
    model_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Use 50-point bins, but combine 1450–1499 and the lone 1504 model."""
    available = [row for row in model_rows if row["elo"] != ""]
    ranges = [
        (1200, 1249),
        (1250, 1299),
        (1300, 1349),
        (1350, 1399),
        (1400, 1449),
        (1450, 1504),
    ]
    output: list[dict[str, object]] = []
    for lower, upper in ranges:
        selected = [row for row in available if lower <= int(row["elo"]) <= upper]
        denominator = sum(int(row["eligible_game1_runs_containing_model"]) for row in selected)
        harmful = sum(int(row["harmful_proposals"]) for row in selected)
        zero = sum(int(row["literal_zero_proposals"]) for row in selected)
        harmful_low, harmful_high = wilson_interval(harmful, denominator)
        zero_low, zero_high = wilson_interval(zero, denominator)
        output.append(
            {
                "elo_bin": f"{lower}–{upper}",
                "elo_bin_lower": lower,
                "elo_bin_upper_inclusive": upper,
                "models": len(selected),
                "eligible_model_run_appearances": denominator,
                "harmful_proposals": harmful,
                "harmful_proposal_pct": round(100 * harmful / denominator, 4),
                "harmful_wilson_low_pct": round(harmful_low, 4),
                "harmful_wilson_high_pct": round(harmful_high, 4),
                "literal_zero_proposals": zero,
                "literal_zero_proposal_pct": round(100 * zero / denominator, 4),
                "literal_zero_wilson_low_pct": round(zero_low, 4),
                "literal_zero_wilson_high_pct": round(zero_high, 4),
            }
        )
    return output


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "axes.linewidth": 1.15,
            "xtick.major.width": 1.15,
            "ytick.major.width": 1.15,
        }
    )


def plot_model_rates(model_rows: list[dict[str, object]]) -> None:
    configure_style()
    available = [row for row in model_rows if row["elo"] != ""]
    unavailable = [row for row in model_rows if row["elo"] == ""]
    panels = [
        ("harmful_proposal_pct", "All harmful proposals", "#2563EB"),
        ("literal_zero_proposal_pct", "Literal-zero proposals", "#DC2626"),
    ]
    annotation_positions = {
        "harmful_proposal_pct": {
            "claude-sonnet-4-20250514": (1354, 1.15),
            "claude-haiku-4-5-20251001": (1360, 4.75),
            "deepseek-r1-0528": (1395, 1.15),
            "qwen3-max-preview": (1390, 3.25),
            "gemini-2.5-pro": (1415, 1.75),
            "claude-opus-4-5-20251101": (1415, 2.65),
            "claude-opus-4-5-20251101-thinking-32k": (1440, 0.35),
            "gpt-5.4-high": (1452, 5.10),
            "gemini-3.1-pro": (1440, 12.85),
            "claude-opus-4-6": (1484, 3.20),
            "claude-opus-4-6-thinking": (1448, 3.65),
        },
        "literal_zero_proposal_pct": {
            "claude-haiku-4-5-20251001": (1360, 4.75),
            "qwen3-max-preview": (1395, 2.20),
            "claude-opus-4-5-20251101": (1415, 1.20),
            "gpt-5.4-high": (1447, 3.30),
            "gemini-3.1-pro": (1440, 8.90),
            "claude-opus-4-6-thinking": (1445, 2.15),
        },
    }
    maximum = max(float(row["harmful_proposal_pct"]) for row in available)
    y_max = math.ceil(maximum * 1.13 * 2) / 2
    fig, axes = plt.subplots(1, 2, figsize=(16.8, 6.1), sharex=True, sharey=True)

    for axis, (field, label, color) in zip(axes, panels):
        x = np.asarray([int(row["elo"]) for row in available], dtype=float)
        y = np.asarray([float(row[field]) for row in available], dtype=float)
        axis.plot(x, y, color=color, alpha=0.30, linewidth=1.35, zorder=1)
        axis.scatter(
            x, y, s=74, color=color, edgecolor="white", linewidth=0.8,
            alpha=0.80, zorder=3, clip_on=False,
        )
        for row in available:
            rate = float(row[field])
            if rate == 0:
                continue
            text_x, text_y = annotation_positions[field][row["model"]]
            axis.annotate(
                f"{row['model_display']}\n{rate:.1f}%",
                (row["elo"], rate),
                xytext=(text_x, text_y),
                textcoords="data",
                ha="left",
                va="center",
                fontsize=9.5,
                color="#111827",
                arrowprops={"arrowstyle": "-", "color": "#9CA3AF", "linewidth": 0.7},
                annotation_clip=False,
            )

        axis.set_xlabel(f"Model Elo\n{label}")
        axis.set_xlim(1225, 1515)
        axis.set_ylim(0, y_max)
        axis.set_xticks([1250, 1300, 1350, 1400, 1450, 1500])
        axis.grid(color="#D1D5DB", alpha=0.52, linewidth=0.75)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if unavailable:
            missing_text = "; ".join(
                f"{row['model_display']}: {row['harmful_proposals'] if field == 'harmful_proposal_pct' else row['literal_zero_proposals']}/{row['eligible_game1_runs_containing_model']}"
                for row in unavailable
            )
            axis.text(
                0.01, 0.965, f"Elo unavailable: {missing_text}", transform=axis.transAxes,
                ha="left", va="top", fontsize=10.5, color="#6B7280",
            )

    axes[0].set_ylabel("Eligible runs with a proposal (%)")
    fig.subplots_adjust(left=0.075, right=0.995, top=0.955, bottom=0.16, wspace=0.10)
    fig.savefig(MODEL_PNG, dpi=360, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(MODEL_PDF, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def plot_bin_rates(
    bin_rows: list[dict[str, object]],
    png_path: Path = BIN_PNG,
    pdf_path: Path = BIN_PDF,
) -> None:
    configure_style()
    labels = [row["elo_bin"] for row in bin_rows]
    x = np.arange(len(labels), dtype=float)
    figure_width = 11.8 if len(labels) >= 7 else 9.3
    fig, axis = plt.subplots(figsize=(figure_width, 5.8))
    series = [
        ("harmful_proposal_pct", "All harmful", "#2563EB", "o", -0.05),
        ("literal_zero_proposal_pct", "Literal zero", "#DC2626", "s", 0.05),
    ]
    maximum = 0.0
    for series_index, (field, label, color, marker, offset) in enumerate(series):
        values = np.asarray([float(row[field]) for row in bin_rows])
        maximum = max(maximum, float(values.max()))
        axis.plot(x + offset, values, color=color, linewidth=2.1, alpha=0.82, zorder=2)
        axis.scatter(
            x + offset, values, marker=marker, s=92, color=color,
            edgecolor="white", linewidth=0.85, label=label, zorder=3,
        )
        numerator_key = {
            "harmful_proposal_pct": "harmful_proposals",
            "literal_zero_proposal_pct": "literal_zero_proposals",
        }[field]
        for x_value, value, row in zip(x + offset, values, bin_rows):
            vertical_offset = 0.24 if series_index == 0 else 0.07
            axis.text(
                x_value, value + vertical_offset,
                f"{int(row[numerator_key])}/{int(row['eligible_model_run_appearances'])}",
                ha="center", va="bottom", fontsize=11, color=color,
            )
    axis.set_ylim(0, math.ceil(maximum * 1.18 * 2) / 2)
    axis.set_xticks(x, labels)
    axis.set_xlabel("Model Elo bin")
    axis.set_ylabel("Proposal rate (%)")
    axis.grid(axis="y", color="#D1D5DB", alpha=0.52, linewidth=0.75)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, loc="upper left", fontsize=13)
    fig.subplots_adjust(left=0.15, right=0.99, top=0.98, bottom=0.17)
    fig.savefig(png_path, dpi=360, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    model_rows = build_model_rows()
    bin_rows = build_bin_rows(model_rows)
    bin_100_rows = build_equal_width_bin_rows(model_rows, width=100)
    bin_50_rows = build_elo_50_rows_with_merged_terminal_bin(model_rows)
    write_rows(MODEL_CSV, model_rows)
    write_rows(BIN_CSV, bin_rows)
    write_rows(BIN_100_CSV, bin_100_rows)
    write_rows(BIN_50_CSV, bin_50_rows)
    plot_model_rates(model_rows)
    plot_bin_rates(bin_rows)
    plot_bin_rates(bin_100_rows, BIN_100_PNG, BIN_100_PDF)
    plot_bin_rates(bin_50_rows, BIN_50_PNG, BIN_50_PDF)
    for path in (
        MODEL_CSV,
        BIN_CSV,
        BIN_100_CSV,
        BIN_50_CSV,
        MODEL_PNG,
        MODEL_PDF,
        BIN_PNG,
        BIN_PDF,
        BIN_100_PNG,
        BIN_100_PDF,
        BIN_50_PNG,
        BIN_50_PDF,
    ):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()

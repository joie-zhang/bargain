#!/usr/bin/env python3
"""Plot Game 1 harmful-coalition conversion and voting support by family."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("/scratch/gpfs/DANQIC/jz4391/bargain")
ASSET_DIR = ROOT / "docs/analysis/assets/minimum_winning_coalition_20260817"
FAMILY_SOURCE = ASSET_DIR / "game1_coalitions_by_family.csv"
CASE_SOURCE = ASSET_DIR / "strict_proposal_case_audit.csv"
FAMILY_GAME_SOURCE = ASSET_DIR / "harmful_strict_rates_by_family_game.csv"
GAME3_ATTEMPT_SOURCE = (
    ROOT / "analysis/minimum_winning_coalition_audit_20260816/subagent_outputs/mono_game23.csv"
)
OUT_CSV = ASSET_DIR / "game1_coalition_conversion_by_family.csv"
ALL_GAMES_OUT_CSV = ASSET_DIR / "all_games_coalition_conversion_by_family.csv"
OUT_PNG = ASSET_DIR / "game1_coalition_conversion_by_family.png"
OUT_PDF = ASSET_DIR / "game1_coalition_conversion_by_family.pdf"

FAMILY_ORDER = [
    "random_monoculture",
    "heterogeneous_random",
    "homogeneous_adversary",
]
FAMILY_LABELS = {
    "random_monoculture": "Homogeneous\ncontrol",
    "heterogeneous_random": "Heterogeneous",
    "homogeneous_adversary": "Homogeneous\nadversary",
    "homogeneous_control": "Homogeneous\ncontrol\nGPT-5 Nano",
}

# The individual-ballot audit found exact support from every final planned
# coalition member in these accepted harmful cases. Config 0412 uses the final
# three-voter plan; using its earlier superseded plan would classify it as a
# replacement-voter case instead.
CLEAN_ACCEPTED_CONFIGS = {
    "random_monoculture": {107, 108, 114, 115, 117, 118, 119, 122, 123, 124},
    "heterogeneous_random": {363, 412, 756, 835},
    "homogeneous_adversary": {609},
}

ALL_GAMES_CLEAN_ACCEPTED_CASES = {
    "random_monoculture": {
        *(('game1', config_id) for config_id in CLEAN_ACCEPTED_CONFIGS['random_monoculture']),
        ("game2", 210),
    },
    "heterogeneous_random": {
        *(('game1', config_id) for config_id in CLEAN_ACCEPTED_CONFIGS['heterogeneous_random']),
        ("game2", 1417),
        ("game3", 2518),
    },
    "homogeneous_adversary": {
        *(('game1', config_id) for config_id in CLEAN_ACCEPTED_CONFIGS['homogeneous_adversary']),
    },
}

AUDIT_FAMILY_NAMES = {
    "random_monoculture": "random_monoculture",
    "random_monoculture_control": "random_monoculture",
    "heterogeneous_random": "heterogeneous_random",
    "homogeneous_adversary": "homogeneous_adversary",
    "homogeneous_control": "homogeneous_control",
}


def pct(numerator: int, denominator: int) -> float:
    return 100 * numerator / denominator if denominator else 0.0


def load_rows() -> list[dict[str, object]]:
    with FAMILY_SOURCE.open(newline="") as handle:
        family_rows = {row["family"]: row for row in csv.DictReader(handle)}

    with CASE_SOURCE.open(newline="") as handle:
        cases = list(csv.DictReader(handle))
    accepted_case_keys = {
        (row["experiment_family"], int(row["config_id"]))
        for row in cases
        if row["game"] == "game1" and row["accepted"].lower() == "true"
    }

    output: list[dict[str, object]] = []
    for family in FAMILY_ORDER:
        source = family_rows[family]
        proposed = int(source["harmful_proposed"])
        accepted = int(source["harmful_accepted"])
        clean_configs = CLEAN_ACCEPTED_CONFIGS[family]
        missing = {
            config_id
            for config_id in clean_configs
            if (family, config_id) not in accepted_case_keys
        }
        if missing:
            raise ValueError(f"Clean accepted cases missing for {family}: {sorted(missing)}")
        clean = len(clean_configs)
        replacement = accepted - clean
        not_accepted = proposed - accepted
        if min(clean, replacement, not_accepted) < 0:
            raise ValueError(f"Invalid family funnel for {family}")
        eligible = int(source["eligible_runs"])
        output.append(
            {
                "family": family,
                "family_label": FAMILY_LABELS[family].replace("\n", " "),
                "eligible_runs": eligible,
                "harmful_proposed": proposed,
                "proposed_pct_eligible": round(pct(proposed, eligible), 3),
                "harmful_accepted": accepted,
                "accepted_pct_proposed": round(pct(accepted, proposed), 3),
                "clean_accepted": clean,
                "clean_pct_eligible": round(pct(clean, eligible), 3),
                "clean_pct_proposed": round(pct(clean, proposed), 3),
                "clean_pct_accepted": round(pct(clean, accepted), 3),
                "replacement_voter_accepted": replacement,
                "replacement_voter_pct_eligible": round(
                    pct(replacement, eligible), 3
                ),
                "replacement_voter_pct_proposed": round(
                    pct(replacement, proposed), 3
                ),
                "not_accepted": not_accepted,
                "not_accepted_pct_eligible": round(
                    pct(not_accepted, eligible), 3
                ),
                "not_accepted_pct_proposed": round(pct(not_accepted, proposed), 3),
            }
        )

    expected = {
        "random_monoculture": (100, 12, 10, 10, 0, 2),
        "heterogeneous_random": (400, 28, 16, 4, 12, 12),
        "homogeneous_adversary": (400, 5, 4, 1, 3, 1),
    }
    observed = {
        str(row["family"]): (
            int(row["eligible_runs"]),
            int(row["harmful_proposed"]),
            int(row["harmful_accepted"]),
            int(row["clean_accepted"]),
            int(row["replacement_voter_accepted"]),
            int(row["not_accepted"]),
        )
        for row in output
    }
    if observed != expected:
        raise ValueError(f"Family conversion totals changed: {observed}")
    return output


def load_all_game_rows() -> list[dict[str, object]]:
    eligible_by_family = {family: 0 for family in FAMILY_ORDER}
    with FAMILY_GAME_SOURCE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            family = AUDIT_FAMILY_NAMES[row["run_family"]]
            if family in eligible_by_family:
                eligible_by_family[family] += int(row["eligible_runs"])

    proposed_by_family = {family: 0 for family in FAMILY_ORDER}
    accepted_by_family = {family: 0 for family in FAMILY_ORDER}
    accepted_case_keys: set[tuple[str, str, int]] = set()
    with CASE_SOURCE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            family = AUDIT_FAMILY_NAMES[row["experiment_family"]]
            if family not in proposed_by_family:
                continue
            game = row["game"]
            config_id = int(row["config_id"])
            proposed_by_family[family] += 1
            if row["accepted"].lower() == "true":
                accepted_by_family[family] += 1
                accepted_case_keys.add((family, game, config_id))

    # The allocation-level case audit omits Game 3 config 0270 because the
    # selected agreement broadened from the proposed 3-of-4 coalition to 4-of-4.
    # Count the explicit harmful coalition proposal as a failed proposal.
    with GAME3_ATTEMPT_SOURCE.open(newline="") as handle:
        attempt_rows = [
            row for row in csv.DictReader(handle) if row["config_id"] == "config_0270"
        ]
    if len(attempt_rows) != 1:
        raise ValueError(f"Expected one config_0270 row, found {len(attempt_rows)}")
    attempt = attempt_rows[0]
    expected_attempt = {
        "game_label": "game3",
        "explicit_exact_subset_attempt": "True",
        "exact_targeted_coalition_selected": "False",
    }
    for field, expected_value in expected_attempt.items():
        if attempt[field] != expected_value:
            raise ValueError(
                f"config_0270 {field} changed: {attempt[field]!r} != {expected_value!r}"
            )
    proposed_by_family["random_monoculture"] += 1

    output: list[dict[str, object]] = []
    for family in FAMILY_ORDER:
        proposed = proposed_by_family[family]
        accepted = accepted_by_family[family]
        clean_cases = ALL_GAMES_CLEAN_ACCEPTED_CASES[family]
        missing = {
            (game, config_id)
            for game, config_id in clean_cases
            if (family, game, config_id) not in accepted_case_keys
        }
        if missing:
            raise ValueError(
                f"Full-support accepted cases missing for {family}: {sorted(missing)}"
            )
        clean = len(clean_cases)
        replacement = accepted - clean
        not_accepted = proposed - accepted
        eligible = eligible_by_family[family]
        output.append(
            {
                "family": family,
                "family_label": FAMILY_LABELS[family].replace("\n", " "),
                "eligible_runs": eligible,
                "harmful_proposed": proposed,
                "proposed_pct_eligible": round(pct(proposed, eligible), 3),
                "harmful_accepted": accepted,
                "accepted_pct_proposed": round(pct(accepted, proposed), 3),
                "clean_accepted": clean,
                "clean_pct_eligible": round(pct(clean, eligible), 3),
                "clean_pct_proposed": round(pct(clean, proposed), 3),
                "clean_pct_accepted": round(pct(clean, accepted), 3),
                "replacement_voter_accepted": replacement,
                "replacement_voter_pct_eligible": round(
                    pct(replacement, eligible), 3
                ),
                "replacement_voter_pct_proposed": round(
                    pct(replacement, proposed), 3
                ),
                "not_accepted": not_accepted,
                "not_accepted_pct_eligible": round(
                    pct(not_accepted, eligible), 3
                ),
                "not_accepted_pct_proposed": round(
                    pct(not_accepted, proposed), 3
                ),
            }
        )

    expected = {
        "random_monoculture": (260, 14, 11, 11, 0, 3),
        "heterogeneous_random": (1040, 33, 18, 6, 12, 15),
        "homogeneous_adversary": (1040, 6, 4, 1, 3, 2),
    }
    observed = {
        str(row["family"]): (
            int(row["eligible_runs"]),
            int(row["harmful_proposed"]),
            int(row["harmful_accepted"]),
            int(row["clean_accepted"]),
            int(row["replacement_voter_accepted"]),
            int(row["not_accepted"]),
        )
        for row in output
    }
    if observed != expected:
        raise ValueError(f"All-game family conversion totals changed: {observed}")
    return output


def write_csv(
    rows: list[dict[str, object]], output_path: Path = OUT_CSV
) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_family_conversion(
    axis: plt.Axes,
    rows: list[dict[str, object]],
    *,
    compact: bool = False,
    y_max: float = 17,
    y_label: str = "Percentage of runs with a coalition proposal",
    legend_title: str | None = None,
    show_segment_counts: bool = True,
    annotation_fontsize: float = 9.2,
    legend_fontsize: float | None = None,
    xtick_fontsize: float | None = None,
    annotation_x_offsets: list[float] | None = None,
    x_positions: list[float] | None = None,
    x_limits: tuple[float, float] | None = None,
) -> None:
    x = x_positions or [0.0, 1.15, 2.3]
    if len(x) != len(rows):
        raise ValueError("x_positions must match the number of rows")
    clean = [float(row["clean_pct_eligible"]) for row in rows]
    replacement = [float(row["replacement_voter_pct_eligible"]) for row in rows]
    failed = [float(row["not_accepted_pct_eligible"]) for row in rows]

    clean_bars = axis.bar(
        x,
        clean,
        width=0.58,
        color="#1D4ED8",
        edgecolor="white",
        linewidth=1.0,
        zorder=3,
    )
    replacement_bars = axis.bar(
        x,
        replacement,
        width=0.58,
        bottom=clean,
        color="#60A5FA",
        edgecolor="white",
        linewidth=1.0,
        zorder=3,
    )
    accepted_height = [a + b for a, b in zip(clean, replacement)]
    failed_bars = axis.bar(
        x,
        failed,
        width=0.58,
        bottom=accepted_height,
        color="#B8C0CC",
        edgecolor="white",
        linewidth=1.0,
        zorder=3,
    )

    proposal_heights = [a + b for a, b in zip(accepted_height, failed)]
    for index, total_height in enumerate(proposal_heights):
        if total_height == 0:
            for bars in (clean_bars, replacement_bars, failed_bars):
                bars.patches[index].set_visible(False)
    if annotation_x_offsets is None:
        annotation_x_offsets = [0.0] * len(rows)
    if len(annotation_x_offsets) != len(rows):
        raise ValueError("annotation_x_offsets must match the number of rows")
    for center, total_height, row, x_offset in zip(
        x, proposal_heights, rows, annotation_x_offsets
    ):
        annotation_text = row.get("annotation_text") or (
            f"{row['clean_accepted']} of {row['harmful_proposed']} accepted\n"
            "with full coalition\n"
            f"support ({float(row['clean_pct_proposed']):.1f}%)"
        )
        axis.text(
            center + x_offset,
            total_height + 0.35,
            annotation_text,
            ha="center",
            va="bottom",
            fontsize=annotation_fontsize,
            color="#1E3A8A",
            linespacing=1.04,
        )

    if show_segment_counts:
        segment_specs = [
            (clean, [0.0] * len(rows), "clean_accepted", "white"),
            (replacement, clean, "replacement_voter_accepted", "#0F172A"),
            (failed, accepted_height, "not_accepted", "#374151"),
        ]
        for heights, bottoms, count_field, color in segment_specs:
            for center, height, bottom, row in zip(x, heights, bottoms, rows):
                count = int(row[count_field])
                if count == 0:
                    continue
                axis.text(
                    center,
                    bottom + height / 2,
                    str(count),
                    ha="center",
                    va="center",
                    fontsize=7.2 if height < 0.5 else 9.5,
                    color=color,
                    weight="bold",
                )

    axis.legend(
        [
            "Accepted, full coalition support",
            "Accepted, outside votes needed",
            "Not accepted",
        ],
        loc="upper right",
        frameon=False,
        fontsize=legend_fontsize or (10.5 if compact else 9.5),
        title=legend_title,
        title_fontsize=legend_fontsize or (10.5 if compact else 9.5),
    )

    axis.set_xticks(x, [FAMILY_LABELS[str(row["family"])] for row in rows])
    axis.set_ylabel(y_label)
    axis.set_xlim(*(x_limits or (-0.52, 2.65)))
    axis.set_ylim(0, y_max)
    axis.set_yticks(
        [value for value in range(2, math.ceil(y_max) + 1, 2) if value < y_max]
    )
    if compact:
        axis.tick_params(axis="x", labelsize=xtick_fontsize or 12.5)
    axis.grid(axis="y", color="#D1D5DB", alpha=0.52, linewidth=0.75)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def make_plot(rows: list[dict[str, object]]) -> None:
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
    fig, axis = plt.subplots(figsize=(6.2, 8.6))
    draw_family_conversion(axis, rows)
    fig.subplots_adjust(left=0.19, right=0.99, top=0.98, bottom=0.11)
    fig.savefig(OUT_PNG, dpi=360, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    rows = load_rows()
    all_game_rows = load_all_game_rows()
    write_csv(rows)
    write_csv(all_game_rows, ALL_GAMES_OUT_CSV)
    make_plot(rows)
    for path in (OUT_CSV, ALL_GAMES_OUT_CSV, OUT_PNG, OUT_PDF):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate a text-only Markdown edition of all ICML AIWILD main figures."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/icml_aiwild_main_figures_ascii.md"

PRIMARY = (
    ROOT
    / "experiments/results/n2_baseline_comparison_analysis_20260505"
    / "primary_runs_with_metrics.csv"
)
FAIRSHARE = (
    ROOT
    / "overleaf/icml_aiwild_template/graphics/n2_gpt5_nano"
    / "fairshare_residual_combined_summary.csv"
)
N2_CORR = (
    ROOT
    / "overleaf/icml_aiwild_template/graphics/qualitative_n2"
    / "n2_group_payoff_corr_dedup.csv"
)
N2_INTENSITY = (
    ROOT
    / "overleaf/icml_aiwild_template/graphics/qualitative_n2"
    / "n2_group_intensity_dedup.csv"
)
TTC_PAYOFF = ROOT / "analysis/neurips_revision_20260504/ttc_game_averaged_by_effort.csv"
TTC_INTENSITY = (
    ROOT
    / "analysis/ttc_group_intensity_turn_dedup_verification_20260701"
    / "ttc_group_intensity_turn_dedup_summary.csv"
)
HET_AGENTS = (
    ROOT
    / "experiments/results/n2_plus_multiagent_comparison_analysis_20260505"
    / "tables_multiagent/heterogeneous_agents_fresh.csv"
)
HET_SLOPES = (
    ROOT
    / "experiments/results/n2_plus_multiagent_comparison_analysis_20260505"
    / "tables_multiagent/heterogeneous_payoff_vs_arena_elo_slopes.csv"
)
GINI_SUMMARY = ROOT / "analysis/recreated_figures/figure7_from_script/gini_summary.csv"
GINI_HOM_RUNS = (
    ROOT / "analysis/recreated_figures/figure7_from_script/homogeneous_gini_run_metrics.csv"
)
HOM_GINI = (
    ROOT
    / "overleaf/neurips/graphics/n_gt_2_report"
    / "homogeneous_adversary_baseline_only_vs_all_payoff_gini_summary.csv"
)
HOM_ROLE = (
    ROOT
    / "overleaf/neurips/graphics/n_gt_2_report"
    / "role_payoff_with_within_run_variance_bars_summary.csv"
)

GAME_NAMES = {
    "game1": "G1 Item Allocation",
    "game2": "G2 Diplomatic Treaty",
    "game3": "G3 Co-funding",
}
CAT_ORDER = [
    "trade/compromise",
    "emotional persuasion",
    "logical persuasion",
    "pressure",
    "self-interest/exploitation",
    "formalization",
]
CAT_SHORT = {
    "trade/compromise": "Trade / compromise",
    "emotional persuasion": "Emotional persuasion",
    "logical persuasion": "Logical persuasion",
    "pressure": "Pressure",
    "self-interest/exploitation": "Self-interest / exploitation",
    "formalization": "Formalization",
}


def sem(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return 0.0 if len(clean) <= 1 else float(clean.std(ddof=1) / math.sqrt(len(clean)))


def linear_stats(x: Sequence[float], y: Sequence[float]) -> dict[str, float]:
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    mask = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[mask], yv[mask]
    slope, intercept = np.polyfit(xv, yv, 1)
    fitted = slope * xv + intercept
    residual = yv - fitted
    sse = float(np.sum(residual**2))
    sst = float(np.sum((yv - yv.mean()) ** 2))
    slope_se = math.sqrt((sse / (len(xv) - 2)) / float(np.sum((xv - xv.mean()) ** 2)))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "slope_se": slope_se,
        "r2": 1.0 - sse / sst if sst else 1.0,
    }


def md_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    escaped = [[str(value).replace("|", "\\|") for value in row] for row in rows]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(":--" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in escaped)
    return "\n".join(lines)


def fenced(lines: Sequence[str]) -> str:
    return "```\n" + "\n".join(line.rstrip() for line in lines) + "\n```"


def collision_mapper(
    values: Iterable[float],
    *,
    width: int,
    xmin: float,
    xmax: float,
) -> tuple[dict[float, int], Callable[[float], int]]:
    """Assign distinct, order-preserving columns while minimizing displacement."""
    observed = np.asarray(sorted(set(float(value) for value in values)), dtype=float)
    if len(observed) > width:
        raise ValueError(f"{len(observed)} x values cannot occupy {width} distinct columns")
    ideal = (observed - xmin) / (xmax - xmin) * (width - 1)
    n = len(observed)
    dp = np.full((n, width), np.inf)
    prev = np.full((n, width), -1, dtype=int)
    dp[0] = (np.arange(width) - ideal[0]) ** 2
    for index in range(1, n):
        best = np.inf
        best_index = -1
        for column in range(width):
            if column > 0 and dp[index - 1, column - 1] < best:
                best = dp[index - 1, column - 1]
                best_index = column - 1
            if best_index >= 0:
                dp[index, column] = best + (column - ideal[index]) ** 2
                prev[index, column] = best_index
    column = int(np.argmin(dp[-1]))
    columns = [column]
    for index in range(n - 1, 0, -1):
        column = int(prev[index, column])
        columns.append(column)
    columns = np.asarray(columns[::-1], dtype=int)
    exact = dict(zip(observed.tolist(), columns.tolist()))

    def mapper(value: float) -> int:
        return int(round(float(np.interp(float(value), observed, columns))))

    return exact, mapper


def linear_mapper(
    *, width: int, xmin: float, xmax: float
) -> tuple[dict[float, int], Callable[[float], int]]:
    def mapper(value: float) -> int:
        column = round((float(value) - xmin) / (xmax - xmin) * (width - 1))
        return max(0, min(width - 1, int(column)))

    return {}, mapper


def categorical_mapper(
    values: Sequence[object], width: int
) -> tuple[dict[object, int], Callable[[object], int]]:
    columns = np.rint(np.linspace(2, width - 3, len(values))).astype(int)
    exact = dict(zip(values, columns.tolist()))

    def mapper(value: object) -> int:
        return exact[value]

    return exact, mapper


def render_plot(
    series: Sequence[dict[str, object]],
    *,
    width: int,
    height: int,
    ymin: float,
    ymax: float,
    x_mapper: Callable[[object], int],
    x_ticks: Sequence[tuple[object, str]],
    y_ticks: Sequence[float],
    x_label: str,
    y_format: Callable[[float], str] | None = None,
    hline: float | None = None,
    bars: Sequence[dict[str, object]] | None = None,
) -> list[str]:
    grid = [[" "] * width for _ in range(height + 1)]

    def yrow(value: float) -> int:
        row = round((ymax - float(value)) / (ymax - ymin) * height)
        return max(0, min(height, int(row)))

    def put(row: int, column: int, char: str) -> None:
        old = grid[row][column]
        low_priority = {"-", ".", ":"}
        if old == " " or (old in low_priority and char not in low_priority):
            grid[row][column] = char
        elif char in low_priority and old not in low_priority:
            return
        elif old != char:
            grid[row][column] = "*"

    def draw(a: tuple[int, int], b: tuple[int, int], char: str) -> None:
        x0, y0 = a
        x1, y1 = b
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for step in range(steps + 1):
            x = round(x0 + (x1 - x0) * step / steps)
            y = round(y0 + (y1 - y0) * step / steps)
            put(y, x, char)

    if hline is not None and ymin <= hline <= ymax:
        row = yrow(hline)
        for column in range(width):
            put(row, column, "-")

    if bars:
        for bar in bars:
            column = x_mapper(bar["x"])
            base = yrow(float(bar.get("base", ymin)))
            top = yrow(float(bar["y"]))
            for row in range(min(base, top), max(base, top) + 1):
                put(row, column, str(bar.get("char", "#")))

    for item in series:
        xs = list(item["x"])
        ys = [float(value) for value in item["y"]]
        char = str(item.get("char", "o"))
        line_char = str(item.get("line_char", char))
        points = [(x_mapper(x), yrow(y)) for x, y in zip(xs, ys)]
        if item.get("connect", False):
            for first, second in zip(points, points[1:]):
                draw(first, second, line_char)
        yerr = item.get("yerr")
        if yerr is not None:
            for x, y, error in zip(xs, ys, yerr):
                column = x_mapper(x)
                upper = yrow(y + float(error))
                lower = yrow(y - float(error))
                for row in range(min(upper, lower), max(upper, lower) + 1):
                    put(row, column, ":")
        if item.get("points", True):
            for column, row in points:
                put(row, column, char)

    formatter = y_format or (lambda value: f"{value:.0f}")
    label_by_row: dict[int, str] = {}
    for tick in y_ticks:
        label_by_row[yrow(float(tick))] = formatter(float(tick))
    label_width = max(3, max((len(value) for value in label_by_row.values()), default=3))
    lines = []
    for row, values in enumerate(grid):
        label = label_by_row.get(row, "").rjust(label_width)
        lines.append(f"{label} |{''.join(values)}|")
    lines.append(" " * (label_width + 1) + "+" + "-" * width + "+")

    tick_chars = [" "] * width
    for value, label in x_ticks:
        center = x_mapper(value)
        start = max(0, min(width - len(label), center - len(label) // 2))
        for offset, char in enumerate(label):
            tick_chars[start + offset] = char
    lines.append(" " * (label_width + 2) + "".join(tick_chars))
    lines.append(" " * (label_width + 2) + x_label.center(width))
    return lines


def combine_panels(
    panels: Sequence[tuple[str, list[str]]],
    *,
    columns: int,
    gap: str = "   ",
) -> list[str]:
    output: list[str] = []
    for start in range(0, len(panels), columns):
        row = list(panels[start : start + columns])
        widths = [max(len(line) for line in lines) for _, lines in row]
        output.append(gap.join(title.center(width) for (title, _), width in zip(row, widths)))
        max_lines = max(len(lines) for _, lines in row)
        for line_index in range(max_lines):
            parts = []
            for (_, lines), width in zip(row, widths):
                value = lines[line_index] if line_index < len(lines) else ""
                parts.append(value.ljust(width))
            output.append(gap.join(parts).rstrip())
        if start + columns < len(panels):
            output.append("")
    return output


def effort_label(level: str) -> str:
    return {"minimal": "min", "low": "low", "medium": "med", "high": "high", "max": "max"}[level]


def figure_1() -> str:
    diagram = [
        "+----------------------+       +-------------------------+",
        "| PRIVATE PREFERENCES  |       | PUBLIC NEGOTIATION LOOP |",
        "| Agent 1: Stone 43    |       +-------------------------+",
        "|          Apple 33    |                    |",
        "+----------+-----------+                    v",
        "           +------------------------> [1. START]",
        "                                           |",
        "                                           v",
        "                                    [2. DISCUSSION]",
        "                                     share / conceal",
        "                                     preferences",
        "                                           |",
        "                                           v",
        "                                    [3. PROPOSALS]",
        "                                     allocation +",
        "                                     stated reasoning",
        "                                           |",
        "                                           v",
        "                                    [4. VOTING]",
        "                                  accept / reject",
        "                                      /       \\",
        "                               unanimous       rejected",
        "                                  |               |",
        "                                  v               v",
        "                            [FINAL OUTCOME] [5. REFLECTION]",
        "                              utilities       diagnose round",
        "                                                 |",
        "                                                 +----> back to",
        "                                                       discussion",
    ]
    table = md_table(
        ["Stage", "Measured state", "Example from the figure"],
        [
            ["Initialization", "Private utility vector", "Stone 43; Apple 33; Jewel 5"],
            ["Discussion", "Natural-language transcript", "Agents reveal or frame preferences"],
            ["Proposal", "Structured allocation + rationale", "Agent 1 requests Apple and Stone"],
            ["Voting", "Per-agent accept/reject", "Agent 1 accepts; Agent 2 rejects"],
            ["Reflection", "Private diagnosis", "Agent 1 identifies a promising non-overlap"],
            ["Termination", "Allocation, utilities, consensus, rounds", "Unanimous acceptance"],
        ],
    )
    return (
        "## Figure 1 — Negotiation episode workflow\n\n"
        "**Old NeurIPS relationship:** exact old Figure 1; the number is unchanged.\n\n"
        + fenced(diagram)
        + "\n\n"
        + table
        + "\n\n**Reading:** private preferences feed a public negotiation cycle; rejection "
        "causes reflection and another round, while unanimous acceptance terminates."
    )


def load_primary() -> pd.DataFrame:
    frame = pd.read_csv(PRIMARY)
    return frame[frame["baseline_key"].eq("gpt5_nano")].copy()


def figure_2() -> str:
    frame = load_primary()
    elos = sorted(frame["adversary_elo"].unique())
    exact, mapper = collision_mapper(elos, width=40, xmin=1088, xmax=1512)
    _ = exact
    panel_a = []
    slope_rows = []
    ranges = {
        "game1": (35.0, 90.0, list(range(40, 91, 10))),
        "game2": (35.0, 105.0, list(range(40, 101, 10))),
        "game3": (-10.0, 55.0, list(range(-10, 51, 10))),
    }
    for game in ["game1", "game2", "game3"]:
        game_frame = frame[frame["game_id"].eq(game)]
        aggregate = (
            game_frame.groupby(["adversary_model", "adversary_elo"], as_index=False)
            .agg(mean=("adversary_utility", "mean"), error=("adversary_utility", sem))
            .sort_values("adversary_elo")
        )
        stats = linear_stats(aggregate["adversary_elo"], aggregate["mean"])
        fit_x = np.linspace(min(elos), max(elos), 120)
        fit_y = stats["slope"] * fit_x + stats["intercept"]
        ymin, ymax, yticks = ranges[game]
        lines = render_plot(
            [
                {
                    "x": fit_x,
                    "y": fit_y,
                    "char": ".",
                    "line_char": ".",
                    "connect": True,
                    "points": False,
                },
                {
                    "x": aggregate["adversary_elo"],
                    "y": aggregate["mean"],
                    "yerr": aggregate["error"],
                    "char": "o",
                    "connect": False,
                },
            ],
            width=40,
            height=36,
            ymin=ymin,
            ymax=ymax,
            x_mapper=mapper,
            x_ticks=[(1100, "1100"), (1300, "1300"), (1500, "1500")],
            y_ticks=yticks,
            x_label="Adversary Elo (ordered)",
        )
        panel_a.append((GAME_NAMES[game], lines))
        half = 1.96 * stats["slope_se"] * 100
        slope_rows.append(
            [
                GAME_NAMES[game],
                "30",
                f"{stats['slope'] * 100:+.2f}",
                f"[{stats['slope'] * 100 - half:.2f}, {stats['slope'] * 100 + half:.2f}]",
                f"mean ± {aggregate['error'].median():.2f}",
                f"{stats['r2']:.2f}",
            ]
        )

    panel_b = []
    smooth_by_game: dict[str, dict[str, pd.DataFrame]] = {}
    for game in ["game1", "game2", "game3"]:
        game_frame = frame[frame["game_id"].eq(game)]
        curves = {}
        for endpoint, value in [
            ("cooperative", float(game_frame["competition_value"].min())),
            ("competitive", float(game_frame["competition_value"].max())),
        ]:
            aggregate = (
                game_frame[np.isclose(game_frame["competition_value"].astype(float), value)]
                .groupby("adversary_elo", as_index=False)
                .agg(mean=("baseline_utility", "mean"), error=("baseline_utility", sem))
                .sort_values("adversary_elo")
            )
            aggregate["smooth"] = aggregate["mean"].ewm(alpha=0.24, adjust=False).mean()
            curves[endpoint] = aggregate
        smooth_by_game[game] = curves
        ymin, ymax = (-5.0, 105.0) if game != "game3" else (-5.0, 65.0)
        y_ticks = (
            [-5, *range(0, 101, 10), 105]
            if game != "game3"
            else [-5, *range(0, 61, 10), 65]
        )
        lines = render_plot(
            [
                {
                    "x": curves["cooperative"]["adversary_elo"],
                    "y": curves["cooperative"]["smooth"],
                    "char": "C",
                    "connect": True,
                },
                {
                    "x": curves["competitive"]["adversary_elo"],
                    "y": curves["competitive"]["smooth"],
                    "char": "x",
                    "connect": True,
                },
            ],
            width=40,
            height=60,
            ymin=ymin,
            ymax=ymax,
            x_mapper=mapper,
            x_ticks=[(1100, "1100"), (1300, "1300"), (1500, "1500")],
            y_ticks=y_ticks,
            x_label="Adversary Elo (ordered)",
        )
        panel_b.append((GAME_NAMES[game], lines))

    models = (
        frame[["adversary_model", "adversary_elo"]]
        .drop_duplicates()
        .sort_values(["adversary_elo", "adversary_model"])
        .reset_index(drop=True)
    )
    models["quartile"] = np.minimum(
        np.floor(np.arange(len(models)) * 4 / len(models)).astype(int) + 1, 4
    )
    with_quartile = frame.merge(models, on=["adversary_model", "adversary_elo"])
    endpoint_rows = []
    for game in ["game1", "game2", "game3"]:
        game_frame = with_quartile[with_quartile["game_id"].eq(game)]
        for quartile in range(1, 5):
            subset = game_frame[game_frame["quartile"].eq(quartile)]
            elo_range = models[models["quartile"].eq(quartile)]["adversary_elo"]
            values = []
            for competition in [
                float(subset["competition_value"].min()),
                float(subset["competition_value"].max()),
            ]:
                payoffs = subset[
                    np.isclose(subset["competition_value"].astype(float), competition)
                ]["baseline_utility"]
                values.append(f"{payoffs.mean():.1f} ± {sem(payoffs):.1f}")
            endpoint_rows.append(
                [
                    game.upper().replace("GAME", "G"),
                    f"Q{quartile} ({int(elo_range.min())}–{int(elo_range.max())})",
                    values[0],
                    values[1],
                ]
            )

    return "\n\n".join(
        [
            "## Figure 2 — Bilateral capability scaling against GPT-5-nano\n\n"
            "**Old NeurIPS relationship:** no exact counterpart; closest are old "
            "main-text Figures 2–3.",
            "### Panel (a): adversary payoff\n\n"
            "`o` = model mean; `:` = ±SEM; `.` = fitted trend; `*` = overlap.\n\n"
            + fenced(combine_panels(panel_a, columns=3)),
            md_table(
                [
                    "Game",
                    "Model means",
                    "Payoff / 100 Elo",
                    "95% slope CI",
                    "Typical point",
                    "R²",
                ],
                slope_rows,
            ),
            "### Panel (b): baseline payoff at competition endpoints\n\n"
            "`C` = maximally cooperative EWM; `x` = maximally competitive EWM; "
            "`*` = overlap. All 30 distinct Elo values occupy distinct columns; "
            "local x-spacing is collision-resolved.\n\n"
            + fenced(combine_panels(panel_b, columns=3)),
            md_table(
                [
                    "Game",
                    "Adversary Elo band",
                    "Max-cooperative payoff",
                    "Max-competitive payoff",
                ],
                endpoint_rows,
            ),
            "**Reading:** adversary payoff rises in every game. Cooperation also "
            "lifts the baseline, whereas competition produces a flat-to-declining "
            "baseline curve.",
        ]
    )


def figure_3() -> str:
    summary = pd.read_csv(FAIRSHARE)
    bilateral = summary[summary["panel"].eq("bilateral")].copy()
    elos = sorted(bilateral["reference_elo"].dropna().unique())
    _, mapper = collision_mapper(elos, width=40, xmin=1088, xmax=1512)
    panels = []
    table_rows = []
    for game in ["game1", "game2", "game3"]:
        subset = bilateral[bilateral["game_id"].eq(game)].sort_values("reference_elo")
        stats = linear_stats(subset["reference_elo"], subset["residual_mean"])
        fit_x = np.linspace(subset["reference_elo"].min(), subset["reference_elo"].max(), 120)
        fit_y = stats["slope"] * fit_x + stats["intercept"]
        panels.append(
            (
                GAME_NAMES[game],
                render_plot(
                    [
                        {
                            "x": fit_x,
                            "y": fit_y,
                            "char": ".",
                            "connect": True,
                            "points": False,
                        },
                        {
                            "x": subset["reference_elo"],
                            "y": subset["residual_mean"],
                            "char": "o",
                        },
                    ],
                    width=40,
                    height=40,
                    ymin=-30,
                    ymax=15,
                    x_mapper=mapper,
                    x_ticks=[(1100, "1100"), (1300, "1300"), (1500, "1500")],
                    y_ticks=list(range(-30, 16, 5)),
                    x_label="Reference Elo (ordered)",
                    hline=0,
                ),
            )
        )
        lo = float(subset["reference_elo"].min())
        hi = float(subset["reference_elo"].max())
        crossing = -stats["intercept"] / stats["slope"]
        table_rows.append(
            [
                f"Bilateral — {game.upper().replace('GAME', 'G')}",
                f"{lo:.0f}–{hi:.0f}",
                f"{stats['intercept'] + stats['slope'] * lo:+.2f}",
                f"{stats['intercept'] + stats['slope'] * hi:+.2f}",
                f"{stats['slope'] * 100:+.2f}",
                f"{crossing:.0f}",
            ]
        )

    multi = summary[summary["panel"].eq("multiagent")].copy()
    multi_elos = sorted(multi["reference_elo"].dropna().unique())
    _, multi_mapper = collision_mapper(multi_elos, width=55, xmin=1220, xmax=1512)
    styles = {
        "heterogeneous_agent": ("H", True),
        "homogeneous_adversary_adversary": ("A", True),
        "homogeneous_adversary_baseline": ("B", True),
        "homogeneous_control": ("O", False),
    }
    multi_series = []
    labels = {
        "heterogeneous_agent": "Multi-agent — heterogeneous focal",
        "homogeneous_adversary_adversary": "Multi-agent — inserted adversary",
        "homogeneous_adversary_baseline": "Multi-agent — baseline mean",
        "homogeneous_control": "Multi-agent — GPT-5-nano control",
    }
    for key, (char, connect) in styles.items():
        subset = multi[multi["series_key"].eq(key)].sort_values("reference_elo")
        multi_series.append(
            {
                "x": subset["reference_elo"],
                "y": subset["residual_mean"],
                "char": char,
                "connect": connect,
            }
        )
        if len(subset) > 1:
            stats = linear_stats(subset["reference_elo"], subset["residual_mean"])
            lo = float(subset["reference_elo"].min())
            hi = float(subset["reference_elo"].max())
            crossing = -stats["intercept"] / stats["slope"]
            crossing_text = (
                f"{crossing:.0f}" if lo <= crossing <= hi else "No crossing in range"
            )
            table_rows.append(
                [
                    labels[key],
                    f"{lo:.0f}–{hi:.0f}",
                    f"{stats['intercept'] + stats['slope'] * lo:+.2f}",
                    f"{stats['intercept'] + stats['slope'] * hi:+.2f}",
                    f"{stats['slope'] * 100:+.2f}",
                    crossing_text,
                ]
            )
        else:
            value = float(subset["residual_mean"].iloc[0])
            elo = float(subset["reference_elo"].iloc[0])
            table_rows.append(
                [labels[key], f"{elo:.0f}", f"{value:+.2f}", f"{value:+.2f}", "—", "—"]
            )
    multi_plot = render_plot(
        multi_series,
        width=55,
        height=40,
        ymin=-12,
        ymax=5,
        x_mapper=multi_mapper,
        x_ticks=[(1240, "1240"), (1389, "1389"), (1504, "1504")],
        y_ticks=list(range(-12, 5, 2)),
        x_label="Reference Elo (ordered)",
        hline=0,
    )

    return "\n\n".join(
        [
            "## Figure 3 — Utility relative to fair share\n\n"
            "**Old NeurIPS relationship:** no exact counterpart; closest are old "
            "Figure 5 (main text) and Figure 29 (appendix).",
            "### Bilateral panel\n\n`o` = model residual; `.` = fitted trend; "
            "`-` = zero/fair-share line.\n\n"
            + fenced(combine_panels(panels, columns=3)),
            "### Multi-agent panel\n\n`H` = heterogeneous focal agent; `A` = "
            "inserted adversary; `B` = baseline-agent mean; `O` = homogeneous "
            "control; `-` = zero/fair-share line.\n\n"
            + fenced(multi_plot),
            md_table(
                [
                    "Series",
                    "Elo span",
                    "Fitted low-Elo residual",
                    "Fitted high-Elo residual",
                    "Δ / 100 Elo",
                    "Zero crossing",
                ],
                table_rows,
            ),
            "**Reading:** focal-agent trends rise through zero around Elo 1400–1470. "
            "Baseline agents improve only mildly and remain below fair share.",
        ]
    )


def correlation_bars(correlations: pd.DataFrame) -> list[str]:
    xmin, xmax, width = -0.40, 0.20, 60
    zero = round((0 - xmin) / (xmax - xmin) * (width - 1))
    lines = ["Payoff correlation (Spearman rho)".center(90)]
    axis = ["-"] * width
    axis[zero] = "+"
    lines.append(" " * 27 + "".join(axis))
    for category in CAT_ORDER:
        value = float(
            correlations.loc[
                correlations["category"].eq(category), "spearman_event_count_r_utility"
            ].iloc[0]
        )
        end = round((value - xmin) / (xmax - xmin) * (width - 1))
        row = [" "] * width
        for column in range(min(zero, end), max(zero, end) + 1):
            row[column] = "="
        row[zero] = "|"
        row[end] = "o"
        lines.append(f"{CAT_SHORT[category]:<25} {value:+.2f} {''.join(row)}")
    labels = [" "] * width
    for value in [-0.4, -0.2, 0.0, 0.2]:
        text = f"{value:+.1f}"
        center = round((value - xmin) / (xmax - xmin) * (width - 1))
        start = max(0, min(width - len(text), center - len(text) // 2))
        labels[start : start + len(text)] = text
    lines.append(" " * 27 + "".join(labels))
    return lines


def figure_4() -> str:
    corr = pd.read_csv(N2_CORR)
    intensity = pd.read_csv(N2_INTENSITY)
    panel_specs = {
        "trade/compromise": 2.2,
        "emotional persuasion": 0.8,
        "logical persuasion": 2.0,
        "pressure": 2.2,
        "self-interest/exploitation": 0.8,
        "formalization": 0.30,
    }
    panels = []
    rows = []
    for category in CAT_ORDER:
        subset = (
            intensity[intensity["category"].eq(category)]
            .sort_values(["speaker_elo", "speaker_model"])
            .reset_index(drop=True)
        )
        baseline = subset["intensity"].rolling(5, center=True, min_periods=1).median()
        subset = subset.drop(index=(subset["intensity"] - baseline).abs().idxmax())
        subset = subset.sort_values(["speaker_elo", "speaker_model"]).reset_index(drop=True)
        subset["smooth"] = subset["intensity"].rolling(5, center=True, min_periods=1).mean()
        _, mapper = collision_mapper(
            subset["speaker_elo"], width=38, xmin=1090, xmax=1524
        )
        ymax = panel_specs[category]
        panels.append(
            (
                CAT_SHORT[category],
                render_plot(
                    [
                        {
                            "x": subset["speaker_elo"],
                            "y": subset["smooth"],
                            "char": "o",
                            "connect": True,
                        }
                    ],
                    width=38,
                    height=20,
                    ymin=0,
                    ymax=ymax,
                    x_mapper=mapper,
                    x_ticks=[(1110, "1110"), (1350, "1350"), (1504, "1504")],
                    y_ticks=np.linspace(0, ymax, 5),
                    y_format=lambda value: f"{value:g}",
                    x_label="Adversary Elo (ordered)",
                ),
            )
        )
        payoff = corr[corr["category"].eq(category)].iloc[0]
        stats = linear_stats(subset["speaker_elo"], subset["intensity"])
        rows.append(
            [
                CAT_SHORT[category],
                f"{float(payoff['spearman_event_count_r_utility']):+.2f}",
                f"{float(payoff['spearman_event_count_p_utility']):.2g}",
                f"{stats['slope'] * 100:+.2f}",
            ]
        )
    return "\n\n".join(
        [
            "## Figure 4 — Strategic behavior in bilateral play\n\n"
            "**Old NeurIPS relationship:** no counterpart; this systematic "
            "six-category analysis is new.",
            "### Payoff association\n\n`o` marks the correlation; `|` is zero.\n\n"
            + fenced(correlation_bars(corr)),
            "### Behavior frequency versus capability\n\nEach `o` curve is the "
            "same centered five-model smooth used by the figure after its "
            "one-outlier-per-category removal.\n\n"
            + fenced(combine_panels(panels, columns=3)),
            md_table(
                [
                    "Behavior",
                    "Payoff Spearman ρ",
                    "Payoff p",
                    "Δ events / rollout / 100 Elo",
                ],
                rows,
            ),
            "**Reading:** trade/compromise and logical persuasion both rise with "
            "capability and predict payoff. Pressure rises fastest but is slightly "
            "payoff-negative.",
        ]
    )


def figure_5() -> str:
    summary = pd.read_csv(TTC_PAYOFF)
    families = [
        ("gpt-5", "GPT-5"),
        ("claude-sonnet-4-6", "Claude Sonnet 4.6"),
        ("gemini-3-flash", "Gemini 3 Flash"),
    ]
    panels = []
    rows = []
    for family, title in families:
        subset = summary[summary["family"].eq(family)].sort_values("level_index")
        levels = subset["level"].astype(str).tolist()
        _, mapper = categorical_mapper(levels, width=28)
        panels.append(
            (
                title,
                render_plot(
                    [
                        {
                            "x": levels,
                            "y": subset["target_utility_mean"],
                            "yerr": subset["target_utility_sem"],
                            "char": "o",
                            "line_char": ".",
                            "connect": True,
                        }
                    ],
                    width=28,
                    height=34,
                    ymin=48,
                    ymax=82,
                    x_mapper=mapper,
                    x_ticks=[(level, effort_label(level)) for level in levels],
                    y_ticks=[50, 55, 60, 65, 70, 75, 80],
                    x_label="Requested effort",
                ),
            )
        )
        for row in subset.itertuples(index=False):
            rows.append(
                [
                    title,
                    str(row.level).title(),
                    f"{row.target_utility_mean:.1f} ± {row.target_utility_sem:.1f}",
                    f"{row.target_tokens_mean:,.0f}",
                    str(int(row.game_cell_count)),
                ]
            )
    return "\n\n".join(
        [
            "## Figure 5 — Test-time compute payoff\n\n"
            "**Old NeurIPS relationship:** no exact counterpart; closest is old "
            "appendix Figure 18.\n\n`o` = mean payoff; `:` = ±SEM; `.` connects "
            "effort levels.",
            fenced(combine_panels(panels, columns=3)),
            md_table(
                [
                    "Model",
                    "Requested effort",
                    "Mean payoff ± SEM",
                    "Observed tokens / call",
                    "Game cells",
                ],
                rows,
            ),
            "**Reading:** all effort-level means lie inside broad, strongly "
            "overlapping SEM intervals despite large token-count changes.",
        ]
    )


def figure_6() -> str:
    summary = pd.read_csv(TTC_INTENSITY)
    focus = summary[
        summary["family"].isin(["gpt-5", "gemini-3-flash"])
        & summary["category"].isin(CAT_ORDER)
        & summary["level"].isin(["minimal", "low", "medium", "high"])
    ].copy()
    levels = ["minimal", "low", "medium", "high"]
    panel_max = {
        "trade/compromise": 4.8,
        "emotional persuasion": 1.0,
        "logical persuasion": 4.8,
        "pressure": 4.0,
        "self-interest/exploitation": 3.0,
        "formalization": 1.6,
    }
    panels = []
    for category in CAT_ORDER:
        ymax = panel_max[category]
        _, mapper = categorical_mapper(levels, width=34)
        series = []
        for family, char in [("gpt-5", "G"), ("gemini-3-flash", "M")]:
            subset = focus[
                focus["family"].eq(family) & focus["category"].eq(category)
            ].set_index("level").loc[levels]
            series.append(
                {
                    "x": levels,
                    "y": subset["unique_turn_events_per_rollout"],
                    "char": char,
                    "connect": True,
                }
            )
        panels.append(
            (
                CAT_SHORT[category],
                render_plot(
                    series,
                    width=34,
                    height=22,
                    ymin=0,
                    ymax=ymax,
                    x_mapper=mapper,
                    x_ticks=[(level, effort_label(level)) for level in levels],
                    y_ticks=np.linspace(0, ymax, 5),
                    y_format=lambda value: f"{value:g}",
                    x_label="Requested effort",
                ),
            )
        )

    tables = []
    for family, label in [("gpt-5", "GPT-5"), ("gemini-3-flash", "Gemini 3 Flash")]:
        rows = []
        for category in CAT_ORDER:
            subset = focus[
                focus["family"].eq(family) & focus["category"].eq(category)
            ].set_index("level").loc[levels]
            values = subset["unique_turn_events_per_rollout"].tolist()
            rows.append(
                [CAT_SHORT[category]]
                + [f"{value:.2f}" for value in values]
                + [f"{values[-1] - values[0]:+.2f}"]
            )
        tables.append(
            f"#### {label}\n\n"
            + md_table(
                ["Behavior", "Minimal", "Low", "Medium", "High", "Minimal → high"],
                rows,
            )
        )
    return "\n\n".join(
        [
            "## Figure 6 — Test-time compute and strategic behavior\n\n"
            "**Old NeurIPS relationship:** no counterpart; this TTC behavior "
            "analysis is new.\n\n`G` = GPT-5; `M` = Gemini 3 Flash; `*` = overlap. "
            "Each point averages 18 rollouts.",
            fenced(combine_panels(panels, columns=3)),
            *tables,
            "**Reading:** reasoning effort changes the behavioral mix, but it "
            "simultaneously strengthens payoff-positive and payoff-negative behaviors.",
        ]
    )


def figure_7() -> str:
    agents = pd.read_csv(HET_AGENTS)
    aggregate = (
        agents[
            agents["experiment_family"].eq("heterogeneous_random")
            & agents["game_label"].eq("game1")
        ]
        .groupby(["n_agents", "model", "model_short", "elo"], as_index=False)
        .agg(mean=("final_utility", "mean"))
    )
    panels = []
    for n_agents in [2, 4, 6, 8, 10]:
        subset = aggregate[aggregate["n_agents"].eq(n_agents)].sort_values("elo")
        _, mapper = collision_mapper(subset["elo"], width=40, xmin=1220, xmax=1512)
        stats = linear_stats(subset["elo"], subset["mean"])
        fit_x = np.linspace(subset["elo"].min(), subset["elo"].max(), 120)
        fit_y = stats["slope"] * fit_x + stats["intercept"]
        ymin = 5 * math.floor((subset["mean"].min() - 5) / 5)
        ymax = 5 * math.ceil((subset["mean"].max() + 5) / 5)
        ticks = np.linspace(ymin, ymax, 6)
        panels.append(
            (
                f"N={n_agents}",
                render_plot(
                    [
                        {
                            "x": fit_x,
                            "y": fit_y,
                            "char": ".",
                            "connect": True,
                            "points": False,
                        },
                        {
                            "x": subset["elo"],
                            "y": subset["mean"],
                            "char": "o",
                        },
                    ],
                    width=40,
                    height=30,
                    ymin=ymin,
                    ymax=ymax,
                    x_mapper=mapper,
                    x_ticks=[(1240, "1240"), (1389, "1389"), (1504, "1504")],
                    y_ticks=ticks,
                    x_label="Arena Elo (ordered)",
                ),
            )
        )
    slopes = pd.read_csv(HET_SLOPES)
    slopes = slopes[slopes["game_label"].eq("game1")].sort_values("n_agents")
    rows = [
        [
            f"N={int(row.n_agents)}",
            str(int(row.n_models)),
            f"{row.mean_obs_per_model:.1f}",
            f"{row.slope_per_100_elo:+.2f}",
            f"{row.r_squared:.2f}",
        ]
        for row in slopes.itertuples(index=False)
    ]
    return "\n\n".join(
        [
            "## Figure 7 — Heterogeneous Game 1 payoff scaling\n\n"
            "**Old NeurIPS relationship:** no exact standalone counterpart; its "
            "Game 1 panel appeared within old main-text Figure 7.\n\n`o` = "
            "model-level mean; `.` = fitted trend. Every distinct model Elo gets "
            "a distinct, order-preserving column.",
            fenced(combine_panels(panels, columns=3)),
            md_table(
                [
                    "Group size",
                    "Models",
                    "Mean observations / model",
                    "Payoff / 100 Elo",
                    "R²",
                ],
                rows,
            ),
            "**Reading:** the fitted trend is positive for every tested group size, "
            "although its strength and explanatory power vary with N.",
        ]
    )


def horizontal_intervals(
    rows: Sequence[tuple[str, float, float]],
    *,
    xmin: float,
    xmax: float,
    width: int,
) -> list[str]:
    lines = []
    for label, mean, error in rows:
        left = round((mean - error - xmin) / (xmax - xmin) * (width - 1))
        center = round((mean - xmin) / (xmax - xmin) * (width - 1))
        right = round((mean + error - xmin) / (xmax - xmin) * (width - 1))
        left, center, right = [
            max(0, min(width - 1, value)) for value in (left, center, right)
        ]
        chars = [" "] * width
        for column in range(min(left, right), max(left, right) + 1):
            chars[column] = "-"
        chars[left] = "["
        chars[right] = "]"
        chars[center] = "o"
        lines.append(f"{label:<24} {mean:.3f} ± {error:.3f} |{''.join(chars)}|")
    ticks = [" "] * width
    for value in np.linspace(xmin, xmax, 5):
        text = f"{value:.2f}"
        center = round((value - xmin) / (xmax - xmin) * (width - 1))
        start = max(0, min(width - len(text), center - len(text) // 2))
        ticks[start : start + len(text)] = text
    lines.append(" " * 40 + "".join(ticks))
    return lines


def figure_8() -> str:
    summary = pd.read_csv(GINI_SUMMARY)
    aggregate = summary[summary["bar_type"].eq("aggregate")]
    aggregate_rows = [
        (
            "Heterogeneous rosters",
            float(aggregate.loc[aggregate["bar_label"].eq("Heterogeneous all"), "payoff_gini_mean"].iloc[0]),
            float(aggregate.loc[aggregate["bar_label"].eq("Heterogeneous all"), "payoff_gini_sem"].iloc[0]),
        ),
        (
            "Homogeneous monocultures",
            float(aggregate.loc[aggregate["bar_label"].eq("Homogeneous all"), "payoff_gini_mean"].iloc[0]),
            float(aggregate.loc[aggregate["bar_label"].eq("Homogeneous all"), "payoff_gini_sem"].iloc[0]),
        ),
    ]
    model = summary[summary["bar_type"].eq("homogeneous_model")].sort_values("model_elo")
    _, mapper = collision_mapper(model["model_elo"], width=70, xmin=1210, xmax=1529)
    hom_runs = pd.read_csv(GINI_HOM_RUNS)
    fit = linear_stats(hom_runs["model_elo"], hom_runs["payoff_gini_corrected"])
    fit_x = np.linspace(model["model_elo"].min(), model["model_elo"].max(), 200)
    fit_y = fit["slope"] * fit_x + fit["intercept"]
    heterogeneous_mean = aggregate_rows[0][1]
    heterogeneous_sem = aggregate_rows[0][2]
    game_chars = {"game1": "1", "game2": "2", "game3": "3"}
    series = [
        {
            "x": [fit_x.min(), fit_x.max()],
            "y": [heterogeneous_mean - heterogeneous_sem] * 2,
            "char": "=",
            "line_char": "=",
            "connect": True,
            "points": False,
        },
        {
            "x": [fit_x.min(), fit_x.max()],
            "y": [heterogeneous_mean + heterogeneous_sem] * 2,
            "char": "=",
            "line_char": "=",
            "connect": True,
            "points": False,
        },
        {
            "x": fit_x,
            "y": fit_y,
            "char": ".",
            "connect": True,
            "points": False,
        }
    ]
    for game, char in game_chars.items():
        subset = model[model["game_label"].eq(game)]
        series.append(
            {
                "x": subset["model_elo"],
                "y": subset["payoff_gini_mean"],
                "yerr": subset["payoff_gini_sem"],
                "char": char,
            }
        )
    scatter = render_plot(
        series,
        width=70,
        height=44,
        ymin=-0.025,
        ymax=0.525,
        x_mapper=mapper,
        x_ticks=[
            (1250, "1250"),
            (1300, "1300"),
            (1350, "1350"),
            (1400, "1400"),
            (1450, "1450"),
            (1500, "1500"),
        ],
        y_ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
        y_format=lambda value: f"{value:.1f}",
        x_label="Monoculture model Elo (ordered)",
        hline=heterogeneous_mean,
    )
    aggregate_table = md_table(
        ["Roster condition", "Runs", "Corrected Gini ± SEM"],
        [
            ["Heterogeneous random rosters", "1,300", f"{aggregate_rows[0][1]:.3f} ± {aggregate_rows[0][2]:.3f}"],
            ["Homogeneous monocultures", "325", f"{aggregate_rows[1][1]:.3f} ± {aggregate_rows[1][2]:.3f}"],
        ],
    )
    model_rows = [
        [
            str(row.game_label).replace("game", "G"),
            str(row.model_short),
            str(int(row.model_elo)),
            str(int(row.n_runs)),
            f"{row.payoff_gini_mean:.3f} ± {row.payoff_gini_sem:.3f}",
        ]
        for row in model.sort_values(["game_label", "model_elo"]).itertuples(index=False)
    ]
    return "\n\n".join(
        [
            "## Figure 8 — Heterogeneous versus homogeneous-control Gini\n\n"
            "**Old NeurIPS relationship:** no counterpart; the 325-run random "
            "monoculture control is new.",
            "### Panel (a): aggregate comparison\n\n`[---o---]` is mean ± SEM.\n\n"
            + fenced(horizontal_intervals(aggregate_rows, xmin=0.13, xmax=0.19, width=60)),
            aggregate_table,
            "### Panel (b): monoculture capability\n\n`1`, `2`, and `3` are "
            "G1–G3 model means; `:` is ±SEM; `.` is the run-level fit; `-` is "
            "the heterogeneous mean and `=` bounds its ±SEM envelope.\n\n"
            + fenced(scatter),
            md_table(
                ["Game", "Monoculture model", "Elo", "Runs", "Corrected Gini ± SEM"],
                model_rows,
            ),
            "**Reading:** aggregate heterogeneous and homogeneous inequality are "
            "nearly tied; within monocultures, capability is the stronger gradient.",
        ]
    )


def figure_9() -> str:
    gini = pd.read_csv(HOM_GINI)
    gini = gini[gini["scope"].eq("overall")].sort_values("bucket_code")
    role = pd.read_csv(HOM_ROLE)
    role = role[role["scenario"].eq("homogeneous_adversary")].sort_values("bucket_code")
    quartiles = ["Q1", "Q2", "Q3", "Q4"]
    _, mapper_left = categorical_mapper(quartiles, width=42)
    left = render_plot(
        [
            {
                "x": quartiles,
                "y": gini["baseline_only_payoff_gini_mean"],
                "yerr": gini["baseline_only_payoff_gini_sem"],
                "char": "o",
                "connect": True,
            }
        ],
        width=42,
        height=40,
        ymin=0.10,
        ymax=0.23,
        x_mapper=mapper_left,
        x_ticks=list(zip(quartiles, quartiles)),
        y_ticks=[0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22],
        y_format=lambda value: f"{value:.2f}",
        x_label="Adversary Elo quartile",
    )
    _, mapper_right = categorical_mapper(quartiles, width=42)
    right = render_plot(
        [
            {
                "x": quartiles,
                "y": role["high_role_payoff_mean"],
                "char": "A",
                "connect": True,
            },
            {
                "x": quartiles,
                "y": role["low_role_payoff_mean"],
                "char": "B",
                "connect": True,
            },
        ],
        width=42,
        height=40,
        ymin=40,
        ymax=60,
        x_mapper=mapper_right,
        x_ticks=list(zip(quartiles, quartiles)),
        y_ticks=[40, 42.5, 45, 47.5, 50, 52.5, 55, 57.5, 60],
        y_format=lambda value: f"{value:g}",
        x_label="Adversary Elo quartile",
    )
    rows = []
    for gini_row, role_row in zip(gini.itertuples(index=False), role.itertuples(index=False)):
        label = str(gini_row.bucket_label).replace("\n", " ")
        rows.append(
            [
                label,
                str(int(gini_row.n_runs)),
                f"{gini_row.baseline_only_payoff_gini_mean:.3f} ± {gini_row.baseline_only_payoff_gini_sem:.3f}",
                f"{role_row.high_role_payoff_mean:.1f}",
                f"{role_row.low_role_payoff_mean:.1f}",
                f"{role_row.high_role_payoff_mean - role_row.low_role_payoff_mean:+.1f}",
            ]
        )
    return "\n\n".join(
        [
            "## Figure 9 — Homogeneous-adversary inequality and role payoff\n\n"
            "**Old NeurIPS relationship:** no exact counterpart; closest are old "
            "main-text Figure 8 and appendix Figure 27.\n\nLeft: `o` = "
            "baseline-only Gini mean and `:` = ±SEM. Right: `A` = adversary "
            "payoff and `B` = mean baseline-agent payoff.",
            fenced(combine_panels([("Baseline-only Gini", left), ("Role payoff", right)], columns=2)),
            md_table(
                [
                    "Adversary Elo quartile",
                    "Runs",
                    "Baseline-only Gini ± SEM",
                    "Adversary payoff",
                    "Mean baseline payoff",
                    "Adversary gap",
                ],
                rows,
            ),
            "**Reading:** the baseline fleet becomes internally more equal while "
            "the inserted adversary's payoff advantage grows from -1.5 to +9.6.",
        ]
    )


def main() -> None:
    sections = [
        "# ICML AIWILD main-text figures as ASCII plots",
        (
            "This is a standalone, text-only companion to "
            "`docs/icml_aiwild_main_figure_tables.md`. It converts every active "
            "main-text figure in `overleaf/icml_aiwild_template/icml_aiwild_2026.pdf` "
            "into Markdown-safe ASCII and keeps exact numeric tables immediately "
            "below the plots. It does not modify the manuscript or the original "
            "table document."
        ),
        (
            "## Rendering guide\n\n"
            "- Every plot is inside a fenced monospaced block; no image embedding is used.\n"
            "- Numeric y-resolution is stated or evident from the labeled ticks.\n"
            "- Dense Elo plots assign distinct, order-preserving columns to distinct "
            "observed Elo values. Local x-spacing is collision-resolved when 1–2 Elo "
            "differences cannot be represented literally.\n"
            "- `:` denotes an uncertainty interval where the active figure uses one; "
            "exact `mean ± SEM` values appear in the table below the plot.\n"
            "- G1 = Item Allocation; G2 = Diplomatic Treaty; G3 = Co-funding."
        ),
        figure_1(),
        figure_2(),
        figure_3(),
        figure_4(),
        figure_5(),
        figure_6(),
        figure_7(),
        figure_8(),
        figure_9(),
        (
            "## Source scope\n\n"
            + md_table(
                ["Figure", "Active numerical scope"],
                [
                    ["1", "Manual conceptual workflow"],
                    ["2", "1,500 primary bilateral runs; 30 adversary models"],
                    [
                        "3",
                        "1,500 bilateral + 1,300 heterogeneous + 1,300 "
                        "homogeneous-adversary + 130 control runs",
                    ],
                    ["4", "1,920 accepted qualitative rollouts; 1,891 payoff-valid"],
                    ["5", "216 TTC runs; 18 per family-effort cell"],
                    ["6", "144 displayed GPT-5/Gemini TTC rollouts"],
                    ["7", "1,300 canonical heterogeneous multi-agent runs; G1 shown"],
                    ["8", "1,300 heterogeneous + 325 monoculture-control runs"],
                    ["9", "1,300 canonical homogeneous-adversary runs"],
                ],
            )
        ),
    ]
    OUT.write_text("\n\n".join(sections).rstrip() + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()

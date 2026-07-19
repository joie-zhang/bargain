#!/usr/bin/env python3
"""N=8/10 filtered version of the bucketed Gini/mean-payoff bar plots."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import plot_n_gt_2_bucketed_homogeneous_heterogeneous_breakdowns as base


N_FILTER = [8, 10]
FILTER_TAG = "n8_n10"


def main() -> None:
    base.OUT_DIR.mkdir(parents=True, exist_ok=True)

    runs = base.compute_run_metrics(base.load_agents())
    runs = runs[runs["n_agents"].isin(N_FILTER)].copy()
    bar_rows, specs = base.build_legacy4_bar_rows(runs)

    all_summaries: list[pd.DataFrame] = []
    plot_paths: list[Path] = []

    n_cols = [(f"N={n}", {"n_agents": n}, {"n_agents": n}) for n in N_FILTER]
    game_cols = [(base.GAME_TITLES[g], {"game_label": g}, {"game_label": g}) for g in base.GAME_ORDER]
    comp_cols = [
        (base.COMPETITION_TITLES[c], {"competition_band": c}, {"competition_band": c})
        for c in base.COMPETITION_ORDER
    ]
    plot_metrics = base.METRIC_SETS["gini_mean_payoff"]
    title_prefix = "Gini inequality and average payoff"
    file_prefix = f"homogeneous_heterogeneous_bucketed_gini_mean_payoff_{FILTER_TAG}"

    plot_specs = [
        (
            [("All", {}, {})],
            n_cols,
            f"by_n_{FILTER_TAG}",
            "broken down by N",
            "by_n",
            (9.2, 6.2),
        ),
        (
            [("All", {}, {})],
            comp_cols,
            f"by_competition_{FILTER_TAG}",
            "broken down by competition band",
            "by_competition",
            (14.0, 6.2),
        ),
        (
            [("All", {}, {})],
            game_cols,
            f"by_game_{FILTER_TAG}",
            "broken down by game",
            "by_game",
            (14.0, 6.2),
        ),
        (
            game_cols,
            comp_cols,
            f"by_game_competition_{FILTER_TAG}",
            "broken down by game and competition band",
            "by_game_competition",
            (14.0, 13.5),
        ),
        (
            game_cols,
            n_cols,
            f"by_game_n_{FILTER_TAG}",
            "broken down by game and N",
            "by_game_n",
            (9.2, 13.5),
        ),
        (
            comp_cols,
            n_cols,
            f"by_competition_n_{FILTER_TAG}",
            "broken down by competition band and N",
            "by_competition_n",
            (9.2, 13.5),
        ),
    ]

    path, summary = base.plot_grid(
        bar_rows,
        specs,
        [("All", {}, {})],
        [("All", {}, {})],
        plot_metrics,
        f"base_{FILTER_TAG}",
        f"N=8/10 {title_prefix} by roster bucket",
        f"{file_prefix}_bars.png",
        (9.5, 6.2),
    )
    plot_paths.append(path)
    all_summaries.append(summary.assign(metric_family="gini_mean_payoff", filter=FILTER_TAG))

    for row_groups, col_groups, scope, title_suffix, filename_suffix, figsize in plot_specs:
        path, summary = base.plot_grid(
            bar_rows,
            specs,
            row_groups,
            col_groups,
            plot_metrics,
            scope,
            f"N=8/10 {title_prefix} by roster bucket, {title_suffix}",
            f"{file_prefix}_{filename_suffix}.png",
            figsize,
        )
        plot_paths.append(path)
        all_summaries.append(summary.assign(metric_family="gini_mean_payoff", filter=FILTER_TAG))

    for n in N_FILTER:
        title = (
            f"N={n} {title_prefix} by roster bucket, "
            "broken down by game and competition band"
        )
        path, summary = base.plot_grid(
            base.filter_frame(bar_rows, {"n_agents": n}),
            specs,
            game_cols,
            comp_cols,
            plot_metrics,
            f"by_n_game_competition_n{n}_{FILTER_TAG}",
            title,
            f"{file_prefix}_by_n{n}_game_competition.png",
            (14.0, 13.5),
        )
        plot_paths.append(path)
        all_summaries.append(summary.assign(metric_family="gini_mean_payoff", filter=FILTER_TAG))

    summary_path = base.OUT_DIR / f"homogeneous_heterogeneous_bucketed_gini_mean_payoff_{FILTER_TAG}_breakdown_summary.csv"
    run_metrics_path = base.OUT_DIR / f"homogeneous_heterogeneous_bucketed_gini_mean_payoff_{FILTER_TAG}_run_metrics.csv"
    pd.concat(all_summaries, ignore_index=True).to_csv(summary_path, index=False)
    runs.to_csv(run_metrics_path, index=False)

    print(f"Wrote {run_metrics_path}")
    print(f"Wrote {summary_path}")
    for path in plot_paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()

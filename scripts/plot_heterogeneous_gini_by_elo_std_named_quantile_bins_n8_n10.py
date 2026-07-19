#!/usr/bin/env python3
"""N=8/10 filtered payoff-Gini bar plots by Elo-std quantile bins."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_heterogeneous_gini_by_elo_std_named_quantile_bins as base


N_FILTER = [8, 10]
FILTER_TAG = "n8_n10"


def plot_overall(frame: pd.DataFrame, config: base.BinConfig) -> tuple[Path, pd.DataFrame]:
    summary = base.summarize_bins(frame, config)
    y_max = base.y_max_for_summaries([summary])
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    base.draw_bar_axis(ax, summary, config, y_max, show_counts=True)
    ax.set_xlabel(f"Within-roster Elo standard deviation {config.name_singular}", fontsize=10.5)
    ax.set_ylabel("Mean corrected payoff Gini", fontsize=10.5)
    ax.set_title(
        f"N=8/10 heterogeneous runs: corrected payoff Gini by Elo-spread {config.name_singular}",
        fontsize=13,
        pad=10,
    )
    fig.tight_layout()
    out_path = base.OUT_DIR / f"heterogeneous_payoff_gini_by_elo_std_{config.file_tag}_{FILTER_TAG}_overall.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path, base.add_metadata(summary, "overall_n8_n10", "All", "All", {"n_agents": "8_or_10"})


def plot_grid(
    frame: pd.DataFrame,
    config: base.BinConfig,
    row_groups: list[tuple[str, dict[str, object], dict[str, object]]],
    col_groups: list[tuple[str, dict[str, object], dict[str, object]]],
    scope: str,
    title: str,
    filename: str,
    figsize: tuple[float, float],
) -> tuple[Path, pd.DataFrame]:
    summaries: dict[tuple[int, int], pd.DataFrame] = {}
    summary_rows: list[pd.DataFrame] = []
    for row_idx, (row_label, row_filter, row_extra) in enumerate(row_groups):
        for col_idx, (col_label, col_filter, col_extra) in enumerate(col_groups):
            filters = {**row_filter, **col_filter}
            summary = base.summarize_bins(frame, config, filters)
            summaries[(row_idx, col_idx)] = summary
            metadata = base.add_metadata(summary, scope, row_label, col_label, filters)
            for key, value in {**row_extra, **col_extra}.items():
                metadata[key] = value
            summary_rows.append(metadata)

    y_max = base.y_max_for_summaries(list(summaries.values()))
    fig, axes = plt.subplots(len(row_groups), len(col_groups), figsize=figsize, sharex=True, sharey=True, squeeze=False)
    show_counts = len(row_groups) * len(col_groups) <= 5
    for row_idx, (row_label, _, _) in enumerate(row_groups):
        for col_idx, (col_label, _, _) in enumerate(col_groups):
            ax = axes[row_idx, col_idx]
            base.draw_bar_axis(ax, summaries[(row_idx, col_idx)], config, y_max, show_counts=show_counts)
            if row_idx == 0:
                ax.set_title(col_label, fontsize=9, pad=5)
            if col_idx == 0:
                ax.set_ylabel(f"{row_label}\nMean Gini" if len(row_groups) > 1 else "Mean Gini", fontsize=8.5)
            if row_idx == len(row_groups) - 1:
                ax.set_xlabel(config.name_singular.title(), fontsize=8.5)
            else:
                ax.tick_params(axis="x", labelbottom=False)
    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    out_path = base.OUT_DIR / filename
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path, pd.concat(summary_rows, ignore_index=True)


def make_outputs_for_config(run_metrics: pd.DataFrame, config: base.BinConfig) -> tuple[list[Path], pd.DataFrame]:
    paths: list[Path] = []
    summaries: list[pd.DataFrame] = []

    path, summary = plot_overall(run_metrics, config)
    paths.append(path)
    summaries.append(summary)

    n_cols = [(f"N={n}", {"n_agents": n}, {"n_agents": n}) for n in N_FILTER]
    game_cols = [(base.GAME_TITLES[g], {"game_label": g}, {"game_label": g}) for g in base.GAME_ORDER]
    comp_cols = [
        (base.COMPETITION_TITLES[c], {"competition_band": c}, {"competition_band": c})
        for c in base.COMPETITION_ORDER
    ]
    all_rows = [("All", {}, {})]

    specs = [
        (
            all_rows,
            game_cols,
            f"by_game_{FILTER_TAG}",
            f"N=8/10 corrected payoff Gini by Elo-spread {config.name_singular}, by game",
            f"heterogeneous_payoff_gini_by_elo_std_{config.file_tag}_{FILTER_TAG}_by_game.png",
            (10.2, 4.0),
        ),
        (
            all_rows,
            n_cols,
            f"by_n_{FILTER_TAG}",
            f"N=8/10 corrected payoff Gini by Elo-spread {config.name_singular}, by N",
            f"heterogeneous_payoff_gini_by_elo_std_{config.file_tag}_{FILTER_TAG}_by_n.png",
            (7.2, 4.0),
        ),
        (
            all_rows,
            comp_cols,
            f"by_competition_{FILTER_TAG}",
            f"N=8/10 corrected payoff Gini by Elo-spread {config.name_singular}, by competition band",
            f"heterogeneous_payoff_gini_by_elo_std_{config.file_tag}_{FILTER_TAG}_by_competition.png",
            (10.2, 4.0),
        ),
        (
            game_cols,
            n_cols,
            f"by_game_n_{FILTER_TAG}",
            f"N=8/10 corrected payoff Gini by Elo-spread {config.name_singular}, by game and N",
            f"heterogeneous_payoff_gini_by_elo_std_{config.file_tag}_{FILTER_TAG}_by_game_n.png",
            (7.2, 8.0),
        ),
        (
            comp_cols,
            game_cols,
            f"by_competition_game_{FILTER_TAG}",
            f"N=8/10 corrected payoff Gini by Elo-spread {config.name_singular}, by competition band and game",
            f"heterogeneous_payoff_gini_by_elo_std_{config.file_tag}_{FILTER_TAG}_by_competition_game.png",
            (10.2, 8.0),
        ),
    ]

    for row_groups, col_groups, scope, title, filename, figsize in specs:
        path, summary = plot_grid(run_metrics, config, row_groups, col_groups, scope, title, filename, figsize)
        paths.append(path)
        summaries.append(summary)

    for n in N_FILTER:
        filtered = base.subset_frame(run_metrics, {"n_agents": n})
        path, summary = plot_grid(
            filtered,
            config,
            game_cols,
            comp_cols,
            f"by_game_competition_n{n}_{FILTER_TAG}",
            f"N={n} corrected payoff Gini by Elo-spread {config.name_singular}, by game and competition band",
            f"heterogeneous_payoff_gini_by_elo_std_{config.file_tag}_{FILTER_TAG}_by_game_competition_n{n}.png",
            (10.2, 8.0),
        )
        summary["n_agents"] = n
        paths.append(path)
        summaries.append(summary)

    return paths, pd.concat(summaries, ignore_index=True)


def main() -> None:
    base.OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_metrics = base.load_run_metrics()
    run_metrics = run_metrics[run_metrics["n_agents"].isin(N_FILTER)].copy()

    run_metrics_path = base.OUT_DIR / f"heterogeneous_payoff_gini_by_elo_std_named_quantile_bin_{FILTER_TAG}_run_metrics.csv"
    run_metrics.to_csv(run_metrics_path, index=False)

    all_paths: list[Path] = []
    all_summaries: list[pd.DataFrame] = []
    for config in base.BIN_CONFIGS:
        paths, summary = make_outputs_for_config(run_metrics, config)
        summary_path = base.OUT_DIR / f"heterogeneous_payoff_gini_by_elo_std_{config.file_tag}_{FILTER_TAG}_summary.csv"
        summary.to_csv(summary_path, index=False)
        all_paths.extend(paths)
        all_paths.append(summary_path)
        all_summaries.append(summary)

    combined_summary_path = base.OUT_DIR / f"heterogeneous_payoff_gini_by_elo_std_named_quantile_bin_{FILTER_TAG}_summary.csv"
    pd.concat(all_summaries, ignore_index=True).to_csv(combined_summary_path, index=False)

    for path in all_paths:
        print(f"Wrote {path}")
    print(f"Wrote {run_metrics_path}")
    print(f"Wrote {combined_summary_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Minimal CSV-only plotting script for Lewis slides.

Reads:
  visualization/figures/gpt5_nano_full_data.csv

Writes:
  - MAIN_PLOT_1_BASELINE_PAYOFF.png
  - MAIN_PLOT_2_ADVERSARY_PAYOFF.png
  - Optional variant plots (raw/compressed/smoothed/compressed+smoothed)
  - Markdown report with model coverage and per-competition run counts
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 11


COMP_BUCKET_LABELS = {
    0.0: "gamma=0.0",
    0.2: "gamma=0.1-0.3 avg",
    0.5: "gamma=0.4-0.6 avg",
    0.8: "gamma=0.7-0.9 avg",
    1.0: "gamma=1.0",
}

MEAN_COLS = {
    "baseline_utility",
    "adversary_utility",
    "total_utility",
    "utility_share",
    "payoff_diff",
    "consensus_reached",
    "final_round",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate slide payoff plots from CSV only.")
    parser.add_argument(
        "--input-csv",
        default="figures/gpt5_nano_full_data.csv",
        help="Path to input CSV, relative to this script directory by default.",
    )
    parser.add_argument(
        "--output-dir",
        default="figures",
        help="Directory for output plots/report, relative to this script directory by default.",
    )
    parser.add_argument(
        "--min-runs-per-model",
        type=int,
        default=10,
        help="Filter models with fewer than this many successful runs. Use 0 to disable.",
    )
    parser.add_argument(
        "--smoothing-method",
        choices=["ewm", "moving_average"],
        default="ewm",
        help="Smoothing method for smoothed variants.",
    )
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=0.35,
        help="EWM alpha (used when smoothing method is ewm).",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=3,
        help="Moving-average window (used when smoothing method is moving_average).",
    )
    parser.add_argument(
        "--make-all-variants",
        action="store_true",
        help="Also generate raw/compressed/smoothed/compressed+smoothed variant files.",
    )
    parser.add_argument(
        "--report-name",
        default="lewis_slides_plot_report_mar_1.md",
        help="Markdown report filename.",
    )
    return parser.parse_args()


def resolve_path(script_dir: Path, path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else script_dir / path


def bucket_competition_level(comp_level: float) -> float:
    comp_level = round(float(comp_level), 1)
    if np.isclose(comp_level, 0.0) or np.isclose(comp_level, 1.0):
        return comp_level
    if 0.1 <= comp_level <= 0.3:
        return 0.2
    if 0.4 <= comp_level <= 0.6:
        return 0.5
    if 0.7 <= comp_level <= 0.9:
        return 0.8
    return comp_level


def comp_label(comp_level: float, compressed: bool) -> str:
    if compressed:
        return COMP_BUCKET_LABELS.get(round(float(comp_level), 1), f"gamma={comp_level:.1f}")
    return f"gamma={comp_level:.1f}"


def smooth_series(
    values: np.ndarray,
    method: str,
    alpha: float,
    window: int,
) -> np.ndarray:
    series = pd.Series(values, dtype="float64")
    if method == "moving_average":
        return series.rolling(window=window, min_periods=1).mean().to_numpy()
    if method == "ewm":
        return series.ewm(alpha=alpha, adjust=False).mean().to_numpy()
    raise ValueError(f"Unsupported smoothing method: {method}")


def build_avg_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["adversary_model", "competition_level"]
    agg_map: Dict[str, str] = {}
    for col in df_raw.columns:
        if col in group_cols:
            continue
        if col in MEAN_COLS:
            agg_map[col] = "mean"
        elif col == "model_order":
            agg_map[col] = "count"
        else:
            agg_map[col] = "first"

    df_avg = df_raw.groupby(group_cols, as_index=False).agg(agg_map)
    if "model_order" in df_avg.columns:
        df_avg = df_avg.rename(columns={"model_order": "n_orders_averaged"})
    return df_avg.sort_values(["competition_level", "adversary_elo"])


def compress_competition_levels(df_plot: pd.DataFrame) -> pd.DataFrame:
    compressed = df_plot.copy()
    compressed["competition_level"] = compressed["competition_level"].apply(bucket_competition_level)

    group_cols = ["adversary_model", "competition_level"]
    agg_map: Dict[str, str] = {}
    for col in compressed.columns:
        if col in group_cols:
            continue
        if col in MEAN_COLS:
            agg_map[col] = "mean"
        elif col == "n_orders_averaged":
            agg_map[col] = "sum"
        else:
            agg_map[col] = "first"

    return compressed.groupby(group_cols, as_index=False).agg(agg_map).sort_values(
        ["competition_level", "adversary_elo"]
    )


def plot_payoff_lines_by_comp(
    plot_df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
    compress_levels: bool,
    apply_smoothing: bool,
    smoothing_method: str,
    smoothing_alpha: float,
    smoothing_window: int,
) -> None:
    df_line = plot_df.copy()
    if compress_levels:
        df_line = compress_competition_levels(df_line)
    if df_line.empty:
        raise ValueError("No rows available for plotting after filtering.")

    comp_levels = sorted(df_line["competition_level"].dropna().unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(comp_levels)))

    fig, ax = plt.subplots(figsize=(12, 8))
    for color, comp_level in zip(colors, comp_levels):
        comp_df = df_line[df_line["competition_level"] == comp_level].sort_values("adversary_elo")
        if comp_df.empty:
            continue

        x_vals = comp_df["adversary_elo"].to_numpy()
        y_raw = comp_df[value_col].to_numpy()
        if apply_smoothing:
            y_vals = smooth_series(
                values=y_raw,
                method=smoothing_method,
                alpha=smoothing_alpha,
                window=smoothing_window,
            )
            ax.plot(x_vals, y_raw, color=color, linewidth=1.0, alpha=0.22)
        else:
            y_vals = y_raw

        ax.plot(
            x_vals,
            y_vals,
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=3.5,
            alpha=0.95,
            label=comp_label(comp_level, compressed=compress_levels),
        )

    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, linewidth=1.1, label="Equal split (50)")
    ax.set_xlabel("Adversary Elo")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(df_line["adversary_elo"].min() - 5, df_line["adversary_elo"].max() + 5)
    ax.legend(title="Competition", fontsize=8.5, title_fontsize=9.5, ncol=2, loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def get_model_filter(df_raw: pd.DataFrame, min_runs_per_model: int) -> Tuple[List[str], pd.Series]:
    run_counts = df_raw["adversary_model"].value_counts().sort_index()
    if min_runs_per_model <= 0:
        return sorted(run_counts.index.tolist()), run_counts
    eligible = run_counts[run_counts >= min_runs_per_model].index.tolist()
    return sorted(eligible), run_counts


def format_model_list(models: Iterable[str]) -> str:
    model_list = sorted(models)
    return ", ".join(model_list) if model_list else "(none)"


def write_report(
    report_path: Path,
    input_csv: Path,
    min_runs_per_model: int,
    generated_plots: List[str],
    models_used_by_plot: Dict[str, List[str]],
    run_counts: pd.Series,
    df_raw: pd.DataFrame,
) -> None:
    comp_levels = sorted(df_raw["competition_level"].dropna().unique())
    coverage = (
        df_raw.pivot_table(
            index="adversary_model",
            columns="competition_level",
            values="baseline_utility",
            aggfunc="count",
            fill_value=0,
        )
        .reindex(columns=comp_levels, fill_value=0)
        .copy()
    )

    elo = df_raw.groupby("adversary_model")["adversary_elo"].first()
    coverage["total_successful_runs"] = run_counts
    coverage["elo"] = coverage.index.map(elo)
    coverage = coverage.sort_values("elo", ascending=False)

    cols = ["elo", "total_successful_runs"] + comp_levels
    coverage = coverage[cols]

    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Lewis Slides Plot Inputs (Mar 1)\n\n")
        f.write("This report is generated from CSV only (no raw JSON dependency).\n\n")
        f.write(f"- Input CSV: `{input_csv}`\n")
        f.write(f"- Model inclusion rule for plotted lines: `successful runs >= {min_runs_per_model}`\n")
        f.write("- Main plots are generated with: compression + smoothing\n\n")

        f.write("## Generated Plot Files\n\n")
        for name in generated_plots:
            f.write(f"- `{name}`\n")

        f.write("\n## Models Used Per Plot\n\n")
        for plot_name, models in models_used_by_plot.items():
            f.write(f"### `{plot_name}`\n\n")
            f.write(f"- Number of models: `{len(models)}`\n")
            f.write(f"- Models: {format_model_list(models)}\n\n")

        f.write("## Successful Runs Per Model Per Competition Level\n\n")
        f.write("| adversary_model | elo | total_successful_runs |")
        for c in comp_levels:
            f.write(f" comp_{c:.1f} |")
        f.write("\n")
        f.write("|---|---:|---:|")
        for _ in comp_levels:
            f.write("---:|")
        f.write("\n")

        for model, row in coverage.iterrows():
            f.write(f"| {model} | {int(row['elo'])} | {int(row['total_successful_runs'])} |")
            for c in comp_levels:
                f.write(f" {int(row[c])} |")
            f.write("\n")


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    input_csv = resolve_path(script_dir, args.input_csv)
    output_dir = resolve_path(script_dir, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"Missing CSV: {input_csv}")

    df_raw = pd.read_csv(input_csv)
    required_cols = {
        "adversary_model",
        "adversary_elo",
        "competition_level",
        "baseline_utility",
        "adversary_utility",
    }
    missing = sorted(required_cols - set(df_raw.columns))
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    df_raw = df_raw[df_raw["adversary_elo"] > 0].copy()
    df_avg = build_avg_df(df_raw)

    included_models, run_counts = get_model_filter(df_raw, args.min_runs_per_model)
    df_plot = df_avg[df_avg["adversary_model"].isin(included_models)].copy()
    if df_plot.empty:
        raise ValueError("No models remain after filtering. Lower --min-runs-per-model.")

    generated_plots: List[str] = []
    models_used_by_plot: Dict[str, List[str]] = {}

    # Always generate the two main slide plots (compression + smoothing).
    main_baseline = "MAIN_PLOT_1_BASELINE_PAYOFF.png"
    main_adversary = "MAIN_PLOT_2_ADVERSARY_PAYOFF.png"

    plot_payoff_lines_by_comp(
        plot_df=df_plot,
        value_col="baseline_utility",
        ylabel="Baseline Model Payoff",
        title=(
            "Baseline Model Payoff vs Adversary Elo\n"
            f"(Compressed buckets + {args.smoothing_method.upper()} smoothing)"
        ),
        output_path=output_dir / main_baseline,
        compress_levels=True,
        apply_smoothing=True,
        smoothing_method=args.smoothing_method,
        smoothing_alpha=args.smoothing_alpha,
        smoothing_window=args.smoothing_window,
    )
    generated_plots.append(main_baseline)
    models_used_by_plot[main_baseline] = included_models

    plot_payoff_lines_by_comp(
        plot_df=df_plot,
        value_col="adversary_utility",
        ylabel="Adversary Model Payoff",
        title=(
            "Adversary Model Payoff vs Adversary Elo\n"
            f"(Compressed buckets + {args.smoothing_method.upper()} smoothing)"
        ),
        output_path=output_dir / main_adversary,
        compress_levels=True,
        apply_smoothing=True,
        smoothing_method=args.smoothing_method,
        smoothing_alpha=args.smoothing_alpha,
        smoothing_window=args.smoothing_window,
    )
    generated_plots.append(main_adversary)
    models_used_by_plot[main_adversary] = included_models

    if args.make_all_variants:
        variants = [
            ("baseline_raw", "baseline_utility", False, False),
            ("baseline_compressed", "baseline_utility", True, False),
            ("baseline_smoothed", "baseline_utility", False, True),
            ("baseline_compressed_smoothed", "baseline_utility", True, True),
            ("adversary_raw", "adversary_utility", False, False),
            ("adversary_compressed", "adversary_utility", True, False),
            ("adversary_smoothed", "adversary_utility", False, True),
            ("adversary_compressed_smoothed", "adversary_utility", True, True),
        ]
        for suffix, value_col, compress_levels, apply_smoothing in variants:
            filename = f"lewis_mar_1_{suffix}.png"
            ylabel = "Baseline Model Payoff" if value_col == "baseline_utility" else "Adversary Model Payoff"
            title = f"{ylabel} vs Adversary Elo"
            if compress_levels and apply_smoothing:
                title += f"\n(Compressed + {args.smoothing_method.upper()} smoothed)"
            elif compress_levels:
                title += "\n(Compressed)"
            elif apply_smoothing:
                title += f"\n({args.smoothing_method.upper()} smoothed)"
            else:
                title += "\n(Raw competition lines)"

            plot_payoff_lines_by_comp(
                plot_df=df_plot,
                value_col=value_col,
                ylabel=ylabel,
                title=title,
                output_path=output_dir / filename,
                compress_levels=compress_levels,
                apply_smoothing=apply_smoothing,
                smoothing_method=args.smoothing_method,
                smoothing_alpha=args.smoothing_alpha,
                smoothing_window=args.smoothing_window,
            )
            generated_plots.append(filename)
            models_used_by_plot[filename] = included_models

    report_path = output_dir / args.report_name
    write_report(
        report_path=report_path,
        input_csv=input_csv,
        min_runs_per_model=args.min_runs_per_model,
        generated_plots=generated_plots,
        models_used_by_plot=models_used_by_plot,
        run_counts=run_counts,
        df_raw=df_raw,
    )

    print(f"Input CSV: {input_csv}")
    print(f"Output directory: {output_dir}")
    print(f"Generated {len(generated_plots)} plot files.")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()

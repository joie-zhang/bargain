from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "analysis/recreated_figures/figure4_n2_qualitative"
PAYOFF_CSV = (
    ROOT
    / "overleaf/icml_aiwild_template/graphics/qualitative_n2/n2_group_payoff_corr_dedup.csv"
)
INTENSITY_CSV = (
    ROOT
    / "overleaf/icml_aiwild_template/graphics/qualitative_n2/n2_group_intensity_dedup.csv"
)
OUT_PATH = OUT_DIR / (
    "figure4_combined_lhs_rhs_smooth5_outlier_removed_square_font_balanced_aligned.png"
)
OUT_PDF = OUT_PATH.with_suffix(".pdf")

CATEGORY_ORDER = [
    "trade/compromise",
    "emotional persuasion",
    "logical persuasion",
    "pressure",
    "self-interest/exploitation",
    "formalization",
]

DISPLAY_LABELS = {
    "trade/compromise": "Trade / Compromise",
    "emotional persuasion": "Emotional persuasion",
    "logical persuasion": "Logical persuasion",
    "pressure": "Pressure",
    "self-interest/exploitation": "Self-interest / Exploitation",
    "formalization": "Formalization",
}

COLORS = {
    "trade/compromise": "#1f77b4",
    "emotional persuasion": "#2ca02c",
    "logical persuasion": "#17becf",
    "pressure": "#ff7f0e",
    "self-interest/exploitation": "#d62728",
    "formalization": "#9467bd",
}


def remove_one_outlier_per_category(df: pd.DataFrame, window: int) -> pd.DataFrame:
    kept = []
    for category in CATEGORY_ORDER:
        group = (
            df[df["category"] == category]
            .sort_values(["speaker_elo", "speaker_model"])
            .reset_index(drop=True)
        )
        baseline = group["intensity"].rolling(window=window, center=True, min_periods=1).median()
        outlier_pos = (group["intensity"] - baseline).abs().idxmax()
        kept.append(group.drop(index=outlier_pos))
    return pd.concat(kept, ignore_index=True)


def centered_smooth(values: pd.Series, window: int) -> pd.Series:
    return values.rolling(window=window, center=True, min_periods=1).mean()


def plot_left(ax: plt.Axes) -> None:
    df = pd.read_csv(PAYOFF_CSV)
    df["category"] = pd.Categorical(df["category"], CATEGORY_ORDER, ordered=True)
    df = df.sort_values("category")

    y_positions = list(range(len(df)))
    values = df["spearman_event_count_r_utility"].to_numpy()
    colors = ["#4c78a8" if value >= 0 else "#e45756" for value in values]

    ax.barh(
        y_positions,
        values,
        color=colors,
        edgecolor="black",
        linewidth=1.0,
        height=0.72,
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels([DISPLAY_LABELS[c] for c in df["category"]], fontsize=23)
    ax.invert_yaxis()

    for y, value in zip(y_positions, values):
        label = f"{value:+.2f}"
        if value >= 0:
            ax.text(value + 0.015, y, label, va="center", ha="left", fontsize=22)
        else:
            ax.text(value - 0.015, y, label, va="center", ha="right", fontsize=22)

    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_xlim(-0.45, 0.32)
    ax.set_xticks([-0.4, -0.2, 0.0, 0.2])
    ax.set_xlabel("Spearman correlation with utility", fontsize=27, labelpad=8)
    ax.tick_params(axis="x", labelsize=22, width=1.2, length=6)
    ax.tick_params(axis="y", width=1.2, length=5)
    ax.grid(True, axis="x", linestyle=":", linewidth=1.2, alpha=0.45)
    ax.grid(False, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)


def plot_right(ax: plt.Axes) -> None:
    df = pd.read_csv(INTENSITY_CSV)
    df = df[df["category"].isin(CATEGORY_ORDER)].copy()
    df["speaker_elo"] = pd.to_numeric(df["speaker_elo"], errors="coerce")
    df["intensity"] = pd.to_numeric(df["intensity"], errors="coerce")
    df = remove_one_outlier_per_category(df, window=5)

    for category in CATEGORY_ORDER:
        group = (
            df[df["category"] == category]
            .sort_values(["speaker_elo", "speaker_model"])
            .reset_index(drop=True)
        )
        smoothed = centered_smooth(group["intensity"], window=5)
        color = COLORS[category]

        ax.plot(
            group["speaker_elo"],
            group["intensity"],
            marker="o",
            linewidth=1.7,
            markersize=4.8,
            alpha=0.14,
            color=color,
            zorder=1,
        )
        ax.plot(
            group["speaker_elo"],
            smoothed,
            marker="o",
            linewidth=3.2,
            markersize=9.0,
            alpha=0.98,
            color=color,
            label=DISPLAY_LABELS[category],
            zorder=3,
        )

    ax.set_xlabel("Adversary Elo", fontsize=29, labelpad=8)
    ax.set_ylabel("Mean events per rollout", fontsize=29, labelpad=8)
    ax.set_xlim(1090, 1524)
    ax.set_ylim(-0.13, 3.1)
    ax.set_xticks([1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500])
    ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax.tick_params(axis="both", labelsize=21, width=1.2, length=6)
    ax.grid(True, which="major", linestyle=":", linewidth=1.2, alpha=0.55)
    ax.legend(
        loc="upper left",
        fontsize=18,
        framealpha=0.84,
        facecolor="white",
        edgecolor="0.75",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "lines.solid_capstyle": "round",
        }
    )

    fig, (left_ax, right_ax) = plt.subplots(
        1,
        2,
        figsize=(24.0, 10.0),
        dpi=150,
        gridspec_kw={"width_ratios": [1.0, 1.08], "wspace": 0.20},
    )
    plot_left(left_ax)
    plot_right(right_ax)
    fig.subplots_adjust(left=0.22, right=0.985, bottom=0.14, top=0.965, wspace=0.20)
    fig.savefig(OUT_PATH, dpi=300)
    fig.savefig(OUT_PDF)
    plt.close(fig)
    print(OUT_PATH)
    print(OUT_PDF)


if __name__ == "__main__":
    main()

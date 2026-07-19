#!/usr/bin/env python3
"""Compare current random-monoculture Gini against heterogeneous Games 1-3 Gini."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import full_games123_multiagent_batch as full  # noqa: E402
import random_monoculture_control_batch as rmc  # noqa: E402


DEFAULT_HOM_ROOT = (
    PROJECT_ROOT
    / "experiments/results/full_games123_random_monoculture_control_20260628_014357"
)
DEFAULT_HETERO_RUNS = (
    PROJECT_ROOT
    / "experiments/results/n2_plus_multiagent_comparison_analysis_20260505"
    / "tables_multiagent/heterogeneous_runs_fresh.csv"
)

GAME_COLORS = {
    "game1": "#4E79A7",
    "game2": "#59A14F",
    "game3": "#E15759",
}
GAME_ORDER = ["game1", "game2", "game3"]
GAME_LABELS = {"game1": "G1", "game2": "G2", "game3": "G3"}
AGG_COLORS = {
    "Heterogeneous all": "#D54E6A",
    "Homogeneous all": "#4E79A7",
}
MODEL_SHORT_NAMES = {
    "amazon-nova-micro-v1.0": "Nova Micro",
    "amazon-nova-pro-v1.0": "Nova Pro",
    "claude-3-haiku-20240307": "Claude 3 Haiku",
    "claude-opus-4-5-20251101": "Opus 4.5",
    "claude-opus-4-5-20251101-thinking-32k": "Opus 4.5 Think",
    "claude-opus-4-6": "Opus 4.6",
    "deepseek-r1-0528": "DeepSeek R1",
    "deepseek-v3": "DeepSeek V3",
    "gemini-3.1-pro": "Gemini 3.1 Pro",
    "gpt-4o-2024-05-13": "GPT-4o",
    "gpt-5-nano-high": "GPT-5 nano",
    "gpt-5.2-chat-latest-20260210": "GPT-5.2 Chat",
    "gpt-5.4-high": "GPT-5.4 High",
    "o3-mini-high": "o3-mini",
    "qwen3-max-preview": "Qwen3 Max",
}


def short_model_name(model: str) -> str:
    return MODEL_SHORT_NAMES.get(model, model)


def shifted_gini(values: list[float] | np.ndarray) -> tuple[float, float, bool]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan, math.nan, False
    shifted = False
    if float(arr.min()) < 0:
        arr = arr - float(arr.min())
        shifted = True
    if arr.size < 2 or np.allclose(arr, arr[0]) or np.allclose(arr, 0.0):
        return 0.0, 0.0, shifted
    mean_value = float(arr.mean())
    if math.isclose(mean_value, 0.0):
        return 0.0, 0.0, shifted
    diffs = np.abs(arr[:, None] - arr[None, :])
    raw = float(np.mean(diffs) / (2.0 * mean_value))
    corrected = min(raw * float(arr.size / (arr.size - 1)), 1.0)
    return raw, corrected, shifted


def sem(values: pd.Series) -> float:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return 0.0
    return float(clean.std(ddof=1) / math.sqrt(len(clean)))


def config_id_string(config: dict[str, Any]) -> str:
    return f"config_{rmc.config_number(config['config_id']):04d}"


def load_homogeneous_runs(results_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for config in full.load_configs(results_root):
        result_path = full.result_path_for(config)
        if (
            result_path is None
            or not result_path.exists()
            or full.validate_result_file(rmc.runtime_config(config), result_path) is not None
        ):
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        utilities = [float(v) for v in (result.get("final_utilities") or {}).values()]
        raw_gini, corrected_gini, shifted = shifted_gini(utilities)
        rows.append(
            {
                "run_key": config_id_string(config),
                "config_id": config_id_string(config),
                "comparison_group": "homogeneous_random_monoculture",
                "game_label": str(config["game_label"]),
                "n_agents": int(config["n_agents"]),
                "model": str(config["monoculture_model"]),
                "model_short": short_model_name(str(config["monoculture_model"])),
                "model_elo": float(config.get("model_elo", math.nan)),
                "payoff_gini_raw_shifted": raw_gini,
                "payoff_gini_corrected": corrected_gini,
                "payoff_gini_shifted_for_negative": shifted,
                "mean_payoff": float(np.mean(utilities)) if utilities else math.nan,
                "final_round": int(result.get("final_round") or 0),
                "consensus_reached": bool(result.get("consensus_reached")),
            }
        )
    return pd.DataFrame(rows)


def load_heterogeneous_runs(path: Path) -> pd.DataFrame:
    runs = pd.read_csv(path)
    runs = runs[runs["experiment_family"].eq("heterogeneous_random")].copy()
    runs["payoff_gini_corrected"] = pd.to_numeric(runs["utility_gini_corrected"], errors="coerce")
    runs["payoff_gini_raw_shifted"] = pd.to_numeric(runs["utility_gini_raw"], errors="coerce")
    runs["n_agents"] = pd.to_numeric(runs["n_agents"], errors="coerce")
    runs["mean_payoff"] = pd.to_numeric(runs["mean_utility"], errors="coerce")
    runs = runs.dropna(subset=["run_key", "game_label", "n_agents", "payoff_gini_corrected"])
    runs["comparison_group"] = "heterogeneous_random"
    return runs


def summarize_bar(label: str, values: pd.Series) -> dict[str, Any]:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "bar_label": label,
        "n_runs": int(len(clean)),
        "payoff_gini_mean": float(clean.mean()) if len(clean) else math.nan,
        "payoff_gini_sem": sem(clean),
    }


def expected_homogeneous_counts(results_root: Path) -> pd.DataFrame:
    rows = []
    for config in full.load_configs(results_root):
        rows.append(
            {
                "model": str(config["monoculture_model"]),
                "game_label": str(config["game_label"]),
                "expected": 1,
            }
        )
    return (
        pd.DataFrame(rows)
        .groupby(["model", "game_label"], as_index=False, dropna=False)["expected"]
        .sum()
    )


def build_summary(hetero: pd.DataFrame, hom: pd.DataFrame, hom_root: Path) -> pd.DataFrame:
    expected_by_model = expected_homogeneous_counts(hom_root)
    expected_by_game = expected_by_model.groupby("game_label", as_index=False)["expected"].sum()
    rows = [
        {
            **summarize_bar("Heterogeneous all", hetero["payoff_gini_corrected"]),
            "bar_type": "aggregate",
            "game_label": "all",
            "model": "heterogeneous_all",
            "model_short": "Heterogeneous all",
            "model_elo": math.nan,
            "completed": len(hetero),
            "expected": len(hetero),
        },
        {
            **summarize_bar("Homogeneous all", hom["payoff_gini_corrected"]),
            "bar_type": "aggregate",
            "game_label": "all",
            "model": "homogeneous_all",
            "model_short": "Homogeneous all",
            "model_elo": math.nan,
            "completed": len(hom),
            "expected": int(expected_by_game["expected"].sum()),
        },
    ]
    for game in GAME_ORDER:
        hetero_game = hetero[hetero["game_label"].eq(game)]
        hom_game = hom[hom["game_label"].eq(game)]
        expected_hit = expected_by_game[expected_by_game["game_label"].eq(game)]
        expected = int(expected_hit["expected"].iloc[0]) if not expected_hit.empty else math.nan
        rows.extend(
            [
                {
                    **summarize_bar(f"Heterogeneous {GAME_LABELS[game]}", hetero_game["payoff_gini_corrected"]),
                    "bar_type": "game_aggregate",
                    "source_group": "heterogeneous",
                    "game_label": game,
                    "model": f"heterogeneous_{game}",
                    "model_short": f"Heterogeneous {GAME_LABELS[game]}",
                    "model_elo": math.nan,
                    "completed": len(hetero_game),
                    "expected": len(hetero_game),
                },
                {
                    **summarize_bar(f"Homogeneous {GAME_LABELS[game]}", hom_game["payoff_gini_corrected"]),
                    "bar_type": "game_aggregate",
                    "source_group": "homogeneous",
                    "game_label": game,
                    "model": f"homogeneous_{game}",
                    "model_short": f"Homogeneous {GAME_LABELS[game]}",
                    "model_elo": math.nan,
                    "completed": len(hom_game),
                    "expected": expected,
                },
            ]
        )
    for (model, game), sub in hom.groupby(["model", "game_label"], sort=False):
        expected = math.nan
        hit = expected_by_model[
            expected_by_model["model"].eq(model) & expected_by_model["game_label"].eq(game)
        ]
        if not hit.empty:
            expected = int(hit["expected"].iloc[0])
        rows.append(
            {
                **summarize_bar(str(sub["model_short"].iloc[0]), sub["payoff_gini_corrected"]),
                "bar_type": "homogeneous_model",
                "source_group": "homogeneous",
                "game_label": game,
                "model": model,
                "model_short": str(sub["model_short"].iloc[0]),
                "model_elo": float(sub["model_elo"].iloc[0]),
                "completed": int(len(sub)),
                "expected": expected,
            }
        )
    summary = pd.DataFrame(rows)
    aggregates = summary[summary["bar_type"].eq("aggregate")]
    game_aggregates = summary[summary["bar_type"].eq("game_aggregate")]
    models = summary[summary["bar_type"].eq("homogeneous_model")].sort_values(
        ["game_label", "model_elo", "model_short"]
    )
    return pd.concat([aggregates, game_aggregates, models], ignore_index=True)


def plot_summary(summary: pd.DataFrame, out_path: Path) -> None:
    x_positions = []
    x = 0.0
    for i, row in enumerate(summary.itertuples()):
        x_positions.append(x)
        x += 1.0
        if i in {1, 3, 5, 7}:
            x += 0.65

    colors = []
    labels = []
    hatches = []
    for row in summary.itertuples():
        if row.bar_type == "aggregate":
            colors.append(AGG_COLORS[row.bar_label])
            labels.append(row.bar_label)
            hatches.append("")
        elif row.bar_type == "game_aggregate":
            if row.source_group == "heterogeneous":
                colors.append(AGG_COLORS["Heterogeneous all"])
                labels.append(f"Hetero\n{GAME_LABELS[row.game_label]}")
                hatches.append("\\\\")
            else:
                colors.append(GAME_COLORS[row.game_label])
                labels.append(f"Hom\n{GAME_LABELS[row.game_label]}")
                hatches.append("//")
        else:
            colors.append(GAME_COLORS[row.game_label])
            labels.append(f"{row.model_short}\n{row.game_label.replace('game', 'G')}")
            hatches.append("")

    fig_width = max(15.5, 0.72 * len(summary) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_width, 5.8))
    means = summary["payoff_gini_mean"].to_numpy(dtype=float)
    sems = summary["payoff_gini_sem"].to_numpy(dtype=float)
    bars = ax.bar(
        x_positions,
        means,
        yerr=sems,
        capsize=3,
        color=colors,
        edgecolor="#333333",
        linewidth=0.6,
        alpha=0.9,
    )
    for bar, hatch in zip(bars, hatches, strict=True):
        bar.set_hatch(hatch)
    section_cuts = []
    for left_index, right_index in [(1, 2), (7, 8)]:
        if right_index < len(x_positions):
            section_cuts.append((x_positions[left_index] + x_positions[right_index]) / 2)
    for cut in section_cuts:
        ax.axvline(cut, color="#777777", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=42, ha="right")
    ax.set_ylabel("Mean corrected within-run payoff Gini")
    ax.set_title("Payoff inequality: heterogeneous vs current homogeneous controls")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(0, max(0.45, float(np.nanmax(means + sems)) * 1.22))

    for bar, row in zip(bars, summary.itertuples(), strict=True):
        expected = row.expected
        if isinstance(expected, float) and math.isnan(expected):
            count_text = f"n={int(row.n_runs)}"
        elif row.bar_type == "aggregate" and row.bar_label == "Heterogeneous all":
            count_text = f"n={int(row.n_runs)}"
        else:
            count_text = f"{int(row.completed)}/{int(expected)}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.008,
            f"{row.payoff_gini_mean:.3f}\n{count_text}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=AGG_COLORS["Heterogeneous all"], label="Heterogeneous aggregate"),
        plt.Rectangle((0, 0), 1, 1, color=AGG_COLORS["Homogeneous all"], label="Homogeneous aggregate"),
        plt.Rectangle((0, 0), 1, 1, color=AGG_COLORS["Heterogeneous all"], hatch="\\\\", label="Heterogeneous game aggregate"),
        plt.Rectangle((0, 0), 1, 1, color="#777777", hatch="//", label="Homogeneous game aggregate"),
        plt.Rectangle((0, 0), 1, 1, color=GAME_COLORS["game1"], label="Homogeneous Game 1 model"),
        plt.Rectangle((0, 0), 1, 1, color=GAME_COLORS["game2"], label="Homogeneous Game 2 model"),
        plt.Rectangle((0, 0), 1, 1, color=GAME_COLORS["game3"], label="Homogeneous Game 3 model"),
    ]
    ax.legend(handles=legend_handles, frameon=False, ncol=4, loc="upper left")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_markdown(
    out_path: Path,
    plot_path: Path,
    summary: pd.DataFrame,
    hetero_path: Path,
    hom_root: Path,
) -> None:
    table = summary[
        [
            "bar_label",
            "bar_type",
            "game_label",
            "model_elo",
            "completed",
            "expected",
            "payoff_gini_mean",
            "payoff_gini_sem",
        ]
    ].copy()
    for col in ["model_elo", "payoff_gini_mean", "payoff_gini_sem"]:
        table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
    lines = [
        "# Heterogeneous vs Homogeneous Control Gini",
        "",
        f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Heterogeneous source: `{hetero_path}`",
        f"- Homogeneous source: `{hom_root}`",
        "- Metric: mean shifted, small-N-corrected within-run payoff Gini.",
        "",
        f"![Gini comparison]({plot_path.relative_to(out_path.parent).as_posix()})",
        "",
        "## Values",
        "",
        table.to_markdown(index=False),
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hom-results-root", type=Path, default=DEFAULT_HOM_ROOT)
    parser.add_argument("--heterogeneous-runs", type=Path, default=DEFAULT_HETERO_RUNS)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hom_root = args.hom_results_root.resolve()
    hetero_path = args.heterogeneous_runs.resolve()
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = hom_root / "analysis" / f"gini_vs_heterogeneous_{stamp}"
    else:
        out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    hetero = load_heterogeneous_runs(hetero_path)
    hom = load_homogeneous_runs(hom_root)
    if hetero.empty:
        raise SystemExit(f"No heterogeneous rows loaded from {hetero_path}")
    if hom.empty:
        raise SystemExit(f"No homogeneous completed rows loaded from {hom_root}")

    summary = build_summary(hetero, hom, hom_root)
    summary_path = out_dir / "gini_summary.csv"
    hetero_path_out = out_dir / "heterogeneous_gini_run_metrics.csv"
    hom_path_out = out_dir / "homogeneous_gini_run_metrics.csv"
    plot_path = out_dir / "heterogeneous_vs_homogeneous_gini_bars.png"
    md_path = out_dir / "heterogeneous_vs_homogeneous_gini_report.md"

    summary.to_csv(summary_path, index=False)
    hetero.to_csv(hetero_path_out, index=False)
    hom.to_csv(hom_path_out, index=False)
    plot_summary(summary, plot_path)
    write_markdown(md_path, plot_path, summary, hetero_path, hom_root)

    print(f"Wrote plot: {plot_path}")
    print(f"Wrote report: {md_path}")
    print(f"Wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

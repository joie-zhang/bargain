#!/usr/bin/env python3
"""Plot endpoint fair-share gaps from the canonical bilateral raw results.

The filename is retained for compatibility with the figure verification tools.
Following the rebuttal-era sensitivity analysis, the lowest-Elo Game 2 model
(`llama-3.2-1b-instruct`) is excluded only from the maximally competitive
baseline/adversary curves. All accepted Game 1 and Game 3 observations remain.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_n2_baseline_comparison import (  # noqa: E402
    BASELINES,
    load_baseline_rows,
    load_combined_elo_map,
    primary_protocol_rows,
)
from scripts.analyze_nash_lindahl_fairness import game3_lindahl_nbs  # noqa: E402
from strong_models_experiment.analysis.active_model_roster import (  # noqa: E402
    elo_for_model,
)


OUT_DIR = PROJECT_ROOT / "overleaf" / "neurips" / "graphics" / "n2_gpt5_nano" / "fairness_explanation"
ICML_OUT_DIR = PROJECT_ROOT / "overleaf" / "icml_aiwild_template" / "graphics" / "n2_gpt5_nano" / "fairness_explanation"

OUT_PNG = OUT_DIR / "baseline_adversary_fair_share_symmetric_percent_endpoints_tall_ewm_drop_game2_lowest_elo_style_matched.png"
OUT_BASELINE_ELO_PNG = (
    OUT_DIR
    / "baseline_adversary_fair_share_symmetric_percent_endpoints_tall_ewm_drop_game2_lowest_elo_style_matched_baseline_elo.png"
)
OUT_SYMMETRIC_BASELINE_ELO_PNG = (
    OUT_DIR
    / "baseline_adversary_fair_share_symmetric_percent_endpoints_tall_symmetric_ewm_drop_game2_lowest_elo_style_matched_baseline_elo.png"
)
OUT_CSV = OUT_DIR / "baseline_adversary_fair_share_symmetric_percent_endpoints_tall_ewm_drop_game2_lowest_elo_style_matched_model_means.csv"
OUT_CELLS = OUT_DIR / "baseline_adversary_fair_share_symmetric_percent_endpoints_tall_ewm_drop_game2_lowest_elo_style_matched_cells.csv"

EWM_ALPHA = 0.10
BASELINE_CANONICAL = "gpt-5-nano-high"
DROP_GAME2_MODEL = "llama-3.2-1b-instruct"

GAME_ORDER = ("game1", "game2", "game3")
GAME_LABELS = {
    "game1": "Game 1",
    "game2": "Game 2",
    "game3": "Game 3",
}
ENDPOINT_COLORS = {
    "max_cooperative": "#0b4f63",
    "max_competitive": "#48c7df",
}
ENDPOINT_LABELS = {
    "max_cooperative": "Max Cooperative",
    "max_competitive": "Max Competitive",
}


def symmetric_percent(actual: pd.Series, fair: pd.Series) -> pd.Series:
    actual_num = pd.to_numeric(actual, errors="coerce")
    fair_num = pd.to_numeric(fair, errors="coerce")
    denom = actual_num.abs() + fair_num.abs()
    out = 200.0 * (actual_num - fair_num) / denom
    out = out.mask(denom <= 1e-12, 0.0)
    return out.replace([np.inf, -np.inf], np.nan)


def game3_rebuttal_benchmark_utilities(
    primary: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Recompute the Game 3 benchmark used by the anonymous rebuttal figure.

    Unlike the realized-set Lindahl reference in the general bilateral loader,
    this enumerates feasible funded-project sets and selects the
    Lindahl-cost-sharing allocation with the largest Nash product.
    """

    game3 = primary[primary["game_id"].eq("game3")]
    benchmarks: dict[str, dict[str, float]] = {"baseline": {}, "adversary": {}}
    for row in game3.itertuples(index=False):
        result_path = str(row.result_path)
        payload = json.loads((PROJECT_ROOT / result_path).read_text(encoding="utf-8"))
        config = payload.get("config") or {}
        valuations = {
            str(agent): [float(value) for value in values]
            for agent, values in (payload.get("agent_preferences") or {}).items()
        }
        costs = [float(item["cost"]) for item in config.get("items", [])]
        budgets = {
            str(agent): float(value)
            for agent, value in (config.get("agent_budgets") or {}).items()
        }
        total_budget = float(config.get("total_budget") or sum(budgets.values()) or sum(costs))
        benchmark, _funded_set = game3_lindahl_nbs(
            valuations,
            costs,
            budgets,
            total_budget,
        )
        baseline_agent = str(row.baseline_agent)
        adversary_agent = str(row.adversary_agent)
        benchmarks["baseline"][result_path] = (
            float(benchmark[baseline_agent]) if baseline_agent in benchmark else np.nan
        )
        benchmarks["adversary"][result_path] = (
            float(benchmark[adversary_agent]) if adversary_agent in benchmark else np.nan
        )

    expected = int(len(game3))
    for role, values in benchmarks.items():
        if len(values) != expected:
            raise RuntimeError(
                f"Expected {expected} Game 3 {role} rebuttal benchmarks, found {len(values)}"
            )
    return benchmarks


def load_model_means() -> pd.DataFrame:
    spec = next(spec for spec in BASELINES if spec.key == "gpt5_nano")
    all_rows = load_baseline_rows(spec, load_combined_elo_map())
    primary = primary_protocol_rows(all_rows)
    if len(primary) != 1500:
        raise RuntimeError(f"Expected 1,500 primary GPT-5-nano runs, found {len(primary):,}")
    if primary["adversary_model"].astype(str).str.contains("phi", case=False).any():
        raise RuntimeError("Phi rows remain in the primary bilateral data")
    game1_turns = set(pd.to_numeric(primary.loc[primary["game_id"].eq("game1"), "discussion_turns"]))
    if game1_turns != {2}:
        raise RuntimeError(f"Expected only two-turn Game 1 rows, found {sorted(game1_turns)}")

    game3_benchmarks = game3_rebuttal_benchmark_utilities(primary)
    role_frames: list[pd.DataFrame] = []
    for role in ("baseline", "adversary"):
        actual_col = f"{role}_actual_utility_undiscounted"
        fair_col = f"{role}_fair_utility"
        role_rows = primary[
            ["result_path", "game_id", "competition_value", "adversary_model", "adversary_elo", actual_col, fair_col]
        ].copy()
        role_rows["role"] = role
        role_rows["actual_raw_utility"] = pd.to_numeric(role_rows[actual_col], errors="coerce")
        role_rows["fair_utility"] = pd.to_numeric(role_rows[fair_col], errors="coerce")
        game3_mask = role_rows["game_id"].eq("game3")
        role_rows.loc[game3_mask, "fair_utility"] = role_rows.loc[
            game3_mask, "result_path"
        ].map(game3_benchmarks[role])
        role_rows["signed_relative_gap_pct"] = symmetric_percent(
            role_rows["actual_raw_utility"], role_rows["fair_utility"]
        )
        role_frames.append(role_rows)
    metrics = pd.concat(role_frames, ignore_index=True)
    metrics = metrics.rename(columns={"adversary_model": "adversary_canonical", "adversary_elo": "adv_elo"})

    endpoint_frames: list[pd.DataFrame] = []
    for game_id, sub in metrics.groupby("game_id", sort=False):
        min_comp = float(sub["competition_value"].min())
        max_comp = float(sub["competition_value"].max())
        for endpoint, comp in (("max_cooperative", min_comp), ("max_competitive", max_comp)):
            endpoint_sub = sub[np.isclose(sub["competition_value"], comp)].copy()
            endpoint_sub["endpoint"] = endpoint
            endpoint_frames.append(endpoint_sub)

    endpoints = pd.concat(endpoint_frames, ignore_index=True)
    drop_mask = (
        endpoints["game_id"].eq("game2")
        & endpoints["endpoint"].eq("max_competitive")
        & endpoints["adversary_canonical"].eq(DROP_GAME2_MODEL)
    )
    dropped = endpoints.loc[drop_mask]
    if len(dropped) != 4 or set(dropped["role"]) != {"baseline", "adversary"}:
        raise RuntimeError(
            "Expected four Game 2 max-competitive run-role rows for the "
            f"rebuttal-era exclusion, found {len(dropped)}"
        )
    endpoints = endpoints.loc[~drop_mask].copy()

    means = (
        endpoints.groupby(
            ["game_id", "endpoint", "competition_value", "role", "adversary_canonical", "adv_elo"],
            as_index=False,
        )
        .agg(
            signed_relative_gap_pct=("signed_relative_gap_pct", "mean"),
            n_runs=("signed_relative_gap_pct", "count"),
        )
        .sort_values(["game_id", "endpoint", "role", "adv_elo"])
    )
    return means


def endpoint_handle(endpoint: str) -> mlines.Line2D:
    return mlines.Line2D(
        [],
        [],
        color=ENDPOINT_COLORS[endpoint],
        marker="o",
        linestyle="-",
        linewidth=3.2,
        markersize=6.0,
        label=ENDPOINT_LABELS[endpoint],
    )


def role_handle(role: str) -> mlines.Line2D:
    if role == "baseline":
        return mlines.Line2D(
            [],
            [],
            color="#111827",
            marker="o",
            markerfacecolor="#111827",
            markeredgecolor="#111827",
            linestyle="-",
            linewidth=3.2,
            markersize=6.0,
            label="Baseline",
        )
    return mlines.Line2D(
        [],
        [],
        color="#111827",
        marker="o",
        markerfacecolor="white",
        markeredgecolor="#111827",
        markeredgewidth=1.4,
        linestyle="-",
        linewidth=3.2,
        markersize=6.0,
        label="Adversary",
    )


def padded_ylim(values: pd.Series) -> tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return -1.0, 1.0
    lo = float(clean.min())
    hi = float(clean.max())
    lo = min(lo, 0.0)
    hi = max(hi, 0.0)
    pad = max((hi - lo) * 0.10, 4.0)
    return lo - pad, hi + pad


def baseline_elo() -> float:
    elo = elo_for_model(BASELINE_CANONICAL)
    if elo is None:
        raise RuntimeError(f"Could not find baseline Elo for {BASELINE_CANONICAL}")
    return float(elo)


def smooth_series(values: pd.Series, symmetric: bool = False) -> pd.Series:
    forward = values.ewm(alpha=EWM_ALPHA, adjust=False).mean()
    if not symmetric:
        return forward
    backward = values.iloc[::-1].ewm(alpha=EWM_ALPHA, adjust=False).mean().iloc[::-1]
    backward.index = values.index
    return (forward + backward) / 2.0


def plot(
    means: pd.DataFrame,
    out_png: Path,
    show_baseline_elo: bool = False,
    symmetric_smoothing: bool = False,
) -> pd.DataFrame:
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    # The 70%-scale canvas pairs with the 70%-scale LaTeX inclusion below, so
    # paper text remains legible while the displayed figure is 30% smaller.
    fig, axes = plt.subplots(1, 3, figsize=(7.46, 6.45), sharex=False, sharey=False)
    cell_rows: list[dict[str, object]] = []
    base_elo = baseline_elo()

    for ax, game_id in zip(axes, GAME_ORDER):
        game_sub = means[means["game_id"].eq(game_id)].copy()
        plotted_values: list[float] = []

        for endpoint in ("max_cooperative", "max_competitive"):
            for role in ("baseline", "adversary"):
                sub = game_sub[
                    game_sub["endpoint"].eq(endpoint) & game_sub["role"].eq(role)
                ].sort_values("adv_elo")
                if sub.empty:
                    continue

                y = smooth_series(sub["signed_relative_gap_pct"], symmetric=symmetric_smoothing)
                plotted_values.extend(y.dropna().tolist())
                color = ENDPOINT_COLORS[endpoint]
                is_baseline = role == "baseline"

                ax.plot(
                    sub["adv_elo"],
                    y,
                    color=color,
                    linestyle="-",
                    linewidth=3.0,
                    marker="o",
                    markersize=5.2,
                    markerfacecolor=color if is_baseline else "white",
                    markeredgecolor=color,
                    markeredgewidth=1.2,
                    alpha=0.96,
                )
                cell_rows.append(
                    {
                        "plot": out_png.name,
                        "role": role,
                        "game_id": game_id,
                        "endpoint": endpoint,
                        "competition_value": float(sub["competition_value"].iloc[0]),
                        "n_points": int(len(sub)),
                        "ewm_alpha": EWM_ALPHA,
                        "filter": (
                            "active roster; Game 1 discussion_turns=2; "
                            f"exclude {DROP_GAME2_MODEL} from Game 2 max_competitive only; "
                            "Game 3 uses enumerated Lindahl-cost-sharing Nash benchmark"
                        ),
                    }
                )

        ax.axhline(0, color="#6b7280", linewidth=1.1, alpha=0.82)
        if show_baseline_elo:
            ax.axvline(
                base_elo,
                color="#64748b",
                linestyle=(0, (4, 4)),
                linewidth=1.15,
                alpha=0.34,
                zorder=0,
            )
            ax.text(
                base_elo + 4,
                0.965,
                f"baseline Elo {base_elo:.0f}",
                transform=ax.get_xaxis_transform(),
                ha="left",
                va="top",
                fontsize=9.4,
                color="#64748b",
                alpha=0.78,
                rotation=90,
            )
        ax.grid(alpha=0.23, linewidth=0.8)
        ax.set_title(GAME_LABELS[game_id], fontsize=22, pad=11)
        ax.set_xlabel("Adversary Elo", fontsize=15, labelpad=8)
        if game_id == "game1":
            ax.set_ylabel("Signed relative fair-share gap (%)", fontsize=15, labelpad=10)
        ax.tick_params(axis="both", labelsize=12)
        ax.set_xlim(1088, 1515)
        ax.set_ylim(*padded_ylim(pd.Series(plotted_values)))
    endpoint_legend = fig.legend(
        handles=[endpoint_handle("max_cooperative"), endpoint_handle("max_competitive")],
        title="Competition",
        loc="upper center",
        bbox_to_anchor=(0.29, 0.985),
        ncol=2,
        fontsize=12.5,
        title_fontsize=12.5,
        frameon=True,
        handlelength=2.1,
        columnspacing=1.5,
    )
    endpoint_legend.get_frame().set_edgecolor("#d1d5db")
    endpoint_legend.get_frame().set_alpha(0.96)

    role_legend = fig.legend(
        handles=[role_handle("baseline"), role_handle("adversary")],
        title="Role",
        loc="upper center",
        bbox_to_anchor=(0.81, 0.985),
        ncol=2,
        fontsize=12.5,
        title_fontsize=12.5,
        frameon=True,
        handlelength=2.1,
        columnspacing=1.5,
    )
    role_legend.get_frame().set_edgecolor("#d1d5db")
    role_legend.get_frame().set_alpha(0.96)

    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.11, top=0.79, wspace=0.31)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return pd.DataFrame(cell_rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ICML_OUT_DIR.mkdir(parents=True, exist_ok=True)
    means = load_model_means()
    active_neurips = OUT_DIR / "baseline_adversary_fair_share_symmetric_percent_endpoints_tall_ewm.png"
    active_icml = ICML_OUT_DIR / active_neurips.name
    cells = plot(means, active_neurips)
    plot(means, active_icml)
    plot(means, OUT_BASELINE_ELO_PNG, show_baseline_elo=True)
    plot(
        means,
        OUT_SYMMETRIC_BASELINE_ELO_PNG,
        show_baseline_elo=True,
        symmetric_smoothing=True,
    )
    means.to_csv(OUT_CSV, index=False)
    cells.to_csv(OUT_CELLS, index=False)
    print(f"Wrote {active_neurips}")
    print(f"Wrote {active_icml}")
    print(f"Wrote {OUT_BASELINE_ELO_PNG}")
    print(f"Wrote {OUT_SYMMETRIC_BASELINE_ELO_PNG}")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_CELLS}")


if __name__ == "__main__":
    main()

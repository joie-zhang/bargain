#!/usr/bin/env python3
"""Plot N=2 group-intensity/payoff Spearman correlations with bootstrap CIs."""

from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ANALYSIS_DIR = Path("analysis/llm_strategic_tag_elo_exploration_n2_gpt5_intensity_20260629")


def spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 4 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.nan
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    if np.std(xr) == 0 or np.std(yr) == 0:
        return np.nan
    return float(np.corrcoef(xr, yr)[0, 1])


def bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    n = len(x)
    vals: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r = spearman_r(x[idx], y[idx])
        if np.isfinite(r):
            vals.append(r)
    if not vals:
        return np.nan, np.nan, 0
    arr = np.asarray(vals, dtype=float)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)), int(len(arr))


def group_seed(seed: int, group: object) -> int:
    return seed + zlib.crc32(str(group).encode("utf-8")) % 1_000_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG output path. Defaults to plots/payoff/group_model_rate_payoff_corr_bootstrap_ci.png",
    )
    args = parser.parse_args()

    analysis_dir = args.analysis_dir
    freq = pd.read_csv(analysis_dir / "group_model_frequency.csv")
    speaker = pd.read_csv(analysis_dir / "speaker_payoffs.csv")
    reported = pd.read_csv(analysis_dir / "group_payoff_model_correlations.csv")

    payoff = (
        speaker.dropna(subset=["final_utility"])
        .groupby(["experiment_family", "speaker_model", "speaker_role", "speaker_elo"], dropna=False)
        .agg(
            mean_final_utility=("final_utility", "mean"),
            speaker_rollouts=("speaker_key", "nunique"),
        )
        .reset_index()
    )

    merged = freq.merge(
        payoff,
        on=["experiment_family", "speaker_model", "speaker_role", "speaker_elo", "speaker_rollouts"],
        how="left",
    )

    rows: list[dict[str, object]] = []
    for group, sub in merged.groupby("group", dropna=False):
        sub = sub.dropna(subset=["events_per_speaker_rollout", "mean_final_utility"]).copy()
        x = sub["events_per_speaker_rollout"].to_numpy(dtype=float)
        y = sub["mean_final_utility"].to_numpy(dtype=float)
        rho = spearman_r(x, y)
        lo, hi, valid_boot = bootstrap_ci(x, y, args.n_bootstrap, group_seed(args.seed, group))
        report_row = reported.loc[reported["group"] == group]
        reported_n = int(report_row["n_models"].iloc[0]) if len(report_row) else np.nan
        reported_rho = (
            float(report_row["spearman_model_rate_vs_mean_utility"].iloc[0])
            if len(report_row)
            else np.nan
        )
        rows.append(
            {
                "group": group,
                "reported_n_models": reported_n,
                "effective_n_models": int(sub["speaker_model"].nunique()),
                "spearman_model_intensity_vs_mean_utility": rho,
                "reported_spearman_model_rate_vs_mean_utility": reported_rho,
                "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi,
                "valid_bootstrap_samples": valid_boot,
            }
        )

    out = pd.DataFrame(rows).sort_values("spearman_model_intensity_vs_mean_utility")
    out_path = args.output or analysis_dir / "plots" / "payoff" / "group_model_rate_payoff_corr_bootstrap_ci.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_path.with_suffix(".csv")
    out.to_csv(csv_path, index=False)

    y_pos = np.arange(len(out))
    rho = out["spearman_model_intensity_vs_mean_utility"].to_numpy(dtype=float)
    lo = out["bootstrap_ci_low"].to_numpy(dtype=float)
    hi = out["bootstrap_ci_high"].to_numpy(dtype=float)
    xerr = np.vstack([rho - lo, hi - rho])
    xerr = np.nan_to_num(xerr, nan=0.0)
    colors = np.where(rho >= 0, "#2563eb", "#dc2626")

    fig_h = max(4.8, 0.55 * len(out) + 1.6)
    fig, ax = plt.subplots(figsize=(8.8, fig_h))
    ax.barh(y_pos, rho, color=colors, alpha=0.82)
    ax.errorbar(
        rho,
        y_pos,
        xerr=xerr,
        fmt="none",
        ecolor="#111827",
        elinewidth=1.2,
        capsize=3.5,
        capthick=1.2,
        zorder=3,
    )
    ax.axvline(0, color="#111827", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(out["group"].astype(str), fontsize=9)
    effective_n = ", ".join(str(v) for v in sorted(out["effective_n_models"].dropna().unique()))
    reported_n = ", ".join(str(v) for v in sorted(out["reported_n_models"].dropna().unique()))
    ax.set_xlabel("Spearman rho: group intensity vs mean adversary utility")
    ax.set_title(
        "N=2 model-level payoff correlation by strategic category\n"
        f"bootstrap 95% CI over matched adversary-model payoff points (effective n={effective_n}; report n={reported_n})"
    )
    ax.set_xlim(min(-1.0, np.nanmin(lo) - 0.05), max(1.0, np.nanmax(hi) + 0.05))
    ax.grid(True, axis="x", alpha=0.32)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {out_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()

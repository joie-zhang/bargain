#!/usr/bin/env python3
"""Build the paper fair-share composite from canonical experiment results."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
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
from scripts.analyze_nash_lindahl_fairness import (  # noqa: E402
    AnalysisRow,
    analyze_row,
    normalize_game_id,
    recover_game1_preferences,
)
from strong_models_experiment.analysis.active_model_roster import elo_for_model  # noqa: E402


MANIFEST_CSV = PROJECT_ROOT / "docs" / "reproducibility" / "paper_experiment_data_manifest.csv"
FAIRNESS_CACHE_CSV = PROJECT_ROOT / "analysis" / "nash_lindahl_fairness_20260505" / "agent_metrics.csv"
OUT_DIRS = (
    PROJECT_ROOT / "overleaf" / "icml_aiwild_template" / "graphics" / "n2_gpt5_nano",
    PROJECT_ROOT / "overleaf" / "neurips" / "graphics" / "n2_gpt5_nano",
    PROJECT_ROOT / "overleaf" / "NExT_Game_2026_style_new" / "graphics" / "n2_gpt5_nano",
)
OUT_NAME = "fairshare_residual_combined.png"
SOURCE_ROWS_NAME = "fairshare_residual_combined_source_rows.csv"
SUMMARY_NAME = "fairshare_residual_combined_summary.csv"
PROVENANCE_NAME = "fairshare_residual_combined_provenance.json"

MULTIAGENT_FAMILIES = (
    "heterogeneous_random",
    "homogeneous_adversary",
    "homogeneous_control",
)
EXPECTED_MULTIAGENT_RUNS = {
    "heterogeneous_random": 1300,
    "homogeneous_adversary": 1300,
    "homogeneous_control": 130,
}
GAME_ORDER = ("game1", "game2", "game3")
GAME_LABELS = {
    "game1": "Game 1: Item allocation",
    "game2": "Game 2: Diplomacy",
    "game3": "Game 3: Co-funding",
}
GAME_COLORS = {"game1": "#1f77b4", "game2": "#d62728", "game3": "#2ca02c"}
SERIES_STYLES = {
    "heterogeneous_agent": ("Heterogeneous", "#17becf", "o", "-"),
    "homogeneous_adversary_adversary": ("Homogeneous adversary", "#9467bd", "s", "-"),
    "homogeneous_adversary_baseline": ("Homogeneous: baseline", "#9467bd", "D", "--"),
    "homogeneous_control": ("Homogeneous control", "#ff7f0e", "^", ":"),
}
MAIN_SERIES_KEYS = (
    "heterogeneous_agent",
    "homogeneous_adversary_adversary",
)


def sem(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 2:
        return 0.0
    return float(clean.std(ddof=1) / np.sqrt(len(clean)))


def path_set_hash(paths: set[str]) -> str:
    payload = "\n".join(sorted(paths)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_bilateral_rows() -> pd.DataFrame:
    spec = next(spec for spec in BASELINES if spec.key == "gpt5_nano")
    all_rows = load_baseline_rows(spec, load_combined_elo_map())
    rows = primary_protocol_rows(all_rows)
    if len(rows) != 1500:
        raise RuntimeError(f"Expected 1,500 primary bilateral runs, found {len(rows):,}")
    if rows["adversary_model"].astype(str).str.contains("phi", case=False).any():
        raise RuntimeError("Phi rows remain in the primary bilateral data")
    if rows["adversary_model"].nunique() != 30:
        raise RuntimeError(f"Expected 30 bilateral models, found {rows['adversary_model'].nunique()}")
    game1_turns = set(pd.to_numeric(rows.loc[rows["game_id"].eq("game1"), "discussion_turns"]))
    if game1_turns != {2}:
        raise RuntimeError(f"Expected only two-turn Game 1 rows, found {sorted(game1_turns)}")
    return rows


def analysis_row_from_result(relative_path: str) -> AnalysisRow:
    result_path = PROJECT_ROOT / relative_path
    with result_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if normalize_game_id(payload) == "game1" and not (payload.get("agent_preferences") or {}):
        payload = dict(payload)
        payload["agent_preferences"] = recover_game1_preferences(
            result_path,
            payload.get("config") or {},
        )
    config = payload.get("config") or {}
    family = str(config.get("experiment_family") or config.get("experiment_type") or "")
    source_group = "n_gt_2_heterogeneous" if family == "heterogeneous_random" else "n_gt_2_homogeneous"
    return AnalysisRow(
        source_group=source_group,
        dataset="paper_multiagent",
        game_id=normalize_game_id(payload),
        result_path=result_path,
        payload=payload,
        config=config,
        agent_model_map=dict(config.get("agent_model_map") or {}),
        agent_role_map=dict(config.get("agent_role_map") or {}),
        agent_elo_map=dict(config.get("agent_elo_map") or {}),
    )


def load_multiagent_agent_rows() -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = pd.read_csv(MANIFEST_CSV)
    selected = manifest[manifest["experiment_family"].isin(MULTIAGENT_FAMILIES)].copy()
    counts = selected.groupby("experiment_family").size().to_dict()
    if counts != EXPECTED_MULTIAGENT_RUNS:
        raise RuntimeError(f"Unexpected canonical multi-agent counts: {counts}")

    canonical_paths = set(selected["result_path"].astype(str))
    cache = pd.read_csv(FAIRNESS_CACHE_CSV)
    cache = cache[cache["result_path"].astype(str).isin(canonical_paths)].copy()
    cache = cache.drop_duplicates(["result_path", "agent_id"], keep="last")
    cached_paths = set(cache["result_path"].astype(str))
    missing_paths = sorted(canonical_paths - cached_paths)

    recomputed: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for index, relative_path in enumerate(missing_paths, start=1):
        try:
            _, agent_records = analyze_row(analysis_row_from_result(relative_path))
            recomputed.extend(agent_records)
        except Exception as exc:
            errors.append({"result_path": relative_path, "error": repr(exc)})
        if index % 100 == 0:
            print(f"Recomputed fairness for {index}/{len(missing_paths)} uncached canonical runs", flush=True)

    agents = pd.concat([cache, pd.DataFrame(recomputed)], ignore_index=True, sort=False)
    agents = agents[agents["result_path"].astype(str).isin(canonical_paths)].copy()
    agents = agents.drop_duplicates(["result_path", "agent_id"], keep="last")
    if agents["model"].astype(str).str.contains("phi", case=False).any():
        raise RuntimeError("Phi rows remain in the canonical multi-agent fairness data")

    covered_paths = set(agents["result_path"].astype(str))
    provenance = {
        "canonical_run_count": len(canonical_paths),
        "canonical_family_counts": counts,
        "canonical_result_path_set_sha256": path_set_hash(canonical_paths),
        "cache_hit_run_count": len(cached_paths),
        "recomputed_run_count": len(set(pd.DataFrame(recomputed).get("result_path", pd.Series(dtype=str)))),
        "covered_run_count": len(covered_paths),
        "unanalyzable_runs": errors,
        "phi_agent_row_count": int(agents["model"].astype(str).str.contains("phi", case=False).sum()),
    }
    return agents, provenance


def build_multiagent_series(agents: pd.DataFrame) -> pd.DataFrame:
    agents = agents.copy()
    agents["fairness_residual"] = np.where(
        agents["game_id"].eq("game3"),
        pd.to_numeric(agents["lindahl_residual"], errors="coerce"),
        pd.to_numeric(agents["nbs_residual"], errors="coerce"),
    )
    agents["elo"] = pd.to_numeric(agents["elo"], errors="coerce")
    frames: list[pd.DataFrame] = []

    heterogeneous = agents[agents["experiment_family"].eq("heterogeneous_random")].copy()
    heterogeneous["series_key"] = "heterogeneous_agent"
    heterogeneous["reference_elo"] = heterogeneous["elo"]
    heterogeneous["unit_kind"] = "agent"
    frames.append(heterogeneous)

    homogeneous = agents[agents["experiment_family"].eq("homogeneous_adversary")].copy()
    adversary = homogeneous[homogeneous["role"].eq("adversary")].copy()
    adversary["series_key"] = "homogeneous_adversary_adversary"
    adversary["reference_elo"] = adversary["elo"]
    adversary["unit_kind"] = "agent"
    frames.append(adversary)

    adversary_elo = adversary.groupby("result_path")["reference_elo"].first()
    baseline = homogeneous[homogeneous["role"].eq("baseline")].copy()
    baseline = (
        baseline.groupby(["result_path", "game_id", "n_agents", "experiment_family"], as_index=False)
        .agg(fairness_residual=("fairness_residual", "mean"))
    )
    baseline["series_key"] = "homogeneous_adversary_baseline"
    baseline["reference_elo"] = baseline["result_path"].map(adversary_elo)
    baseline["unit_kind"] = "within-run baseline mean"
    frames.append(baseline)

    control = agents[agents["experiment_family"].eq("homogeneous_control")].copy()
    control = (
        control.groupby(["result_path", "game_id", "n_agents", "experiment_family"], as_index=False)
        .agg(fairness_residual=("fairness_residual", "mean"))
    )
    control["series_key"] = "homogeneous_control"
    control["reference_elo"] = float(elo_for_model("gpt-5-nano-high") or 1337)
    control["unit_kind"] = "within-run agent mean"
    frames.append(control)

    columns = [
        "result_path",
        "game_id",
        "n_agents",
        "experiment_family",
        "series_key",
        "unit_kind",
        "reference_elo",
        "fairness_residual",
    ]
    rows = pd.concat([frame.reindex(columns=columns) for frame in frames], ignore_index=True)
    return rows.dropna(subset=["reference_elo", "fairness_residual"])


def summarize_bilateral(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["game_id", "adversary_model", "adversary_elo"], as_index=False)
        .agg(
            residual_mean=("adversary_fairness_excess", "mean"),
            residual_sem=("adversary_fairness_excess", sem),
            run_count=("result_path", "nunique"),
        )
        .sort_values(["game_id", "adversary_elo"])
    )


def summarize_multiagent(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["series_key", "reference_elo"], as_index=False)
        .agg(
            residual_mean=("fairness_residual", "mean"),
            residual_sem=("fairness_residual", sem),
            observation_count=("fairness_residual", "size"),
            run_count=("result_path", "nunique"),
        )
        .sort_values(["series_key", "reference_elo"])
    )


def add_fit(ax: plt.Axes, x: pd.Series, y: pd.Series, color: str) -> None:
    clean = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(clean) < 2 or clean["x"].nunique() < 2:
        return
    slope, intercept = np.polyfit(clean["x"], clean["y"], 1)
    xs = np.linspace(float(clean["x"].min()), float(clean["x"].max()), 200)
    ax.plot(xs, slope * xs + intercept, color=color, linestyle="--", linewidth=2.2)


def draw_figure(bilateral: pd.DataFrame, multiagent: pd.DataFrame) -> plt.Figure:
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 17, "axes.labelsize": 15})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))

    left = axes[0]
    for game_id in GAME_ORDER:
        sub = bilateral[bilateral["game_id"].eq(game_id)].sort_values("adversary_elo")
        color = GAME_COLORS[game_id]
        left.scatter(
            sub["adversary_elo"],
            sub["residual_mean"],
            s=27,
            color=color,
            alpha=0.55,
            edgecolors="none",
            label=GAME_LABELS[game_id],
        )
        add_fit(left, sub["adversary_elo"], sub["residual_mean"], color)
    left.axhline(0, color="#374151", linewidth=1.0)
    left.set_title("Bilateral (n=2)")
    left.set_xlabel("Adversary Elo")
    left.set_ylabel("Adversary utility above fair share")
    left.grid(alpha=0.22)
    left.legend(fontsize=9, frameon=True, loc="lower right")

    right = axes[1]
    for series_key in MAIN_SERIES_KEYS:
        label, color, marker, linestyle = SERIES_STYLES[series_key]
        sub = multiagent[multiagent["series_key"].eq(series_key)].sort_values("reference_elo")
        if sub.empty:
            continue
        right.errorbar(
            sub["reference_elo"],
            sub["residual_mean"],
            yerr=sub["residual_sem"],
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.0,
            markersize=5.2,
            elinewidth=1.05,
            capsize=2.5,
            capthick=1.05,
            alpha=0.95,
            label=label,
        )
    right.axhline(0, color="#374151", linewidth=1.0)
    right.set_title("Multi-agent (n=2, 4, 6, 8, 10)")
    right.set_xlabel("Elo")
    right.set_ylabel("Utility above fair share")
    right.grid(alpha=0.22)
    right.legend(fontsize=9, frameon=True, loc="lower right")

    fig.tight_layout(w_pad=3.0)
    return fig


def write_outputs(
    figure: plt.Figure,
    bilateral_summary: pd.DataFrame,
    multiagent_rows: pd.DataFrame,
    multiagent_summary: pd.DataFrame,
    provenance: dict[str, Any],
    out_dirs: tuple[Path, ...] = OUT_DIRS,
) -> None:
    combined_summary = pd.concat(
        [
            bilateral_summary.assign(panel="bilateral").rename(columns={"adversary_elo": "reference_elo"}),
            multiagent_summary.assign(panel="multiagent"),
        ],
        ignore_index=True,
        sort=False,
    )
    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(out_dir / OUT_NAME, dpi=220, bbox_inches="tight")
        multiagent_rows.to_csv(out_dir / SOURCE_ROWS_NAME, index=False)
        combined_summary.to_csv(out_dir / SUMMARY_NAME, index=False)
        (out_dir / PROVENANCE_NAME).write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out_dir / OUT_NAME}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--icml-only",
        action="store_true",
        help="Write only the ICML paper asset and its supporting files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bilateral_rows = load_bilateral_rows()
    multiagent_agents, provenance = load_multiagent_agent_rows()
    multiagent_rows = build_multiagent_series(multiagent_agents)
    bilateral_summary = summarize_bilateral(bilateral_rows)
    multiagent_summary = summarize_multiagent(multiagent_rows)

    provenance.update(
        {
            "bilateral_primary_run_count": int(len(bilateral_rows)),
            "bilateral_game_counts": bilateral_rows.groupby("game_id").size().to_dict(),
            "bilateral_model_count": int(bilateral_rows["adversary_model"].nunique()),
            "bilateral_game1_discussion_turns": sorted(
                pd.to_numeric(bilateral_rows.loc[bilateral_rows["game_id"].eq("game1"), "discussion_turns"])
                .astype(int)
                .unique()
                .tolist()
            ),
            "bilateral_phi_row_count": int(
                bilateral_rows["adversary_model"].astype(str).str.contains("phi", case=False).sum()
            ),
            "multiagent_source_row_count": int(len(multiagent_rows)),
            "inputs": {
                "paper_manifest": str(MANIFEST_CSV.relative_to(PROJECT_ROOT)),
                "fairness_cache": str(FAIRNESS_CACHE_CSV.relative_to(PROJECT_ROOT)),
                "bilateral_source": "canonical raw roots loaded by scripts/analyze_n2_baseline_comparison.py",
            },
        }
    )
    figure = draw_figure(bilateral_summary, multiagent_summary)
    out_dirs = (OUT_DIRS[0],) if args.icml_only else OUT_DIRS
    write_outputs(
        figure,
        bilateral_summary,
        multiagent_rows,
        multiagent_summary,
        provenance,
        out_dirs=out_dirs,
    )
    plt.close(figure)


if __name__ == "__main__":
    main()

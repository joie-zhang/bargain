#!/usr/bin/env python3
"""Generate the partial completed-run multi-agent plot report.

This script intentionally reads only completed SUCCESS runs from the two result
roots named in the report. It writes one Markdown report plus plot/CSV assets.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOMOGENEOUS_ROOT = (
    PROJECT_ROOT
    / "experiments/results/full_games123_multiagent_production_20260428_085255"
)
HETEROGENEOUS_ROOT = (
    PROJECT_ROOT
    / "experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848"
)
ELO_DOC = (
    PROJECT_ROOT
    / "docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md"
)
REPORT_PATH = (
    PROJECT_ROOT
    / "experiments/results/partial_multiagent_results_plot_report_20260503.md"
)
ASSET_DIR = REPORT_PATH.with_name(REPORT_PATH.stem + "_assets")


GAME_TITLES = {
    "game1": "Game 1: Item Allocation",
    "game2": "Game 2: Diplomacy",
    "game3": "Game 3: Co-funding",
}

FAMILY_TITLES = {
    "homogeneous_control": "Homogeneous control",
    "homogeneous_adversary": "Homogeneous adversary",
    "heterogeneous_random": "Heterogeneous",
}

MODEL_SHORT_NAMES = {
    "amazon-nova-micro-v1.0": "Nova Micro",
    "amazon-nova-pro-v1.0": "Nova Pro",
    "claude-3-haiku-20240307": "Claude 3 Haiku",
    "claude-haiku-4-5-20251001": "Haiku 4.5",
    "claude-opus-4-5-20251101": "Opus 4.5",
    "claude-opus-4-5-20251101-thinking-32k": "Opus 4.5 Thinking",
    "claude-opus-4-6": "Opus 4.6",
    "claude-opus-4-6-thinking": "Opus 4.6 Thinking",
    "claude-sonnet-4-20250514": "Sonnet 4",
    "command-r-plus-08-2024": "Command R+",
    "deepseek-r1": "DeepSeek R1",
    "deepseek-r1-0528": "DeepSeek R1-0528",
    "deepseek-v3": "DeepSeek V3",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-3.1-pro": "Gemini 3.1 Pro",
    "gemma-3-27b-it": "Gemma 3 27B",
    "gpt-4.1-nano-2025-04-14": "GPT-4.1 nano",
    "gpt-4o-2024-05-13": "GPT-4o",
    "gpt-4o-mini-2024-07-18": "GPT-4o mini",
    "gpt-5-nano": "GPT-5 nano",
    "gpt-5-nano-high": "GPT-5 nano",
    "gpt-5.2-chat-latest-20260210": "GPT-5.2 Chat",
    "gpt-5.4-high": "GPT-5.4 High",
    "llama-3.1-8b-instruct": "Llama 3.1 8B",
    "llama-3.2-1b-instruct": "Llama 3.2 1B",
    "llama-3.2-3b-instruct": "Llama 3.2 3B",
    "llama-3.3-70b-instruct": "Llama 3.3 70B",
    "o3-mini-high": "o3-mini-high",
    "qwen2.5-72b-instruct": "Qwen2.5 72B",
    "qwen3-max-preview": "Qwen3 Max",
    "qwq-32b": "QwQ 32B",
}

N_COLORS = {
    2: "#1f77b4",
    4: "#2ca02c",
    6: "#ff7f0e",
    8: "#9467bd",
    10: "#d62728",
}
LINE_COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#7f7f7f",
    "#bcbd22",
    "#e377c2",
]
MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h"]


def parse_model_elos(path: Path) -> dict[str, int]:
    elos: dict[str, int] = {}
    row_pattern = re.compile(r"^\|\s*\d+\s*\|\s*`([^`]+)`\s*\|\s*(\d+)\s*\|")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = row_pattern.match(line)
        if match:
            elos[match.group(1)] = int(match.group(2))
    if "gpt-5-nano-high" in elos:
        elos.setdefault("gpt-5-nano", elos["gpt-5-nano-high"])
    elos.setdefault("gpt-5-nano", 1337)
    return elos


def short_model_name(model: Any) -> str:
    if model is None or (isinstance(model, float) and math.isnan(model)):
        return ""
    return MODEL_SHORT_NAMES.get(str(model), str(model))


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return slug or "plot"


def agent_sort_key(agent_id: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", str(agent_id))
    if match:
        return int(match.group(1)), str(agent_id)
    return 10**9, str(agent_id)


def parse_config_id(path: Path) -> int | None:
    for part in path.parts:
        match = re.match(r"config_(\d{4})", part)
        if match:
            return int(match.group(1))
    match = re.search(r"config_(\d{4})", path.name)
    return int(match.group(1)) if match else None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def format_float(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    if abs(number - round(number)) < 10 ** (-(digits + 1)):
        return str(int(round(number)))
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def format_bool(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return ""


def competition_fields(config: dict[str, Any]) -> tuple[str, int, str]:
    game = config.get("game_label")
    competition_id = str(config.get("competition_id") or "")
    if game == "game1":
        comp = as_float(config.get("competition_level"))
        label = f"comp={comp:.2f}"
        return label, int(round(comp * 1000)), competition_id

    if game == "game2":
        theta = as_float(config.get("theta"))
        rho = as_float(config.get("rho"))
        rho_label = config.get("rho_label")
        if not rho_label:
            rho_label = "high_alignment" if rho >= 0 else "negative_lower_bound"
        pretty_rho = str(rho_label).replace("_", " ")
        label = f"{pretty_rho}, theta={theta:.1f}"
        rho_order = 0 if rho_label == "high_alignment" else 1
        theta_order = 0 if theta < 0.5 else 1
        detail = f"{competition_id}; rho={rho:.6g}; theta={theta:.1f}"
        return label, rho_order * 10 + theta_order, detail

    if game == "game3":
        sigma = as_float(config.get("sigma"))
        alpha = as_float(config.get("alpha"))
        label = f"sigma={sigma:.1f}, alpha={alpha:.1f}"
        order = int(round(sigma * 10)) * 10 + int(round(alpha * 10))
        detail = f"{competition_id}; sigma={sigma:.1f}; alpha={alpha:.1f}"
        return label, order, detail

    return competition_id or "unknown", 9999, competition_id or "unknown"


def resolve_elo(
    agent_id: str,
    model: Any,
    agent_elo_map: dict[str, Any],
    model_elos: dict[str, int],
) -> float:
    value = agent_elo_map.get(agent_id)
    if value is not None and value != "":
        return float(value)
    if model is not None and str(model) in model_elos:
        return float(model_elos[str(model)])
    return math.nan


def gini_nonnegative(values: Iterable[float]) -> tuple[float, bool]:
    arr = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=float)
    if arr.size == 0:
        return math.nan, False
    shifted = False
    min_value = float(arr.min())
    if min_value < 0:
        arr = arr - min_value
        shifted = True
    if np.allclose(arr, 0.0):
        return 0.0, shifted
    mean_value = float(arr.mean())
    if math.isclose(mean_value, 0.0):
        return 0.0, shifted
    diffs = np.abs(arr[:, None] - arr[None, :])
    return float(diffs.mean() / (2.0 * mean_value)), shifted


def linear_stats(frame: pd.DataFrame, x_col: str, y_col: str) -> dict[str, float]:
    clean = frame[[x_col, y_col]].dropna()
    clean = clean[np.isfinite(clean[x_col]) & np.isfinite(clean[y_col])]
    if len(clean) < 2 or clean[x_col].nunique() < 2:
        return {"n": float(len(clean)), "slope": math.nan, "corr": math.nan}
    x = clean[x_col].astype(float).to_numpy()
    y = clean[y_col].astype(float).to_numpy()
    slope, _ = np.polyfit(x, y, deg=1)
    corr = 0.0 if np.allclose(y, y[0]) else float(np.corrcoef(x, y)[0, 1])
    return {"n": float(len(clean)), "slope": float(slope), "corr": corr}


def iter_completed_payloads(results_root: Path) -> Iterable[tuple[dict[str, Any], dict[str, Any], Path]]:
    for status_path in sorted((results_root / "status").glob("config_*.json")):
        status = load_json(status_path)
        if status.get("state") != "SUCCESS":
            continue
        result_path_value = status.get("result_path")
        if not result_path_value:
            continue
        result_path = Path(result_path_value)
        if not result_path.exists():
            continue
        yield status, load_json(result_path), result_path


def build_tables(model_elos: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, Any]] = []
    agent_rows: list[dict[str, Any]] = []

    for source_label, results_root in [
        ("homogeneous", HOMOGENEOUS_ROOT),
        ("heterogeneous", HETEROGENEOUS_ROOT),
    ]:
        for status, payload, result_path in iter_completed_payloads(results_root):
            config = payload.get("config") or {}
            final_utilities = payload.get("final_utilities") or {}
            agent_model_map = config.get("agent_model_map") or {}
            agent_role_map = config.get("agent_role_map") or {}
            agent_elo_map = config.get("agent_elo_map") or {}
            config_id = int(config.get("config_id") or status.get("config_id") or parse_config_id(result_path))
            competition_label, competition_order, competition_detail = competition_fields(config)
            n_agents = int(config.get("n_agents") or len(final_utilities))
            game_label = config.get("game_label") or "unknown"
            family = config.get("experiment_family") or config.get("experiment_type") or source_label
            utilities = [
                float(value)
                for value in final_utilities.values()
                if value is not None and math.isfinite(float(value))
            ]
            gini_value, gini_shifted = gini_nonnegative(utilities)
            elos_for_run = [
                resolve_elo(agent_id, agent_model_map.get(agent_id), agent_elo_map, model_elos)
                for agent_id in sorted(set(agent_model_map) | set(final_utilities), key=agent_sort_key)
            ]
            elos_for_run = [elo for elo in elos_for_run if math.isfinite(elo)]
            recomputed_elo_variance = float(np.var(elos_for_run)) if elos_for_run else math.nan
            stored_elo_variance = as_float(config.get("elo_variance"))
            elo_variance = stored_elo_variance if math.isfinite(stored_elo_variance) else recomputed_elo_variance

            run_row = {
                "source": source_label,
                "config_id": config_id,
                "result_path": str(result_path),
                "experiment_family": family,
                "game_label": game_label,
                "game_title": GAME_TITLES.get(game_label, game_label),
                "game_type": config.get("game_type"),
                "n_agents": n_agents,
                "competition_label": competition_label,
                "competition_order": competition_order,
                "competition_detail": competition_detail,
                "competition_id": config.get("competition_id"),
                "competition_level": config.get("competition_level"),
                "rho": config.get("rho"),
                "rho_label": config.get("rho_label"),
                "theta": config.get("theta"),
                "sigma": config.get("sigma"),
                "alpha": config.get("alpha"),
                "seed_replicate": config.get("seed_replicate"),
                "heterogeneous_run_index": config.get("heterogeneous_run_index"),
                "run_index": config.get("seed_replicate")
                if family != "heterogeneous_random"
                else config.get("heterogeneous_run_index"),
                "random_seed": config.get("random_seed") or config.get("seed"),
                "model_order": config.get("model_order"),
                "adversary_model": config.get("adversary_model"),
                "adversary_position": config.get("adversary_position"),
                "consensus_reached": payload.get("consensus_reached"),
                "final_round": payload.get("final_round"),
                "utility_count": len(utilities),
                "mean_utility": float(np.mean(utilities)) if utilities else math.nan,
                "sum_utility": float(np.sum(utilities)) if utilities else math.nan,
                "utility_min": float(np.min(utilities)) if utilities else math.nan,
                "utility_max": float(np.max(utilities)) if utilities else math.nan,
                "utility_std": float(np.std(utilities)) if utilities else math.nan,
                "utility_gini": gini_value,
                "utility_gini_shifted": bool(gini_shifted),
                "negative_utility_count": int(sum(value < 0 for value in utilities)),
                "all_zero_utilities": bool(utilities and np.allclose(utilities, 0.0)),
                "elo_variance": elo_variance,
                "elo_variance_recomputed": recomputed_elo_variance,
                "elo_stddev": math.sqrt(elo_variance) if math.isfinite(elo_variance) else math.nan,
            }
            run_rows.append(run_row)

            agent_ids = sorted(set(final_utilities) | set(agent_model_map) | set(agent_role_map), key=agent_sort_key)
            for agent_id in agent_ids:
                utility = final_utilities.get(agent_id)
                utility_float = float(utility) if utility is not None else math.nan
                model = agent_model_map.get(agent_id)
                role = agent_role_map.get(agent_id)
                elo = resolve_elo(agent_id, model, agent_elo_map, model_elos)
                agent_index = agent_sort_key(agent_id)[0]
                normalized_position = (
                    (agent_index - 1) / (n_agents - 1) if n_agents > 1 and agent_index < 10**9 else math.nan
                )
                agent_rows.append(
                    {
                        **run_row,
                        "agent_id": agent_id,
                        "agent_index": agent_index,
                        "normalized_position": normalized_position,
                        "model": model,
                        "model_short": short_model_name(model),
                        "role": role,
                        "elo": elo,
                        "final_utility": utility_float,
                        "is_adversary": role == "adversary",
                        "is_baseline": role in {"baseline", "baseline_control"},
                    }
                )

    runs = pd.DataFrame(run_rows).sort_values(["source", "config_id"]).reset_index(drop=True)
    agents = pd.DataFrame(agent_rows).sort_values(["source", "config_id", "agent_index"]).reset_index(drop=True)
    return runs, agents


def aggregate_agents(
    agents: pd.DataFrame,
    group_cols: list[str],
    utility_col: str = "final_utility",
) -> pd.DataFrame:
    return (
        agents.groupby(group_cols, dropna=False)
        .agg(
            mean_utility=(utility_col, "mean"),
            utility_std=(utility_col, "std"),
            sample_count=(utility_col, "count"),
            completed_runs=("config_id", "nunique"),
        )
        .reset_index()
    )


def aggregate_runs(runs: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        runs.groupby(group_cols, dropna=False)
        .agg(
            mean_utility=("mean_utility", "mean"),
            utility_gini=("utility_gini", "mean"),
            elo_variance=("elo_variance", "mean"),
            elo_stddev=("elo_stddev", "mean"),
            sample_count=("config_id", "count"),
            consensus_rate=("consensus_reached", "mean"),
            mean_final_round=("final_round", "mean"),
        )
        .reset_index()
    )


def sort_summary(frame: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [
        col
        for col in [
            "game_label",
            "competition_order",
            "competition_label",
            "n_agents",
            "adversary_position",
            "elo",
            "agent_index",
            "elo_bin",
        ]
        if col in frame.columns
    ]
    return frame.sort_values(sort_cols).reset_index(drop=True) if sort_cols else frame


def rel_path(path: Path) -> str:
    return str(path.relative_to(REPORT_PATH.parent))


def setup_axis(ax: plt.Axes, title: str, x_label: str, y_label: str) -> None:
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.grid(True, alpha=0.22)


def ordered_group_values(frame: pd.DataFrame, group_col: str) -> list[Any]:
    values = frame[group_col].dropna().unique().tolist()
    if group_col == "n_agents":
        return sorted(values, key=lambda value: int(value))
    if group_col == "competition_label" and "competition_order" in frame.columns:
        order_frame = frame[["competition_label", "competition_order"]].drop_duplicates()
        order_frame = order_frame.sort_values(["competition_order", "competition_label"])
        return order_frame["competition_label"].tolist()
    return sorted(values, key=lambda value: str(value))


def plot_grouped_line(
    frame: pd.DataFrame,
    path: Path,
    *,
    x_col: str,
    y_col: str,
    group_col: str,
    title: str,
    x_label: str,
    y_label: str,
    sample_label: str,
    annotate: bool = True,
    width: float = 10.5,
    height: float = 6.6,
) -> None:
    plot_df = frame.dropna(subset=[x_col, y_col]).copy()
    if plot_df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(width, height))
    max_count = max(1, int(plot_df["sample_count"].max())) if "sample_count" in plot_df else 1

    for idx, group_value in enumerate(ordered_group_values(plot_df, group_col)):
        group = plot_df[plot_df[group_col].eq(group_value)].sort_values(x_col)
        if group.empty:
            continue
        color = N_COLORS.get(int(group_value), LINE_COLORS[idx % len(LINE_COLORS)]) if group_col == "n_agents" else LINE_COLORS[idx % len(LINE_COLORS)]
        marker = MARKERS[idx % len(MARKERS)]
        label = f"{group_value} ({sample_label}={int(group['sample_count'].sum())})"
        if group_col == "n_agents":
            label = f"N={int(group_value)} ({sample_label}={int(group['sample_count'].sum())})"
        ax.plot(group[x_col], group[y_col], color=color, linewidth=1.45, alpha=0.78)
        sizes = 34 + 126 * np.sqrt(group["sample_count"].astype(float).clip(lower=1) / max_count)
        ax.scatter(
            group[x_col],
            group[y_col],
            s=sizes,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            alpha=0.9,
            label=label,
            zorder=3,
        )
        if annotate:
            for _, row in group.iterrows():
                ax.annotate(
                    f"n={int(row['sample_count'])}",
                    (row[x_col], row[y_col]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=6.5,
                    color=color,
                    alpha=0.85,
                )

    setup_axis(ax, title, x_label, y_label)
    ax.legend(fontsize=7.5, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_grouped_scatter(
    frame: pd.DataFrame,
    path: Path,
    *,
    x_col: str,
    y_col: str,
    group_col: str,
    title: str,
    x_label: str,
    y_label: str,
    width: float = 10.5,
    height: float = 6.7,
) -> None:
    plot_df = frame.dropna(subset=[x_col, y_col]).copy()
    if plot_df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(width, height))
    for idx, group_value in enumerate(ordered_group_values(plot_df, group_col)):
        group = plot_df[plot_df[group_col].eq(group_value)]
        color = N_COLORS.get(int(group_value), LINE_COLORS[idx % len(LINE_COLORS)]) if group_col == "n_agents" else LINE_COLORS[idx % len(LINE_COLORS)]
        marker = MARKERS[idx % len(MARKERS)]
        label = f"{group_value} (runs={len(group)})"
        if group_col == "n_agents":
            label = f"N={int(group_value)} (runs={len(group)})"
        ax.scatter(
            group[x_col],
            group[y_col],
            s=34,
            color=color,
            marker=marker,
            alpha=0.58,
            edgecolor="none",
            label=label,
        )
        if len(group) >= 2 and group[x_col].nunique() >= 2:
            x = group[x_col].astype(float).to_numpy()
            y = group[y_col].astype(float).to_numpy()
            slope, intercept = np.polyfit(x, y, deg=1)
            xs = np.linspace(float(np.min(x)), float(np.max(x)), 60)
            ax.plot(xs, slope * xs + intercept, color=color, linewidth=1.25, alpha=0.8)

    setup_axis(ax, title, x_label, y_label)
    ax.legend(fontsize=7.5, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def make_binned_summary(
    frame: pd.DataFrame,
    group_cols: list[str],
    *,
    x_col: str,
    y_col: str,
    max_bins: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.dropna(subset=[x_col, y_col]).groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        group = group.copy()
        unique_x = group[x_col].nunique()
        bins = min(max_bins, max(1, unique_x), max(1, len(group) // 4))
        if bins <= 1:
            group["_bin"] = 0
        else:
            try:
                group["_bin"] = pd.qcut(group[x_col], q=bins, labels=False, duplicates="drop")
            except ValueError:
                group["_bin"] = 0
        for bin_id, bin_group in group.groupby("_bin", dropna=False):
            row = {col: key for col, key in zip(group_cols, keys)}
            row.update(
                {
                    "elo_bin": int(bin_id) if pd.notna(bin_id) else 0,
                    "elo_variance": float(bin_group[x_col].mean()),
                    "utility_gini": float(bin_group[y_col].mean()),
                    "sample_count": int(len(bin_group)),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def df_to_markdown(frame: pd.DataFrame, columns: list[str] | None = None, max_rows: int | None = None) -> str:
    if frame.empty:
        return "_No rows._"
    table = frame.copy()
    if columns is not None:
        table = table[[col for col in columns if col in table.columns]]
    if max_rows is not None:
        table = table.head(max_rows)
    display_rows: list[list[str]] = []
    headers = list(table.columns)
    for _, row in table.iterrows():
        display_row: list[str] = []
        for col in headers:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                display_row.append(format_float(value, 3))
            elif isinstance(value, (bool, np.bool_)):
                display_row.append(format_bool(bool(value)))
            elif pd.isna(value):
                display_row.append("")
            else:
                display_row.append(str(value))
        display_rows.append(display_row)
    widths = [
        max(len(str(header)), *(len(row[idx]) for row in display_rows))
        for idx, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
        for row in display_rows
    ]
    return "\n".join([header_line, sep_line, *body])


def write_csv(frame: pd.DataFrame, filename: str) -> Path:
    path = ASSET_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def build_plot_data(runs: pd.DataFrame, agents: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], list[dict[str, str]]]:
    plot_tables: dict[str, pd.DataFrame] = {}
    plot_inventory: list[dict[str, str]] = []
    plot_paths: dict[str, list[Path]] = defaultdict(list)

    hom_adv = agents[
        agents["experiment_family"].eq("homogeneous_adversary") & agents["role"].eq("adversary")
    ].copy()
    hom_ctrl = agents[agents["experiment_family"].eq("homogeneous_control")].copy()
    hetero_agents = agents[agents["experiment_family"].eq("heterogeneous_random")].copy()
    hetero_runs = runs[runs["experiment_family"].eq("heterogeneous_random")].copy()

    hom_adv_by_n = sort_summary(
        aggregate_agents(
            hom_adv,
            ["game_label", "n_agents", "model", "model_short", "elo"],
        )
    )
    plot_tables["hom_adv_by_n"] = hom_adv_by_n

    hom_adv_by_comp = sort_summary(
        aggregate_agents(
            hom_adv,
            ["game_label", "competition_label", "competition_order", "model", "model_short", "elo"],
        )
    )
    plot_tables["hom_adv_by_competition"] = hom_adv_by_comp

    hom_adv_by_position = sort_summary(
        aggregate_agents(
            hom_adv,
            ["game_label", "adversary_position", "model", "model_short", "elo"],
        )
    )
    plot_tables["hom_adv_by_position"] = hom_adv_by_position

    pos_base = sort_summary(
        aggregate_agents(
            hom_adv,
            ["game_label", "n_agents", "adversary_position", "model", "model_short", "elo"],
        )
    )
    gap_rows: list[dict[str, Any]] = []
    for keys, group in pos_base.groupby(["game_label", "n_agents", "model", "model_short", "elo"], dropna=False):
        game_label, n_agents, model, model_short, elo = keys
        first = group[group["adversary_position"].eq("first")]
        last = group[group["adversary_position"].eq("last")]
        if first.empty or last.empty:
            continue
        gap_rows.append(
            {
                "game_label": game_label,
                "n_agents": n_agents,
                "model": model,
                "model_short": model_short,
                "elo": elo,
                "position_utility_gap_first_minus_last": float(first["mean_utility"].iloc[0] - last["mean_utility"].iloc[0]),
                "sample_count": int(first["sample_count"].iloc[0] + last["sample_count"].iloc[0]),
                "first_sample_count": int(first["sample_count"].iloc[0]),
                "last_sample_count": int(last["sample_count"].iloc[0]),
            }
        )
    hom_adv_position_gap = sort_summary(pd.DataFrame(gap_rows))
    plot_tables["hom_adv_position_gap_by_n"] = hom_adv_position_gap

    ctrl_by_position_n = sort_summary(
        aggregate_agents(
            hom_ctrl,
            ["game_label", "n_agents", "agent_index"],
        )
    )
    plot_tables["hom_ctrl_by_position_n"] = ctrl_by_position_n

    ctrl_by_position_comp = sort_summary(
        aggregate_agents(
            hom_ctrl,
            ["game_label", "competition_label", "competition_order", "agent_index"],
        )
    )
    plot_tables["hom_ctrl_by_position_competition"] = ctrl_by_position_comp

    ctrl_scaling = sort_summary(
        aggregate_runs(
            runs[runs["experiment_family"].eq("homogeneous_control")],
            ["game_label", "n_agents"],
        )
    )
    plot_tables["hom_ctrl_scaling"] = ctrl_scaling

    ctrl_scaling_comp = sort_summary(
        aggregate_runs(
            runs[runs["experiment_family"].eq("homogeneous_control")],
            ["game_label", "competition_label", "competition_order", "n_agents"],
        )
    )
    plot_tables["hom_ctrl_scaling_competition"] = ctrl_scaling_comp

    hetero_perf_by_n = sort_summary(
        aggregate_agents(
            hetero_agents,
            ["game_label", "n_agents", "model", "model_short", "elo"],
        )
    )
    plot_tables["hetero_perf_by_n"] = hetero_perf_by_n

    hetero_perf_by_comp = sort_summary(
        aggregate_agents(
            hetero_agents,
            ["game_label", "competition_label", "competition_order", "model", "model_short", "elo"],
        )
    )
    plot_tables["hetero_perf_by_competition"] = hetero_perf_by_comp

    hetero_position = sort_summary(
        aggregate_agents(
            hetero_agents,
            ["game_label", "n_agents", "agent_index"],
        )
    )
    plot_tables["hetero_position"] = hetero_position

    hetero_gini_by_n_binned = make_binned_summary(
        hetero_runs,
        ["game_label", "n_agents"],
        x_col="elo_variance",
        y_col="utility_gini",
    )
    hetero_gini_by_n_binned = sort_summary(hetero_gini_by_n_binned)
    plot_tables["hetero_gini_by_n_binned"] = hetero_gini_by_n_binned

    hetero_gini_by_comp_binned = make_binned_summary(
        hetero_runs,
        ["game_label", "competition_label", "competition_order"],
        x_col="elo_variance",
        y_col="utility_gini",
    )
    hetero_gini_by_comp_binned = sort_summary(hetero_gini_by_comp_binned)
    plot_tables["hetero_gini_by_competition_binned"] = hetero_gini_by_comp_binned

    for game_label in ["game1", "game2", "game3"]:
        game_title = GAME_TITLES[game_label]

        frame = hom_adv_by_n[hom_adv_by_n["game_label"].eq(game_label)]
        path = ASSET_DIR / f"hom_adv_{game_label}_utility_vs_elo_by_n.png"
        plot_grouped_line(
            frame,
            path,
            x_col="elo",
            y_col="mean_utility",
            group_col="n_agents",
            title=f"{game_title}: homogeneous adversary utility vs Elo by N",
            x_label="Adversary model Elo",
            y_label="Mean adversary final utility",
            sample_label="runs",
            annotate=True,
        )
        plot_paths["hom_adv"].append(path)

        frame = hom_adv_by_comp[hom_adv_by_comp["game_label"].eq(game_label)]
        path = ASSET_DIR / f"hom_adv_{game_label}_utility_vs_elo_by_competition.png"
        plot_grouped_line(
            frame,
            path,
            x_col="elo",
            y_col="mean_utility",
            group_col="competition_label",
            title=f"{game_title}: homogeneous adversary utility vs Elo by competition",
            x_label="Adversary model Elo",
            y_label="Mean adversary final utility",
            sample_label="runs",
            annotate=True,
        )
        plot_paths["hom_adv"].append(path)

        frame = hom_adv_by_position[hom_adv_by_position["game_label"].eq(game_label)]
        path = ASSET_DIR / f"hom_adv_{game_label}_utility_vs_elo_by_position.png"
        plot_grouped_line(
            frame,
            path,
            x_col="elo",
            y_col="mean_utility",
            group_col="adversary_position",
            title=f"{game_title}: homogeneous adversary utility vs Elo by model order",
            x_label="Adversary model Elo",
            y_label="Mean adversary final utility",
            sample_label="runs",
            annotate=True,
        )
        plot_paths["hom_adv"].append(path)

        frame = hom_adv_position_gap[hom_adv_position_gap["game_label"].eq(game_label)]
        path = ASSET_DIR / f"hom_adv_{game_label}_first_minus_last_gap_by_n.png"
        plot_grouped_line(
            frame,
            path,
            x_col="elo",
            y_col="position_utility_gap_first_minus_last",
            group_col="n_agents",
            title=f"{game_title}: adversary order gap by N (first - last)",
            x_label="Adversary model Elo",
            y_label="Mean utility gap: first minus last",
            sample_label="runs",
            annotate=True,
        )
        plot_paths["hom_adv"].append(path)

        frame = ctrl_by_position_n[ctrl_by_position_n["game_label"].eq(game_label)]
        path = ASSET_DIR / f"hom_ctrl_{game_label}_utility_by_position_and_n.png"
        plot_grouped_line(
            frame,
            path,
            x_col="agent_index",
            y_col="mean_utility",
            group_col="n_agents",
            title=f"{game_title}: homogeneous control utility by position and N",
            x_label="Agent position",
            y_label="Mean final utility",
            sample_label="agent obs",
            annotate=True,
        )
        plot_paths["hom_ctrl"].append(path)

        frame = ctrl_by_position_comp[ctrl_by_position_comp["game_label"].eq(game_label)]
        path = ASSET_DIR / f"hom_ctrl_{game_label}_utility_by_position_and_competition.png"
        plot_grouped_line(
            frame,
            path,
            x_col="agent_index",
            y_col="mean_utility",
            group_col="competition_label",
            title=f"{game_title}: homogeneous control utility by position and competition",
            x_label="Agent position",
            y_label="Mean final utility",
            sample_label="agent obs",
            annotate=True,
        )
        plot_paths["hom_ctrl"].append(path)

        frame = ctrl_scaling_comp[ctrl_scaling_comp["game_label"].eq(game_label)]
        path = ASSET_DIR / f"hom_ctrl_{game_label}_scaling_by_competition.png"
        plot_grouped_line(
            frame,
            path,
            x_col="n_agents",
            y_col="mean_utility",
            group_col="competition_label",
            title=f"{game_title}: homogeneous control mean utility as N scales",
            x_label="N",
            y_label="Mean run-level utility",
            sample_label="runs",
            annotate=True,
        )
        plot_paths["hom_ctrl"].append(path)

        frame = hetero_perf_by_n[hetero_perf_by_n["game_label"].eq(game_label)]
        path = ASSET_DIR / f"hetero_{game_label}_utility_vs_elo_by_n.png"
        plot_grouped_line(
            frame,
            path,
            x_col="elo",
            y_col="mean_utility",
            group_col="n_agents",
            title=f"{game_title}: heterogeneous utility vs Elo by N",
            x_label="Agent model Elo",
            y_label="Mean final utility",
            sample_label="agent obs",
            annotate=True,
            width=11.8,
            height=7.2,
        )
        plot_paths["hetero_perf"].append(path)

        frame = hetero_perf_by_comp[hetero_perf_by_comp["game_label"].eq(game_label)]
        path = ASSET_DIR / f"hetero_{game_label}_utility_vs_elo_by_competition.png"
        plot_grouped_line(
            frame,
            path,
            x_col="elo",
            y_col="mean_utility",
            group_col="competition_label",
            title=f"{game_title}: heterogeneous utility vs Elo by competition",
            x_label="Agent model Elo",
            y_label="Mean final utility",
            sample_label="agent obs",
            annotate=True,
            width=11.8,
            height=7.2,
        )
        plot_paths["hetero_perf"].append(path)

        frame = hetero_position[hetero_position["game_label"].eq(game_label)]
        path = ASSET_DIR / f"hetero_{game_label}_utility_by_position_and_n.png"
        plot_grouped_line(
            frame,
            path,
            x_col="agent_index",
            y_col="mean_utility",
            group_col="n_agents",
            title=f"{game_title}: heterogeneous utility by random order position",
            x_label="Agent position",
            y_label="Mean final utility",
            sample_label="agent obs",
            annotate=True,
        )
        plot_paths["hetero_perf"].append(path)

        frame = hetero_runs[hetero_runs["game_label"].eq(game_label)]
        path = ASSET_DIR / f"hetero_{game_label}_gini_vs_elo_variance_scatter_by_n.png"
        plot_grouped_scatter(
            frame,
            path,
            x_col="elo_variance",
            y_col="utility_gini",
            group_col="n_agents",
            title=f"{game_title}: run utility Gini vs within-run Elo variance by N",
            x_label="Within-run Elo variance",
            y_label="Final utility Gini",
        )
        plot_paths["hetero_gini"].append(path)

        frame = hetero_gini_by_n_binned[hetero_gini_by_n_binned["game_label"].eq(game_label)]
        path = ASSET_DIR / f"hetero_{game_label}_gini_vs_elo_variance_binned_by_n.png"
        plot_grouped_line(
            frame,
            path,
            x_col="elo_variance",
            y_col="utility_gini",
            group_col="n_agents",
            title=f"{game_title}: binned mean Gini vs Elo variance by N",
            x_label="Mean within-bin Elo variance",
            y_label="Mean final utility Gini",
            sample_label="runs",
            annotate=True,
        )
        plot_paths["hetero_gini"].append(path)

        frame = hetero_runs[hetero_runs["game_label"].eq(game_label)]
        path = ASSET_DIR / f"hetero_{game_label}_gini_vs_elo_variance_scatter_by_competition.png"
        plot_grouped_scatter(
            frame,
            path,
            x_col="elo_variance",
            y_col="utility_gini",
            group_col="competition_label",
            title=f"{game_title}: run utility Gini vs Elo variance by competition",
            x_label="Within-run Elo variance",
            y_label="Final utility Gini",
        )
        plot_paths["hetero_gini"].append(path)

        frame = hetero_gini_by_comp_binned[
            hetero_gini_by_comp_binned["game_label"].eq(game_label)
        ]
        path = ASSET_DIR / f"hetero_{game_label}_gini_vs_elo_variance_binned_by_competition.png"
        plot_grouped_line(
            frame,
            path,
            x_col="elo_variance",
            y_col="utility_gini",
            group_col="competition_label",
            title=f"{game_title}: binned mean Gini vs Elo variance by competition",
            x_label="Mean within-bin Elo variance",
            y_label="Mean final utility Gini",
            sample_label="runs",
            annotate=True,
        )
        plot_paths["hetero_gini"].append(path)

    path = ASSET_DIR / "hom_ctrl_all_games_scaling_by_n.png"
    frame = ctrl_scaling.copy()
    plot_grouped_line(
        frame,
        path,
        x_col="n_agents",
        y_col="mean_utility",
        group_col="game_label",
        title="Homogeneous control: mean utility as N scales",
        x_label="N",
        y_label="Mean run-level utility",
        sample_label="runs",
        annotate=True,
    )
    plot_paths["hom_ctrl"].append(path)

    for category, paths in plot_paths.items():
        for path in paths:
            if path.exists():
                plot_inventory.append(
                    {
                        "category": category,
                        "file": rel_path(path),
                    }
                )

    return plot_tables, plot_inventory


def summarize_for_report(runs: pd.DataFrame, agents: pd.DataFrame) -> dict[str, pd.DataFrame]:
    overview = (
        runs.groupby(["game_label", "experiment_family"], dropna=False)
        .agg(
            completed_runs=("config_id", "count"),
            mean_run_utility=("mean_utility", "mean"),
            consensus_rate=("consensus_reached", "mean"),
        )
        .reset_index()
        .sort_values(["game_label", "experiment_family"])
    )
    overview["category"] = overview["experiment_family"].map(FAMILY_TITLES).fillna(overview["experiment_family"])
    overview = overview[["game_label", "category", "completed_runs", "mean_run_utility", "consensus_rate"]]

    n_counts = (
        runs.groupby(["game_label", "experiment_family", "n_agents"], dropna=False)
        .size()
        .reset_index(name="completed_runs")
        .sort_values(["game_label", "experiment_family", "n_agents"])
    )

    hom_adv = agents[
        agents["experiment_family"].eq("homogeneous_adversary") & agents["role"].eq("adversary")
    ].copy()
    hom_adv_stats = []
    for game_label, group in hom_adv.groupby("game_label"):
        stats = linear_stats(group, "elo", "final_utility")
        hom_adv_stats.append(
            {
                "game_label": game_label,
                "adversary_agent_obs": int(stats["n"]),
                "utility_per_100_elo_slope": stats["slope"] * 100 if math.isfinite(stats["slope"]) else math.nan,
                "elo_utility_corr": stats["corr"],
            }
        )
    hom_adv_stats = pd.DataFrame(hom_adv_stats)

    hetero = agents[agents["experiment_family"].eq("heterogeneous_random")].copy()
    hetero_stats = []
    for game_label, group in hetero.groupby("game_label"):
        stats = linear_stats(group, "elo", "final_utility")
        hetero_stats.append(
            {
                "game_label": game_label,
                "agent_obs": int(stats["n"]),
                "utility_per_100_elo_slope": stats["slope"] * 100 if math.isfinite(stats["slope"]) else math.nan,
                "elo_utility_corr": stats["corr"],
            }
        )
    hetero_stats = pd.DataFrame(hetero_stats)

    hetero_runs = runs[runs["experiment_family"].eq("heterogeneous_random")].copy()
    gini_stats = []
    for game_label, group in hetero_runs.groupby("game_label"):
        stats = linear_stats(group, "elo_variance", "utility_gini")
        gini_stats.append(
            {
                "game_label": game_label,
                "runs": int(stats["n"]),
                "gini_per_1000_elo_variance_slope": stats["slope"] * 1000 if math.isfinite(stats["slope"]) else math.nan,
                "elo_variance_gini_corr": stats["corr"],
                "negative_utility_runs": int(group["negative_utility_count"].gt(0).sum()),
                "all_zero_runs": int(group["all_zero_utilities"].sum()),
            }
        )
    gini_stats = pd.DataFrame(gini_stats)

    control_agents = agents[agents["experiment_family"].eq("homogeneous_control")].copy()
    control_position_stats = []
    for (game_label, n_agents), group in control_agents.groupby(["game_label", "n_agents"]):
        stats = linear_stats(group, "agent_index", "final_utility")
        control_position_stats.append(
            {
                "game_label": game_label,
                "n_agents": n_agents,
                "agent_obs": int(stats["n"]),
                "utility_per_position_slope": stats["slope"],
                "position_utility_corr": stats["corr"],
            }
        )
    control_position_stats = pd.DataFrame(control_position_stats).sort_values(["game_label", "n_agents"])

    return {
        "overview": overview,
        "n_counts": n_counts,
        "hom_adv_stats": hom_adv_stats,
        "hetero_stats": hetero_stats,
        "gini_stats": gini_stats,
        "control_position_stats": control_position_stats,
    }


def append_plot_block(lines: list[str], title: str, paths: list[str]) -> None:
    lines.append(f"### {title}")
    for path in paths:
        lines.append(f"![{Path(path).stem}]({path})")
        lines.append("")


def build_markdown(
    runs: pd.DataFrame,
    agents: pd.DataFrame,
    plot_tables: dict[str, pd.DataFrame],
    plot_inventory: list[dict[str, str]],
    report_tables: dict[str, pd.DataFrame],
    csv_paths: dict[str, Path],
) -> str:
    completed_runs = len(runs)
    completed_pct = completed_runs / 2730 * 100
    homogeneous_runs = int(runs["source"].eq("homogeneous").sum())
    heterogeneous_runs = int(runs["source"].eq("heterogeneous").sum())
    agent_obs = len(agents)
    hetero_runs = runs[runs["experiment_family"].eq("heterogeneous_random")]
    shifted_gini_runs = int(hetero_runs["utility_gini_shifted"].sum())
    all_zero_runs = int(hetero_runs["all_zero_utilities"].sum())

    lines: list[str] = []
    lines.append("# Partial Multi-Agent Results Plot Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        f"This report analyzes the completed subset only: {completed_runs:,} / 2,730 runs "
        f"({completed_pct:.1f}%). That is {homogeneous_runs:,} homogeneous runs and "
        f"{heterogeneous_runs:,} heterogeneous runs, yielding {agent_obs:,} agent-level utility observations."
    )
    lines.append("")
    lines.append(f"- Homogeneous root: `{HOMOGENEOUS_ROOT.relative_to(PROJECT_ROOT)}`")
    lines.append(f"- Heterogeneous root: `{HETEROGENEOUS_ROOT.relative_to(PROJECT_ROOT)}`")
    lines.append(f"- Elo source: `{ELO_DOC.relative_to(PROJECT_ROOT)}`")
    lines.append("- Payoff/utility field: top-level `final_utilities[Agent_i]` in each `experiment_results.json`.")
    lines.append(
        "- Gini convention: standard Gini on final utilities after shifting a run by "
        "subtracting its minimum only when at least one utility is negative; all-zero utility vectors use Gini=0. "
        f"This affected {shifted_gini_runs} completed heterogeneous runs; "
        f"{all_zero_runs} completed heterogeneous runs were all-zero."
    )
    lines.append(
        "- Sample sizes are shown as `n=` labels on aggregated points where possible; for dense heterogeneous Elo plots, "
        "marker size also scales with the completed agent observations behind that point. Exact point counts are in the appendix CSVs and tables."
    )
    lines.append("")
    lines.append("The missing and failed runs are not imputed. The completed subset is therefore a partial-results view, not a balanced full-factorial estimate.")
    lines.append("")

    lines.append("## Completion Overview")
    lines.append("")
    lines.append(df_to_markdown(report_tables["overview"]))
    lines.append("")
    lines.append("### Completed Runs By N")
    lines.append("")
    lines.append(df_to_markdown(report_tables["n_counts"]))
    lines.append("")

    lines.append("## High-Level Slopes")
    lines.append("")
    lines.append("These are descriptive linear slopes on the completed observations, not causal estimates.")
    lines.append("")
    lines.append("### Homogeneous Adversary: Utility vs Elo")
    lines.append("")
    lines.append(df_to_markdown(report_tables["hom_adv_stats"]))
    lines.append("")
    lines.append("### Heterogeneous: Utility vs Elo")
    lines.append("")
    lines.append(df_to_markdown(report_tables["hetero_stats"]))
    lines.append("")
    lines.append("### Heterogeneous: Gini vs Elo Variance")
    lines.append("")
    lines.append(df_to_markdown(report_tables["gini_stats"]))
    lines.append("")

    coverage_rows: list[dict[str, Any]] = []
    expected_n_values = [2, 4, 6, 8, 10]
    hom_adv_by_n = plot_tables.get("hom_adv_by_n", pd.DataFrame())
    if not hom_adv_by_n.empty:
        adv_models = (
            hom_adv_by_n[["model_short", "elo"]]
            .drop_duplicates()
            .sort_values("elo")
            .to_dict("records")
        )
        for game_label in ["game1", "game2", "game3"]:
            game_frame = hom_adv_by_n[hom_adv_by_n["game_label"].eq(game_label)]
            for n_agents in expected_n_values:
                for model_row in adv_models:
                    mask = (
                        game_frame["n_agents"].eq(n_agents)
                        & game_frame["model_short"].eq(model_row["model_short"])
                    )
                    if not mask.any():
                        coverage_rows.append(
                            {
                                "section": "homogeneous adversary by N",
                                "game_label": game_label,
                                "n_agents": n_agents,
                                "missing": f"{model_row['model_short']} (Elo {int(model_row['elo'])})",
                            }
                        )
    hetero_runs_for_coverage = runs[runs["experiment_family"].eq("heterogeneous_random")]
    for game_label in ["game1", "game2", "game3"]:
        game_frame = hetero_runs_for_coverage[hetero_runs_for_coverage["game_label"].eq(game_label)]
        for n_agents in expected_n_values:
            if not game_frame["n_agents"].eq(n_agents).any():
                coverage_rows.append(
                    {
                        "section": "heterogeneous by N",
                        "game_label": game_label,
                        "n_agents": n_agents,
                        "missing": "no completed heterogeneous runs",
                    }
                )
    if coverage_rows:
        lines.append("### No-Completed-Data Plot Gaps")
        lines.append("")
        lines.append(
            "These are aggregate points or curves that cannot appear in the plots because the completed subset has zero observations for that cell."
        )
        lines.append("")
        lines.append(df_to_markdown(pd.DataFrame(coverage_rows)))
        lines.append("")

    plot_lookup = defaultdict(list)
    for item in plot_inventory:
        plot_lookup[item["category"]].append(item["file"])

    lines.append("## Homogeneous Adversary")
    lines.append("")
    lines.append(
        "Each point is the completed-run mean utility of the adversary model at that Elo. "
        "The broad N plots average over competition level, model order, and seeds; competition plots average over N, model order, and seeds; "
        "model-order plots average over N, competition level, and seeds."
    )
    lines.append("")
    for game_label in ["game1", "game2", "game3"]:
        paths = [
            rel_path(ASSET_DIR / f"hom_adv_{game_label}_utility_vs_elo_by_n.png"),
            rel_path(ASSET_DIR / f"hom_adv_{game_label}_utility_vs_elo_by_competition.png"),
            rel_path(ASSET_DIR / f"hom_adv_{game_label}_utility_vs_elo_by_position.png"),
            rel_path(ASSET_DIR / f"hom_adv_{game_label}_first_minus_last_gap_by_n.png"),
        ]
        append_plot_block(lines, GAME_TITLES[game_label], paths)

    lines.append("## Homogeneous Control")
    lines.append("")
    lines.append(
        "All agents use the same baseline model, so these plots treat `Agent_i` as the model position. "
        "The position plots expose order effects; the scaling plots show the run-level mean utility as N increases."
    )
    lines.append("")
    lines.append("### Position Slope Summary")
    lines.append("")
    lines.append(df_to_markdown(report_tables["control_position_stats"]))
    lines.append("")
    append_plot_block(lines, "All games scaling", [rel_path(ASSET_DIR / "hom_ctrl_all_games_scaling_by_n.png")])
    for game_label in ["game1", "game2", "game3"]:
        paths = [
            rel_path(ASSET_DIR / f"hom_ctrl_{game_label}_utility_by_position_and_n.png"),
            rel_path(ASSET_DIR / f"hom_ctrl_{game_label}_utility_by_position_and_competition.png"),
            rel_path(ASSET_DIR / f"hom_ctrl_{game_label}_scaling_by_competition.png"),
        ]
        append_plot_block(lines, GAME_TITLES[game_label], paths)

    lines.append("## Heterogeneous Performance vs Elo")
    lines.append("")
    lines.append(
        "These plots use every completed heterogeneous agent observation. The N and competition curves aggregate by model/Elo; "
        "the random-order position plot is included as a check on model order effects."
    )
    lines.append("")
    for game_label in ["game1", "game2", "game3"]:
        paths = [
            rel_path(ASSET_DIR / f"hetero_{game_label}_utility_vs_elo_by_n.png"),
            rel_path(ASSET_DIR / f"hetero_{game_label}_utility_vs_elo_by_competition.png"),
            rel_path(ASSET_DIR / f"hetero_{game_label}_utility_by_position_and_n.png"),
        ]
        append_plot_block(lines, GAME_TITLES[game_label], paths)

    lines.append("## Heterogeneous Ecosystem Inequality")
    lines.append("")
    lines.append(
        "Each scatter point is one completed experiment. The binned plots average neighboring Elo-variance runs within each N or competition curve, "
        "with `n=` showing completed runs in that bin."
    )
    lines.append("")
    for game_label in ["game1", "game2", "game3"]:
        paths = [
            rel_path(ASSET_DIR / f"hetero_{game_label}_gini_vs_elo_variance_scatter_by_n.png"),
            rel_path(ASSET_DIR / f"hetero_{game_label}_gini_vs_elo_variance_binned_by_n.png"),
            rel_path(ASSET_DIR / f"hetero_{game_label}_gini_vs_elo_variance_scatter_by_competition.png"),
            rel_path(ASSET_DIR / f"hetero_{game_label}_gini_vs_elo_variance_binned_by_competition.png"),
        ]
        append_plot_block(lines, GAME_TITLES[game_label], paths)

    lines.append("## Appendix: Data Files")
    lines.append("")
    file_rows = pd.DataFrame(
        [
            {"dataset": key, "file": rel_path(path)}
            for key, path in sorted(csv_paths.items())
        ]
    )
    lines.append(df_to_markdown(file_rows))
    lines.append("")
    lines.append("## Appendix: Plot Inventory")
    lines.append("")
    lines.append(df_to_markdown(pd.DataFrame(plot_inventory)))
    lines.append("")

    lines.append("## Appendix: Aggregated Point Data")
    lines.append("")
    lines.append("The tables below are the exact aggregated points used in the plots. `sample_count` is the plotted `n`.")
    lines.append("")
    point_columns = {
        "hom_adv_by_n": ["game_label", "n_agents", "model_short", "elo", "mean_utility", "sample_count"],
        "hom_adv_by_competition": ["game_label", "competition_label", "model_short", "elo", "mean_utility", "sample_count"],
        "hom_adv_by_position": ["game_label", "adversary_position", "model_short", "elo", "mean_utility", "sample_count"],
        "hom_adv_position_gap_by_n": [
            "game_label",
            "n_agents",
            "model_short",
            "elo",
            "position_utility_gap_first_minus_last",
            "sample_count",
            "first_sample_count",
            "last_sample_count",
        ],
        "hom_ctrl_by_position_n": ["game_label", "n_agents", "agent_index", "mean_utility", "sample_count"],
        "hom_ctrl_by_position_competition": ["game_label", "competition_label", "agent_index", "mean_utility", "sample_count"],
        "hom_ctrl_scaling_competition": ["game_label", "competition_label", "n_agents", "mean_utility", "sample_count"],
        "hetero_perf_by_n": ["game_label", "n_agents", "model_short", "elo", "mean_utility", "sample_count"],
        "hetero_perf_by_competition": ["game_label", "competition_label", "model_short", "elo", "mean_utility", "sample_count"],
        "hetero_position": ["game_label", "n_agents", "agent_index", "mean_utility", "sample_count"],
        "hetero_gini_by_n_binned": ["game_label", "n_agents", "elo_bin", "elo_variance", "utility_gini", "sample_count"],
        "hetero_gini_by_competition_binned": ["game_label", "competition_label", "elo_bin", "elo_variance", "utility_gini", "sample_count"],
    }
    for key, table in plot_tables.items():
        lines.append(f"<details><summary>{key} ({len(table):,} rows)</summary>")
        lines.append("")
        lines.append(df_to_markdown(sort_summary(table), columns=point_columns.get(key)))
        lines.append("")
        lines.append("</details>")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    model_elos = parse_model_elos(ELO_DOC)
    runs, agents = build_tables(model_elos)
    plot_tables, plot_inventory = build_plot_data(runs, agents)
    report_tables = summarize_for_report(runs, agents)

    csv_paths: dict[str, Path] = {
        "completed_runs": write_csv(runs, "completed_runs.csv"),
        "agent_observations": write_csv(agents, "agent_observations.csv"),
    }
    for key, table in plot_tables.items():
        csv_paths[key] = write_csv(sort_summary(table), f"{key}.csv")
    for key, table in report_tables.items():
        csv_paths[f"report_{key}"] = write_csv(table, f"report_{key}.csv")

    markdown = build_markdown(runs, agents, plot_tables, plot_inventory, report_tables, csv_paths)
    REPORT_PATH.write_text(markdown, encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote assets to {ASSET_DIR}")
    print(f"Completed runs: {len(runs)}; agent observations: {len(agents)}")
    print(f"Plots: {sum(1 for item in plot_inventory if (REPORT_PATH.parent / item['file']).exists())}")


if __name__ == "__main__":
    main()

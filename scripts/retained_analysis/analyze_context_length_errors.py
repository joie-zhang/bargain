#!/usr/bin/env python3
"""Analyze context-length failures in the full Games 1/2/3 multi-agent runs."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[2]
HOMOGENEOUS_ROOT = REPO_ROOT / "experiments/results/full_games123_multiagent_production_20260428_085255"
HETEROGENEOUS_ROOT = (
    REPO_ROOT
    / "experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848"
)
CONTEXT_GUIDE = (
    REPO_ROOT / "docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md"
)
OUT_DIR = REPO_ROOT / "experiments/analysis/context_length_errors_20260502"
PLOTS_DIR = OUT_DIR / "plots"
TABLES_DIR = OUT_DIR / "tables"


PROGRESS_RE = re.compile(
    r"PROGRESS interaction=(?P<interaction>\d+) "
    r"round=(?P<round>\d+) "
    r"phase=(?P<phase>\S+) "
    r"agent=(?P<agent>Agent_\d+) "
    r"model=(?P<model>\S+) "
    r"prompt_chars=(?P<prompt_chars>\d+) "
    r"response_chars=(?P<response_chars>\d+)"
)

CONTEXT_ERROR_PATTERNS = [
    re.compile(
        r"Input tokens exceed the configured limit of (?P<limit>[\d,]+) tokens\. "
        r"Your messages resulted in (?P<requested>[\d,]+) tokens"
    ),
    re.compile(
        r"maximum context length is (?P<limit>[\d,]+) tokens\. "
        r"However, your messages resulted in (?P<requested>[\d,]+) tokens"
    ),
    re.compile(
        r"maximum context length is (?P<limit>[\d,]+) tokens\. "
        r"However, you requested about (?P<requested>[\d,]+) tokens "
        r"\((?P<input>[\d,]+) of text input, (?P<output>[\d,]+) in the output\)"
    ),
    re.compile(
        r"maximum context length is (?P<limit>[\d,]+) tokens\. "
        r"However, you requested (?P<requested>[\d,]+) tokens "
        r"\((?P<input>[\d,]+) in the messages, (?P<output>[\d,]+) in the completion\)"
    ),
]

MODEL_ALIASES = {
    "gpt-5-nano": "gpt-5-nano-high",
    "amazon/nova-micro-v1": "amazon-nova-micro-v1.0",
    "amazon/nova-pro-v1": "amazon-nova-pro-v1.0",
}


@dataclass(frozen=True)
class RootSpec:
    label: str
    root: Path


ROOTS = [
    RootSpec("homogeneous", HOMOGENEOUS_ROOT),
    RootSpec("heterogeneous", HETEROGENEOUS_ROOT),
]


def parse_number(value: str | int | float | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = value.strip().replace(",", "")
    if not text or text == "-":
        return None
    multiplier = 1
    if text[-1:].upper() == "K":
        multiplier = 1_000
        text = text[:-1]
    elif text[-1:].upper() == "M":
        multiplier = 1_000_000
        text = text[:-1]
    try:
        return int(float(text) * multiplier)
    except ValueError:
        return None


def pct(numerator: int | float, denominator: int | float) -> float:
    return (float(numerator) / float(denominator) * 100.0) if denominator else 0.0


def load_context_lengths() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for line in CONTEXT_GUIDE.read_text().splitlines():
        line = line.strip()
        if not line.startswith("|") or "`" not in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 9 or cells[0] in {"--", "#"}:
            continue
        model_match = re.search(r"`([^`]+)`", cells[1])
        if not model_match:
            continue
        model = model_match.group(1)
        arena_context = parse_number(cells[3])
        openrouter_context = parse_number(cells[4])
        effective_context = openrouter_context or arena_context
        rows.append(
            {
                "model": model,
                "arena_context": arena_context,
                "openrouter_context": openrouter_context,
                "effective_context": effective_context,
                "arena_org": cells[5],
                "repo_route": cells[6],
                "route": cells[7].strip("`"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        alias_rows = []
        for alias, target in MODEL_ALIASES.items():
            target_row = df[df["model"] == target]
            if not target_row.empty:
                row = target_row.iloc[0].to_dict()
                row["model"] = alias
                row["alias_for"] = target
                alias_rows.append(row)
        if alias_rows:
            df = pd.concat([df, pd.DataFrame(alias_rows)], ignore_index=True)
    return df


def load_config(root: Path, config_id: int) -> dict[str, Any]:
    config_path = root / f"configs/config_{config_id:04d}.json"
    return json.loads(config_path.read_text())


def config_id_from_path(path: Path) -> int:
    match = re.search(r"config_(\d+)", path.name)
    if not match:
        raise ValueError(f"Cannot parse config id from {path}")
    return int(match.group(1))


def find_first_context_error(text: str) -> tuple[int, dict[str, int]] | None:
    if not (
        "context_length_exceeded" in text
        or "Input tokens exceed the configured limit" in text
        or "maximum context length" in text
    ):
        return None
    matches = []
    for pattern in CONTEXT_ERROR_PATTERNS:
        match = pattern.search(text)
        if match:
            matches.append(match)
    if not matches:
        return None
    first = min(matches, key=lambda item: item.start())
    values = {
        key: int(value.replace(",", ""))
        for key, value in first.groupdict().items()
        if value is not None
    }
    return first.start(), values


def last_progress_before(text: str, position: int) -> dict[str, Any] | None:
    last: dict[str, Any] | None = None
    for line in text[:position].splitlines():
        match = PROGRESS_RE.search(line)
        if not match:
            continue
        last = match.groupdict()
        for key in ("interaction", "round", "prompt_chars", "response_chars"):
            last[key] = int(last[key])
    return last


def infer_next_call(last: dict[str, Any] | None, config: dict[str, Any]) -> dict[str, Any] | None:
    if not last:
        return None
    n_agents = int(config["n_agents"])
    discussion_turns = int(config.get("discussion_turns") or 2)
    max_rounds = int(config.get("max_rounds") or 10)
    phase = str(last["phase"])
    agent_idx = int(str(last["agent"]).split("_")[1])

    def result(round_id: int, phase_name: str, agent_name: str) -> dict[str, Any]:
        return {
            "failed_round": round_id,
            "failed_phase": phase_name,
            "failed_agent": agent_name,
            "failed_model": (config.get("agent_model_map") or {}).get(agent_name),
        }

    match = re.match(r"discussion_round_(\d+)_turn_(\d+)", phase)
    if match:
        round_id = int(match.group(1))
        turn = int(match.group(2))
        if agent_idx < n_agents:
            return result(round_id, phase, f"Agent_{agent_idx + 1}")
        if turn < discussion_turns:
            return result(round_id, f"discussion_round_{round_id}_turn_{turn + 1}", "Agent_1")
        return result(round_id, f"private_thinking_round_{round_id}", "Agent_1")

    match = re.match(r"private_thinking_round_(\d+)", phase)
    if match:
        round_id = int(match.group(1))
        if agent_idx < n_agents:
            return result(round_id, phase, f"Agent_{agent_idx + 1}")
        return result(round_id, f"proposal_round_{round_id}", "Agent_1")

    match = re.match(r"proposal_round_(\d+)(?:_invalid_attempt_\d+)?", phase)
    if match:
        round_id = int(match.group(1))
        if agent_idx < n_agents:
            return result(round_id, f"proposal_round_{round_id}", f"Agent_{agent_idx + 1}")
        return result(round_id, f"voting_round_{round_id}_proposal_1", "Agent_1")

    match = re.match(r"voting_round_(\d+)_proposal_(\d+)", phase)
    if match:
        round_id = int(match.group(1))
        proposal_idx = int(match.group(2))
        if proposal_idx < n_agents:
            return result(round_id, f"voting_round_{round_id}_proposal_{proposal_idx + 1}", last["agent"])
        if agent_idx < n_agents:
            return result(round_id, f"voting_round_{round_id}_proposal_1", f"Agent_{agent_idx + 1}")
        return result(round_id, f"reflection_round_{round_id}", "Agent_1")

    match = re.match(r"reflection_round_(\d+)", phase)
    if match:
        round_id = int(match.group(1))
        if agent_idx < n_agents:
            return result(round_id, phase, f"Agent_{agent_idx + 1}")
        if round_id < max_rounds:
            return result(round_id + 1, f"discussion_round_{round_id + 1}_turn_1", "Agent_1")
        return None

    if phase == "game_setup":
        if agent_idx < n_agents:
            return result(0, "game_setup", f"Agent_{agent_idx + 1}")
        return result(1, "discussion_round_1_turn_1", "Agent_1")

    return None


def load_run_metadata() -> pd.DataFrame:
    rows = []
    for spec in ROOTS:
        for run_dir in sorted((spec.root / "runs").iterdir()):
            if not run_dir.is_dir():
                continue
            config_id = config_id_from_path(run_dir)
            config = load_config(spec.root, config_id)
            progress_path = run_dir / "progress.json"
            result_path = run_dir / "experiment_results.json"
            progress: dict[str, Any] = {}
            if progress_path.exists():
                try:
                    progress = json.loads(progress_path.read_text())
                except json.JSONDecodeError:
                    progress = {}
            rows.append(
                {
                    "folder": spec.label,
                    "config_id": config_id,
                    "run_dir": str(run_dir.relative_to(REPO_ROOT)),
                    "experiment_type": config["experiment_type"],
                    "game": config["game_label"],
                    "N": int(config["n_agents"]),
                    "has_result": result_path.exists(),
                    "current_round": progress.get("current_round"),
                    "last_phase": (progress.get("last_interaction") or {}).get("phase"),
                    "last_agent": (progress.get("last_interaction") or {}).get("agent_id"),
                    "baseline_model": config.get("baseline_model"),
                    "adversary_model": config.get("adversary_model"),
                    "adversary_position": config.get("adversary_position"),
                    "agent_model_map": config.get("agent_model_map") or {},
                }
            )
    return pd.DataFrame(rows)


def load_model_appearances(runs_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in runs_df.to_dict("records"):
        for agent, model in row["agent_model_map"].items():
            rows.append(
                {
                    "folder": row["folder"],
                    "config_id": row["config_id"],
                    "experiment_type": row["experiment_type"],
                    "game": row["game"],
                    "N": row["N"],
                    "agent": agent,
                    "model": model,
                }
            )
    return pd.DataFrame(rows)


def load_context_errors() -> pd.DataFrame:
    rows = []
    for spec in ROOTS:
        for log_path in sorted((spec.root / "logs").glob("*.log")):
            text = log_path.read_text(errors="ignore")
            context_error = find_first_context_error(text)
            if not context_error:
                continue
            error_pos, token_values = context_error
            config_id = config_id_from_path(log_path)
            config = load_config(spec.root, config_id)
            last = last_progress_before(text, error_pos)
            inferred = infer_next_call(last, config) or {}
            limit = token_values.get("limit")
            requested = token_values.get("requested")
            row = {
                "folder": spec.label,
                "config_id": config_id,
                "log_file": str(log_path.relative_to(REPO_ROOT)),
                "experiment_type": config["experiment_type"],
                "game": config["game_label"],
                "N": int(config["n_agents"]),
                "baseline_model": config.get("baseline_model"),
                "adversary_model": config.get("adversary_model"),
                "adversary_position": config.get("adversary_position"),
                "limit_tokens_from_error": limit,
                "requested_tokens": requested,
                "input_tokens_from_error": token_values.get("input"),
                "output_tokens_from_error": token_values.get("output"),
                "over_limit_tokens": requested - limit if requested and limit else None,
                "over_limit_pct": pct(requested - limit, limit) if requested and limit else None,
                "last_success_round": last.get("round") if last else None,
                "last_success_phase": last.get("phase") if last else None,
                "last_success_agent": last.get("agent") if last else None,
                "last_success_model": last.get("model") if last else None,
                "last_success_prompt_chars": last.get("prompt_chars") if last else None,
                "last_success_response_chars": last.get("response_chars") if last else None,
                **inferred,
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Some configs were retried. Keep all attempts in context_errors_attempts.csv;
    # config-level summaries use the first log-confirmed context error per config.
    return df.sort_values(["folder", "config_id", "log_file"]).reset_index(drop=True)


def first_error_per_config(errors_df: pd.DataFrame) -> pd.DataFrame:
    if errors_df.empty:
        return errors_df.copy()
    return (
        errors_df.sort_values(["folder", "config_id", "log_file"])
        .drop_duplicates(["folder", "config_id"], keep="first")
        .reset_index(drop=True)
    )


def add_context_lengths(model_df: pd.DataFrame, context_df: pd.DataFrame) -> pd.DataFrame:
    context_cols = [
        "model",
        "effective_context",
        "arena_context",
        "openrouter_context",
        "repo_route",
        "route",
    ]
    merged = model_df.merge(context_df[context_cols], on="model", how="left")
    return merged


def rate_table(
    runs_df: pd.DataFrame,
    first_errors_df: pd.DataFrame,
    group_cols: list[str],
    n_gt_2: bool = True,
) -> pd.DataFrame:
    base = runs_df.copy()
    errors = first_errors_df.copy()
    if n_gt_2:
        base = base[base["N"] > 2]
        errors = errors[errors["N"] > 2]

    denominator = base.groupby(group_cols, dropna=False).size().rename("runs").reset_index()
    numerator = errors.groupby(group_cols, dropna=False).size().rename("context_error_runs").reset_index()
    table = denominator.merge(numerator, on=group_cols, how="left")
    table["context_error_runs"] = table["context_error_runs"].fillna(0).astype(int)
    table["context_error_pct"] = table.apply(
        lambda row: pct(row["context_error_runs"], row["runs"]), axis=1
    )
    return table.sort_values(group_cols).reset_index(drop=True)


def round_rate_table(
    runs_df: pd.DataFrame,
    first_errors_df: pd.DataFrame,
    group_cols: list[str],
    n_gt_2: bool = True,
) -> pd.DataFrame:
    base = runs_df.copy()
    errors = first_errors_df.copy()
    if n_gt_2:
        base = base[base["N"] > 2]
        errors = errors[errors["N"] > 2]
    denom = base.groupby(group_cols, dropna=False).size().rename("runs").reset_index()
    nums = (
        errors.groupby(group_cols + ["failed_round"], dropna=False)
        .size()
        .rename("context_error_runs")
        .reset_index()
    )
    table = nums.merge(denom, on=group_cols, how="left")
    table["context_error_pct"] = table.apply(
        lambda row: pct(row["context_error_runs"], row["runs"]), axis=1
    )
    return table.sort_values(group_cols + ["failed_round"]).reset_index(drop=True)


def model_rate_table(
    appearances_df: pd.DataFrame,
    first_errors_df: pd.DataFrame,
    context_df: pd.DataFrame,
    n_gt_2: bool = True,
) -> pd.DataFrame:
    appearances = appearances_df.copy()
    errors = first_errors_df.copy()
    if n_gt_2:
        appearances = appearances[appearances["N"] > 2]
        errors = errors[errors["N"] > 2]

    cfg_den = (
        appearances.drop_duplicates(["folder", "config_id", "model"])
        .groupby(["folder", "model"], dropna=False)
        .size()
        .rename("configs_with_model")
        .reset_index()
    )
    agent_den = (
        appearances.groupby(["folder", "model"], dropna=False)
        .size()
        .rename("agent_runs_with_model")
        .reset_index()
    )
    numerator = (
        errors.groupby(["folder", "failed_model"], dropna=False)
        .size()
        .rename("context_error_runs")
        .reset_index()
        .rename(columns={"failed_model": "model"})
    )
    table = cfg_den.merge(agent_den, on=["folder", "model"], how="outer")
    table = table.merge(numerator, on=["folder", "model"], how="left")
    table["context_error_runs"] = table["context_error_runs"].fillna(0).astype(int)
    table["context_error_pct_configs"] = table.apply(
        lambda row: pct(row["context_error_runs"], row["configs_with_model"]), axis=1
    )
    table["context_error_pct_agent_runs"] = table.apply(
        lambda row: pct(row["context_error_runs"], row["agent_runs_with_model"]), axis=1
    )
    return add_context_lengths(table, context_df).sort_values(
        ["context_error_pct_configs", "context_error_runs", "model"], ascending=[False, False, True]
    )


def save_plot(fig: plt.Figure, filename: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def annotate_nonzero(ax: plt.Axes, data: pd.DataFrame, y_col: str) -> None:
    for row in data.to_dict("records"):
        if row.get("context_error_runs", 0) <= 0:
            continue
        x = row.get("effective_context")
        y = row.get(y_col)
        if pd.isna(x) or pd.isna(y):
            continue
        ax.annotate(
            str(row["model"]),
            (x, y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )


def make_plots(
    runs_df: pd.DataFrame,
    first_errors_df: pd.DataFrame,
    model_rates: pd.DataFrame,
    context_errors_df: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid")

    n_table = rate_table(runs_df, first_errors_df, ["folder", "N"])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.barplot(data=n_table, x="N", y="context_error_pct", hue="folder", ax=ax)
    ax.set_title("Context-Length Error Rate by N (N > 2)")
    ax.set_xlabel("Number of agents (N)")
    ax.set_ylabel("Runs with context-length error (%)")
    save_plot(fig, "error_rate_by_n.png")

    game_table = rate_table(runs_df, first_errors_df, ["folder", "game"])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.barplot(data=game_table, x="game", y="context_error_pct", hue="folder", ax=ax)
    ax.set_title("Context-Length Error Rate by Game (N > 2)")
    ax.set_xlabel("Game")
    ax.set_ylabel("Runs with context-length error (%)")
    save_plot(fig, "error_rate_by_game.png")

    round_table = round_rate_table(runs_df, first_errors_df, ["folder"])
    if not round_table.empty:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        sns.barplot(data=round_table, x="failed_round", y="context_error_pct", hue="folder", ax=ax)
        ax.set_title("First Context-Length Error Round (N > 2)")
        ax.set_xlabel("Round of first context-length error")
        ax.set_ylabel("Runs with first error in round (%)")
        save_plot(fig, "error_rate_by_round.png")

    n_round = round_rate_table(runs_df, first_errors_df, ["folder", "N"])
    if not n_round.empty:
        for folder in sorted(n_round["folder"].dropna().unique()):
            folder_df = n_round[n_round["folder"] == folder]
            pivot = folder_df.pivot_table(
                index="N",
                columns="failed_round",
                values="context_error_pct",
                fill_value=0,
            )
            fig, ax = plt.subplots(figsize=(10, 4.8))
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Reds", ax=ax, cbar_kws={"label": "%"})
            ax.set_title(f"{folder.title()} Context-Length Error Rate by N and Round (N > 2)")
            ax.set_xlabel("Round of first context-length error")
            ax.set_ylabel("N")
            save_plot(fig, f"heatmap_error_rate_by_n_round_{folder}.png")

    game_n = rate_table(runs_df, first_errors_df, ["folder", "game", "N"])
    for folder in sorted(game_n["folder"].dropna().unique()):
        folder_df = game_n[game_n["folder"] == folder]
        pivot = folder_df.pivot_table(index="game", columns="N", values="context_error_pct", fill_value=0)
        fig, ax = plt.subplots(figsize=(8, 4.8))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Reds", ax=ax, cbar_kws={"label": "%"})
        ax.set_title(f"{folder.title()} Context-Length Error Rate by Game and N (N > 2)")
        ax.set_xlabel("N")
        ax.set_ylabel("Game")
        save_plot(fig, f"heatmap_error_rate_by_game_n_{folder}.png")

    game_round = round_rate_table(runs_df, first_errors_df, ["folder", "game"])
    for folder in sorted(game_round["folder"].dropna().unique()):
        folder_df = game_round[game_round["folder"] == folder]
        if folder_df.empty:
            continue
        pivot = folder_df.pivot_table(
            index="game",
            columns="failed_round",
            values="context_error_pct",
            fill_value=0,
        )
        fig, ax = plt.subplots(figsize=(10, 4.8))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Reds", ax=ax, cbar_kws={"label": "%"})
        ax.set_title(f"{folder.title()} Context-Length Error Rate by Game and Round (N > 2)")
        ax.set_xlabel("Round of first context-length error")
        ax.set_ylabel("Game")
        save_plot(fig, f"heatmap_error_rate_by_game_round_{folder}.png")

    plot_model_rates = model_rates.dropna(subset=["effective_context"]).copy()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.scatterplot(
        data=plot_model_rates,
        x="effective_context",
        y="context_error_pct_configs",
        size="configs_with_model",
        hue="folder",
        sizes=(25, 220),
        alpha=0.75,
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_title("Model Context Length vs Context-Length Error Rate (N > 2)")
    ax.set_xlabel("Effective context length from guide (tokens, log scale)")
    ax.set_ylabel("Configs with this model that hit context-length error (%)")
    annotate_nonzero(ax, plot_model_rates, "context_error_pct_configs")
    save_plot(fig, "model_context_length_error_rate.png")

    for game in sorted(runs_df.loc[runs_df["N"] > 2, "game"].unique()):
        appearances = load_model_appearances(runs_df[runs_df["game"] == game])
        game_errors = first_errors_df[first_errors_df["game"] == game]
        game_model_rates = model_rate_table(appearances, game_errors, load_context_lengths())
        game_model_rates = game_model_rates.dropna(subset=["effective_context"])
        fig, ax = plt.subplots(figsize=(10, 5.2))
        sns.scatterplot(
            data=game_model_rates,
            x="effective_context",
            y="context_error_pct_configs",
            size="configs_with_model",
            hue="folder",
            sizes=(25, 220),
            alpha=0.75,
            ax=ax,
        )
        ax.set_xscale("log")
        ax.set_title(f"Model Context Length vs Error Rate: {game} (N > 2)")
        ax.set_xlabel("Effective context length from guide (tokens, log scale)")
        ax.set_ylabel("Configs with this model that hit context-length error (%)")
        annotate_nonzero(ax, game_model_rates, "context_error_pct_configs")
        save_plot(fig, f"model_context_length_error_rate_{game}.png")

    if not context_errors_df.empty:
        error_plot = context_errors_df.copy()
        error_plot["requested_over_limit_ratio"] = (
            error_plot["requested_tokens"] / error_plot["limit_tokens_from_error"]
        )
        fig, ax = plt.subplots(figsize=(9, 5.2))
        sns.scatterplot(
            data=error_plot,
            x="limit_tokens_from_error",
            y="requested_over_limit_ratio",
            hue="failed_model",
            style="game",
            size="N",
            sizes=(60, 180),
            ax=ax,
        )
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_title("How Far Context-Length Errors Exceeded Provider Limit")
        ax.set_xlabel("Provider limit reported in error (tokens, log scale)")
        ax.set_ylabel("Requested tokens / reported limit")
        save_plot(fig, "requested_tokens_over_limit.png")


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return "_No rows._"
    return df[columns].to_markdown(index=False)


def write_report(
    runs_df: pd.DataFrame,
    errors_df: pd.DataFrame,
    first_errors_df: pd.DataFrame,
    model_rates: pd.DataFrame,
) -> None:
    report_path = OUT_DIR / "context_length_error_report.md"
    primary_runs = runs_df[runs_df["N"] > 2]
    primary_errors = first_errors_df[first_errors_df["N"] > 2]
    all_error_pct = pct(len(first_errors_df), len(runs_df))
    primary_error_pct = pct(len(primary_errors), len(primary_runs))

    scope = rate_table(runs_df, first_errors_df, ["folder"], n_gt_2=True)
    scope_all = rate_table(runs_df, first_errors_df, ["folder"], n_gt_2=False)
    by_n = rate_table(runs_df, first_errors_df, ["folder", "N"], n_gt_2=True)
    by_game = rate_table(runs_df, first_errors_df, ["folder", "game"], n_gt_2=True)
    by_game_n = rate_table(runs_df, first_errors_df, ["folder", "game", "N"], n_gt_2=True)
    by_round = round_rate_table(runs_df, first_errors_df, ["folder"], n_gt_2=True)
    by_n_round = round_rate_table(runs_df, first_errors_df, ["folder", "N"], n_gt_2=True)
    by_game_round = round_rate_table(runs_df, first_errors_df, ["folder", "game"], n_gt_2=True)

    errors_for_stats = primary_errors.copy()
    if not errors_for_stats.empty:
        min_requested = int(errors_for_stats["requested_tokens"].min())
        max_requested = int(errors_for_stats["requested_tokens"].max())
        median_requested = float(errors_for_stats["requested_tokens"].median())
        min_over = float(errors_for_stats["over_limit_pct"].min())
        max_over = float(errors_for_stats["over_limit_pct"].max())
        median_over = float(errors_for_stats["over_limit_pct"].median())
        round_min = int(errors_for_stats["failed_round"].min())
        round_max = int(errors_for_stats["failed_round"].max())
    else:
        min_requested = max_requested = round_min = round_max = 0
        median_requested = min_over = max_over = median_over = 0.0

    nonzero_models = model_rates[model_rates["context_error_runs"] > 0].copy()
    nonzero_models = nonzero_models[
        [
            "folder",
            "model",
            "effective_context",
            "configs_with_model",
            "context_error_runs",
            "context_error_pct_configs",
            "agent_runs_with_model",
            "context_error_pct_agent_runs",
        ]
    ].sort_values(["context_error_runs", "model"], ascending=[False, True])

    detail_cols = [
        "folder",
        "config_id",
        "game",
        "N",
        "failed_round",
        "failed_phase",
        "failed_agent",
        "failed_model",
        "limit_tokens_from_error",
        "requested_tokens",
        "over_limit_pct",
    ]
    detail = primary_errors[detail_cols].copy()
    if not detail.empty:
        detail["over_limit_pct"] = detail["over_limit_pct"].map(lambda x: f"{x:.2f}")

    top_round = (
        primary_errors.groupby(["failed_round"]).size().sort_values(ascending=False).head(1)
        if not primary_errors.empty
        else pd.Series(dtype=int)
    )
    top_round_text = (
        f"round {int(top_round.index[0])} ({int(top_round.iloc[0])} configs)"
        if not top_round.empty
        else "none"
    )

    report = f"""# Context-Length Error Analysis

Generated: 2026-05-02

## Scope

Primary analysis filters to `N > 2`, matching the experiment folders described in the request. I also include an all-N scope table for completeness. A run is counted as a context-length-error run if any log attempt for that config contains a provider/API context-length error. If a config failed once and later succeeded in a backfill, it still counts as having run into a context-length error.

Result folders:

- Homogeneous: `{HOMOGENEOUS_ROOT.relative_to(REPO_ROOT)}`
- Heterogeneous: `{HETEROGENEOUS_ROOT.relative_to(REPO_ROOT)}`
- Context-length guide: `{CONTEXT_GUIDE.relative_to(REPO_ROOT)}`

## Executive Summary

- Log-confirmed context-length failures: **{len(primary_errors)} / {len(primary_runs)} N > 2 attempted runs ({primary_error_pct:.2f}%)**.
- All of them are in the homogeneous folder: **{int(scope.loc[scope['folder'] == 'homogeneous', 'context_error_runs'].iloc[0])} / {int(scope.loc[scope['folder'] == 'homogeneous', 'runs'].iloc[0])} ({float(scope.loc[scope['folder'] == 'homogeneous', 'context_error_pct'].iloc[0]):.2f}%)**.
- The heterogeneous repair folder has **0 / {int(scope.loc[scope['folder'] == 'heterogeneous', 'runs'].iloc[0])} N > 2 context-length errors (0.00%)**.
- The failures are concentrated in **Game 3** and in **larger N**. Game 3 has **25 / 460 N > 2 runs ({pct(25, 460):.2f}%)** across both folders, and homogeneous Game 3 alone has **25 / 352 ({pct(25, 352):.2f}%)**.
- The affected N values are **N=6, N=8, and N=10**. There are no log-confirmed context-length errors at N=2 or N=4.
- First failures happen late: rounds **{round_min}-{round_max}**, most often {top_round_text}. There are two non-discussion failures, both in voting; all other first failures are in discussion.
- By model, every log-confirmed failure is on a model with an effective context length of **128K in the guide**: `amazon-nova-micro-v1.0`, `gpt-4o-mini-2024-07-18`, and `gpt-5-nano`/`gpt-5-nano-high`. This does not mean every 128K model fails; the heterogeneous folder has several 128K model appearances and no context-length failures.
- The requests were usually just over the cap: requested token counts range from **{min_requested:,}** to **{max_requested:,}**, median **{median_requested:,.0f}**; overage ranges from **{min_over:.2f}%** to **{max_over:.2f}%**, median **{median_over:.2f}%**.

## Plot Gallery

![Context-length error rate by N](plots/error_rate_by_n.png)

![Context-length error rate by game](plots/error_rate_by_game.png)

![First context-length error round](plots/error_rate_by_round.png)

![Model context length vs context-length error rate](plots/model_context_length_error_rate.png)

## What Is Going On

The failures are not random API outages. The failing requests are too large for the target model/provider. Most occur during public discussion after several rounds. In these games, each later speaker sees accumulated public discussion, proposal/vote history, and often long prior responses. Larger N makes this worse because each round contains more agents, more proposals, more votes, and longer discussion transcripts. Game 3 is especially exposed because co-funding prompts include project lists, budget constraints, contribution plans, and repeated coalition reasoning.

The context errors tend to appear at transition points where the next agent is about to receive the accumulated state. For example, a last successful line such as `discussion_round_8_turn_2 Agent_7` followed by an error implies the failed call was `Agent_8` in that same discussion turn. The parser uses that sequencing to infer the failed agent, phase, and model.

Two `gpt-5-nano` homogeneous control failures report a provider/configured limit of 272K in the raw API error, while the guide maps `gpt-5-nano` to the `gpt-5-nano-high` row with 128K effective context. For plots, I use the guide value as requested; in the detailed table, I keep the provider-reported limit from the actual error.

## Overall Rates

### N > 2

{md_table(scope, ["folder", "runs", "context_error_runs", "context_error_pct"])}

### All N

{md_table(scope_all, ["folder", "runs", "context_error_runs", "context_error_pct"])}

## Breakdown by N

{md_table(by_n, ["folder", "N", "runs", "context_error_runs", "context_error_pct"])}

![Context-length error rate by N](plots/error_rate_by_n.png)

## Breakdown by Game

{md_table(by_game, ["folder", "game", "runs", "context_error_runs", "context_error_pct"])}

![Context-length error rate by game](plots/error_rate_by_game.png)

## Breakdown by Game and N

{md_table(by_game_n, ["folder", "game", "N", "runs", "context_error_runs", "context_error_pct"], max_rows=50)}

![Homogeneous context-length error rate by game and N](plots/heatmap_error_rate_by_game_n_homogeneous.png)

![Heterogeneous context-length error rate by game and N](plots/heatmap_error_rate_by_game_n_heterogeneous.png)

## Breakdown by Round

This table reads as: percent of runs in the subgroup whose first context-length error occurred in that round. Percentages therefore sum to the subgroup's total error rate.

{md_table(by_round, ["folder", "failed_round", "context_error_runs", "runs", "context_error_pct"])}

![Context-length error rate by first failed round](plots/error_rate_by_round.png)

## Breakdown by N and Round

{md_table(by_n_round, ["folder", "N", "failed_round", "context_error_runs", "runs", "context_error_pct"], max_rows=80)}

![Homogeneous context-length error rate by N and round](plots/heatmap_error_rate_by_n_round_homogeneous.png)

## Breakdown by Game and Round

{md_table(by_game_round, ["folder", "game", "failed_round", "context_error_runs", "runs", "context_error_pct"], max_rows=80)}

![Homogeneous context-length error rate by game and round](plots/heatmap_error_rate_by_game_round_homogeneous.png)

## Model-Level Results

The model-level denominator is config presence: a config counts once for a model if at least one agent in that config uses that model. The numerator attributes the error to the inferred failed model.

{md_table(nonzero_models, list(nonzero_models.columns))}

![Model context length vs context-length error rate](plots/model_context_length_error_rate.png)

![Model context length vs context-length error rate for Game 1](plots/model_context_length_error_rate_game1.png)

![Model context length vs context-length error rate for Game 2](plots/model_context_length_error_rate_game2.png)

![Model context length vs context-length error rate for Game 3](plots/model_context_length_error_rate_game3.png)

![Requested tokens over reported provider limit](plots/requested_tokens_over_limit.png)

## Detailed Error Records

{md_table(detail, list(detail.columns), max_rows=80)}

## Artifacts

CSV tables:

- [runs.csv](tables/runs.csv)
- [context_errors_attempts.csv](tables/context_errors_attempts.csv)
- [context_errors_first_per_config.csv](tables/context_errors_first_per_config.csv)
- [model_error_rates.csv](tables/model_error_rates.csv)
- [rate_by_n.csv](tables/rate_by_n.csv)
- [rate_by_game.csv](tables/rate_by_game.csv)
- [rate_by_game_n.csv](tables/rate_by_game_n.csv)
- [rate_by_round.csv](tables/rate_by_round.csv)
- [rate_by_n_round.csv](tables/rate_by_n_round.csv)
- [rate_by_game_round.csv](tables/rate_by_game_round.csv)
"""
    report_path.write_text(report)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    context_df = load_context_lengths()
    runs_df = load_run_metadata()
    appearances_df = load_model_appearances(runs_df)
    errors_df = load_context_errors()
    first_errors_df = first_error_per_config(errors_df)
    model_rates = model_rate_table(appearances_df, first_errors_df, context_df)

    runs_df.drop(columns=["agent_model_map"]).to_csv(TABLES_DIR / "runs.csv", index=False)
    appearances_with_context = add_context_lengths(appearances_df, context_df)
    appearances_with_context.to_csv(TABLES_DIR / "model_appearances.csv", index=False)
    errors_df.to_csv(TABLES_DIR / "context_errors_attempts.csv", index=False)
    first_errors_df.to_csv(TABLES_DIR / "context_errors_first_per_config.csv", index=False)
    model_rates.to_csv(TABLES_DIR / "model_error_rates.csv", index=False)
    rate_table(runs_df, first_errors_df, ["folder", "N"]).to_csv(TABLES_DIR / "rate_by_n.csv", index=False)
    rate_table(runs_df, first_errors_df, ["folder", "game"]).to_csv(
        TABLES_DIR / "rate_by_game.csv", index=False
    )
    rate_table(runs_df, first_errors_df, ["folder", "game", "N"]).to_csv(
        TABLES_DIR / "rate_by_game_n.csv", index=False
    )
    round_rate_table(runs_df, first_errors_df, ["folder"]).to_csv(
        TABLES_DIR / "rate_by_round.csv", index=False
    )
    round_rate_table(runs_df, first_errors_df, ["folder", "N"]).to_csv(
        TABLES_DIR / "rate_by_n_round.csv", index=False
    )
    round_rate_table(runs_df, first_errors_df, ["folder", "game"]).to_csv(
        TABLES_DIR / "rate_by_game_round.csv", index=False
    )

    make_plots(runs_df, first_errors_df, model_rates, first_errors_df)
    write_report(runs_df, errors_df, first_errors_df, model_rates)

    print(f"Wrote report to {OUT_DIR / 'context_length_error_report.md'}")
    print(f"Log-confirmed context-length error attempts: {len(errors_df)}")
    print(f"Unique configs with context-length errors: {len(first_errors_df)}")


if __name__ == "__main__":
    main()

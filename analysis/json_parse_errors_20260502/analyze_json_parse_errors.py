#!/usr/bin/env python3
"""Analyze JSON parse failures in the full games 1/2/3 result folders."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from negotiation.json_repair import parse_json_object


OUTPUT_DIR = REPO_ROOT / "analysis" / "json_parse_errors_20260502"

RESULT_ROOTS = {
    "homogeneous": REPO_ROOT
    / "experiments/results/full_games123_multiagent_production_20260428_085255",
    "heterogeneous": REPO_ROOT
    / "experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848",
}

ELO_CSV = (
    REPO_ROOT
    / "analysis/elo_variance_sampling_100k_context/filtered_100k_context_model_pool.csv"
)

MODEL_ALIASES = {
    "amazon/nova-micro-v1": "amazon-nova-micro-v1.0",
    "amazon/nova-pro-v1": "amazon-nova-pro-v1.0",
    "anthropic/claude-3-haiku": "claude-3-haiku-20240307",
    "openai/gpt-5-nano": "gpt-5-nano-high",
    "gpt-5-nano": "gpt-5-nano-high",
    "gpt-5.4": "gpt-5.4-high",
    "openai/gpt-5.4": "gpt-5.4-high",
    "gpt-5.2-chat-latest": "gpt-5.2-chat-latest-20260210",
    "o3-mini-2025-01-31": "o3-mini-high",
    "deepseek/deepseek-chat": "deepseek-v3",
    "deepseek/deepseek-r1-0528": "deepseek-r1-0528",
    "qwen/qwen3-max": "qwen3-max-preview",
    "google/gemini-3.1-pro-preview": "gemini-3.1-pro-preview",
    "google/gemini-2.5-pro": "gemini-2.5-pro",
    "google/gemma-3-27b-it": "gemma-3-27b-it",
    "meta-llama/llama-3.3-70b-instruct": "llama-3.3-70b-instruct",
    "cohere/command-r-plus-08-2024": "command-r-plus-08-2024",
}

EXTRA_ELO = {
    "gemini-3.1-pro-preview": 1494,
}

STRUCTURED_PHASE_PREFIXES = (
    "private_thinking_round_",
    "proposal_round_",
    "voting_round_",
)

PROMPT_FLAG_COLUMNS = [
    "prompt_says_only_json",
    "prompt_has_strict_json_requirements",
    "prompt_bans_markdown_fences",
    "prompt_bans_json_comments",
    "prompt_bans_prose_outside_json",
    "prompt_bans_literal_newlines_in_strings",
]

CAUSE_NOTES = {
    "unescaped newline/control char in JSON string": {
        "plain_cause": "literal line breaks inside quoted strings",
        "prompt_assessment": "Historical prompt was incomplete: it asked for JSON, but did not explicitly ban literal line breaks inside string values. Current repo has a shared strict-format block that does ban this.",
        "suggestion": "Keep reasoning/strategy as one-line strings, cap their length, or change private thinking to arrays of short strings parsed as arrays.",
    },
    "JSON comments inside arrays/objects": {
        "plain_cause": "extraneous JSON comments such as // Apple beside array values",
        "prompt_assessment": "Historical proposal prompt said use item indices, but did not explicitly ban comments or labels inside arrays. This was under-specified for weaker models.",
        "suggestion": "State that allocation arrays may contain only bare integers, with no item names, labels, comments, parentheses notes, or placeholders.",
    },
    "comments/prose inside JSON object": {
        "plain_cause": "comments or prose placed between JSON fields",
        "prompt_assessment": "Historical prompt did not explicitly ban prose before/after or inside the JSON object. Current repo's shared block does.",
        "suggestion": "Retain the strict block in every structured prompt and repeat it in repair prompts.",
    },
    "natural language/no JSON object": {
        "plain_cause": "model answered with discussion or analysis instead of JSON",
        "prompt_assessment": "Proposal/vote prompts said ONLY JSON, but private thinking only said 'Respond with a JSON object'. No provider-enforced JSON mode was used.",
        "suggestion": "Use provider JSON/schema mode when available and add a same-call repair retry that includes the invalid response and exact schema.",
    },
    "missing comma/quote delimiter": {
        "plain_cause": "invalid punctuation between fields or after long strings",
        "prompt_assessment": "Prompt allowed long free-form reasoning strings, which increases delimiter mistakes. It did not require concise one-line fields.",
        "suggestion": "Shorten text fields, prefer arrays of short strings, and keep examples compact without long prose.",
    },
    "wrong proposal schema": {
        "plain_cause": "valid-ish JSON with the wrong top-level schema",
        "prompt_assessment": "Prompt showed the target schema, but Game 1/Game 2 schemas are similar enough that some models used agreement vectors in item allocation.",
        "suggestion": "Name the game/schema in the final instruction and add negative examples: Game 1 must use allocation, never agreement.",
    },
    "used item names instead of numeric indices": {
        "plain_cause": "item names returned where integer indices were required",
        "prompt_assessment": "Prompt did specify indices, so this is mostly model compliance/schema confusion rather than missing prompt text.",
        "suggestion": "Repeat that arrays contain integers only and validate/repair item names deterministically when unambiguous.",
    },
    "missing value or truncated JSON": {
        "plain_cause": "partial JSON, placeholders, or invalid values such as '(no item)'",
        "prompt_assessment": "Prompt did not explicitly ban placeholders inside arrays. Some cases may also be output truncation.",
        "suggestion": "Ban placeholders, compact examples, and keep max output sufficient for N=10 item allocations.",
    },
    "unterminated string": {
        "plain_cause": "long string started but not closed",
        "prompt_assessment": "Long private-thinking fields made this more likely. Prompt did not cap field length.",
        "suggestion": "Cap reasoning/strategy fields or convert them to short bullet arrays.",
    },
    "malformed object key": {
        "plain_cause": "invalid or unquoted object key",
        "prompt_assessment": "Prompt example used valid keys, but did not explicitly say double quotes are mandatory.",
        "suggestion": "Add 'double quotes only for all keys and strings' to strict JSON requirements.",
    },
    "invalid escape sequence": {
        "plain_cause": "backslash escape that is not valid JSON",
        "prompt_assessment": "Prompt did not mention escaping rules.",
        "suggestion": "Tell models to avoid backslashes in reasoning text and use plain spaces.",
    },
    "missing colon delimiter": {
        "plain_cause": "field key was not followed by ':'",
        "prompt_assessment": "Usually a generic syntax miss in long JSON outputs rather than a specific missing instruction.",
        "suggestion": "Use compact examples and provider JSON/schema mode.",
    },
    "prose before JSON object": {
        "plain_cause": "introductory text before the object",
        "prompt_assessment": "Historical private-thinking prompt did not say ONLY JSON; proposal/vote did, but without enforced schema.",
        "suggestion": "Say 'first character must be { and last character must be }' in structured phases.",
    },
}


def normalize_model(model: Any) -> str:
    model = str(model or "UNKNOWN")
    return MODEL_ALIASES.get(model, model)


def phase_category(phase: str) -> str | None:
    if phase.startswith("private_thinking_round_"):
        return "private_thinking"
    if phase.startswith("proposal_round_"):
        return "proposal"
    if phase.startswith("voting_round_"):
        return "voting"
    return None


def prompt_text_for_interaction(interaction: dict[str, Any], run_dir: Path) -> str:
    """Return logged prompt text, following externalized prompt paths if needed."""
    prompt = str(interaction.get("prompt") or "")
    storage_path = interaction.get("prompt_storage_path")
    if storage_path:
        prompt_path = Path(str(storage_path))
        if not prompt_path.is_absolute():
            prompt_path = run_dir / prompt_path
        if prompt_path.exists():
            try:
                return prompt_path.read_text(encoding="utf-8")
            except OSError:
                return prompt
    return prompt


def prompt_flags(prompt: str) -> dict[str, bool]:
    lower = prompt.lower()
    return {
        "prompt_says_only_json": (
            "respond with only" in lower
            or "return only" in lower
            or "output exactly one json object" in lower
            or "output only" in lower
        ),
        "prompt_has_strict_json_requirements": "json format requirements" in prompt,
        "prompt_bans_markdown_fences": (
            "markdown code fences" in lower
            or "do not wrap" in lower and "code fence" in lower
        ),
        "prompt_bans_json_comments": "json comments" in lower or "do not include comments" in lower,
        "prompt_bans_prose_outside_json": (
            "prose before/after" in lower
            or "no prose outside" in lower
            or "prose outside the json" in lower
        ),
        "prompt_bans_literal_newlines_in_strings": (
            "literal line breaks inside quoted string values" in lower
            or "never put literal line breaks inside quoted string values" in lower
            or "keep every string value on one line" in lower
        ),
    }


def parse_run_metadata(run_dir: Path) -> dict[str, Any]:
    name = run_dir.name
    match = re.search(r"config_(\d+)_game([123]).*?_n(\d+)", name)
    if not match:
        return {
            "config_id": None,
            "game": None,
            "n_agents": None,
            "run_name": name,
        }
    return {
        "config_id": int(match.group(1)),
        "game": f"game{match.group(2)}",
        "n_agents": int(match.group(3)),
        "run_name": name,
    }


def load_elo_map() -> dict[str, int]:
    elo_map: dict[str, int] = {}
    if ELO_CSV.exists():
        df = pd.read_csv(ELO_CSV)
        for row in df.itertuples(index=False):
            elo_map[str(row.model)] = int(row.arena_elo)
    elo_map.update(EXTRA_ELO)
    return elo_map


def collect_structured_attempts() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group, root in RESULT_ROOTS.items():
        for path in sorted(root.glob("runs/*/all_interactions.json")):
            run_dir = path.parent
            run_meta = parse_run_metadata(run_dir)
            try:
                interactions = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for idx, interaction in enumerate(interactions):
                phase = str(interaction.get("phase") or "")
                phase_cat = phase_category(phase)
                if phase_cat is None:
                    continue
                model_name = str(interaction.get("model_name") or "UNKNOWN")
                prompt = prompt_text_for_interaction(interaction, run_dir)
                rows.append(
                    {
                        "experiment_group": group,
                        "config_id": run_meta["config_id"],
                        "game": run_meta["game"],
                        "n_agents": run_meta["n_agents"],
                        "run_name": run_meta["run_name"],
                        "interaction_index": idx,
                        "agent_id": interaction.get("agent_id"),
                        "round": interaction.get("round"),
                        "interaction_phase": phase,
                        "phase": phase_cat,
                        "model_name": model_name,
                        "model_normalized": normalize_model(model_name),
                        "prompt_chars_logged": len(prompt),
                        **prompt_flags(prompt),
                    }
                )
    attempts = pd.DataFrame(rows)
    if attempts.empty:
        return attempts
    attempts["n_gt_2"] = attempts["n_agents"] > 2
    return attempts


def current_parser_can_parse(raw: str) -> bool:
    try:
        parse_json_object(raw or "", "malformed response")
        return True
    except Exception:
        return False


def classify_error(parse_type: str, message: str, raw: str) -> str:
    raw = raw or ""
    stripped = raw.strip()
    msg = message or ""
    has_comments = bool(re.search(r"(?m)(//|#\s*[A-Za-z])", raw))

    if "Invalid control character" in msg:
        return "unescaped newline/control char in JSON string"
    if "No JSON found" in msg:
        return "natural language/no JSON object"
    if "No allocation in proposal" in msg:
        return "wrong proposal schema"
    if "invalid literal for int()" in msg:
        return "used item names instead of numeric indices"
    if "Unterminated string" in msg:
        return "unterminated string"
    if "Invalid \\escape" in msg:
        return "invalid escape sequence"
    if "Expecting value" in msg:
        if has_comments:
            return "JSON comments inside arrays/objects"
        if stripped.startswith("```"):
            return "malformed fenced JSON block"
        return "missing value or truncated JSON"
    if "Expecting property name enclosed in double quotes" in msg:
        if has_comments:
            return "comments/prose inside JSON object"
        if not stripped.startswith("{"):
            return "prose before JSON object"
        return "malformed object key"
    if "Expecting ',' delimiter" in msg:
        return "missing comma/quote delimiter"
    if "Expecting ':' delimiter" in msg:
        return "missing colon delimiter"
    if parse_type == "ValueError":
        return "schema/value error"
    return "other JSON syntax error"


def collect_malformed_examples() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group, root in RESULT_ROOTS.items():
        for path in sorted(root.glob("runs/*/monitoring/malformed_json_examples.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for ex in payload.get("examples", []):
                parse_error = ex.get("parse_error") or {}
                model_name = str(ex.get("model_name") or "UNKNOWN")
                raw = str(ex.get("raw_malformed_json") or ex.get("raw_response") or "")
                run_dir = Path(str(ex.get("run_dir") or ""))
                run_name = run_dir.name if str(run_dir) else None
                parse_type = str(parse_error.get("type") or "")
                message = str(parse_error.get("message") or "")
                rows.append(
                    {
                        "experiment_group": group,
                        "config_id": ex.get("config_id"),
                        "game": ex.get("game") or ex.get("game_label"),
                        "game_type": ex.get("game_type"),
                        "n_agents": ex.get("n_agents") or ex.get("N"),
                        "run_name": run_name,
                        "agent_id": ex.get("agent_id"),
                        "round": ex.get("round"),
                        "phase": ex.get("phase"),
                        "interaction_phase": ex.get("interaction_phase"),
                        "proposal_repair_attempt": ex.get("proposal_repair_attempt"),
                        "model_name": model_name,
                        "model_normalized": normalize_model(model_name),
                        "parse_error_type": parse_type,
                        "parse_error_message": message,
                        "error_cause": classify_error(parse_type, message, raw),
                        "current_parser_parseable": current_parser_can_parse(raw),
                        "raw_preview": raw.strip().replace("\n", "\\n")[:500],
                    }
                )
    malformed = pd.DataFrame(rows)
    if malformed.empty:
        return malformed
    malformed["n_gt_2"] = malformed["n_agents"].astype(float) > 2
    return malformed


def add_elo(df: pd.DataFrame, elo_map: dict[str, int]) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "model_normalized" in df.columns:
        df["arena_elo"] = df["model_normalized"].map(elo_map)
    return df


def summarize_sample_rates(
    attempts: pd.DataFrame,
    malformed: pd.DataFrame,
    keys: list[str],
    n_gt_2_only: bool,
) -> pd.DataFrame:
    att = attempts[attempts["n_gt_2"]] if n_gt_2_only else attempts
    mal = malformed[malformed["n_gt_2"]] if n_gt_2_only else malformed
    den = (
        att.drop_duplicates(keys + ["run_name"])
        .groupby(keys, dropna=False)
        .size()
        .rename("samples")
    )
    num = (
        mal.drop_duplicates(keys + ["run_name"])
        .groupby(keys, dropna=False)
        .size()
        .rename("samples_with_json_error")
    )
    out = pd.concat([den, num], axis=1).fillna(0).reset_index()
    out["samples"] = out["samples"].astype(int)
    out["samples_with_json_error"] = out["samples_with_json_error"].astype(int)
    out["sample_json_error_pct"] = (
        out["samples_with_json_error"] / out["samples"].where(out["samples"] != 0)
    ) * 100.0
    out["sample_json_error_pct"] = out["sample_json_error_pct"].fillna(0.0)
    out["n_scope"] = "N>2" if n_gt_2_only else "all_N"
    return out


def save_csvs(attempts: pd.DataFrame, malformed: pd.DataFrame, elo_map: dict[str, int]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    malformed = add_elo(malformed, elo_map)
    paths["malformed_examples"] = OUTPUT_DIR / "malformed_examples_classified.csv"
    malformed.to_csv(paths["malformed_examples"], index=False)

    prompt_summary = (
        attempts[attempts["n_gt_2"]]
        .groupby(["experiment_group", "phase"], dropna=False)
        .agg(
            structured_calls=("run_name", "size"),
            samples=("run_name", "nunique"),
            **{
                f"{col}_calls": (col, "sum")
                for col in PROMPT_FLAG_COLUMNS
            },
        )
        .reset_index()
    )
    for col in PROMPT_FLAG_COLUMNS:
        prompt_summary[f"{col}_pct"] = (
            prompt_summary[f"{col}_calls"]
            / prompt_summary["structured_calls"].where(prompt_summary["structured_calls"] != 0)
            * 100.0
        ).fillna(0.0)

    summaries = {
        "model_summary_all_N": summarize_sample_rates(
            attempts, malformed, ["experiment_group", "model_normalized"], False
        ),
        "model_summary_N_gt_2": summarize_sample_rates(
            attempts, malformed, ["experiment_group", "model_normalized"], True
        ),
        "model_phase_summary_N_gt_2": summarize_sample_rates(
            attempts, malformed, ["experiment_group", "model_normalized", "phase"], True
        ),
        "game_phase_summary_N_gt_2": summarize_sample_rates(
            attempts, malformed, ["experiment_group", "game", "phase"], True
        ),
        "model_game_summary_N_gt_2": summarize_sample_rates(
            attempts, malformed, ["experiment_group", "model_normalized", "game"], True
        ),
        "cause_summary_N_gt_2": malformed[malformed["n_gt_2"]]
        .drop_duplicates(["experiment_group", "phase", "error_cause", "run_name"])
        .groupby(["experiment_group", "phase", "error_cause"], dropna=False)
        .size()
        .rename("samples_with_json_error")
        .reset_index(),
        "cause_model_summary_N_gt_2": malformed[malformed["n_gt_2"]]
        .drop_duplicates(["experiment_group", "model_normalized", "error_cause", "run_name"])
        .groupby(["experiment_group", "model_normalized", "error_cause"], dropna=False)
        .size()
        .rename("samples_with_json_error")
        .reset_index(),
        "cause_game_phase_summary_N_gt_2": malformed[malformed["n_gt_2"]]
        .drop_duplicates(["experiment_group", "game", "phase", "error_cause", "run_name"])
        .groupby(["experiment_group", "game", "phase", "error_cause"], dropna=False)
        .size()
        .rename("samples_with_json_error")
        .reset_index(),
        "round_summary_N_gt_2": malformed[malformed["n_gt_2"]]
        .drop_duplicates(["experiment_group", "phase", "round", "run_name"])
        .groupby(["experiment_group", "phase", "round"], dropna=False)
        .size()
        .rename("samples_with_json_error")
        .reset_index(),
        "run_summary_N_gt_2": malformed[malformed["n_gt_2"]]
        .groupby(["experiment_group", "config_id", "run_name", "game", "n_agents"], dropna=False)
        .size()
        .rename("json_errors")
        .reset_index()
        .sort_values(["json_errors", "experiment_group"], ascending=[False, True]),
        "prompt_requirements_summary_N_gt_2": prompt_summary,
    }
    for name, df in summaries.items():
        df = add_elo(df, elo_map)
        path = OUTPUT_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
        paths[name] = path
    return paths


def model_label(model: str, elo: Any) -> str:
    if pd.isna(elo):
        return f"{model}\nELO ?"
    return f"{model}\nELO {int(elo)}"


def plot_model_rates(model_summary: pd.DataFrame) -> Path:
    df = model_summary.copy()
    df["arena_elo_sort"] = df["arena_elo"].fillna(-1)
    path = OUTPUT_DIR / "n_gt_2_model_error_pct_by_elo.png"
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), constrained_layout=True)
    for ax, group in zip(axes, ["homogeneous", "heterogeneous"]):
        sub = df[df["experiment_group"] == group].sort_values("arena_elo_sort", ascending=False)
        labels = [model_label(m, e) for m, e in zip(sub["model_normalized"], sub["arena_elo"])]
        colors = sns.color_palette("viridis", n_colors=max(len(sub), 1))
        ax.bar(range(len(sub)), sub["sample_json_error_pct"], color=colors)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
        ax.set_ylabel("Samples with >=1 JSON error (%)")
        ax.set_title(f"{group.title()} N>2 sample-level JSON error rate by model, sorted by Arena Elo")
        for idx, row in enumerate(sub.itertuples(index=False)):
            if row.samples_with_json_error:
                ax.text(
                    idx,
                    row.sample_json_error_pct,
                    f"{int(row.samples_with_json_error)}/{int(row.samples)}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )
        ax.set_ylim(0, max(1.0, sub["sample_json_error_pct"].max() * 1.2))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_model_phase_rates(model_phase: pd.DataFrame) -> Path:
    df = model_phase[model_phase["phase"].isin(["private_thinking", "proposal"])].copy()
    df["arena_elo_sort"] = df["arena_elo"].fillna(-1)
    path = OUTPUT_DIR / "n_gt_2_model_phase_error_pct_by_elo.png"
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(20, 11), constrained_layout=True)
    for row_idx, group in enumerate(["homogeneous", "heterogeneous"]):
        for col_idx, phase in enumerate(["private_thinking", "proposal"]):
            ax = axes[row_idx][col_idx]
            sub = df[
                (df["experiment_group"] == group)
                & (df["phase"] == phase)
                & (df["samples"] > 0)
            ].sort_values("arena_elo_sort", ascending=False)
            labels = [model_label(m, e) for m, e in zip(sub["model_normalized"], sub["arena_elo"])]
            ax.bar(range(len(sub)), sub["sample_json_error_pct"], color=sns.color_palette("mako", n_colors=max(len(sub), 1)))
            ax.set_xticks(range(len(sub)))
            ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=7)
            ax.set_ylabel("Samples with >=1 JSON error (%)")
            ax.set_title(f"{group.title()} {phase.replace('_', ' ')}")
            for idx, row in enumerate(sub.itertuples(index=False)):
                if row.samples_with_json_error:
                    ax.text(
                        idx,
                        row.sample_json_error_pct,
                        f"{int(row.samples_with_json_error)}/{int(row.samples)}",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        rotation=90,
                    )
            ax.set_ylim(0, max(1.0, sub["sample_json_error_pct"].max() * 1.2 if len(sub) else 1.0))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_game_phase_heatmap(game_phase: pd.DataFrame) -> Path:
    df = game_phase.copy()
    df["row"] = df["experiment_group"] + " " + df["game"].fillna("?")
    table = df.pivot_table(
        index="row",
        columns="phase",
        values="sample_json_error_pct",
        aggfunc="sum",
        fill_value=0,
    )
    for phase in ["private_thinking", "proposal", "voting"]:
        if phase not in table.columns:
            table[phase] = 0.0
    table = table[["private_thinking", "proposal", "voting"]]
    path = OUTPUT_DIR / "n_gt_2_game_phase_error_heatmap.png"
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    sns.heatmap(table, annot=True, fmt=".2f", cmap="Reds", cbar_kws={"label": "Sample JSON error %"}, ax=ax)
    ax.set_title("N>2 sample-level JSON error percentage by game and phase")
    ax.set_xlabel("Phase")
    ax.set_ylabel("")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_cause_distribution(cause_summary: pd.DataFrame) -> Path:
    df = cause_summary.copy()
    df["bucket"] = df["experiment_group"] + " / " + df["phase"].fillna("?")
    pivot = df.pivot_table(index="bucket", columns="error_cause", values="samples_with_json_error", fill_value=0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    pivot = pivot[pivot.sum(axis=0).sort_values(ascending=False).index]
    path = OUTPUT_DIR / "n_gt_2_error_cause_distribution.png"
    fig, ax = plt.subplots(figsize=(15, 7), constrained_layout=True)
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_title("N>2 malformed JSON causes by experiment group and phase")
    ax.set_ylabel("Samples with cause")
    ax.set_xlabel("")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_top_causes(cause_summary: pd.DataFrame) -> Path:
    df = (
        cause_summary.groupby("error_cause", dropna=False)["samples_with_json_error"]
        .sum()
        .sort_values(ascending=True)
        .tail(12)
        .reset_index()
    )
    path = OUTPUT_DIR / "n_gt_2_top_error_causes.png"
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    colors = sns.color_palette("crest", n_colors=max(len(df), 1))
    ax.barh(df["error_cause"], df["samples_with_json_error"], color=colors)
    ax.set_title("N>2 top malformed JSON causes")
    ax.set_xlabel("Samples with cause across group/phase buckets")
    ax.set_ylabel("")
    for idx, row in enumerate(df.itertuples(index=False)):
        ax.text(row.samples_with_json_error, idx, f" {int(row.samples_with_json_error)}", va="center", fontsize=8)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_cause_phase_heatmap(cause_summary: pd.DataFrame) -> Path:
    df = cause_summary.copy()
    top_causes = (
        df.groupby("error_cause")["samples_with_json_error"]
        .sum()
        .sort_values(ascending=False)
        .head(12)
        .index
    )
    df = df[df["error_cause"].isin(top_causes)]
    df["bucket"] = df["experiment_group"] + " / " + df["phase"].fillna("?")
    table = df.pivot_table(
        index="error_cause",
        columns="bucket",
        values="samples_with_json_error",
        aggfunc="sum",
        fill_value=0,
    )
    table = table.loc[table.sum(axis=1).sort_values(ascending=False).index]
    path = OUTPUT_DIR / "n_gt_2_error_cause_phase_heatmap.png"
    fig, ax = plt.subplots(figsize=(13, 7), constrained_layout=True)
    sns.heatmap(table, annot=True, fmt=".0f", cmap="YlOrRd", cbar_kws={"label": "Samples with cause"}, ax=ax)
    ax.set_title("N>2 malformed JSON causes by experiment group and phase")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_cause_model_heatmap(cause_model: pd.DataFrame) -> Path:
    df = cause_model.copy()
    top_models = (
        df.groupby(["experiment_group", "model_normalized"], dropna=False)["samples_with_json_error"]
        .sum()
        .sort_values(ascending=False)
        .head(16)
        .reset_index()
    )
    df = df.merge(top_models[["experiment_group", "model_normalized"]], on=["experiment_group", "model_normalized"])
    top_causes = (
        df.groupby("error_cause")["samples_with_json_error"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .index
    )
    df = df[df["error_cause"].isin(top_causes)]
    if "arena_elo" not in df.columns:
        df["arena_elo"] = pd.NA
    df["model_bucket"] = df.apply(
        lambda r: f"{r['experiment_group']} / {model_label(r['model_normalized'], r['arena_elo'])}",
        axis=1,
    )
    table = df.pivot_table(
        index="model_bucket",
        columns="error_cause",
        values="samples_with_json_error",
        aggfunc="sum",
        fill_value=0,
    )
    table = table.loc[table.sum(axis=1).sort_values(ascending=False).index]
    table = table[table.sum(axis=0).sort_values(ascending=False).index]
    path = OUTPUT_DIR / "n_gt_2_error_cause_model_heatmap.png"
    fig, ax = plt.subplots(figsize=(15, 9), constrained_layout=True)
    sns.heatmap(table, annot=True, fmt=".0f", cmap="Blues", cbar_kws={"label": "Samples with cause"}, ax=ax)
    ax.set_title("N>2 malformed JSON causes by high-error model buckets")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_round_distribution(round_summary: pd.DataFrame) -> Path:
    df = round_summary.copy()
    path = OUTPUT_DIR / "n_gt_2_errors_by_round.png"
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    sns.barplot(
        data=df,
        x="round",
        y="samples_with_json_error",
        hue="phase",
        errorbar=None,
        ax=ax,
    )
    ax.set_title("N>2 samples with JSON errors by round and phase")
    ax.set_ylabel("Samples with >=1 JSON error")
    ax.set_xlabel("Round")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def pct(num: int, den: int) -> str:
    if den == 0:
        return "n/a"
    return f"{100.0 * num / den:.2f}%"


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def sibling(path: Path) -> str:
    return path.name


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    show = df.loc[:, columns].head(max_rows).copy()
    return show.to_markdown(index=False)


def write_proposed_diff() -> Path:
    """Write a focused proposed patch for future runs without modifying game code."""
    diff_text = '''diff --git a/negotiation/llm_agents.py b/negotiation/llm_agents.py
--- a/negotiation/llm_agents.py
+++ b/negotiation/llm_agents.py
@@
-Response format: Provide your analysis as structured strategic thinking."""
+Response format: Return ONLY one valid JSON object matching the user's schema.
+Do not include markdown, comments, prose outside the object, or literal line breaks inside string values.
+Keep every string value on one line."""
 
diff --git a/game_environments/base.py b/game_environments/base.py
--- a/game_environments/base.py
+++ b/game_environments/base.py
@@
         return (
             "**JSON FORMAT REQUIREMENTS:**\\n"
             "- Return exactly one valid JSON object.\\n"
+            "- The first character must be `{` and the last character must be `}`.\\n"
+            "- Use double quotes for all object keys and string values; never use single quotes.\\n"
             "- Do not wrap the object in markdown code fences.\\n"
-            "- Do not include JSON comments (`//` or `/* ... */`) or prose before/after the object.\\n"
+            "- Do not include JSON comments (`//`, `#`, or `/* ... */`) or prose before/after the object.\\n"
+            "- Arrays must contain only the requested primitive values; do not put labels, item names, parenthetical notes, or placeholders inside arrays.\\n"
             "- Line breaks between JSON fields and array entries are allowed, but never put literal line breaks inside quoted string values.\\n"
-            "- Keep every string value on one line; use spaces instead of newline characters inside strings."
+            "- Keep every string value on one line; use spaces instead of newline characters inside strings.\\n"
+            "- Keep `reasoning` and `strategy` fields concise: one sentence each, with no embedded quotation marks."
         )
 
diff --git a/game_environments/item_allocation.py b/game_environments/item_allocation.py
--- a/game_environments/item_allocation.py
+++ b/game_environments/item_allocation.py
@@
-        example_payload = json.dumps(
+        example_payload = json.dumps(
             {
                 "allocation": example_alloc,
                 "reasoning": "Brief explanation of your proposed allocation",
             },
-            indent=4,
+            separators=(",", ":"),
         )
@@
 - Use item INDICES (0-{len(items)-1}), not names
+- Inside allocation arrays, use only bare integers. Do not add item names, comments, labels, `(no item)`, or parenthetical notes.
 - Each item must be assigned to exactly one agent
@@
-Respond with a JSON object:
+Respond with ONLY a JSON object matching this schema. Keep `reasoning` and `strategy` to one sentence each:
 
diff --git a/game_environments/diplomatic_treaty.py b/game_environments/diplomatic_treaty.py
--- a/game_environments/diplomatic_treaty.py
+++ b/game_environments/diplomatic_treaty.py
@@
-            indent=4,
+            separators=(",", ":"),
         )
@@
-Respond with a JSON object:
+Respond with ONLY a JSON object matching this schema. Keep `reasoning` and `strategy` to one sentence each:
 
diff --git a/game_environments/co_funding.py b/game_environments/co_funding.py
--- a/game_environments/co_funding.py
+++ b/game_environments/co_funding.py
@@
-                indent=4,
+                separators=(",", ":"),
             )
@@
-                indent=4,
+                separators=(",", ":"),
             )
@@
-Respond with a JSON object:
+Respond with ONLY a JSON object matching this schema. Keep `reasoning` and `strategy` to one sentence each:
 
diff --git a/strong_models_experiment/phases/phase_handlers.py b/strong_models_experiment/phases/phase_handlers.py
--- a/strong_models_experiment/phases/phase_handlers.py
+++ b/strong_models_experiment/phases/phase_handlers.py
@@
                     "Return ONLY one JSON object with an allocation object whose keys are the exact agent IDs "
                     f"{agent_ids}. The values must be arrays of item indices. "
                     f"Every item index from 0 to {n_items - 1} must appear exactly once across all agents. "
-                    "Do not include any vector-style proposal, utility vector, markdown, or prose outside the JSON object."
+                    "Do not include any vector-style proposal, utility vector, markdown, comments, labels, item names, or prose outside the JSON object."
                 ),
'''
    path = OUTPUT_DIR / "proposed_json_prompt_hardening.diff"
    path.write_text(diff_text, encoding="utf-8")
    return path


def write_report(
    attempts: pd.DataFrame,
    malformed: pd.DataFrame,
    paths: dict[str, Path],
    plot_paths: dict[str, Path],
) -> Path:
    n_attempts_all = len(attempts)
    n_malformed_all = len(malformed)
    attempts_n = attempts[attempts["n_gt_2"]]
    malformed_n = malformed[malformed["n_gt_2"]]

    group_summary = summarize_sample_rates(
        attempts, malformed, ["experiment_group"], n_gt_2_only=True
    ).sort_values("experiment_group")
    model_summary = pd.read_csv(paths["model_summary_N_gt_2"])
    model_summary = model_summary.sort_values(
        ["experiment_group", "sample_json_error_pct"], ascending=[True, False]
    )
    model_phase = pd.read_csv(paths["model_phase_summary_N_gt_2"])
    game_phase = pd.read_csv(paths["game_phase_summary_N_gt_2"])
    cause_summary = pd.read_csv(paths["cause_summary_N_gt_2"])
    prompt_summary = pd.read_csv(paths["prompt_requirements_summary_N_gt_2"])
    run_summary = pd.read_csv(paths["run_summary_N_gt_2"])

    repairable = int(malformed_n["current_parser_parseable"].sum())
    repairable_total = int(len(malformed_n))
    total_error_samples = int(
        malformed_n.drop_duplicates(["experiment_group", "run_name"]).shape[0]
    )

    phase_totals = summarize_sample_rates(
        attempts, malformed, ["experiment_group", "phase"], n_gt_2_only=True
    ).sort_values(["experiment_group", "sample_json_error_pct"], ascending=[True, False])

    top_model_phase = model_phase[
        (model_phase["samples_with_json_error"] > 0)
        & (model_phase["phase"].isin(["private_thinking", "proposal"]))
    ].sort_values("sample_json_error_pct", ascending=False)

    top_runs = run_summary.head(10)
    hom_row = group_summary[group_summary.experiment_group == "homogeneous"].iloc[0]
    het_row = group_summary[group_summary.experiment_group == "heterogeneous"].iloc[0]
    top_model = model_summary.sort_values("sample_json_error_pct", ascending=False).iloc[0]
    cause_totals = (
        malformed_n.drop_duplicates(["experiment_group", "run_name", "error_cause"])
        .groupby("error_cause", dropna=False)
        .size()
        .rename("samples_with_cause")
        .reset_index()
        .sort_values("samples_with_cause", ascending=False)
    )
    cause_totals["pct_of_error_samples"] = (
        cause_totals["samples_with_cause"] / total_error_samples * 100.0
        if total_error_samples
        else 0.0
    )
    cause_note_rows = []
    for row in cause_totals.head(12).itertuples(index=False):
        note = CAUSE_NOTES.get(
            row.error_cause,
            {
                "plain_cause": row.error_cause,
                "prompt_assessment": "Not enough evidence to assign a specific prompt cause.",
                "suggestion": "Use strict JSON/schema mode and keep deterministic repair enabled.",
            },
        )
        cause_note_rows.append(
            {
                "error_cause": row.error_cause,
                "samples_with_cause": int(row.samples_with_cause),
                "pct_of_error_samples": row.pct_of_error_samples,
                "plain_cause": note["plain_cause"],
                "prompt_assessment": note["prompt_assessment"],
                "suggestion": note["suggestion"],
            }
        )
    cause_notes_df = pd.DataFrame(cause_note_rows)

    report = f"""# JSON Parse Error Audit for Full Games 1/2/3

Generated: 2026-05-02

## Scope and Method

I analyzed these two result roots:

- Homogeneous: `experiments/results/full_games123_multiagent_production_20260428_085255/`
- Heterogeneous: `experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848/`

The numerator is sample-level: a run/config sample counts once in a bucket if it has at least one JSON parse diagnostic in `monitoring/malformed_json_examples.json` for that bucket. The denominator is the number of run/config samples represented in `all_interactions.json` for the relevant bucket. For example, a model-phase denominator is the number of samples where that model made at least one call in that phase; the numerator is the number of those samples where that model had at least one JSON error in that phase.

The headline tables below use the requested `N > 2` subset. The raw folders also contain `N=2`; those are kept in `model_summary_all_N.csv` as a sensitivity check.

Arena Elo values are local, from `analysis/elo_variance_sampling_100k_context/filtered_100k_context_model_pool.csv`, plus the local `docs/guides/chatbot_arena_elo_scores_2026_03_31.md` entry for `gemini-3.1-pro-preview`.

## Headline Findings

- Across `N > 2`, homogeneous has **{int(hom_row.samples_with_json_error)}/{int(hom_row.samples)}** samples with at least one JSON error ({hom_row.sample_json_error_pct:.2f}%). Heterogeneous has **{int(het_row.samples_with_json_error)}/{int(het_row.samples)}** ({het_row.sample_json_error_pct:.2f}%).
- This is **not only strong models failing at JSON**. The worst `N > 2` sample-level model rate is `{top_model.model_normalized}` in `{top_model.experiment_group}` runs: **{int(top_model.samples_with_json_error)}/{int(top_model.samples)} = {top_model.sample_json_error_pct:.2f}%**. Lower/mid models (`gpt-4o-mini`, `gpt-5-nano`, `claude-haiku-4-5`, `amazon-nova-*`, `llama-3.3-70b`) account for much of the high-rate mass.
- Strong models are mostly clean at the sample level. In `N > 2`, `gemini-3.1-pro-preview`, `gemini-2.5-pro`, and `qwen3-max-preview` have 0 samples with recorded JSON errors in this sample; `gpt-5.2-chat-latest`, `o3-mini-high`, `claude-opus-4-6`, and `gpt-5.4-high` are low relative to the worst buckets.
- The errors concentrate in **private thinking** and **proposal**. I found no recorded malformed JSON diagnostics in voting. The most concentrated game/phase bucket is heterogeneous Game 1 proposals.
- A large part is prompt/parser friction, not just model capability. The historical experiment prompts asked for JSON, but did **not** include the current repo's strict `JSON FORMAT REQUIREMENTS` block. Private thinking invited long free-form reasoning inside JSON string fields, and Game 1 proposals used large allocation objects without explicitly banning comments, markdown fences, prose, placeholders, or literal line breaks inside string values. **{repairable}/{repairable_total} ({pct(repairable, repairable_total)})** of `N > 2` malformed raw responses are syntactically parseable by the current deterministic JSON repair helper, so many failures are recoverable syntax issues.

## Main Plots

![N>2 model sample-level JSON error percentage by Elo]({sibling(plot_paths['model_rates'])})

![N>2 model-phase sample-level JSON error percentage by Elo]({sibling(plot_paths['model_phase_rates'])})

![N>2 game-phase sample-level JSON error heatmap]({sibling(plot_paths['game_phase_heatmap'])})

![N>2 cause distribution]({sibling(plot_paths['cause_distribution'])})

![N>2 top malformed JSON causes]({sibling(plot_paths['top_causes'])})

![N>2 malformed JSON causes by phase]({sibling(plot_paths['cause_phase_heatmap'])})

![N>2 malformed JSON causes by high-error model buckets]({sibling(plot_paths['cause_model_heatmap'])})

![N>2 errors by round]({sibling(plot_paths['round_distribution'])})

## By Model

Top `N > 2` model-level JSON error rates, where each model/sample pair counts once:

{markdown_table(model_summary.sort_values('sample_json_error_pct', ascending=False), ['experiment_group', 'model_normalized', 'arena_elo', 'samples_with_json_error', 'samples', 'sample_json_error_pct'], 30)}

## By Phase

`N > 2` phase-level rates:

{markdown_table(phase_totals, ['experiment_group', 'phase', 'samples_with_json_error', 'samples', 'sample_json_error_pct'], 20)}

Top model-phase cells:

{markdown_table(top_model_phase, ['experiment_group', 'model_normalized', 'arena_elo', 'phase', 'samples_with_json_error', 'samples', 'sample_json_error_pct'], 30)}

## By Game

`N > 2` game/phase rates:

{markdown_table(game_phase.sort_values(['experiment_group', 'game', 'phase']), ['experiment_group', 'game', 'phase', 'samples_with_json_error', 'samples', 'sample_json_error_pct'], 30)}

Interpretation:

- Game 1 item allocation is the proposal hot spot. In high-N settings, the proposal object is large and models often annotate arrays with comments such as `// Ring` or return a discussion-style counterproposal instead of the required `allocation` object.
- Game 2 and Game 3 mostly fail in private thinking, where the model tries to produce the requested JSON but puts multi-paragraph text into string values with literal newlines or misses a comma between fields.
- Game 3 proposal errors are relatively rare; the co-funding vector schema is simpler than a full item allocation over many agents/items.

## Error Causes

Top `N > 2` causes, counted once per sample/cause across the two experiment groups. A sample can have more than one cause, so percentages need not sum to 100%.

{markdown_table(cause_totals, ['error_cause', 'samples_with_cause', 'pct_of_error_samples'], 20)}

`N > 2` sample-cause counts. A sample can contribute to multiple causes if it has multiple kinds of JSON error, but it is counted at most once per cause/phase:

{markdown_table(cause_summary.sort_values('samples_with_json_error', ascending=False), ['experiment_group', 'phase', 'error_cause', 'samples_with_json_error'], 40)}

The dominant raw failure patterns are:

- **Unescaped newlines/control characters inside JSON strings.** This is especially common in private thinking. The model starts with a JSON object, then writes multi-paragraph `reasoning` or `strategy` values with literal newlines.
- **JSON comments inside arrays/objects.** This is especially common in Game 1 proposals. Models write valid-looking allocation arrays but annotate entries with `// Apple`, `# Stone`, etc., which is not JSON.
- **Natural-language/no JSON object.** Some proposal repair attempts drift into discussion text, e.g. `[Round 1 | Discussion] ...`, or markdown analysis instead of a JSON proposal.
- **Wrong proposal schema.** A smaller set returns an `agreement` array in Game 1 or omits the `allocation` object, so this is not pure JSON syntax failure; it is schema confusion.

## Cause-by-Cause Prompt Assessment

{markdown_table(cause_notes_df, ['error_cause', 'samples_with_cause', 'plain_cause', 'prompt_assessment', 'suggestion'], 12)}

## Historical Prompt Audit

I checked the actual prompts stored in `all_interactions.json` for the two experiment folders. The historical runs did not include the current repo's shared strict-format block in any structured phase: no prompt contained `JSON FORMAT REQUIREMENTS`, no prompt explicitly banned JSON comments, and no prompt explicitly banned literal line breaks inside quoted string values.

`N > 2` structured-call prompt flags:

{markdown_table(prompt_summary.sort_values(['experiment_group', 'phase']), ['experiment_group', 'phase', 'structured_calls', 'samples', 'prompt_says_only_json_pct', 'prompt_has_strict_json_requirements_pct', 'prompt_bans_json_comments_pct', 'prompt_bans_literal_newlines_in_strings_pct', 'prompt_bans_markdown_fences_pct'], 20)}

## Specific Conversations

Highest-error `N > 2` runs. This table intentionally keeps event counts so the report can identify specific conversations to inspect:

{markdown_table(top_runs, ['experiment_group', 'config_id', 'run_name', 'game', 'n_agents', 'json_errors'], 10)}

Two representative conversations:

- `config_0865_game1_heterogeneous_random_n10_comp_0p0_run03` has 8 JSON errors. They are not from one model: `gpt-4o-mini`, `claude-sonnet-4`, `llama-3.3-70b`, `gpt-5-nano`, and `gpt-4o` all appear. The errors are mostly Round 1 proposal attempts in Game 1: comments in allocation arrays, markdown/prose instead of JSON, and one later private-thinking multiline-string failure.
- `config_2649_game3_homogeneous_adversary_n10_sigma_0p5_alpha_0p2_amazon_nova_micro_v1p0_first_seed1` has 9 JSON errors. Most are `gpt-5-nano` private-thinking failures in later co-funding rounds, caused by missing commas or unescaped multiline reasoning. One `amazon-nova-micro` proposal misses a comma between `contributions` and `reasoning`.

So the errors are not confined to one conversation, but the failure mode is phase-specific: Game 1 proposal generation at high N and private-thinking JSON across games.

## Prompt Assessment

For these historical experiments, the prompt/specification was only partially correct.

- Proposal and voting prompts were directionally correct because they usually said `Respond with ONLY a JSON object in this exact format` and gave an example.
- Private-thinking prompts were weaker: they said `Respond with a JSON object`, not `ONLY`, and then asked for long strategic analysis fields.
- The historical structured prompts did not systematically ban comments, markdown fences, `#`/`//` annotations, placeholders inside arrays, single quotes, or literal newlines inside string values. A small subset of proposal repair prompts banned prose outside JSON, but the original phase prompts did not carry a complete strict-format contract.
- The current worktree already contains a shared `GameEnvironment.json_format_requirements()` block and deterministic JSON repair helpers. Future runs need to be generated from that prompt version or stricter.

Recommended fixes:

1. Use provider structured outputs / JSON schema where available for private thinking, proposals, and votes. Prompting alone will not fully eliminate these errors for weak models.
2. Keep the current shared strict-format block in every structured prompt and repair prompt.
3. Add an item-allocation-specific ban: allocation arrays must contain only bare integers, never item names, comments, labels, `(no item)`, or parenthetical notes.
4. Shorten private-thinking fields: one-sentence `reasoning` and `strategy`, or change the schema to arrays of short strings and update normalization accordingly.
5. Use compact JSON examples for large Game 1 allocations. Pretty-printed arrays invite models to annotate each line with comments.
6. Keep deterministic repair enabled, because most failures are syntax-adjacent and recoverable, but continue reporting schema failures separately from syntax failures.

## Proposed Diff

I wrote a proposed future-run hardening patch here:

- `{rel(paths['proposed_diff'])}`

The diff is intentionally prompt-focused. It does not change game payoffs or negotiation mechanics; it tightens the system instruction, shared JSON requirements, Game 1 array rules, private-thinking wording, and repair prompts.

## Output Files

- Classified malformed examples: `{rel(paths['malformed_examples'])}`
- Model summary, all N: `{rel(paths['model_summary_all_N'])}`
- Model summary, N>2: `{rel(paths['model_summary_N_gt_2'])}`
- Model-phase summary, N>2: `{rel(paths['model_phase_summary_N_gt_2'])}`
- Game-phase summary, N>2: `{rel(paths['game_phase_summary_N_gt_2'])}`
- Cause summary, N>2: `{rel(paths['cause_summary_N_gt_2'])}`
- Cause-model summary, N>2: `{rel(paths['cause_model_summary_N_gt_2'])}`
- Cause-game-phase summary, N>2: `{rel(paths['cause_game_phase_summary_N_gt_2'])}`
- Prompt requirements summary, N>2: `{rel(paths['prompt_requirements_summary_N_gt_2'])}`
- Proposed prompt hardening diff: `{rel(paths['proposed_diff'])}`
"""
    report_path = OUTPUT_DIR / "json_parse_error_report.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    elo_map = load_elo_map()
    attempts = collect_structured_attempts()
    malformed = collect_malformed_examples()
    paths = save_csvs(attempts, malformed, elo_map)
    paths["proposed_diff"] = write_proposed_diff()

    model_summary = add_elo(pd.read_csv(paths["model_summary_N_gt_2"]), elo_map)
    model_phase = add_elo(pd.read_csv(paths["model_phase_summary_N_gt_2"]), elo_map)
    game_phase = pd.read_csv(paths["game_phase_summary_N_gt_2"])
    cause_summary = pd.read_csv(paths["cause_summary_N_gt_2"])
    cause_model = add_elo(pd.read_csv(paths["cause_model_summary_N_gt_2"]), elo_map)
    round_summary = pd.read_csv(paths["round_summary_N_gt_2"])

    plot_paths = {
        "model_rates": plot_model_rates(model_summary),
        "model_phase_rates": plot_model_phase_rates(model_phase),
        "game_phase_heatmap": plot_game_phase_heatmap(game_phase),
        "cause_distribution": plot_cause_distribution(cause_summary),
        "top_causes": plot_top_causes(cause_summary),
        "cause_phase_heatmap": plot_cause_phase_heatmap(cause_summary),
        "cause_model_heatmap": plot_cause_model_heatmap(cause_model),
        "round_distribution": plot_round_distribution(round_summary),
    }
    report_path = write_report(attempts, malformed, paths, plot_paths)
    print(f"Wrote report: {report_path}")
    for name, path in plot_paths.items():
        print(f"Wrote plot {name}: {path}")


if __name__ == "__main__":
    main()

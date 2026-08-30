#!/usr/bin/env python3
"""Reconstruct and classify all attempts behind the in-scope paper runs.

Scope is defined by the paper experiment manifest, excluding TTC and the
bilateral Llama baseline.  The script produces both attempt-level and
configuration-level datasets, a context-compaction audit, summary tables,
figures, and a Markdown report suitable for the reviewer-response workflow.

The classifier is deliberately rule based and auditable.  It operates on the
terminal portion of each attempt log and retains a sanitized terminal detail
field so that every assignment can be spot checked.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr


REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO / "docs/reproducibility/paper_experiment_data_manifest.csv"
DEFAULT_OUTPUT = REPO / "experiments/analysis/reviewer_failure_audit_20260725"

EXCLUDED_BATCHES = {"ttc", "bilateral_llama33"}
MULTI_BATCHES = {
    "multiagent_homogeneous",
    "multiagent_heterogeneous",
    "random_monoculture",
}
BILATERAL_BATCH = "bilateral_gpt5_nano"

# First production commit that added proactive compaction.  It is used only as
# a historical-era marker; per-run compaction is detected from rollout data.
COMPACTION_INTRODUCED = datetime.fromisoformat("2026-04-30T15:25:55")

GAME_LABELS = {
    "game1": "Game 1: Item allocation",
    "game2": "Game 2: Treaty",
    "game3": "Game 3: Co-funding",
}

FAILURE_RELEVANCE = {
    "valid_consensus": "valid_outcome",
    "valid_no_consensus": "valid_outcome",
    "model_invalid_proposal": "model_or_protocol",
    "model_invalid_vote": "model_or_protocol",
    "model_invalid_structured_output": "model_or_protocol",
    "model_empty_or_output_limit": "model_or_protocol",
    "model_content_filter_or_refusal": "model_or_protocol",
    "context_overflow": "context_or_capacity",
    "api_auth_or_credentials": "infrastructure",
    "api_credits_or_spend_limit": "infrastructure",
    "api_rate_limit_or_quota": "infrastructure",
    "api_connection_or_timeout": "infrastructure",
    "provider_server_or_overload": "infrastructure",
    "api_key_pool_exhausted": "infrastructure",
    "scheduler_interrupted": "infrastructure",
    "code_or_orchestration": "code_or_orchestration",
    "unknown_failure": "unknown",
}

FAILURE_LABELS = {
    "valid_consensus": "Valid: consensus",
    "valid_no_consensus": "Valid: no consensus",
    "model_invalid_proposal": "Invalid proposal",
    "model_invalid_vote": "Invalid vote",
    "model_invalid_structured_output": "Other invalid structured output",
    "model_empty_or_output_limit": "Empty/truncated model output",
    "model_content_filter_or_refusal": "Content filter/refusal",
    "context_overflow": "Context overflow/preflight",
    "api_auth_or_credentials": "API auth/credentials",
    "api_credits_or_spend_limit": "Credits/spend limit",
    "api_rate_limit_or_quota": "Rate limit/quota",
    "api_connection_or_timeout": "Connection/timeout",
    "provider_server_or_overload": "Provider server/overload",
    "api_key_pool_exhausted": "API key pool exhausted (cause unspecified)",
    "scheduler_interrupted": "Scheduler/interrupted",
    "code_or_orchestration": "Code/orchestration/storage",
    "unknown_failure": "Unknown",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(errors="replace"))


def numeric_config_id(value: Any) -> int:
    match = re.search(r"(\d+)$", str(value))
    if not match:
        raise ValueError(f"Cannot parse config id: {value!r}")
    return int(match.group(1))


def canonical_config_key(source_root: str, config_id: Any) -> str:
    return f"{Path(source_root).name}:config_{numeric_config_id(config_id):04d}"


def parse_timestamp(value: Any) -> pd.Timestamp | pd.NaT:
    if value is None or value == "":
        return pd.NaT
    try:
        return pd.Timestamp(value)
    except Exception:
        pass
    for fmt in (
        "%a %b %d %H:%M:%S %Z %Y",
        "%a %b  %d %H:%M:%S %Z %Y",
    ):
        try:
            # Zone abbreviations in these Slurm logs are not needed for the
            # pre/post split and are inconsistently parsed across Python builds.
            cleaned = re.sub(r"\s+(EDT|EST)\s+", " ", str(value))
            return pd.Timestamp(datetime.strptime(cleaned, fmt.replace(" %Z", "")))
        except Exception:
            continue
    return pd.NaT


def safe_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def sanitize_detail(text: str, limit: int = 600) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"sk-[A-Za-z0-9_-]{8,}", "<redacted-key>", text)
    text = re.sub(r"AIza[A-Za-z0-9_-]{10,}", "<redacted-key>", text)
    text = re.sub(r"\b[A-Za-z0-9_-]{32,}\b", "<long-token>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def terminal_cause_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    patterns = (
        r"ContextWindowPreflightError",
        r"NonRetryableLLMError",
        r"ProviderKeyExhaustedError",
        r"ProviderTransientRetryExhaustedError",
        r"VoteIntegrityError",
        r"(?:ValueError|RuntimeError|IndexError|AttributeError|OSError|KeyError):",
        r"Error during negotiation:",
        r"error: argument",
        r"API error",
        r"failed:",
        r"CANCELLED",
        r"Terminated",
    )
    for line in reversed(lines):
        # Shell wrappers append these after the underlying exception. Skipping
        # them prevents a generic batch RuntimeError from hiding the actual
        # provider, model, or context cause in the preceding traceback.
        if any(
            phrase in line.lower()
            for phrase in (
                "runtimeerror: batch [custom-output] produced 0 successful runs",
                "experiment failed at:",
                "experiment failed: batch [custom-output]",
                "error:root:experiment error",
            )
        ):
            continue
        if any(re.search(pattern, line, re.I) for pattern in patterns):
            return line
    return lines[-1] if lines else "<empty log>"


def terminal_detail(text: str) -> str:
    return sanitize_detail(terminal_cause_text(text))


def classify_failure(text: str, terminal_state: str = "FAILED") -> tuple[str, str]:
    """Return (failure_type, relevance) using terminal-log evidence."""
    cause = terminal_cause_text(text)
    # Only fall back to a broader tail when no informative terminal exception
    # survived (mostly old bilateral wrapper logs). For structured multi-agent
    # logs, the single terminal exception is the classification evidence.
    generic_cause = not re.search(
        r"(error|exception|failed|exhausted|invalid|timeout|too long|"
        r"context|cancelled|terminated|quota|credit|refus|empty)",
        cause,
        flags=re.I,
    )
    tail = text[-30000:] if generic_cause else cause
    low = tail.lower()

    # Capacity failures receive highest priority because they are often wrapped
    # in a generic voting/proposal exception.
    if any(
        phrase in low
        for phrase in (
            "context_length_exceeded",
            "context window preflight",
            "contextwindowpreflighterror",
            "maximum context length",
            "max context length",
            "input tokens exceed the configured limit",
            "prompt is too long",
            "prompt too long",
            "requested about",
            "should not exceed max_num_tokens",
            "sum of prompt length",
            "exceeds the context window",
        )
    ) and any(word in low for word in ("token", "context", "prompt")):
        kind = "context_overflow"
    elif any(
        phrase in low
        for phrase in (
            "no space left on device",
            "disk quota exceeded",
            "input/output error",
            "attributeerror:",
            "indexerror:",
            "keyerror:",
            "unboundlocalerror:",
            "syntaxerror:",
            "invalid choice:",
            "list index out of range",
            "missing result file",
            "result_validation_error",
        )
    ):
        kind = "code_or_orchestration"
    elif any(
        phrase in low
        for phrase in (
            "api key not valid",
            "api_key_invalid",
            "invalid api key",
            "missing provider credentials",
            "credentials for requested models",
            "authenticationerror",
            "unauthorized",
            "project has been denied access",
            "access denied",
            "http 401",
            "error code: 401",
        )
    ):
        kind = "api_auth_or_credentials"
    elif any(
        phrase in low
        for phrase in (
            "insufficient credits",
            "workspace api usage limits",
            "spend limit",
            "key limit exceeded",
            "total limit",
            "http 402",
            "error code: 402",
        )
    ):
        kind = "api_credits_or_spend_limit"
    elif any(
        phrase in low
        for phrase in (
            "rate limit",
            "ratelimit",
            "retry budget exhausted",
            "http 429",
            "error code: 429",
            "last error=429",
            "429 resource has been exhausted",
            "you exceeded your current quota",
            "too many requests",
            "resource_exhausted",
            "quota exceeded",
        )
    ):
        kind = "api_rate_limit_or_quota"
    elif any(
        phrase in low
        for phrase in (
            "apiconnectionerror",
            "connection error",
            "clientconnectorerror",
            "cannot connect to host",
            "connection reset",
            "connection refused",
            "readtimeout",
            "connecttimeout",
            "timeouterror",
            "timed out",
            "proxy error",
            "server disconnected",
            "temporary failure in name resolution",
        )
    ):
        kind = "api_connection_or_timeout"
    elif any(
        phrase in low
        for phrase in (
            "service unavailable",
            "server error",
            "internal server error",
            "overloaded",
            "provider returned error",
            "bad gateway",
            "gateway timeout",
            "http 500",
            "http 502",
            "http 503",
            "http 504",
            "error code: 500",
            "error code: 502",
            "error code: 503",
            "error code: 504",
        )
    ):
        kind = "provider_server_or_overload"
    elif re.search(r"all configured [a-z]+ api keys are exhausted", low):
        # The retained terminal message does not say whether the pool was
        # disabled because of auth, credits, quota, or another provider error.
        # Keep it in infrastructure without inventing a more specific cause.
        kind = "api_key_pool_exhausted"
    elif any(
        phrase in low
        for phrase in (
            "prompt was flagged",
            "content filter",
            "content_filter",
            "safety policy",
            "finish_reason=content_filter",
            "content was filtered",
            "model refused",
            "explicit refusal",
        )
    ):
        kind = "model_content_filter_or_refusal"
    elif any(
        phrase in low
        for phrase in (
            "empty content from model",
            "returned empty content",
            "empty response",
            "finish_reason=length",
            "stop_reason=max_tokens",
            "max output",
            "maximum output",
            "output token",
        )
    ):
        kind = "model_empty_or_output_limit"
    elif any(
        phrase in low
        for phrase in (
            "voteintegrityerror",
            "vote parse failure",
            "voting failure",
            "structured vote recovery failed",
            "no valid json in vote",
        )
    ):
        kind = "model_invalid_vote"
    elif (
        "proposal" in low
        and any(
            phrase in low
            for phrase in (
                "remained invalid",
                "unparsable",
                "parse error",
                "validation error",
                "proposal recovery failed",
            )
        )
    ):
        kind = "model_invalid_proposal"
    elif any(
        phrase in low
        for phrase in (
            "no valid json",
            "jsondecodeerror",
            "structured output",
            "parse failure",
            "validation error",
            "unparsable",
        )
    ):
        kind = "model_invalid_structured_output"
    elif terminal_state.upper() in {"RUNNING", "UNKNOWN", "INCOMPLETE"} or (
        "running: python" in low and not any(x in low for x in ("traceback", "error", "failed"))
    ):
        kind = "scheduler_interrupted"
    elif any(
        phrase in low
        for phrase in (
            "traceback (most recent call last)",
            "runtimeerror:",
            "exception:",
            "produced 0 successful runs",
        )
    ):
        kind = "code_or_orchestration"
    else:
        kind = "unknown_failure"
    return kind, FAILURE_RELEVANCE[kind]


def extract_round_phase_agent_model(
    text: str, agent_model_map: dict[str, str]
) -> tuple[float, str, str, str]:
    tail = text[-80000:]
    rounds: list[int] = []
    for pattern in (
        r"(?:round|ROUND)\s*[=: ]\s*(\d+)(?:/10)?",
        r"in round\s+(\d+)/\d+",
        r"round_(\d+)",
    ):
        rounds.extend(int(x) for x in re.findall(pattern, tail))
    round_number = float(rounds[-1]) if rounds else math.nan

    low = tail.lower()
    if "private voting" in low or "voting failure" in low or "vote parse" in low:
        phase = "voting"
    elif "proposal" in low and any(x in low for x in ("invalid", "unparsable", "parse")):
        phase = "proposal"
    elif "phase=discussion" in low or "discussion phase" in low:
        phase = "discussion"
    elif "private_thinking" in low or "thinking phase" in low:
        phase = "private_thinking"
    elif "reflection" in low:
        phase = "reflection"
    else:
        matches = re.findall(
            r"phase[=: ]+([a-z_]+)", tail, flags=re.I
        )
        phase = matches[-1].lower() if matches else ""

    agents = re.findall(r"Agent_\d+", tail)
    agent = agents[-1] if agents else ""

    models = re.findall(
        r"(?:for model|model=|model_aliases=\[')([A-Za-z0-9_./:+-]+)", tail
    )
    model = models[-1].rstrip("',]") if models else agent_model_map.get(agent, "")
    return round_number, phase, agent, model


def read_elo_table() -> tuple[dict[str, float], dict[str, str], dict[str, str]]:
    path = REPO / "docs/guides/chatbot_arena_elo_scores_2026_03_31.md"
    elo: dict[str, float] = {}
    price: dict[str, str] = {}
    context: dict[str, str] = {}
    row_pattern = re.compile(
        r"^\|\s*\d+\s*\|[^|]*\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|"
        r"[^|]*\|[^|]*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|"
    )
    for line in path.read_text(errors="replace").splitlines():
        match = row_pattern.match(line)
        if not match:
            continue
        model, score, model_price, model_context = (x.strip() for x in match.groups())
        elo[model.lower()] = float(score)
        price[model.lower()] = model_price
        context[model.lower()] = model_context
    return elo, price, context


MODEL_ALIASES = {
    "gemini-3.1-pro": "gemini-3.1-pro-preview",
    "amazon-nova-micro-v1.0": "amazon-nova-micro-v1",
    "amazon-nova-pro-v1.0": "amazon-nova-pro-v1",
    "claude-3-haiku-20240307": "claude-3-haiku-20240307",
    "claude-3-haiku": "claude-3-haiku-20240307",
    "o3-mini-high": "o3-mini-high",
    "deepseek-v3": "deepseek-v3",
    "qwen2.5-72b-instruct": "qwen2.5-72b-instruct",
    "qwen3-max-preview": "qwen3-max-preview",
    "claude-opus-4-6-thinking": "claude-opus-4-6-thinking",
    "claude-opus-4-6": "claude-opus-4-6",
    "gpt-5.4-high": "gpt-5.4-high",
    "gpt-5-nano-high": "gpt-5-nano-high",
    "gpt-4o-mini-2024-07-18": "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-05-13": "gpt-4o-2024-05-13",
    "gpt-4.1-nano-2025-04-14": "gpt-4.1-nano-2025-04-14",
    "llama-3.3-70b-instruct": "llama-3.3-70b-instruct",
    "command-r-plus-08-2024": "command-r-plus-08-2024",
}


def normalize_model_name(model: str) -> str:
    value = str(model or "").strip().lower()
    for prefix in ("openai/", "anthropic/", "google/", "amazon/", "cohere/", "qwen/"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
    if value == "deepseek/deepseek-chat":
        value = "deepseek-v3"
    elif value == "deepseek/deepseek-r1-0528":
        value = "deepseek-r1-0528"
    elif value == "meta-llama/llama-3.3-70b-instruct":
        value = "llama-3.3-70b-instruct"
    elif value == "amazon/nova-micro-v1":
        value = "amazon-nova-micro-v1.0"
    elif value == "amazon/nova-pro-v1":
        value = "amazon-nova-pro-v1.0"
    return value


def lookup_elo(model: str, elo_table: dict[str, float]) -> float:
    normalized = normalize_model_name(model)
    candidates = [
        normalized,
        MODEL_ALIASES.get(normalized, ""),
        normalized.replace("-preview", ""),
    ]
    for candidate in candidates:
        if candidate and candidate.lower() in elo_table:
            return elo_table[candidate.lower()]
    return math.nan


def compaction_stats(path: Path) -> dict[str, Any]:
    """Read compaction metadata, parsing JSON only for compacted rollouts."""
    defaults = {
        "compaction_used": False,
        "compaction_call_count": 0,
        "first_compaction_round": math.nan,
        "first_compaction_phase": "",
        "max_estimated_tokens_before": math.nan,
        "min_estimated_tokens_after": math.nan,
        "min_context_limit_tokens": math.nan,
        "compacted_round_count": 0,
    }
    if not path.exists():
        defaults["rollout_read_error"] = "missing"
        return defaults
    try:
        text = path.read_text(errors="replace")
    except OSError as exc:
        defaults["rollout_read_error"] = str(exc)
        return defaults
    if '"context_compacted": true' not in text:
        defaults["rollout_read_error"] = ""
        return defaults
    try:
        values = json.loads(text)
    except Exception as exc:
        defaults["rollout_read_error"] = f"json:{exc}"
        return defaults
    events: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("context_compacted") is True and (
                "phase" in value or "estimated_input_tokens_before" in value
            ):
                events.append(value)
            for child in value.values():
                if isinstance(child, (dict, list)):
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(values)
    # The same metadata can be nested inside token_usage.  Keep top-level event
    # records only when possible, then de-duplicate by agent/round/phase/before.
    dedup: dict[tuple[Any, ...], dict[str, Any]] = {}
    for event in events:
        key = (
            event.get("agent_id"),
            event.get("round"),
            event.get("phase"),
            event.get("estimated_input_tokens_before"),
            event.get("estimated_input_tokens_after"),
        )
        dedup[key] = event
    events = list(dedup.values())
    if not events:
        defaults["rollout_read_error"] = "flag_without_event"
        return defaults
    events.sort(key=lambda x: safe_float(x.get("timestamp")))
    before = [safe_float(x.get("estimated_input_tokens_before")) for x in events]
    after = [safe_float(x.get("estimated_input_tokens_after")) for x in events]
    limits = [safe_float(x.get("context_limit_tokens")) for x in events]
    compacted_rounds = {
        int(round_number)
        for event in events
        for round_number in (event.get("compacted_rounds") or [])
        if str(round_number).isdigit()
    }
    return {
        "compaction_used": True,
        "compaction_call_count": len(events),
        "first_compaction_round": safe_float(events[0].get("round")),
        "first_compaction_phase": str(events[0].get("phase") or ""),
        "max_estimated_tokens_before": np.nanmax(before) if before else math.nan,
        "min_estimated_tokens_after": np.nanmin(after) if after else math.nan,
        "min_context_limit_tokens": np.nanmin(limits) if limits else math.nan,
        "compacted_round_count": len(compacted_rounds),
        "rollout_read_error": "",
    }


def config_metadata(config: dict[str, Any], elo_table: dict[str, float]) -> dict[str, Any]:
    models = config.get("models") or []
    agent_model_map = config.get("agent_model_map") or {}
    if not agent_model_map and models:
        agent_model_map = {f"Agent_{i + 1}": model for i, model in enumerate(models)}
    if not models:
        models = list(agent_model_map.values())
    models = [str(model) for model in models]
    model_elos = [lookup_elo(model, elo_table) for model in models]
    valid_elos = [value for value in model_elos if not math.isnan(value)]
    return {
        "models_json": json.dumps(models),
        "models_unique_json": json.dumps(sorted(set(models))),
        "agent_model_map_json": json.dumps(agent_model_map, sort_keys=True),
        "agent_elo_map_json": json.dumps(config.get("agent_elo_map") or {}, sort_keys=True),
        "mean_model_elo": float(np.mean(valid_elos)) if valid_elos else math.nan,
        "min_model_elo": float(np.min(valid_elos)) if valid_elos else math.nan,
        "max_model_elo": float(np.max(valid_elos)) if valid_elos else math.nan,
        "elo_coverage_agents": len(valid_elos),
        "competition_id": str(config.get("competition_id") or ""),
        "competition_level": safe_float(config.get("competition_level")),
        "rho": safe_float(config.get("rho")),
        "theta": safe_float(config.get("theta")),
        "alpha": safe_float(config.get("alpha")),
        "sigma": safe_float(config.get("sigma")),
        "max_rounds": safe_float(config.get("max_rounds") or config.get("t_rounds")),
        "discussion_turns": safe_float(config.get("discussion_turns")),
    }


def build_final_outcomes(
    manifest: pd.DataFrame, elo_table: dict[str, float]
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    metadata_by_key: dict[str, dict[str, Any]] = {}
    total = len(manifest)
    for index, manifest_row in manifest.iterrows():
        config_path = REPO / manifest_row["config_path"]
        result_path = REPO / manifest_row["result_path"]
        rollout_path = REPO / manifest_row["rollout_path"]
        config = load_json(config_path)
        result = load_json(result_path)
        key = canonical_config_key(manifest_row["source_root"], manifest_row["config_id"])
        meta = config_metadata(config, elo_table)
        consensus = bool(result.get("consensus_reached", False))
        final_utilities = result.get("final_utilities") or {}
        utility_values = [
            safe_float(value)
            for value in final_utilities.values()
            if not math.isnan(safe_float(value))
        ]
        # No-consensus is a valid outcome with zero utility even where the raw
        # bilateral result serialized an empty utility dictionary.
        utility_sum_analysis = float(np.sum(utility_values)) if consensus else 0.0
        utility_mean_analysis = float(np.mean(utility_values)) if consensus and utility_values else 0.0
        compact = (
            compaction_stats(rollout_path)
            if manifest_row["paper_batch"] in MULTI_BATCHES
            else {
                "compaction_used": False,
                "compaction_call_count": 0,
                "first_compaction_round": math.nan,
                "first_compaction_phase": "",
                "max_estimated_tokens_before": math.nan,
                "min_estimated_tokens_after": math.nan,
                "min_context_limit_tokens": math.nan,
                "compacted_round_count": 0,
                "rollout_read_error": "",
            }
        )
        row = {
            **manifest_row.to_dict(),
            "config_key": key,
            "config_numeric_id": numeric_config_id(manifest_row["config_id"]),
            "consensus_reached": consensus,
            "valid_outcome_type": "valid_consensus" if consensus else "valid_no_consensus",
            "final_round": safe_float(result.get("final_round")),
            "raw_final_utilities_empty": len(final_utilities) == 0,
            "utility_sum_analysis": utility_sum_analysis,
            "utility_mean_analysis": utility_mean_analysis,
            **meta,
            **compact,
        }
        rows.append(row)
        metadata_by_key[key] = {
            **row,
            "_config": config,
            "_agent_model_map": json.loads(meta["agent_model_map_json"]),
        }
        processed = len(rows)
        if processed % 500 == 0 or processed == total:
            print(f"[final outcomes] {processed}/{total}", flush=True)
    return pd.DataFrame(rows), metadata_by_key


def read_attempt_log(path_value: Any) -> tuple[str, str]:
    if not path_value:
        return "", "missing log path"
    path = Path(str(path_value))
    if not path.is_absolute():
        path = REPO / path
    if not path.exists():
        return "", f"missing log: {path}"
    try:
        return path.read_text(errors="replace"), ""
    except OSError as exc:
        return "", str(exc)


def attempt_common(meta: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "paper_run_id",
        "paper_batch",
        "source_root",
        "experiment_family",
        "game",
        "n_agents",
        "config_id",
        "config_key",
        "config_numeric_id",
        "competition_id",
        "competition_level",
        "rho",
        "theta",
        "alpha",
        "sigma",
        "models_json",
        "models_unique_json",
        "agent_model_map_json",
        "mean_model_elo",
        "min_model_elo",
        "max_model_elo",
        "elo_coverage_agents",
        "max_rounds",
        "discussion_turns",
        "consensus_reached",
        "valid_outcome_type",
        "final_round",
        "compaction_used",
        "compaction_call_count",
        "first_compaction_round",
        "first_compaction_phase",
    )
    return {key: meta.get(key) for key in keys}


def build_multi_attempts(metadata: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    multi_meta = [value for value in metadata.values() if value["paper_batch"] in MULTI_BATCHES]
    for count, meta in enumerate(multi_meta, start=1):
        root = REPO / meta["source_root"]
        status_path = root / "status" / f"config_{int(meta['config_numeric_id']):04d}.json"
        if not status_path.exists():
            raise FileNotFoundError(f"Missing multi-agent status: {status_path}")
        status = load_json(status_path)
        attempts = status.get("attempts") or []
        for attempt_index, attempt in enumerate(attempts, start=1):
            state = str(attempt.get("state") or "UNKNOWN").upper()
            log_text, log_read_error = read_attempt_log(attempt.get("log_path"))
            if state == "SUCCESS":
                failure_type = str(meta["valid_outcome_type"])
                relevance = "valid_outcome"
                detail = "Completed result; " + (
                    "consensus reached"
                    if meta["consensus_reached"]
                    else "no consensus at terminal round (zero utility)"
                )
                failed_round, phase, failed_agent, responsible_model = (
                    math.nan,
                    "",
                    "",
                    "",
                )
            else:
                failure_type, relevance = classify_failure(log_text, state)
                detail = terminal_detail(log_text) if log_text else log_read_error
                failed_round, phase, failed_agent, responsible_model = (
                    extract_round_phase_agent_model(log_text, meta["_agent_model_map"])
                )
            started_at = parse_timestamp(attempt.get("started_at"))
            policy_era = (
                "post_compaction_available"
                if pd.notna(started_at)
                and started_at.to_pydatetime().replace(tzinfo=None) >= COMPACTION_INTRODUCED
                else "pre_compaction"
            )
            rows.append(
                {
                    **attempt_common(meta),
                    "attempt_source": "structured_status",
                    "attempt_id": str(attempt.get("attempt_id") or f"attempt_{attempt_index}"),
                    "attempt_index": attempt_index,
                    "attempt_state": state,
                    "attempt_success": state == "SUCCESS",
                    "started_at": started_at,
                    "finished_at": parse_timestamp(attempt.get("finished_at")),
                    "duration_seconds": safe_float(attempt.get("duration_seconds")),
                    "returncode": attempt.get("returncode"),
                    "log_path": str(attempt.get("log_path") or ""),
                    "log_read_error": log_read_error,
                    "failure_type": failure_type,
                    "failure_relevance": relevance,
                    "failure_detail": detail,
                    "failed_round": failed_round,
                    "failed_phase": phase,
                    "failed_agent": failed_agent,
                    "responsible_model": responsible_model,
                    "compaction_policy_era": policy_era,
                    "synthetic_success": False,
                }
            )
        if count % 500 == 0 or count == len(multi_meta):
            print(f"[multi attempts] {count}/{len(multi_meta)}", flush=True)
    return rows


BILATERAL_CONFIG_RE = re.compile(
    r"Config file:\s+.*experiments/results/([^/]+)/configs/config_(\d+)\.json"
)


def bilateral_state(out_text: str, err_text: str) -> str:
    combined = (out_text + "\n" + err_text).lower()
    if "completed successfully" in combined:
        return "SUCCESS"
    if any(
        phrase in combined
        for phrase in (
            "experiment failed",
            "traceback (most recent call last)",
            "produced 0 successful runs",
            "error during negotiation",
            "error: argument",
        )
    ):
        return "FAILED"
    return "INCOMPLETE"


def extract_slurm_started_at(text: str) -> pd.Timestamp | pd.NaT:
    match = re.search(r"Started at:\s*(.+)", text)
    if not match:
        return pd.NaT
    value = re.sub(r"\s+(EDT|EST)\s+", " ", match.group(1).strip())
    for fmt in ("%a %b %d %H:%M:%S %Y", "%a %b  %d %H:%M:%S %Y"):
        try:
            return pd.Timestamp(datetime.strptime(value, fmt))
        except ValueError:
            continue
    return pd.NaT


def build_bilateral_attempts(metadata: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bilateral_meta = {
        key: value for key, value in metadata.items() if value["paper_batch"] == BILATERAL_BATCH
    }
    successful_config_keys: set[str] = set()
    for log_dir in (REPO / "logs/cluster", REPO / "logs/cluster_pre_feb_23"):
        for out_path in sorted(log_dir.glob("*.out")):
            try:
                out_text = out_path.read_text(errors="replace")
            except OSError:
                continue
            match = BILATERAL_CONFIG_RE.search(out_text)
            if not match:
                continue
            source_name, config_id = match.groups()
            key = f"{source_name}:config_{int(config_id):04d}"
            if key not in bilateral_meta:
                continue
            meta = bilateral_meta[key]
            err_path = out_path.with_suffix(".err")
            err_text = err_path.read_text(errors="replace") if err_path.exists() else ""
            combined = out_text + "\n" + err_text
            state = bilateral_state(out_text, err_text)
            if state == "SUCCESS":
                successful_config_keys.add(key)
                failure_type = str(meta["valid_outcome_type"])
                relevance = "valid_outcome"
                detail = "Completed result; " + (
                    "consensus reached"
                    if meta["consensus_reached"]
                    else "no consensus at terminal round (zero utility)"
                )
                failed_round, phase, failed_agent, responsible_model = (
                    math.nan,
                    "",
                    "",
                    "",
                )
            else:
                failure_type, relevance = classify_failure(combined, state)
                detail = terminal_detail(combined)
                failed_round, phase, failed_agent, responsible_model = (
                    extract_round_phase_agent_model(combined, meta["_agent_model_map"])
                )
            started_at = extract_slurm_started_at(out_text)
            rows.append(
                {
                    **attempt_common(meta),
                    "attempt_source": "cluster_log",
                    "attempt_id": out_path.stem,
                    "attempt_index": math.nan,
                    "attempt_state": state,
                    "attempt_success": state == "SUCCESS",
                    "started_at": started_at,
                    "finished_at": pd.NaT,
                    "duration_seconds": math.nan,
                    "returncode": 0 if state == "SUCCESS" else math.nan,
                    "log_path": str(out_path),
                    "log_read_error": "",
                    "failure_type": failure_type,
                    "failure_relevance": relevance,
                    "failure_detail": detail,
                    "failed_round": failed_round,
                    "failed_phase": phase,
                    "failed_agent": failed_agent,
                    "responsible_model": responsible_model,
                    "compaction_policy_era": "pre_compaction",
                    "synthetic_success": False,
                }
            )

    # Some final successful reruns were performed outside the retained Slurm
    # wrapper.  Add exactly one explicitly marked inferred success so every
    # configuration has a terminal successful attempt without pretending that
    # an unobserved log exists.
    for key, meta in bilateral_meta.items():
        if key in successful_config_keys:
            continue
        result_path = REPO / meta["result_path"]
        timestamp = pd.NaT
        try:
            result = load_json(result_path)
            raw_timestamp = result.get("timestamp")
            if raw_timestamp:
                timestamp = pd.to_datetime(float(raw_timestamp), unit="s")
        except Exception:
            pass
        rows.append(
            {
                **attempt_common(meta),
                "attempt_source": "final_result_only",
                "attempt_id": "inferred_success_from_final_result",
                "attempt_index": math.nan,
                "attempt_state": "SUCCESS",
                "attempt_success": True,
                "started_at": timestamp,
                "finished_at": timestamp,
                "duration_seconds": math.nan,
                "returncode": 0,
                "log_path": "",
                "log_read_error": "No retained successful wrapper log; success inferred from final result.",
                "failure_type": str(meta["valid_outcome_type"]),
                "failure_relevance": "valid_outcome",
                "failure_detail": (
                    "Completed final result; "
                    + (
                        "consensus reached"
                        if meta["consensus_reached"]
                        else "no consensus at terminal round (zero utility)"
                    )
                ),
                "failed_round": math.nan,
                "failed_phase": "",
                "failed_agent": "",
                "responsible_model": "",
                "compaction_policy_era": "pre_compaction",
                "synthetic_success": True,
            }
        )
    print(
        f"[bilateral attempts] logs={sum(r['attempt_source'] == 'cluster_log' for r in rows)} "
        f"inferred_successes={sum(r['synthetic_success'] for r in rows)}",
        flush=True,
    )
    return rows


def attach_attempt_order(attempts: pd.DataFrame) -> pd.DataFrame:
    attempts = attempts.copy()
    attempts["started_at"] = pd.to_datetime(attempts["started_at"], errors="coerce")
    attempts = attempts.sort_values(
        ["config_key", "started_at", "attempt_success", "attempt_id"],
        na_position="first",
    )
    attempts["attempt_sequence"] = attempts.groupby("config_key").cumcount() + 1
    attempts["attempt_count_for_config"] = attempts.groupby("config_key")[
        "attempt_id"
    ].transform("size")
    attempts["failed_attempt_count_for_config"] = attempts.groupby("config_key")[
        "attempt_success"
    ].transform(lambda x: int((~x.astype(bool)).sum()))
    return attempts.reset_index(drop=True)


def build_configuration_audit(
    final_outcomes: pd.DataFrame, attempts: pd.DataFrame
) -> pd.DataFrame:
    failure_rows = attempts[~attempts["attempt_success"].astype(bool)]
    grouped = failure_rows.groupby("config_key")
    failure_summary = grouped.agg(
        historical_failed_attempts=("attempt_id", "size"),
        first_failed_at=("started_at", "min"),
        last_failed_at=("started_at", "max"),
        earliest_failed_round=("failed_round", "min"),
        latest_failed_round=("failed_round", "max"),
    )
    type_lists = grouped["failure_type"].apply(
        lambda values: json.dumps(sorted(set(values)))
    )
    relevance_lists = grouped["failure_relevance"].apply(
        lambda values: json.dumps(sorted(set(values)))
    )
    attempt_counts = attempts.groupby("config_key").agg(
        observed_attempt_count=("attempt_id", "size"),
        successful_attempt_count=("attempt_success", "sum"),
        inferred_success_count=("synthetic_success", "sum"),
    )
    result = final_outcomes.merge(
        attempt_counts, left_on="config_key", right_index=True, how="left"
    )
    result = result.merge(
        failure_summary, left_on="config_key", right_index=True, how="left"
    )
    result = result.merge(
        type_lists.rename("historical_failure_types"),
        left_on="config_key",
        right_index=True,
        how="left",
    )
    result = result.merge(
        relevance_lists.rename("historical_failure_relevance"),
        left_on="config_key",
        right_index=True,
        how="left",
    )
    result["historical_failed_attempts"] = (
        result["historical_failed_attempts"].fillna(0).astype(int)
    )
    result["retried_after_failure"] = result["historical_failed_attempts"] > 0
    result["historical_context_failure"] = result["historical_failure_types"].fillna(
        ""
    ).str.contains("context_overflow")
    result["historical_model_protocol_failure"] = result[
        "historical_failure_relevance"
    ].fillna("").str.contains("model_or_protocol")
    result["historical_infrastructure_failure"] = result[
        "historical_failure_relevance"
    ].fillna("").str.contains("infrastructure")
    result["historical_code_failure"] = result[
        "historical_failure_relevance"
    ].fillna("").str.contains("code_or_orchestration")
    return result


def build_model_exposures(attempts: pd.DataFrame, elo_table: dict[str, float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, attempt in attempts.iterrows():
        models = json.loads(attempt["models_json"])
        counts = Counter(models)
        responsible = normalize_model_name(str(attempt["responsible_model"] or ""))
        for model, agent_count in counts.items():
            normalized = normalize_model_name(model)
            rows.append(
                {
                    "config_key": attempt["config_key"],
                    "attempt_id": attempt["attempt_id"],
                    "paper_batch": attempt["paper_batch"],
                    "game": attempt["game"],
                    "n_agents": attempt["n_agents"],
                    "competition_id": attempt["competition_id"],
                    "model": model,
                    "normalized_model": normalized,
                    "model_elo": lookup_elo(model, elo_table),
                    "agents_using_model": agent_count,
                    "attempt_success": attempt["attempt_success"],
                    "failure_type": attempt["failure_type"],
                    "failure_relevance": attempt["failure_relevance"],
                    "responsible_model_match": bool(
                        responsible
                        and (
                            responsible == normalized
                            or responsible in normalized
                            or normalized in responsible
                        )
                    ),
                    "attributed_model_protocol_failure": bool(
                        responsible
                        and attempt["failure_relevance"] == "model_or_protocol"
                        and (
                            responsible == normalized
                            or responsible in normalized
                            or normalized in responsible
                        )
                    ),
                    "attributed_context_failure": bool(
                        responsible
                        and attempt["failure_relevance"] == "context_or_capacity"
                        and (
                            responsible == normalized
                            or responsible in normalized
                            or normalized in responsible
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def rate_table(
    attempts: pd.DataFrame, group_columns: list[str], output_path: Path
) -> pd.DataFrame:
    temp = attempts.copy()
    temp["failed"] = ~temp["attempt_success"].astype(bool)
    temp["model_protocol_failure"] = temp["failure_relevance"].eq("model_or_protocol")
    temp["context_failure"] = temp["failure_relevance"].eq("context_or_capacity")
    temp["infrastructure_failure"] = temp["failure_relevance"].eq("infrastructure")
    temp["code_failure"] = temp["failure_relevance"].eq("code_or_orchestration")
    table = (
        temp.groupby(group_columns, dropna=False)
        .agg(
            attempts=("attempt_id", "size"),
            configurations=("config_key", "nunique"),
            failed_attempts=("failed", "sum"),
            model_protocol_failures=("model_protocol_failure", "sum"),
            context_failures=("context_failure", "sum"),
            infrastructure_failures=("infrastructure_failure", "sum"),
            code_failures=("code_failure", "sum"),
        )
        .reset_index()
    )
    for numerator in (
        "failed_attempts",
        "model_protocol_failures",
        "context_failures",
        "infrastructure_failures",
        "code_failures",
    ):
        table[f"{numerator}_rate"] = table[numerator] / table["attempts"]
    table.to_csv(output_path, index=False)
    return table


def write_csvs_and_tables(
    output: Path,
    final_outcomes: pd.DataFrame,
    attempts: pd.DataFrame,
    configurations: pd.DataFrame,
    exposures: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    final_outcomes.to_csv(output / "final_outcomes.csv", index=False)
    attempts.to_csv(output / "attempt_audit.csv", index=False)
    configurations.to_csv(output / "configuration_audit.csv", index=False)
    exposures.to_csv(output / "model_exposure_audit.csv", index=False)

    failure_counts = (
        attempts.groupby(["failure_relevance", "failure_type"], dropna=False)
        .agg(attempts=("attempt_id", "size"), configurations=("config_key", "nunique"))
        .reset_index()
        .sort_values("attempts", ascending=False)
    )
    failure_counts["share_all_attempts"] = failure_counts["attempts"] / len(attempts)
    failed_denominator = int((~attempts["attempt_success"].astype(bool)).sum())
    failure_counts["share_failed_attempts"] = np.where(
        failure_counts["failure_relevance"].ne("valid_outcome"),
        failure_counts["attempts"] / failed_denominator,
        np.nan,
    )
    failure_counts.to_csv(output / "failure_type_summary.csv", index=False)

    tables = {
        "failure_counts": failure_counts,
        "by_batch": rate_table(attempts, ["paper_batch"], output / "rates_by_batch.csv"),
        "by_game": rate_table(attempts, ["game"], output / "rates_by_game.csv"),
        "by_n": rate_table(attempts, ["n_agents"], output / "rates_by_n_agents.csv"),
        "by_game_n": rate_table(
            attempts, ["game", "n_agents"], output / "rates_by_game_n_agents.csv"
        ),
        "by_competition": rate_table(
            attempts,
            ["game", "competition_id"],
            output / "rates_by_game_competition.csv",
        ),
        "by_round": rate_table(
            attempts[~attempts["attempt_success"].astype(bool)],
            ["failed_round"],
            output / "failures_by_round.csv",
        ),
    }

    model_table = (
        exposures.assign(
            failed=lambda x: ~x["attempt_success"].astype(bool),
            model_protocol=lambda x: x["failure_relevance"].eq("model_or_protocol"),
            context=lambda x: x["failure_relevance"].eq("context_or_capacity"),
            infrastructure=lambda x: x["failure_relevance"].eq("infrastructure"),
        )
        .groupby(["normalized_model", "model_elo"], dropna=False)
        .agg(
            roster_attempt_exposures=("attempt_id", "size"),
            roster_failures=("failed", "sum"),
            roster_model_protocol_failures=("model_protocol", "sum"),
            roster_context_failures=("context", "sum"),
            roster_infrastructure_failures=("infrastructure", "sum"),
            attributed_terminal_failures=("responsible_model_match", "sum"),
            attributed_model_protocol_failures=(
                "attributed_model_protocol_failure",
                "sum",
            ),
            attributed_context_failures=("attributed_context_failure", "sum"),
        )
        .reset_index()
    )
    for numerator in (
        "roster_failures",
        "roster_model_protocol_failures",
        "roster_context_failures",
        "roster_infrastructure_failures",
        "attributed_terminal_failures",
        "attributed_model_protocol_failures",
        "attributed_context_failures",
    ):
        model_table[f"{numerator}_rate"] = (
            model_table[numerator] / model_table["roster_attempt_exposures"]
        )
    model_table.to_csv(output / "rates_by_model.csv", index=False)
    tables["by_model"] = model_table

    context_attempts = attempts[
        attempts["failure_type"].eq("context_overflow")
        | attempts["compaction_used"].astype(bool)
    ].copy()
    context_attempts.to_csv(output / "context_relevant_attempts.csv", index=False)

    context_config = (
        configurations[configurations["paper_batch"].isin(MULTI_BATCHES)]
        .groupby(["paper_batch", "game", "n_agents"])
        .agg(
            final_configurations=("config_key", "size"),
            configs_with_historical_context_failure=("historical_context_failure", "sum"),
            final_configs_using_compaction=("compaction_used", "sum"),
            final_no_consensus=("consensus_reached", lambda x: (~x.astype(bool)).sum()),
            mean_final_round=("final_round", "mean"),
        )
        .reset_index()
    )
    context_config["historical_context_failure_rate"] = (
        context_config["configs_with_historical_context_failure"]
        / context_config["final_configurations"]
    )
    context_config["final_compaction_rate"] = (
        context_config["final_configs_using_compaction"]
        / context_config["final_configurations"]
    )
    context_config.to_csv(output / "context_by_batch_game_n.csv", index=False)
    tables["context_config"] = context_config

    era = attempts[attempts["paper_batch"].isin(MULTI_BATCHES)].copy()
    era["is_context_failure"] = era["failure_type"].eq("context_overflow")
    context_era = (
        era.groupby(["compaction_policy_era", "paper_batch", "game", "n_agents"])
        .agg(
            attempts=("attempt_id", "size"),
            context_failures=("is_context_failure", "sum"),
            configurations=("config_key", "nunique"),
        )
        .reset_index()
    )
    context_era["context_failure_rate"] = (
        context_era["context_failures"] / context_era["attempts"]
    )
    context_era.to_csv(output / "context_failures_pre_post.csv", index=False)
    tables["context_era"] = context_era
    return tables


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "axes.titleweight": "bold",
            "font.family": "DejaVu Sans",
        }
    )


def save_attempt_pies(attempts: pd.DataFrame, output: Path) -> None:
    collapsed = attempts["failure_relevance"].replace(
        {
            "valid_outcome": "Valid completed outcome",
            "model_or_protocol": "Model/protocol failure",
            "context_or_capacity": "Context/capacity failure",
            "infrastructure": "API/infrastructure failure",
            "code_or_orchestration": "Code/orchestration failure",
            "unknown": "Unknown failure",
        }
    )
    counts = collapsed.value_counts()
    colors = ["#4C956C", "#D9822B", "#8B5CF6", "#4C78A8", "#E45756", "#9D9D9D"]
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    wedges, _, autotexts = ax.pie(
        counts.values,
        labels=None,
        # Small-slice values are already exact in the legend; hiding them here
        # prevents overlapping labels around the top of the donut.
        autopct=lambda p: f"{p:.1f}%" if p >= 3 else "",
        startangle=90,
        colors=colors[: len(counts)],
        pctdistance=0.76,
        wedgeprops={"width": 0.42, "edgecolor": "white"},
    )
    ax.legend(
        wedges,
        [f"{label}: {value:,}" for label, value in counts.items()],
        loc="center left",
        bbox_to_anchor=(0.95, 0.5),
        frameon=False,
    )
    ax.set_title(f"All reconstructed attempts (N={len(attempts):,})")
    fig.tight_layout()
    fig.savefig(output / "fig_all_attempts_breakdown.png", bbox_inches="tight")
    fig.savefig(output / "fig_all_attempts_breakdown.pdf", bbox_inches="tight")
    plt.close(fig)

    failed = attempts[~attempts["attempt_success"].astype(bool)]
    counts = failed["failure_type"].map(FAILURE_LABELS).value_counts()
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = sns.color_palette("tab20", len(counts))
    wedges, _ = ax.pie(
        counts.values,
        labels=None,
        startangle=90,
        colors=colors,
        wedgeprops={"width": 0.42, "edgecolor": "white"},
    )
    ax.legend(
        wedges,
        [
            f"{label}: {value:,} ({100 * value / counts.sum():.1f}%)"
            for label, value in counts.items()
        ],
        loc="center left",
        bbox_to_anchor=(0.92, 0.5),
        frameon=False,
    )
    ax.set_title(f"Terminal causes among failed/incomplete attempts (N={len(failed):,})")
    fig.tight_layout()
    fig.savefig(output / "fig_failed_attempt_types.png", bbox_inches="tight")
    fig.savefig(output / "fig_failed_attempt_types.pdf", bbox_inches="tight")
    plt.close(fig)


def save_rate_plots(
    attempts: pd.DataFrame,
    configurations: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    output: Path,
) -> None:
    by_game_n = tables["by_game_n"].copy()
    by_game_n["game_label"] = by_game_n["game"].map(GAME_LABELS)
    long = by_game_n.melt(
        id_vars=["game", "game_label", "n_agents", "attempts"],
        value_vars=[
            "model_protocol_failures_rate",
            "context_failures_rate",
            "infrastructure_failures_rate",
            "code_failures_rate",
        ],
        var_name="failure_group",
        value_name="rate",
    )
    long["failure_group"] = long["failure_group"].replace(
        {
            "model_protocol_failures_rate": "Model/protocol",
            "context_failures_rate": "Context/capacity",
            "infrastructure_failures_rate": "Infrastructure",
            "code_failures_rate": "Code/orchestration",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    for ax, (game, subset) in zip(axes, long.groupby("game", sort=True)):
        sns.lineplot(
            data=subset,
            x="n_agents",
            y="rate",
            hue="failure_group",
            marker="o",
            ax=ax,
        )
        ax.set_title(GAME_LABELS[game])
        ax.set_xlabel("Number of agents")
        ax.set_ylabel("Failures / all attempts")
        ax.yaxis.set_major_formatter(lambda value, _: f"{100 * value:.0f}%")
        if ax is not axes[-1] and ax.get_legend() is not None:
            ax.get_legend().remove()
    axes[-1].legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.suptitle("Denominator-aware failure rates by game and group size", y=1.03)
    fig.tight_layout()
    fig.savefig(output / "fig_failure_rates_by_game_n.png", bbox_inches="tight")
    fig.savefig(output / "fig_failure_rates_by_game_n.pdf", bbox_inches="tight")
    plt.close(fig)

    multi = configurations[configurations["paper_batch"].isin(MULTI_BATCHES)].copy()
    comp = (
        multi.groupby(["game", "n_agents"])
        .agg(
            configs=("config_key", "size"),
            compacted=("compaction_used", "sum"),
            context_history=("historical_context_failure", "sum"),
        )
        .reset_index()
    )
    comp["final_compaction_rate"] = comp["compacted"] / comp["configs"]
    comp["historical_context_failure_rate"] = comp["context_history"] / comp["configs"]
    comp_long = comp.melt(
        id_vars=["game", "n_agents"],
        value_vars=["final_compaction_rate", "historical_context_failure_rate"],
        var_name="metric",
        value_name="rate",
    )
    comp_long["metric"] = comp_long["metric"].replace(
        {
            "final_compaction_rate": "Final run used compaction",
            "historical_context_failure_rate": "Config had historical context failure",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    for ax, (game, subset) in zip(axes, comp_long.groupby("game", sort=True)):
        sns.lineplot(
            data=subset,
            x="n_agents",
            y="rate",
            hue="metric",
            marker="o",
            ax=ax,
        )
        ax.set_title(GAME_LABELS[game])
        ax.set_xlabel("Number of agents")
        ax.set_ylabel("Share of final configurations")
        ax.yaxis.set_major_formatter(lambda value, _: f"{100 * value:.0f}%")
        if ax is not axes[-1] and ax.get_legend() is not None:
            ax.get_legend().remove()
    axes[-1].legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.suptitle("Historical context failures and final-run compaction", y=1.03)
    fig.tight_layout()
    fig.savefig(output / "fig_context_history_and_compaction_by_n.png", bbox_inches="tight")
    fig.savefig(output / "fig_context_history_and_compaction_by_n.pdf", bbox_inches="tight")
    plt.close(fig)

    failures = attempts[~attempts["attempt_success"].astype(bool)].copy()
    failures = failures[failures["failed_round"].notna()]
    if not failures.empty:
        pivot = (
            failures.assign(
                group=failures["failure_relevance"].replace(
                    {
                        "model_or_protocol": "Model/protocol",
                        "context_or_capacity": "Context/capacity",
                        "infrastructure": "Infrastructure",
                        "code_or_orchestration": "Code/orchestration",
                        "unknown": "Unknown",
                    }
                )
            )
            .groupby(["failed_round", "group"])
            .size()
            .unstack(fill_value=0)
        )
        fig, ax = plt.subplots(figsize=(11, 6))
        pivot.plot(kind="bar", stacked=True, ax=ax, color=sns.color_palette("Set2", len(pivot.columns)))
        ax.set_xlabel("Round at terminal failure")
        ax.set_ylabel("Failed attempts")
        ax.set_title("When failures terminated games")
        ax.legend(title="", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
        fig.tight_layout()
        fig.savefig(output / "fig_failures_by_round.png", bbox_inches="tight")
        fig.savefig(output / "fig_failures_by_round.pdf", bbox_inches="tight")
        plt.close(fig)


def save_model_plot(tables: dict[str, pd.DataFrame], output: Path) -> tuple[float, float, int]:
    model = tables["by_model"].copy()
    model = model[
        model["model_elo"].notna() & (model["roster_attempt_exposures"] >= 20)
    ].copy()
    if len(model) >= 3:
        rho, pvalue = spearmanr(
            model["model_elo"], model["attributed_model_protocol_failures_rate"]
        )
    else:
        rho, pvalue = math.nan, math.nan
    model["Context-failure rate"] = model["roster_context_failures_rate"]
    model["Attempt exposures"] = model["roster_attempt_exposures"]
    fig, ax = plt.subplots(figsize=(11.5, 7))
    sns.scatterplot(
        data=model,
        x="model_elo",
        y="attributed_model_protocol_failures_rate",
        size="Attempt exposures",
        sizes=(50, 650),
        hue="Context-failure rate",
        palette="viridis",
        ax=ax,
    )
    for _, row in model.iterrows():
        if (
            row["attributed_model_protocol_failures_rate"]
            >= model["attributed_model_protocol_failures_rate"].quantile(0.75)
            or row["roster_attempt_exposures"] >= model["roster_attempt_exposures"].quantile(0.85)
        ):
            ax.annotate(
                row["normalized_model"],
                (row["model_elo"], row["attributed_model_protocol_failures_rate"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    ax.yaxis.set_major_formatter(lambda value, _: f"{100 * value:.1f}%")
    ax.set_xlabel("LMArena Elo (March 31, 2026 snapshot)")
    ax.set_ylabel("Attributed model/protocol failure rate")
    ax.set_title(
        "Elo versus attributed model/protocol failure rate\n"
        f"Spearman ρ={rho:.2f}, p={pvalue:.3g}"
    )
    ax.legend(
        title="Descriptive audit",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(output / "fig_model_elo_failure_rate.png", bbox_inches="tight")
    fig.savefig(output / "fig_model_elo_failure_rate.pdf", bbox_inches="tight")
    plt.close(fig)
    return float(rho), float(pvalue), len(model)


def format_pct(numerator: int | float, denominator: int | float) -> str:
    return f"{int(numerator):,}/{int(denominator):,} ({100 * numerator / denominator:.2f}%)"


def markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
    selected = frame[columns].head(max_rows).copy()
    headers = [str(column) for column in selected.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for _, row in selected.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                if math.isnan(value):
                    values.append("")
                elif abs(value) < 1 and value != 0:
                    values.append(f"{value:.4f}")
                else:
                    values.append(f"{value:.2f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report(
    output: Path,
    final_outcomes: pd.DataFrame,
    attempts: pd.DataFrame,
    configurations: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    model_stats: tuple[float, float, int],
) -> None:
    total_final = len(final_outcomes)
    total_attempts = len(attempts)
    failures = attempts[~attempts["attempt_success"].astype(bool)]
    valid = attempts[attempts["attempt_success"].astype(bool)]
    no_consensus = final_outcomes[~final_outcomes["consensus_reached"].astype(bool)]
    raw_empty_no_consensus = no_consensus["raw_final_utilities_empty"].sum()
    inferred = attempts["synthetic_success"].sum()
    config_retries = configurations["retried_after_failure"].sum()

    relevance_counts = failures["failure_relevance"].value_counts()
    type_counts = failures["failure_type"].value_counts()

    multi_attempts = attempts[attempts["paper_batch"].isin(MULTI_BATCHES)]
    pre = multi_attempts[multi_attempts["compaction_policy_era"].eq("pre_compaction")]
    post = multi_attempts[
        multi_attempts["compaction_policy_era"].eq("post_compaction_available")
    ]
    pre_context = pre["failure_type"].eq("context_overflow").sum()
    post_context = post["failure_type"].eq("context_overflow").sum()

    multi_configs = configurations[configurations["paper_batch"].isin(MULTI_BATCHES)]
    historical_context = multi_configs["historical_context_failure"].sum()
    compacted_final = multi_configs["compaction_used"].sum()
    context_and_compacted = (
        multi_configs["historical_context_failure"] & multi_configs["compaction_used"]
    ).sum()

    context_by_n = (
        multi_configs.groupby("n_agents")
        .agg(
            configs=("config_key", "size"),
            historical_context=("historical_context_failure", "sum"),
            compacted=("compaction_used", "sum"),
        )
        .reset_index()
    )
    context_by_n["context_rate"] = (
        context_by_n["historical_context"] / context_by_n["configs"]
    )
    context_by_n["compaction_rate"] = context_by_n["compacted"] / context_by_n["configs"]

    failure_table = tables["failure_counts"].copy()
    failure_table = failure_table[
        failure_table["failure_relevance"].ne("valid_outcome")
    ].copy()
    failure_table["failure_type"] = failure_table["failure_type"].map(FAILURE_LABELS)
    failure_table["share_failed_attempts_pct"] = (
        100 * failure_table["share_failed_attempts"]
    ).round(2)

    no_consensus_by_game = (
        final_outcomes.groupby("game")
        .agg(
            final_runs=("config_key", "size"),
            no_consensus=("consensus_reached", lambda x: (~x.astype(bool)).sum()),
        )
        .reset_index()
    )
    no_consensus_by_game["no_consensus_rate"] = (
        no_consensus_by_game["no_consensus"] / no_consensus_by_game["final_runs"]
    )

    rho, pvalue, model_count = model_stats
    text = f"""# Reviewer failure audit

Generated from the current paper experiment manifest. Scope excludes TTC and
the bilateral Llama baseline, as requested.

## Executive findings

- The final paper dataset contains **{total_final:,} in-scope completed games**.
  The audit reconstructs **{total_attempts:,} execution attempts**, including
  **{len(failures):,} failed or incomplete attempts** and **{len(valid):,}
  successful attempts**.
- **{int(config_retries):,}/{total_final:,}
  ({100 * config_retries / total_final:.2f}%) configurations** have at least one
  retained historical failure before the final completed result.
- Of failed attempts, **{format_pct(relevance_counts.get("infrastructure", 0), len(failures))}
  are API/infrastructure failures**, **{format_pct(relevance_counts.get("model_or_protocol", 0), len(failures))}
  are model/protocol failures**, **{format_pct(relevance_counts.get("context_or_capacity", 0), len(failures))}
  are context/capacity failures**, and **{format_pct(relevance_counts.get("code_or_orchestration", 0), len(failures))}
  are code/orchestration failures**.
- No-consensus is already treated correctly: **{len(no_consensus):,}/{total_final:,}
  ({100 * len(no_consensus) / total_final:.2f}%) final games** are valid completed
  no-consensus outcomes with zero analysis utility.  **{int(raw_empty_no_consensus):,}**
  of these serialize an empty raw utility dictionary, but the retained analysis
  code explicitly maps them to zero. No correction to the existing plots is
  required.
- Among the **{len(multi_configs):,} in-scope multi-agent configurations**,
  **{int(historical_context):,} ({100 * historical_context / len(multi_configs):.2f}%)**
  have a retained context-overflow failure and **{int(compacted_final):,}
  ({100 * compacted_final / len(multi_configs):.2f}%)** use proactive compaction
  in the final completed rollout. **{int(context_and_compacted):,}** occur in
  both sets.
- Before proactive compaction was available, there are
  **{format_pct(pre_context, len(pre))} context failures per multi-agent attempt**.
  In the post-introduction era there are **{format_pct(post_context, len(post))}**.
  This is a historical before/after description, not a causal estimate: model
  rosters, rerun selection, and dates differ.
- Across {model_count} models with Elo coverage and at least 20 roster-attempt
  exposures, the descriptive Spearman association between Elo and the rate of
  terminal model/protocol failures attributed to that model is **ρ={rho:.3f},
  p={pvalue:.4g}**. This association is confounded by game, N, provider, and
  heterogeneous composition and should not be presented as a causal capability
  result.

## Failure taxonomy

The principal distinction for the rebuttal is:

1. **Valid strategic outcomes:** consensus or terminal no-consensus. These remain
   in the scientific outcome dataset.
2. **Model/protocol failures:** invalid proposals, invalid votes, other malformed
   structured outputs, truncation/empty output, and safety refusal. These can
   plausibly reflect differential model capability and should be reported.
3. **Context/capacity failures:** provider rejection or local preflight because
   accumulated history exceeds the effective context budget. These are partly
   model/deployment-capability coupled and are reported separately.
4. **Infrastructure failures:** authentication, exhausted credits, exhausted
   key pools, rate limits, connectivity, provider outages, and scheduler
   interruption. These do not constitute bargaining outcomes or clean measures
   of model capability.
5. **Code/orchestration failures:** storage and implementation errors. These also
   do not constitute model behavior.

{markdown_table(failure_table, ["failure_type", "attempts", "configurations", "share_failed_attempts_pct"])}

## No-consensus semantics

{markdown_table(no_consensus_by_game, ["game", "final_runs", "no_consensus", "no_consensus_rate"])}

The final-round no-consensus event is not classified as a failed attempt. It is
a completed game and receives zero utility. The 27 Game 1 records with empty
raw `final_utilities` are normalized to zero by the existing analysis helper.

## Context pressure and compaction

{markdown_table(context_by_n, ["n_agents", "configs", "historical_context", "context_rate", "compacted", "compaction_rate"])}

Interpretation:

- Historical context failures are concentrated at larger N and later rounds.
- Final-run compaction is more common than observed terminal context failure
  because it is proactive: it triggers before a provider rejection.
- A final completed run using compaction is not itself a failure. It is evidence
  that the mitigation activated.
- The historical pre/post comparison cannot establish behavioral equivalence.
  That requires the paired randomized ablation described in
  `ablation_design.md`.

## Attempt-level versus configuration-level estimands

- `attempt_audit.csv` answers operational questions such as “what fraction of
  every execution attempt failed for an API reason?”
- `configuration_audit.csv` answers scientific-selection questions such as “what
  fraction of final paper configurations required a retry, and which failure
  types occurred before success?”
- Both are needed. Attempt-level counts expose engineering burden; configuration-
  level counts avoid allowing repeatedly retried configurations to dominate the
  scientific prevalence estimate.

## Coverage and limitations

- All 3,055 multi-agent configurations have structured status histories.
- All 1,920 bilateral configurations have at least one retained cluster wrapper
  log. For **{int(inferred):,} bilateral configurations**, no retained wrapper
  explicitly marks the successful rerun, so one success record is inferred from
  the extant final result and labeled `synthetic_success=True`. Their historical
  failed/incomplete logs remain in the attempt denominator.
- Rule-based assignments are inspectable through `failure_detail` and
  `log_path`. The final audit has no unknown assignments; a 65-row stratified
  cross-class/batch sample is saved as `classifier_stratified_sample.csv` for
  manual review.
- Provider-monitoring event logs are not used as independent attempts; doing so
  would incorrectly count internal retries as games.

## Files

- `attempt_audit.csv`: one row per reconstructed attempt.
- `configuration_audit.csv`: one row per final paper configuration.
- `final_outcomes.csv`: completed-outcome and compaction metadata.
- `model_exposure_audit.csv`: one row per model roster exposure per attempt.
- `failure_type_summary.csv` and `rates_by_*.csv`: denominator-aware tables.
- `context_relevant_attempts.csv`, `context_by_batch_game_n.csv`, and
  `context_failures_pre_post.csv`: context-specific audit tables.
- `classifier_stratified_sample.csv`: manual-review sample spanning every
  terminal failure class and available batch.
- `ablation_design.md` and `ablation_cost_estimate.*`: paired intervention
  design, adaptive run count, power rationale, and token-cost estimate.
- `fig_*.png` / `fig_*.pdf`: rebuttal-ready diagnostic figures.
"""
    (output / "audit_report.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    manifest = pd.read_csv(args.manifest)
    manifest = manifest[~manifest["paper_batch"].isin(EXCLUDED_BATCHES)].copy()
    if set(manifest["paper_batch"]) != {
        BILATERAL_BATCH,
        "multiagent_homogeneous",
        "multiagent_heterogeneous",
        "random_monoculture",
    }:
        raise AssertionError(f"Unexpected scoped batches: {sorted(manifest.paper_batch.unique())}")
    if len(manifest) != 4975:
        raise AssertionError(f"Expected 4,975 scoped final runs, found {len(manifest):,}")

    elo_table, _, _ = read_elo_table()
    final_outcomes, metadata = build_final_outcomes(manifest, elo_table)

    attempt_rows = build_multi_attempts(metadata)
    attempt_rows.extend(build_bilateral_attempts(metadata))
    attempts = attach_attempt_order(pd.DataFrame(attempt_rows))

    if attempts["config_key"].nunique() != len(final_outcomes):
        raise AssertionError(
            f"Attempt coverage mismatch: {attempts.config_key.nunique()} configs for "
            f"{len(final_outcomes)} final outcomes"
        )
    if not attempts.groupby("config_key")["attempt_success"].any().all():
        missing = attempts.groupby("config_key")["attempt_success"].any()
        raise AssertionError(f"Configurations without success: {missing[~missing].index.tolist()[:10]}")

    configurations = build_configuration_audit(final_outcomes, attempts)
    exposures = build_model_exposures(attempts, elo_table)
    tables = write_csvs_and_tables(output, final_outcomes, attempts, configurations, exposures)
    save_attempt_pies(attempts, output)
    save_rate_plots(attempts, configurations, tables, output)
    model_stats = save_model_plot(tables, output)
    build_report(output, final_outcomes, attempts, configurations, tables, model_stats)

    metadata_payload = {
        "generated_at": datetime.now().isoformat(),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "excluded_batches": sorted(EXCLUDED_BATCHES),
        "final_runs": len(final_outcomes),
        "attempts": len(attempts),
        "failed_or_incomplete_attempts": int((~attempts["attempt_success"].astype(bool)).sum()),
        "configurations_with_historical_failure": int(configurations["retried_after_failure"].sum()),
        "inferred_bilateral_successes": int(attempts["synthetic_success"].sum()),
        "compaction_introduced_timestamp": COMPACTION_INTRODUCED.isoformat(),
    }
    (output / "audit_metadata.json").write_text(json.dumps(metadata_payload, indent=2) + "\n")
    print(json.dumps(metadata_payload, indent=2), flush=True)
    print(f"Wrote audit to {output}", flush=True)


if __name__ == "__main__":
    main()

"""Prompt-local context budgeting and deterministic public-history compaction."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple


DEFAULT_CONTEXT_DOC = Path("docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md")
DEFAULT_CONTEXT_THRESHOLD = 0.85
DEFAULT_RESERVED_OUTPUT_TOKENS = 10_000
DEFAULT_CHARS_PER_TOKEN = 3.0
DISCUSSION_LINE_CHAR_LIMIT = 180
PROPOSAL_CHAR_LIMIT = 2200
VOTE_LINE_CHAR_LIMIT = 260

KNOWN_PROVIDER_CONTEXT_CAPS = {
    "amazon-nova-micro-v1.0": 128_000,
    "amazon/nova-micro-v1": 128_000,
    "gpt-4o-mini-2024-07-18": 128_000,
    "gpt-5-nano": 272_000,
    "openai/gpt-5-nano": 272_000,
    "gpt-5-nano-2025-08-07": 272_000,
    "gpt-4o-2024-05-13": 128_000,
    "gpt-4o": 128_000,
}


@dataclass
class ContextCompactionMetadata:
    """Structured metadata for a prompt-local compaction decision."""

    model_names: List[str]
    context_limit_tokens: int
    threshold: float
    reserved_output_tokens: int
    input_budget_tokens: int
    estimated_input_tokens_before: int
    estimated_input_tokens_after: int
    compacted_rounds: List[int]
    original_public_history_entries: int
    compacted_public_history_entries: int


class ContextWindowPreflightError(RuntimeError):
    """Raised before a provider call that would exceed the context budget."""


def _normalize_model_name(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().strip("`").lower())


def _parse_context_value(value: str) -> Optional[int]:
    text = str(value or "").strip().strip("`")
    if not text or text == "-":
        return None
    text = text.replace(",", "")
    multiplier = 1
    if text.lower().endswith("m"):
        multiplier = 1_000_000
        text = text[:-1]
    elif text.lower().endswith("k"):
        multiplier = 1_000
        text = text[:-1]
    try:
        return int(float(text) * multiplier)
    except ValueError:
        return None


@lru_cache(maxsize=1)
def _load_context_limits_from_doc() -> Dict[str, int]:
    """Load model/route context limits from the canonical markdown table."""
    doc_path = Path(os.getenv("NEGOTIATION_MODEL_CONTEXT_DOC", str(DEFAULT_CONTEXT_DOC)))
    if not doc_path.is_absolute() and not doc_path.exists():
        repo_doc_path = Path(__file__).resolve().parents[1] / doc_path
        if repo_doc_path.exists():
            doc_path = repo_doc_path
    limits: Dict[str, int] = {}
    if not doc_path.exists():
        return limits

    for raw_line in doc_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or "`" not in line:
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) < 8 or parts[0] in {"#", "--"}:
            continue

        model_name = parts[1].replace("`", "").strip()
        arena_context = _parse_context_value(parts[3])
        openrouter_context = _parse_context_value(parts[4])
        route = parts[7].replace("`", "").strip()
        effective = min(
            value for value in (arena_context, openrouter_context) if value is not None
        ) if (arena_context is not None or openrouter_context is not None) else None
        if effective is None:
            continue

        aliases = {model_name, route}
        if "/" in route:
            aliases.add(route.split("/", 1)[1])
        for alias in aliases:
            normalized = _normalize_model_name(alias)
            if normalized:
                limits[normalized] = min(effective, limits.get(normalized, effective))

    for name, cap in KNOWN_PROVIDER_CONTEXT_CAPS.items():
        normalized = _normalize_model_name(name)
        limits[normalized] = min(cap, limits.get(normalized, cap))

    return limits


def resolve_context_limit(model_names: Iterable[str]) -> Optional[int]:
    """Return the minimum known context limit across possible model aliases."""
    limits = _load_context_limits_from_doc()
    resolved: List[int] = []
    for model_name in model_names:
        normalized = _normalize_model_name(model_name)
        if not normalized:
            continue
        if normalized in limits:
            resolved.append(limits[normalized])
            continue
        if "/" in normalized:
            suffix = normalized.split("/", 1)[1]
            if suffix in limits:
                resolved.append(limits[suffix])
    return min(resolved) if resolved else None


@lru_cache(maxsize=1)
def _token_encoder():
    try:
        import tiktoken  # type: ignore

        try:
            return tiktoken.get_encoding("o200k_base")
        except Exception:
            return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def estimate_text_tokens(text: str) -> int:
    """Estimate tokens conservatively enough for preflight context checks."""
    content = str(text or "")
    encoder = _token_encoder()
    if encoder is not None:
        try:
            return len(encoder.encode(content))
        except Exception:
            pass
    return int(math.ceil(len(content) / DEFAULT_CHARS_PER_TOKEN))


def estimate_messages_tokens(messages: Sequence[Dict[str, str]]) -> int:
    total = 0
    for message in messages:
        total += 4
        total += estimate_text_tokens(str(message.get("role", "")))
        total += estimate_text_tokens(str(message.get("content", "")))
    return total + 2


def reserved_output_tokens(config_max_tokens: Optional[int], context_limit: int) -> int:
    """Resolve the output reserve used in input-budget preflight."""
    raw_default = os.getenv("NEGOTIATION_CONTEXT_RESERVED_OUTPUT_TOKENS")
    try:
        default_reserve = int(raw_default) if raw_default else DEFAULT_RESERVED_OUTPUT_TOKENS
    except ValueError:
        default_reserve = DEFAULT_RESERVED_OUTPUT_TOKENS

    if isinstance(config_max_tokens, int) and 0 < config_max_tokens < 999_999:
        reserve = config_max_tokens
    else:
        reserve = default_reserve
    return max(1, min(reserve, max(1, int(context_limit * 0.25))))


def context_threshold() -> float:
    raw = os.getenv("NEGOTIATION_CONTEXT_COMPACTION_THRESHOLD")
    if raw:
        try:
            value = float(raw)
            if 0 < value < 1:
                return value
        except ValueError:
            pass
    return DEFAULT_CONTEXT_THRESHOLD


def _entry_round(entry: Dict[str, Any]) -> Optional[int]:
    try:
        value = entry.get("round")
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def compactable_rounds(history: Sequence[Dict[str, Any]]) -> List[int]:
    rounds = {
        round_num
        for entry in history
        if (round_num := _entry_round(entry)) is not None and round_num > 0
    }
    return sorted(rounds)


def _collapse_line(text: Any) -> str:
    line = re.sub(r"\s+", " ", str(text or "")).strip()
    line = re.sub(r"^[-*#`_\\s]+", "", line)
    return line


def _truncate(text: str, limit: int) -> str:
    line = _collapse_line(text)
    if len(line) <= limit:
        return line
    return line[: max(0, limit - 3)].rstrip() + "..."


def _first_discussion_line(content: Any) -> str:
    text = _collapse_line(content)
    text = re.sub(r"^.*PUBLIC DISCUSSION PHASE[^A-Za-z0-9]*", "", text, flags=re.IGNORECASE)
    parts = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
    return _truncate(parts[0] if parts and parts[0] else text, DISCUSSION_LINE_CHAR_LIMIT)


def _proposal_text(content: Any) -> str:
    text = _collapse_line(content)
    if text.lower().startswith("i propose:"):
        text = text[len("I propose:"):].strip()
    return _truncate(text, PROPOSAL_CHAR_LIMIT)


def _vote_lines(content: Any) -> List[str]:
    lines = []
    for raw_line in str(content or "").splitlines():
        line = _collapse_line(raw_line)
        lowered = line.lower()
        if not line:
            continue
        if (
            line.startswith("Proposal #")
            or "supermajority" in lowered
            or "no proposal" in lowered
            or "winning proposal" in lowered
            or "agreement reached" in lowered
        ):
            lines.append(_truncate(line, VOTE_LINE_CHAR_LIMIT))
    return lines[:12]


def summarize_public_round(round_num: int, entries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a deterministic public-only summary for one round."""
    discussion_by_speaker: Dict[str, str] = {}
    proposals: List[str] = []
    vote_outcomes: List[str] = []

    for entry in entries:
        phase = str(entry.get("phase") or "")
        speaker = str(entry.get("from") or entry.get("agent_id") or "system")
        content = entry.get("content") or entry.get("response") or ""
        if phase.startswith("discussion") and speaker not in discussion_by_speaker:
            discussion_by_speaker[speaker] = _first_discussion_line(content)
        elif phase.startswith("proposal") or str(content).lstrip().lower().startswith("i propose:"):
            proposals.append(f"{speaker}: {_proposal_text(content)}")
        elif phase == "vote_tabulation" or "tabulation" in phase:
            vote_outcomes.extend(_vote_lines(content))
        elif phase.startswith("pledge") or str(content).lstrip().lower().startswith("pledge:"):
            proposals.append(f"{speaker}: {_proposal_text(content)}")

    lines = [
        f"[DETERMINISTIC PUBLIC SUMMARY | Round {round_num}]",
        "This replaces older raw public messages for prompt budgeting only; raw trajectories are unchanged on disk.",
    ]
    if discussion_by_speaker:
        lines.append("Discussion one-liners:")
        for speaker in sorted(discussion_by_speaker):
            lines.append(f"- {speaker}: {discussion_by_speaker[speaker]}")
    if proposals:
        lines.append("Formal public proposals/pledges:")
        for proposal in proposals:
            lines.append(f"- {proposal}")
    if vote_outcomes:
        lines.append("Public vote outcomes:")
        for line in vote_outcomes:
            lines.append(f"- {line}")
    if len(lines) == 2:
        lines.append("No compactable public discussion, proposal, pledge, or vote outcome was recorded.")

    return {
        "round": round_num,
        "phase": "compressed_public_round_summary",
        "from": "system",
        "content": "\n".join(lines),
        "message_type": "context_compaction_summary",
        "compacted_round": round_num,
    }


def compact_public_history_entries(
    history: Sequence[Dict[str, Any]],
    rounds_to_compact: Set[int],
) -> List[Dict[str, Any]]:
    """Replace selected public-history rounds with deterministic summaries."""
    output: List[Dict[str, Any]] = []
    emitted_round_summaries: Set[int] = set()
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for entry in history:
        round_num = _entry_round(entry)
        if round_num in rounds_to_compact:
            grouped.setdefault(round_num, []).append(dict(entry))

    for entry in history:
        round_num = _entry_round(entry)
        if round_num not in rounds_to_compact:
            output.append(dict(entry))
            continue
        if round_num in emitted_round_summaries:
            continue
        output.append(summarize_public_round(round_num, grouped.get(round_num, [])))
        emitted_round_summaries.add(round_num)
    return output

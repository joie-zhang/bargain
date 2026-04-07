from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ELO_MARKDOWN = (
    PROJECT_ROOT
    / "docs"
    / "guides"
    / "chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md"
)


# Canonical April 2026 adversary roster used across Games 1-3 plots.
ACTIVE_ADVERSARY_MODELS: Tuple[str, ...] = (
    "claude-opus-4-6-thinking",
    "claude-opus-4-6",
    "gemini-3-pro",
    "gpt-5.4-high",
    "gpt-5.2-chat-latest-20260210",
    "claude-opus-4-5-20251101-thinking-32k",
    "claude-opus-4-5-20251101",
    "gemini-2.5-pro",
    "qwen3-max-preview",
    "deepseek-r1-0528",
    "claude-haiku-4-5-20251001",
    "deepseek-r1",
    "claude-sonnet-4-20250514",
    "gemma-3-27b-it",
    "o3-mini-high",
    "deepseek-v3",
    "gpt-4o-2024-05-13",
    "qwq-32b",
    "gpt-4.1-nano-2025-04-14",
    "llama-3.3-70b-instruct",
    "gpt-4o-mini-2024-07-18",
    "qwen2.5-72b-instruct",
    "amazon-nova-pro-v1.0",
    "command-r-plus-08-2024",
    "claude-3-haiku-20240307",
    "amazon-nova-micro-v1.0",
    "llama-3.1-8b-instruct",
    "llama-3.2-3b-instruct",
    "llama-3.2-1b-instruct",
    "gpt-5-nano-high",
)
ACTIVE_ADVERSARY_MODEL_SET = frozenset(ACTIVE_ADVERSARY_MODELS)

LEGACY_EXCLUDED_MODELS = frozenset(
    {
        "claude-3-5-sonnet-20241022",
        "phi-3-mini-128k-instruct",
        "qwq-32b-preview",
    }
)

MODEL_ALIASES: Dict[str, str] = {
    "amazon-nova-micro": "amazon-nova-micro-v1.0",
    "claude-haiku-4-5": "claude-haiku-4-5-20251001",
    "claude-opus-4-5": "claude-opus-4-5-20251101",
    "claude-opus-4-5-thinking-32k": "claude-opus-4-5-20251101-thinking-32k",
    "claude-sonnet-4": "claude-sonnet-4-20250514",
    "gpt-4o": "gpt-4o-2024-05-13",
    "Llama-3.2-1B-Instruct": "llama-3.2-1b-instruct",
    "Llama-3.2-3B-Instruct": "llama-3.2-3b-instruct",
    "o3": "o3-mini-high",
    "QwQ-32B": "qwq-32b",
    "Qwen2.5-72B-Instruct": "qwen2.5-72b-instruct",
    "qwen3-max": "qwen3-max-preview",
}

MODEL_SHORT_NAMES: Dict[str, str] = {
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
    "gemini-3-pro": "Gemini 3 Pro",
    "gemma-3-27b-it": "Gemma 3 27B",
    "gpt-4.1-nano-2025-04-14": "GPT-4.1 nano",
    "gpt-4o-2024-05-13": "GPT-4o",
    "gpt-4o-mini-2024-07-18": "GPT-4o mini",
    "gpt-5-nano": "GPT-5-nano",
    "gpt-5-nano-high": "GPT-5-nano",
    "gpt-5.2-chat-latest-20260210": "GPT-5.2 Chat",
    "gpt-5.4-high": "GPT-5.4 High",
    "llama-3.1-8b-instruct": "Llama 3.1 8B",
    "llama-3.2-1b-instruct": "Llama 3.2 1B",
    "llama-3.2-3b-instruct": "Llama 3.2 3B",
    "llama-3.3-70b-instruct": "Llama 3.3 70B",
    "o3-mini-high": "o3-mini-high",
    "qwen2.5-72b-instruct": "Qwen2.5 72B",
    "qwen3-max-preview": "Qwen3 Max",
    "qwq-32b": "QwQ-32B",
}


def canonical_model_name(model_name: str) -> str:
    return MODEL_ALIASES.get(model_name, model_name)


def short_model_name(model_name: str) -> str:
    canonical = canonical_model_name(model_name)
    return MODEL_SHORT_NAMES.get(canonical, canonical)


def is_active_adversary_model(model_name: str) -> bool:
    return canonical_model_name(model_name) in ACTIVE_ADVERSARY_MODEL_SET


def filter_active_adversary_models(model_names: Iterable[str]) -> List[str]:
    filtered: List[str] = []
    seen = set()
    for model_name in model_names:
        canonical = canonical_model_name(model_name)
        if canonical not in ACTIVE_ADVERSARY_MODEL_SET or canonical in seen:
            continue
        seen.add(canonical)
        filtered.append(canonical)
    return filtered


@lru_cache(maxsize=1)
def active_model_elo_map() -> Dict[str, int]:
    table_row = re.compile(r"^\|\s*\d+\s*\|\s*`?([^|`]+?)`?\s*\|\s*(\d+)\s*\|")
    elo_by_model: Dict[str, int] = {}
    for line in DEFAULT_ELO_MARKDOWN.read_text(encoding="utf-8").splitlines():
        match = table_row.match(line.strip())
        if not match:
            continue
        model_name = canonical_model_name(match.group(1).strip())
        if model_name in ACTIVE_ADVERSARY_MODEL_SET:
            elo_by_model[model_name] = int(match.group(2))
    missing = [model for model in ACTIVE_ADVERSARY_MODELS if model not in elo_by_model]
    if missing:
        raise ValueError(
            "Missing Elo entries for active adversary models: "
            + ", ".join(missing)
        )
    return elo_by_model


def elo_for_model(model_name: str) -> int | None:
    canonical = canonical_model_name(model_name)
    return active_model_elo_map().get(canonical)


def tier_from_elo(elo: int | float | None) -> str:
    if elo is None:
        return "Unknown"
    if float(elo) >= 1415:
        return "Strong"
    if float(elo) >= 1290:
        return "Medium"
    return "Weak"


def active_model_info_map() -> Dict[str, Dict[str, object]]:
    return {
        model_name: {
            "elo": elo,
            "tier": tier_from_elo(elo),
            "short_name": short_model_name(model_name),
        }
        for model_name, elo in active_model_elo_map().items()
    }

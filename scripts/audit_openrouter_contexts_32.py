#!/usr/bin/env python3
"""Audit native-vs-OpenRouter context windows for the 32-model slate."""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strong_models_experiment.configs import STRONG_MODELS_CONFIG
from scripts.game2_derisk_32 import FINAL_32_MODELS

CANONICAL_MARKDOWN = REPO_ROOT / "docs" / "guides" / "chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "guides" / "openrouter_context_windows_final32_2026_04_06.md"
OPENROUTER_MODELS_API = "https://openrouter.ai/api/v1/models"
ALIAS_NATIVE_CONTEXT = {
    "qwq-32b-preview": "qwq-32b",
}


def parse_context_tokens(raw_value: str) -> Optional[int]:
    value = raw_value.strip().upper()
    if not value:
        return None
    if value.endswith("K"):
        return int(round(float(value[:-1]) * 1000))
    if value.endswith("M"):
        return int(round(float(value[:-1]) * 1_000_000))
    digits = value.replace(",", "")
    return int(digits) if digits.isdigit() else None


def format_tokens(value: Optional[int]) -> str:
    if value is None:
        return "n/a"
    return f"{value:,}"


def format_delta(native_tokens: Optional[int], openrouter_tokens: Optional[int]) -> str:
    if native_tokens is None or openrouter_tokens is None:
        return "n/a"
    delta = openrouter_tokens - native_tokens
    if delta == 0:
        return "same"
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:,}"


def format_ratio(native_tokens: Optional[int], openrouter_tokens: Optional[int]) -> str:
    if native_tokens in (None, 0) or openrouter_tokens is None:
        return "n/a"
    pct = (openrouter_tokens / native_tokens) * 100.0
    return f"{pct:.1f}%"


def load_canonical_rows() -> Dict[str, Dict[str, str]]:
    rows: Dict[str, Dict[str, str]] = {}
    for line_number, line in enumerate(CANONICAL_MARKDOWN.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.startswith("|"):
            continue
        parts = [part.strip() for part in line.split("|")[1:-1]]
        if len(parts) < 10 or parts[0] == "Rank":
            continue
        model = parts[1].strip("`")
        route = parts[6].strip("`")
        rows[model] = {
            "rank": parts[0],
            "context": parts[3],
            "route": route,
            "notes": parts[9],
            "line_number": str(line_number),
        }
    return rows


def fetch_openrouter_model_index() -> Dict[str, Dict]:
    response = requests.get(OPENROUTER_MODELS_API, timeout=60)
    response.raise_for_status()
    payload = response.json()
    return {item["id"]: item for item in payload["data"]}


def fetch_openrouter_context_from_page(route: str) -> Optional[int]:
    response = requests.get(f"https://openrouter.ai/{route}", timeout=60)
    response.raise_for_status()
    match = re.search(r'(?:\\"|")context_length(?:\\"|"):(\d+)', response.text)
    if not match:
        return None
    return int(match.group(1))


def build_rows() -> List[Dict[str, str]]:
    canonical_rows = load_canonical_rows()
    openrouter_models = fetch_openrouter_model_index()
    rows: List[Dict[str, str]] = []

    for entry in FINAL_32_MODELS:
        if entry["provider"] != "openrouter":
            continue

        model = entry["model"]
        config = STRONG_MODELS_CONFIG[model]
        route = config["model_id"]
        native_key = ALIAS_NATIVE_CONTEXT.get(model, model)
        canonical = canonical_rows.get(native_key)
        if not canonical:
            raise KeyError(f"No canonical row found for {model} (native key {native_key})")

        openrouter_entry = openrouter_models.get(route)
        openrouter_context = openrouter_entry["context_length"] if openrouter_entry else fetch_openrouter_context_from_page(route)
        source_kind = "models_api" if openrouter_entry else "page_html"

        native_context_raw = canonical["context"]
        native_context_tokens = parse_context_tokens(native_context_raw)

        notes: List[str] = []
        if model != native_key:
            notes.append(f"alias of `{native_key}`")
        if model == "qwen3-max-preview":
            notes.append("repo alias to current `qwen/qwen3-max` route")
        if model == "claude-3-5-sonnet-20241022":
            notes.append("dated Anthropic snapshot is retired; repo uses best-effort OpenRouter route")
        if source_kind == "page_html":
            notes.append("not present in current OpenRouter models API; context parsed from route page HTML")

        rows.append(
            {
                "rank": str(entry["rank"]),
                "model": model,
                "route": route,
                "native_context_raw": native_context_raw,
                "native_context_tokens": format_tokens(native_context_tokens),
                "openrouter_context_tokens": format_tokens(openrouter_context),
                "delta": format_delta(native_context_tokens, openrouter_context),
                "ratio": format_ratio(native_context_tokens, openrouter_context),
                "route_url": f"https://openrouter.ai/{route}",
                "canonical_line": canonical["line_number"],
                "source_kind": source_kind,
                "notes": "; ".join(notes) if notes else "",
            }
        )

    return rows


def render_markdown(rows: List[Dict[str, str]]) -> str:
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    shorter_count = sum(
        row["delta"] not in ("same", "n/a") and row["delta"].startswith("-")
        for row in rows
    )
    unavailable_count = sum(row["openrouter_context_tokens"] == "n/a" for row in rows)
    same_count = sum(row["delta"] == "same" for row in rows)

    lines = [
        "# OpenRouter Context Window Audit For The Final 32-Model Slate",
        "",
        f"Generated: {generated_at}",
        "",
        "This compares the repo's canonical native/full context window for each OpenRouter-backed model in the final 32-model slate against the current context window reported by OpenRouter.",
        "",
        "Sources:",
        f"- Native/full context reference: `{CANONICAL_MARKDOWN}`",
        f"- Current OpenRouter model metadata: `{OPENROUTER_MODELS_API}`",
        "- Fallback when a route is missing from the live OpenRouter model index: the public model page at `https://openrouter.ai/<route>`",
        "",
        "Summary:",
        f"- OpenRouter-backed entries in the 32-model slate: {len(rows)}",
        f"- Same context on OpenRouter: {same_count}",
        f"- Shorter context on OpenRouter: {shorter_count}",
        f"- Missing from live OpenRouter models API and resolved from page HTML: {sum(row['source_kind'] == 'page_html' for row in rows)}",
        f"- Missing OpenRouter context even after fallback: {unavailable_count}",
        "",
        "| Rank | Model | OpenRouter Route | Native / Full Context | OpenRouter Context Now | OpenRouter vs Native | OpenRouter / Native | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in rows:
        native_label = f"{row['native_context_raw']} ({row['native_context_tokens']})"
        route_label = f"[`{row['route']}`]({row['route_url']})"
        notes = row["notes"] or ""
        lines.append(
            f"| {row['rank']} | `{row['model']}` | {route_label} | {native_label} | {row['openrouter_context_tokens']} | {row['delta']} | {row['ratio']} | {notes} |"
        )

    lines.extend(
        [
            "",
            "Canonical slate rows consulted:",
        ]
    )
    for row in rows:
        lines.append(
            f"- `{row['model']}`: `{CANONICAL_MARKDOWN}#L{row['canonical_line']}`"
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    rows = build_rows()
    markdown = render_markdown(rows)
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_text(markdown, encoding="utf-8")
    print(DEFAULT_OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

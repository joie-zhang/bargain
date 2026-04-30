"""Deterministic helpers for parsing lightly malformed model JSON.

These utilities intentionally avoid model-assisted rewriting. They only fix
syntax-level issues that are common in structured-output responses and leave
schema validation to the game-specific parsers.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


def json_object_candidates(response: str) -> List[str]:
    """Return plausible JSON-object substrings from model output."""
    text = str(response or "").strip()
    if not text:
        return []

    candidates = [text] if text.startswith("{") else []
    for fence_match in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE):
        candidates.append(fence_match.group(1).strip())

    for start_idx, char in enumerate(text):
        if char != "{":
            continue

        stack: List[str] = []
        in_string = False
        escaped = False
        for end_idx in range(start_idx, len(text)):
            current = text[end_idx]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue

            if current == '"':
                in_string = True
            elif current == "{":
                stack.append("}")
            elif current == "[":
                stack.append("]")
            elif current in ("}", "]"):
                if stack and stack[-1] == current:
                    stack.pop()
                if not stack:
                    candidates.append(text[start_idx:end_idx + 1])
                    break
        else:
            candidates.append(text[start_idx:])

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if candidate and candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)
    return unique_candidates


def _strip_comments_and_markdown(candidate: str) -> str:
    """Strip comments and simple markdown emphasis outside JSON strings."""
    output: List[str] = []
    in_string = False
    escaped = False
    i = 0
    while i < len(candidate):
        char = candidate[i]
        next_char = candidate[i + 1] if i + 1 < len(candidate) else ""

        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            i += 1
            continue

        if char == '"':
            output.append(char)
            in_string = True
            i += 1
            continue

        if char == "/" and next_char == "/":
            i += 2
            while i < len(candidate) and candidate[i] not in "\r\n":
                i += 1
            continue

        if char == "/" and next_char == "*":
            i += 2
            while i + 1 < len(candidate) and not (candidate[i] == "*" and candidate[i + 1] == "/"):
                i += 1
            i = min(len(candidate), i + 2)
            continue

        if char == "*" and next_char == "*":
            i += 2
            continue

        output.append(char)
        i += 1

    return "".join(output)


def _escape_control_chars_and_balance(candidate: str) -> str:
    """Escape control chars inside strings and close obviously-open braces."""
    repaired: List[str] = []
    stack: List[str] = []
    in_string = False
    escaped = False

    for char in candidate.strip():
        if in_string:
            if escaped:
                repaired.append(char)
                escaped = False
            elif char == "\\":
                repaired.append(char)
                escaped = True
            elif char == '"':
                repaired.append(char)
                in_string = False
            elif char == "\n":
                repaired.append("\\n")
            elif char == "\r":
                repaired.append("\\r")
            elif char == "\t":
                repaired.append("\\t")
            elif ord(char) < 0x20:
                repaired.append(f"\\u{ord(char):04x}")
            else:
                repaired.append(char)
            continue

        repaired.append(char)
        if char == '"':
            in_string = True
        elif char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in ("}", "]") and stack and stack[-1] == char:
            stack.pop()

    if in_string:
        repaired.append('"')
    repaired.extend(reversed(stack))
    return "".join(repaired)


def _remove_trailing_commas(candidate: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", candidate)


def _insert_obvious_missing_commas(candidate: str) -> str:
    """Repair common model omission: value newline followed by next key."""
    return re.sub(
        r'([}\]"0-9])(\s*\n\s*)("[-A-Za-z0-9_ ]+"\s*:)',
        r"\1,\2\3",
        candidate,
    )


def repair_json_candidate(candidate: str) -> str:
    """Repair common model JSON issues without changing valid JSON."""
    repaired = _strip_comments_and_markdown(candidate)
    repaired = _escape_control_chars_and_balance(repaired)
    repaired = _insert_obvious_missing_commas(repaired)
    repaired = _remove_trailing_commas(repaired)
    return repaired


def json_repair_attempts(candidate: str) -> List[str]:
    """Return deterministic repair variants in least-to-most invasive order."""
    stripped = _strip_comments_and_markdown(candidate)
    repaired = repair_json_candidate(candidate)
    attempts = [
        candidate.strip(),
        stripped.strip(),
        _escape_control_chars_and_balance(candidate).strip(),
        repaired.strip(),
    ]
    unique_attempts = []
    seen = set()
    for attempt in attempts:
        if attempt and attempt not in seen:
            unique_attempts.append(attempt)
            seen.add(attempt)
    return unique_attempts


def parse_json_object(response: str, label: str = "response") -> Dict[str, Any]:
    """Parse a JSON object from direct, fenced, embedded, or repaired output."""
    first_error: Optional[Exception] = None
    for candidate in json_object_candidates(response):
        for attempt in json_repair_attempts(candidate):
            try:
                payload = json.loads(attempt)
                if isinstance(payload, dict):
                    return payload
                raise ValueError(f"{label} JSON payload was not an object")
            except (json.JSONDecodeError, ValueError) as exc:
                if first_error is None:
                    first_error = exc

    if first_error is not None:
        raise first_error
    raise ValueError(f"No JSON found in {label}")

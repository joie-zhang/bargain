"""Shared API-key rotation and provider failure reporting.

This module keeps provider key handling out of experiment/game logic. It never
logs secret values; reports use only environment-variable labels.
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime as dt
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - non-Unix fallback
    fcntl = None


PROVIDER_ENV_BASE = {
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

DEFAULT_REPORT_PATH = Path("experiments/results/provider_failures.md")
DEFAULT_TRANSIENT_RETRY_SECONDS = 300.0


@dataclasses.dataclass(frozen=True)
class ProviderKey:
    provider: str
    value: str
    label: str
    group: str


class ProviderKeyExhaustedError(RuntimeError):
    """Raised when every configured key for a provider has failed."""


class ProviderTransientRetryExhaustedError(RuntimeError):
    """Raised after the per-call transient retry budget is spent."""


_DISABLED_KEY_LABELS_BY_PROVIDER: Dict[str, set[str]] = {}


def _provider_name(provider: str) -> str:
    provider_name = provider.strip().lower()
    if provider_name not in PROVIDER_ENV_BASE:
        raise ValueError(f"Unsupported provider for key rotation: {provider!r}")
    return provider_name


def _key_sort_tuple(env_name: str, prefix: str) -> tuple[int, str]:
    suffix = env_name.removeprefix(prefix)
    if not suffix:
        return (0, env_name)
    if suffix.startswith("_") and suffix[1:].isdigit():
        return (int(suffix[1:]), env_name)
    return (10_000, env_name)


def _iter_group_key_names(group: str, env_base: str) -> Iterable[str]:
    prefix = f"{group}_{env_base}"
    matches = [
        name
        for name in os.environ
        if name == prefix or name.startswith(f"{prefix}_")
    ]
    yield from sorted(matches, key=lambda name: _key_sort_tuple(name, prefix))


def _group_order() -> List[str]:
    raw = os.getenv("LLM_KEY_GROUP_ORDER", "")
    return [part.strip().upper() for part in raw.split(",") if part.strip()]


def discover_provider_keys(
    provider: str,
    fallback_key: Optional[str] = None,
) -> List[ProviderKey]:
    """Discover ordered keys for one provider.

    Grouped variables are considered only when LLM_KEY_GROUP_ORDER is set.
    Legacy provider variables remain a fallback for older workflows.
    """

    provider_name = _provider_name(provider)
    env_base = PROVIDER_ENV_BASE[provider_name]
    keys: List[ProviderKey] = []
    seen_values: set[str] = set()

    for group in _group_order():
        for env_name in _iter_group_key_names(group, env_base):
            value = os.getenv(env_name)
            if not value or value in seen_values:
                continue
            seen_values.add(value)
            keys.append(
                ProviderKey(
                    provider=provider_name,
                    value=value,
                    label=env_name,
                    group=group,
                )
            )

    legacy_value = fallback_key or os.getenv(env_base)
    if legacy_value and legacy_value not in seen_values:
        keys.append(
            ProviderKey(
                provider=provider_name,
                value=legacy_value,
                label=env_base,
                group="LEGACY",
            )
        )

    return keys


def has_provider_keys(provider: str, fallback_key: Optional[str] = None) -> bool:
    return bool(discover_provider_keys(provider, fallback_key=fallback_key))


def _extract_status_code(exc: BaseException) -> Optional[int]:
    for attr_name in ("status_code", "status", "http_status"):
        value = getattr(exc, attr_name, None)
        if isinstance(value, int):
            return value

    response = getattr(exc, "response", None)
    if response is not None:
        value = getattr(response, "status_code", None) or getattr(response, "status", None)
        if isinstance(value, int):
            return value

    text = _error_text(exc).lower()
    match = re.search(r"\b(?:http\s*)?([45]\d{2})\b", text)
    if match:
        return int(match.group(1))
    return None


def _error_text(exc: BaseException) -> str:
    parts = [type(exc).__name__, str(exc), repr(exc)]
    body = getattr(exc, "body", None)
    if body is not None:
        parts.append(str(body))
    return " ".join(part for part in parts if part)


FUNDS_MARKERS = (
    "insufficient_quota",
    "insufficient quota",
    "insufficient funds",
    "insufficient credits",
    "credit balance",
    "billing",
    "payment required",
    "402",
)

RATE_LIMIT_MARKERS = (
    "429",
    "rate limit",
    "rate_limit",
    "resource_exhausted",
    "resource exhausted",
    "quota exceeded",
    "daily quota",
    "free tier",
    "requests per day",
    "requests per minute",
    "tokens per minute",
    "generate_requests_per_model_per_day",
)

TRANSIENT_MARKERS = (
    "500",
    "502",
    "503",
    "504",
    "internal server error",
    "service unavailable",
    "temporarily unavailable",
    "bad gateway",
    "gateway timeout",
    "connection reset",
    "connection aborted",
    "connection refused",
    "read timeout",
    "timeout",
    "timed out",
)


def classify_key_scoped_failure(
    provider: str,
    exc: BaseException,
) -> Optional[str]:
    """Return a key-scoped failure kind, or None if this is not key-scoped."""

    _provider_name(provider)
    text = _error_text(exc).lower()
    status = _extract_status_code(exc)

    if status == 402 or any(marker in text for marker in FUNDS_MARKERS):
        return "insufficient_funds"
    if status == 429 or any(marker in text for marker in RATE_LIMIT_MARKERS):
        return "rate_limit_or_quota"
    return None


def is_transient_provider_failure(exc: BaseException) -> bool:
    status = _extract_status_code(exc)
    if status is not None and 500 <= status < 600:
        return True
    text = _error_text(exc).lower()
    return any(marker in text for marker in TRANSIENT_MARKERS)


def is_deterministic_provider_failure(exc: BaseException) -> bool:
    """Return True for provider/config failures that should fail fast."""

    status = _extract_status_code(exc)
    if status is not None:
        return 400 <= status < 500 and status not in {402, 408, 409, 429}
    text = _error_text(exc).lower()
    return any(
        marker in text
        for marker in (
            "max_tokens must be greater than thinking.budget_tokens",
            "invalid model",
            "invalid_request_error",
            "bad request",
            "not_found_error",
            "no endpoints found",
        )
    )


def _transient_retry_budget_seconds() -> float:
    raw = os.getenv("LLM_TRANSIENT_RETRY_SECONDS")
    if raw is None or raw == "":
        return DEFAULT_TRANSIENT_RETRY_SECONDS
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_TRANSIENT_RETRY_SECONDS


class ProviderKeyPool:
    """Ordered key pool for one provider.

    Key exhaustion is shared within the current process by provider/key label so
    concurrent agents do not keep retrying a key that already failed.
    """

    def __init__(
        self,
        provider: str,
        fallback_key: Optional[str] = None,
    ):
        self.provider = _provider_name(provider)
        self.keys = discover_provider_keys(self.provider, fallback_key=fallback_key)
        if not self.keys:
            raise ValueError(f"No API keys configured for provider {self.provider}")
        self.index = 0

    def current(self) -> ProviderKey:
        disabled = _DISABLED_KEY_LABELS_BY_PROVIDER.setdefault(self.provider, set())
        while self.index < len(self.keys) and self.keys[self.index].label in disabled:
            self.index += 1
        if self.index >= len(self.keys):
            labels = [key.label for key in self.keys]
            raise ProviderKeyExhaustedError(
                f"All configured {self.provider} API keys are exhausted "
                f"(labels={labels})"
            )
        return self.keys[self.index]

    def rotate_after_failure(self, failed_key: ProviderKey) -> Optional[ProviderKey]:
        disabled = _DISABLED_KEY_LABELS_BY_PROVIDER.setdefault(self.provider, set())
        disabled.add(failed_key.label)
        if self.index < len(self.keys) and self.keys[self.index].label == failed_key.label:
            self.index += 1
        try:
            return self.current()
        except ProviderKeyExhaustedError:
            return None

    def labels(self) -> List[str]:
        return [key.label for key in self.keys]


def report_path() -> Path:
    raw_path = os.getenv("LLM_FAILURE_REPORT_PATH")
    path = Path(raw_path) if raw_path else DEFAULT_REPORT_PATH
    return path.expanduser()


def _jsonl_path(markdown_path: Path) -> Path:
    return markdown_path.with_suffix(".jsonl")


def _lock_path(markdown_path: Path) -> Path:
    return markdown_path.with_suffix(".lock")


def _solution_distance(
    *,
    failure_kind: str,
    exhausted: bool,
    next_key_label: Optional[str],
) -> str:
    if exhausted:
        if failure_kind == "rate_limit_or_quota":
            return "all-keys-exhausted; requeue after provider quota reset"
        if failure_kind == "insufficient_funds":
            return "all-keys-exhausted; add funds or add another key"
        return "all-keys-exhausted; inspect provider error"
    if next_key_label:
        return f"auto-rotated-to-{next_key_label}"
    return "logged"


def _safe_message(exc: BaseException, max_chars: int = 2000) -> str:
    text = str(exc) or repr(exc)
    text = text.replace("\n", "\\n")
    return text[:max_chars]


def _read_events(jsonl_path: Path) -> List[Dict[str, Any]]:
    if not jsonl_path.exists():
        return []
    events = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _render_markdown(events: List[Dict[str, Any]]) -> str:
    generated_at = dt.datetime.now().isoformat(timespec="seconds")
    if not events:
        return (
            "# Provider Failure Report\n\n"
            f"- Generated at: {generated_at}\n"
            "- No provider failures recorded.\n"
        )

    aggregate: Dict[tuple[str, str, str, str], Dict[str, Any]] = {}
    for event in events:
        key = (
            str(event.get("provider", "")),
            str(event.get("model", "")),
            str(event.get("failure_kind", "")),
            str(event.get("key_label", "")),
        )
        row = aggregate.setdefault(
            key,
            {
                "provider": key[0],
                "model": key[1],
                "failure_kind": key[2],
                "key_label": key[3],
                "count": 0,
                "latest_seen": "",
                "latest_message": "",
                "solution_distance": "",
            },
        )
        row["count"] += 1
        row["latest_seen"] = str(event.get("timestamp", ""))
        row["latest_message"] = str(event.get("message", ""))
        row["solution_distance"] = str(event.get("solution_distance", ""))

    rows = sorted(
        aggregate.values(),
        key=lambda row: (row["provider"], row["model"], row["failure_kind"], row["key_label"]),
    )

    lines = [
        "# Provider Failure Report",
        "",
        f"- Generated at: {generated_at}",
        f"- Total failure events: {len(events)}",
        "",
        "| Provider | Model | Failure Kind | Key Label | Count | Latest Seen | Solution Distance | Latest Provider Message |",
        "|---|---|---|---|---:|---|---|---|",
    ]
    for row in rows:
        message = row["latest_message"].replace("|", "\\|")
        lines.append(
            "| {provider} | {model} | {failure_kind} | `{key_label}` | {count} | "
            "{latest_seen} | {solution_distance} | `{message}` |".format(
                **row,
                message=message,
            )
        )
    lines.append("")
    return "\n".join(lines)


def _ensure_report_links(markdown_path: Path) -> None:
    try:
        if markdown_path.name != "provider_failures.md" or markdown_path.parent.name != "monitoring":
            return
        run_root = markdown_path.parent.parent
        central_dir = Path("experiments/results/provider_failure_reports")
        central_dir.mkdir(parents=True, exist_ok=True)
        targets = [
            central_dir / f"{run_root.name}.md",
            central_dir / "latest.md",
        ]
        for link_path in targets:
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
            try:
                link_path.symlink_to(markdown_path.resolve())
            except OSError:
                link_path.write_text(markdown_path.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        logging.getLogger(__name__).debug("Failed to update provider failure report links", exc_info=True)


def record_provider_failure(
    *,
    provider: str,
    model: str,
    failure_kind: str,
    key: ProviderKey,
    exc: BaseException,
    exhausted: bool,
    next_key_label: Optional[str] = None,
) -> None:
    """Append one provider failure event and refresh the markdown summary."""

    markdown_path = report_path()
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = _jsonl_path(markdown_path)
    lock_path = _lock_path(markdown_path)
    event = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "provider": _provider_name(provider),
        "model": model,
        "failure_kind": failure_kind,
        "key_label": key.label,
        "key_group": key.group,
        "exhausted": exhausted,
        "next_key_label": next_key_label,
        "solution_distance": _solution_distance(
            failure_kind=failure_kind,
            exhausted=exhausted,
            next_key_label=next_key_label,
        ),
        "message": _safe_message(exc),
    }

    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        if fcntl is not None:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            with jsonl_path.open("a", encoding="utf-8") as jsonl_handle:
                jsonl_handle.write(json.dumps(event, sort_keys=True) + "\n")
                jsonl_handle.flush()
                os.fsync(jsonl_handle.fileno())
            events = _read_events(jsonl_path)
            temp_path = markdown_path.with_suffix(".tmp")
            temp_path.write_text(_render_markdown(events), encoding="utf-8")
            os.replace(temp_path, markdown_path)
            _ensure_report_links(markdown_path)
        finally:
            if fcntl is not None:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


async def call_with_key_rotation(
    *,
    provider: str,
    model: str,
    key_pool: ProviderKeyPool,
    request_coro_factory: Callable[[ProviderKey], Awaitable[Any]],
    logger: Optional[logging.Logger] = None,
    sleep_func: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> Any:
    """Run one provider API call with key rotation and transient retries."""

    provider_name = _provider_name(provider)
    log = logger or logging.getLogger(__name__)
    started = time.monotonic()
    transient_attempt = 0
    last_key_error: Optional[BaseException] = None
    last_key: Optional[ProviderKey] = None

    while True:
        key = key_pool.current()
        try:
            return await request_coro_factory(key)
        except Exception as exc:
            key_failure_kind = classify_key_scoped_failure(provider_name, exc)
            if key_failure_kind is not None:
                next_key = key_pool.rotate_after_failure(key)
                record_provider_failure(
                    provider=provider_name,
                    model=model,
                    failure_kind=key_failure_kind,
                    key=key,
                    exc=exc,
                    exhausted=next_key is None,
                    next_key_label=next_key.label if next_key else None,
                )
                last_key_error = exc
                last_key = key
                transient_attempt = 0
                if next_key is not None:
                    log.warning(
                        "%s key %s failed for model %s (%s); rotating to %s",
                        provider_name,
                        key.label,
                        model,
                        key_failure_kind,
                        next_key.label,
                    )
                    continue

                labels = key_pool.labels()
                raise ProviderKeyExhaustedError(
                    f"All configured {provider_name} API keys failed for model {model}. "
                    f"Last failed key={last_key.label if last_key else 'unknown'}; "
                    f"key labels={labels}; last error={_safe_message(last_key_error or exc)}. "
                    "Recommended fix: requeue after the provider quota reset, add funds, "
                    "or add another key to the key pool."
                ) from exc

            if is_transient_provider_failure(exc):
                retry_budget = _transient_retry_budget_seconds()
                elapsed = time.monotonic() - started
                remaining = retry_budget - elapsed
                if remaining <= 0:
                    raise ProviderTransientRetryExhaustedError(
                        f"{provider_name} transient retry budget exhausted after "
                        f"{retry_budget:.1f}s for model {model}: {_safe_message(exc)}"
                    ) from exc

                wait_time = min(
                    remaining,
                    60.0,
                    (2.0 ** transient_attempt) + random.uniform(0, 1),
                )
                transient_attempt += 1
                log.warning(
                    "%s transient API error for model %s on key %s; retrying in %.2fs "
                    "(elapsed %.2fs / %.2fs): %s",
                    provider_name,
                    model,
                    key.label,
                    wait_time,
                    elapsed,
                    retry_budget,
                    exc,
                )
                await sleep_func(wait_time)
                continue

            raise

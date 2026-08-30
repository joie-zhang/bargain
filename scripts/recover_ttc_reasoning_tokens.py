#!/usr/bin/env python3
"""Recover per-call hidden-reasoning usage for the five historical TTC seeds.

The historical Claude and Gemini artifacts retained inclusive provider output
counts but not their provider-specific reasoning breakdowns.  This script:

* matches seeds 984/526/423/1024 Gemini interactions to the exact raw response
  text retained by the OpenRouter file-proxy archive;
* uses the saved interaction response for Claude and seed-42 Gemini;
* asks each provider's free token-counting endpoint to count the visible text;
* reports the signed residual

      inclusive output tokens - estimated visible output tokens

without clamping or the older Gemini minimal-effort zero anchoring.

Historical result JSON and proxy files are read-only.  All generated artifacts
are written to a separate, versioned analysis directory.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from negotiation.provider_key_rotation import (  # noqa: E402
    ProviderKey,
    discover_provider_keys,
)


HISTORICAL_ROOTS: dict[int, Path] = {
    42: PROJECT_ROOT
    / "experiments/results/ttc_native_scaling_20260502_212943",
    984: PROJECT_ROOT
    / "experiments/results/ttc_native_scaling_seed984_20260725_025700",
    526: PROJECT_ROOT
    / "experiments/results/ttc_native_scaling_seed526_20260725_181400",
    423: PROJECT_ROOT
    / "experiments/results/ttc_native_scaling_seed423_20260725_211500",
    1024: PROJECT_ROOT
    / "experiments/results/ttc_native_scaling_seed1024_20260725_211500",
}
ARCHIVE_GEMINI_SEEDS = {984, 526, 423, 1024}
VALIDATION_ROOTS: dict[int, Path] = {
    seed: PROJECT_ROOT
    / f"experiments/results/ttc_native_scaling_seed{seed}_20260727_021100"
    for seed in (128, 256, 612, 2048, 4096)
}
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "experiments/analysis/reviewer_ttc_token_recovery_20260727"
)
DEFAULT_PROXY_DIR = Path("/home/jz4391/openrouter_proxy/processed")

FAMILIES = ("claude-sonnet-4-6", "gemini-3-flash")
FAMILY_MODEL_NEEDLE = {
    "claude-sonnet-4-6": "claude",
    "gemini-3-flash": "gemini",
}
COUNT_MODEL = {
    "claude-sonnet-4-6": "claude-sonnet-4-6",
    "gemini-3-flash": "gemini-3-flash-preview",
}
LEVEL_ORDER = {
    "claude-sonnet-4-6": ["low", "medium", "high", "max"],
    "gemini-3-flash": ["minimal", "low", "medium", "high"],
}
PARSED_PHASE_PREFIXES = ("private_thinking_", "proposal_", "voting_")
MATCH_WINDOW_SECONDS = 600.0
CACHE_VERSION = 1
RECOVERY_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--proxy-dir", type=Path, default=DEFAULT_PROXY_DIR)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inventory and match records without provider calls or output writes.",
    )
    parser.add_argument(
        "--require-validation",
        action="store_true",
        help="Fail unless capture-enabled validation interactions are available.",
    )
    parser.add_argument(
        "--run-anthropic-validation-probes",
        action="store_true",
        help=(
            "Make four small paid Claude calls (one per effort) when the "
            "direct-probe cache does not already exist."
        ),
    )
    parser.add_argument(
        "--validation-root",
        type=Path,
        action="append",
        default=None,
        help="Additional/replacement capture-enabled root; may be repeated.",
    )
    return parser.parse_args()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_level(path: Path) -> str:
    for part in path.parts:
        if part.startswith("level_"):
            return part.removeprefix("level_")
    raise ValueError(f"No level component in {path}")


def extract_run_dimensions(path: Path) -> tuple[str, str]:
    parts = list(path.parts)
    level_index = next(i for i, part in enumerate(parts) if part.startswith("level_"))
    return parts[level_index + 1], parts[level_index + 2]


def target_interaction(family: str, row: Mapping[str, Any]) -> bool:
    return FAMILY_MODEL_NEEDLE[family] in str(row.get("model_name") or "").lower()


def saved_text_fidelity(phase: str) -> str:
    if str(phase or "").startswith(PARSED_PHASE_PREFIXES):
        return "parsed_or_canonicalized_interaction_text"
    return "likely_raw_interaction_text"


def interaction_record(
    *,
    seed: int,
    family: str,
    run_path: Path,
    interaction_index: int,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    level = extract_level(run_path)
    game_cell, order = extract_run_dimensions(run_path)
    usage = row.get("token_usage") or {}
    response = str(row.get("response") or "")
    record_material = f"{run_path.resolve()}:{interaction_index}"
    return {
        "record_id": sha256_text(record_material),
        "seed": int(seed),
        "family": family,
        "provider": "anthropic" if family.startswith("claude") else "google",
        "model_name": str(row.get("model_name") or ""),
        "reasoning_effort": level,
        "level_index": LEVEL_ORDER[family].index(level),
        "game_cell": game_cell,
        "model_order": order,
        "run_path": str(run_path.resolve()),
        "interaction_index": int(interaction_index),
        "experiment_id": row.get("experiment_id"),
        "agent_id": row.get("agent_id"),
        "phase": row.get("phase"),
        "round": row.get("round"),
        "interaction_timestamp": row.get("timestamp"),
        "prompt_sha256": row.get("prompt_sha256"),
        "provider_input_tokens": usage.get("input_tokens"),
        "provider_output_tokens_inclusive": usage.get("output_tokens"),
        "provider_total_tokens": usage.get("total_tokens"),
        "direct_reasoning_tokens": (
            usage.get("reasoning_tokens")
            if usage.get("reasoning_tokens") is not None
            else usage.get("thinking_tokens")
        ),
        "direct_reasoning_source": usage.get("reasoning_token_source"),
        "openrouter_transport": usage.get("openrouter_transport"),
        "interaction_response_chars": len(response),
        "interaction_response_sha256": sha256_text(response),
        "visible_text_source": "saved_interaction_response",
        "visible_text_fidelity": saved_text_fidelity(str(row.get("phase") or "")),
        "visible_text_chars": len(response),
        "visible_text_sha256": sha256_text(response),
        "_visible_text": response,
    }


def discover_historical_interactions(
    roots: Mapping[int, Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    run_counts: dict[str, int] = {}
    target_counts: collections.Counter[tuple[int, str]] = collections.Counter()

    for seed, root in roots.items():
        for family in FAMILIES:
            paths = sorted(
                (root / family).glob(
                    "level_*/*/*/seed_*/run_1_all_interactions.json"
                )
            )
            run_counts[f"{seed}:{family}"] = len(paths)
            if len(paths) != 72:
                raise RuntimeError(
                    f"Expected 72 canonical {family} run files for seed {seed}; "
                    f"found {len(paths)} under {root}"
                )
            for run_path in paths:
                interactions = json_load(run_path)
                for index, row in enumerate(interactions):
                    if not isinstance(row, dict) or not target_interaction(family, row):
                        continue
                    target_counts[(seed, family)] += 1
                    usage = row.get("token_usage")
                    reasons: list[str] = []
                    if not isinstance(usage, dict):
                        reasons.append("missing_token_usage")
                    elif usage.get("output_tokens") is None:
                        reasons.append("missing_provider_output_tokens")
                    if not isinstance(row.get("response"), str) or not row.get("response"):
                        reasons.append("missing_visible_response_text")
                    if reasons:
                        exclusions.append(
                            {
                                "seed": seed,
                                "family": family,
                                "run_path": str(run_path.resolve()),
                                "interaction_index": index,
                                "phase": row.get("phase"),
                                "model_name": row.get("model_name"),
                                "exclusion_reason": ";".join(reasons),
                            }
                        )
                        continue
                    records.append(
                        interaction_record(
                            seed=seed,
                            family=family,
                            run_path=run_path,
                            interaction_index=index,
                            row=row,
                        )
                    )

    inventory = {
        "run_counts": run_counts,
        "target_interaction_counts": {
            f"{seed}:{family}": count
            for (seed, family), count in sorted(target_counts.items())
        },
        "included_usage_calls": len(records),
        "excluded_target_interactions": len(exclusions),
        "included_by_seed_family": dict(
            sorted(
                collections.Counter(
                    f"{record['seed']}:{record['family']}" for record in records
                ).items()
            )
        ),
    }
    return records, exclusions, inventory


@dataclass(frozen=True)
class ProxyRecord:
    suffix: str
    request_timestamp: float
    effort: str | None
    prompt_sha256: str
    completion_tokens: int
    direct_reasoning_tokens: int | None
    response_sha256: str
    response_text: str
    request_path: Path
    response_path: Path

    @property
    def key3(self) -> tuple[str | None, str, int]:
        return (self.effort, self.prompt_sha256, self.completion_tokens)


def request_last_content(payload: Mapping[str, Any]) -> str:
    messages = payload.get("messages") or []
    if not messages or not isinstance(messages[-1], dict):
        return ""
    content = messages[-1].get("content")
    if isinstance(content, str):
        return content
    return json.dumps(content, sort_keys=True, ensure_ascii=False)


def load_proxy_records(proxy_dir: Path) -> list[ProxyRecord]:
    records: list[ProxyRecord] = []
    request_paths = sorted(proxy_dir.glob("request_*.json"))
    for request_path in request_paths:
        suffix = request_path.name.removeprefix("request_")
        response_path = proxy_dir / f"response_{suffix}"
        if not response_path.exists():
            continue
        try:
            request = json_load(request_path)
            response = json_load(response_path)
        except (OSError, json.JSONDecodeError):
            continue
        payload = request.get("payload") or {}
        usage = response.get("usage") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        response_text = response.get("result")
        if "gemini" not in str(payload.get("model") or "").lower():
            continue
        if not isinstance(response_text, str) or not response_text:
            continue
        if usage.get("completion_tokens") is None:
            continue
        try:
            timestamp = int(suffix.split("_", 1)[0]) / 1_000_000_000
        except (TypeError, ValueError):
            continue
        effort = (payload.get("reasoning") or {}).get("effort")
        prompt_text = request_last_content(payload)
        records.append(
            ProxyRecord(
                suffix=suffix,
                request_timestamp=timestamp,
                effort=effort,
                prompt_sha256=sha256_text(prompt_text),
                completion_tokens=int(usage["completion_tokens"]),
                direct_reasoning_tokens=(
                    int(completion_details["reasoning_tokens"])
                    if completion_details.get("reasoning_tokens") is not None
                    else (
                        int(usage["reasoning_tokens"])
                        if usage.get("reasoning_tokens") is not None
                        else None
                    )
                ),
                response_sha256=sha256_text(response_text),
                response_text=response_text,
                request_path=request_path.resolve(),
                response_path=response_path.resolve(),
            )
        )
    return records


def interaction_key3(record: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(record["reasoning_effort"]),
        str(record["prompt_sha256"]),
        int(record["provider_output_tokens_inclusive"]),
    )


def _timestamp_delta(record: Mapping[str, Any], proxy: ProxyRecord) -> float:
    return float(record["interaction_timestamp"]) - proxy.request_timestamp


def match_proxy_group(
    interactions: Sequence[dict[str, Any]],
    candidates: Sequence[ProxyRecord],
    *,
    window_seconds: float = MATCH_WINDOW_SECONDS,
) -> list[tuple[dict[str, Any], ProxyRecord, str, int]]:
    """Return a duplicate-free assignment for one effort/prompt/count group."""
    unmatched_interactions = list(interactions)
    unmatched_candidates = list(candidates)
    assignments: list[tuple[dict[str, Any], ProxyRecord, str, int]] = []

    # Strongest case: the saved interaction response is byte-identical to raw
    # proxy output.  Pick the closest candidate if identical text occurred more
    # than once.
    for interaction in list(unmatched_interactions):
        possible = [
            candidate
            for candidate in unmatched_candidates
            if candidate.response_sha256
            == interaction["interaction_response_sha256"]
            and abs(_timestamp_delta(interaction, candidate)) <= window_seconds
        ]
        if not possible:
            continue
        chosen = min(
            possible,
            key=lambda candidate: abs(_timestamp_delta(interaction, candidate)),
        )
        assignments.append(
            (interaction, chosen, "exact_response_hash", len(possible))
        )
        unmatched_interactions.remove(interaction)
        unmatched_candidates.remove(chosen)

    if not unmatched_interactions:
        return assignments
    if len(unmatched_candidates) < len(unmatched_interactions):
        raise RuntimeError(
            "Too few proxy candidates after exact matching: "
            f"{len(unmatched_candidates)} candidates for "
            f"{len(unmatched_interactions)} interactions"
        )

    # Parsed/canonicalized phases cannot be compared byte-for-byte.  Use a
    # global minimum-cost, one-to-one timestamp assignment within the key group.
    costs = np.full(
        (len(unmatched_interactions), len(unmatched_candidates)),
        fill_value=1e12,
        dtype=float,
    )
    candidate_counts: list[int] = []
    for row_index, interaction in enumerate(unmatched_interactions):
        valid_count = 0
        for column_index, candidate in enumerate(unmatched_candidates):
            distance = abs(_timestamp_delta(interaction, candidate))
            if distance <= window_seconds:
                costs[row_index, column_index] = distance
                valid_count += 1
        candidate_counts.append(valid_count)
        if valid_count == 0:
            raise RuntimeError(
                "No proxy candidate within timestamp window for "
                f"{interaction['run_path']} interaction "
                f"{interaction['interaction_index']}"
            )

    row_indexes, column_indexes = linear_sum_assignment(costs)
    if len(row_indexes) != len(unmatched_interactions):
        raise RuntimeError("Proxy assignment did not cover every interaction")
    for row_index, column_index in zip(row_indexes, column_indexes, strict=True):
        if costs[row_index, column_index] >= 1e12:
            raise RuntimeError("Proxy assignment required an invalid candidate")
        method = (
            "unique_effort_prompt_completion"
            if candidate_counts[row_index] == 1
            else "minimum_timestamp_assignment"
        )
        assignments.append(
            (
                unmatched_interactions[row_index],
                unmatched_candidates[column_index],
                method,
                candidate_counts[row_index],
            )
        )
    return assignments


def attach_proxy_responses(
    records: list[dict[str, Any]],
    proxy_records: Sequence[ProxyRecord],
) -> list[dict[str, Any]]:
    target_records = [
        record
        for record in records
        if record["family"] == "gemini-3-flash"
        and int(record["seed"]) in ARCHIVE_GEMINI_SEEDS
    ]
    interactions_by_key: dict[
        tuple[str, str, int], list[dict[str, Any]]
    ] = collections.defaultdict(list)
    candidates_by_key: dict[
        tuple[str | None, str, int], list[ProxyRecord]
    ] = collections.defaultdict(list)
    for record in target_records:
        interactions_by_key[interaction_key3(record)].append(record)
    for proxy in proxy_records:
        candidates_by_key[proxy.key3].append(proxy)

    audit: list[dict[str, Any]] = []
    used_suffixes: set[str] = set()
    for key, interactions in interactions_by_key.items():
        candidates = candidates_by_key.get(key, [])
        assignments = match_proxy_group(interactions, candidates)
        for record, proxy, method, candidate_count in assignments:
            if proxy.suffix in used_suffixes:
                raise RuntimeError(f"Duplicate proxy assignment: {proxy.suffix}")
            used_suffixes.add(proxy.suffix)
            record["_visible_text"] = proxy.response_text
            record["visible_text_source"] = "openrouter_proxy_raw_result"
            record["visible_text_fidelity"] = "exact_raw_provider_output"
            record["visible_text_chars"] = len(proxy.response_text)
            record["visible_text_sha256"] = proxy.response_sha256
            record["proxy_suffix"] = proxy.suffix
            record["proxy_request_path"] = str(proxy.request_path)
            record["proxy_response_path"] = str(proxy.response_path)
            record["proxy_match_method"] = method
            record["proxy_match_candidate_count"] = candidate_count
            record["proxy_timestamp_delta_seconds"] = _timestamp_delta(
                record, proxy
            )
            audit.append(
                {
                    "record_id": record["record_id"],
                    "seed": record["seed"],
                    "run_path": record["run_path"],
                    "interaction_index": record["interaction_index"],
                    "phase": record["phase"],
                    "reasoning_effort": record["reasoning_effort"],
                    "proxy_suffix": proxy.suffix,
                    "proxy_request_path": str(proxy.request_path),
                    "proxy_response_path": str(proxy.response_path),
                    "match_method": method,
                    "candidate_count": candidate_count,
                    "timestamp_delta_seconds": _timestamp_delta(record, proxy),
                    "interaction_response_sha256": record[
                        "interaction_response_sha256"
                    ],
                    "proxy_response_sha256": proxy.response_sha256,
                    "response_text_identical": (
                        record["interaction_response_sha256"]
                        == proxy.response_sha256
                    ),
                }
            )

    if len(audit) != len(target_records):
        raise RuntimeError(
            f"Matched {len(audit)} proxy calls for {len(target_records)} "
            "archive-backed Gemini interactions"
        )
    return audit


class ProviderCountCache:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.payload: dict[str, Any] = {
            "version": CACHE_VERSION,
            "entries": {},
            "controls": {},
        }
        if path.exists():
            loaded = json_load(path)
            if loaded.get("version") != CACHE_VERSION:
                raise RuntimeError(f"Unsupported token-count cache: {path}")
            self.payload = loaded

    @property
    def entries(self) -> dict[str, dict[str, Any]]:
        return self.payload["entries"]

    @property
    def controls(self) -> dict[str, dict[str, Any]]:
        return self.payload["controls"]

    def save(self) -> None:
        with self.lock:
            atomic_write_json(self.path, self.payload)

    def set_control(self, key: str, value: dict[str, Any]) -> None:
        with self.lock:
            self.controls[key] = value
            atomic_write_json(self.path, self.payload)

    def set_entry(self, key: str, value: dict[str, Any], *, save: bool) -> None:
        with self.lock:
            self.entries[key] = value
            if save:
                atomic_write_json(self.path, self.payload)


def count_cache_key(provider: str, model: str, role: str, text: str) -> str:
    return sha256_text(
        json.dumps(
            {
                "provider": provider,
                "model": model,
                "role": role,
                "text_sha256": sha256_text(text),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def preferred_keys(provider: str) -> list[ProviderKey]:
    keys = discover_provider_keys(provider)
    return sorted(
        keys,
        key=lambda key: (
            0 if key.label.startswith("JOIE_") else 1,
            key.label,
        ),
    )


def retry_provider_call(
    call: Callable[[], int],
    *,
    label: str,
    attempts: int = 8,
) -> int:
    last_error: BaseException | None = None
    for attempt in range(attempts):
        try:
            return int(call())
        except BaseException as exc:
            last_error = exc
            if attempt == attempts - 1:
                break
            delay = min(2**attempt, 30) + random.random()
            print(
                f"Retrying {label} after {type(exc).__name__} "
                f"(attempt {attempt + 2}/{attempts}, sleep={delay:.1f}s)",
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError(f"Provider token count failed for {label}") from last_error


def anthropic_counter() -> Callable[[str], int]:
    import anthropic

    keys = preferred_keys("anthropic")
    if not keys:
        raise RuntimeError("No Anthropic API key discovered")
    key = keys[0]
    local = threading.local()

    def client() -> anthropic.Anthropic:
        if not hasattr(local, "client"):
            local.client = anthropic.Anthropic(api_key=key.value)
        return local.client

    def count(text: str) -> int:
        return retry_provider_call(
            lambda: client()
            .messages.count_tokens(
                model=COUNT_MODEL["claude-sonnet-4-6"],
                messages=[{"role": "assistant", "content": text}],
            )
            .input_tokens,
            label=f"Anthropic text {sha256_text(text)[:12]}",
        )

    count.key_label = key.label  # type: ignore[attr-defined]
    return count


def gemini_counter() -> Callable[[str], int]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import google.generativeai as genai

    keys = preferred_keys("google")
    if not keys:
        raise RuntimeError("No Google API key discovered")
    key = keys[0]
    genai.configure(api_key=key.value)
    local = threading.local()

    def model() -> Any:
        if not hasattr(local, "model"):
            local.model = genai.GenerativeModel(
                COUNT_MODEL["gemini-3-flash"]
            )
        return local.model

    def count(text: str) -> int:
        content = {"role": "model", "parts": [{"text": text}]}
        return retry_provider_call(
            lambda: model().count_tokens(content).total_tokens,
            label=f"Gemini text {sha256_text(text)[:12]}",
        )

    count.key_label = key.label  # type: ignore[attr-defined]
    return count


def ensure_control(
    cache: ProviderCountCache,
    *,
    family: str,
    counter: Callable[[str], int],
) -> int:
    provider = "anthropic" if family.startswith("claude") else "google"
    role = "assistant" if provider == "anthropic" else "model"
    model = COUNT_MODEL[family]
    control_key = f"{provider}:{model}:{role}:empty"
    cached = cache.controls.get(control_key)
    if cached is not None:
        return int(cached["raw_count_tokens"])
    value = counter("")
    cache.set_control(
        control_key,
        {
            "provider": provider,
            "model": model,
            "role": role,
            "raw_count_tokens": value,
            "key_label": getattr(counter, "key_label", None),
            "counted_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    )
    return value


def fill_provider_counts(
    records: Sequence[dict[str, Any]],
    cache: ProviderCountCache,
    *,
    workers: int,
) -> None:
    for family, factory in (
        ("claude-sonnet-4-6", anthropic_counter),
        ("gemini-3-flash", gemini_counter),
    ):
        family_records = [record for record in records if record["family"] == family]
        provider = "anthropic" if family.startswith("claude") else "google"
        role = "assistant" if provider == "anthropic" else "model"
        model = COUNT_MODEL[family]
        unique_texts: dict[str, str] = {}
        for record in family_records:
            text = record["_visible_text"]
            key = count_cache_key(provider, model, role, text)
            unique_texts.setdefault(key, text)
        missing = {
            key: text for key, text in unique_texts.items() if key not in cache.entries
        }
        counter: Callable[[str], int] | None = None
        if missing or not any(
            key.startswith(f"{provider}:{model}:{role}:")
            for key in cache.controls
        ):
            counter = factory()
        if counter is None:
            control_key = f"{provider}:{model}:{role}:empty"
            empty_count = int(cache.controls[control_key]["raw_count_tokens"])
        else:
            empty_count = ensure_control(cache, family=family, counter=counter)

        print(
            f"{family}: {len(family_records)} records, "
            f"{len(unique_texts)} unique texts, {len(missing)} uncached",
            flush=True,
        )
        if missing:
            assert counter is not None

            def one(item: tuple[str, str]) -> tuple[str, dict[str, Any]]:
                key, text = item
                raw_count = counter(text)
                return key, {
                    "provider": provider,
                    "model": model,
                    "role": role,
                    "text_sha256": sha256_text(text),
                    "text_chars": len(text),
                    "raw_count_tokens": raw_count,
                    "empty_control_tokens": empty_count,
                    "estimated_visible_tokens": raw_count - empty_count,
                    "key_label": getattr(counter, "key_label", None),
                    "counted_at_utc": dt.datetime.now(
                        dt.timezone.utc
                    ).isoformat(),
                }

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, workers)
            ) as executor:
                futures = {
                    executor.submit(one, item): item[0]
                    for item in missing.items()
                }
                for index, future in enumerate(
                    concurrent.futures.as_completed(futures), start=1
                ):
                    key, entry = future.result()
                    cache.set_entry(key, entry, save=index % 25 == 0)
                    if index % 100 == 0 or index == len(futures):
                        print(
                            f"{family}: counted {index}/{len(futures)}",
                            flush=True,
                        )
            cache.save()

        for record in family_records:
            key = count_cache_key(provider, model, role, record["_visible_text"])
            entry = cache.entries[key]
            visible = int(entry["estimated_visible_tokens"])
            output = int(record["provider_output_tokens_inclusive"])
            hidden = output - visible
            record["count_provider"] = provider
            record["count_model"] = model
            record["count_role"] = role
            record["count_endpoint"] = (
                "POST /v1/messages/count_tokens"
                if provider == "anthropic"
                else "models.countTokens"
            )
            record["count_cache_key"] = key
            record["raw_provider_count_tokens"] = int(
                entry["raw_count_tokens"]
            )
            record["empty_control_tokens"] = int(
                entry["empty_control_tokens"]
            )
            record["estimated_visible_output_tokens"] = visible
            record["reconstructed_hidden_tokens"] = hidden
            record["negative_hidden_residual"] = hidden < 0
            if (
                family == "gemini-3-flash"
                and record.get("visible_text_source")
                == "openrouter_proxy_raw_result"
            ):
                provenance = (
                    "proxy_raw_text_provider_tokenizer_reconstruction"
                )
            else:
                provenance = "saved_text_provider_tokenizer_estimate"
            record["token_recovery_provenance"] = provenance


def proxy_direct_validation_records(
    proxy_records: Sequence[ProxyRecord],
) -> list[dict[str, Any]]:
    """Build raw-text validation rows from capture-enabled Gemini responses."""
    records: list[dict[str, Any]] = []
    for proxy in proxy_records:
        if proxy.direct_reasoning_tokens is None:
            continue
        records.append(
            {
                "record_id": sha256_text(f"proxy-validation:{proxy.suffix}"),
                "seed": None,
                "family": "gemini-3-flash",
                "provider": "google",
                "model_name": "google/gemini-3-flash-preview",
                "reasoning_effort": proxy.effort,
                "level_index": (
                    LEVEL_ORDER["gemini-3-flash"].index(proxy.effort)
                    if proxy.effort in LEVEL_ORDER["gemini-3-flash"]
                    else None
                ),
                "run_path": None,
                "interaction_index": None,
                "phase": None,
                "provider_output_tokens_inclusive": proxy.completion_tokens,
                "direct_reasoning_tokens": proxy.direct_reasoning_tokens,
                "direct_reasoning_source": (
                    "openrouter_completion_tokens_details.reasoning_tokens"
                ),
                "visible_text_source": "openrouter_proxy_raw_result",
                "visible_text_fidelity": "exact_raw_provider_output",
                "visible_text_chars": len(proxy.response_text),
                "visible_text_sha256": proxy.response_sha256,
                "proxy_suffix": proxy.suffix,
                "proxy_request_path": str(proxy.request_path),
                "proxy_response_path": str(proxy.response_path),
                "validation_source": "openrouter_proxy_direct_usage",
                "_visible_text": proxy.response_text,
            }
        )
    return records


def anthropic_probe_records(
    cache_path: Path,
    *,
    run_if_missing: bool,
) -> list[dict[str, Any]]:
    """Load or create a tiny direct-usage calibration set for Claude."""
    if cache_path.exists():
        cached = json_load(cache_path)
        return list(cached.get("records") or [])
    if not run_if_missing:
        return []

    import anthropic

    keys = preferred_keys("anthropic")
    if not keys:
        raise RuntimeError("No Anthropic API key discovered for validation probes")
    key = keys[0]
    client = anthropic.Anthropic(api_key=key.value)
    prompt = (
        "Mentally compute 127 times 389, check the result, then reply with "
        "only the integer."
    )
    records: list[dict[str, Any]] = []
    for level_index, effort in enumerate(
        LEVEL_ORDER["claude-sonnet-4-6"]
    ):
        response = client.messages.create(
            model=COUNT_MODEL["claude-sonnet-4-6"],
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            thinking={"type": "adaptive"},
            extra_body={"output_config": {"effort": effort}},
        )
        text = "".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        )
        usage = response.usage
        details = getattr(usage, "output_tokens_details", None)
        thinking_tokens = (
            details.get("thinking_tokens")
            if isinstance(details, Mapping)
            else (
                getattr(details, "thinking_tokens", None)
                if details is not None
                else None
            )
        )
        # With adaptive thinking, Anthropic omits output_tokens_details when
        # it elects to use no thinking tokens.  A text-only response therefore
        # has a ground-truth thinking count of zero.
        if thinking_tokens is None:
            has_thinking_block = any(
                getattr(block, "type", None) == "thinking"
                for block in response.content
            )
            if has_thinking_block:
                raise RuntimeError(
                    "Claude probe contained thinking but no thinking-token count"
                )
            thinking_tokens = 0
            direct_source = "anthropic_adaptive_thinking_absent_implies_zero"
        else:
            direct_source = (
                "anthropic_usage.output_tokens_details.thinking_tokens"
            )
        records.append(
            {
                "record_id": sha256_text(
                    f"anthropic-validation:{response.id}:{effort}"
                ),
                "seed": None,
                "family": "claude-sonnet-4-6",
                "provider": "anthropic",
                "model_name": response.model,
                "reasoning_effort": effort,
                "level_index": level_index,
                "run_path": None,
                "interaction_index": None,
                "phase": "validation_probe",
                "provider_output_tokens_inclusive": int(usage.output_tokens),
                "direct_reasoning_tokens": int(thinking_tokens),
                "direct_reasoning_source": direct_source,
                "visible_text_source": "anthropic_direct_probe_text",
                "visible_text_fidelity": "exact_raw_provider_output",
                "visible_text_chars": len(text),
                "visible_text_sha256": sha256_text(text),
                "validation_source": "anthropic_direct_usage_probe",
                "anthropic_message_id": response.id,
                "anthropic_stop_reason": response.stop_reason,
                "_visible_text": text,
            }
        )
    atomic_write_json(
        cache_path,
        {
            "version": 1,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "model": COUNT_MODEL["claude-sonnet-4-6"],
            "prompt_sha256": sha256_text(prompt),
            "key_label": key.label,
            "records": records,
        },
    )
    return records


def validation_root_map(paths: Sequence[Path] | None) -> dict[str, Path]:
    if paths:
        return {path.resolve().name: path.resolve() for path in paths}
    return {str(seed): root.resolve() for seed, root in VALIDATION_ROOTS.items()}


def discover_validation_records(
    roots: Mapping[str, Path],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for root_label, root in roots.items():
        if not root.exists():
            continue
        for family in FAMILIES:
            for run_path in sorted(
                (root / family).glob(
                    "level_*/*/*/seed_*/run_1_all_interactions.json"
                )
            ):
                for index, row in enumerate(json_load(run_path)):
                    if not isinstance(row, dict) or not target_interaction(family, row):
                        continue
                    usage = row.get("token_usage") or {}
                    direct = (
                        usage.get("reasoning_tokens")
                        if usage.get("reasoning_tokens") is not None
                        else usage.get("thinking_tokens")
                    )
                    if (
                        direct is None
                        or usage.get("output_tokens") is None
                        or not isinstance(row.get("response"), str)
                        or not row.get("response")
                    ):
                        continue
                    fidelity = saved_text_fidelity(str(row.get("phase") or ""))
                    if fidelity != "likely_raw_interaction_text":
                        continue
                    record = interaction_record(
                        seed=int(
                            next(
                                part.removeprefix("seed_")
                                for part in run_path.parts
                                if part.startswith("seed_")
                            )
                        ),
                        family=family,
                        run_path=run_path,
                        interaction_index=index,
                        row=row,
                    )
                    record["validation_root_label"] = root_label
                    record["validation_source"] = (
                        "capture_enabled_ttc_result"
                    )
                    records.append(record)
    return records


def build_validation_frame(
    records: list[dict[str, Any]],
    cache: ProviderCountCache,
    *,
    workers: int,
) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    fill_provider_counts(records, cache, workers=workers)
    rows: list[dict[str, Any]] = []
    for record in records:
        expected_visible = int(record["provider_output_tokens_inclusive"]) - int(
            record["direct_reasoning_tokens"]
        )
        estimated_visible = int(record["estimated_visible_output_tokens"])
        signed_error = estimated_visible - expected_visible
        rows.append(
            {
                key: value
                for key, value in record.items()
                if not key.startswith("_")
            }
            | {
                "direct_implied_visible_tokens": expected_visible,
                "visible_count_signed_error": signed_error,
                "visible_count_absolute_error": abs(signed_error),
                "visible_count_exact_match": signed_error == 0,
            }
        )
    return pd.DataFrame(rows)


def validation_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "family",
                "reasoning_effort",
                "calls",
                "exact_match_rate",
                "mean_signed_error",
                "median_signed_error",
                "mean_absolute_error",
                "absolute_error_p95",
            ]
        )
    rows: list[dict[str, Any]] = []
    for (family, effort), group in frame.groupby(
        ["family", "reasoning_effort"], sort=False
    ):
        signed = group["visible_count_signed_error"].to_numpy(dtype=float)
        absolute = np.abs(signed)
        rows.append(
            {
                "family": family,
                "reasoning_effort": effort,
                "calls": len(group),
                "exact_match_rate": float(np.mean(signed == 0)),
                "mean_signed_error": float(np.mean(signed)),
                "median_signed_error": float(np.median(signed)),
                "mean_absolute_error": float(np.mean(absolute)),
                "absolute_error_p95": float(np.percentile(absolute, 95)),
            }
        )
    return pd.DataFrame(rows).sort_values(["family", "reasoning_effort"])


def export_frame(records: Sequence[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {key: value for key, value in record.items() if not key.startswith("_")}
            for record in records
        ]
    )
    return frame.sort_values(
        ["family", "seed", "level_index", "run_path", "interaction_index"]
    ).reset_index(drop=True)


def aggregate_recovery(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(list(group_columns), sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        calls = len(group)
        output = int(group["provider_output_tokens_inclusive"].sum())
        visible = int(group["estimated_visible_output_tokens"].sum())
        hidden = int(group["reconstructed_hidden_tokens"].sum())
        row = dict(zip(group_columns, keys, strict=True))
        row.update(
            {
                "target_calls": calls,
                "provider_output_tokens_inclusive_sum": output,
                "estimated_visible_output_tokens_sum": visible,
                "reconstructed_hidden_tokens_sum": hidden,
                "provider_output_tokens_per_call": output / calls,
                "estimated_visible_tokens_per_call": visible / calls,
                "reconstructed_hidden_tokens_per_call": hidden / calls,
                "reconstructed_hidden_fraction": (
                    hidden / output if output else math.nan
                ),
                "negative_residual_calls": int(
                    group["negative_hidden_residual"].sum()
                ),
                "exact_raw_text_calls": int(
                    group["visible_text_fidelity"]
                    .eq("exact_raw_provider_output")
                    .sum()
                ),
                "parsed_or_canonicalized_calls": int(
                    group["visible_text_fidelity"]
                    .eq("parsed_or_canonicalized_interaction_text")
                    .sum()
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def effort_summary(seed_effort: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (family, effort, level_index), group in seed_effort.groupby(
        ["family", "reasoning_effort", "level_index"], sort=False
    ):
        values = group["reconstructed_hidden_tokens_per_call"].to_numpy(
            dtype=float
        )
        n = len(values)
        mean = float(np.mean(values))
        sem = float(stats.sem(values)) if n > 1 else math.nan
        half = (
            float(stats.t.ppf(0.975, n - 1) * sem)
            if n > 1
            else math.nan
        )
        rows.append(
            {
                "family": family,
                "reasoning_effort": effort,
                "level_index": int(level_index),
                "seed_count": n,
                "hidden_tokens_per_call_mean": mean,
                "hidden_tokens_per_call_seed_sem": sem,
                "hidden_tokens_per_call_ci95_low": mean - half,
                "hidden_tokens_per_call_ci95_high": mean + half,
                "negative_residual_calls": int(
                    group["negative_residual_calls"].sum()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["family", "level_index"])


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_readme(
    path: Path,
    *,
    inventory: Mapping[str, Any],
    master: pd.DataFrame,
    validation: pd.DataFrame,
    validation_stats: pd.DataFrame,
) -> None:
    by_family_seed = (
        master.groupby(["family", "seed"]).size().rename("calls").reset_index()
    )
    lines = [
        "# Historical TTC reasoning-token recovery",
        "",
        "This directory is an additive, auditable reconstruction. Historical result",
        "JSON and OpenRouter proxy files were not modified.",
        "",
        "The reconstructed quantity is:",
        "",
        "`historical inclusive output tokens - provider-counted visible text`",
        "",
        "Negative residuals are preserved and flagged. No values are clamped, and",
        "Gemini is not zero-anchored to its minimal-effort condition.",
        "",
        "## Provenance",
        "",
        "- Gemini seeds 984, 526, 423, and 1024 use exact raw provider output from",
        "  matched OpenRouter proxy response files.",
        "- Gemini seed 42 and all Claude seeds use the saved interaction response.",
        "  Structured private-thinking, proposal, and voting phases may have been",
        "  parsed or canonicalized before saving and are flagged accordingly.",
        "- Provider token-count APIs estimate current tokenizer counts. These are",
        "  validated reconstructions, not directly observed historical reasoning",
        "  fields.",
        "",
        "## Coverage",
        "",
        by_family_seed.to_markdown(index=False),
        "",
        f"Included calls: {len(master):,}.",
        f"Excluded target interactions: "
        f"{inventory['excluded_target_interactions']:,}.",
        "",
        "## Direct-field validation",
        "",
    ]
    if validation.empty:
        lines.extend(
            [
                "No capture-enabled raw-fidelity validation calls were available",
                "when this package was generated. Re-run after the restarted jobs",
                "finish to populate the validation files.",
            ]
        )
    else:
        lines.extend(
            [
                f"Validation calls: {len(validation):,}.",
                "",
                validation_stats.to_markdown(index=False),
            ]
        )
        gemini_errors = validation.loc[
            validation["family"].eq("gemini-3-flash"),
            "visible_count_signed_error",
        ]
        claude_errors = validation.loc[
            validation["family"].eq("claude-sonnet-4-6"),
            "visible_count_signed_error",
        ]
        if not gemini_errors.empty:
            lines.extend(
                [
                    "",
                    "The Gemini countTokens reconstruction was exact on "
                    f"{len(gemini_errors):,}/{len(gemini_errors):,} direct-field "
                    "raw-response checks.",
                ]
            )
        if not claude_errors.empty:
            claude_absolute_errors = claude_errors.abs()
            lines.extend(
                [
                    "",
                    "For Claude, assistant-message token counting omitted "
                    f"{int(claude_absolute_errors.min())}–"
                    f"{int(claude_absolute_errors.max())} output-envelope tokens "
                    "in these probes. The reported historical residual therefore "
                    "overestimates direct thinking by that small per-call amount "
                    "when the saved text is raw.",
                ]
            )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `interaction_token_recovery.csv`: one row per usage-bearing call.",
            "- `run_token_recovery.csv`: call sums and per-call metrics by run.",
            "- `seed_effort_token_recovery.csv`: aggregation by seed and effort.",
            "- `effort_summary.csv`: across-seed mean and 95% t interval.",
            "- `proxy_match_audit.csv`: one-to-one proxy matching diagnostics.",
            "- `excluded_interactions.csv`: target rows without usable usage/text.",
            "- `fidelity_quality_summary.csv`: coverage and signed-negative",
            "  diagnostics by family, seed, and text fidelity.",
            "- `validation_against_direct_counts.csv`: comparison with new direct",
            "  reasoning fields.",
            "- `anthropic_direct_probe_cache.json`: four small direct Claude",
            "  calibration calls, when explicitly requested.",
            "- `provider_token_count_cache.json`: resumable, text-hash-based cache.",
            "- `manifest.json`: source roots, counts, versions, and checksums.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    proxy_dir = args.proxy_dir.resolve()
    roots = {seed: path.resolve() for seed, path in HISTORICAL_ROOTS.items()}

    print("Discovering canonical historical interactions", flush=True)
    records, exclusions, inventory = discover_historical_interactions(roots)
    print(json.dumps(inventory, indent=2), flush=True)
    print("Loading the read-only OpenRouter proxy archive snapshot", flush=True)
    proxy_records = load_proxy_records(proxy_dir)
    print(f"Loaded {len(proxy_records):,} usage-bearing Gemini proxy pairs")
    proxy_audit = attach_proxy_responses(records, proxy_records)
    print(f"Matched {len(proxy_audit):,} archive-backed Gemini calls")

    if args.dry_run:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    cache = ProviderCountCache(output_dir / "provider_token_count_cache.json")
    fill_provider_counts(records, cache, workers=max(1, args.workers))

    validation_records = proxy_direct_validation_records(proxy_records)
    validation_records.extend(
        anthropic_probe_records(
            output_dir / "anthropic_direct_probe_cache.json",
            run_if_missing=args.run_anthropic_validation_probes,
        )
    )
    validation_records.extend(
        discover_validation_records(
        validation_root_map(args.validation_root)
        )
    )
    if args.require_validation and not validation_records:
        raise RuntimeError(
            "No capture-enabled raw-fidelity validation calls were found"
        )
    validation = build_validation_frame(
        validation_records,
        cache,
        workers=max(1, args.workers),
    )
    validation_stats = validation_summary(validation)

    master = export_frame(records)
    run_summary = aggregate_recovery(
        master,
        [
            "seed",
            "family",
            "reasoning_effort",
            "level_index",
            "game_cell",
            "model_order",
            "run_path",
        ],
    )
    seed_summary = aggregate_recovery(
        master,
        ["seed", "family", "reasoning_effort", "level_index"],
    )
    across_seed = effort_summary(seed_summary)
    quality_summary = (
        master.groupby(
            ["family", "seed", "visible_text_source", "visible_text_fidelity"],
            dropna=False,
        )
        .agg(
            calls=("record_id", "size"),
            negative_residual_calls=("negative_hidden_residual", "sum"),
            reconstructed_hidden_tokens_min=(
                "reconstructed_hidden_tokens",
                "min",
            ),
            reconstructed_hidden_tokens_mean=(
                "reconstructed_hidden_tokens",
                "mean",
            ),
        )
        .reset_index()
    )
    quality_summary["negative_residual_rate"] = (
        quality_summary["negative_residual_calls"] / quality_summary["calls"]
    )

    master_path = output_dir / "interaction_token_recovery.csv"
    run_path = output_dir / "run_token_recovery.csv"
    seed_path = output_dir / "seed_effort_token_recovery.csv"
    effort_path = output_dir / "effort_summary.csv"
    proxy_audit_path = output_dir / "proxy_match_audit.csv"
    exclusions_path = output_dir / "excluded_interactions.csv"
    quality_path = output_dir / "fidelity_quality_summary.csv"
    validation_path = output_dir / "validation_against_direct_counts.csv"
    validation_summary_path = output_dir / "validation_summary.csv"

    master.to_csv(master_path, index=False)
    run_summary.to_csv(run_path, index=False)
    seed_summary.to_csv(seed_path, index=False)
    across_seed.to_csv(effort_path, index=False)
    pd.DataFrame(proxy_audit).sort_values(
        ["seed", "run_path", "interaction_index"]
    ).to_csv(proxy_audit_path, index=False)
    pd.DataFrame(exclusions).to_csv(exclusions_path, index=False)
    quality_summary.to_csv(quality_path, index=False)
    validation.to_csv(validation_path, index=False)
    validation_stats.to_csv(validation_summary_path, index=False)

    manifest = {
        "recovery_version": RECOVERY_VERSION,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "script_path": str(Path(__file__).resolve()),
        "historical_roots": {
            str(seed): str(path) for seed, path in roots.items()
        },
        "validation_roots": {
            label: str(path)
            for label, path in validation_root_map(args.validation_root).items()
        },
        "proxy_dir": str(proxy_dir),
        "proxy_snapshot_usage_pairs": len(proxy_records),
        "inventory": inventory,
        "master_rows": len(master),
        "proxy_match_rows": len(proxy_audit),
        "validation_rows": len(validation),
        "validation_rows_by_source": (
            validation.groupby("validation_source").size().to_dict()
            if not validation.empty and "validation_source" in validation
            else {}
        ),
        "negative_residual_rows": int(
            master["negative_hidden_residual"].sum()
        ),
        "parsed_or_canonicalized_rows": int(
            master["visible_text_fidelity"]
            .eq("parsed_or_canonicalized_interaction_text")
            .sum()
        ),
        "models": COUNT_MODEL,
        "sdk_versions": {
            "anthropic": package_version("anthropic"),
            "google-generativeai": package_version("google-generativeai"),
            "pandas": package_version("pandas"),
            "scipy": package_version("scipy"),
        },
        "method": {
            "formula": (
                "provider_output_tokens_inclusive - "
                "estimated_visible_output_tokens"
            ),
            "negative_residual_policy": "preserve_and_flag",
            "gemini_zero_anchor": False,
            "full_response_text_exported": False,
        },
    }
    output_checksums = {}
    for path in (
        master_path,
        run_path,
        seed_path,
        effort_path,
        proxy_audit_path,
        exclusions_path,
        quality_path,
        validation_path,
        validation_summary_path,
        cache.path,
    ):
        output_checksums[path.name] = sha256_file(path)
    anthropic_probe_path = output_dir / "anthropic_direct_probe_cache.json"
    if anthropic_probe_path.exists():
        output_checksums[anthropic_probe_path.name] = sha256_file(
            anthropic_probe_path
        )
    manifest["output_sha256"] = output_checksums
    atomic_write_json(output_dir / "manifest.json", manifest)
    write_readme(
        output_dir / "README.md",
        inventory=inventory,
        master=master,
        validation=validation,
        validation_stats=validation_stats,
    )

    print(f"Wrote recovery package to {output_dir}")
    print(f"Master rows: {len(master):,}")
    print(f"Proxy matches: {len(proxy_audit):,}")
    print(f"Validation rows: {len(validation):,}")
    print(
        "Negative residual rows: "
        f"{int(master['negative_hidden_residual'].sum()):,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

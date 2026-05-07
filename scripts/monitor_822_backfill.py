#!/usr/bin/env python3
"""Monitor the 822-sample homogeneous/heterogeneous backfill slice.

This script intentionally focuses on the Appendix A backfill slice discussed in
the monitoring thread. It treats valid result files as successful, overlays
active Slurm array state, and recovers failed statuses that were only printed to
Slurm stdout when the per-config status artifact was not written.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
HOMOGENEOUS_ROOT = PROJECT_ROOT / "experiments/results/full_games123_multiagent_production_20260428_085255"
HETEROGENEOUS_ROOT = (
    PROJECT_ROOT
    / "experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848"
)
SCOPE_SELECTIONS = {
    "homogeneous": [
        HOMOGENEOUS_ROOT / "selections/backfill_n_gt2_appendixA_20260502_homogeneous_failed_config_ids.txt",
        HOMOGENEOUS_ROOT / "selections/backfill_n2_appendixA_20260502_homogeneous_failed_config_ids.txt",
    ],
    "heterogeneous": [
        HETEROGENEOUS_ROOT / "selections/backfill_n_gt2_appendixA_20260502_heterogeneous_failed_config_ids.txt",
        HETEROGENEOUS_ROOT / "selections/backfill_n2_appendixA_20260502_heterogeneous_failed_config_ids.txt",
        HETEROGENEOUS_ROOT / "selections/release_n_gt2_appendixA_20260502_heterogeneous_held_tail_config_ids.txt",
    ],
}
SCOPE_ROOTS = {
    "homogeneous": HOMOGENEOUS_ROOT,
    "heterogeneous": HETEROGENEOUS_ROOT,
}

STATE_ORDER = ("SUCCEEDED", "FAILED", "RUNNING", "QUEUED", "ERRORS")
STATE_LABELS = {
    "SUCCEEDED": "Succeeded",
    "FAILED": "Failed",
    "RUNNING": "Running",
    "QUEUED": "Queued",
    "ERRORS": "Errors",
}
OPENROUTER_BUCKET = "OpenRouter pool exhausted or failed after Lewis-to-Joie rotation"
STDOUT_ONLY_BUCKET = "failed_task_stdout_only_missing_per_config_artifacts"
SLURM_JSON = json.JSONDecoder()


def load_batch_module() -> Any:
    module_path = PROJECT_ROOT / "scripts/full_games123_multiagent_batch.py"
    spec = importlib.util.spec_from_file_location("fg123_batch", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FG123 = load_batch_module()


def now_local() -> dt.datetime:
    return dt.datetime.now().astimezone()


def read_config_ids(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"state": "STATUS_PARSE_ERROR", "status_parse_error_path": str(path)}


def read_tail(path: Path, max_bytes: int = 160_000) -> str:
    if not path.exists():
        return ""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > max_bytes:
                handle.seek(-max_bytes, os.SEEK_END)
            data = handle.read()
        return data.decode("utf-8", errors="replace")
    except OSError:
        return ""


def parse_stdout_status(path: Path) -> dict[str, Any]:
    text = read_tail(path, max_bytes=260_000).strip()
    if not text:
        return {}
    for match in reversed(list(re.finditer(r"{", text))):
        candidate = text[match.start() :].lstrip()
        try:
            payload, _ = SLURM_JSON.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and ("state" in payload or "config_id" in payload):
            payload["_recovered_from_slurm_stdout"] = str(path)
            return payload
    return {}


def selected_path_from_scontrol(job_id: str) -> tuple[Path | None, Path | None, int | None]:
    try:
        output = subprocess.check_output(
            ["scontrol", "show", "job", str(job_id)],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None, None

    submit_line = " ".join(
        line.strip()
        for line in output.splitlines()
        if "SubmitLine=" in line or line.lstrip().startswith("sbatch ")
    )
    if "SubmitLine=" in submit_line:
        submit_line = submit_line.split("SubmitLine=", 1)[1]

    run_dir_match = re.search(r"(?:^|[, ])RUN_DIR=([^,\s]+)", submit_line)
    selected_match = re.search(r"(?:^|[, ])SELECTED_CONFIG_IDS_FILE=([^,\s]+)", submit_line)
    offset_match = re.search(r"(?:^|[, ])CONFIG_OFFSET=([^,\s]+)", submit_line)

    run_dir = Path(shlex.split(run_dir_match.group(1))[0]) if run_dir_match else None
    selected_path = Path(shlex.split(selected_match.group(1))[0]) if selected_match else None
    offset = int(offset_match.group(1)) if offset_match else None
    return run_dir, selected_path, offset


def known_submission_maps() -> dict[str, dict[str, Any]]:
    maps: dict[str, dict[str, Any]] = {}
    for scope, root in SCOPE_ROOTS.items():
        for record in FG123.load_submission_records(root):
            job_id = str(record.get("job_id") or "")
            if not job_id:
                continue
            config_ids = [int(value) for value in record.get("config_ids") or []]
            maps[job_id] = {
                "scope": scope,
                "root": root,
                "config_ids": config_ids,
                "config_offset": int(record.get("config_offset", 0)),
                "source": "submissions.json",
            }
    return maps


def active_slurm_states(slice_ids: dict[str, set[int]]) -> dict[tuple[str, int], dict[str, str]]:
    states: dict[tuple[str, int], dict[str, str]] = {}
    submission_maps = known_submission_maps()
    scontrol_cache: dict[str, dict[str, Any]] = {}

    try:
        output = subprocess.check_output(
            ["squeue", "-r", "-h", "-u", os.getenv("USER", ""), "-o", "%i|%T|%M|%R|%j"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return states

    for line in output.splitlines():
        parts = line.split("|", 4)
        if len(parts) != 5:
            continue
        slurm_id, slurm_state, elapsed, reason, job_name = parts
        if job_name not in {"fg123", "fg123bf"} or "_" not in slurm_id:
            continue
        job_id, raw_task_id = slurm_id.rsplit("_", 1)
        try:
            task_id = int(raw_task_id)
        except ValueError:
            continue

        job_map = submission_maps.get(job_id)
        if job_map is None:
            if job_id not in scontrol_cache:
                run_dir, selected_path, offset = selected_path_from_scontrol(job_id)
                config_ids: list[int] = []
                scope = None
                if selected_path and selected_path.exists():
                    config_ids = read_config_ids(selected_path)
                if run_dir:
                    for candidate_scope, candidate_root in SCOPE_ROOTS.items():
                        try:
                            if run_dir.resolve() == candidate_root.resolve():
                                scope = candidate_scope
                        except OSError:
                            pass
                scontrol_cache[job_id] = {
                    "scope": scope,
                    "root": run_dir,
                    "config_ids": config_ids,
                    "config_offset": int(offset or 0),
                    "source": "scontrol",
                }
            job_map = scontrol_cache[job_id]

        config_ids = job_map.get("config_ids") or []
        if config_ids:
            if task_id < 1 or task_id > len(config_ids):
                continue
            config_id = int(config_ids[task_id - 1])
        else:
            config_id = task_id + int(job_map.get("config_offset", 0))

        scope = job_map.get("scope")
        if scope not in slice_ids or config_id not in slice_ids[scope]:
            continue

        state = "RUNNING" if slurm_state == "RUNNING" else "QUEUED" if slurm_state == "PENDING" else slurm_state
        key = (str(scope), config_id)
        old = states.get(key)
        if old and old.get("state") == "RUNNING":
            continue
        states[key] = {
            "state": state,
            "slurm_state": slurm_state,
            "elapsed": elapsed,
            "reason": reason,
            "job_name": job_name,
            "job_id": job_id,
            "task_id": str(task_id),
            "mapping_source": str(job_map.get("source")),
        }
    return states


def task_maps_for_stdout(root: Path) -> dict[int, list[tuple[str, int]]]:
    mapping: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for record in FG123.load_submission_records(root):
        job_id = str(record.get("job_id") or "")
        if not job_id:
            continue
        config_ids = [int(value) for value in record.get("config_ids") or []]
        if config_ids:
            for task_id, config_id in enumerate(config_ids, start=1):
                mapping[config_id].append((job_id, task_id))
        else:
            offset = int(record.get("config_offset", 0))
            array_spec = str(record.get("array_spec") or "")
            match = re.match(r"1-(\d+)", array_spec)
            if match:
                for task_id in range(1, int(match.group(1)) + 1):
                    mapping[task_id + offset].append((job_id, task_id))
    return mapping


def recover_status_from_stdout(root: Path, config_id: int, stdout_map: dict[int, list[tuple[str, int]]]) -> dict[str, Any]:
    for job_id, task_id in reversed(stdout_map.get(config_id, [])):
        path = PROJECT_ROOT / "slurm" / f"full_games123_{job_id}_{task_id}.out"
        payload = parse_stdout_status(path)
        if payload and int(payload.get("config_id", config_id)) == int(config_id):
            return payload
    return {}


def status_attempts(status: dict[str, Any]) -> list[dict[str, Any]]:
    attempts = status.get("attempts")
    return attempts if isinstance(attempts, list) else []


def latest_attempt(status: dict[str, Any]) -> dict[str, Any]:
    attempts = status_attempts(status)
    return attempts[-1] if attempts else {}


def status_job(status: dict[str, Any]) -> tuple[str | None, str | None]:
    attempt = latest_attempt(status)
    job_id = attempt.get("slurm_array_job_id") or status.get("slurm_array_job_id")
    task_id = attempt.get("slurm_array_task_id") or status.get("slurm_array_task_id")
    return (str(job_id), str(task_id)) if job_id and task_id else (None, None)


def result_success(config: dict[str, Any]) -> bool:
    result_path = FG123.result_path_for(config)
    return result_path is not None and FG123.validate_result_file(config, result_path) is None


def detail_text(root: Path, config_id: int, status: dict[str, Any], stdout_status: bool) -> str:
    pieces = [json.dumps(status, default=str, sort_keys=True)]
    log_paths: list[Path] = []
    for key in ("attempt_log_path", "log_path"):
        raw = status.get(key)
        if raw:
            log_paths.append(Path(raw))
    for attempt in reversed(status_attempts(status)[-3:]):
        for key in ("log_path", "attempt_log_path"):
            raw = attempt.get(key)
            if raw:
                log_paths.append(Path(raw))
    log_paths.append(root / "logs" / f"config_{config_id:04d}.log")
    for path in dict.fromkeys(log_paths):
        pieces.append(read_tail(path))
    if stdout_status:
        recovered = status.get("_recovered_from_slurm_stdout")
        if recovered:
            pieces.append(read_tail(Path(recovered)))
    return "\n".join(piece for piece in pieces if piece)


def openrouter_tags(text: str) -> set[str]:
    lower = text.lower()
    tags: set[str] = set()
    if (
        "all configured openrouter api keys failed" in lower
        or ("all openrouter" in lower and ("exhausted" in lower or "failed" in lower))
    ):
        tags.add("all OpenRouter keys exhausted")
    if "openrouter" in lower and ("timeout" in lower or "timed out" in lower or "read timed out" in lower):
        tags.add("OpenRouter timeout")
    if "openrouter" in lower and (
        "empty response" in lower
        or "upstream" in lower
        or "no response body" in lower
        or "returned empty" in lower
    ):
        tags.add("OpenRouter empty/upstream response")
    lewis_failed = bool(
        re.search(r"openrouter key LEWIS_OPENROUTER_API_KEY_\d+ failed", text)
        or re.search(r"labels=\[[^\]]*LEWIS_OPENROUTER_API_KEY_\d+[^\]]*\].*exhausted", text, flags=re.I)
    )
    joie_failed = bool(
        re.search(r"Last failed key=JOIE_OPENROUTER_API_KEY_\d+", text)
        or re.search(r"labels=\[[^\]]*JOIE_OPENROUTER_API_KEY_\d+[^\]]*\].*exhausted", text, flags=re.I)
        or re.search(r"exhausted \(labels=\[[^\]]*JOIE_OPENROUTER_API_KEY_\d+[^\]]*\]\)", text, flags=re.I)
    )
    joie_success = bool(
        re.search(r"JOIE_OPENROUTER_API_KEY_\d+.*(?:succeeded|success)", text, flags=re.I)
    )
    if lewis_failed and joie_failed:
        tags.add("Lewis failed, Joie also failed")
    if lewis_failed and joie_success:
        tags.add("Lewis failed, Joie succeeded later")
    if "native" in lower and "fallback" in lower and "openrouter" in lower and (
        "overloaded" in lower or "overload" in lower or "fallback" in lower
    ):
        tags.add("native-provider fallback overloaded OpenRouter")
    return tags


def normalized_openrouter_error(raw_error: str) -> str:
    if "HTTP 402" in raw_error or "Insufficient credits" in raw_error:
        return "HTTP 402 insufficient credits"
    if "HTTP 429" in raw_error or "rate-limited upstream" in raw_error:
        return "HTTP 429 upstream/provider rate-limited"
    if "HTTP 403" in raw_error or "Key limit exceeded" in raw_error:
        return "HTTP 403 key limit exceeded"
    if "TimeoutError" in raw_error:
        return "TimeoutError"
    if "Empty content from model" in raw_error:
        return "Empty content from model"
    if "Anthropic returned empty content" in raw_error:
        return "Anthropic empty content at max_tokens"
    return raw_error.strip()[:180]


def joie_error_evidence(text: str) -> set[str]:
    evidence: set[str] = set()
    for match in re.finditer(
        r"Last failed key=JOIE_OPENROUTER_API_KEY_\d+;[^\n]*last error=([^\n]+?)(?:\. Recommended fix|\n|$)",
        text,
    ):
        evidence.add(normalized_openrouter_error(match.group(1)))
    if re.search(r"exhausted \(labels=\[[^\]]*JOIE_OPENROUTER_API_KEY_\d+[^\]]*\]\)", text, flags=re.I):
        evidence.add("OpenRouter key pool exhausted with Joie label")
    return evidence


def joie_attempt_evidence(text: str) -> set[str]:
    evidence: set[str] = set()
    if re.search(r"rotating to JOIE_OPENROUTER_API_KEY_\d+", text):
        evidence.add("Lewis failed, rotated to Joie")
    if re.search(r"Last failed key=JOIE_OPENROUTER_API_KEY_\d+", text):
        evidence.add("Joie was terminal last failed key")
    if re.search(r"exhausted \(labels=\[[^\]]*JOIE_OPENROUTER_API_KEY_\d+[^\]]*\]\)", text, flags=re.I):
        evidence.add("Joie appeared in exhausted key-pool labels")
    return evidence


def classify_failure(status: dict[str, Any], text: str, stdout_status: bool) -> tuple[str, set[str]]:
    lower = text.lower()
    tags = openrouter_tags(text)

    has_specific_detail = any(
        marker in lower
        for marker in (
            "traceback",
            "exception",
            "error:",
            "openrouter",
            "anthropic",
            "gemini",
            "max_tokens",
            "max_num_tokens",
            "invalid proposal",
            "proposal",
            "context",
            "prompt length",
        )
    )
    if stdout_status and not has_specific_detail:
        return STDOUT_ONLY_BUCKET, tags

    if "max_tokens is too large" in lower:
        return "provider_max_tokens_parameter_too_high", tags
    if "workspace" in lower and "usage limit" in lower and "anthropic" in lower:
        return "anthropic_workspace_api_usage_limit", tags
    if (
        "sum of prompt length" in lower
        or "should not exceed max_num_tokens" in lower
        or "context length" in lower
        or "context_length" in lower
        or "maximum context" in lower
        or "request too large" in lower
    ):
        return "context_or_request_length_exceeded", tags
    if tags & {
        "all OpenRouter keys exhausted",
        "OpenRouter timeout",
        "OpenRouter empty/upstream response",
        "Lewis failed, Joie also failed",
        "native-provider fallback overloaded OpenRouter",
    }:
        return OPENROUTER_BUCKET, tags
    if (
        "proposal remained invalid" in lower
        or "invalid proposal" in lower
        or "proposal validation" in lower
        or "repair exhausted" in lower
    ):
        return "proposal_validation_repair_exhausted", tags
    if "vote" in lower and ("parse" in lower or "failed after" in lower):
        return "vote_recovery_failed_after_retries", tags
    if "api key" in lower or "quota" in lower or "rate limit" in lower:
        return "provider_key_quota_or_rate_limit", tags
    if "empty" in lower or "truncated" in lower:
        return "empty_or_truncated_provider_response", tags
    if any(marker in lower for marker in ("502", "503", "504", "gateway", "server error", "overloaded")):
        return "provider_server_or_gateway_error", tags
    if status.get("state") == "STATUS_PARSE_ERROR":
        return "status_parse_error", tags
    return "unknown_failure", tags


def proposal_for(reason: str) -> str:
    proposals = {
        STDOUT_ONLY_BUCKET: (
            "Patch the runner to write status/log artifacts before every early exit; inspect Slurm stdout for these IDs before "
            "deciding whether to rerun."
        ),
        "context_or_request_length_exceeded": (
            "Reduce transcript/prompt payload for high-N heterogeneous games, summarize older rounds before repair/vote calls, "
            "and cap proposal enumeration text."
        ),
        OPENROUTER_BUCKET: (
            "Keep Lewis-to-Joie rotation, but isolate timeout/empty-response/all-keys-exhausted cases and lower concurrency for "
            "OpenRouter-heavy fallback cohorts."
        ),
        "proposal_validation_repair_exhausted": (
            "Constrain proposal JSON/schema generation earlier and rerun with the stronger JSON repair path now in the codebase."
        ),
        "provider_max_tokens_parameter_too_high": "Clamp provider-specific max_tokens before dispatch.",
        "anthropic_workspace_api_usage_limit": "Route these away from Anthropic until workspace limits reset or use fallback keys.",
        "provider_key_quota_or_rate_limit": "Separate key/quota failures from model failures and retry after rotating credentials.",
        "empty_or_truncated_provider_response": "Retry with stricter response validation and shorter payloads.",
        "provider_server_or_gateway_error": "Retry with exponential backoff and reduce concurrent calls to that provider.",
        "vote_recovery_failed_after_retries": "Apply the JSON repair path to vote parsing and add transcript-local validation.",
        "unknown_failure": "Open the per-config attempt log and Slurm stdout; add a classifier once the repeated signature is clear.",
    }
    return proposals.get(reason, proposals["unknown_failure"])


def collect_snapshot() -> dict[str, Any]:
    selected: dict[str, list[int]] = {}
    slice_ids: dict[str, set[int]] = {}
    configs_by_scope: dict[str, dict[int, dict[str, Any]]] = {}

    for scope, paths in SCOPE_SELECTIONS.items():
        ids: list[int] = []
        for path in paths:
            ids.extend(read_config_ids(path))
        selected[scope] = ids
        slice_ids[scope] = set(ids)
        configs = FG123.load_configs(SCOPE_ROOTS[scope])
        configs_by_scope[scope] = {int(config["config_id"]): config for config in configs}

    active_states = active_slurm_states(slice_ids)
    stdout_maps = {scope: task_maps_for_stdout(root) for scope, root in SCOPE_ROOTS.items()}

    top = Counter()
    scopes: dict[str, Any] = {}
    active_by_job: dict[str, Counter] = defaultdict(Counter)
    openrouter_detail = Counter()
    joie_error_detail = Counter()
    joie_attempt_detail = Counter()
    recovered_from_slurm_stdout = 0
    failure_rows: list[dict[str, Any]] = []

    for scope, ids in selected.items():
        root = SCOPE_ROOTS[scope]
        counts = Counter()
        by_game: dict[str, Counter] = defaultdict(Counter)
        failure_reasons = Counter()
        failure_reasons_by_game: dict[str, Counter] = defaultdict(Counter)
        failure_examples: dict[str, int] = {}

        for config_id in ids:
            config = configs_by_scope[scope][config_id]
            game_label = str(config.get("game_label") or "unknown")
            status_path = root / "status" / f"config_{config_id:04d}.json"
            status = read_json_file(status_path)
            stdout_status = False

            if not status:
                recovered = recover_status_from_stdout(root, config_id, stdout_maps[scope])
                if recovered:
                    status = recovered
                    stdout_status = True
                    recovered_from_slurm_stdout += 1

            active = active_states.get((scope, config_id), {})
            if result_success(config):
                state = "SUCCEEDED"
            elif active.get("state") == "RUNNING":
                state = "RUNNING"
            elif active.get("state") == "QUEUED":
                state = "QUEUED"
            elif (status.get("state") or "").upper() == "RUNNING":
                state = "RUNNING"
            elif (status.get("state") or "").upper() in {
                "FAILED",
                "TIMEOUT",
                "CANCELLED",
                "NODE_FAIL",
                "OUT_OF_MEMORY",
            }:
                state = "FAILED"
            elif (status.get("state") or "").upper() == "STATUS_PARSE_ERROR":
                state = "ERRORS"
            else:
                state = "QUEUED"

            counts[state] += 1
            top[state] += 1
            by_game[game_label][state] += 1

            job_id = active.get("job_id")
            if active:
                active_by_job[f"{scope}:{job_id}"][state] += 1
            elif state == "RUNNING":
                job_id, _ = status_job(status)
                if job_id:
                    active_by_job[f"{scope}:{job_id}"][state] += 1

            if state == "FAILED":
                text = detail_text(root, config_id, status, stdout_status)
                reason, tags = classify_failure(status, text, stdout_status)
                failure_reasons[reason] += 1
                failure_reasons_by_game[game_label][reason] += 1
                failure_examples.setdefault(reason, config_id)
                for tag in tags:
                    openrouter_detail[tag] += 1
                for evidence in joie_error_evidence(text):
                    joie_error_detail[evidence] += 1
                for evidence in joie_attempt_evidence(text):
                    joie_attempt_detail[evidence] += 1
                failure_rows.append(
                    {
                        "scope": scope,
                        "config_id": config_id,
                        "game_label": game_label,
                        "reason": reason,
                        "openrouter_tags": sorted(tags),
                        "stdout_status": stdout_status,
                        "status_path_exists": status_path.exists(),
                        "slurm_stdout": status.get("_recovered_from_slurm_stdout"),
                    }
                )

        scopes[scope] = {
            "sample_count": len(ids),
            "counts": dict(sorted(counts.items())),
            "by_game": {game: dict(sorted(counter.items())) for game, counter in sorted(by_game.items())},
            "failure_reasons": dict(failure_reasons.most_common()),
            "failure_reasons_by_game": {
                game: dict(counter.most_common()) for game, counter in sorted(failure_reasons_by_game.items())
            },
            "failure_examples": failure_examples,
        }

    snapshot = {
        "at": now_local().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "top": {state: int(top.get(state, 0)) for state in STATE_ORDER},
        "scopes": scopes,
        "active_by_job": {key: dict(counter) for key, counter in sorted(active_by_job.items())},
        "openrouter_rotation_detail": dict(sorted(openrouter_detail.items())),
        "openrouter_joie_error_detail": dict(joie_error_detail.most_common()),
        "openrouter_joie_attempt_evidence": dict(joie_attempt_detail.most_common()),
        "recovered_from_slurm_stdout": recovered_from_slurm_stdout,
        "failure_rows": failure_rows,
    }
    return snapshot


def pct(count: int, total: int) -> str:
    return f"{(100.0 * count / total):.1f}%" if total else "0.0%"


def format_counts(counts: dict[str, int], total: int) -> list[str]:
    return [f"- {STATE_LABELS[label]}: {int(counts.get(label, 0))} / {total}" for label in STATE_ORDER]


def format_game_label(game_label: str) -> str:
    match = re.fullmatch(r"game(\d+)", game_label)
    if match:
        return f"Game {match.group(1)}"
    return game_label


def format_active_job(job_key: str) -> str:
    scope, _, job_id = job_key.partition(":")
    label = "Homogeneous" if scope == "homogeneous" else "Heterogeneous"
    if job_id == "7620669":
        return f"{label} requeue {job_id}"
    if scope == "heterogeneous":
        return f"Status-reported hetero rerun {job_id}"
    return f"{label} job {job_id}"


def format_report(snapshot: dict[str, Any], snapshot_path: Path) -> str:
    lines: list[str] = []
    top_total = sum(int(snapshot["scopes"][scope]["sample_count"]) for scope in ("homogeneous", "heterogeneous"))
    lines.append(f"**Backfill Health Snapshot - {snapshot['at']}**")
    lines.append(f"Snapshot: `{snapshot_path}`")
    lines.append("")
    lines.append("**Top Level, 822 Samples**")
    lines.append("- Sample split: homogeneous 241 + heterogeneous 581 = 822")
    lines.extend(format_counts(snapshot["top"], top_total))
    lines.append("")

    for scope in ("homogeneous", "heterogeneous"):
        scope_payload = snapshot["scopes"][scope]
        total = int(scope_payload["sample_count"])
        title = "Homogeneous" if scope == "homogeneous" else "Heterogeneous"
        lines.append(f"**{title}, {total} Samples**")
        lines.extend(format_counts(scope_payload["counts"], total))
        lines.append("By game:")
        for game, counts in scope_payload["by_game"].items():
            lines.append(
                f"- {format_game_label(game)}: {int(counts.get('SUCCEEDED', 0))} succeeded, "
                f"{int(counts.get('FAILED', 0))} failed, {int(counts.get('RUNNING', 0))} running, "
                f"{int(counts.get('QUEUED', 0))} queued, {int(counts.get('ERRORS', 0))} errors"
            )
        lines.append("")

    failed_total = int(snapshot["top"].get("FAILED", 0))
    lines.append(f"**Failure Modes, {failed_total} Current Failures**")
    if failed_total:
        combined = Counter()
        for scope in ("homogeneous", "heterogeneous"):
            combined.update(snapshot["scopes"][scope].get("failure_reasons", {}))
        for reason, count in combined.most_common():
            example = None
            for scope in ("homogeneous", "heterogeneous"):
                example = snapshot["scopes"][scope].get("failure_examples", {}).get(reason)
                if example is not None:
                    break
            lines.append(f"- {reason}: {count} / {failed_total}, {pct(count, failed_total)}")
            lines.append(f"  Proposed fix: {proposal_for(reason)}")
            if example is not None:
                lines.append(f"  Example config: {example}")
    else:
        lines.append("- None.")
    lines.append("")

    lines.append("**OpenRouter Rotation Detail**")
    lines.append("These are evidence tags from available transcript/log text and can overlap.")
    detail = snapshot.get("openrouter_rotation_detail") or {}
    if detail:
        preferred = [
            "Lewis failed, Joie succeeded later",
            "Lewis failed, Joie also failed",
            "all OpenRouter keys exhausted",
            "OpenRouter timeout",
            "OpenRouter empty/upstream response",
            "native-provider fallback overloaded OpenRouter",
        ]
        for key in preferred:
            lines.append(f"- {key}: {int(detail.get(key, 0))}")
        for key, value in sorted(detail.items()):
            if key not in preferred:
                lines.append(f"- {key}: {int(value)}")
    else:
        lines.append("- No OpenRouter-specific evidence in current terminal failures.")
    lines.append("")

    lines.append("**Joie Key Evidence**")
    attempt_detail = snapshot.get("openrouter_joie_attempt_evidence") or {}
    error_detail = snapshot.get("openrouter_joie_error_detail") or {}
    if attempt_detail:
        lines.append("Attempt evidence, counted once per failed config when present:")
        for key, value in attempt_detail.items():
            lines.append(f"- {key}: {int(value)}")
    else:
        lines.append("Attempt evidence: none in current terminal failures.")
    if error_detail:
        lines.append("Joie-side terminal error evidence, counted once per failed config when present:")
        for key, value in error_detail.items():
            lines.append(f"- {key}: {int(value)}")
    else:
        lines.append("Joie-side terminal error evidence: none in current terminal failures.")
    lines.append("")

    lines.append("**Active Slurm Work**")
    active = snapshot.get("active_by_job") or {}
    if active:
        for job_key, counts in active.items():
            pretty = ", ".join(f"{state.lower()} {count}" for state, count in sorted(counts.items()))
            lines.append(f"- {format_active_job(job_key)}: {pretty}")
    else:
        lines.append("- No mapped active Slurm work remains for the 822-sample slice.")
    lines.append("")

    lines.append("**Readout**")
    hom = snapshot["scopes"]["homogeneous"]["counts"]
    het = snapshot["scopes"]["heterogeneous"]["counts"]
    if int(hom.get("FAILED", 0)) == 0 and int(hom.get("RUNNING", 0)) == 0 and int(hom.get("QUEUED", 0)) == 0:
        lines.append("- Homogeneous is complete and clean: all 241 / 241 succeeded.")
    lines.append(
        f"- Heterogeneous is the remaining work: {int(het.get('SUCCEEDED', 0))} succeeded, "
        f"{int(het.get('FAILED', 0))} failed, {int(het.get('RUNNING', 0))} running, "
        f"{int(het.get('QUEUED', 0))} queued."
    )
    lines.append(f"- Recovered stdout-only terminal statuses in this snapshot: {snapshot['recovered_from_slurm_stdout']}.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print only the JSON snapshot path.")
    args = parser.parse_args()

    snapshot = collect_snapshot()
    stamp = now_local().strftime("%Y%m%d_%H%M%S")
    snapshot_path = PROJECT_ROOT / "experiments/results" / f"monitor_822_snapshot_openrouter_detail_recovered_{stamp}.json"
    snapshot_path.write_text(json.dumps(snapshot, indent=2, default=str) + "\n", encoding="utf-8")
    if args.json:
        print(snapshot_path)
    else:
        print(format_report(snapshot, snapshot_path.relative_to(PROJECT_ROOT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

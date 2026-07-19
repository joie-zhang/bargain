#!/usr/bin/env python3
"""Monitor the random-monoculture derisk Slurm array."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ERROR_STATES = {
    "FAILED",
    "FAILED_VALIDATION",
    "TIMEOUT",
    "CANCELLED",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
}


def run_command(cmd: str) -> str:
    return subprocess.run(
        cmd,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    ).stdout


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def fmt_age(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def parse_squeue(job_id: str) -> tuple[dict[str, dict[str, str]], str]:
    out = run_command(f"squeue -j {shlex.quote(job_id)} -o '%i|%T|%M|%D|%R'")
    rows: dict[str, dict[str, str]] = {}
    for line in out.splitlines()[1:]:
        parts = line.split("|")
        if len(parts) >= 5:
            rows[parts[0]] = {
                "state": parts[1],
                "elapsed": parts[2],
                "nodes": parts[3],
                "reason": parts[4],
            }
    return rows, out


def parse_sacct(job_id: str) -> tuple[dict[str, dict[str, str]], str]:
    out = run_command(
        "sacct -j "
        + shlex.quote(job_id)
        + " --format=JobID,State,ExitCode,Elapsed,Start,End -P"
    )
    rows: dict[str, dict[str, str]] = {}
    for line in out.splitlines()[1:]:
        parts = line.split("|")
        if len(parts) >= 6 and "." not in parts[0]:
            rows[parts[0]] = {
                "state": parts[1],
                "exit_code": parts[2],
                "elapsed": parts[3],
                "start": parts[4],
                "end": parts[5],
            }
    return rows, out


def compressed_array_contains(job_key: str, job_id: str, task_id: str) -> bool:
    prefix = f"{job_id}_["
    if not job_key.startswith(prefix) or not job_key.endswith("]"):
        return False
    task_value = safe_int(task_id)
    if task_value is None:
        return False
    body = job_key[len(prefix) : -1]
    body = body.split("%", 1)[0]
    for part in body.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = safe_int(start_text)
            end = safe_int(end_text)
            if start is not None and end is not None and start <= task_value <= end:
                return True
        elif safe_int(part) == task_value:
            return True
    return False


def array_row_for(
    rows: dict[str, dict[str, str]],
    job_id: str,
    task_id: str,
) -> tuple[str, dict[str, str]]:
    array_job_id = f"{job_id}_{task_id}" if task_id else job_id
    if array_job_id in rows:
        return array_job_id, rows[array_job_id]
    for key, row in rows.items():
        if compressed_array_contains(key, job_id, task_id):
            return key, row
    return array_job_id, {}


def latest_submission(results_root: Path) -> dict[str, Any]:
    submissions = sorted((results_root / "submissions").glob("derisk_*_submission.json"))
    if not submissions:
        raise FileNotFoundError(f"No derisk submission JSON under {results_root / 'submissions'}")
    return read_json(submissions[-1])


def config_number(config_id: str) -> int:
    match = re.search(r"(\d+)$", config_id)
    if not match:
        raise ValueError(f"Cannot parse config number from {config_id!r}")
    return int(match.group(1))


def output_dir_from_status(status: dict[str, Any]) -> Path | None:
    command = status.get("command")
    if not command:
        return None
    try:
        parts = shlex.split(command)
    except ValueError:
        return None
    for idx, part in enumerate(parts):
        if part == "--output-dir" and idx + 1 < len(parts):
            return Path(parts[idx + 1])
    return None


def tail_lines(path: Path, max_lines: int = 80) -> list[str]:
    if not path.exists():
        return []
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-max_lines:]
    except OSError:
        return []


def infer_log_stage(lines: list[str]) -> str:
    text = "\n".join(lines)
    for line in reversed(lines):
        match = re.search(
            r"PROGRESS interaction=(?P<interaction>\d+) round=(?P<round>\d+) "
            r"phase=(?P<phase>\S+) agent=(?P<agent>\S+)",
            line,
        )
        if match:
            return (
                f"round {match.group('round')}; "
                f"phase={match.group('phase')}; "
                f"agent={match.group('agent')}; "
                f"interactions={match.group('interaction')}"
            )
    round_matches = re.findall(r"\b[Rr]ound\s+(\d+)\b", text)
    if "Single Experiment Results" in text:
        return "finalizing result summary"
    if round_matches:
        return f"negotiation round {round_matches[-1]}"
    if "Created 25 items" in text or "Creating game environment" in text:
        return "round 0 setup/model-call phase"
    if "Starting experiment" in text:
        return "experiment startup"
    if not lines:
        return "waiting for log output"
    return "running"


def interesting_error(lines: list[str]) -> str:
    needles = ("Traceback", "Error", "ERROR", "Exception", "failed", "FAILED", "Returncode:")
    hits = [line.strip() for line in lines if any(needle in line for needle in needles)]
    return " | ".join(hits[-4:])


def progress_stage(output_dir: Path | None) -> str | None:
    if output_dir is None:
        return None
    progress = read_json(output_dir / "progress.json")
    if not progress:
        return None
    last = progress.get("last_interaction") or {}
    phase = str(last.get("phase") or "unknown_phase")
    phase_counts = progress.get("phase_counts") or {}
    phase_count = phase_counts.get(phase)
    round_value = last.get("round", progress.get("current_round", "unknown"))
    agent = last.get("agent_id") or "unknown_agent"
    interaction_count = progress.get("interaction_count", "unknown")
    parts = [
        f"round {round_value}",
        f"phase={phase}",
        f"agent={agent}",
        f"interactions={interaction_count}",
    ]
    if phase_count is not None:
        parts.append(f"phase_count={phase_count}")
    return "; ".join(parts)


def proxy_queue_summary(proxy_dir: Path) -> dict[str, Any]:
    now = time.time()
    by_model: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "oldest_age_seconds": None, "newest_age_seconds": None}
    )
    total = 0
    for path in proxy_dir.glob("request_*.json"):
        if path.parent.name == "processed":
            continue
        try:
            age = now - path.stat().st_mtime
        except FileNotFoundError:
            continue
        total += 1
        model = "unknown"
        try:
            payload = json.loads(path.read_text(encoding="utf-8")).get("payload") or {}
            model = str(payload.get("model") or payload.get("model_id") or "unknown")
        except Exception:
            pass
        row = by_model[model]
        row["count"] += 1
        row["oldest_age_seconds"] = max(row["oldest_age_seconds"] or 0, age)
        newest = row["newest_age_seconds"]
        row["newest_age_seconds"] = age if newest is None else min(newest, age)

    heartbeat = read_json(proxy_dir / "monitor_heartbeat.json")
    heartbeat_age = None
    if heartbeat.get("updated_at_unix") is not None:
        try:
            heartbeat_age = now - float(heartbeat["updated_at_unix"])
        except (TypeError, ValueError):
            heartbeat_age = None
    return {
        "total_pending_requests": total,
        "by_model": dict(sorted(by_model.items())),
        "heartbeat_updated_at": heartbeat.get("updated_at_iso"),
        "heartbeat_age_seconds": heartbeat_age,
    }


def proxy_model_hint(config: dict[str, Any]) -> str:
    model = str(config.get("monoculture_model") or (config.get("models") or [""])[0])
    mapping = {
        "claude-3-haiku-20240307": "anthropic/claude-3-haiku",
        "gpt-5-nano-high": "openai/gpt-5-nano",
        "qwen3-max-preview": "qwen/qwen3-max",
        "claude-opus-4-5-20251101": "anthropic/claude-opus-4.5",
        "claude-opus-4-5-20251101-thinking-32k": "anthropic/claude-opus-4.5",
        "claude-opus-4-6": "anthropic/claude-opus-4.6",
        "gemini-3.1-pro": "google/gemini-3.1-pro-preview",
        "gpt-4o-2024-05-13": "openai/gpt-4o-2024-05-13",
        "o3-mini-high": "openai/o3-mini",
        "gpt-5.2-chat-latest-20260210": "openai/gpt-5.2-chat",
        "gpt-5.4-high": "openai/gpt-5.4",
        "amazon-nova-pro-v1.0": "amazon/nova-pro-v1",
        "amazon-nova-micro-v1.0": "amazon/nova-micro-v1",
        "deepseek-v3": "deepseek/deepseek-chat",
        "deepseek-r1-0528": "deepseek/deepseek-r1-0528",
    }
    return mapping.get(model, model)


def safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def round_sort_key(label: Any) -> tuple[int, Any]:
    value = safe_int(label)
    if value is not None:
        return (0, value)
    return (1, str(label))


def completed_rollout_summary(config: dict[str, Any], status: dict[str, Any]) -> str:
    result_path = Path(str(status.get("result_path") or ""))
    if not result_path.exists():
        return f"health=WARN missing result file: {result_path}"

    payload = read_json(result_path)
    if not payload:
        return f"health=WARN unreadable result file: {result_path}"

    result_config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    game = result_config.get("game_label") or config.get("game_label") or result_config.get("game_type")
    game_type = result_config.get("game_type") or config.get("game_type") or "unknown_game"
    max_rounds = (
        safe_int(result_config.get("max_rounds"))
        or safe_int(result_config.get("t_rounds"))
        or safe_int(config.get("max_rounds"))
        or safe_int(config.get("t_rounds"))
    )
    n_agents = (
        safe_int(result_config.get("n_agents"))
        or safe_int(result_config.get("num_agents"))
        or safe_int(config.get("n_agents"))
        or safe_int(config.get("num_agents"))
    )
    discussion_turns = safe_int(result_config.get("discussion_turns")) or safe_int(
        config.get("discussion_turns")
    )
    final_round = safe_int(payload.get("final_round"))
    consensus = bool(payload.get("consensus_reached"))
    exploitation = bool(payload.get("exploitation_detected"))

    logs = payload.get("conversation_logs") or []
    if not isinstance(logs, list):
        logs = []

    round_phase_counts: dict[Any, dict[str, int]] = {}
    for entry in logs:
        if not isinstance(entry, dict):
            continue
        round_label = entry.get("round", "?")
        phase = str(entry.get("phase") or "unknown")
        phase_counts = round_phase_counts.setdefault(round_label, {})
        phase_counts[phase] = phase_counts.get(phase, 0) + 1

    round_parts = []
    for round_label in sorted(round_phase_counts, key=round_sort_key):
        phase_counts = round_phase_counts[round_label]
        phases = ",".join(f"{phase}={count}" for phase, count in phase_counts.items())
        round_parts.append(f"r{round_label}[{phases}]")
    rounds_text = ";".join(round_parts) if round_parts else "none"

    final_utilities = payload.get("final_utilities") or {}
    numeric_utilities = [
        value
        for value in (safe_float(v) for v in final_utilities.values())
        if value is not None
    ] if isinstance(final_utilities, dict) else []
    if numeric_utilities:
        utilities_text = (
            f"utilities agents={len(numeric_utilities)} "
            f"sum={sum(numeric_utilities):.1f} "
            f"min={min(numeric_utilities):.1f} "
            f"max={max(numeric_utilities):.1f}"
        )
    else:
        utilities_text = "utilities unavailable"

    strategic = payload.get("strategic_behaviors") or {}
    strategic_parts = []
    if isinstance(strategic, dict):
        for key in (
            "discussion_message_count",
            "pledge_submission_count",
            "feedback_message_count",
            "commit_vote_count",
            "unanimous_commit_round_count",
        ):
            if key in strategic:
                strategic_parts.append(f"{key}={strategic[key]}")
    strategic_text = ",".join(strategic_parts) if strategic_parts else "none"

    vote_integrity = payload.get("vote_integrity") or {}
    if isinstance(vote_integrity, dict):
        vote_events = vote_integrity.get("events") or []
        vote_text = (
            f"synthetic={bool(vote_integrity.get('synthetic_vote_used'))}"
            f"/{vote_integrity.get('synthetic_vote_count', 0)} "
            f"contaminated={bool(vote_integrity.get('contaminated'))} "
            f"hard_failed={bool(vote_integrity.get('hard_failed'))} "
            f"events={len(vote_events) if isinstance(vote_events, list) else 'n/a'}"
        )
    else:
        vote_text = "unavailable"

    warnings = []
    if not logs:
        warnings.append("no conversation logs")
    if final_round is None:
        warnings.append("missing final_round")
    elif max_rounds is not None and (final_round < 0 or final_round > max_rounds):
        warnings.append(f"final_round {final_round} outside 0..{max_rounds}")
    if consensus and final_round is not None and final_round not in round_phase_counts:
        warnings.append(f"consensus final_round r{final_round} absent from logs")
    if consensus and final_round is not None:
        terminal_phases = round_phase_counts.get(final_round, {})
        has_terminal_vote = any(
            "vote" in phase or "tabulation" in phase or "commit" in phase
            for phase in terminal_phases
        )
        if not has_terminal_vote:
            warnings.append(f"consensus round r{final_round} has no terminal vote/commit phase")
    if n_agents is not None and discussion_turns is not None:
        expected_discussion = n_agents * discussion_turns
        for round_label, phase_counts in round_phase_counts.items():
            observed_discussion = phase_counts.get("discussion")
            if observed_discussion is not None and observed_discussion != expected_discussion:
                warnings.append(
                    f"r{round_label} discussion count {observed_discussion}!={expected_discussion}"
                )
    if n_agents is not None:
        for round_label, phase_counts in round_phase_counts.items():
            observed_proposals = phase_counts.get("proposal")
            if observed_proposals is not None and observed_proposals != n_agents:
                warnings.append(f"r{round_label} proposal count {observed_proposals}!={n_agents}")
    if isinstance(vote_integrity, dict):
        if vote_integrity.get("synthetic_vote_used"):
            warnings.append("synthetic vote used")
        if vote_integrity.get("contaminated"):
            warnings.append("vote contaminated")
        if vote_integrity.get("hard_failed"):
            warnings.append("vote hard_failed")

    health = "OK" if not warnings else "WARN " + "; ".join(warnings[:5])
    if len(warnings) > 5:
        health += f"; +{len(warnings) - 5} more"

    max_round_text = str(max_rounds) if max_rounds is not None else "?"
    final_round_text = str(final_round) if final_round is not None else "?"
    return (
        f"health={health}; game={game}/{game_type}; consensus={consensus}; "
        f"final_round={final_round_text}/{max_round_text}; logs={len(logs)}; "
        f"rounds={rounds_text}; {utilities_text}; exploitation={exploitation}; "
        f"vote_integrity={vote_text}; strategic={strategic_text}"
    )


def model_set_summary(config: dict[str, Any]) -> str:
    agent_model_map = config.get("agent_model_map")
    if isinstance(agent_model_map, dict) and agent_model_map:
        counts = Counter(str(model) for model in agent_model_map.values())
    else:
        models = config.get("models")
        if isinstance(models, list) and models:
            counts = Counter(str(model) for model in models)
        else:
            model = config.get("monoculture_model") or config.get("baseline_model") or "unknown"
            n_agents = safe_int(config.get("n_agents")) or safe_int(config.get("num_agents")) or 1
            counts = Counter({str(model): n_agents})
    return ",".join(f"{model}x{count}" for model, count in sorted(counts.items()))


def config_summary(config: dict[str, Any]) -> str:
    keys = [
        "config_id",
        "game_label",
        "game_type",
        "n_agents",
        "num_agents",
        "max_rounds",
        "t_rounds",
        "discussion_turns",
        "competition_level",
        "random_seed",
        "m_items",
        "n_issues",
        "m_projects",
        "rho",
        "theta",
        "alpha",
        "sigma",
        "cofunding_enable_commit_vote",
        "monoculture_model",
    ]
    parts = []
    seen = set()
    for key in keys:
        if key in seen or key not in config:
            continue
        value = config.get(key)
        if value is None:
            continue
        parts.append(f"{key}={value}")
        seen.add(key)
    return ";".join(parts)


def classify(state: str, queue_state: str, acct_state: str) -> str:
    if state == "SUCCESS":
        return "completed"
    if state == "RUNNING" or queue_state == "RUNNING" or acct_state == "RUNNING":
        return "in_progress"
    if state in ERROR_STATES or acct_state in ERROR_STATES:
        return "errored"
    return "queued"


def has_valid_result(status: dict[str, Any]) -> bool:
    result_raw = str(status.get("result_path") or "").strip()
    if not result_raw:
        return False
    result_path = Path(result_raw)
    if not result_path.exists():
        return False
    return status.get("result_validation_error") in (None, "", False)


def status_for_registered_attempt(status: dict[str, Any], backfill: dict[str, Any]) -> dict[str, Any]:
    record = backfill.get("record") or {}
    job_id = str(record.get("job_id") or "")
    task_id = str(record.get("array_task_id") or "")
    if not job_id:
        return status
    for attempt in status.get("attempts") or []:
        if not isinstance(attempt, dict):
            continue
        attempt_job = str(attempt.get("slurm_job_id") or "")
        attempt_array = str(attempt.get("slurm_array_job_id") or "")
        attempt_task = str(attempt.get("slurm_array_task_id") or "")
        if job_id not in {attempt_job, attempt_array}:
            continue
        if task_id and attempt_task and task_id != attempt_task:
            continue
        merged = dict(status)
        for key in (
            "attempt_id",
            "attempt_log_path",
            "log_path",
            "started_at",
            "finished_at",
            "duration_seconds",
            "result_path",
            "result_validation_error",
            "returncode",
        ):
            if key in attempt:
                merged[key] = attempt[key]
        merged["_prefer_attempt_log_stage"] = True
        return merged
    return status


def stage_progress(
    *,
    category: str,
    config: dict[str, Any],
    status: dict[str, Any],
    queue_row: dict[str, str],
    acct_row: dict[str, str],
    proxy_summary: dict[str, Any],
) -> str:
    if category == "queued":
        reason = queue_row.get("reason") or acct_row.get("state") or "not started"
        return f"queued: {reason}"
    if category == "completed":
        duration = status.get("duration_seconds")
        result_path = status.get("result_path") or ""
        return f"success; duration={fmt_age(float(duration)) if duration is not None else 'n/a'}; result={result_path}"

    log_path = Path(status.get("attempt_log_path") or status.get("log_path") or "")
    lines = tail_lines(log_path)
    now = time.time()
    log_age = None
    log_size = None
    if log_path.exists():
        stat = log_path.stat()
        log_age = now - stat.st_mtime
        log_size = stat.st_size
    out_dir = output_dir_from_status(status)
    output_files = 0
    newest_output_age = None
    if out_dir and out_dir.exists():
        files = [path for path in out_dir.rglob("*") if path.is_file()]
        output_files = len(files)
        if files:
            newest_output_age = now - max(path.stat().st_mtime for path in files)
    if status.get("_prefer_attempt_log_stage"):
        stage = infer_log_stage(lines) or progress_stage(out_dir)
    else:
        stage = progress_stage(out_dir) or infer_log_stage(lines)

    model_hint = proxy_model_hint(config)
    proxy_by_model = proxy_summary.get("by_model", {})
    proxy_row = proxy_by_model.get(model_hint) or {}
    proxy_text = ""
    if proxy_row.get("count"):
        proxy_text = (
            f"; proxy pending={proxy_row['count']} for {model_hint}, "
            f"oldest={fmt_age(proxy_row.get('oldest_age_seconds'))}"
        )
    heartbeat_age = proxy_summary.get("heartbeat_age_seconds")
    if heartbeat_age is not None and heartbeat_age > 300:
        proxy_text += f"; proxy heartbeat stale={fmt_age(heartbeat_age)}"

    if category == "errored":
        detail = status.get("result_validation_error") or interesting_error(lines)
        return f"error at {stage}; exit={acct_row.get('exit_code') or status.get('returncode')}; {detail}"

    return (
        f"{stage}; slurm_elapsed={acct_row.get('elapsed') or queue_row.get('elapsed') or 'n/a'}; "
        f"log_size={log_size if log_size is not None else 'n/a'}; "
        f"log_idle={fmt_age(log_age)}; output_files={output_files}; "
        f"output_idle={fmt_age(newest_output_age)}{proxy_text}"
    )


def load_backfill_states(backfill_file: Path | None) -> dict[str, dict[str, Any]]:
    if backfill_file is None:
        return {}
    data = read_json(backfill_file)
    states: dict[str, dict[str, Any]] = {}
    for config_id, record in data.items():
        if not isinstance(record, dict):
            continue
        job_id = str(record.get("job_id") or "")
        task_id = str(record.get("array_task_id") or "")
        if not job_id:
            continue
        q_rows, _ = parse_squeue(job_id)
        a_rows, _ = parse_sacct(job_id)
        array_job_id, queue_row = array_row_for(q_rows, job_id, task_id)
        acct_job_id, acct_row = array_row_for(a_rows, job_id, task_id)
        if not queue_row and acct_job_id:
            array_job_id = acct_job_id
        states[config_id] = {
            "record": record,
            "array_job_id": array_job_id,
            "queue": queue_row,
            "acct": acct_row,
        }
    return states


def apply_backfill_category(
    *,
    category: str,
    config_id: str,
    backfill_states: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    backfill = backfill_states.get(config_id) or {}
    queue_state = (backfill.get("queue") or {}).get("state", "")
    acct_state = (backfill.get("acct") or {}).get("state", "")
    if category == "errored":
        if queue_state == "RUNNING" or acct_state == "RUNNING":
            return "in_progress", backfill
        if queue_state == "PENDING" or acct_state == "PENDING":
            return "queued", backfill
    return category, backfill


def classify_backfill(
    *,
    status_state: str,
    queue_state: str,
    acct_state: str,
) -> str | None:
    if status_state == "SUCCESS":
        return "completed"
    if queue_state == "RUNNING" or acct_state == "RUNNING":
        return "in_progress"
    if queue_state == "PENDING" or acct_state == "PENDING":
        return "queued"
    if acct_state in ERROR_STATES:
        return "errored"
    return None


def build_snapshot(
    results_root: Path,
    job_id: str,
    task_file: Path,
    proxy_dir: Path,
    backfill_file: Path | None = None,
) -> dict[str, Any]:
    config_ids = [line.strip() for line in task_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    squeue, squeue_raw = parse_squeue(job_id)
    sacct, sacct_raw = parse_sacct(job_id)
    proxy = proxy_queue_summary(proxy_dir)
    backfill_states = load_backfill_states(backfill_file)
    rows = []
    counts: Counter[str] = Counter()

    for idx, config_id in enumerate(config_ids, start=1):
        config = read_json(results_root / "configs" / f"{config_id}.json")
        num = config_number(config_id)
        status = read_json(results_root / "status" / f"config_{num:04d}.json")
        state = str(status.get("state") or "NOT_STARTED")
        if has_valid_result(status):
            state = "SUCCESS"
        array_id = f"{job_id}_{idx}"
        queue_row = squeue.get(array_id, {})
        acct_row = sacct.get(array_id, {})
        effective_queue_row = queue_row
        effective_acct_row = acct_row
        category = classify(state, queue_row.get("state", ""), acct_row.get("state", ""))
        category, backfill = apply_backfill_category(
            category=category,
            config_id=config_id,
            backfill_states=backfill_states,
        )
        if backfill:
            backfill_queue = backfill.get("queue") or {}
            backfill_acct = backfill.get("acct") or {}
            if backfill_queue or backfill_acct:
                status = status_for_registered_attempt(status, backfill)
                effective_queue_row = backfill_queue
                effective_acct_row = backfill_acct
                category = classify_backfill(
                    status_state=state,
                    queue_state=effective_queue_row.get("state", ""),
                    acct_state=effective_acct_row.get("state", ""),
                ) or classify(
                    state,
                    effective_queue_row.get("state", ""),
                    effective_acct_row.get("state", ""),
                )
                category, backfill = apply_backfill_category(
                    category=category,
                    config_id=config_id,
                    backfill_states=backfill_states,
                )
        counts[category] += 1
        row = {
            "task": idx,
            "config_id": config_id,
            "category": category,
            "status_state": state,
            "slurm_queue_state": effective_queue_row.get("state", ""),
            "slurm_acct_state": effective_acct_row.get("state", ""),
            "exit_code": effective_acct_row.get("exit_code") or status.get("returncode") or "",
            "game": config.get("game_label"),
            "n_agents": config.get("n_agents"),
            "model": config.get("monoculture_model") or (config.get("models") or [""])[0],
            "model_set": model_set_summary(config),
            "config_summary": config_summary(config),
            "started_at": status.get("started_at", ""),
            "finished_at": status.get("finished_at", ""),
            "duration_seconds": status.get("duration_seconds", ""),
            "result_path": status.get("result_path", ""),
            "result_validation_error": status.get("result_validation_error"),
            "attempt_log_path": status.get("attempt_log_path", ""),
            "backfill_job_id": (backfill.get("record") or {}).get("job_id", ""),
        }
        if backfill and state == "FAILED" and category in {"queued", "in_progress"}:
            backfill_queue = backfill.get("queue") or {}
            backfill_acct = backfill.get("acct") or {}
            backfill_id = backfill.get("array_job_id")
            if category == "queued":
                reason = backfill_queue.get("reason") or backfill_acct.get("state") or "pending"
                row["stage_progress"] = f"backfill queued: {backfill_id}; {reason}"
            else:
                row["stage_progress"] = (
                    f"backfill running: {backfill_id}; "
                    f"slurm_elapsed={backfill_acct.get('elapsed') or backfill_queue.get('elapsed') or 'n/a'}"
                )
        else:
            stage_text = stage_progress(
                category=category,
                config=config,
                status=status,
                queue_row=effective_queue_row,
                acct_row=effective_acct_row,
                proxy_summary=proxy,
            )
            if backfill and category in {"queued", "in_progress"}:
                stage_text = f"backfill {backfill.get('array_job_id')}; {stage_text}"
            row["stage_progress"] = stage_text
        if category == "completed":
            row["completed_rollout"] = completed_rollout_summary(config, status)
        rows.append(row)

    completed = counts["completed"]
    errored = counts["errored"]
    snapshot_counts = {
        "completed": completed,
        "in_progress": counts["in_progress"],
        "errored": errored,
        "queued": counts["queued"],
        "need_backfilled": errored,
    }
    return {
        "results_root": str(results_root),
        "job_id": job_id,
        "task_file": str(task_file),
        "counts": snapshot_counts,
        "success_count": completed,
        "rows": rows,
        "proxy": proxy,
        "squeue": squeue_raw,
        "sacct": sacct_raw,
    }


def print_text(snapshot: dict[str, Any]) -> None:
    counts = snapshot["counts"]
    print(
        "COUNTS completed={completed} in_progress={in_progress} errored={errored} "
        "queued={queued} need_backfilled={need_backfilled}".format(**counts)
    )
    proxy = snapshot["proxy"]
    print(
        "PROXY pending_requests={} heartbeat_age={}".format(
            proxy["total_pending_requests"],
            fmt_age(proxy.get("heartbeat_age_seconds")),
        )
    )
    if "newly_completed" in snapshot:
        completed = ", ".join(row["config_id"] for row in snapshot["newly_completed"]) or "none"
        errored = ", ".join(row["config_id"] for row in snapshot["newly_errored"]) or "none"
        print(f"NEW newly_completed={completed} newly_errored={errored}")
    print("ROWS task|config|category|status|qstate|acct|exit|game|model|stage_progress")
    for row in snapshot["rows"]:
        print(
            "|".join(
                str(row.get(key) or "")
                for key in (
                    "task",
                    "config_id",
                    "category",
                    "status_state",
                    "slurm_queue_state",
                    "slurm_acct_state",
                    "exit_code",
                    "game",
                    "model",
                    "stage_progress",
                )
            )
        )
    completed_rows = [row for row in snapshot["rows"] if row.get("category") == "completed"]
    if completed_rows:
        print("COMPLETED_ROLLOUTS config|rollout_health")
        for row in completed_rows:
            print(f"{row['config_id']}|{row.get('completed_rollout') or 'unavailable'}")
    active_or_completed_rows = [
        row
        for row in snapshot["rows"]
        if row.get("category") in {"completed", "in_progress"}
    ]
    if active_or_completed_rows:
        print("ACTIVE_COMPLETED_CONFIGS config|category|game|model_set|config_summary")
        for row in active_or_completed_rows:
            print(
                "|".join(
                    str(row.get(key) or "")
                    for key in (
                        "config_id",
                        "category",
                        "game",
                        "model_set",
                        "config_summary",
                    )
                )
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("experiments/results/full_games123_random_monoculture_control_20260628_014357"),
    )
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--task-file", type=Path, default=None)
    parser.add_argument("--proxy-dir", type=Path, default=Path("/home/jz4391/openrouter_proxy"))
    parser.add_argument(
        "--backfill-file",
        type=Path,
        default=None,
        help="Optional JSON mapping config IDs to active backfill Slurm job IDs.",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help="Optional file used to track newly completed/errored configs across monitor runs.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    results_root = args.results_root.resolve()
    submission = latest_submission(results_root)
    job_id = args.job_id or str(submission["job_id"])
    task_file = args.task_file or Path(submission["task_file"])
    backfill_file = args.backfill_file
    if backfill_file is None:
        candidate = results_root / "monitoring" / "derisk_backfill_jobs.json"
        backfill_file = candidate if candidate.exists() else None
    snapshot = build_snapshot(results_root, job_id, task_file, args.proxy_dir, backfill_file)
    if args.state_file is not None:
        previous = read_json(args.state_file)
        previous_completed = set(previous.get("completed") or [])
        previous_errored = set(previous.get("errored") or [])
        current_completed = {
            row["config_id"] for row in snapshot["rows"] if row["category"] == "completed"
        }
        current_errored = {
            row["config_id"] for row in snapshot["rows"] if row["category"] == "errored"
        }
        snapshot["newly_completed"] = [
            row
            for row in snapshot["rows"]
            if row["category"] == "completed" and row["config_id"] not in previous_completed
        ]
        snapshot["newly_errored"] = [
            row
            for row in snapshot["rows"]
            if row["category"] == "errored" and row["config_id"] not in previous_errored
        ]
        args.state_file.parent.mkdir(parents=True, exist_ok=True)
        args.state_file.write_text(
            json.dumps(
                {
                    "completed": sorted(current_completed),
                    "errored": sorted(current_errored),
                    "updated_at_unix": time.time(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(snapshot, indent=2, sort_keys=True))
    else:
        print_text(snapshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

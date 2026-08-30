"""Persistence and export helpers for semantic behavior-label review."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import io
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


VALID_RESPONSES = {"yes", "no", "unsure", "skip"}
REVIEWER_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def validate_reviewer_id(reviewer_id: str) -> str:
    reviewer_id = reviewer_id.strip()
    if not REVIEWER_PATTERN.fullmatch(reviewer_id):
        raise ValueError(
            "Reviewer ID must contain 1 to 64 letters, numbers, periods, underscores, or hyphens"
        )
    return reviewer_id


def latest_decisions(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["reviewer_id"]), str(row["item_id"]))
        latest[key] = row
    return latest


def append_decision(
    journal_path: Path,
    manifest_path: Path,
    item: dict[str, Any],
    reviewer_id: str,
    response: str,
    note: str = "",
) -> dict[str, Any]:
    reviewer_id = validate_reviewer_id(reviewer_id)
    if response not in VALID_RESPONSES:
        raise ValueError(f"Unknown response: {response}")
    record = {
        "schema_version": 1,
        "decision_id": str(uuid.uuid4()),
        "decided_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reviewer_id": reviewer_id,
        "item_id": item["item_id"],
        "reviewer_response": response,
        "reviewer_binary": 1 if response == "yes" else 0 if response == "no" else None,
        "note": note.strip(),
        "sampling_manifest_path": str(manifest_path.resolve()),
        "sampling_manifest_sha256": sha256_file(manifest_path),
    }
    payload = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(journal_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        with os.fdopen(descriptor, "ab", closefd=False) as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        os.close(descriptor)
    return record


def source_record(view: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    source_kind = item["source_kind"]
    source_index = int(item["source_index"])
    if source_kind == "conversation_log":
        rows = view.get("conversation_logs") or []
        matches = [row for row in rows if int(row.get("log_index", -1)) == source_index]
        text_key = "content"
    elif source_kind == "interaction":
        rows = view.get("agent_authored_interactions") or view.get("target_private_interactions") or []
        matches = [row for row in rows if int(row.get("interaction_index", -1)) == source_index]
        text_key = "response"
    else:
        raise ValueError(f"Unsupported source kind: {source_kind}")
    if len(matches) != 1:
        raise ValueError(f"Expected one matching source record, found {len(matches)}")
    text = str(matches[0].get(text_key) or matches[0].get("content") or "")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest != item["source_text_sha256"]:
        raise ValueError("The focused source text does not match the sampling manifest")
    return matches[0]


EXPORT_FIELDS = (
    "decision_id",
    "decided_at_utc",
    "reviewer_id",
    "item_id",
    "reviewer_response",
    "reviewer_binary",
    "included_in_primary_binary_analysis",
    "machine_positive",
    "machine_binary",
    "agrees_with_machine",
    "note",
    "sampling_weight",
    "candidate_count_in_stratum",
    "sample_count_in_stratum",
    "sampling_stratum",
    "dataset",
    "family",
    "level",
    "seed",
    "rollout_id",
    "config_id",
    "game_label",
    "game_cell",
    "order",
    "target_agent",
    "speaker_agent",
    "source_kind",
    "source_index",
    "phase",
    "round",
    "discussion_turn",
    "tag_code",
    "tag_title",
    "tag_category",
    "sampling_manifest_path",
    "sampling_manifest_sha256",
)


def agreement_rows(
    manifest_rows: Iterable[dict[str, Any]],
    decision_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_item = {str(row["item_id"]): row for row in manifest_rows}
    latest = latest_decisions(decision_rows)
    output = []
    for (reviewer_id, item_id), decision in latest.items():
        item = by_item.get(item_id)
        if item is None:
            continue
        human = decision.get("reviewer_binary")
        machine = int(bool(item["machine_positive"]))
        merged = {**item, **decision}
        merged["machine_binary"] = machine
        merged["included_in_primary_binary_analysis"] = human in {0, 1}
        merged["agrees_with_machine"] = (int(human) == machine) if human in {0, 1} else None
        merged["reviewer_id"] = reviewer_id
        output.append(merged)
    return sorted(output, key=lambda row: (row["reviewer_id"], row["item_id"]))


def agreement_csv(rows: Iterable[dict[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=EXPORT_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return stream.getvalue()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


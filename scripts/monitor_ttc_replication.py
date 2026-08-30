#!/usr/bin/env python3
"""Summarize completion and Slurm health for a TTC replication batch."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


BAD_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "TIMEOUT",
}
ACTIVE_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}


def load_submissions(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        row["config_id"] = int(row["config_id"])
    return rows


def accounting(job_ids: List[str]) -> Dict[str, Dict[str, str]]:
    states: Dict[str, Dict[str, str]] = {}
    for offset in range(0, len(job_ids), 80):
        chunk = job_ids[offset : offset + 80]
        completed = subprocess.run(
            [
                "sacct",
                "-j",
                ",".join(chunk),
                "--format=JobIDRaw,JobName,State,Elapsed,ExitCode",
                "-n",
                "-P",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        for line in completed.stdout.splitlines():
            fields = line.split("|")
            if len(fields) < 5 or "." in fields[0]:
                continue
            states[fields[0]] = {
                "name": fields[1],
                "state": fields[2].split()[0],
                "elapsed": fields[3],
                "exit_code": fields[4],
            }
    return states


def result_health(run_root: Path, configs: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    health: Dict[int, Dict[str, Any]] = {}
    for config in configs:
        config_id = int(config["config_id"])
        result_path = Path(config["output_dir"]) / "run_1_experiment_results.json"
        if not result_path.exists():
            continue
        record: Dict[str, Any] = {"path": str(result_path)}
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            vote_integrity = result.get("vote_integrity") or (result.get("config") or {}).get(
                "vote_integrity"
            ) or {}
            record.update(
                {
                    "parse_ok": True,
                    "hard_failed": bool(vote_integrity.get("hard_failed")),
                    "consensus_reached": bool(result.get("consensus_reached")),
                    "final_round": result.get("final_round"),
                }
            )
        except Exception as exc:
            record.update({"parse_ok": False, "error": repr(exc), "hard_failed": True})
        health[config_id] = record
    return health


def scan_logs(run_root: Path, completed_ids: set[int]) -> Dict[str, Any]:
    fatal_ids = set()
    cap_signal_ids = set()
    provider_signal_ids = set()
    examples = []
    fatal_pattern = re.compile(
        r"Traceback \(most recent call last\)|slurmstepd: error:|"
        r"OUT_OF_MEMORY|DUE TO TIME LIMIT|RuntimeError:"
    )
    cap_pattern = re.compile(
        r"maximum output tokens|maximum context|context length exceeded|"
        r"stop_reason.?max_tokens|finish_reason.?length|"
        r"(?:hit|reached).{0,30}(?:token|output).{0,20}limit|"
        r"max_tokens.{0,40}(?:error|exceed)",
        re.IGNORECASE,
    )
    provider_pattern = re.compile(
        r"ProviderKeyExhaustedError|Native .* call failed|falling back to OpenRouter|"
        r"HTTP (?:4|5)\d\d|rate.?limit",
        re.IGNORECASE,
    )
    for path in sorted((run_root / "slurm" / "logs").glob("ttc*_*.*")):
        match = re.search(r"ttc\d+_(\d{4})_", path.name)
        if not match:
            continue
        config_id = int(match.group(1))
        text = path.read_text(encoding="utf-8", errors="replace")
        if fatal_pattern.search(text):
            fatal_ids.add(config_id)
        if cap_pattern.search(text):
            cap_signal_ids.add(config_id)
        if provider_pattern.search(text):
            provider_signal_ids.add(config_id)
        if config_id not in completed_ids and len(examples) < 12:
            for line in text.splitlines():
                if fatal_pattern.search(line) or provider_pattern.search(line):
                    examples.append(
                        {"config_id": config_id, "log": path.name, "line": line[:500]}
                    )
                    break
    return {
        "fatal_log_config_ids": sorted(fatal_ids),
        "fatal_log_without_result_config_ids": sorted(fatal_ids - completed_ids),
        "cap_signal_config_ids": sorted(cap_signal_ids),
        "provider_signal_config_ids": sorted(provider_signal_ids),
        "examples": examples,
    }


def summarize(run_root: Path) -> Dict[str, Any]:
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    configs = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((run_root / "configs").glob("config_*.json"))
    ]
    submissions = load_submissions(run_root / "slurm" / "submitted_jobs.tsv")
    state_by_job = accounting([row["job_id"] for row in submissions])
    attempts_by_config: Dict[int, List[Dict[str, str]]] = {}
    for submission in submissions:
        state = state_by_job.get(
            submission["job_id"],
            {"name": "", "state": "UNKNOWN", "elapsed": "", "exit_code": ""},
        )
        attempt = dict(state)
        attempt["job_id"] = submission["job_id"]
        attempts_by_config.setdefault(submission["config_id"], []).append(attempt)

    results = result_health(run_root, configs)
    healthy_ids = {
        config_id
        for config_id, record in results.items()
        if record.get("parse_ok") and not record.get("hard_failed")
    }
    hard_failed_result_ids = {
        config_id
        for config_id, record in results.items()
        if record.get("hard_failed")
    }
    latest_states = {
        config_id: attempts[-1]["state"]
        for config_id, attempts in attempts_by_config.items()
    }
    terminal_failed_ids = {
        config_id
        for config_id, state in latest_states.items()
        if state in BAD_STATES and config_id not in healthy_ids
    }
    active_ids = {
        config_id
        for config_id, state in latest_states.items()
        if state in ACTIVE_STATES and config_id not in healthy_ids
    }
    missing_submission_ids = set(range(1, int(manifest["num_configs"]) + 1)) - set(
        attempts_by_config
    )
    missing_result_ids = set(range(1, int(manifest["num_configs"]) + 1)) - healthy_ids
    log_scan = scan_logs(run_root, set(results))
    report = {
        "run_root": str(run_root),
        "expected_configs": int(manifest["num_configs"]),
        "submission_attempts": len(submissions),
        "submitted_unique_configs": len(attempts_by_config),
        "healthy_results": len(healthy_ids),
        "healthy_config_ids": sorted(healthy_ids),
        "results_present": len(results),
        "hard_failed_result_config_ids": sorted(hard_failed_result_ids),
        "missing_healthy_result_config_ids": sorted(missing_result_ids),
        "missing_submission_config_ids": sorted(missing_submission_ids),
        "active_without_result": len(active_ids),
        "active_without_result_config_ids": sorted(active_ids),
        "terminal_failed_without_healthy_result_config_ids": sorted(terminal_failed_ids),
        "latest_slurm_state_counts": dict(Counter(latest_states.values())),
        "no_consensus_count": sum(
            1
            for config_id, record in results.items()
            if config_id in healthy_ids and not record.get("consensus_reached")
        ),
        "no_consensus_config_ids": sorted(
            config_id
            for config_id, record in results.items()
            if config_id in healthy_ids and not record.get("consensus_reached")
        ),
        "log_scan": log_scan,
    }
    monitoring_dir = run_root / "monitoring"
    monitoring_dir.mkdir(parents=True, exist_ok=True)
    (monitoring_dir / "status_latest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = summarize(args.run_root.resolve())
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

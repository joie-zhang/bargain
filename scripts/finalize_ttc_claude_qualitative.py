#!/usr/bin/env python3
"""Wait for all 648 sources, backfill labels, and build final summaries."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import prepare_ttc_claude_seed_qualitative as prepare


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATUS_PATH = prepare.OUTPUT_ROOT / "finalizer_status.json"
MANIFEST_PATH = prepare.OUTPUT_ROOT / "all_available_rollouts_manifest.jsonl"


def source_counts() -> tuple[int, dict[str, list[int]]]:
    available = 0
    missing: dict[str, list[int]] = {}
    for seed in prepare.SEED_ORDER:
        root = prepare.RESULTS_ROOT / prepare.SEED_ROOT_NAMES[seed]
        seed_missing = []
        for config_id in prepare.SOURCE_CONFIG_IDS:
            config_path = root / "configs" / f"config_{config_id:04d}.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            output_dir = Path(config["output_dir"])
            result_path = output_dir / "run_1_experiment_results.json"
            interactions_path = output_dir / "run_1_all_interactions.json"
            if result_path.exists() and interactions_path.exists():
                available += 1
            else:
                seed_missing.append(config_id)
        missing[str(seed)] = seed_missing
    return available, missing


def write_status(payload: dict) -> None:
    prepare.write_json(STATUS_PATH, payload)
    print(json.dumps(payload, sort_keys=True), flush=True)


def run_checked(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def annotation_counts() -> tuple[int, list[str]]:
    expected_ids = []
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                expected_ids.append(json.loads(line)["rollout_id"])
    missing = [
        rollout_id
        for rollout_id in expected_ids
        if not (
            prepare.OUTPUT_ROOT
            / "judge_outputs"
            / "completions"
            / f"{rollout_id}.json"
        ).exists()
    ]
    return len(expected_ids) - len(missing), missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--wait-hours", type=float, default=10.0)
    parser.add_argument("--judge-concurrency", type=int, default=4)
    parser.add_argument("--judge-passes", type=int, default=8)
    args = parser.parse_args()

    deadline = time.monotonic() + args.wait_hours * 3600
    while True:
        available, missing = source_counts()
        write_status(
            {
                "stage": "waiting_for_sources",
                "available_source_rollouts": available,
                "expected_source_rollouts": prepare.EXPECTED_TOTAL_ROLLOUTS,
                "missing_source_config_ids": missing,
            }
        )
        if available == prepare.EXPECTED_TOTAL_ROLLOUTS:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out with {available}/{prepare.EXPECTED_TOTAL_ROLLOUTS} sources"
            )
        time.sleep(max(1, args.poll_seconds))

    python = str(PROJECT_ROOT / ".venv/bin/python")
    run_checked(
        [
            python,
            str(PROJECT_ROOT / "scripts/prepare_ttc_claude_seed_qualitative.py"),
            "--require-complete",
        ]
    )
    judge_command = [
        python,
        str(PROJECT_ROOT / "scripts/run_ttc_claude_qualitative_judge.py"),
        "--concurrency",
        str(args.judge_concurrency),
        "--transport",
        "openai_proxy",
        "--model",
        "openai/gpt-5.5",
        "--reasoning-effort",
        "xhigh",
        "--max-tokens",
        "65536",
        "--max-attempts",
        "6",
    ]
    for pass_number in range(1, args.judge_passes + 1):
        completed, missing = annotation_counts()
        write_status(
            {
                "stage": "backfilling_annotations",
                "pass": pass_number,
                "completed_annotations": completed,
                "expected_annotations": prepare.EXPECTED_TOTAL_ROLLOUTS,
                "missing_rollout_ids": missing,
            }
        )
        if not missing:
            break
        print("+ " + " ".join(judge_command), flush=True)
        result = subprocess.run(judge_command, cwd=PROJECT_ROOT, check=False)
        completed, missing = annotation_counts()
        write_status(
            {
                "stage": "backfill_pass_complete",
                "pass": pass_number,
                "judge_exit_code": result.returncode,
                "completed_annotations": completed,
                "expected_annotations": prepare.EXPECTED_TOTAL_ROLLOUTS,
                "missing_rollout_ids": missing,
            }
        )
        if not missing:
            break
        time.sleep(5)
    completed, missing = annotation_counts()
    if missing:
        raise RuntimeError(
            f"Backfill exhausted with {completed}/{prepare.EXPECTED_TOTAL_ROLLOUTS} "
            f"annotations; missing={missing}"
        )
    write_status({"stage": "validating_and_summarizing"})
    run_checked(
        [
            python,
            str(PROJECT_ROOT / "scripts/analyze_ttc_claude_seed_qualitative.py"),
            "--require-complete",
        ]
    )
    write_status({"stage": "complete", "rollouts": prepare.EXPECTED_TOTAL_ROLLOUTS})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

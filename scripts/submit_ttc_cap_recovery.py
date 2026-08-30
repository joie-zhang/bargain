#!/usr/bin/env python3
"""Archive a failed TTC attempt and submit its approved 16384-cap recovery."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict


RECOVERY_CAP = 16_384
EXPECTED_ORIGINAL_CAP = 10_500
DEFAULT_REASON = (
    "empty content; finish_reason=length; native_finish_reason=max_output_tokens"
)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def submission_attempts(run_root: Path, config_id: int) -> int:
    path = run_root / "slurm" / "submitted_jobs.tsv"
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(
            1
            for row in csv.DictReader(handle, delimiter="\t")
            if int(row["config_id"]) == config_id
        )


def submit_recovery(run_root: Path, config_id: int, reason: str) -> Dict[str, Any]:
    original_path = run_root / "configs" / f"config_{config_id:04d}.json"
    if not original_path.exists():
        raise RuntimeError(f"Original config does not exist: {original_path}")
    original = json.loads(original_path.read_text(encoding="utf-8"))
    original_cap = int(original["max_tokens_per_phase"])
    if original_cap != EXPECTED_ORIGINAL_CAP:
        raise RuntimeError(
            f"Expected original cap {EXPECTED_ORIGINAL_CAP}, found {original_cap}"
        )
    output_dir = Path(original["output_dir"])
    result_path = output_dir / "run_1_experiment_results.json"
    if result_path.exists():
        raise RuntimeError(f"Refusing to replace an existing result: {result_path}")
    if not output_dir.is_dir():
        raise RuntimeError(f"Failed-attempt output directory is missing: {output_dir}")

    attempt = submission_attempts(run_root, config_id) + 1
    if attempt < 2:
        raise RuntimeError("Recovery requires a recorded original submission")
    archive = (
        run_root
        / "recovery"
        / "failed_attempts"
        / f"config_{config_id:04d}_attempt{attempt - 1}_cap{original_cap}"
    )
    if archive.exists():
        raise RuntimeError(f"Recovery archive already exists: {archive}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(output_dir), str(archive))

    recovery = dict(original)
    recovery["max_tokens_per_phase"] = RECOVERY_CAP
    recovery["recovery_attempt"] = attempt
    recovery["recovery_reason"] = reason
    recovery_path = (
        run_root
        / "recovery"
        / "configs"
        / f"config_{config_id:04d}_cap{RECOVERY_CAP}_attempt{attempt}.json"
    )
    if recovery_path.exists():
        raise RuntimeError(f"Recovery config already exists: {recovery_path}")
    write_json(recovery_path, recovery)

    slurm_script = run_root / "slurm" / "run_one.sbatch"
    seed = int(original["random_seed"])
    completed = subprocess.run(
        [
            "sbatch",
            "--parsable",
            f"--job-name=ttc{seed}_{config_id:04d}bf{attempt}",
            str(slurm_script),
            str(recovery_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    job_id = completed.stdout.strip()
    submitted_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")

    with (run_root / "slurm" / "submitted_jobs.tsv").open(
        "a", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow([submitted_at, config_id, job_id, recovery_path])

    recovery_log = run_root / "recovery" / "recovery_log.tsv"
    recovery_log.parent.mkdir(parents=True, exist_ok=True)
    new_log = not recovery_log.exists()
    with recovery_log.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        if new_log:
            writer.writerow(
                [
                    "submitted_at",
                    "config_id",
                    "attempt",
                    "job_id",
                    "original_cap",
                    "recovery_cap",
                    "reason",
                    "config_file",
                    "failed_attempt_archive",
                ]
            )
        writer.writerow(
            [
                submitted_at,
                config_id,
                attempt,
                job_id,
                original_cap,
                RECOVERY_CAP,
                reason,
                recovery_path,
                archive,
            ]
        )
    return {
        "config_id": config_id,
        "attempt": attempt,
        "job_id": job_id,
        "original_cap": original_cap,
        "recovery_cap": RECOVERY_CAP,
        "recovery_config": str(recovery_path),
        "failed_attempt_archive": str(archive),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("config_id", type=int)
    parser.add_argument("--reason", default=DEFAULT_REASON)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = submit_recovery(args.run_root.resolve(), args.config_id, args.reason)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Archive and resubmit a TTC cap recovery without increasing its 16384 cap."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List


RECOVERY_CAP = 16_384


def recovery_configs(run_root: Path, config_id: int) -> List[Path]:
    paths = list(
        (run_root / "recovery" / "configs").glob(
            f"config_{config_id:04d}_cap{RECOVERY_CAP}_attempt*.json"
        )
    )
    return sorted(
        paths,
        key=lambda path: int(re.search(r"_attempt(\d+)\.json$", path.name).group(1)),
    )


def submission_attempts(run_root: Path, config_id: int) -> int:
    path = run_root / "slurm" / "submitted_jobs.tsv"
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(
            1
            for row in csv.DictReader(handle, delimiter="\t")
            if int(row["config_id"]) == config_id
        )


def resubmit(run_root: Path, config_id: int, reason: str) -> Dict[str, Any]:
    paths = recovery_configs(run_root, config_id)
    if not paths:
        raise RuntimeError(f"No 16384-cap recovery config exists for config {config_id}")
    prior_path = paths[-1]
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    prior_attempt = int(prior["recovery_attempt"])
    if int(prior["max_tokens_per_phase"]) != RECOVERY_CAP:
        raise RuntimeError(f"Latest recovery is not capped at {RECOVERY_CAP}")

    recorded_attempts = submission_attempts(run_root, config_id)
    if recorded_attempts != prior_attempt:
        raise RuntimeError(
            f"Recorded attempts={recorded_attempts}, latest recovery attempt={prior_attempt}"
        )
    output_dir = Path(prior["output_dir"])
    result_path = output_dir / "run_1_experiment_results.json"
    if result_path.exists():
        raise RuntimeError(f"Refusing to replace an existing result: {result_path}")
    if not output_dir.is_dir():
        raise RuntimeError(f"Failed recovery output is missing: {output_dir}")

    archive = (
        run_root
        / "recovery"
        / "failed_attempts"
        / f"config_{config_id:04d}_attempt{prior_attempt}_cap{RECOVERY_CAP}"
    )
    if archive.exists():
        raise RuntimeError(f"Recovery archive already exists: {archive}")
    shutil.move(str(output_dir), str(archive))

    attempt = prior_attempt + 1
    retry = dict(prior)
    retry["recovery_attempt"] = attempt
    retry["recovery_reason"] = reason
    retry_path = (
        run_root
        / "recovery"
        / "configs"
        / f"config_{config_id:04d}_cap{RECOVERY_CAP}_attempt{attempt}.json"
    )
    retry_path.parent.mkdir(parents=True, exist_ok=True)
    retry_path.write_text(
        json.dumps(retry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    seed = int(prior["random_seed"])
    completed = subprocess.run(
        [
            "sbatch",
            "--parsable",
            f"--job-name=ttc{seed}_{config_id:04d}bf{attempt}",
            str(run_root / "slurm" / "run_one.sbatch"),
            str(retry_path),
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
        csv.writer(handle, delimiter="\t", lineterminator="\n").writerow(
            [submitted_at, config_id, job_id, retry_path]
        )

    recovery_log = run_root / "recovery" / "recovery_log.tsv"
    with recovery_log.open("a", encoding="utf-8", newline="") as handle:
        csv.writer(handle, delimiter="\t", lineterminator="\n").writerow(
            [
                submitted_at,
                config_id,
                attempt,
                job_id,
                RECOVERY_CAP,
                RECOVERY_CAP,
                reason,
                retry_path,
                archive,
            ]
        )
    return {
        "config_id": config_id,
        "attempt": attempt,
        "job_id": job_id,
        "cap_unchanged": RECOVERY_CAP,
        "recovery_config": str(retry_path),
        "failed_attempt_archive": str(archive),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("config_id", type=int)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            resubmit(args.run_root.resolve(), args.config_id, args.reason),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

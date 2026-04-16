#!/usr/bin/env python3
"""Backfill exact OpenRouter failures into an existing experiment tree.

This script is designed for rescue/backfill work after API transport failures.
It:
1. reads generated config JSONs by id,
2. archives any partial output directory for each target,
3. reruns the exact config into the same visible output directory,
4. emits minute-by-minute status ticks until all targets are done.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = REPO_ROOT / "experiments/results/scaling_experiment_20260404_064451"
DEFAULT_IDS = [
    130, 132, 155, 156, 159, 161, 162, 163, 164, 165, 166, 167, 258, 260, 262,
    264, 265, 408, 410, 412, 413, 414, 416, 417, 418, 419, 574, 576, 580, 581,
    584, 586, 604, 605, 606, 609, 612, 614, 700, 702, 704, 705, 706, 707, 708,
    709, 710, 711, 712, 713, 766, 846, 848, 850, 852, 856, 858, 859, 860, 862,
    864, 866, 867, 885, 889, 891,
]


@dataclass
class Task:
    config_id: int
    config_path: Path
    output_dir: Path
    stdout_path: Path
    stderr_path: Path
    cmd: List[str]
    model_pair: str
    process: Optional[subprocess.Popen] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    archived_from: Optional[Path] = None
    returncode: Optional[int] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--config-ids", nargs="+", type=int, default=DEFAULT_IDS)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--transport", choices=["direct", "proxy"], default="direct")
    parser.add_argument("--tick-seconds", type=int, default=60)
    parser.add_argument("--archive-root", type=Path, default=None)
    return parser.parse_args()


def resolve_config_path(experiment_dir: Path, config_id: int) -> Path:
    configs_dir = experiment_dir / "configs"
    exact = configs_dir / f"config_{config_id}.json"
    if exact.exists():
        return exact

    padded = sorted(configs_dir.glob(f"config_*{config_id:04d}.json"))
    if len(padded) == 1:
        return padded[0]

    matches = []
    for candidate in configs_dir.glob("config_*.json"):
        suffix = candidate.stem.split("_", 1)[1]
        try:
            if int(suffix) == config_id:
                matches.append(candidate)
        except ValueError:
            continue
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No config file found for config_id={config_id} in {configs_dir}")
    raise RuntimeError(f"Multiple config files matched config_id={config_id}: {matches}")


def model_pair_label(config: Dict) -> str:
    if "weak_model" in config and "strong_model" in config:
        return f"{config['weak_model']}_vs_{config['strong_model']}"
    models = config.get("models", [])
    if len(models) == 2:
        return f"{models[0]}_vs_{models[1]}"
    return "_vs_".join(str(model) for model in models)


def build_command(config: Dict, output_dir: Path, config_id: int) -> List[str]:
    models = config.get("models")
    if not models:
        models = [config["weak_model"], config["strong_model"]]

    cmd = [
        sys.executable,
        "run_strong_models_experiment.py",
        "--models", *models,
        "--batch",
        "--num-runs", str(config.get("num_runs", 1)),
        "--run-number", str(config["run_number"]),
        "--random-seed", str(config["random_seed"]),
        "--discussion-turns", str(config["discussion_turns"]),
        "--model-order", config["model_order"],
        "--output-dir", str(output_dir),
        "--job-id", str(config_id),
    ]

    if "competition_level" in config:
        cmd.extend(["--competition-level", str(config["competition_level"])])
    if "num_items" in config:
        cmd.extend(["--num-items", str(config["num_items"])])
    if "max_rounds" in config:
        cmd.extend(["--max-rounds", str(config["max_rounds"])])
    if "max_tokens_per_phase" in config:
        cmd.extend(["--max-tokens-per-phase", str(config["max_tokens_per_phase"])])
    if "gamma_discount" in config:
        cmd.extend(["--gamma-discount", str(config["gamma_discount"])])
    if config.get("game_type"):
        cmd.extend(["--game-type", config["game_type"]])

    game_type = config.get("game_type")
    if game_type == "diplomacy":
        cmd.extend(["--n-issues", str(config["n_issues"])])
        cmd.extend(["--rho", str(config["rho"])])
        cmd.extend(["--theta", str(config["theta"])])
    elif game_type == "co_funding":
        cmd.extend(["--m-projects", str(config["m_projects"])])
        cmd.extend(["--alpha", str(config["alpha"])])
        cmd.extend(["--sigma", str(config["sigma"])])
        cmd.extend(["--c-min", str(config["c_min"])])
        cmd.extend(["--c-max", str(config["c_max"])])
        cmd.extend([
            "--cofunding-discussion-transparency",
            str(config.get("cofunding_discussion_transparency", "own")),
        ])
        cmd.extend([
            "--cofunding-time-discount",
            str(config.get("cofunding_time_discount", 0.9)),
        ])
        if not config.get("cofunding_enable_commit_vote", True):
            cmd.append("--cofunding-disable-commit-vote")
        if not config.get("cofunding_enable_time_discount", True):
            cmd.append("--cofunding-disable-time-discount")

    return cmd


def load_task(experiment_dir: Path, config_id: int, logs_dir: Path) -> Task:
    config_path = resolve_config_path(experiment_dir, config_id)
    config = json.loads(config_path.read_text())
    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    model_pair = model_pair_label(config)
    stdout_path = logs_dir / f"backfill_exact_{config_id}.out"
    stderr_path = logs_dir / f"backfill_exact_{config_id}.err"
    cmd = build_command(config, output_dir, config_id)

    return Task(
        config_id=config_id,
        config_path=config_path,
        output_dir=output_dir,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        cmd=cmd,
        model_pair=model_pair,
    )


def archive_partial_output(task: Task, archive_root: Path) -> None:
    if not task.output_dir.exists():
        return
    if (task.output_dir / "experiment_results.json").exists():
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = archive_root / f"config_{task.config_id}_{timestamp}"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(task.output_dir), str(archive_path))
    task.archived_from = archive_path


def start_task(task: Task, transport: str) -> None:
    task.output_dir.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["OPENROUTER_TRANSPORT"] = transport
    with task.stdout_path.open("w") as out_f, task.stderr_path.open("w") as err_f:
        process = subprocess.Popen(
            task.cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=out_f,
            stderr=err_f,
        )
    task.process = process
    task.started_at = time.time()


def summarize_status(start_time: float, tasks: Dict[int, Task], pending: List[Task]) -> str:
    started = sum(task.started_at is not None for task in tasks.values())
    succeeded = sum((task.returncode == 0 and (task.output_dir / "experiment_results.json").exists()) for task in tasks.values())
    failed = sum(task.returncode not in (None, 0) for task in tasks.values())
    running = sum(task.process is not None and task.returncode is None for task in tasks.values())
    pending_count = len(pending)
    healthy = failed == 0
    reason = "healthy: active reruns are progressing and completed outputs are landing in place" if healthy else "attention: at least one rerun exited non-zero"
    elapsed = int(time.time() - start_time)
    return (
        f"TICK|elapsed_s={elapsed}|started={started}|succeeded={succeeded}|"
        f"failed={failed}|running={running}|pending={pending_count}|{reason}"
    )


def final_summary(tasks: Dict[int, Task]) -> str:
    succeeded = sorted(task.config_id for task in tasks.values() if task.returncode == 0 and (task.output_dir / "experiment_results.json").exists())
    failed = sorted(task.config_id for task in tasks.values() if not ((task.returncode == 0) and (task.output_dir / "experiment_results.json").exists()))
    return json.dumps({
        "succeeded": succeeded,
        "failed": failed,
        "succeeded_count": len(succeeded),
        "failed_count": len(failed),
    }, indent=2)


def main() -> int:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    logs_dir = REPO_ROOT / "logs/cluster"
    logs_dir.mkdir(parents=True, exist_ok=True)
    archive_root = args.archive_root or experiment_dir / "_backfill_archives" / "openrouter_exact_backfill"
    archive_root = archive_root.resolve()

    tasks = {config_id: load_task(experiment_dir, config_id, logs_dir) for config_id in args.config_ids}
    pending = [tasks[i] for i in args.config_ids]
    start_time = time.time()
    last_tick = 0.0

    print(
        f"START|targets={len(pending)}|concurrency={args.concurrency}|transport={args.transport}|"
        f"experiment_dir={experiment_dir}",
        flush=True,
    )

    for task in pending:
        archive_partial_output(task, archive_root)

    while pending or any(task.process is not None and task.returncode is None for task in tasks.values()):
        while pending and sum(task.process is not None and task.returncode is None for task in tasks.values()) < args.concurrency:
            task = pending.pop(0)
            if (task.output_dir / "experiment_results.json").exists():
                task.returncode = 0
                task.finished_at = time.time()
                print(f"SKIP|config_id={task.config_id}|reason=already_has_result", flush=True)
                continue

            print(
                f"START_TASK|config_id={task.config_id}|models={task.model_pair}|output_dir={task.output_dir}",
                flush=True,
            )
            start_task(task, args.transport)

        for task in tasks.values():
            if task.process is None or task.returncode is not None:
                continue
            rc = task.process.poll()
            if rc is None:
                continue
            task.returncode = rc
            task.finished_at = time.time()
            result_exists = (task.output_dir / "experiment_results.json").exists()
            status = "SUCCESS" if rc == 0 and result_exists else "FAIL"
            print(
                f"{status}|config_id={task.config_id}|returncode={rc}|result_exists={int(result_exists)}|"
                f"stdout={task.stdout_path}|stderr={task.stderr_path}",
                flush=True,
            )

        now = time.time()
        if now - last_tick >= args.tick_seconds:
            print(summarize_status(start_time, tasks, pending), flush=True)
            last_tick = now

        time.sleep(5)

    print(f"DONE|{final_summary(tasks)}", flush=True)
    failed = [task.config_id for task in tasks.values() if not ((task.returncode == 0) and (task.output_dir / "experiment_results.json").exists())]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

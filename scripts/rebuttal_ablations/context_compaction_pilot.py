#!/usr/bin/env python3
"""Generate, run, submit, and monitor paired context-compaction batches.

Each batch fixes Game 3 at sigma=0.2 and alpha=0.2, uses five environment
seeds at each N in {2, 4, 6, 8, 10}, and runs every seeded environment once
with deterministic public-history compaction enabled and once with it disabled.
The replicate range is configurable so a follow-up batch can add fresh seeds
without altering the original pilot.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import full_games123_multiagent_batch as full  # noqa: E402


MASTER_SEED = 20260726
N_VALUES = (2, 4, 6, 8, 10)
REPLICATES_PER_N = 5
ARMS = ("on", "off")
MODEL = "gpt-4o-mini-2024-07-18"
BATCH_TYPE = "reviewer_context_compaction_pilot"
MODEL_ORDER = "context_compaction_pilot"


def timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def read_ids(path: Path) -> list[int]:
    return [
        int(line.strip().removeprefix("config_"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_ids(path: Path, ids: Iterable[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{int(config_id)}\n" for config_id in ids), encoding="utf-8")


def paired_randomized_ids(
    configs: list[dict[str, Any]],
    n_agents: int,
    *,
    exclude_pair_ids: set[str] | None = None,
) -> list[int]:
    """Randomize pair order and arm order while keeping paired runs adjacent."""
    excluded = exclude_pair_ids or set()
    grouped: dict[str, list[int]] = {}
    for config in configs:
        if int(config["n_agents"]) != n_agents:
            continue
        pair_id = str(config["pair_id"])
        if pair_id in excluded:
            continue
        grouped.setdefault(pair_id, []).append(int(config["config_id"]))

    rng = random.Random(full.stable_seed(MASTER_SEED, "selection", n_agents))
    pair_ids = sorted(grouped)
    rng.shuffle(pair_ids)
    selected: list[int] = []
    for pair_id in pair_ids:
        arm_ids = sorted(grouped[pair_id])
        rng.shuffle(arm_ids)
        selected.extend(arm_ids)
    return selected


def config_path(results_root: Path, config_id: int) -> Path:
    return results_root / "configs" / f"config_{config_id:04d}.json"


def load_config(results_root: Path, config_id: int) -> dict[str, Any]:
    return json.loads(config_path(results_root, config_id).read_text(encoding="utf-8"))


def output_dir_for(
    results_root: Path,
    n_agents: int,
    seed_replicate: int,
    arm: str,
) -> Path:
    return (
        results_root
        / "runs"
        / f"n_{n_agents}"
        / f"seed_replicate_{seed_replicate}"
        / f"compaction_{arm}"
    )


def build_configs(
    results_root: Path,
    *,
    replicate_start: int = 1,
) -> list[dict[str, Any]]:
    if replicate_start < 1:
        raise ValueError("replicate_start must be positive")
    configs: list[dict[str, Any]] = []
    config_id = 1
    for n_agents in N_VALUES:
        for seed_replicate in range(
            replicate_start,
            replicate_start + REPLICATES_PER_N,
        ):
            environment_seed = full.stable_seed(
                MASTER_SEED,
                BATCH_TYPE,
                n_agents,
                seed_replicate,
            )
            pair_id = f"n{n_agents:02d}_seed{seed_replicate:02d}"
            for arm in ARMS:
                models = [MODEL] * n_agents
                run_dir = output_dir_for(
                    results_root,
                    n_agents,
                    seed_replicate,
                    arm,
                )
                configs.append(
                    {
                        "config_id": config_id,
                        "experiment_id": f"{BATCH_TYPE}_config_{config_id:04d}",
                        "batch_type": BATCH_TYPE,
                        "experiment_family": BATCH_TYPE,
                        "experiment_type": "paired_compaction_flag_ablation",
                        "game_label": "game3",
                        "game_type": "co_funding",
                        "competition_id": "sigma_0p2_alpha_0p2",
                        "n_agents": n_agents,
                        "num_agents": n_agents,
                        "models": models,
                        "baseline_model": MODEL,
                        "monoculture_model": MODEL,
                        "model_order": MODEL_ORDER,
                        "agent_model_map": {
                            f"Agent_{index}": MODEL
                            for index in range(1, n_agents + 1)
                        },
                        "agent_role_map": {
                            f"Agent_{index}": "compaction_ablation_agent"
                            for index in range(1, n_agents + 1)
                        },
                        "max_rounds": 10,
                        "discussion_turns": 2,
                        "gamma_discount": 0.9,
                        "parallel_phases": True,
                        "random_seed": environment_seed,
                        "seed": environment_seed,
                        "seed_replicate": seed_replicate,
                        "pair_id": pair_id,
                        "treatment_arm": arm,
                        "compaction_enabled": arm == "on",
                        "run_number": 1,
                        "output_dir": full.relative_or_absolute(run_dir),
                        "m_projects": int(2.5 * n_agents),
                        "alpha": 0.2,
                        "sigma": 0.2,
                        "c_min": 10.0,
                        "c_max": 30.0,
                        "cofunding_discussion_transparency": "own",
                        "cofunding_enable_commit_vote": True,
                        "cofunding_enable_time_discount": True,
                        "cofunding_time_discount": 0.9,
                        "compaction_disable_env": (
                            None
                            if arm == "on"
                            else "NEGOTIATION_DISABLE_CONTEXT_COMPACTION=1"
                        ),
                        "notes": (
                            "Paired reviewer pilot: identical environment seeds across "
                            "compaction arms; only the deployed compaction flag differs."
                        ),
                    }
                )
                config_id += 1
    return configs


def validate_configs(configs: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    if len(configs) != 50:
        errors.append(f"Expected 50 configs, found {len(configs)}")

    ids = [int(config["config_id"]) for config in configs]
    if ids != list(range(1, 51)):
        errors.append("Config IDs must be contiguous from 1 through 50")

    pairs: dict[str, list[dict[str, Any]]] = {}
    for config in configs:
        pairs.setdefault(str(config["pair_id"]), []).append(config)
        n_agents = int(config["n_agents"])
        if n_agents not in N_VALUES:
            errors.append(f"Unexpected N={n_agents} in config {config['config_id']}")
        if config["models"] != [MODEL] * n_agents:
            errors.append(f"Wrong model roster in config {config['config_id']}")
        if config.get("alpha") != 0.2 or config.get("sigma") != 0.2:
            errors.append(f"Wrong Game 3 cell in config {config['config_id']}")
        if config.get("parallel_phases") is not True:
            errors.append(f"Parallel phases disabled in config {config['config_id']}")
        if config.get("treatment_arm") not in ARMS:
            errors.append(f"Unknown arm in config {config['config_id']}")

    if len(pairs) != 25:
        errors.append(f"Expected 25 paired blocks, found {len(pairs)}")

    for pair_id, pair in pairs.items():
        if len(pair) != 2:
            errors.append(f"{pair_id}: expected two configs, found {len(pair)}")
            continue
        by_arm = {str(config["treatment_arm"]): config for config in pair}
        if set(by_arm) != set(ARMS):
            errors.append(f"{pair_id}: expected arms {ARMS}, found {sorted(by_arm)}")
            continue
        on = by_arm["on"]
        off = by_arm["off"]
        for field in (
            "n_agents",
            "random_seed",
            "seed_replicate",
            "m_projects",
            "alpha",
            "sigma",
            "max_rounds",
            "discussion_turns",
            "models",
        ):
            if on[field] != off[field]:
                errors.append(f"{pair_id}: paired field {field} differs across arms")
        if on["compaction_enabled"] is not True:
            errors.append(f"{pair_id}: on arm is not marked enabled")
        if off["compaction_enabled"] is not False:
            errors.append(f"{pair_id}: off arm is not marked disabled")

    by_n = Counter(int(config["n_agents"]) for config in configs)
    for n_agents in N_VALUES:
        if by_n[n_agents] != 10:
            errors.append(f"N={n_agents}: expected 10 runs, found {by_n[n_agents]}")
    return errors


def generate(args: argparse.Namespace) -> None:
    results_root = args.results_root.resolve()
    if results_root.exists() and any(results_root.iterdir()) and not args.force:
        raise FileExistsError(f"Results root already exists and is not empty: {results_root}")

    for dirname in ("configs", "runs", "status", "logs", "monitoring", "selections", "submissions", "slurm"):
        (results_root / dirname).mkdir(parents=True, exist_ok=True)

    replicate_start = int(args.replicate_start)
    configs = build_configs(results_root, replicate_start=replicate_start)
    errors = validate_configs(configs)
    if errors:
        raise ValueError("Invalid pilot grid:\n" + "\n".join(f"- {error}" for error in errors))

    for config in configs:
        write_json(config_path(results_root, int(config["config_id"])), config)

    all_ids = [int(config["config_id"]) for config in configs]
    first_pair = [
        int(config["config_id"])
        for config in configs
        if config["n_agents"] == 2
        and config["seed_replicate"] == replicate_start
    ]
    randomized_first = list(first_pair)
    random.Random(MASTER_SEED).shuffle(randomized_first)
    remaining = [config_id for config_id in all_ids if config_id not in first_pair]
    random.Random(MASTER_SEED + 1).shuffle(remaining)
    randomized_all = list(all_ids)
    random.Random(MASTER_SEED + 2).shuffle(randomized_all)

    write_ids(results_root / "selections" / "first_pair_config_ids.txt", randomized_first)
    write_ids(results_root / "selections" / "remaining_config_ids.txt", remaining)
    write_ids(results_root / "selections" / "all_config_ids.txt", randomized_all)
    write_n_selections(results_root, configs)

    write_json(
        results_root / "manifest.json",
        {
            "batch_type": BATCH_TYPE,
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
            "results_root": str(results_root),
            "master_seed": MASTER_SEED,
            "environment_seed_definition": (
                "stable_seed(master_seed, batch_type, n_agents, seed_replicate)"
            ),
            "n_values": list(N_VALUES),
            "replicates_per_n": REPLICATES_PER_N,
            "replicate_start": replicate_start,
            "replicate_stop_inclusive": (
                replicate_start + REPLICATES_PER_N - 1
            ),
            "treatment_arms": list(ARMS),
            "model": MODEL,
            "game": "game3",
            "sigma": 0.2,
            "alpha": 0.2,
            "expected_pairs": 25,
            "expected_runs": 50,
            "pilot_runs_are_included_in_any_later_full_design": True,
            "compaction_off_switch": "NEGOTIATION_DISABLE_CONTEXT_COMPACTION=1",
        },
    )
    print(f"Generated 50 configs (25 paired blocks) under {results_root}")
    print(f"First validation pair: {randomized_first}")


def write_n_selections(
    results_root: Path,
    configs: list[dict[str, Any]] | None = None,
) -> None:
    if configs is None:
        configs = [
            load_config(results_root, config_id)
            for config_id in range(1, 51)
            if config_path(results_root, config_id).exists()
        ]
    for n_agents in N_VALUES:
        excluded = {"n02_seed01"} if n_agents == 2 else set()
        ids = paired_randomized_ids(
            configs,
            n_agents,
            exclude_pair_ids=excluded,
        )
        write_ids(
            results_root / "selections" / f"n{n_agents}_config_ids.txt",
            ids,
        )


def refresh_selections(args: argparse.Namespace) -> None:
    results_root = args.results_root.resolve()
    write_n_selections(results_root)
    counts = {
        f"n{n_agents}": len(
            read_ids(results_root / "selections" / f"n{n_agents}_config_ids.txt")
        )
        for n_agents in N_VALUES
    }
    print(json.dumps(counts, indent=2))


def validate(args: argparse.Namespace) -> None:
    results_root = args.results_root.resolve()
    configs = [
        load_config(results_root, config_id)
        for config_id in range(1, 51)
        if config_path(results_root, config_id).exists()
    ]
    errors = validate_configs(configs)
    if errors:
        print("Validation failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print(f"Validation passed: 50 runs, 25 pairs, 10 runs at each N under {results_root}")


def add_status_metadata(results_root: Path, config: dict[str, Any]) -> None:
    status_path = full.config_status_path(results_root, int(config["config_id"]))
    status = full.read_json_file(status_path)
    status.update(
        {
            "pair_id": config["pair_id"],
            "treatment_arm": config["treatment_arm"],
            "compaction_enabled": config["compaction_enabled"],
            "n_agents": config["n_agents"],
            "random_seed": config["random_seed"],
            "seed_replicate": config["seed_replicate"],
        }
    )
    write_json(status_path, status)


def execute_config(results_root: Path, config_id: int) -> bool:
    config = load_config(results_root, config_id)
    if config["treatment_arm"] == "off":
        os.environ["NEGOTIATION_DISABLE_CONTEXT_COMPACTION"] = "1"
    else:
        os.environ.pop("NEGOTIATION_DISABLE_CONTEXT_COMPACTION", None)

    status = full.run_config(results_root, config)
    add_status_metadata(results_root, config)
    return status.get("state") == "SUCCESS"


def run_one(args: argparse.Namespace) -> None:
    results_root = args.results_root.resolve()
    if not execute_config(results_root, int(args.config_id)):
        raise SystemExit(1)


def run_pair(args: argparse.Namespace) -> None:
    """Run both arms in one Slurm task, preserving the randomized arm order."""
    results_root = args.results_root.resolve()
    config_ids = [
        int(token.strip())
        for token in str(args.config_ids).split(",")
        if token.strip()
    ]
    if not config_ids or len(config_ids) > 2:
        raise ValueError("--config-ids must contain one or two comma-separated IDs")

    failed: list[int] = []
    if args.concurrent and len(config_ids) == 2:
        processes: list[tuple[int, subprocess.Popen[str]]] = []
        for config_id in config_ids:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "run-one",
                "--results-root",
                str(results_root),
                "--config-id",
                str(config_id),
            ]
            processes.append(
                (
                    config_id,
                    subprocess.Popen(
                        command,
                        cwd=PROJECT_ROOT,
                        text=True,
                    ),
                )
            )
        for config_id, process in processes:
            if process.wait() != 0:
                failed.append(config_id)
    else:
        for config_id in config_ids:
            if not execute_config(results_root, config_id):
                failed.append(config_id)
    if failed:
        print(f"Failed configs in paired task: {failed}", file=sys.stderr)
        raise SystemExit(1)


def write_slurm_script(
    results_root: Path,
    task_file: Path,
    job_name: str,
    slurm_time: str,
    *,
    pair_tasks: bool,
    concurrent_arms: bool,
) -> Path:
    slurm_path = results_root / "slurm" / f"{job_name}.sbatch"
    log_prefix = results_root / "slurm" / f"{job_name}_%A_%a"
    cpus_per_task = 2 if concurrent_arms else 1
    content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=cpu
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem=8G
#SBATCH --time={slurm_time}
#SBATCH --output={log_prefix}.out
#SBATCH --error={log_prefix}.err

set -eo pipefail

cd {PROJECT_ROOT}
module purge
module load anaconda3/2024.2
module load proxy/default

export PYTHONPATH="{PROJECT_ROOT}:${{PYTHONPATH:-}}"
export OPENROUTER_TRANSPORT="${{OPENROUTER_TRANSPORT:-proxy}}"
export OPENROUTER_PROXY_POLL_DIR="${{OPENROUTER_PROXY_POLL_DIR:-/home/jz4391/openrouter_proxy}}"
export OPENROUTER_PROXY_CLIENT_TIMEOUT="${{OPENROUTER_PROXY_CLIENT_TIMEOUT:-9000}}"
export OPENROUTER_API_TIMEOUT="${{OPENROUTER_API_TIMEOUT:-1800}}"
export LLM_FAILURE_REPORT_PATH="{results_root}/monitoring/provider_failures.md"

TASK_VALUE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {task_file})
if [ -z "$TASK_VALUE" ]; then
  echo "No config for SLURM_ARRAY_TASK_ID=${{SLURM_ARRAY_TASK_ID}}" >&2
  exit 2
fi

{PROJECT_ROOT}/.venv/bin/python {Path(__file__).resolve()} {"run-pair" if pair_tasks else "run-one"} \
  --results-root {results_root} \
  --{"config-ids" if pair_tasks else "config-id"} "$TASK_VALUE"{" --concurrent" if concurrent_arms else ""}
"""
    slurm_path.write_text(content, encoding="utf-8")
    slurm_path.chmod(0o755)
    return slurm_path


def successful_config(results_root: Path, config: dict[str, Any]) -> bool:
    result_path = full.result_path_for(config)
    return (
        result_path is not None
        and full.validate_result_file(config, result_path) is None
    )


def scientific_context_failure(results_root: Path, config: dict[str, Any]) -> bool:
    status = full.read_json_file(
        full.config_status_path(results_root, int(config["config_id"]))
    )
    if status.get("state") != "FAILED":
        return False
    markers = (
        "context_length_exceeded",
        "maximum context length",
        "context window",
        "requested too many tokens",
    )
    for attempt in status.get("attempts", []):
        log_path = attempt.get("log_path")
        if not log_path or not Path(log_path).exists():
            continue
        with Path(log_path).open("rb") as handle:
            handle.seek(max(0, Path(log_path).stat().st_size - 1_000_000))
            text = handle.read().decode(errors="replace").lower()
        if any(marker in text for marker in markers):
            return True
    return False


def terminal_config(results_root: Path, config: dict[str, Any]) -> bool:
    return successful_config(results_root, config) or scientific_context_failure(
        results_root, config
    )


def submit(args: argparse.Namespace) -> None:
    results_root = args.results_root.resolve()
    selection = args.selection
    source_selection = results_root / "selections" / f"{selection}_config_ids.txt"
    if not source_selection.exists():
        raise FileNotFoundError(f"Missing selection: {source_selection}")

    source_ids = read_ids(source_selection)
    selected_ids = [
        config_id
        for config_id in source_ids
        if args.rerun_existing
        or not terminal_config(results_root, load_config(results_root, config_id))
    ]
    if not selected_ids:
        print(f"No unfinished configs in selection {selection}")
        return
    if args.concurrent_arms and not args.pair_tasks:
        raise ValueError("--concurrent-arms requires --pair-tasks")

    stamp = timestamp()
    task_file = results_root / "submissions" / f"{selection}_{stamp}_tasks.txt"
    if args.pair_tasks:
        source_order = {config_id: index for index, config_id in enumerate(source_ids)}
        grouped: dict[str, list[int]] = {}
        for config_id in selected_ids:
            config = load_config(results_root, config_id)
            grouped.setdefault(str(config["pair_id"]), []).append(config_id)
        task_rows = [
            ",".join(
                str(config_id)
                for config_id in sorted(ids, key=source_order.__getitem__)
            )
            for _, ids in sorted(
                grouped.items(),
                key=lambda item: min(source_order[config_id] for config_id in item[1]),
            )
        ]
        task_file.write_text("\n".join(task_rows) + "\n", encoding="utf-8")
    else:
        task_rows = [str(config_id) for config_id in selected_ids]
        write_ids(task_file, selected_ids)
    job_name = f"cmpct_{selection}_{stamp}"[:48]
    slurm_path = write_slurm_script(
        results_root,
        task_file,
        job_name,
        args.slurm_time,
        pair_tasks=args.pair_tasks,
        concurrent_arms=args.concurrent_arms,
    )
    array_spec = f"1-{len(task_rows)}%{max(1, args.max_concurrent)}"
    command = ["sbatch"]
    if args.dependency_afterany:
        dependency_job_id = str(args.dependency_afterany)
        if not dependency_job_id.isdigit():
            raise ValueError("--dependency-afterany must be a numeric Slurm job ID")
        command.extend(["--dependency", f"afterany:{dependency_job_id}"])
    command.extend(["--array", array_spec, str(slurm_path)])
    record: dict[str, Any] = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "selection": selection,
        "source_selection": str(source_selection),
        "task_file": str(task_file),
        "config_ids": selected_ids,
        "config_count": len(selected_ids),
        "pair_tasks": args.pair_tasks,
        "concurrent_arms": args.concurrent_arms,
        "task_count": len(task_rows),
        "array_spec": array_spec,
        "dependency_afterany": args.dependency_afterany,
        "slurm_file": str(slurm_path),
        "command": command,
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        write_json(
            results_root / "submissions" / f"{selection}_{stamp}_dry_run.json",
            record,
        )
        print(" ".join(command))
        return

    submitted = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    output = (submitted.stdout or submitted.stderr).strip()
    record["sbatch_output"] = output
    record["job_id"] = full.parse_job_id(output)
    write_json(
        results_root / "submissions" / f"{selection}_{stamp}_submission.json",
        record,
    )
    print(output)
    print(
        f"Submitted {len(selected_ids)} runs in {len(task_rows)} "
        f"{'paired ' if args.pair_tasks else ''}tasks from {selection}"
    )


def summarize(results_root: Path) -> dict[str, Any]:
    state_counts: Counter[str] = Counter()
    by_n_arm: Counter[tuple[int, str, str]] = Counter()
    complete_pairs = 0
    terminal_pairs = 0
    pair_states: dict[str, set[str]] = {}
    configs = [load_config(results_root, config_id) for config_id in range(1, 51)]
    for config in configs:
        result_path = full.result_path_for(config)
        if result_path is not None and full.validate_result_file(config, result_path) is None:
            state = "SUCCESS"
        elif scientific_context_failure(results_root, config):
            state = "CONTEXT_FAILURE"
        else:
            status = full.read_json_file(
                full.config_status_path(results_root, int(config["config_id"]))
            )
            state = str(status.get("state") or "NOT_STARTED")
        state_counts[state] += 1
        by_n_arm[(int(config["n_agents"]), str(config["treatment_arm"]), state)] += 1
        if state in {"SUCCESS", "CONTEXT_FAILURE"}:
            pair_states.setdefault(str(config["pair_id"]), set()).add(
                f"{config['treatment_arm']}:{state}"
            )
    for states in pair_states.values():
        terminal_arms = {state.split(":", 1)[0] for state in states}
        if terminal_arms == set(ARMS):
            terminal_pairs += 1
        if states == {"on:SUCCESS", "off:SUCCESS"}:
            complete_pairs += 1
    return {
        "results_root": str(results_root),
        "state_counts": dict(state_counts),
        "complete_pairs": complete_pairs,
        "terminal_pairs": terminal_pairs,
        "by_n_arm_state": [
            {
                "n_agents": n_agents,
                "treatment_arm": arm,
                "state": state,
                "count": count,
            }
            for (n_agents, arm, state), count in sorted(by_n_arm.items())
        ],
    }


def status(args: argparse.Namespace) -> None:
    payload = summarize(args.results_root.resolve())
    print(json.dumps(payload, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--results-root", type=Path, required=True)
    generate_parser.add_argument(
        "--replicate-start",
        type=int,
        default=1,
        help="First seed-replicate label; use 6 to extend the five-seed pilot.",
    )
    generate_parser.add_argument("--force", action="store_true")
    generate_parser.set_defaults(func=generate)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--results-root", type=Path, required=True)
    validate_parser.set_defaults(func=validate)

    run_parser = subparsers.add_parser("run-one")
    run_parser.add_argument("--results-root", type=Path, required=True)
    run_parser.add_argument("--config-id", type=int, required=True)
    run_parser.set_defaults(func=run_one)

    pair_parser = subparsers.add_parser("run-pair")
    pair_parser.add_argument("--results-root", type=Path, required=True)
    pair_parser.add_argument("--config-ids", required=True)
    pair_parser.add_argument("--concurrent", action="store_true")
    pair_parser.set_defaults(func=run_pair)

    submit_parser = subparsers.add_parser("submit")
    submit_parser.add_argument("--results-root", type=Path, required=True)
    submit_parser.add_argument(
        "--selection",
        default="all",
        help="Stem of a file under selections/ named <stem>_config_ids.txt.",
    )
    submit_parser.add_argument("--max-concurrent", type=int, default=5)
    submit_parser.add_argument("--slurm-time", default="12:00:00")
    submit_parser.add_argument("--rerun-existing", action="store_true")
    submit_parser.add_argument("--pair-tasks", action="store_true")
    submit_parser.add_argument("--concurrent-arms", action="store_true")
    submit_parser.add_argument("--dry-run", action="store_true")
    submit_parser.add_argument(
        "--dependency-afterany",
        help="Submit with a Slurm afterany dependency on this numeric job ID.",
    )
    submit_parser.set_defaults(func=submit)

    refresh_parser = subparsers.add_parser("refresh-selections")
    refresh_parser.add_argument("--results-root", type=Path, required=True)
    refresh_parser.set_defaults(func=refresh_selections)

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--results-root", type=Path, required=True)
    status_parser.set_defaults(func=status)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

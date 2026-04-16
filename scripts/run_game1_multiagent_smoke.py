#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
import json
import random
import shlex
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strong_models_experiment.analysis.game1_multiagent_smoke import (  # noqa: E402
    resolve_results_root,
    write_analysis_outputs,
)


DEFAULT_BASELINE_MODEL = "gpt-5-nano"
DEFAULT_FOCAL_MODELS = [
    "llama-3.2-1b-instruct",
    "amazon-nova-micro-v1.0",
    "gpt-4.1-nano-2025-04-14",
    "o3-mini-high",
    "gemini-2.5-pro",
    "claude-opus-4-6",
]

DEFAULT_ECOLOGY_POOLS = {
    "weak": [
        "llama-3.2-3b-instruct",
        "llama-3.1-8b-instruct",
        "claude-3-haiku-20240307",
    ],
    "medium": [
        "qwen2.5-72b-instruct",
        "gpt-4o-mini-2024-07-18",
        "deepseek-v3",
    ],
    "strong": [
        "deepseek-r1-0528",
        "qwen3-max-preview",
        "gemini-3-pro",
    ],
}

DEFAULT_COMPETITION_LEVELS = [0.0, 0.5, 0.9, 1.0]
DEFAULT_N_VALUES = [3, 5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate, run, and analyze Game 1 multi-agent smoke experiments."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--results-root", type=str, default=None)

    generate = subparsers.add_parser("generate", parents=[common])
    add_generation_args(generate)

    run = subparsers.add_parser("run", parents=[common])
    run.add_argument("--max-workers", type=int, default=4)
    run.add_argument("--shuffle", action="store_true")
    run.add_argument("--seed", type=int, default=17)
    run.add_argument("--limit", type=int, default=None)

    analyze = subparsers.add_parser("analyze", parents=[common])

    all_cmd = subparsers.add_parser("all", parents=[common])
    add_generation_args(all_cmd)
    all_cmd.add_argument("--max-workers", type=int, default=4)
    all_cmd.add_argument("--shuffle", action="store_true")
    all_cmd.add_argument("--seed", type=int, default=17)
    all_cmd.add_argument("--limit", type=int, default=None)

    return parser.parse_args()


def add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--baseline-model", type=str, default=DEFAULT_BASELINE_MODEL)
    parser.add_argument("--focal-models", nargs="+", default=DEFAULT_FOCAL_MODELS)
    parser.add_argument("--competition-levels", nargs="+", type=float, default=DEFAULT_COMPETITION_LEVELS)
    parser.add_argument("--n-values", nargs="+", type=int, default=DEFAULT_N_VALUES)
    parser.add_argument("--proposal1-reps", type=int, default=2)
    parser.add_argument("--proposal2-fields", type=int, default=3)
    parser.add_argument("--discussion-turns", type=int, default=2)
    parser.add_argument("--num-items", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--gamma-discount", type=float, default=0.9)
    parser.add_argument("--base-seed", type=int, default=4242)


def sanitize_float(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".").replace("-", "n").replace(".", "p")


def latest_root_or_new(raw_value: str | None) -> Path:
    if raw_value:
        path = Path(raw_value)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        return path
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return (PROJECT_ROOT / "experiments" / "results" / f"game1_multiagent_smoke_{timestamp}").resolve()


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def create_latest_symlink(results_root: Path) -> None:
    symlink = results_root.parent / "game1_multiagent_smoke_latest"
    if symlink.is_symlink() or symlink.exists():
        symlink.unlink()
    symlink.symlink_to(results_root.name)


def build_field_templates(n_agents: int) -> List[List[str]]:
    if n_agents == 3:
        return [
            ["weak", "medium"],
            ["weak", "strong"],
            ["medium", "strong"],
        ]
    if n_agents == 5:
        return [
            ["weak", "weak", "medium", "strong"],
            ["weak", "medium", "medium", "strong"],
            ["weak", "medium", "strong", "strong"],
        ]
    raise ValueError(f"Unsupported n_agents={n_agents} for mixed ecology")


def build_mixed_fields(
    n_agents: int,
    num_fields: int,
    ecology_pools: Dict[str, Sequence[str]],
    seed: int,
) -> List[Dict]:
    rng = random.Random(seed + n_agents * 101)
    templates = build_field_templates(n_agents)
    fields: List[Dict] = []
    for field_idx in range(num_fields):
        template = templates[field_idx % len(templates)]
        opponents: List[str] = []
        for tier in ("weak", "medium", "strong"):
            count = sum(1 for item in template if item == tier)
            if count == 0:
                continue
            pool = list(ecology_pools[tier])
            if count > len(pool):
                raise ValueError(f"Not enough models in ecology tier {tier} for n={n_agents}")
            sampled = rng.sample(pool, count)
            opponents.extend(sampled)
        rng.shuffle(opponents)
        fields.append(
            {
                "field_id": field_idx + 1,
                "template": template,
                "opponents": opponents,
            }
        )
    return fields


def build_manifest(args: argparse.Namespace, results_root: Path) -> Dict:
    ecology_fields = {
        str(n_agents): build_mixed_fields(
            n_agents=n_agents,
            num_fields=args.proposal2_fields,
            ecology_pools=DEFAULT_ECOLOGY_POOLS,
            seed=args.base_seed,
        )
        for n_agents in args.n_values
    }
    return {
        "results_root": str(results_root),
        "game_type": "item_allocation",
        "baseline_model": args.baseline_model,
        "focal_models": list(args.focal_models),
        "competition_levels": list(args.competition_levels),
        "n_values": list(args.n_values),
        "proposal1_reps": int(args.proposal1_reps),
        "proposal2_fields": int(args.proposal2_fields),
        "discussion_turns": int(args.discussion_turns),
        "num_items": int(args.num_items),
        "max_rounds": int(args.max_rounds),
        "gamma_discount": float(args.gamma_discount),
        "base_seed": int(args.base_seed),
        "ecology_pools": DEFAULT_ECOLOGY_POOLS,
        "ecology_fields": ecology_fields,
        "notes": "Smoke batch for Game 1 multi-agent proposals 1 and 2.",
    }


def pairing_id(parts: Iterable[object]) -> str:
    return "|".join(str(part) for part in parts)


def build_config_records(manifest: Dict, results_root: Path) -> List[Dict]:
    configs: List[Dict] = []
    config_id = 1
    baseline = manifest["baseline_model"]

    for focal_model in manifest["focal_models"]:
        for n_agents in manifest["n_values"]:
            for competition_level in manifest["competition_levels"]:
                for replicate_id in range(1, manifest["proposal1_reps"] + 1):
                    seed = manifest["base_seed"] + config_id * 13
                    pair_id = pairing_id(
                        [
                            "proposal1",
                            focal_model,
                            f"n{n_agents}",
                            f"comp{competition_level}",
                            f"rep{replicate_id}",
                        ]
                    )
                    configs.extend(
                        [
                            make_config_record(
                                config_id=config_id,
                                proposal_family="proposal1_invasion",
                                condition_role="focal",
                                pairing=pair_id,
                                focal_model=focal_model,
                                baseline_model=baseline,
                                models=[focal_model] + [baseline] * (n_agents - 1),
                                n_agents=n_agents,
                                competition_level=competition_level,
                                random_seed=seed,
                                replicate_id=replicate_id,
                                field_id=None,
                                ecology_models=[],
                                results_root=results_root,
                                manifest=manifest,
                            ),
                            make_config_record(
                                config_id=config_id + 1,
                                proposal_family="proposal1_invasion",
                                condition_role="control",
                                pairing=pair_id,
                                focal_model=focal_model,
                                baseline_model=baseline,
                                models=[baseline] * n_agents,
                                n_agents=n_agents,
                                competition_level=competition_level,
                                random_seed=seed,
                                replicate_id=replicate_id,
                                field_id=None,
                                ecology_models=[],
                                results_root=results_root,
                                manifest=manifest,
                            ),
                        ]
                    )
                    config_id += 2

    for focal_model in manifest["focal_models"]:
        for n_agents in manifest["n_values"]:
            ecology_fields = manifest["ecology_fields"][str(n_agents)]
            for competition_level in manifest["competition_levels"]:
                for field in ecology_fields:
                    seed = manifest["base_seed"] + config_id * 13
                    pair_id = pairing_id(
                        [
                            "proposal2",
                            focal_model,
                            f"n{n_agents}",
                            f"comp{competition_level}",
                            f"field{field['field_id']}",
                        ]
                    )
                    opponents = list(field["opponents"])
                    configs.extend(
                        [
                            make_config_record(
                                config_id=config_id,
                                proposal_family="proposal2_mixed_ecology",
                                condition_role="focal",
                                pairing=pair_id,
                                focal_model=focal_model,
                                baseline_model=baseline,
                                models=[focal_model] + opponents,
                                n_agents=n_agents,
                                competition_level=competition_level,
                                random_seed=seed,
                                replicate_id=None,
                                field_id=field["field_id"],
                                ecology_models=opponents,
                                results_root=results_root,
                                manifest=manifest,
                            ),
                            make_config_record(
                                config_id=config_id + 1,
                                proposal_family="proposal2_mixed_ecology",
                                condition_role="control",
                                pairing=pair_id,
                                focal_model=focal_model,
                                baseline_model=baseline,
                                models=[baseline] + opponents,
                                n_agents=n_agents,
                                competition_level=competition_level,
                                random_seed=seed,
                                replicate_id=None,
                                field_id=field["field_id"],
                                ecology_models=opponents,
                                results_root=results_root,
                                manifest=manifest,
                            ),
                        ]
                    )
                    config_id += 2

    return configs


def make_config_record(
    *,
    config_id: int,
    proposal_family: str,
    condition_role: str,
    pairing: str,
    focal_model: str,
    baseline_model: str,
    models: List[str],
    n_agents: int,
    competition_level: float,
    random_seed: int,
    replicate_id: int | None,
    field_id: int | None,
    ecology_models: List[str],
    results_root: Path,
    manifest: Dict,
) -> Dict:
    comp_token = sanitize_float(competition_level)
    run_token = (
        f"rep_{replicate_id:02d}" if replicate_id is not None else f"field_{field_id:02d}"
    )
    output_dir = (
        results_root
        / proposal_family
        / f"focal_{focal_model}"
        / f"n_{n_agents}"
        / f"comp_{comp_token}"
        / run_token
        / condition_role
    )
    return {
        "config_id": config_id,
        "proposal_family": proposal_family,
        "condition_role": condition_role,
        "pairing_id": pairing,
        "game_type": "item_allocation",
        "focal_model": focal_model,
        "baseline_model": baseline_model,
        "models": models,
        "ecology_models": ecology_models,
        "n_agents": n_agents,
        "competition_level": competition_level,
        "num_items": manifest["num_items"],
        "max_rounds": manifest["max_rounds"],
        "gamma_discount": manifest["gamma_discount"],
        "discussion_turns": manifest["discussion_turns"],
        "random_seed": random_seed,
        "replicate_id": replicate_id,
        "field_id": field_id,
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
    }


def write_generated_files(results_root: Path, manifest: Dict, configs: List[Dict]) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    config_dir = results_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    write_json(results_root / "manifest.json", manifest)

    for config in configs:
        config_path = config_dir / f"config_{config['config_id']:04d}.json"
        write_json(config_path, config)

    all_configs = config_dir / "all_configs.txt"
    all_config_lines = [
        str(config_dir / f"config_{cfg['config_id']:04d}.json") for cfg in configs
    ]
    all_configs.write_text("\n".join(all_config_lines) + "\n", encoding="utf-8")

    index_path = config_dir / "experiment_index.csv"
    fieldnames = [
        "config_id",
        "proposal_family",
        "condition_role",
        "pairing_id",
        "focal_model",
        "baseline_model",
        "n_agents",
        "competition_level",
        "replicate_id",
        "field_id",
        "random_seed",
        "models",
        "ecology_models",
        "output_dir",
        "config_file",
    ]
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for config in configs:
            row = {key: config.get(key) for key in fieldnames if key not in {"models", "ecology_models", "config_file"}}
            row["models"] = "+".join(config["models"])
            row["ecology_models"] = "+".join(config["ecology_models"])
            row["config_file"] = f"config_{config['config_id']:04d}.json"
            writer.writerow(row)


def generate_results_root(args: argparse.Namespace) -> Path:
    results_root = latest_root_or_new(args.results_root)
    manifest = build_manifest(args, results_root)
    configs = build_config_records(manifest, results_root)
    write_generated_files(results_root, manifest, configs)
    create_latest_symlink(results_root)
    print(results_root)
    print(f"Generated {len(configs)} single-run configs")
    return results_root


def config_result_exists(config: Dict) -> bool:
    output_dir = PROJECT_ROOT / config["output_dir"]
    return (output_dir / "experiment_results.json").exists() or (output_dir / "run_1_experiment_results.json").exists()


def load_configs_from_root(results_root: Path) -> List[Dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((results_root / "configs").glob("config_*.json"))
    ]


def build_command(config: Dict) -> List[str]:
    output_dir = (PROJECT_ROOT / config["output_dir"]).resolve()
    cmd = [
        sys.executable,
        "run_strong_models_experiment.py",
        "--models",
        *config["models"],
        "--game-type",
        "item_allocation",
        "--num-items",
        str(config["num_items"]),
        "--max-rounds",
        str(config["max_rounds"]),
        "--competition-level",
        str(config["competition_level"]),
        "--gamma-discount",
        str(config["gamma_discount"]),
        "--discussion-turns",
        str(config["discussion_turns"]),
        "--model-order",
        "weak_first",
        "--random-seed",
        str(config["random_seed"]),
        "--output-dir",
        str(output_dir),
    ]
    return cmd


async def run_one_config(config: Dict, log_dir: Path, semaphore: asyncio.Semaphore) -> Dict:
    async with semaphore:
        log_path = log_dir / f"config_{config['config_id']:04d}.log"
        output_dir = (PROJECT_ROOT / config["output_dir"]).resolve()
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        if config_result_exists(config):
            return {"config_id": config["config_id"], "status": "SKIPPED", "log_path": str(log_path)}

        cmd = build_command(config)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(PROJECT_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        with log_path.open("wb") as handle:
            if proc.stdout is not None:
                while True:
                    chunk = await proc.stdout.read(4096)
                    if not chunk:
                        break
                    handle.write(chunk)
                    handle.flush()
        await proc.wait()
        status = "SUCCESS" if proc.returncode == 0 and config_result_exists(config) else "FAILED"
        return {
            "config_id": config["config_id"],
            "status": status,
            "returncode": proc.returncode,
            "log_path": str(log_path),
            "command": shlex.join(cmd),
        }


async def run_pending_configs(
    results_root: Path,
    max_workers: int,
    shuffle: bool,
    seed: int,
    limit: int | None,
) -> List[Dict]:
    configs = load_configs_from_root(results_root)
    pending = [config for config in configs if not config_result_exists(config)]
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(pending)
    if limit is not None:
        pending = pending[:limit]

    log_dir = results_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(max_workers)
    tasks = [asyncio.create_task(run_one_config(config, log_dir, semaphore)) for config in pending]
    results: List[Dict] = []
    for task in asyncio.as_completed(tasks):
        result = await task
        results.append(result)
        print(
            f"[{len(results)}/{len(tasks)}] config_{result['config_id']:04d} -> {result['status']}",
            flush=True,
        )
    return results


def cmd_generate(args: argparse.Namespace) -> int:
    generate_results_root(args)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    results_root = resolve_results_root(args.results_root)
    asyncio.run(
        run_pending_configs(
            results_root=results_root,
            max_workers=args.max_workers,
            shuffle=args.shuffle,
            seed=args.seed,
            limit=args.limit,
        )
    )
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    results_root = resolve_results_root(args.results_root)
    outputs = write_analysis_outputs(results_root)
    for key, value in outputs.items():
        print(f"{key}: {value}")
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    results_root = generate_results_root(args)
    asyncio.run(
        run_pending_configs(
            results_root=results_root,
            max_workers=args.max_workers,
            shuffle=args.shuffle,
            seed=args.seed,
            limit=args.limit,
        )
    )
    outputs = write_analysis_outputs(results_root)
    for key, value in outputs.items():
        print(f"{key}: {value}")
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "generate":
        return cmd_generate(args)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "analyze":
        return cmd_analyze(args)
    if args.command == "all":
        return cmd_all(args)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

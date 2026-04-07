#!/usr/bin/env python3
"""
Generate a Game 3 reference batch that mirrors the prior Game 1/Game 2
config-indexed mini-slate workflow.

The batch defaults to:
- baseline: gpt-5-nano
- adversaries: claude-opus-4-6, gpt-3.5-turbo-0125, grok-4, o3-mini-high
- alpha grid: 0.0, 0.5, 1.0
- sigma grid: 0.2, 0.6, 1.0
- model orders: weak_first, strong_first

This yields 72 model-scale configs.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "experiments" / "results"

DEFAULT_BASELINE = "gpt-5-nano"
DEFAULT_ADVERSARIES = [
    "claude-opus-4-6",
    "gpt-3.5-turbo-0125",
    "grok-4",
    "o3-mini-high",
]
DEFAULT_ALPHA_VALUES = [0.0, 0.5, 1.0]
DEFAULT_SIGMA_VALUES = [0.2, 0.6, 1.0]
DEFAULT_MODEL_ORDERS = ["weak_first", "strong_first"]


@dataclass(frozen=True)
class BatchSettings:
    batch_name: str
    baseline_model: str
    adversaries: List[str]
    alpha_values: List[float]
    sigma_values: List[float]
    model_orders: List[str]
    m_projects: int
    c_min: float
    c_max: float
    max_rounds: int
    discussion_turns: int
    max_tokens_per_phase: int
    base_seed: int
    num_runs: int
    transparency: str
    time_discount: float
    enable_commit_vote: bool
    enable_time_discount: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-name", default=None)
    parser.add_argument("--baseline-model", default=DEFAULT_BASELINE)
    parser.add_argument("--adversaries", nargs="+", default=DEFAULT_ADVERSARIES)
    parser.add_argument("--alpha-values", nargs="+", type=float, default=DEFAULT_ALPHA_VALUES)
    parser.add_argument("--sigma-values", nargs="+", type=float, default=DEFAULT_SIGMA_VALUES)
    parser.add_argument("--model-orders", nargs="+", default=DEFAULT_MODEL_ORDERS)
    parser.add_argument("--m-projects", type=int, default=5)
    parser.add_argument("--c-min", type=float, default=10.0)
    parser.add_argument("--c-max", type=float, default=30.0)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--discussion-turns", type=int, default=2)
    parser.add_argument("--max-tokens-per-phase", type=int, default=10500)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--transparency", choices=["aggregate", "own", "full"], default="own")
    parser.add_argument("--time-discount", type=float, default=0.9)
    parser.add_argument("--disable-commit-vote", action="store_true")
    parser.add_argument("--disable-time-discount", action="store_true")
    return parser.parse_args()


def _float_slug(value: float) -> str:
    return f"{value:.1f}".replace(".", "_")


def _competition_index(alpha: float, sigma: float) -> float:
    return (1.0 - float(alpha)) * (1.0 - float(sigma))


def _timestamped_batch_name(explicit_name: str | None) -> str:
    if explicit_name:
        return explicit_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"cofunding_{timestamp}_reference_slate"


def _iter_configs(settings: BatchSettings) -> Iterable[dict]:
    experiment_id = 0
    for model2 in settings.adversaries:
        for alpha in settings.alpha_values:
            for sigma in settings.sigma_values:
                for model_order in settings.model_orders:
                    for run_number in range(1, settings.num_runs + 1):
                        model1 = settings.baseline_model
                        models = [model1, model2]
                        if model_order == "strong_first":
                            models = [model2, model1]

                        output_dir = (
                            f"experiments/results/{settings.batch_name}/model_scale/"
                            f"{model1}_vs_{model2}/{model_order}/"
                            f"alpha_{_float_slug(alpha)}_sigma_{_float_slug(sigma)}"
                        )
                        if settings.num_runs > 1:
                            output_dir = f"{output_dir}/run_{run_number}"

                        yield {
                            "experiment_id": experiment_id,
                            "experiment_type": "model_scale",
                            "game_type": "co_funding",
                            "model1": model1,
                            "model2": model2,
                            "models": models,
                            "model_order": model_order,
                            "run_number": run_number,
                            "num_runs": settings.num_runs,
                            "max_tokens_per_phase": settings.max_tokens_per_phase,
                            "m_projects": settings.m_projects,
                            "alpha": alpha,
                            "sigma": sigma,
                            "c_min": settings.c_min,
                            "c_max": settings.c_max,
                            "cofunding_discussion_transparency": settings.transparency,
                            "cofunding_enable_commit_vote": settings.enable_commit_vote,
                            "cofunding_enable_time_discount": settings.enable_time_discount,
                            "cofunding_time_discount": settings.time_discount,
                            "max_rounds": settings.max_rounds,
                            "discussion_turns": settings.discussion_turns,
                            "random_seed": settings.base_seed + experiment_id,
                            "output_dir": output_dir,
                        }
                        experiment_id += 1


def main() -> None:
    args = parse_args()
    settings = BatchSettings(
        batch_name=_timestamped_batch_name(args.batch_name),
        baseline_model=args.baseline_model,
        adversaries=list(args.adversaries),
        alpha_values=list(args.alpha_values),
        sigma_values=list(args.sigma_values),
        model_orders=list(args.model_orders),
        m_projects=args.m_projects,
        c_min=args.c_min,
        c_max=args.c_max,
        max_rounds=args.max_rounds,
        discussion_turns=args.discussion_turns,
        max_tokens_per_phase=args.max_tokens_per_phase,
        base_seed=args.base_seed,
        num_runs=args.num_runs,
        transparency=args.transparency,
        time_discount=args.time_discount,
        enable_commit_vote=not args.disable_commit_vote,
        enable_time_discount=not args.disable_time_discount,
    )

    batch_root = RESULTS_ROOT / settings.batch_name
    config_dir = batch_root / "configs"
    log_dir = batch_root / "logs"
    config_dir.mkdir(parents=True, exist_ok=False)
    log_dir.mkdir(parents=True, exist_ok=True)

    configs = list(_iter_configs(settings))
    padding = max(4, len(str(len(configs) - 1)))

    all_configs_path = config_dir / "all_configs.txt"
    summary_path = config_dir / "summary.txt"
    index_path = config_dir / "experiment_index.csv"

    all_config_paths: List[Path] = []
    for payload in configs:
        config_name = f"config_{payload['experiment_id']:0{padding}d}.json"
        config_path = config_dir / config_name
        config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        all_config_paths.append(config_path)

    all_configs_path.write_text(
        "".join(f"{path}\n" for path in all_config_paths),
        encoding="utf-8",
    )

    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment_id",
                "experiment_type",
                "model1",
                "model2",
                "model_order",
                "token_budget",
                "alpha",
                "sigma",
                "competition_index",
                "run_number",
                "seed",
                "config_file",
            ],
        )
        writer.writeheader()
        for payload, config_path in zip(configs, all_config_paths):
            writer.writerow(
                {
                    "experiment_id": payload["experiment_id"],
                    "experiment_type": payload["experiment_type"],
                    "model1": payload["model1"],
                    "model2": payload["model2"],
                    "model_order": payload["model_order"],
                    "token_budget": "NA",
                    "alpha": payload["alpha"],
                    "sigma": payload["sigma"],
                    "competition_index": _competition_index(payload["alpha"], payload["sigma"]),
                    "run_number": payload["run_number"],
                    "seed": payload["random_seed"],
                    "config_file": config_path.name,
                }
            )

    summary_lines = [
        "Game 3 Reference Batch Summary",
        "==============================",
        f"Batch root: {batch_root}",
        f"Total configs: {len(configs)}",
        "",
        "Reference pattern:",
        "  Mirrors the prior Game 1 / Game 2 config-indexed mini-slate workflow.",
        f"  Baseline: {settings.baseline_model}",
        f"  Adversaries: {' '.join(settings.adversaries)}",
        f"  Alpha values: {' '.join(str(v) for v in settings.alpha_values)}",
        f"  Sigma values: {' '.join(str(v) for v in settings.sigma_values)}",
        f"  Model orders: {' '.join(settings.model_orders)}",
        "",
        "Game 3 parameters:",
        f"  m_projects={settings.m_projects}",
        f"  c_min={settings.c_min}",
        f"  c_max={settings.c_max}",
        f"  max_rounds={settings.max_rounds}",
        f"  discussion_turns={settings.discussion_turns}",
        f"  transparency={settings.transparency}",
        f"  enable_commit_vote={settings.enable_commit_vote}",
        f"  enable_time_discount={settings.enable_time_discount}",
        f"  time_discount={settings.time_discount}",
        "",
        "Competition index:",
        "  CI3 = (1 - alpha) * (1 - sigma)",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Created batch root: {batch_root}")
    print(f"Config dir: {config_dir}")
    print(f"Config index: {index_path}")
    print(f"Total configs: {len(configs)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate and run the full Games 1/2/3 random-monoculture control sweep.

This experiment mirrors the heterogeneous Games 1/2/3 grid, but each run uses
N copies of one sampled model. The 15 monoculture models are sampled once from
the historical heterogeneous 24-model pool after excluding unavailable models.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import full_games123_multiagent_batch as full  # noqa: E402
from strong_models_experiment import STRONG_MODELS_CONFIG  # noqa: E402


DEFAULT_POOL_CSV = (
    PROJECT_ROOT
    / "experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848"
    / "configs/heterogeneous_subset_maps/model_pool_24.csv"
)
DEFAULT_SEED = 20260628
EXCLUDED_MODELS = ("claude-sonnet-4-20250514",)
N_VALUES = (2, 4, 6, 8, 10)
GAME_LABELS = ("game1", "game2", "game3")
MODEL_ORDER_LABEL = "random_monoculture_control"
POOL_SIZE_AFTER_EXCLUSION = 23
SELECTED_MODEL_COUNT = 15
MODELS_PER_GAME = 5
ELO_BAND_COUNT = 5


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def default_results_root() -> Path:
    return PROJECT_ROOT / "experiments/results" / f"full_games123_random_monoculture_control_{timestamp()}"


def read_pool(pool_csv: Path) -> list[dict[str, Any]]:
    if not pool_csv.exists():
        raise FileNotFoundError(f"Model pool CSV not found: {pool_csv}")

    rows: list[dict[str, Any]] = []
    with pool_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            model = row.get("model") or row.get("model_name")
            elo_raw = row.get("elo") or row.get("elo_score") or row.get("arena_elo")
            if not model or elo_raw is None:
                raise ValueError(f"Could not parse model/elo from row in {pool_csv}: {row}")
            rows.append(
                {
                    "pool_index": int(row.get("pool_index", len(rows))),
                    "model": model,
                    "elo": int(float(elo_raw)),
                }
            )
    return rows


def split_evenly(items: list[dict[str, Any]], n_bands: int) -> list[list[dict[str, Any]]]:
    base = len(items) // n_bands
    extra = len(items) % n_bands
    bands: list[list[dict[str, Any]]] = []
    cursor = 0
    for band_index in range(n_bands):
        size = base + (1 if band_index < extra else 0)
        bands.append(items[cursor : cursor + size])
        cursor += size
    return bands


def select_models(
    pool_rows: list[dict[str, Any]],
    seed: int,
    excluded_models: tuple[str, ...] = EXCLUDED_MODELS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    excluded = set(excluded_models)
    filtered = [row for row in pool_rows if row["model"] not in excluded]
    if len(filtered) != POOL_SIZE_AFTER_EXCLUSION:
        raise ValueError(
            f"Expected {POOL_SIZE_AFTER_EXCLUSION} models after exclusions, got {len(filtered)}"
        )

    sorted_pool = sorted(filtered, key=lambda row: (row["elo"], row["model"]))
    bands = split_evenly(sorted_pool, ELO_BAND_COUNT)
    rng = random.Random(seed)

    band_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    assigned_by_game: dict[str, list[str]] = {game_label: [] for game_label in GAME_LABELS}

    for band_index, band in enumerate(bands, start=1):
        if len(band) < 3:
            raise ValueError(f"Elo band {band_index} has only {len(band)} models; need at least 3")

        elo_min = min(row["elo"] for row in band)
        elo_max = max(row["elo"] for row in band)
        band_label = f"band{band_index}_{elo_min}_{elo_max}"
        sampled = rng.sample(band, 3)
        rng.shuffle(sampled)

        for row in band:
            band_rows.append(
                {
                    "band_index": band_index,
                    "band_label": band_label,
                    "band_min_elo": elo_min,
                    "band_max_elo": elo_max,
                    "pool_index": row["pool_index"],
                    "model": row["model"],
                    "elo": row["elo"],
                    "selected": row in sampled,
                }
            )

        for selected_rank, row in enumerate(sampled, start=1):
            game_label = GAME_LABELS[selected_rank - 1]
            selected_row = {
                "band_index": band_index,
                "band_label": band_label,
                "band_min_elo": elo_min,
                "band_max_elo": elo_max,
                "pool_index": row["pool_index"],
                "model": row["model"],
                "elo": row["elo"],
                "assigned_game": game_label,
            }
            selected_rows.append(selected_row)
            assigned_by_game[game_label].append(row["model"])
            assignment_rows.append(
                {
                    "game_label": game_label,
                    "band_index": band_index,
                    "band_label": band_label,
                    "model": row["model"],
                    "elo": row["elo"],
                    "pool_index": row["pool_index"],
                }
            )

    if len(selected_rows) != SELECTED_MODEL_COUNT:
        raise ValueError(f"Expected {SELECTED_MODEL_COUNT} selected models, got {len(selected_rows)}")
    for game_label, models in assigned_by_game.items():
        if len(models) != MODELS_PER_GAME:
            raise ValueError(f"{game_label} has {len(models)} assigned models, expected {MODELS_PER_GAME}")

    selected_rows.sort(key=lambda row: (row["assigned_game"], row["band_index"]))
    assignment_rows.sort(key=lambda row: (row["game_label"], row["band_index"]))
    return band_rows, selected_rows, assignment_rows


def agent_model_map(models: list[str]) -> dict[str, str]:
    return {f"Agent_{index}": model for index, model in enumerate(models, start=1)}


def agent_elo_map(models: list[str]) -> dict[str, int | None]:
    return {
        f"Agent_{index}": STRONG_MODELS_CONFIG.get(model, {}).get("elo")
        for index, model in enumerate(models, start=1)
    }


def build_config(
    *,
    results_root: Path,
    config_number: int,
    seed: int,
    selected_row: dict[str, Any],
    game_label: str,
    n_agents: int,
    params: dict[str, Any],
    selected_models: list[str],
    pool_models: list[str],
    pool_source: Path,
) -> dict[str, Any]:
    model = selected_row["model"]
    models = [model] * n_agents
    competition_id = str(params["competition_id"])
    config_id = f"config_{config_number:04d}"
    model_token = full.sanitize_token(model)
    run_dir = (
        results_root
        / "runs"
        / f"{config_id}_{game_label}_n{n_agents}_{competition_id}_{model_token}"
    )
    stable = full.stable_seed(
        seed,
        "random_monoculture_control",
        game_label,
        n_agents,
        competition_id,
        model,
    )
    config: dict[str, Any] = {
        "config_id": config_id,
        "experiment_id": f"random_monoculture_control_{config_id}",
        "batch_type": "full_games123_random_monoculture_control",
        "experiment_family": "random_monoculture_control",
        "experiment_type": "random_monoculture_control",
        "game_label": game_label,
        "game_type": full.game_type_for_label(game_label),
        "n_agents": n_agents,
        "num_agents": n_agents,
        "models": models,
        "baseline_model": model,
        "monoculture_model": model,
        "model_order": MODEL_ORDER_LABEL,
        "agent_model_map": agent_model_map(models),
        "agent_elo_map": agent_elo_map(models),
        "agent_role_map": {agent_id: "random_monoculture_control" for agent_id in full.agent_ids(n_agents)},
        "max_rounds": 10,
        "discussion_turns": 2,
        "gamma_discount": 0.9,
        "parallel_phases": True,
        "random_seed": stable,
        "seed": stable,
        "run_number": 1,
        "output_dir": full.relative_or_absolute(run_dir),
        "competition_id": competition_id,
        "model_selection_seed": seed,
        "model_pool_size": len(pool_models),
        "model_pool": pool_models,
        "model_pool_source": str(pool_source),
        "excluded_models": list(EXCLUDED_MODELS),
        "selected_models": selected_models,
        "selected_model_count": len(selected_models),
        "elo_band_index": selected_row["band_index"],
        "elo_band_label": selected_row["band_label"],
        "elo_band_min": selected_row["band_min_elo"],
        "elo_band_max": selected_row["band_max_elo"],
        "model_pool_index": selected_row["pool_index"],
        "model_elo": selected_row["elo"],
        "game_assignment": game_label,
        "notes": (
            "Random-monoculture control: this run uses N copies of one model sampled "
            "from the historical heterogeneous model pool."
        ),
    }
    for key, value in params.items():
        if key != "competition_id":
            config[key] = value
    return config


def is_derisk_config(config: dict[str, Any]) -> bool:
    if int(config["n_agents"]) != 10:
        return False
    game_label = config["game_label"]
    if game_label == "game1":
        return float(config.get("competition_level", -1.0)) == 1.0
    if game_label == "game2":
        return (
            str(config.get("rho_label")) == "negative_lower_bound"
            and float(config.get("theta", -1.0)) == 0.8
        )
    if game_label == "game3":
        return float(config.get("sigma", -1.0)) == 0.2 and float(config.get("alpha", -1.0)) == 0.2
    return False


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_selection(
    *,
    results_root: Path,
    name: str,
    configs: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> Path:
    selections_dir = results_root / "selections"
    selections_dir.mkdir(parents=True, exist_ok=True)
    ids_path = selections_dir / f"{name}_config_ids.txt"
    ids_path.write_text("\n".join(config["config_id"] for config in configs) + "\n", encoding="utf-8")

    rows = [
        {
            "config_id": config["config_id"],
            "game_label": config["game_label"],
            "n_agents": config["n_agents"],
            "competition_id": config["competition_id"],
            "model": config["monoculture_model"],
            "model_elo": config["model_elo"],
            "elo_band_index": config["elo_band_index"],
            "output_dir": config["output_dir"],
        }
        for config in configs
    ]
    write_csv(selections_dir / f"{name}_index.csv", rows)
    full.write_json(
        selections_dir / f"{name}_manifest.json",
        {
            "selection_name": name,
            "results_root": str(results_root),
            "count": len(configs),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "model_selection_seed": manifest["model_selection_seed"],
            "config_ids_file": str(ids_path),
            "config_ids": [config["config_id"] for config in configs],
        },
    )
    return ids_path


def write_run_notes(
    results_root: Path,
    manifest: dict[str, Any],
    assignment_rows: list[dict[str, Any]],
    derisk_configs: list[dict[str, Any]],
) -> None:
    lines = [
        "# Random-Monoculture Control Batch",
        "",
        "This batch compares against the existing heterogeneous Games 1/2/3 sweep.",
        "Each config uses `models = [sampled_model] * N`.",
        "",
        "## Grid",
        "",
        "- Game 1: 5 N values x 5 competition levels x 5 models = 125 configs",
        "- Game 2: 5 N values x 4 rho/theta cells x 5 models = 100 configs",
        "- Game 3: 5 N values x 4 sigma/alpha cells x 5 models = 100 configs",
        "- Total: 325 configs",
        "",
        "## Model Selection",
        "",
        f"- Seed: `{manifest['model_selection_seed']}`",
        f"- Source pool: `{manifest['model_pool_source']}`",
        f"- Excluded models: `{', '.join(manifest['excluded_models'])}`",
        "- The remaining 23 models were sorted by Elo, split into five bands,",
        "  and three models were sampled from each band.",
        "- Each game receives one sampled model from each Elo band.",
        "",
        "## Game Assignments",
        "",
        "| Game | Band | Model | Elo |",
        "| --- | ---: | --- | ---: |",
    ]
    for row in assignment_rows:
        lines.append(
            f"| {row['game_label']} | {row['band_index']} | `{row['model']}` | {row['elo']} |"
        )
    lines.extend(
        [
            "",
            "## Derisk Selection",
            "",
            "The 15-config derisk set runs the highest-risk N=10 condition once per selected model:",
            "",
            "- Game 1: competition level 1.0",
            "- Game 2: rho at the negative lower bound, theta 0.8",
            "- Game 3: sigma 0.2, alpha 0.2",
            "",
            "Derisk config IDs:",
            "",
        ]
    )
    for config in derisk_configs:
        lines.append(
            f"- `{config['config_id']}`: {config['game_label']} N={config['n_agents']} "
            f"{config['competition_id']} `{config['monoculture_model']}`"
        )
    (results_root / "RUN_NOTES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_configs(args: argparse.Namespace) -> None:
    results_root = args.results_root.resolve()
    if results_root.exists() and any(results_root.iterdir()) and not args.force:
        raise FileExistsError(f"Results root already exists and is not empty: {results_root}")

    configs_dir = results_root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    (results_root / "runs").mkdir(parents=True, exist_ok=True)
    (results_root / "monitoring").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "slurm").mkdir(parents=True, exist_ok=True)

    pool_rows = read_pool(args.pool_csv)
    band_rows, selected_rows, assignment_rows = select_models(pool_rows, args.seed)
    pool_models = [row["model"] for row in sorted(pool_rows, key=lambda row: (row["elo"], row["model"])) if row["model"] not in EXCLUDED_MODELS]
    selected_models = [row["model"] for row in selected_rows]

    missing = sorted(model for model in selected_models if model not in STRONG_MODELS_CONFIG)
    if missing:
        raise ValueError(f"Selected models missing from STRONG_MODELS_CONFIG: {missing}")

    assigned_by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        assigned_by_game[row["assigned_game"]].append(row)

    configs: list[dict[str, Any]] = []
    config_number = 1
    for game_label in GAME_LABELS:
        for selected_row in sorted(assigned_by_game[game_label], key=lambda row: row["band_index"]):
            for n_agents in N_VALUES:
                for params in full.game_parameter_grid(game_label, n_agents):
                    configs.append(
                        build_config(
                            results_root=results_root,
                            config_number=config_number,
                            seed=args.seed,
                            selected_row=selected_row,
                            game_label=game_label,
                            n_agents=n_agents,
                            params=params,
                            selected_models=selected_models,
                            pool_models=pool_models,
                            pool_source=args.pool_csv,
                        )
                    )
                    config_number += 1

    manifest = {
        "batch_type": "full_games123_random_monoculture_control",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "results_root": str(results_root),
        "model_selection_seed": args.seed,
        "model_pool_source": str(args.pool_csv),
        "original_pool_size": len(pool_rows),
        "model_pool_size_after_exclusion": len(pool_models),
        "excluded_models": list(EXCLUDED_MODELS),
        "selected_model_count": len(selected_models),
        "selected_models": selected_models,
        "n_values": list(N_VALUES),
        "game_labels": list(GAME_LABELS),
        "expected_total_configs": 325,
        "expected_game_counts": {"game1": 125, "game2": 100, "game3": 100},
        "expected_derisk_configs": 15,
        "model_order": MODEL_ORDER_LABEL,
        "slurm_time": args.slurm_time,
        "slurm_max_concurrent": args.max_concurrent,
        "openrouter_proxy_poll_dir": "/home/jz4391/openrouter_proxy",
    }

    derisk_configs = [config for config in configs if is_derisk_config(config)]
    all_ids = [config["config_id"] for config in configs]
    for config in configs:
        full.write_json(configs_dir / f"{config['config_id']}.json", config)
    (configs_dir / "all_configs.txt").write_text("\n".join(all_ids) + "\n", encoding="utf-8")
    write_csv(configs_dir / "model_pool_23.csv", [row for row in sorted(pool_rows, key=lambda row: (row["elo"], row["model"])) if row["model"] not in EXCLUDED_MODELS])
    write_csv(configs_dir / "elo_bands.csv", band_rows)
    write_csv(configs_dir / "model_assignments.csv", assignment_rows)
    write_csv(
        configs_dir / "experiment_index.csv",
        [
            {
                "config_id": config["config_id"],
                "game_label": config["game_label"],
                "n_agents": config["n_agents"],
                "competition_id": config["competition_id"],
                "model": config["monoculture_model"],
                "model_elo": config["model_elo"],
                "elo_band_index": config["elo_band_index"],
                "output_dir": config["output_dir"],
            }
            for config in configs
        ],
    )
    full.write_json(results_root / "manifest.json", manifest)
    write_selection(results_root=results_root, name="all", configs=configs, manifest=manifest)
    write_selection(results_root=results_root, name="derisk", configs=derisk_configs, manifest=manifest)
    write_run_notes(results_root, manifest, assignment_rows, derisk_configs)

    validation_errors = validate_config_list(configs, manifest)
    if validation_errors:
        raise ValueError("Generated invalid config set:\n" + "\n".join(validation_errors))

    print(f"Generated {len(configs)} configs under {results_root}")
    print(f"Derisk selection: {results_root / 'selections/derisk_config_ids.txt'}")


def validate_config_list(configs: list[dict[str, Any]], manifest: dict[str, Any] | None = None) -> list[str]:
    errors: list[str] = []
    if manifest is None:
        manifest = {}

    if len(configs) != 325:
        errors.append(f"Expected 325 configs, found {len(configs)}")

    game_counts = Counter(config["game_label"] for config in configs)
    expected_game_counts = {"game1": 125, "game2": 100, "game3": 100}
    for game_label, expected in expected_game_counts.items():
        if game_counts[game_label] != expected:
            errors.append(f"{game_label}: expected {expected} configs, found {game_counts[game_label]}")

    selected_models = set(manifest.get("selected_models") or [])
    if selected_models and len(selected_models) != SELECTED_MODEL_COUNT:
        errors.append(f"Expected {SELECTED_MODEL_COUNT} selected models, found {len(selected_models)}")

    excluded = set(manifest.get("excluded_models") or EXCLUDED_MODELS)
    seen_ids: set[str] = set()
    per_game_model_counts: Counter[tuple[str, str]] = Counter()
    per_cell_counts: Counter[tuple[str, int, str]] = Counter()
    per_game_band: Counter[tuple[str, int]] = Counter()

    for config in configs:
        config_id = str(config.get("config_id"))
        if config_id in seen_ids:
            errors.append(f"Duplicate config_id: {config_id}")
        seen_ids.add(config_id)

        models = config.get("models") or []
        n_agents = int(config.get("n_agents", -1))
        monoculture_model = config.get("monoculture_model")
        game_label = config.get("game_label")

        if len(models) != n_agents:
            errors.append(f"{config_id}: model list length {len(models)} != n_agents {n_agents}")
        if len(set(models)) != 1:
            errors.append(f"{config_id}: models are not monoculture: {models}")
        if models and models[0] != monoculture_model:
            errors.append(f"{config_id}: monoculture_model does not match models[0]")
        if monoculture_model in excluded:
            errors.append(f"{config_id}: uses excluded model {monoculture_model}")
        if selected_models and monoculture_model not in selected_models:
            errors.append(f"{config_id}: model {monoculture_model} is not in selected set")
        if monoculture_model not in STRONG_MODELS_CONFIG:
            errors.append(f"{config_id}: model {monoculture_model} missing from STRONG_MODELS_CONFIG")
        if config.get("model_order") != MODEL_ORDER_LABEL:
            errors.append(f"{config_id}: unexpected model_order {config.get('model_order')}")
        if config.get("parallel_phases") is not True:
            errors.append(f"{config_id}: parallel_phases is not true")
        if int(config.get("max_rounds", -1)) != 10:
            errors.append(f"{config_id}: max_rounds is not 10")
        if int(config.get("discussion_turns", -1)) != 2:
            errors.append(f"{config_id}: discussion_turns is not 2")

        per_game_model_counts[(str(game_label), str(monoculture_model))] += 1
        per_cell_counts[(str(game_label), n_agents, str(config.get("competition_id")))] += 1
        per_game_band[(str(game_label), int(config.get("elo_band_index", -1)))] += 1

    for (game_label, model), count in per_game_model_counts.items():
        expected = 25 if game_label == "game1" else 20
        if count != expected:
            errors.append(f"{game_label}/{model}: expected {expected} configs, found {count}")

    for game_label in GAME_LABELS:
        expected_cells = 5 if game_label == "game1" else 4
        for n_agents in N_VALUES:
            cells = [count for (g, n, _), count in per_cell_counts.items() if g == game_label and n == n_agents]
            if len(cells) != expected_cells:
                errors.append(f"{game_label} N={n_agents}: expected {expected_cells} competition cells, found {len(cells)}")
            for count in cells:
                if count != MODELS_PER_GAME:
                    errors.append(f"{game_label} N={n_agents}: expected {MODELS_PER_GAME} models per cell, found {count}")

    for game_label in GAME_LABELS:
        for band_index in range(1, ELO_BAND_COUNT + 1):
            expected = 25 if game_label == "game1" else 20
            observed = per_game_band[(game_label, band_index)]
            if observed != expected:
                errors.append(
                    f"{game_label} band {band_index}: expected {expected} configs, found {observed}"
                )

    derisk_configs = [config for config in configs if is_derisk_config(config)]
    if len(derisk_configs) != 15:
        errors.append(f"Expected 15 derisk configs, found {len(derisk_configs)}")

    return errors


def validate_results_root(args: argparse.Namespace) -> None:
    results_root = args.results_root.resolve()
    manifest = full.read_json_file(results_root / "manifest.json")
    configs = full.load_configs(results_root)
    errors = validate_config_list(configs, manifest)
    if errors:
        print("Validation failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print(f"Validation passed for {len(configs)} configs under {results_root}")


def load_config_by_id(results_root: Path, config_id: str | int) -> dict[str, Any]:
    if isinstance(config_id, int):
        filename = f"config_{config_id:04d}.json"
    else:
        text_id = str(config_id)
        filename = f"{text_id}.json" if text_id.startswith("config_") else f"config_{int(text_id):04d}.json"
    path = results_root / "configs" / filename
    return full.read_json_file(path)


def run_one(args: argparse.Namespace) -> None:
    results_root = args.results_root.resolve()
    config = load_config_by_id(results_root, args.config_id)
    max_tokens_per_phase_raw = os.getenv("RMC_RUNTIME_MAX_TOKENS_PER_PHASE")
    if max_tokens_per_phase_raw:
        max_tokens_per_phase = int(max_tokens_per_phase_raw)
        for key in (
            "max_tokens_proposal",
            "max_tokens_voting",
            "max_tokens_reflection",
            "max_tokens_thinking",
        ):
            config[key] = max_tokens_per_phase
    overrides_raw = os.getenv("RMC_RUNTIME_CONFIG_OVERRIDES_JSON")
    if overrides_raw:
        overrides = json.loads(overrides_raw)
        if not isinstance(overrides, dict):
            raise ValueError("RMC_RUNTIME_CONFIG_OVERRIDES_JSON must decode to a JSON object")
        config.update(overrides)
    status = full.run_config(results_root, runtime_config(config))
    if status.get("state") != "SUCCESS":
        raise SystemExit(1)


def read_config_ids(selection_path: Path) -> list[str]:
    return [line.strip() for line in selection_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def config_number(config_id: str | int) -> int:
    if isinstance(config_id, int):
        return config_id
    text_id = str(config_id)
    if text_id.startswith("config_"):
        return int(text_id.removeprefix("config_"))
    return int(text_id)


def config_succeeded(config: dict[str, Any]) -> bool:
    result_path = full.result_path_for(config)
    if result_path is None or not result_path.exists():
        return False
    return full.validate_result_file(runtime_config(config), result_path) is None


def runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    """Adapt readable config IDs to the existing runner's numeric ID convention."""
    adapted = dict(config)
    adapted["config_id"] = config_number(config["config_id"])
    return adapted


def write_slurm_file(
    *,
    results_root: Path,
    task_file: Path,
    job_name: str,
    slurm_time: str,
) -> Path:
    slurm_dir = results_root / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    slurm_path = slurm_dir / f"{job_name}.sbatch"
    log_prefix = PROJECT_ROOT / "slurm" / f"{job_name}_%A_%a"
    script_path = PROJECT_ROOT / "scripts/random_monoculture_control_batch.py"
    content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time={slurm_time}
#SBATCH --output={log_prefix}.out
#SBATCH --error={log_prefix}.err

set -eo pipefail

cd {PROJECT_ROOT}

module load anaconda3/2024.2
module load proxy/default

KEY_ENV_FILE="${{BARGAIN_API_KEYS_ENV:-bargain/api_keys.env}}"
if [ -f "$KEY_ENV_FILE" ]; then
  set -a
  source "$KEY_ENV_FILE"
  set +a
fi

export PYTHONPATH="{PROJECT_ROOT}:${{PYTHONPATH:-}}"
export OPENROUTER_TRANSPORT="${{OPENROUTER_TRANSPORT:-proxy}}"
export OPENROUTER_PROXY_POLL_DIR="${{OPENROUTER_PROXY_POLL_DIR:-/home/jz4391/openrouter_proxy}}"
export OPENROUTER_PROXY_CLIENT_TIMEOUT="${{OPENROUTER_PROXY_CLIENT_TIMEOUT:-9000}}"
export OPENROUTER_API_TIMEOUT="${{OPENROUTER_API_TIMEOUT:-1800}}"
export LLM_FAILURE_REPORT_PATH="${{LLM_FAILURE_REPORT_PATH:-{results_root}/monitoring/provider_failures.md}}"

CONFIG_ID=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {task_file})
if [ -z "$CONFIG_ID" ]; then
  echo "No config id found for SLURM_ARRAY_TASK_ID=${{SLURM_ARRAY_TASK_ID}}" >&2
  exit 1
fi

python {script_path} run-one --results-root {results_root} --config-id "$CONFIG_ID"
"""
    slurm_path.write_text(content, encoding="utf-8")
    return slurm_path


def submit_selection(args: argparse.Namespace) -> None:
    results_root = args.results_root.resolve()
    manifest = full.read_json_file(results_root / "manifest.json")

    if args.selection_file:
        selection_path = args.selection_file.resolve()
        selection_name = full.selection_slug(selection_path.stem)
    else:
        selection_name = full.selection_slug(args.selection_name)
        selection_path = results_root / "selections" / f"{selection_name}_config_ids.txt"
    if not selection_path.exists():
        raise FileNotFoundError(f"Selection file not found: {selection_path}")

    config_ids = read_config_ids(selection_path)
    if not config_ids:
        raise ValueError(f"Selection is empty: {selection_path}")

    selected_configs: list[dict[str, Any]] = []
    for config_id in config_ids:
        config = load_config_by_id(results_root, config_id)
        if args.rerun_existing or not config_succeeded(config):
            selected_configs.append(config)

    if not selected_configs:
        print(f"No configs need submission for selection {selection_name}")
        return

    submit_dir = results_root / "submissions"
    submit_dir.mkdir(parents=True, exist_ok=True)
    stamp = timestamp()
    task_file = submit_dir / f"{selection_name}_{stamp}_tasks.txt"
    task_file.write_text(
        "\n".join(config["config_id"] for config in selected_configs) + "\n",
        encoding="utf-8",
    )
    job_name = full.selection_slug(f"rmc_{selection_name}_{stamp}")[:40]
    slurm_path = write_slurm_file(
        results_root=results_root,
        task_file=task_file,
        job_name=job_name,
        slurm_time=args.slurm_time or manifest.get("slurm_time", "08:00:00"),
    )
    array_spec = f"1-{len(selected_configs)}%{max(1, args.max_concurrent)}"
    command = ["sbatch", "--array", array_spec, str(slurm_path)]

    submission_record = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "selection_name": selection_name,
        "selection_file": str(selection_path),
        "task_file": str(task_file),
        "slurm_file": str(slurm_path),
        "config_count": len(selected_configs),
        "array_spec": array_spec,
        "command": command,
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        print("Dry run:")
        print(" ".join(command))
        full.write_json(submit_dir / f"{selection_name}_{stamp}_dry_run.json", submission_record)
        return

    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=True)
    output = (result.stdout or result.stderr).strip()
    job_id = full.parse_job_id(output)
    submission_record["sbatch_output"] = output
    submission_record["job_id"] = job_id
    full.write_json(submit_dir / f"{selection_name}_{stamp}_submission.json", submission_record)
    print(output)
    print(f"Submitted {len(selected_configs)} configs from {selection_name}: job_id={job_id}")
    print(f"Task file: {task_file}")
    print(f"Slurm file: {slurm_path}")


def status_path_for(results_root: Path, config: dict[str, Any]) -> Path:
    return results_root / "status" / f"config_{config_number(config['config_id']):04d}.json"


def summarize_results_root(results_root: Path) -> dict[str, Any]:
    configs = full.load_configs(results_root)
    state_counts: Counter[str] = Counter()
    game_counts: Counter[str] = Counter()
    game_state_counts: Counter[tuple[str, str]] = Counter()
    derisk_state_counts: Counter[str] = Counter()
    durations: list[float] = []

    for config in configs:
        game_label = str(config["game_label"])
        game_counts[game_label] += 1
        result_path = full.result_path_for(config)
        runtime = runtime_config(config)
        if result_path is not None and full.validate_result_file(runtime, result_path) is None:
            state = "SUCCESS"
        else:
            status = full.read_json_file(status_path_for(results_root, config))
            state = str(status.get("state") or "NOT_STARTED")
            if state == "SUCCESS" and result_path is not None:
                result_error = full.validate_result_file(runtime, result_path)
                if result_error is not None:
                    state = "FAILED_VALIDATION"
            if status.get("duration_seconds") is not None:
                try:
                    durations.append(float(status["duration_seconds"]))
                except (TypeError, ValueError):
                    pass
        state_counts[state] += 1
        game_state_counts[(game_label, state)] += 1
        if is_derisk_config(config):
            derisk_state_counts[state] += 1

    return {
        "results_root": str(results_root),
        "total_configs": len(configs),
        "state_counts": dict(sorted(state_counts.items())),
        "game_counts": dict(sorted(game_counts.items())),
        "game_state_counts": {
            f"{game_label}:{state}": count
            for (game_label, state), count in sorted(game_state_counts.items())
        },
        "derisk_state_counts": dict(sorted(derisk_state_counts.items())),
        "duration_seconds": {
            "count": len(durations),
            "min": min(durations) if durations else None,
            "max": max(durations) if durations else None,
            "mean": (sum(durations) / len(durations)) if durations else None,
        },
    }


def print_local_summary(summary_obj: dict[str, Any]) -> None:
    print(f"Results root: {summary_obj['results_root']}")
    print(f"Total configs: {summary_obj['total_configs']}")
    print("States:")
    for state, count in summary_obj["state_counts"].items():
        print(f"  {state}: {count}")
    print("Derisk states:")
    for state, count in summary_obj["derisk_state_counts"].items():
        print(f"  {state}: {count}")
    print("Game counts:")
    for game_label, count in summary_obj["game_counts"].items():
        print(f"  {game_label}: {count}")


def summary(args: argparse.Namespace) -> None:
    results_root = args.results_root.resolve()
    summary_obj = summarize_results_root(results_root)
    if args.json:
        print(json.dumps(summary_obj, indent=2, sort_keys=True))
    else:
        print_local_summary(summary_obj)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate the 325-config batch")
    generate.add_argument("--results-root", type=Path, default=default_results_root())
    generate.add_argument("--pool-csv", type=Path, default=DEFAULT_POOL_CSV)
    generate.add_argument("--seed", type=int, default=DEFAULT_SEED)
    generate.add_argument("--slurm-time", default="08:00:00")
    generate.add_argument("--max-concurrent", type=int, default=40)
    generate.add_argument("--force", action="store_true")
    generate.set_defaults(func=generate_configs)

    validate = subparsers.add_parser("validate", help="Validate generated configs")
    validate.add_argument("--results-root", type=Path, required=True)
    validate.set_defaults(func=validate_results_root)

    run_one_parser = subparsers.add_parser("run-one", help="Run one config by config_id")
    run_one_parser.add_argument("--results-root", type=Path, required=True)
    run_one_parser.add_argument("--config-id", required=True)
    run_one_parser.set_defaults(func=run_one)

    submit = subparsers.add_parser("submit-selection", help="Submit a config-id selection as a Slurm array")
    submit.add_argument("--results-root", type=Path, required=True)
    submit.add_argument("--selection-name", default="derisk")
    submit.add_argument("--selection-file", type=Path)
    submit.add_argument("--max-concurrent", type=int, default=40)
    submit.add_argument("--slurm-time", default=None)
    submit.add_argument("--rerun-existing", action="store_true")
    submit.add_argument("--dry-run", action="store_true")
    submit.set_defaults(func=submit_selection)

    summary_parser = subparsers.add_parser("summary", help="Summarize config/result status")
    summary_parser.add_argument("--results-root", type=Path, required=True)
    summary_parser.add_argument("--json", action="store_true")
    summary_parser.set_defaults(func=summary)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    started = time.time()
    args.func(args)
    elapsed = time.time() - started
    if args.command not in {"summary"}:
        print(f"Done in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

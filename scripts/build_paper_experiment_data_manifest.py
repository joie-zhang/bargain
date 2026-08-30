#!/usr/bin/env python3
"""Build and validate the paper's 7,143-row retained-corpus manifest."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import full_games123_multiagent_batch as multiagent  # noqa: E402
import random_monoculture_control_batch as monoculture  # noqa: E402


N2_RUNS = (
    PROJECT_ROOT
    / "experiments/results/n2_baseline_comparison_analysis_20260505/primary_runs_with_metrics.csv"
)
MULTIAGENT_TABLE_DIR = (
    PROJECT_ROOT
    / "experiments/results/n2_plus_multiagent_comparison_analysis_20260505/tables_multiagent"
)
MONOCULTURE_ROOT = (
    PROJECT_ROOT
    / "experiments/results/full_games123_random_monoculture_control_20260628_014357"
)
TTC_ANALYSIS_DIR = PROJECT_ROOT / "analysis/ttc_complete_family_seed_panels_20260810"
TTC_RETAINED_RUNS = TTC_ANALYSIS_DIR / "run_level_complete_panels.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "docs/reproducibility/paper_experiment_data_manifest.csv"
DEFAULT_IDENTIFIER_REPORT = (
    PROJECT_ROOT / "docs/reproducibility/paper_experiment_identifier_validation.json"
)

EXPECTED_COUNTS = {
    ("bilateral_gpt5_nano", "game1"): 420,
    ("bilateral_gpt5_nano", "game2"): 540,
    ("bilateral_gpt5_nano", "game3"): 540,
    ("bilateral_llama33", "game1"): 140,
    ("bilateral_llama33", "game2"): 180,
    ("bilateral_llama33", "game3"): 180,
    ("multiagent_homogeneous", "game1"): 550,
    ("multiagent_homogeneous", "game2"): 440,
    ("multiagent_homogeneous", "game3"): 440,
    ("multiagent_heterogeneous", "game1"): 500,
    ("multiagent_heterogeneous", "game2"): 400,
    ("multiagent_heterogeneous", "game3"): 400,
    ("random_monoculture", "game1"): 125,
    ("random_monoculture", "game2"): 100,
    ("random_monoculture", "game3"): 100,
    ("ttc", "game1"): 696,
    ("ttc", "game2"): 696,
    ("ttc", "game3"): 696,
}

FIELDNAMES = (
    "paper_run_id",
    "paper_batch",
    "source_root",
    "experiment_family",
    "game",
    "n_agents",
    "config_id",
    "config_path",
    "result_path",
    "rollout_path",
    "result_bytes",
    "rollout_bytes",
    "canonical_index",
)


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def resolve_repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def source_root_for(path: Path) -> str:
    relative_path = path.resolve().relative_to(PROJECT_ROOT / "experiments/results")
    return str(Path("experiments/results") / relative_path.parts[0])


def rollout_for_result(result_path: Path) -> Path:
    match = re.fullmatch(r"(run_\d+)_experiment_results\.json", result_path.name)
    candidates = []
    if match:
        candidates.append(result_path.with_name(f"{match.group(1)}_all_interactions.json"))
    candidates.extend(
        [
            result_path.with_name("all_interactions.json"),
            result_path.with_name("run_1_all_interactions.json"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No rollout file for {result_path}")


def validate_result(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("final_utilities"), dict):
        raise ValueError(f"Result does not contain a final_utilities object: {path}")


def make_row(
    *,
    paper_batch: str,
    experiment_family: str,
    game: str,
    n_agents: int,
    config_id: str,
    config_path: Path,
    result_path: Path,
    canonical_index: Path,
) -> dict[str, Any]:
    validate_result(result_path)
    rollout_path = rollout_for_result(result_path)
    return {
        "paper_run_id": "",
        "paper_batch": paper_batch,
        "source_root": source_root_for(result_path),
        "experiment_family": experiment_family,
        "game": game,
        "n_agents": n_agents,
        "config_id": config_id,
        "config_path": relative(config_path),
        "result_path": relative(result_path),
        "rollout_path": relative(rollout_path),
        "result_bytes": result_path.stat().st_size,
        "rollout_bytes": rollout_path.stat().st_size,
        "canonical_index": relative(canonical_index),
    }


def load_n2_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with N2_RUNS.open(newline="", encoding="utf-8") as handle:
        for item in csv.DictReader(handle):
            result_path = resolve_repo_path(item["result_path"])
            source_root = Path(source_root_for(result_path))
            config_path = PROJECT_ROOT / source_root / "configs" / item["config_file"]
            paper_batch = (
                "bilateral_gpt5_nano"
                if item["baseline_key"] == "gpt5_nano"
                else "bilateral_llama33"
            )
            rows.append(
                make_row(
                    paper_batch=paper_batch,
                    experiment_family="bilateral_fixed_baseline",
                    game=item["game_id"],
                    n_agents=2,
                    config_id=item["config_file"].removesuffix(".json"),
                    config_path=config_path,
                    result_path=result_path,
                    canonical_index=N2_RUNS,
                )
            )
    return rows


def load_multiagent_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sources = (
        (
            "multiagent_homogeneous",
            "homogeneous_runs_fresh.csv",
            PROJECT_ROOT / "experiments/results/full_games123_multiagent_production_20260428_085255",
        ),
        (
            "multiagent_heterogeneous",
            "heterogeneous_runs_fresh.csv",
            PROJECT_ROOT
            / "experiments/results/full_games123_multiagent_heterogeneous_equal_width_openrouter_repair_20260429_113848",
        ),
    )
    for paper_batch, filename, root in sources:
        index_path = MULTIAGENT_TABLE_DIR / filename
        with index_path.open(newline="", encoding="utf-8") as handle:
            for item in csv.DictReader(handle):
                config_number = int(float(item["config_id"]))
                rows.append(
                    make_row(
                        paper_batch=paper_batch,
                        experiment_family=item["experiment_family"],
                        game=item["game_label"],
                        n_agents=int(float(item["n_agents"])),
                        config_id=f"config_{config_number:04d}",
                        config_path=root / "configs" / f"config_{config_number:04d}.json",
                        result_path=resolve_repo_path(item["result_path"]),
                        canonical_index=index_path,
                    )
                )
    return rows


def load_monoculture_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config in multiagent.load_configs(MONOCULTURE_ROOT):
        result_path = multiagent.result_path_for(config)
        if result_path is None:
            raise FileNotFoundError(f"No result for {config['config_id']}")
        error = multiagent.validate_result_file(monoculture.runtime_config(config), result_path)
        if error is not None:
            raise ValueError(f"Invalid result for {config['config_id']}: {error}")
        config_id = str(config["config_id"])
        rows.append(
            make_row(
                paper_batch="random_monoculture",
                experiment_family="random_monoculture_control",
                game=str(config["game_label"]),
                n_agents=int(config["n_agents"]),
                config_id=config_id,
                config_path=MONOCULTURE_ROOT / "configs" / f"{config_id}.json",
                result_path=result_path,
                canonical_index=MONOCULTURE_ROOT / "manifest.json",
            )
        )
    return rows


def load_ttc_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with TTC_RETAINED_RUNS.open(newline="", encoding="utf-8") as handle:
        for item in csv.DictReader(handle):
            if item.get("family") == "claude-sonnet-4-6" and int(item["seed"]) == 612:
                raise ValueError("Off-protocol seed-612 Claude panel entered retained TTC manifest")
            rows.append(
                make_row(
                    paper_batch="ttc",
                    experiment_family="test_time_compute_comparable_complete_panels",
                    game=str(item["game"]),
                    n_agents=2,
                    config_id=(
                        f"seed_{int(item['seed']):04d}_config_{int(item['config_id']):04d}"
                    ),
                    config_path=PROJECT_ROOT / item["config_path"],
                    result_path=PROJECT_ROOT / item["result_path"],
                    canonical_index=TTC_RETAINED_RUNS,
                )
            )
    return rows


def validate_inventory(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 7143:
        raise ValueError(f"Expected 7,143 rows, got {len(rows):,}")
    counts = Counter((row["paper_batch"], row["game"]) for row in rows)
    if counts != Counter(EXPECTED_COUNTS):
        raise ValueError(f"Count mismatch: expected {EXPECTED_COUNTS}, got {dict(counts)}")
    for field in ("config_path", "result_path", "rollout_path"):
        missing = [row[field] for row in rows if not (PROJECT_ROOT / row[field]).exists()]
        if missing:
            raise FileNotFoundError(f"Missing {field} files: {missing[:10]}")
    for field in ("result_path", "rollout_path"):
        values = [row[field] for row in rows]
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate {field} entries")


def validate_identifier_equality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Require each result and all records in its rollout to name one experiment."""
    failures: list[dict[str, Any]] = []
    records_checked = 0
    for index, row in enumerate(rows, start=1):
        result_path = PROJECT_ROOT / row["result_path"]
        rollout_path = PROJECT_ROOT / row["rollout_path"]
        result = json.loads(result_path.read_text(encoding="utf-8"))
        rollout = json.loads(rollout_path.read_text(encoding="utf-8"))
        result_id = result.get("experiment_id") if isinstance(result, dict) else None
        rollout_ids: set[str | None] = set()
        missing_id_records = 0
        if not isinstance(rollout, list):
            failures.append(
                {
                    "paper_run_id": row["paper_run_id"],
                    "result_path": row["result_path"],
                    "rollout_path": row["rollout_path"],
                    "error": "rollout is not a JSON list",
                }
            )
            continue
        for record in rollout:
            records_checked += 1
            if not isinstance(record, dict) or not record.get("experiment_id"):
                missing_id_records += 1
                continue
            rollout_ids.add(str(record["experiment_id"]))
        if not result_id or rollout_ids != {str(result_id)} or missing_id_records:
            failures.append(
                {
                    "paper_run_id": row["paper_run_id"],
                    "result_path": row["result_path"],
                    "rollout_path": row["rollout_path"],
                    "result_experiment_id": result_id,
                    "rollout_experiment_ids": sorted(value for value in rollout_ids if value is not None),
                    "rollout_records": len(rollout),
                    "rollout_records_without_experiment_id": missing_id_records,
                }
            )
        if index % 500 == 0:
            print(f"Checked result-rollout identifiers for {index:,}/{len(rows):,} runs", flush=True)
    return {
        "manifest_rows": len(rows),
        "rollout_records_checked": records_checked,
        "failure_count": len(failures),
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="Validate without writing the CSV.")
    parser.add_argument(
        "--check-identifiers",
        action="store_true",
        help="Also require result experiment_id equality with every rollout record.",
    )
    parser.add_argument("--identifier-report", type=Path, default=DEFAULT_IDENTIFIER_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_n2_rows() + load_multiagent_rows() + load_monoculture_rows() + load_ttc_rows()
    rows.sort(key=lambda row: (row["paper_batch"], row["game"], row["config_id"], row["result_path"]))
    for index, row in enumerate(rows, start=1):
        row["paper_run_id"] = f"paper_run_{index:04d}"
    validate_inventory(rows)
    if args.check_identifiers:
        identifier_report = validate_identifier_equality(rows)
        report_path = (
            args.identifier_report
            if args.identifier_report.is_absolute()
            else PROJECT_ROOT / args.identifier_report
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(identifier_report, indent=2) + "\n", encoding="utf-8")
        if identifier_report["failure_count"]:
            raise ValueError(
                f"Result-rollout identifier mismatches: {identifier_report['failure_count']}; "
                f"see {report_path}"
            )
        print(
            f"Validated identifier equality for {identifier_report['manifest_rows']:,} runs "
            f"and {identifier_report['rollout_records_checked']:,} rollout records."
        )
    if not args.check:
        output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        print(output)
    print(
        f"Validated {len(rows):,} paper-run rows and the existence of their config, "
        "result, and rollout paths."
    )


if __name__ == "__main__":
    main()

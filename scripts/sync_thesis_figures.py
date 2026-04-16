#!/usr/bin/env python3
"""
Copy the thesis figure set into a repo-root Figures/ directory.

This keeps a stable folder layout for exporting plots to Overleaf:

    Figures/
      game_1/
      game_2/
      game_3/

Usage:
    python scripts/sync_thesis_figures.py
    python scripts/sync_thesis_figures.py --zip
    python scripts/sync_thesis_figures.py --output-root Figures
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FIGURES_BY_GAME = {
    "game_1": [
        "experiments/results/scaling_experiment_20260404_064451/analysis/average_utility_vs_elo.png",
        "experiments/results/scaling_experiment_20260404_064451/analysis/average_utility_vs_elo_by_competition_level.png",
        "experiments/results/scaling_experiment_20260404_064451/analysis/average_rounds_to_consensus_vs_elo.png",
        "experiments/results/scaling_experiment_20260404_064451/analysis/average_utility_vs_competition_level_aggregated_over_models.png",
    ],
    "game_2": [
        "visualization/figures/diplomacy_20260405_082215_summary/utility_vs_elo_overall.png",
        "visualization/figures/diplomacy_20260405_082215_summary/utility_vs_elo_by_competition_index.png",
        "visualization/figures/diplomacy_20260405_082215_summary/utility_vs_elo_by_rho_theta.png",
    ],
    "game_3": [
        "experiments/results/cofunding_20260405_083548/analysis/utility_vs_elo_all_models.png",
        "experiments/results/cofunding_20260405_083548/analysis/utility_vs_elo_by_competition_index.png",
        "experiments/results/cofunding_20260405_083548/analysis/utility_vs_elo_by_alpha_sigma.png",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy the thesis plot set into repo-root Figures/game_{1,2,3}."
    )
    parser.add_argument(
        "--output-root",
        default="Figures",
        help="Output directory, relative to the repo root unless absolute.",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Also write a zip archive containing the synced Figures directory.",
    )
    parser.add_argument(
        "--zip-name",
        default="Figures.zip",
        help="Zip archive filename, relative to the repo root unless absolute.",
    )
    return parser.parse_args()


def resolve_repo_path(path_value: str) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def validate_sources() -> list[tuple[str, Path]]:
    missing: list[tuple[str, Path]] = []
    for game_name, rel_paths in FIGURES_BY_GAME.items():
        for rel_path in rel_paths:
            src = PROJECT_ROOT / rel_path
            if not src.exists():
                missing.append((game_name, src))
    return missing


def copy_figures(output_root: Path) -> list[Path]:
    copied_paths: list[Path] = []
    output_root.mkdir(parents=True, exist_ok=True)

    for game_name, rel_paths in FIGURES_BY_GAME.items():
        game_dir = output_root / game_name
        game_dir.mkdir(parents=True, exist_ok=True)
        for rel_path in rel_paths:
            src = PROJECT_ROOT / rel_path
            dest = game_dir / src.name
            shutil.copy2(src, dest)
            copied_paths.append(dest)
            print(f"Copied {display_path(src)} -> {display_path(dest)}")

    return copied_paths


def write_zip(output_root: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_root.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=path.relative_to(PROJECT_ROOT))

    print(f"Wrote zip archive: {display_path(zip_path)}")


def main() -> int:
    args = parse_args()
    output_root = resolve_repo_path(args.output_root)
    zip_path = resolve_repo_path(args.zip_name)

    missing = validate_sources()
    if missing:
        print("Missing source figures:", file=sys.stderr)
        for game_name, src in missing:
            print(f"  [{game_name}] {src}", file=sys.stderr)
        return 1

    copied_paths = copy_figures(output_root)
    print(f"Copied {len(copied_paths)} figure(s) into {display_path(output_root)}")

    if args.zip:
        write_zip(output_root, zip_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

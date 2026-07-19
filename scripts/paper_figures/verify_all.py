#!/usr/bin/env python3
"""Regenerate paper figures without changing the saved paper assets."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VENV_PYTHON = ROOT / ".venv/bin/python"
LOG_ROOT = ROOT / "analysis/paper_figure_verification"


@dataclass(frozen=True)
class Task:
    name: str
    args: tuple[str, ...]
    outputs: tuple[str, ...]
    inputs: tuple[str, ...] = ()


TASKS = (
    Task(
        "bilateral overview",
        ("scripts/paper_figures/render_figure2_large_fonts.py",),
        ("analysis/recreated_figures/figure2_bilateral_overview_combined_large_fonts.png",),
        ("experiments/results/n2_baseline_comparison_analysis_20260505/all_runs_with_metrics.csv",),
    ),
    Task(
        "bilateral qualitative figure",
        ("scripts/paper_figures/make_combined_aligned_font_balanced.py",),
        ("analysis/recreated_figures/figure4_n2_qualitative/figure4_combined_lhs_rhs_smooth5_outlier_removed_square_font_balanced_aligned.*",),
        (
            "overleaf/icml_aiwild_template/graphics/qualitative_n2/n2_group_payoff_corr_dedup.csv",
            "overleaf/icml_aiwild_template/graphics/qualitative_n2/n2_group_intensity_dedup.csv",
        ),
    ),
    Task(
        "TTC payoff by observed tokens",
        ("scripts/paper_figures/plot_ttc_game_averaged_observed_tokens.py",),
        (
            "overleaf/neurips/graphics/ttc_game_averaged_target_payoff_vs_compute.png",
            "overleaf/neurips/graphics/ttc_game_averaged_target_payoff_vs_compute_gray_points.png",
            "analysis/neurips_revision_20260504/ttc_game_averaged_by_effort.csv",
        ),
        ("analysis/neurips_revision_20260504/ttc_order_averaged.csv",),
    ),
    Task(
        "bilateral baseline payoff",
        ("scripts/paper_figures/plot_figure3_baseline_by_competition_ewma_iteration.py",),
        (
            "overleaf/neurips/graphics/n2_gpt5_nano/04_baseline_payoff_by_competition.png",
            "overleaf/neurips/graphics/n2_gpt5_nano/04_baseline_payoff_by_competition_raw_per_elo.png",
            "experiments/results/figure_iteration_20260507/gpt5_nano/figure3_baseline_payoff_by_competition_iteration.png",
        ),
        ("experiments/results/n2_baseline_comparison_analysis_20260505/all_runs_with_metrics.csv",),
    ),
    Task(
        "bilateral total welfare",
        ("scripts/paper_figures/plot_figure4_total_welfare_by_competition_ewma.py",),
        (
            "overleaf/neurips/graphics/n2_gpt5_nano/10_total_welfare_by_competition_ewma.png",
            "experiments/results/figure_iteration_20260507/gpt5_nano/figure4_welfare_by_competition_attainable_welfare.csv",
        ),
        ("experiments/results/n2_baseline_comparison_analysis_20260505/all_runs_with_metrics.csv",),
    ),
    Task(
        "fair-share residual",
        ("scripts/paper_figures/plot_fairshare_residual_combined.py",),
        (
            "overleaf/icml_aiwild_template/graphics/n2_gpt5_nano/fairshare_residual_combined.png",
            "overleaf/neurips/graphics/n2_gpt5_nano/fairshare_residual_combined.png",
        ),
        ("scripts/paper_figures/assets/fairshare_residual_combined_base.png",),
    ),
    Task(
        "bilateral fairness distance",
        ("scripts/paper_figures/plot_n2_fairness_distance_three_game_curves.py",),
        ("overleaf/neurips/graphics/n2_gpt5_nano/11_fairness_distance_three_game_curves*",),
        ("experiments/results/n2_baseline_comparison_analysis_20260505/overall_by_model_game.csv",),
    ),
    Task(
        "endpoint fairness",
        ("scripts/paper_figures/plot_n2_endpoint_fairness_style_matched_drop_game2_outlier.py",),
        ("overleaf/neurips/graphics/n2_gpt5_nano/fairness_explanation/baseline_adversary_fair_share_symmetric_percent_endpoints_tall_*",),
        (
            "analysis/nash_lindahl_fairness_20260505/agent_metrics.csv",
            "scripts/paper_figures/assets/endpoint_fairness_elo_snapshot.csv",
        ),
    ),
    Task(
        "TTC payoff by effort",
        ("scripts/paper_figures/plot_ttc_effort_adversary_baseline.py",),
        ("overleaf/neurips/graphics/ttc_full/overall_by_effort_adversary_baseline_combined.png",),
        ("experiments/results/ttc_native_scaling_20260502_212943/monitoring/partial_results_latest.csv",),
    ),
    Task(
        "ICML TTC figures",
        ("scripts/paper_figures/plot_icml_ttc_main_figures.py",),
        (
            "overleaf/icml_aiwild_template/graphics/ttc_game_averaged_target_payoff_vs_compute.png",
            "overleaf/icml_aiwild_template/graphics/qualitative_ttc/ttc_group_intensity_singlecolumn_3x2.*",
            "overleaf/icml_aiwild_template/graphics/qualitative_ttc/ttc_group_intensity_fullwidth_2x3_compact.*",
        ),
        (
            "analysis/neurips_revision_20260504/ttc_game_averaged_by_effort.csv",
            "analysis/ttc_group_intensity_turn_dedup_verification_20260701/ttc_group_intensity_turn_dedup_summary.csv",
        ),
    ),
    Task(
        "heterogeneous utility distribution",
        ("scripts/paper_figures/plot_n_gt_2_heterogeneous_utility_distribution.py",),
        (
            "overleaf/neurips/graphics/n_gt_2_report/heterogeneous_payoff_vs_arena_elo_by_n_recreated.png",
            "overleaf/neurips/graphics/n_gt_2_report/heterogeneous_utility_by_elo_bucket_by_n*",
            "overleaf/neurips/graphics/n_gt_2_report/heterogeneous_elo_bucket_ranges.csv",
            "experiments/results/figure_iteration_20260626/multiagent_utility_distribution/*",
        ),
        ("experiments/results/n2_plus_multiagent_comparison_analysis_20260505/tables_multiagent/heterogeneous_agents_fresh.csv",),
    ),
    Task(
        "heterogeneous versus monoculture Gini",
        (
            "scripts/paper_figures/plot_random_monoculture_gini_vs_heterogeneous.py",
            "--output-dir",
            "analysis/recreated_figures/figure7_from_script",
        ),
        ("analysis/recreated_figures/figure7_from_script/*",),
        (
            "experiments/results/full_games123_random_monoculture_control_20260628_014357",
            "experiments/results/n2_plus_multiagent_comparison_analysis_20260505/tables_multiagent/heterogeneous_runs_fresh.csv",
        ),
    ),
    Task(
        "final Gini label rendering",
        ("scripts/paper_figures/render_figure7_label_edits.py",),
        ("analysis/recreated_figures/figure7_label_edits/heterogeneous_vs_homogeneous_gini_bars_label_edits.png",),
        (
            "analysis/recreated_figures/figure7_from_script/heterogeneous_gini_run_metrics.csv",
            "analysis/recreated_figures/figure7_from_script/homogeneous_gini_run_metrics.csv",
        ),
    ),
    Task(
        "NeurIPS homogeneous-adversary Gini",
        ("scripts/paper_figures/plot_homogeneous_adversary_baseline_vs_all_gini.py",),
        ("overleaf/neurips/graphics/n_gt_2_report/homogeneous_adversary_baseline_only_vs_all_payoff_gini*",),
        ("experiments/results/n2_plus_multiagent_comparison_analysis_20260505/tables_multiagent/homogeneous_agents_fresh.csv",),
    ),
    Task(
        "NeurIPS role payoff",
        ("scripts/paper_figures/plot_role_payoff_with_within_run_variance_bars.py",),
        (
            "overleaf/neurips/graphics/n_gt_2_report/*_within_run_variance_bars*.png",
            "overleaf/neurips/graphics/n_gt_2_report/role_payoff_with_within_run_variance_bars_summary.csv",
        ),
        (
            "overleaf/neurips/graphics/n_gt_2_report/role_payoff_curves_by_strength_run_metrics.csv",
            "overleaf/neurips/graphics/n_gt_2_report/homogeneous_heterogeneous_bucketed_variance_mean_payoff_breakdown_run_metrics.csv",
        ),
    ),
    Task(
        "ICML homogeneous-adversary panels",
        ("scripts/paper_figures/plot_icml_homogeneous_adversary_main_panels.py",),
        ("overleaf/icml_aiwild_template/graphics/n_gt_2_report/homogeneous_adversary*.png",),
        (
            "overleaf/icml_aiwild_template/graphics/n_gt_2_report/homogeneous_adversary_baseline_only_vs_all_payoff_gini_summary.csv",
            "overleaf/icml_aiwild_template/graphics/n_gt_2_report/role_payoff_with_within_run_variance_bars_summary.csv",
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List the render tasks and exit.")
    parser.add_argument("--keep-outputs", action="store_true", help="Keep regenerated files instead of restoring the saved files.")
    return parser.parse_args()


def use_repository_python() -> None:
    if Path(sys.prefix).resolve() == (ROOT / ".venv").resolve():
        return
    if not VENV_PYTHON.is_file():
        raise SystemExit(f"Missing repository Python: {VENV_PYTHON}")
    print(f"Re-running with {VENV_PYTHON}", flush=True)
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])


def expand_outputs() -> set[Path]:
    paths: set[Path] = set()
    for task in TASKS:
        for pattern in task.outputs:
            paths.update(path for path in ROOT.glob(pattern) if path.is_file())
    return paths


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_images(before: Path, after: Path) -> tuple[str, str]:
    import numpy as np
    from PIL import Image

    with Image.open(before) as old_image, Image.open(after) as new_image:
        old = old_image.convert("RGBA")
        new = new_image.convert("RGBA")
    if old.size != new.size:
        return "REVIEW", f"canvas changed from {old.size[0]}x{old.size[1]} to {new.size[0]}x{new.size[1]}"

    old_array = np.asarray(old, dtype=np.int16)
    new_array = np.asarray(new, dtype=np.int16)
    changed = np.any(old_array != new_array, axis=2)
    changed_fraction = float(changed.mean())
    rms = float(np.sqrt(np.mean((old_array - new_array) ** 2)))
    if changed_fraction == 0.0:
        return "PASS", "pixels are identical; only file metadata or encoding changed"
    if changed_fraction <= 0.001 or rms <= 0.5:
        return "PASS", f"near-identical pixels ({changed_fraction:.3%} changed; RMS {rms:.3f}/255)"
    return "REVIEW", f"pixel rendering changed ({changed_fraction:.3%} changed; RMS {rms:.2f}/255)"


def compare_text(before: Path, after: Path) -> tuple[str, str]:
    def stable_lines(path: Path) -> list[str]:
        return [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if not line.startswith("- Generated: ")
        ]

    if stable_lines(before) == stable_lines(after):
        return "PASS", "only the generated timestamp changed"
    return "REVIEW", "text contents changed"


def main() -> int:
    args = parse_args()
    if args.list:
        for index, task in enumerate(TASKS, start=1):
            print(f"{index:2}. {task.name}: python {' '.join(task.args)}")
        return 0

    use_repository_python()
    if sys.version_info < (3, 10):
        raise SystemExit(f"Python 3.10 or newer is required; found {sys.version.split()[0]}")
    try:
        import matplotlib  # noqa: F401
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        import PIL  # noqa: F401
    except ImportError as error:
        raise SystemExit(f"The repository environment is incomplete: {error}") from error

    missing = sorted({item for task in TASKS for item in task.inputs if not (ROOT / item).exists()})
    if missing:
        print("Missing required inputs:")
        for item in missing:
            print(f"  - {item}")
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = LOG_ROOT / timestamp
    log_dir.mkdir(parents=True, exist_ok=False)
    before_paths = expand_outputs()
    failures: list[str] = []
    reviews: list[str] = []

    with tempfile.TemporaryDirectory(prefix="paper-figure-snapshot-") as temp_name:
        snapshot_root = Path(temp_name)
        for path in before_paths:
            target = snapshot_root / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)

        try:
            for index, task in enumerate(TASKS, start=1):
                command = [sys.executable, *task.args]
                log_path = log_dir / f"{index:02d}_{task.args[0].split('/')[-1]}.log"
                print(f"[{index:02d}/{len(TASKS)}] {task.name}", flush=True)
                started = time.monotonic()
                result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
                elapsed = time.monotonic() - started
                log_path.write_text(
                    f"$ {' '.join(command)}\n\nSTDOUT\n{result.stdout}\nSTDERR\n{result.stderr}",
                    encoding="utf-8",
                )
                if result.returncode:
                    failures.append(f"{task.name} exited {result.returncode}; see {log_path.relative_to(ROOT)}")
                    print(f"  FAIL after {elapsed:.1f}s; see {log_path.relative_to(ROOT)}")
                    break
                print(f"  completed in {elapsed:.1f}s")

            after_paths = expand_outputs()
            print("\nOutput comparison")
            for path in sorted(before_paths & after_paths):
                saved = snapshot_root / path.relative_to(ROOT)
                if sha256(saved) == sha256(path):
                    continue
                suffix = path.suffix.lower()
                if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
                    status, detail = compare_images(saved, path)
                elif suffix == ".pdf":
                    status, detail = "INFO", "PDF bytes changed; PDF metadata is not deterministic"
                elif suffix in {".md", ".txt"}:
                    status, detail = compare_text(saved, path)
                else:
                    status, detail = "REVIEW", "file contents changed"
                relative = path.relative_to(ROOT)
                print(f"  {status}: {relative}: {detail}")
                if status == "REVIEW":
                    reviews.append(f"{relative}: {detail}")
            for path in sorted(after_paths - before_paths):
                print(f"  NEW: {path.relative_to(ROOT)}")
        finally:
            if not args.keep_outputs:
                after_paths = expand_outputs()
                for path in after_paths - before_paths:
                    path.unlink()
                for path in before_paths:
                    saved = snapshot_root / path.relative_to(ROOT)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(saved, path)
                print("\nRestored all declared outputs to their pre-run bytes.")
            else:
                print("\nKept regenerated outputs because --keep-outputs was set.")

    validator = subprocess.run(
        [sys.executable, "scripts/validate_paper_figure_manifest.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    (log_dir / "manifest_validation.log").write_text(
        validator.stdout + validator.stderr,
        encoding="utf-8",
    )
    if validator.returncode:
        failures.append(f"saved manifest validation failed; see {(log_dir / 'manifest_validation.log').relative_to(ROOT)}")

    print(f"\nLogs: {log_dir.relative_to(ROOT)}")
    if reviews:
        print(f"Review required for {len(reviews)} regenerated output(s).")
    if failures:
        print("Failures:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("All render commands completed. No saved paper asset was changed.")
    return 2 if reviews else 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from strong_models_experiment.analysis.active_model_roster import (
    canonical_model_name,
    elo_for_model,
    short_model_name,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "experiments" / "results"

PROPOSAL_LABELS = {
    "proposal1_invasion": "Proposal 1: Invasion Into Nano Population",
    "proposal2_mixed_ecology": "Proposal 2: Matched Mixed Ecology",
}


def latest_results_root() -> Optional[Path]:
    candidates = sorted(RESULTS_ROOT.glob("game1_multiagent_smoke_*"))
    return candidates[-1].resolve() if candidates else None


def resolve_results_root(raw_value: Optional[str] = None) -> Path:
    if raw_value:
        path = Path(raw_value)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        return path
    latest = latest_results_root()
    if latest is None:
        raise FileNotFoundError("No game1_multiagent_smoke_* directories found.")
    return latest


def load_manifest(results_root: Path) -> Dict:
    manifest_path = results_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_config_rows(results_root: Path) -> List[Dict]:
    config_dir = results_root / "configs"
    rows: List[Dict] = []
    for config_path in sorted(config_dir.glob("config_*.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["config_path"] = str(config_path)
        rows.append(config)
    return rows


def resolve_output_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def result_file_for_output(output_dir: Path) -> Optional[Path]:
    candidates = [
        output_dir / "experiment_results.json",
        output_dir / "run_1_experiment_results.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def all_interactions_file_for_output(output_dir: Path) -> Optional[Path]:
    candidates = [
        output_dir / "all_interactions.json",
        output_dir / "run_1_all_interactions.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _result_status(output_dir: Path) -> str:
    if result_file_for_output(output_dir):
        return "SUCCESS"
    if output_dir.exists():
        return "IN_PROGRESS"
    return "NOT_STARTED"


def _log_status(results_root: Path, config_id: int) -> Optional[str]:
    log_path = results_root / "logs" / f"config_{int(config_id):04d}.log"
    if not log_path.exists():
        return None
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    failure_markers = [
        "Experiment failed:",
        "HTTP 400:",
        "Traceback (most recent call last):",
        "❌ Experiment failed:",
    ]
    if any(marker in text for marker in failure_markers):
        return "FAILED"
    success_markers = [
        "completed successfully",
        "✅ Strong models experiment completed successfully!",
    ]
    if any(marker in text for marker in success_markers):
        return "SUCCESS"
    return "IN_PROGRESS"


def _active_output_dirs(results_root: Path) -> set[str]:
    pattern = f"run_strong_models_experiment.py .*{results_root.name}"
    try:
        output = subprocess.check_output(
            ["pgrep", "-af", pattern],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return set()

    active: set[str] = set()
    for line in output.splitlines():
        marker = "--output-dir "
        idx = line.find(marker)
        if idx == -1:
            continue
        output_dir = line[idx + len(marker) :].strip()
        if not output_dir:
            continue
        active.add(str(Path(output_dir).resolve()))
    return active


def collect_condition_rows(results_root: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    active_output_dirs = _active_output_dirs(results_root)
    for config in load_config_rows(results_root):
        output_dir = resolve_output_dir(config["output_dir"])
        status = _result_status(output_dir)
        status_path = results_root / "status" / f"config_{int(config['config_id']):04d}.json"
        if status_path.exists():
            try:
                status_payload = json.loads(status_path.read_text(encoding="utf-8"))
            except OSError:
                status_payload = {}
            if status_payload.get("state") == "SKIPPED" and status != "SUCCESS":
                status = "SKIPPED"
        if status != "SUCCESS":
            log_status = _log_status(results_root, int(config["config_id"]))
            if status == "SKIPPED":
                pass
            elif log_status == "FAILED":
                status = "FAILED"
            elif (
                log_status is not None
                and str(output_dir.resolve()) not in active_output_dirs
                and log_status != "SUCCESS"
            ):
                status = "FAILED"
            elif log_status is not None:
                status = log_status
        row = {
            "config_id": config["config_id"],
            "proposal_family": config["proposal_family"],
            "condition_role": config["condition_role"],
            "pairing_id": config["pairing_id"],
            "focal_model": canonical_model_name(config["focal_model"]),
            "baseline_model": canonical_model_name(config["baseline_model"]),
            "n_agents": int(config["n_agents"]),
            "competition_level": float(config["competition_level"]),
            "replicate_id": config.get("replicate_id"),
            "field_id": config.get("field_id"),
            "random_seed": int(config["random_seed"]),
            "models": list(config["models"]),
            "ecology_models": list(config.get("ecology_models", [])),
            "output_dir": str(output_dir),
            "status": status,
        }

        result_path = result_file_for_output(output_dir)
        if result_path is not None:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            final_utilities = payload.get("final_utilities") or {}
            if not isinstance(final_utilities, dict):
                final_utilities = {}
            row["consensus_reached"] = bool(payload.get("consensus_reached", False))
            row["final_round"] = int(payload.get("final_round") or 0)
            row["focal_agent_utility"] = float(final_utilities.get("Agent_1", 0.0) or 0.0)
            row["mean_utility_all_agents"] = (
                float(sum(final_utilities.values()) / len(final_utilities))
                if final_utilities
                else 0.0
            )
        else:
            row["consensus_reached"] = None
            row["final_round"] = None
            row["focal_agent_utility"] = None
            row["mean_utility_all_agents"] = None

        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["proposal_family", "focal_model", "n_agents", "competition_level", "config_id"]).reset_index(drop=True)


def collect_paired_rows(results_root: Path) -> pd.DataFrame:
    conditions = collect_condition_rows(results_root)
    if conditions.empty:
        return conditions

    paired_rows: List[Dict] = []
    grouped = conditions.groupby("pairing_id", sort=False)
    for pairing_id, group in grouped:
        focal = group[group["condition_role"] == "focal"]
        control = group[group["condition_role"] == "control"]
        if len(focal) != 1 or len(control) != 1:
            continue
        focal_row = focal.iloc[0]
        control_row = control.iloc[0]
        if "SKIPPED" in {focal_row["status"], control_row["status"]}:
            continue
        status = "SUCCESS" if focal_row["status"] == "SUCCESS" and control_row["status"] == "SUCCESS" else "INCOMPLETE"
        paired_rows.append(
            {
                "pairing_id": pairing_id,
                "proposal_family": focal_row["proposal_family"],
                "focal_model": focal_row["focal_model"],
                "baseline_model": focal_row["baseline_model"],
                "n_agents": int(focal_row["n_agents"]),
                "competition_level": float(focal_row["competition_level"]),
                "replicate_id": focal_row["replicate_id"],
                "field_id": focal_row["field_id"],
                "random_seed": int(focal_row["random_seed"]),
                "status": status,
                "focal_output_dir": focal_row["output_dir"],
                "control_output_dir": control_row["output_dir"],
                "ecology_models": focal_row["ecology_models"],
                "focal_agent_utility": focal_row["focal_agent_utility"],
                "control_agent_utility": control_row["focal_agent_utility"],
                "utility_advantage_vs_nano": (
                    float(focal_row["focal_agent_utility"] - control_row["focal_agent_utility"])
                    if status == "SUCCESS"
                    else None
                ),
                "focal_consensus_reached": focal_row["consensus_reached"],
                "control_consensus_reached": control_row["consensus_reached"],
                "focal_final_round": focal_row["final_round"],
                "control_final_round": control_row["final_round"],
                "elo": elo_for_model(focal_row["focal_model"]),
                "short_name": short_model_name(focal_row["focal_model"]),
            }
        )

    frame = pd.DataFrame(paired_rows)
    if frame.empty:
        return frame
    return frame.sort_values(["proposal_family", "focal_model", "n_agents", "competition_level", "pairing_id"]).reset_index(drop=True)


def aggregate_paired_rows(paired_rows: pd.DataFrame) -> pd.DataFrame:
    if paired_rows.empty:
        return paired_rows
    successful = paired_rows[paired_rows["status"] == "SUCCESS"].copy()
    if successful.empty:
        return successful

    summaries: List[Dict] = []
    grouped = successful.groupby(["proposal_family", "focal_model", "n_agents", "competition_level"], sort=False)
    for (proposal_family, focal_model, n_agents, competition_level), group in grouped:
        deltas = group["utility_advantage_vs_nano"].tolist()
        focal_utils = group["focal_agent_utility"].tolist()
        control_utils = group["control_agent_utility"].tolist()
        summaries.append(
            {
                "proposal_family": proposal_family,
                "proposal_label": PROPOSAL_LABELS.get(proposal_family, proposal_family),
                "focal_model": focal_model,
                "short_name": short_model_name(focal_model),
                "elo": elo_for_model(focal_model),
                "n_agents": int(n_agents),
                "competition_level": float(competition_level),
                "num_pairs": int(len(group)),
                "mean_utility_advantage_vs_nano": mean(deltas),
                "std_utility_advantage_vs_nano": pstdev(deltas) if len(deltas) > 1 else 0.0,
                "mean_focal_utility": mean(focal_utils),
                "mean_control_utility": mean(control_utils),
                "focal_consensus_rate": float(group["focal_consensus_reached"].mean()),
                "control_consensus_rate": float(group["control_consensus_reached"].mean()),
                "mean_focal_final_round": float(group["focal_final_round"].mean()),
                "mean_control_final_round": float(group["control_final_round"].mean()),
            }
        )

    summary = pd.DataFrame(summaries)
    return summary.sort_values(["proposal_family", "n_agents", "competition_level", "elo"]).reset_index(drop=True)


def completion_summary(conditions: pd.DataFrame) -> Dict[str, int]:
    if conditions.empty:
        return {"total": 0, "successful": 0, "in_progress": 0, "not_started": 0, "skipped": 0}
    counts = conditions["status"].value_counts().to_dict()
    return {
        "total": int(len(conditions)),
        "successful": int(counts.get("SUCCESS", 0)),
        "in_progress": int(counts.get("IN_PROGRESS", 0)),
        "not_started": int(counts.get("NOT_STARTED", 0)),
        "skipped": int(counts.get("SKIPPED", 0)),
    }


def write_csv(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def write_analysis_outputs(results_root: Path) -> Dict[str, Path]:
    analysis_dir = results_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    conditions = collect_condition_rows(results_root)
    paired = collect_paired_rows(results_root)
    summary = aggregate_paired_rows(paired)

    outputs = {
        "conditions_csv": analysis_dir / "condition_rows.csv",
        "paired_csv": analysis_dir / "paired_rows.csv",
        "summary_csv": analysis_dir / "summary_by_model.csv",
    }
    write_csv(conditions, outputs["conditions_csv"])
    write_csv(paired, outputs["paired_csv"])
    write_csv(summary, outputs["summary_csv"])

    manifest = load_manifest(results_root)
    completion = completion_summary(conditions)
    completion_path = analysis_dir / "completion_summary.json"
    completion_path.write_text(
        json.dumps({"manifest": manifest, "completion": completion}, indent=2) + "\n",
        encoding="utf-8",
    )
    outputs["completion_json"] = completion_path

    if not summary.empty:
        outputs["proposal1_plot"] = analysis_dir / "proposal1_advantage_vs_elo.png"
        outputs["proposal2_plot"] = analysis_dir / "proposal2_advantage_vs_elo.png"
        plot_advantage_vs_elo(
            summary[summary["proposal_family"] == "proposal1_invasion"],
            outputs["proposal1_plot"],
            PROPOSAL_LABELS["proposal1_invasion"],
        )
        plot_advantage_vs_elo(
            summary[summary["proposal_family"] == "proposal2_mixed_ecology"],
            outputs["proposal2_plot"],
            PROPOSAL_LABELS["proposal2_mixed_ecology"],
        )
    return outputs


def _competition_palette() -> Dict[float, str]:
    return {
        0.0: "#0f766e",
        0.5: "#2563eb",
        0.9: "#d97706",
        1.0: "#b91c1c",
    }


def plot_advantage_vs_elo(summary: pd.DataFrame, output_path: Path, title: str) -> None:
    if summary.empty:
        return

    palette = _competition_palette()
    n_values = sorted(summary["n_agents"].unique().tolist())
    fig, axes = plt.subplots(1, len(n_values), figsize=(8 * len(n_values), 5), squeeze=False)

    for ax, n_agents in zip(axes[0], n_values):
        subset_n = summary[summary["n_agents"] == n_agents].sort_values("elo")
        for competition_level in sorted(subset_n["competition_level"].unique().tolist()):
            subset = subset_n[subset_n["competition_level"] == competition_level].sort_values("elo")
            ax.plot(
                subset["elo"],
                subset["mean_utility_advantage_vs_nano"],
                marker="o",
                linewidth=2,
                color=palette.get(float(competition_level), "#475569"),
                label=f"comp={competition_level:g}",
            )
            for _, row in subset.iterrows():
                ax.annotate(
                    row["short_name"],
                    (row["elo"], row["mean_utility_advantage_vs_nano"]),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=8,
                )

        ax.axhline(0.0, color="#94a3b8", linewidth=1, linestyle="--")
        ax.set_title(f"n = {n_agents}")
        ax.set_xlabel("Chatbot Arena Elo")
        ax.set_ylabel("Mean Utility Advantage vs Nano Control")
        ax.grid(alpha=0.25)
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def completion_table_rows(conditions: pd.DataFrame) -> List[Dict]:
    if conditions.empty:
        return []
    rows: List[Dict] = []
    grouped = conditions.groupby(["proposal_family", "n_agents", "competition_level"], sort=False)
    for (proposal_family, n_agents, competition_level), group in grouped:
        counts = group["status"].value_counts().to_dict()
        rows.append(
            {
                "proposal_family": proposal_family,
                "proposal_label": PROPOSAL_LABELS.get(proposal_family, proposal_family),
                "n_agents": int(n_agents),
                "competition_level": float(competition_level),
                "total_conditions": int(len(group)),
                "successful": int(counts.get("SUCCESS", 0)),
                "in_progress": int(counts.get("IN_PROGRESS", 0)),
                "not_started": int(counts.get("NOT_STARTED", 0)),
            }
        )
    return rows

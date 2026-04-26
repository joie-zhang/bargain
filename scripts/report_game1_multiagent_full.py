#!/usr/bin/env python3
"""Queue-aware status and performance reporting for Game 1 multi-agent full runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "experiments" / "results" / "game1_multiagent_full_latest"
RESULT_FILE_CANDIDATES = ("experiment_results.json", "run_1_experiment_results.json")

FAILURE_KIND_PATTERNS = {
    "quota_or_billing": (
        "insufficient_quota",
        "credit balance is too low",
        "daily quota exceeded",
        "quota exceeded",
        "rate limit exceeded for model",
    ),
    "timeout_or_transport": (
        "timed out",
        "timeout",
        "connection error",
        "remoteprotocolerror",
        "server disconnected without sending a response",
        "502",
        "503",
        "504",
    ),
    "context_or_request": (
        "maximum context length",
        "context window",
        "invalid_request_error",
        "could not parse the json body",
    ),
}


@dataclass
class ConfigRecord:
    config_id: int
    payload: Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build queue-aware status + n>2 performance summaries for game1_multiagent_full results."
    )
    parser.add_argument("--results-root", type=str, default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="analysis_extended",
        help="Subdirectory under results-root for generated outputs.",
    )
    return parser.parse_args()


def resolve_results_root(raw_value: str) -> Path:
    path = Path(raw_value)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path.resolve()


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_configs(results_root: Path) -> List[ConfigRecord]:
    configs: List[ConfigRecord] = []
    for path in sorted((results_root / "configs").glob("config_*.json")):
        payload = load_json(path)
        configs.append(ConfigRecord(config_id=int(payload["config_id"]), payload=payload))
    return configs


def status_path(results_root: Path, config_id: int) -> Path:
    return results_root / "status" / f"config_{config_id:04d}.json"


def log_path(results_root: Path, config_id: int) -> Path:
    return results_root / "logs" / f"config_{config_id:04d}.log"


def resolve_output_dir(output_dir_raw: str) -> Path:
    path = Path(output_dir_raw)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def find_result_file(output_dir: Path) -> Optional[Path]:
    for candidate in RESULT_FILE_CANDIDATES:
        path = output_dir / candidate
        if path.exists():
            return path
    return None


def classify_failure_kind(failure_reason: str, log_text: str) -> str:
    lowered = f"{failure_reason}\n{log_text}".lower()
    for label, patterns in FAILURE_KIND_PATTERNS.items():
        if any(pattern in lowered for pattern in patterns):
            return label
    if "traceback (most recent call last):" in lowered:
        return "python_traceback"
    return "other"


def read_status_payload(results_root: Path, config_id: int) -> Optional[Dict]:
    path = status_path(results_root, config_id)
    if not path.exists():
        return None
    return load_json(path)


def derive_state(result_exists: bool, status_payload: Optional[Dict]) -> str:
    if result_exists:
        return "SUCCESS"
    if status_payload is None:
        return "NOT_STARTED"
    state = str(status_payload.get("state", "")).upper()
    if state in {"RUNNING", "SUBMITTED", "SKIPPED", "FAILED", "SUCCESS"}:
        return state
    return "UNKNOWN"


def utility_from_result(result_payload: Dict, agent_id: str = "Agent_1") -> Tuple[Optional[float], Optional[float]]:
    utilities = result_payload.get("final_utilities", {})
    if not isinstance(utilities, dict) or not utilities:
        return None, None
    values = []
    for value in utilities.values():
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not values:
        return None, None
    focal = utilities.get(agent_id)
    try:
        focal_value = float(focal) if focal is not None else values[0]
    except (TypeError, ValueError):
        focal_value = values[0]
    return focal_value, mean(values)


def build_condition_rows(results_root: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for record in load_configs(results_root):
        config = record.payload
        output_dir = resolve_output_dir(config["output_dir"])
        result_file = find_result_file(output_dir)
        result_payload: Dict = {}
        if result_file is not None:
            try:
                result_payload = load_json(result_file)
            except Exception:
                result_payload = {}

        status_payload = read_status_payload(results_root, record.config_id)
        state = derive_state(result_file is not None, status_payload)

        attempts = int(status_payload.get("attempts", 0)) if status_payload else 0
        retryable = bool(status_payload.get("retryable", False)) if status_payload else False
        failure_reason = str(status_payload.get("failure_reason", "")) if status_payload else ""
        failure_kind = ""
        if state == "FAILED":
            raw_log = ""
            log_file = log_path(results_root, record.config_id)
            if log_file.exists():
                raw_log = log_file.read_text(encoding="utf-8", errors="ignore")[-25000:]
            failure_kind = classify_failure_kind(failure_reason, raw_log)

        consensus = result_payload.get("consensus_reached")
        if consensus is not None:
            consensus = bool(consensus)
        final_round = result_payload.get("final_round")
        try:
            final_round = float(final_round) if final_round is not None else None
        except (TypeError, ValueError):
            final_round = None

        focal_utility, mean_utility = utility_from_result(result_payload)

        rows.append(
            {
                "config_id": record.config_id,
                "pairing_id": config.get("pairing_id", ""),
                "proposal_family": config.get("proposal_family", ""),
                "condition_role": config.get("condition_role", ""),
                "focal_model": config.get("focal_model", ""),
                "baseline_model": config.get("baseline_model", ""),
                "n_agents": int(config.get("n_agents", 0)),
                "competition_level": float(config.get("competition_level", 0.0)),
                "replicate_id": config.get("replicate_id"),
                "field_id": config.get("field_id"),
                "state": state,
                "attempts": attempts,
                "retryable": retryable,
                "failure_reason": failure_reason,
                "failure_kind": failure_kind,
                "consensus_reached": consensus,
                "final_round": final_round,
                "focal_agent_utility": focal_utility,
                "mean_utility_all_agents": mean_utility,
                "output_dir": str(output_dir),
            }
        )
    return pd.DataFrame(rows)


def build_pair_rows(condition_df: pd.DataFrame) -> pd.DataFrame:
    focal = condition_df[condition_df["condition_role"] == "focal"].copy()
    control = condition_df[condition_df["condition_role"] == "control"].copy()

    use_cols = [
        "pairing_id",
        "state",
        "focal_agent_utility",
        "consensus_reached",
        "final_round",
        "output_dir",
    ]
    focal = focal[
        [
            "pairing_id",
            "focal_model",
            "baseline_model",
            "n_agents",
            "competition_level",
            "proposal_family",
            "replicate_id",
            "field_id",
        ]
        + use_cols[1:]
    ].rename(
        columns={
            "state": "focal_state",
            "focal_agent_utility": "focal_agent_utility",
            "consensus_reached": "focal_consensus_reached",
            "final_round": "focal_final_round",
            "output_dir": "focal_output_dir",
        }
    )
    control = control[use_cols].rename(
        columns={
            "state": "control_state",
            "focal_agent_utility": "control_agent_utility",
            "consensus_reached": "control_consensus_reached",
            "final_round": "control_final_round",
            "output_dir": "control_output_dir",
        }
    )

    merged = focal.merge(control, on="pairing_id", how="outer")
    merged["pair_state"] = merged.apply(
        lambda row: "SUCCESS"
        if row.get("focal_state") == "SUCCESS" and row.get("control_state") == "SUCCESS"
        else "INCOMPLETE",
        axis=1,
    )
    merged["utility_advantage_vs_nano"] = merged["focal_agent_utility"] - merged["control_agent_utility"]
    focal_consensus = merged["focal_consensus_reached"].astype("boolean").fillna(False)
    control_consensus = merged["control_consensus_reached"].astype("boolean").fillna(False)
    merged["both_reached_consensus"] = focal_consensus & control_consensus
    return merged


def write_overview_json(
    output_path: Path,
    condition_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    manifest: Dict,
) -> None:
    state_counts = condition_df["state"].value_counts().to_dict()
    focal_df = condition_df[condition_df["condition_role"] == "focal"]
    success_focal = focal_df[focal_df["state"] == "SUCCESS"]
    complete_pairs = pair_df[pair_df["pair_state"] == "SUCCESS"]
    failed_df = condition_df[condition_df["state"] == "FAILED"]

    summary = {
        "results_root": str(output_path.parent.parent),
        "generated_files_dir": str(output_path.parent),
        "total_configs": int(len(condition_df)),
        "state_counts": {k: int(v) for k, v in state_counts.items()},
        "focal_success_count": int(len(success_focal)),
        "complete_pair_count": int(len(complete_pairs)),
        "focal_consensus_rate_success_only": float(success_focal["consensus_reached"].mean())
        if not success_focal.empty
        else None,
        "focal_non_consensus_rate_success_only": float(1.0 - success_focal["consensus_reached"].mean())
        if not success_focal.empty
        else None,
        "mean_focal_utility_success_only": float(success_focal["focal_agent_utility"].mean())
        if not success_focal.empty
        else None,
        "mean_advantage_vs_nano_complete_pairs": float(complete_pairs["utility_advantage_vs_nano"].mean())
        if not complete_pairs.empty
        else None,
        "failed_breakdown_by_kind": {
            str(k): int(v) for k, v in failed_df["failure_kind"].value_counts().to_dict().items()
        },
        "manifest_max_attempts": manifest.get("max_attempts"),
        "manifest_n_values": manifest.get("n_values"),
        "manifest_competition_levels": manifest.get("competition_levels"),
    }
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def plot_state_counts_by_n(condition_df: pd.DataFrame, output_path: Path) -> None:
    counts = (
        condition_df.groupby(["n_agents", "state"], dropna=False)
        .size()
        .reset_index(name="count")
        .pivot(index="n_agents", columns="state", values="count")
        .fillna(0)
    )
    if counts.empty:
        return

    preferred_order = ["SUCCESS", "RUNNING", "SUBMITTED", "FAILED", "NOT_STARTED", "SKIPPED", "UNKNOWN"]
    ordered_cols = [col for col in preferred_order if col in counts.columns] + [
        col for col in counts.columns if col not in preferred_order
    ]
    counts = counts[ordered_cols]

    ax = counts.plot(kind="bar", stacked=True, figsize=(10, 6))
    ax.set_title("Config States by Agent Count (Queue-Aware)")
    ax.set_xlabel("n_agents")
    ax.set_ylabel("Config Count")
    ax.legend(title="State", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_metric_by_competition(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for n_agents, subset in sorted(df.groupby("n_agents"), key=lambda item: item[0]):
        ordered = subset.sort_values("competition_level")
        ax.plot(
            ordered["competition_level"],
            ordered[value_col],
            marker="o",
            linewidth=2,
            label=f"n={int(n_agents)}",
        )
    ax.set_title(title)
    ax.set_xlabel("Competition Level")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_tables(condition_df: pd.DataFrame, pair_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    condition_df.to_csv(output_dir / "condition_status_rows.csv", index=False)
    pair_df.to_csv(output_dir / "pair_status_rows.csv", index=False)

    focal_df = condition_df[condition_df["condition_role"] == "focal"].copy()
    model_completion = (
        focal_df.assign(count=1)
        .pivot_table(
            index="focal_model",
            columns="state",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    state_cols = [col for col in model_completion.columns if col != "focal_model"]
    model_completion["total_configs"] = model_completion[state_cols].sum(axis=1)
    model_completion["success_rate"] = model_completion.get("SUCCESS", 0) / model_completion["total_configs"]
    model_completion = model_completion.sort_values(
        ["success_rate", "SUCCESS", "total_configs"], ascending=[False, False, False]
    )
    model_completion.to_csv(output_dir / "model_completion_table.csv", index=False)

    success_focal = focal_df[focal_df["state"] == "SUCCESS"].copy()
    model_perf = (
        success_focal.groupby(["focal_model", "n_agents", "competition_level"], dropna=False)
        .agg(
            runs=("config_id", "count"),
            utility_observed_runs=("focal_agent_utility", "count"),
            consensus_rate=("consensus_reached", "mean"),
            mean_focal_utility=("focal_agent_utility", "mean"),
            mean_final_round=("final_round", "mean"),
        )
        .reset_index()
    )
    model_perf["non_consensus_rate"] = 1.0 - model_perf["consensus_rate"]
    model_perf.to_csv(output_dir / "model_performance_table.csv", index=False)

    consensus_by_n_comp = (
        success_focal.groupby(["n_agents", "competition_level"], dropna=False)
        .agg(
            runs=("config_id", "count"),
            consensus_rate=("consensus_reached", "mean"),
        )
        .reset_index()
    )
    consensus_by_n_comp["non_consensus_rate"] = 1.0 - consensus_by_n_comp["consensus_rate"]
    consensus_by_n_comp.to_csv(output_dir / "consensus_by_n_competition.csv", index=False)

    utility_by_n_comp = (
        success_focal.groupby(["n_agents", "competition_level"], dropna=False)
        .agg(
            runs=("config_id", "count"),
            utility_observed_runs=("focal_agent_utility", "count"),
            mean_focal_utility=("focal_agent_utility", "mean"),
            mean_all_agents_utility=("mean_utility_all_agents", "mean"),
        )
        .reset_index()
    )
    utility_by_n_comp.to_csv(output_dir / "utility_by_n_competition.csv", index=False)

    complete_pairs = pair_df[pair_df["pair_state"] == "SUCCESS"].copy()
    pair_summary = (
        complete_pairs.groupby(["n_agents", "competition_level"], dropna=False)
        .agg(
            pairs=("pairing_id", "count"),
            mean_advantage_vs_nano=("utility_advantage_vs_nano", "mean"),
            both_consensus_rate=("both_reached_consensus", "mean"),
        )
        .reset_index()
    )
    pair_summary.to_csv(output_dir / "pair_advantage_by_n_competition.csv", index=False)

    plot_state_counts_by_n(condition_df, output_dir / "state_counts_by_n_agents.png")
    plot_metric_by_competition(
        consensus_by_n_comp,
        value_col="consensus_rate",
        title="Consensus Rate vs Competition Level (Successful Runs, n>2)",
        ylabel="Consensus Rate",
        output_path=output_dir / "consensus_rate_by_competition_n.png",
    )
    plot_metric_by_competition(
        utility_by_n_comp,
        value_col="mean_focal_utility",
        title="Mean Focal Utility vs Competition Level (Successful Runs, n>2)",
        ylabel="Mean Focal Utility",
        output_path=output_dir / "mean_focal_utility_by_competition_n.png",
    )
    plot_metric_by_competition(
        pair_summary,
        value_col="mean_advantage_vs_nano",
        title="Mean Utility Advantage vs Nano Control (Complete Pairs, n>2)",
        ylabel="Mean Utility Advantage vs Nano",
        output_path=output_dir / "mean_advantage_vs_nano_by_competition_n.png",
    )


def main() -> int:
    args = parse_args()
    results_root = resolve_results_root(args.results_root)
    if not results_root.exists():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")
    manifest_path = results_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.json in: {results_root}")
    manifest = load_json(manifest_path)

    output_dir = results_root / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    condition_df = build_condition_rows(results_root)
    pair_df = build_pair_rows(condition_df)
    write_tables(condition_df, pair_df, output_dir)
    write_overview_json(output_dir / "overview.json", condition_df, pair_df, manifest)

    print(f"condition_rows_csv: {output_dir / 'condition_status_rows.csv'}")
    print(f"pair_rows_csv: {output_dir / 'pair_status_rows.csv'}")
    print(f"overview_json: {output_dir / 'overview.json'}")
    print(f"state_plot: {output_dir / 'state_counts_by_n_agents.png'}")
    print(f"consensus_plot: {output_dir / 'consensus_rate_by_competition_n.png'}")
    print(f"utility_plot: {output_dir / 'mean_focal_utility_by_competition_n.png'}")
    print(f"advantage_plot: {output_dir / 'mean_advantage_vs_nano_by_competition_n.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

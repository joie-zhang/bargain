#!/usr/bin/env python3
"""Estimate context-compaction ablation cost from retained Game 3 rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = (
    REPO / "experiments/analysis/reviewer_failure_audit_20260725/final_outcomes.csv"
)
DEFAULT_OUTPUT = REPO / "experiments/analysis/reviewer_failure_audit_20260725"

MULTI_BATCHES = {
    "multiagent_homogeneous",
    "multiagent_heterogeneous",
    "random_monoculture",
}

# Final adaptive design: one Game 3 cell (sigma=0.2, alpha=0.2), 10 pairs
# per low-N cell and at most 20 pairs per stress-N cell, with two treatment
# arms. The five-pair-per-N pilot is included in these cumulative totals.
PLANNED_RUNS_BY_N = {
    2: 10 * 2,
    4: 10 * 2,
    6: 20 * 2,
    8: 20 * 2,
    10: 20 * 2,
}

MODEL_PRICES = {
    "amazon_nova_micro": {"input_per_million": 0.035, "output_per_million": 0.14},
    "gpt_4o_mini": {"input_per_million": 0.15, "output_per_million": 0.60},
}


def event_tokens(event: dict[str, Any]) -> tuple[int, int]:
    usage = event.get("token_usage") or {}
    input_tokens = event.get("provider_input_tokens")
    if input_tokens is None:
        input_tokens = usage.get("provider_input_tokens", usage.get("input_tokens", 0))
    output_tokens = usage.get("output_tokens", 0)
    try:
        return int(input_tokens or 0), int(output_tokens or 0)
    except (TypeError, ValueError):
        return 0, 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-per-batch-n", type=int, default=10)
    parser.add_argument("--cap-per-n", type=int, default=25)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    final = pd.read_csv(args.audit)
    eligible = final[
        final["paper_batch"].isin(MULTI_BATCHES)
        & final["game"].eq("game3")
        & final["sigma"].eq(0.2)
        & final["alpha"].eq(0.2)
        & ~final["compaction_used"].astype(bool)
    ].copy()

    samples = []
    for n_agents, n_group in eligible.groupby("n_agents"):
        selected = (
            n_group.sort_values("config_key")
            .groupby("paper_batch", group_keys=False)
            .head(args.sample_per_batch_n)
            .head(args.cap_per_n)
        )
        for _, row in selected.iterrows():
            path = REPO / row["rollout_path"]
            interactions = json.loads(path.read_text(errors="replace"))
            input_tokens = 0
            output_tokens = 0
            calls = 0
            for event in interactions if isinstance(interactions, list) else []:
                if not isinstance(event, dict):
                    continue
                event_input, event_output = event_tokens(event)
                input_tokens += event_input
                output_tokens += event_output
                if event_input or event_output:
                    calls += 1
            samples.append(
                {
                    "config_key": row["config_key"],
                    "paper_batch": row["paper_batch"],
                    "n_agents": int(n_agents),
                    "final_round": row["final_round"],
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "model_calls": calls,
                }
            )
    sample_frame = pd.DataFrame(samples)
    sample_frame.to_csv(output / "ablation_cost_token_sample.csv", index=False)

    summary = (
        sample_frame.groupby("n_agents")
        .agg(
            sampled_runs=("config_key", "size"),
            median_input_tokens=("input_tokens", "median"),
            input_tokens_p25=("input_tokens", lambda values: values.quantile(0.25)),
            input_tokens_p75=("input_tokens", lambda values: values.quantile(0.75)),
            median_output_tokens=("output_tokens", "median"),
            median_model_calls=("model_calls", "median"),
            median_final_round=("final_round", "median"),
        )
        .reset_index()
    )
    summary["planned_ablation_runs"] = summary["n_agents"].map(PLANNED_RUNS_BY_N)
    for model_name, price in MODEL_PRICES.items():
        summary[f"{model_name}_median_cost_per_run_usd"] = (
            summary["median_input_tokens"] / 1_000_000 * price["input_per_million"]
            + summary["median_output_tokens"] / 1_000_000 * price["output_per_million"]
        )
        summary[f"{model_name}_planned_cell_cost_usd"] = (
            summary[f"{model_name}_median_cost_per_run_usd"]
            * summary["planned_ablation_runs"]
        )
    summary.to_csv(output / "ablation_cost_estimate.csv", index=False)

    totals = {
        "planned_runs": int(sum(PLANNED_RUNS_BY_N.values())),
        "sampled_historical_runs": len(sample_frame),
    }
    for model_name in MODEL_PRICES:
        estimate = float(summary[f"{model_name}_planned_cell_cost_usd"].sum())
        totals[f"{model_name}_median_based_estimate_usd"] = estimate
        totals[f"{model_name}_two_x_contingency_usd"] = 2 * estimate
    (output / "ablation_cost_estimate.json").write_text(
        json.dumps(totals, indent=2) + "\n"
    )
    print(summary.to_string(index=False))
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()

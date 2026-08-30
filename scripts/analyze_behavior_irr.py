#!/usr/bin/env python3
"""Compute agreement metrics from the human behavior-review export."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = PROJECT_ROOT / "analysis" / "behavior_annotation_irr_review_20260814"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_ROOT / "agreement_ready.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT / "agreement_metrics.json")
    return parser.parse_args()


def safe_divide(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def metric_block(rows: Iterable[dict[str, Any]], weighted: bool) -> dict[str, Any]:
    cells = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
    decisions = list(rows)
    for row in decisions:
        human = int(row["reviewer_binary"])
        machine = int(row["machine_binary"])
        weight = float(row["sampling_weight"]) if weighted else 1.0
        key = "tp" if machine and human else "fp" if machine else "fn" if human else "tn"
        cells[key] += weight
    tp, fp, fn, tn = (cells[key] for key in ("tp", "fp", "fn", "tn"))
    total = tp + fp + fn + tn
    observed = safe_divide(tp + tn, total)
    machine_positive = safe_divide(tp + fp, total)
    human_positive = safe_divide(tp + fn, total)
    expected = None
    kappa = None
    ac1 = None
    if machine_positive is not None and human_positive is not None and observed is not None:
        expected = machine_positive * human_positive + (1 - machine_positive) * (1 - human_positive)
        kappa = safe_divide(observed - expected, 1 - expected)
        mean_positive = (machine_positive + human_positive) / 2
        ac1_expected = 2 * mean_positive * (1 - mean_positive)
        ac1 = safe_divide(observed - ac1_expected, 1 - ac1_expected)
    return {
        "weighted": weighted,
        "binary_decision_count": len(decisions),
        "effective_weight_total": total,
        "confusion_machine_against_human": cells,
        "raw_agreement": observed,
        "cohen_kappa": kappa,
        "gwet_ac1": ac1,
        "machine_sensitivity": safe_divide(tp, tp + fn),
        "machine_specificity": safe_divide(tn, tn + fp),
        "machine_positive_predictive_value": safe_divide(tp, tp + fp),
        "machine_negative_predictive_value": safe_divide(tn, tn + fn),
        "human_positive_prevalence": human_positive,
        "machine_positive_prevalence": machine_positive,
        "prevalence_index": safe_divide(abs(tp - tn), total),
        "bias_index": safe_divide(abs(fp - fn), total),
    }


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["reviewer_binary"] = int(row["reviewer_binary"]) if row["reviewer_binary"] else None
        row["machine_binary"] = int(row["machine_binary"])
        row["sampling_weight"] = float(row["sampling_weight"])
    return rows


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    binary = [row for row in rows if row["reviewer_binary"] in {0, 1}]
    response_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        response_counts[row["reviewer_response"]] += 1

    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_reviewer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in binary:
        by_label[row["tag_code"]].append(row)
        by_reviewer[row["reviewer_id"]].append(row)

    report = {
        "input_path": str(args.input.resolve()),
        "decision_count": len(rows),
        "response_counts": dict(sorted(response_counts.items())),
        "primary_excludes_unsure_and_skip": True,
        "overall_unweighted": metric_block(binary, weighted=False),
        "overall_sampling_weighted": metric_block(binary, weighted=True),
        "per_label": {
            label: {
                "unweighted": metric_block(label_rows, weighted=False),
                "sampling_weighted": metric_block(label_rows, weighted=True),
            }
            for label, label_rows in sorted(by_label.items())
        },
        "per_reviewer": {
            reviewer: {
                "unweighted": metric_block(reviewer_rows, weighted=False),
                "sampling_weighted": metric_block(reviewer_rows, weighted=True),
            }
            for reviewer, reviewer_rows in sorted(by_reviewer.items())
        },
        "interpretation_limits": [
            "The machine-positive and machine-negative classes were sampled at different rates.",
            "Use sampling-weighted metrics for candidate-population estimates.",
            "Use rollout-clustered confidence intervals before making an inferential claim.",
            "Cohen's kappa can fall when prevalence is very low even if raw agreement is high.",
            "Report Gwet's AC1, prevalence, the confusion table, and unsure frequency with kappa.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

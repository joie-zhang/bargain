#!/usr/bin/env python3
"""Judge harness for validating qualitative extraction quality."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strong_models_experiment.analysis.qualitative_judge import (
    RUBRIC_VERSION,
    aggregate_judge_scores,
    build_judge_packet,
)


def _load_result(path: Path) -> Dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _iter_cofunding_results(results_dir: Path) -> List[Dict[str, Any]]:
    records = []
    for path in sorted(results_dir.rglob("*experiment_results.json")):
        data = _load_result(path)
        if not data:
            continue
        cfg = data.get("config", {})
        if cfg.get("game_type") != "co_funding":
            continue
        data["_file_path"] = str(path)
        records.append(data)
    return records


def _stratified_sample(
    records: List[Dict[str, Any]],
    per_stratum: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    strata: Dict[tuple, List[Dict[str, Any]]] = {}
    for rec in records:
        cfg = rec.get("config", {})
        key = (
            cfg.get("alpha"),
            cfg.get("sigma"),
            bool(rec.get("consensus_reached", False)),
        )
        strata.setdefault(key, []).append(rec)

    sampled = []
    for _, bucket in sorted(strata.items(), key=lambda kv: str(kv[0])):
        rng.shuffle(bucket)
        sampled.extend(bucket[:per_stratum])
    return sampled


def _cmd_build_packets(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise SystemExit(f"Results directory not found: {results_dir}")

    records = _iter_cofunding_results(results_dir)
    if not records:
        raise SystemExit("No co-funding experiment results found.")

    sampled = _stratified_sample(records, per_stratum=args.per_stratum, seed=args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    packets_written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in sampled:
            packet = build_judge_packet(rec, max_messages=args.max_messages)
            if not packet:
                continue
            packet["source_file"] = rec.get("_file_path")
            f.write(json.dumps(packet, ensure_ascii=False) + "\n")
            packets_written += 1

    print(f"Rubric version: {RUBRIC_VERSION}")
    print(f"Total co-funding records: {len(records)}")
    print(f"Sampled packets written: {packets_written}")
    print(f"Output: {out_path}")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _cmd_score(args: argparse.Namespace) -> None:
    responses_path = Path(args.responses)
    if not responses_path.exists():
        raise SystemExit(f"Judge responses file not found: {responses_path}")

    rows = _load_jsonl(responses_path)
    summary = aggregate_judge_scores(rows)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Scored packets: {summary['num_packets_scored']}")
    print(f"Overall quality mean: {summary['overall_quality_mean']}")
    print(f"Output: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-packets", help="Create stratified judge packets JSONL.")
    build.add_argument("--results-dir", type=str, default="experiments/results/cofunding_latest")
    build.add_argument("--output", type=str, default="analysis/qualitative_judge_packets.jsonl")
    build.add_argument("--per-stratum", type=int, default=3)
    build.add_argument("--max-messages", type=int, default=40)
    build.add_argument("--seed", type=int, default=42)
    build.set_defaults(func=_cmd_build_packets)

    score = sub.add_parser("score", help="Aggregate judge responses into quality metrics.")
    score.add_argument("--responses", type=str, required=True, help="Judge response JSONL path.")
    score.add_argument("--output", type=str, default="analysis/qualitative_judge_report.json")
    score.set_defaults(func=_cmd_score)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

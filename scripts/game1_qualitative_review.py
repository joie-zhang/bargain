#!/usr/bin/env python3
"""Game 1 qualitative review pipeline.

This script builds a broad-coverage qualitative review for the current
GPT-5-nano-vs-model Game 1 item-allocation batch.

Outputs:
- run-level deterministic feature table
- adversary summary table
- stratified LLM label packets
- cached Anthropic labels for selected runs
- figures and markdown report
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import contextlib
import json
import math
import os
import random
import re
import ssl
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - optional plotting dependency
    sns = None

try:
    import requests
except ImportError:  # pragma: no cover - runtime fallback
    requests = None

try:
    import httpx
except ImportError:  # pragma: no cover - runtime fallback
    httpx = None

try:
    import anthropic
except ImportError:  # pragma: no cover - handled at runtime
    anthropic = None

try:
    import certifi
except ImportError:  # pragma: no cover - runtime fallback
    certifi = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strong_models_experiment.analysis.active_model_roster import (  # noqa: E402
    LEGACY_EXCLUDED_MODELS,
    active_model_info_map,
    canonical_model_name,
    short_model_name,
)


DEFAULT_RESULTS_DIR = PROJECT_ROOT / "experiments" / "results" / "scaling_experiment_20260404_064451"
DEFAULT_ANALYSIS_DIR = PROJECT_ROOT / "analysis" / "game1_qualitative_review"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "visualization" / "figures" / "game1_qualitative_review"
BASELINE_MODEL = "gpt-5-nano"
LLM_MODEL = "claude-haiku-4-5-20251001"
STRATIFY_COMP_ORDER = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]

MODEL_INFO = active_model_info_map()

if sns is not None:
    sns.set_theme(style="whitegrid")
else:
    plt.style.use("default")

LABEL_SCHEMA = {
    "opening_style": [
        "parser_or_degenerate",
        "maximalist_anchor",
        "targeted_anchor",
        "balanced_tradeoff",
        "cooperative_exploration",
    ],
    "adaptation_style": [
        "rigid_repetition",
        "incremental_concession",
        "responsive_tradeoff",
        "oscillating_or_incoherent",
        "minimal_evidence",
    ],
    "failure_mode": [
        "none",
        "repetitive_deadlock",
        "top_item_conflict",
        "parser_failure",
        "incompatible_fairness_frame",
        "late_round_brinkmanship",
        "other",
    ],
    "resolution_driver": [
        "adversary_frame_accepted",
        "baseline_frame_accepted",
        "hybrid_compromise",
        "no_resolution",
    ],
}

LABEL_COLUMNS = [
    "opening_style",
    "adaptation_style",
    "failure_mode",
    "resolution_driver",
    "relative_stubbornness",
]

ELO_BANDS = [
    ("low (<1250)", float("-inf"), 1250.0),
    ("mid (1250-1399)", 1250.0, 1400.0),
    ("high (>=1400)", 1400.0, float("inf")),
]

PREFERRED_EXEMPLARS = {
    "opening_style": {
        "cooperative_exploration": "experiments_results_scaling_experiment_20260404_064451_gpt_5_nano_vs_deepseek_r1_weak_first_comp_0_0_turns_1_run_1_experiment_results_json",
        "balanced_tradeoff": "experiments_results_scaling_experiment_20260404_064451_gpt_5_nano_vs_gpt_5_nano_high_strong_first_comp_1_0_turns_2_run_2_experiment_results_json",
        "targeted_anchor": "experiments_results_scaling_experiment_20260404_064451_gpt_5_nano_vs_gpt_5_4_high_strong_first_comp_0_95_turns_2_run_2_experiment_results_json",
        "maximalist_anchor": "experiments_results_scaling_experiment_20260404_064451_gpt_5_nano_vs_claude_opus_4_5_20251101_strong_first_comp_1_0_turns_2_run_2_experiment_results_json",
        "parser_or_degenerate": "experiments_results_scaling_experiment_20260404_064451_gpt_5_nano_vs_llama_3_2_3b_instruct_strong_first_comp_0_9_turns_1_run_2_experiment_results_json",
    },
    "adaptation_style": {
        "responsive_tradeoff": "experiments_results_scaling_experiment_20260404_064451_gpt_5_nano_vs_gpt_5_4_high_strong_first_comp_0_95_turns_2_run_2_experiment_results_json",
        "minimal_evidence": "experiments_results_scaling_experiment_20260404_064451_gpt_5_nano_vs_deepseek_r1_weak_first_comp_0_0_turns_1_run_1_experiment_results_json",
        "rigid_repetition": "experiments_results_scaling_experiment_20260404_064451_gpt_5_nano_vs_deepseek_v3_weak_first_comp_0_9_turns_2_run_1_experiment_results_json",
        "incremental_concession": "experiments_results_scaling_experiment_20260404_064451_gpt_5_nano_vs_amazon_nova_micro_v1_0_weak_first_comp_0_25_turns_2_run_1_experiment_results_json",
        "oscillating_or_incoherent": "experiments_results_scaling_experiment_20260404_064451_gpt_5_nano_vs_llama_3_2_1b_instruct_strong_first_comp_1_0_turns_2_run_2_experiment_results_json",
    },
    "failure_mode": {
        "none": "experiments_results_scaling_experiment_20260404_064451_gpt_5_nano_vs_gpt_5_4_high_strong_first_comp_0_95_turns_2_run_2_experiment_results_json",
        "repetitive_deadlock": "experiments_results_scaling_experiment_20260404_064451_gpt_5_nano_vs_deepseek_v3_weak_first_comp_0_9_turns_2_run_1_experiment_results_json",
        "parser_failure": "experiments_results_scaling_experiment_20260404_064451_gpt_5_nano_vs_llama_3_2_3b_instruct_strong_first_comp_0_9_turns_1_run_2_experiment_results_json",
        "top_item_conflict": "experiments_results_scaling_experiment_20260404_064451_gpt_5_nano_vs_claude_opus_4_5_20251101_strong_first_comp_1_0_turns_2_run_2_experiment_results_json",
        "incompatible_fairness_frame": "experiments_results_scaling_experiment_20260404_064451_gpt_5_nano_vs_gemma_3_27b_it_weak_first_comp_0_0_turns_2_run_1_experiment_results_json",
        "late_round_brinkmanship": "experiments_results_scaling_experiment_20260404_064451_gpt_5_nano_vs_claude_opus_4_6_thinking_weak_first_comp_0_5_turns_2_run_1_experiment_results_json",
    },
    "resolution_driver": {
        "hybrid_compromise": "experiments_results_scaling_experiment_20260404_064451_gpt_5_nano_vs_deepseek_r1_weak_first_comp_0_0_turns_1_run_1_experiment_results_json",
        "baseline_frame_accepted": "experiments_results_scaling_experiment_20260404_064451_gpt_5_nano_vs_llama_3_1_8b_instruct_weak_first_comp_0_5_turns_1_run_1_experiment_results_json",
        "adversary_frame_accepted": "experiments_results_scaling_experiment_20260404_064451_gpt_5_nano_vs_gpt_5_4_high_strong_first_comp_0_95_turns_2_run_2_experiment_results_json",
        "no_resolution": "experiments_results_scaling_experiment_20260404_064451_gpt_5_nano_vs_gemma_3_27b_it_weak_first_comp_0_0_turns_2_run_1_experiment_results_json",
    },
    "relative_stubbornness": {
        "neither": "experiments_results_scaling_experiment_20260404_064451_gpt_5_nano_vs_deepseek_r1_weak_first_comp_0_0_turns_1_run_1_experiment_results_json",
        "baseline_more_stubborn": "experiments_results_scaling_experiment_20260404_064451_gpt_5_nano_vs_gemma_3_27b_it_weak_first_comp_0_0_turns_2_run_1_experiment_results_json",
        "adversary_more_stubborn": "experiments_results_scaling_experiment_20260404_064451_gpt_5_nano_vs_gpt_5_4_high_strong_first_comp_0_95_turns_2_run_2_experiment_results_json",
        "both_stubborn": "experiments_results_scaling_experiment_20260404_064451_gpt_5_nano_vs_deepseek_v3_weak_first_comp_0_9_turns_2_run_1_experiment_results_json",
    },
}

PAYOFF_MECHANISM_EXEMPLARS = [
    {
        "mechanism": "Complementary package recognition",
        "packet_id": "experiments_results_scaling_experiment_20260404_064451_gpt_5_nano_vs_deepseek_r1_weak_first_comp_0_0_turns_1_run_1_experiment_results_json",
        "quantitative_hook": "High-welfare complementarity case: both agents reach 100 utility in round 1.",
        "what_to_look_for": "Both sides explicitly state non-overlapping priorities and independently converge on the same Pareto-efficient package.",
    },
    {
        "mechanism": "Frame control without much extra welfare",
        "packet_id": "experiments_results_scaling_experiment_20260404_064451_gpt_5_nano_vs_gpt_5_4_high_strong_first_comp_0_95_turns_2_run_2_experiment_results_json",
        "quantitative_hook": "Resolved quickly, but the final package strongly favors the adversary (60.3 vs 42.3).",
        "what_to_look_for": "The adversary reuses the same package and the baseline eventually accepts that frame despite initial resistance.",
    },
    {
        "mechanism": "Low-payoff repetition / deadlock",
        "packet_id": "experiments_results_scaling_experiment_20260404_064451_gpt_5_nano_vs_deepseek_v3_weak_first_comp_0_9_turns_2_run_1_experiment_results_json",
        "quantitative_hook": "Rounds are consumed by repetition, ending in a low-value deal after 10 rounds (27.5 vs 17.4).",
        "what_to_look_for": "Near-identical proposals for most of the run, with little evidence that either side is updating on the other's stated priorities.",
    },
    {
        "mechanism": "Parser reliability as a payoff bottleneck",
        "packet_id": "experiments_results_scaling_experiment_20260404_064451_gpt_5_nano_vs_llama_3_2_3b_instruct_strong_first_comp_0_9_turns_1_run_2_experiment_results_json",
        "quantitative_hook": "No agreement; adversary parse-failure rate is 1.0.",
        "what_to_look_for": "Discussion text is not the whole story. The actual proposal channel collapses into proposer-gets-all fallbacks, so the run never reaches a real negotiation.",
    },
    {
        "mechanism": "Hard top-item conflict remains hard",
        "packet_id": "experiments_results_scaling_experiment_20260404_064451_gpt_5_nano_vs_claude_opus_4_5_20251101_strong_first_comp_1_0_turns_2_run_2_experiment_results_json",
        "quantitative_hook": "Even a strong adversary reaches only a low-joint-payoff deal under top-item conflict (15.3 vs 57.6).",
        "what_to_look_for": "The agents can bargain coherently, but the same high-value item remains the central collision point, limiting achievable welfare.",
    },
]

LABEL_PROMPT_SYSTEM = """You are a careful research assistant labeling negotiation transcripts.

Your task is to classify a two-agent item-allocation negotiation between a fixed baseline model and an adversary model.
Use only the supplied packet. Do not invent evidence. Prefer conservative labels.

Return strict JSON matching this schema:
{
  "opening_style": "parser_or_degenerate|maximalist_anchor|targeted_anchor|balanced_tradeoff|cooperative_exploration",
  "adaptation_style": "rigid_repetition|incremental_concession|responsive_tradeoff|oscillating_or_incoherent|minimal_evidence",
  "failure_mode": "none|repetitive_deadlock|top_item_conflict|parser_failure|incompatible_fairness_frame|late_round_brinkmanship|other",
  "resolution_driver": "adversary_frame_accepted|baseline_frame_accepted|hybrid_compromise|no_resolution",
  "relative_stubbornness": "adversary_more_stubborn|baseline_more_stubborn|both_stubborn|neither",
  "stubbornness_scores": {
    "baseline": 1,
    "adversary": 1
  },
  "adversary_effective_against_baseline": true,
  "surprising_feature": "short string",
  "evidence": ["short quote or paraphrase", "short quote or paraphrase"]
}

Scoring guidance:
- 1 means very flexible / cooperative.
- 5 means extremely rigid, repetitive, or unwilling to concede.
- `parser_or_degenerate` is only a valid value for `opening_style`.
- If the negotiation is malformed or parser-driven and you are choosing `adaptation_style`, use `minimal_evidence` when there is too little real adaptation to classify, or `rigid_repetition` when the same degenerate fallback is repeated across rounds.
- Use `minimal_evidence` only when the packet is too thin to classify adaptation style.
"""


@dataclass
class ProposalRecord:
    round_num: int
    proposer: str
    allocation: Dict[str, List[int]]
    signature: str
    reasoning: str
    self_raw_utility: float
    other_raw_utility: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Game 1 qualitative review pipeline")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--label-model", default=LLM_MODEL)
    parser.add_argument("--per-stratum", type=int, default=3,
                        help="Number of representative runs to label per (tier, competition) stratum.")
    parser.add_argument("--extreme-count", type=int, default=18,
                        help="Additional extreme-case runs to label.")
    parser.add_argument("--label-limit", type=int, default=None,
                        help="Optional hard cap on number of packets to label.")
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--label-workers", type=int, default=6)
    parser.add_argument("--skip-labeling", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ensure_dirs(*dirs: Path) -> None:
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Unsupported JSON type: {type(value)!r}")


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def canonical_tier(model_name: str) -> str:
    return str(MODEL_INFO.get(canonical_model_name(model_name), {}).get("tier", "Unknown"))


def model_elo(model_name: str) -> float:
    value = MODEL_INFO.get(canonical_model_name(model_name), {}).get("elo")
    return float(value) if value is not None else float("nan")


def _adversary_from_pair_dir(path: Path) -> Optional[str]:
    for part in path.parts:
        if "_vs_" in part:
            left, right = part.split("_vs_", 1)
            if canonical_model_name(left) == BASELINE_MODEL:
                return canonical_model_name(right)
            return canonical_model_name(left)
    return None


def load_results(results_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen_dirs = set()
    for result_file in sorted(results_dir.rglob("experiment_results.json")):
        parent_key = str(result_file.parent)
        if parent_key in seen_dirs:
            continue
        seen_dirs.add(parent_key)
        try:
            data = json.loads(result_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        config = data.get("config", {})
        if config.get("game_type") != "item_allocation":
            continue
        if "gpt-5-nano_vs_" not in str(result_file):
            continue
        adversary_from_dir = _adversary_from_pair_dir(result_file)
        if adversary_from_dir in LEGACY_EXCLUDED_MODELS:
            continue
        data["_file_path"] = str(result_file)
        records.append(data)
    return records


def parse_path_metadata(result_path: str) -> Dict[str, Any]:
    path = Path(result_path)
    meta: Dict[str, Any] = {"pair_dir": None, "model_order": None, "competition_level": None, "discussion_turns": None}
    for part in path.parts:
        if "_vs_" in part:
            meta["pair_dir"] = part
        elif part in {"weak_first", "strong_first"}:
            meta["model_order"] = part
        elif part.startswith("comp_"):
            meta["competition_level"] = float(part.replace("comp_", ""))
        elif part.startswith("turns_"):
            meta["discussion_turns"] = int(part.replace("turns_", ""))
    return meta


def identify_agents(result: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, str]:
    # The pair directory is the ground-truth adversary model: the experiment
    # runner occasionally strips the `-thinking` or `-thinking-32k` suffix in
    # `agent_performance[*].model`, which would silently merge thinking variants
    # with their non-thinking counterparts. The directory name never does that.
    pair_dir = meta.get("pair_dir") or ""
    model_a, model_b = pair_dir.split("_vs_")
    model_a = canonical_model_name(model_a)
    model_b = canonical_model_name(model_b)
    inferred_adversary_model = model_b if model_a == BASELINE_MODEL else model_a

    perf = result.get("agent_performance", {}) or {}
    baseline_agent = None
    adversary_agent = None
    baseline_model = BASELINE_MODEL

    for agent_id, info in perf.items():
        model_name = canonical_model_name(str((info or {}).get("model", "")))
        if model_name == BASELINE_MODEL:
            baseline_agent = agent_id
            baseline_model = model_name
        else:
            adversary_agent = agent_id

    if baseline_agent and adversary_agent:
        return {
            "baseline_agent": baseline_agent,
            "adversary_agent": adversary_agent,
            "baseline_model": baseline_model,
            "adversary_model": inferred_adversary_model,
        }

    order = meta.get("model_order")
    if order == "weak_first":
        baseline_agent, adversary_agent = "Agent_1", "Agent_2"
    else:
        baseline_agent, adversary_agent = "Agent_2", "Agent_1"
    return {
        "baseline_agent": baseline_agent,
        "adversary_agent": adversary_agent,
        "baseline_model": BASELINE_MODEL,
        "adversary_model": inferred_adversary_model,
    }


def allocation_signature(allocation: Dict[str, Sequence[int]]) -> str:
    parts = []
    for agent_id in sorted(allocation):
        items = ",".join(str(i) for i in sorted(int(x) for x in allocation.get(agent_id, [])))
        parts.append(f"{agent_id}:{items}")
    return "|".join(parts)


def utility_for_allocation(agent_id: str, allocation: Dict[str, Sequence[int]], prefs: Dict[str, Sequence[float]]) -> float:
    agent_prefs = prefs.get(agent_id, [])
    total = 0.0
    for item_idx in allocation.get(agent_id, []):
        if isinstance(item_idx, int) and 0 <= item_idx < len(agent_prefs):
            total += float(agent_prefs[item_idx])
    return total


def top_item_overlap(prefs: Dict[str, Sequence[float]], baseline_agent: str, adversary_agent: str) -> Tuple[bool, int]:
    baseline_vals = list(map(float, prefs.get(baseline_agent, [])))
    adversary_vals = list(map(float, prefs.get(adversary_agent, [])))
    if not baseline_vals or not adversary_vals:
        return False, 0
    baseline_top = int(np.argmax(baseline_vals))
    adversary_top = int(np.argmax(adversary_vals))
    baseline_top2 = set(np.argsort(baseline_vals)[-2:])
    adversary_top2 = set(np.argsort(adversary_vals)[-2:])
    return baseline_top == adversary_top, len(baseline_top2 & adversary_top2)


def extract_proposals(result: Dict[str, Any], focal_agent: str, other_agent: str) -> List[ProposalRecord]:
    records: List[ProposalRecord] = []
    prefs = result.get("agent_preferences", {}) or {}
    for log in result.get("conversation_logs", []):
        if log.get("phase") != "proposal":
            continue
        if log.get("from") != focal_agent:
            continue
        payload = log.get("proposal", {}) or {}
        allocation = payload.get("allocation", {}) or {}
        if not allocation:
            continue
        records.append(
            ProposalRecord(
                round_num=int(log.get("round", 0)),
                proposer=focal_agent,
                allocation=allocation,
                signature=allocation_signature(allocation),
                reasoning=str(payload.get("reasoning", "")),
                self_raw_utility=utility_for_allocation(focal_agent, allocation, prefs),
                other_raw_utility=utility_for_allocation(other_agent, allocation, prefs),
            )
        )
    return records


def summarize_proposal_path(records: List[ProposalRecord], max_rounds: int) -> Dict[str, Any]:
    if not records:
        return {
            "proposal_count": 0,
            "unique_proposals": 0,
            "repeat_rate": float("nan"),
            "all_items_rate": float("nan"),
            "parse_failure_rate": float("nan"),
            "opening_self_raw": float("nan"),
            "opening_other_raw": float("nan"),
            "opening_self_share": float("nan"),
            "first_concession_round": float("nan"),
            "late_concession_score": float("nan"),
            "total_concession": float("nan"),
            "largest_single_concession": float("nan"),
            "final_self_raw": float("nan"),
            "final_other_raw": float("nan"),
        }

    first = records[0]
    repeats = 0
    all_items = 0
    parse_failures = 0
    first_concession_round = None
    largest_single_concession = 0.0
    total_concession = max(first.self_raw_utility - min(r.self_raw_utility for r in records), 0.0)
    prev = first
    for current in records:
        allocation_values = list(current.allocation.values())
        if allocation_values:
            largest_bundle = max(len(bundle) for bundle in allocation_values if isinstance(bundle, list))
            if len(current.allocation.get(current.proposer, [])) == largest_bundle == sum(len(bundle) for bundle in allocation_values):
                all_items += 1
        if current.reasoning.lower().startswith("failed to parse response"):
            parse_failures += 1
        if current is not first and current.signature == prev.signature:
            repeats += 1
        if current is not first:
            concession = max(prev.self_raw_utility - current.self_raw_utility, 0.0)
            largest_single_concession = max(largest_single_concession, concession)
            if first_concession_round is None and current.self_raw_utility < first.self_raw_utility - 1e-9:
                first_concession_round = current.round_num
        prev = current

    opening_total = first.self_raw_utility + first.other_raw_utility
    if first_concession_round is None:
        late_concession_score = 1.0 if len(records) > 1 else 0.5
    else:
        late_concession_score = min((first_concession_round - 1) / max(max_rounds - 1, 1), 1.0)

    return {
        "proposal_count": len(records),
        "unique_proposals": len({r.signature for r in records}),
        "repeat_rate": repeats / max(len(records) - 1, 1),
        "all_items_rate": all_items / len(records),
        "parse_failure_rate": parse_failures / len(records),
        "opening_self_raw": first.self_raw_utility,
        "opening_other_raw": first.other_raw_utility,
        "opening_self_share": first.self_raw_utility / opening_total if opening_total > 0 else float("nan"),
        "first_concession_round": first_concession_round if first_concession_round is not None else float("nan"),
        "late_concession_score": late_concession_score,
        "total_concession": total_concession,
        "largest_single_concession": largest_single_concession,
        "final_self_raw": records[-1].self_raw_utility,
        "final_other_raw": records[-1].other_raw_utility,
    }


def stubbornness_proxy(summary: Dict[str, Any]) -> float:
    values = [
        summary.get("repeat_rate"),
        summary.get("all_items_rate"),
        summary.get("late_concession_score"),
        summary.get("parse_failure_rate"),
    ]
    clean = [float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    return float(sum(clean) / len(clean)) if clean else float("nan")


def find_accepted_proposer(result: Dict[str, Any]) -> str:
    if not result.get("consensus_reached"):
        return "none"
    final_allocation = result.get("final_allocation", {}) or {}
    target = allocation_signature(final_allocation)
    for log in result.get("conversation_logs", []):
        if log.get("phase") != "proposal_enumeration":
            continue
        if int(log.get("round", 0)) != int(result.get("final_round", 0)):
            continue
        for entry in log.get("enumerated_proposals", []) or []:
            allocation = (entry or {}).get("allocation", {}) or {}
            if allocation_signature(allocation) == target:
                return str((entry or {}).get("proposer", "unknown"))
    return "unknown"


def build_discussion_excerpts(result: Dict[str, Any], focus_rounds: Iterable[int], max_chars: int = 240) -> List[Dict[str, Any]]:
    excerpts: List[Dict[str, Any]] = []
    focus_set = set(int(r) for r in focus_rounds if isinstance(r, (int, float)) and not math.isnan(r))
    for log in result.get("conversation_logs", []):
        if log.get("phase") != "discussion":
            continue
        round_num = int(log.get("round", 0))
        if round_num not in focus_set and round_num != 1 and round_num != int(result.get("final_round", 0)):
            continue
        content = str(log.get("content", "")).strip().replace("\n", " ")
        excerpts.append({
            "round": round_num,
            "speaker": str(log.get("from", "")),
            "content": content[:max_chars],
        })
    # keep stable order but cap total size
    return excerpts[:12]


def packet_text(packet: Dict[str, Any]) -> str:
    parts = [
        f"Packet ID: {packet['packet_id']}",
        "Metadata:",
        json.dumps(packet["metadata"], indent=2),
        "Preferences:",
        json.dumps(packet["preferences"], indent=2),
        "Deterministic Features:",
        json.dumps(packet["feature_snapshot"], indent=2),
        "Round Summary:",
        json.dumps(packet["round_summary"], indent=2),
        "Discussion Excerpts:",
        json.dumps(packet["discussion_excerpts"], indent=2),
    ]
    return "\n".join(parts)


def extract_round_summary(result: Dict[str, Any], baseline_agent: str, adversary_agent: str) -> List[Dict[str, Any]]:
    prefs = result.get("agent_preferences", {}) or {}
    by_round: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for log in result.get("conversation_logs", []):
        if log.get("phase") != "proposal":
            continue
        round_num = int(log.get("round", 0))
        allocation = ((log.get("proposal") or {}).get("allocation") or {})
        proposer = str(log.get("from", ""))
        if not allocation or proposer not in {baseline_agent, adversary_agent}:
            continue
        other = adversary_agent if proposer == baseline_agent else baseline_agent
        by_round.setdefault(round_num, {})[proposer] = {
            "allocation": allocation,
            "self_raw_utility": utility_for_allocation(proposer, allocation, prefs),
            "other_raw_utility": utility_for_allocation(other, allocation, prefs),
            "reasoning": str(((log.get("proposal") or {}).get("reasoning") or ""))[:200],
        }
    summary: List[Dict[str, Any]] = []
    for round_num in sorted(by_round):
        round_entry = {"round": round_num}
        for proposer, prefix in ((baseline_agent, "baseline"), (adversary_agent, "adversary")):
            proposal_info = by_round[round_num].get(proposer)
            if proposal_info:
                round_entry[f"{prefix}_proposal"] = proposal_info
        summary.append(round_entry)
    return summary


def build_label_packet(row: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    baseline_agent = row["baseline_agent"]
    adversary_agent = row["adversary_agent"]
    prefs = result.get("agent_preferences", {}) or {}
    item_names = [item.get("name", f"Item_{idx}") for idx, item in enumerate((result.get("config") or {}).get("items", []))]
    preferences = []
    baseline_vals = list(prefs.get(baseline_agent, []))
    adversary_vals = list(prefs.get(adversary_agent, []))
    for idx, item_name in enumerate(item_names):
        preferences.append({
            "item": item_name,
            "baseline_value": float(baseline_vals[idx]) if idx < len(baseline_vals) else float("nan"),
            "adversary_value": float(adversary_vals[idx]) if idx < len(adversary_vals) else float("nan"),
        })
    focus_rounds = [
        row.get("baseline_first_concession_round"),
        row.get("adversary_first_concession_round"),
        row.get("final_round"),
    ]
    return {
        "packet_id": row["packet_id"],
        "metadata": {
            "adversary_model": row["adversary_model"],
            "adversary_tier": row["adversary_tier"],
            "adversary_elo": row["adversary_elo"],
            "competition_level": row["competition_level"],
            "discussion_turns": row["discussion_turns"],
            "model_order": row["model_order"],
            "consensus_reached": row["consensus_reached"],
            "final_round": row["final_round"],
            "accepted_proposer_role": row["accepted_proposer_role"],
        },
        "preferences": preferences,
        "feature_snapshot": {
            "top_item_conflict": row["top_item_conflict"],
            "top2_overlap_count": row["top2_overlap_count"],
            "baseline_stubbornness_proxy": row["baseline_stubbornness_proxy"],
            "adversary_stubbornness_proxy": row["adversary_stubbornness_proxy"],
            "baseline_parse_failure_rate": row["baseline_parse_failure_rate"],
            "adversary_parse_failure_rate": row["adversary_parse_failure_rate"],
            "baseline_repeat_rate": row["baseline_repeat_rate"],
            "adversary_repeat_rate": row["adversary_repeat_rate"],
        },
        "round_summary": extract_round_summary(result, baseline_agent, adversary_agent),
        "discussion_excerpts": build_discussion_excerpts(result, focus_rounds),
    }


def normalize_role(proposer: str, baseline_agent: str, adversary_agent: str) -> str:
    if proposer == baseline_agent:
        return "baseline"
    if proposer == adversary_agent:
        return "adversary"
    return proposer


def extract_run_features(result: Dict[str, Any]) -> Dict[str, Any]:
    meta = parse_path_metadata(result["_file_path"])
    agents = identify_agents(result, meta)
    baseline_agent = agents["baseline_agent"]
    adversary_agent = agents["adversary_agent"]
    prefs = result.get("agent_preferences", {}) or {}
    config = result.get("config", {}) or {}
    top_conflict, top2_overlap = top_item_overlap(prefs, baseline_agent, adversary_agent)

    baseline_summary = summarize_proposal_path(
        extract_proposals(result, baseline_agent, adversary_agent),
        int(config.get("t_rounds", 10)),
    )
    adversary_summary = summarize_proposal_path(
        extract_proposals(result, adversary_agent, baseline_agent),
        int(config.get("t_rounds", 10)),
    )

    baseline_model = canonical_model_name(agents["baseline_model"])
    adversary_model = canonical_model_name(agents["adversary_model"])
    accepted_proposer = find_accepted_proposer(result)
    accepted_role = normalize_role(accepted_proposer, baseline_agent, adversary_agent)

    final_utilities = result.get("final_utilities", {}) or {}
    packet_id = re.sub(r"[^a-zA-Z0-9_]+", "_", str(result["_file_path"]).replace(str(PROJECT_ROOT), "")).strip("_")

    row: Dict[str, Any] = {
        "packet_id": packet_id,
        "file_path": result["_file_path"],
        "baseline_agent": baseline_agent,
        "adversary_agent": adversary_agent,
        "baseline_model": baseline_model,
        "adversary_model": adversary_model,
        "adversary_short": short_model_name(adversary_model),
        "adversary_tier": canonical_tier(adversary_model),
        "adversary_elo": model_elo(adversary_model),
        "model_order": meta["model_order"],
        "competition_level": meta["competition_level"],
        "discussion_turns": meta["discussion_turns"],
        "consensus_reached": bool(result.get("consensus_reached")),
        "final_round": int(result.get("final_round", 0) or 0),
        "baseline_utility": float(final_utilities.get(baseline_agent, float("nan"))),
        "adversary_utility": float(final_utilities.get(adversary_agent, float("nan"))),
        "utility_gap": float(final_utilities.get(adversary_agent, float("nan"))) - float(final_utilities.get(baseline_agent, float("nan"))),
        "top_item_conflict": bool(top_conflict),
        "top2_overlap_count": int(top2_overlap),
        "accepted_proposer": accepted_proposer,
        "accepted_proposer_role": accepted_role,
    }

    for prefix, summary in (("baseline", baseline_summary), ("adversary", adversary_summary)):
        for key, value in summary.items():
            row[f"{prefix}_{key}"] = value
        row[f"{prefix}_stubbornness_proxy"] = stubbornness_proxy(summary)

    return row


def build_feature_table(results: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    rows = []
    result_index = {}
    for result in results:
        row = extract_run_features(result)
        rows.append(row)
        result_index[row["packet_id"]] = result
    return pd.DataFrame(rows), result_index


def select_runs_for_labeling(df: pd.DataFrame, per_stratum: int, extreme_count: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    chosen_ids = []
    grouped = df.groupby(["adversary_tier", "competition_level"], dropna=False)
    for _, group in grouped:
        if group.empty:
            continue
        sample_n = min(per_stratum, len(group))
        sample_idx = list(group.index)
        rng.shuffle(sample_idx)
        chosen_ids.extend(sample_idx[:sample_n])

    extreme_frames = []
    for column, ascending in (
        ("adversary_stubbornness_proxy", False),
        ("utility_gap", False),
        ("utility_gap", True),
        ("final_round", False),
    ):
        extreme_frames.append(df.sort_values(column, ascending=ascending).head(extreme_count))
    if "consensus_reached" in df.columns:
        extreme_frames.append(df[df["consensus_reached"] == False].sort_values("adversary_stubbornness_proxy", ascending=False).head(extreme_count))

    if extreme_frames:
        extreme_df = pd.concat(extreme_frames, ignore_index=False)
        chosen_ids.extend(list(extreme_df.index))

    chosen = df.loc[sorted(set(chosen_ids))].copy()
    return chosen.sort_values(["adversary_tier", "competition_level", "adversary_elo", "model_order"])


def normalize_label_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload)

    # Claude occasionally applies the opening-style token to adaptation when the
    # run is dominated by parser fallback or degenerate proposals. That should be
    # treated conservatively as missing adaptation evidence rather than a hard
    # pipeline failure.
    if normalized.get("adaptation_style") == "parser_or_degenerate":
        normalized["adaptation_style"] = "minimal_evidence"

    return normalized


def validate_label(payload: Dict[str, Any]) -> Dict[str, Any]:
    for key, allowed in LABEL_SCHEMA.items():
        value = payload.get(key)
        if value not in allowed:
            raise ValueError(f"Invalid label for {key}: {value!r}")
    stubborn = payload.get("stubbornness_scores", {})
    for role in ("baseline", "adversary"):
        score = int(stubborn.get(role))
        if score < 1 or score > 5:
            raise ValueError(f"Invalid stubbornness score for {role}: {score}")
    if not isinstance(payload.get("adversary_effective_against_baseline"), bool):
        raise ValueError("adversary_effective_against_baseline must be bool")
    evidence = payload.get("evidence", [])
    if not isinstance(evidence, list):
        raise ValueError("evidence must be a list")
    return payload


def record_label_failure(
    analysis_dir: Path,
    packet_id: str,
    attempt: int,
    error: Exception,
    raw_text: Optional[str],
    parsed_payload: Optional[Dict[str, Any]],
) -> None:
    failure_path = analysis_dir / "label_failures.jsonl"
    row = {
        "packet_id": packet_id,
        "attempt": attempt,
        "error_type": type(error).__name__,
        "error": str(error),
        "raw_text": raw_text,
        "parsed_payload": parsed_payload,
    }
    with failure_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=_json_default) + "\n")


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("No JSON object found in model response")
    return json.loads(text[start:end + 1])


def build_shared_anthropic_client(api_key: str) -> Any:
    if anthropic is None:
        return None

    if httpx is None:
        return anthropic.Anthropic(api_key=api_key)

    http_client_kwargs: Dict[str, Any] = {
        "timeout": 300.0,
        "follow_redirects": True,
    }
    if certifi is not None:
        # Princeton's Python 3.14 runtime intermittently fails while creating TLS
        # contexts from the environment CA bundle. Pinning certifi and disabling
        # trust_env avoids that path and is stable across concurrent requests.
        http_client_kwargs["verify"] = certifi.where()
        http_client_kwargs["trust_env"] = False

    http_client = httpx.Client(**http_client_kwargs)
    try:
        return anthropic.Anthropic(api_key=api_key, http_client=http_client)
    except Exception:
        with contextlib.suppress(Exception):
            http_client.close()
        raise


def should_prefer_requests_backend() -> bool:
    return requests is not None and sys.version_info >= (3, 14)


def describe_labeling_exception(exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, ssl.SSLError):
        message += (
            " | TLS initialization failed. On the uv Python 3.14 runtime this can be"
            " triggered by concurrent SSL context creation."
        )
    return message


def call_anthropic_messages(
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    client: Any = None,
) -> str:
    use_requests_backend = should_prefer_requests_backend()

    if client is not None:
        response = client.messages.create(
            model=model,
            max_tokens=700,
            temperature=temperature,
            system=LABEL_PROMPT_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )

    if anthropic is not None and not use_requests_backend:
        local_client = build_shared_anthropic_client(api_key)
        try:
            response = local_client.messages.create(
                model=model,
                max_tokens=700,
                temperature=temperature,
                system=LABEL_PROMPT_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(
                block.text for block in response.content if getattr(block, "type", "") == "text"
            )
        finally:
            with contextlib.suppress(Exception):
                local_client.close()

    if requests is None:
        raise RuntimeError("Neither anthropic SDK nor requests is available")

    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 700,
            "temperature": temperature,
            "system": LABEL_PROMPT_SYSTEM,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=300,
    )
    response.raise_for_status()
    payload = response.json()
    return "".join(
        block.get("text", "")
        for block in payload.get("content", [])
        if block.get("type") == "text"
    )


def label_packets(
    packets: List[Dict[str, Any]],
    analysis_dir: Path,
    model: str,
    temperature: float,
    max_retries: int,
    label_workers: int,
) -> List[Dict[str, Any]]:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    labels_path = analysis_dir / "llm_labels.jsonl"
    cached: Dict[str, Dict[str, Any]] = {}
    if labels_path.exists():
        for line in labels_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            cached[row["packet_id"]] = row

    new_rows: List[Dict[str, Any]] = []
    pending_packets: List[Dict[str, Any]] = []
    for packet in packets:
        packet_id = packet["packet_id"]
        if packet_id in cached:
            new_rows.append(cached[packet_id])
        else:
            pending_packets.append(packet)

    lock = threading.Lock()
    shared_client = (
        None
        if should_prefer_requests_backend()
        else build_shared_anthropic_client(api_key) if anthropic is not None else None
    )

    def _label_one(packet: Dict[str, Any]) -> Dict[str, Any]:
        packet_id = packet["packet_id"]
        prompt = packet_text(packet)
        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            raw_text: Optional[str] = None
            parsed_payload: Optional[Dict[str, Any]] = None
            try:
                raw_text = call_anthropic_messages(
                    api_key=api_key,
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    client=shared_client,
                )
                parsed_payload = extract_json_object(raw_text)
                parsed_payload = normalize_label_payload(parsed_payload)
                parsed = validate_label(parsed_payload)
                row = {"packet_id": packet_id, "model": model, **parsed}
                with lock:
                    with labels_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row) + "\n")
                return row
            except Exception as exc:  # pragma: no cover - runtime API handling
                last_error = exc
                with lock:
                    record_label_failure(
                        analysis_dir=analysis_dir,
                        packet_id=packet_id,
                        attempt=attempt,
                        error=exc,
                        raw_text=raw_text,
                        parsed_payload=parsed_payload,
                    )
                sleep_s = min(2 ** attempt, 20)
                time.sleep(sleep_s)
        raise RuntimeError(
            f"Labeling failed for {packet_id}: {describe_labeling_exception(last_error)}"
        ) from last_error

    try:
        if pending_packets:
            workers = max(1, min(label_workers, len(pending_packets)))
            if sys.version_info >= (3, 14):
                if label_workers != 1:
                    print(
                        "Python 3.14 runtime detected; forcing label_workers=1 for stable Anthropic HTTPS calls.",
                        file=sys.stderr,
                    )
                if should_prefer_requests_backend():
                    print(
                        "Python 3.14 runtime detected; using direct requests backend instead of the Anthropic SDK.",
                        file=sys.stderr,
                    )
                workers = 1
            if workers == 1:
                for packet in pending_packets:
                    new_rows.append(_label_one(packet))
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(_label_one, packet) for packet in pending_packets]
                    for future in concurrent.futures.as_completed(futures):
                        new_rows.append(future.result())
    finally:
        if shared_client is not None:
            with contextlib.suppress(Exception):
                shared_client.close()
    return new_rows


def aggregate_adversary_summary(df: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "consensus_reached": "mean",
        "final_round": "mean",
        "utility_gap": "mean",
        "adversary_stubbornness_proxy": "mean",
        "adversary_repeat_rate": "mean",
        "adversary_parse_failure_rate": "mean",
        "adversary_all_items_rate": "mean",
    }
    summary = (
        df.groupby(["adversary_model", "adversary_short", "adversary_tier", "competition_level"], as_index=False)
        .agg(columns)
        .rename(columns={"consensus_reached": "consensus_rate"})
    )
    return summary


def plot_stubbornness_by_competition(df: pd.DataFrame, figures_dir: Path) -> None:
    plot_df = (
        df.groupby(["competition_level", "adversary_tier"], as_index=False)["adversary_stubbornness_proxy"]
        .mean()
        .sort_values("competition_level")
    )
    plt.figure(figsize=(12, 7))
    for tier, group in plot_df.groupby("adversary_tier"):
        plt.plot(
            group["competition_level"],
            group["adversary_stubbornness_proxy"],
            marker="o",
            linewidth=2,
            label=tier,
        )
    plt.title("Game 1 Adversary Stubbornness Proxy by Competition")
    plt.xlabel("Competition Level")
    plt.ylabel("Adversary Stubbornness Proxy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "stubbornness_by_competition.png", dpi=200)
    plt.close()


def plot_stubbornness_vs_consensus(df: pd.DataFrame, figures_dir: Path) -> None:
    plot_df = (
        df.groupby(["consensus_reached", "adversary_tier"], as_index=False)["adversary_stubbornness_proxy"]
        .mean()
    )
    plt.figure(figsize=(10, 7))
    tiers = list(plot_df["adversary_tier"].drop_duplicates())
    x = np.arange(len(tiers))
    width = 0.35
    for offset, consensus in enumerate(sorted(plot_df["consensus_reached"].unique())):
        subset = (
            plot_df[plot_df["consensus_reached"] == consensus]
            .set_index("adversary_tier")
            .reindex(tiers)
        )
        plt.bar(
            x + (offset - 0.5) * width,
            subset["adversary_stubbornness_proxy"],
            width=width,
            label=f"consensus={consensus}",
        )
    plt.title("Game 1 Adversary Stubbornness Proxy vs Consensus Outcome")
    plt.xlabel("Adversary Tier")
    plt.ylabel("Adversary Stubbornness Proxy")
    plt.xticks(x, tiers)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "stubbornness_vs_consensus.png", dpi=200)
    plt.close()


def plot_first_concession_vs_elo(df: pd.DataFrame, figures_dir: Path) -> None:
    plot_df = df.dropna(subset=["adversary_first_concession_round", "adversary_elo"]).copy()
    if plot_df.empty:
        return
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        plot_df["adversary_elo"],
        plot_df["adversary_first_concession_round"],
        c=plot_df["competition_level"],
        cmap="viridis",
        s=70,
        alpha=0.8,
    )
    plt.title("Game 1 Adversary First Concession Round vs Elo")
    plt.xlabel("Adversary Elo")
    plt.ylabel("First Concession Round")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Competition Level")
    plt.tight_layout()
    plt.savefig(figures_dir / "first_concession_vs_elo.png", dpi=200)
    plt.close()


def plot_labeled_distribution(
    labeled_df: pd.DataFrame,
    figures_dir: Path,
    column: str,
    filename: str,
    title: str,
    x_col: str,
) -> None:
    if labeled_df.empty or column not in labeled_df.columns:
        return
    plot_df = pd.crosstab(labeled_df[x_col], labeled_df[column])
    plt.figure(figsize=(12, 7))
    plot_df.plot(kind="bar", stacked=True, ax=plt.gca())
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(figures_dir / filename, dpi=200)
    plt.close()


def plot_labeled_sample_counts_by_elo(labeled_df: pd.DataFrame, figures_dir: Path) -> None:
    if labeled_df.empty or "adversary_elo" not in labeled_df.columns:
        return
    counts = (
        labeled_df.groupby("adversary_elo", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("adversary_elo")
    )
    plt.figure(figsize=(12, 6))
    plt.bar(counts["adversary_elo"], counts["count"], width=10, color="#7f8c8d")
    plt.title("LLM-Labeled Game 1 Sample Coverage by Adversary Elo")
    plt.xlabel("Adversary Elo")
    plt.ylabel("Labeled Runs")
    plt.tight_layout()
    plt.savefig(figures_dir / "labeled_sample_count_by_elo.png", dpi=200)
    plt.close()


def plot_labeled_share_vs_elo(
    labeled_df: pd.DataFrame,
    figures_dir: Path,
    column: str,
    filename: str,
    title: str,
) -> None:
    if labeled_df.empty or column not in labeled_df.columns or "adversary_elo" not in labeled_df.columns:
        return

    plot_df = labeled_df.dropna(subset=["adversary_elo", column]).copy()
    if plot_df.empty:
        return

    counts = pd.crosstab(plot_df["adversary_elo"], plot_df[column]).sort_index()
    shares = counts.div(counts.sum(axis=1), axis=0)
    if shares.empty:
        return

    plt.figure(figsize=(12, 7))
    shares.index = [str(int(x)) if float(x).is_integer() else str(x) for x in shares.index]
    shares.plot(kind="bar", stacked=True, ax=plt.gca(), width=0.85)
    plt.title(title)
    plt.xlabel("Adversary Elo")
    plt.ylabel("Share Within Labeled Runs")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(figures_dir / filename, dpi=200)
    plt.close()


def plot_payoff_mechanisms(df: pd.DataFrame, figures_dir: Path) -> None:
    plot_df = df.dropna(subset=["adversary_elo"]).copy()
    if plot_df.empty:
        return

    plot_df["elo_band"] = plot_df["adversary_elo"].apply(assign_elo_band)
    plot_df["total_payoff"] = plot_df["baseline_utility"] + plot_df["adversary_utility"]
    band_order = [name for name, _, _ in ELO_BANDS]

    utility_summary = (
        plot_df.groupby("elo_band", observed=True)[["baseline_utility", "adversary_utility"]]
        .mean()
        .reindex(band_order)
    )
    conflict_summary = (
        plot_df.groupby(["elo_band", "top_item_conflict"], observed=True)["total_payoff"]
        .mean()
        .unstack(fill_value=np.nan)
        .reindex(band_order)
    )
    friction_summary = (
        plot_df.groupby("elo_band", observed=True)[
            ["adversary_parse_failure_rate", "adversary_repeat_rate", "adversary_stubbornness_proxy"]
        ]
        .mean()
        .reindex(band_order)
    )

    x = np.arange(len(band_order))
    width = 0.36

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    axes[0].bar(x, utility_summary["baseline_utility"], width=0.6, label="Baseline utility", color="#8fb9a8")
    axes[0].bar(
        x,
        utility_summary["adversary_utility"],
        width=0.6,
        bottom=utility_summary["baseline_utility"],
        label="Adversary utility",
        color="#d37a5b",
    )
    axes[0].set_title("Mean Utility by Elo Band")
    axes[0].set_xlabel("Adversary Elo band")
    axes[0].set_ylabel("Utility")
    axes[0].set_xticks(x, band_order)
    axes[0].legend(fontsize=9)

    no_conflict = conflict_summary.get(False, pd.Series(index=band_order, dtype=float))
    conflict = conflict_summary.get(True, pd.Series(index=band_order, dtype=float))
    axes[1].bar(x - width / 2, no_conflict.values, width=width, label="No top-item conflict", color="#5b8e7d")
    axes[1].bar(x + width / 2, conflict.values, width=width, label="Top-item conflict", color="#b85c38")
    axes[1].set_title("Total Payoff by Conflict Structure")
    axes[1].set_xlabel("Adversary Elo band")
    axes[1].set_ylabel("Mean total payoff")
    axes[1].set_xticks(x, band_order)
    axes[1].legend(fontsize=9)

    metric_labels = [
        ("adversary_parse_failure_rate", "Parse failure", "#c44e52"),
        ("adversary_repeat_rate", "Repeat rate", "#8172b2"),
        ("adversary_stubbornness_proxy", "Stubbornness proxy", "#4c72b0"),
    ]
    metric_width = 0.22
    for idx, (column, label, color) in enumerate(metric_labels):
        axes[2].bar(
            x + (idx - 1) * metric_width,
            friction_summary[column].values,
            width=metric_width,
            label=label,
            color=color,
        )
    axes[2].set_title("Negotiation Frictions by Elo Band")
    axes[2].set_xlabel("Adversary Elo band")
    axes[2].set_ylabel("Mean rate / score")
    axes[2].set_xticks(x, band_order)
    axes[2].set_ylim(0, 0.5)
    axes[2].legend(fontsize=9)

    fig.suptitle("Mechanisms Behind the Payoff Increase with Adversary Elo", fontsize=14)
    fig.tight_layout()
    fig.savefig(figures_dir / "payoff_mechanisms_by_elo.png", dpi=220)
    plt.close(fig)


def parse_list_field(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except Exception:
            continue
    return [text]


def format_ratio(count: int, total: int) -> str:
    if total <= 0:
        return "0/0 (0%)"
    return f"{count}/{total} ({count / total:.0%})"


def format_elo(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return str(int(value)) if float(value).is_integer() else f"{float(value):.0f}"


def markdown_file_link(path_str: str, label: str) -> str:
    return f"[{label}]({path_str}:1)"


def markdown_path_link(path_str: str, label: str) -> str:
    return f"[{label}]({path_str})"


def label_bundle(row: pd.Series) -> List[Tuple[str, str]]:
    pairs = []
    for column in LABEL_COLUMNS:
        value = row.get(column)
        if pd.notna(value):
            pairs.append((column, str(value)))
    return pairs


def choose_exemplar_row(labeled_df: pd.DataFrame, column: str, value: str) -> Optional[pd.Series]:
    subset = labeled_df[labeled_df[column] == value].copy()
    if subset.empty:
        return None

    preferred_id = PREFERRED_EXEMPLARS.get(column, {}).get(value)
    if preferred_id:
        preferred = subset[subset["packet_id"] == preferred_id]
        if not preferred.empty:
            return preferred.iloc[0]

    subset["evidence_count"] = subset["evidence"].apply(lambda item: len(parse_list_field(item)))
    if column == "failure_mode" and value != "none":
        subset = subset.sort_values(
            ["consensus_reached", "final_round", "evidence_count"],
            ascending=[True, False, False],
        )
    else:
        subset = subset.sort_values(
            ["evidence_count", "final_round", "consensus_reached"],
            ascending=[False, False, False],
        )
    return subset.iloc[0]


def select_audit_pack(exemplar_rows: Dict[Tuple[str, str], pd.Series]) -> List[pd.Series]:
    by_packet: Dict[str, pd.Series] = {}
    for row in exemplar_rows.values():
        by_packet.setdefault(str(row["packet_id"]), row)

    rows = list(by_packet.values())
    rows.sort(
        key=lambda row: (
            len(label_bundle(row)),
            len(parse_list_field(row.get("evidence"))),
            int(bool(row.get("consensus_reached"))),
            int(row.get("final_round", 0) or 0),
        ),
        reverse=True,
    )
    return rows


def format_evidence_snippet(row: pd.Series, packet_lookup: Dict[str, Dict[str, Any]], max_items: int = 2) -> str:
    evidence_items = parse_list_field(row.get("evidence"))
    selected = evidence_items[:max_items]
    if not selected:
        packet = packet_lookup.get(str(row["packet_id"]), {})
        excerpts = packet.get("discussion_excerpts", []) or []
        selected = [
            f"Round {excerpt['round']} {excerpt['speaker']}: {excerpt['content']}"
            for excerpt in excerpts[:max_items]
        ]
    return " / ".join(selected)


def count_label(subset: pd.DataFrame, column: str, value: str) -> int:
    if subset.empty:
        return 0
    return int((subset[column] == value).sum())


def assign_elo_band(elo: Any) -> Optional[str]:
    if elo is None or (isinstance(elo, float) and math.isnan(elo)):
        return None
    elo = float(elo)
    for name, lower, upper in ELO_BANDS:
        if lower <= elo < upper:
            return name
    return None


def build_elo_writeup(labeled_df: pd.DataFrame, analysis_dir: Path, figures_dir: Path) -> None:
    if labeled_df.empty or "adversary_elo" not in labeled_df.columns:
        return

    elo_df = labeled_df.dropna(subset=["adversary_elo"]).copy()
    if elo_df.empty:
        return

    elo_df["elo_band"] = elo_df["adversary_elo"].apply(assign_elo_band)
    band_groups = {name: elo_df[elo_df["elo_band"] == name].copy() for name, _, _ in ELO_BANDS}
    missing_elo = int(labeled_df["adversary_elo"].isna().sum())

    low = band_groups["low (<1250)"]
    mid = band_groups["mid (1250-1399)"]
    high = band_groups["high (>=1400)"]

    paragraph_1 = (
        f"Across the Elo-conditioned labeled subset ({len(elo_df)} runs with recorded Elo; "
        f"{missing_elo} labeled runs had missing Elo and are excluded from the band summaries), "
        f"the lowest-Elo region looks qualitatively different from the top of the ladder. "
        f"In the low band (<1250, n={len(low)}), openings are dominated by `targeted_anchor` "
        f"({format_ratio(count_label(low, 'opening_style', 'targeted_anchor'), len(low))}), "
        f"with another {format_ratio(count_label(low, 'opening_style', 'maximalist_anchor') + count_label(low, 'opening_style', 'parser_or_degenerate'), len(low))} "
        f"falling into `maximalist_anchor` or `parser_or_degenerate`. Adaptation is most often `rigid_repetition` "
        f"({format_ratio(count_label(low, 'adaptation_style', 'rigid_repetition'), len(low))}), and the dominant failure labels are "
        f"`parser_failure` ({format_ratio(count_label(low, 'failure_mode', 'parser_failure'), len(low))}) and "
        f"`repetitive_deadlock` ({format_ratio(count_label(low, 'failure_mode', 'repetitive_deadlock'), len(low))}). "
        f"In this same band, the adversary is labeled more stubborn than the baseline in "
        f"{format_ratio(count_label(low, 'relative_stubbornness', 'adversary_more_stubborn'), len(low))} of runs."
    )

    paragraph_2 = (
        f"The middle band (1250-1399, n={len(mid)}) already looks more functional. "
        f"`balanced_tradeoff` and `targeted_anchor` appear at nearly identical rates "
        f"({format_ratio(count_label(mid, 'opening_style', 'balanced_tradeoff'), len(mid))} vs "
        f"{format_ratio(count_label(mid, 'opening_style', 'targeted_anchor'), len(mid))}), while "
        f"`responsive_tradeoff` becomes the modal adaptation style "
        f"({format_ratio(count_label(mid, 'adaptation_style', 'responsive_tradeoff'), len(mid))}). "
        f"Most mid-Elo runs receive no failure label "
        f"({format_ratio(count_label(mid, 'failure_mode', 'none'), len(mid))}), although `repetitive_deadlock` remains visible "
        f"({format_ratio(count_label(mid, 'failure_mode', 'repetitive_deadlock'), len(mid))}). "
        f"Resolution is usually driven by `hybrid_compromise` "
        f"({format_ratio(count_label(mid, 'resolution_driver', 'hybrid_compromise'), len(mid))}), and the dominant relative-stubbornness label shifts to `neither` "
        f"({format_ratio(count_label(mid, 'relative_stubbornness', 'neither'), len(mid))})."
    )

    paragraph_3 = (
        f"At the top of the Elo range (>=1400, n={len(high)}), the main change is not simply that models become softer; it is that they become more strategically coherent. "
        f"`cooperative_exploration` is the single most common opening label "
        f"({format_ratio(count_label(high, 'opening_style', 'cooperative_exploration'), len(high))}), "
        f"`responsive_tradeoff` remains dominant "
        f"({format_ratio(count_label(high, 'adaptation_style', 'responsive_tradeoff'), len(high))}), and "
        f"{format_ratio(count_label(high, 'failure_mode', 'none'), len(high))} of runs receive no failure label at all. "
        f"`hybrid_compromise` remains the modal resolution driver "
        f"({format_ratio(count_label(high, 'resolution_driver', 'hybrid_compromise'), len(high))}), while `neither` dominates relative stubbornness "
        f"({format_ratio(count_label(high, 'relative_stubbornness', 'neither'), len(high))}). "
        f"Taken together, the Elo plots support a capability story about parser reliability and strategic coherence more than a simple monotone stubbornness story: the bottom of the ladder fails disproportionately because negotiations become malformed or cyclic, whereas stronger models more often converge through package-level tradeoffs."
    )

    lines = [
        "# Game 1 Elo Paper-Ready Paragraphs",
        "",
        paragraph_1,
        "",
        paragraph_2,
        "",
        paragraph_3,
        "",
        "## Figure References",
        "",
        f"- {markdown_path_link(str(figures_dir / 'opening_style_by_elo.png'), 'opening_style_by_elo.png')}",
        f"- {markdown_path_link(str(figures_dir / 'adaptation_style_by_elo.png'), 'adaptation_style_by_elo.png')}",
        f"- {markdown_path_link(str(figures_dir / 'failure_mode_by_elo.png'), 'failure_mode_by_elo.png')}",
        f"- {markdown_path_link(str(figures_dir / 'resolution_driver_by_elo.png'), 'resolution_driver_by_elo.png')}",
        f"- {markdown_path_link(str(figures_dir / 'relative_stubbornness_by_elo.png'), 'relative_stubbornness_by_elo.png')}",
    ]
    (analysis_dir / "game1_elo_paper_ready_paragraphs.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_label_exemplar_index(
    labeled_df: pd.DataFrame,
    packet_lookup: Dict[str, Dict[str, Any]],
    analysis_dir: Path,
) -> None:
    if labeled_df.empty:
        return

    exemplar_rows: Dict[Tuple[str, str], pd.Series] = {}
    csv_rows: List[Dict[str, Any]] = []
    for column in LABEL_COLUMNS:
        values = [value for value in labeled_df[column].value_counts().index.tolist() if pd.notna(value)]
        for value in values:
            row = choose_exemplar_row(labeled_df, column, str(value))
            if row is None:
                continue
            exemplar_rows[(column, str(value))] = row
            csv_rows.append({
                "label_family": column,
                "label_value": str(value),
                "packet_id": str(row["packet_id"]),
                "adversary_model": str(row["adversary_model"]),
                "adversary_elo": row["adversary_elo"],
                "competition_level": row["competition_level"],
                "consensus_reached": bool(row["consensus_reached"]),
                "final_round": int(row["final_round"]),
                "file_path": str(row["file_path"]),
                "surprising_feature": str(row.get("surprising_feature", "")),
                "evidence_snippet": format_evidence_snippet(row, packet_lookup, max_items=2),
            })

    pd.DataFrame(csv_rows).to_csv(analysis_dir / "label_exemplars.csv", index=False)

    audit_pack = select_audit_pack(exemplar_rows)
    lines = [
        "# Game 1 Label Exemplar Index",
        "",
        "This file is for auditing the current Claude Haiku labels. Repeated transcripts across sections are intentional because one run can be a clean exemplar for multiple label values.",
        "",
        "## Curated Audit Pack",
        "",
        f"These {len(audit_pack)} transcripts collectively cover every label value currently present in `labeled_runs.csv`.",
        "",
    ]

    for row in audit_pack:
        covered = ", ".join(f"`{value}`" for _, value in label_bundle(row))
        lines.extend([
            f"### {row['adversary_model']} | Elo {format_elo(row['adversary_elo'])} | comp {row['competition_level']}",
            "",
            markdown_file_link(str(row["file_path"]), f"{row['adversary_model']} transcript"),
            "",
            f"Covered labels: {covered}",
            f"Evidence: {format_evidence_snippet(row, packet_lookup, max_items=2)}",
            "",
        ])

    lines.extend([
        "## Exemplars By Label",
        "",
        "Use the audit-pack links above for a compact reading list, then use the sections below when you want the cleanest available example of a specific label value.",
        "",
    ])

    ordered_values = {
        **LABEL_SCHEMA,
        "relative_stubbornness": ["neither", "adversary_more_stubborn", "baseline_more_stubborn", "both_stubborn"],
    }

    for column in LABEL_COLUMNS:
        lines.extend([f"## {column}", ""])
        for value in ordered_values[column]:
            row = exemplar_rows.get((column, value))
            if row is None:
                continue
            other_labels = ", ".join(
                f"`{other_value}`"
                for other_column, other_value in label_bundle(row)
                if not (other_column == column and other_value == value)
            )
            lines.extend([
                f"### {value}",
                "",
                markdown_file_link(
                    str(row["file_path"]),
                    f"{row['adversary_model']} | Elo {format_elo(row['adversary_elo'])} | comp {row['competition_level']}",
                ),
                "",
                f"Context: consensus={bool(row['consensus_reached'])}, final_round={int(row['final_round'])}, model_order=`{row['model_order']}`.",
                f"Other labels on this run: {other_labels or 'none'}",
                f"Evidence: {format_evidence_snippet(row, packet_lookup, max_items=2)}",
                "",
            ])

    (analysis_dir / "game1_label_exemplar_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_payoff_mechanism_artifacts(
    df: pd.DataFrame,
    labeled_df: pd.DataFrame,
    analysis_dir: Path,
    figures_dir: Path,
) -> None:
    plot_df = df.dropna(subset=["adversary_elo"]).copy()
    if plot_df.empty:
        return

    plot_df["elo_band"] = plot_df["adversary_elo"].apply(assign_elo_band)
    plot_df["total_payoff"] = plot_df["baseline_utility"] + plot_df["adversary_utility"]
    band_order = [name for name, _, _ in ELO_BANDS]

    band_summary = (
        plot_df.groupby("elo_band", observed=True)
        .agg(
            n=("packet_id", "size"),
            consensus=("consensus_reached", "mean"),
            baseline_utility=("baseline_utility", "mean"),
            adversary_utility=("adversary_utility", "mean"),
            total_payoff=("total_payoff", "mean"),
            adversary_parse_failure_rate=("adversary_parse_failure_rate", "mean"),
            adversary_repeat_rate=("adversary_repeat_rate", "mean"),
            adversary_stubbornness_proxy=("adversary_stubbornness_proxy", "mean"),
        )
        .reindex(band_order)
    )
    conflict_summary = (
        plot_df.groupby(["elo_band", "top_item_conflict"], observed=True)
        .agg(
            n=("packet_id", "size"),
            total_payoff=("total_payoff", "mean"),
            adversary_utility=("adversary_utility", "mean"),
            baseline_utility=("baseline_utility", "mean"),
        )
        .reset_index()
    )

    known_labeled = labeled_df.dropna(subset=["adversary_elo"]).copy()
    known_labeled["total_payoff"] = known_labeled["baseline_utility"] + known_labeled["adversary_utility"]
    opening_summary = pd.DataFrame()
    adaptation_summary = pd.DataFrame()
    resolution_summary = pd.DataFrame()
    if not known_labeled.empty:
        opening_summary = (
            known_labeled.groupby("opening_style")
            .agg(n=("packet_id", "size"), mean_total_payoff=("total_payoff", "mean"), mean_consensus=("consensus_reached", "mean"))
            .sort_values("mean_total_payoff", ascending=False)
        )
        adaptation_summary = (
            known_labeled.groupby("adaptation_style")
            .agg(n=("packet_id", "size"), mean_total_payoff=("total_payoff", "mean"), mean_consensus=("consensus_reached", "mean"))
            .sort_values("mean_total_payoff", ascending=False)
        )
        resolution_summary = (
            known_labeled.groupby("resolution_driver")
            .agg(n=("packet_id", "size"), mean_total_payoff=("total_payoff", "mean"), mean_consensus=("consensus_reached", "mean"))
            .sort_values("mean_total_payoff", ascending=False)
        )

    labeled_index = labeled_df.set_index("packet_id", drop=False)
    table_lines = [
        "# Game 1 Payoff-Mechanism Table",
        "",
        "This table is designed as a paper-facing companion to the Elo/payoff figure.",
        "",
        "| Mechanism | Quantitative hook | Exemplar | What to look for |",
        "| --- | --- | --- | --- |",
    ]
    for item in PAYOFF_MECHANISM_EXEMPLARS:
        row = labeled_index.loc[item["packet_id"]]
        exemplar_link = markdown_file_link(
            str(row["file_path"]),
            f"{row['adversary_model']} (Elo {format_elo(row['adversary_elo'])}, comp {row['competition_level']})",
        )
        table_lines.append(
            f"| {item['mechanism']} | {item['quantitative_hook']} | {exemplar_link} | {item['what_to_look_for']} |"
        )
    (analysis_dir / "game1_payoff_mechanism_table.md").write_text("\n".join(table_lines) + "\n", encoding="utf-8")

    p1 = (
        f"Figure 1 decomposes the positive relationship between adversary Elo and payoff into two separate effects. "
        f"First, the payoff increase is asymmetric: across the full Game 1 batch with recorded Elo, baseline utility is nearly flat across Elo bands "
        f"({band_summary.loc['low (<1250)', 'baseline_utility']:.1f} in the low band, "
        f"{band_summary.loc['mid (1250-1399)', 'baseline_utility']:.1f} in the middle band, "
        f"and {band_summary.loc['high (>=1400)', 'baseline_utility']:.1f} in the high band), whereas adversary utility rises substantially "
        f"({band_summary.loc['low (<1250)', 'adversary_utility']:.1f} -> "
        f"{band_summary.loc['mid (1250-1399)', 'adversary_utility']:.1f} -> "
        f"{band_summary.loc['high (>=1400)', 'adversary_utility']:.1f}). "
        f"Total payoff also rises ({band_summary.loc['low (<1250)', 'total_payoff']:.1f} -> "
        f"{band_summary.loc['mid (1250-1399)', 'total_payoff']:.1f} -> "
        f"{band_summary.loc['high (>=1400)', 'total_payoff']:.1f}), but more modestly than adversary utility alone. "
        f"This implies that stronger adversaries are not simply making bargaining more efficient for everyone; they are also capturing a larger share of the available surplus."
    )

    no_conflict = conflict_summary[conflict_summary["top_item_conflict"] == False].set_index("elo_band")
    conflict = conflict_summary[conflict_summary["top_item_conflict"] == True].set_index("elo_band")
    p2 = (
        f"The welfare gain appears to come mainly from stronger models handling easy-to-trade preference structures more cleanly, not from a universal ability to solve hard conflicts. "
        f"When there is no top-item conflict, mean total payoff rises sharply from "
        f"{no_conflict.loc['low (<1250)', 'total_payoff']:.1f} in the low-Elo band to "
        f"{no_conflict.loc['high (>=1400)', 'total_payoff']:.1f} in the high-Elo band. "
        f"By contrast, when both agents want the same top item, total payoff remains low across all three bands "
        f"({conflict.loc['low (<1250)', 'total_payoff']:.1f}, "
        f"{conflict.loc['mid (1250-1399)', 'total_payoff']:.1f}, "
        f"{conflict.loc['high (>=1400)', 'total_payoff']:.1f}). "
        f"This suggests that the main capability gain is better recognition and closure of complementary package deals, rather than a qualitative breakthrough on the hardest top-item collisions."
    )

    if not known_labeled.empty:
        p3 = (
            f"The labeled subset supports the same mechanism qualitatively. High-welfare runs are disproportionately associated with "
            f"`cooperative_exploration` openings (mean total payoff {opening_summary.loc['cooperative_exploration', 'mean_total_payoff']:.1f}), "
            f"`responsive_tradeoff` or low-friction `minimal_evidence` adaptation "
            f"({adaptation_summary.loc['responsive_tradeoff', 'mean_total_payoff']:.1f} and {adaptation_summary.loc['minimal_evidence', 'mean_total_payoff']:.1f}, respectively), "
            f"and `hybrid_compromise` resolutions (mean total payoff {resolution_summary.loc['hybrid_compromise', 'mean_total_payoff']:.1f}). "
            f"Low-payoff runs, in contrast, are dominated by `rigid_repetition` "
            f"(mean total payoff {adaptation_summary.loc['rigid_repetition', 'mean_total_payoff']:.1f}) or by negotiations that collapse into the adversary's frame without much joint surplus "
            f"(`adversary_frame_accepted`, mean total payoff {resolution_summary.loc['adversary_frame_accepted', 'mean_total_payoff']:.1f}). "
            f"Taken together, the payoff-vs-Elo slope seems to combine two mechanisms: stronger models waste fewer rounds on malformed or cyclic bargaining, and they are better at turning complementary preferences into high-value package deals."
        )
    else:
        p3 = ""

    draft_lines = [
        "# Game 1 Payoff-vs-Elo Subsection Draft",
        "",
        p1,
        "",
        p2,
    ]
    if p3:
        draft_lines.extend(["", p3])
    draft_lines.extend([
        "",
        "## Figure and Table",
        "",
        f"- Figure: {markdown_path_link(str(figures_dir / 'payoff_mechanisms_by_elo.png'), 'payoff_mechanisms_by_elo.png')}",
        f"- Table: {markdown_file_link(str(analysis_dir / 'game1_payoff_mechanism_table.md'), 'game1_payoff_mechanism_table.md')}",
    ])
    (analysis_dir / "game1_payoff_vs_elo_subsection.md").write_text("\n".join(draft_lines) + "\n", encoding="utf-8")


def build_report(df: pd.DataFrame, labeled_df: pd.DataFrame, analysis_dir: Path) -> None:
    comp1 = df[df["competition_level"] == 1.0].copy()
    lines = [
        "# Game 1 Qualitative Review",
        "",
        "## Coverage",
        "",
        f"- Runs analyzed: {len(df)}",
        f"- Unique adversary models: {df['adversary_model'].nunique()}",
        f"- No-consensus runs: {int((~df['consensus_reached']).sum())}",
        f"- LLM-labeled runs: {len(labeled_df)}",
        "",
        "## Deterministic Signals",
        "",
    ]

    if not comp1.empty:
        worst_consensus = (
            comp1.groupby("adversary_model", as_index=False)["consensus_reached"]
            .mean()
            .sort_values("consensus_reached")
            .head(8)
        )
        lines.append("Lowest consensus-rate adversaries at competition 1.0:")
        for _, row in worst_consensus.iterrows():
            lines.append(f"- `{row['adversary_model']}`: consensus rate {row['consensus_reached']:.2f}")
        lines.append("")

        stubborn = (
            comp1.groupby("adversary_model", as_index=False)["adversary_stubbornness_proxy"]
            .mean()
            .sort_values("adversary_stubbornness_proxy", ascending=False)
            .head(8)
        )
        lines.append("Highest adversary stubbornness proxy at competition 1.0:")
        for _, row in stubborn.iterrows():
            lines.append(f"- `{row['adversary_model']}`: proxy {row['adversary_stubbornness_proxy']:.2f}")
        lines.append("")

    lines.extend([
        "## Exemplar Candidates",
        "",
    ])
    exemplar_df = df.sort_values(
        ["consensus_reached", "adversary_stubbornness_proxy", "final_round"],
        ascending=[True, False, False],
    ).head(10)
    for _, row in exemplar_df.iterrows():
        lines.append(
            f"- `{row['adversary_model']}` at comp `{row['competition_level']}`"
            f" | consensus={bool(row['consensus_reached'])}"
            f" | rounds={int(row['final_round'])}"
            f" | stubbornness={row['adversary_stubbornness_proxy']:.2f}"
            f" | file={row['file_path']}"
        )

    if not labeled_df.empty:
        lines.extend([
            "",
            "## LLM Label Snapshot",
            "",
            "These counts come from the labeled subset, not the full 890-run batch.",
            "",
        ])
        for column in ["opening_style", "adaptation_style", "failure_mode", "resolution_driver", "relative_stubbornness"]:
            counts = labeled_df[column].value_counts().sort_values(ascending=False)
            lines.append(f"{column}:")
            for label, count in counts.items():
                lines.append(f"- `{label}`: {int(count)}")
            lines.append("")

        lines.extend([
            "## Companion Docs",
            "",
            "- `game1_elo_paper_ready_paragraphs.md`",
            "- `game1_label_exemplar_index.md`",
            "- `label_exemplars.csv`",
            "- `game1_payoff_mechanism_table.md`",
            "- `game1_payoff_vs_elo_subsection.md`",
            "- `why_payoff_rises_with_elo_notes.md`",
            "",
        ])

    (analysis_dir / "game1_qualitative_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dirs(args.analysis_dir, args.figures_dir)

    results = load_results(args.results_dir)
    if not results:
        raise SystemExit(f"No Game 1 results found under {args.results_dir}")

    df, result_index = build_feature_table(results)
    df = df.sort_values(["adversary_tier", "competition_level", "adversary_elo", "model_order"]).reset_index(drop=True)
    df.to_csv(args.analysis_dir / "run_features.csv", index=False)

    summary_df = aggregate_adversary_summary(df)
    summary_df.to_csv(args.analysis_dir / "adversary_summary.csv", index=False)

    packets_df = select_runs_for_labeling(df, args.per_stratum, args.extreme_count, args.seed)
    if args.label_limit is not None:
        packets_df = packets_df.head(args.label_limit).copy()

    packets = []
    for _, row in packets_df.iterrows():
        packet = build_label_packet(row.to_dict(), result_index[row["packet_id"]])
        packets.append(packet)
    packet_lookup = {packet["packet_id"]: packet for packet in packets}
    with (args.analysis_dir / "label_packets.jsonl").open("w", encoding="utf-8") as handle:
        for packet in packets:
            handle.write(json.dumps(packet, default=_json_default) + "\n")

    labeled_df = pd.DataFrame()
    if not args.skip_labeling and packets:
        labels = label_packets(
            packets=packets,
            analysis_dir=args.analysis_dir,
            model=args.label_model,
            temperature=args.temperature,
            max_retries=args.max_retries,
            label_workers=args.label_workers,
        )
        labels_df = pd.DataFrame(labels)
        labeled_df = packets_df.merge(labels_df, on="packet_id", how="left")
        labeled_df = labeled_df.dropna(subset=["opening_style"]).copy()
        labeled_df.to_csv(args.analysis_dir / "labeled_runs.csv", index=False)
    elif (args.analysis_dir / "llm_labels.jsonl").exists():
        rows = [
            json.loads(line)
            for line in (args.analysis_dir / "llm_labels.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if rows:
            labels_df = pd.DataFrame(rows)
            labeled_df = packets_df.merge(labels_df, on="packet_id", how="left")
            labeled_df = labeled_df.dropna(subset=["opening_style"]).copy()
            labeled_df.to_csv(args.analysis_dir / "labeled_runs.csv", index=False)

    plot_stubbornness_by_competition(df, args.figures_dir)
    plot_stubbornness_vs_consensus(df, args.figures_dir)
    plot_first_concession_vs_elo(df, args.figures_dir)
    plot_payoff_mechanisms(df, args.figures_dir)

    if not labeled_df.empty:
        plot_labeled_sample_counts_by_elo(labeled_df, args.figures_dir)
        plot_labeled_distribution(
            labeled_df=labeled_df,
            figures_dir=args.figures_dir,
            column="opening_style",
            filename="opening_style_by_tier.png",
            title="LLM Opening Style Labels by Adversary Tier",
            x_col="adversary_tier",
        )
        plot_labeled_distribution(
            labeled_df=labeled_df,
            figures_dir=args.figures_dir,
            column="failure_mode",
            filename="failure_mode_by_competition.png",
            title="LLM Failure Modes by Competition Level",
            x_col="competition_level",
        )
        plot_labeled_share_vs_elo(
            labeled_df=labeled_df,
            figures_dir=args.figures_dir,
            column="opening_style",
            filename="opening_style_by_elo.png",
            title="LLM Opening Style Share by Adversary Elo",
        )
        plot_labeled_share_vs_elo(
            labeled_df=labeled_df,
            figures_dir=args.figures_dir,
            column="adaptation_style",
            filename="adaptation_style_by_elo.png",
            title="LLM Adaptation Style Share by Adversary Elo",
        )
        plot_labeled_share_vs_elo(
            labeled_df=labeled_df,
            figures_dir=args.figures_dir,
            column="failure_mode",
            filename="failure_mode_by_elo.png",
            title="LLM Failure Mode Share by Adversary Elo",
        )
        plot_labeled_share_vs_elo(
            labeled_df=labeled_df,
            figures_dir=args.figures_dir,
            column="resolution_driver",
            filename="resolution_driver_by_elo.png",
            title="LLM Resolution Driver Share by Adversary Elo",
        )
        plot_labeled_share_vs_elo(
            labeled_df=labeled_df,
            figures_dir=args.figures_dir,
            column="relative_stubbornness",
            filename="relative_stubbornness_by_elo.png",
            title="LLM Relative Stubbornness Share by Adversary Elo",
        )

    build_report(df, labeled_df, args.analysis_dir)
    build_elo_writeup(labeled_df, args.analysis_dir, args.figures_dir)
    build_label_exemplar_index(labeled_df, packet_lookup, args.analysis_dir)
    build_payoff_mechanism_artifacts(df, labeled_df, args.analysis_dir, args.figures_dir)

    metadata = {
        "results_dir": str(args.results_dir),
        "analysis_dir": str(args.analysis_dir),
        "figures_dir": str(args.figures_dir),
        "run_count": int(len(df)),
        "labeled_count": int(len(labeled_df)),
        "label_model": args.label_model if not args.skip_labeling else None,
    }
    save_json(args.analysis_dir / "run_metadata.json", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

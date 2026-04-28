#!/usr/bin/env python3
"""Run the five approved Game 2 multi-agent sample tests.

This script creates a small batch-shaped results directory so the Streamlit
viewer can browse all samples from one root.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
from scipy.stats import pearsonr, spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "experiments" / "results"
BASELINE_MODEL = "gpt-5-nano"
N_AGENTS = 10
N_ISSUES = 10
MAX_ROUNDS = 10
DISCUSSION_TURNS = 2
GAMMA_DISCOUNT = 0.9
THETA = 0.9
BASE_SEED = 27000
HOMOGENEOUS_PHASE_TOKEN_LIMITS = {
    "discussion": 32768,
    "thinking": 32768,
    "proposal": 32768,
    "voting": 32768,
    "reflection": 32768,
}
HETEROGENEOUS_PHASE_TOKEN_LIMITS = {
    "discussion": 32768,
    "thinking": 32768,
    "proposal": 32768,
    "voting": 32768,
    "reflection": 32768,
}

MODEL_POOL_25 = [
    "claude-opus-4-6-thinking",
    "claude-opus-4-6",
    "gemini-3-pro",
    "gpt-5.4-high",
    "gpt-5.2-chat-latest-20260210",
    "claude-opus-4-5-20251101-thinking-32k",
    "claude-opus-4-5-20251101",
    "gemini-2.5-pro",
    "qwen3-max-preview",
    "deepseek-r1-0528",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-20250514",
    "gemma-3-27b-it",
    "o3-mini-high",
    "deepseek-v3",
    "gpt-4o-2024-05-13",
    "gpt-5-nano-high",
    "qwq-32b",
    "gpt-4.1-nano-2025-04-14",
    "llama-3.3-70b-instruct",
    "gpt-4o-mini-2024-07-18",
    "amazon-nova-pro-v1.0",
    "command-r-plus-08-2024",
    "claude-3-haiku-20240307",
    "amazon-nova-micro-v1.0",
]

MODEL_ELO = {
    "claude-opus-4-6-thinking": 1504,
    "claude-opus-4-6": 1499,
    "gemini-3-pro": 1486,
    "gpt-5.4-high": 1484,
    "gpt-5.2-chat-latest-20260210": 1478,
    "claude-opus-4-5-20251101-thinking-32k": 1474,
    "claude-opus-4-5-20251101": 1468,
    "gemini-2.5-pro": 1448,
    "qwen3-max-preview": 1435,
    "deepseek-r1-0528": 1422,
    "claude-haiku-4-5-20251001": 1407,
    "claude-sonnet-4-20250514": 1389,
    "gemma-3-27b-it": 1365,
    "o3-mini-high": 1363,
    "deepseek-v3": 1358,
    "gpt-4o-2024-05-13": 1345,
    "gpt-5-nano-high": 1337,
    "qwq-32b": 1336,
    "gpt-4.1-nano-2025-04-14": 1322,
    "llama-3.3-70b-instruct": 1318,
    "gpt-4o-mini-2024-07-18": 1317,
    "amazon-nova-pro-v1.0": 1290,
    "command-r-plus-08-2024": 1276,
    "claude-3-haiku-20240307": 1260,
    "amazon-nova-micro-v1.0": 1240,
}


def rho_low(n_agents: int) -> float:
    if n_agents == 2:
        return -1.0
    return (6.0 / math.pi) * math.asin(-1.0 / (2.0 * (n_agents - 1)))


def timestamp_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def slugify(value: str) -> str:
    chars = [ch.lower() if ch.isalnum() else "_" for ch in value]
    slug = "".join(chars)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_samples(run_dir: Path) -> List[Dict[str, Any]]:
    rho = rho_low(N_AGENTS)
    samples: List[Dict[str, Any]] = []

    adversaries = [
        ("claude-opus-4-6-thinking", "claude_opus_4_6_thinking"),
        ("amazon-nova-micro-v1.0", "amazon_nova_micro"),
    ]
    sample_id = 0
    for adversary, adversary_slug in adversaries:
        for position in ("first", "last"):
            if position == "first":
                models = [adversary] + [BASELINE_MODEL] * (N_AGENTS - 1)
                model_order = "strong_first"
            else:
                models = [BASELINE_MODEL] * (N_AGENTS - 1) + [adversary]
                model_order = "weak_first"

            output_dir = run_dir / "runs" / f"{sample_id:02d}_homogeneous_{adversary_slug}_{position}"
            samples.append({
                "experiment_id": sample_id,
                "sample_name": f"homogeneous_{adversary_slug}_{position}",
                "experiment_type": "homogeneous_adversary_sample",
                "models": models,
                "model1": BASELINE_MODEL,
                "model2": adversary,
                "model_order": model_order,
                "adversary_model": adversary,
                "adversary_position": position,
                "n_agents": N_AGENTS,
                "n_issues": N_ISSUES,
                "rho": rho,
                "theta": THETA,
                "max_rounds": MAX_ROUNDS,
                "gamma_discount": GAMMA_DISCOUNT,
                "discussion_turns": DISCUSSION_TURNS,
                "phase_token_limits": HOMOGENEOUS_PHASE_TOKEN_LIMITS,
                "random_seed": BASE_SEED + sample_id,
                "output_dir": str(output_dir),
            })
            sample_id += 1

    # Fixed after the initial sample draw selected claude-3-haiku-20240307, which
    # currently returns a provider 404 from Anthropic. Keep the sample in the
    # approved 25-model pool while avoiding that unavailable route.
    heterogeneous_models = [
        "amazon-nova-micro-v1.0",
        "gpt-5.4-high",
        "claude-opus-4-6-thinking",
        "gpt-5-nano-high",
        "deepseek-r1-0528",
        "deepseek-v3",
        "gpt-4o-2024-05-13",
        "command-r-plus-08-2024",
        "gpt-4o-mini-2024-07-18",
        "claude-opus-4-5-20251101",
    ]
    output_dir = run_dir / "runs" / f"{sample_id:02d}_heterogeneous_random_pool"
    samples.append({
        "experiment_id": sample_id,
        "sample_name": "heterogeneous_random_pool",
        "experiment_type": "heterogeneous_random_sample",
        "models": heterogeneous_models,
        "model1": "heterogeneous_pool",
        "model2": "random_10_models",
        "model_order": "random_sampled_order",
        "adversary_model": None,
        "adversary_position": None,
        "n_agents": N_AGENTS,
        "n_issues": N_ISSUES,
        "rho": rho,
        "theta": THETA,
        "max_rounds": MAX_ROUNDS,
        "gamma_discount": GAMMA_DISCOUNT,
        "discussion_turns": DISCUSSION_TURNS,
        "phase_token_limits": HETEROGENEOUS_PHASE_TOKEN_LIMITS,
        "random_seed": BASE_SEED + sample_id,
        "model_pool_source": "docs/guides/chatbot_arena_elo_scores_2026_03_31_smooth_33_models.md filtered to 25 models with usable >=100K context and Elo>=1240",
        "output_dir": str(output_dir),
    })
    return samples


def pairwise_stats(values: Iterable[float]) -> Dict[str, Any]:
    arr = np.array(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p05": float(np.quantile(arr, 0.05)),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "values": [float(x) for x in arr.tolist()],
    }


def empirical_diagnostics(result_payload: Dict[str, Any]) -> Dict[str, Any]:
    config = result_payload.get("config", {})
    positions = config.get("agent_positions", {})
    weights = config.get("agent_weights", {})
    agents = sorted(positions)

    spearman_values = []
    pearson_values = []
    cosine_values = []
    for i, agent_a in enumerate(agents):
        for agent_b in agents[i + 1:]:
            pos_a = np.asarray(positions[agent_a], dtype=float)
            pos_b = np.asarray(positions[agent_b], dtype=float)
            if pos_a.size and pos_b.size and not np.allclose(pos_a, pos_a[0]) and not np.allclose(pos_b, pos_b[0]):
                spearman_values.append(float(spearmanr(pos_a, pos_b).statistic))
                pearson_values.append(float(pearsonr(pos_a, pos_b).statistic))

            w_a = np.asarray(weights.get(agent_a, []), dtype=float)
            w_b = np.asarray(weights.get(agent_b, []), dtype=float)
            if w_a.size and w_b.size:
                denom = float(np.linalg.norm(w_a) * np.linalg.norm(w_b))
                if denom > 0:
                    cosine_values.append(float(np.dot(w_a, w_b) / denom))

    return {
        "commanded": {
            "rho": config.get("rho"),
            "theta": config.get("theta"),
        },
        "empirical_rho_spearman": pairwise_stats(spearman_values),
        "empirical_rho_pearson": pairwise_stats(pearson_values),
        "empirical_theta_cosine": pairwise_stats(cosine_values),
    }


def find_result_file(output_dir: Path) -> Path | None:
    for name in ("run_1_experiment_results.json", "experiment_results.json"):
        path = output_dir / name
        if path.exists():
            return path
    return None


def looks_like_refusal(value: Any) -> bool:
    normalized = " ".join(str(value or "").strip().lower().split())
    if not normalized:
        return False
    return normalized.startswith((
        "i'm sorry, but i cannot assist",
        "i am sorry, but i cannot assist",
        "sorry, but i can't assist",
        "i can't assist with that request",
        "i cannot assist with that request",
    ))


def dirty_interaction_summary(output_dir: Path) -> Dict[str, Any]:
    interactions_path = output_dir / "run_1_all_interactions.json"
    if not interactions_path.exists():
        return {"count": 0, "interactions_file": str(interactions_path), "examples": []}

    records = json.loads(interactions_path.read_text(encoding="utf-8"))
    examples = []
    for idx, record in enumerate(records):
        response = record.get("response", "")
        parsed_response = None
        if isinstance(response, str):
            try:
                parsed_response = json.loads(response)
            except json.JSONDecodeError:
                parsed_response = None

        dirty_reason = None
        if isinstance(parsed_response, dict):
            if "parse_error" in parsed_response:
                dirty_reason = f"parse_error: {parsed_response['parse_error']}"
            elif parsed_response.get("reasoning") == "Missing or invalid vote entry":
                dirty_reason = "missing_or_invalid_vote_entry"
            elif parsed_response.get("raw_response") == "":
                dirty_reason = "empty_raw_response"
            elif "validation_error" in parsed_response:
                dirty_reason = f"validation_error: {parsed_response['validation_error']}"
        elif isinstance(response, str) and (
            '"parse_error"' in response
            or "Missing or invalid vote entry" in response
            or '"validation_error"' in response
        ):
            dirty_reason = "dirty_marker_in_response"
        elif looks_like_refusal(response):
            dirty_reason = "refusal_response"

        if dirty_reason:
            examples.append({
                "interaction_index": idx,
                "agent_id": record.get("agent_id"),
                "phase": record.get("phase"),
                "round": record.get("round"),
                "reason": dirty_reason,
            })

    return {
        "count": len(examples),
        "interactions_file": str(interactions_path),
        "examples": examples[:20],
    }


def postprocess_result(sample: Dict[str, Any]) -> Dict[str, Any]:
    output_dir = Path(sample["output_dir"])
    result_path = find_result_file(output_dir)
    if result_path is None:
        return {"status": "missing_result", "result_file": None}

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    agents = payload.get("config", {}).get("agents", [])
    agent_model_map = {
        agent_id: sample["models"][idx]
        for idx, agent_id in enumerate(agents)
        if idx < len(sample["models"])
    }
    agent_elo_map = {
        agent_id: MODEL_ELO.get(model_name)
        for agent_id, model_name in agent_model_map.items()
    }

    sample_metadata = {
        key: sample.get(key)
        for key in [
            "sample_name",
            "experiment_type",
            "n_agents",
            "n_issues",
            "adversary_model",
            "adversary_position",
            "model_pool_source",
            "phase_token_limits",
        ]
        if key in sample
    }
    sample_metadata["agent_model_map"] = agent_model_map
    sample_metadata["agent_elo_map"] = agent_elo_map
    sample_metadata["models_in_order"] = sample["models"]

    payload.setdefault("config", {})
    payload["config"]["sample_metadata"] = sample_metadata
    payload["config"]["agent_model_map"] = agent_model_map
    payload["config"]["agent_elo_map"] = agent_elo_map
    payload["config"]["models_in_order"] = sample["models"]
    payload["config"]["game2_parameter_diagnostics"] = empirical_diagnostics(payload)

    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    dirty_summary = dirty_interaction_summary(output_dir)
    if dirty_summary["count"]:
        return {
            "status": "dirty_output",
            "result_file": str(result_path),
            "dirty_interactions": dirty_summary,
            "diagnostics": payload["config"]["game2_parameter_diagnostics"],
        }

    return {
        "status": "completed",
        "result_file": str(result_path),
        "consensus_reached": payload.get("consensus_reached"),
        "final_round": payload.get("final_round"),
        "diagnostics": payload["config"]["game2_parameter_diagnostics"],
    }


def command_for_sample(sample: Dict[str, Any]) -> List[str]:
    cmd = [
        sys.executable,
        "run_strong_models_experiment.py",
        "--game-type", "diplomacy",
        "--models", *sample["models"],
        "--batch",
        "--num-runs", "1",
        "--run-number", "1",
        "--max-rounds", str(sample["max_rounds"]),
        "--n-issues", str(sample["n_issues"]),
        "--rho", str(sample["rho"]),
        "--theta", str(sample["theta"]),
        "--gamma-discount", str(sample["gamma_discount"]),
        "--discussion-turns", str(sample["discussion_turns"]),
        "--max-tokens-discussion", str(sample["phase_token_limits"]["discussion"]),
        "--max-tokens-thinking", str(sample["phase_token_limits"]["thinking"]),
        "--max-tokens-proposal", str(sample["phase_token_limits"]["proposal"]),
        "--max-tokens-voting", str(sample["phase_token_limits"]["voting"]),
        "--max-tokens-reflection", str(sample["phase_token_limits"]["reflection"]),
        "--random-seed", str(sample["random_seed"]),
        "--model-order", "strong_first" if sample["model_order"] == "strong_first" else "weak_first",
        "--output-dir", sample["output_dir"],
        "--job-id", str(sample["experiment_id"]),
    ]
    return cmd


def write_batch_files(run_dir: Path, samples: List[Dict[str, Any]]) -> None:
    config_dir = run_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        write_json(config_dir / f"config_{sample['experiment_id']:04d}.json", sample)

    index_path = config_dir / "experiment_index.csv"
    with open(index_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment_id",
                "experiment_type",
                "model1",
                "model2",
                "model_order",
                "rho",
                "theta",
                "run_number",
                "seed",
                "config_file",
            ],
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow({
                "experiment_id": sample["experiment_id"],
                "experiment_type": sample["experiment_type"],
                "model1": sample["model1"],
                "model2": sample["model2"],
                "model_order": sample["model_order"],
                "rho": sample["rho"],
                "theta": sample["theta"],
                "run_number": 1,
                "seed": sample["random_seed"],
                "config_file": f"config_{sample['experiment_id']:04d}.json",
            })

    (config_dir / "all_configs.txt").write_text(
        "\n".join(str(config_dir / f"config_{sample['experiment_id']:04d}.json") for sample in samples) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--only-create-configs", action="store_true")
    parser.add_argument("--sample-id", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
    else:
        run_dir = RESULTS_ROOT / f"game2_samples_{timestamp_now()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    samples = build_samples(run_dir)
    if args.sample_id is not None:
        samples = [sample for sample in samples if sample["experiment_id"] == args.sample_id]
        if not samples:
            raise ValueError(f"No sample with experiment_id={args.sample_id}")
    manifest = {
        "created_at": datetime.now().isoformat(),
        "repo_root": str(REPO_ROOT),
        "run_dir": str(run_dir),
        "n_agents": N_AGENTS,
        "n_issues": N_ISSUES,
        "rho_low": rho_low(N_AGENTS),
        "theta": THETA,
        "phase_token_limits": {
            "homogeneous": HOMOGENEOUS_PHASE_TOKEN_LIMITS,
            "heterogeneous": HETEROGENEOUS_PHASE_TOKEN_LIMITS,
        },
        "model_pool_25": MODEL_POOL_25,
        "samples": samples,
    }
    if args.sample_id is None or not (run_dir / "manifest.json").exists():
        write_json(run_dir / "manifest.json", manifest)
    if args.sample_id is None or not (run_dir / "configs" / "experiment_index.csv").exists():
        write_batch_files(run_dir, build_samples(run_dir))

    if args.only_create_configs:
        print(run_dir)
        return 0

    statuses = []
    status_path = (
        run_dir / "status" / f"sample_{args.sample_id:04d}.json"
        if args.sample_id is not None
        else run_dir / "status.json"
    )
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        output_dir = Path(sample["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        existing_result = find_result_file(output_dir)
        if args.resume and existing_result is not None:
            status = postprocess_result(sample)
            status.update({
                "sample_name": sample["sample_name"],
                "returncode": 0 if status.get("status") == "completed" else 1,
                "skipped_existing": True,
            })
            statuses.append(status)
            write_json(status_path, {"samples": statuses})
            continue

        cmd = command_for_sample(sample)
        log_path = logs_dir / f"{sample['experiment_id']:04d}_{sample['sample_name']}.log"
        print(f"Running sample {sample['experiment_id']}: {sample['sample_name']}", flush=True)
        print(" ".join(cmd), flush=True)
        start = time.time()
        env = os.environ.copy()
        env.setdefault(
            "LLM_FAILURE_REPORT_PATH",
            str(run_dir / "monitoring" / "provider_failures.md"),
        )
        with open(log_path, "w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                text=True,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=env,
            )
        status = {
            "sample_name": sample["sample_name"],
            "returncode": completed.returncode,
            "duration_seconds": round(time.time() - start, 3),
            "log_file": str(log_path),
            "output_dir": str(output_dir),
        }
        if completed.returncode == 0:
            status.update(postprocess_result(sample))
            if status.get("status") != "completed":
                status["returncode"] = 1
        else:
            status["status"] = "failed"
        statuses.append(status)
        write_json(status_path, {"samples": statuses})

    failures = [status for status in statuses if status.get("returncode") != 0]
    print(json.dumps({"run_dir": str(run_dir), "failures": failures}, indent=2), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

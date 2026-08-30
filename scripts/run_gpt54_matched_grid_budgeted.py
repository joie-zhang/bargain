#!/usr/bin/env python3
"""Run the remaining matched GPT-5.4 Game 1 cells within a spend cap."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PRICE_PER_MILLION = 2.50
OUTPUT_PRICE_PER_MILLION = 15.00
RESERVE_BY_N = {2: 4.0, 4: 4.0, 6: 8.0, 8: 13.0, 10: 20.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--budget-usd", type=float, required=True)
    parser.add_argument("--prior-spend-usd", type=float, default=0.0)
    parser.add_argument("--ledger", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def config_path(results_root: Path, config_id: str) -> Path:
    return results_root / "configs" / f"{config_id}.json"


def interaction_cost(output_dir: Path) -> tuple[int, int, float, bool]:
    path = output_dir / "all_interactions.json"
    if not path.exists():
        return 0, 0, 0.0, False
    interactions = read_json(path)
    input_tokens = 0
    output_tokens = 0
    openrouter_used = False
    for interaction in interactions:
        usage = interaction.get("token_usage") or {}
        input_tokens += int(usage.get("input_tokens") or 0)
        output_tokens += int(usage.get("output_tokens") or 0)
        if usage.get("openrouter_transport") or usage.get("provider_fallback"):
            openrouter_used = True
    cost = (
        input_tokens * INPUT_PRICE_PER_MILLION
        + output_tokens * OUTPUT_PRICE_PER_MILLION
    ) / 1_000_000
    return input_tokens, output_tokens, cost, openrouter_used


def result_is_clean(output_dir: Path) -> bool:
    result_path = output_dir / "experiment_results.json"
    if not result_path.exists():
        return False
    payload = read_json(result_path)
    integrity = payload.get("vote_integrity") or {}
    return not any(
        bool(integrity.get(key))
        for key in ("synthetic_vote_used", "contaminated", "hard_failed")
    )


def direct_openai_env() -> dict[str, str]:
    env = dict(os.environ)
    env["OPENAI_TRANSPORT"] = "direct"
    env["OPENROUTER_PROVIDER_FALLBACK"] = "0"
    for key in list(env):
        if "OPENROUTER" in key:
            env[key] = ""
    env["OPENROUTER_PROVIDER_FALLBACK"] = "0"
    return env


def main() -> int:
    args = parse_args()
    results_root = args.results_root.resolve()
    selection = [
        line.strip()
        for line in args.selection.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ledger: dict[str, Any] = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "budget_usd": float(args.budget_usd),
        "prior_spend_usd": float(args.prior_spend_usd),
        "provider": "openai",
        "model": "gpt-5.4",
        "openrouter_provider_fallback_allowed": False,
        "runs": [],
    }
    spent = float(args.prior_spend_usd)
    write_json(args.ledger, ledger)

    for config_id in selection:
        config = read_json(config_path(results_root, config_id))
        if config.get("provider_route") != "direct-openai":
            raise RuntimeError(f"{config_id} is not configured for direct OpenAI")
        n_agents = int(config["n_agents"])
        reserve = RESERVE_BY_N[n_agents]
        remaining = float(args.budget_usd) - spent
        if remaining < reserve:
            ledger["stop_reason"] = (
                f"Remaining recorded budget ${remaining:.4f} is below the "
                f"${reserve:.2f} reserve for the next N={n_agents} run"
            )
            break

        output_dir = Path(config["output_dir"])
        before_input, before_output, before_cost, before_openrouter = interaction_cost(output_dir)
        if before_openrouter:
            raise RuntimeError(f"{config_id} already contains OpenRouter usage")
        command = [
            str(PROJECT_ROOT / ".venv/bin/python"),
            str(PROJECT_ROOT / "scripts/random_monoculture_control_batch.py"),
            "run-one",
            "--results-root",
            str(results_root),
            "--config-id",
            config_id,
        ]
        started_at = datetime.now().isoformat(timespec="seconds")
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=direct_openai_env(),
            text=True,
        )
        after_input, after_output, after_cost, after_openrouter = interaction_cost(output_dir)
        if after_openrouter:
            raise RuntimeError(f"OpenRouter usage detected in {config_id}; aborting")
        incremental_cost = max(0.0, after_cost - before_cost)
        spent += incremental_cost
        status_path = results_root / "status" / f"{config_id}.json"
        status = read_json(status_path) if status_path.exists() else {}
        row = {
            "config_id": config_id,
            "source_config_id": config.get("source_config_id"),
            "n_agents": n_agents,
            "competition_level": config.get("competition_level"),
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "returncode": int(completed.returncode),
            "status_state": status.get("state"),
            "clean_result": result_is_clean(output_dir),
            "input_tokens": max(0, after_input - before_input),
            "output_tokens": max(0, after_output - before_output),
            "estimated_cost_usd": incremental_cost,
            "cumulative_spend_usd": spent,
            "remaining_recorded_budget_usd": float(args.budget_usd) - spent,
            "openrouter_used": False,
            "output_dir": str(output_dir),
        }
        ledger["runs"].append(row)
        ledger["recorded_spend_usd"] = spent
        write_json(args.ledger, ledger)
        print(json.dumps(row, sort_keys=True), flush=True)

        if spent >= float(args.budget_usd):
            ledger["stop_reason"] = "Recorded spend reached the configured budget"
            break
        if completed.returncode != 0:
            latest_log = status.get("attempt_log_path") or status.get("log_path")
            log_text = ""
            if latest_log and Path(latest_log).exists():
                log_text = Path(latest_log).read_text(encoding="utf-8", errors="replace")[-20000:].lower()
            billing_markers = (
                "insufficient_quota",
                "billing hard limit",
                "insufficient credits",
                "exceeded your current quota",
            )
            if any(marker in log_text for marker in billing_markers):
                ledger["stop_reason"] = "Native OpenAI account budget was exhausted"
                break
            if incremental_cost == 0.0:
                ledger["stop_reason"] = (
                    f"{config_id} failed before any billable model response; stopping for diagnosis"
                )
                break

    ledger["finished_at"] = datetime.now().isoformat(timespec="seconds")
    ledger["recorded_spend_usd"] = spent
    ledger["remaining_recorded_budget_usd"] = float(args.budget_usd) - spent
    ledger.setdefault("stop_reason", "Selection completed")
    write_json(args.ledger, ledger)
    print(json.dumps({
        "ledger": str(args.ledger.resolve()),
        "recorded_spend_usd": spent,
        "remaining_recorded_budget_usd": float(args.budget_usd) - spent,
        "stop_reason": ledger["stop_reason"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Static GPT-5.2 reasoning_effort alias check.

This script instantiates agents with a placeholder OpenAI key if needed. It
does not call the OpenAI API.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from strong_models_experiment.agents import StrongModelAgentFactory  # noqa: E402
from strong_models_experiment.configs import STRONG_MODELS_CONFIG  # noqa: E402

BASELINE_MODEL = "gpt-5-nano"
EFFORT_ALIASES = {
    "low": "gpt-5.2-low",
    "medium": "gpt-5.2-medium",
    "high": "gpt-5.2-high",
    "xhigh": "gpt-5.2-xhigh",
}


async def inspect_order(factory: StrongModelAgentFactory, effort: str, alias: str, model_order: str) -> bool:
    models = [BASELINE_MODEL, alias] if model_order == "weak_first" else [alias, BASELINE_MODEL]
    agents = await factory.create_agents(
        models,
        {
            "model_order": model_order,
            "max_tokens_default": 1000,
        },
    )

    ok = True
    print(f"\n{alias} ({effort}) order={model_order}")
    for model_name, agent in zip(models, agents):
        configured_effort = agent.config.custom_parameters.get("reasoning_effort")
        actual_model_id = getattr(agent.config, "_actual_model_id", None)
        print(
            f"  {agent.agent_id}: alias={model_name} "
            f"model_id={actual_model_id} reasoning_effort={configured_effort}"
        )
        if model_name == alias and configured_effort != effort:
            ok = False
            print(f"    ERROR: expected reasoning_effort={effort}")
        if model_name == BASELINE_MODEL and configured_effort is not None:
            ok = False
            print("    ERROR: baseline should not carry the treatment effort in factory config")
    return ok


async def main() -> int:
    os.environ.setdefault("OPENAI_API_KEY", "debug-placeholder-key")

    ok = True
    for effort, alias in EFFORT_ALIASES.items():
        cfg = STRONG_MODELS_CONFIG.get(alias)
        if cfg is None:
            print(f"ERROR: missing alias {alias}")
            ok = False
            continue
        expected = {
            "model_id": "gpt-5.2-2025-12-11",
            "api_type": "openai",
            "reasoning_effort": effort,
        }
        for key, value in expected.items():
            if cfg.get(key) != value:
                print(f"ERROR: {alias} {key}={cfg.get(key)!r}, expected {value!r}")
                ok = False

    baseline_cfg = STRONG_MODELS_CONFIG.get(BASELINE_MODEL)
    if not baseline_cfg:
        print(f"ERROR: missing baseline {BASELINE_MODEL}")
        ok = False
    else:
        print(f"Baseline: {BASELINE_MODEL} model_id={baseline_cfg.get('model_id')}")

    factory = StrongModelAgentFactory()
    for effort, alias in EFFORT_ALIASES.items():
        for model_order in ("weak_first", "strong_first"):
            ok = await inspect_order(factory, effort, alias, model_order) and ok

    if ok:
        print("\nAll GPT-5.2 effort aliases resolve with provider-native reasoning_effort.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

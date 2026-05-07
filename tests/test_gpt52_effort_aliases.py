import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import strong_models_experiment.agents.agent_factory as agent_factory_module
from strong_models_experiment.agents import StrongModelAgentFactory
from strong_models_experiment.configs import STRONG_MODELS_CONFIG


GPT52_EFFORTS = {
    "gpt-5.2-low": "low",
    "gpt-5.2-medium": "medium",
    "gpt-5.2-high": "high",
    "gpt-5.2-xhigh": "xhigh",
}


class DummyOpenAIAgent:
    def __init__(self, agent_id, config, api_key):
        self.agent_id = agent_id
        self.config = config
        self.api_key = api_key


def test_gpt52_effort_aliases_use_same_snapshot_and_native_effort():
    for alias, effort in GPT52_EFFORTS.items():
        cfg = STRONG_MODELS_CONFIG[alias]
        assert cfg["model_id"] == "gpt-5.2-2025-12-11"
        assert cfg["api_type"] == "openai"
        assert cfg["reasoning_effort"] == effort


def test_agent_factory_preserves_gpt52_efforts_without_budget_override(monkeypatch):
    monkeypatch.setattr(agent_factory_module, "OpenAIAgent", DummyOpenAIAgent)
    monkeypatch.setattr(agent_factory_module, "has_provider_keys", lambda provider, fallback_key=None: True)

    async def create_and_check() -> None:
        factory = StrongModelAgentFactory()
        for alias, effort in GPT52_EFFORTS.items():
            agents = await factory.create_agents(
                ["gpt-5-nano", alias],
                {
                    "model_order": "weak_first",
                    "max_tokens_default": 1000,
                },
            )
            baseline_agent, treatment_agent = agents

            assert baseline_agent.config.custom_parameters.get("reasoning_effort") is None
            assert treatment_agent.config.custom_parameters["reasoning_effort"] == effort

    asyncio.run(create_and_check())

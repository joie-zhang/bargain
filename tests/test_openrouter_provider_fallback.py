import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import negotiation.llm_agents as llm_agents_module
from negotiation.llm_agents import (
    AgentResponse,
    BaseLLMAgent,
    LLMConfig,
    ModelType,
    build_openrouter_fallback_custom_parameters,
    infer_openrouter_fallback_model_id,
)
from negotiation.provider_key_rotation import ProviderKeyExhaustedError
from negotiation.provider_key_rotation import _DISABLED_KEY_LABELS_BY_PROVIDER


class FallbackHarnessAgent(BaseLLMAgent):
    async def _call_llm_api(self, messages, **kwargs):
        raise NotImplementedError

    def get_model_info(self):
        return {}


def clear_provider_env(monkeypatch):
    _DISABLED_KEY_LABELS_BY_PROVIDER.clear()
    for name in list(os.environ):
        if any(
            token in name
            for token in (
                "ANTHROPIC_API_KEY",
                "GOOGLE_API_KEY",
                "OPENAI_API_KEY",
                "OPENROUTER_API_KEY",
            )
        ):
            monkeypatch.delenv(name, raising=False)


def test_openrouter_fallback_route_inference_for_native_providers():
    assert (
        infer_openrouter_fallback_model_id("openai", "gpt-5.2-2025-12-11")
        == "openai/gpt-5.2"
    )
    assert (
        infer_openrouter_fallback_model_id("openai", "gpt-5.2-chat-latest")
        == "openai/gpt-5.2-chat"
    )
    assert (
        infer_openrouter_fallback_model_id("anthropic", "claude-opus-4-6")
        == "anthropic/claude-opus-4.6"
    )
    assert (
        infer_openrouter_fallback_model_id("anthropic", "claude-sonnet-4-5-20250929")
        == "anthropic/claude-sonnet-4.5"
    )
    assert (
        infer_openrouter_fallback_model_id("google", "gemini-2.0-flash")
        == "google/gemini-2.0-flash-001"
    )


def test_openrouter_fallback_translates_native_reasoning_controls():
    assert build_openrouter_fallback_custom_parameters(
        "openai",
        {"reasoning_effort": "xhigh"},
    ) == {"reasoning": {"effort": "xhigh", "exclude": True}}

    assert build_openrouter_fallback_custom_parameters(
        "anthropic",
        {"thinking_budget_tokens": 32000},
    ) == {"reasoning": {"max_tokens": 32000, "exclude": True}}


def test_runtime_provider_failure_uses_openrouter_last_resort(monkeypatch):
    import negotiation.openrouter_client as openrouter_client_module

    created_agents = []

    class FakeOpenRouterAgent:
        def __init__(
            self,
            agent_id,
            llm_config,
            api_key,
            model_id=None,
            key_pool=None,
            rotate_unclassified_failures=False,
        ):
            self.agent_id = agent_id
            self.llm_config = llm_config
            self.api_key = api_key
            self.model_id = model_id
            self.key_pool = key_pool
            self.rotate_unclassified_failures = rotate_unclassified_failures
            self.closed = False
            created_agents.append(self)

        async def _call_llm_api(self, messages, **kwargs):
            return AgentResponse(
                content="fallback ok",
                model_used=self.model_id,
                response_time=0.01,
                tokens_used=3,
                metadata={"usage": {"total_tokens": 3}},
            )

        async def close(self):
            self.closed = True

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    monkeypatch.setattr(
        llm_agents_module,
        "has_provider_keys",
        lambda provider, fallback_key=None: provider == "openrouter",
    )
    monkeypatch.setattr(openrouter_client_module, "OpenRouterAgent", FakeOpenRouterAgent)

    agent = FallbackHarnessAgent(
        "Agent_1",
        LLMConfig(
            model_type=ModelType.GPT_5,
            max_tokens=64,
            custom_parameters={"reasoning_effort": "xhigh"},
        ),
    )
    response = asyncio.run(
        agent._call_openrouter_last_resort(
            messages=[{"role": "user", "content": "hi"}],
            source_provider="openai",
            source_model="gpt-5.2-2025-12-11",
            source_error=ProviderKeyExhaustedError("all native keys failed"),
        )
    )

    assert response.content == "fallback ok"
    assert response.metadata["provider_fallback"]["used"] is True
    assert response.metadata["provider_fallback"]["from_provider"] == "openai"
    assert response.metadata["provider_fallback"]["to_model"] == "openai/gpt-5.2"
    assert created_agents[0].model_id == "openai/gpt-5.2"
    assert created_agents[0].rotate_unclassified_failures is True
    assert created_agents[0].llm_config.custom_parameters == {
        "reasoning": {"effort": "xhigh", "exclude": True}
    }
    assert created_agents[0].closed is True


def test_openai_route_interleaves_native_and_openrouter_by_group(monkeypatch, tmp_path):
    import negotiation.openrouter_client as openrouter_client_module

    clear_provider_env(monkeypatch)
    monkeypatch.setenv("LLM_KEY_GROUP_ORDER", "LEWIS,JOIE")
    monkeypatch.setenv("LEWIS_OPENAI_API_KEY_1", "sk-lewis-openai")
    monkeypatch.setenv("JOIE_OPENAI_API_KEY_1", "sk-joie-openai")
    monkeypatch.setenv("LEWIS_OPENROUTER_API_KEY_1", "sk-or-v1-lewis")
    monkeypatch.setenv("JOIE_OPENROUTER_API_KEY_1", "sk-or-v1-joie")
    monkeypatch.setenv("LLM_FAILURE_REPORT_PATH", str(tmp_path / "provider_failures.md"))
    attempts = []

    class FakeOpenRouterAgent:
        def __init__(
            self,
            agent_id,
            llm_config,
            api_key,
            model_id=None,
            key_pool=None,
            rotate_unclassified_failures=False,
        ):
            self.key_pool = key_pool
            self.model_id = model_id
            self.rotate_unclassified_failures = rotate_unclassified_failures

        async def _call_llm_api(self, messages, **kwargs):
            label = self.key_pool.labels()[0]
            attempts.append(("openrouter", label))
            if label == "LEWIS_OPENROUTER_API_KEY_1":
                raise RuntimeError("opaque Lewis OpenRouter failure")
            return AgentResponse(
                content="openrouter joie ok",
                model_used=self.model_id,
                response_time=0.01,
            )

        async def close(self):
            pass

    monkeypatch.setattr(openrouter_client_module, "OpenRouterAgent", FakeOpenRouterAgent)
    agent = FallbackHarnessAgent("Agent_1", LLMConfig(model_type=ModelType.GPT_5))

    async def native_request(key):
        attempts.append(("openai", key.label))
        raise RuntimeError("opaque native provider failure")

    response = asyncio.run(
        agent._call_provider_route_sequence(
            source_provider="openai",
            source_model="gpt-5.2-2025-12-11",
            messages=[{"role": "user", "content": "hi"}],
            native_request_coro_factory=native_request,
        )
    )

    assert response.content == "openrouter joie ok"
    assert attempts == [
        ("openai", "LEWIS_OPENAI_API_KEY_1"),
        ("openrouter", "LEWIS_OPENROUTER_API_KEY_1"),
        ("openai", "JOIE_OPENAI_API_KEY_1"),
        ("openrouter", "JOIE_OPENROUTER_API_KEY_1"),
    ]


def test_google_route_uses_lewis_then_polaris_before_openrouter(monkeypatch, tmp_path):
    clear_provider_env(monkeypatch)
    monkeypatch.setenv("LLM_KEY_GROUP_ORDER", "LEWIS,POLARIS,JOIE")
    monkeypatch.setenv("LEWIS_GOOGLE_API_KEY_1", "lewis-google")
    monkeypatch.setenv("POLARIS_GOOGLE_API_KEY_1", "polaris-google-1")
    monkeypatch.setenv("POLARIS_GOOGLE_API_KEY_2", "polaris-google-2")
    monkeypatch.setenv("LEWIS_OPENROUTER_API_KEY_1", "sk-or-v1-lewis")
    monkeypatch.setenv("JOIE_OPENROUTER_API_KEY_1", "sk-or-v1-joie")
    monkeypatch.setenv("LLM_FAILURE_REPORT_PATH", str(tmp_path / "provider_failures.md"))
    attempts = []
    agent = FallbackHarnessAgent("Agent_1", LLMConfig(model_type=ModelType.GPT_4))

    async def native_request(key):
        attempts.append(("google", key.label))
        if key.label != "POLARIS_GOOGLE_API_KEY_2":
            raise RuntimeError("ResourceExhausted: 429 quota exceeded")
        return AgentResponse(
            content="polaris 2 ok",
            model_used="gemini-2.0-flash",
            response_time=0.01,
        )

    response = asyncio.run(
        agent._call_provider_route_sequence(
            source_provider="google",
            source_model="gemini-2.0-flash",
            messages=[{"role": "user", "content": "hi"}],
            native_request_coro_factory=native_request,
        )
    )

    assert response.content == "polaris 2 ok"
    assert attempts == [
        ("google", "LEWIS_GOOGLE_API_KEY_1"),
        ("google", "POLARIS_GOOGLE_API_KEY_1"),
        ("google", "POLARIS_GOOGLE_API_KEY_2"),
    ]


def test_factory_uses_openrouter_when_native_provider_keys_are_missing(monkeypatch):
    import strong_models_experiment.agents.agent_factory as agent_factory_module
    from strong_models_experiment.agents import StrongModelAgentFactory

    class DummyOpenRouterAgent:
        def __init__(self, agent_id, llm_config, api_key, model_id=None):
            self.agent_id = agent_id
            self.llm_config = llm_config
            self.api_key = api_key
            self.model_id = model_id

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    monkeypatch.setattr(agent_factory_module, "OpenRouterAgent", DummyOpenRouterAgent)
    monkeypatch.setattr(
        agent_factory_module,
        "has_provider_keys",
        lambda provider, fallback_key=None: provider == "openrouter",
    )

    async def create_agent():
        factory = StrongModelAgentFactory()
        agents = await factory.create_agents(
            ["gpt-5.2-high"],
            {"max_tokens_default": 128},
        )
        return agents[0]

    agent = asyncio.run(create_agent())

    assert isinstance(agent, DummyOpenRouterAgent)
    assert agent.model_id == "openai/gpt-5.2"
    assert agent.llm_config.custom_parameters["model_id"] == "openai/gpt-5.2"
    assert agent.llm_config.custom_parameters["reasoning"] == {
        "effort": "high",
        "exclude": True,
    }

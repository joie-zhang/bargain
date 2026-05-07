import asyncio
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import negotiation.llm_agents as llm_agents_module
from negotiation.llm_agents import (
    AgentResponse,
    BaseLLMAgent,
    LLMConfig,
    ModelType,
    build_openrouter_fallback_custom_parameters,
    infer_openrouter_fallback_model_id,
    openrouter_fallback_unsupported_reason,
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

    assert openrouter_fallback_unsupported_reason(
        "anthropic",
        {"extra_body": {"output_config": {"effort": "max"}}},
    )


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


def test_openai_route_uses_all_native_groups_before_openrouter(monkeypatch, tmp_path):
    import negotiation.openrouter_client as openrouter_client_module

    clear_provider_env(monkeypatch)
    monkeypatch.setenv("LLM_KEY_GROUP_ORDER", "PRIMARY,SECONDARY")
    monkeypatch.setenv("PRIMARY_OPENAI_API_KEY_1", "sk-primary-openai")
    monkeypatch.setenv("SECONDARY_OPENAI_API_KEY_1", "sk-secondary-openai")
    monkeypatch.setenv("PRIMARY_OPENROUTER_API_KEY_1", "sk-or-v1-primary")
    monkeypatch.setenv("SECONDARY_OPENROUTER_API_KEY_1", "sk-or-v1-secondary")
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
            if label == "PRIMARY_OPENROUTER_API_KEY_1":
                raise RuntimeError("opaque Primary OpenRouter failure")
            return AgentResponse(
                content="openrouter secondary ok",
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

    assert response.content == "openrouter secondary ok"
    assert attempts == [
        ("openai", "PRIMARY_OPENAI_API_KEY_1"),
        ("openai", "SECONDARY_OPENAI_API_KEY_1"),
        ("openrouter", "PRIMARY_OPENROUTER_API_KEY_1"),
        ("openrouter", "SECONDARY_OPENROUTER_API_KEY_1"),
    ]


def test_anthropic_workspace_limit_tries_secondary_native_before_primary_openrouter(monkeypatch, tmp_path):
    import negotiation.openrouter_client as openrouter_client_module

    clear_provider_env(monkeypatch)
    monkeypatch.setenv("LLM_KEY_GROUP_ORDER", "PRIMARY,SECONDARY")
    monkeypatch.setenv("PRIMARY_ANTHROPIC_API_KEY_1", "sk-ant-primary")
    monkeypatch.setenv("SECONDARY_ANTHROPIC_API_KEY_1", "sk-ant-secondary")
    monkeypatch.setenv("PRIMARY_OPENROUTER_API_KEY_1", "sk-or-v1-primary")
    monkeypatch.setenv("SECONDARY_OPENROUTER_API_KEY_1", "sk-or-v1-secondary")
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
            return AgentResponse(
                content="primary openrouter ok",
                model_used=self.model_id,
                response_time=0.01,
                metadata={"provider": "openrouter"},
            )

        async def close(self):
            pass

    monkeypatch.setattr(openrouter_client_module, "OpenRouterAgent", FakeOpenRouterAgent)
    agent = FallbackHarnessAgent("Agent_1", LLMConfig(model_type=ModelType.CLAUDE_3_5_SONNET))

    async def native_request(key):
        attempts.append(("anthropic", key.label))
        raise RuntimeError(
            "Error code: 400 - {'type': 'error', 'error': "
            "{'type': 'invalid_request_error', 'message': "
            "'You have reached your specified workspace API usage limits. "
            "You will regain access on 2026-06-01 at 00:00 UTC.'}}"
        )

    response = asyncio.run(
        agent._call_provider_route_sequence(
            source_provider="anthropic",
            source_model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "hi"}],
            native_request_coro_factory=native_request,
        )
    )

    assert response.content == "primary openrouter ok"
    assert response.metadata["provider_fallback"]["used"] is True
    assert response.metadata["provider_fallback"]["from_provider"] == "anthropic"
    assert response.metadata["provider_fallback"]["to_provider"] == "openrouter"
    assert attempts == [
        ("anthropic", "PRIMARY_ANTHROPIC_API_KEY_1"),
        ("anthropic", "SECONDARY_ANTHROPIC_API_KEY_1"),
        ("openrouter", "PRIMARY_OPENROUTER_API_KEY_1"),
    ]


def test_anthropic_max_effort_skips_openrouter_and_uses_secondary_native(monkeypatch, tmp_path):
    import negotiation.openrouter_client as openrouter_client_module

    clear_provider_env(monkeypatch)
    monkeypatch.setenv("LLM_KEY_GROUP_ORDER", "PRIMARY,SECONDARY")
    monkeypatch.setenv("PRIMARY_ANTHROPIC_API_KEY_1", "sk-ant-primary")
    monkeypatch.setenv("SECONDARY_ANTHROPIC_API_KEY_1", "sk-ant-secondary")
    monkeypatch.setenv("PRIMARY_OPENROUTER_API_KEY_1", "sk-or-v1-primary")
    monkeypatch.setenv("SECONDARY_OPENROUTER_API_KEY_1", "sk-or-v1-secondary")
    monkeypatch.setenv("LLM_FAILURE_REPORT_PATH", str(tmp_path / "provider_failures.md"))
    attempts = []

    class FakeOpenRouterAgent:
        def __init__(self, *args, **kwargs):
            raise AssertionError("OpenRouter fallback should be skipped for Anthropic effort=max")

    monkeypatch.setattr(openrouter_client_module, "OpenRouterAgent", FakeOpenRouterAgent)
    agent = FallbackHarnessAgent(
        "Agent_1",
        LLMConfig(
            model_type=ModelType.CLAUDE_3_5_SONNET,
            custom_parameters={
                "thinking": {"type": "adaptive"},
                "extra_body": {"output_config": {"effort": "max"}},
            },
        ),
    )

    async def native_request(key):
        attempts.append(("anthropic", key.label))
        if key.label == "PRIMARY_ANTHROPIC_API_KEY_1":
            raise RuntimeError("workspace API usage limits")
        return AgentResponse(content="secondary native ok", model_used="claude-sonnet-4-6", response_time=0.01)

    response = asyncio.run(
        agent._call_provider_route_sequence(
            source_provider="anthropic",
            source_model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hi"}],
            native_request_coro_factory=native_request,
        )
    )

    assert response.content == "secondary native ok"
    assert attempts == [
        ("anthropic", "PRIMARY_ANTHROPIC_API_KEY_1"),
        ("anthropic", "SECONDARY_ANTHROPIC_API_KEY_1"),
    ]


def test_anthropic_max_effort_does_not_fall_through_to_openrouter(monkeypatch, tmp_path):
    import negotiation.openrouter_client as openrouter_client_module

    clear_provider_env(monkeypatch)
    monkeypatch.setenv("LLM_KEY_GROUP_ORDER", "SECONDARY")
    monkeypatch.setenv("SECONDARY_ANTHROPIC_API_KEY_1", "sk-ant-secondary")
    monkeypatch.setenv("SECONDARY_OPENROUTER_API_KEY_1", "sk-or-v1-secondary")
    monkeypatch.setenv("LLM_FAILURE_REPORT_PATH", str(tmp_path / "provider_failures.md"))

    class FakeOpenRouterAgent:
        def __init__(self, *args, **kwargs):
            raise AssertionError("OpenRouter fallback should not be used for Anthropic effort=max")

    monkeypatch.setattr(openrouter_client_module, "OpenRouterAgent", FakeOpenRouterAgent)
    agent = FallbackHarnessAgent(
        "Agent_1",
        LLMConfig(
            model_type=ModelType.CLAUDE_3_5_SONNET,
            custom_parameters={
                "thinking": {"type": "adaptive"},
                "extra_body": {"output_config": {"effort": "max"}},
            },
        ),
    )

    async def native_request(key):
        raise RuntimeError("Anthropic returned empty content (stop_reason=max_tokens, output_tokens=10500)")

    with pytest.raises(RuntimeError, match="stop_reason=max_tokens"):
        asyncio.run(
            agent._call_provider_route_sequence(
                source_provider="anthropic",
                source_model="claude-sonnet-4-6",
                messages=[{"role": "user", "content": "hi"}],
                native_request_coro_factory=native_request,
            )
        )


def test_google_route_uses_primary_then_group_a_before_openrouter(monkeypatch, tmp_path):
    clear_provider_env(monkeypatch)
    monkeypatch.setenv("LLM_KEY_GROUP_ORDER", "PRIMARY,GROUP_A,SECONDARY")
    monkeypatch.setenv("PRIMARY_GOOGLE_API_KEY_1", "primary-google")
    monkeypatch.setenv("GROUP_A_GOOGLE_API_KEY_1", "group_a-google-1")
    monkeypatch.setenv("GROUP_A_GOOGLE_API_KEY_2", "group_a-google-2")
    monkeypatch.setenv("PRIMARY_OPENROUTER_API_KEY_1", "sk-or-v1-primary")
    monkeypatch.setenv("SECONDARY_OPENROUTER_API_KEY_1", "sk-or-v1-secondary")
    monkeypatch.setenv("LLM_FAILURE_REPORT_PATH", str(tmp_path / "provider_failures.md"))
    attempts = []
    agent = FallbackHarnessAgent("Agent_1", LLMConfig(model_type=ModelType.GPT_4))

    async def native_request(key):
        attempts.append(("google", key.label))
        if key.label != "GROUP_A_GOOGLE_API_KEY_2":
            raise RuntimeError("ResourceExhausted: 429 quota exceeded")
        return AgentResponse(
            content="group_a 2 ok",
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

    assert response.content == "group_a 2 ok"
    assert attempts == [
        ("google", "PRIMARY_GOOGLE_API_KEY_1"),
        ("google", "GROUP_A_GOOGLE_API_KEY_1"),
        ("google", "GROUP_A_GOOGLE_API_KEY_2"),
    ]


def test_google_route_uses_secondary_native_before_openrouter(monkeypatch, tmp_path):
    import negotiation.openrouter_client as openrouter_client_module

    clear_provider_env(monkeypatch)
    monkeypatch.setenv("LLM_KEY_GROUP_ORDER", "PRIMARY,GROUP_A,SECONDARY")
    monkeypatch.setenv("PRIMARY_GOOGLE_API_KEY_1", "primary-google")
    monkeypatch.setenv("GROUP_A_GOOGLE_API_KEY_1", "group_a-google-1")
    monkeypatch.setenv("GROUP_A_GOOGLE_API_KEY_2", "group_a-google-2")
    monkeypatch.setenv("SECONDARY_GOOGLE_API_KEY_1", "secondary-google")
    monkeypatch.setenv("PRIMARY_OPENROUTER_API_KEY_1", "sk-or-v1-primary")
    monkeypatch.setenv("SECONDARY_OPENROUTER_API_KEY_1", "sk-or-v1-secondary")
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
            return AgentResponse(
                content="primary openrouter ok",
                model_used=self.model_id,
                response_time=0.01,
            )

        async def close(self):
            pass

    monkeypatch.setattr(openrouter_client_module, "OpenRouterAgent", FakeOpenRouterAgent)
    agent = FallbackHarnessAgent("Agent_1", LLMConfig(model_type=ModelType.GPT_4))

    async def native_request(key):
        attempts.append(("google", key.label))
        raise RuntimeError("ResourceExhausted: 429 quota exceeded")

    response = asyncio.run(
        agent._call_provider_route_sequence(
            source_provider="google",
            source_model="gemini-2.0-flash",
            messages=[{"role": "user", "content": "hi"}],
            native_request_coro_factory=native_request,
        )
    )

    assert response.content == "primary openrouter ok"
    assert attempts == [
        ("google", "PRIMARY_GOOGLE_API_KEY_1"),
        ("google", "GROUP_A_GOOGLE_API_KEY_1"),
        ("google", "GROUP_A_GOOGLE_API_KEY_2"),
        ("google", "SECONDARY_GOOGLE_API_KEY_1"),
        ("openrouter", "PRIMARY_OPENROUTER_API_KEY_1"),
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

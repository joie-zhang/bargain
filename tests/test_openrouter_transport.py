import asyncio

import pytest

from negotiation.llm_agents import LLMConfig, ModelType
from negotiation.openrouter_client import OpenRouterAgent, ProxyMonitorUnavailableError


def make_agent(monkeypatch, transport=None):
    if transport is None:
        monkeypatch.delenv("OPENROUTER_TRANSPORT", raising=False)
    else:
        monkeypatch.setenv("OPENROUTER_TRANSPORT", transport)
    return OpenRouterAgent(
        agent_id="test-agent",
        llm_config=LLMConfig(model_type=ModelType.GEMMA_2_9B, max_tokens=64),
        api_key="sk-or-v1-test-key",
        model_id="google/gemma-2-9b-it",
    )


def test_default_transport_is_auto(monkeypatch):
    agent = make_agent(monkeypatch)
    assert agent.openrouter_config.transport == "auto"


def test_auto_mode_tries_proxy_before_direct(monkeypatch):
    agent = make_agent(monkeypatch, transport="auto")
    calls = []

    async def fake_proxy(url, headers, payload, timeout):
        calls.append("proxy")
        raise ProxyMonitorUnavailableError("proxy monitor unavailable")

    async def fake_direct(url, headers, payload, timeout):
        calls.append("direct")
        return "ok", None, {"total_tokens": 1}

    monkeypatch.setattr(agent, "_send_request_via_proxy", fake_proxy)
    monkeypatch.setattr(agent, "_send_request_direct", fake_direct)

    result, usage = asyncio.run(
        agent._make_request([{"role": "user", "content": "hi"}])
    )

    assert result == "ok"
    assert usage == {"total_tokens": 1}
    assert calls == ["proxy", "direct"]


def test_proxy_mode_does_not_fall_back_to_direct(monkeypatch):
    agent = make_agent(monkeypatch, transport="proxy")

    async def fake_proxy(url, headers, payload, timeout):
        raise ProxyMonitorUnavailableError("proxy monitor unavailable")

    async def fake_direct(url, headers, payload, timeout):
        raise AssertionError("direct fallback should not be used in proxy mode")

    monkeypatch.setattr(agent, "_send_request_via_proxy", fake_proxy)
    monkeypatch.setattr(agent, "_send_request_direct", fake_direct)

    with pytest.raises(ProxyMonitorUnavailableError):
        asyncio.run(agent._make_request([{"role": "user", "content": "hi"}]))

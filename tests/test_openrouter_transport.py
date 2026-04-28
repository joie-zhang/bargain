import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from negotiation.llm_agents import LLMConfig, ModelType, BaseLLMAgent, NonRetryableLLMError
from negotiation.openrouter_client import (
    DEFAULT_OPENROUTER_MAX_TOKENS_CAP,
    OpenRouterAgent,
    ProxyMonitorUnavailableError,
    get_openrouter_max_tokens_cap,
)


def make_agent(monkeypatch, transport=None, slurm_job_id=None, max_tokens=64):
    if transport is None:
        monkeypatch.delenv("OPENROUTER_TRANSPORT", raising=False)
    else:
        monkeypatch.setenv("OPENROUTER_TRANSPORT", transport)

    if slurm_job_id is None:
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    else:
        monkeypatch.setenv("SLURM_JOB_ID", slurm_job_id)

    return OpenRouterAgent(
        agent_id="test-agent",
        llm_config=LLMConfig(model_type=ModelType.GEMMA_2_9B, max_tokens=max_tokens),
        api_key="sk-or-v1-test-key",
        model_id="google/gemma-2-9b-it",
    )


def test_default_transport_resolves_to_direct_outside_slurm(monkeypatch):
    agent = make_agent(monkeypatch)
    assert agent.requested_transport == "auto"
    assert agent.openrouter_config.transport == "direct"


def test_auto_transport_resolves_to_proxy_in_slurm(monkeypatch):
    agent = make_agent(monkeypatch, transport="auto", slurm_job_id="12345")
    assert agent.requested_transport == "auto"
    assert agent.openrouter_config.transport == "proxy"


def test_auto_transport_in_slurm_does_not_fall_back_to_direct(monkeypatch):
    agent = make_agent(monkeypatch, transport="auto", slurm_job_id="12345")

    async def fake_proxy(url, headers, payload, timeout):
        raise ProxyMonitorUnavailableError("proxy monitor unavailable")

    async def fake_direct(url, headers, payload, timeout):
        raise AssertionError("direct fallback should not be used in Slurm proxy mode")

    monkeypatch.setattr(agent, "_send_request_via_proxy", fake_proxy)
    monkeypatch.setattr(agent, "_send_request_direct", fake_direct)

    with pytest.raises(ProxyMonitorUnavailableError):
        asyncio.run(agent._make_request([{"role": "user", "content": "hi"}]))


def test_auto_transport_outside_slurm_uses_direct_only(monkeypatch):
    agent = make_agent(monkeypatch, transport="auto")
    calls = []

    async def fake_proxy(url, headers, payload, timeout):
        calls.append("proxy")
        raise AssertionError("proxy should not be used outside Slurm when auto resolves to direct")

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
    assert calls == ["direct"]


def test_openrouter_max_tokens_cap_defaults_to_raised_limit(monkeypatch):
    monkeypatch.delenv("OPENROUTER_MAX_TOKENS_CAP", raising=False)

    assert DEFAULT_OPENROUTER_MAX_TOKENS_CAP == 10000
    assert get_openrouter_max_tokens_cap() == DEFAULT_OPENROUTER_MAX_TOKENS_CAP


def test_openrouter_max_tokens_cap_is_configurable(monkeypatch):
    monkeypatch.setenv("OPENROUTER_MAX_TOKENS_CAP", "8192")

    assert get_openrouter_max_tokens_cap() == 8192


def test_openrouter_request_applies_configurable_max_tokens_cap(monkeypatch):
    monkeypatch.setenv("OPENROUTER_MAX_TOKENS_CAP", "8192")
    agent = make_agent(monkeypatch, transport="direct", max_tokens=999999)
    payloads = []

    async def fake_direct(url, headers, payload, timeout):
        payloads.append(payload)
        return "ok", None, {"total_tokens": 1}

    monkeypatch.setattr(agent, "_send_request_direct", fake_direct)

    result, usage = asyncio.run(
        agent._make_request([{"role": "user", "content": "hi"}])
    )

    assert result == "ok"
    assert usage == {"total_tokens": 1}
    assert payloads[0]["max_tokens"] == 8192


def test_openrouter_request_preserves_lower_explicit_max_tokens(monkeypatch):
    monkeypatch.setenv("OPENROUTER_MAX_TOKENS_CAP", "8192")
    agent = make_agent(monkeypatch, transport="direct", max_tokens=512)
    payloads = []

    async def fake_direct(url, headers, payload, timeout):
        payloads.append(payload)
        return "ok", None, {"total_tokens": 1}

    monkeypatch.setattr(agent, "_send_request_direct", fake_direct)

    asyncio.run(agent._make_request([{"role": "user", "content": "hi"}]))

    assert payloads[0]["max_tokens"] == 512


def test_proxy_provider_error_is_non_retryable(monkeypatch):
    agent = make_agent(monkeypatch, transport="proxy", slurm_job_id="12345")

    async def fake_proxy(url, headers, payload, timeout):
        return None, 'Exception: HTTP 400: {"error":{"message":"Provider returned error","code":400,"metadata":{"raw":"{\\"message\\":\\"The provided model identifier is invalid.\\"}"}}}', None

    monkeypatch.setattr(agent, "_send_request_via_proxy", fake_proxy)

    with pytest.raises(NonRetryableLLMError):
        asyncio.run(agent._make_request([{"role": "user", "content": "hi"}]))


def test_proxy_output_limit_error_is_non_retryable(monkeypatch):
    agent = make_agent(monkeypatch, transport="proxy", slurm_job_id="12345")

    async def fake_proxy(url, headers, payload, timeout):
        return None, "Exception: Empty content from model. finish_reason=length", None

    monkeypatch.setattr(agent, "_send_request_via_proxy", fake_proxy)

    with pytest.raises(NonRetryableLLMError):
        asyncio.run(agent._make_request([{"role": "user", "content": "hi"}]))


def test_proxy_waits_for_complete_response_file(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_PROXY_POLL_DIR", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_PROXY_CLIENT_POLL_INTERVAL", "0.01")
    monkeypatch.setenv("OPENROUTER_PROXY_CLIENT_TIMEOUT", "2")
    agent = make_agent(monkeypatch, transport="proxy", slurm_job_id="12345")

    async def writer():
        request_path = None
        for _ in range(100):
            matches = list(tmp_path.glob("request_*.json"))
            if matches:
                request_path = matches[0]
                break
            await asyncio.sleep(0.01)
        assert request_path is not None
        suffix = request_path.stem.removeprefix("request_")
        response_path = tmp_path / f"response_{suffix}.json"
        response_path.write_text("{")
        await asyncio.sleep(0.05)
        response_path.write_text('{"result":"ok","error":null,"usage":{"total_tokens":1}}')

    async def run_test():
        writer_task = asyncio.create_task(writer())
        result = await agent._make_request([{"role": "user", "content": "hi"}])
        await writer_task
        return result

    result, usage = asyncio.run(run_test())
    assert result == "ok"
    assert usage == {"total_tokens": 1}


class DummyNoRetryAgent(BaseLLMAgent):
    def __init__(self):
        super().__init__(
            "dummy",
            LLMConfig(model_type=ModelType.GEMMA_2_9B, max_tokens=64, max_retries=12),
        )
        self.calls = 0

    async def _call_llm_api(self, messages, **kwargs):
        self.calls += 1
        raise NonRetryableLLMError("invalid provider/model")

    def get_model_info(self):
        return {}

    def _build_context_messages(self, context, prompt):
        return [{"role": "user", "content": prompt}]


def test_generate_response_does_not_retry_non_retryable_errors():
    agent = DummyNoRetryAgent()

    with pytest.raises(NonRetryableLLMError):
        asyncio.run(agent.generate_response(context=None, prompt="hi"))

    assert agent.calls == 1

import asyncio
import logging
from types import SimpleNamespace

import negotiation.openrouter_client as openrouter_client
from anthropic.types import MessageDeltaUsage
from negotiation.llm_agents import (
    AgentResponse,
    BaseLLMAgent,
    _extract_anthropic_thinking_tokens,
)
from negotiation.openrouter_client import (
    OpenRouterAgent,
    normalize_openrouter_usage,
)
from negotiation.openrouter_proxy_monitor import build_success_response


def test_claude_nested_thinking_count_is_authoritative_and_zero_is_preserved():
    usage = MessageDeltaUsage(
        output_tokens=25,
        output_tokens_details={"thinking_tokens": 0},
        thinking_tokens=999,
    )

    count, source = _extract_anthropic_thinking_tokens(usage)

    assert count == 0
    assert source == "output_tokens_details.thinking_tokens"


def test_claude_adaptive_thinking_omission_is_recorded_as_zero():
    usage = MessageDeltaUsage(output_tokens=25)

    count, source = _extract_anthropic_thinking_tokens(
        usage,
        thinking_enabled=True,
    )

    assert count == 0
    assert source == "thinking_enabled_without_reported_detail"


def test_saved_token_usage_keeps_claude_nested_details_and_source():
    response = AgentResponse(
        content="answer",
        model_used="claude-sonnet-4-6",
        response_time=0.1,
        tokens_used=125,
        metadata={
            "usage": {
                "input_tokens": 100,
                "output_tokens": 25,
                "total_tokens": 125,
                "output_tokens_details": {"thinking_tokens": 0},
                "reasoning_token_source": (
                    "output_tokens_details.thinking_tokens"
                ),
                "reasoning_tokens": 999,
                "thinking_tokens": 998,
                "output_tokens_includes_reasoning": True,
            }
        },
    )

    usage = BaseLLMAgent._extract_token_usage_from_response(response)

    assert usage["reasoning_tokens"] == 0
    assert usage["thinking_tokens"] == 0
    assert usage["output_tokens_details"] == {"thinking_tokens": 0}
    assert (
        usage["reasoning_token_source"]
        == "output_tokens_details.thinking_tokens"
    )
    assert usage["output_tokens_includes_reasoning"] is True


def test_openrouter_nested_reasoning_overrides_flat_compatibility_value():
    raw_usage = {
        "prompt_tokens": 100,
        "completion_tokens": 700,
        "total_tokens": 800,
        "reasoning_tokens": 999,
        "completion_tokens_details": {
            "reasoning_tokens": 512,
            "accepted_prediction_tokens": 3,
        },
    }

    normalized = normalize_openrouter_usage(raw_usage)

    assert normalized["reasoning_tokens"] == 512
    assert normalized["thinking_tokens"] == 512
    assert normalized["completion_tokens_details"] == {
        "reasoning_tokens": 512,
        "accepted_prediction_tokens": 3,
    }
    assert (
        normalized["reasoning_token_source"]
        == "completion_tokens_details.reasoning_tokens"
    )
    assert normalized["output_tokens_includes_reasoning"] is True


def test_openrouter_nested_zero_reasoning_is_not_treated_as_missing():
    normalized = normalize_openrouter_usage(
        {
            "completion_tokens": 42,
            "completion_tokens_details": {"reasoning_tokens": 0},
        }
    )

    assert normalized["reasoning_tokens"] == 0
    assert normalized["thinking_tokens"] == 0
    assert (
        normalized["reasoning_token_source"]
        == "completion_tokens_details.reasoning_tokens"
    )


def test_file_proxy_preserves_complete_openrouter_usage_object():
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 700,
        "total_tokens": 800,
        "completion_tokens_details": {"reasoning_tokens": 512},
        "cost": 0.00123,
        "cost_details": {"upstream_inference_cost": 0.001},
    }

    response = build_success_response("answer", usage)

    assert response["usage"] == usage
    assert response["usage"] is not usage


def test_auto_openrouter_transport_is_direct_first_with_latched_proxy_backup():
    agent = object.__new__(OpenRouterAgent)
    agent.requested_transport = "auto"
    agent._auto_proxy_fallback_active = False

    assert agent._transport_order() == ("direct", "proxy")

    agent._auto_proxy_fallback_active = True
    assert agent._transport_order() == ("proxy",)


def test_openrouter_aiohttp_session_honors_proxy_environment():
    agent = object.__new__(OpenRouterAgent)
    agent.session = None

    async def check_session():
        await agent._ensure_session()
        try:
            assert agent.session is not None
            assert agent.session.trust_env is True
        finally:
            await agent.close()

    asyncio.run(check_session())


def test_auto_openrouter_request_falls_back_and_records_proxy_transport(
    monkeypatch,
):
    agent = object.__new__(OpenRouterAgent)
    agent.model_id = "google/gemini-3-flash-preview"
    agent.requested_transport = "auto"
    agent._auto_proxy_fallback_active = False
    agent._last_transport_used = None
    agent.logger = logging.getLogger("test-openrouter-transport")
    agent.llm_config = SimpleNamespace(
        temperature=0.0,
        max_tokens=100,
        custom_parameters={},
    )
    agent.openrouter_config = SimpleNamespace(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        timeout=1.0,
        site_url=None,
        site_name=None,
    )
    agent.key_pool = object()
    agent.rotate_unclassified_failures = False
    calls = []

    async def fake_direct(*args):
        calls.append("direct")
        raise OSError("simulated connection failure")

    async def fake_proxy(*args):
        calls.append("proxy")
        return (
            "answer",
            None,
            {
                "completion_tokens": 12,
                "completion_tokens_details": {"reasoning_tokens": 7},
            },
        )

    async def fake_key_rotation(**kwargs):
        return await kwargs["request_coro_factory"](
            SimpleNamespace(value="test-key")
        )

    agent._send_request_direct = fake_direct
    agent._send_request_via_proxy = fake_proxy
    monkeypatch.setattr(
        openrouter_client,
        "call_with_key_rotation",
        fake_key_rotation,
    )
    monkeypatch.setattr(
        openrouter_client,
        "get_openrouter_max_tokens_cap",
        lambda logger, model_id: 100,
    )

    content, usage = asyncio.run(
        agent._make_request([{"role": "user", "content": "test"}])
    )

    assert content == "answer"
    assert calls == ["direct", "proxy"]
    assert usage["openrouter_transport"] == "proxy"
    assert agent._last_transport_used == "proxy"
    assert agent._auto_proxy_fallback_active is True

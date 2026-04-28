import asyncio
import sys
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from negotiation.provider_key_rotation import (
    ProviderKeyExhaustedError,
    ProviderKeyPool,
    ProviderTransientRetryExhaustedError,
    _DISABLED_KEY_LABELS_BY_PROVIDER,
    call_with_key_rotation,
    discover_provider_keys,
    is_deterministic_provider_failure,
)


def clear_rotation_state():
    _DISABLED_KEY_LABELS_BY_PROVIDER.clear()


def test_discovers_grouped_keys_before_legacy(monkeypatch):
    clear_rotation_state()
    monkeypatch.setenv("LLM_KEY_GROUP_ORDER", "LEWIS,JOIE")
    monkeypatch.setenv("LEWIS_GOOGLE_API_KEY_2", "lewis-2")
    monkeypatch.setenv("LEWIS_GOOGLE_API_KEY_1", "lewis-1")
    monkeypatch.setenv("JOIE_GOOGLE_API_KEY_1", "joie-1")
    monkeypatch.setenv("GOOGLE_API_KEY", "legacy")

    keys = discover_provider_keys("google")

    assert [key.label for key in keys] == [
        "LEWIS_GOOGLE_API_KEY_1",
        "LEWIS_GOOGLE_API_KEY_2",
        "JOIE_GOOGLE_API_KEY_1",
        "GOOGLE_API_KEY",
    ]
    assert [key.value for key in keys] == ["lewis-1", "lewis-2", "joie-1", "legacy"]


def test_legacy_key_is_supported_without_group_order(monkeypatch):
    clear_rotation_state()
    monkeypatch.delenv("LLM_KEY_GROUP_ORDER", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-openai")

    keys = discover_provider_keys("openai")

    assert len(keys) == 1
    assert keys[0].label == "OPENAI_API_KEY"
    assert keys[0].value == "legacy-openai"


def test_rate_limit_rotates_immediately_and_writes_report(monkeypatch, tmp_path):
    clear_rotation_state()
    monkeypatch.setenv("LLM_KEY_GROUP_ORDER", "LEWIS,JOIE")
    monkeypatch.setenv("LEWIS_GOOGLE_API_KEY_1", "lewis")
    monkeypatch.setenv("JOIE_GOOGLE_API_KEY_1", "joie")
    monkeypatch.setenv("LLM_FAILURE_REPORT_PATH", str(tmp_path / "provider_failures.md"))
    sleep_calls = []
    attempts = []

    async def fake_sleep(seconds):
        sleep_calls.append(seconds)

    async def request(key):
        attempts.append(key.label)
        if key.label == "LEWIS_GOOGLE_API_KEY_1":
            raise RuntimeError("ResourceExhausted: 429 quota exceeded for generate_requests_per_model_per_day")
        return key.label

    result = asyncio.run(
        call_with_key_rotation(
            provider="google",
            model="gemini-2.5-pro",
            key_pool=ProviderKeyPool("google"),
            request_coro_factory=request,
            sleep_func=fake_sleep,
        )
    )

    assert result == "JOIE_GOOGLE_API_KEY_1"
    assert attempts == ["LEWIS_GOOGLE_API_KEY_1", "JOIE_GOOGLE_API_KEY_1"]
    assert sleep_calls == []
    report = tmp_path / "provider_failures.md"
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "LEWIS_GOOGLE_API_KEY_1" in text
    assert "auto-rotated-to-JOIE_GOOGLE_API_KEY_1" in text


def test_all_keys_exhausted_fails_current_call(monkeypatch, tmp_path):
    clear_rotation_state()
    monkeypatch.setenv("LLM_KEY_GROUP_ORDER", "LEWIS")
    monkeypatch.setenv("LEWIS_ANTHROPIC_API_KEY_1", "lewis")
    monkeypatch.setenv("LLM_FAILURE_REPORT_PATH", str(tmp_path / "provider_failures.md"))

    async def request(key):
        raise RuntimeError("HTTP 429: rate limit exceeded")

    with pytest.raises(ProviderKeyExhaustedError, match="All configured anthropic API keys failed"):
        asyncio.run(
            call_with_key_rotation(
                provider="anthropic",
                model="claude-sonnet-4-20250514",
                key_pool=ProviderKeyPool("anthropic"),
                request_coro_factory=request,
            )
        )

    text = (tmp_path / "provider_failures.md").read_text(encoding="utf-8")
    assert "all-keys-exhausted" in text
    assert "requeue after provider quota reset" in text


def test_transient_error_retries_same_key(monkeypatch):
    clear_rotation_state()
    monkeypatch.setenv("LLM_KEY_GROUP_ORDER", "LEWIS")
    monkeypatch.setenv("LEWIS_OPENROUTER_API_KEY_1", "sk-or-v1-lewis")
    sleep_calls = []
    attempts = []

    async def fake_sleep(seconds):
        sleep_calls.append(seconds)

    async def request(key):
        attempts.append(key.label)
        if len(attempts) == 1:
            raise RuntimeError("HTTP 503: service unavailable")
        return "ok"

    result = asyncio.run(
        call_with_key_rotation(
            provider="openrouter",
            model="google/gemini-2.5-pro",
            key_pool=ProviderKeyPool("openrouter"),
            request_coro_factory=request,
            sleep_func=fake_sleep,
        )
    )

    assert result == "ok"
    assert attempts == ["LEWIS_OPENROUTER_API_KEY_1", "LEWIS_OPENROUTER_API_KEY_1"]
    assert len(sleep_calls) == 1


def test_transient_budget_exhaustion_fails_without_rotation(monkeypatch):
    clear_rotation_state()
    monkeypatch.setenv("LLM_KEY_GROUP_ORDER", "LEWIS")
    monkeypatch.setenv("LEWIS_OPENAI_API_KEY_1", "sk-lewis")
    monkeypatch.setenv("LLM_TRANSIENT_RETRY_SECONDS", "0")

    async def request(key):
        raise RuntimeError("HTTP 503: service unavailable")

    with pytest.raises(ProviderTransientRetryExhaustedError):
        asyncio.run(
            call_with_key_rotation(
                provider="openai",
                model="gpt-5-nano",
                key_pool=ProviderKeyPool("openai"),
                request_coro_factory=request,
            )
        )


def test_deterministic_provider_failure_catches_output_limit_messages():
    assert is_deterministic_provider_failure(
        RuntimeError(
            "Error code: 400 - {'error': {'type': 'invalid_request_error', "
            "'message': '`max_tokens` must be greater than `thinking.budget_tokens`.'}}"
        )
    )
    assert is_deterministic_provider_failure(
        RuntimeError(
            "Could not finish the message because max_tokens or model output limit was reached."
        )
    )
    assert is_deterministic_provider_failure(
        RuntimeError("Empty content from model. finish_reason=length")
    )


def test_google_configure_and_generate_are_serialized(monkeypatch):
    clear_rotation_state()
    monkeypatch.setenv("LLM_KEY_GROUP_ORDER", "TEST")
    monkeypatch.setenv("TEST_GOOGLE_API_KEY_1", "test-google-key")

    active = {"count": 0, "max": 0}

    class FakeGenerationConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakePart:
        text = "ok"

    class FakeContent:
        parts = [FakePart()]

    class FakeFinishReason:
        name = "STOP"
        value = 1

    class FakeCandidate:
        finish_reason = FakeFinishReason()
        safety_ratings = []
        content = FakeContent()

    class FakeResponse:
        candidates = [FakeCandidate()]
        usage_metadata = None

    class FakeModel:
        def __init__(self, model_name):
            self.model_name = model_name

        def generate_content(self, *args, **kwargs):
            active["count"] += 1
            active["max"] = max(active["max"], active["count"])
            time.sleep(0.02)
            active["count"] -= 1
            return FakeResponse()

    fake_genai = types.ModuleType("google.generativeai")
    fake_genai.configure = lambda api_key: None
    fake_genai.GenerativeModel = FakeModel
    fake_genai.types = types.SimpleNamespace(GenerationConfig=FakeGenerationConfig)
    google_pkg = types.ModuleType("google")
    google_pkg.generativeai = fake_genai
    monkeypatch.setitem(sys.modules, "google", google_pkg)
    monkeypatch.setitem(sys.modules, "google.generativeai", fake_genai)

    from negotiation.llm_agents import GoogleAgent, LLMConfig, ModelType, NegotiationContext

    agent_a = GoogleAgent("Agent_1", LLMConfig(model_type=ModelType.GPT_4, max_tokens=32), None)
    agent_b = GoogleAgent("Agent_2", LLMConfig(model_type=ModelType.GPT_4, max_tokens=32), None)
    context = NegotiationContext(
        current_round=1,
        max_rounds=1,
        items=[{"name": "Apple"}],
        agents=["Agent_1", "Agent_2"],
        agent_id="Agent_1",
        preferences=[1.0],
    )

    async def run_calls():
        await asyncio.gather(
            agent_a.generate_response(context, "hi"),
            agent_b.generate_response(context, "hi"),
        )

    asyncio.run(run_calls())

    assert active["max"] == 1

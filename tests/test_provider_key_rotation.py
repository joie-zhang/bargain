import asyncio
import sys
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from negotiation.llm_agents import NonRetryableLLMError
from negotiation.provider_key_rotation import (
    ProviderKeyExhaustedError,
    ProviderKeyPool,
    ProviderTransientRetryExhaustedError,
    _DISABLED_KEY_LABELS_BY_PROVIDER,
    call_with_key_rotation,
    classify_key_scoped_failure,
    discover_provider_keys,
    is_deterministic_provider_failure,
    is_retryable_model_output_failure,
    is_upstream_provider_rate_limit,
)


def clear_rotation_state():
    _DISABLED_KEY_LABELS_BY_PROVIDER.clear()


def test_discovers_grouped_keys_before_legacy(monkeypatch):
    clear_rotation_state()
    monkeypatch.setenv("LLM_KEY_GROUP_ORDER", "PRIMARY,SECONDARY")
    monkeypatch.setenv("PRIMARY_GOOGLE_API_KEY_2", "primary-2")
    monkeypatch.setenv("PRIMARY_GOOGLE_API_KEY_1", "primary-1")
    monkeypatch.setenv("SECONDARY_GOOGLE_API_KEY_1", "secondary-1")
    monkeypatch.setenv("GOOGLE_API_KEY", "legacy")

    keys = discover_provider_keys("google")

    assert [key.label for key in keys] == [
        "PRIMARY_GOOGLE_API_KEY_1",
        "PRIMARY_GOOGLE_API_KEY_2",
        "SECONDARY_GOOGLE_API_KEY_1",
        "GOOGLE_API_KEY",
    ]
    assert [key.value for key in keys] == ["primary-1", "primary-2", "secondary-1", "legacy"]


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
    monkeypatch.setenv("LLM_KEY_GROUP_ORDER", "PRIMARY,SECONDARY")
    monkeypatch.setenv("PRIMARY_GOOGLE_API_KEY_1", "primary")
    monkeypatch.setenv("SECONDARY_GOOGLE_API_KEY_1", "secondary")
    monkeypatch.setenv("LLM_FAILURE_REPORT_PATH", str(tmp_path / "provider_failures.md"))
    sleep_calls = []
    attempts = []

    async def fake_sleep(seconds):
        sleep_calls.append(seconds)

    async def request(key):
        attempts.append(key.label)
        if key.label == "PRIMARY_GOOGLE_API_KEY_1":
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

    assert result == "SECONDARY_GOOGLE_API_KEY_1"
    assert attempts == ["PRIMARY_GOOGLE_API_KEY_1", "SECONDARY_GOOGLE_API_KEY_1"]
    assert sleep_calls == []
    report = tmp_path / "provider_failures.md"
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "PRIMARY_GOOGLE_API_KEY_1" in text
    assert "auto-rotated-to-SECONDARY_GOOGLE_API_KEY_1" in text


def test_all_keys_exhausted_fails_current_call(monkeypatch, tmp_path):
    clear_rotation_state()
    monkeypatch.setenv("LLM_KEY_GROUP_ORDER", "PRIMARY")
    monkeypatch.setenv("PRIMARY_ANTHROPIC_API_KEY_1", "primary")
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


def test_transient_error_retries_same_key(monkeypatch, tmp_path):
    clear_rotation_state()
    monkeypatch.setenv("LLM_KEY_GROUP_ORDER", "PRIMARY,SECONDARY")
    monkeypatch.setenv("PRIMARY_OPENROUTER_API_KEY_1", "sk-or-v1-primary")
    monkeypatch.setenv("SECONDARY_OPENROUTER_API_KEY_1", "sk-or-v1-secondary")
    monkeypatch.setenv("LLM_FAILURE_REPORT_PATH", str(tmp_path / "provider_failures.md"))
    monkeypatch.setenv("LLM_TRANSIENT_RETRY_SECONDS", "30")
    sleep_calls = []
    attempts = []

    async def fake_sleep(seconds):
        sleep_calls.append(seconds)

    async def request(key):
        attempts.append(key.label)
        if len(attempts) == 1:
            raise RuntimeError("HTTP 503: service unavailable")
        return "ok"

    assert classify_key_scoped_failure("openrouter", RuntimeError("HTTP 503: service unavailable")) is None

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
    assert attempts == ["PRIMARY_OPENROUTER_API_KEY_1", "PRIMARY_OPENROUTER_API_KEY_1"]
    assert len(sleep_calls) == 1
    assert _DISABLED_KEY_LABELS_BY_PROVIDER.get("openrouter", set()) == set()
    assert not (tmp_path / "provider_failures.md").exists()


def test_openrouter_nonretryable_insufficient_credits_rotates_to_secondary(monkeypatch, tmp_path):
    clear_rotation_state()
    monkeypatch.setenv("LLM_KEY_GROUP_ORDER", "PRIMARY,SECONDARY")
    monkeypatch.setenv("PRIMARY_OPENROUTER_API_KEY_1", "sk-or-v1-primary")
    monkeypatch.setenv("SECONDARY_OPENROUTER_API_KEY_1", "sk-or-v1-secondary")
    monkeypatch.setenv("LLM_FAILURE_REPORT_PATH", str(tmp_path / "provider_failures.md"))
    attempts = []

    error = NonRetryableLLMError(
        'Exception: HTTP 402: {"error":{"message":"Insufficient credits. '
        'Add more using https://openrouter.ai/settings/credits","code":402}}'
    )

    assert classify_key_scoped_failure("openrouter", error) == "insufficient_funds"

    async def request(key):
        attempts.append(key.label)
        if key.label == "PRIMARY_OPENROUTER_API_KEY_1":
            raise error
        return "ok"

    result = asyncio.run(
        call_with_key_rotation(
            provider="openrouter",
            model="deepseek/deepseek-r1-0528",
            key_pool=ProviderKeyPool("openrouter"),
            request_coro_factory=request,
        )
    )

    assert result == "ok"
    assert attempts == ["PRIMARY_OPENROUTER_API_KEY_1", "SECONDARY_OPENROUTER_API_KEY_1"]
    text = (tmp_path / "provider_failures.md").read_text(encoding="utf-8")
    assert "insufficient_funds" in text
    assert "auto-rotated-to-SECONDARY_OPENROUTER_API_KEY_1" in text


def test_openrouter_nonretryable_key_limit_rotates_to_secondary(monkeypatch, tmp_path):
    clear_rotation_state()
    monkeypatch.setenv("LLM_KEY_GROUP_ORDER", "PRIMARY,SECONDARY")
    monkeypatch.setenv("PRIMARY_OPENROUTER_API_KEY_1", "sk-or-v1-primary")
    monkeypatch.setenv("SECONDARY_OPENROUTER_API_KEY_1", "sk-or-v1-secondary")
    monkeypatch.setenv("LLM_FAILURE_REPORT_PATH", str(tmp_path / "provider_failures.md"))
    attempts = []

    error = NonRetryableLLMError(
        'Exception: HTTP 403: {"error":{"message":"Key limit exceeded '
        '(total limit). Manage it using https://openrouter.ai/settings/keys","code":403}}'
    )

    assert classify_key_scoped_failure("openrouter", error) == "rate_limit_or_quota"

    async def request(key):
        attempts.append(key.label)
        if key.label == "PRIMARY_OPENROUTER_API_KEY_1":
            raise error
        return "ok"

    result = asyncio.run(
        call_with_key_rotation(
            provider="openrouter",
            model="amazon/nova-micro-v1",
            key_pool=ProviderKeyPool("openrouter"),
            request_coro_factory=request,
        )
    )

    assert result == "ok"
    assert attempts == ["PRIMARY_OPENROUTER_API_KEY_1", "SECONDARY_OPENROUTER_API_KEY_1"]
    text = (tmp_path / "provider_failures.md").read_text(encoding="utf-8")
    assert "rate_limit_or_quota" in text
    assert "auto-rotated-to-SECONDARY_OPENROUTER_API_KEY_1" in text


def test_openrouter_nonretryable_user_not_found_rotates_as_invalid_key(monkeypatch, tmp_path):
    clear_rotation_state()
    monkeypatch.setenv("LLM_KEY_GROUP_ORDER", "PRIMARY,SECONDARY")
    monkeypatch.setenv("PRIMARY_OPENROUTER_API_KEY_1", "sk-or-v1-primary")
    monkeypatch.setenv("SECONDARY_OPENROUTER_API_KEY_1", "sk-or-v1-secondary")
    monkeypatch.setenv("LLM_FAILURE_REPORT_PATH", str(tmp_path / "provider_failures.md"))
    attempts = []

    error = NonRetryableLLMError(
        'Exception: HTTP 401: {"error":{"message":"User not found.","code":401}}'
    )

    assert classify_key_scoped_failure("openrouter", error) == "invalid_api_key"

    async def request(key):
        attempts.append(key.label)
        if key.label == "PRIMARY_OPENROUTER_API_KEY_1":
            raise error
        return "ok"

    result = asyncio.run(
        call_with_key_rotation(
            provider="openrouter",
            model="qwen/qwen3-max",
            key_pool=ProviderKeyPool("openrouter"),
            request_coro_factory=request,
        )
    )

    assert result == "ok"
    assert attempts == ["PRIMARY_OPENROUTER_API_KEY_1", "SECONDARY_OPENROUTER_API_KEY_1"]
    text = (tmp_path / "provider_failures.md").read_text(encoding="utf-8")
    assert "invalid_api_key" in text
    assert "auto-rotated-to-SECONDARY_OPENROUTER_API_KEY_1" in text


def test_openrouter_deepseek_upstream_429_retries_same_key(monkeypatch, tmp_path):
    clear_rotation_state()
    monkeypatch.setenv("LLM_KEY_GROUP_ORDER", "PRIMARY,SECONDARY")
    monkeypatch.setenv("PRIMARY_OPENROUTER_API_KEY_1", "sk-or-v1-primary")
    monkeypatch.setenv("SECONDARY_OPENROUTER_API_KEY_1", "sk-or-v1-secondary")
    monkeypatch.setenv("LLM_FAILURE_REPORT_PATH", str(tmp_path / "provider_failures.md"))
    monkeypatch.setenv("LLM_TRANSIENT_RETRY_SECONDS", "30")
    attempts = []
    sleep_calls = []

    error = NonRetryableLLMError(
        'Exception: HTTP 429: {"error":{"message":"Provider returned error",'
        '"code":429,"metadata":{"raw":"deepseek/deepseek-chat is temporarily '
        'rate-limited upstream. Please retry shortly, or add your own key to '
        'accumulate your rate limits","provider_name":"DeepInfra","is_byok":false}}}'
    )

    assert classify_key_scoped_failure("openrouter", error) == "rate_limit_or_quota"
    assert is_upstream_provider_rate_limit("openrouter", "deepseek/deepseek-chat", error)

    async def fake_sleep(seconds):
        sleep_calls.append(seconds)

    async def request(key):
        attempts.append(key.label)
        if len(attempts) == 1:
            raise error
        return "ok"

    result = asyncio.run(
        call_with_key_rotation(
            provider="openrouter",
            model="deepseek/deepseek-chat",
            key_pool=ProviderKeyPool("openrouter"),
            request_coro_factory=request,
            sleep_func=fake_sleep,
        )
    )

    assert result == "ok"
    assert attempts == ["PRIMARY_OPENROUTER_API_KEY_1", "PRIMARY_OPENROUTER_API_KEY_1"]
    assert len(sleep_calls) == 1
    assert _DISABLED_KEY_LABELS_BY_PROVIDER.get("openrouter", set()) == set()
    assert not (tmp_path / "provider_failures.md").exists()


def test_openrouter_qwen_high_demand_429_retries_same_key(monkeypatch, tmp_path):
    clear_rotation_state()
    monkeypatch.setenv("LLM_KEY_GROUP_ORDER", "PRIMARY,SECONDARY")
    monkeypatch.setenv("PRIMARY_OPENROUTER_API_KEY_1", "sk-or-v1-primary")
    monkeypatch.setenv("SECONDARY_OPENROUTER_API_KEY_1", "sk-or-v1-secondary")
    monkeypatch.setenv("LLM_FAILURE_REPORT_PATH", str(tmp_path / "provider_failures.md"))
    monkeypatch.setenv("LLM_TRANSIENT_RETRY_SECONDS", "30")
    attempts = []

    error = NonRetryableLLMError(
        'Exception: HTTP 429: {"error":{"message":"Rate limit exceeded: '
        'limit_rpm/qwen/qwen3-max/example. High demand for qwen/qwen3-max '
        'on OpenRouter - limited to 20 requests per minute. Please retry '
        'shortly.","code":429}}'
    )

    assert classify_key_scoped_failure("openrouter", error) == "rate_limit_or_quota"
    assert is_upstream_provider_rate_limit("openrouter", "qwen/qwen3-max", error)

    async def fake_sleep(seconds):
        pass

    async def request(key):
        attempts.append(key.label)
        if len(attempts) == 1:
            raise error
        return "ok"

    result = asyncio.run(
        call_with_key_rotation(
            provider="openrouter",
            model="qwen/qwen3-max",
            key_pool=ProviderKeyPool("openrouter"),
            request_coro_factory=request,
            sleep_func=fake_sleep,
        )
    )

    assert result == "ok"
    assert attempts == ["PRIMARY_OPENROUTER_API_KEY_1", "PRIMARY_OPENROUTER_API_KEY_1"]
    assert _DISABLED_KEY_LABELS_BY_PROVIDER.get("openrouter", set()) == set()
    assert not (tmp_path / "provider_failures.md").exists()


def test_openrouter_deepseek_upstream_429_exhausts_retry_budget_without_disabling_keys(
    monkeypatch,
    tmp_path,
):
    clear_rotation_state()
    monkeypatch.setenv("LLM_KEY_GROUP_ORDER", "PRIMARY,SECONDARY")
    monkeypatch.setenv("PRIMARY_OPENROUTER_API_KEY_1", "sk-or-v1-primary")
    monkeypatch.setenv("SECONDARY_OPENROUTER_API_KEY_1", "sk-or-v1-secondary")
    monkeypatch.setenv("LLM_FAILURE_REPORT_PATH", str(tmp_path / "provider_failures.md"))
    monkeypatch.setenv("LLM_TRANSIENT_RETRY_SECONDS", "0")
    attempts = []

    error = NonRetryableLLMError(
        'Exception: HTTP 429: {"error":{"message":"Provider returned error",'
        '"code":429,"metadata":{"raw":"deepseek/deepseek-chat is temporarily '
        'rate-limited upstream. Please retry shortly","provider_name":"DeepInfra"}}}'
    )

    async def request(key):
        attempts.append(key.label)
        raise error

    with pytest.raises(ProviderTransientRetryExhaustedError):
        asyncio.run(
            call_with_key_rotation(
                provider="openrouter",
                model="deepseek/deepseek-chat",
                key_pool=ProviderKeyPool("openrouter"),
                request_coro_factory=request,
            )
        )

    assert attempts == ["PRIMARY_OPENROUTER_API_KEY_1"]
    assert _DISABLED_KEY_LABELS_BY_PROVIDER.get("openrouter", set()) == set()
    text = (tmp_path / "provider_failures.md").read_text(encoding="utf-8")
    assert "upstream_rate_limit_retry_exhausted" in text
    assert "SECONDARY_OPENROUTER_API_KEY_1" not in text


def test_openrouter_nonretryable_invalid_model_still_fails_fast(monkeypatch, tmp_path):
    clear_rotation_state()
    monkeypatch.setenv("LLM_KEY_GROUP_ORDER", "PRIMARY,SECONDARY")
    monkeypatch.setenv("PRIMARY_OPENROUTER_API_KEY_1", "sk-or-v1-primary")
    monkeypatch.setenv("SECONDARY_OPENROUTER_API_KEY_1", "sk-or-v1-secondary")
    monkeypatch.setenv("LLM_FAILURE_REPORT_PATH", str(tmp_path / "provider_failures.md"))
    attempts = []

    async def request(key):
        attempts.append(key.label)
        raise NonRetryableLLMError(
            'Exception: HTTP 400: {"error":{"message":"The provided model '
            'identifier is invalid.","code":400}}'
        )

    with pytest.raises(NonRetryableLLMError, match="provided model identifier is invalid"):
        asyncio.run(
            call_with_key_rotation(
                provider="openrouter",
                model="bad/model",
                key_pool=ProviderKeyPool("openrouter"),
                request_coro_factory=request,
            )
        )

    assert attempts == ["PRIMARY_OPENROUTER_API_KEY_1"]
    assert not (tmp_path / "provider_failures.md").exists()


def test_transient_error_exhausts_retry_budget_without_disabling_key(monkeypatch, tmp_path):
    clear_rotation_state()
    monkeypatch.setenv("LLM_KEY_GROUP_ORDER", "PRIMARY")
    monkeypatch.setenv("PRIMARY_OPENAI_API_KEY_1", "sk-primary")
    monkeypatch.setenv("LLM_TRANSIENT_RETRY_SECONDS", "0")
    monkeypatch.setenv("LLM_FAILURE_REPORT_PATH", str(tmp_path / "provider_failures.md"))

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
    text = (tmp_path / "provider_failures.md").read_text(encoding="utf-8")
    assert "provider_transient_retry_exhausted" in text
    assert "all-keys-exhausted" not in text
    assert _DISABLED_KEY_LABELS_BY_PROVIDER.get("openai", set()) == set()


def test_anthropic_workspace_usage_limit_rotates_to_secondary(monkeypatch, tmp_path):
    clear_rotation_state()
    monkeypatch.setenv("LLM_KEY_GROUP_ORDER", "PRIMARY,SECONDARY")
    monkeypatch.setenv("PRIMARY_ANTHROPIC_API_KEY_1", "sk-ant-primary")
    monkeypatch.setenv("SECONDARY_ANTHROPIC_API_KEY_1", "sk-ant-secondary")
    monkeypatch.setenv("LLM_FAILURE_REPORT_PATH", str(tmp_path / "provider_failures.md"))
    attempts = []

    workspace_limit_error = RuntimeError(
        "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
        "'message': 'You have reached your specified workspace API usage limits. "
        "You will regain access on 2026-05-01 at 00:00 UTC.'}}"
    )

    assert classify_key_scoped_failure("anthropic", workspace_limit_error) == "rate_limit_or_quota"

    async def request(key):
        attempts.append(key.label)
        if key.label == "PRIMARY_ANTHROPIC_API_KEY_1":
            raise workspace_limit_error
        return "ok"

    result = asyncio.run(
        call_with_key_rotation(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            key_pool=ProviderKeyPool("anthropic"),
            request_coro_factory=request,
        )
    )

    assert result == "ok"
    assert attempts == ["PRIMARY_ANTHROPIC_API_KEY_1", "SECONDARY_ANTHROPIC_API_KEY_1"]
    text = (tmp_path / "provider_failures.md").read_text(encoding="utf-8")
    assert "PRIMARY_ANTHROPIC_API_KEY_1" in text
    assert "auto-rotated-to-SECONDARY_ANTHROPIC_API_KEY_1" in text


def test_anthropic_empty_max_tokens_is_not_key_scoped_transient():
    error = RuntimeError(
        "Anthropic returned empty content (stop_reason=max_tokens, output_tokens=10500)"
    )

    assert classify_key_scoped_failure("anthropic", error) is None
    assert not is_retryable_model_output_failure(error)
    assert is_deterministic_provider_failure(error)


def test_provider_connection_error_retries_same_openai_key(monkeypatch, tmp_path):
    clear_rotation_state()
    monkeypatch.setenv("LLM_KEY_GROUP_ORDER", "PRIMARY,SECONDARY")
    monkeypatch.setenv("PRIMARY_OPENAI_API_KEY_1", "sk-primary")
    monkeypatch.setenv("SECONDARY_OPENAI_API_KEY_1", "sk-secondary")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_FAILURE_REPORT_PATH", str(tmp_path / "provider_failures.md"))
    attempts = []

    connection_error = RuntimeError(
        "openai.APIConnectionError: Connection error. "
        "Server disconnected without sending a response."
    )

    assert classify_key_scoped_failure("openai", connection_error) is None

    async def request(key):
        attempts.append(key.label)
        if len(attempts) == 1:
            raise connection_error
        return "ok"

    result = asyncio.run(
        call_with_key_rotation(
            provider="openai",
            model="gpt-5.4-high",
            key_pool=ProviderKeyPool("openai"),
            request_coro_factory=request,
        )
    )

    assert result == "ok"
    assert attempts == ["PRIMARY_OPENAI_API_KEY_1", "PRIMARY_OPENAI_API_KEY_1"]
    assert _DISABLED_KEY_LABELS_BY_PROVIDER.get("openai", set()) == set()
    assert not (tmp_path / "provider_failures.md").exists()


def test_rotate_unclassified_failures_exhausts_all_openai_keys(monkeypatch, tmp_path):
    clear_rotation_state()
    monkeypatch.setenv("LLM_KEY_GROUP_ORDER", "PRIMARY,SECONDARY")
    monkeypatch.setenv("PRIMARY_OPENAI_API_KEY_1", "sk-primary")
    monkeypatch.setenv("SECONDARY_OPENAI_API_KEY_1", "sk-secondary")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_FAILURE_REPORT_PATH", str(tmp_path / "provider_failures.md"))
    attempts = []

    async def request(key):
        attempts.append(key.label)
        raise RuntimeError("unexpected provider failure without a recognizable status")

    with pytest.raises(ProviderKeyExhaustedError):
        asyncio.run(
            call_with_key_rotation(
                provider="openai",
                model="gpt-5.2-2025-12-11",
                key_pool=ProviderKeyPool("openai"),
                request_coro_factory=request,
                rotate_unclassified_failures=True,
            )
        )

    assert attempts == ["PRIMARY_OPENAI_API_KEY_1", "SECONDARY_OPENAI_API_KEY_1"]
    text = (tmp_path / "provider_failures.md").read_text(encoding="utf-8")
    assert "unclassified_provider_error" in text
    assert "auto-rotated-to-SECONDARY_OPENAI_API_KEY_1" in text
    assert "all-keys-exhausted" in text


def test_unclassified_failures_do_not_rotate_by_default(monkeypatch, tmp_path):
    clear_rotation_state()
    monkeypatch.setenv("LLM_KEY_GROUP_ORDER", "PRIMARY,SECONDARY")
    monkeypatch.setenv("PRIMARY_OPENROUTER_API_KEY_1", "sk-or-v1-primary")
    monkeypatch.setenv("SECONDARY_OPENROUTER_API_KEY_1", "sk-or-v1-secondary")
    monkeypatch.setenv("LLM_FAILURE_REPORT_PATH", str(tmp_path / "provider_failures.md"))
    attempts = []

    async def request(key):
        attempts.append(key.label)
        raise RuntimeError("unexpected local client bug")

    with pytest.raises(RuntimeError, match="unexpected local client bug"):
        asyncio.run(
            call_with_key_rotation(
                provider="openrouter",
                model="openai/gpt-5.2",
                key_pool=ProviderKeyPool("openrouter"),
                request_coro_factory=request,
            )
        )

    assert attempts == ["PRIMARY_OPENROUTER_API_KEY_1"]
    assert not (tmp_path / "provider_failures.md").exists()


def test_empty_model_output_retries_same_key_without_rotating(monkeypatch, tmp_path):
    clear_rotation_state()
    monkeypatch.setenv("LLM_KEY_GROUP_ORDER", "PRIMARY,SECONDARY")
    monkeypatch.setenv("PRIMARY_OPENROUTER_API_KEY_1", "sk-or-v1-primary")
    monkeypatch.setenv("SECONDARY_OPENROUTER_API_KEY_1", "sk-or-v1-secondary")
    monkeypatch.setenv("LLM_FAILURE_REPORT_PATH", str(tmp_path / "provider_failures.md"))
    monkeypatch.setenv("LLM_TRANSIENT_RETRY_SECONDS", "30")
    sleep_calls = []
    attempts = []

    async def fake_sleep(seconds):
        sleep_calls.append(seconds)

    async def request(key):
        attempts.append(key.label)
        if len(attempts) == 1:
            raise RuntimeError(
                "Empty content from model. finish_reason=stop, "
                "message keys=['role', 'content']"
            )
        return "ok"

    result = asyncio.run(
        call_with_key_rotation(
            provider="openrouter",
            model="amazon/nova-micro-v1",
            key_pool=ProviderKeyPool("openrouter"),
            request_coro_factory=request,
            sleep_func=fake_sleep,
        )
    )

    assert result == "ok"
    assert attempts == ["PRIMARY_OPENROUTER_API_KEY_1", "PRIMARY_OPENROUTER_API_KEY_1"]
    assert len(sleep_calls) == 1
    assert not (tmp_path / "provider_failures.md").exists()


def test_empty_model_output_retry_exhaustion_does_not_disable_next_key(monkeypatch, tmp_path):
    clear_rotation_state()
    monkeypatch.setenv("LLM_KEY_GROUP_ORDER", "PRIMARY,SECONDARY")
    monkeypatch.setenv("PRIMARY_OPENROUTER_API_KEY_1", "sk-or-v1-primary")
    monkeypatch.setenv("SECONDARY_OPENROUTER_API_KEY_1", "sk-or-v1-secondary")
    monkeypatch.setenv("LLM_FAILURE_REPORT_PATH", str(tmp_path / "provider_failures.md"))
    monkeypatch.setenv("LLM_TRANSIENT_RETRY_SECONDS", "0")
    attempts = []

    output_error = RuntimeError("Empty content from model. finish_reason=stop")
    assert classify_key_scoped_failure("openrouter", output_error) is None
    assert is_retryable_model_output_failure(output_error)

    async def request(key):
        attempts.append(key.label)
        raise output_error

    with pytest.raises(ProviderTransientRetryExhaustedError):
        asyncio.run(
            call_with_key_rotation(
                provider="openrouter",
                model="amazon/nova-micro-v1",
                key_pool=ProviderKeyPool("openrouter"),
                request_coro_factory=request,
            )
        )

    assert attempts == ["PRIMARY_OPENROUTER_API_KEY_1"]
    assert _DISABLED_KEY_LABELS_BY_PROVIDER.get("openrouter", set()) == set()
    text = (tmp_path / "provider_failures.md").read_text(encoding="utf-8")
    assert "provider_output_retry_exhausted" in text
    assert "SECONDARY_OPENROUTER_API_KEY_1" not in text


def test_empty_model_output_with_token_counts_is_not_key_scoped():
    output_error = RuntimeError(
        'Exception: Empty content from model. finish_reason=None, '
        'full response: {"choices": [{"finish_reason": null, '
        '"message": {"content": null}}], "usage": {"prompt_tokens": 41068, '
        '"completion_tokens": 0, "total_tokens": 41068}}'
    )

    assert classify_key_scoped_failure("openrouter", output_error) is None
    assert is_retryable_model_output_failure(output_error)


def test_local_disk_quota_is_not_provider_key_scoped():
    error = OSError("[Errno 122] Disk quota exceeded")

    assert classify_key_scoped_failure("openrouter", error) is None
    assert is_deterministic_provider_failure(error)


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
    assert not is_retryable_model_output_failure(
        RuntimeError("Empty content from model. finish_reason=length")
    )
    assert is_deterministic_provider_failure(
        RuntimeError("Response blocked by safety filter")
    )


def test_google_configure_and_generate_are_serialized(monkeypatch):
    clear_rotation_state()
    monkeypatch.setenv("LLM_KEY_GROUP_ORDER", "TEST")
    monkeypatch.setenv("TEST_GOOGLE_API_KEY_1", "test-google-key")

    active = {"count": 0, "max": 0}
    generation_configs = []

    class FakeGenerationConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            generation_configs.append(kwargs)

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
    assert [config.get("max_output_tokens") for config in generation_configs] == [32, 32]

    generation_configs.clear()
    unlimited_agent = GoogleAgent(
        "Agent_3",
        LLMConfig(model_type=ModelType.GPT_4, max_tokens=999999),
        None,
    )
    asyncio.run(unlimited_agent.generate_response(context, "hi"))
    assert generation_configs == [{"temperature": 0.7}]

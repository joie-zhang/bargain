"""Tests for prompt-local context compaction preflight."""

import pytest

from negotiation.context_compaction import (
    ContextWindowPreflightError,
    compact_public_history_entries,
    estimate_text_tokens,
    reserved_output_tokens,
    resolve_context_limit,
)
from negotiation.llm_agents import AgentResponse, BaseLLMAgent, LLMConfig, ModelType, NegotiationContext
from strong_models_experiment.phases.phase_handlers import PhaseHandler


class DummyContextAgent(BaseLLMAgent):
    """Minimal model wrapper for prompt-building tests."""

    async def _call_llm_api(self, messages, **kwargs):
        raise NotImplementedError("API calls are not used in this test")

    def get_model_info(self):
        return {"model_name": "amazon-nova-micro-v1.0"}


class DummyDeepSeekContextAgent(DummyContextAgent):
    def get_model_info(self):
        return {"model_name": "deepseek/deepseek-chat"}


def _context(history):
    return NegotiationContext(
        current_round=4,
        max_rounds=10,
        items=[{"name": "Item 1"}],
        agents=["Agent_1", "Agent_2"],
        agent_id="Agent_1",
        preferences={"Item 1": 1.0},
        turn_type="discussion",
        conversation_history=history,
        strategic_notes=[],
    )


def test_resolve_context_limit_uses_canonical_metadata_and_hard_caps():
    assert resolve_context_limit(["amazon-nova-micro-v1.0"]) == 128_000
    assert resolve_context_limit(["amazon/nova-micro-v1"]) == 128_000
    assert resolve_context_limit(["deepseek-v3"]) == 32_768
    assert resolve_context_limit(["deepseek/deepseek-chat"]) == 32_768
    assert resolve_context_limit(["deepseek/deepseek-chat", "deepseek-v3"]) == 32_768
    assert resolve_context_limit(["gpt-5-nano-2025-08-07"]) == 128_000
    assert resolve_context_limit(["claude-sonnet-4-20250514"]) == 200_000


def test_reserved_output_tokens_env_caps_configured_reserve(monkeypatch):
    monkeypatch.setenv("NEGOTIATION_CONTEXT_RESERVED_OUTPUT_TOKENS", "2048")

    assert reserved_output_tokens(16_384, 32_768) == 2048


def test_compact_public_history_preserves_only_public_summaries():
    history = [
        {"round": 1, "phase": "discussion_turn_1", "from": "Agent_2", "content": "Public offer. More text."},
        {"round": 1, "phase": "private_thinking", "from": "Agent_2", "content": "private strategy"},
        {"round": 1, "phase": "proposal", "from": "Agent_1", "content": "I propose: formal allocation"},
        {"round": 2, "phase": "discussion_turn_1", "from": "Agent_1", "content": "Keep raw round 2."},
    ]

    compacted = compact_public_history_entries(history, {1})

    assert len(compacted) == 2
    assert compacted[0]["phase"] == "compressed_public_round_summary"
    assert "Public offer." in compacted[0]["content"]
    assert "formal allocation" in compacted[0]["content"]
    assert "private strategy" not in compacted[0]["content"]
    assert compacted[1]["content"] == "Keep raw round 2."


def test_build_context_messages_compacts_prompt_local_history(monkeypatch):
    agent = DummyContextAgent("Agent_1", LLMConfig(model_type=ModelType.GPT_4O, max_tokens=1))
    history = [
        {
            "round": 1,
            "phase": "discussion_turn_1",
            "from": "Agent_2",
            "content": "Round one public discussion. " + ("x " * 4000),
        },
        {
            "round": 2,
            "phase": "discussion_turn_1",
            "from": "Agent_2",
            "content": "Round two public discussion. " + ("y " * 120),
        },
    ]
    context = _context(history)

    monkeypatch.setenv("NEGOTIATION_CONTEXT_COMPACTION_THRESHOLD", "0.85")
    monkeypatch.setattr("negotiation.llm_agents.resolve_context_limit", lambda _names: 600)
    messages, metadata = agent._build_context_messages_with_metadata(context, "Respond.")

    assert metadata is not None
    assert metadata.compacted_rounds == [1]
    assert metadata.context_compacted is True
    assert metadata.phase_prompt_chars == len("Respond.")
    assert metadata.estimated_provider_input_tokens == metadata.estimated_input_tokens_after
    assert "DETERMINISTIC PUBLIC SUMMARY" in "\n".join(message["content"] for message in messages)
    assert context.conversation_history == history


def test_deepseek_uses_terse_compaction_when_standard_summaries_still_exceed_budget(monkeypatch):
    agent = DummyDeepSeekContextAgent("Agent_1", LLMConfig(model_type=ModelType.GPT_4O, max_tokens=1))
    history = [
        {
            "round": 1,
            "phase": "proposal",
            "from": f"Agent_{idx + 1}",
            "content": "I propose: " + ("item allocation utility " * 500),
        }
        for idx in range(10)
    ]
    context = _context(history)

    monkeypatch.setenv("NEGOTIATION_CONTEXT_COMPACTION_THRESHOLD", "0.85")
    monkeypatch.setattr("negotiation.llm_agents.resolve_context_limit", lambda _names: 4_000)
    messages, metadata = agent._build_context_messages_with_metadata(context, "Respond.")
    rendered = "\n".join(message["content"] for message in messages)

    assert metadata is not None
    assert metadata.context_compacted is True
    assert metadata.compacted_rounds == [1]
    assert metadata.estimated_input_tokens_after <= metadata.input_budget_tokens
    assert "TERSE PUBLIC SUMMARY" in rendered
    assert "DETERMINISTIC PUBLIC SUMMARY" not in rendered
    assert context.conversation_history == history


def test_build_context_messages_records_budget_metadata_without_compaction(monkeypatch):
    agent = DummyContextAgent("Agent_1", LLMConfig(model_type=ModelType.GPT_4O, max_tokens=1))
    context = _context([
        {
            "round": 1,
            "phase": "discussion_turn_1",
            "from": "Agent_2",
            "content": "Short public discussion.",
        },
    ])

    monkeypatch.setenv("NEGOTIATION_CONTEXT_COMPACTION_THRESHOLD", "0.85")
    monkeypatch.setattr("negotiation.llm_agents.resolve_context_limit", lambda _names: 10_000)
    _messages, metadata = agent._build_context_messages_with_metadata(context, "Respond.")

    assert metadata is not None
    assert metadata.context_compacted is False
    assert metadata.compacted_rounds == []
    assert metadata.phase_prompt_chars == len("Respond.")
    assert metadata.estimated_provider_input_tokens == metadata.estimated_input_tokens_before
    assert metadata.context_limit_tokens == 10_000
    assert metadata.input_budget_tokens is not None


def test_discussion_phase_bounds_current_round_history_for_deepseek():
    handler = PhaseHandler()
    agent = DummyDeepSeekContextAgent(
        "Agent_1",
        LLMConfig(model_type=ModelType.GPT_4O, max_tokens=999999),
    )
    messages = [
        {
            "from": f"Agent_{idx % 3 + 1}",
            "content": f"Message {idx}. " + ("current round detail " * 2000),
        }
        for idx in range(8)
    ]

    raw_history = [
        f"**{message['from']}**: {message['content']}"
        for message in messages
    ]
    bounded_history = handler._build_current_discussion_history(messages, agent)

    assert estimate_text_tokens("\n".join(raw_history)) > handler.CURRENT_DISCUSSION_HISTORY_MAX_TOKENS
    assert estimate_text_tokens("\n".join(bounded_history)) <= handler.CURRENT_DISCUSSION_HISTORY_MAX_TOKENS
    assert bounded_history[-1].startswith("**Agent_2**:")
    assert len(bounded_history[-1]) < len(raw_history[-1])


def test_discussion_phase_keeps_current_round_history_for_large_context_model():
    handler = PhaseHandler()
    agent = DummyContextAgent(
        "Agent_1",
        LLMConfig(model_type=ModelType.GPT_4O, max_tokens=999999),
    )
    messages = [{"from": "Agent_2", "content": "Line one.\nLine two."}]

    assert handler._build_current_discussion_history(messages, agent) == [
        "**Agent_2**: Line one.\nLine two."
    ]


def test_token_usage_exposes_provider_context_observability_fields():
    response = AgentResponse(
        content="ok",
        model_used="test-model",
        response_time=0.25,
        tokens_used=30,
        metadata={
            "input_tokens": 20,
            "output_tokens": 10,
            "context_compaction": {
                "phase_prompt_chars": 123,
                "estimated_provider_input_tokens": 456,
                "context_limit_tokens": 1000,
                "input_budget_tokens": 850,
                "context_compacted": False,
            },
        },
    )

    usage = BaseLLMAgent._extract_token_usage_from_response(response)

    assert usage["provider_input_tokens"] == 20
    assert usage["phase_prompt_chars"] == 123
    assert usage["estimated_provider_input_tokens"] == 456
    assert usage["context_limit_tokens"] == 1000
    assert usage["input_budget_tokens"] == 850
    assert usage["context_compacted"] is False


def test_context_preflight_fails_when_compaction_cannot_fit(monkeypatch):
    agent = DummyContextAgent("Agent_1", LLMConfig(model_type=ModelType.GPT_4O, max_tokens=1))
    history = [
        {
            "round": 1,
            "phase": "discussion_turn_1",
            "from": "Agent_2",
            "content": "Round one public discussion. " + ("x " * 120),
        },
    ]
    context = _context(history)

    monkeypatch.setenv("NEGOTIATION_CONTEXT_COMPACTION_THRESHOLD", "0.85")
    monkeypatch.setattr("negotiation.llm_agents.resolve_context_limit", lambda _names: 60)

    with pytest.raises(ContextWindowPreflightError, match="context_length_exceeded preflight"):
        agent._build_context_messages_with_metadata(context, "Respond.")

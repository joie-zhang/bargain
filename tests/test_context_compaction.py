"""Tests for prompt-local context compaction preflight."""

import pytest

from negotiation.context_compaction import (
    ContextWindowPreflightError,
    compact_public_history_entries,
    resolve_context_limit,
)
from negotiation.llm_agents import BaseLLMAgent, LLMConfig, ModelType, NegotiationContext


class DummyContextAgent(BaseLLMAgent):
    """Minimal model wrapper for prompt-building tests."""

    async def _call_llm_api(self, messages, **kwargs):
        raise NotImplementedError("API calls are not used in this test")

    def get_model_info(self):
        return {"model_name": "amazon-nova-micro-v1.0"}


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
    assert resolve_context_limit(["gpt-5-nano-2025-08-07"]) == 128_000


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
    assert "DETERMINISTIC PUBLIC SUMMARY" in "\n".join(message["content"] for message in messages)
    assert context.conversation_history == history


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

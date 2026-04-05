#!/usr/bin/env python3
"""Focused tests for diplomatic preference replay in agent system prompts."""

from negotiation.llm_agents import BaseLLMAgent, LLMConfig, ModelType, NegotiationContext


class DummyAgent(BaseLLMAgent):
    """Minimal agent for testing message construction only."""

    async def _call_llm_api(self, messages, **kwargs):
        raise NotImplementedError("API calls are not used in this test")

    def get_model_info(self):
        return {"model_name": "dummy-model"}


def test_context_messages_render_diplomatic_positions_and_weights():
    """Diplomatic contexts should replay both ideal positions and importance weights."""
    config = LLMConfig(model_type=ModelType.GPT_4O)
    agent = DummyAgent("Agent_1", config)

    context = NegotiationContext(
        current_round=2,
        max_rounds=5,
        items=[{"name": "Issue A"}, {"name": "Issue B"}],
        agents=["Agent_1", "Agent_2"],
        agent_id="Agent_1",
        preferences={
            "issues": ["Issue A", "Issue B"],
            "positions": [0.25, 0.80],
            "weights": [0.60, 0.40],
        },
    )

    messages = agent._build_context_messages(context, "Respond.")
    system_prompt = messages[0]["content"]
    assert "Issue A: ideal=25%, weight=60%" in system_prompt
    assert "Issue B: ideal=80%, weight=40%" in system_prompt

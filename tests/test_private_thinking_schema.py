"""Tests for shared private-thinking schema normalization."""

from negotiation.llm_agents import BaseLLMAgent, LLMConfig, ModelType


class DummyAgent(BaseLLMAgent):
    """Minimal agent for testing BaseLLMAgent parsing helpers."""

    async def _call_llm_api(self, messages, **kwargs):
        raise NotImplementedError

    def get_model_info(self):
        return {"model_name": "dummy"}


def test_thinking_response_normalizes_shared_schema():
    """Shared private-thinking keys should be normalized with backward-compatible aliases."""
    agent = DummyAgent("test_agent", LLMConfig(model_type=ModelType.GPT_4O))

    parsed = agent._parse_thinking_response("""{
  "reasoning": "I should push hardest for Apple.",
  "strategy": "Trade lower-value items to secure my top pick.",
  "key_priorities": ["Apple", "Jewel"],
  "potential_concessions": ["Pencil"]
}""")

    assert parsed["key_priorities"] == ["Apple", "Jewel"]
    assert parsed["potential_concessions"] == ["Pencil"]
    assert parsed["target_items"] == ["Apple", "Jewel"]
    assert parsed["anticipated_resistance"] == ["Pencil"]

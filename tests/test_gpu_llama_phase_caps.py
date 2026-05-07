from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from negotiation.llm_agents import DEFAULT_OPENAI_MAX_TOKENS_CAP, OPENAI_MODEL_MAX_TOKENS_CAPS
from strong_models_experiment.agents.agent_factory import StrongModelAgentFactory
from strong_models_experiment.configs import (
    AMAZON_NOVA_MICRO_V1_MAX_TOKENS_PER_PHASE,
    AMAZON_NOVA_PRO_V1_MAX_TOKENS_PER_PHASE,
    CLAUDE_3_HAIKU_MAX_TOKENS_PER_PHASE,
    COMMAND_R_PLUS_08_2024_MAX_TOKENS_PER_PHASE,
    DEFAULT_MAX_TOKENS_PER_PHASE,
    GPT_4O_2024_05_13_MAX_TOKENS_PER_PHASE,
    STRONG_MODELS_CONFIG,
)
from strong_models_experiment.experiment import build_phase_token_config
from strong_models_experiment.phases.phase_handlers import PhaseHandler


EXPECTED_PHASE_CAPS = {
    "max_tokens_discussion": 16384,
    "max_tokens_thinking": 16384,
    "max_tokens_proposal": 16384,
    "max_tokens_voting": 16384,
    "max_tokens_reflection": 16384,
}

EXPECTED_GPT4O_PHASE_CAPS = {
    "max_tokens_discussion": GPT_4O_2024_05_13_MAX_TOKENS_PER_PHASE,
    "max_tokens_thinking": GPT_4O_2024_05_13_MAX_TOKENS_PER_PHASE,
    "max_tokens_proposal": GPT_4O_2024_05_13_MAX_TOKENS_PER_PHASE,
    "max_tokens_voting": GPT_4O_2024_05_13_MAX_TOKENS_PER_PHASE,
    "max_tokens_reflection": GPT_4O_2024_05_13_MAX_TOKENS_PER_PHASE,
    "max_tokens_default": GPT_4O_2024_05_13_MAX_TOKENS_PER_PHASE,
}


@dataclass
class DummyConfig:
    custom_parameters: Dict[str, Any] = field(default_factory=dict)


class DummyAgent:
    def __init__(self, phase_token_caps: Dict[str, int] | None = None):
        custom_parameters = {}
        if phase_token_caps is not None:
            custom_parameters["phase_token_caps"] = phase_token_caps
        self.config = DummyConfig(custom_parameters=custom_parameters)
        self.updated_limits: List[int] = []

    def update_max_tokens(self, max_tokens):
        self.updated_limits.append(max_tokens)


def _empty_token_config() -> Dict[str, Any]:
    return {
        "discussion": None,
        "proposal": None,
        "voting": None,
        "reflection": None,
        "thinking": None,
        "default": None,
    }


def test_experiment_token_config_defaults_all_phases_to_16k():
    assert build_phase_token_config({}) == {
        "discussion": DEFAULT_MAX_TOKENS_PER_PHASE,
        "proposal": DEFAULT_MAX_TOKENS_PER_PHASE,
        "voting": DEFAULT_MAX_TOKENS_PER_PHASE,
        "reflection": DEFAULT_MAX_TOKENS_PER_PHASE,
        "thinking": DEFAULT_MAX_TOKENS_PER_PHASE,
        "default": DEFAULT_MAX_TOKENS_PER_PHASE,
    }


def test_experiment_token_config_applies_max_tokens_per_phase_with_specific_override():
    assert build_phase_token_config(
        {
            "max_tokens_per_phase": 12000,
            "max_tokens_voting": 16384,
        }
    ) == {
        "discussion": 12000,
        "proposal": 12000,
        "voting": 16384,
        "reflection": 12000,
        "thinking": 12000,
        "default": 12000,
    }


def test_gpt4o_2024_05_13_aliases_have_4k_phase_caps():
    for alias in ("gpt-4o", "gpt-4o-2024-05-13"):
        config = STRONG_MODELS_CONFIG[alias]
        assert config["model_id"] == "gpt-4o-2024-05-13"
        for key, value in EXPECTED_GPT4O_PHASE_CAPS.items():
            assert config[key] == value


def test_openrouter_provider_limited_aliases_have_phase_caps():
    capped_aliases = {
        "amazon-nova-pro-v1.0": (
            "amazon/nova-pro-v1",
            AMAZON_NOVA_PRO_V1_MAX_TOKENS_PER_PHASE,
        ),
        "command-r-plus-08-2024": (
            "cohere/command-r-plus-08-2024",
            COMMAND_R_PLUS_08_2024_MAX_TOKENS_PER_PHASE,
        ),
        "claude-3-haiku": (
            "anthropic/claude-3-haiku",
            CLAUDE_3_HAIKU_MAX_TOKENS_PER_PHASE,
        ),
        "claude-3-haiku-20240307": (
            "anthropic/claude-3-haiku",
            CLAUDE_3_HAIKU_MAX_TOKENS_PER_PHASE,
        ),
        "amazon-nova-micro": (
            "amazon/nova-micro-v1",
            AMAZON_NOVA_MICRO_V1_MAX_TOKENS_PER_PHASE,
        ),
        "amazon-nova-micro-v1.0": (
            "amazon/nova-micro-v1",
            AMAZON_NOVA_MICRO_V1_MAX_TOKENS_PER_PHASE,
        ),
    }

    phase_cap_fields = (
        "max_tokens_discussion",
        "max_tokens_thinking",
        "max_tokens_proposal",
        "max_tokens_voting",
        "max_tokens_reflection",
        "max_tokens_default",
    )

    for alias, (model_id, expected_cap) in capped_aliases.items():
        config = STRONG_MODELS_CONFIG[alias]
        assert config["model_id"] == model_id
        assert config["api_type"] == "openrouter"
        for field in phase_cap_fields:
            assert config[field] == expected_cap


def test_openai_call_site_has_gpt4o_specific_output_cap():
    assert DEFAULT_OPENAI_MAX_TOKENS_CAP == DEFAULT_MAX_TOKENS_PER_PHASE
    assert (
        OPENAI_MODEL_MAX_TOKENS_CAPS["gpt-4o-2024-05-13"]
        == GPT_4O_2024_05_13_MAX_TOKENS_PER_PHASE
    )


def test_cluster_llama_aliases_have_caps_but_api_aliases_do_not():
    cluster_aliases = [
        "llama-3.1-8b-instruct-cluster",
        "llama-3.2-3b-instruct-cluster",
        "llama-3.2-1b-instruct-cluster",
    ]
    api_aliases = [
        "llama-3.1-8b-instruct",
        "llama-3.2-3b-instruct",
        "llama-3.2-1b-instruct",
    ]

    for alias in cluster_aliases:
        config = STRONG_MODELS_CONFIG[alias]
        assert config["provider"] == "Princeton Cluster"
        for key, value in EXPECTED_PHASE_CAPS.items():
            assert config[key] == value

    for alias in api_aliases:
        config = STRONG_MODELS_CONFIG[alias]
        for key in EXPECTED_PHASE_CAPS:
            assert key not in config


def test_phase_handler_applies_agent_specific_caps_without_touching_uncapped_agents():
    handler = PhaseHandler(token_config=_empty_token_config())
    capped_agent = DummyAgent({"discussion": 2048})
    uncapped_agent = DummyAgent()

    handler._apply_phase_token_limits([capped_agent, uncapped_agent], "discussion")

    assert capped_agent.updated_limits == [2048]
    assert uncapped_agent.updated_limits == []


def test_phase_handler_uses_min_of_global_limit_and_agent_cap():
    token_config = _empty_token_config()
    token_config["discussion"] = 1024
    handler = PhaseHandler(token_config=token_config)
    capped_agent = DummyAgent({"discussion": 2048})
    uncapped_agent = DummyAgent()

    handler._apply_phase_token_limits([capped_agent, uncapped_agent], "discussion")

    assert capped_agent.updated_limits == [1024]
    assert uncapped_agent.updated_limits == [1024]


def test_phase_handler_voting_falls_back_to_default_but_prefers_phase_specific_cap():
    token_config = _empty_token_config()
    token_config["default"] = 2000
    handler = PhaseHandler(token_config=token_config)
    capped_agent = DummyAgent({"voting": 16384})
    uncapped_agent = DummyAgent()

    handler._apply_phase_token_limits([capped_agent, uncapped_agent], "voting", fallback_phase="default")

    assert capped_agent.updated_limits == [16384]
    assert uncapped_agent.updated_limits == [2000]


def test_phase_handler_uses_gpt4o_cap_below_global_16k_limit():
    token_config = {
        "discussion": DEFAULT_MAX_TOKENS_PER_PHASE,
        "proposal": DEFAULT_MAX_TOKENS_PER_PHASE,
        "voting": DEFAULT_MAX_TOKENS_PER_PHASE,
        "reflection": DEFAULT_MAX_TOKENS_PER_PHASE,
        "thinking": DEFAULT_MAX_TOKENS_PER_PHASE,
        "default": DEFAULT_MAX_TOKENS_PER_PHASE,
    }
    handler = PhaseHandler(token_config=token_config)
    capped_agent = DummyAgent({"discussion": 4096, "default": 4096})

    handler._apply_phase_token_limits([capped_agent], "discussion")
    handler._apply_phase_token_limits([capped_agent], "default")

    assert capped_agent.updated_limits == [4096, 4096]


def test_local_model_factory_passes_phase_caps_into_agent_config(monkeypatch):
    captured = {}

    class StubLocalModelAgent:
        def __init__(self, agent_id, config, local_path):
            captured["agent_id"] = agent_id
            captured["config"] = config
            captured["local_path"] = local_path

    monkeypatch.setattr(
        "strong_models_experiment.agents.agent_factory.LocalModelAgent",
        StubLocalModelAgent,
    )

    factory = StrongModelAgentFactory()
    config = STRONG_MODELS_CONFIG["llama-3.2-3b-instruct-cluster"]
    factory._create_local_model_agent(
        "llama-3.2-3b-instruct-cluster",
        config,
        "Agent_1",
        max_tokens=999999,
    )

    assert captured["agent_id"] == "Agent_1"
    assert captured["local_path"] == "bargain/models/Llama-3.2-3B-Instruct"
    assert captured["config"].custom_parameters["phase_token_caps"] == {
        "discussion": 16384,
        "thinking": 16384,
        "proposal": 16384,
        "voting": 16384,
        "reflection": 16384,
    }


def test_openai_model_factory_passes_gpt4o_phase_caps_into_agent_config(monkeypatch):
    captured = {}

    class StubOpenAIAgent:
        def __init__(self, agent_id, config, api_key):
            captured["agent_id"] = agent_id
            captured["config"] = config
            captured["api_key"] = api_key

    monkeypatch.setattr(
        "strong_models_experiment.agents.agent_factory.OpenAIAgent",
        StubOpenAIAgent,
    )
    monkeypatch.setattr(
        "strong_models_experiment.agents.agent_factory.has_provider_keys",
        lambda provider, fallback_key=None: True,
    )

    factory = StrongModelAgentFactory()
    config = STRONG_MODELS_CONFIG["gpt-4o-2024-05-13"]
    factory._create_openai_agent(
        "gpt-4o-2024-05-13",
        config,
        "Agent_1",
        api_key="sk-test",
        max_tokens=DEFAULT_MAX_TOKENS_PER_PHASE,
    )

    assert captured["agent_id"] == "Agent_1"
    assert captured["config"].custom_parameters["phase_token_caps"] == {
        "discussion": 4096,
        "thinking": 4096,
        "proposal": 4096,
        "voting": 4096,
        "reflection": 4096,
        "default": 4096,
    }


def test_openrouter_model_factory_passes_provider_limited_phase_caps_into_agent_config(monkeypatch):
    captured = {}

    class StubOpenRouterAgent:
        def __init__(self, agent_id, llm_config, api_key, model_id):
            captured["agent_id"] = agent_id
            captured["config"] = llm_config
            captured["api_key"] = api_key
            captured["model_id"] = model_id

    monkeypatch.setattr(
        "strong_models_experiment.agents.agent_factory.OpenRouterAgent",
        StubOpenRouterAgent,
    )
    monkeypatch.setattr(
        "strong_models_experiment.agents.agent_factory.has_provider_keys",
        lambda provider, fallback_key=None: True,
    )

    factory = StrongModelAgentFactory()
    config = STRONG_MODELS_CONFIG["command-r-plus-08-2024"]
    factory._create_openrouter_agent(
        "command-r-plus-08-2024",
        config,
        "Agent_1",
        api_key="sk-or-test",
        max_tokens=DEFAULT_MAX_TOKENS_PER_PHASE,
    )

    assert captured["agent_id"] == "Agent_1"
    assert captured["model_id"] == "cohere/command-r-plus-08-2024"
    assert captured["config"].custom_parameters["phase_token_caps"] == {
        "discussion": 4000,
        "thinking": 4000,
        "proposal": 4000,
        "voting": 4000,
        "reflection": 4000,
        "default": 4000,
    }

"""Agent creation utilities for different model providers."""

import os
import logging
from typing import List, Dict, Any, Optional
from negotiation import AgentFactory, AgentConfiguration
from negotiation.llm_agents import ModelType, BaseLLMAgent, AnthropicAgent, OpenAIAgent, LocalModelAgent, GoogleAgent, LLMConfig
from negotiation.openrouter_client import OpenRouterAgent
from ..configs import STRONG_MODELS_CONFIG


class StrongModelAgentFactory:
    """Factory for creating agents with strong language models."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.factory = AgentFactory()
    
    async def create_agents(self, models: List[str], config: Dict[str, Any]) -> List[BaseLLMAgent]:
        """Create agents for the specified models.
        
        Args:
            models: List of model names to use
            config: Configuration dictionary including token limits
            
        Returns:
            List of created agents
        """
        agents = []
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        xai_key = os.getenv("XAI_API_KEY")
        google_key = os.getenv("GOOGLE_API_KEY")
        
        # Get max_tokens from config, default to None (unlimited)
        max_tokens = config.get("max_tokens_default", None)
        # Convert None to effectively unlimited for agent creation
        if max_tokens is None:
            max_tokens = 999999

        # Get reasoning token budget for API-based control (test-time compute scaling)
        reasoning_token_budget = config.get("reasoning_token_budget", None)
        
        # If only one model specified, create 3 agents of that model for negotiation
        if len(models) == 1:
            models = models * 3
        
        # Use Greek letters for agent names to maintain anonymity
        agent_names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta"]

        # DEBUG: Log the model order as received
        self.logger.info(f"DEBUG: Models received in create_agents: {models}")

        for i, model_name in enumerate(models):
            if model_name not in STRONG_MODELS_CONFIG:
                self.logger.warning(f"Unknown model: {model_name}, skipping")
                continue

            model_config = STRONG_MODELS_CONFIG[model_name]
            api_type = model_config.get("api_type", "openrouter")

            # Use anonymous agent names instead of model names
            if i < len(agent_names):
                agent_id = f"Agent_{agent_names[i]}"
            else:
                agent_id = f"Agent_{i+1}"  # Fallback to numbers if we run out of Greek letters

            # DEBUG: Log the mapping
            self.logger.info(f"DEBUG: Creating {agent_id} -> {model_name} (api_type={api_type})")
            
            agent = self._create_agent_by_type(
                api_type, model_name, model_config, agent_id,
                anthropic_key, openai_key, openrouter_key, xai_key, google_key, max_tokens,
                reasoning_token_budget=reasoning_token_budget
            )
            
            if agent:
                agents.append(agent)
        
        if not agents:
            raise ValueError("No agents could be created. Check your API keys.")
        
        return agents
    
    def _create_agent_by_type(self, api_type: str, model_name: str, model_config: Dict,
                             agent_id: str, anthropic_key: Optional[str],
                             openai_key: Optional[str], openrouter_key: Optional[str],
                             xai_key: Optional[str], google_key: Optional[str],
                             max_tokens: int = 999999,
                             reasoning_token_budget: Optional[int] = None) -> Optional[BaseLLMAgent]:
        """Create an agent based on API type.

        Args:
            reasoning_token_budget: Optional token budget for API-based reasoning control.
                - For Anthropic: Sets thinking.budget_tokens parameter
                - For OpenAI: Maps to reasoning_effort (low/medium/high)
                - For others: No API-level control available
        """

        if api_type == "anthropic":
            return self._create_anthropic_agent(model_name, model_config, agent_id, anthropic_key, max_tokens, reasoning_token_budget)
        elif api_type == "openai":
            return self._create_openai_agent(model_name, model_config, agent_id, openai_key, max_tokens, reasoning_token_budget)
        elif api_type == "xai":
            return self._create_xai_agent(model_name, model_config, agent_id, xai_key, max_tokens)
        elif api_type == "google":
            return self._create_google_agent(model_name, model_config, agent_id, google_key, max_tokens)
        elif api_type == "princeton_cluster":
            return self._create_local_model_agent(model_name, model_config, agent_id, max_tokens)
        else:  # openrouter
            return self._create_openrouter_agent(model_name, model_config, agent_id, openrouter_key, max_tokens)
    
    def _create_anthropic_agent(self, model_name: str, model_config: Dict,
                               agent_id: str, api_key: Optional[str], max_tokens: int = 999999,
                               reasoning_token_budget: Optional[int] = None) -> Optional[AnthropicAgent]:
        """Create an Anthropic agent.

        Args:
            reasoning_token_budget: If set, enables extended thinking with this budget.
                Sets thinking.budget_tokens in the API call. Minimum is 1024 tokens.
        """
        # Check if model is deprecated
        if model_config.get("deprecated", False):
            deprecation_msg = model_config.get("deprecation_message", f"{model_name} is deprecated")
            error_msg = f"ERROR: {deprecation_msg}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        if not api_key:
            self.logger.warning(f"ANTHROPIC_API_KEY not set, skipping {model_name}")
            return None
        
        # Map model names to ModelType enum values
        # Note: This is mainly for compatibility/fallback. The actual model_id is always
        # set via _actual_model_id (line 139), which takes precedence in AnthropicAgent.
        model_type_map = {
            "claude-3-opus": ModelType.CLAUDE_3_OPUS,
            "claude-3-sonnet": ModelType.CLAUDE_3_SONNET,
            "claude-3-haiku": ModelType.CLAUDE_3_HAIKU,
            "claude-3-5-sonnet": ModelType.CLAUDE_3_5_SONNET,
            "claude-3-5-haiku": ModelType.CLAUDE_3_5_HAIKU,
            "claude-3-7-sonnet": ModelType.CLAUDE_3_7_SONNET,
            "claude-4-sonnet": ModelType.CLAUDE_4_SONNET,
            "claude-4-1-opus": ModelType.CLAUDE_4_1_OPUS,
            "claude-4-5-haiku": ModelType.CLAUDE_4_5_HAIKU,
        }
        
        # Use the enum if available, otherwise use a default
        model_type = model_type_map.get(model_name, ModelType.CLAUDE_3_5_SONNET)
        
        # Set appropriate max_tokens based on model type
        # Haiku models have a 4096 token limit, others can go higher
        if "haiku" in model_name.lower():
            actual_max_tokens = min(max_tokens, 4096)
        elif "opus" in model_name.lower():
            actual_max_tokens = min(max_tokens, 4096)
        else:
            actual_max_tokens = min(max_tokens, 8192)
        
        # Build custom_parameters for Anthropic
        custom_params = {}

        # Add reasoning token budget for extended thinking if specified
        # This enables API-level control via thinking.budget_tokens
        if reasoning_token_budget is not None:
            # Anthropic requires minimum of 1024 tokens for extended thinking
            budget = max(reasoning_token_budget, 1024)
            custom_params["thinking_budget_tokens"] = budget
            self.logger.info(f"Anthropic extended thinking enabled with budget_tokens={budget}")

        llm_config = LLMConfig(
            model_type=model_type,
            temperature=model_config["temperature"],
            max_tokens=actual_max_tokens,
            system_prompt=model_config["system_prompt"],
            custom_parameters=custom_params
        )

        # Always use model_id from config if available (more up-to-date than enum)
        if "model_id" in model_config:
            llm_config._actual_model_id = model_config["model_id"]

        return AnthropicAgent(
            agent_id=agent_id,
            config=llm_config,
            api_key=api_key
        )
    
    def _create_openai_agent(self, model_name: str, model_config: Dict,
                            agent_id: str, api_key: Optional[str], max_tokens: int = 999999,
                            reasoning_token_budget: Optional[int] = None) -> Optional[OpenAIAgent]:
        """Create an OpenAI agent.

        Args:
            reasoning_token_budget: If set, maps to reasoning_effort for O3/O1 models:
                - budget <= 2000: "low"
                - 2000 < budget <= 5000: "medium"
                - budget > 5000: "high"
        """
        if not api_key:
            self.logger.warning(f"OPENAI_API_KEY not set, skipping {model_name}")
            return None

        # Determine correct model type
        # Note: This is mainly for compatibility/fallback. The actual model_id is always
        # set via _actual_model_id (line 178), which takes precedence in OpenAIAgent.
        if "gpt-4o" in model_name:
            model_type = ModelType.GPT_4O
        elif "gpt-5" in model_name:
            model_type = ModelType.GPT_5
        elif "gpt-4" in model_name:
            model_type = ModelType.GPT_4
        elif "o3" in model_name or "o1" in model_name:
            model_type = ModelType.O3
        else:
            model_type = ModelType.GPT_4  # default fallback

        # Extract reasoning_effort if available and add to custom_parameters
        custom_params = {}
        if "reasoning_effort" in model_config:
            custom_params["reasoning_effort"] = model_config["reasoning_effort"]

        # Map reasoning_token_budget to reasoning_effort for O3/O1/GPT-5 models
        # This overrides model_config's reasoning_effort if budget is specified
        if reasoning_token_budget is not None and ("o3" in model_name.lower() or "o1" in model_name.lower() or "gpt-5" in model_name.lower()):
            if reasoning_token_budget <= 2000:
                custom_params["reasoning_effort"] = "low"
            elif reasoning_token_budget <= 5000:
                custom_params["reasoning_effort"] = "medium"
            else:
                custom_params["reasoning_effort"] = "high"
            self.logger.info(f"OpenAI reasoning_effort set to '{custom_params['reasoning_effort']}' based on budget={reasoning_token_budget}")

        llm_config = LLMConfig(
            model_type=model_type,
            temperature=model_config["temperature"],
            max_tokens=max_tokens,
            system_prompt=model_config["system_prompt"],
            custom_parameters=custom_params
        )
        
        # Store the actual model_id for the agent to use
        llm_config._actual_model_id = model_config["model_id"]
        
        return OpenAIAgent(
            agent_id=agent_id,
            config=llm_config,
            api_key=api_key
        )
    
    def _create_xai_agent(self, model_name: str, model_config: Dict,
                         agent_id: str, api_key: Optional[str], max_tokens: int = 999999) -> Optional['XAIAgent']:
        """Create an XAI Grok agent."""
        if not api_key:
            self.logger.warning(f"XAI_API_KEY not set, skipping {model_name}")
            return None
        
        from negotiation.llm_agents import XAIAgent, LLMConfig, ModelType
        
        # Use GPT_4 as the base model type for configuration compatibility
        llm_config = LLMConfig(
            model_type=ModelType.GPT_4,  # Base type for config compatibility
            temperature=model_config["temperature"],
            max_tokens=max_tokens,
            system_prompt=model_config["system_prompt"],
            custom_parameters={}
        )
        
        # Store the actual model_id for the agent to use
        llm_config._actual_model_id = model_config["model_id"]
        
        return XAIAgent(
            agent_id=agent_id,
            config=llm_config,
            api_key=api_key
        )

    def _create_google_agent(self, model_name: str, model_config: Dict,
                            agent_id: str, api_key: Optional[str], max_tokens: int = 999999) -> Optional[GoogleAgent]:
        """Create a Google Gemini agent."""
        if not api_key:
            self.logger.warning(f"GOOGLE_API_KEY not set, skipping {model_name}")
            return None

        # Use GPT_4 as the base model type for configuration compatibility
        llm_config = LLMConfig(
            model_type=ModelType.GPT_4,  # Base type for config compatibility
            temperature=model_config["temperature"],
            max_tokens=max_tokens,
            system_prompt=model_config["system_prompt"],
            custom_parameters={}
        )

        # Store the actual model_id for the agent to use
        llm_config._actual_model_id = model_config["model_id"]

        return GoogleAgent(
            agent_id=agent_id,
            config=llm_config,
            api_key=api_key
        )

    def _create_local_model_agent(self, model_name: str, model_config: Dict,
                                 agent_id: str, max_tokens: int = 999999) -> Optional[LocalModelAgent]:
        """Create a local model agent for Princeton cluster models."""
        local_path = model_config.get("local_path")
        if not local_path:
            self.logger.warning(f"local_path not specified for {model_name}, skipping")
            return None
        
        # Check if path exists
        import os
        if not os.path.exists(local_path):
            self.logger.warning(f"Model path does not exist: {local_path}, skipping {model_name}")
            return None
        
        # Use GEMMA_2_27B as base type (just for config compatibility, won't be used)
        llm_config = LLMConfig(
            model_type=ModelType.GEMMA_2_27B,  # Base type for compatibility
            temperature=model_config["temperature"],
            max_tokens=max_tokens,
            system_prompt=model_config["system_prompt"],
            custom_parameters={}
        )
        
        # Store the actual model_id for reference
        llm_config._actual_model_id = model_config.get("model_id", model_name)
        
        return LocalModelAgent(
            agent_id=agent_id,
            config=llm_config,
            local_path=local_path
        )
    
    def _create_openrouter_agent(self, model_name: str, model_config: Dict,
                                agent_id: str, api_key: Optional[str], max_tokens: int = 999999) -> Optional[OpenRouterAgent]:
        """Create an OpenRouter agent."""
        if not api_key:
            self.logger.warning(f"OPENROUTER_API_KEY not set, skipping {model_name}")
            return None
        
        agent_config = AgentConfiguration(
            agent_id=agent_id,
            model_type=ModelType.GEMMA_2_27B,  # Base type for OpenRouter
            api_key=api_key,
            temperature=model_config["temperature"],
            max_tokens=max_tokens,
            system_prompt=model_config["system_prompt"],
            custom_parameters={"model_id": model_config["model_id"]}
        )
        
        llm_config = agent_config.to_llm_config()
        
        return OpenRouterAgent(
            agent_id=agent_id,
            llm_config=llm_config,
            api_key=api_key,
            model_id=model_config["model_id"]
        )
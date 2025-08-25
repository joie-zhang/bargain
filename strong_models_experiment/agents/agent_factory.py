"""Agent creation utilities for different model providers."""

import os
import logging
from typing import List, Dict, Any, Optional
from negotiation import AgentFactory, AgentConfiguration
from negotiation.llm_agents import ModelType, BaseLLMAgent, AnthropicAgent, OpenAIAgent, LLMConfig
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
        
        # Get max_tokens from config, default to None (unlimited)
        max_tokens = config.get("max_tokens_default", None)
        # Convert None to effectively unlimited for agent creation
        if max_tokens is None:
            max_tokens = 999999
        
        # If only one model specified, create 3 agents of that model for negotiation
        if len(models) == 1:
            models = models * 3
        
        for i, model_name in enumerate(models):
            if model_name not in STRONG_MODELS_CONFIG:
                self.logger.warning(f"Unknown model: {model_name}, skipping")
                continue
            
            model_config = STRONG_MODELS_CONFIG[model_name]
            api_type = model_config.get("api_type", "openrouter")
            
            agent_id = f"{model_name.replace('-', '_')}_{i+1}"
            
            agent = self._create_agent_by_type(
                api_type, model_name, model_config, agent_id,
                anthropic_key, openai_key, openrouter_key, max_tokens
            )
            
            if agent:
                agents.append(agent)
        
        if not agents:
            raise ValueError("No agents could be created. Check your API keys.")
        
        return agents
    
    def _create_agent_by_type(self, api_type: str, model_name: str, model_config: Dict,
                             agent_id: str, anthropic_key: Optional[str],
                             openai_key: Optional[str], openrouter_key: Optional[str],
                             max_tokens: int = 999999) -> Optional[BaseLLMAgent]:
        """Create an agent based on API type."""
        
        if api_type == "anthropic":
            return self._create_anthropic_agent(model_name, model_config, agent_id, anthropic_key, max_tokens)
        elif api_type == "openai":
            return self._create_openai_agent(model_name, model_config, agent_id, openai_key, max_tokens)
        else:  # openrouter
            return self._create_openrouter_agent(model_name, model_config, agent_id, openrouter_key, max_tokens)
    
    def _create_anthropic_agent(self, model_name: str, model_config: Dict,
                               agent_id: str, api_key: Optional[str], max_tokens: int = 999999) -> Optional[AnthropicAgent]:
        """Create an Anthropic agent."""
        if not api_key:
            self.logger.warning(f"ANTHROPIC_API_KEY not set, skipping {model_name}")
            return None
        
        llm_config = LLMConfig(
            model_type=ModelType.CLAUDE_3_5_SONNET if "sonnet" in model_name else ModelType.CLAUDE_3_HAIKU,
            temperature=model_config["temperature"],
            max_tokens=max_tokens,
            system_prompt=model_config["system_prompt"]
        )
        
        return AnthropicAgent(
            agent_id=agent_id,
            config=llm_config,
            api_key=api_key
        )
    
    def _create_openai_agent(self, model_name: str, model_config: Dict,
                            agent_id: str, api_key: Optional[str], max_tokens: int = 999999) -> Optional[OpenAIAgent]:
        """Create an OpenAI agent."""
        if not api_key:
            self.logger.warning(f"OPENAI_API_KEY not set, skipping {model_name}")
            return None
        
        # Determine correct model type
        if "gpt-4o" in model_name:
            model_type = ModelType.GPT_4O
        elif "gpt-4" in model_name:
            model_type = ModelType.GPT_4
        elif "o3" in model_name:
            model_type = ModelType.O3
        else:
            model_type = ModelType.GPT_4  # default
        
        llm_config = LLMConfig(
            model_type=model_type,
            temperature=model_config["temperature"],
            max_tokens=max_tokens,
            system_prompt=model_config["system_prompt"],
            custom_parameters={}
        )
        
        return OpenAIAgent(
            agent_id=agent_id,
            config=llm_config,
            api_key=api_key
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
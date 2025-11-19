"""Agent factory for name-based experiments.

This factory extends StrongModelAgentFactory to use custom agent names
instead of Greek letters, and ensures both agents use the same model.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from negotiation.llm_agents import BaseLLMAgent
from strong_models_experiment.agents.agent_factory import StrongModelAgentFactory
from strong_models_experiment.configs import STRONG_MODELS_CONFIG


class NameAgentFactory(StrongModelAgentFactory):
    """Factory for creating agents with custom names for name-based experiments."""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
    
    async def create_agents(
        self, 
        model: str, 
        agent_names: List[str], 
        config: Dict[str, Any]
    ) -> List[BaseLLMAgent]:
        """Create agents with custom names, all using the same model.
        
        Args:
            model: Single model name to use for all agents
            agent_names: List of custom names for agents (e.g., ["Alice", "Bob"])
            config: Configuration dictionary including token limits
            
        Returns:
            List of created agents with custom names
        """
        agents = []
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        xai_key = os.getenv("XAI_API_KEY")
        
        # Get max_tokens from config, default to None (unlimited)
        max_tokens = config.get("max_tokens_default", None)
        # Convert None to effectively unlimited for agent creation
        if max_tokens is None:
            max_tokens = 999999
        
        # Validate model exists
        if model not in STRONG_MODELS_CONFIG:
            raise ValueError(f"Unknown model: {model}. Available models: {list(STRONG_MODELS_CONFIG.keys())}")
        
        model_config = STRONG_MODELS_CONFIG[model]
        api_type = model_config.get("api_type", "openrouter")
        
        # Create agents with custom names, all using the same model
        for agent_name in agent_names:
            # Use the custom name directly as agent_id (without "Agent_" prefix for cleaner names)
            agent_id = agent_name
            
            agent = self._create_agent_by_type(
                api_type, model, model_config, agent_id,
                anthropic_key, openai_key, openrouter_key, xai_key, max_tokens
            )
            
            if agent:
                agents.append(agent)
            else:
                self.logger.warning(f"Failed to create agent with name: {agent_name}")
        
        if not agents:
            raise ValueError("No agents could be created. Check your API keys and model configuration.")
        
        self.logger.info(f"Created {len(agents)} agents with names: {[a.agent_id for a in agents]}")
        return agents


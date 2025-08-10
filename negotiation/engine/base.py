"""
Base classes for the modular negotiation engine.

This module defines the core interfaces and configuration classes that form the
foundation of the parameterizable negotiation engine system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import time


@dataclass  
class NegotiationEngineConfig:
    """Configuration for negotiation engine behavior."""
    # Environment parameters
    n_agents: int
    m_items: int  
    t_rounds: int
    gamma_discount: float
    
    # Phase configuration
    max_discussion_turns: int = 3
    allow_private_thinking: bool = True
    require_unanimous_consensus: bool = True
    
    # Communication settings
    randomized_proposal_order: bool = False
    max_reflection_chars: int = 2000
    turn_timeout_seconds: int = 60
    
    # Behavioral options
    enable_strategic_analysis: bool = True
    enable_conversation_logging: bool = True
    enable_performance_tracking: bool = True
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.n_agents < 2:
            raise ValueError("n_agents must be at least 2")
        if self.m_items < 1:
            raise ValueError("m_items must be at least 1")
        if self.t_rounds < 1:
            raise ValueError("t_rounds must be at least 1")
        if not (0 <= self.gamma_discount <= 1):
            raise ValueError("gamma_discount must be between 0 and 1")
        if self.max_discussion_turns < 1:
            raise ValueError("max_discussion_turns must be at least 1")
        if self.turn_timeout_seconds < 1:
            raise ValueError("turn_timeout_seconds must be at least 1")


class NegotiationEngine(ABC):
    """Abstract base class for negotiation engines."""
    
    def __init__(self, config: NegotiationEngineConfig):
        """Initialize the negotiation engine with configuration."""
        config.validate()
        self.config = config
        self.engine_id = f"engine_{int(time.time())}_{id(self)}"
    
    @abstractmethod
    async def run_negotiation(self, 
                            agents: List[Any],  # BaseLLMAgent - avoiding circular import
                            env: Any,           # NegotiationEnvironment
                            preferences: Dict[str, Any]) -> 'NegotiationResult':
        """
        Run the complete negotiation process.
        
        Args:
            agents: List of LLM agents participating in negotiation
            env: The negotiation environment containing items and rules
            preferences: Preference specifications for agents (vector or matrix based)
            
        Returns:
            NegotiationResult: Complete results and analysis of the negotiation
        """
        pass
    
    def get_config(self) -> NegotiationEngineConfig:
        """Get the current configuration."""
        return self.config
        
    def get_engine_info(self) -> Dict[str, Any]:
        """Get basic information about this engine instance."""
        return {
            "engine_id": self.engine_id,
            "engine_type": self.__class__.__name__,
            "config": self.config
        }
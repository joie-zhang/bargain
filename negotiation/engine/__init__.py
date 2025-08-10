"""
Modular negotiation engine package.

This package provides a configurable, reusable negotiation engine that can be
parameterized for different experiment setups and research questions.
"""

from .base import NegotiationEngine, NegotiationEngineConfig
from .results import NegotiationResult, PhaseResult
from .communication import ConversationManager, ConversationContext
from .phases import PhaseManager
from .consensus import ConsensusTracker, UtilityCalculator
from .orchestrator import StandardNegotiationEngine

__version__ = "0.1.0"

__all__ = [
    # Core interfaces and configuration
    "NegotiationEngine",
    "NegotiationEngineConfig",
    
    # Result types
    "NegotiationResult", 
    "PhaseResult",
    
    # Communication management
    "ConversationManager",
    "ConversationContext",
    
    # Phase management
    "PhaseManager",
    
    # Consensus and utility calculation
    "ConsensusTracker",
    "UtilityCalculator",
    
    # Main engine implementation
    "StandardNegotiationEngine",
]
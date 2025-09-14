"""
Multi-Agent Negotiation Environment

This package implements a research environment for studying strategic interactions
between AI agents in negotiation settings, with a focus on understanding how
stronger models might exploit weaker models.
"""

from .environment import (
    NegotiationEnvironment,
    NegotiationConfig,
    NegotiationStatus,
    Item,
    ItemPool,
    Round,
    create_negotiation_environment
)

from .communication import (
    Message,
    MessageType,
    Turn,
    TurnType,
    AgentInterface,
    SimpleAgent,
    CommunicationManager,
    TurnManager,
    create_communication_system
)

from .preferences import (
    PreferenceType,
    PreferenceConfig,
    BasePreferenceSystem,
    VectorPreferenceSystem,
    MatrixPreferenceSystem,
    PreferenceManager,
    create_competitive_preferences,
    create_cooperative_preferences,
    analyze_preference_competition_level
)

from .llm_agents import (
    ModelProvider,
    ModelType,
    LLMConfig,
    AgentResponse,
    NegotiationContext,
    RateLimiter,
    BaseLLMAgent,
    AnthropicAgent,
    OpenAIAgent,
)

from .agent_factory import (
    AgentConfiguration,
    ExperimentConfiguration,
    AgentFactory,
    create_o3_vs_haiku_experiment,
    create_cooperative_experiment,
    create_scaling_study_experiment,
    agent_factory
)

from .utility_engine import (
    UtilityEngine,
    UtilityCalculationResult,
    create_utility_engine,
    calculate_discounted_utility,
    compare_discount_factors
)

from .negotiation_runner import (
    ModularNegotiationRunner,
    NegotiationOutcome,
    NegotiationPhase
)

__version__ = "0.1.0"

__all__ = [
    # Environment components
    "NegotiationEnvironment",
    "NegotiationConfig", 
    "NegotiationStatus",
    "Item",
    "ItemPool",
    "Round",
    "create_negotiation_environment",
    
    # Communication components
    "Message",
    "MessageType",
    "Turn", 
    "TurnType",
    "AgentInterface",
    "SimpleAgent",
    "CommunicationManager",
    "TurnManager",
    "create_communication_system",
    
    # Preference components
    "PreferenceType",
    "PreferenceConfig",
    "BasePreferenceSystem",
    "VectorPreferenceSystem",
    "MatrixPreferenceSystem", 
    "PreferenceManager",
    "create_competitive_preferences",
    "create_cooperative_preferences",
    "analyze_preference_competition_level",
    
    # LLM Agent components
    "ModelProvider",
    "ModelType",
    "LLMConfig",
    "AgentResponse", 
    "NegotiationContext",
    "RateLimiter",
    "BaseLLMAgent",
    "AnthropicAgent",
    "OpenAIAgent",
    
    # Agent Factory components
    "AgentConfiguration",
    "ExperimentConfiguration",
    "AgentFactory",
    "create_o3_vs_haiku_experiment",
    "create_cooperative_experiment", 
    "create_scaling_study_experiment",
    "agent_factory",
    
    # Utility Engine components
    "UtilityEngine",
    "UtilityCalculationResult",
    "create_utility_engine",
    "calculate_discounted_utility",
    "compare_discount_factors",
    
    # Negotiation Runner components
    "ModularNegotiationRunner",
    "NegotiationOutcome",
    "NegotiationPhase"
]
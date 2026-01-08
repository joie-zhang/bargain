"""
Base game environment classes and interfaces.

This module defines the abstract interface that all negotiation game environments
must implement, enabling modular game types within the same experiment framework.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class GameType(Enum):
    """Supported game types."""
    ITEM_ALLOCATION = "item_allocation"
    DIPLOMATIC_TREATY = "diplomacy"


@dataclass
class GameConfig:
    """Base configuration for any negotiation game."""
    n_agents: int
    t_rounds: int
    gamma_discount: float = 0.9
    random_seed: Optional[int] = None


@dataclass
class ItemAllocationConfig(GameConfig):
    """Configuration specific to Item Allocation game."""
    m_items: int = 5
    competition_level: float = 0.95  # cosine similarity target


@dataclass
class DiplomaticTreatyConfig(GameConfig):
    """Configuration specific to Diplomatic Treaty game."""
    n_issues: int = 5
    rho: float = 0.0       # Preference correlation [-1, 1]
    theta: float = 0.5     # Interest overlap [0, 1]
    lam: float = 0.0       # Issue compatibility [-1, 1]

    def __post_init__(self):
        """Validate parameter bounds."""
        if not -1 <= self.rho <= 1:
            raise ValueError(f"rho must be in [-1, 1], got {self.rho}")
        if not 0 <= self.theta <= 1:
            raise ValueError(f"theta must be in [0, 1], got {self.theta}")
        if not -1 <= self.lam <= 1:
            raise ValueError(f"lam must be in [-1, 1], got {self.lam}")


class GameEnvironment(ABC):
    """
    Abstract base class for negotiation game environments.

    All game types (Item Allocation, Diplomatic Treaty, etc.) must implement
    this interface to be compatible with the experiment framework.

    The interface separates game-specific logic (prompts, proposals, utilities)
    from the shared negotiation infrastructure (agents, phases, logging).
    """

    def __init__(self, config: GameConfig):
        """
        Initialize the game environment.

        Args:
            config: Game-specific configuration
        """
        self.config = config

    @abstractmethod
    def create_game_state(self, agents: List[Any]) -> Dict[str, Any]:
        """
        Initialize game-specific state (items/issues, preferences, etc.).

        Args:
            agents: List of agent objects with agent_id attributes

        Returns:
            Dictionary containing all game state needed for the negotiation
        """
        pass

    @abstractmethod
    def get_game_rules_prompt(self, game_state: Dict[str, Any]) -> str:
        """
        Generate the game rules explanation prompt.

        Args:
            game_state: Current game state from create_game_state()

        Returns:
            Prompt string explaining game rules to agents
        """
        pass

    @abstractmethod
    def get_preference_assignment_prompt(
        self,
        agent_id: str,
        game_state: Dict[str, Any]
    ) -> str:
        """
        Generate preference assignment prompt for a specific agent.

        Args:
            agent_id: ID of the agent receiving preferences
            game_state: Current game state

        Returns:
            Prompt string with agent's private preferences
        """
        pass

    @abstractmethod
    def get_proposal_prompt(
        self,
        agent_id: str,
        game_state: Dict[str, Any],
        round_num: int,
        agents: List[str]
    ) -> str:
        """
        Generate proposal prompt for a specific agent.

        Args:
            agent_id: ID of the proposing agent
            game_state: Current game state
            round_num: Current round number
            agents: List of all agent IDs

        Returns:
            Prompt string requesting a proposal
        """
        pass

    @abstractmethod
    def parse_proposal(
        self,
        response: str,
        agent_id: str,
        game_state: Dict[str, Any],
        agents: List[str]
    ) -> Dict[str, Any]:
        """
        Parse agent response into a valid proposal.

        Args:
            response: Raw response string from agent
            agent_id: ID of the proposing agent
            game_state: Current game state
            agents: List of all agent IDs

        Returns:
            Parsed proposal dictionary
        """
        pass

    @abstractmethod
    def validate_proposal(
        self,
        proposal: Dict[str, Any],
        game_state: Dict[str, Any]
    ) -> bool:
        """
        Check if a proposal is valid according to game rules.

        Args:
            proposal: Proposal to validate
            game_state: Current game state

        Returns:
            True if proposal is valid, False otherwise
        """
        pass

    @abstractmethod
    def calculate_utility(
        self,
        agent_id: str,
        proposal: Dict[str, Any],
        game_state: Dict[str, Any],
        round_num: int
    ) -> float:
        """
        Calculate utility for an agent given a proposal.

        Args:
            agent_id: ID of the agent
            proposal: The proposal to evaluate
            game_state: Current game state
            round_num: Current round (for discounting)

        Returns:
            Utility value (discounted)
        """
        pass

    @abstractmethod
    def get_discussion_prompt(
        self,
        agent_id: str,
        game_state: Dict[str, Any],
        round_num: int,
        max_rounds: int,
        discussion_history: List[Dict[str, Any]]
    ) -> str:
        """
        Generate discussion prompt.

        Args:
            agent_id: ID of the speaking agent
            game_state: Current game state
            round_num: Current round number
            max_rounds: Total rounds in negotiation
            discussion_history: Previous discussion messages

        Returns:
            Prompt string for discussion phase
        """
        pass

    @abstractmethod
    def get_voting_prompt(
        self,
        agent_id: str,
        proposal: Dict[str, Any],
        game_state: Dict[str, Any],
        round_num: int
    ) -> str:
        """
        Generate voting prompt for a specific proposal.

        Args:
            agent_id: ID of the voting agent
            proposal: The proposal to vote on
            game_state: Current game state
            round_num: Current round number

        Returns:
            Prompt string requesting a vote
        """
        pass

    @abstractmethod
    def get_thinking_prompt(
        self,
        agent_id: str,
        game_state: Dict[str, Any],
        round_num: int,
        max_rounds: int,
        discussion_history: List[Dict[str, Any]]
    ) -> str:
        """
        Generate private thinking prompt.

        Args:
            agent_id: ID of the thinking agent
            game_state: Current game state
            round_num: Current round number
            max_rounds: Total rounds
            discussion_history: Previous discussion messages

        Returns:
            Prompt string for strategic thinking
        """
        pass

    @abstractmethod
    def format_proposal_display(
        self,
        proposal: Dict[str, Any],
        game_state: Dict[str, Any]
    ) -> str:
        """
        Format a proposal for human-readable display.

        Args:
            proposal: The proposal to format
            game_state: Current game state

        Returns:
            Formatted string representation
        """
        pass

    @abstractmethod
    def get_game_type(self) -> GameType:
        """
        Return the game type identifier.

        Returns:
            GameType enum value
        """
        pass

    @abstractmethod
    def get_agent_preferences_summary(
        self,
        agent_id: str,
        game_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get agent preferences in a format suitable for logging/results.

        Args:
            agent_id: ID of the agent
            game_state: Current game state

        Returns:
            Dictionary with preference information
        """
        pass

    def get_reflection_prompt(
        self,
        agent_id: str,
        game_state: Dict[str, Any],
        round_num: int,
        max_rounds: int,
        tabulation_result: Dict[str, Any]
    ) -> str:
        """
        Generate reflection prompt after a round.

        Default implementation provided; override for game-specific reflection.

        Args:
            agent_id: ID of the reflecting agent
            game_state: Current game state
            round_num: Current round number
            max_rounds: Total rounds
            tabulation_result: Results from vote tabulation

        Returns:
            Prompt string for reflection
        """
        return f"""Reflect on the outcome of round {round_num}.
No proposal achieved unanimous acceptance.
Consider what adjustments might lead to consensus in future rounds."""

    def get_contextual_discussion_prompt(
        self,
        base_prompt: str,
        agent_id: str,
        discussion_history: List[str],
        speaker_order: int,
        total_speakers: int
    ) -> str:
        """
        Add context from current round's discussion to base prompt.

        Default implementation provided; override for game-specific context.

        Args:
            base_prompt: Base discussion prompt
            agent_id: ID of the speaking agent
            discussion_history: Messages from current round
            speaker_order: This agent's position in speaking order
            total_speakers: Total number of speakers

        Returns:
            Enhanced prompt with context
        """
        if discussion_history:
            context_section = f"""
**DISCUSSION SO FAR THIS ROUND:**
{len(discussion_history)} agent(s) have already spoken.

**YOUR TURN ({speaker_order}/{total_speakers})**:
Consider what others have said and respond strategically.

"""
        else:
            context_section = f"""**YOU'RE SPEAKING FIRST ({speaker_order}/{total_speakers})**:
Set the tone for this round's discussion.

"""

        return base_prompt + "\n\n" + context_section

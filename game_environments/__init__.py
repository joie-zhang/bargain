"""
Game Environments Package

This package provides modular game environments for multi-agent negotiation research.
Each game type implements the GameEnvironment interface, allowing the experiment
framework to run different negotiation games with the same infrastructure.

Supported game types:
- Item Allocation: Discrete item allocation with preference vectors
- Diplomatic Treaty: Multi-issue continuous negotiation with position/weight preferences

Usage:
    from game_environments import create_game_environment, ItemAllocationConfig

    config = ItemAllocationConfig(n_agents=2, t_rounds=10, m_items=5)
    game = create_game_environment("item_allocation", config)
"""

from .base import (
    GameConfig,
    GameEnvironment,
    GameType,
    ItemAllocationConfig,
    DiplomaticTreatyConfig,
)
from .item_allocation import ItemAllocationGame
from .diplomatic_treaty import DiplomaticTreatyGame


def create_game_environment(
    game_type: str,
    n_agents: int,
    t_rounds: int,
    gamma_discount: float = 0.9,
    random_seed: int = None,
    **kwargs
) -> GameEnvironment:
    """
    Factory function to create game environments.

    Args:
        game_type: Type of game - "item_allocation" or "diplomacy"
        n_agents: Number of agents in the negotiation
        t_rounds: Maximum number of rounds
        gamma_discount: Discount factor per round (default: 0.9)
        random_seed: Random seed for reproducibility
        **kwargs: Game-specific parameters:
            For item_allocation:
                - m_items: Number of items (default: 5)
                - competition_level: Cosine similarity target (default: 0.95)
            For diplomacy:
                - n_issues: Number of issues (default: 5)
                - rho: Preference correlation [-1, 1] (default: 0.0)
                - theta: Interest overlap [0, 1] (default: 0.5)
                - lam: Issue compatibility [-1, 1] (default: 0.0)

    Returns:
        GameEnvironment instance for the specified game type

    Raises:
        ValueError: If game_type is not recognized

    Example:
        # Item Allocation game
        game = create_game_environment(
            game_type="item_allocation",
            n_agents=2,
            t_rounds=10,
            m_items=5,
            competition_level=0.95
        )

        # Diplomatic Treaty game
        game = create_game_environment(
            game_type="diplomacy",
            n_agents=2,
            t_rounds=10,
            n_issues=5,
            rho=0.0,
            theta=0.3,
            lam=0.5
        )
    """
    if game_type == "item_allocation":
        config = ItemAllocationConfig(
            n_agents=n_agents,
            t_rounds=t_rounds,
            gamma_discount=gamma_discount,
            random_seed=random_seed,
            m_items=kwargs.get("m_items", 5),
            competition_level=kwargs.get("competition_level", 0.95)
        )
        return ItemAllocationGame(config)

    elif game_type == "diplomacy":
        config = DiplomaticTreatyConfig(
            n_agents=n_agents,
            t_rounds=t_rounds,
            gamma_discount=gamma_discount,
            random_seed=random_seed,
            n_issues=kwargs.get("n_issues", 5),
            rho=kwargs.get("rho", 0.0),
            theta=kwargs.get("theta", 0.5),
            lam=kwargs.get("lam", 0.0)
        )
        return DiplomaticTreatyGame(config)

    else:
        valid_types = [gt.value for gt in GameType]
        raise ValueError(
            f"Unknown game type: '{game_type}'. "
            f"Valid types are: {valid_types}"
        )


def create_game_from_config(config: GameConfig) -> GameEnvironment:
    """
    Create a game environment from an existing config object.

    Args:
        config: A GameConfig subclass instance (ItemAllocationConfig or DiplomaticTreatyConfig)

    Returns:
        GameEnvironment instance matching the config type

    Raises:
        ValueError: If config type is not recognized
    """
    if isinstance(config, ItemAllocationConfig):
        return ItemAllocationGame(config)
    elif isinstance(config, DiplomaticTreatyConfig):
        return DiplomaticTreatyGame(config)
    else:
        raise ValueError(
            f"Unknown config type: {type(config).__name__}. "
            f"Expected ItemAllocationConfig or DiplomaticTreatyConfig."
        )


__all__ = [
    # Base classes and types
    "GameEnvironment",
    "GameConfig",
    "GameType",
    "ItemAllocationConfig",
    "DiplomaticTreatyConfig",
    # Game implementations
    "ItemAllocationGame",
    "DiplomaticTreatyGame",
    # Factory functions
    "create_game_environment",
    "create_game_from_config",
]

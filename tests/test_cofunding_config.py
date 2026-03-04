#!/usr/bin/env python3
"""
Tests for CoFundingConfig validation and get_protocol_type() default.
"""

import pytest

from game_environments.base import (
    CoFundingConfig,
    GameConfig,
    GameEnvironment,
    ItemAllocationConfig,
)


class TestCoFundingConfig:
    """Tests for CoFundingConfig validation."""

    def test_valid_config_defaults(self):
        """Default config creates successfully."""
        config = CoFundingConfig(n_agents=2, t_rounds=5)
        assert config.m_projects == 5
        assert config.alpha == 0.5
        assert config.sigma == 0.5
        assert config.c_min == 10.0
        assert config.c_max == 30.0
        assert config.discussion_transparency == "own"
        assert config.enable_commit_vote is True
        assert config.enable_time_discount is True
        assert config.gamma_discount == 0.9

    def test_valid_config_custom(self):
        """Custom valid config creates successfully."""
        config = CoFundingConfig(
            n_agents=3,
            t_rounds=10,
            m_projects=8,
            alpha=0.8,
            sigma=0.3,
            c_min=5.0,
            c_max=100.0,
        )
        assert config.m_projects == 8
        assert config.alpha == 0.8
        assert config.sigma == 0.3

    def test_alpha_boundary_zero(self):
        """alpha=0 (orthogonal) is valid."""
        config = CoFundingConfig(n_agents=2, t_rounds=5, alpha=0.0)
        assert config.alpha == 0.0

    def test_alpha_boundary_one(self):
        """alpha=1 (identical) is valid."""
        config = CoFundingConfig(n_agents=2, t_rounds=5, alpha=1.0)
        assert config.alpha == 1.0

    def test_alpha_too_low(self):
        """alpha < 0 raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be in"):
            CoFundingConfig(n_agents=2, t_rounds=5, alpha=-0.1)

    def test_alpha_too_high(self):
        """alpha > 1 raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be in"):
            CoFundingConfig(n_agents=2, t_rounds=5, alpha=1.5)

    def test_sigma_boundary_one(self):
        """sigma=1 (can afford everything) is valid."""
        config = CoFundingConfig(n_agents=2, t_rounds=5, sigma=1.0)
        assert config.sigma == 1.0

    def test_sigma_small_positive(self):
        """sigma=0.01 (very scarce) is valid."""
        config = CoFundingConfig(n_agents=2, t_rounds=5, sigma=0.01)
        assert config.sigma == 0.01

    def test_sigma_zero_raises(self):
        """sigma=0 raises ValueError (strict inequality)."""
        with pytest.raises(ValueError, match="sigma must be in"):
            CoFundingConfig(n_agents=2, t_rounds=5, sigma=0.0)

    def test_sigma_negative_raises(self):
        """sigma < 0 raises ValueError."""
        with pytest.raises(ValueError, match="sigma must be in"):
            CoFundingConfig(n_agents=2, t_rounds=5, sigma=-0.5)

    def test_sigma_too_high(self):
        """sigma > 1 raises ValueError."""
        with pytest.raises(ValueError, match="sigma must be in"):
            CoFundingConfig(n_agents=2, t_rounds=5, sigma=1.5)

    def test_c_min_positive(self):
        """c_min must be positive."""
        with pytest.raises(ValueError, match="c_min must be positive"):
            CoFundingConfig(n_agents=2, t_rounds=5, c_min=0.0)

    def test_c_max_less_than_c_min(self):
        """c_max < c_min raises ValueError."""
        with pytest.raises(ValueError, match="c_max must be >= c_min"):
            CoFundingConfig(n_agents=2, t_rounds=5, c_min=50.0, c_max=10.0)

    def test_m_projects_zero(self):
        """m_projects < 1 raises ValueError."""
        with pytest.raises(ValueError, match="m_projects must be >= 1"):
            CoFundingConfig(n_agents=2, t_rounds=5, m_projects=0)

    def test_invalid_discussion_transparency(self):
        """Invalid discussion transparency mode should raise ValueError."""
        with pytest.raises(ValueError, match="discussion_transparency must be"):
            CoFundingConfig(
                n_agents=2,
                t_rounds=5,
                discussion_transparency="invalid_mode"
            )

    def test_invalid_commit_vote_type(self):
        """Commit vote toggle must be bool."""
        with pytest.raises(ValueError, match="enable_commit_vote must be bool"):
            CoFundingConfig(n_agents=2, t_rounds=5, enable_commit_vote="yes")

    def test_invalid_time_discount_toggle_type(self):
        """Time discount toggle must be bool."""
        with pytest.raises(ValueError, match="enable_time_discount must be bool"):
            CoFundingConfig(n_agents=2, t_rounds=5, enable_time_discount="yes")

    def test_invalid_gamma_discount(self):
        """gamma_discount must be in [0,1]."""
        with pytest.raises(ValueError, match="gamma_discount must be in"):
            CoFundingConfig(n_agents=2, t_rounds=5, gamma_discount=1.2)


class TestGetProtocolType:
    """Tests for get_protocol_type() default on base class."""

    def test_item_allocation_default_protocol(self):
        """ItemAllocation should use propose_and_vote."""
        from game_environments import ItemAllocationGame

        config = ItemAllocationConfig(n_agents=2, t_rounds=5, m_items=3)
        game = ItemAllocationGame(config)
        assert game.get_protocol_type() == "propose_and_vote"

    def test_diplomacy_default_protocol(self):
        """DiplomaticTreaty should use propose_and_vote."""
        from game_environments import DiplomaticTreatyGame, DiplomaticTreatyConfig

        config = DiplomaticTreatyConfig(n_agents=2, t_rounds=5)
        game = DiplomaticTreatyGame(config)
        assert game.get_protocol_type() == "propose_and_vote"

"""
Tests for the preference system implementation.

This module tests both vector and matrix preference systems,
including utility calculations, preference generation, and analysis tools.
"""

import pytest
import numpy as np
from negotiation.preferences import (
    PreferenceType,
    PreferenceConfig,
    VectorPreferenceSystem,
    MatrixPreferenceSystem,
    PreferenceManager,
    create_competitive_preferences,
    create_cooperative_preferences,
    analyze_preference_competition_level
)


class TestPreferenceConfig:
    """Test PreferenceConfig validation and initialization."""
    
    def test_valid_config(self):
        """Test valid configuration creation."""
        config = PreferenceConfig(
            preference_type=PreferenceType.VECTOR,
            m_items=5,
            n_agents=3,
            min_value=0.0,
            max_value=10.0,
            random_seed=42
        )
        
        assert config.preference_type == PreferenceType.VECTOR
        assert config.m_items == 5
        assert config.n_agents == 3
        assert config.min_value == 0.0
        assert config.max_value == 10.0
        assert config.random_seed == 42
    
    def test_invalid_items(self):
        """Test validation for invalid number of items."""
        with pytest.raises(ValueError, match="Must have at least 1 item"):
            PreferenceConfig(
                preference_type=PreferenceType.VECTOR,
                m_items=0,
                n_agents=3
            )
    
    def test_invalid_agents(self):
        """Test validation for invalid number of agents."""
        with pytest.raises(ValueError, match="Need at least 2 agents"):
            PreferenceConfig(
                preference_type=PreferenceType.VECTOR,
                m_items=5,
                n_agents=1
            )
    
    def test_invalid_value_range(self):
        """Test validation for invalid value range."""
        with pytest.raises(ValueError, match="min_value must be <= max_value"):
            PreferenceConfig(
                preference_type=PreferenceType.VECTOR,
                m_items=5,
                n_agents=3,
                min_value=10.0,
                max_value=5.0
            )
    
    def test_invalid_cosine_similarity(self):
        """Test validation for invalid cosine similarity."""
        with pytest.raises(ValueError, match="Cosine similarity must be between -1 and 1"):
            PreferenceConfig(
                preference_type=PreferenceType.VECTOR,
                m_items=5,
                n_agents=3,
                target_cosine_similarity=1.5
            )
    
    def test_invalid_cooperation_factor(self):
        """Test validation for invalid cooperation factor."""
        with pytest.raises(ValueError, match="Cooperation factor must be between 0 and 1"):
            PreferenceConfig(
                preference_type=PreferenceType.MATRIX,
                m_items=5,
                n_agents=3,
                cooperation_factor=1.5
            )


class TestVectorPreferenceSystem:
    """Test vector preference system."""
    
    def test_random_preference_generation(self):
        """Test random preference vector generation."""
        config = PreferenceConfig(
            preference_type=PreferenceType.VECTOR,
            m_items=5,
            n_agents=3,
            random_seed=42
        )
        
        system = VectorPreferenceSystem(config)
        preferences = system.generate_preferences()
        
        assert preferences["type"] == "vector"
        assert len(preferences["agent_preferences"]) == 3
        
        # Check each agent has m_items preferences
        for agent_id, prefs in preferences["agent_preferences"].items():
            assert len(prefs) == 5
            assert all(0.0 <= p <= 10.0 for p in prefs)
        
        # Check cosine similarities are calculated
        assert "cosine_similarities" in preferences
    
    def test_targeted_similarity_generation(self):
        """Test preference generation with target cosine similarity."""
        config = PreferenceConfig(
            preference_type=PreferenceType.VECTOR,
            m_items=5,
            n_agents=3,
            target_cosine_similarity=0.9,
            random_seed=42
        )
        
        system = VectorPreferenceSystem(config)
        preferences = system.generate_preferences()
        
        # Check similarities are close to target
        similarities = list(preferences["cosine_similarities"].values())
        for sim in similarities:
            assert abs(sim - 0.9) < 0.2  # Allow some tolerance
    
    def test_utility_calculation(self):
        """Test vector preference utility calculation."""
        config = PreferenceConfig(
            preference_type=PreferenceType.VECTOR,
            m_items=3,
            n_agents=2,
            random_seed=42
        )
        
        system = VectorPreferenceSystem(config)
        
        # Create test preferences
        preferences = {
            "type": "vector",
            "agent_preferences": {
                "agent_0": [5.0, 3.0, 8.0],
                "agent_1": [2.0, 7.0, 4.0]
            }
        }
        
        # Test allocation: agent_0 gets items 0,2, agent_1 gets item 1
        allocation = {
            "agent_0": [0, 2],
            "agent_1": [1]
        }
        
        utility_0 = system.calculate_utility("agent_0", allocation, preferences)
        utility_1 = system.calculate_utility("agent_1", allocation, preferences)
        
        assert utility_0 == 5.0 + 8.0  # Items 0 and 2
        assert utility_1 == 7.0  # Item 1
    
    def test_get_agent_preferences(self):
        """Test getting preferences for specific agent."""
        config = PreferenceConfig(
            preference_type=PreferenceType.VECTOR,
            m_items=3,
            n_agents=2
        )
        
        system = VectorPreferenceSystem(config)
        preferences = {
            "agent_preferences": {
                "agent_0": [5.0, 3.0, 8.0],
                "agent_1": [2.0, 7.0, 4.0]
            }
        }
        
        agent_0_prefs = system.get_agent_preferences("agent_0", preferences)
        assert agent_0_prefs == [5.0, 3.0, 8.0]


class TestMatrixPreferenceSystem:
    """Test matrix preference system."""
    
    def test_preference_generation(self):
        """Test matrix preference generation."""
        config = PreferenceConfig(
            preference_type=PreferenceType.MATRIX,
            m_items=3,
            n_agents=2,
            cooperation_factor=0.5,
            random_seed=42
        )
        
        system = MatrixPreferenceSystem(config)
        preferences = system.generate_preferences()
        
        assert preferences["type"] == "matrix"
        assert len(preferences["agent_preferences"]) == 2
        
        # Check each agent has m×n matrix
        for agent_id, matrix in preferences["agent_preferences"].items():
            assert len(matrix) == 3  # m_items rows
            assert len(matrix[0]) == 2  # n_agents columns
    
    def test_utility_calculation(self):
        """Test matrix preference utility calculation."""
        config = PreferenceConfig(
            preference_type=PreferenceType.MATRIX,
            m_items=3,
            n_agents=2
        )
        
        system = MatrixPreferenceSystem(config)
        
        # Create test preferences: agent_0's matrix
        preferences = {
            "type": "matrix",
            "agent_preferences": {
                "agent_0": [
                    [5.0, 2.0],  # Item 0: values agent_0=5, agent_1=2
                    [3.0, 4.0],  # Item 1: values agent_0=3, agent_1=4
                    [8.0, 1.0]   # Item 2: values agent_0=8, agent_1=1
                ]
            }
        }
        
        # Test allocation: agent_0 gets items 0,2, agent_1 gets item 1
        allocation = {
            "agent_0": [0, 2],
            "agent_1": [1]
        }
        
        utility = system.calculate_utility("agent_0", allocation, preferences)
        
        # Expected: 5.0 (agent_0 gets item 0) + 8.0 (agent_0 gets item 2) + 4.0 (agent_1 gets item 1)
        expected = 5.0 + 8.0 + 4.0
        assert utility == expected


class TestPreferenceManager:
    """Test PreferenceManager functionality."""
    
    def test_vector_preference_manager(self):
        """Test preference manager with vector preferences."""
        config = PreferenceConfig(
            preference_type=PreferenceType.VECTOR,
            m_items=3,
            n_agents=2,
            random_seed=42
        )
        
        manager = PreferenceManager(config)
        preferences = manager.generate_preferences()
        
        assert preferences["type"] == "vector"
        assert len(preferences["agent_preferences"]) == 2
    
    def test_matrix_preference_manager(self):
        """Test preference manager with matrix preferences."""
        config = PreferenceConfig(
            preference_type=PreferenceType.MATRIX,
            m_items=3,
            n_agents=2,
            random_seed=42
        )
        
        manager = PreferenceManager(config)
        preferences = manager.generate_preferences()
        
        assert preferences["type"] == "matrix"
        assert len(preferences["agent_preferences"]) == 2
    
    def test_calculate_all_utilities(self):
        """Test utility calculation for all agents."""
        config = PreferenceConfig(
            preference_type=PreferenceType.VECTOR,
            m_items=3,
            n_agents=2
        )
        
        manager = PreferenceManager(config)
        
        preferences = {
            "type": "vector",
            "agent_preferences": {
                "agent_0": [5.0, 3.0, 8.0],
                "agent_1": [2.0, 7.0, 4.0]
            }
        }
        
        allocation = {
            "agent_0": [0, 2],
            "agent_1": [1]
        }
        
        utilities = manager.calculate_all_utilities(allocation, preferences)
        
        assert utilities["agent_0"] == 13.0  # 5.0 + 8.0
        assert utilities["agent_1"] == 7.0   # 7.0
    
    def test_competitive_scenario_detection(self):
        """Test detection of competitive scenarios."""
        config = PreferenceConfig(
            preference_type=PreferenceType.VECTOR,
            m_items=3,
            n_agents=2
        )
        
        manager = PreferenceManager(config)
        
        # High similarity preferences (competitive)
        competitive_prefs = {
            "type": "vector",
            "cosine_similarities": {"agent_0_vs_agent_1": 0.9}
        }
        
        # Low similarity preferences (less competitive)
        cooperative_prefs = {
            "type": "vector", 
            "cosine_similarities": {"agent_0_vs_agent_1": 0.3}
        }
        
        assert manager.is_competitive_scenario(competitive_prefs, threshold=0.7)
        assert not manager.is_competitive_scenario(cooperative_prefs, threshold=0.7)


class TestFactoryFunctions:
    """Test factory functions for creating preference systems."""
    
    def test_create_competitive_preferences(self):
        """Test competitive preference factory function."""
        manager = create_competitive_preferences(
            m_items=5,
            n_agents=3,
            cosine_similarity=0.9,
            random_seed=42
        )
        
        preferences = manager.generate_preferences()
        
        assert preferences["type"] == "vector"
        assert len(preferences["agent_preferences"]) == 3
        
        # Check that similarities are high (competitive)
        similarities = list(preferences["cosine_similarities"].values())
        assert all(sim > 0.7 for sim in similarities)
    
    def test_create_cooperative_preferences(self):
        """Test cooperative preference factory function."""
        manager = create_cooperative_preferences(
            m_items=4,
            n_agents=2,
            cooperation_factor=0.8,
            random_seed=42
        )
        
        preferences = manager.generate_preferences()
        
        assert preferences["type"] == "matrix"
        assert len(preferences["agent_preferences"]) == 2
        assert preferences["config"]["cooperation_factor"] == 0.8


class TestPreferenceAnalysis:
    """Test preference analysis functions."""
    
    def test_analyze_vector_preferences(self):
        """Test analysis of vector preferences."""
        preferences = {
            "type": "vector",
            "cosine_similarities": {
                "agent_0_vs_agent_1": 0.9,
                "agent_0_vs_agent_2": 0.8,
                "agent_1_vs_agent_2": 0.85
            }
        }
        
        analysis = analyze_preference_competition_level(preferences)
        
        assert analysis["preference_type"] == "vector"
        assert analysis["competition_level"] == "high"
        assert abs(analysis["avg_cosine_similarity"] - 0.85) < 0.01
    
    def test_analyze_matrix_preferences(self):
        """Test analysis of matrix preferences."""
        preferences = {
            "type": "matrix",
            "config": {"cooperation_factor": 0.2}
        }
        
        analysis = analyze_preference_competition_level(preferences)
        
        assert analysis["preference_type"] == "matrix"
        assert analysis["competition_level"] == "high"
        assert analysis["cooperation_factor"] == 0.2


class TestReproducibility:
    """Test reproducibility with random seeds."""
    
    def test_vector_preferences_reproducible(self):
        """Test that vector preferences are reproducible with same seed."""
        config = PreferenceConfig(
            preference_type=PreferenceType.VECTOR,
            m_items=5,
            n_agents=3,
            random_seed=42
        )
        
        system1 = VectorPreferenceSystem(config)
        system2 = VectorPreferenceSystem(config)
        
        prefs1 = system1.generate_preferences()
        prefs2 = system2.generate_preferences()
        
        # Should be identical
        assert prefs1["agent_preferences"] == prefs2["agent_preferences"]
    
    def test_matrix_preferences_reproducible(self):
        """Test that matrix preferences are reproducible with same seed."""
        config = PreferenceConfig(
            preference_type=PreferenceType.MATRIX,
            m_items=3,
            n_agents=2,
            random_seed=42
        )
        
        system1 = MatrixPreferenceSystem(config)
        system2 = MatrixPreferenceSystem(config)
        
        prefs1 = system1.generate_preferences()
        prefs2 = system2.generate_preferences()
        
        # Should be identical (convert to numpy for comparison)
        for agent_id in prefs1["agent_preferences"]:
            matrix1 = np.array(prefs1["agent_preferences"][agent_id])
            matrix2 = np.array(prefs2["agent_preferences"][agent_id])
            assert np.allclose(matrix1, matrix2)


if __name__ == "__main__":
    pytest.main([__file__])
#!/usr/bin/env python3
"""
Test Suite for Parameterized Experiment System

Tests the full parameterization system including:
- Configuration loading and validation
- Environment parameter configuration (n, m, t, γ)
- Model configuration system
- Preference system selection (vector vs matrix)
- Competition level configuration
- Proposal order analysis
"""

import pytest
import asyncio
import yaml
import json
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, AsyncMock, patch

# Import our parameterized experiment system
from negotiation.experiment_config import (
    ExperimentConfigManager, ExperimentConfig, ConfigurationError,
    PreferenceSystemType, StrategicLevel, ReasoningCapability
)
from experiments.parameterized_experiment import (
    ParameterizedExperimentRunner,
    PreferenceSystemManager,
    ProposalOrderAnalyzer
)


class TestExperimentConfigManager:
    """Test the experiment configuration manager."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ExperimentConfigManager(config_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_load_basic_config(self):
        """Test loading a basic configuration file."""
        # Create a minimal config file
        config_data = {
            "schema_version": "1.0",
            "description": "Test configuration",
            "environment": {
                "n_agents": 3,
                "m_items": 5,
                "t_rounds": 6,
                "gamma_discount": 0.9
            },
            "models": {
                "providers": {
                    "test_provider": {
                        "provider": "test",
                        "api_key_env": "TEST_API_KEY"
                    }
                },
                "available_models": {
                    "test_model": {
                        "display_name": "Test Model",
                        "family": "test",
                        "provider": "test_provider",
                        "api_model_name": "test-model",
                        "context_window": 4096,
                        "reasoning_capability": "medium"
                    }
                }
            },
            "agents": [
                {
                    "agent_id": "test_agent",
                    "model_id": "test_model"
                }
            ]
        }
        
        config_path = Path(self.temp_dir) / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        # Load and validate the configuration
        config = self.config_manager.load_config(config_path)
        
        assert config.schema_version == "1.0"
        assert config.environment.n_agents == 3
        assert config.environment.m_items == 5
        assert config.environment.t_rounds == 6
        assert config.environment.gamma_discount == 0.9
        assert len(config.agents) == 1
        assert config.agents[0].agent_id == "test_agent"
    
    def test_environment_parameter_configuration(self):
        """Test that environment parameters (n, m, t, γ) are properly configured."""
        config_data = {
            "environment": {
                "n_agents": 4,
                "m_items": 8,
                "t_rounds": 10,
                "gamma_discount": 0.95,
                "randomized_proposal_order": False
            }
        }
        
        config = self.config_manager._parse_config_dict(config_data)
        
        assert config.environment.n_agents == 4
        assert config.environment.m_items == 8  
        assert config.environment.t_rounds == 10
        assert config.environment.gamma_discount == 0.95
        assert config.environment.randomized_proposal_order == False
    
    def test_model_configuration_system(self):
        """Test the model configuration system with multiple providers."""
        config_data = {
            "models": {
                "providers": {
                    "openai": {
                        "provider": "openai",
                        "requests_per_minute": 30,
                        "tokens_per_minute": 50000
                    },
                    "anthropic": {
                        "provider": "anthropic", 
                        "requests_per_minute": 60,
                        "tokens_per_minute": 100000
                    }
                },
                "available_models": {
                    "gpt-4o": {
                        "display_name": "GPT-4o",
                        "family": "gpt4",
                        "provider": "openai",
                        "api_model_name": "gpt-4o",
                        "context_window": 128000,
                        "reasoning_capability": "high"
                    },
                    "claude-haiku": {
                        "display_name": "Claude Haiku",
                        "family": "claude",
                        "provider": "anthropic",
                        "api_model_name": "claude-3-haiku",
                        "context_window": 200000,
                        "reasoning_capability": "medium"
                    }
                }
            }
        }
        
        config = self.config_manager._parse_config_dict(config_data)
        
        # Test provider configuration
        assert "openai" in config.providers
        assert "anthropic" in config.providers
        assert config.providers["openai"].requests_per_minute == 30
        assert config.providers["anthropic"].tokens_per_minute == 100000
        
        # Test model configuration
        assert "gpt-4o" in config.models
        assert "claude-haiku" in config.models
        assert config.models["gpt-4o"].reasoning_capability == ReasoningCapability.HIGH
        assert config.models["claude-haiku"].context_window == 200000
    
    def test_preference_system_selection(self):
        """Test vector vs matrix preference system configuration."""
        # Test vector preferences
        vector_config = {
            "preferences": {
                "system_type": "vector",
                "vector_preferences": {
                    "value_range": [0, 10],
                    "distribution": "uniform"
                },
                "competition_level": {
                    "cosine_similarity": 0.95
                }
            }
        }
        
        config = self.config_manager._parse_config_dict(vector_config)
        assert config.preferences.system_type == PreferenceSystemType.VECTOR
        assert config.preferences.value_range == (0, 10)
        assert config.preferences.cosine_similarity == 0.95
        
        # Test matrix preferences
        matrix_config = {
            "preferences": {
                "system_type": "matrix",
                "matrix_preferences": {
                    "self_weight": 0.6,
                    "others_weight": 0.4
                }
            }
        }
        
        config = self.config_manager._parse_config_dict(matrix_config)
        assert config.preferences.system_type == PreferenceSystemType.MATRIX
        assert config.preferences.self_weight == 0.6
        assert config.preferences.others_weight == 0.4
    
    def test_competition_level_configuration(self):
        """Test cosine similarity configuration for competition levels."""
        test_cases = [
            {"cosine_similarity": 0.1, "expected_competition": "low"},
            {"cosine_similarity": 0.5, "expected_competition": "medium"},
            {"cosine_similarity": 0.95, "expected_competition": "high"}
        ]
        
        for case in test_cases:
            config_data = {
                "preferences": {
                    "competition_level": {
                        "cosine_similarity": case["cosine_similarity"],
                        "tolerance": 0.05
                    }
                }
            }
            
            config = self.config_manager._parse_config_dict(config_data)
            assert config.preferences.cosine_similarity == case["cosine_similarity"]
            assert config.preferences.tolerance == 0.05
    
    def test_config_validation(self):
        """Test configuration validation system."""
        # Test valid configuration
        valid_config = ExperimentConfig()
        valid_config.environment.n_agents = 3
        valid_config.environment.m_items = 5
        valid_config.agents = [
            Mock(agent_id="agent1", model_id="test_model", temperature=0.7, max_output_tokens=2000),
            Mock(agent_id="agent2", model_id="test_model", temperature=0.7, max_output_tokens=2000),
            Mock(agent_id="agent3", model_id="test_model", temperature=0.7, max_output_tokens=2000)
        ]
        valid_config.models = {
            "test_model": Mock(supports_system_prompt=True)
        }
        
        # Should not raise exception
        self.config_manager.validate_config(valid_config)
        
        # Test invalid configuration - wrong number of agents
        invalid_config = ExperimentConfig()
        invalid_config.environment.n_agents = 3
        invalid_config.agents = [Mock()]  # Only 1 agent
        
        with pytest.raises(ConfigurationError):
            self.config_manager.validate_config(invalid_config)
    
    def test_cost_estimation(self):
        """Test cost estimation functionality."""
        config = ExperimentConfig()
        config.environment.t_rounds = 6
        config.agents = [
            Mock(model_id="expensive_model"),
            Mock(model_id="cheap_model")
        ]
        config.models = {
            "expensive_model": Mock(input_cost_per_1m=50.0, output_cost_per_1m=150.0),
            "cheap_model": Mock(input_cost_per_1m=1.0, output_cost_per_1m=3.0)
        }
        
        estimated_cost = self.config_manager.estimate_cost(config)
        assert estimated_cost > 0
        assert isinstance(estimated_cost, float)


class TestPreferenceSystemManager:
    """Test the preference system manager."""
    
    def setup_method(self):
        """Set up test environment."""
        self.preference_manager = PreferenceSystemManager()
    
    def test_vector_preference_generation(self):
        """Test vector preference generation with cosine similarity control."""
        # Create a mock config
        config = Mock()
        config.preferences.system_type = PreferenceSystemType.VECTOR
        config.preferences.cosine_similarity = 0.95
        config.preferences.tolerance = 0.05
        config.preferences.value_range = (0, 10)
        config.preferences.distribution = "uniform"
        config.preferences.max_generation_attempts = 50
        config.environment.n_agents = 3
        config.environment.m_items = 5
        config.agents = [
            Mock(agent_id="agent1"),
            Mock(agent_id="agent2"),
            Mock(agent_id="agent3")
        ]
        
        preferences, actual_similarity = self.preference_manager.generate_preferences(config)
        
        assert preferences["type"] == "vector"
        assert "preferences" in preferences
        assert len(preferences["preferences"]) == 3
        assert "agent1" in preferences["preferences"]
        assert len(preferences["preferences"]["agent1"]) == 5  # m_items
        assert abs(actual_similarity - 0.95) <= 0.1  # Within reasonable tolerance
    
    def test_matrix_preference_generation(self):
        """Test matrix preference generation."""
        config = Mock()
        config.preferences.system_type = PreferenceSystemType.MATRIX
        config.preferences.value_range = (1, 10)
        config.preferences.self_weight = 0.7
        config.preferences.others_weight = 0.3
        config.environment.n_agents = 3
        config.environment.m_items = 5
        config.agents = [
            Mock(agent_id="agent1"),
            Mock(agent_id="agent2"),
            Mock(agent_id="agent3")
        ]
        
        preferences, actual_similarity = self.preference_manager.generate_preferences(config)
        
        assert preferences["type"] == "matrix"
        assert "preferences" in preferences
        assert len(preferences["preferences"]) == 3
        assert "agent1" in preferences["preferences"]
        # Matrix should be m_items x n_agents
        assert len(preferences["preferences"]["agent1"]) == 5  # m_items
        assert len(preferences["preferences"]["agent1"][0]) == 3  # n_agents
        assert preferences["self_weight"] == 0.7
        assert preferences["others_weight"] == 0.3
    
    def test_cosine_similarity_calculation(self):
        """Test cosine similarity calculation."""
        # Test with known vectors
        vectors = [
            [1.0, 0.0, 0.0],  # Orthogonal vectors
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
        
        similarity = self.preference_manager._calculate_mean_cosine_similarity(vectors)
        assert abs(similarity - 0.0) < 0.001  # Should be close to 0 for orthogonal vectors
        
        # Test with identical vectors
        identical_vectors = [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ]
        
        similarity = self.preference_manager._calculate_mean_cosine_similarity(identical_vectors)
        assert abs(similarity - 1.0) < 0.001  # Should be close to 1 for identical vectors


class TestProposalOrderAnalyzer:
    """Test the proposal order analyzer."""
    
    def setup_method(self):
        """Set up test environment."""
        self.analyzer = ProposalOrderAnalyzer()
    
    def test_order_effect_analysis_insufficient_data(self):
        """Test that analyzer handles insufficient data gracefully."""
        # Create minimal results data
        mock_results = []
        for i in range(5):  # Less than minimum 10
            result = Mock()
            result.proposal_orders = [["agent1", "agent2", "agent3"]]
            result.winner_agent_id = "agent1"
            result.final_utilities = {"agent1": 10, "agent2": 5, "agent3": 3}
            result.experiment_id = f"exp_{i}"
            mock_results.append(result)
        
        analysis = self.analyzer.analyze_order_effects(mock_results)
        assert analysis["insufficient_data"] == True
    
    def test_order_effect_analysis_sufficient_data(self):
        """Test order effect analysis with sufficient data."""
        mock_results = []
        for i in range(15):  # Sufficient data
            result = Mock()
            # Vary the order and outcomes
            if i % 3 == 0:
                result.proposal_orders = [["agent1", "agent2", "agent3"]]
                result.winner_agent_id = "agent1"
            elif i % 3 == 1:
                result.proposal_orders = [["agent2", "agent1", "agent3"]]
                result.winner_agent_id = "agent2"
            else:
                result.proposal_orders = [["agent3", "agent1", "agent2"]]
                result.winner_agent_id = "agent3"
            
            result.final_utilities = {
                "agent1": 10 if result.winner_agent_id == "agent1" else 5,
                "agent2": 10 if result.winner_agent_id == "agent2" else 5,
                "agent3": 10 if result.winner_agent_id == "agent3" else 5
            }
            result.experiment_id = f"exp_{i}"
            mock_results.append(result)
        
        analysis = self.analyzer.analyze_order_effects(mock_results)
        
        assert analysis["sufficient_data"] == True
        assert "total_observations" in analysis
        assert "position_bias_stats" in analysis
        assert "position_win_correlation" in analysis
        assert analysis["total_observations"] > 0


class TestParameterizedExperimentRunner:
    """Test the main experiment runner."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.runner = ParameterizedExperimentRunner(
            results_dir=str(Path(self.temp_dir) / "results"),
            logs_dir=str(Path(self.temp_dir) / "logs")
        )
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_config_loading_and_validation(self):
        """Test that the runner can load and validate configurations."""
        # Create a valid config dictionary
        config_dict = {
            "schema_version": "1.0",
            "environment": {
                "n_agents": 3,
                "m_items": 5,
                "t_rounds": 6,
                "gamma_discount": 0.9
            },
            "models": {
                "providers": {
                    "test": {"provider": "test"}
                },
                "available_models": {
                    "test_model": {
                        "display_name": "Test Model",
                        "family": "test",
                        "provider": "test",
                        "api_model_name": "test-model",
                        "context_window": 4096,
                        "reasoning_capability": "medium"
                    }
                }
            },
            "agents": [
                {"agent_id": "agent1", "model_id": "test_model"},
                {"agent_id": "agent2", "model_id": "test_model"},
                {"agent_id": "agent3", "model_id": "test_model"}
            ]
        }
        
        # Parse the configuration
        config = self.runner.config_manager._parse_config_dict(config_dict)
        
        # Validate it
        self.runner.config_manager.validate_config(config)
        
        assert config.environment.n_agents == 3
        assert len(config.agents) == 3
    
    @patch('experiments.parameterized_experiment.ParameterizedExperimentRunner._create_configured_agents')
    @patch('experiments.parameterized_experiment.ParameterizedExperimentRunner._create_configured_environment')
    @patch('experiments.parameterized_experiment.ParameterizedExperimentRunner._run_parameterized_negotiation')
    @patch('experiments.parameterized_experiment.ParameterizedExperimentRunner._analyze_experiment_results')
    @patch('experiments.parameterized_experiment.ParameterizedExperimentRunner._save_experiment_results')
    async def test_single_experiment_execution(self, mock_save, mock_analyze, mock_negotiate, mock_env, mock_agents):
        """Test single experiment execution flow."""
        # Mock the async methods
        mock_agents.return_value = [Mock(agent_id="agent1"), Mock(agent_id="agent2")]
        mock_env.return_value = Mock()
        mock_negotiate.return_value = {
            "consensus_reached": True,
            "final_round": 4,
            "winner_agent_id": "agent1",
            "final_utilities": {"agent1": 10, "agent2": 5},
            "proposal_orders": [["agent1", "agent2"]],
            "conversation_logs": []
        }
        mock_analyze.return_value = {
            "strategic_behaviors": {},
            "agent_performance": {},
            "model_performance": {},
            "exploitation_evidence": {},
            "order_correlation": 0.0
        }
        mock_save.return_value = None
        
        # Create simple config
        config_dict = {
            "environment": {"n_agents": 2, "m_items": 3, "t_rounds": 5, "gamma_discount": 0.9},
            "models": {
                "providers": {"test": {"provider": "test"}},
                "available_models": {
                    "test_model": {
                        "display_name": "Test",
                        "family": "test", 
                        "provider": "test",
                        "api_model_name": "test",
                        "context_window": 4096,
                        "reasoning_capability": "medium"
                    }
                }
            },
            "agents": [
                {"agent_id": "agent1", "model_id": "test_model"},
                {"agent_id": "agent2", "model_id": "test_model"}
            ],
            "execution": {"random_seed": 42}
        }
        
        # Run the experiment
        result = await self.runner.run_single_experiment(config_dict)
        
        # Verify result structure
        assert result.consensus_reached == True
        assert result.final_round == 4
        assert result.winner_agent_id == "agent1"
        assert result.final_utilities == {"agent1": 10, "agent2": 5}
        assert len(result.proposal_orders) == 1
        assert result.experiment_id is not None
    
    def test_cost_estimation_integration(self):
        """Test that cost estimation works with the full system."""
        config_dict = {
            "environment": {"t_rounds": 6},
            "models": {
                "available_models": {
                    "expensive": {
                        "input_cost_per_1m": 60.0,
                        "output_cost_per_1m": 180.0,
                        "display_name": "Expensive",
                        "family": "test",
                        "provider": "test",
                        "api_model_name": "expensive",
                        "context_window": 4096,
                        "reasoning_capability": "high"
                    }
                }
            },
            "agents": [{"agent_id": "agent1", "model_id": "expensive"}]
        }
        
        config = self.runner.config_manager._parse_config_dict(config_dict)
        estimated_cost = self.runner.config_manager.estimate_cost(config)
        
        assert estimated_cost > 0
        # High-cost model should produce significant cost estimate
        assert estimated_cost > 1.0


class TestIntegrationScenarios:
    """Integration tests for complete experiment scenarios."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_o3_vs_haiku_baseline_config_loading(self):
        """Test loading the actual O3 vs Haiku baseline configuration."""
        config_manager = ExperimentConfigManager()
        
        # Test that we can load the actual config file (if it exists)
        config_path = Path("experiments/configs/o3_vs_haiku_baseline_parameterized.yaml")
        if config_path.exists():
            config = config_manager.load_config(config_path)
            
            # Validate key properties
            assert config.environment.n_agents == 3
            assert config.environment.m_items == 5
            assert config.environment.t_rounds == 6
            assert config.environment.gamma_discount == 0.9
            assert len(config.agents) == 3
            assert config.preferences.system_type == PreferenceSystemType.VECTOR
            assert config.preferences.cosine_similarity == 0.95
    
    def test_scaling_laws_config_loading(self):
        """Test loading the scaling laws study configuration."""
        config_manager = ExperimentConfigManager()
        
        config_path = Path("experiments/configs/scaling_laws_study.yaml")
        if config_path.exists():
            config = config_manager.load_config(config_path)
            
            # Validate scaling laws specific properties
            assert config.environment.n_agents == 3
            assert config.execution.batch_size == 20  # Large batch for statistics
            assert "o3" in config.models
            assert "claude-3-haiku" in config.models
    
    def test_cooperative_matrix_config_loading(self):
        """Test loading the cooperative matrix preferences configuration."""
        config_manager = ExperimentConfigManager()
        
        config_path = Path("experiments/configs/cooperative_matrix_preferences.yaml")
        if config_path.exists():
            config = config_manager.load_config(config_path)
            
            # Validate cooperative experiment properties
            assert config.environment.n_agents == 4
            assert config.environment.m_items == 8
            assert config.preferences.system_type == PreferenceSystemType.MATRIX
            assert config.preferences.cosine_similarity == 0.2  # Low competition
            assert all(agent.strategic_level == StrategicLevel.COOPERATIVE for agent in config.agents)


def run_parameterization_tests():
    """Run all parameterization tests."""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    # Run the test suite
    run_parameterization_tests()
"""
Test suite for the enhanced model configuration system.

Tests configuration loading, validation, client creation, and integration
with different model providers and combinations.
"""

import pytest
import tempfile
import os
import yaml
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from negotiation.model_config import (
    ModelRegistry,
    ExperimentModelConfig,
    ConfigLoader,
    ConfigValidator,
    ModelSpec,
    ModelProvider,
    ModelFamily,
    ProviderConfig,
    AgentModelConfig,
    create_default_registry,
    create_o3_vs_haiku_config
)

from negotiation.model_clients import (
    ModelClientFactory,
    UnifiedModelManager,
    OpenAIClient,
    AnthropicClient,
    GoogleClient,
    OpenRouterClient,
    PrincetonClusterClient,
    ModelResponse
)

from negotiation.config_integration import (
    ConfigurationManager,
    create_model_manager_from_config_name,
    load_o3_vs_haiku_config
)


class TestModelRegistry:
    """Test the model registry functionality."""
    
    def test_default_registry_initialization(self):
        """Test that default registry loads all expected models."""
        registry = create_default_registry()
        
        # Test specific models exist
        assert registry.get_model("o3") is not None
        assert registry.get_model("claude-3-haiku") is not None
        assert registry.get_model("gpt-4o") is not None
        assert registry.get_model("gemini-1.5-pro") is not None
        
        # Test model properties
        o3_model = registry.get_model("o3")
        assert o3_model.family == ModelFamily.OPENAI_O_SERIES
        assert o3_model.provider == ModelProvider.OPENAI
        assert o3_model.reasoning_capability == "high"
    
    def test_list_models_filtering(self):
        """Test filtering models by family and provider."""
        registry = create_default_registry()
        
        # Filter by family
        claude_models = registry.list_models(family=ModelFamily.CLAUDE)
        assert len(claude_models) >= 3
        assert all(m.family == ModelFamily.CLAUDE for m in claude_models)
        
        # Filter by provider
        openai_models = registry.list_models(provider=ModelProvider.OPENAI)
        assert len(openai_models) >= 3
        assert all(m.provider == ModelProvider.OPENAI for m in openai_models)
    
    def test_add_princeton_cluster_model(self):
        """Test adding custom Princeton cluster models."""
        registry = create_default_registry()
        
        model_spec = registry.add_princeton_cluster_model(
            model_id="test-llama-70b",
            display_name="Test Llama 70B",
            family=ModelFamily.LLAMA,
            local_path="/scratch/gpfs/DANQIC/models/llama-70b-instruct",
            context_window=32000,
            requires_gpu=True,
            estimated_vram_gb=80
        )
        
        assert model_spec.model_id == "test-llama-70b"
        assert model_spec.provider == ModelProvider.PRINCETON_CLUSTER
        assert model_spec.local_path.startswith("/scratch/gpfs/DANQIC/models/")
        assert model_spec.requires_gpu is True
        
        # Verify it's retrievable
        retrieved = registry.get_model("test-llama-70b")
        assert retrieved == model_spec


class TestConfigLoader:
    """Test configuration loading and saving."""
    
    def test_config_loading_and_saving(self):
        """Test round-trip config loading and saving."""
        registry = create_default_registry()
        loader = ConfigLoader(registry)
        
        # Create a test config
        original_config = create_o3_vs_haiku_config()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "test_config.yaml"
            
            # Save config
            loader.save_to_yaml(original_config, config_path)
            assert config_path.exists()
            
            # Load config back
            loaded_config = loader.load_from_yaml(config_path)
            
            # Verify basic properties
            assert loaded_config.config_name == original_config.config_name
            assert loaded_config.description == original_config.description
            assert len(loaded_config.agents) == len(original_config.agents)
            assert len(loaded_config.providers) == len(original_config.providers)
    
    def test_yaml_config_structure(self):
        """Test that saved YAML has expected structure."""
        registry = create_default_registry()
        loader = ConfigLoader(registry)
        config = create_o3_vs_haiku_config()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "test_config.yaml"
            loader.save_to_yaml(config, config_path)
            
            # Load raw YAML and verify structure
            with open(config_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
            
            assert 'config_name' in yaml_data
            assert 'providers' in yaml_data
            assert 'available_models' in yaml_data
            assert 'agents' in yaml_data
            
            # Verify providers have correct structure
            for provider_name, provider_data in yaml_data['providers'].items():
                assert 'provider' in provider_data
                assert 'requests_per_minute' in provider_data
            
            # Verify agents have model_id references
            for agent_data in yaml_data['agents']:
                assert 'agent_id' in agent_data
                assert 'model_id' in agent_data
                assert agent_data['model_id'] in yaml_data['available_models']


class TestConfigValidator:
    """Test configuration validation."""
    
    def test_valid_config_passes(self):
        """Test that a valid config passes validation."""
        config = create_o3_vs_haiku_config()
        
        # Add API keys to make config valid
        config.providers["openai"].api_key = "test-openai-key"
        config.providers["anthropic"].api_key = "test-anthropic-key"
        
        errors = ConfigValidator.validate_config(config)
        if errors:
            print("Validation errors:", errors)
        assert len(errors) == 0
    
    def test_missing_provider_fails(self):
        """Test that missing provider configuration fails validation."""
        config = create_o3_vs_haiku_config()
        # Remove a provider that's used by agents
        del config.providers["openai"]
        
        errors = ConfigValidator.validate_config(config)
        assert len(errors) > 0
        assert any("not configured" in error for error in errors)
    
    def test_unknown_model_fails(self):
        """Test that referencing unknown models fails validation."""
        config = create_o3_vs_haiku_config()
        # Reference a non-existent model
        config.agents[0].model_spec.model_id = "non-existent-model"
        
        errors = ConfigValidator.validate_config(config)
        assert len(errors) > 0
        assert any("unknown model" in error.lower() for error in errors)


class TestModelClientFactory:
    """Test model client creation."""
    
    @patch('negotiation.model_clients.OPENAI_AVAILABLE', True)
    @patch('negotiation.model_clients.openai')
    def test_openai_client_creation(self, mock_openai):
        """Test creating OpenAI client."""
        registry = create_default_registry()
        model_spec = registry.get_model("o3")
        provider_config = ProviderConfig(
            provider=ModelProvider.OPENAI,
            api_key="test-key"
        )
        
        client = ModelClientFactory.create_client(provider_config, model_spec)
        assert isinstance(client, OpenAIClient)
    
    @patch('negotiation.model_clients.ANTHROPIC_AVAILABLE', True)
    @patch('negotiation.model_clients.anthropic')
    def test_anthropic_client_creation(self, mock_anthropic):
        """Test creating Anthropic client."""
        registry = create_default_registry()
        model_spec = registry.get_model("claude-3-haiku")
        provider_config = ProviderConfig(
            provider=ModelProvider.ANTHROPIC,
            api_key="test-key"
        )
        
        client = ModelClientFactory.create_client(provider_config, model_spec)
        assert isinstance(client, AnthropicClient)
    
    def test_unsupported_provider_fails(self):
        """Test that unsupported provider raises error."""
        model_spec = ModelSpec(
            model_id="test",
            display_name="Test Model",
            family=ModelFamily.CUSTOM,
            provider=ModelProvider("unsupported")
        )
        provider_config = ProviderConfig(provider=ModelProvider("unsupported"))
        
        with pytest.raises(ValueError, match="Unsupported provider"):
            ModelClientFactory.create_client(provider_config, model_spec)


class TestUnifiedModelManager:
    """Test the unified model manager."""
    
    @patch('negotiation.model_clients.OPENAI_AVAILABLE', True)
    @patch('negotiation.model_clients.ANTHROPIC_AVAILABLE', True)
    @patch('negotiation.model_clients.openai')
    @patch('negotiation.model_clients.anthropic')
    def test_manager_registration_and_info(self, mock_anthropic, mock_openai):
        """Test agent registration and info retrieval."""
        registry = create_default_registry()
        manager = UnifiedModelManager()
        
        # Register agents
        o3_config = AgentModelConfig(
            agent_id="test_o3",
            model_spec=registry.get_model("o3")
        )
        openai_provider = ProviderConfig(
            provider=ModelProvider.OPENAI,
            api_key="test-key"
        )
        
        manager.register_agent(o3_config, openai_provider)
        
        # Test agent listing
        agents = manager.list_agents()
        assert "test_o3" in agents
        
        # Test agent info
        info = manager.get_agent_info("test_o3")
        assert info['agent_id'] == "test_o3"
        assert info['model_id'] == "o3"
        assert info['provider'] == "openai"
        assert 'capabilities' in info
    
    def test_unregistered_agent_fails(self):
        """Test that using unregistered agent raises error."""
        manager = UnifiedModelManager()
        
        with pytest.raises(ValueError, match="not registered"):
            asyncio.run(manager.generate("unknown_agent", []))


class TestConfigurationManager:
    """Test the configuration management system."""
    
    def test_config_creation_and_loading(self):
        """Test creating and loading configurations."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = ConfigurationManager(config_dir=tmp_dir)
            
            # Create and save a config
            config = create_o3_vs_haiku_config()
            config_path = manager.save_config(config, "test_experiment")
            
            assert Path(config_path).exists()
            
            # Load it back
            loaded_config = manager.load_config("test_experiment")
            assert loaded_config.config_name == config.config_name
    
    def test_list_configs(self):
        """Test listing available configurations."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            manager = ConfigurationManager(config_dir=tmp_dir)
            
            # Create multiple configs
            config1 = create_o3_vs_haiku_config()
            config1.config_name = "experiment_1"
            manager.save_config(config1)
            
            config2 = create_o3_vs_haiku_config()
            config2.config_name = "experiment_2"
            manager.save_config(config2)
            
            configs = manager.list_configs()
            assert "experiment_1" in configs
            assert "experiment_2" in configs
    
    def test_environment_validation(self):
        """Test environment validation functionality."""
        config = create_o3_vs_haiku_config()
        manager = ConfigurationManager()
        
        # Test validation (will likely show warnings about missing API keys)
        results = manager.validate_environment(config)
        
        assert 'valid' in results
        assert 'errors' in results
        assert 'warnings' in results
        assert 'info' in results


class TestConfigurationIntegration:
    """Test integration with existing system."""
    
    @patch('negotiation.model_clients.OPENAI_AVAILABLE', True)
    @patch('negotiation.model_clients.ANTHROPIC_AVAILABLE', True)
    @patch('negotiation.model_clients.openai')
    @patch('negotiation.model_clients.anthropic')
    def test_model_manager_creation_from_config(self, mock_anthropic, mock_openai):
        """Test creating model manager from configuration."""
        config = create_o3_vs_haiku_config()
        manager_class = ConfigurationManager()
        
        # This should create a unified model manager
        model_manager = manager_class.create_model_manager(config)
        
        assert isinstance(model_manager, UnifiedModelManager)
        
        # Check that agents are registered
        agents = model_manager.list_agents()
        expected_agents = [agent.agent_id for agent in config.agents]
        for expected in expected_agents:
            assert expected in agents
    
    def test_legacy_compatibility(self):
        """Test conversion to legacy configuration format."""
        new_config = create_o3_vs_haiku_config()
        manager = ConfigurationManager()
        
        legacy_config = manager.create_legacy_experiment_config(new_config)
        
        assert legacy_config.experiment_name == new_config.config_name
        assert legacy_config.description == new_config.description
        assert len(legacy_config.agents) == len(new_config.agents)


class TestRealWorldScenarios:
    """Test realistic usage scenarios."""
    
    def test_o3_vs_multiple_models_config(self):
        """Test creating configuration with O3 vs multiple different models."""
        registry = create_default_registry()
        
        config = ExperimentModelConfig(
            config_name="o3_vs_mixed_models",
            description="O3 against various other models",
            providers={
                "openai": ProviderConfig(provider=ModelProvider.OPENAI),
                "anthropic": ProviderConfig(provider=ModelProvider.ANTHROPIC),
                "openrouter": ProviderConfig(provider=ModelProvider.OPENROUTER)
            },
            available_models={
                "o3": registry.get_model("o3"),
                "claude-3-haiku": registry.get_model("claude-3-haiku"),
                "llama-3-70b": registry.get_model("llama-3-70b"),
                "qwen-2.5-72b": registry.get_model("qwen-2.5-72b")
            },
            agents=[
                AgentModelConfig(
                    agent_id="strong_o3",
                    model_spec=registry.get_model("o3"),
                    strategic_level="balanced"
                ),
                AgentModelConfig(
                    agent_id="fast_haiku",
                    model_spec=registry.get_model("claude-3-haiku"),
                    strategic_level="balanced"
                ),
                AgentModelConfig(
                    agent_id="large_llama",
                    model_spec=registry.get_model("llama-3-70b"),
                    strategic_level="balanced"
                ),
                AgentModelConfig(
                    agent_id="chinese_qwen",
                    model_spec=registry.get_model("qwen-2.5-72b"),
                    strategic_level="balanced"
                )
            ]
        )
        
        # Validate the configuration
        errors = ConfigValidator.validate_config(config)
        
        # Should be valid (though may have warnings about API keys)
        assert len([e for e in errors if "not configured" in e]) == 0
    
    def test_princeton_cluster_configuration(self):
        """Test configuration with Princeton cluster models."""
        registry = create_default_registry()
        
        # Add a cluster model
        cluster_model = registry.add_princeton_cluster_model(
            model_id="llama-3-8b-cluster",
            display_name="Llama 3 8B (Cluster)",
            family=ModelFamily.LLAMA,
            local_path="/scratch/gpfs/DANQIC/models/llama-3-8b-instruct"
        )
        
        config = ExperimentModelConfig(
            config_name="cluster_experiment",
            description="Experiment using Princeton cluster models",
            providers={
                "princeton_cluster": ProviderConfig(provider=ModelProvider.PRINCETON_CLUSTER),
                "anthropic": ProviderConfig(provider=ModelProvider.ANTHROPIC)
            },
            available_models={
                "llama-3-8b-cluster": cluster_model,
                "claude-3-haiku": registry.get_model("claude-3-haiku")
            },
            agents=[
                AgentModelConfig(
                    agent_id="cluster_llama",
                    model_spec=cluster_model
                ),
                AgentModelConfig(
                    agent_id="cloud_claude",
                    model_spec=registry.get_model("claude-3-haiku")
                )
            ],
            cluster_config={
                "slurm_partition": "gpu",
                "slurm_time": "02:00:00",
                "conda_env": "negotiation"
            }
        )
        
        # Validate
        errors = ConfigValidator.validate_config(config)
        
        # Should not have errors about cluster configuration
        cluster_errors = [e for e in errors if "cluster" in e.lower()]
        assert len(cluster_errors) == 0


# Test fixtures and utilities
@pytest.fixture
def temp_config_dir():
    """Create a temporary configuration directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return create_o3_vs_haiku_config()


# Run tests
if __name__ == "__main__":
    # Run a subset of tests that don't require external dependencies
    test_registry = TestModelRegistry()
    test_registry.test_default_registry_initialization()
    test_registry.test_list_models_filtering()
    test_registry.test_add_princeton_cluster_model()
    
    test_loader = TestConfigLoader()
    test_loader.test_config_loading_and_saving()
    test_loader.test_yaml_config_structure()
    
    test_validator = TestConfigValidator()
    test_validator.test_valid_config_passes()
    test_validator.test_missing_provider_fails()
    
    test_scenarios = TestRealWorldScenarios()
    test_scenarios.test_o3_vs_multiple_models_config()
    test_scenarios.test_princeton_cluster_configuration()
    
    print("✅ All basic configuration tests passed!")
    print("📋 Model configuration system is working correctly")
    print("🔧 Run full test suite with: pytest tests/test_model_configuration.py")
"""
Configuration Integration for Multi-Agent Negotiation Experiments

This module integrates the enhanced model configuration system with
the existing negotiation framework, providing seamless backward compatibility
while enabling new parameterization features.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import yaml
import os
import logging

from .model_config import (
    ExperimentModelConfig,
    ConfigLoader,
    ConfigValidator,
    ModelRegistry,
    create_default_registry
)
from .model_clients import (
    UnifiedModelManager,
    ModelClientFactory
)
from .agent_factory import (
    AgentConfiguration,
    ExperimentConfiguration
)


class ConfigurationManager:
    """Manages model configurations for negotiation experiments."""
    
    def __init__(self, 
                 config_dir: Union[str, Path] = "experiments/configs",
                 registry: Optional[ModelRegistry] = None):
        """Initialize the configuration manager."""
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.registry = registry or create_default_registry()
        self.loader = ConfigLoader(self.registry)
        self.validator = ConfigValidator()
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Cache for loaded configurations
        self._config_cache: Dict[str, ExperimentModelConfig] = {}
    
    def load_config(self, config_name: str) -> ExperimentModelConfig:
        """Load a configuration by name."""
        # Check cache first
        if config_name in self._config_cache:
            return self._config_cache[config_name]
        
        # Look for config file
        config_path = self.config_dir / f"{config_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load and validate
        config = self.loader.load_from_yaml(config_path)
        errors = self.validator.validate_config(config)
        
        if errors:
            error_msg = f"Configuration validation failed for {config_name}:\n" + "\n".join(f"- {err}" for err in errors)
            raise ValueError(error_msg)
        
        # Cache and return
        self._config_cache[config_name] = config
        self.logger.info(f"Loaded configuration: {config_name}")
        return config
    
    def save_config(self, config: ExperimentModelConfig, config_name: Optional[str] = None) -> str:
        """Save a configuration to file."""
        name = config_name or config.config_name
        config_path = self.config_dir / f"{name}.yaml"
        
        self.loader.save_to_yaml(config, config_path)
        self._config_cache[name] = config
        
        self.logger.info(f"Saved configuration: {name}")
        return str(config_path)
    
    def list_configs(self) -> List[str]:
        """List available configuration names."""
        yaml_files = list(self.config_dir.glob("*.yaml"))
        return [f.stem for f in yaml_files]
    
    def create_model_manager(self, config: ExperimentModelConfig) -> UnifiedModelManager:
        """Create a UnifiedModelManager from a configuration."""
        manager = UnifiedModelManager()
        
        # Register all agents with their model configurations
        for agent_config in config.agents:
            provider_name = agent_config.model_spec.provider.value
            
            if provider_name not in config.providers:
                raise ValueError(f"Provider {provider_name} not configured for agent {agent_config.agent_id}")
            
            provider_config = config.providers[provider_name]
            
            # Ensure API keys are available
            if not provider_config.api_key:
                env_key = self._get_env_key_name(provider_name)
                if env_key in os.environ:
                    provider_config.api_key = os.environ[env_key]
                elif provider_name in config.default_api_keys:
                    provider_config.api_key = config.default_api_keys[provider_name]
                else:
                    # Only warn for providers that need API keys
                    if provider_name not in ["princeton_cluster"]:
                        self.logger.warning(f"No API key found for provider {provider_name}")
            
            manager.register_agent(agent_config, provider_config)
        
        return manager
    
    def create_legacy_experiment_config(self, config: ExperimentModelConfig) -> ExperimentConfiguration:
        """Convert new model config to legacy ExperimentConfiguration for backward compatibility."""
        # Create agent configurations
        legacy_agents = []
        for agent_config in config.agents:
            legacy_agent = AgentConfiguration(
                agent_id=agent_config.agent_id,
                model_type=self._get_legacy_model_type(agent_config.model_spec),
                temperature=agent_config.temperature,
                system_prompt=agent_config.system_prompt,
                strategic_level=agent_config.strategic_level
            )
            legacy_agents.append(legacy_agent)
        
        return ExperimentConfiguration(
            experiment_name=config.config_name,
            description=config.description,
            agents=legacy_agents,
            # Set reasonable defaults for environment parameters
            m_items=5,
            n_agents=len(config.agents),
            t_rounds=10,
            gamma_discount=0.9,
            preference_type="vector",
            competition_level=0.9,
            known_to_all=False,
            tags=[f"version-{config.version}"]
        )
    
    def _get_env_key_name(self, provider_name: str) -> str:
        """Get the expected environment variable name for a provider's API key."""
        env_key_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "openrouter": "OPENROUTER_API_KEY"
        }
        return env_key_map.get(provider_name, f"{provider_name.upper()}_API_KEY")
    
    def _get_legacy_model_type(self, model_spec):
        """Convert new model spec to legacy ModelType enum."""
        # This is a compatibility mapping - you'd import the actual ModelType enum
        # from the existing llm_agents module
        from .llm_agents import ModelType
        
        model_mapping = {
            "o3": ModelType.O3,
            "o3-mini": ModelType.O3_MINI,
            "gpt-4o": ModelType.GPT_4O,
            "claude-3-haiku": ModelType.CLAUDE_3_HAIKU,
            "claude-3-sonnet": ModelType.CLAUDE_3_SONNET,
            "claude-3-opus": ModelType.CLAUDE_3_OPUS
        }
        
        return model_mapping.get(model_spec.model_id, ModelType.CLAUDE_3_HAIKU)  # Default fallback
    
    def validate_environment(self, config: ExperimentModelConfig) -> Dict[str, Any]:
        """Validate that the environment is set up correctly for the configuration."""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "info": []
        }
        
        # Check API keys
        for agent_config in config.agents:
            provider_name = agent_config.model_spec.provider.value
            provider_config = config.providers.get(provider_name)
            
            if not provider_config:
                validation_results["errors"].append(f"Provider {provider_name} not configured")
                validation_results["valid"] = False
                continue
            
            # Check API key availability
            if not provider_config.api_key:
                env_key = self._get_env_key_name(provider_name)
                if env_key not in os.environ and provider_name not in config.default_api_keys:
                    if provider_name not in ["princeton_cluster"]:
                        validation_results["errors"].append(f"API key missing for {provider_name}")
                        validation_results["valid"] = False
        
        # Check Princeton cluster models
        cluster_agents = [a for a in config.agents if a.model_spec.provider.value == "princeton_cluster"]
        if cluster_agents:
            if not config.cluster_config:
                validation_results["warnings"].append("Princeton cluster agents specified but no cluster_config provided")
            
            for agent_config in cluster_agents:
                if not agent_config.model_spec.local_path:
                    validation_results["errors"].append(f"No local_path specified for cluster model {agent_config.model_spec.model_id}")
                    validation_results["valid"] = False
        
        # Check model capabilities
        for agent_config in config.agents:
            model_spec = agent_config.model_spec
            
            if agent_config.system_prompt and not model_spec.supports_system_prompt:
                validation_results["warnings"].append(
                    f"Agent {agent_config.agent_id} has system prompt but model {model_spec.model_id} doesn't support it"
                )
        
        return validation_results


def create_configuration_from_legacy(legacy_config: ExperimentConfiguration) -> ExperimentModelConfig:
    """Create a new model configuration from a legacy ExperimentConfiguration."""
    registry = create_default_registry()
    
    # Map legacy agents to new format
    new_agents = []
    providers = {}
    available_models = {}
    
    for legacy_agent in legacy_config.agents:
        # Get model spec from registry based on legacy model type
        model_id = _legacy_model_type_to_id(legacy_agent.model_type)
        model_spec = registry.get_model(model_id)
        
        if not model_spec:
            raise ValueError(f"Unknown model type: {legacy_agent.model_type}")
        
        # Add to available models
        available_models[model_id] = model_spec
        
        # Add provider if not exists
        provider_name = model_spec.provider.value
        if provider_name not in providers:
            from .model_config import ProviderConfig
            providers[provider_name] = ProviderConfig(
                provider=model_spec.provider,
                requests_per_minute=60,
                tokens_per_minute=100000
            )
        
        # Create new agent config
        from .model_config import AgentModelConfig
        new_agent = AgentModelConfig(
            agent_id=legacy_agent.agent_id,
            model_spec=model_spec,
            temperature=legacy_agent.temperature,
            system_prompt=legacy_agent.system_prompt,
            strategic_level=legacy_agent.strategic_level
        )
        new_agents.append(new_agent)
    
    return ExperimentModelConfig(
        config_name=legacy_config.experiment_name,
        description=legacy_config.description,
        version="1.0",
        providers=providers,
        available_models=available_models,
        agents=new_agents
    )


def _legacy_model_type_to_id(model_type) -> str:
    """Convert legacy ModelType enum to model ID string."""
    # This would map the old enum values to new string IDs
    type_mapping = {
        "O3": "o3",
        "O3_MINI": "o3-mini", 
        "GPT_4O": "gpt-4o",
        "CLAUDE_3_HAIKU": "claude-3-haiku",
        "CLAUDE_3_SONNET": "claude-3-sonnet",
        "CLAUDE_3_OPUS": "claude-3-opus"
    }
    
    return type_mapping.get(str(model_type), "claude-3-haiku")


# Utility functions for common operations
def load_o3_vs_haiku_config() -> ExperimentModelConfig:
    """Load the enhanced O3 vs Haiku configuration."""
    manager = ConfigurationManager()
    return manager.load_config("o3_vs_haiku_enhanced")


def create_model_manager_from_config_name(config_name: str) -> UnifiedModelManager:
    """Create a model manager from a configuration name."""
    manager = ConfigurationManager()
    config = manager.load_config(config_name)
    return manager.create_model_manager(config)


def add_princeton_model_to_registry(model_id: str, 
                                   display_name: str,
                                   local_path: str,
                                   family: str = "custom") -> None:
    """Add a Princeton cluster model to the default registry."""
    from .model_config import ModelFamily
    registry = create_default_registry()
    
    # Convert family string to enum
    family_enum = ModelFamily.CUSTOM
    if hasattr(ModelFamily, family.upper()):
        family_enum = getattr(ModelFamily, family.upper())
    
    registry.add_princeton_cluster_model(
        model_id=model_id,
        display_name=display_name,
        family=family_enum,
        local_path=local_path,
        requires_gpu=True
    )
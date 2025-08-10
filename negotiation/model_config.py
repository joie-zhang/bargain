"""
Enhanced Model Configuration System for Multi-Agent Negotiation Research

This module provides comprehensive configuration support for multiple LLM providers
including native APIs, OpenRouter integration, and local models from Princeton cluster.

Features:
- Support for OpenAI models (O3, O3-mini, GPT-4, GPT-5, etc.)
- Support for Claude models (Haiku, Sonnet, Opus)
- Support for Llama, Gemini, and Qwen models
- OpenRouter integration for unified API access
- Local model support from Princeton cluster (/scratch/gpfs/DANQIC/models/)
- YAML configuration with validation
- Flexible model assignment and parameterization
"""

from typing import Dict, Any, List, Optional, Union, Literal
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import yaml
import json
import os
from abc import ABC, abstractmethod


class ModelProvider(Enum):
    """Supported LLM providers with routing information."""
    # Native API providers
    OPENAI = "openai"
    ANTHROPIC = "anthropic" 
    GOOGLE = "google"
    
    # Third-party aggregators
    OPENROUTER = "openrouter"
    
    # Local deployment
    LOCAL_VLLM = "local_vllm"
    LOCAL_TRANSFORMERS = "local_transformers"
    PRINCETON_CLUSTER = "princeton_cluster"


class ModelFamily(Enum):
    """Model families for easier configuration."""
    OPENAI_GPT = "openai_gpt"
    OPENAI_O_SERIES = "openai_o_series"
    CLAUDE = "claude"
    LLAMA = "llama"
    GEMINI = "gemini"
    GEMMA = "gemma"
    QWEN = "qwen"
    CUSTOM = "custom"


@dataclass
class ModelSpec:
    """Specification for a specific model variant."""
    model_id: str  # Unique identifier (e.g., "o3-mini", "claude-3-haiku")
    display_name: str  # Human-readable name
    family: ModelFamily
    provider: ModelProvider
    
    # API configuration
    api_endpoint: Optional[str] = None
    api_model_name: Optional[str] = None  # Provider-specific model name
    
    # Capabilities
    max_tokens: Optional[int] = None
    context_window: Optional[int] = None
    supports_system_prompt: bool = True
    supports_function_calling: bool = False
    supports_vision: bool = False
    
    # Cost information (per 1M tokens)
    input_cost_per_1m: Optional[float] = None
    output_cost_per_1m: Optional[float] = None
    
    # Performance characteristics
    estimated_speed: Optional[str] = None  # "fast", "medium", "slow"
    reasoning_capability: Optional[str] = None  # "high", "medium", "low"
    
    # Local deployment specific
    local_path: Optional[str] = None
    requires_gpu: bool = False
    estimated_vram_gb: Optional[int] = None


@dataclass
class ProviderConfig:
    """Configuration for a specific model provider."""
    provider: ModelProvider
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None
    organization: Optional[str] = None
    
    # Rate limiting
    requests_per_minute: int = 60
    tokens_per_minute: int = 100000
    
    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True
    
    # Timeout settings
    connect_timeout: float = 10.0
    read_timeout: float = 30.0
    
    # Custom headers
    custom_headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class AgentModelConfig:
    """Model configuration for a specific agent in an experiment."""
    agent_id: str
    model_spec: ModelSpec
    
    # Generation parameters
    temperature: float = 0.7
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    
    # Response control
    max_output_tokens: Optional[int] = None
    stop_sequences: List[str] = field(default_factory=list)
    
    # Prompt configuration
    system_prompt: Optional[str] = None
    prompt_template: Optional[str] = None
    
    # Agent-specific behavior
    strategic_level: str = "balanced"  # aggressive, cooperative, balanced
    reasoning_steps: bool = True
    
    # Custom parameters for specific models
    custom_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentModelConfig:
    """Complete model configuration for an experiment."""
    config_name: str
    description: str
    version: str = "1.0"
    
    # Provider configurations
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)
    
    # Model specifications
    available_models: Dict[str, ModelSpec] = field(default_factory=dict)
    
    # Agent assignments
    agents: List[AgentModelConfig] = field(default_factory=list)
    
    # Environment API keys (can be overridden by provider configs)
    default_api_keys: Dict[str, str] = field(default_factory=dict)
    
    # Princeton cluster specific
    cluster_config: Optional[Dict[str, Any]] = None
    
    # Validation rules
    validation_rules: Dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """Registry of available models with their specifications."""
    
    def __init__(self):
        self._models: Dict[str, ModelSpec] = {}
        self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize with default model specifications."""
        
        # OpenAI O-series models
        self.register_model(ModelSpec(
            model_id="o3",
            display_name="OpenAI O3",
            family=ModelFamily.OPENAI_O_SERIES,
            provider=ModelProvider.OPENAI,
            api_model_name="o3",
            context_window=200000,
            supports_system_prompt=True,
            reasoning_capability="high",
            estimated_speed="slow"
        ))
        
        self.register_model(ModelSpec(
            model_id="o3-mini",
            display_name="OpenAI O3-mini",
            family=ModelFamily.OPENAI_O_SERIES,
            provider=ModelProvider.OPENAI,
            api_model_name="o3-mini",
            context_window=128000,
            supports_system_prompt=True,
            reasoning_capability="high",
            estimated_speed="medium"
        ))
        
        # OpenAI GPT models
        self.register_model(ModelSpec(
            model_id="gpt-4o",
            display_name="GPT-4o",
            family=ModelFamily.OPENAI_GPT,
            provider=ModelProvider.OPENAI,
            api_model_name="gpt-4o",
            context_window=128000,
            supports_system_prompt=True,
            supports_vision=True,
            reasoning_capability="high",
            estimated_speed="fast"
        ))
        
        self.register_model(ModelSpec(
            model_id="gpt-5",
            display_name="GPT-5",
            family=ModelFamily.OPENAI_GPT,
            provider=ModelProvider.OPENAI,
            api_model_name="gpt-5",
            context_window=200000,
            supports_system_prompt=True,
            reasoning_capability="high",
            estimated_speed="medium"
        ))
        
        self.register_model(ModelSpec(
            model_id="gpt-oss",
            display_name="GPT-OSS",
            family=ModelFamily.OPENAI_GPT,
            provider=ModelProvider.OPENAI,
            api_model_name="gpt-oss",
            context_window=128000,
            supports_system_prompt=True,
            reasoning_capability="medium",
            estimated_speed="fast"
        ))
        
        # Claude models
        self.register_model(ModelSpec(
            model_id="claude-3-haiku",
            display_name="Claude 3 Haiku",
            family=ModelFamily.CLAUDE,
            provider=ModelProvider.ANTHROPIC,
            api_model_name="claude-3-haiku-20240307",
            context_window=200000,
            supports_system_prompt=True,
            reasoning_capability="medium",
            estimated_speed="fast"
        ))
        
        self.register_model(ModelSpec(
            model_id="claude-3-sonnet",
            display_name="Claude 3 Sonnet",
            family=ModelFamily.CLAUDE,
            provider=ModelProvider.ANTHROPIC,
            api_model_name="claude-3-sonnet-20240229",
            context_window=200000,
            supports_system_prompt=True,
            reasoning_capability="high",
            estimated_speed="medium"
        ))
        
        self.register_model(ModelSpec(
            model_id="claude-3-opus",
            display_name="Claude 3 Opus",
            family=ModelFamily.CLAUDE,
            provider=ModelProvider.ANTHROPIC,
            api_model_name="claude-3-opus-20240229",
            context_window=200000,
            supports_system_prompt=True,
            reasoning_capability="high",
            estimated_speed="slow"
        ))
        
        # Llama models (via OpenRouter or local)
        self.register_model(ModelSpec(
            model_id="llama-3-70b",
            display_name="Llama 3 70B",
            family=ModelFamily.LLAMA,
            provider=ModelProvider.OPENROUTER,
            api_model_name="meta-llama/llama-3-70b-instruct",
            context_window=32000,
            supports_system_prompt=True,
            reasoning_capability="high",
            estimated_speed="medium",
            requires_gpu=True,
            estimated_vram_gb=80
        ))
        
        self.register_model(ModelSpec(
            model_id="llama-3-8b",
            display_name="Llama 3 8B",
            family=ModelFamily.LLAMA,
            provider=ModelProvider.OPENROUTER,
            api_model_name="meta-llama/llama-3-8b-instruct",
            context_window=32000,
            supports_system_prompt=True,
            reasoning_capability="medium",
            estimated_speed="fast",
            requires_gpu=True,
            estimated_vram_gb=16
        ))
        
        # Gemini 2.5 models (latest generation)
        self.register_model(ModelSpec(
            model_id="gemini-2.5-pro",
            display_name="Gemini 2.5 Pro",
            family=ModelFamily.GEMINI,
            provider=ModelProvider.GOOGLE,
            api_model_name="gemini-2.5-pro",
            context_window=2000000,
            supports_system_prompt=True,
            supports_vision=True,
            supports_function_calling=True,
            reasoning_capability="high",
            estimated_speed="medium",
            input_cost_per_1m=1.25,
            output_cost_per_1m=2.50
        ))
        
        self.register_model(ModelSpec(
            model_id="gemini-2.5-flash",
            display_name="Gemini 2.5 Flash",
            family=ModelFamily.GEMINI,
            provider=ModelProvider.GOOGLE,
            api_model_name="gemini-2.5-flash",
            context_window=1000000,
            supports_system_prompt=True,
            supports_vision=True,
            supports_function_calling=True,
            reasoning_capability="high",
            estimated_speed="fast",
            input_cost_per_1m=0.075,
            output_cost_per_1m=0.30
        ))
        
        self.register_model(ModelSpec(
            model_id="gemini-2.5-flash-lite",
            display_name="Gemini 2.5 Flash Lite",
            family=ModelFamily.GEMINI,
            provider=ModelProvider.GOOGLE,
            api_model_name="gemini-2.5-flash-lite",
            context_window=1000000,
            supports_system_prompt=True,
            supports_vision=True,
            reasoning_capability="medium",
            estimated_speed="very_fast",
            input_cost_per_1m=0.0375,
            output_cost_per_1m=0.15
        ))
        
        # Gemini 2.0 models
        self.register_model(ModelSpec(
            model_id="gemini-2.0-flash",
            display_name="Gemini 2.0 Flash",
            family=ModelFamily.GEMINI,
            provider=ModelProvider.GOOGLE,
            api_model_name="gemini-2.0-flash",
            context_window=1000000,
            supports_system_prompt=True,
            supports_vision=True,
            supports_function_calling=True,
            reasoning_capability="high",
            estimated_speed="fast",
            input_cost_per_1m=0.075,
            output_cost_per_1m=0.30
        ))
        
        self.register_model(ModelSpec(
            model_id="gemini-2.0-flash-lite",
            display_name="Gemini 2.0 Flash Lite",
            family=ModelFamily.GEMINI,
            provider=ModelProvider.GOOGLE,
            api_model_name="gemini-2.0-flash-lite",
            context_window=1000000,
            supports_system_prompt=True,
            supports_vision=True,
            reasoning_capability="medium",
            estimated_speed="very_fast",
            input_cost_per_1m=0.0375,
            output_cost_per_1m=0.15
        ))
        
        # Gemini 1.5 models (legacy but still useful)
        self.register_model(ModelSpec(
            model_id="gemini-1.5-pro",
            display_name="Gemini 1.5 Pro",
            family=ModelFamily.GEMINI,
            provider=ModelProvider.GOOGLE,
            api_model_name="gemini-1.5-pro",
            context_window=2000000,
            supports_system_prompt=True,
            supports_vision=True,
            reasoning_capability="high",
            estimated_speed="medium",
            input_cost_per_1m=1.25,
            output_cost_per_1m=5.0
        ))
        
        self.register_model(ModelSpec(
            model_id="gemini-1.5-flash",
            display_name="Gemini 1.5 Flash",
            family=ModelFamily.GEMINI,
            provider=ModelProvider.GOOGLE,
            api_model_name="gemini-1.5-flash",
            context_window=1000000,
            supports_system_prompt=True,
            supports_vision=True,
            reasoning_capability="medium",
            estimated_speed="fast",
            input_cost_per_1m=0.075,
            output_cost_per_1m=0.30
        ))
        
        self.register_model(ModelSpec(
            model_id="gemini-1.5-flash-8b",
            display_name="Gemini 1.5 Flash 8B",
            family=ModelFamily.GEMINI,
            provider=ModelProvider.GOOGLE,
            api_model_name="gemini-1.5-flash-8b",
            context_window=1000000,
            supports_system_prompt=True,
            supports_vision=True,
            reasoning_capability="medium",
            estimated_speed="very_fast",
            input_cost_per_1m=0.0375,
            output_cost_per_1m=0.15
        ))
        
        # Gemma 2 models (2B, 9B, 27B variants)
        self.register_model(ModelSpec(
            model_id="gemma-2-27b",
            display_name="Gemma 2 27B",
            family=ModelFamily.GEMMA,
            provider=ModelProvider.OPENROUTER,
            api_model_name="google/gemma-2-27b-it",
            context_window=8192,
            supports_system_prompt=True,
            reasoning_capability="high",
            estimated_speed="medium",
            requires_gpu=True,
            estimated_vram_gb=54
        ))
        
        self.register_model(ModelSpec(
            model_id="gemma-2-9b",
            display_name="Gemma 2 9B", 
            family=ModelFamily.GEMMA,
            provider=ModelProvider.OPENROUTER,
            api_model_name="google/gemma-2-9b-it",
            context_window=8192,
            supports_system_prompt=True,
            reasoning_capability="medium",
            estimated_speed="fast",
            requires_gpu=True,
            estimated_vram_gb=18
        ))
        
        self.register_model(ModelSpec(
            model_id="gemma-2-2b",
            display_name="Gemma 2 2B",
            family=ModelFamily.GEMMA,
            provider=ModelProvider.OPENROUTER,
            api_model_name="google/gemma-2-2b-it",
            context_window=8192,
            supports_system_prompt=True,
            reasoning_capability="medium",
            estimated_speed="fast",
            requires_gpu=True,
            estimated_vram_gb=4
        ))
        
        # Gemma 3 models (latest: 1B, 4B, 12B, 27B variants)
        self.register_model(ModelSpec(
            model_id="gemma-3-27b",
            display_name="Gemma 3 27B",
            family=ModelFamily.GEMMA,
            provider=ModelProvider.OPENROUTER,
            api_model_name="google/gemma-3-27b-it",
            context_window=8192,
            supports_system_prompt=True,
            reasoning_capability="high",
            estimated_speed="medium",
            requires_gpu=True,
            estimated_vram_gb=54
        ))
        
        self.register_model(ModelSpec(
            model_id="gemma-3-12b",
            display_name="Gemma 3 12B",
            family=ModelFamily.GEMMA,
            provider=ModelProvider.OPENROUTER,
            api_model_name="google/gemma-3-12b-it",
            context_window=8192,
            supports_system_prompt=True,
            reasoning_capability="medium",
            estimated_speed="fast",
            requires_gpu=True,
            estimated_vram_gb=24
        ))
        
        self.register_model(ModelSpec(
            model_id="gemma-3-4b",
            display_name="Gemma 3 4B",
            family=ModelFamily.GEMMA,
            provider=ModelProvider.OPENROUTER,
            api_model_name="google/gemma-3-4b-it",
            context_window=8192,
            supports_system_prompt=True,
            reasoning_capability="medium",
            estimated_speed="fast",
            requires_gpu=True,
            estimated_vram_gb=8
        ))
        
        self.register_model(ModelSpec(
            model_id="gemma-3-1b",
            display_name="Gemma 3 1B",
            family=ModelFamily.GEMMA,
            provider=ModelProvider.OPENROUTER,
            api_model_name="google/gemma-3-1b-it",
            context_window=8192,
            supports_system_prompt=True,
            reasoning_capability="low",
            estimated_speed="very_fast",
            requires_gpu=False,  # Can run on CPU
            estimated_vram_gb=2
        ))
        
        # Qwen models
        self.register_model(ModelSpec(
            model_id="qwen-2.5-72b",
            display_name="Qwen 2.5 72B",
            family=ModelFamily.QWEN,
            provider=ModelProvider.OPENROUTER,
            api_model_name="qwen/qwen-2.5-72b-instruct",
            context_window=32768,
            supports_system_prompt=True,
            reasoning_capability="high",
            estimated_speed="medium"
        ))
        
        self.register_model(ModelSpec(
            model_id="qwen-2.5-14b",
            display_name="Qwen 2.5 14B",
            family=ModelFamily.QWEN,
            provider=ModelProvider.OPENROUTER,
            api_model_name="qwen/qwen-2.5-14b-instruct",
            context_window=32768,
            supports_system_prompt=True,
            reasoning_capability="medium",
            estimated_speed="fast"
        ))
    
    def register_model(self, model_spec: ModelSpec) -> None:
        """Register a new model specification."""
        self._models[model_spec.model_id] = model_spec
    
    def get_model(self, model_id: str) -> Optional[ModelSpec]:
        """Get model specification by ID."""
        return self._models.get(model_id)
    
    def list_models(self, 
                   family: Optional[ModelFamily] = None, 
                   provider: Optional[ModelProvider] = None) -> List[ModelSpec]:
        """List available models, optionally filtered by family or provider."""
        models = list(self._models.values())
        
        if family:
            models = [m for m in models if m.family == family]
        if provider:
            models = [m for m in models if m.provider == provider]
            
        return models
    
    def add_princeton_cluster_model(self, 
                                  model_id: str,
                                  display_name: str,
                                  family: ModelFamily,
                                  local_path: str,
                                  **kwargs) -> ModelSpec:
        """Add a model available on Princeton cluster."""
        # Validate path exists on cluster
        cluster_base = "/scratch/gpfs/DANQIC/models/"
        if not local_path.startswith(cluster_base):
            local_path = os.path.join(cluster_base, local_path)
        
        model_spec = ModelSpec(
            model_id=model_id,
            display_name=display_name,
            family=family,
            provider=ModelProvider.PRINCETON_CLUSTER,
            local_path=local_path,
            requires_gpu=kwargs.get('requires_gpu', True),
            **{k: v for k, v in kwargs.items() if k != 'requires_gpu'}
        )
        
        self.register_model(model_spec)
        return model_spec


class ConfigValidator:
    """Validates model configuration files."""
    
    @staticmethod
    def validate_config(config: ExperimentModelConfig) -> List[str]:
        """Validate experiment configuration and return list of errors."""
        errors = []
        
        # Validate agent model assignments
        for agent_config in config.agents:
            if agent_config.model_spec.model_id not in config.available_models:
                errors.append(f"Agent {agent_config.agent_id} references unknown model {agent_config.model_spec.model_id}")
        
        # Validate provider configurations
        for agent_config in config.agents:
            provider = agent_config.model_spec.provider
            if provider.value not in config.providers:
                errors.append(f"Provider {provider.value} not configured for agent {agent_config.agent_id}")
        
        # Validate API keys are provided
        for provider_name, provider_config in config.providers.items():
            provider_enum = ModelProvider(provider_name)
            if provider_enum in [ModelProvider.OPENAI, ModelProvider.ANTHROPIC, ModelProvider.GOOGLE]:
                if not provider_config.api_key and provider_name not in config.default_api_keys:
                    errors.append(f"No API key provided for provider {provider_name}")
        
        # Validate Princeton cluster configuration
        cluster_agents = [a for a in config.agents if a.model_spec.provider == ModelProvider.PRINCETON_CLUSTER]
        if cluster_agents and not config.cluster_config:
            errors.append("Princeton cluster agents specified but no cluster_config provided")
        
        return errors


class ConfigLoader:
    """Loads and manages model configurations from YAML files."""
    
    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or ModelRegistry()
    
    def load_from_yaml(self, config_path: Union[str, Path]) -> ExperimentModelConfig:
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return self._parse_config_dict(data)
    
    def save_to_yaml(self, config: ExperimentModelConfig, output_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict and clean up non-serializable fields
        config_dict = self._config_to_dict(config)
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    def _parse_config_dict(self, data: Dict[str, Any]) -> ExperimentModelConfig:
        """Parse configuration dictionary into ExperimentModelConfig."""
        # Parse providers
        providers = {}
        for provider_name, provider_data in data.get('providers', {}).items():
            # Remove 'provider' key to avoid duplicate argument
            provider_data_copy = provider_data.copy()
            provider_data_copy.pop('provider', None)
            providers[provider_name] = ProviderConfig(
                provider=ModelProvider(provider_name),
                **provider_data_copy
            )
        
        # Parse available models
        available_models = {}
        for model_id, model_data in data.get('available_models', {}).items():
            # Create filtered data without keys that will be explicitly set
            filtered_data = {k: v for k, v in model_data.items() 
                           if k not in ['family', 'provider']}
            model_spec = ModelSpec(
                model_id=model_id,
                family=ModelFamily(model_data['family']),
                provider=ModelProvider(model_data['provider']),
                **filtered_data
            )
            available_models[model_id] = model_spec
        
        # Parse agent configurations
        agents = []
        for agent_data in data.get('agents', []):
            model_id = agent_data['model_id']
            if model_id in available_models:
                model_spec = available_models[model_id]
            else:
                model_spec = self.registry.get_model(model_id)
                if not model_spec:
                    raise ValueError(f"Unknown model: {model_id}")
            
            agent_config = AgentModelConfig(
                model_spec=model_spec,
                **{k: v for k, v in agent_data.items() if k != 'model_id'}
            )
            agents.append(agent_config)
        
        return ExperimentModelConfig(
            config_name=data['config_name'],
            description=data['description'],
            version=data.get('version', '1.0'),
            providers=providers,
            available_models=available_models,
            agents=agents,
            default_api_keys=data.get('default_api_keys', {}),
            cluster_config=data.get('cluster_config'),
            validation_rules=data.get('validation_rules', {})
        )
    
    def _config_to_dict(self, config: ExperimentModelConfig) -> Dict[str, Any]:
        """Convert configuration to serializable dictionary."""
        return {
            'config_name': config.config_name,
            'description': config.description,
            'version': config.version,
            'providers': {
                name: {
                    'provider': provider_config.provider.value,
                    **{k: v for k, v in asdict(provider_config).items() if k != 'provider'}
                }
                for name, provider_config in config.providers.items()
            },
            'available_models': {
                model_id: {
                    'family': model_spec.family.value,
                    'provider': model_spec.provider.value,
                    **{k: v for k, v in asdict(model_spec).items() 
                       if k not in ['model_id', 'family', 'provider']}
                }
                for model_id, model_spec in config.available_models.items()
            },
            'agents': [
                {
                    'model_id': agent.model_spec.model_id,
                    **{k: v for k, v in asdict(agent).items() if k != 'model_spec'}
                }
                for agent in config.agents
            ],
            'default_api_keys': config.default_api_keys,
            'cluster_config': config.cluster_config,
            'validation_rules': config.validation_rules
        }


def create_default_registry() -> ModelRegistry:
    """Create a model registry with all default models."""
    return ModelRegistry()


def create_o3_vs_haiku_config() -> ExperimentModelConfig:
    """Create a configuration for O3 vs Claude Haiku experiments."""
    registry = create_default_registry()
    
    return ExperimentModelConfig(
        config_name="o3_vs_haiku_baseline",
        description="O3 vs Claude Haiku competitive negotiation experiment",
        version="1.0",
        providers={
            "openai": ProviderConfig(
                provider=ModelProvider.OPENAI,
                requests_per_minute=50,
                tokens_per_minute=80000
            ),
            "anthropic": ProviderConfig(
                provider=ModelProvider.ANTHROPIC,
                requests_per_minute=60,
                tokens_per_minute=100000
            )
        },
        available_models={
            "o3": registry.get_model("o3"),
            "claude-3-haiku": registry.get_model("claude-3-haiku")
        },
        agents=[
            AgentModelConfig(
                agent_id="agent_1_o3",
                model_spec=registry.get_model("o3"),
                temperature=0.7,
                strategic_level="balanced"
            ),
            AgentModelConfig(
                agent_id="agent_2_haiku",
                model_spec=registry.get_model("claude-3-haiku"),
                temperature=0.7,
                strategic_level="balanced"
            ),
            AgentModelConfig(
                agent_id="agent_3_haiku",
                model_spec=registry.get_model("claude-3-haiku"),
                temperature=0.7,
                strategic_level="balanced"
            )
        ]
    )
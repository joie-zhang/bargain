#!/usr/bin/env python3
"""
Experiment Configuration Management System

This module provides comprehensive configuration management for parameterized 
multi-agent negotiation experiments, supporting:
- YAML configuration loading and validation
- Multiple LLM provider support 
- Environment parameter configuration
- Preference system selection
- Competition level tuning
- Proposal order analysis
- Cost and resource estimation
"""

import yaml
import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import random
import time
from copy import deepcopy


class PreferenceSystemType(Enum):
    VECTOR = "vector"
    MATRIX = "matrix"


class StrategicLevel(Enum):
    COOPERATIVE = "cooperative"
    BALANCED = "balanced" 
    COMPETITIVE = "competitive"


class ReasoningCapability(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    provider: str
    base_url: Optional[str] = None
    api_key_env: str = ""
    requests_per_minute: int = 60
    tokens_per_minute: int = 100000
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True
    connect_timeout: float = 10.0
    read_timeout: float = 30.0
    
    # Local cluster specific
    base_path: Optional[str] = None
    gpu_allocation: str = "1"
    memory_gb: int = 16
    timeout_minutes: int = 60


@dataclass  
class ModelConfig:
    """Configuration for a specific model."""
    display_name: str
    family: str
    provider: str
    api_model_name: str
    context_window: int
    supports_system_prompt: bool = True
    supports_function_calling: bool = False
    reasoning_capability: ReasoningCapability = ReasoningCapability.MEDIUM
    estimated_speed: str = "medium"
    input_cost_per_1m: float = 1.0
    output_cost_per_1m: float = 3.0
    
    # Local cluster specific
    model_path: Optional[str] = None
    gpu_memory_gb: Optional[int] = None


@dataclass
class AgentConfig:
    """Configuration for a negotiation agent."""
    agent_id: str
    model_id: str
    temperature: float = 0.7
    max_output_tokens: int = 2048
    strategic_level: StrategicLevel = StrategicLevel.BALANCED
    reasoning_steps: bool = True
    use_chain_of_thought: bool = True
    system_prompt: Optional[str] = None
    custom_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentConfig:
    """Environment parameters for negotiation."""
    n_agents: int = 3
    m_items: int = 5
    t_rounds: int = 6
    gamma_discount: float = 0.9
    randomized_proposal_order: bool = True
    require_unanimous_consensus: bool = True
    allow_partial_allocations: bool = False
    max_conversation_turns_per_round: int = 10
    timeout_minutes: int = 30


@dataclass
class PreferenceConfig:
    """Preference system configuration."""
    system_type: PreferenceSystemType = PreferenceSystemType.VECTOR
    
    # Vector preferences
    value_range: Tuple[int, int] = (0, 10)
    distribution: str = "uniform"
    ensure_variation: bool = True
    
    # Matrix preferences
    self_weight: float = 0.7
    others_weight: float = 0.3
    asymmetric_weights: bool = False
    
    # Competition level
    cosine_similarity: float = 0.95
    tolerance: float = 0.05
    max_generation_attempts: int = 100
    
    # Visibility
    known_to_all: bool = False


@dataclass
class ProposalOrderConfig:
    """Configuration for proposal order analysis."""
    randomized: bool = True
    track_order_effects: bool = True
    correlation_threshold: float = 0.3
    min_samples_for_analysis: int = 10
    statistical_significance: float = 0.05
    fixed_order_experiments: int = 5
    random_order_experiments: int = 15
    analyze_position_bias: bool = True


@dataclass  
class ExecutionConfig:
    """Experiment execution configuration."""
    experiment_name: str = "parameterized_negotiation"
    experiment_version: str = "1.0"
    random_seed: Optional[int] = None
    seed_increment_per_run: int = 1
    batch_size: int = 1
    parallel_runs: int = 1
    save_full_logs: bool = True
    save_decision_traces: bool = True
    save_intermediate_states: bool = True
    validate_config_before_run: bool = True
    max_cost_per_run: float = 10.0
    stop_on_api_errors: bool = False


@dataclass
class ValidationConfig:
    """Validation rules for experiments."""
    min_agents: int = 2
    max_agents: int = 10
    min_items: int = 1
    max_items: int = 20
    min_rounds: int = 1
    max_rounds: int = 20
    gamma_range: Tuple[float, float] = (0.1, 1.0)
    max_expensive_models_per_experiment: int = 2
    require_system_prompt_support: bool = True
    temperature_range: Tuple[float, float] = (0.0, 2.0)
    max_tokens_range: Tuple[int, int] = (100, 8192)
    max_estimated_cost_per_run: float = 50.0
    warn_if_cost_above: float = 5.0
    max_expected_runtime_minutes: int = 120
    warn_if_runtime_above_minutes: int = 30


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    schema_version: str = "1.0"
    description: str = ""
    
    # Core configuration sections
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    agents: List[AgentConfig] = field(default_factory=list)
    preferences: PreferenceConfig = field(default_factory=PreferenceConfig)
    proposal_order: ProposalOrderConfig = field(default_factory=ProposalOrderConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    
    # Output configuration
    results_directory: str = "experiments/results"
    log_directory: str = "experiments/logs"
    output_formats: List[str] = field(default_factory=lambda: ["json", "csv", "markdown"])


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass


class ExperimentConfigManager:
    """
    Manages loading, validation, and manipulation of experiment configurations.
    
    Supports:
    - Loading YAML configurations with schema validation
    - Multiple LLM provider support
    - Dynamic agent configuration
    - Cost estimation
    - Configuration templates and presets
    """
    
    def __init__(self, config_dir: str = "experiments/configs"):
        """Initialize configuration manager."""
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Cache for loaded configurations
        self._config_cache: Dict[str, ExperimentConfig] = {}
        
        # Schema for validation
        self._schema_cache: Optional[Dict] = None
    
    def load_config(self, config_path: Union[str, Path]) -> ExperimentConfig:
        """
        Load experiment configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            ExperimentConfig object
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        config_path = Path(config_path)
        
        # Check cache first
        cache_key = str(config_path.absolute())
        if cache_key in self._config_cache:
            return deepcopy(self._config_cache[cache_key])
        
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {config_path}: {e}")
        
        # Parse configuration sections
        try:
            config = self._parse_config_dict(raw_config)
        except Exception as e:
            raise ConfigurationError(f"Error parsing configuration {config_path}: {e}")
        
        # Validate configuration
        self.validate_config(config)
        
        # Cache and return
        self._config_cache[cache_key] = deepcopy(config)
        return config
    
    def _parse_config_dict(self, raw_config: Dict[str, Any]) -> ExperimentConfig:
        """Parse raw configuration dictionary into ExperimentConfig."""
        
        # Parse environment configuration
        env_config = EnvironmentConfig()
        if 'environment' in raw_config:
            env_data = raw_config['environment']
            env_config = EnvironmentConfig(
                n_agents=env_data.get('n_agents', 3),
                m_items=env_data.get('m_items', 5),
                t_rounds=env_data.get('t_rounds', 6),
                gamma_discount=env_data.get('gamma_discount', 0.9),
                randomized_proposal_order=env_data.get('randomized_proposal_order', True),
                require_unanimous_consensus=env_data.get('require_unanimous_consensus', True),
                allow_partial_allocations=env_data.get('allow_partial_allocations', False),
                max_conversation_turns_per_round=env_data.get('max_conversation_turns_per_round', 10),
                timeout_minutes=env_data.get('timeout_minutes', 30)
            )
        
        # Parse provider configurations
        providers = {}
        if 'models' in raw_config and 'providers' in raw_config['models']:
            for name, provider_data in raw_config['models']['providers'].items():
                providers[name] = ProviderConfig(
                    provider=provider_data.get('provider', name),
                    base_url=provider_data.get('base_url'),
                    api_key_env=provider_data.get('api_key_env', ''),
                    requests_per_minute=provider_data.get('requests_per_minute', 60),
                    tokens_per_minute=provider_data.get('tokens_per_minute', 100000),
                    max_retries=provider_data.get('max_retries', 3),
                    retry_delay=provider_data.get('retry_delay', 1.0),
                    exponential_backoff=provider_data.get('exponential_backoff', True),
                    connect_timeout=provider_data.get('connect_timeout', 10.0),
                    read_timeout=provider_data.get('read_timeout', 30.0),
                    base_path=provider_data.get('base_path'),
                    gpu_allocation=provider_data.get('gpu_allocation', '1'),
                    memory_gb=provider_data.get('memory_gb', 16),
                    timeout_minutes=provider_data.get('timeout_minutes', 60)
                )
        
        # Parse model configurations
        models = {}
        if 'models' in raw_config and 'available_models' in raw_config['models']:
            for name, model_data in raw_config['models']['available_models'].items():
                capability_str = model_data.get('reasoning_capability', 'medium')
                capability = ReasoningCapability(capability_str)
                
                models[name] = ModelConfig(
                    display_name=model_data.get('display_name', name),
                    family=model_data.get('family', 'unknown'),
                    provider=model_data.get('provider', 'unknown'),
                    api_model_name=model_data.get('api_model_name', name),
                    context_window=model_data.get('context_window', 4096),
                    supports_system_prompt=model_data.get('supports_system_prompt', True),
                    supports_function_calling=model_data.get('supports_function_calling', False),
                    reasoning_capability=capability,
                    estimated_speed=model_data.get('estimated_speed', 'medium'),
                    input_cost_per_1m=model_data.get('input_cost_per_1m', 1.0),
                    output_cost_per_1m=model_data.get('output_cost_per_1m', 3.0),
                    model_path=model_data.get('model_path'),
                    gpu_memory_gb=model_data.get('gpu_memory_gb')
                )
        
        # Parse agent configurations
        agents = []
        if 'agents' in raw_config:
            if isinstance(raw_config['agents'], list):
                # Direct agent list
                for i, agent_data in enumerate(raw_config['agents']):
                    strategic_level_str = agent_data.get('strategic_level', 'balanced')
                    strategic_level = StrategicLevel(strategic_level_str)
                    
                    agents.append(AgentConfig(
                        agent_id=agent_data.get('agent_id', f'agent_{i+1}'),
                        model_id=agent_data.get('model_id', agent_data.get('model', 'gpt-4o-mini')),
                        temperature=agent_data.get('temperature', 0.7),
                        max_output_tokens=agent_data.get('max_output_tokens', 2048),
                        strategic_level=strategic_level,
                        reasoning_steps=agent_data.get('reasoning_steps', True),
                        use_chain_of_thought=agent_data.get('use_chain_of_thought', True),
                        system_prompt=agent_data.get('system_prompt'),
                        custom_parameters=agent_data.get('custom_parameters', {})
                    ))
            else:
                # Template-based agent generation  
                template = raw_config['agents'].get('template', {})
                # This will be used for dynamic agent generation
                pass
        
        # Parse preference configuration
        pref_config = PreferenceConfig()
        if 'preferences' in raw_config:
            pref_data = raw_config['preferences']
            
            system_type_str = pref_data.get('system_type', 'vector')
            system_type = PreferenceSystemType(system_type_str)
            
            pref_config = PreferenceConfig(
                system_type=system_type,
                value_range=tuple(pref_data.get('vector_preferences', {}).get('value_range', [0, 10])),
                distribution=pref_data.get('vector_preferences', {}).get('distribution', 'uniform'),
                ensure_variation=pref_data.get('vector_preferences', {}).get('ensure_variation', True),
                self_weight=pref_data.get('matrix_preferences', {}).get('self_weight', 0.7),
                others_weight=pref_data.get('matrix_preferences', {}).get('others_weight', 0.3),
                asymmetric_weights=pref_data.get('matrix_preferences', {}).get('asymmetric_weights', False),
                cosine_similarity=pref_data.get('competition_level', {}).get('cosine_similarity', 0.95),
                tolerance=pref_data.get('competition_level', {}).get('tolerance', 0.05),
                max_generation_attempts=pref_data.get('competition_level', {}).get('max_generation_attempts', 100),
                known_to_all=pref_data.get('known_to_all', False)
            )
        
        # Parse proposal order configuration  
        order_config = ProposalOrderConfig()
        if 'proposal_order' in raw_config:
            order_data = raw_config['proposal_order']
            analysis_data = order_data.get('analysis', {})
            ablation_data = order_data.get('ablation_studies', {})
            
            order_config = ProposalOrderConfig(
                randomized=order_data.get('randomized', True),
                track_order_effects=order_data.get('track_order_effects', True),
                correlation_threshold=analysis_data.get('correlation_threshold', 0.3),
                min_samples_for_analysis=analysis_data.get('min_samples_for_analysis', 10),
                statistical_significance=analysis_data.get('statistical_significance', 0.05),
                fixed_order_experiments=ablation_data.get('fixed_order_experiments', 5),
                random_order_experiments=ablation_data.get('random_order_experiments', 15),
                analyze_position_bias=ablation_data.get('analyze_position_bias', True)
            )
        
        # Parse execution configuration
        exec_config = ExecutionConfig()
        if 'execution' in raw_config:
            exec_data = raw_config['execution']
            exec_config = ExecutionConfig(
                experiment_name=exec_data.get('experiment_name', 'parameterized_negotiation'),
                experiment_version=exec_data.get('experiment_version', '1.0'),
                random_seed=exec_data.get('random_seed'),
                seed_increment_per_run=exec_data.get('seed_increment_per_run', 1),
                batch_size=exec_data.get('batch_size', 1),
                parallel_runs=exec_data.get('parallel_runs', 1),
                save_full_logs=exec_data.get('save_full_logs', True),
                save_decision_traces=exec_data.get('save_decision_traces', True),
                save_intermediate_states=exec_data.get('save_intermediate_states', True),
                validate_config_before_run=exec_data.get('validate_config_before_run', True),
                max_cost_per_run=exec_data.get('max_cost_per_run', 10.0),
                stop_on_api_errors=exec_data.get('stop_on_api_errors', False)
            )
        
        # Parse validation configuration
        validation_config = ValidationConfig()
        if 'validation' in raw_config:
            val_data = raw_config['validation']
            validation_config = ValidationConfig(
                min_agents=val_data.get('min_agents', 2),
                max_agents=val_data.get('max_agents', 10),
                min_items=val_data.get('min_items', 1),
                max_items=val_data.get('max_items', 20),
                min_rounds=val_data.get('min_rounds', 1),
                max_rounds=val_data.get('max_rounds', 20),
                gamma_range=tuple(val_data.get('gamma_range', [0.1, 1.0])),
                max_expensive_models_per_experiment=val_data.get('max_expensive_models_per_experiment', 2),
                require_system_prompt_support=val_data.get('require_system_prompt_support', True),
                temperature_range=tuple(val_data.get('temperature_range', [0.0, 2.0])),
                max_tokens_range=tuple(val_data.get('max_tokens_range', [100, 8192])),
                max_estimated_cost_per_run=val_data.get('max_estimated_cost_per_run', 50.0),
                warn_if_cost_above=val_data.get('warn_if_cost_above', 5.0),
                max_expected_runtime_minutes=val_data.get('max_expected_runtime_minutes', 120),
                warn_if_runtime_above_minutes=val_data.get('warn_if_runtime_above_minutes', 30)
            )
        
        # Create complete configuration
        config = ExperimentConfig(
            schema_version=raw_config.get('schema_version', '1.0'),
            description=raw_config.get('description', ''),
            environment=env_config,
            providers=providers,
            models=models,
            agents=agents,
            preferences=pref_config,
            proposal_order=order_config,
            execution=exec_config,
            validation=validation_config,
            results_directory=raw_config.get('output', {}).get('results_directory', 'experiments/results'),
            log_directory=raw_config.get('output', {}).get('log_directory', 'experiments/logs'),
            output_formats=raw_config.get('output', {}).get('formats', ['json', 'csv', 'markdown'])
        )
        
        return config
    
    def validate_config(self, config: ExperimentConfig) -> None:
        """
        Validate experiment configuration.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        validation_errors = []
        
        # Environment validation
        if config.environment.n_agents < config.validation.min_agents:
            validation_errors.append(f"n_agents ({config.environment.n_agents}) < minimum ({config.validation.min_agents})")
        if config.environment.n_agents > config.validation.max_agents:
            validation_errors.append(f"n_agents ({config.environment.n_agents}) > maximum ({config.validation.max_agents})")
            
        if config.environment.m_items < config.validation.min_items:
            validation_errors.append(f"m_items ({config.environment.m_items}) < minimum ({config.validation.min_items})")
        if config.environment.m_items > config.validation.max_items:
            validation_errors.append(f"m_items ({config.environment.m_items}) > maximum ({config.validation.max_items})")
            
        if config.environment.t_rounds < config.validation.min_rounds:
            validation_errors.append(f"t_rounds ({config.environment.t_rounds}) < minimum ({config.validation.min_rounds})")
        if config.environment.t_rounds > config.validation.max_rounds:
            validation_errors.append(f"t_rounds ({config.environment.t_rounds}) > maximum ({config.validation.max_rounds})")
            
        if not (config.validation.gamma_range[0] <= config.environment.gamma_discount <= config.validation.gamma_range[1]):
            validation_errors.append(f"gamma_discount ({config.environment.gamma_discount}) outside valid range {config.validation.gamma_range}")
        
        # Agent validation
        if len(config.agents) != config.environment.n_agents:
            validation_errors.append(f"Number of agents ({len(config.agents)}) != n_agents ({config.environment.n_agents})")
        
        # Model validation
        expensive_models = ['o3', 'o3-mini', 'claude-3-5-sonnet', 'gpt-4o']
        expensive_model_count = sum(1 for agent in config.agents 
                                  if any(exp_model in agent.model_id for exp_model in expensive_models))
        
        if expensive_model_count > config.validation.max_expensive_models_per_experiment:
            validation_errors.append(f"Too many expensive models ({expensive_model_count}) > maximum ({config.validation.max_expensive_models_per_experiment})")
        
        # Agent parameter validation
        for agent in config.agents:
            if not (config.validation.temperature_range[0] <= agent.temperature <= config.validation.temperature_range[1]):
                validation_errors.append(f"Agent {agent.agent_id} temperature ({agent.temperature}) outside valid range {config.validation.temperature_range}")
            
            if not (config.validation.max_tokens_range[0] <= agent.max_output_tokens <= config.validation.max_tokens_range[1]):
                validation_errors.append(f"Agent {agent.agent_id} max_output_tokens ({agent.max_output_tokens}) outside valid range {config.validation.max_tokens_range}")
            
            # Check if model exists
            if agent.model_id not in config.models:
                validation_errors.append(f"Agent {agent.agent_id} references unknown model: {agent.model_id}")
            else:
                model = config.models[agent.model_id]
                if config.validation.require_system_prompt_support and not model.supports_system_prompt:
                    validation_errors.append(f"Agent {agent.agent_id} uses model {agent.model_id} which doesn't support system prompts")
        
        # Cost estimation and validation
        estimated_cost = self.estimate_cost(config)
        if estimated_cost > config.validation.max_estimated_cost_per_run:
            validation_errors.append(f"Estimated cost ({estimated_cost:.2f}) > maximum ({config.validation.max_estimated_cost_per_run})")
        elif estimated_cost > config.validation.warn_if_cost_above:
            self.logger.warning(f"High estimated cost: ${estimated_cost:.2f} per run")
        
        if validation_errors:
            raise ConfigurationError("Configuration validation failed:\n" + "\n".join(f"- {error}" for error in validation_errors))
    
    def estimate_cost(self, config: ExperimentConfig) -> float:
        """
        Estimate the cost per experiment run.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Estimated cost in USD per run
        """
        total_cost = 0.0
        
        # Estimate tokens per agent per round
        estimated_input_tokens_per_round = 2000  # Context + preferences + discussion
        estimated_output_tokens_per_round = 500  # Proposals + reasoning
        
        for agent in config.agents:
            if agent.model_id in config.models:
                model = config.models[agent.model_id]
                
                # Calculate token usage for this agent
                total_input_tokens = estimated_input_tokens_per_round * config.environment.t_rounds
                total_output_tokens = estimated_output_tokens_per_round * config.environment.t_rounds
                
                # Add cost
                input_cost = (total_input_tokens / 1_000_000) * model.input_cost_per_1m
                output_cost = (total_output_tokens / 1_000_000) * model.output_cost_per_1m
                
                total_cost += input_cost + output_cost
        
        return total_cost
    
    def create_agent_configs_from_models(self, 
                                       model_ids: List[str],
                                       config: ExperimentConfig,
                                       agent_template: Optional[Dict[str, Any]] = None) -> List[AgentConfig]:
        """
        Create agent configurations from a list of model IDs.
        
        Args:
            model_ids: List of model IDs to create agents for
            config: Base configuration
            agent_template: Template for agent parameters
            
        Returns:
            List of AgentConfig objects
        """
        template = agent_template or {}
        agents = []
        
        for i, model_id in enumerate(model_ids):
            strategic_level_str = template.get('strategic_level', 'balanced')
            strategic_level = StrategicLevel(strategic_level_str)
            
            # Generate system prompt if not provided
            system_prompt = template.get('system_prompt')
            if system_prompt is None:
                system_prompt = self._generate_system_prompt(
                    i + 1, model_id, config.environment, strategic_level
                )
            
            agents.append(AgentConfig(
                agent_id=f"agent_{i+1}_{model_id.replace('-', '_')}",
                model_id=model_id,
                temperature=template.get('temperature', 0.7),
                max_output_tokens=template.get('max_output_tokens', 2048),
                strategic_level=strategic_level,
                reasoning_steps=template.get('reasoning_steps', True),
                use_chain_of_thought=template.get('use_chain_of_thought', True),
                system_prompt=system_prompt,
                custom_parameters=template.get('custom_parameters', {})
            ))
        
        return agents
    
    def _generate_system_prompt(self, 
                              agent_num: int,
                              model_id: str, 
                              environment: EnvironmentConfig,
                              strategic_level: StrategicLevel) -> str:
        """Generate a system prompt for an agent."""
        return f"""You are Agent {agent_num} in a multi-party negotiation for allocating {environment.m_items} items among {environment.n_agents} participants.

Your goal is to maximize your own utility while engaging in {strategic_level.value} negotiation.
You have your own preferences for different items, and other agents have theirs.

Key principles for {strategic_level.value} strategy:
- Be strategic but fair in your proposals  
- Consider the preferences of other agents to build consensus
- Use logical reasoning to support your proposals
- Adapt your strategy based on how the negotiation progresses

The negotiation will last up to {environment.t_rounds} rounds.
Remember: You need unanimous agreement to reach a consensus."""
    
    def save_config(self, config: ExperimentConfig, output_path: Union[str, Path]) -> None:
        """
        Save experiment configuration to YAML file.
        
        Args:
            config: Configuration to save
            output_path: Path to save configuration
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary for serialization
        config_dict = asdict(config)
        
        # Convert enums back to strings
        def convert_enums(obj):
            if isinstance(obj, dict):
                return {k: convert_enums(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_enums(item) for item in obj]
            elif isinstance(obj, Enum):
                return obj.value
            else:
                return obj
        
        config_dict = convert_enums(config_dict)
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    def create_preset_config(self, preset_name: str) -> ExperimentConfig:
        """
        Create a configuration from a preset.
        
        Args:
            preset_name: Name of the preset
            
        Returns:
            ExperimentConfig for the preset
            
        Raises:
            ConfigurationError: If preset is unknown
        """
        if preset_name == "baseline_o3_vs_haiku":
            return self._create_o3_vs_haiku_preset()
        elif preset_name == "cooperative_matrix":
            return self._create_cooperative_matrix_preset()
        elif preset_name == "scaling_laws_study":
            return self._create_scaling_laws_preset()
        else:
            raise ConfigurationError(f"Unknown preset: {preset_name}")
    
    def _create_o3_vs_haiku_preset(self) -> ExperimentConfig:
        """Create O3 vs Haiku baseline preset."""
        # Load the base schema and modify for this preset
        schema_path = self.config_dir / "experiment_schema.yaml"
        if schema_path.exists():
            base_config = self.load_config(schema_path)
        else:
            base_config = ExperimentConfig()
        
        # Override for O3 vs Haiku experiment
        base_config.environment.n_agents = 3
        base_config.environment.m_items = 5
        base_config.environment.t_rounds = 6
        base_config.environment.gamma_discount = 0.9
        
        base_config.preferences.system_type = PreferenceSystemType.VECTOR
        base_config.preferences.cosine_similarity = 0.95
        
        base_config.execution.experiment_name = "o3_vs_haiku_baseline"
        
        return base_config
    
    def _create_cooperative_matrix_preset(self) -> ExperimentConfig:
        """Create cooperative matrix preset."""
        config = ExperimentConfig()
        config.environment.n_agents = 4
        config.environment.m_items = 8
        config.environment.t_rounds = 10
        config.environment.gamma_discount = 0.95
        
        config.preferences.system_type = PreferenceSystemType.MATRIX
        config.preferences.self_weight = 0.5
        config.preferences.others_weight = 0.5
        config.preferences.cosine_similarity = 0.1  # Low competition
        
        config.execution.experiment_name = "cooperative_matrix_experiment"
        
        return config
    
    def _create_scaling_laws_preset(self) -> ExperimentConfig:
        """Create scaling laws study preset."""
        config = ExperimentConfig()
        config.environment.n_agents = 3
        config.environment.m_items = 5 
        config.environment.t_rounds = 6
        
        config.execution.experiment_name = "scaling_laws_study"
        config.execution.batch_size = 20
        
        return config


# Export main classes
__all__ = [
    'ExperimentConfig',
    'ExperimentConfigManager', 
    'EnvironmentConfig',
    'ModelConfig',
    'AgentConfig',
    'PreferenceConfig',
    'ConfigurationError',
    'PreferenceSystemType',
    'StrategicLevel',
    'ReasoningCapability'
]
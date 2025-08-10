#!/usr/bin/env python3
"""
Model Configuration System Demonstration

This script demonstrates how to use the enhanced model configuration system
for multi-agent negotiation experiments with support for multiple LLM providers.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from negotiation.model_config import (
    create_default_registry,
    create_o3_vs_haiku_config,
    ModelRegistry,
    ExperimentModelConfig,
    ConfigLoader,
    ConfigValidator,
    ModelFamily,
    ProviderConfig,
    ModelProvider,
    AgentModelConfig
)

from negotiation.config_integration import (
    ConfigurationManager,
    create_model_manager_from_config_name
)


def demo_model_registry():
    """Demonstrate model registry functionality."""
    print("🔍 Model Registry Demonstration")
    print("=" * 50)
    
    registry = create_default_registry()
    
    # List all models
    all_models = registry.list_models()
    print(f"📋 Total available models: {len(all_models)}")
    
    # Group by family
    families = {}
    for model in all_models:
        family_name = model.family.value
        if family_name not in families:
            families[family_name] = []
        families[family_name].append(model)
    
    print("\n📊 Models by family:")
    for family, models in families.items():
        print(f"  {family}: {len(models)} models")
        for model in models[:3]:  # Show first 3 examples
            print(f"    - {model.model_id} ({model.display_name})")
        if len(models) > 3:
            print(f"    ... and {len(models) - 3} more")
    
    # Show specific model details
    print("\n🔍 Model Details Example:")
    o3_model = registry.get_model("o3")
    if o3_model:
        print(f"  Model ID: {o3_model.model_id}")
        print(f"  Display Name: {o3_model.display_name}")
        print(f"  Provider: {o3_model.provider.value}")
        print(f"  Context Window: {o3_model.context_window:,} tokens")
        print(f"  Reasoning Capability: {o3_model.reasoning_capability}")
        print(f"  Supports System Prompt: {o3_model.supports_system_prompt}")
    
    # Add custom Princeton cluster model
    print("\n🏫 Adding Princeton Cluster Model:")
    cluster_model = registry.add_princeton_cluster_model(
        model_id="custom-llama-70b",
        display_name="Custom Llama 70B (Princeton)",
        family=ModelFamily.LLAMA,
        local_path="/scratch/gpfs/DANQIC/models/llama-70b-custom",
        context_window=32000,
        requires_gpu=True,
        estimated_vram_gb=80
    )
    print(f"  ✅ Added: {cluster_model.display_name}")
    print(f"  📁 Path: {cluster_model.local_path}")


def demo_configuration_creation():
    """Demonstrate creating configurations."""
    print("\n⚙️  Configuration Creation Demonstration")
    print("=" * 50)
    
    # Create basic O3 vs Haiku config
    print("📝 Creating O3 vs Haiku configuration...")
    config = create_o3_vs_haiku_config()
    
    print(f"  Config Name: {config.config_name}")
    print(f"  Description: {config.description}")
    print(f"  Agents: {len(config.agents)}")
    print(f"  Providers: {list(config.providers.keys())}")
    
    # Show agent details
    print("\n👥 Agent Configuration:")
    for agent in config.agents:
        print(f"  - {agent.agent_id}:")
        print(f"    Model: {agent.model_spec.display_name}")
        print(f"    Provider: {agent.model_spec.provider.value}")
        print(f"    Temperature: {agent.temperature}")
        print(f"    Strategic Level: {agent.strategic_level}")
    
    # Create a more complex multi-model configuration
    print("\n🌟 Creating Multi-Model Configuration...")
    registry = create_default_registry()
    
    complex_config = ExperimentModelConfig(
        config_name="multi_model_showdown",
        description="Multi-model negotiation experiment featuring diverse AI systems",
        version="1.0",
        providers={
            "openai": ProviderConfig(
                provider=ModelProvider.OPENAI,
                requests_per_minute=30,
                api_key="dummy-openai-key"  # For demo purposes
            ),
            "anthropic": ProviderConfig(
                provider=ModelProvider.ANTHROPIC,
                requests_per_minute=60,
                api_key="dummy-anthropic-key"
            ),
            "openrouter": ProviderConfig(
                provider=ModelProvider.OPENROUTER,
                requests_per_minute=100,
                api_key="dummy-openrouter-key"
            )
        },
        available_models={
            "o3": registry.get_model("o3"),
            "claude-3-sonnet": registry.get_model("claude-3-sonnet"),
            "llama-3-70b": registry.get_model("llama-3-70b"),
            "qwen-2.5-72b": registry.get_model("qwen-2.5-72b")
        },
        agents=[
            AgentModelConfig(
                agent_id="reasoning_champion_o3",
                model_spec=registry.get_model("o3"),
                temperature=0.7,
                strategic_level="balanced",
                system_prompt="You are a highly analytical negotiator who uses logical reasoning."
            ),
            AgentModelConfig(
                agent_id="creative_claude",
                model_spec=registry.get_model("claude-3-sonnet"),
                temperature=0.8,
                strategic_level="cooperative",
                system_prompt="You are a creative and empathetic negotiator seeking win-win outcomes."
            ),
            AgentModelConfig(
                agent_id="open_llama",
                model_spec=registry.get_model("llama-3-70b"),
                temperature=0.7,
                strategic_level="balanced",
                system_prompt="You are a straightforward negotiator representing open-source values."
            ),
            AgentModelConfig(
                agent_id="multilingual_qwen",
                model_spec=registry.get_model("qwen-2.5-72b"),
                temperature=0.6,
                strategic_level="balanced",
                system_prompt="You are a strategic negotiator with global perspective."
            )
        ]
    )
    
    print(f"  ✅ Created complex config with {len(complex_config.agents)} diverse agents")
    return complex_config


def demo_yaml_operations():
    """Demonstrate YAML configuration operations."""
    print("\n📄 YAML Configuration Operations")
    print("=" * 50)
    
    # Create configuration manager
    config_dir = Path("experiments/configs")
    manager = ConfigurationManager(config_dir=config_dir)
    
    # Create and save a configuration
    print("💾 Creating and saving configuration...")
    config = create_o3_vs_haiku_config()
    
    # Add dummy API keys for validation
    config.providers["openai"].api_key = "demo-openai-key"
    config.providers["anthropic"].api_key = "demo-anthropic-key"
    
    config_path = manager.save_config(config, "demo_o3_vs_haiku")
    print(f"  ✅ Saved to: {config_path}")
    
    # List available configurations
    print("\n📋 Available configurations:")
    configs = manager.list_configs()
    for config_name in configs:
        print(f"  - {config_name}")
    
    # Load configuration back
    print("\n📖 Loading configuration...")
    loaded_config = manager.load_config("demo_o3_vs_haiku")
    print(f"  ✅ Loaded: {loaded_config.config_name}")
    print(f"  📝 Description: {loaded_config.description}")
    
    # Validate environment
    print("\n🔍 Environment validation:")
    validation_results = manager.validate_environment(loaded_config)
    print(f"  Valid: {validation_results['valid']}")
    if validation_results['errors']:
        print("  Errors:")
        for error in validation_results['errors']:
            print(f"    - {error}")
    if validation_results['warnings']:
        print("  Warnings:")
        for warning in validation_results['warnings']:
            print(f"    - {warning}")


def demo_validation():
    """Demonstrate configuration validation."""
    print("\n✅ Configuration Validation Demonstration")
    print("=" * 50)
    
    validator = ConfigValidator()
    
    # Test valid configuration
    print("🔍 Testing valid configuration...")
    valid_config = create_o3_vs_haiku_config()
    valid_config.providers["openai"].api_key = "test-key"
    valid_config.providers["anthropic"].api_key = "test-key"
    
    errors = validator.validate_config(valid_config)
    if errors:
        print(f"  ❌ Unexpected errors: {errors}")
    else:
        print("  ✅ Configuration is valid!")
    
    # Test invalid configuration
    print("\n🔍 Testing invalid configuration...")
    invalid_config = create_o3_vs_haiku_config()
    # Remove a required provider
    del invalid_config.providers["openai"]
    
    errors = validator.validate_config(invalid_config)
    print(f"  Found {len(errors)} validation errors:")
    for i, error in enumerate(errors, 1):
        print(f"    {i}. {error}")


def demo_model_manager_integration():
    """Demonstrate integration with model manager."""
    print("\n🤖 Model Manager Integration Demonstration")
    print("=" * 50)
    
    # Note: This demo won't actually make API calls
    print("📋 Creating model manager from configuration...")
    
    try:
        config = create_o3_vs_haiku_config()
        config.providers["openai"].api_key = "demo-key"
        config.providers["anthropic"].api_key = "demo-key"
        
        manager_class = ConfigurationManager()
        model_manager = manager_class.create_model_manager(config)
        
        print(f"  ✅ Created model manager")
        print(f"  👥 Registered agents: {model_manager.list_agents()}")
        
        # Show agent information
        for agent_id in model_manager.list_agents():
            info = model_manager.get_agent_info(agent_id)
            print(f"\n  🤖 Agent: {agent_id}")
            print(f"    Model: {info['display_name']}")
            print(f"    Provider: {info['provider']}")
            print(f"    Context Window: {info['capabilities']['context_window']:,} tokens")
            print(f"    Reasoning: {info['capabilities']['reasoning_capability']}")
    
    except Exception as e:
        print(f"  ⚠️  Demo mode - would create manager (missing dependencies: {e})")


def demo_princeton_cluster_usage():
    """Demonstrate Princeton cluster model configuration."""
    print("\n🏫 Princeton Cluster Configuration Demonstration")
    print("=" * 50)
    
    registry = create_default_registry()
    
    # Add cluster models
    print("➕ Adding Princeton cluster models...")
    
    cluster_models = [
        {
            "model_id": "llama-3-8b-princeton",
            "display_name": "Llama 3 8B (Princeton Cluster)",
            "family": ModelFamily.LLAMA,
            "local_path": "/scratch/gpfs/DANQIC/models/llama-3-8b-instruct",
            "estimated_vram_gb": 16
        },
        {
            "model_id": "llama-3-70b-princeton", 
            "display_name": "Llama 3 70B (Princeton Cluster)",
            "family": ModelFamily.LLAMA,
            "local_path": "/scratch/gpfs/DANQIC/models/llama-3-70b-instruct",
            "estimated_vram_gb": 80
        }
    ]
    
    for model_info in cluster_models:
        model_spec = registry.add_princeton_cluster_model(**model_info)
        print(f"  ✅ {model_spec.display_name}")
        print(f"    📁 Path: {model_spec.local_path}")
        print(f"    🎮 VRAM: {model_spec.estimated_vram_gb}GB")
    
    # Create cluster-based experiment configuration
    print("\n⚙️  Creating cluster experiment configuration...")
    
    cluster_config = ExperimentModelConfig(
        config_name="princeton_cluster_experiment",
        description="Negotiation experiment using Princeton cluster resources",
        version="1.0",
        providers={
            "princeton_cluster": ProviderConfig(
                provider=ModelProvider.PRINCETON_CLUSTER,
                requests_per_minute=1000  # No API rate limits for local models
            ),
            "anthropic": ProviderConfig(
                provider=ModelProvider.ANTHROPIC,
                api_key="dummy-key"  # For comparison with cloud model
            )
        },
        available_models={
            "llama-3-8b-princeton": registry.get_model("llama-3-8b-princeton"),
            "llama-3-70b-princeton": registry.get_model("llama-3-70b-princeton"),
            "claude-3-haiku": registry.get_model("claude-3-haiku")
        },
        agents=[
            AgentModelConfig(
                agent_id="cluster_small_llama",
                model_spec=registry.get_model("llama-3-8b-princeton"),
                temperature=0.7,
                strategic_level="balanced"
            ),
            AgentModelConfig(
                agent_id="cluster_large_llama",
                model_spec=registry.get_model("llama-3-70b-princeton"),
                temperature=0.7,
                strategic_level="balanced"
            ),
            AgentModelConfig(
                agent_id="cloud_claude_baseline",
                model_spec=registry.get_model("claude-3-haiku"),
                temperature=0.7,
                strategic_level="balanced"
            )
        ],
        cluster_config={
            "slurm_partition": "gpu",
            "slurm_time": "02:00:00",
            "slurm_nodes": 1,
            "slurm_gpus_per_node": 2,
            "conda_env": "negotiation",
            "python_path": "/opt/conda/envs/negotiation/bin/python"
        }
    )
    
    print(f"  ✅ Configuration created with {len(cluster_config.agents)} agents")
    print("  🖥️  SLURM configuration included for cluster deployment")
    
    # Validate cluster configuration
    validator = ConfigValidator()
    errors = validator.validate_config(cluster_config)
    if errors:
        print(f"  ⚠️  Validation issues: {len(errors)}")
        for error in errors:
            print(f"    - {error}")
    else:
        print("  ✅ Cluster configuration is valid!")


def demo_gemini_gemma_showcase():
    """Demonstrate the new Gemini and Gemma models."""
    print("\n🌟 Gemini & Gemma Models Showcase")
    print("=" * 50)
    
    registry = create_default_registry()
    
    # Show Gemini models by generation
    print("🧠 Google Gemini Models by Generation:")
    
    gemini_generations = {
        "2.5": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"],
        "2.0": ["gemini-2.0-flash", "gemini-2.0-flash-lite"],
        "1.5": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.5-flash-8b"]
    }
    
    for generation, model_ids in gemini_generations.items():
        print(f"\n  📅 Gemini {generation} Series:")
        for model_id in model_ids:
            model = registry.get_model(model_id)
            if model:
                print(f"    • {model.display_name}")
                print(f"      Context: {model.context_window:,} tokens")
                print(f"      Vision: {'✅' if model.supports_vision else '❌'}")
                print(f"      Functions: {'✅' if model.supports_function_calling else '❌'}")
                if model.input_cost_per_1m:
                    print(f"      Cost: ${model.input_cost_per_1m:.3f}/${model.output_cost_per_1m:.2f} per 1M tokens")
    
    # Show Gemma models
    print("\n🔓 Open Source Gemma Models:")
    gemma_models = ["gemma-2-27b", "gemma-2-12b", "gemma-2-4b", "gemma-2-1b"]
    
    for model_id in gemma_models:
        model = registry.get_model(model_id)
        if model:
            print(f"\n  🤖 {model.display_name}:")
            print(f"    VRAM: {model.estimated_vram_gb}GB")
            print(f"    Speed: {model.estimated_speed}")
            print(f"    GPU Required: {'Yes' if model.requires_gpu else 'No'}")
            print(f"    Reasoning: {model.reasoning_capability}")
    
    # Load and show the showcase configuration
    print("\n⚙️  Gemini/Gemma Showcase Configuration:")
    try:
        manager = ConfigurationManager()
        showcase_config = manager.load_config("gemini_gemma_showcase")
        
        print(f"  📋 {showcase_config.config_name}")
        print(f"  📝 {showcase_config.description}")
        print(f"  👥 {len(showcase_config.agents)} agents configured")
        
        print("\n  🎭 Agent Lineup:")
        for agent in showcase_config.agents:
            print(f"    • {agent.agent_id}: {agent.model_spec.display_name}")
            print(f"      Strategy: {agent.strategic_level}")
            print(f"      Temperature: {agent.temperature}")
        
    except Exception as e:
        print(f"  ⚠️  Showcase config not available: {e}")


def demo_cost_estimation():
    """Demonstrate cost estimation features."""
    print("\n💰 Cost Estimation Demonstration")
    print("=" * 50)
    
    registry = create_default_registry()
    
    # Show cost information for different models including new Gemini ones
    print("📊 Model Cost Comparison (per 1M tokens):")
    
    models_to_compare = [
        "o3", "claude-3-haiku", "claude-3-sonnet", "gpt-4o",
        "gemini-2.5-pro", "gemini-2.5-flash", "gemini-1.5-pro"
    ]
    
    for model_id in models_to_compare:
        model = registry.get_model(model_id)
        if model and model.input_cost_per_1m and model.output_cost_per_1m:
            print(f"\n  💰 {model.display_name}:")
            print(f"    Input:  ${model.input_cost_per_1m:.3f}/1M tokens")
            print(f"    Output: ${model.output_cost_per_1m:.2f}/1M tokens")
            print(f"    Speed:  {model.estimated_speed}")
            print(f"    Reasoning: {model.reasoning_capability}")
        elif model:
            print(f"\n  💰 {model.display_name}: Cost info not available")
    
    # Show cost-effectiveness analysis
    print("\n📈 Cost-Effectiveness Analysis:")
    print("  For budget-conscious experiments:")
    budget_models = ["gemini-2.5-flash", "gemini-1.5-flash", "claude-3-haiku", "gemma-2-4b"]
    for model_id in budget_models:
        model = registry.get_model(model_id)
        if model:
            cost_indicator = "💰" if model.input_cost_per_1m and model.input_cost_per_1m < 0.1 else "🔓" if model.provider.value == "openrouter" and "gemma" in model_id else "💎"
            print(f"    {cost_indicator} {model.display_name} - {model.reasoning_capability} reasoning, {model.estimated_speed} speed")
    
    # Estimate cost for a typical negotiation
    print("\n🧮 Cost Estimation for Typical 3-Agent Negotiation:")
    estimated_tokens_per_agent = {
        "input": 5000,   # System prompt + context + other agents' messages
        "output": 1500   # Agent's own responses
    }
    
    config = create_o3_vs_haiku_config()
    total_estimated_cost = 0
    
    for agent in config.agents:
        model_spec = agent.model_spec
        if model_spec.input_cost_per_1m and model_spec.output_cost_per_1m:
            input_cost = (estimated_tokens_per_agent["input"] / 1_000_000) * model_spec.input_cost_per_1m
            output_cost = (estimated_tokens_per_agent["output"] / 1_000_000) * model_spec.output_cost_per_1m
            agent_cost = input_cost + output_cost
            total_estimated_cost += agent_cost
            
            print(f"  🤖 {agent.agent_id} ({model_spec.display_name}):")
            print(f"    Estimated cost: ${agent_cost:.4f}")
    
    print(f"\n  💵 Total estimated cost per negotiation: ${total_estimated_cost:.4f}")
    print(f"  📈 For 100 experiments: ${total_estimated_cost * 100:.2f}")


async def main():
    """Run all demonstrations."""
    print("🚀 Model Configuration System Demonstration")
    print("=" * 80)
    
    # Run all demonstrations
    demo_model_registry()
    complex_config = demo_configuration_creation()
    demo_yaml_operations()
    demo_validation()
    demo_model_manager_integration()
    demo_princeton_cluster_usage()
    demo_gemini_gemma_showcase()
    demo_cost_estimation()
    
    print("\n" + "=" * 80)
    print("✅ All demonstrations completed successfully!")
    print("\n📚 Key Features Demonstrated:")
    print("  • Model registry with 20+ pre-configured models")
    print("  • Support for OpenAI, Anthropic, Google, OpenRouter, and Princeton cluster")
    print("  • Complete Gemini lineup: 2.5, 2.0, and 1.5 series (8 models)")
    print("  • Open-source Gemma models: 27B, 12B, 4B, 1B variants")
    print("  • YAML-based configuration with validation")
    print("  • Cost estimation and performance characteristics")
    print("  • Integration with existing experiment framework")
    print("  • Princeton cluster SLURM job configuration")
    
    print("\n🎯 Next Steps:")
    print("  1. Set up API keys in environment variables")
    print("  2. Create your own experiment configuration")
    print("  3. Run: python experiments/o3_vs_haiku_baseline.py --config your_config.yaml")
    print("  4. For Princeton cluster: Use SLURM scripts with cluster_config settings")


if __name__ == "__main__":
    asyncio.run(main())
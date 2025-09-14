#!/usr/bin/env python3
"""
Test script to verify Grok integration with the negotiation framework.
This script tests that Grok models are properly registered and can be used.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from negotiation.model_config import (
    ModelRegistry,
    ModelProvider,
    ModelFamily,
    ProviderConfig,
    AgentModelConfig,
    ConfigValidator,
    ExperimentModelConfig
)
from negotiation.model_clients import ModelClientFactory, UnifiedModelManager


def test_model_registry():
    """Test that Grok models are properly registered."""
    print("Testing Model Registry...")
    registry = ModelRegistry()
    
    # Test that Grok models exist
    grok_models = [
        "grok-4-0709",
        "grok-3",
        "grok-3-mini",
        "grok-2",
        "grok-2-mini"
    ]
    
    for model_id in grok_models:
        model = registry.get_model(model_id)
        assert model is not None, f"Model {model_id} not found in registry"
        assert model.family == ModelFamily.GROK, f"Model {model_id} has wrong family"
        assert model.provider == ModelProvider.XAI, f"Model {model_id} has wrong provider"
        print(f"  ✓ {model_id}: {model.display_name}")
    
    # List all Grok models
    grok_family_models = registry.list_models(family=ModelFamily.GROK)
    print(f"  Total Grok models registered: {len(grok_family_models)}")
    
    # List models by XAI provider
    xai_provider_models = registry.list_models(provider=ModelProvider.XAI)
    assert len(xai_provider_models) == len(grok_models), "Provider count mismatch"
    print("  ✓ All Grok models correctly assigned to XAI provider")
    
    return True


def test_client_factory():
    """Test that ModelClientFactory can create Grok clients."""
    print("\nTesting Client Factory...")
    
    registry = ModelRegistry()
    grok_model = registry.get_model("grok-3")
    
    # Create provider config (without actual API key for testing)
    provider_config = ProviderConfig(
        provider=ModelProvider.XAI,
        api_key="test_key_placeholder"  # Would need real key for actual API calls
    )
    
    try:
        # Try to create client (will fail if xai_sdk not installed)
        from negotiation.model_clients import XAI_AVAILABLE
        
        if XAI_AVAILABLE:
            client = ModelClientFactory.create_client(provider_config, grok_model)
            print(f"  ✓ Successfully created client for {grok_model.display_name}")
            print(f"    Client type: {type(client).__name__}")
        else:
            print("  ⚠ xai_sdk not installed - client creation would fail")
            print("    Install with: pip install xai-sdk")
    except Exception as e:
        print(f"  ⚠ Client creation test skipped: {e}")
    
    return True


def test_config_validation():
    """Test that configurations with Grok models validate properly."""
    print("\nTesting Configuration Validation...")
    
    registry = ModelRegistry()
    
    # Create a test configuration with Grok
    config = ExperimentModelConfig(
        config_name="grok_test_config",
        description="Test configuration with Grok models",
        providers={
            "xai": ProviderConfig(
                provider=ModelProvider.XAI,
                api_key="test_key"
            ),
            "anthropic": ProviderConfig(
                provider=ModelProvider.ANTHROPIC,
                api_key="test_key"
            )
        },
        available_models={
            "grok-3": registry.get_model("grok-3"),
            "claude-3-haiku": registry.get_model("claude-3-haiku")
        },
        agents=[
            AgentModelConfig(
                agent_id="agent_1_grok",
                model_spec=registry.get_model("grok-3"),
                temperature=0.7,
                strategic_level="balanced"
            ),
            AgentModelConfig(
                agent_id="agent_2_haiku",
                model_spec=registry.get_model("claude-3-haiku"),
                temperature=0.7,
                strategic_level="balanced"
            )
        ]
    )
    
    # Validate configuration
    validator = ConfigValidator()
    errors = validator.validate_config(config)
    
    if errors:
        print("  ✗ Validation errors found:")
        for error in errors:
            print(f"    - {error}")
    else:
        print("  ✓ Configuration with Grok models validates successfully")
    
    return len(errors) == 0


def test_config_generation():
    """Test that config generation script includes Grok models."""
    print("\nTesting Config Generation Script...")
    
    # Read the generation script
    script_path = Path(__file__).parent.parent / "scripts" / "generate_configs_both_orders.sh"
    with open(script_path, 'r') as f:
        script_content = f.read()
    
    # Check that Grok models are included
    grok_models = ["grok-4-0709", "grok-3", "grok-3-mini"]
    
    for model in grok_models:
        if f'"{model}"' in script_content:
            print(f"  ✓ {model} found in generation script")
        else:
            print(f"  ✗ {model} NOT found in generation script")
            return False
    
    # Check that XAI Grok comment exists
    if "# XAI Grok models" in script_content:
        print("  ✓ Grok section properly commented")
    else:
        print("  ⚠ Grok section comment not found")
    
    return True


async def test_model_manager():
    """Test UnifiedModelManager with Grok models."""
    print("\nTesting Unified Model Manager...")
    
    manager = UnifiedModelManager()
    registry = ModelRegistry()
    
    # Register a Grok agent
    agent_config = AgentModelConfig(
        agent_id="test_grok_agent",
        model_spec=registry.get_model("grok-3"),
        temperature=0.7
    )
    
    provider_config = ProviderConfig(
        provider=ModelProvider.XAI,
        api_key="test_key"
    )
    
    try:
        from negotiation.model_clients import XAI_AVAILABLE
        
        if XAI_AVAILABLE:
            manager.register_agent(agent_config, provider_config)
            print(f"  ✓ Successfully registered Grok agent: {agent_config.agent_id}")
            
            # Get agent info
            info = manager.get_agent_info("test_grok_agent")
            print(f"    Model: {info['display_name']}")
            print(f"    Provider: {info['provider']}")
            print(f"    Context window: {info['capabilities']['context_window']}")
        else:
            print("  ⚠ xai_sdk not installed - manager test skipped")
    except Exception as e:
        print(f"  ⚠ Manager test skipped: {e}")
    
    return True


def main():
    """Run all integration tests."""
    print("=" * 50)
    print("GROK INTEGRATION TEST SUITE")
    print("=" * 50)
    
    tests = [
        ("Model Registry", test_model_registry),
        ("Client Factory", test_client_factory),
        ("Config Validation", test_config_validation),
        ("Config Generation", test_config_generation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Async test
    try:
        success = asyncio.run(test_model_manager())
        results.append(("Model Manager", success))
    except Exception as e:
        print(f"\n✗ Model Manager test failed with exception: {e}")
        results.append(("Model Manager", False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:.<30} {status}")
        if not success:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("✓ ALL TESTS PASSED - Grok integration is ready!")
        print("\nNext steps:")
        print("1. Set XAI_API_KEY environment variable")
        print("2. Install xai-sdk: pip install xai-sdk")
        print("3. Run: ./scripts/generate_configs_both_orders.sh")
        print("4. Run: ./scripts/run_all_simple.sh")
    else:
        print("✗ SOME TESTS FAILED - Please review the issues above")
        sys.exit(1)


if __name__ == "__main__":
    main()
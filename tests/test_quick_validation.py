#!/usr/bin/env python3
"""
Quick validation test for Steps 8.1 and 8.2 without running full experiments.
"""

import sys
from pathlib import Path
import json
import yaml
import time

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.o3_vs_haiku_baseline import O3VsHaikuExperiment
from negotiation.llm_agents import LLMConfig, ModelType, SimulatedAgent


def test_enhanced_config_creation():
    """Test Step 8.1: Enhanced configuration creation only."""
    print("🧪 Testing Enhanced Config Creation...")
    
    experiment = O3VsHaikuExperiment()
    
    # Create test agents
    agents = [
        SimulatedAgent("test_o3", LLMConfig(
            model_type=ModelType.O3,
            temperature=0.8,
            max_tokens=2000,
            system_prompt="O3 test prompt",
            custom_parameters={"reasoning_steps": True}
        )),
        SimulatedAgent("test_haiku", LLMConfig(
            model_type=ModelType.CLAUDE_3_HAIKU,
            temperature=0.7,
            max_tokens=1500
        ))
    ]
    
    # Test preferences and items
    preferences = {
        "test_o3": [8.5, 2.3, 6.7, 1.9, 9.1],
        "test_haiku": [3.2, 7.8, 4.5, 8.9, 2.1]
    }
    items = ["Item1", "Item2", "Item3", "Item4", "Item5"]
    
    base_config = {
        "m_items": 5,
        "n_agents": 2,
        "t_rounds": 4,
        "gamma_discount": 0.9,
        "competition_level": 0.85,
        "known_to_all": False,
        "random_seed": 12345
    }
    
    start_time = time.time()
    
    # Test enhanced config creation
    enhanced_config = experiment._create_enhanced_config(
        base_config, agents, preferences, items, start_time, "test_exp_001"
    )
    
    print("✅ Enhanced config created successfully!")
    
    # Verify all required sections exist
    required_sections = [
        "experiment_metadata",
        "agent_configurations",
        "preference_configuration", 
        "item_configuration",
        "proposal_order_configuration",
        "negotiation_flow_configuration"
    ]
    
    missing_sections = []
    for section in required_sections:
        if section in enhanced_config:
            print(f"  ✅ {section}")
        else:
            missing_sections.append(section)
            print(f"  ❌ {section}: MISSING")
    
    if missing_sections:
        print(f"❌ Missing sections: {missing_sections}")
        return False
    
    # Test JSON serialization
    json_str = json.dumps(enhanced_config, indent=2, default=str)
    print(f"  📄 JSON serialization successful ({len(json_str)} chars)")
    
    # Verify some key values
    print(f"  📊 Agent configurations: {len(enhanced_config['agent_configurations'])}")
    print(f"  🎯 Preference vectors: {len(enhanced_config['preference_configuration']['preferences_by_agent'])}")
    print(f"  ⏱️ Duration: {enhanced_config['experiment_metadata']['duration_seconds']:.3f}s")
    
    return True


def test_yaml_schema_comprehensive():
    """Test Step 8.2: Complete YAML schema loading and validation."""
    print("🧪 Testing Complete YAML Schema...")
    
    schema_path = Path("experiments/configs/model_config_schema.yaml")
    if not schema_path.exists():
        print(f"❌ Schema file not found: {schema_path}")
        return False
    
    # Load YAML
    with open(schema_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("✅ YAML loaded successfully!")
    
    # Test all major sections
    expected_sections = {
        "config_name": "Basic info",
        "providers": "Model providers", 
        "available_models": "Model definitions",
        "agents": "Agent configs",
        "environment": "Core parameters (n,m,t,γ)",
        "preferences": "Preference system", 
        "negotiation_flow": "Phase configuration",
        "proposal_order_analysis": "Order tracking",
        "analysis": "Metrics and detection",
        "experiment_metadata": "Metadata config",
        "cluster_config": "Princeton cluster",
        "validation_rules": "Parameter validation"
    }
    
    missing_sections = []
    for section, description in expected_sections.items():
        if section in config:
            print(f"  ✅ {section}: {description}")
        else:
            missing_sections.append(section)
            print(f"  ❌ {section}: MISSING")
    
    if missing_sections:
        print(f"❌ Missing YAML sections: {missing_sections}")
        return False
    
    # Test specific subsections
    print("  🔧 Environment subsections:")
    env = config["environment"]
    for param in ["num_agents", "num_items", "max_rounds", "discount_factor"]:
        if param in env:
            print(f"    ✅ {param}: {env[param]}")
        else:
            print(f"    ❌ {param}: MISSING")
            return False
    
    print(f"  📊 Available models: {len(config['available_models'])}")
    print(f"  🎯 Analysis categories: {len(config['analysis'])}")
    
    return True


def main():
    """Run quick validation tests."""
    print("🚀 Quick Validation for Steps 8.1 and 8.2")
    print("=" * 50)
    
    # Test Step 8.1
    print("\n📋 STEP 8.1: Enhanced Hyperparameter Logging")
    step_8_1_success = test_enhanced_config_creation()
    
    # Test Step 8.2  
    print("\n📋 STEP 8.2: Complete YAML Schema Structure")
    step_8_2_success = test_yaml_schema_comprehensive()
    
    # Summary
    print("\n" + "=" * 50)
    if step_8_1_success and step_8_2_success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Steps 8.1 and 8.2 are working correctly!")
        return 0
    else:
        print("❌ Some tests failed.")
        print(f"Step 8.1: {'✅ PASSED' if step_8_1_success else '❌ FAILED'}")
        print(f"Step 8.2: {'✅ PASSED' if step_8_2_success else '❌ FAILED'}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
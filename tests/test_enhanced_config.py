#!/usr/bin/env python3
"""
Test script to verify enhanced configuration capture works correctly.
"""

import asyncio
import sys
from pathlib import Path
import json

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.o3_vs_haiku_baseline import O3VsHaikuExperiment
from negotiation.llm_agents import LLMConfig, ModelType

async def test_enhanced_config():
    """Test the enhanced configuration capture functionality."""
    
    print("🧪 Testing Enhanced Configuration Capture...")
    
    # Create a simple experiment instance
    experiment = O3VsHaikuExperiment()
    
    # Create mock agents to test config capture
    from negotiation.llm_agents import SimulatedAgent
    
    agents = [
        SimulatedAgent("test_o3_agent", LLMConfig(
            model_type=ModelType.O3,
            temperature=0.8,
            max_tokens=2000,
            timeout=30.0,
            system_prompt="Test O3 agent prompt"
        )),
        SimulatedAgent("test_haiku_agent", LLMConfig(
            model_type=ModelType.CLAUDE_3_HAIKU,
            temperature=0.7,
            max_tokens=1500,
            custom_parameters={"reasoning_steps": True}
        ))
    ]
    
    # Mock preferences and items
    preferences = {
        "test_o3_agent": [8.5, 2.3, 6.7, 1.9, 9.1],
        "test_haiku_agent": [3.2, 7.8, 4.5, 8.9, 2.1]
    }
    items = ["Item1", "Item2", "Item3", "Item4", "Item5"]
    
    # Test config
    base_config = {
        "m_items": 5,
        "n_agents": 2,
        "t_rounds": 4,
        "gamma_discount": 0.9,
        "competition_level": 0.85,
        "known_to_all": False,
        "random_seed": 12345
    }
    
    # Test the enhanced config creation
    import time
    start_time = time.time()
    
    try:
        enhanced_config = experiment._create_enhanced_config(
            base_config, agents, preferences, items, start_time, "test_experiment_123"
        )
        
        print("✅ Enhanced configuration created successfully!")
        print(f"📊 Config sections: {list(enhanced_config.keys())}")
        
        # Verify required sections exist
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
                print(f"✅ {section}: Present")
            else:
                missing_sections.append(section)
                print(f"❌ {section}: Missing")
        
        if not missing_sections:
            print("\n🎉 All required configuration sections are present!")
            
            # Test JSON serialization
            json_str = json.dumps(enhanced_config, indent=2, default=str)
            print(f"📄 JSON serialization: Success ({len(json_str)} chars)")
            
            # Sample some key values
            print(f"\n📋 Sample Configuration Values:")
            print(f"  Experiment ID: {enhanced_config['experiment_metadata']['experiment_id']}")
            print(f"  Duration: {enhanced_config['experiment_metadata']['duration_seconds']:.3f}s")
            print(f"  Agent Count: {len(enhanced_config['agent_configurations'])}")
            
            # Check agent-specific hyperparameters
            for agent_id, agent_config in enhanced_config['agent_configurations'].items():
                print(f"  {agent_id}:")
                print(f"    Model: {agent_config['model_type']}")
                print(f"    Temperature: {agent_config['temperature']}")
                print(f"    Max Tokens: {agent_config['max_tokens']}")
            
            # Check preference configuration
            pref_count = len(enhanced_config['preference_configuration']['preferences_by_agent'])
            print(f"  Preference Vectors: {pref_count} agents configured")
            
            return True
            
        else:
            print(f"❌ Missing sections: {missing_sections}")
            return False
            
    except Exception as e:
        print(f"❌ Enhanced config creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_enhanced_config())
    if success:
        print("\n✅ Enhanced configuration test PASSED!")
        sys.exit(0)
    else:
        print("\n❌ Enhanced configuration test FAILED!")
        sys.exit(1)
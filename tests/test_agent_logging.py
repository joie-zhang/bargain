#!/usr/bin/env python3
"""
Test script to verify agent experience logging functionality.

This script runs a minimal negotiation to test that the agent experience logging
system properly captures input prompts and output responses.
"""

import asyncio
import json
import tempfile
from pathlib import Path
import os

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from negotiation import (
    AgentFactory,
    AgentConfiguration,
    ExperimentConfiguration,
    create_competitive_preferences,
    ModularNegotiationRunner,
)
from negotiation.llm_agents import ModelType, SimulatedAgent, LLMConfig


async def test_agent_logging():
    """Test the agent experience logging system."""
    print("🧪 Testing Agent Experience Logging System")
    print("=" * 60)
    
    # Create temporary directory for test results
    with tempfile.TemporaryDirectory() as temp_dir:
        results_dir = Path(temp_dir) / "test_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Using temporary results directory: {results_dir}")
        
        # Create simulated agents for testing
        agent_configs = []
        agents = []
        
        for i in range(2):  # Just 2 agents for simple test
            config = LLMConfig(
                model_type=ModelType.TEST_STRONG,
                temperature=0.7,
                max_tokens=1000,
                system_prompt=f"You are test agent {i+1}."
            )
            
            agent = SimulatedAgent(
                agent_id=f"test_agent_{i+1}",
                config=config,
                strategic_level="balanced"
            )
            agents.append(agent)
        
        print(f"👥 Created {len(agents)} test agents: {[a.agent_id for a in agents]}")
        
        # Create simple preferences
        preference_manager = create_competitive_preferences(
            n_agents=len(agents),
            m_items=3,  # Just 3 items for simplicity
            cosine_similarity=0.8
        )
        
        items = ["Apple", "Orange", "Banana"]
        
        # Create negotiation runner WITH logging enabled
        runner = ModularNegotiationRunner(
            agents=agents,
            preferences=preference_manager,
            items=items,
            max_rounds=2,  # Short test
            discount_factor=0.9,
            log_level="INFO",
            results_dir=str(results_dir),
            enable_agent_logging=True  # This is the key line!
        )
        
        print("\n🚀 Starting test negotiation...")
        
        # Run the negotiation
        outcome = await runner.run_negotiation()
        
        print(f"\n✅ Negotiation completed!")
        print(f"   Consensus reached: {outcome.consensus_reached}")
        print(f"   Final round: {outcome.final_round}")
        
        # Check if logging files were created
        agent_log_dir = results_dir / "agent_experiences"
        
        if agent_log_dir.exists():
            print(f"\n📊 Agent experience logs created in: {agent_log_dir}")
            
            # List and analyze each agent's log file
            for agent in agents:
                log_file = agent_log_dir / f"{agent.agent_id}_experience.json"
                
                if log_file.exists():
                    print(f"\n📄 {agent.agent_id} log file: {log_file}")
                    
                    # Load and analyze the log
                    with open(log_file, 'r') as f:
                        log_data = json.load(f)
                    
                    # Print summary
                    metadata = log_data.get("metadata", {})
                    interactions = log_data.get("interactions", [])
                    
                    print(f"   📈 Total interactions: {len(interactions)}")
                    print(f"   🎭 Phases participated: {metadata.get('phases_participated', [])}")
                    
                    # Show sample interaction
                    if interactions:
                        sample = interactions[0]
                        print(f"   📝 Sample interaction (Round {sample['round_number']}, {sample['phase']}):")
                        print(f"       Input prompt length: {len(sample['input_prompt'])} chars")
                        print(f"       Response length: {len(sample['raw_response'])} chars")
                        print(f"       Response time: {sample['response_time_seconds']:.2f}s")
                        
                        # Show first 200 chars of prompt
                        prompt_preview = sample['input_prompt'][:200] + "..." if len(sample['input_prompt']) > 200 else sample['input_prompt']
                        print(f"       Prompt preview: {prompt_preview}")
                        
                        # Show first 100 chars of response
                        response_preview = sample['raw_response'][:100] + "..." if len(sample['raw_response']) > 100 else sample['raw_response']
                        print(f"       Response preview: {response_preview}")
                    
                    print(f"   ✅ Log validation: PASSED")
                else:
                    print(f"   ❌ ERROR: Log file not found for {agent.agent_id}")
                    return False
        else:
            print(f"❌ ERROR: Agent log directory not found: {agent_log_dir}")
            return False
        
        print("\n🎉 Agent Experience Logging Test: SUCCESS")
        print(f"💡 The logging system successfully captured {len(agents)} agents' complete experiences!")
        return True


async def main():
    """Main test function."""
    try:
        success = await test_agent_logging()
        if success:
            print("\n✅ ALL TESTS PASSED!")
            return 0
        else:
            print("\n❌ TESTS FAILED!")
            return 1
    except Exception as e:
        print(f"\n💥 Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
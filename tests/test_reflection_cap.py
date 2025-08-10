#!/usr/bin/env python3
"""
Test script to verify the reflection character cap is working correctly.
"""

from experiments.o3_vs_haiku_baseline import O3VsHaikuExperiment

def test_reflection_truncation():
    """Test that reflection content is properly truncated."""
    experiment = O3VsHaikuExperiment()
    
    # Test the _extract_key_takeaways method with different inputs
    class MockResponse:
        def __init__(self, content):
            self.content = content
    
    class MockAgent:
        def __init__(self, agent_id):
            self.agent_id = agent_id
    
    agent = MockAgent("test_agent")
    
    # Test 1: Short content (should not be truncated)
    short_content = "This is a short reflection about the negotiation."
    response = MockResponse(short_content)
    
    import asyncio
    
    async def test_async():
        result = await experiment._extract_key_takeaways(agent, response, 1, False, max_chars=2000)
        print(f"Short content test:")
        print(f"Original: {len(short_content)} chars")
        print(f"Result: {len(result)} chars")
        print(f"Content preserved: {result == short_content}")
        print()
        
        # Test 2: Long content (should be truncated)
        long_content = "This is a very long reflection. " * 100  # Should be ~3200 chars
        long_response = MockResponse(long_content)
        
        result = await experiment._extract_key_takeaways(agent, long_response, 1, False, max_chars=500)
        print(f"Long content test (max_chars=500):")
        print(f"Original: {len(long_content)} chars")
        print(f"Result: {len(result)} chars")
        print(f"Truncated: {len(result) <= 500}")
        print(f"Preview: {result[:100]}...")
        print()
        
        # Test 3: Default character limit from config
        config_max = experiment.default_config.get("max_reflection_chars", 2000)
        result = await experiment._extract_key_takeaways(agent, long_response, 1, False)
        print(f"Default config test (max_chars={config_max}):")
        print(f"Original: {len(long_content)} chars")
        print(f"Result: {len(result)} chars")
        print(f"Within limit: {len(result) <= config_max}")
        print()
    
    asyncio.run(test_async())

if __name__ == "__main__":
    test_reflection_truncation()
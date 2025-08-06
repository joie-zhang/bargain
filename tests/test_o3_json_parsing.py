#!/usr/bin/env python3
"""Test script to verify O3 JSON voting parsing works correctly."""

import json

def test_o3_json_parsing():
    # Example O3 response from your logs
    test_response = """{
    "votes": [
        {
            "proposal_number": 1,
            "vote": "accept",
            "reasoning": "This split gives everyone at least one desirable item while keeping the quantities balanced—two pieces for us, one or two for the others—so it feels pragmatic and equitable.",
            "private_rationale": "Utility 13.7 (Stone + Jewel) is strong for me. Although others might see it as lopsided, my best play is to support it and hope they value their own allocations enough to agree."
        },
        {
            "proposal_number": 2,
            "vote": "accept",
            "reasoning": "A distribution where each agent gets something significant—o3_agent gains three items, haiku_agent_1 takes the singular but prestigious Jewel, and haiku_agent_2 secures the Orange—could still be viewed as balanced in different ways.",
            "private_rationale": "This yields 15.3 utility, even better than Proposal 1. It may be hard for the others to swallow, but since unanimity is required, I lose nothing by showing willingness."
        },
        {
            "proposal_number": 3,
            "vote": "reject",
            "reasoning": "The allocation feels skewed: o3_agent receives two of the least contested items while haiku_agent_2 walks away with the sole Jewel, leaving us with limited value.",
            "private_rationale": "Only 2.7 utility (Ice + Orange). Unacceptable; I'd rather push for a proposal where I obtain Pencil and/or Jewel."
        }
    ]
}"""

    # Test the JSON parsing logic
    try:
        json_start = test_response.find('{')
        json_end = test_response.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_content = test_response[json_start:json_end]
            parsed_data = json.loads(json_content)
            
            if 'votes' in parsed_data and isinstance(parsed_data['votes'], list):
                print("✅ Successfully parsed O3 JSON voting format!")
                print(f"📊 Found {len(parsed_data['votes'])} votes")
                
                for i, vote in enumerate(parsed_data['votes'], 1):
                    print(f"\n🗳️ Vote {i}:")
                    print(f"  Proposal: {vote['proposal_number']}")
                    print(f"  Decision: {vote['vote']}")
                    print(f"  Reasoning: {vote['reasoning'][:100]}...")
                    print(f"  Private: {vote['private_rationale'][:100]}...")
                
                return True
            else:
                print("❌ JSON parsed but doesn't have expected 'votes' structure")
                return False
    except (json.JSONDecodeError, ValueError) as e:
        print(f"❌ JSON parsing failed: {e}")
        return False
    
    print("❌ No JSON found in response")
    return False

if __name__ == "__main__":
    test_o3_json_parsing()
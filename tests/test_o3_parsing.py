#!/usr/bin/env python3
"""
Test O3 voting response parsing
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Sample O3 voting response based on the patterns we've seen
sample_o3_response = """
Looking at these proposals, I need to evaluate each one strategically:

For Proposal #1: I believe this allocation leaves me with too little value relative to others, so I can't support it. The utility for me is only 4.9, far below the other proposals and my reservation level. I'm better off holding out for a fairer split.

For Proposal #2: This proposal gives me significant value and seems a reasonable starting point for agreement. The utility for me is 17.0—excellent. Even though haiku_agent_2 currently receives nothing, signaling acceptance keeps pressure on them to adjust.

For Proposal #3: This division also provides a solid share for me while appearing balanced overall. Utility for me is 14.4, still strong, and haiku_agent_2 authored it, so it has the best chance of unanimous approval. Accepting maximizes the likelihood of closing the deal with good value.

My votes are: reject proposal 1, accept proposal 2, accept proposal 3.
"""

# Test the parsing functions
def test_o3_format_detection():
    from experiments.o3_vs_haiku_baseline import O3VsHaikuExperiment
    
    # Create experiment instance 
    exp = O3VsHaikuExperiment()
    
    # Test format detection
    is_o3_format = exp._is_o3_voting_format(sample_o3_response, "o3_agent")
    print(f"O3 format detected: {is_o3_format}")
    
    # Test parsing
    enumerated_proposals = [
        {"proposal_number": 1, "allocation": {"o3_agent": [3, 4], "haiku_agent_1": [0], "haiku_agent_2": [1]}},
        {"proposal_number": 2, "allocation": {"o3_agent": [0, 1, 4], "haiku_agent_1": [2, 3], "haiku_agent_2": []}},
        {"proposal_number": 3, "allocation": {"o3_agent": [0, 2, 4], "haiku_agent_1": [1, 3], "haiku_agent_2": []}}
    ]
    
    if is_o3_format:
        result = exp._parse_o3_voting_format(sample_o3_response, "o3_agent", enumerated_proposals)
        print(f"Parsed votes: {result}")
        
        for vote in result['votes']:
            print(f"Proposal {vote['proposal_number']}: {vote['vote']} - {vote['reasoning']}")
    else:
        print("O3 format not detected - checking indicators...")
        content_lower = sample_o3_response.lower()
        o3_indicators = [
            'proposal' in content_lower,
            'accept' in content_lower or 'reject' in content_lower,
            'vote' in content_lower or 'voting' in content_lower,
            'utility' in content_lower,
            'reasoning' in content_lower or 'rationale' in content_lower,
            len(sample_o3_response) > 100,
            any(item in content_lower for item in ['camera', 'apple', 'notebook', 'flower', 'pencil', 'stone', 'kite', 'hat'])
        ]
        
        print(f"Indicators: {o3_indicators}")
        print(f"Sum: {sum(o3_indicators)} (need >=4)")

if __name__ == "__main__":
    test_o3_format_detection()
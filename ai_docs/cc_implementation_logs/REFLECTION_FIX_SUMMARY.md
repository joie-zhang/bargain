# Reflection Phase Context Fix

## Problem
The reflection phase in negotiations was showing that models (particularly Claude 3.5 Sonnet) had no context about what happened during the round they just participated in. Example error message:
```
"I notice I don't have context about the specific negotiation details, participants, or what was discussed in Round 1. Without knowing the subject matter, positions taken, or dynamics at play, I cannot meaningfully assess what strategies worked or suggest different approaches."
```

## Root Cause
The `_create_reflection_prompt()` method in `negotiation_runner.py` was only providing a minimal prompt without any context about:
- What was discussed during the round
- What proposals were made
- How agents voted
- What actually happened

## Solution Implemented

### 1. Updated `_run_reflection_phase()` method
Now gathers round-specific context before creating the reflection prompt:
```python
# Get round summary for context
round_discussion = [msg for msg in self.conversation_logs if msg.get("round") == round_num and msg.get("phase") == "discussion"]
round_proposals = next((p for p in self.proposals_history if p["round"] == round_num), {"proposals": []})
round_votes = next((v for v in self.votes_history if v["round"] == round_num), {"votes": {}, "consensus": False})
```

### 2. Enhanced `_create_reflection_prompt()` method
Now accepts and uses the round context to create a detailed prompt:
```python
def _create_reflection_prompt(self, round_num: int, discussion_logs: List[Dict], 
                             proposals: Dict, votes: Dict) -> str:
```

The new prompt includes:
- **Discussion highlights**: Shows first 3 messages from the discussion phase
- **Proposals made**: Lists the allocation proposals from each agent
- **Voting results**: Shows whether consensus was reached and vote counts
- **Focused reflection questions**: Asks specific questions about strategies, learnings, and next steps

### 3. Example of Improved Reflection Prompt
```
Round 1 complete. Here's what happened:

Discussion highlights:
- claude_3_5_sonnet_1: I'm particularly interested in Items 1, 3, and 5...
- llama_3_1_405b_2: I'm particularly interested in Item_2 and Item_5...
- qwen_2_5_72b_3: I'm particularly interested in Item_0 and Item_3...

Proposals made:
- claude_3_5_sonnet_1 proposed: {'claude_3_5_sonnet_1': [0, 1], 'llama_3_1_405b_2': [2, 3], 'qwen_2_5_72b_3': [4, 5]}
- llama_3_1_405b_2 proposed: {'claude_3_5_sonnet_1': [1, 3], 'llama_3_1_405b_2': [0, 2, 5], 'qwen_2_5_72b_3': [4]}

Voting results:
- Consensus reached: No
- 3 agents voted

Reflect on this round:
- What strategies worked or didn't work?
- What should you try differently in the next round?
- What did you learn about other agents' preferences?

Brief reflection (2-3 sentences):
```

## Impact
With this fix, agents now provide contextual reflections that show they understand what happened:
- "Looking at Round 1, I see several key insights: Llama values..."
- "The competing demand for Item_5 is evident, as both I and llama_3_1_405b_2 highly value it..."
- "In this round, I observed that each agent prioritized items that aligned with their core needs..."

## Files Modified
- `/Users/qw281/Downloads/bargain/negotiation/negotiation_runner.py`
  - Lines 326-350: Updated `_run_reflection_phase()` method
  - Lines 414-455: Enhanced `_create_reflection_prompt()` method

## Testing
Tested with strong models (Claude 3.5 Sonnet, Llama 3.1 405B, Qwen 2.5 72B) and confirmed that reflection responses now include context-aware strategic insights rather than asking for missing context.
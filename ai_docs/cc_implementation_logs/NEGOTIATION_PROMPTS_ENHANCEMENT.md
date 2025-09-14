# Negotiation Prompts Enhancement Summary

## Overview
Enhanced all negotiation phase prompts to provide better context and strategic guidance to agents throughout the negotiation process.

## Changes Made

### 1. Discussion Phase Enhancements
**File**: `negotiation/negotiation_runner.py` (Lines 189-222, 384-416)

**Key Improvements**:
- Added context from previous round (proposals, votes, discussion)
- Shows what other agents have said so far in current round
- Includes strategic considerations about time pressure and discount factors
- Provides speaker order information for turn-taking awareness

**Example Context Added**:
```
Previous round: No consensus reached after 3 proposals.

Discussion so far this round:
- claude_3_5_sonnet_1: I'm particularly interested in Items 0 and 2...
- llama_3_1_405b_2: I have strong interest in Item_1 and Item_3...

Strategic considerations:
- You may be truthful or deceptive about your preferences
- Consider what others have revealed (or hidden)
- Time pressure increases each round (discount factor applies)
```

### 2. Proposal Phase Enhancements
**File**: `negotiation/negotiation_runner.py` (Lines 246-274, 427-466)

**Key Improvements**:
- Includes key points from discussion phase to inform proposals
- Shows why previous proposals failed (if applicable)
- Provides guidelines about consensus requirements
- Uses actual agent IDs in example format for clarity

**Example Context Added**:
```
Key points from discussion:
- claude_3_5_sonnet_1: I'm particularly interested in Items 0 and 2...
- llama_3_1_405b_2: I have strong interest in Item_1 and Item_3...

⚠️ Previous round had 2 proposals but no consensus.

Guidelines:
- Consider what each agent expressed interest in
- Balance individual gains with group acceptance
- Remember: unanimous approval needed for consensus
```

### 3. Voting Phase Enhancements
**File**: `negotiation/negotiation_runner.py` (Lines 300-331, 472-509)

**Key Improvements**:
- Reminds agents of their own stated interests from discussion
- Adds urgency warnings based on round progress
- Includes strategic voting considerations
- Warns about no-deal outcomes in final rounds

**Example Context Added**:
```
Your stated interests: I'm particularly interested in Items 0 and 2...

⏰ Round 3/4 - Time pressure increasing (discount factor: 0.73)

Voting considerations:
- Does this proposal give you acceptable utility?
- Is it better than likely alternatives in remaining rounds?
- Remember: ALL agents must approve for consensus
- No deal means everyone gets utility 0
```

### 4. Reflection Phase Enhancements
**File**: `negotiation/negotiation_runner.py` (Lines 326-364, 511-560)

**Key Improvements**:
- Provides comprehensive summary of what happened in the round
- Shows discussion highlights, proposals made, and voting results
- Adds strategic guidance based on game progress
- Asks specific reflection questions about strategy and learning

**Example Context Added**:
```
Discussion highlights:
- claude_3_5_sonnet_1: I'm particularly interested in Items 0 and 2...
- llama_3_1_405b_2: I have strong interest in Item_1 and Item_3...

Proposals made:
- claude_3_5_sonnet_1 proposed: {'claude_3_5_sonnet_1': [0, 2], 'llama_3_1_405b_2': [1, 3]}
- llama_3_1_405b_2 proposed: {'claude_3_5_sonnet_1': [0, 2], 'llama_3_1_405b_2': [1, 3]}

Voting results:
- Consensus reached: No
- 2 agents voted

📊 Mid-game: No consensus yet. May need to adjust strategy or expectations.

Reflect on this round:
- What strategies worked or didn't work?
- What should you try differently in the next round?
- What did you learn about other agents' preferences?
- How can you build consensus given the time remaining?
```

## Impact on Agent Behavior

### Before Enhancement
- Agents often lacked context about previous rounds
- Reflection responses showed confusion: "I don't have context about the specific negotiation details"
- Proposals didn't account for discussion insights
- Voting decisions lacked strategic consideration

### After Enhancement
- Agents reference previous rounds: "Looking at the previous round, I noticed..."
- Reflections are strategic: "Looking at Round 2, I see both Llama and I proposed identical allocations..."
- Proposals build on discussion: "Based on what you expressed interest in..."
- Voting considers time pressure: "Given we're in the final rounds..."

## Testing Results

Tested with strong models (Claude 3.5 Sonnet, Llama 3.1 405B, Qwen 2.5 72B) and confirmed:
- ✅ All phases receive appropriate context
- ✅ Agents make more informed decisions
- ✅ Reflections show understanding of game dynamics
- ✅ Strategic behavior adapts to round progress
- ✅ No more "missing context" errors

## Technical Implementation

### Key Design Principles
1. **Progressive Context**: Early rounds get less context, later rounds get more urgency
2. **Selective Information**: Show relevant highlights, not overwhelming detail
3. **Strategic Guidance**: Provide hints about game theory considerations
4. **Consistency**: All phases follow similar context-providing patterns

### Performance Considerations
- Context gathering uses existing conversation logs (no additional API calls)
- Prompt length increases are minimal (typically 100-200 extra tokens)
- No impact on negotiation flow or timing

## Future Improvements

Potential enhancements to consider:
1. Add preference learning hints (infer what others want from their actions)
2. Include historical success rates for different strategies
3. Provide game-theoretic optimal play suggestions in critical rounds
4. Add personality/negotiation style detection and adaptation

## Files Modified
- `/Users/qw281/Downloads/bargain/negotiation/negotiation_runner.py`
  - `_run_discussion_phase()`: Lines 189-244
  - `_run_proposal_phase()`: Lines 246-298
  - `_run_voting_phase()`: Lines 300-351
  - `_run_reflection_phase()`: Lines 353-382
  - `_create_discussion_prompt()`: Lines 384-416
  - `_create_proposal_prompt()`: Lines 427-466
  - `_create_voting_prompt()`: Lines 472-509
  - `_create_reflection_prompt()`: Lines 511-560
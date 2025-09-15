# Prompt Clarity Improvement for 2-Agent Negotiations

## Problem Identified
The original negotiation prompts contained an ambiguity when there were exactly 2 agents negotiating:
- Original wording: "You are participating in a strategic negotiation with 2 agents"
- Ambiguity: Could mean either:
  - 2 agents total (including the recipient)
  - 2 other agents (3 total)

This confusion could lead to agents misunderstanding the negotiation setup and potentially making suboptimal decisions.

## Solution Implemented

### Changes Made
Modified two files to use clearer, unambiguous language:

1. **`strong_models_experiment/prompts/prompt_generator.py`**
2. **`negotiation/experiment_phases.py`**

### New Logic
```python
# Create clearer phrasing for 2-agent negotiations
if num_agents == 2:
    agent_phrase = "another agent"
else:
    agent_phrase = f"{num_agents - 1} other agents"
```

### Result
- **2 agents**: "You are participating in a strategic negotiation with **another agent**"
- **3 agents**: "You are participating in a strategic negotiation with **2 other agents**"
- **5 agents**: "You are participating in a strategic negotiation with **4 other agents**"

## Benefits
1. **Eliminates ambiguity**: Clear that the number refers to OTHER agents, not total agents
2. **Natural language**: "another agent" sounds more natural for 1-on-1 negotiations
3. **Consistent interpretation**: All agents will have the same understanding of the setup
4. **Better strategic decisions**: Agents can make more informed choices knowing the exact number of participants

## Testing
Created test file `tests/test_prompt_clarity.py` that verifies:
- 2-agent negotiations use "another agent"
- 3+ agent negotiations use "N other agents"
- All tests pass successfully

## Impact on Existing Experiments
This change only affects future experiments. Past experiment results remain valid as the second line of the prompt ("There are N agents participating (including you)") provided clarification, though the initial ambiguity may have caused some confusion.

## Recommendation
Re-run any critical experiments where 2-agent negotiation dynamics are being studied to ensure clean, unambiguous results.
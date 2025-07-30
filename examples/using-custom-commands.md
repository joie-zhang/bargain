# Using Custom Claude Commands

This guide demonstrates how to use the custom commands included in this template.

## Multi-Mind Analysis

Use `/multi-mind` when you need multiple perspectives on a complex problem:

```
> /multi-mind "Analyze the safety implications of this reward function"
```

This will spawn multiple specialist agents that independently analyze the problem and then cross-review each other's findings.

## Search Previous Conversations

Use `/search-all` to find insights from previous sessions:

```
> /search-all "gradient hacking"
```

This searches through:
- Your current session
- Saved conversation history
- Exported session JSONs

## Save Session State

Use `/page` to checkpoint your work before context fills:

```
> /page "reward-modeling-session-1"
```

Resume later with:
```bash
claude --resume reward-modeling-session-1
```

## Deep Function Analysis

Use `/analyze-function` for detailed code analysis:

```
> /analyze-function "def compute_reward(state, action, next_state):"
```

This provides:
- Line-by-line performance analysis
- Complexity assessment
- Edge case identification
- Optimization suggestions

## Create Your Own Commands

Use `/crud-claude-commands` to build custom commands:

```
> /crud-claude-commands create safety-check "Analyze this code for potential safety issues including: reward hacking, goal misgeneralization, and unintended optimization"
```

Now you can use:
```
> /safety-check
```

## Best Practices

1. **Start Complex Tasks with Multi-Mind**: Get diverse perspectives before diving deep
2. **Search Before Implementing**: Check if similar problems were solved before
3. **Page Frequently**: Save state every 30-45 minutes or before major transitions
4. **Create Domain-Specific Commands**: Build commands for your specific research area
5. **Combine Commands**: Use multiple commands together for comprehensive analysis

## Example Workflow

```
# 1. Start with multi-agent analysis
> /multi-mind "Design a safe exploration strategy for this RL agent"

# 2. Search for related work
> /search-all "safe exploration strategies"

# 3. Analyze specific implementations
> /analyze-function "def exploration_policy(state, epsilon):"

# 4. Save progress
> /page "safe-exploration-design-1"

# 5. Create a reusable command for future use
> /crud-claude-commands create safe-exploration-review "Review this exploration strategy for safety considerations including distributional shift, reward hacking, and catastrophic actions"
```
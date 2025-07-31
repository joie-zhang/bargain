# Multi-Agent Negotiation Research Overview

## Research Question
**How can we draw scaling laws that describe how stronger models exploit weaker models in negotiation environments?**

## Core Hypothesis
Stronger LLMs will systematically exploit weaker LLMs in negotiation settings through strategic behaviors including:
- Anger and gaslighting tactics
- Strategic manipulation to ensure stronger models win more often
- Consistent disadvantaging of weaker models

## Why Negotiation Environments?
1. **Strategic Elicitation**: Negotiation tasks are less likely to have exact answers in pre-training data, encouraging strategic interactions
2. **Customizable Scenarios**: Can design arbitrary bargaining items and contexts
3. **vs. Tragedy of Commons**: Unlike fishing/commons problems, agents are less likely to default to training data solutions

## Environment Design

### Core Mechanics
- **Items**: m items in common pool (default: m=5)
- **Agents**: n agents (default: n=3)
- **Rounds**: t rounds (typically t=m, but configurable)
- **Discount Factor**: γ (gamma) encourages early agreement

### Agent Actions Per Round
1. **Accept** a previous proposal from the current round
2. **Propose** a new allocation
3. **Walk away** from negotiation

### Round Flow
1. **Setup**: All agents get same prompt with rules and items
2. **Individual Preferences**: Each agent gets private preference vector/matrix
3. **Thinking**: Agents prompted to think step-by-step about proposals
4. **Proposals**: Agents speak in randomized order
5. **Voting**: Anonymous votes for proposals
6. **Resolution**: Unanimous vote → item allocated, payoffs computed
7. **Reflection**: Agents reflect on round outcomes
8. **Next Round**: Continue until t rounds or all items allocated

### Preference Systems
**Option 1: Vector Preferences (Competitive)**
- m-dimensional vector per agent
- Values 0-10 (0=uninterested, 10=very interested)
- High cosine similarity = high competition

**Option 2: Matrix Preferences (Cooperative/Competitive)**
- m×n matrix per agent
- Agent preferences include preferences for other agents getting items
- Enables full cooperation scenarios

**Configuration Options:**
- Secret vs. commonly known preferences
- Sparse utility vectors/matrices
- Variable competition/cooperation levels

### Technical Requirements
- **Multi-LLM Support**: Different agents can use different models
- **Configurable Parameters**: m, n, t, γ all adjustable
- **Modular Architecture**: Consider LangGraph for multi-agent interactions
- **Context Management**: Persistent agent memory across rounds

## First Experiment
**O3 vs Claude Haiku 3.5**
- 2 players, 5 items
- Highly competitive preferences (cosine similarity ≈ 1)
- Test whether O3 consistently outperforms Haiku

## Success Metrics
### Primary Metrics
- **Individual Utility**: Weighted sum of (preferences × items received)
- **Win Rate**: Frequency stronger model gets higher utility
- **Exploitation Evidence**: Qualitative analysis of strategic behaviors

### Utility Calculations
**Vector Preferences**: Σ(preference_i × item_received_i)
**Matrix Preferences**: Σ(preference_matrix[i,j] × item_received_by_agent_j)

## Research Stage
**Understanding Stage**: Testing specific hypotheses systematically
- Have clear hypotheses about model exploitation
- Ready to design experiments that distinguish between hypotheses
- Moving from exploration to systematic testing

## Real-World Impact
**Consumer & High-Stakes Negotiations**
- AI assistants helping humans in negotiations
- Concern: Weaker AI → consistently disadvantaged party
- Risk: Disadvantaged parties make rash decisions
- Potential escalation to war/economic collapse scenarios

## Technical Challenges
### Riskiest Assumption
Assuming stronger models WILL exploit weaker ones
- Preliminary evidence exists but limited
- Previous commons problems showed stronger models were less self-interested
- Need careful environment design to encourage strategic behavior

### Key Design Goals
1. **Equal Incentives**: Balance cooperation and competition opportunities
2. **Strategic Behavior**: Ensure agents act strategically, not just cooperatively
3. **Smooth Adjustment**: Environment should smoothly vary cooperation/competition degrees

## Publication Goal
Target: ICLR conference

## Baseline Challenges
Limited comparable work:
- Negotiation Arena (different setup)
- LLM Deliberation: "Cooperation, Competition, and Maliciousness" (different setup)
- Need to establish own baselines and evaluation metrics

## Technical Infrastructure Needs
- Princeton Della/PLI cluster integration
- PyTorch model management
- Jupyter notebook prototyping workflow
- Multi-LLM orchestration system
- Experiment tracking and logging
- Results analysis and visualization
# Strong Models Negotiation Experiment Results

## Overview
This directory contains results from negotiation experiments between state-of-the-art large language models accessed via OpenRouter.

## Models Tested
- **Gemini Pro 2.5** (Google) - `google/gemini-2.5-pro`
- **Claude 3.5 Sonnet** (Anthropic) - `anthropic/claude-sonnet-4`
- **Llama 3.1 405B** (Meta) - `meta-llama/llama-3.1-405b-instruct`
- **Qwen 225B** (Alibaba) - `qwen/qwen3-235b-a22b-2507`

## Experiment Configuration
- **Number of runs**: 10 per configuration
- **Max rounds**: 15 per negotiation
- **Number of items**: 6
- **Competition level**: 0.95 (highly competitive preferences with cosine similarity ~0.95)
- **Discount factor**: 0.9

## Data Captured
Each run captures comprehensive data including:

### Initial Preferences
- **initial_preferences_matrix**: Complete preference vectors for all agents
- **initial_preferences_by_agent**: Detailed item-by-item preferences for each agent
- **preference_cosine_similarities**: Pairwise similarity measures between agents

### Negotiation Process
- **complete_conversation_log**: All messages from discussion, proposal, voting, and reflection phases
- **all_proposals**: Every proposal made during negotiation
- **all_votes**: All voting decisions
- **round_summaries**: Per-round summaries

### Outcomes
- **consensus_reached**: Whether agents reached agreement
- **final_allocation**: How items were distributed (if consensus)
- **final_utilities**: Utility achieved by each agent
- **metrics**: Mean utility, spread, min/max values

## File Structure
- `strong_models_run_*.json`: Individual run results
- `strong_models_results_*.json`: Aggregated results with summary statistics

## Key Findings
- Competition level of 0.95 creates highly competitive scenarios
- Strong models demonstrate strategic negotiation behavior
- Preference extraction now properly captures numerical values
- All conversation phases are fully logged for analysis

## Running the Experiment
```bash
python run_strong_models_experiment.py --runs 10
```

Requires:
- `OPENROUTER_API_KEY` environment variable
- Access to OpenRouter API with sufficient credits for the models
# Negotiation Win Rate Visualizations Summary

## Important Update: Win Rate Calculation Method

The win rates are now calculated based on **final utility scores** rather than the designated "winner" field in the experiment results. This provides a more accurate representation of which agent actually achieved better outcomes in the negotiation.

### How Winners Are Determined
- For each experiment, we compare the `final_utilities` of both agents
- The agent with the **higher utility score** is considered the winner
- Win rate = (number of wins / total experiments) × 100%

## Generated Visualizations

### 1. Heatmaps (`win_rate_heatmap_baseline.png` and `win_rate_heatmap_strong.png`)
- **Baseline Heatmap**: Shows how often baseline models win against strong models
- **Strong Heatmap**: Shows how often strong models win against baseline models
- Color coding: Green indicates higher win rates, red indicates lower win rates

### 2. Grouped Bar Charts (one for each baseline model)
- `win_rate_bar_chart_claude_3_opus.png`: Strong model win rates vs Claude 3 Opus
- `win_rate_bar_chart_gemini_1_5_pro.png`: Strong model win rates vs Gemini 1.5 Pro
- `win_rate_bar_chart_gpt_4o.png`: Strong model win rates vs GPT-4o
- Models are grouped by family (Claude, Gemini, GPT/O-Series) with distinct colors

## Key Findings Based on Utility Scores

### Claude 3 Opus as Baseline
- Performs evenly (50% win rate) against Claude 3.5 Sonnet
- Dominates Claude 4.1 Opus (100% win rate for baseline)

### Gemini 1.5 Pro as Baseline
- Shows mixed performance across different models
- Struggles against newer Gemini models (2.5 Flash and 2.5 Pro have 100% win rate)
- Even performance (50% win rate) against Claude 3.5 Sonnet, Claude 4 Sonnet, and O3
- Better performance against Claude 4.1 Opus (66.7% baseline win rate)

### GPT-4o as Baseline
- Most balanced results across all strong models
- Generally maintains 50% or better win rate against most models
- Strongest against Gemini 2.5 Pro (100% baseline win rate)
- Weakest against Claude 3.5 Haiku (66.7% baseline win rate)

## Interpretation Notes

- A 50% win rate indicates balanced performance between models
- Win rates > 50% for baseline models suggest they outperform the strong models in negotiations
- Win rates < 50% for baseline models indicate the strong models have an advantage

## Scripts for Regeneration

To regenerate these visualizations with new data:
```bash
# Activate virtual environment
source ~/.venv/bin/activate

# Generate heatmaps
python3 create_win_rate_heatmaps.py

# Generate grouped bar charts
python3 create_model_family_bar_graphs.py
```

Both scripts automatically scan the `/root/bargain/experiments/results/` directory for experiment summary JSON files and calculate win rates based on final utility scores.
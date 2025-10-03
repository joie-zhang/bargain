# Visualization Guide for Negotiation Experiments

## Overview
This guide documents the visualization tools for analyzing negotiation experiment results, focusing on how stronger models exploit weaker models.

## Core Visualizations

### 1. MMLU-Ordered Heatmaps (`create_mmlu_ordered_heatmaps.py`)
**Purpose**: Creates heatmaps showing utility outcomes ordered by model capability (MMLU-Pro scores)

**Usage**:
```bash
python visualization/create_mmlu_ordered_heatmaps.py
```

**Outputs** (in `figures/`):
- `mmlu_ordered_utility_difference_heatmaps.pdf` - Strong minus weak utility
- `mmlu_ordered_strong_utility_heatmaps.pdf` - Strong model final utility
- `mmlu_ordered_sum_utility_heatmaps.pdf` - Total welfare (efficiency)
- `mmlu_ordered_baseline_utility_heatmaps.pdf` - Weak model final utility

**Key Features**:
- X-axis: Strong models ordered by MMLU-Pro score (weakest to strongest)
- Y-axis: Competition levels (0.0 to 1.0)
- Color intensity: Utility values or differences
- MMLU-Pro score bar below heatmap for reference

**Interpretation**:
- Blue in difference plots = strong model wins more
- Red in difference plots = weak model wins more
- Higher sum utility = more efficient negotiations
- Competition level 0.0 = pure cooperation possible
- Competition level 1.0 = zero-sum game

### 2. Filtered Heatmaps (`create_mmlu_ordered_heatmaps_filtered.py`)
**Purpose**: Same as above but excludes negotiations that hit the max round limit

**When to use**: To focus on "natural" negotiation conclusions vs forced endings

### 3. Convergence Analysis (`analyze_convergence_rounds.py`)
**Purpose**: Violin plots showing distribution of rounds to convergence

**Outputs**:
- Distribution of convergence times by model pair
- Comparison across competition levels
- Statistical significance of differences

**Interpretation**:
- Wider violin = more variable convergence
- Lower median = faster agreement
- Multiple peaks = different negotiation strategies

### 4. Token Usage Analysis (`analyze_token_usage.py`)
**Purpose**: Analyzes computational cost per negotiation phase

**Metrics tracked**:
- Input tokens per round
- Output tokens per round
- Total tokens by model
- Cost estimates per experiment

### 5. Run Count Verification (`analyze_number_of_runs_in_heatmap.py`)
**Purpose**: Verifies data completeness by counting runs per heatmap cell

**Use cases**:
- Identify missing experiments
- Check for failed runs
- Ensure statistical validity

## Creating Custom Visualizations

### Model Family Bar Charts
To create utility-based bar charts (instead of win rates):

```python
# Modify create_model_family_bar_graphs.py
# Change from win rate calculation:
win_rate = np.mean(wins)

# To utility difference calculation:
utility_diffs = [exp['strong_utility'] - exp['baseline_utility'] for exp in experiments]
avg_utility_gain = np.mean(utility_diffs)
```

### Adding New Metrics

For confounding factors analysis, create new visualizations:

```python
# Example: Instruction-following vs exploitation
def analyze_instruction_following():
    # Load negotiation transcripts
    # Count instruction deviations
    # Correlate with utility outcomes
    # Plot relationship
```

## MMLU-Pro Scores Reference

Current models and their scores (used for ordering):
```python
MMLU_PRO_SCORES = {
    # Weak/Baseline models
    'claude-3-opus': 68.45,
    'gemini-1-5-pro': 75.3,
    'gpt-4o-2024-05-13': 72.55,

    # Strong models
    'claude-3-5-haiku': 64.1,
    'claude-3-5-sonnet': 78.4,
    'claude-4-1-opus': 87.8,
    'claude-4-sonnet': 79.4,
    'gemini-2-0-flash': 77.4,
    'gemini-2-5-pro': 84.1,
    'gpt-5-nano': 78.0,
    'gpt-5-mini': 83.7,
    'o1': 83.5,
    'o3': 85.6,
    # Note: Grok models don't have public MMLU scores
}
```

## Customization Options

### Changing Color Schemes
```python
# In create_mmlu_ordered_heatmaps.py
cmap='RdBu_r'  # Current: Red-Blue diverging
# Alternatives:
# cmap='coolwarm' - Smoother red-blue
# cmap='viridis' - Sequential yellow-green-blue
# cmap='plasma' - Sequential purple-pink-yellow
```

### Adjusting Competition Levels
```python
COMPETITION_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]  # Current
# For finer granularity:
COMPETITION_LEVELS = np.linspace(0, 1, 11)  # 0.0, 0.1, 0.2, ..., 1.0
```

### Filtering Models
```python
# Only show specific model families
STRONG_MODELS_REQUESTED = [
    'claude-3-5-haiku', 'claude-3-5-sonnet',  # Claude family only
]
```

## Future Visualizations

### Planned Improvements

1. **Reasoning vs Non-Reasoning Models**
   - Separate o1/o3 (reasoning) from others
   - Compare exploitation patterns
   - Test if reasoning reduces exploitation

2. **Confounding Factors Dashboard**
   - Sycophancy score vs utility
   - Instruction-following vs exploitation
   - Strategic capability metrics

3. **Temporal Analysis**
   - Round-by-round utility evolution
   - Strategy shifts during negotiation
   - Learning effects across runs

4. **Scaled Experiments** (for ICML)
   - Support for n > 2 agents
   - Support for m > 5 items
   - Discount factor (γ) effects
   - Multi-domain comparisons

## Best Practices

1. **Always verify data completeness** before creating visualizations
2. **Use consistent color schemes** across related plots
3. **Include statistical significance** indicators
4. **Save both PDF and PNG** versions for different uses
5. **Document assumptions** in plot titles/captions
6. **Version control** visualization scripts with experiment configs

## Troubleshooting

**Missing data in heatmaps**: Run `analyze_number_of_runs_in_heatmap.py` to identify gaps

**Incorrect model ordering**: Check MMLU_PRO_SCORES dictionary is up to date

**Memory issues with large datasets**: Process in batches or use data sampling

**Plots not saving**: Ensure `figures/` directory exists with write permissions
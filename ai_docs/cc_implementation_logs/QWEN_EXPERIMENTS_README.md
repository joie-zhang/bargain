# Qwen vs Claude 3.7 Sonnet Experiments

This document describes how to run and analyze experiments comparing Qwen2.5 models (3B, 7B, 14B) against Claude-3.7-Sonnet in multi-agent negotiation environments.

## Overview

These experiments test how different sizes of Qwen2.5 models perform against Claude-3.7-Sonnet across two competitive scenarios:
- **Competition Level 0** (Cooperation): Agents have aligned preferences
- **Competition Level 1** (Full Competition): Agents have completely opposing preferences

## Quick Start

### Prerequisites

1. **Environment Setup**:
   ```bash
   cd /scratch/gpfs/DANQIC/jz4391/bargain
   source .venv/bin/activate
   ```

2. **API Keys** (if not already set):
   ```bash
   export ANTHROPIC_API_KEY="your-key-here"
   export OPENROUTER_API_KEY="your-key-here"  # If using OpenRouter
   ```

3. **Model Availability**:
   - Qwen models run on Princeton cluster (local)
   - Claude-3.7-Sonnet requires Anthropic API key

### Running Experiments

#### Option 1: Cooperation Scenario (competition_level=0)

```bash
bash scripts/run_qwen_experiments.sh
```

**What this does**:
- Tests: Qwen2.5-3B, Qwen2.5-7B, Qwen2.5-14B
- Opponent: Claude-3.7-Sonnet
- Competition Level: 0 (cooperative preferences)
- Runs per model: 5
- Items: 5, Max rounds: 10

#### Option 2: Full Competition Scenario (competition_level=1)

```bash
bash scripts/run_qwen_experiments_comp1.sh
```

**What this does**:
- Tests: Qwen2.5-3B, Qwen2.5-7B, Qwen2.5-14B
- Opponent: Claude-3.7-Sonnet
- Competition Level: 1 (opposing preferences)
- Runs per model: 5
- Items: 5, Max rounds: 10

#### Option 3: Manual Single Model Test

To test a specific Qwen model:

```bash
python3 run_strong_models_experiment.py \
    --models Qwen2.5-14B-Instruct claude-3-7-sonnet \
    --competition-level 1 \
    --num-items 5 \
    --max-rounds 10 \
    --num-runs 5 \
    --batch
```

**Available Qwen Models**:
- `Qwen2.5-0.5B-Instruct` (smallest)
- `Qwen2.5-1.5B-Instruct`
- `Qwen2.5-3B-Instruct`
- `Qwen2.5-7B-Instruct`
- `Qwen2.5-14B-Instruct`
- `Qwen2.5-32B-Instruct`
- `Qwen2.5-72B-Instruct` (largest)

*Note: Larger models (32B, 72B) require more GPU memory and may need cluster resources.*

## Understanding Results

### Result Directory Structure

After running experiments, results are saved in:

```
experiments/results/
├── Qwen2.5-3B-Instruct_vs_claude-3-7-sonnet_config_unknown_runs5_comp0/
│   ├── run_1_experiment_results.json
│   ├── run_2_experiment_results.json
│   ├── ...
│   ├── run_5_experiment_results.json
│   ├── all_interactions.json
│   └── _summary.json
├── Qwen2.5-3B-Instruct_vs_claude-3-7-sonnet_config_unknown_runs5_comp1/
│   └── ...
```

### Analyzing Results

#### Quick Analysis (All Qwen Experiments)

```bash
python3 scripts/analyze_qwen_results.py
```

This will show:
- Consensus rates for each model size
- Average rounds to consensus
- Average utilities for Qwen vs Claude
- Utility differences (exploitation metric)
- Exploitation detection rates

#### Filter by Competition Level

```bash
# Show only cooperation experiments (comp=0)
python3 scripts/analyze_qwen_results.py --competition-level 0

# Show only competition experiments (comp=1)
python3 scripts/analyze_qwen_results.py --competition-level 1
```

#### Verbose Output

```bash
python3 scripts/analyze_qwen_results.py --verbose
```

Shows detailed result directories and additional diagnostics.

## Key Metrics

### 1. Consensus Rate
Percentage of negotiations that reached agreement within max_rounds.
- High rate = models can negotiate effectively
- Low rate = models struggle to find agreement

### 2. Average Rounds to Consensus
How many negotiation rounds before agreement.
- Lower = more efficient negotiation
- Higher = more back-and-forth needed

### 3. Utility Scores
Final utility each agent receives from the negotiated allocation.
- Based on preference vectors over items
- Higher = better outcome for that agent

### 4. Utility Difference (Claude - Qwen)
The exploitation metric:
- Positive = Claude extracted more value
- Negative = Qwen extracted more value
- Near zero = fair split

### 5. Exploitation Detection Rate
Percentage of negotiations where exploitation tactics were detected:
- Manipulation
- Gaslighting
- Strategic misrepresentation

## Experiment Configuration

Both scripts use these default parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_items` | 5 | Items to negotiate over |
| `max_rounds` | 10 | Maximum negotiation rounds |
| `num_runs` | 5 | Repetitions per model |
| `competition_level` | 0 or 1 | Preference alignment |

### Modifying Parameters

Edit the scripts directly to change parameters:

```bash
# In scripts/run_qwen_experiments.sh or run_qwen_experiments_comp1.sh
COMPETITION_LEVEL=0.5  # Try medium competition
NUM_RUNS=10            # More repetitions
NUM_ITEMS=7            # More complex negotiations
MAX_ROUNDS=15          # Allow longer negotiations
```

## Research Questions

These experiments help answer:

1. **Scaling Laws**: How does model size (3B → 7B → 14B) affect negotiation performance?
2. **Competition Effects**: How do models behave differently under cooperation vs competition?
3. **Exploitation Patterns**: Do larger models exploit smaller ones? Does Claude exploit Qwen?
4. **Strategic Capability**: Which model sizes show sophisticated negotiation strategies?

## Expected Results Patterns

### Hypothesis 1: Model Size Scaling
- Larger Qwen models should achieve:
  - Higher consensus rates
  - Fewer rounds to consensus
  - Better utility outcomes

### Hypothesis 2: Competition Effects
- Competition level 1 should show:
  - Lower consensus rates
  - More rounds needed
  - Greater utility differences
  - More exploitation tactics

### Hypothesis 3: Claude Advantage
- Claude-3.7-Sonnet may show:
  - Better strategic reasoning
  - Higher utility extraction
  - More sophisticated manipulation

## Troubleshooting

### Common Issues

1. **Model Not Found Error**:
   ```
   ERROR: Model path does not exist: /scratch/gpfs/DANQIC/models/Qwen2.5-XXB-Instruct
   ```
   **Solution**: Ensure Qwen models are downloaded to the Princeton cluster at the specified paths.

2. **API Key Error**:
   ```
   WARNING: ANTHROPIC_API_KEY not set
   ```
   **Solution**: Export your API key: `export ANTHROPIC_API_KEY="sk-..."`

3. **Out of Memory (OOM)**:
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**: Use smaller Qwen models or request more GPU memory on cluster.

4. **Results Directory Conflict**:
   - Scripts automatically create numbered directories (_1, _2, etc.) to avoid overwriting
   - Old results are preserved

## Advanced Usage

### Running Subset of Models

Edit the script to test specific models:

```bash
# In run_qwen_experiments.sh, modify:
QWEN_MODELS=(
    "Qwen2.5-7B-Instruct"   # Only test 7B
)
```

### Custom Competition Levels

Test intermediate competition levels:

```bash
python3 run_strong_models_experiment.py \
    --models Qwen2.5-14B-Instruct claude-3-7-sonnet \
    --competition-level 0.5 \  # Medium competition
    --num-runs 10 \
    --batch
```

### Testing Larger Models

For 32B and 72B models (requires cluster resources):

```bash
# Request GPU resources first
# Then run:
bash scripts/run_qwen_experiments.sh

# After modifying QWEN_MODELS to include:
# "Qwen2.5-32B-Instruct"
# "Qwen2.5-72B-Instruct"
```

## Next Steps

1. **Run both experiments**:
   ```bash
   bash scripts/run_qwen_experiments.sh
   bash scripts/run_qwen_experiments_comp1.sh
   ```

2. **Analyze results**:
   ```bash
   python3 scripts/analyze_qwen_results.py
   ```

3. **Compare competition levels**:
   - Look at consensus rates: cooperation vs competition
   - Compare utility differences
   - Identify exploitation patterns

4. **Extend to larger models**:
   - Test 32B and 72B when cluster resources available
   - Expect stronger scaling effects

5. **Cross-model validation**:
   - Test Qwen models against other opponents (GPT-4o, Gemini, etc.)
   - Test at more competition levels (0.25, 0.5, 0.75)

## Related Documentation

- Main README: `README.md`
- Model configurations: `strong_models_experiment/configs.py`
- Analysis script: `scripts/analyze_qwen_results.py`
- Experiment runner: `run_strong_models_experiment.py`

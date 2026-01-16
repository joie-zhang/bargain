# Visualization Guide for Experiment Results

## Quick Start

To visualize the most recent batch of results:

```bash
cd /scratch/gpfs/DANQIC/jz4391/bargain

# Generate scaling regression plots (recommended first step)
python visualize_latest_batch.py --regplot

# Generate MMLU-ordered heatmaps
python visualize_latest_batch.py --heatmaps

# Analyze convergence rounds
python visualize_latest_batch.py --convergence

# Analyze token usage
python visualize_latest_batch.py --tokens

# Run all visualizations
python visualize_latest_batch.py --all

# Launch interactive Streamlit UI
python visualize_latest_batch.py --ui
```

## Visualization Options

### 1. Scaling Regression Plots (`--regplot`)
Creates regression plots showing the relationship between model capability (MMLU-Pro score) and negotiation outcomes.

**Output:** `visualization/figures/scaling_regplot_*.pdf`
- `scaling_regplot_combined.pdf` - Overall regression across all competition levels
- `scaling_regplot_by_competition.pdf` - Separate plots for each competition level
- `scaling_regplot_by_baseline.pdf` - Separate plots for each baseline model

### 2. MMLU-Ordered Heatmaps (`--heatmaps`)
Creates heatmaps with models ordered by MMLU-Pro performance.

**Output:** `visualization/figures/mmlu_ordered_*.pdf`
- Utility difference heatmaps
- Strong model utility heatmaps
- Sum of utilities heatmaps
- Baseline utility heatmaps

**Note:** This script may need path adjustments as it uses hardcoded paths. Check `visualization/create_mmlu_ordered_heatmaps.py` if needed.

### 3. Convergence Analysis (`--convergence`)
Analyzes how many rounds it takes for negotiations to reach consensus.

**Output:** `visualization/figures/convergence_*.pdf`

### 4. Token Usage Analysis (`--tokens`)
Analyzes token consumption across experiments.

**Output:** `visualization/figures/token_usage_*.pdf`

### 5. Interactive UI (`--ui`)
Launches a Streamlit-based web interface for exploring results interactively.

**Features:**
- Browse individual experiments
- Compare batches
- View conversation logs
- Analyze metrics

## Specifying a Different Batch

To visualize a specific batch instead of the most recent:

```bash
python visualize_latest_batch.py --batch-dir experiments/results/scaling_experiment --regplot
```

## Custom Output Directory

```bash
python visualize_latest_batch.py --regplot --output-dir my_figures
```

## Direct Script Usage

You can also run visualization scripts directly:

```bash
# Scaling regression plots
python visualization/create_scaling_regplot.py \
    --results-dir experiments/results/scaling_experiment \
    --output-dir visualization/figures

# Other scripts may need path modifications
python visualization/create_mmlu_ordered_heatmaps.py
python visualization/analyze_convergence_rounds.py
python visualization/analyze_token_usage.py
```

## Most Recent Batch

The script automatically detects the most recently modified batch directory in `experiments/results/`.

Currently detected: **scaling_experiment**

## Troubleshooting

### Scripts with Hardcoded Paths
Some visualization scripts (like `create_mmlu_ordered_heatmaps.py`) may have hardcoded paths. You can:
1. Modify the script to accept command-line arguments
2. Create a symlink from the expected path to your results directory
3. Use the Streamlit UI instead, which dynamically finds results

### Missing Dependencies
Install required packages:
```bash
pip install matplotlib seaborn pandas scipy plotly streamlit
```

### No Results Found
Check that your batch directory contains `*_summary.json` files:
```bash
find experiments/results/scaling_experiment -name "*_summary.json" | head -5
```

## Streamlit UI

The Streamlit UI provides the most flexible way to explore results:

```bash
python visualize_latest_batch.py --ui
```

Or directly:
```bash
streamlit run ui/negotiation_viewer.py
```

The UI will:
- Automatically detect all experiment batches
- Allow you to browse and compare experiments
- Show detailed conversation logs
- Generate interactive plots

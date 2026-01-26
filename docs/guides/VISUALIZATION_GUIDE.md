# Visualization Guide for Experiment Results

This guide covers the visualization workflows for analyzing negotiation experiment results. There are two main visualization scripts for different types of experiments:

1. **GPT-5-nano Scaling Analysis** (`visualization/gpt5_nano_analysis.py`) - Analyzes GPT-5-nano paired against various adversary models
2. **TTC Scaling Visualization** (`visualization/visualize_ttc_scaling.py`) - Analyzes test-time compute (reasoning tokens) scaling experiments

---

## Quick Start

### GPT-5-nano Scaling Analysis

To analyze GPT-5-nano negotiation experiments:

```bash
cd /scratch/gpfs/DANQIC/jz4391/bargain
python visualization/gpt5_nano_analysis.py
```

This script automatically discovers experiments from the configured scaling experiment directories and generates comprehensive figures analyzing payoffs, competition levels, Elo ratings, rounds to consensus, and Nash welfare.

**Output:** All figures are saved to `visualization/figures/`

### TTC Scaling Visualization

To visualize test-time compute scaling experiments:

```bash
cd /scratch/gpfs/DANQIC/jz4391/bargain
python visualization/visualize_ttc_scaling.py
```

By default, this aggregates results from **all** `ttc_scaling_*` directories in `experiments/results/`.

**Output:** Figures are saved to `visualization/figures/`

To analyze the most recent TTC scaling run with specific model pairs:

```bash
python visualization/visualize_ttc_scaling.py \
    --single experiments/results/ttc_scaling_20260125_050451 \
    --model-pairs o3-mini-high_vs_gpt-5-nano gpt-5.2-high_vs_gpt-5-nano
```

---

## GPT-5-nano Scaling Analysis

### Overview

The `gpt5_nano_analysis.py` script analyzes negotiation experiments where GPT-5-nano (baseline model, Elo 1338) is paired against various adversary models across different tiers and competition levels.

### Model Tiers

Experiments are organized by adversary model tiers:

- **STRONG TIER** (Elo >= 1415): gemini-3-pro, gemini-3-flash, claude-opus-4-5-thinking-32k, claude-opus-4-5, claude-sonnet-4-5, glm-4.7, gpt-5.2-high, qwen3-max, deepseek-r1-0528, grok-4
- **MEDIUM TIER** (1290 <= Elo < 1415): claude-haiku-4-5, deepseek-r1, claude-sonnet-4, claude-3.5-sonnet, gemma-3-27b-it, o3-mini-high, deepseek-v3, gpt-4o, QwQ-32B, llama-3.3-70b-instruct, Qwen2.5-72B-Instruct, gemma-2-27b-it, Meta-Llama-3-70B-Instruct, claude-3-haiku, phi-4
- **WEAK TIER** (Elo < 1290): amazon-nova-micro, mixtral-8x22b-instruct-v0.1, gpt-3.5-turbo-0125, llama-3.1-8b-instruct, mixtral-8x7b-instruct-v0.1, Llama-3.2-3B-Instruct, Mistral-7B-Instruct-v0.2, Phi-3-mini-128k-instruct, Llama-3.2-1B-Instruct

### Competition Levels

Experiments use competition levels: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0

### Configuration

The script searches for experiments in directories specified in the `SCALING_EXPERIMENTS` list (currently: `scaling_experiment_20260121_070359`). To analyze different experiments, modify this list in the script:

```python
SCALING_EXPERIMENTS = [
    "scaling_experiment_20260121_070359",
    # Add more experiment directories here
]
```

### Generated Figures

The script generates a comprehensive set of figures organized into several categories:

#### Series A: Baseline Model Payoff Analysis
- **A1**: Baseline Model (GPT-5-nano) Payoff vs Adversary Model Elo
  - Overall view and subplots by competition level
- **A2**: Baseline Model Payoff vs Competition Level
  - Overall view and subplots by adversary tier

#### Series B: Adversary Model Payoff Analysis
- **B1**: Adversary Model Payoff vs Adversary Model Elo
  - Overall view and subplots by competition level
- **B2**: Adversary Model Payoff vs Competition Level
  - Overall view and subplots by adversary tier
- **B3**: Payoff Difference (Adversary - Baseline) vs Elo (colored by competition)
- **B4**: Payoff Difference Heatmap (Elo bins × Competition)

#### Series C: Elo Difference Analysis
- **C1**: Baseline Model Payoff vs Elo Difference
- **C2**: Adversary Model Payoff vs Elo Difference

#### Series D: Payoff vs Elo Difference
- **D1**: Payoff Difference vs Elo Difference
- **D2**: Payoff Difference vs Elo Difference (colored by competition)

#### Series E: Rounds to Consensus
- **E1**: Rounds to Consensus vs Competition Level
- **E2**: Rounds to Consensus by Adversary Tier
- **E3**: Rounds to Consensus by Reasoning vs Non-Reasoning Models
- **E4**: Rounds to Consensus Heatmap (Tier × Competition)

#### Series F: Nash Welfare Analysis
- **F1**: Nash Welfare vs Competition Level
- **F2**: Nash Welfare by Adversary Tier
- **F3**: Nash Welfare Heatmap (Tier × Competition)
- **F4**: Efficiency vs Fairness Trade-off
- **F5**: Nash Welfare vs Elo Difference
- **F6**: Nash Welfare vs Competition by Tier
- **F7**: Nash Welfare by Reasoning vs Non-Reasoning
- **F8**: Comprehensive Nash Welfare Summary (4-panel)

All figures are saved as PNG files in `visualization/figures/` with descriptive filenames (e.g., `fig_a1_weak_payoff_vs_elo_overall.png`).

### Usage

```bash
# Run the analysis (no command-line arguments needed)
python visualization/gpt5_nano_analysis.py
```

The script will:
1. Discover all GPT-5-nano experiments from configured scaling experiment directories
2. Load and validate experiment results
3. Extract features and create a DataFrame
4. Generate all figures automatically
5. Print a summary of generated figures at the end

---

## TTC Scaling Visualization

### Overview

The `visualize_ttc_scaling.py` script analyzes test-time compute (TTC) scaling experiments, focusing on the relationship between reasoning tokens and negotiation payoff. It processes experiments where models are given different reasoning token budgets.

### Available Scripts

| Script | Purpose |
|--------|---------|
| `visualization/visualize_ttc_scaling.py` | **Current** - Full TTC analysis with multi-model comparison |
| `visualization/visualize_ttc_scaling_derisk.py` | **Archive** - Original derisk version (simpler, single model pair) |

### Recent TTC Experiment Runs

| Directory | Date | Models | Description |
|-----------|------|--------|-------------|
| `ttc_scaling_20260125_050451` | Jan 25, 2026 | o3-mini-high, gpt-5.2-high, claude-opus-4-5-thinking-32k, deepseek-r1, grok-4, QwQ-32B vs gpt-5-nano | Full-scale TTC experiment with multiple reasoning models |
| `ttc_scaling_20260124_224426` | Jan 24, 2026 | claude-opus-4-5-thinking-32k vs gpt-5-nano | Derisk run (archived) |

### What It Analyzes

- **Reasoning tokens vs payoff**: How reasoning token usage correlates with negotiation outcomes
- **Per-phase analysis**: Reasoning tokens by phase (thinking, reflection, discussion, proposal, voting)
- **Multi-model comparison**: Compare adversary vs baseline model performance (Plot 6)
- **Compliance analysis**: Comparison of instructed token budgets vs actual token usage
- **Budget vs payoff**: How prompted token budgets affect negotiation outcomes
- **Competition level breakdown**: Separate analysis for each competition level (0.0, 0.25, 0.5, 0.75, 1.0)

### Command-Line Options

```bash
# Aggregate all ttc_scaling_* directories (default)
python visualization/visualize_ttc_scaling.py

# Analyze a single directory
python visualization/visualize_ttc_scaling.py --single experiments/results/ttc_scaling_20260125_050451

# Filter for specific model pairs (recommended for focused analysis)
python visualization/visualize_ttc_scaling.py --single experiments/results/ttc_scaling_20260125_050451 \
    --model-pairs o3-mini-high_vs_gpt-5-nano gpt-5.2-high_vs_gpt-5-nano

# Specify custom results base directory
python visualization/visualize_ttc_scaling.py --results-base /path/to/results

# Specify custom output directory
python visualization/visualize_ttc_scaling.py --output-dir /path/to/output
```

### Generated Figures

The script generates 6 main plot types plus a data summary CSV:

1. **plot1_avg_reasoning_vs_payoff.png**
   - Scatter plot: Average reasoning tokens per stage vs payoff
   - Points labeled with prompted token budget

2. **plot2_per_round_reasoning.png**
   - 5 subplots (one per phase: Thinking, Discussion, Proposal, Voting, Reflection)
   - Reasoning tokens vs payoff, color-coded by token budget

3. **plot3_phase_breakdown.png**
   - Stacked bar chart: Reasoning tokens by phase
   - Scatter plot: Payoff vs total reasoning tokens (colored by budget)

4. **plot4_instructed_vs_actual.png**
   - Scatter plot: Instructed token budget vs actual reasoning tokens used
   - Shows compliance ratio annotations

5. **plot5_instructed_vs_payoff.png**
   - Scatter plot: Instructed token budget vs payoff
   - Includes fair split reference line at y=50

6. **plot6_payoff_vs_reasoning_by_phase_comp_{level}.png** (NEW)
   - **5 plots**, one for each competition level (0.0, 0.25, 0.5, 0.75, 1.0)
   - Each plot is a 5×2 grid:
     - 5 rows: Private Thinking, Discussion, Proposal, Voting, Reflection
     - 2 columns: o3-mini-high vs gpt-5-nano (left), gpt-5.2-high vs gpt-5-nano (right)
   - Each subplot shows:
     - Scatter points for adversary model (blue circles) and baseline model (coral squares)
     - Linear trendlines with equation, R², and p-values
     - Normalized axes across all subplots for easy comparison
     - Fair split reference line at y=50

7. **data_summary.csv**
   - Extracted data for further analysis
   - Includes all experiment parameters, token usage metrics, and outcomes

### Output Location

All figures are saved to `visualization/figures/` by default.

To specify a custom output directory, use `--output-dir`.

### Data Collection

The script automatically:
- Finds all `ttc_scaling_*` directories in the results base directory
- Discovers all experiment directories (those containing `experiment_results.json`)
- Loads experiment results and agent interactions
- Extracts reasoning token usage from `run_*_all_interactions.json` files
- Identifies which agent received the prompted reasoning budget vs baseline
- Aggregates data across all experiments into a single DataFrame

### Key Metrics Extracted

For each experiment and agent:
- Total reasoning tokens
- Average reasoning tokens per interaction
- Reasoning tokens by phase (thinking, reflection, discussion, proposal, voting)
- Reasoning tokens by round (rounds 1-10)
- Final utility (payoff)
- Token budget (prompted/instructed)
- Competition level
- Model order (weak_first vs strong_first)
- **is_prompted_reasoning**: Whether this agent received the reasoning budget prompt (True = adversary, False = baseline)
- Consensus reached status
- Final round number

### Example: Analyzing Jan 25 TTC Run

```bash
# Full analysis of the main TTC experiment with o3-mini-high and gpt-5.2-high
python visualization/visualize_ttc_scaling.py \
    --single experiments/results/ttc_scaling_20260125_050451 \
    --model-pairs o3-mini-high_vs_gpt-5-nano gpt-5.2-high_vs_gpt-5-nano
```

This generates:
- `visualization/figures/plot1_avg_reasoning_vs_payoff.png`
- `visualization/figures/plot2_per_round_reasoning.png`
- `visualization/figures/plot3_phase_breakdown.png`
- `visualization/figures/plot4_instructed_vs_actual.png`
- `visualization/figures/plot5_instructed_vs_payoff.png`
- `visualization/figures/plot6_payoff_vs_reasoning_by_phase_comp_0_0.png`
- `visualization/figures/plot6_payoff_vs_reasoning_by_phase_comp_0_25.png`
- `visualization/figures/plot6_payoff_vs_reasoning_by_phase_comp_0_5.png`
- `visualization/figures/plot6_payoff_vs_reasoning_by_phase_comp_0_75.png`
- `visualization/figures/plot6_payoff_vs_reasoning_by_phase_comp_1_0.png`
- `visualization/figures/data_summary.csv`

---

## Troubleshooting

### GPT-5-nano Analysis

**No experiments found:**
- Check that the experiment directories specified in `SCALING_EXPERIMENTS` exist
- Verify that experiments contain `experiment_results.json` files
- Ensure experiments follow the expected directory structure: `{scaling_experiment}/gpt-5-nano_vs_{adversary}/{model_order}/comp_{level}/run_{n}/experiment_results.json`

**Missing figures:**
- Check that `visualization/figures/` directory exists and is writable
- Verify that experiments contain valid data (consensus reached, valid utilities)

**Script errors:**
- Ensure all dependencies are installed: `pandas`, `numpy`, `matplotlib`, `seaborn`
- Check that experiment JSON files are valid and not corrupted

### TTC Scaling Visualization

**No ttc_scaling directories found:**
- Verify that `experiments/results/` contains directories starting with `ttc_scaling_`
- Check the `--results-base` path if using a custom location

**No data in plots:**
- Ensure experiment directories contain `experiment_results.json` files
- Verify that `run_*_all_interactions.json` files exist and contain `reasoning_tokens` in `token_usage`
- Check that experiments completed successfully (consensus reached)

**Missing reasoning tokens:**
- The script extracts `reasoning_tokens` from `token_usage.reasoning_tokens` in interaction files
- Ensure your experiment logging includes reasoning token tracking

### General Issues

**Missing dependencies:**
```bash
pip install matplotlib seaborn pandas numpy scipy
```

**Permission errors:**
- Ensure output directories are writable
- Check file permissions on experiment result files

**Memory issues:**
- For very large datasets, consider analyzing subsets using `--single` mode
- The aggregate mode loads all experiments into memory

---

## Dependencies

Both visualization scripts require:

```bash
pip install matplotlib seaborn pandas numpy scipy
```

Optional (for advanced analysis):
```bash
pip install plotly  # For interactive plots (if used)
```

---

## File Locations

### Scripts
- `visualization/gpt5_nano_analysis.py` - GPT-5-nano scaling analysis
- `visualization/visualize_ttc_scaling.py` - TTC scaling visualization (current)
- `visualization/visualize_ttc_scaling_derisk.py` - TTC scaling visualization (archived original)
- `visualization/visualize_nonreasoning_tokens.py` - Non-reasoning model token analysis

### Output Directories
- `visualization/figures/` - All visualization figures (GPT-5-nano and TTC scaling)

### Experiment Data
- `experiments/results/scaling_experiment_*/` - GPT-5-nano scaling experiments
- `experiments/results/ttc_scaling_*/` - TTC scaling experiments
  - `ttc_scaling_20260125_050451/` - Latest full TTC run (Jan 25, 2026)
  - `ttc_scaling_20260124_224426/` - Derisk run (Jan 24, 2026)

---

## Additional Resources

- Experiment results are stored in JSON format with `experiment_results.json` files
- Agent interactions are stored in `run_*_all_interactions.json` files
- The Streamlit UI (`ui/negotiation_viewer.py`) provides an interactive way to explore results
- See `scripts/README.md` for information on running experiments

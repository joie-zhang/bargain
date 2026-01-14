# Implementation Plan: Configurable Discussion Rounds and Bidirectional Ordering

## Summary of Requested Changes

### Action Item 1: Configurable Number of Discussion Exchanges
**Parameter Name**: `discussion_turns` (default: 3)

- **Current State**: Discussion phase runs **once per round** - each agent speaks once (in `phase_handlers.py:200-286`), then moves to proposal phase
- **Desired State**: Agents should be able to go around the circle N times (default: 3) during each negotiation round before moving to proposals
- **Scope**: Per-round - 3 discussion exchanges happen in each negotiation round

### Action Item 2: Configurable Starting Order with Balanced Runs
**Goal**: Control which agent speaks first to mitigate first-mover advantage.

- **Current State**: Config has `model_order` field (`weak_first`/`strong_first`), but `generate_configs_both_orders.sh` only generates `weak_first` configs
- **Desired State**:
  - When runs=5, split: 2 or 3 runs as `weak_first`, remainder as `strong_first`
  - Support `"random"` option that flips a coin at experiment start
  - Generate configs for both order variants

### Action Item 3: Run Experiments with Both Orders + SLURM Script Generation
**Goal**: For each model pair, run experiments in both orders, with appropriate SLURM scripts.

- **API Models (CPU)**: Reduce memory from 16G to 4G (since utilization is ~5%)
- **Local Princeton Models (GPU on PLI H100 cluster)**:
  - Llama 3.1 8B: 1 GPU (80GB)
  - Llama 3.3 70B: 4 GPUs (320GB) - similar to Qwen 72B
  - Other models: Calculate based on GPU_REQUIREMENTS.md

### Action Item 4: Average Results Across Both Orders in Figures
- **Main figures**: Simple mean with confidence intervals (for paper body)
- **Appendix figures**: Order-specific figures as distinct outputs
- Include confidence intervals wherever aggregating

---

## Files to Modify

### 1. Configuration & SLURM Generation
| File | Change |
|------|--------|
| `scripts/generate_configs_both_orders.sh` | Split runs between orders, generate SLURM scripts |
| `run_strong_models_experiment.py` | Add `--discussion-turns` CLI argument |
| `strong_models_experiment/experiment.py` | Pass `discussion_terms` to phase handler, handle random order |

### 2. Core Experiment Logic
| File | Change |
|------|--------|
| `strong_models_experiment/phases/phase_handlers.py` | Modify `run_discussion_phase` to loop N times |

### 3. Analysis & Visualization
| File | Change |
|------|--------|
| `visualization/create_mmlu_ordered_heatmaps.py` | Aggregate results across both orders with CIs |
| `scripts/analyze_order_effects.py` | Add averaging mode for figure generation |
| `visualization/create_scaling_regplot.py` | **NEW** - Scatter plots with regression lines (seaborn regplot/statsmodels) |

---

## Implementation Details

### Phase 1: Add `discussion_turns` Parameter

**1.1. Update `phase_handlers.py:run_discussion_phase()`**

```python
async def run_discussion_phase(self, agents, items, preferences,
                                round_num, max_rounds,
                                discussion_turns=3):  # NEW PARAM
    messages = []

    for turn in range(discussion_turns):  # NEW: Outer loop
        self.logger.info(f"=== DISCUSSION TURN {turn+1}/{discussion_turns} - Round {round_num} ===")

        for i, agent in enumerate(agents):
            # existing agent speaking logic
            ...
            message["discussion_turn"] = turn + 1  # Track which turn
            messages.append(message)

    return {"messages": messages}
```

**1.2. Update `experiment.py`**

```python
discussion_result = await self.phase_handler.run_discussion_phase(
    agents, items, preferences, round_num, config["t_rounds"],
    discussion_turns=config.get("discussion_turns", 3)
)
```

**1.3. Update CLI (`run_strong_models_experiment.py`)**

```python
parser.add_argument(
    "--discussion-turns",
    type=int,
    default=3,
    help="Number of times agents go around discussing per round (default: 3)"
)
```

### Phase 2: Balance Runs Across Orders

**2.1. Update `generate_configs_both_orders.sh`**

```bash
# Configuration
NUM_RUNS=5  # Total runs per model pair

# Calculate split (e.g., 5 runs → 3 weak_first, 2 strong_first)
WEAK_FIRST_RUNS=$(( (NUM_RUNS + 1) / 2 ))  # Ceiling division
STRONG_FIRST_RUNS=$(( NUM_RUNS - WEAK_FIRST_RUNS ))

# Generate weak_first configs (runs 1 to WEAK_FIRST_RUNS)
for run_idx in $(seq 0 $((WEAK_FIRST_RUNS - 1))); do
    SEED=${RUN_SEEDS[$run_idx]}
    RUN_NUM=$((run_idx + 1))
    # Generate weak_first config with seed and run number
    ...
done

# Generate strong_first configs (remaining runs)
for run_idx in $(seq $WEAK_FIRST_RUNS $((NUM_RUNS - 1))); do
    SEED=${RUN_SEEDS[$run_idx]}
    RUN_NUM=$((run_idx + 1))
    # Generate strong_first config with seed and run number
    ...
done
```

### Phase 3: SLURM Script Generation

**3.1. Detect Model Type and Generate Appropriate SLURM Script**

```bash
# In generate_configs_both_orders.sh

# Model type detection function
get_api_type() {
    local model=$1
    case $model in
        Qwen2.5-*|llama-3.*-*-instruct)
            echo "princeton_cluster"  # Needs GPU
            ;;
        *)
            echo "api"  # CPU only
            ;;
    esac
}

# GPU requirements lookup
get_gpu_requirements() {
    local model=$1
    case $model in
        *-0.5B-*|*-1.5B-*|*-3B-*)
            echo "1:40G"  # 1 GPU, 40GB
            ;;
        *-7B-*|*-8b-*|*-14B-*)
            echo "1:80G"  # 1 GPU, 80GB
            ;;
        *-32B-*|*-70b-*)
            echo "4:320G"  # 4 GPUs, 320GB (similar to 72B)
            ;;
        *-72B-*)
            echo "4:320G"  # 4 GPUs, 320GB
            ;;
        *)
            echo "1:80G"  # Default
            ;;
    esac
}

generate_slurm_script() {
    local config_file=$1
    local exp_id=$2
    local model1=$(jq -r '.models[0]' "$config_file")
    local model2=$(jq -r '.models[1]' "$config_file")

    local api_type1=$(get_api_type "$model1")
    local api_type2=$(get_api_type "$model2")

    if [ "$api_type1" = "princeton_cluster" ] || [ "$api_type2" = "princeton_cluster" ]; then
        # GPU job on PLI cluster
        generate_gpu_slurm "$config_file" "$exp_id" "$model1" "$model2"
    else
        # CPU-only job (API models)
        generate_cpu_slurm "$config_file" "$exp_id" "$model1" "$model2"
    fi
}
```

**3.2. CPU SLURM Template (for API models)**

```bash
generate_cpu_slurm() {
    local config_file=$1
    local exp_id=$2
    local model1=$3
    local model2=$4

    local slurm_file="${SLURM_DIR}/submit_${exp_id}.sbatch"

    cat > "${slurm_file}" << 'SLURM_EOF'
#!/bin/bash
#SBATCH --job-name=bargain_EXP_ID
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/cluster/exp_EXP_ID_%j.out
#SBATCH --error=logs/cluster/exp_EXP_ID_%j.err
#SBATCH --mail-type=FAIL

set -e
cd /scratch/gpfs/DANQIC/jz4391/bargain
module load proxy/default
source .venv/bin/activate

python3 run_strong_models_experiment.py --config CONFIG_FILE

SLURM_EOF

    # Replace placeholders
    sed -i "s|EXP_ID|${exp_id}|g" "${slurm_file}"
    sed -i "s|CONFIG_FILE|${config_file}|g" "${slurm_file}"
}
```

**3.3. GPU SLURM Template (for Princeton cluster models)**

```bash
generate_gpu_slurm() {
    local config_file=$1
    local exp_id=$2
    local model1=$3
    local model2=$4

    # Get GPU requirements for the larger model
    local gpu_req1=$(get_gpu_requirements "$model1")
    local gpu_req2=$(get_gpu_requirements "$model2")

    # Use the larger requirement
    local num_gpus=$(echo "$gpu_req1 $gpu_req2" | tr ' ' '\n' | cut -d: -f1 | sort -rn | head -1)
    local mem=$(echo "$gpu_req1 $gpu_req2" | tr ' ' '\n' | cut -d: -f2 | sort -t G -k1 -rn | head -1)

    local slurm_file="${SLURM_DIR}/submit_${exp_id}.sbatch"

    cat > "${slurm_file}" << 'SLURM_EOF'
#!/bin/bash
#SBATCH --job-name=bargain_EXP_ID
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:NUM_GPUS
#SBATCH --constraint=gpu80
#SBATCH --mem=MEM
#SBATCH --time=04:00:00
#SBATCH --partition=pli-c
#SBATCH --output=logs/cluster/exp_EXP_ID_%j.out
#SBATCH --error=logs/cluster/exp_EXP_ID_%j.err
#SBATCH --mail-type=FAIL

set -e
cd /scratch/gpfs/DANQIC/jz4391/bargain
module load proxy/default
source .venv/bin/activate

python3 run_strong_models_experiment.py --config CONFIG_FILE

SLURM_EOF

    # Replace placeholders
    sed -i "s|EXP_ID|${exp_id}|g" "${slurm_file}"
    sed -i "s|CONFIG_FILE|${config_file}|g" "${slurm_file}"
    sed -i "s|NUM_GPUS|${num_gpus}|g" "${slurm_file}"
    sed -i "s|MEM|${mem}|g" "${slurm_file}"
}
```

### Phase 4: Visualization with Aggregation

**4.1. Aggregation Function with Confidence Intervals**

```python
from scipy import stats
from collections import defaultdict
import numpy as np

def compute_mean_with_ci(values, confidence=0.95):
    """Compute mean and confidence interval."""
    n = len(values)
    mean = np.mean(values)
    if n > 1:
        se = stats.sem(values)
        ci = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    else:
        ci = 0
    return mean, ci

def aggregate_across_orders(results_by_competition):
    """Average results across weak_first and strong_first for each model pair."""
    aggregated = {}

    for comp_level, results in results_by_competition.items():
        aggregated[comp_level] = {}

        # Group by (weak_model, strong_model) pair
        pair_results = defaultdict(list)
        for result in results:
            key = (result['weak_model'], result['strong_model'])
            pair_results[key].append(result)

        # Average across both orders with CI
        for pair, pair_data in pair_results.items():
            utility_diffs = [r['utility_difference'] for r in pair_data]
            weak_utilities = [r['weak_utility'] for r in pair_data]
            strong_utilities = [r['strong_utility'] for r in pair_data]

            aggregated[comp_level][pair] = {
                'utility_difference': compute_mean_with_ci(utility_diffs),
                'weak_utility': compute_mean_with_ci(weak_utilities),
                'strong_utility': compute_mean_with_ci(strong_utilities),
                'n_experiments': len(pair_data),
                'orders_included': list(set(r.get('model_order') for r in pair_data))
            }

    return aggregated
```

**4.2. Separate Figure Generation**

```python
# Main figures (for paper body)
def generate_aggregated_figures(results):
    """Generate figures with results averaged across both orders."""
    aggregated = aggregate_across_orders(results)
    # Plot with error bars from CI
    save_figure("figures/main_heatmap_aggregated.pdf")

# Appendix figures (order-specific)
def generate_order_specific_figures(results):
    """Generate separate figures for each ordering."""
    for order in ['weak_first', 'strong_first']:
        filtered = filter_by_order(results, order)
        save_figure(f"figures/appendix_heatmap_{order}.pdf")
```

---

## GPU Requirements Reference

Based on `scripts/GPU_REQUIREMENTS.md`:

| Model Size | GPUs Required | Memory |
|------------|---------------|--------|
| 0.5B-3B | 1 GPU | 40-80G |
| 7B-14B | 1 GPU | 80G |
| 32B, Llama 3.3 70B | 4 GPUs | 320G |
| 72B | 4 GPUs | 320G |

---

## Testing Plan

1. **Unit Test**: Verify `discussion_turns` parameter works
   - Run with `--discussion-turns 1` (baseline)
   - Run with `--discussion-turns 3` (default)
   - Check conversation logs show correct number of exchanges per round

2. **Config Generation Test**:
   - Run updated `generate_configs_both_orders.sh`
   - Verify runs are split correctly between orders
   - Check SLURM scripts are generated with correct resources

3. **SLURM Test**:
   - Submit one CPU job and verify 4G memory is sufficient
   - Submit one GPU job and verify model loads correctly

4. **Visualization Test**:
   - Generate figures with aggregated data
   - Verify confidence intervals appear correctly
   - Confirm appendix figures are distinct from main figures

---

## Random Order Support

In `experiment.py`, before agent creation:

```python
# Handle random model ordering
if config.get("model_order") == "random":
    import random
    if random.random() < 0.5:
        actual_order = "weak_first"
    else:
        actual_order = "strong_first"
        models = models[::-1]  # Reverse the models list
    config["actual_order"] = actual_order  # Track what was chosen
else:
    config["actual_order"] = config.get("model_order", "weak_first")
```

---

## Phase 5: Scaling Law Regression Plots

**NEW FILE: `visualization/create_scaling_regplot.py`**

Create scatter plots with regression lines showing utility vs model capability (MMLU-Pro score).

```python
#!/usr/bin/env python3
"""
Create scaling law regression plots using seaborn regplot.
Shows utility difference vs model capability with fitted regression lines.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Try seaborn regplot first, fall back to statsmodels if needed
try:
    from seaborn import regplot
    USE_SEABORN = True
except ImportError:
    import statsmodels.api as sm
    USE_SEABORN = False


def create_scaling_regplot(results_df, output_path, x_col='mmlu_pro_score',
                           y_col='utility_difference', hue_col='competition_level'):
    """
    Create scatter plot with regression line showing scaling relationship.

    Args:
        results_df: DataFrame with columns for x, y, and optionally hue
        output_path: Path to save the figure
        x_col: Column name for x-axis (model capability metric)
        y_col: Column name for y-axis (utility metric)
        hue_col: Column name for color grouping (e.g., competition level)
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    if USE_SEABORN:
        # Seaborn regplot with confidence intervals
        if hue_col and hue_col in results_df.columns:
            # Plot separate regression lines for each competition level
            for comp_level in sorted(results_df[hue_col].unique()):
                subset = results_df[results_df[hue_col] == comp_level]
                sns.regplot(
                    data=subset,
                    x=x_col,
                    y=y_col,
                    scatter=True,
                    ci=95,  # 95% confidence interval
                    label=f'Competition {comp_level}',
                    scatter_kws={'alpha': 0.6, 's': 80},
                    line_kws={'linewidth': 2},
                    ax=ax
                )
        else:
            sns.regplot(
                data=results_df,
                x=x_col,
                y=y_col,
                scatter=True,
                ci=95,
                scatter_kws={'alpha': 0.6, 's': 80},
                line_kws={'linewidth': 2},
                ax=ax
            )
    else:
        # Statsmodels fallback
        X = sm.add_constant(results_df[x_col])
        y = results_df[y_col]
        model = sm.OLS(y, X).fit()

        # Scatter plot
        ax.scatter(results_df[x_col], results_df[y_col], alpha=0.6, s=80)

        # Regression line
        x_pred = np.linspace(results_df[x_col].min(), results_df[x_col].max(), 100)
        X_pred = sm.add_constant(x_pred)
        y_pred = model.predict(X_pred)
        ax.plot(x_pred, y_pred, 'r-', linewidth=2, label=f'R² = {model.rsquared:.3f}')

        # Confidence interval
        pred = model.get_prediction(X_pred)
        ci = pred.conf_int(alpha=0.05)
        ax.fill_between(x_pred, ci[:, 0], ci[:, 1], alpha=0.2)

    ax.set_xlabel('Model Capability (MMLU-Pro Score)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Utility Difference (Strong - Weak)', fontsize=12, fontweight='bold')
    ax.set_title('Scaling Law: Model Capability vs Negotiation Advantage',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Scaling regplot saved to {output_path}")
    plt.close()


def create_order_comparison_regplot(results_df, output_path):
    """
    Create side-by-side regplots comparing weak_first vs strong_first.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for idx, order in enumerate(['weak_first', 'strong_first']):
        ax = axes[idx]
        subset = results_df[results_df['model_order'] == order]

        if len(subset) > 0:
            sns.regplot(
                data=subset,
                x='mmlu_pro_score',
                y='utility_difference',
                scatter=True,
                ci=95,
                scatter_kws={'alpha': 0.6, 's': 80},
                line_kws={'linewidth': 2},
                ax=ax
            )

        ax.set_title(f'Order: {order.replace("_", " ").title()}', fontsize=12)
        ax.set_xlabel('Model Capability (MMLU-Pro)', fontsize=11)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Utility Difference', fontsize=11)

    plt.suptitle('Scaling Laws by Model Ordering (Appendix)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Order comparison regplot saved to {output_path}")
    plt.close()
```

### Usage

```bash
# Generate main scaling law figure (aggregated across orders)
python visualization/create_scaling_regplot.py --input results.csv --output figures/scaling_law_regplot.pdf

# Generate appendix figures (order-specific)
python visualization/create_scaling_regplot.py --input results.csv --output figures/appendix_scaling_by_order.pdf --by-order
```

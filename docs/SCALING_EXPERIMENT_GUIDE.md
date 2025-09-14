# Scaling Experiment Guide: Current Workflow

## Overview
This guide documents the current bash script system for running negotiation experiments testing how stronger models exploit weaker models across different competition levels.

### Current Experiment Configuration
- **Weak Models**: `gemini-1-5-pro` (others commented out: `claude-3-opus`, `gpt-4o`)
- **Strong Models**: `grok-4-0709` (others available but commented out)
- **Competition Levels (5)**: 0.0, 0.25, 0.5, 0.75, 1.0
- **Runs per configuration**: 5 (different seeds: 42, 123, 456, 789, 101112)
- **Items (m)**: 5
- **Agents (n)**: 2
- **Max rounds (t)**: 10

## Quick Start

```bash
# 1. Set your API keys
export OPENROUTER_API_KEY='your-key'
export ANTHROPIC_API_KEY='your-key'
export OPENAI_API_KEY='your-key'
export GOOGLE_API_KEY='your-key'  # For Gemini models
export XAI_API_KEY='your-key'      # For Grok models

# 2. Generate configurations
./scripts/generate_configs_both_orders.sh

# 3. Run all experiments (default 4 parallel)
./scripts/run_all_simple.sh

# Or with custom parallelism (e.g., 8 jobs)
./scripts/run_all_simple.sh 8

# 4. Visualize results
python visualization/create_mmlu_ordered_heatmaps.py
```

## Current Script Files

All scripts are located in `scripts/`

### 1. Main Orchestrator: `run_all_simple.sh`
**Purpose**: Simple runner that just executes all experiments without complex timeout handling

**Usage**:
```bash
./scripts/run_all_simple.sh [MAX_PARALLEL]
# Example: ./scripts/run_all_simple.sh 4
```

**What it does**:
1. Generates configs if needed
2. Runs experiments in parallel using xargs
3. Skips already completed experiments
4. Collects results at the end
5. Shows summary statistics

### 2. Configuration Generator: `generate_configs_both_orders.sh`
**Purpose**: Creates JSON configuration files with weak model always first (no order variation)

**Output**: Creates config files in `experiments/results/scaling_experiment/configs/`

**Config format**:
```json
{
    "experiment_id": 0,
    "weak_model": "gemini-1-5-pro",
    "strong_model": "grok-4-0709",
    "models": ["gemini-1-5-pro", "grok-4-0709"],
    "model_order": "weak_first",
    "competition_level": 0.5,
    "run_number": 1,
    "num_items": 5,
    "max_rounds": 10,
    "random_seed": 42,
    "output_dir": "experiments/results/scaling_experiment/gemini-1-5-pro_vs_grok-4-0709/weak_first/comp_0.5/run_1"
}
```

### 3. Single Experiment Runner: `run_single_experiment_simple.sh`
**Purpose**: Runs a single experiment by job ID (no timeout, simple execution)

**Usage**:
```bash
./scripts/run_single_experiment_simple.sh JOB_ID
# Example: ./scripts/run_single_experiment_simple.sh 42
```

**Features**:
- No timeout (lets Python handle retries internally)
- Detailed logging to `logs/experiment_JOB_ID.log`
- Saves results to appropriate output directory
- Creates completion flag for resume capability
- Simpler error handling

### 4. Results Collector: `collect_results.sh`
**Purpose**: Aggregates all experiment results and creates summaries

**Output files**:
- `all_results.json` - Complete results data
- `summary.json` - Statistical summary
- `results.csv` - CSV for analysis

**Summary includes**:
- Total experiments run
- Success/failure rates
- Breakdown by model pair
- Breakdown by competition level

## Directory Structure

```
experiments/results/scaling_experiment/
├── configs/                          # Configuration files
│   ├── config_0.json
│   ├── config_1.json
│   └── ...
├── logs/                             # Execution logs
│   ├── experiment_0.log
│   ├── completed_0.flag             # Completion markers
│   └── ...
├── gemini-1-5-pro_vs_grok-4-0709/  # Results by model pair
│   └── weak_first/                  # Model order
│       ├── comp_0.0/                # By competition level
│       │   └── run_1/              # By run number
│       │       └── result.json
│       └── ...
├── all_results.json                 # Aggregated results
├── summary.json                     # Statistical summary
└── results.csv                      # CSV for analysis
```

## Visualization

After experiments complete, use the visualization scripts:

```bash
# Create MMLU-ordered heatmaps
python visualization/create_mmlu_ordered_heatmaps.py

# Analyze convergence rates
python visualization/analyze_convergence_rounds.py

# Check token usage
python visualization/analyze_token_usage.py

# Verify run counts per cell
python visualization/analyze_number_of_runs_in_heatmap.py
```

## Monitoring Progress

### During execution:
- Check individual logs: `tail -f experiments/results/scaling_experiment/logs/experiment_*.log`
- Count completed: `ls experiments/results/scaling_experiment/logs/completed_*.flag | wc -l`

### After execution:
- View summary: `cat experiments/results/scaling_experiment/summary.json | python3 -m json.tool`
- Check failures: `grep -l "FAILED" experiments/results/scaling_experiment/logs/*.log`

## Resuming Interrupted Runs

The system automatically resumes from where it left off:

```bash
# If interrupted, just run again - completed experiments will be skipped
./scripts/run_all_simple.sh 8
```

## Troubleshooting

### Common Issues:

1. **API Key Errors**:
   ```bash
   # Check keys are set
   echo $OPENROUTER_API_KEY
   echo $ANTHROPIC_API_KEY
   echo $GOOGLE_API_KEY
   echo $XAI_API_KEY
   ```

2. **Permission Denied**:
   ```bash
   # Make scripts executable
   chmod +x scripts/*.sh
   ```

3. **Python Module Not Found**:
   ```bash
   # Install required packages
   pip install numpy pandas matplotlib seaborn
   ```

## Modifying Experiments

### To test different models:
Edit `scripts/generate_configs_both_orders.sh`:
```bash
WEAK_MODELS=(
    "claude-3-opus"
    "gemini-1-5-pro"
    "gpt-4o"
)

STRONG_MODELS=(
    "claude-3-5-haiku"
    "claude-3-5-sonnet"
    "gpt-5-nano"
    # etc...
)
```

### To change competition levels:
```bash
COMPETITION_LEVELS=(0.0 0.25 0.5 0.75 1.0)
```

### To adjust number of runs:
```bash
NUM_RUNS=5  # Number of runs per configuration
RUN_SEEDS=(42 123 456 789 101112)  # Seeds for each run
```

## Notes

- Results are saved incrementally, so partial runs are not lost
- Each experiment is independent - order doesn't matter
- System is idempotent - safe to run multiple times
- The simple runner has no timeouts - relies on Python's internal retry logic
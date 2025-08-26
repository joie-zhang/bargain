# Scaling Experiment Guide: 300 Experiments Setup

## Overview
This guide documents the bash script system for running 300 negotiation experiments testing how stronger models exploit weaker models across different competition levels.

### Experiment Configuration
- **Weak Models (3)**: `gpt-4o`, `claude-3-opus`, `gemini-1-5-pro`
- **Strong Models (10)**: `gemini-2-0-flash`, `gemini-2-5-flash`, `gemini-2-5-pro`, `claude-3-5-haiku`, `claude-4-sonnet`, `claude-4-opus`, `o3-mini`, `o4-mini`, `o3`, `chatgpt-5`
- **Competition Levels (10)**: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
- **Total Experiments**: 3 × 10 × 10 = 300

## Quick Start

```bash
# 1. First time setup
cd /Users/qw281/Downloads/bargain
./scripts/setup_experiment.sh

# 2. Set your API keys
export OPENROUTER_API_KEY='your-key'
export ANTHROPIC_API_KEY='your-key'
export OPENAI_API_KEY='your-key'

# 3. Run all experiments (default 4 parallel)
./scripts/run_all_experiments.sh

# Or with custom parallelism (e.g., 8 jobs)
./scripts/run_all_experiments.sh --parallel 8
```

## Script Files

All scripts are located in `/Users/qw281/Downloads/bargain/scripts/`

### 1. Main Orchestrator: `run_all_experiments.sh`
**Purpose**: Coordinates the entire experiment pipeline

**Usage**:
```bash
./scripts/run_all_experiments.sh [OPTIONS]
  --parallel N     Number of parallel jobs (default: 4)
  --dry-run        Show what would be run without executing
  --help           Show help message
```

**What it does**:
1. Checks requirements (Python, API keys)
2. Generates experiment configurations
3. Runs experiments in parallel
4. Collects and aggregates results
5. Shows summary statistics

### 2. Configuration Generator: `generate_configs.sh`
**Purpose**: Creates JSON configuration files for all 300 experiments

**Output**: Creates 300 config files in `experiments/results/scaling_experiment/configs/`

**Config format**:
```json
{
    "experiment_id": 0,
    "weak_model": "gpt-4o",
    "strong_model": "gemini-2-0-flash",
    "competition_level": 0.1,
    "num_items": 5,
    "max_rounds": 10,
    "random_seed": 42,
    "output_dir": "experiments/results/scaling_experiment/gpt-4o_vs_gemini-2-0-flash/comp_0.1"
}
```

### 3. Single Experiment Runner: `run_single_experiment.sh`
**Purpose**: Runs a single experiment by job ID

**Usage**:
```bash
./scripts/run_single_experiment.sh JOB_ID
# Example: ./scripts/run_single_experiment.sh 42
```

**Features**:
- 10-minute timeout per experiment
- Detailed logging to `logs/experiment_JOB_ID.log`
- Saves results to appropriate output directory
- Creates completion flag for resume capability

### 4. Parallel Job Manager: `run_parallel_jobs.sh`
**Purpose**: Manages parallel execution of experiments

**Usage**:
```bash
./scripts/run_parallel_jobs.sh [MAX_PARALLEL]
# Example: ./scripts/run_parallel_jobs.sh 8
```

**Features**:
- Real-time progress display
- Job slot management
- Resume capability (skips completed jobs)
- Statistics on completion rate and timing
- Lists failed experiments at the end

**Progress display example**:
```
[150/300] Completed: 150 | Failed: 5 | Rate: 2.5/min | Elapsed: 3600s
```

### 5. Results Collector: `collect_results.sh`
**Purpose**: Aggregates all experiment results and creates summaries

**Output files**:
- `all_results.json` - Complete results data
- `summary.json` - Statistical summary
- `results.csv` - CSV for analysis
- `visualize_results.py` - Script for creating plots

**Summary includes**:
- Total experiments run
- Success/failure rates
- Breakdown by model pair
- Breakdown by competition level

### 6. Retry Failed: `retry_failed.sh`
**Purpose**: Retries failed or timed-out experiments

**Usage**:
```bash
./scripts/retry_failed.sh [MAX_PARALLEL]
# Example: ./scripts/retry_failed.sh 4
```

**Features**:
- Automatically identifies failed experiments
- Archives old logs before retry
- Runs failed jobs in parallel
- Updates results after completion

### 7. Setup Script: `setup_experiment.sh`
**Purpose**: Prepares environment for running experiments

**What it checks**:
- Makes all scripts executable
- Verifies Python dependencies
- Checks API key environment variables
- Shows usage instructions

## Directory Structure

```
experiments/results/scaling_experiment/
├── configs/                          # Configuration files
│   ├── config_0.json
│   ├── config_1.json
│   └── ... (300 files total)
├── logs/                             # Execution logs
│   ├── experiment_0.log
│   ├── completed_0.flag             # Completion markers
│   └── ...
├── gpt-4o_vs_gemini-2-0-flash/     # Results by model pair
│   ├── comp_0.1/                    # By competition level
│   │   └── result_0.json
│   └── ...
├── all_results.json                 # Aggregated results
├── summary.json                     # Statistical summary
├── results.csv                      # CSV for analysis
└── visualize_results.py            # Visualization script
```

## Running Individual Steps

Instead of using the main orchestrator, you can run steps individually:

```bash
# Step 1: Generate configurations
./scripts/generate_configs.sh

# Step 2: Run experiments (with 6 parallel jobs)
./scripts/run_parallel_jobs.sh 6

# Step 3: Collect results
./scripts/collect_results.sh

# Optional: Retry any failures
./scripts/retry_failed.sh 4

# Optional: Visualize results (requires matplotlib)
python3 experiments/results/scaling_experiment/visualize_results.py
```

## Monitoring Progress

### During execution:
- Watch real-time progress in terminal
- Check individual logs: `tail -f experiments/results/scaling_experiment/logs/experiment_*.log`
- Count completed: `ls experiments/results/scaling_experiment/logs/completed_*.flag | wc -l`

### After execution:
- View summary: `cat experiments/results/scaling_experiment/summary.json | python3 -m json.tool`
- Check failures: `grep -l "FAILED\|TIMEOUT" experiments/results/scaling_experiment/logs/*.log`

## Resuming Interrupted Runs

The system automatically resumes from where it left off:

```bash
# If interrupted, just run again - completed experiments will be skipped
./scripts/run_all_experiments.sh --parallel 8

# Or run the parallel jobs script directly
./scripts/run_parallel_jobs.sh 8
```

## Performance Tuning

### Parallelism Guidelines:
- **API Rate Limits**: Don't exceed your API rate limits
  - Anthropic: ~5 requests/minute per key
  - OpenAI: ~60 requests/minute per key
  - OpenRouter: Varies by model
- **System Resources**: Each experiment uses ~1GB RAM
- **Recommended**: Start with 4-6 parallel jobs, adjust based on performance

### Time Estimates:
- Single experiment: 2-5 minutes
- Sequential (300 experiments): 10-25 hours
- Parallel (4 jobs): 2.5-6 hours
- Parallel (8 jobs): 1.5-3 hours

## Troubleshooting

### Common Issues:

1. **API Key Errors**:
   ```bash
   # Check keys are set
   echo $OPENROUTER_API_KEY
   echo $ANTHROPIC_API_KEY
   echo $OPENAI_API_KEY
   ```

2. **Permission Denied**:
   ```bash
   # Make scripts executable
   chmod +x scripts/*.sh
   ```

3. **Python Module Not Found**:
   ```bash
   # Install required packages
   pip install numpy pandas matplotlib
   ```

4. **Timeout Issues**:
   - Edit timeout in `run_single_experiment.sh` (line with `timeout 600`)
   - Default is 600 seconds (10 minutes)

5. **Resume Not Working**:
   - Check for completion flags: `ls logs/completed_*.flag`
   - Remove flag to re-run: `rm logs/completed_42.flag`

## Advanced Usage

### Running Subset of Experiments:
```bash
# Modify generate_configs.sh to only generate specific configs
# For example, only competition levels 0.5-0.9:
COMPETITION_LEVELS=(0.5 0.6 0.7 0.8 0.9)

# Or only specific model pairs:
WEAK_MODELS=("gpt-4o")
STRONG_MODELS=("o3" "chatgpt-5")
```

### Custom Analysis:
```python
# Load results in Python
import json
import pandas as pd

with open('experiments/results/scaling_experiment/all_results.json') as f:
    results = json.load(f)

df = pd.DataFrame(results)
# Your analysis here...
```

### Integration with SLURM (for cluster):
If you later want to run on a cluster, replace `run_parallel_jobs.sh` with:
```bash
#!/bin/bash
#SBATCH --array=0-299
#SBATCH --time=00:30:00
#SBATCH --mem=4G

./scripts/run_single_experiment.sh $SLURM_ARRAY_TASK_ID
```

## Notes

- Results are saved incrementally, so partial runs are not lost
- Each experiment is independent - order doesn't matter
- System is idempotent - safe to run multiple times
- Logs are verbose for debugging - check them if issues arise

## Contact & Updates

This experiment setup was created for the bargain negotiation research project.
Last updated: August 2024
Location: `/Users/qw281/Downloads/bargain/`
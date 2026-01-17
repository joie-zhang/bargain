# Experiment Workflow Guide

## Overview

This guide explains the workflow for running large-scale negotiation experiments using the config-based system.

## Step 1: Generate Configurations

Run the config generator to create all experiment configurations:

```bash
cd /scratch/gpfs/DANQIC/jz4391/bargain
./scripts/generate_configs_both_orders.sh
```

**What this creates:**
- `experiments/results/scaling_experiment/configs_YYYYMMDD_HHMMSS/` - **Timestamped config directory** (preserves previous experiments)
  - `config_*.json` - Individual experiment configs
  - `slurm/submit_all.sh` - Job submission script
  - `slurm/run_api_experiments.sbatch` - SLURM script for API models
  - `slurm/run_gpu_small.sbatch` - SLURM script for small GPU models
  - `slurm/run_gpu_large.sbatch` - SLURM script for large GPU models
  - `experiment_index.csv` - Searchable index
  - `summary.txt` - Human-readable summary
- `experiments/results/scaling_experiment/configs` - **Symlink** pointing to the latest timestamped directory

**Important:** 
- Each run of `generate_configs_both_orders.sh` creates a new timestamped folder, so previous experiment configs are preserved
- **The symlink is automatically created/updated** by `generate_configs_both_orders.sh` (defined at line 175-188 in the script)
- The `configs` symlink always points to the most recent timestamped directory for convenience
- **To update the symlink:** Simply run `generate_configs_both_orders.sh` again - it will create a new timestamped directory and update the symlink automatically

## Step 2: Submit Jobs

### Option A: Submit All at Once (Current Default - May Hit Rate Limits)

```bash
cd /scratch/gpfs/DANQIC/jz4391/bargain
./experiments/results/scaling_experiment/configs/slurm/submit_all.sh all
```

**Problem:** This submits all jobs simultaneously using SLURM array jobs, which can overwhelm API rate limits.

### Option B: Staggered Submission (Recommended for API Rate Limiting)

Use the staggered submission option:

```bash
cd /scratch/gpfs/DANQIC/jz4391/bargain
./experiments/results/scaling_experiment/configs/slurm/submit_all.sh all --staggered 2
```

This submits jobs individually with a 2-second delay between each submission to avoid rate limits.

**Recommended delays:**
- **2 seconds**: Good default for most cases
- **5 seconds**: More conservative, use if you're still hitting rate limits
- **1 second**: Faster but may still hit limits with many jobs

### Option C: Concurrency Limit (Control How Many Jobs Run Simultaneously)

Limit the number of concurrent jobs using SLURM's built-in concurrency control:

```bash
cd /scratch/gpfs/DANQIC/jz4391/bargain
./experiments/results/scaling_experiment/configs/slurm/submit_all.sh all --max-concurrent 10
```

This submits all jobs as an array but limits SLURM to run only 10 jobs simultaneously. As jobs complete, new ones start automatically.

**When to use:**
- **Array jobs with rate limiting**: Use `--max-concurrent` to control API load while still using efficient array jobs
- **Combined with staggered**: `--staggered` and `--max-concurrent` can be used together, but staggered mode submits individually so max-concurrent doesn't apply

**Recommended limits:**
- **5-10 concurrent**: Conservative, good for strict API limits
- **10-20 concurrent**: Moderate, balances throughput and rate limits
- **20+ concurrent**: Aggressive, use only if you have high API quotas

### Option D: Submit by Type

```bash
# Only API-based experiments (CPU)
./experiments/results/scaling_experiment/configs/slurm/submit_all.sh api --staggered 2

# Only GPU experiments
./experiments/results/scaling_experiment/configs/slurm/submit_all.sh gpu --staggered 2

# API jobs with concurrency limit
./experiments/results/scaling_experiment/configs/slurm/submit_all.sh api --max-concurrent 5
```

## Step 3: Monitor Jobs

```bash
# Check job status
squeue -u $USER

# View latest logs
tail -f logs/cluster/.latest.out
tail -f logs/cluster/.latest.err

# Check for rate limit errors
grep -r "429\|RateLimit\|rate limit" logs/cluster/*.err
```

## Understanding the Difference

### Array Jobs (Current `submit_all.sh` default)
- **How it works:** `sbatch --array=0,1,2,3...` submits all jobs at once
- **Pros:** Fast submission, SLURM manages scheduling
- **Cons:** All jobs start simultaneously → API rate limit issues

### Staggered Submission
- **How it works:** Submits individual jobs sequentially with delays
- **Pros:** Spreads load over time, avoids rate limits
- **Cons:** Takes longer to submit, but safer for API calls

### Concurrency-Limited Array Jobs
- **How it works:** `sbatch --array=0,1,2,3...%10` submits all jobs but limits to 10 running at once
- **Pros:** Efficient array job submission + controlled concurrency, SLURM manages scheduling
- **Cons:** Still submits all jobs at once (just limits execution), may still hit rate limits if limit is too high

## Why `submit_staggered.sh` Doesn't Work Here

`scripts/submit_staggered.sh` is designed for submitting the **same** `.sbatch` script multiple times. However, the config-based workflow uses:
- Different config IDs per job (`SLURM_ARRAY_TASK_ID`)
- Array jobs that reference config files

The new staggered submission is built into `submit_all.sh` and works with the config-based system.

## Troubleshooting

### Rate Limit Errors
- Use `--staggered` option with longer delays (e.g., `--staggered 5`)
- Reduce per-agent rate limits in agent factory
- Check `docs/RATE_LIMITING_GUIDE.md` for more strategies

### Job Failures
- Check error logs: `logs/cluster/*.err`
- Verify config files exist: `ls experiments/results/scaling_experiment/configs/config_*.json`
- Test single job: `./experiments/results/scaling_experiment/configs/slurm/submit_single.sh 0`

### Missing Configs
- Regenerate configs: `./scripts/generate_configs_both_orders.sh`
- Check summary: `cat experiments/results/scaling_experiment/configs/summary.txt`
- List all timestamped config directories: `ls -d experiments/results/scaling_experiment/configs_*`

### Finding Previous Experiment Configs
Each run of `generate_configs_both_orders.sh` creates a timestamped directory:
- Latest: `experiments/results/scaling_experiment/configs` (symlink)
- All versions: `ls -d experiments/results/scaling_experiment/configs_*`
- Specific version: `experiments/results/scaling_experiment/configs_20250115_143022/`

**Symlink Management:**
- **Location:** The symlink is defined in `scripts/generate_configs_both_orders.sh` at lines 198-218
- **Path:** `experiments/results/scaling_experiment/configs` → points to `scaling_experiment_YYYYMMDD_HHMMSS/`
- **Update:** The symlink is automatically updated every time you run `generate_configs_both_orders.sh`
- **No manual update needed:** Just run `./scripts/generate_configs_both_orders.sh` and the symlink will be refreshed

The symlink always points to the most recent config directory, so scripts continue to work without modification.

## Workflow Safety: Regenerating Configs During Active Experiments

### The Problem (Fixed)

Previously, if you regenerated configs while jobs were queued or running, there was a risk that:
1. Queued jobs would read config files from the updated symlink (pointing to new configs)
2. This could cause jobs to use wrong config files or fail to find their configs
3. Results might be saved to unexpected directories

### The Solution

**SLURM scripts now use absolute paths to timestamped config directories** instead of the symlink. This ensures:

1. **Queued jobs are safe**: Each SLURM script embeds the absolute path to its timestamped config directory at generation time
2. **No symlink confusion**: Jobs always read from the correct config directory, even if configs are regenerated
3. **Results go to correct location**: The `output_dir` in each config file contains the timestamp, so results are always saved to the correct experiment directory

### How It Works

When `generate_configs_both_orders.sh` runs:
1. Creates a new timestamped directory: `scaling_experiment_YYYYMMDD_HHMMSS/`
2. Generates config files with `output_dir` containing the timestamp
3. Generates SLURM scripts with **absolute paths** to the timestamped config directory
4. Updates the symlink for convenience (used by `submit_all.sh` and manual access)

**Example:**
- Config generation creates: `scaling_experiment_20250116_052234/`
- SLURM script embeds: `CONFIG_DIR="/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20250116_052234/configs"`
- Even if you regenerate configs later, queued jobs from this batch will still use `scaling_experiment_20250116_052234/`

### Safe Workflow

✅ **Safe to do:**
- Regenerate configs while jobs are running (they use absolute paths)
- Submit new jobs after regenerating configs (new jobs use new config directory)
- Run multiple experiment batches simultaneously (each uses its own timestamped directory)

⚠️ **Still be careful:**
- Don't delete timestamped config directories while jobs are running
- Check that `submit_all.sh` is using the correct config directory (it uses the symlink, which is fine for submitting new jobs)

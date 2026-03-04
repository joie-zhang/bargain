# Failed Ambitious Tier Jobs - Re-Run TODO (Feb 24, 2026)

## Summary

985 of 2700 ambitious tier jobs need re-running across both games.

| Game | Total | Succeeded | Failed | Success Rate |
|---|---|---|---|---|
| **Diplomacy** | 1350 | 1057 | **293** | 78.3% |
| **Co-Funding** | 1350 | 658 | **692** | 48.7% |

## Prerequisites

- [x] **OpenAI API key re-enabled** (confirmed working Mar 3)
  - Was archived ~00:17 Feb 24, restored by Mar 3
- [x] **Code fix applied**: `negotiation/openrouter_client.py:425` — added `"model_name": self.model_id` to `OpenRouterAgent.get_model_info()`

## Root Causes

### 1. Code Bug: `KeyError: 'model_name'` (300 failures)
- `OpenRouterAgent.get_model_info()` was missing `model_name` key
- Crash at `phase_handlers.py:222` during `save_interaction()`
- Affected: **amazon-nova-micro** (OpenRouter provider)
- **Status: FIXED**

### 2. OpenAI API Key Disabled (542 failures)
- Admin archived the OpenAI project around 00:17 Feb 24
- All cofunding pairs that were still running failed (gpt-5-nano can't authenticate)
- Affected (cofunding only — diplomacy finished before the key died):
  - gpt-5.2-high: 150/150
  - claude-sonnet-4-5: 150/150
  - gemini-3-pro: 150/150
  - claude-haiku-4-5: ~88 (partial, mix with overload)
  - gpt-5-nano self-play: 4

### 3. Anthropic API Overloaded (143 diplomacy failures)
- `overloaded_error` for claude-haiku-4-5 in diplomacy
- Retry logic (3 attempts) exhausted
- Transient issue, should succeed on re-run

## Failure Breakdown by Model

### Diplomacy (293 failed)

| Model Pair | Failed | Total | Failure Rate | Root Cause |
|---|---|---|---|---|
| gpt-5-nano vs amazon-nova-micro | 150 | 150 | 100% | Code bug (fixed) |
| gpt-5-nano vs claude-haiku-4-5 | 143 | 150 | 95.3% | Anthropic overloaded |

### Co-Funding (692 failed)

| Model Pair | Failed | Total | Failure Rate | Root Cause |
|---|---|---|---|---|
| gpt-5-nano vs amazon-nova-micro | 150 | 150 | 100% | Code bug (fixed) |
| gpt-5-nano vs gpt-5.2-high | 150 | 150 | 100% | OpenAI key disabled |
| gpt-5-nano vs claude-sonnet-4-5 | 150 | 150 | 100% | OpenAI key disabled |
| gpt-5-nano vs gemini-3-pro | 150 | 150 | 100% | OpenAI key disabled |
| gpt-5-nano vs claude-haiku-4-5 | 88 | 150 | 58.7% | OpenAI key + overload |
| gpt-5-nano vs gpt-5-nano | 4 | 150 | 2.7% | OpenAI key disabled |

## Re-Run Commands

### Step 1: Verify OpenAI API key is working

```bash
source .venv/bin/activate
python3 -c "
from openai import OpenAI
client = OpenAI()
r = client.chat.completions.create(model='gpt-5-nano-2025-08-07', messages=[{'role':'user','content':'hi'}], max_tokens=5)
print('OK:', r.choices[0].message.content)
"
```

### Step 2: Re-run Diplomacy failures (293 jobs)

```bash
cd /scratch/gpfs/DANQIC/jz4391/bargain

sbatch --array=300-449,750-881,883,885-889,891-895 \
    experiments/results/diplomacy_20260223_032204/configs/slurm/run_diplomacy_experiments.sbatch
```

### Step 3: Re-run Co-Funding failures (692 jobs)

Due to QOSMaxSubmitJobPerUserLimit (~2000 tasks), submit in batches if needed:

```bash
cd /scratch/gpfs/DANQIC/jz4391/bargain

# Batch 1: amazon-nova-micro + gpt-5-nano self-play + claude-haiku-4-5 (242 jobs)
sbatch --array=37,85,89,101,300-449,804-806,808,810-811,813-826,829-834,836-842,844,846-899 \
    experiments/results/cofunding_20260223_032239/configs/slurm/run_cofunding_experiments.sbatch

# Batch 2: gpt-5.2-high + claude-sonnet-4-5 (300 jobs)
sbatch --array=900-1199 \
    experiments/results/cofunding_20260223_032239/configs/slurm/run_cofunding_experiments.sbatch

# Batch 3: gemini-3-pro (150 jobs)
sbatch --array=1200-1349 \
    experiments/results/cofunding_20260223_032239/configs/slurm/run_cofunding_experiments.sbatch
```

### Step 4: Monitor

```bash
# Watch job progress
watch -n 60 'squeue -u jz4391 | tail -5; echo "---"; squeue -u jz4391 | wc -l'

# Check for new failures after completion
python3 -c "
import json, glob, os
for game, d in [('diplomacy', 'experiments/results/diplomacy_20260223_032204'),
                ('cofunding', 'experiments/results/cofunding_20260223_032239')]:
    configs = glob.glob(f'{d}/configs/config_*.json')
    failed = sum(1 for cf in configs
                 if not os.path.isfile(os.path.join(json.load(open(cf))['output_dir'], 'experiment_results.json')))
    print(f'{game}: {len(configs)-failed}/{len(configs)} succeeded ({failed} still missing)')
"
```

### Step 5: Regenerate visualizations

```bash
source .venv/bin/activate

# Diplomacy
python3 visualization/visualize_diplomacy.py \
    --dir experiments/results/diplomacy_20260223_032204

# Co-Funding
python3 visualization/visualize_cofunding.py \
    --results-dir experiments/results/cofunding_20260223_032239
```

## Experiment Directories

| Game | Config Dir | SLURM Script |
|---|---|---|
| Diplomacy | `experiments/results/diplomacy_20260223_032204/configs/` | `configs/slurm/run_diplomacy_experiments.sbatch` |
| Co-Funding | `experiments/results/cofunding_20260223_032239/configs/` | `configs/slurm/run_cofunding_experiments.sbatch` |

## Re-Run Status (Updated Mar 3, 2026)

### Resubmission #1 (Mar 3, ~07:00 EST)
- Diplomacy Job 5289855: 282 tasks → 189 completed, ~93 timed out (6h limit too short for claude-haiku-4-5 with API contention)
- Cofunding Jobs 5289856/5289857/5289862: Cancelled due to API contention (350+ concurrent jobs → 5-10 min per API call)

### Resubmission #2 (Mar 3, 13:08 EST) — CURRENT
- **Diplomacy Job 5302273**: 104 tasks, `--time=24:00:00`, `%20` throttle
  - All claude-haiku-4-5. Running cleanly, no API errors.
- **Cofunding Job 5302284**: 692 tasks, `--time=24:00:00`, `%20` throttle
  - 6 model pairs (amazon-nova-micro, claude-haiku-4-5, claude-sonnet-4-5, gemini-3-pro, gpt-5.2-high, gpt-5-nano)
  - Running cleanly, scaling up.

### Current Completion (Mar 3, 23:39 EST)
| Game | Completed | Missing |
|---|---|---|
| Diplomacy | 1290/1350 | 60 (all claude-haiku-4-5) |
| Cofunding | 658/1350 | 692 (6 model pairs) |

### Bug Fix: `num_runs` redundancy in cofunding
- **Issue**: Each cofunding config had `num_runs=3` causing 3 identical iterations per SLURM task
- Configs already have separate run_1/run_2/run_3 with unique seeds, so `num_runs=3` was redundant
- Fixed sbatch script to force `--num-runs 1`, reducing per-task time from ~15h to ~5h

## Timeline

- **Feb 23 03:22**: Ambitious configs generated (1350 per game)
- **Feb 23 ~03:30**: All jobs submitted
- **Feb 23 ~15:30**: Diplomacy 100% complete (1350/1350 SLURM COMPLETED)
- **Feb 24 00:17**: OpenAI API key disabled — remaining cofunding jobs start failing
- **Feb 24 ~01:30**: All cofunding jobs finished (658 success, 692 fail)
- **Feb 24**: Code fix applied for OpenRouter `model_name` bug
- **Mar 3 07:00**: Resubmission #1 — diplomacy partially completed, cofunding cancelled (API contention)
- **Mar 3 13:08**: Resubmission #2 — all remaining failures resubmitted with 24h time limit and %20 concurrency throttle
- **Mar 3 18:25**: Fixed cofunding `num_runs` redundancy (3→1), cancelled and resubmitted cofunding job 5314832
- **Mar 3 ~22:36**: Cofunding job 5314832 cancelled by user (200 tasks, 0 completed)
- **Mar 3 23:32**: Cofunding resubmitted as job 5325457 with %50 throttle

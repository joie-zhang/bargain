# Failed Ambitious Tier Jobs - Re-Run TODO (Feb 24, 2026)

## Summary

985 of 2700 ambitious tier jobs need re-running across both games.

| Game | Total | Succeeded | Failed | Success Rate |
|---|---|---|---|---|
| **Diplomacy** | 1350 | 1057 | **293** | 78.3% |
| **Co-Funding** | 1350 | 658 | **692** | 48.7% |

## Prerequisites

- [ ] **OpenAI API key must be re-enabled** (project was archived ~00:17 Feb 24)
  - Error: `openai.AuthenticationError: 401 - The project you are requesting has been archived and is no longer accessible`
  - Affects all pairs since gpt-5-nano (baseline) uses OpenAI
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

## Timeline

- **Feb 23 03:22**: Ambitious configs generated (1350 per game)
- **Feb 23 ~03:30**: All jobs submitted
- **Feb 23 ~15:30**: Diplomacy 100% complete (1350/1350 SLURM COMPLETED)
- **Feb 24 00:17**: OpenAI API key disabled — remaining cofunding jobs start failing
- **Feb 24 ~01:30**: All cofunding jobs finished (658 success, 692 fail)
- **Feb 24**: Code fix applied for OpenRouter `model_name` bug

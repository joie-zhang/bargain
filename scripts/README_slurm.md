# Slurm Job Submission for Qwen Large Models Experiments

This directory contains Slurm submission scripts for running Qwen2.5 large model experiments (14B, 32B, 72B) with competition_level=1.

## Available Scripts

### Model-Specific Scripts (Recommended)

These scripts are optimized for each model size with appropriate GPU and memory allocations:

#### 1. `submit_qwen_14b_comp1.sbatch`
- **Model**: Qwen2.5-14B-Instruct
- **GPUs**: 1 H100 GPU (80GB)
- **Memory**: 80GB
- **Submit**: `sbatch scripts/submit_qwen_14b_comp1.sbatch`

#### 2. `submit_qwen_32b_comp1.sbatch`
- **Model**: Qwen2.5-32B-Instruct
- **GPUs**: 2 H100 GPUs (160GB total)
- **Memory**: 160GB
- **Submit**: `sbatch scripts/submit_qwen_32b_comp1.sbatch`

#### 3. `submit_qwen_72b_comp1.sbatch`
- **Model**: Qwen2.5-72B-Instruct
- **GPUs**: 4 H100 GPUs (320GB total)
- **Memory**: 320GB
- **Submit**: `sbatch scripts/submit_qwen_72b_comp1.sbatch`

### Legacy Scripts (Not Recommended - Wrong GPU Allocation)

(all have been deleted, but we should frequently check to make sure we are keeping the codebase clean and intuitive)

## Configuration

### PLI Cluster Scripts (Default)
- **Partition**: `gpu` (PLI cluster with H100 GPUs)
- **H100 GPU Memory**: 80GB per GPU
- **GPUs**: Model-specific allocation (see GPU_REQUIREMENTS.md)
  - 14B model: 1 GPU (80GB)
  - 32B model: 2 GPUs (160GB)
  - 72B model: 4 GPUs (320GB)
- **Memory**: Matches GPU memory allocation
- **Proxy Module**: Loads `proxy/default` to enable API access (Gemini, Anthropic, OpenAI) and Wandb

### Della Cluster Script
- **Script**: `submit_qwen_large_models_comp1_della.sbatch`
- **Partition**: `della-gpu`
- **Memory**: `256G` (configured for 72B model - see MEMORY_REQUIREMENTS.md)
- **Proxy Module**: Automatically loads `proxy/default` to enable API access (Gemini, Anthropic, OpenAI) and Wandb

Before submitting, you may need to adjust:

1. **Time Limit**: Adjust `#SBATCH --time=01:00:00` if needed:
   - Format: `HH:MM:SS` or `D-HH:MM:SS` for days
   - Example: `02:00:00` for 2 hours, `1-00:00:00` for 1 day

2. **GPU and Memory**: Model-specific allocations (see `GPU_REQUIREMENTS.md`):
   - **14B model**: 1 GPU, 80GB memory (~50-70 GB minimum, 80-96 GB recommended)
   - **32B model**: 2 GPUs, 160GB memory (~100-120 GB minimum, 128-160 GB recommended)
   - **72B model**: 4 GPUs, 320GB memory (~190-240 GB minimum, 256 GB recommended)
   - Models use `device_map="auto"` to automatically distribute across GPUs
   - See `MEMORY_REQUIREMENTS.md` and `GPU_REQUIREMENTS.md` for detailed breakdowns
   - Check with `squeue` or `sacct` after first run if you see OOM errors

3. **GPU Requirements**: Currently `--gpus-per-node=1`. Adjust if needed:
   - Some clusters require specific GPU types: `--gres=gpu:1` or `--gres=gpu:h100:1`

4. **Modules**: Uncomment and adjust module loading lines based on your cluster:
   ```bash
   module purge
   module load python/3.9
   module load cuda/11.8
   ```
   - **All scripts**: Automatically load `proxy/default` for API access (Gemini, Anthropic, OpenAI) and Wandb

## Usage

### Submit Sequential Job
```bash
cd /scratch/gpfs/DANQIC/jz4391/bargain
sbatch scripts/submit_qwen_large_models_comp1.sbatch
```

### Submit Model-Specific Jobs (Recommended)
```bash
cd /scratch/gpfs/DANQIC/jz4391/bargain

# Submit 14B model (1 GPU)
sbatch scripts/submit_qwen_14b_comp1.sbatch

# Submit 32B model (2 GPUs)
sbatch scripts/submit_qwen_32b_comp1.sbatch

# Submit 72B model (4 GPUs)
sbatch scripts/submit_qwen_72b_comp1.sbatch
```

### Submit Array Job (Della Cluster - with API access)
```bash
cd /scratch/gpfs/DANQIC/jz4391/bargain
sbatch scripts/submit_qwen_large_models_comp1_della.sbatch
```

### Monitor Jobs
```bash
# Check job status
squeue -u $USER

# Check specific job
squeue -j <JOB_ID>

# View output (while running)
tail -f logs/cluster/qwen_large_comp1_<JOB_ID>.out

# View error log
tail -f logs/cluster/qwen_large_comp1_<JOB_ID>.err
```

### Cancel Jobs
```bash
# Cancel specific job
scancel <JOB_ID>

# Cancel all your jobs
scancel -u $USER

# Cancel array job
scancel <ARRAY_JOB_ID>
```

## Output

Results will be saved in:
- `experiments/results/Qwen2.5-14B-Instruct_vs_claude-3-7-sonnet_runs5_comp1/`
- `experiments/results/Qwen2.5-32B-Instruct_vs_claude-3-7-sonnet_runs5_comp1/`
- `experiments/results/Qwen2.5-72B-Instruct_vs_claude-3-7-sonnet_runs5_comp1/`

If directories already exist, numbered suffixes (`_1`, `_2`, etc.) will be added automatically.

## Troubleshooting

1. **Job fails immediately**: Check error log for module/environment issues
2. **Out of memory**: Increase `--mem` or reduce batch size
3. **Time limit exceeded**: Increase `--time` or split into smaller jobs
4. **GPU not available**: Check partition availability with `sinfo -p gpu`
5. **Module not found**: Check available modules with `module avail`

## Notes

- The scripts automatically detect and activate your virtual environment
- Email notifications are enabled for job completion/failure
- Logs are saved to `logs/cluster/` directory
- Each model runs 5 experiments with competition_level=1, max_rounds=10


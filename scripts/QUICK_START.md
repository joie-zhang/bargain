# Quick Start Guide - Qwen Large Models Experiments

## H100 GPU Specifications
- **Memory per GPU**: 80 GB HBM2e
- **Cluster**: PLI cluster

## Submit Jobs

### Option 1: Submit Each Model Separately (Recommended)

```bash
cd /scratch/gpfs/DANQIC/jz4391/bargain

# 14B model - 1 GPU, 80GB
sbatch scripts/submit_qwen_14b_comp1.sbatch

# 32B model - 2 GPUs, 160GB
sbatch scripts/submit_qwen_32b_comp1.sbatch

# 72B model - 4 GPUs, 320GB
sbatch scripts/submit_qwen_72b_comp1.sbatch
```

### Option 2: Submit All at Once

```bash
cd /scratch/gpfs/DANQIC/jz4391/bargain

# Submit all three models
sbatch scripts/submit_qwen_14b_comp1.sbatch
sbatch scripts/submit_qwen_32b_comp1.sbatch
sbatch scripts/submit_qwen_72b_comp1.sbatch
```

## Monitor Jobs

```bash
# Check job status
squeue -u $USER

# View output (replace JOB_ID with actual job ID)
tail -f logs/cluster/qwen_14b_comp1_<JOB_ID>.out

# View error log
tail -f logs/cluster/qwen_14b_comp1_<JOB_ID>.err
```

## Experiment Configuration

All scripts run:
- **Competition Level**: 1 (full competition)
- **Runs per model**: 5
- **Items**: 5
- **Max Rounds**: 10
- **Adversary**: Claude-3.7-Sonnet

## Results Location

Results are saved in:
- `experiments/results/Qwen2.5-14B-Instruct_vs_claude-3-7-sonnet_runs5_comp1/`
- `experiments/results/Qwen2.5-32B-Instruct_vs_claude-3-7-sonnet_runs5_comp1/`
- `experiments/results/Qwen2.5-72B-Instruct_vs_claude-3-7-sonnet_runs5_comp1/`

If directories already exist, numbered suffixes (`_1`, `_2`, etc.) will be added automatically.

## Troubleshooting

1. **Out of Memory**: Check that you're using the correct script for each model size
2. **Job Fails**: Check error logs in `logs/cluster/`
3. **API Errors**: Ensure `module load proxy/default` is loaded (already in scripts)

## See Also

- `GPU_REQUIREMENTS.md` - Detailed GPU allocation rationale
- `README_slurm.md` - Complete Slurm documentation


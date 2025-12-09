#!/bin/bash
#SBATCH --job-name=qwen_72b_run5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --constraint=gpu80
#SBATCH --mem=320G
#SBATCH --time=01:00:00
#SBATCH --partition=pli-c
#SBATCH --output=logs/cluster/qwen_72b_run5_%j.out
#SBATCH --error=logs/cluster/qwen_72b_run5_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=joie@princeton.edu

# This script runs a single run (Run 5) for Qwen2.5-72B-Instruct vs Claude-3.7-Sonnet
# Seed: 46, competition_level=1, max_rounds=10

set -e

echo "============================================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Model: Qwen2.5-72B-Instruct"
echo "Run Number: 5"
echo "Random Seed: 46"
echo "Started at: $(date)"
echo "Node: $SLURM_NODELIST"
echo "============================================================"

# Change to project directory
cd "/scratch/gpfs/DANQIC/jz4391/bargain"

# Load proxy module to enable API access
module load proxy/default

# Activate virtual environment
if [ -f "/scratch/gpfs/DANQIC/jz4391/bargain/.venv/bin/activate" ]; then
    source "/scratch/gpfs/DANQIC/jz4391/bargain/.venv/bin/activate"
elif [ -f "/scratch/gpfs/DANQIC/jz4391/bargain/venv/bin/activate" ]; then
    source "/scratch/gpfs/DANQIC/jz4391/bargain/venv/bin/activate"
elif [ -f ~/.venv/bin/activate ]; then
    source ~/.venv/bin/activate
fi

# Verify Python and CUDA
echo "Python version: $(python3 --version)"
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs/cluster

echo ""
echo "Running single experiment:"
echo "  Model: Qwen2.5-72B-Instruct"
echo "  Adversary: claude-3-7-sonnet"
echo "  Run Number: 5"
echo "  Random Seed: 46"
echo "  Competition Level: 1"
echo "  Output Directory: experiments/results/Qwen2.5-72B-Instruct_vs_claude-3-7-sonnet_runs5_comp1"
echo ""

# Run a single experiment using batch mode with num-runs=1 to properly handle run-number
if python3 run_strong_models_experiment.py \
    --models "Qwen2.5-72B-Instruct" "claude-3-7-sonnet" \
    --competition-level 1 \
    --num-items 5 \
    --max-rounds 10 \
    --random-seed 46 \
    --batch \
    --num-runs 1 \
    --run-number 5 \
    --output-dir "experiments/results/Qwen2.5-72B-Instruct_vs_claude-3-7-sonnet_runs5_comp1"; then
    echo ""
    echo "✅ Successfully completed Run 5 for Qwen2.5-72B-Instruct"
else
    echo ""
    echo "❌ Failed to complete Run 5 for Qwen2.5-72B-Instruct"
    exit 1
fi

echo ""
echo "============================================================"
echo "Job completed at: $(date)"
echo "============================================================"

#!/bin/bash
#SBATCH --job-name=qwen_32b_comp1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:2
#SBATCH --constraint=gpu80
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --partition=pli-c
#SBATCH --output=logs/cluster/qwen_32b_comp1_%j.out
#SBATCH --error=logs/cluster/qwen_32b_comp1_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=joie@princeton.edu

# This script runs Qwen2.5-32B-Instruct vs Claude-3.7-Sonnet
# 5 runs with competition_level=1, max_rounds=10
# Running on PLI cluster (H100 GPUs - 2 GPUs, 160GB total)

set -e

QWEN_MODEL="Qwen2.5-32B-Instruct"
ADVERSARY_MODEL="claude-3-7-sonnet"
COMPETITION_LEVEL=1
NUM_RUNS=5
NUM_ITEMS=5
MAX_ROUNDS=10

echo "============================================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Model: ${QWEN_MODEL}"
echo "Started at: $(date)"
echo "Node: $SLURM_NODELIST"
echo "Cluster: PLI (H100 GPUs - 2 GPUs, 160GB total)"
echo "============================================================"

# Change to project directory
BASE_DIR="/scratch/gpfs/DANQIC/jz4391/bargain"
cd "${BASE_DIR}"

# Load required modules (PLI cluster)
# Uncomment/modify based on your cluster's available modules
# module purge
# module load python/3.9
# module load cuda/11.8
# module load miniconda3/4.10.3

# Load proxy module to enable API access (Gemini, Anthropic, OpenAI) and Wandb
module load proxy/default

# Activate virtual environment
if [ -f "${BASE_DIR}/.venv/bin/activate" ]; then
    source "${BASE_DIR}/.venv/bin/activate"
    echo "Activated virtual environment: ${BASE_DIR}/.venv"
elif [ -f "${BASE_DIR}/venv/bin/activate" ]; then
    source "${BASE_DIR}/venv/bin/activate"
    echo "Activated virtual environment: ${BASE_DIR}/venv"
elif [ -f ~/.venv/bin/activate ]; then
    source ~/.venv/bin/activate
    echo "Activated virtual environment: ~/.venv"
else
    echo "Warning: No virtual environment found. Using system Python."
fi

# Verify Python and CUDA
echo "Python version: $(python3 --version)"
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA devices: $(python3 -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs/cluster

# Generate unique output directory with number suffix to avoid overwriting
BASE_OUTPUT_DIR="experiments/results/${QWEN_MODEL}_vs_${ADVERSARY_MODEL}_runs${NUM_RUNS}_comp${COMPETITION_LEVEL}"

# Check if base directory exists and find next available number
RUN_NUM=1
if [ -d "${BASE_OUTPUT_DIR}" ]; then
    OUTPUT_DIR="${BASE_OUTPUT_DIR}_${RUN_NUM}"
    while [ -d "${OUTPUT_DIR}" ]; do
        RUN_NUM=$((RUN_NUM + 1))
        OUTPUT_DIR="${BASE_OUTPUT_DIR}_${RUN_NUM}"
    done
    echo "Existing results found. Using numbered directory: ${OUTPUT_DIR}"
elif ls -d "${BASE_OUTPUT_DIR}_"* 2>/dev/null | grep -q .; then
    MAX_NUM=0
    for existing_dir in "${BASE_OUTPUT_DIR}_"*; do
        if [ -d "${existing_dir}" ]; then
            suffix=$(basename "${existing_dir}" | sed "s|${BASE_OUTPUT_DIR}_||")
            if [[ "${suffix}" =~ ^[0-9]+$ ]]; then
                num=${suffix}
                if [ "${num}" -gt "${MAX_NUM}" ]; then
                    MAX_NUM=${num}
                fi
            fi
        fi
    done
    RUN_NUM=$((MAX_NUM + 1))
    OUTPUT_DIR="${BASE_OUTPUT_DIR}_${RUN_NUM}"
    echo "Existing numbered results found. Using directory: ${OUTPUT_DIR}"
else
    OUTPUT_DIR="${BASE_OUTPUT_DIR}"
    echo "No existing results found. Using directory: ${OUTPUT_DIR}"
fi

echo ""
echo "Running experiment for: ${QWEN_MODEL}"
echo "Command: python3 run_strong_models_experiment.py \\"
echo "    --models ${QWEN_MODEL} ${ADVERSARY_MODEL} \\"
echo "    --competition-level ${COMPETITION_LEVEL} \\"
echo "    --num-items ${NUM_ITEMS} \\"
echo "    --max-rounds ${MAX_ROUNDS} \\"
echo "    --batch \\"
echo "    --num-runs ${NUM_RUNS} \\"
echo "    --output-dir ${OUTPUT_DIR}"
echo ""

# Run the experiment
if python3 run_strong_models_experiment.py \
    --models "${QWEN_MODEL}" "${ADVERSARY_MODEL}" \
    --competition-level "${COMPETITION_LEVEL}" \
    --num-items "${NUM_ITEMS}" \
    --max-rounds "${MAX_ROUNDS}" \
    --batch \
    --num-runs "${NUM_RUNS}" \
    --output-dir "${OUTPUT_DIR}"; then
    echo ""
    echo "✅ Successfully completed ${NUM_RUNS} runs for ${QWEN_MODEL}"
else
    echo ""
    echo "❌ Failed to complete experiments for ${QWEN_MODEL}"
    exit 1
fi

echo ""
echo "============================================================"
echo "Job completed at: $(date)"
echo "============================================================"


#!/bin/bash
#SBATCH --job-name=gpt5_med_vs_low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/cluster/gpt5_med_vs_low_%j.out
#SBATCH --error=logs/cluster/gpt5_med_vs_low_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=joie@princeton.edu

# This script runs GPT-5 Medium Reasoning Effort vs GPT-5 Low Reasoning Effort
# 5 runs with competition_level=1, max_rounds=10
# Running on PLI cluster (CPU-only, API calls to OpenAI)

set -e

MODEL1="gpt-5-medium-effort"
MODEL2="gpt-5-low-effort"
COMPETITION_LEVEL=1
NUM_RUNS=5
NUM_ITEMS=5
MAX_ROUNDS=10

echo "============================================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Models: ${MODEL1} vs ${MODEL2}"
echo "Started at: $(date)"
echo "Node: $SLURM_NODELIST"
echo "Cluster: PLI (CPU-only, API calls)"
echo "============================================================"

# Change to project directory
BASE_DIR="/scratch/gpfs/DANQIC/jz4391/bargain"
cd "${BASE_DIR}"

# Load required modules (PLI cluster)
module purge

# Load proxy module to enable API access (OpenAI) and Wandb
module load proxy/default

# Activate virtual environment
source "${BASE_DIR}/.venv/bin/activate"
echo "Activated virtual environment: ${BASE_DIR}/.venv"

# Verify Python
echo "Python version: $(python3 --version)"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs/cluster

# Generate unique output directory with number suffix to avoid overwriting
BASE_OUTPUT_DIR="experiments/results/${MODEL1}_vs_${MODEL2}_runs${NUM_RUNS}_comp${COMPETITION_LEVEL}"

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
echo "Running experiment: ${MODEL1} vs ${MODEL2}"
echo "Command: python3 run_strong_models_experiment.py \\"
echo "    --models ${MODEL1} ${MODEL2} \\"
echo "    --competition-level ${COMPETITION_LEVEL} \\"
echo "    --num-items ${NUM_ITEMS} \\"
echo "    --max-rounds ${MAX_ROUNDS} \\"
echo "    --batch \\"
echo "    --num-runs ${NUM_RUNS} \\"
echo "    --output-dir ${OUTPUT_DIR}"
echo ""

# Run the experiment
if python3 run_strong_models_experiment.py \
    --models "${MODEL1}" "${MODEL2}" \
    --competition-level "${COMPETITION_LEVEL}" \
    --num-items "${NUM_ITEMS}" \
    --max-rounds "${MAX_ROUNDS}" \
    --batch \
    --num-runs "${NUM_RUNS}" \
    --output-dir "${OUTPUT_DIR}"; then
    echo ""
    echo "✅ Successfully completed ${NUM_RUNS} runs for ${MODEL1} vs ${MODEL2}"
else
    echo ""
    echo "❌ Failed to complete experiments for ${MODEL1} vs ${MODEL2}"
    exit 1
fi

echo ""
echo "============================================================"
echo "Job completed at: $(date)"
echo "============================================================"


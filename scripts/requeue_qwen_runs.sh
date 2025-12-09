#!/bin/bash
# Script to requeue specific runs for Qwen models with correct seeds
# Usage: ./requeue_qwen_runs.sh [14b|32b|72b] [run_numbers...]
# Example: ./requeue_qwen_runs.sh 14b 4 5
# Example: ./requeue_qwen_runs.sh 32b 3 4 5

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 [14b|32b|72b] [run_numbers...]"
    echo "Example: $0 14b 4 5"
    echo "Example: $0 32b 3 4 5"
    exit 1
fi

MODEL_SIZE=$1
shift
RUN_NUMBERS=("$@")

# Map model size to full model name and GPU config
case $MODEL_SIZE in
    14b)
        QWEN_MODEL="Qwen2.5-14B-Instruct"
        GPU_COUNT=1
        MEMORY=80G
        SBATCH_FILE="submit_qwen_14b_comp1.sbatch"
        ;;
    32b)
        QWEN_MODEL="Qwen2.5-32B-Instruct"
        GPU_COUNT=2
        MEMORY=160G
        SBATCH_FILE="submit_qwen_32b_comp1.sbatch"
        ;;
    72b)
        QWEN_MODEL="Qwen2.5-72B-Instruct"
        GPU_COUNT=4
        MEMORY=320G
        SBATCH_FILE="submit_qwen_72b_comp1.sbatch"
        ;;
    *)
        echo "Error: Invalid model size. Use 14b, 32b, or 72b"
        exit 1
        ;;
esac

ADVERSARY_MODEL="claude-3-7-sonnet"
COMPETITION_LEVEL=1
NUM_ITEMS=5
MAX_ROUNDS=10
BASE_DIR="/scratch/gpfs/DANQIC/jz4391/bargain"
OUTPUT_DIR="experiments/results/${QWEN_MODEL}_vs_${ADVERSARY_MODEL}_runs5_comp${COMPETITION_LEVEL}"

echo "============================================================"
echo "Requeuing runs for ${QWEN_MODEL}"
echo "Runs: ${RUN_NUMBERS[*]}"
echo "Output directory: ${OUTPUT_DIR}"
echo "============================================================"

# Verify output directory exists
if [ ! -d "${BASE_DIR}/${OUTPUT_DIR}" ]; then
    echo "Error: Output directory does not exist: ${OUTPUT_DIR}"
    exit 1
fi

# Create temporary sbatch script for each run
for RUN_NUM in "${RUN_NUMBERS[@]}"; do
    # Calculate seed: run 1 = 42, run 2 = 43, run 3 = 44, etc.
    SEED=$((41 + RUN_NUM))
    
    echo ""
    echo "Creating job for Run ${RUN_NUM} with seed ${SEED}..."
    
    # Create a temporary sbatch script for this specific run
    TEMP_SBATCH="${BASE_DIR}/scripts/temp_requeue_${MODEL_SIZE}_run${RUN_NUM}.sbatch"
    
    cat > "${TEMP_SBATCH}" <<EOF
#!/bin/bash
#SBATCH --job-name=qwen_${MODEL_SIZE}_run${RUN_NUM}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:${GPU_COUNT}
#SBATCH --constraint=gpu80
#SBATCH --mem=${MEMORY}
#SBATCH --time=01:00:00
#SBATCH --partition=pli-c
#SBATCH --output=logs/cluster/qwen_${MODEL_SIZE}_run${RUN_NUM}_%j.out
#SBATCH --error=logs/cluster/qwen_${MODEL_SIZE}_run${RUN_NUM}_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=joie@princeton.edu

# This script runs a single run (Run ${RUN_NUM}) for ${QWEN_MODEL} vs Claude-3.7-Sonnet
# Seed: ${SEED}, competition_level=${COMPETITION_LEVEL}, max_rounds=${MAX_ROUNDS}

set -e

echo "============================================================"
echo "SLURM Job ID: \$SLURM_JOB_ID"
echo "Model: ${QWEN_MODEL}"
echo "Run Number: ${RUN_NUM}"
echo "Random Seed: ${SEED}"
echo "Started at: \$(date)"
echo "Node: \$SLURM_NODELIST"
echo "============================================================"

# Change to project directory
cd "${BASE_DIR}"

# Load proxy module to enable API access
module load proxy/default

# Activate virtual environment
if [ -f "${BASE_DIR}/.venv/bin/activate" ]; then
    source "${BASE_DIR}/.venv/bin/activate"
elif [ -f "${BASE_DIR}/venv/bin/activate" ]; then
    source "${BASE_DIR}/venv/bin/activate"
elif [ -f ~/.venv/bin/activate ]; then
    source ~/.venv/bin/activate
fi

# Verify Python and CUDA
echo "Python version: \$(python3 --version)"
echo "CUDA available: \$(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs/cluster

echo ""
echo "Running single experiment:"
echo "  Model: ${QWEN_MODEL}"
echo "  Adversary: ${ADVERSARY_MODEL}"
echo "  Run Number: ${RUN_NUM}"
echo "  Random Seed: ${SEED}"
echo "  Competition Level: ${COMPETITION_LEVEL}"
echo "  Output Directory: ${OUTPUT_DIR}"
echo ""

# Run a single experiment using batch mode with num-runs=1 to properly handle run-number
if python3 run_strong_models_experiment.py \\
    --models "${QWEN_MODEL}" "${ADVERSARY_MODEL}" \\
    --competition-level ${COMPETITION_LEVEL} \\
    --num-items ${NUM_ITEMS} \\
    --max-rounds ${MAX_ROUNDS} \\
    --random-seed ${SEED} \\
    --batch \\
    --num-runs 1 \\
    --run-number ${RUN_NUM} \\
    --output-dir "${OUTPUT_DIR}"; then
    echo ""
    echo "✅ Successfully completed Run ${RUN_NUM} for ${QWEN_MODEL}"
else
    echo ""
    echo "❌ Failed to complete Run ${RUN_NUM} for ${QWEN_MODEL}"
    exit 1
fi

echo ""
echo "============================================================"
echo "Job completed at: \$(date)"
echo "============================================================"
EOF

    # Submit the job
    echo "Submitting job for Run ${RUN_NUM}..."
    JOB_ID=$(sbatch "${TEMP_SBATCH}" | grep -oP '\d+$')
    echo "  Job ID: ${JOB_ID}"
    
    # Clean up temporary script (optional - comment out if you want to keep them for debugging)
    # rm "${TEMP_SBATCH}"
done

echo ""
echo "============================================================"
echo "All jobs submitted successfully!"
echo "Check status with: squeue -u \$USER"
echo "============================================================"


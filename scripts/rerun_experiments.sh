#!/bin/bash
# Re-run experiments (gpt-5-nano vs gemini-3-pro, comp_0.75 and comp_1.0)
# Usage: ./scripts/rerun_experiments.sh

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${BASE_DIR}"

# Config directory for this specific experiment batch
CONFIG_DIR="experiments/results/scaling_experiment_20260116_052234/configs"
SBATCH_SCRIPT="${CONFIG_DIR}/slurm/run_api_experiments.sbatch"

# Experiments to re-run
# EXPERIMENT_IDS=(23 24 25 26 27 28 29)
EXPERIMENT_IDS=(97 114)
# EXPERIMENT_IDS=(18)

echo "============================================================"
echo "Re-running experiments: ${EXPERIMENT_IDS[@]}"
echo "Config directory: ${CONFIG_DIR}"
echo "============================================================"
echo ""

# Verify config files exist
for ID in "${EXPERIMENT_IDS[@]}"; do
    CONFIG_FILE="${CONFIG_DIR}/config_$(printf "%03d" ${ID}).json"
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "❌ ERROR: Config file not found: $CONFIG_FILE"
        exit 1
    fi
    echo "✓ Found config: $CONFIG_FILE"
done

echo ""
echo "Submitting jobs to SLURM..."
echo ""

# Submit as SLURM array job
# Format: sbatch --array=23-29 run_api_experiments.sbatch
# Note: The sbatch script uses a hardcoded CONFIG_DIR path, so we need to modify it
# Create a temporary sbatch script with the correct CONFIG_DIR

TEMP_SBATCH=$(mktemp)
# Replace the CONFIG_DIR line in the sbatch script with our specific directory
sed "s|CONFIG_DIR=\"experiments/results/scaling_experiment/configs\"|CONFIG_DIR=\"${CONFIG_DIR}\"|g" "${SBATCH_SCRIPT}" > "${TEMP_SBATCH}"

echo "Submitting SLURM array job..."
JOB_OUTPUT=$(sbatch --array=$(IFS=,; echo "${EXPERIMENT_IDS[*]}") "${TEMP_SBATCH}" 2>&1)
echo "$JOB_OUTPUT"

# Extract job ID from output if available
if echo "$JOB_OUTPUT" | grep -q "Submitted batch job"; then
    JOB_ID=$(echo "$JOB_OUTPUT" | sed -n 's/.*Submitted batch job \([0-9]*\).*/\1/p')
    echo ""
    echo "Job ID: $JOB_ID"
    echo "Array tasks: ${EXPERIMENT_IDS[@]}"
fi

# Clean up temp file
rm "${TEMP_SBATCH}"

echo ""
echo "✅ Submitted experiments ${EXPERIMENT_IDS[@]} to SLURM"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u $USER"
echo ""
echo "Check logs in:"
echo "  logs/cluster/api_<JOB_ID>_<TASK_ID>.out"
echo "  logs/cluster/api_<JOB_ID>_<TASK_ID>.err"

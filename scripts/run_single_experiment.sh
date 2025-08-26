#!/bin/bash
# Run a single experiment based on job ID

set -e

# Get job ID from argument or environment variable (simulating SLURM_ARRAY_TASK_ID)
JOB_ID=${1:-${JOB_ARRAY_ID:-0}}

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="${BASE_DIR}/experiments/results/scaling_experiment/configs"
LOGS_DIR="${BASE_DIR}/experiments/results/scaling_experiment/logs"

# Create log directory
mkdir -p "${LOGS_DIR}"

# Get configuration file
CONFIG_FILE="${CONFIG_DIR}/config_${JOB_ID}.json"

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "❌ Configuration file not found: ${CONFIG_FILE}"
    exit 1
fi

# Extract parameters from JSON config
WEAK_MODEL=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['weak_model'])")
STRONG_MODEL=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['strong_model'])")
COMP_LEVEL=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['competition_level'])")
NUM_ITEMS=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['num_items'])")
MAX_ROUNDS=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['max_rounds'])")
RANDOM_SEED=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['random_seed'])")
OUTPUT_DIR=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['output_dir'])")

# Create output directory
FULL_OUTPUT_DIR="${BASE_DIR}/${OUTPUT_DIR}"
mkdir -p "${FULL_OUTPUT_DIR}"

# Log file for this experiment
LOG_FILE="${LOGS_DIR}/experiment_${JOB_ID}.log"

echo "[$(date)] Starting experiment ${JOB_ID}: ${WEAK_MODEL} vs ${STRONG_MODEL} (competition=${COMP_LEVEL})" | tee "${LOG_FILE}"

# Run the actual experiment
cd "${BASE_DIR}"

# Build the command
CMD="python3 run_strong_models_experiment.py \
    --models ${WEAK_MODEL} ${STRONG_MODEL} \
    --competition-level ${COMP_LEVEL} \
    --num-items ${NUM_ITEMS} \
    --max-rounds ${MAX_ROUNDS} \
    --random-seed ${RANDOM_SEED} \
    --batch \
    --batch-size 5"

echo "[$(date)] Running command: ${CMD}" >> "${LOG_FILE}"

# Execute with timeout (10 minutes per experiment)
# Check if timeout command exists (Linux) or use alternative for macOS
if command -v timeout &> /dev/null; then
    # Linux with timeout command
    if timeout 600 ${CMD} >> "${LOG_FILE}" 2>&1; then
        STATUS="SUCCESS"
        EXIT_CODE=0
    else
        EXIT_CODE=$?
        if [ ${EXIT_CODE} -eq 124 ]; then
            STATUS="TIMEOUT"
        else
            STATUS="FAILED"
        fi
    fi
else
    # macOS without timeout - run directly (no timeout protection)
    # Alternative: Install coreutils via homebrew for gtimeout
    echo "[$(date)] Note: timeout command not available, running without timeout limit" >> "${LOG_FILE}"
    if ${CMD} >> "${LOG_FILE}" 2>&1; then
        STATUS="SUCCESS"
        EXIT_CODE=0
    else
        EXIT_CODE=$?
        STATUS="FAILED"
    fi
fi

# Save status
echo "[$(date)] Experiment ${JOB_ID} completed with status: ${STATUS} (exit code: ${EXIT_CODE})" | tee -a "${LOG_FILE}"

# Create result summary
RESULT_FILE="${FULL_OUTPUT_DIR}/result_${JOB_ID}.json"
cat > "${RESULT_FILE}" << EOF
{
    "job_id": ${JOB_ID},
    "weak_model": "${WEAK_MODEL}",
    "strong_model": "${STRONG_MODEL}",
    "competition_level": ${COMP_LEVEL},
    "status": "${STATUS}",
    "exit_code": ${EXIT_CODE},
    "timestamp": "$(date -Iseconds)",
    "log_file": "${LOG_FILE}",
    "config_file": "${CONFIG_FILE}"
}
EOF

# Mark job as complete
touch "${LOGS_DIR}/completed_${JOB_ID}.flag"

exit ${EXIT_CODE}
#!/bin/bash
# Single experiment runner WITHOUT aggressive timeouts
# This version focuses on handling API rate limits gracefully

set -e

# Get job ID from argument or environment variable
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

# ============================================================================
# KEY CHANGE: Very generous timeout (2 hours) or no timeout at all
# ============================================================================
# The idea is to let the Python script handle retries and rate limiting internally
# rather than killing it from the outside

# Run the actual experiment
cd "${BASE_DIR}"

# Activate virtual environment if it exists
if [ -f ~/.venv/bin/activate ]; then
    source ~/.venv/bin/activate
fi

# Build the command - include job ID for tracking
CMD="python3 run_strong_models_experiment.py \
    --models ${WEAK_MODEL} ${STRONG_MODEL} \
    --competition-level ${COMP_LEVEL} \
    --num-items ${NUM_ITEMS} \
    --max-rounds ${MAX_ROUNDS} \
    --random-seed ${RANDOM_SEED} \
    --batch \
    --batch-size 5 \
    --job-id ${JOB_ID}"

echo "[$(date)] Running command: ${CMD}" >> "${LOG_FILE}"
echo "[$(date)] Note: Using very generous timeout (2 hours) to allow for API rate limiting" >> "${LOG_FILE}"

# Execute with VERY GENEROUS timeout (2 hours = 7200 seconds)
# This should be more than enough for any legitimate experiment
# The Python script should handle its own retries and rate limit backoff
if command -v timeout &> /dev/null; then
    # Linux with timeout command - 2 hour timeout
    if timeout 7200 ${CMD} >> "${LOG_FILE}" 2>&1; then
        STATUS="SUCCESS"
        EXIT_CODE=0
    else
        EXIT_CODE=$?
        if [ ${EXIT_CODE} -eq 124 ]; then
            # If it STILL times out after 2 hours, something is seriously wrong
            STATUS="TIMEOUT_SEVERE"
            echo "[$(date)] SEVERE: Experiment exceeded 2-hour limit - likely stuck or infinite loop" >> "${LOG_FILE}"
        else
            STATUS="FAILED"
        fi
    fi
else
    # macOS or no timeout command - just run without timeout
    echo "[$(date)] Running without timeout limit (timeout command not available)" >> "${LOG_FILE}"
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
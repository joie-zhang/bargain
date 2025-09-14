#!/bin/bash
# Improved single experiment runner with better timeout handling and retry logic
# This version includes model-specific timeouts and better error recovery

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
# IMPROVEMENT 1: Model-specific timeout configuration
# ============================================================================
# Some model pairs are known to be slower, especially at high competition levels
get_timeout_for_models() {
    local weak=$1
    local strong=$2
    local comp=$3
    
    # Default timeout (10 minutes)
    local timeout=600
    
    # Known problematic pairings get extra time
    if [[ "$weak" == "gemini-1-5-pro" ]] || [[ "$weak" == "gpt-4o" ]]; then
        # These weak models are prone to timeouts
        timeout=900  # 15 minutes
    fi
    
    if [[ "$strong" == "claude-3-5-haiku" ]] || [[ "$strong" == "claude-4-sonnet" ]]; then
        # These strong models cause more timeouts
        timeout=$((timeout + 300))  # Add 5 more minutes
    fi
    
    # High competition levels need more negotiation rounds
    if (( $(echo "$comp >= 0.75" | bc -l) )); then
        timeout=$((timeout + 300))  # Add 5 more minutes for high competition
    fi
    
    echo $timeout
}

TIMEOUT_SECONDS=$(get_timeout_for_models "$WEAK_MODEL" "$STRONG_MODEL" "$COMP_LEVEL")
echo "[$(date)] Using timeout of ${TIMEOUT_SECONDS} seconds for this model pairing" >> "${LOG_FILE}"

# ============================================================================
# IMPROVEMENT 2: Retry mechanism with exponential backoff
# ============================================================================
MAX_RETRIES=3
RETRY_COUNT=0
SUCCESS=false

# Run the actual experiment with retries
cd "${BASE_DIR}"

# Activate virtual environment if it exists
if [ -f ~/.venv/bin/activate ]; then
    source ~/.venv/bin/activate
fi

while [ $RETRY_COUNT -lt $MAX_RETRIES ] && [ "$SUCCESS" = false ]; do
    if [ $RETRY_COUNT -gt 0 ]; then
        # Exponential backoff: wait 30s, 60s, 120s between retries
        WAIT_TIME=$((30 * (2 ** ($RETRY_COUNT - 1))))
        echo "[$(date)] Retry $RETRY_COUNT/$MAX_RETRIES - waiting ${WAIT_TIME}s before retry..." | tee -a "${LOG_FILE}"
        sleep $WAIT_TIME
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
    
    echo "[$(date)] Attempt $((RETRY_COUNT + 1))/$MAX_RETRIES - Running command: ${CMD}" >> "${LOG_FILE}"
    
    # ============================================================================
    # IMPROVEMENT 3: Better timeout handling with signal trapping
    # ============================================================================
    # Execute with timeout
    if command -v timeout &> /dev/null; then
        # Linux with timeout command - use SIGTERM first, then SIGKILL
        if timeout --preserve-status --signal=TERM --kill-after=30 ${TIMEOUT_SECONDS} ${CMD} >> "${LOG_FILE}" 2>&1; then
            STATUS="SUCCESS"
            EXIT_CODE=0
            SUCCESS=true
        else
            EXIT_CODE=$?
            if [ ${EXIT_CODE} -eq 124 ]; then
                STATUS="TIMEOUT"
                echo "[$(date)] Experiment timed out after ${TIMEOUT_SECONDS} seconds" >> "${LOG_FILE}"
                
                # Check if this is a known problematic pairing
                if [ $RETRY_COUNT -eq 0 ]; then
                    echo "[$(date)] This model pairing is known to be slow, will retry with extended timeout" >> "${LOG_FILE}"
                    # Increase timeout for retry
                    TIMEOUT_SECONDS=$((TIMEOUT_SECONDS + 600))
                fi
            else
                STATUS="FAILED"
                echo "[$(date)] Experiment failed with exit code ${EXIT_CODE}" >> "${LOG_FILE}"
            fi
        fi
    else
        # macOS without timeout - use background job with timer
        echo "[$(date)] Note: Using custom timeout implementation" >> "${LOG_FILE}"
        
        # Run command in background
        ${CMD} >> "${LOG_FILE}" 2>&1 &
        CMD_PID=$!
        
        # Set up timer
        TIMER=0
        while [ $TIMER -lt $TIMEOUT_SECONDS ]; do
            if ! kill -0 $CMD_PID 2>/dev/null; then
                # Process has finished
                wait $CMD_PID
                EXIT_CODE=$?
                if [ $EXIT_CODE -eq 0 ]; then
                    STATUS="SUCCESS"
                    SUCCESS=true
                else
                    STATUS="FAILED"
                fi
                break
            fi
            sleep 5
            TIMER=$((TIMER + 5))
        done
        
        # Check if we timed out
        if [ $TIMER -ge $TIMEOUT_SECONDS ]; then
            echo "[$(date)] Timeout reached, terminating process..." >> "${LOG_FILE}"
            kill -TERM $CMD_PID 2>/dev/null || true
            sleep 5
            kill -KILL $CMD_PID 2>/dev/null || true
            STATUS="TIMEOUT"
            EXIT_CODE=124
        fi
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

# ============================================================================
# IMPROVEMENT 4: Enhanced result tracking
# ============================================================================
# Save detailed status including retry information
echo "[$(date)] Experiment ${JOB_ID} completed with status: ${STATUS} (exit code: ${EXIT_CODE}, retries: $((RETRY_COUNT - 1)))" | tee -a "${LOG_FILE}"

# Create detailed result summary
RESULT_FILE="${FULL_OUTPUT_DIR}/result_${JOB_ID}.json"
cat > "${RESULT_FILE}" << EOF
{
    "job_id": ${JOB_ID},
    "weak_model": "${WEAK_MODEL}",
    "strong_model": "${STRONG_MODEL}",
    "competition_level": ${COMP_LEVEL},
    "status": "${STATUS}",
    "exit_code": ${EXIT_CODE},
    "retries": $((RETRY_COUNT - 1)),
    "timeout_used": ${TIMEOUT_SECONDS},
    "timestamp": "$(date -Iseconds)",
    "log_file": "${LOG_FILE}",
    "config_file": "${CONFIG_FILE}"
}
EOF

# Mark job as complete only if successful
if [ "$STATUS" = "SUCCESS" ]; then
    touch "${LOGS_DIR}/completed_${JOB_ID}.flag"
else
    # Mark as failed for later retry with different strategy
    touch "${LOGS_DIR}/failed_${JOB_ID}.flag"
    
    # Log to separate timeout tracking file if timeout
    if [ "$STATUS" = "TIMEOUT" ]; then
        echo "${JOB_ID},${WEAK_MODEL},${STRONG_MODEL},${COMP_LEVEL},${TIMEOUT_SECONDS},$((RETRY_COUNT - 1))" >> "${LOGS_DIR}/timeouts.csv"
    fi
fi

exit ${EXIT_CODE}
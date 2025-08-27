#!/bin/bash
# Rate-limit-aware parallel job runner
# This version spaces out job starts to avoid API rate limiting

set -e

# Number of parallel jobs (default 4)
MAX_PARALLEL=${1:-4}

# Delay between job starts (in seconds) to avoid rate limit bursts
JOB_START_DELAY=${JOB_START_DELAY:-2}

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="${BASE_DIR}/scripts"
LOGS_DIR="${BASE_DIR}/experiments/results/scaling_experiment/logs"
CONFIG_DIR="${BASE_DIR}/experiments/results/scaling_experiment/configs"

# Create necessary directories
mkdir -p "${LOGS_DIR}"

# Count total experiments
TOTAL_JOBS=$(ls -1 "${CONFIG_DIR}"/config_*.json 2>/dev/null | wc -l)

if [ ${TOTAL_JOBS} -eq 0 ]; then
    echo "❌ No configuration files found. Run generate_configs.sh first."
    exit 1
fi

echo "============================================================"
echo "Rate-Limited Parallel Execution"
echo "============================================================"
echo "Total experiments: ${TOTAL_JOBS}"
echo "Max parallel jobs: ${MAX_PARALLEL}"
echo "Delay between starts: ${JOB_START_DELAY}s (to avoid rate limit bursts)"
echo "Using 2-hour timeout per experiment (generous for rate limiting)"
echo ""

# Function to check running jobs
count_running_jobs() {
    jobs -r | wc -l
}

# Function to wait for a job slot
wait_for_slot() {
    while [ $(count_running_jobs) -ge ${MAX_PARALLEL} ]; do
        sleep 1
    done
}

# Progress tracking
COMPLETED=0
FAILED=0
START_TIME=$(date +%s)

# Function to update progress
update_progress() {
    local job_id=$1
    local status=$2
    
    COMPLETED=$((COMPLETED + 1))
    if [ "${status}" != "0" ]; then
        FAILED=$((FAILED + 1))
    fi
    
    local CURRENT_TIME=$(date +%s)
    local ELAPSED=$((CURRENT_TIME - START_TIME))
    local RATE=$(echo "scale=2; ${COMPLETED} * 60 / ${ELAPSED}" | bc 2>/dev/null || echo "0")
    local ETA=$((ELAPSED * TOTAL_JOBS / COMPLETED - ELAPSED))
    
    printf "\r[%3d/%3d] Completed: %3d | Failed: %3d | Rate: %.1f/min | ETA: %ds        " \
           ${COMPLETED} ${TOTAL_JOBS} ${COMPLETED} ${FAILED} ${RATE} ${ETA}
}

# Function to run a single job
run_job() {
    local job_id=$1
    
    # Use the no-timeout version if available
    if [ -f "${SCRIPTS_DIR}/run_single_experiment_no_timeout.sh" ]; then
        "${SCRIPTS_DIR}/run_single_experiment_no_timeout.sh" ${job_id}
    else
        # Fall back to original with modified timeout
        TIMEOUT=7200 "${SCRIPTS_DIR}/run_single_experiment.sh" ${job_id}
    fi
    local exit_code=$?
    
    # Update progress
    update_progress ${job_id} ${exit_code}
    
    return ${exit_code}
}

# ============================================================================
# SMART SCHEDULING: Interleave different model providers
# ============================================================================
# Group experiments by API provider to avoid hitting single provider limits

echo "Analyzing experiments by API provider..."

ANTHROPIC_JOBS=()
OPENAI_JOBS=()
GOOGLE_JOBS=()
OTHER_JOBS=()

for job_id in $(seq 0 $((TOTAL_JOBS - 1))); do
    CONFIG_FILE="${CONFIG_DIR}/config_${job_id}.json"
    
    if [ -f "${CONFIG_FILE}" ]; then
        # Check if already completed
        if [ -f "${LOGS_DIR}/completed_${job_id}.flag" ]; then
            echo "Skipping job ${job_id} (already completed)"
            COMPLETED=$((COMPLETED + 1))
            continue
        fi
        
        # Categorize by provider
        MODELS=$(python3 -c "
import json
config = json.load(open('${CONFIG_FILE}'))
weak = config.get('weak_model', '')
strong = config.get('strong_model', '')
print(f'{weak},{strong}')
" 2>/dev/null || echo "unknown,unknown")
        
        if [[ "$MODELS" == *"claude"* ]]; then
            ANTHROPIC_JOBS+=($job_id)
        elif [[ "$MODELS" == *"gpt"* ]] || [[ "$MODELS" == *"o3"* ]]; then
            OPENAI_JOBS+=($job_id)
        elif [[ "$MODELS" == *"gemini"* ]]; then
            GOOGLE_JOBS+=($job_id)
        else
            OTHER_JOBS+=($job_id)
        fi
    fi
done

echo "Jobs by provider:"
echo "  Anthropic (Claude): ${#ANTHROPIC_JOBS[@]}"
echo "  OpenAI (GPT/O3): ${#OPENAI_JOBS[@]}"
echo "  Google (Gemini): ${#GOOGLE_JOBS[@]}"
echo "  Other: ${#OTHER_JOBS[@]}"
echo ""

# ============================================================================
# INTERLEAVED EXECUTION: Rotate between providers
# ============================================================================
echo "Starting interleaved execution (rotating between API providers)..."
echo "Progress:"

# Create interleaved job list
INTERLEAVED_JOBS=()
MAX_PROVIDER_JOBS=$(( ${#ANTHROPIC_JOBS[@]} > ${#OPENAI_JOBS[@]} ? ${#ANTHROPIC_JOBS[@]} : ${#OPENAI_JOBS[@]} ))
MAX_PROVIDER_JOBS=$(( MAX_PROVIDER_JOBS > ${#GOOGLE_JOBS[@]} ? MAX_PROVIDER_JOBS : ${#GOOGLE_JOBS[@]} ))

for i in $(seq 0 $((MAX_PROVIDER_JOBS - 1))); do
    # Add one from each provider in rotation
    if [ $i -lt ${#ANTHROPIC_JOBS[@]} ]; then
        INTERLEAVED_JOBS+=(${ANTHROPIC_JOBS[$i]})
    fi
    if [ $i -lt ${#OPENAI_JOBS[@]} ]; then
        INTERLEAVED_JOBS+=(${OPENAI_JOBS[$i]})
    fi
    if [ $i -lt ${#GOOGLE_JOBS[@]} ]; then
        INTERLEAVED_JOBS+=(${GOOGLE_JOBS[$i]})
    fi
done

# Add any remaining jobs
INTERLEAVED_JOBS+=("${OTHER_JOBS[@]}")

# Execute jobs with rate limiting
for job_id in "${INTERLEAVED_JOBS[@]}"; do
    # Wait for an available slot
    wait_for_slot
    
    # Launch job in background
    {
        run_job ${job_id}
    } &
    
    # ============================================================================
    # KEY CHANGE: Add delay between job starts to avoid rate limit bursts
    # ============================================================================
    # This prevents multiple jobs from hitting the API simultaneously at startup
    sleep ${JOB_START_DELAY}
done

# Wait for all remaining jobs to complete
echo ""
echo "Waiting for all jobs to complete (this may take several hours)..."
wait

# Final statistics
CURRENT_TIME=$(date +%s)
TOTAL_TIME=$((CURRENT_TIME - START_TIME))
echo ""
echo ""
echo "============================================================"
echo "Parallel Execution Complete"
echo "============================================================"
echo "Total experiments: ${TOTAL_JOBS}"
echo "Completed: ${COMPLETED}"
echo "Failed: ${FAILED}"
echo "Success rate: $(echo "scale=1; (${COMPLETED} - ${FAILED}) * 100 / ${COMPLETED}" | bc)%"
echo "Total time: ${TOTAL_TIME} seconds ($(echo "scale=1; ${TOTAL_TIME} / 60" | bc) minutes)"
echo "Average time per experiment: $(echo "scale=1; ${TOTAL_TIME} / ${COMPLETED}" | bc) seconds"
echo ""

# Check for failures
if [ ${FAILED} -gt 0 ]; then
    echo "⚠️  Warning: ${FAILED} experiments failed. Check logs in ${LOGS_DIR}"
    echo ""
    
    # Check for rate limit errors
    RATE_LIMIT_ERRORS=$(grep -l "rate.*limit\|429\|quota" "${LOGS_DIR}"/experiment_*.log 2>/dev/null | wc -l || echo "0")
    
    if [ ${RATE_LIMIT_ERRORS} -gt 0 ]; then
        echo "⚠️  Found ${RATE_LIMIT_ERRORS} experiments with rate limit errors"
        echo ""
        echo "Recommendations:"
        echo "  1. Reduce parallel jobs: --parallel 2"
        echo "  2. Increase delay between starts: export JOB_START_DELAY=5"
        echo "  3. Use provider rotation (already enabled)"
        echo "  4. Wait before retry: sleep 300 && ${SCRIPTS_DIR}/retry_failed.sh"
    fi
fi

echo "✅ All jobs processed"
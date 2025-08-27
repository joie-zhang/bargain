#!/bin/bash
# Run experiments in parallel, simulating a job array

set -e

# Number of parallel jobs (default 4)
MAX_PARALLEL=${1:-4}

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

echo "Running ${TOTAL_JOBS} experiments with ${MAX_PARALLEL} parallel workers"
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
    
    printf "\r[%3d/%3d] Completed: %3d | Failed: %3d | Rate: %.1f/min | Elapsed: %ds" \
           ${COMPLETED} ${TOTAL_JOBS} ${COMPLETED} ${FAILED} ${RATE} ${ELAPSED}
}

# Function to run a single job
run_job() {
    local job_id=$1
    
    # Run the experiment
    "${SCRIPTS_DIR}/run_single_experiment.sh" ${job_id}
    local exit_code=$?
    
    # Update progress
    update_progress ${job_id} ${exit_code}
    
    return ${exit_code}
}

# Main execution loop
echo "Starting parallel execution..."
echo "Progress:"

for job_id in $(seq 0 $((TOTAL_JOBS - 1))); do
    # Check if already completed (for resume capability)
    if [ -f "${LOGS_DIR}/completed_${job_id}.flag" ]; then
        echo "Skipping job ${job_id} (already completed)"
        COMPLETED=$((COMPLETED + 1))
        continue
    fi
    
    # Wait for an available slot
    wait_for_slot
    
    # Launch job in background
    {
        run_job ${job_id}
    } &
    
    # Small delay to avoid overwhelming the system
    sleep 0.1
done

# Wait for all remaining jobs to complete
echo ""
echo "Waiting for final jobs to complete..."
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
    echo "Failed experiments:"
    for job_id in $(seq 0 $((TOTAL_JOBS - 1))); do
        if [ -f "${LOGS_DIR}/experiment_${job_id}.log" ]; then
            if grep -q "status: FAILED\|status: TIMEOUT" "${LOGS_DIR}/experiment_${job_id}.log"; then
                CONFIG="${CONFIG_DIR}/config_${job_id}.json"
                if [ -f "${CONFIG}" ]; then
                    WEAK=$(python3 -c "import json; print(json.load(open('${CONFIG}'))['weak_model'])")
                    STRONG=$(python3 -c "import json; print(json.load(open('${CONFIG}'))['strong_model'])")
                    COMP=$(python3 -c "import json; print(json.load(open('${CONFIG}'))['competition_level'])")
                    echo "  Job ${job_id}: ${WEAK} vs ${STRONG} (competition=${COMP})"
                fi
            fi
        fi
    done
    echo ""
    echo "To retry failed experiments, run:"
    echo "  ${SCRIPTS_DIR}/retry_failed.sh"
fi

echo "✅ All jobs processed"
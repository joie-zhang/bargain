#!/bin/bash
# Simplest possible runner - just run everything, no timeouts
set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${BASE_DIR}/experiments/results/scaling_experiment"
SCRIPTS_DIR="${BASE_DIR}/scripts"
LOGS_DIR="${RESULTS_DIR}/logs"
CONFIG_DIR="${RESULTS_DIR}/configs"

# Configuration
MAX_PARALLEL_JOBS=${1:-4}

# Create directories
mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}" "${CONFIG_DIR}"

echo "============================================================"
echo "SIMPLE EXPERIMENT RUNNER (No Timeouts)"
echo "============================================================"
echo "Max parallel jobs: ${MAX_PARALLEL_JOBS}"
echo "Strategy: Just run them all, let Python handle retries"
echo ""

# Step 1: Generate configs if needed
if [ ! -f "${CONFIG_DIR}/config_0.json" ]; then
    echo "Generating experiment configurations..."
    "${SCRIPTS_DIR}/generate_configs.sh"
fi

# Step 2: Count experiments
TOTAL_JOBS=$(ls -1 "${CONFIG_DIR}"/config_*.json 2>/dev/null | wc -l)
echo "Total experiments: ${TOTAL_JOBS}"
echo ""

# Step 3: Run them all
echo "Starting parallel execution with ${MAX_PARALLEL_JOBS} workers..."
START_TIME=$(date +%s)

# Function to run jobs
run_job() {
    local job_id=$1
    
    # Skip if already done
    if [ -f "${LOGS_DIR}/completed_${job_id}.flag" ]; then
        return 0
    fi
    
    # Run it - NO TIMEOUT
    "${SCRIPTS_DIR}/run_single_experiment_simple.sh" ${job_id}
}

export -f run_job
export LOGS_DIR SCRIPTS_DIR

# Run all jobs in parallel with xargs (simple and efficient)
seq 0 $((TOTAL_JOBS - 1)) | xargs -P ${MAX_PARALLEL_JOBS} -I {} bash -c 'run_job {}'

# Step 4: Summary
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "============================================================"
echo "COMPLETE"
echo "============================================================"

# Count results
COMPLETED=$(ls -1 "${LOGS_DIR}"/completed_*.flag 2>/dev/null | wc -l || echo "0")
FAILED=$(grep -l "FAILED" "${LOGS_DIR}"/experiment_*.log 2>/dev/null | wc -l || echo "0")

echo "Total time: $((ELAPSED / 60)) minutes"
echo "Completed: ${COMPLETED}/${TOTAL_JOBS}"
echo "Failed: ${FAILED}"

if [ ${FAILED} -gt 0 ]; then
    echo ""
    echo "To retry failed experiments:"
    echo "  ${SCRIPTS_DIR}/run_all_simple.sh ${MAX_PARALLEL_JOBS}"
    echo "(It will skip completed experiments automatically)"
fi

# Step 5: Collect results
"${SCRIPTS_DIR}/collect_results.sh"

echo "✅ Done"
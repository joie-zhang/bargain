#!/bin/bash
# Run experiments with selection options (first order, second order, or both)
set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${BASE_DIR}/experiments/results/scaling_experiment"
SCRIPTS_DIR="${BASE_DIR}/scripts"
LOGS_DIR="${RESULTS_DIR}/logs"
CONFIG_DIR="${RESULTS_DIR}/configs"

# Parse arguments
MAX_PARALLEL=${1:-4}
ORDER_TYPE=${2:-"both"}  # "weak_first", "strong_first", or "both"

# Create directories
mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}" "${CONFIG_DIR}"

echo "============================================================"
echo "SELECTIVE EXPERIMENT RUNNER"
echo "============================================================"
echo "Max parallel jobs: ${MAX_PARALLEL}"
echo "Order type: ${ORDER_TYPE}"
echo ""

# Step 1: Generate configs if needed
if [ ! -f "${CONFIG_DIR}/config_0.json" ]; then
    echo "Generating experiment configurations with both orderings..."
    "${SCRIPTS_DIR}/generate_configs_both_orders.sh"
fi

# Step 2: Determine which experiments to run
case ${ORDER_TYPE} in
    weak_first)
        echo "Running experiments with WEAK model first (IDs 0-299)"
        START_ID=0
        END_ID=299
        ;;
    strong_first)
        echo "Running experiments with STRONG model first (IDs 300-599)"
        START_ID=300
        END_ID=599
        ;;
    both)
        echo "Running ALL experiments (both orderings, IDs 0-599)"
        START_ID=0
        END_ID=599
        ;;
    *)
        echo "Invalid order type. Use: weak_first, strong_first, or both"
        exit 1
        ;;
esac

# Count experiments to run
JOBS_TO_RUN=()
for i in $(seq ${START_ID} ${END_ID}); do
    if [ -f "${CONFIG_DIR}/config_${i}.json" ] && [ ! -f "${LOGS_DIR}/completed_${i}.flag" ]; then
        JOBS_TO_RUN+=($i)
    fi
done

TOTAL_PENDING=${#JOBS_TO_RUN[@]}
echo "Experiments to run: ${TOTAL_PENDING}"
echo ""

if [ ${TOTAL_PENDING} -eq 0 ]; then
    echo "✅ All selected experiments already completed!"
    exit 0
fi

# Step 3: Run experiments
echo "Starting parallel execution..."
START_TIME=$(date +%s)

# Function to run single job
run_job() {
    local job_id=$1
    
    # Run it - NO TIMEOUT
    "${SCRIPTS_DIR}/run_single_experiment_simple.sh" ${job_id}
}

export -f run_job
export LOGS_DIR SCRIPTS_DIR

# Run jobs in parallel with xargs
printf "%s\n" "${JOBS_TO_RUN[@]}" | xargs -P ${MAX_PARALLEL} -I {} bash -c 'run_job {}'

# Step 4: Summary
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "============================================================"
echo "EXECUTION COMPLETE"
echo "============================================================"
echo "Total time: $((ELAPSED / 60)) minutes"

# Count results for selected range
COMPLETED=0
FAILED=0
for i in $(seq ${START_ID} ${END_ID}); do
    if [ -f "${LOGS_DIR}/completed_${i}.flag" ]; then
        COMPLETED=$((COMPLETED + 1))
    elif [ -f "${LOGS_DIR}/experiment_${i}.log" ]; then
        if grep -q "FAILED" "${LOGS_DIR}/experiment_${i}.log"; then
            FAILED=$((FAILED + 1))
        fi
    fi
done

echo "Selected range: ${START_ID}-${END_ID}"
echo "Completed: ${COMPLETED}/$((END_ID - START_ID + 1))"
echo "Failed: ${FAILED}"

if [ ${FAILED} -gt 0 ]; then
    echo ""
    echo "To retry failed experiments:"
    echo "  ${SCRIPTS_DIR}/run_experiments_selective.sh ${MAX_PARALLEL} ${ORDER_TYPE}"
fi

# Collect results
"${SCRIPTS_DIR}/collect_results.sh"

echo "✅ Done"
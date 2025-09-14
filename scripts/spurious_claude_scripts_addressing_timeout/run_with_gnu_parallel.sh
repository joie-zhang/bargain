#!/bin/bash
# Ultra-efficient runner using GNU parallel if available
set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${BASE_DIR}/experiments/results/scaling_experiment"
SCRIPTS_DIR="${BASE_DIR}/scripts"
LOGS_DIR="${RESULTS_DIR}/logs"
CONFIG_DIR="${RESULTS_DIR}/configs"

# Configuration
MAX_PARALLEL=${1:-4}
DELAY=${2:-1}  # Delay between job starts

# Create directories
mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}" "${CONFIG_DIR}"

echo "============================================================"
echo "GNU PARALLEL EXPERIMENT RUNNER (Most Efficient)"
echo "============================================================"

# Check for GNU parallel
if ! command -v parallel &> /dev/null; then
    echo "GNU parallel not found. Install it for best performance:"
    echo "  Ubuntu/Debian: sudo apt-get install parallel"
    echo "  Mac: brew install parallel"
    echo ""
    echo "Falling back to simple runner..."
    exec "${SCRIPTS_DIR}/run_all_simple.sh" ${MAX_PARALLEL}
fi

echo "Using GNU parallel for maximum efficiency"
echo "Max parallel jobs: ${MAX_PARALLEL}"
echo "Delay between starts: ${DELAY}s"
echo ""

# Generate configs if needed
if [ ! -f "${CONFIG_DIR}/config_0.json" ]; then
    echo "Generating experiment configurations..."
    "${SCRIPTS_DIR}/generate_configs.sh"
fi

# Get list of experiments to run
JOBS_TO_RUN=()
for i in $(seq 0 299); do
    if [ ! -f "${LOGS_DIR}/completed_${i}.flag" ]; then
        JOBS_TO_RUN+=($i)
    fi
done

TOTAL_PENDING=${#JOBS_TO_RUN[@]}
echo "Experiments to run: ${TOTAL_PENDING}"

if [ ${TOTAL_PENDING} -eq 0 ]; then
    echo "✅ All experiments already completed!"
    exit 0
fi

echo ""
echo "Starting GNU parallel execution..."
START_TIME=$(date +%s)

# Run with GNU parallel
# --delay: Add delay between job starts to avoid API bursts
# --progress: Show progress bar
# --joblog: Keep log of jobs
# --retry-failed: Retry failed jobs
# --halt soon,fail=20%: Stop if 20% of jobs fail
printf "%s\n" "${JOBS_TO_RUN[@]}" | \
    parallel --delay ${DELAY} \
             --jobs ${MAX_PARALLEL} \
             --progress \
             --joblog "${LOGS_DIR}/parallel.log" \
             --retry-failed \
             --halt soon,fail=20% \
             "${SCRIPTS_DIR}/run_single_experiment_simple.sh {}"

# Summary
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "============================================================"
echo "EXECUTION COMPLETE"
echo "============================================================"
echo "Total time: $((ELAPSED / 60)) minutes ($((ELAPSED / 3600)) hours)"

# Analyze results
COMPLETED=$(ls -1 "${LOGS_DIR}"/completed_*.flag 2>/dev/null | wc -l)
echo "Completed: ${COMPLETED}/300"

# Check for failures
if [ -f "${LOGS_DIR}/parallel.log" ]; then
    FAILED=$(awk '$7!=0' "${LOGS_DIR}/parallel.log" | wc -l)
    if [ ${FAILED} -gt 0 ]; then
        echo "Failed: ${FAILED}"
        echo ""
        echo "Failed jobs (from parallel.log):"
        awk '$7!=0 {print "  Job " $NF ": exit code " $7}' "${LOGS_DIR}/parallel.log" | head -10
        
        echo ""
        echo "To retry only failed jobs:"
        echo "  parallel --retry-failed --joblog ${LOGS_DIR}/parallel.log"
    fi
fi

# Collect results
"${SCRIPTS_DIR}/collect_results.sh"

echo "✅ Done"
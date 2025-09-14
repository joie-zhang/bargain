#!/bin/bash
# Improved retry script with special handling for timeout-prone experiments

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="${BASE_DIR}/scripts"
LOGS_DIR="${BASE_DIR}/experiments/results/scaling_experiment/logs"
CONFIG_DIR="${BASE_DIR}/experiments/results/scaling_experiment/configs"

echo "Analyzing failed experiments..."

# Separate timeouts from other failures
TIMEOUT_JOBS=()
OTHER_FAILED_JOBS=()

for log_file in "${LOGS_DIR}"/experiment_*.log; do
    if [ -f "${log_file}" ]; then
        JOB_ID=$(basename "${log_file}" | sed 's/experiment_\(.*\)\.log/\1/')
        
        # Check failure type
        if grep -q "status: TIMEOUT" "${log_file}"; then
            TIMEOUT_JOBS+=("${JOB_ID}")
        elif grep -q "status: FAILED" "${log_file}"; then
            OTHER_FAILED_JOBS+=("${JOB_ID}")
        fi
    fi
done

TOTAL_FAILED=$((${#TIMEOUT_JOBS[@]} + ${#OTHER_FAILED_JOBS[@]}))

if [ ${TOTAL_FAILED} -eq 0 ]; then
    echo "✅ No failed experiments found!"
    exit 0
fi

echo ""
echo "Failed experiment breakdown:"
echo "  Timeouts: ${#TIMEOUT_JOBS[@]}"
echo "  Other failures: ${#OTHER_FAILED_JOBS[@]}"
echo "  Total: ${TOTAL_FAILED}"
echo ""

# ============================================================================
# STRATEGY 1: For timeout experiments, consider alternative approaches
# ============================================================================
if [ ${#TIMEOUT_JOBS[@]} -gt 0 ]; then
    echo "Timeout experiments analysis:"
    echo "=============================="
    
    # Analyze timeout patterns
    python3 << EOF
import json
import os

timeout_jobs = [${TIMEOUT_JOBS[@]}]
config_dir = "${CONFIG_DIR}"

# Count model combinations
model_pairs = {}
for job_id in timeout_jobs:
    config_file = f"{config_dir}/config_{job_id}.json"
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
            pair = f"{config['weak_model']} vs {config['strong_model']}"
            comp = config['competition_level']
            if pair not in model_pairs:
                model_pairs[pair] = []
            model_pairs[pair].append(comp)

print("Model pairs with timeouts:")
for pair, comps in sorted(model_pairs.items()):
    print(f"  {pair}: {len(comps)} timeouts at competition levels {sorted(set(comps))}")
EOF
    
    echo ""
    echo "Suggested actions for timeout experiments:"
    echo "1. Run with extended timeout (20-30 minutes)"
    echo "2. Run sequentially (not in parallel)"
    echo "3. Skip highest competition levels for problematic pairs"
    echo ""
fi

# Ask for retry strategy
echo "Select retry strategy:"
echo "1. Retry all with extended timeouts (recommended for timeouts)"
echo "2. Retry only non-timeout failures"
echo "3. Manual selection"
echo "4. Cancel"
read -p "Choice (1-4): " CHOICE

case $CHOICE in
    1)
        RETRY_JOBS=("${TIMEOUT_JOBS[@]}" "${OTHER_FAILED_JOBS[@]}")
        USE_EXTENDED_TIMEOUT=true
        ;;
    2)
        RETRY_JOBS=("${OTHER_FAILED_JOBS[@]}")
        USE_EXTENDED_TIMEOUT=false
        ;;
    3)
        # Manual selection
        echo "Enter job IDs to retry (space-separated):"
        read -a RETRY_JOBS
        USE_EXTENDED_TIMEOUT=true
        ;;
    4)
        echo "Cancelled"
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

if [ ${#RETRY_JOBS[@]} -eq 0 ]; then
    echo "No jobs selected for retry"
    exit 0
fi

echo ""
echo "Preparing to retry ${#RETRY_JOBS[@]} experiments..."

# Remove completion flags for selected jobs
for job_id in "${RETRY_JOBS[@]}"; do
    rm -f "${LOGS_DIR}/completed_${job_id}.flag"
    rm -f "${LOGS_DIR}/failed_${job_id}.flag"
    
    # Archive old logs
    if [ -f "${LOGS_DIR}/experiment_${job_id}.log" ]; then
        mv "${LOGS_DIR}/experiment_${job_id}.log" "${LOGS_DIR}/experiment_${job_id}.log.retry_$(date +%s)"
    fi
done

# ============================================================================
# STRATEGY 2: Use different parallelism based on job type
# ============================================================================
MAX_PARALLEL=${1:-2}  # Default to 2 for retries (more conservative)

echo ""
echo "Retrying ${#RETRY_JOBS[@]} experiments with max parallelism of ${MAX_PARALLEL}..."

# Sort jobs: non-timeouts first, then timeouts
NON_TIMEOUT_RETRY=()
TIMEOUT_RETRY=()

for job_id in "${RETRY_JOBS[@]}"; do
    if [[ " ${TIMEOUT_JOBS[@]} " =~ " ${job_id} " ]]; then
        TIMEOUT_RETRY+=("${job_id}")
    else
        NON_TIMEOUT_RETRY+=("${job_id}")
    fi
done

# Function to run job with extended timeout if needed
run_retry_job() {
    local job_id=$1
    local extended=$2
    
    if [ "$extended" = true ] && [ -f "${SCRIPTS_DIR}/run_single_experiment_improved.sh" ]; then
        # Use improved script with extended timeout
        TIMEOUT_OVERRIDE=1800 "${SCRIPTS_DIR}/run_single_experiment_improved.sh" ${job_id}
    elif [ -f "${SCRIPTS_DIR}/run_single_experiment_improved.sh" ]; then
        "${SCRIPTS_DIR}/run_single_experiment_improved.sh" ${job_id}
    else
        "${SCRIPTS_DIR}/run_single_experiment.sh" ${job_id}
    fi
}

# Retry non-timeout failures first with normal parallelism
if [ ${#NON_TIMEOUT_RETRY[@]} -gt 0 ]; then
    echo ""
    echo "Phase 1: Retrying non-timeout failures..."
    
    for job_id in "${NON_TIMEOUT_RETRY[@]}"; do
        echo "  Retrying job ${job_id}..."
        run_retry_job ${job_id} false &
        
        # Limit parallel jobs
        while [ $(jobs -r | wc -l) -ge ${MAX_PARALLEL} ]; do
            sleep 1
        done
    done
    
    # Wait for these to complete
    wait
fi

# Retry timeout experiments with reduced parallelism and extended timeout
if [ ${#TIMEOUT_RETRY[@]} -gt 0 ]; then
    echo ""
    echo "Phase 2: Retrying timeout experiments with extended timeout (30 min each)..."
    
    # Use only 1 parallel job for timeout retries to maximize success
    for job_id in "${TIMEOUT_RETRY[@]}"; do
        CONFIG_FILE="${CONFIG_DIR}/config_${job_id}.json"
        if [ -f "${CONFIG_FILE}" ]; then
            WEAK=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['weak_model'])")
            STRONG=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['strong_model'])")
            COMP=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['competition_level'])")
            echo "  Retrying job ${job_id}: ${WEAK} vs ${STRONG} (competition=${COMP})..."
        else
            echo "  Retrying job ${job_id}..."
        fi
        
        run_retry_job ${job_id} true &
        
        # Run timeout experiments one at a time
        wait
    done
fi

echo ""
echo "✅ Retry complete"
echo ""

# Check results
NEW_FAILURES=0
NEW_TIMEOUTS=0

for job_id in "${RETRY_JOBS[@]}"; do
    if [ -f "${LOGS_DIR}/experiment_${job_id}.log" ]; then
        if grep -q "status: SUCCESS" "${LOGS_DIR}/experiment_${job_id}.log"; then
            :  # Success
        elif grep -q "status: TIMEOUT" "${LOGS_DIR}/experiment_${job_id}.log"; then
            NEW_TIMEOUTS=$((NEW_TIMEOUTS + 1))
        else
            NEW_FAILURES=$((NEW_FAILURES + 1))
        fi
    fi
done

echo "Retry results:"
echo "  Successful: $((${#RETRY_JOBS[@]} - NEW_FAILURES - NEW_TIMEOUTS))"
echo "  Still timing out: ${NEW_TIMEOUTS}"
echo "  Other failures: ${NEW_FAILURES}"

if [ $((NEW_TIMEOUTS + NEW_FAILURES)) -gt 0 ]; then
    echo ""
    echo "⚠️  Some experiments still failing. Consider:"
    echo "  1. Running problem pairs individually with very long timeout"
    echo "  2. Skipping these specific model/competition combinations"
    echo "  3. Checking API rate limits and keys"
fi

echo ""
echo "Run collect_results.sh to update the summary:"
echo "  ${SCRIPTS_DIR}/collect_results.sh"
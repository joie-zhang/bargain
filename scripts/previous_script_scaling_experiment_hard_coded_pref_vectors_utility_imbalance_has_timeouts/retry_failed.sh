#!/bin/bash
# Retry failed experiments

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="${BASE_DIR}/scripts"
LOGS_DIR="${BASE_DIR}/experiments/results/scaling_experiment/logs"
CONFIG_DIR="${BASE_DIR}/experiments/results/scaling_experiment/configs"

echo "Identifying failed experiments..."

# Find failed experiments
FAILED_JOBS=()

for log_file in "${LOGS_DIR}"/experiment_*.log; do
    if [ -f "${log_file}" ]; then
        JOB_ID=$(basename "${log_file}" | sed 's/experiment_\(.*\)\.log/\1/')
        
        # Check if experiment failed or timed out
        if grep -q "status: FAILED\|status: TIMEOUT" "${log_file}"; then
            FAILED_JOBS+=("${JOB_ID}")
        fi
    fi
done

if [ ${#FAILED_JOBS[@]} -eq 0 ]; then
    echo "✅ No failed experiments found!"
    exit 0
fi

echo "Found ${#FAILED_JOBS[@]} failed experiments"
echo ""

# Ask for confirmation
read -p "Retry all failed experiments? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 0
fi

# Remove completion flags for failed jobs
for job_id in "${FAILED_JOBS[@]}"; do
    rm -f "${LOGS_DIR}/completed_${job_id}.flag"
    
    # Archive old log
    if [ -f "${LOGS_DIR}/experiment_${job_id}.log" ]; then
        mv "${LOGS_DIR}/experiment_${job_id}.log" "${LOGS_DIR}/experiment_${job_id}.log.old"
    fi
done

# Create a temporary job list file
RETRY_LIST="${LOGS_DIR}/retry_jobs.txt"
printf "%s\n" "${FAILED_JOBS[@]}" > "${RETRY_LIST}"

echo ""
echo "Retrying ${#FAILED_JOBS[@]} experiments..."

# Run failed jobs with parallel execution
MAX_PARALLEL=${1:-4}

for job_id in "${FAILED_JOBS[@]}"; do
    echo "Retrying job ${job_id}..."
    "${SCRIPTS_DIR}/run_single_experiment.sh" ${job_id} &
    
    # Limit parallel jobs
    while [ $(jobs -r | wc -l) -ge ${MAX_PARALLEL} ]; do
        sleep 1
    done
done

# Wait for all jobs to complete
wait

echo ""
echo "✅ Retry complete"
echo ""
echo "Run collect_results.sh to update the summary:"
echo "  ${SCRIPTS_DIR}/collect_results.sh"
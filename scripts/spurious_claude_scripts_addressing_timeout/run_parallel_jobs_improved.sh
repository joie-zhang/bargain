#!/bin/bash
# Improved parallel job runner with smart scheduling for timeout-prone model pairs
# This version prioritizes easy experiments first and handles problematic pairs separately

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

# ============================================================================
# IMPROVEMENT 1: Categorize experiments by difficulty
# ============================================================================
echo "Analyzing experiment difficulty..."

# Arrays to hold job IDs by category
EASY_JOBS=()
MODERATE_JOBS=()
DIFFICULT_JOBS=()

# Known problematic model combinations (from analysis)
PROBLEMATIC_WEAK_MODELS=("gemini-1-5-pro" "gpt-4o")
PROBLEMATIC_STRONG_MODELS=("claude-3-5-haiku" "claude-4-sonnet" "claude-3-5-sonnet" "gemini-2-5-pro" "gemini-2-5-flash")

for job_id in $(seq 0 $((TOTAL_JOBS - 1))); do
    CONFIG_FILE="${CONFIG_DIR}/config_${job_id}.json"
    
    if [ -f "${CONFIG_FILE}" ]; then
        WEAK_MODEL=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['weak_model'])" 2>/dev/null || echo "unknown")
        STRONG_MODEL=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['strong_model'])" 2>/dev/null || echo "unknown")
        COMP_LEVEL=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['competition_level'])" 2>/dev/null || echo "0.5")
        
        # Categorize based on known timeout patterns
        IS_PROBLEMATIC=false
        
        # Check if weak model is problematic
        for prob_model in "${PROBLEMATIC_WEAK_MODELS[@]}"; do
            if [[ "$WEAK_MODEL" == "$prob_model" ]]; then
                IS_PROBLEMATIC=true
                break
            fi
        done
        
        # Check if strong model is problematic
        for prob_model in "${PROBLEMATIC_STRONG_MODELS[@]}"; do
            if [[ "$STRONG_MODEL" == "$prob_model" ]]; then
                IS_PROBLEMATIC=true
                break
            fi
        done
        
        # Categorize based on difficulty
        if [ "$IS_PROBLEMATIC" = true ] && (( $(echo "$COMP_LEVEL >= 0.75" | bc -l) )); then
            # High competition + problematic models = difficult
            DIFFICULT_JOBS+=($job_id)
        elif [ "$IS_PROBLEMATIC" = true ] || (( $(echo "$COMP_LEVEL >= 0.75" | bc -l) )); then
            # Either problematic OR high competition = moderate
            MODERATE_JOBS+=($job_id)
        else
            # Everything else is easy
            EASY_JOBS+=($job_id)
        fi
    fi
done

echo "Experiment difficulty breakdown:"
echo "  Easy: ${#EASY_JOBS[@]} experiments"
echo "  Moderate: ${#MODERATE_JOBS[@]} experiments"
echo "  Difficult: ${#DIFFICULT_JOBS[@]} experiments (timeout-prone)"
echo ""

# ============================================================================
# IMPROVEMENT 2: Smart scheduling - run easy jobs first
# ============================================================================
# Combine all jobs in order: easy first, then moderate, then difficult
ALL_JOBS=("${EASY_JOBS[@]}" "${MODERATE_JOBS[@]}" "${DIFFICULT_JOBS[@]}")

# Function to check running jobs
count_running_jobs() {
    jobs -r | wc -l
}

# Function to wait for a job slot
wait_for_slot() {
    local max_slots=$1
    while [ $(count_running_jobs) -ge ${max_slots} ]; do
        sleep 1
    done
}

# Progress tracking
COMPLETED=0
FAILED=0
TIMEOUTS=0
START_TIME=$(date +%s)

# Function to update progress
update_progress() {
    local job_id=$1
    local status=$2
    local category=$3
    
    COMPLETED=$((COMPLETED + 1))
    if [ "${status}" != "0" ]; then
        FAILED=$((FAILED + 1))
        # Check if it was a timeout
        if grep -q "TIMEOUT" "${LOGS_DIR}/experiment_${job_id}.log" 2>/dev/null; then
            TIMEOUTS=$((TIMEOUTS + 1))
        fi
    fi
    
    local CURRENT_TIME=$(date +%s)
    local ELAPSED=$((CURRENT_TIME - START_TIME))
    local RATE=$(echo "scale=2; ${COMPLETED} * 60 / ${ELAPSED}" | bc 2>/dev/null || echo "0")
    local ETA=$((ELAPSED * TOTAL_JOBS / COMPLETED - ELAPSED))
    
    printf "\r[%3d/%3d] Done: %3d | Failed: %3d | Timeouts: %3d | Rate: %.1f/min | ETA: %ds | Current: %s    " \
           ${COMPLETED} ${TOTAL_JOBS} ${COMPLETED} ${FAILED} ${TIMEOUTS} ${RATE} ${ETA} "${category}"
}

# Function to run a single job
run_job() {
    local job_id=$1
    local category=$2
    
    # Use improved script if available, otherwise fall back to original
    if [ -f "${SCRIPTS_DIR}/run_single_experiment_improved.sh" ]; then
        "${SCRIPTS_DIR}/run_single_experiment_improved.sh" ${job_id}
    else
        "${SCRIPTS_DIR}/run_single_experiment.sh" ${job_id}
    fi
    local exit_code=$?
    
    # Update progress
    update_progress ${job_id} ${exit_code} "${category}"
    
    return ${exit_code}
}

# ============================================================================
# IMPROVEMENT 3: Adaptive parallelism
# ============================================================================
echo "Starting parallel execution with adaptive scheduling..."
echo "Progress:"

# Process easy jobs with full parallelism
echo ""
echo "Phase 1: Processing easy experiments (${#EASY_JOBS[@]} jobs)..."
CURRENT_MAX_PARALLEL=${MAX_PARALLEL}

for job_id in "${EASY_JOBS[@]}"; do
    # Check if already completed (for resume capability)
    if [ -f "${LOGS_DIR}/completed_${job_id}.flag" ]; then
        COMPLETED=$((COMPLETED + 1))
        continue
    fi
    
    # Wait for an available slot
    wait_for_slot ${CURRENT_MAX_PARALLEL}
    
    # Launch job in background
    {
        run_job ${job_id} "Easy"
    } &
    
    sleep 0.1
done

# Wait for easy jobs to complete
wait

# Process moderate jobs with full parallelism
echo ""
echo "Phase 2: Processing moderate experiments (${#MODERATE_JOBS[@]} jobs)..."

for job_id in "${MODERATE_JOBS[@]}"; do
    if [ -f "${LOGS_DIR}/completed_${job_id}.flag" ]; then
        COMPLETED=$((COMPLETED + 1))
        continue
    fi
    
    wait_for_slot ${CURRENT_MAX_PARALLEL}
    
    {
        run_job ${job_id} "Moderate"
    } &
    
    sleep 0.2  # Slightly longer delay for moderate jobs
done

wait

# ============================================================================
# IMPROVEMENT 4: Handle difficult jobs with reduced parallelism
# ============================================================================
# Reduce parallelism for difficult jobs to avoid overwhelming the system
echo ""
echo "Phase 3: Processing difficult experiments (${#DIFFICULT_JOBS[@]} jobs) with reduced parallelism..."
REDUCED_PARALLEL=$((MAX_PARALLEL / 2))
if [ ${REDUCED_PARALLEL} -lt 1 ]; then
    REDUCED_PARALLEL=1
fi

for job_id in "${DIFFICULT_JOBS[@]}"; do
    if [ -f "${LOGS_DIR}/completed_${job_id}.flag" ]; then
        COMPLETED=$((COMPLETED + 1))
        continue
    fi
    
    wait_for_slot ${REDUCED_PARALLEL}
    
    {
        run_job ${job_id} "Difficult"
    } &
    
    sleep 0.5  # Longer delay for difficult jobs
done

# Wait for all remaining jobs to complete
echo ""
echo "Waiting for final jobs to complete..."
wait

# ============================================================================
# IMPROVEMENT 5: Enhanced reporting
# ============================================================================
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
echo "Timeouts: ${TIMEOUTS}"
echo "Success rate: $(echo "scale=1; (${COMPLETED} - ${FAILED}) * 100 / ${COMPLETED}" | bc)%"
echo "Total time: ${TOTAL_TIME} seconds ($(echo "scale=1; ${TOTAL_TIME} / 60" | bc) minutes)"
echo "Average time per experiment: $(echo "scale=1; ${TOTAL_TIME} / ${COMPLETED}" | bc) seconds"
echo ""

# Generate timeout analysis if any timeouts occurred
if [ ${TIMEOUTS} -gt 0 ]; then
    echo "⚠️  Timeout Analysis:"
    echo "===================="
    
    if [ -f "${LOGS_DIR}/timeouts.csv" ]; then
        echo "Timeout summary saved to: ${LOGS_DIR}/timeouts.csv"
        
        # Show most problematic pairings
        echo ""
        echo "Most timeout-prone model pairings:"
        cut -d',' -f2,3 "${LOGS_DIR}/timeouts.csv" | sort | uniq -c | sort -rn | head -5 | while read count weak strong; do
            echo "  ${weak} vs ${strong}: ${count} timeouts"
        done
    fi
    echo ""
fi

# Check for failures
if [ ${FAILED} -gt 0 ]; then
    echo "⚠️  Warning: ${FAILED} experiments failed. Check logs in ${LOGS_DIR}"
    echo ""
    
    # Separate timeouts from other failures
    echo "Failed experiments by type:"
    TIMEOUT_COUNT=0
    OTHER_FAILURES=0
    
    for job_id in $(seq 0 $((TOTAL_JOBS - 1))); do
        if [ -f "${LOGS_DIR}/experiment_${job_id}.log" ]; then
            if grep -q "status: TIMEOUT" "${LOGS_DIR}/experiment_${job_id}.log" 2>/dev/null; then
                TIMEOUT_COUNT=$((TIMEOUT_COUNT + 1))
            elif grep -q "status: FAILED" "${LOGS_DIR}/experiment_${job_id}.log" 2>/dev/null; then
                OTHER_FAILURES=$((OTHER_FAILURES + 1))
            fi
        fi
    done
    
    echo "  Timeouts: ${TIMEOUT_COUNT}"
    echo "  Other failures: ${OTHER_FAILURES}"
    echo ""
    echo "To retry failed experiments with extended timeouts, run:"
    echo "  ${SCRIPTS_DIR}/retry_failed_improved.sh"
fi

echo "✅ All jobs processed"
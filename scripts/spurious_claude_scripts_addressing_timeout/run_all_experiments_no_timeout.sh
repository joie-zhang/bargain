#!/bin/bash
# Main orchestrator WITHOUT aggressive timeouts
# This version focuses on handling API rate limits rather than killing processes

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

# Base directory
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${BASE_DIR}/experiments/results/scaling_experiment"
SCRIPTS_DIR="${BASE_DIR}/scripts"
LOGS_DIR="${RESULTS_DIR}/logs"

# Create necessary directories
mkdir -p "${RESULTS_DIR}"
mkdir -p "${LOGS_DIR}"
mkdir -p "${RESULTS_DIR}/configs"

# Number of parallel jobs (reduced default for rate limiting)
MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS:-3}

# Delay between job starts (seconds) - prevents rate limit bursts
JOB_START_DELAY=${JOB_START_DELAY:-3}

# Total number of experiments
TOTAL_EXPERIMENTS=300

# ============================================================================
# FUNCTIONS
# ============================================================================

print_header() {
    echo ""
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

check_requirements() {
    print_header "Checking Requirements"
    
    # Make scripts executable
    chmod +x "${SCRIPTS_DIR}"/*.sh 2>/dev/null || true
    
    # Activate virtual environment if it exists
    if [ -f ~/.venv/bin/activate ]; then
        source ~/.venv/bin/activate
        echo "✅ Virtual environment activated"
    fi
    
    # Check for Python
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python3 is required but not installed."
        exit 1
    fi
    
    # Check for API keys
    API_KEYS_FOUND=0
    if [ -n "$OPENROUTER_API_KEY" ]; then
        echo "✅ OpenRouter API key found"
        API_KEYS_FOUND=$((API_KEYS_FOUND + 1))
    fi
    if [ -n "$ANTHROPIC_API_KEY" ]; then
        echo "✅ Anthropic API key found"
        API_KEYS_FOUND=$((API_KEYS_FOUND + 1))
    fi
    if [ -n "$OPENAI_API_KEY" ]; then
        echo "✅ OpenAI API key found"
        API_KEYS_FOUND=$((API_KEYS_FOUND + 1))
    fi
    
    if [ $API_KEYS_FOUND -eq 0 ]; then
        echo "❌ No API keys found. Please set at least one of:"
        echo "   export OPENROUTER_API_KEY='your-key'"
        echo "   export ANTHROPIC_API_KEY='your-key'"
        echo "   export OPENAI_API_KEY='your-key'"
        exit 1
    fi
    
    echo "✅ All requirements satisfied"
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    print_header "Rate-Limit-Aware Experiment Runner"
    echo "Configuration:"
    echo "  Total experiments: ${TOTAL_EXPERIMENTS}"
    echo "  Max parallel jobs: ${MAX_PARALLEL_JOBS} (reduced for rate limits)"
    echo "  Job start delay: ${JOB_START_DELAY}s (prevents burst traffic)"
    echo "  Timeout per job: 2 hours (generous for rate limiting)"
    echo "  Results directory: ${RESULTS_DIR}"
    echo ""
    echo "KEY CHANGES:"
    echo "  ✓ NO aggressive timeouts - let experiments complete"
    echo "  ✓ Reduced parallelism to avoid rate limits"
    echo "  ✓ Delays between job starts"
    echo "  ✓ Provider rotation to distribute load"
    echo ""
    
    # Check requirements
    check_requirements
    
    # Step 1: Generate all experiment configurations
    print_header "Step 1: Generating Experiment Configurations"
    "${SCRIPTS_DIR}/generate_configs.sh"
    
    # Step 2: Run experiments with rate-limit-aware scheduling
    print_header "Step 2: Running Experiments (Rate-Limit Aware)"
    echo ""
    echo "Strategy:"
    echo "  1. Rotate between API providers (Anthropic, OpenAI, Google)"
    echo "  2. Add ${JOB_START_DELAY}s delay between job starts"
    echo "  3. Use generous 2-hour timeout per experiment"
    echo "  4. Let Python handle retries internally"
    echo ""
    
    # Use rate-limited runner if available
    if [ -f "${SCRIPTS_DIR}/run_parallel_jobs_rate_limited.sh" ]; then
        echo "Using rate-limited parallel runner..."
        "${SCRIPTS_DIR}/run_parallel_jobs_rate_limited.sh" ${MAX_PARALLEL_JOBS}
    else
        echo "Rate-limited runner not found, using standard runner..."
        echo "WARNING: This may cause more rate limit errors!"
        "${SCRIPTS_DIR}/run_parallel_jobs.sh" ${MAX_PARALLEL_JOBS}
    fi
    
    # Step 3: Analyze results
    print_header "Step 3: Analyzing Results"
    
    # Check for rate limit errors
    RATE_LIMIT_ERRORS=$(grep -l "429\|rate.*limit\|quota\|Too Many Requests" "${LOGS_DIR}"/experiment_*.log 2>/dev/null | wc -l || echo "0")
    TIMEOUT_ERRORS=$(grep -l "TIMEOUT\|timeout" "${LOGS_DIR}"/experiment_*.log 2>/dev/null | wc -l || echo "0")
    TOTAL_FAILURES=$(grep -l "FAILED\|TIMEOUT" "${LOGS_DIR}"/experiment_*.log 2>/dev/null | wc -l || echo "0")
    
    echo "Result Analysis:"
    echo "  Rate limit errors: ${RATE_LIMIT_ERRORS}"
    echo "  Timeout errors: ${TIMEOUT_ERRORS}"
    echo "  Total failures: ${TOTAL_FAILURES}"
    echo ""
    
    if [ ${RATE_LIMIT_ERRORS} -gt 0 ]; then
        print_header "Rate Limit Recommendations"
        echo "Found ${RATE_LIMIT_ERRORS} rate limit errors. Suggestions:"
        echo ""
        echo "1. REDUCE PARALLELISM:"
        echo "   export MAX_PARALLEL_JOBS=2"
        echo "   ./scripts/run_all_experiments_no_timeout.sh"
        echo ""
        echo "2. INCREASE DELAYS:"
        echo "   export JOB_START_DELAY=5"
        echo "   ./scripts/run_all_experiments_no_timeout.sh"
        echo ""
        echo "3. WAIT AND RETRY:"
        echo "   sleep 300  # Wait 5 minutes"
        echo "   ./scripts/retry_failed.sh"
        echo ""
        echo "4. USE DIFFERENT API KEYS:"
        echo "   Consider using multiple API keys and rotating them"
    fi
    
    # Step 4: Collect results
    print_header "Step 4: Collecting Results"
    "${SCRIPTS_DIR}/collect_results.sh"
    
    # Final summary
    print_header "Experiment Complete!"
    
    if [ -f "${RESULTS_DIR}/summary.json" ]; then
        echo ""
        echo "Summary Statistics:"
        python3 -c "
import json
with open('${RESULTS_DIR}/summary.json') as f:
    data = json.load(f)
    print(f'  Total experiments: {data.get(\"total\", 0)}')
    print(f'  Successful: {data.get(\"successful\", 0)}')
    print(f'  Failed: {data.get(\"failed\", 0)}')
    print(f'  Success rate: {data.get(\"success_rate\", 0)*100:.1f}%')
    
    # Calculate estimated time for retries
    if data.get('failed', 0) > 0:
        est_retry_time = data.get('failed', 0) * 10  # 10 min average per experiment
        print(f'')
        print(f'  Estimated retry time: {est_retry_time} minutes')
        print(f'  (assuming 10 min/experiment with proper delays)')
"
    fi
    
    echo ""
    echo "Results saved to: ${RESULTS_DIR}"
    echo ""
    
    # Provide specific retry guidance
    if [ ${TOTAL_FAILURES} -gt 0 ]; then
        echo "To retry failed experiments:"
        echo "  1. Wait a few minutes for rate limits to reset"
        echo "  2. Run with even fewer parallel jobs:"
        echo "     MAX_PARALLEL_JOBS=1 ${SCRIPTS_DIR}/retry_failed.sh"
    fi
}

# ============================================================================
# COMMAND LINE ARGUMENT PARSING
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            MAX_PARALLEL_JOBS="$2"
            shift 2
            ;;
        --delay)
            JOB_START_DELAY="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --parallel N     Number of parallel jobs (default: 3)"
            echo "  --delay S        Delay between job starts in seconds (default: 3)"
            echo "  --help           Show this help message"
            echo ""
            echo "This version focuses on avoiding rate limits rather than timeouts:"
            echo "  - Uses 2-hour timeout (very generous)"
            echo "  - Reduces parallel jobs to avoid rate limits"
            echo "  - Adds delays between job starts"
            echo "  - Rotates between API providers"
            echo ""
            echo "Example for very conservative run:"
            echo "  $0 --parallel 2 --delay 5"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done

# Export for child scripts
export MAX_PARALLEL_JOBS
export JOB_START_DELAY

# Run main function
main
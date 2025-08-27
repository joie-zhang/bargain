#!/bin/bash
# Improved main orchestrator with timeout-aware scheduling and recovery
# This version uses all the improvements to minimize timeout issues

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

# Number of parallel jobs (can be overridden)
MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS:-8}

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
    
    # Check for improved scripts
    if [ ! -f "${SCRIPTS_DIR}/run_single_experiment_improved.sh" ] || \
       [ ! -f "${SCRIPTS_DIR}/run_parallel_jobs_improved.sh" ] || \
       [ ! -f "${SCRIPTS_DIR}/retry_failed_improved.sh" ]; then
        echo "⚠️  Warning: Improved scripts not found. Using original scripts."
        echo "   Run this command to use improved timeout handling:"
        echo "   chmod +x ${SCRIPTS_DIR}/*_improved.sh"
    else
        echo "✅ Improved timeout handling scripts found"
        chmod +x "${SCRIPTS_DIR}"/*_improved.sh 2>/dev/null || true
    fi
    
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
    if [ -z "$OPENROUTER_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
        echo "❌ No API keys found. Please set at least one of:"
        echo "   export OPENROUTER_API_KEY='your-key'"
        echo "   export ANTHROPIC_API_KEY='your-key'"
        echo "   export OPENAI_API_KEY='your-key'"
        exit 1
    fi
    
    echo "✅ All requirements satisfied"
}

analyze_previous_runs() {
    print_header "Analyzing Previous Run Data"
    
    if [ -f "${LOGS_DIR}/timeouts.csv" ]; then
        echo "Found timeout history from previous runs"
        PREV_TIMEOUTS=$(wc -l < "${LOGS_DIR}/timeouts.csv")
        echo "  Previous timeout count: ${PREV_TIMEOUTS}"
        
        # Show most problematic pairs from history
        echo "  Most problematic model pairs:"
        cut -d',' -f2,3 "${LOGS_DIR}/timeouts.csv" 2>/dev/null | \
            sort | uniq -c | sort -rn | head -3 | \
            while read count models; do
                echo "    ${models}: ${count} timeouts"
            done
        
        # Ask if user wants to skip known problematic combinations
        read -p "Skip known highly problematic combinations? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            export SKIP_PROBLEMATIC=true
            echo "  Will skip combinations with >3 historical timeouts"
        fi
    else
        echo "No previous timeout data found (first run)"
    fi
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    print_header "Enhanced Scaling Experiment Runner"
    echo "Configuration:"
    echo "  Total experiments: ${TOTAL_EXPERIMENTS}"
    echo "  Max parallel jobs: ${MAX_PARALLEL_JOBS}"
    echo "  Results directory: ${RESULTS_DIR}"
    echo "  Timeout handling: ENHANCED"
    
    # Check requirements
    check_requirements
    
    # Analyze previous runs for patterns
    analyze_previous_runs
    
    # Step 1: Generate all experiment configurations
    print_header "Step 1: Generating Experiment Configurations"
    "${SCRIPTS_DIR}/generate_configs.sh"
    
    # Step 2: Run experiments with improved parallel scheduling
    print_header "Step 2: Running Experiments with Smart Scheduling"
    echo ""
    echo "Using enhanced parallel runner with:"
    echo "  - Difficulty-based scheduling (easy → moderate → difficult)"
    echo "  - Model-specific timeouts"
    echo "  - Automatic retry with backoff"
    echo "  - Reduced parallelism for problematic pairs"
    echo ""
    
    if [ -f "${SCRIPTS_DIR}/run_parallel_jobs_improved.sh" ]; then
        "${SCRIPTS_DIR}/run_parallel_jobs_improved.sh" ${MAX_PARALLEL_JOBS}
    else
        echo "⚠️  Falling back to original parallel runner"
        "${SCRIPTS_DIR}/run_parallel_jobs.sh" ${MAX_PARALLEL_JOBS}
    fi
    
    # Step 3: Check for failures and offer smart retry
    print_header "Step 3: Checking for Failures"
    
    FAILED_COUNT=$(grep -l "status: FAILED\|status: TIMEOUT" "${LOGS_DIR}"/experiment_*.log 2>/dev/null | wc -l || echo "0")
    
    if [ ${FAILED_COUNT} -gt 0 ]; then
        echo "Found ${FAILED_COUNT} failed experiments"
        
        # Automatically retry with improved strategy
        read -p "Automatically retry failed experiments with extended timeouts? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_header "Step 3b: Smart Retry of Failed Experiments"
            
            if [ -f "${SCRIPTS_DIR}/retry_failed_improved.sh" ]; then
                "${SCRIPTS_DIR}/retry_failed_improved.sh" 2  # Max 2 parallel for retries
            else
                echo "⚠️  Falling back to original retry script"
                "${SCRIPTS_DIR}/retry_failed.sh" 2
            fi
        fi
    else
        echo "✅ No failed experiments!"
    fi
    
    # Step 4: Collect and aggregate results
    print_header "Step 4: Collecting Results"
    "${SCRIPTS_DIR}/collect_results.sh"
    
    # Step 5: Generate timeout analysis report
    print_header "Step 5: Generating Analysis Report"
    
    python3 << 'EOF'
import json
import os
import glob

results_dir = "${RESULTS_DIR}"
logs_dir = f"{results_dir}/logs"

# Analyze results
total = successful = failed = timeouts = 0

for log_file in glob.glob(f"{logs_dir}/experiment_*.log"):
    total += 1
    with open(log_file, 'r') as f:
        content = f.read()
        if "status: SUCCESS" in content:
            successful += 1
        elif "status: TIMEOUT" in content:
            timeouts += 1
        else:
            failed += 1

print(f"Final Results Summary:")
print(f"  Total experiments: {total}")
print(f"  Successful: {successful} ({successful*100/total:.1f}%)")
print(f"  Timeouts: {timeouts} ({timeouts*100/total:.1f}%)")
print(f"  Other failures: {failed-timeouts} ({(failed-timeouts)*100/total:.1f}%)")
print(f"  Overall success rate: {successful*100/total:.1f}%")

if timeouts > 0:
    print(f"\n⚠️  Timeout Recommendations:")
    print(f"  - Consider increasing base timeout beyond current settings")
    print(f"  - Run problematic pairs separately with 30+ minute timeouts")
    print(f"  - Check API rate limits and consider adding delays")

# Save summary
summary = {
    "total": total,
    "successful": successful,
    "failed": failed,
    "timeouts": timeouts,
    "success_rate": successful/total if total > 0 else 0
}

with open(f"{results_dir}/final_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
EOF
    
    print_header "Experiment Complete!"
    echo ""
    echo "Results saved to: ${RESULTS_DIR}"
    echo "Detailed logs in: ${LOGS_DIR}"
    echo ""
    
    # Provide next steps
    echo "Next steps:"
    echo "1. Review results: cat ${RESULTS_DIR}/final_summary.json"
    echo "2. Analyze timeouts: cat ${LOGS_DIR}/timeouts.csv"
    echo "3. Generate visualizations: python3 ${BASE_DIR}/aggregate_results.py"
}

# ============================================================================
# COMMAND LINE ARGUMENT PARSING
# ============================================================================

SKIP_PROBLEMATIC=false
USE_ORIGINAL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            MAX_PARALLEL_JOBS="$2"
            shift 2
            ;;
        --skip-problematic)
            SKIP_PROBLEMATIC=true
            shift
            ;;
        --use-original)
            USE_ORIGINAL=true
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --parallel N          Number of parallel jobs (default: 8)"
            echo "  --skip-problematic    Skip known timeout-prone combinations"
            echo "  --use-original        Use original scripts (no improvements)"
            echo "  --dry-run            Show what would be run without executing"
            echo "  --help               Show this help message"
            echo ""
            echo "Improvements in this version:"
            echo "  - Smart scheduling based on experiment difficulty"
            echo "  - Model-specific timeout configuration"
            echo "  - Automatic retry with exponential backoff"
            echo "  - Reduced parallelism for problematic pairs"
            echo "  - Detailed timeout analysis and reporting"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done

# Export settings for child scripts
export SKIP_PROBLEMATIC
export USE_ORIGINAL

# Run main function
main
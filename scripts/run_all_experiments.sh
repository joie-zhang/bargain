#!/bin/bash
# Main orchestrator script for running 300 scaling experiments
# This simulates SLURM job arrays using local bash

set -e  # Exit on error

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

# Number of parallel jobs to run
MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS:-4}

# Total number of experiments (3 weak × 10 strong × 10 competition levels)
TOTAL_EXPERIMENTS=300

# ============================================================================
# FUNCTIONS
# ============================================================================

print_header() {
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

check_requirements() {
    print_header "Checking Requirements"
    
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
    echo ""
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    print_header "Scaling Experiment Runner"
    echo "Total experiments: ${TOTAL_EXPERIMENTS}"
    echo "Max parallel jobs: ${MAX_PARALLEL_JOBS}"
    echo "Results directory: ${RESULTS_DIR}"
    echo ""
    
    # Check requirements
    check_requirements
    
    # Step 1: Generate all experiment configurations
    print_header "Step 1: Generating Experiment Configurations"
    "${SCRIPTS_DIR}/generate_configs.sh"
    
    # Step 2: Run experiments in parallel batches
    print_header "Step 2: Running Experiments"
    echo "Starting parallel execution with ${MAX_PARALLEL_JOBS} workers..."
    "${SCRIPTS_DIR}/run_parallel_jobs.sh" ${MAX_PARALLEL_JOBS}
    
    # Step 3: Collect and aggregate results
    print_header "Step 3: Collecting Results"
    "${SCRIPTS_DIR}/collect_results.sh"
    
    # Final summary
    print_header "Experiment Complete!"
    echo "Results saved to: ${RESULTS_DIR}"
    echo ""
    
    # Show summary statistics
    if [ -f "${RESULTS_DIR}/summary.json" ]; then
        echo "Summary Statistics:"
        python3 -c "
import json
with open('${RESULTS_DIR}/summary.json') as f:
    data = json.load(f)
    print(f\"  Total experiments: {data.get('total', 0)}\")
    print(f\"  Successful: {data.get('successful', 0)}\")
    print(f\"  Failed: {data.get('failed', 0)}\")
    print(f\"  Success rate: {data.get('success_rate', 0)*100:.1f}%\")
"
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            MAX_PARALLEL_JOBS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --parallel N     Number of parallel jobs (default: 4)"
            echo "  --dry-run        Show what would be run without executing"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main
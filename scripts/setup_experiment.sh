#!/bin/bash
# Setup script to prepare everything for running experiments

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="${BASE_DIR}/scripts"

echo "============================================================"
echo "Setting up Scaling Experiment Scripts"
echo "============================================================"

# Make all scripts executable
echo "Making scripts executable..."
chmod +x "${SCRIPTS_DIR}/run_all_experiments.sh"
chmod +x "${SCRIPTS_DIR}/generate_configs.sh"
chmod +x "${SCRIPTS_DIR}/run_single_experiment.sh"
chmod +x "${SCRIPTS_DIR}/run_parallel_jobs.sh"
chmod +x "${SCRIPTS_DIR}/collect_results.sh"
chmod +x "${SCRIPTS_DIR}/retry_failed.sh"
chmod +x "${SCRIPTS_DIR}/setup_experiment.sh"

echo "✅ Scripts are now executable"

# Check for required Python packages
echo ""
echo "Checking Python dependencies..."
python3 -c "import json, sys, pathlib" 2>/dev/null && echo "✅ Core Python packages available" || echo "❌ Missing core Python packages"

# Check for optional packages
python3 -c "import pandas" 2>/dev/null && echo "✅ pandas available (for analysis)" || echo "⚠️  pandas not installed (optional, for analysis)"
python3 -c "import matplotlib" 2>/dev/null && echo "✅ matplotlib available (for visualization)" || echo "⚠️  matplotlib not installed (optional, for visualization)"

# Check for API keys
echo ""
echo "Checking API keys..."
[ -n "$OPENROUTER_API_KEY" ] && echo "✅ OPENROUTER_API_KEY set" || echo "⚠️  OPENROUTER_API_KEY not set"
[ -n "$ANTHROPIC_API_KEY" ] && echo "✅ ANTHROPIC_API_KEY set" || echo "⚠️  ANTHROPIC_API_KEY not set"
[ -n "$OPENAI_API_KEY" ] && echo "✅ OPENAI_API_KEY set" || echo "⚠️  OPENAI_API_KEY not set"

echo ""
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "To run the experiments:"
echo ""
echo "1. Run all 300 experiments (default 4 parallel):"
echo "   ${SCRIPTS_DIR}/run_all_experiments.sh"
echo ""
echo "2. Run with custom parallelism (e.g., 8 parallel):"
echo "   ${SCRIPTS_DIR}/run_all_experiments.sh --parallel 8"
echo ""
echo "3. Do a dry run to see what would be executed:"
echo "   ${SCRIPTS_DIR}/run_all_experiments.sh --dry-run"
echo ""
echo "4. Run individual steps manually:"
echo "   ${SCRIPTS_DIR}/generate_configs.sh    # Generate configurations"
echo "   ${SCRIPTS_DIR}/run_parallel_jobs.sh 4  # Run with 4 parallel jobs"
echo "   ${SCRIPTS_DIR}/collect_results.sh      # Collect and analyze results"
echo ""
echo "5. Retry failed experiments:"
echo "   ${SCRIPTS_DIR}/retry_failed.sh"
echo ""
echo "Results will be saved to:"
echo "   ${BASE_DIR}/experiments/results/scaling_experiment/"
echo ""
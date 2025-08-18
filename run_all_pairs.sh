#!/bin/bash

# Run All Model Pairs Experiment
# This script runs negotiations between all possible pairs of strong models

# Check if OpenRouter API key is set
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "ERROR: OPENROUTER_API_KEY environment variable is required"
    echo "Please set it with: export OPENROUTER_API_KEY='your-key-here'"
    exit 1
fi

# Configuration
NUM_RUNS=10
MAX_ROUNDS=10
NUM_ITEMS=6
COMPETITION_LEVEL=0.95

# Models to test
MODELS=(
    "gemini-pro"
    "claude-3-5-sonnet"
    "llama-3-1-405b"
    "qwen-2-5-72b"
)

# Create timestamp for this batch of experiments
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="experiments/results/all_pairs_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# Create log file
LOG_FILE="${OUTPUT_DIR}/experiment_log.txt"

# Function to log messages
log_message() {
    echo "$1" | tee -a "$LOG_FILE"
}

# Print header
log_message "================================================="
log_message "ALL MODEL PAIRS NEGOTIATION EXPERIMENT"
log_message "================================================="
log_message "Timestamp: $(date)"
log_message "Models: ${MODELS[@]}"
log_message "Configuration:"
log_message "  - Runs per pair: $NUM_RUNS"
log_message "  - Max rounds: $MAX_ROUNDS"
log_message "  - Items: $NUM_ITEMS"
log_message "  - Competition level: $COMPETITION_LEVEL"
log_message "Output directory: $OUTPUT_DIR"
log_message "================================================="

# Counter for tracking progress
PAIR_COUNT=0
TOTAL_PAIRS=10  # 4 self-play + 6 cross-play pairs
SUCCESS_COUNT=0
FAIL_COUNT=0

# Function to run a single experiment
run_experiment() {
    local model1=$1
    local model2=$2
    local pair_name="${model1}_vs_${model2}"
    
    PAIR_COUNT=$((PAIR_COUNT + 1))
    
    log_message ""
    log_message "================================================="
    log_message "PAIR $PAIR_COUNT/$TOTAL_PAIRS: $pair_name"
    log_message "================================================="
    log_message "Starting at: $(date)"
    
    # Create subdirectory for this pair
    PAIR_DIR="${OUTPUT_DIR}/${pair_name}"
    mkdir -p "$PAIR_DIR"
    
    # Run the experiment
    python run_strong_models_experiment.py \
        --models "$model1" "$model2" \
        --runs $NUM_RUNS \
        --rounds $MAX_ROUNDS \
        --items $NUM_ITEMS \
        --competition $COMPETITION_LEVEL \
        2>&1 | tee "${PAIR_DIR}/output.log"
    
    # Check if successful
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        log_message "✓ SUCCESS: $pair_name completed"
        
        # Move results to pair directory
        LATEST_RESULT=$(ls -t experiments/results/strong_models/strong_models_results_*.json 2>/dev/null | head -1)
        if [ -n "$LATEST_RESULT" ]; then
            cp "$LATEST_RESULT" "${PAIR_DIR}/"
            log_message "  Results saved to: ${PAIR_DIR}/"
        fi
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        log_message "✗ FAILED: $pair_name"
    fi
    
    log_message "Progress: $PAIR_COUNT/$TOTAL_PAIRS completed ($SUCCESS_COUNT successful, $FAIL_COUNT failed)"
    log_message "Current time: $(date)"
}

# Run all pairs
# Self-play experiments (models against themselves)
log_message ""
log_message "RUNNING SELF-PLAY EXPERIMENTS"
log_message "================================================="

for model in "${MODELS[@]}"; do
    run_experiment "$model" "$model"
done

# Cross-play experiments (different models against each other)
log_message ""
log_message "RUNNING CROSS-PLAY EXPERIMENTS"
log_message "================================================="

# Generate all unique pairs
for ((i=0; i<${#MODELS[@]}; i++)); do
    for ((j=i+1; j<${#MODELS[@]}; j++)); do
        run_experiment "${MODELS[i]}" "${MODELS[j]}"
    done
done

# Print final summary
log_message ""
log_message "================================================="
log_message "ALL EXPERIMENTS COMPLETE"
log_message "================================================="
log_message "Total pairs tested: $TOTAL_PAIRS"
log_message "Successful: $SUCCESS_COUNT"
log_message "Failed: $FAIL_COUNT"
log_message "End time: $(date)"
log_message "Results saved to: $OUTPUT_DIR"
log_message ""

# Create summary file
SUMMARY_FILE="${OUTPUT_DIR}/summary.txt"
{
    echo "All Model Pairs Experiment Summary"
    echo "=================================="
    echo "Timestamp: ${TIMESTAMP}"
    echo "Total pairs: $TOTAL_PAIRS"
    echo "Successful: $SUCCESS_COUNT"
    echo "Failed: $FAIL_COUNT"
    echo ""
    echo "Configuration:"
    echo "  - Models: ${MODELS[@]}"
    echo "  - Runs per pair: $NUM_RUNS"
    echo "  - Max rounds: $MAX_ROUNDS"
    echo "  - Items: $NUM_ITEMS"
    echo "  - Competition level: $COMPETITION_LEVEL"
    echo ""
    echo "Pairs tested:"
    
    # List all pairs
    for model in "${MODELS[@]}"; do
        echo "  - ${model} vs ${model} (self-play)"
    done
    
    for ((i=0; i<${#MODELS[@]}; i++)); do
        for ((j=i+1; j<${#MODELS[@]}; j++)); do
            echo "  - ${MODELS[i]} vs ${MODELS[j]}"
        done
    done
} > "$SUMMARY_FILE"

log_message "Summary saved to: $SUMMARY_FILE"

# Check if all experiments were successful
if [ $SUCCESS_COUNT -eq $TOTAL_PAIRS ]; then
    log_message "✓ All experiments completed successfully!"
    exit 0
else
    log_message "⚠ Some experiments failed. Check logs for details."
    exit 1
fi
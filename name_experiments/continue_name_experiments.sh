#!/bin/bash
# Continue name experiments from a specific pair, re-running the last failed pair
# Usage: ./continue_name_experiments.sh [START_PAIR] [LAST_FAILED_PAIR]

set -e

# Activate conda environment if available
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate negotiation-research
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Model to use (can be overridden with environment variable)
MODEL="${MODEL:-claude-4-5-haiku}"

# Competition level (can be overridden)
COMPETITION_LEVEL="${COMPETITION_LEVEL:-0.95}"

# Other experiment parameters
MAX_ROUNDS="${MAX_ROUNDS:-10}"
NUM_ITEMS="${NUM_ITEMS:-5}"
GAMMA_DISCOUNT="${GAMMA_DISCOUNT:-0.9}"

# Parse command line arguments
START_PAIR="${1:-9}"  # Default: start from pair 9
LAST_FAILED_PAIR="${2:-8}"  # Default: re-run pair 8

echo "=========================================="
echo "Continuing Name Experiments"
echo "=========================================="
echo "Model: $MODEL"
echo "Competition Level: $COMPETITION_LEVEL"
echo "Max Rounds: $MAX_ROUNDS"
echo "Items: $NUM_ITEMS"
echo "Re-running pair: $LAST_FAILED_PAIR"
echo "Continuing from pair: $START_PAIR"
echo "=========================================="
echo ""

# Generate all pairs (original + reverse) using Python
PAIRS_FILE=$(mktemp)
python3 << PYTHON_SCRIPT > "$PAIRS_FILE"
import sys
import os
import importlib.util

# Import name_experiment_config directly without going through package __init__
config_path = os.path.join("$SCRIPT_DIR", "name_experiment_config.py")
spec = importlib.util.spec_from_file_location("name_experiment_config", config_path)
name_experiment_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(name_experiment_config)

ALL_NAME_COMBINATIONS = name_experiment_config.ALL_NAME_COMBINATIONS

# Get all combinations from config
seen_pairs = set()
all_ordered_pairs = []

for name1, name2 in ALL_NAME_COMBINATIONS:
    # Add original order
    pair_key = (name1, name2)
    if pair_key not in seen_pairs:
        all_ordered_pairs.append((name1, name2))
        seen_pairs.add(pair_key)
    
    # Add reverse order if not already seen
    reverse_key = (name2, name1)
    if reverse_key not in seen_pairs:
        all_ordered_pairs.append((name2, name1))
        seen_pairs.add(reverse_key)

# Print all pairs (one per line, format: name1|name2)
for name1, name2 in all_ordered_pairs:
    print(f"{name1}|{name2}")
PYTHON_SCRIPT

# Count total pairs
TOTAL_PAIRS=$(wc -l < "$PAIRS_FILE")
echo "Total pairs available: $TOTAL_PAIRS"
echo ""

# Validate pair numbers
if [ "$LAST_FAILED_PAIR" -lt 1 ] || [ "$LAST_FAILED_PAIR" -gt "$TOTAL_PAIRS" ]; then
    echo "ERROR: Last failed pair ($LAST_FAILED_PAIR) is out of range (1-$TOTAL_PAIRS)"
    rm "$PAIRS_FILE"
    exit 1
fi

if [ "$START_PAIR" -lt 1 ] || [ "$START_PAIR" -gt "$TOTAL_PAIRS" ]; then
    echo "ERROR: Start pair ($START_PAIR) is out of range (1-$TOTAL_PAIRS)"
    rm "$PAIRS_FILE"
    exit 1
fi

if [ "$START_PAIR" -le "$LAST_FAILED_PAIR" ]; then
    echo "ERROR: Start pair ($START_PAIR) must be greater than last failed pair ($LAST_FAILED_PAIR)"
    rm "$PAIRS_FILE"
    exit 1
fi

# First, re-run the last failed pair
echo "=========================================="
echo "Re-running Pair $LAST_FAILED_PAIR/$TOTAL_PAIRS"
echo "=========================================="

# Extract the specific pair to re-run
FAILED_PAIR=$(sed -n "${LAST_FAILED_PAIR}p" "$PAIRS_FILE")
IFS='|' read -r name1 name2 <<< "$FAILED_PAIR"

echo "Pair: $name1 vs $name2"
echo ""

python3 "$SCRIPT_DIR/run_name_experiment.py" \
    --model "$MODEL" \
    --agent-names "$name1" "$name2" \
    --competition-level "$COMPETITION_LEVEL" \
    --max-rounds "$MAX_ROUNDS" \
    --num-items "$NUM_ITEMS" \
    --gamma-discount "$GAMMA_DISCOUNT"

echo ""
echo "✅ Re-run of pair $LAST_FAILED_PAIR completed"
echo ""

# Now continue with remaining pairs
echo "=========================================="
echo "Continuing with remaining pairs"
echo "=========================================="
echo ""

PAIR_COUNT=0

# Process all pairs, skipping to START_PAIR
while IFS='|' read -r name1 name2; do
    PAIR_COUNT=$((PAIR_COUNT + 1))
    
    # Skip pairs before START_PAIR
    if [ $PAIR_COUNT -lt $START_PAIR ]; then
        continue
    fi
    
    echo ""
    echo "=========================================="
    echo "Pair $PAIR_COUNT/$TOTAL_PAIRS: $name1 vs $name2"
    echo "=========================================="
    
    python3 "$SCRIPT_DIR/run_name_experiment.py" \
        --model "$MODEL" \
        --agent-names "$name1" "$name2" \
        --competition-level "$COMPETITION_LEVEL" \
        --max-rounds "$MAX_ROUNDS" \
        --num-items "$NUM_ITEMS" \
        --gamma-discount "$GAMMA_DISCOUNT"
done < "$PAIRS_FILE"

# Clean up
rm "$PAIRS_FILE"

REMAINING_PAIRS=$((TOTAL_PAIRS - START_PAIR + 1))
echo ""
echo "=========================================="
echo "✅ All remaining experiments completed!"
echo "Re-ran pair: $LAST_FAILED_PAIR"
echo "Continued from pair: $START_PAIR to $TOTAL_PAIRS"
echo "Total pairs processed: $((REMAINING_PAIRS + 1)) (1 re-run + $REMAINING_PAIRS new)"
echo "=========================================="


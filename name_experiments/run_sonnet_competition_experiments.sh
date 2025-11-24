#!/bin/bash
# Run name experiments with Claude 3.7 Sonnet at competition levels 1.0 and 0.0
# This script runs all name combinations (including company names) for both competition levels

set -e

# Activate conda environment if available
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate negotiation-research
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Model to use
MODEL="claude-3-7-sonnet"

# Competition levels to test
COMPETITION_LEVELS=(1.0 0.5)

# Other experiment parameters
MAX_ROUNDS="${MAX_ROUNDS:-10}"
NUM_ITEMS="${NUM_ITEMS:-5}"
GAMMA_DISCOUNT="${GAMMA_DISCOUNT:-0.9}"

echo "=========================================="
echo "Claude 3.7 Sonnet Competition Level Experiments"
echo "=========================================="
echo "Model: $MODEL"
echo "Competition Levels: ${COMPETITION_LEVELS[*]}"
echo "Max Rounds: $MAX_ROUNDS"
echo "Items: $NUM_ITEMS"
echo "Discount Factor: $GAMMA_DISCOUNT"
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
echo "Total pairs to run per competition level: $TOTAL_PAIRS"
echo "Total experiments: $((TOTAL_PAIRS * ${#COMPETITION_LEVELS[@]}))"
echo ""

# Run experiments for each competition level
for COMP_LEVEL in "${COMPETITION_LEVELS[@]}"; do
    echo ""
    echo "=========================================="
    echo "COMPETITION LEVEL: $COMP_LEVEL"
    echo "=========================================="
    echo ""
    
    PAIR_COUNT=0
    while IFS='|' read -r name1 name2; do
        PAIR_COUNT=$((PAIR_COUNT + 1))
        echo ""
        echo "=========================================="
        echo "Competition Level $COMP_LEVEL - Pair $PAIR_COUNT/$TOTAL_PAIRS: $name1 vs $name2"
        echo "=========================================="
        
        python3 "$SCRIPT_DIR/run_name_experiment.py" \
            --model "$MODEL" \
            --agent-names "$name1" "$name2" \
            --competition-level "$COMP_LEVEL" \
            --max-rounds "$MAX_ROUNDS" \
            --num-items "$NUM_ITEMS" \
            --gamma-discount "$GAMMA_DISCOUNT"
    done < "$PAIRS_FILE"
    
    echo ""
    echo "✅ Completed all pairs for competition level $COMP_LEVEL"
    echo ""
done

# Clean up
rm "$PAIRS_FILE"

echo ""
echo "=========================================="
echo "✅ All experiments completed!"
echo "Model: $MODEL"
echo "Competition Levels: ${COMPETITION_LEVELS[*]}"
echo "Total pairs per level: $TOTAL_PAIRS"
echo "Total experiments run: $((TOTAL_PAIRS * ${#COMPETITION_LEVELS[@]}))"
echo "=========================================="


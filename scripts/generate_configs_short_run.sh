#!/bin/bash
# Generate configuration files for SHORT experiment run
# Only claude-3-opus (weak) vs all strong models, competition level 1.0, single run

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="${BASE_DIR}/experiments/results/short_experiment/configs"
mkdir -p "${CONFIG_DIR}"

# Model definitions for short run
WEAK_MODEL="claude-3-opus"  # Only one weak model

# All strong models
STRONG_MODELS=(
    # Claude models
    "claude-3-5-haiku"
    "claude-3-5-sonnet"
    "claude-4-sonnet"
    "claude-4-1-opus"
    
    # OpenAI models
    "gpt-4o-latest"
    "gpt-4o-mini"
    "o1"
    "o3"
    
    # Google models
    "gemini-2-0-flash"
    "gemini-2-5-pro"
    "gemma-3-27b"
)

# Only competition level 1.0 (fully competitive)
COMPETITION_LEVEL=1.0

# Base parameters
NUM_ITEMS=5
MAX_ROUNDS=10
BASE_SEED=42

echo "Generating SHORT experiment configurations..."
echo "  Weak model: ${WEAK_MODEL}"
echo "  Strong models: ${#STRONG_MODELS[@]}"
echo "  Competition level: ${COMPETITION_LEVEL}"
echo "  Will generate: ${#STRONG_MODELS[@]} configs (single direction)"
echo ""

# Counter for experiment ID
EXPERIMENT_ID=0

# Generate configs - only one direction (weak model first)
echo "Generating configs..."
for strong_model in "${STRONG_MODELS[@]}"; do
    # Create config file
    CONFIG_FILE="${CONFIG_DIR}/config_${EXPERIMENT_ID}.json"
    
    # Use the same base seed for all experiments (will vary by run, not by model pair)
    SEED=${BASE_SEED}
    
    # Write configuration as JSON - WEAK MODEL FIRST
    cat > "${CONFIG_FILE}" << EOF
{
    "experiment_id": ${EXPERIMENT_ID},
    "weak_model": "${WEAK_MODEL}",
    "strong_model": "${strong_model}",
    "models": ["${WEAK_MODEL}", "${strong_model}"],
    "model_order": "weak_first",
    "competition_level": ${COMPETITION_LEVEL},
    "num_items": ${NUM_ITEMS},
    "max_rounds": ${MAX_ROUNDS},
    "random_seed": ${SEED},
    "output_dir": "experiments/results/short_experiment/${WEAK_MODEL}_vs_${strong_model}/comp_${COMPETITION_LEVEL}"
}
EOF
    
    EXPERIMENT_ID=$((EXPERIMENT_ID + 1))
done

echo ""
echo "✅ Generated ${EXPERIMENT_ID} configuration files:"
echo "   - Total configs: ${EXPERIMENT_ID} (IDs 0-$((EXPERIMENT_ID-1)))"
echo "   - Location: ${CONFIG_DIR}"

# Create master config list
MASTER_CONFIG="${CONFIG_DIR}/all_configs.txt"
ls -1 "${CONFIG_DIR}"/config_*.json > "${MASTER_CONFIG}"
echo "✅ Created master config list: ${MASTER_CONFIG}"

# Create a detailed summary file
SUMMARY_FILE="${CONFIG_DIR}/summary.txt"
cat > "${SUMMARY_FILE}" << EOF
Short Experiment Configuration Summary
=======================================
Total experiments: ${EXPERIMENT_ID}

Weak model: ${WEAK_MODEL}
Strong models (${#STRONG_MODELS[@]}): ${STRONG_MODELS[@]}
Competition level: ${COMPETITION_LEVEL} (fully competitive)

Items per negotiation: ${NUM_ITEMS}
Max rounds: ${MAX_ROUNDS}
Base random seed: ${BASE_SEED}

Model Order: Weak model (${WEAK_MODEL}) always goes first
  
Purpose: Quick test run with fully competitive settings only
EOF

echo "✅ Created summary: ${SUMMARY_FILE}"

# Create a CSV index for easier analysis
CSV_FILE="${CONFIG_DIR}/experiment_index.csv"
echo "experiment_id,weak_model,strong_model,model_order,competition_level,config_file" > "${CSV_FILE}"

for config_file in "${CONFIG_DIR}"/config_*.json; do
    if [[ -f "$config_file" ]]; then
        # Extract values from JSON
        exp_id=$(grep -o '"experiment_id": [0-9]*' "$config_file" | grep -o '[0-9]*')
        weak=$(grep -o '"weak_model": "[^"]*"' "$config_file" | cut -d'"' -f4)
        strong=$(grep -o '"strong_model": "[^"]*"' "$config_file" | cut -d'"' -f4)
        order=$(grep -o '"model_order": "[^"]*"' "$config_file" | cut -d'"' -f4)
        comp=$(grep -o '"competition_level": [0-9.]*' "$config_file" | grep -o '[0-9.]*')
        
        echo "${exp_id},${weak},${strong},${order},${comp},$(basename $config_file)" >> "${CSV_FILE}"
    fi
done

echo "✅ Created experiment index: ${CSV_FILE}"
echo ""
echo "To run this short experiment:"
echo "  bash scripts/run_short_experiment.sh"
#!/bin/bash
# Generate configuration files for all 300 experiments

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="${BASE_DIR}/experiments/results/scaling_experiment/configs"
mkdir -p "${CONFIG_DIR}"

# Model definitions
WEAK_MODELS=(
    "claude-3-opus"
    "gpt-4o"
    "gemini-1-5-pro"
)

STRONG_MODELS=(
    "claude-3-5-haiku"
    "claude-3-5-sonnet"
    "claude-4-sonnet"
    "claude-4-1-opus"
    "o3-mini"
    "o4-mini"
    "o3"
    "gpt-5"
    "gemini-2-0-flash"
    "gemini-2-5-flash"
    "gemini-2-5-pro"
)

# Competition levels
COMPETITION_LEVELS=(0.0 0.25 0.5 0.75 1.0)

# Base parameters
NUM_ITEMS=5
MAX_ROUNDS=10
BASE_SEED=42

echo "Generating experiment configurations..."
echo "  Weak models: ${#WEAK_MODELS[@]}"
echo "  Strong models: ${#STRONG_MODELS[@]}"
echo "  Competition levels: ${#COMPETITION_LEVELS[@]}"

# Counter for experiment ID
EXPERIMENT_ID=0

# Generate configuration for each combination
for weak_model in "${WEAK_MODELS[@]}"; do
    for strong_model in "${STRONG_MODELS[@]}"; do
        for comp_level in "${COMPETITION_LEVELS[@]}"; do
            # Create config file
            CONFIG_FILE="${CONFIG_DIR}/config_${EXPERIMENT_ID}.json"
            
            # Calculate seed for this experiment
            SEED=$((BASE_SEED + EXPERIMENT_ID))
            
            # Write configuration as JSON
            cat > "${CONFIG_FILE}" << EOF
{
    "experiment_id": ${EXPERIMENT_ID},
    "weak_model": "${weak_model}",
    "strong_model": "${strong_model}",
    "models": ["${weak_model}", "${strong_model}"],
    "competition_level": ${comp_level},
    "num_items": ${NUM_ITEMS},
    "max_rounds": ${MAX_ROUNDS},
    "random_seed": ${SEED},
    "output_dir": "experiments/results/scaling_experiment/${weak_model}_vs_${strong_model}/comp_${comp_level}"
}
EOF
            
            EXPERIMENT_ID=$((EXPERIMENT_ID + 1))
        done
    done
done

echo "✅ Generated ${EXPERIMENT_ID} configuration files in ${CONFIG_DIR}"

# Create master config list
MASTER_CONFIG="${CONFIG_DIR}/all_configs.txt"
ls -1 "${CONFIG_DIR}"/config_*.json > "${MASTER_CONFIG}"
echo "✅ Created master config list: ${MASTER_CONFIG}"

# Create a summary file
SUMMARY_FILE="${CONFIG_DIR}/summary.txt"
cat > "${SUMMARY_FILE}" << EOF
Experiment Configuration Summary
================================
Total experiments: ${EXPERIMENT_ID}
Weak models: ${WEAK_MODELS[@]}
Strong models: ${STRONG_MODELS[@]}
Competition levels: ${COMPETITION_LEVELS[@]}
Items per negotiation: ${NUM_ITEMS}
Max rounds: ${MAX_ROUNDS}
Base random seed: ${BASE_SEED}
EOF

echo "✅ Created summary: ${SUMMARY_FILE}"
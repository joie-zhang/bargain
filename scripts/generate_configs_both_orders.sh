#!/bin/bash
# Generate configuration files for experiments with BOTH orderings
# This creates 600 configs total: 300 with weak model first, 300 with weak model second

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="${BASE_DIR}/experiments/results/scaling_experiment/configs"
mkdir -p "${CONFIG_DIR}"

# Model definitions
# Weak models - baseline models for exploitation experiments
WEAK_MODELS=(
    "claude-3-opus"
    "gemini-1-5-pro"
    "gpt-4o"  # May 2024 version
)

# Strong models - newer/more capable models that may exploit weak models
STRONG_MODELS=(
    # Claude models
    "claude-3-5-haiku"
    "claude-3-5-sonnet"
    "claude-4-sonnet"
    "claude-4-1-opus"
    
    # OpenAI models
    "gpt-4o-latest"  # Nov 2024 version
    "gpt-4o-mini"
    "o1"
    "o3"
    
    # Google models
    "gemini-2-0-flash"
    "gemini-2-5-pro"
    "gemma-3-27b"
)

# Competition levels
COMPETITION_LEVELS=(0.0 0.25 0.5 0.75 1.0)

# Base parameters
NUM_ITEMS=5
MAX_ROUNDS=10
BASE_SEED=42

echo "Generating experiment configurations with BOTH orderings..."
echo "  Weak models: ${#WEAK_MODELS[@]}"
echo "  Strong models: ${#STRONG_MODELS[@]}"
echo "  Competition levels: ${#COMPETITION_LEVELS[@]}"
echo "  Will generate 2x configs (both model orderings)"
echo ""

# Counter for experiment ID
EXPERIMENT_ID=0

# First pass: weak model first (IDs 0-299)
echo "Generating configs with weak model first (0-299)..."
for weak_model in "${WEAK_MODELS[@]}"; do
    for strong_model in "${STRONG_MODELS[@]}"; do
        for comp_level in "${COMPETITION_LEVELS[@]}"; do
            # Create config file
            CONFIG_FILE="${CONFIG_DIR}/config_${EXPERIMENT_ID}.json"
            
            # Calculate seed for this experiment
            SEED=$((BASE_SEED + EXPERIMENT_ID))
            
            # Write configuration as JSON - WEAK MODEL FIRST
            cat > "${CONFIG_FILE}" << EOF
{
    "experiment_id": ${EXPERIMENT_ID},
    "weak_model": "${weak_model}",
    "strong_model": "${strong_model}",
    "models": ["${weak_model}", "${strong_model}"],
    "model_order": "weak_first",
    "competition_level": ${comp_level},
    "num_items": ${NUM_ITEMS},
    "max_rounds": ${MAX_ROUNDS},
    "random_seed": ${SEED},
    "output_dir": "experiments/results/scaling_experiment/${weak_model}_vs_${strong_model}/weak_first/comp_${comp_level}"
}
EOF
            
            EXPERIMENT_ID=$((EXPERIMENT_ID + 1))
        done
    done
done

WEAK_FIRST_COUNT=${EXPERIMENT_ID}

# Second pass: strong model first (IDs 300-599)
echo "Generating configs with strong model first (300-599)..."
for weak_model in "${WEAK_MODELS[@]}"; do
    for strong_model in "${STRONG_MODELS[@]}"; do
        for comp_level in "${COMPETITION_LEVELS[@]}"; do
            # Create config file
            CONFIG_FILE="${CONFIG_DIR}/config_${EXPERIMENT_ID}.json"
            
            # Calculate seed for this experiment
            SEED=$((BASE_SEED + EXPERIMENT_ID))
            
            # Write configuration as JSON - STRONG MODEL FIRST
            cat > "${CONFIG_FILE}" << EOF
{
    "experiment_id": ${EXPERIMENT_ID},
    "weak_model": "${weak_model}",
    "strong_model": "${strong_model}",
    "models": ["${strong_model}", "${weak_model}"],
    "model_order": "strong_first",
    "competition_level": ${comp_level},
    "num_items": ${NUM_ITEMS},
    "max_rounds": ${MAX_ROUNDS},
    "random_seed": ${SEED},
    "output_dir": "experiments/results/scaling_experiment/${weak_model}_vs_${strong_model}/strong_first/comp_${comp_level}"
}
EOF
            
            EXPERIMENT_ID=$((EXPERIMENT_ID + 1))
        done
    done
done

STRONG_FIRST_COUNT=$((EXPERIMENT_ID - WEAK_FIRST_COUNT))

echo ""
echo "✅ Generated ${EXPERIMENT_ID} total configuration files:"
echo "   - Weak model first: ${WEAK_FIRST_COUNT} configs (IDs 0-$((WEAK_FIRST_COUNT-1)))"
echo "   - Strong model first: ${STRONG_FIRST_COUNT} configs (IDs ${WEAK_FIRST_COUNT}-$((EXPERIMENT_ID-1)))"
echo "   - Location: ${CONFIG_DIR}"

# Create master config list
MASTER_CONFIG="${CONFIG_DIR}/all_configs.txt"
ls -1 "${CONFIG_DIR}"/config_*.json > "${MASTER_CONFIG}"
echo "✅ Created master config list: ${MASTER_CONFIG}"

# Create a detailed summary file
SUMMARY_FILE="${CONFIG_DIR}/summary.txt"
cat > "${SUMMARY_FILE}" << EOF
Experiment Configuration Summary (Both Orderings)
================================================
Total experiments: ${EXPERIMENT_ID}
  - Weak model first: ${WEAK_FIRST_COUNT} (IDs 0-$((WEAK_FIRST_COUNT-1)))
  - Strong model first: ${STRONG_FIRST_COUNT} (IDs ${WEAK_FIRST_COUNT}-$((EXPERIMENT_ID-1)))

Weak models (3): ${WEAK_MODELS[@]}
Strong models (12): ${STRONG_MODELS[@]}
Competition levels (5): ${COMPETITION_LEVELS[@]}

Items per negotiation: ${NUM_ITEMS}
Max rounds: ${MAX_ROUNDS}
Base random seed: ${BASE_SEED}

Model Order Effects:
  - Configs 0-299: Weak model is agent_0 (goes first)
  - Configs 300-599: Strong model is agent_0 (goes first)
  
This allows testing for:
  - First-mover advantages
  - Order-dependent strategic behavior
  - Asymmetric exploitation patterns
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
echo "To run experiments:"
echo "  - First ordering only: Use job IDs 0-299"
echo "  - Second ordering only: Use job IDs 300-599"
echo "  - Both orderings: Use all job IDs 0-599"
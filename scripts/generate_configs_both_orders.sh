#!/bin/bash
# Generate configuration files for experiments with multiple runs
# This creates configs with 3 runs per model pair/competition level
# Each run has a different seed, but run N has consistent seed across all model pairs

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="${BASE_DIR}/experiments/results/scaling_experiment/configs"
mkdir -p "${CONFIG_DIR}"

# Model definitions
# NOTE: These short names map to full model_id values defined in:
#       strong_models_experiment/configs.py (STRONG_MODELS_CONFIG dictionary)
#       The Python code looks up these short names to get the full model_id (e.g., "gpt-5-mini" -> "gpt-5-mini-2025-08-07")
#
# Weak models - baseline models for exploitation experiments
# From Multi-Agent Strategic Games Evaluation Models (Weak Tier - Elo < 1290)
WEAK_MODELS=(
    "gpt-4o" # May 2024 version
    "gemini-1-5-pro"
)

# Strong models - newer/more capable models that may exploit weak models
# From Multi-Agent Strategic Games Evaluation Models (11 models total)
STRONG_MODELS=(
    # STRONG TIER - Elo ≥ 1415 (6 models)
    "gemini-3-pro"  # Strong tier, Elo: 1492, Closed-source, Reasoning
    "gpt-5.2-high"  # Strong tier, Elo: 1465, Closed-source, Reasoning
    "claude-opus-4-5"  # Strong tier, Elo: 1462, Closed-source, Non-reasoning
    "kimi-k2-thinking"  # Strong tier, Elo: 1438, Open-source, Reasoning
    "deepseek-r1-0528"  # Strong tier, Elo: 1426, Open-source, Reasoning
    "qwen3-235b-a22b-instruct-2507"  # Strong tier, Elo: 1418, Open-source, Non-reasoning
    
    # MEDIUM TIER - 1290 ≤ Elo < 1415 (3 models)
    "claude-4-5-haiku"  # Medium tier, Elo: 1378, Closed-source, Non-reasoning
    "o4-mini-2025-04-16"  # Medium tier, Elo: 1362, Closed-source, Reasoning
    "gpt-oss-20b"  # Medium tier, Elo: 1315, Open-source, Non-reasoning

    # WEAK TIER - Elo < 1290 (2 models)
    "llama-3.3-70b-instruct"  # Weak tier, Elo: 1276, Open-source, Non-reasoning
    "llama-3.1-8b-instruct"  # Weak tier, Elo: 1193, Open-source, Non-reasoning
)

# Competition levels
COMPETITION_LEVELS=(0.0 0.25 0.5 0.75 1.0)

# Base parameters
NUM_ITEMS=5
MAX_ROUNDS=10
NUM_RUNS=5  # Number of runs per configuration

# Seeds for each run (consistent across model pairs)
RUN_SEEDS=(42 123 456 789 101112)

echo "Generating experiment configurations with multiple runs..."
echo "  Weak models: ${#WEAK_MODELS[@]}"
echo "  Strong models: ${#STRONG_MODELS[@]}"
echo "  Competition levels: ${#COMPETITION_LEVELS[@]}"
echo "  Runs per config: ${NUM_RUNS}"
echo "  Total configs: $((${#WEAK_MODELS[@]} * ${#STRONG_MODELS[@]} * ${#COMPETITION_LEVELS[@]} * ${NUM_RUNS}))"
echo ""

# Counter for experiment ID
EXPERIMENT_ID=0

# Generate configs: weak model first only, with multiple runs
echo "Generating configs with weak model first..."
for weak_model in "${WEAK_MODELS[@]}"; do
    for strong_model in "${STRONG_MODELS[@]}"; do
        for comp_level in "${COMPETITION_LEVELS[@]}"; do
            for run_idx in "${!RUN_SEEDS[@]}"; do
                # Create config file
                CONFIG_FILE="${CONFIG_DIR}/config_${EXPERIMENT_ID}.json"
                
                # Get seed for this run (consistent across all model pairs for same run number)
                SEED=${RUN_SEEDS[$run_idx]}
                RUN_NUM=$((run_idx + 1))
                
                # Write configuration as JSON - WEAK MODEL FIRST
                cat > "${CONFIG_FILE}" << EOF
{
    "experiment_id": ${EXPERIMENT_ID},
    "weak_model": "${weak_model}",
    "strong_model": "${strong_model}",
    "models": ["${weak_model}", "${strong_model}"],
    "model_order": "weak_first",
    "competition_level": ${comp_level},
    "run_number": ${RUN_NUM},
    "num_items": ${NUM_ITEMS},
    "max_rounds": ${MAX_ROUNDS},
    "random_seed": ${SEED},
    "output_dir": "experiments/results/scaling_experiment/${weak_model}_vs_${strong_model}/weak_first/comp_${comp_level}/run_${RUN_NUM}"
}
EOF
                
                EXPERIMENT_ID=$((EXPERIMENT_ID + 1))
            done
        done
    done
done

TOTAL_COUNT=${EXPERIMENT_ID}

echo ""
echo "✅ Generated ${EXPERIMENT_ID} total configuration files:"
echo "   - Model pairs: $((${#WEAK_MODELS[@]} * ${#STRONG_MODELS[@]}))"
echo "   - Competition levels: ${#COMPETITION_LEVELS[@]}"
echo "   - Runs per config: ${NUM_RUNS}"
echo "   - Location: ${CONFIG_DIR}"

# Create master config list (only if configs were just generated)
# Skip if file already exists and is up to date
MASTER_CONFIG="${CONFIG_DIR}/all_configs.txt"
if [ ! -f "${MASTER_CONFIG}" ] || [ "${CONFIG_DIR}"/config_*.json -nt "${MASTER_CONFIG}" ]; then
    ls -1 "${CONFIG_DIR}"/config_*.json > "${MASTER_CONFIG}"
    echo "✅ Created/updated master config list: ${MASTER_CONFIG}"
fi

# Create a detailed summary file
SUMMARY_FILE="${CONFIG_DIR}/summary.txt"
cat > "${SUMMARY_FILE}" << EOF
Experiment Configuration Summary (Multiple Runs)
================================================
Total experiments: ${EXPERIMENT_ID}
  - Model pairs: $((${#WEAK_MODELS[@]} * ${#STRONG_MODELS[@]}))
  - Competition levels: ${#COMPETITION_LEVELS[@]}
  - Runs per configuration: ${NUM_RUNS}
  - Total configs: ${EXPERIMENT_ID}

Weak models (${#WEAK_MODELS[@]}): ${WEAK_MODELS[@]}
Strong models (${#STRONG_MODELS[@]}): ${STRONG_MODELS[@]}
Competition levels (${#COMPETITION_LEVELS[@]}): ${COMPETITION_LEVELS[@]}

Items per negotiation: ${NUM_ITEMS}
Max rounds: ${MAX_ROUNDS}

Random Seeds by Run:
  - Run 1: ${RUN_SEEDS[0]} (all model pairs use this for run 1)
  - Run 2: ${RUN_SEEDS[1]} (all model pairs use this for run 2)
  - Run 3: ${RUN_SEEDS[2]} (all model pairs use this for run 3)

Model Order:
  - All configs use weak model first (agent_0)
  
This design ensures:
  - Statistical significance with 3 runs per configuration
  - Comparable results across model pairs (same seeds for same run number)
  - Different scenarios tested (3 different seeds)
EOF

echo "✅ Created summary: ${SUMMARY_FILE}"

# Create a CSV index for easier analysis
CSV_FILE="${CONFIG_DIR}/experiment_index.csv"
echo "experiment_id,weak_model,strong_model,model_order,competition_level,run_number,seed,config_file" > "${CSV_FILE}"

for config_file in "${CONFIG_DIR}"/config_*.json; do
    if [[ -f "$config_file" ]]; then
        # Extract values from JSON
        exp_id=$(grep -o '"experiment_id": [0-9]*' "$config_file" | grep -o '[0-9]*')
        weak=$(grep -o '"weak_model": "[^"]*"' "$config_file" | cut -d'"' -f4)
        strong=$(grep -o '"strong_model": "[^"]*"' "$config_file" | cut -d'"' -f4)
        order=$(grep -o '"model_order": "[^"]*"' "$config_file" | cut -d'"' -f4)
        comp=$(grep -o '"competition_level": [0-9.]*' "$config_file" | grep -o '[0-9.]*')
        run=$(grep -o '"run_number": [0-9]*' "$config_file" | grep -o '[0-9]*')
        seed=$(grep -o '"random_seed": [0-9]*' "$config_file" | grep -o '[0-9]*')
        
        echo "${exp_id},${weak},${strong},${order},${comp},${run},${seed},$(basename $config_file)" >> "${CSV_FILE}"
    fi
done

echo "✅ Created experiment index: ${CSV_FILE}"
echo ""
echo "To run experiments:"
echo "  - All experiments: Use job IDs 0-$((EXPERIMENT_ID-1))"
echo "  - Specific run only: Filter by run_number in CSV"
echo "  - Specific model pair: Filter by weak_model and strong_model in CSV"
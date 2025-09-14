#!/bin/bash
# Generate configuration files for 3-agent experiments with multiple runs
# This creates configs with 5 runs per model triple/competition level
# Each run has a different seed, but run N has consistent seed across all model triples

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="${BASE_DIR}/experiments/results/3agent_experiment/configs"
mkdir -p "${CONFIG_DIR}"

# Model definitions (same as generate_configs_both_orders.sh)
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
    "gpt-5-nano"
    "gpt-5-mini"
    "o1"
    "o3"
    
    # Google models
    "gemini-2-0-flash"
    "gemini-2-0-flash-lite"
    "gemini-2-5-pro"
)

# Competition levels (same as generate_configs_both_orders.sh)
COMPETITION_LEVELS=(0.0 0.25 0.5 0.75 1.0)

# Base parameters
NUM_ITEMS=5
MAX_ROUNDS=10
NUM_RUNS=5  # Number of runs per configuration

# Seeds for each run (consistent across model triples)
RUN_SEEDS=(42 123 456 789 101112)

echo "Generating 3-agent experiment configurations with multiple runs..."
echo "  Weak models: ${#WEAK_MODELS[@]}"
echo "  Strong models: ${#STRONG_MODELS[@]}"
echo "  Competition levels: ${#COMPETITION_LEVELS[@]}"
echo "  Runs per config: ${NUM_RUNS}"
echo ""

# Counter for experiment ID
EXPERIMENT_ID=0

# Configuration 1: Two weak vs one strong (all combinations)
echo "Generating two-weak-one-strong configs..."
for weak_model1 in "${WEAK_MODELS[@]}"; do
    for weak_model2 in "${WEAK_MODELS[@]}"; do
        for strong_model in "${STRONG_MODELS[@]}"; do
            for comp_level in "${COMPETITION_LEVELS[@]}"; do
                for run_idx in "${!RUN_SEEDS[@]}"; do
                    # Create config file
                    CONFIG_FILE="${CONFIG_DIR}/config_${EXPERIMENT_ID}.json"
                    
                    # Get seed for this run (consistent across all model triples for same run number)
                    SEED=${RUN_SEEDS[$run_idx]}
                    RUN_NUM=$((run_idx + 1))
                    
                    # Write configuration as JSON - TWO WEAK VS ONE STRONG
                    cat > "${CONFIG_FILE}" << EOF
{
    "experiment_id": ${EXPERIMENT_ID},
    "experiment_type": "3agent_2weak_1strong",
    "weak_model1": "${weak_model1}",
    "weak_model2": "${weak_model2}",
    "strong_model": "${strong_model}",
    "models": ["${weak_model1}", "${weak_model2}", "${strong_model}"],
    "n_agents": 3,
    "competition_level": ${comp_level},
    "run_number": ${RUN_NUM},
    "m_items": ${NUM_ITEMS},
    "t_rounds": ${MAX_ROUNDS},
    "random_seed": ${SEED},
    "gamma_discount": 0.9,
    "output_dir": "experiments/results/3agent_experiment/2weak_1strong/${weak_model1}_${weak_model2}_vs_${strong_model}/comp_${comp_level}/run_${RUN_NUM}"
}
EOF
                    
                    EXPERIMENT_ID=$((EXPERIMENT_ID + 1))
                done
            done
        done
    done
done

TWO_WEAK_COUNT=${EXPERIMENT_ID}

# Configuration 2: One weak vs two strong (all combinations)
echo "Generating one-weak-two-strong configs..."
for weak_model in "${WEAK_MODELS[@]}"; do
    for strong_model1 in "${STRONG_MODELS[@]}"; do
        for strong_model2 in "${STRONG_MODELS[@]}"; do
            for comp_level in "${COMPETITION_LEVELS[@]}"; do
                for run_idx in "${!RUN_SEEDS[@]}"; do
                    # Create config file
                    CONFIG_FILE="${CONFIG_DIR}/config_${EXPERIMENT_ID}.json"
                    
                    # Get seed for this run
                    SEED=${RUN_SEEDS[$run_idx]}
                    RUN_NUM=$((run_idx + 1))
                    
                    # Write configuration as JSON - ONE WEAK VS TWO STRONG
                    cat > "${CONFIG_FILE}" << EOF
{
    "experiment_id": ${EXPERIMENT_ID},
    "experiment_type": "3agent_1weak_2strong",
    "weak_model": "${weak_model}",
    "strong_model1": "${strong_model1}",
    "strong_model2": "${strong_model2}",
    "models": ["${weak_model}", "${strong_model1}", "${strong_model2}"],
    "n_agents": 3,
    "competition_level": ${comp_level},
    "run_number": ${RUN_NUM},
    "m_items": ${NUM_ITEMS},
    "t_rounds": ${MAX_ROUNDS},
    "random_seed": ${SEED},
    "gamma_discount": 0.9,
    "output_dir": "experiments/results/3agent_experiment/1weak_2strong/${weak_model}_vs_${strong_model1}_${strong_model2}/comp_${comp_level}/run_${RUN_NUM}"
}
EOF
                    
                    EXPERIMENT_ID=$((EXPERIMENT_ID + 1))
                done
            done
        done
    done
done

ONE_WEAK_COUNT=$((EXPERIMENT_ID - TWO_WEAK_COUNT))
TOTAL_COUNT=${EXPERIMENT_ID}

# Calculate totals for summary
CONFIGS_PER_TYPE_2WEAK=$((${#WEAK_MODELS[@]} * ${#WEAK_MODELS[@]} * ${#STRONG_MODELS[@]} * ${#COMPETITION_LEVELS[@]} * ${NUM_RUNS}))
CONFIGS_PER_TYPE_1WEAK=$((${#WEAK_MODELS[@]} * ${#STRONG_MODELS[@]} * ${#STRONG_MODELS[@]} * ${#COMPETITION_LEVELS[@]} * ${NUM_RUNS}))

echo ""
echo "✅ Generated ${EXPERIMENT_ID} total configuration files:"
echo "   - Two weak vs one strong: ${TWO_WEAK_COUNT} configs"
echo "     (${#WEAK_MODELS[@]} weak × ${#WEAK_MODELS[@]} weak × ${#STRONG_MODELS[@]} strong × ${#COMPETITION_LEVELS[@]} comp × ${NUM_RUNS} runs)"
echo "   - One weak vs two strong: ${ONE_WEAK_COUNT} configs"
echo "     (${#WEAK_MODELS[@]} weak × ${#STRONG_MODELS[@]} strong × ${#STRONG_MODELS[@]} strong × ${#COMPETITION_LEVELS[@]} comp × ${NUM_RUNS} runs)"
echo "   - Location: ${CONFIG_DIR}"

# Create master config list
MASTER_CONFIG="${CONFIG_DIR}/all_configs.txt"
ls -1 "${CONFIG_DIR}"/config_*.json > "${MASTER_CONFIG}"
echo "✅ Created master config list: ${MASTER_CONFIG}"

# Create a detailed summary file
SUMMARY_FILE="${CONFIG_DIR}/summary.txt"
cat > "${SUMMARY_FILE}" << EOF
3-Agent Experiment Configuration Summary (Multiple Runs)
========================================================
Total experiments: ${EXPERIMENT_ID}
  - Two weak vs one strong: ${TWO_WEAK_COUNT}
  - One weak vs two strong: ${ONE_WEAK_COUNT}
  - Competition levels: ${#COMPETITION_LEVELS[@]}
  - Runs per configuration: ${NUM_RUNS}

Weak models (${#WEAK_MODELS[@]}): ${WEAK_MODELS[@]}
Strong models (${#STRONG_MODELS[@]}): ${STRONG_MODELS[@]}
Competition levels (${#COMPETITION_LEVELS[@]}): ${COMPETITION_LEVELS[@]}

Items per negotiation: ${NUM_ITEMS}
Max rounds: ${MAX_ROUNDS}

Random Seeds by Run:
  - Run 1: ${RUN_SEEDS[0]} (all model triples use this for run 1)
  - Run 2: ${RUN_SEEDS[1]} (all model triples use this for run 2)
  - Run 3: ${RUN_SEEDS[2]} (all model triples use this for run 3)
  - Run 4: ${RUN_SEEDS[3]} (all model triples use this for run 4)
  - Run 5: ${RUN_SEEDS[4]} (all model triples use this for run 5)

Configuration Types:
  - 2 weak vs 1 strong: All combinations of 2 weak models against 1 strong model
  - 1 weak vs 2 strong: All combinations of 1 weak model against 2 strong models

This design ensures:
  - Balanced preferences: All pairwise cosine similarities ≈ competition_level
  - Statistical significance with ${NUM_RUNS} runs per configuration
  - Comparable results across model triples (same seeds for same run number)
  - Complete coverage of all possible weak/strong combinations
EOF

echo "✅ Created summary: ${SUMMARY_FILE}"

# Create a CSV index for easier analysis
CSV_FILE="${CONFIG_DIR}/experiment_index.csv"
echo "experiment_id,experiment_type,model1,model2,model3,competition_level,run_number,seed,config_file" > "${CSV_FILE}"

for config_file in "${CONFIG_DIR}"/config_*.json; do
    if [[ -f "$config_file" ]]; then
        # Extract values from JSON
        exp_id=$(grep -o '"experiment_id": [0-9]*' "$config_file" | grep -o '[0-9]*')
        exp_type=$(grep -o '"experiment_type": "[^"]*"' "$config_file" | cut -d'"' -f4)
        models=$(grep -o '"models": \[[^]]*\]' "$config_file" | sed 's/"models": \[//;s/\]//;s/"//g')
        model1=$(echo "$models" | cut -d',' -f1 | tr -d ' "')
        model2=$(echo "$models" | cut -d',' -f2 | tr -d ' "')
        model3=$(echo "$models" | cut -d',' -f3 | tr -d ' "')
        comp=$(grep -o '"competition_level": [0-9.]*' "$config_file" | grep -o '[0-9.]*')
        run=$(grep -o '"run_number": [0-9]*' "$config_file" | grep -o '[0-9]*')
        seed=$(grep -o '"random_seed": [0-9]*' "$config_file" | grep -o '[0-9]*')
        
        echo "${exp_id},${exp_type},${model1},${model2},${model3},${comp},${run},${seed},$(basename $config_file)" >> "${CSV_FILE}"
    fi
done

echo "✅ Created experiment index: ${CSV_FILE}"
echo ""
echo "To run experiments:"
echo "  - All experiments: Use job IDs 0-$((EXPERIMENT_ID-1))"
echo "  - Specific run only: Filter by run_number in CSV"
echo "  - Specific model triple: Filter by model1, model2, model3 in CSV"
echo "  - Specific type: Filter by experiment_type (3agent_2weak_1strong or 3agent_1weak_2strong)"
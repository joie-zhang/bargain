#!/bin/bash
# Generate a single experiment configuration file
# Usage: ./scripts/generate_single_config.sh [model1] [model2] [competition_level] [seed]

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="${BASE_DIR}/experiments/results/scaling_experiment/configs"
mkdir -p "${CONFIG_DIR}"

# Default parameters (can be overridden via command line)
MODEL1=${1:-"claude-3-5-haiku"}
MODEL2=${2:-"claude-3-5-haiku"}
COMP_LEVEL=${3:-1.0}
SEED=${4:-42}
NUM_ITEMS=${5:-5}
MAX_ROUNDS=${6:-10}

# Use a fixed job ID for single config (0)
JOB_ID=0

CONFIG_FILE="${CONFIG_DIR}/config_${JOB_ID}.json"

echo "Generating single experiment configuration..."
echo "  Model 1: ${MODEL1}"
echo "  Model 2: ${MODEL2}"
echo "  Competition Level: ${COMP_LEVEL}"
echo "  Random Seed: ${SEED}"
echo "  Items: ${NUM_ITEMS}"
echo "  Max Rounds: ${MAX_ROUNDS}"
echo "  Config File: ${CONFIG_FILE}"
echo ""

# Write configuration as JSON
# Note: model_order is metadata only (not used by experiment runner):
#   - "weak_first": weak model is agent_0 (standard in batch experiments)
#   - "strong_first": strong model is agent_0 (for testing order effects)
#   For same-model experiments, we use "weak_first" for consistency
cat > "${CONFIG_FILE}" << EOF
{
    "experiment_id": ${JOB_ID},
    "weak_model": "${MODEL1}",
    "strong_model": "${MODEL2}",
    "models": ["${MODEL1}", "${MODEL2}"],
    "model_order": "weak_first",
    "competition_level": ${COMP_LEVEL},
    "run_number": 1,
    "num_items": ${NUM_ITEMS},
    "max_rounds": ${MAX_ROUNDS},
    "random_seed": ${SEED},
    "output_dir": "experiments/results/scaling_experiment/${MODEL1}_vs_${MODEL2}/test/comp_${COMP_LEVEL}/run_1"
}
EOF

echo "✅ Generated configuration file: ${CONFIG_FILE}"
echo ""

echo ""
echo "Or view the config:"
echo "  cat ${CONFIG_FILE} | python3 -m json.tool"

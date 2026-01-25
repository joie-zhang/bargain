#!/bin/bash
# =============================================================================
# Generate Configuration Files for Test-Time Compute Scaling Experiments
# =============================================================================
#
# This script generates JSON config files for TTC scaling experiments:
#   - Reasoning models vs GPT-5-nano baseline
#   - Variable reasoning token budgets (prompted, not API-enforced)
#   - Multiple competition levels and model orders
#
# Usage:
#   ./scripts/generate_ttc_configs.sh                    # Full experiment (720 configs)
#   ./scripts/generate_ttc_configs.sh --derisk           # Small test run (1 config)
#   ./scripts/generate_ttc_configs.sh --small            # Reduced experiment (~60 configs)
#
# What it creates:
#   experiments/results/ttc_scaling_<timestamp>/configs/
#   ├── config_0000.json ... config_NNNN.json   # Individual experiment configs
#   ├── all_configs.txt                          # List of all config files
#   ├── experiment_index.csv                     # Searchable index of experiments
#   ├── summary.txt                              # Human-readable summary
#   └── slurm/
#       ├── run_ttc_experiments.sbatch           # SLURM script
#       └── submit_all.sh                        # Submit script
#
# Configuration:
#   Edit the arrays below to modify:
#   - REASONING_MODELS: Models to test
#   - TOKEN_BUDGETS: Reasoning token budgets to try
#   - COMPETITION_LEVELS: Competition parameter values
#   - MODEL_ORDERS: weak_first, strong_first
#
# =============================================================================

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Parse arguments
MODE="full"
while [[ $# -gt 0 ]]; do
    case $1 in
        --derisk)
            MODE="derisk"
            shift
            ;;
        --small)
            MODE="small"
            shift
            ;;
        --batch4)
            MODE="batch4"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--derisk|--small|--batch4]"
            echo ""
            echo "Options:"
            echo "  --derisk    Minimal test: 1 model, 1 budget, 1 competition level (1 config)"
            echo "  --small     Reduced test: 2 models, 3 budgets, 2 competition levels (~24 configs)"
            echo "  --batch4    Focused experiment: 3 models (Claude/O3/GPT-5.2), 5 budgets, comp=1.0 (30 configs)"
            echo "  (default)   Full experiment: all combinations (720 configs)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create timestamped config directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TTC_EXPERIMENT_DIR="${BASE_DIR}/experiments/results/ttc_scaling_${TIMESTAMP}"
CONFIG_DIR="${TTC_EXPERIMENT_DIR}/configs"
mkdir -p "${CONFIG_DIR}"

echo "Creating TTC scaling experiment directory: ${TTC_EXPERIMENT_DIR}"
echo "Config directory: ${CONFIG_DIR}"
echo "Mode: ${MODE}"
echo ""

# =============================================================================
# Model and Parameter Definitions
# =============================================================================

# Baseline model (non-reasoning)
BASELINE_MODEL="gpt-5-nano"

# Reasoning models to test
if [[ "$MODE" == "derisk" ]]; then
    # Minimal derisk: just one model
    REASONING_MODELS=(
        "claude-opus-4-5-thinking-32k"
    )
elif [[ "$MODE" == "small" ]]; then
    # Small test: 2 models
    REASONING_MODELS=(
        "claude-opus-4-5-thinking-32k"
        "o3-mini-high"
    )
elif [[ "$MODE" == "batch4" ]]; then
    # Batch 4: Focused on API-controllable models
    REASONING_MODELS=(
        "claude-opus-4-5-thinking-32k"  # Anthropic - thinking.budget_tokens
        "o3-mini-high"                   # OpenAI - reasoning_effort
        "gpt-5.2-high"                   # OpenAI - reasoning_effort
    )
else
    # Full experiment: all 6 reasoning models
    REASONING_MODELS=(
        "claude-opus-4-5-thinking-32k"  # Anthropic extended thinking
        "gpt-5.2-high"                   # OpenAI reasoning
        "o3-mini-high"                   # OpenAI O3
        "grok-4"                         # XAI
        "deepseek-r1"                    # OpenRouter DeepSeek R1
        "QwQ-32B"                        # Princeton cluster local
    )
fi

# Reasoning token budgets (API-enforced where supported)
if [[ "$MODE" == "derisk" ]]; then
    TOKEN_BUDGETS=(100)
elif [[ "$MODE" == "small" ]]; then
    TOKEN_BUDGETS=(100 1000 5000)
elif [[ "$MODE" == "batch4" ]]; then
    TOKEN_BUDGETS=(100 500 1000 5000 10000)
else
    TOKEN_BUDGETS=(100 500 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000)
fi

# Competition levels
if [[ "$MODE" == "derisk" ]]; then
    COMPETITION_LEVELS=(1.0)
elif [[ "$MODE" == "small" ]]; then
    COMPETITION_LEVELS=(0.0 1.0)
elif [[ "$MODE" == "batch4" ]]; then
    COMPETITION_LEVELS=(1.0)  # Fully competitive only
else
    COMPETITION_LEVELS=(0.0 0.25 0.5 0.75 1.0)
fi

# Model orders
if [[ "$MODE" == "derisk" ]]; then
    MODEL_ORDERS=("weak_first")
elif [[ "$MODE" == "small" ]]; then
    MODEL_ORDERS=("weak_first" "strong_first")
elif [[ "$MODE" == "batch4" ]]; then
    MODEL_ORDERS=("weak_first" "strong_first")  # Both orders
else
    MODEL_ORDERS=("weak_first" "strong_first")
fi

# Game configuration
NUM_ITEMS=5
MAX_ROUNDS=10
GAMMA_DISCOUNT=0.9
DISCUSSION_TURNS=3

# Phases to apply reasoning budget instruction
REASONING_PHASES="thinking,reflection"

# Max tokens per phase (high to allow full reasoning)
MAX_TOKENS_PER_PHASE=10500

# Base seed
BASE_SEED=42

echo "Generating TTC scaling experiment configurations..."
echo "  Reasoning models: ${#REASONING_MODELS[@]}"
echo "  Baseline model: ${BASELINE_MODEL}"
echo "  Token budgets: ${#TOKEN_BUDGETS[@]}"
echo "  Competition levels: ${#COMPETITION_LEVELS[@]}"
echo "  Model orders: ${#MODEL_ORDERS[@]}"
echo "  Total configs: $((${#REASONING_MODELS[@]} * ${#TOKEN_BUDGETS[@]} * ${#COMPETITION_LEVELS[@]} * ${#MODEL_ORDERS[@]}))"
echo ""

# Calculate total number of experiments for zero-padding
TOTAL_EXPERIMENTS=$((${#REASONING_MODELS[@]} * ${#TOKEN_BUDGETS[@]} * ${#COMPETITION_LEVELS[@]} * ${#MODEL_ORDERS[@]}))
PADDING_WIDTH=${#TOTAL_EXPERIMENTS}
if [[ $PADDING_WIDTH -lt 4 ]]; then
    PADDING_WIDTH=4
fi

# Counter for experiment ID
EXPERIMENT_ID=0

# Generate configs
for reasoning_model in "${REASONING_MODELS[@]}"; do
    for token_budget in "${TOKEN_BUDGETS[@]}"; do
        for comp_level in "${COMPETITION_LEVELS[@]}"; do
            for model_order in "${MODEL_ORDERS[@]}"; do
                # Create config file with zero-padded experiment ID
                EXPERIMENT_ID_PADDED=$(printf "%0${PADDING_WIDTH}d" ${EXPERIMENT_ID})
                CONFIG_FILE="${CONFIG_DIR}/config_${EXPERIMENT_ID_PADDED}.json"

                # Determine model order in array
                if [[ "$model_order" == "weak_first" ]]; then
                    MODELS_ARRAY="[\"${BASELINE_MODEL}\", \"${reasoning_model}\"]"
                else
                    MODELS_ARRAY="[\"${reasoning_model}\", \"${BASELINE_MODEL}\"]"
                fi

                # Compute seed
                SEED=$((BASE_SEED + EXPERIMENT_ID))

                # Format competition level without trailing zeros for directory name
                COMP_STR=$(echo "${comp_level}" | sed 's/\./_/g')

                # Write configuration as JSON
                cat > "${CONFIG_FILE}" << EOF
{
    "experiment_id": ${EXPERIMENT_ID},
    "reasoning_model": "${reasoning_model}",
    "baseline_model": "${BASELINE_MODEL}",
    "models": ${MODELS_ARRAY},
    "model_order": "${model_order}",
    "reasoning_token_budget": ${token_budget},
    "reasoning_budget_phases": ["thinking", "reflection"],
    "max_tokens_per_phase": ${MAX_TOKENS_PER_PHASE},
    "competition_level": ${comp_level},
    "num_items": ${NUM_ITEMS},
    "max_rounds": ${MAX_ROUNDS},
    "gamma_discount": ${GAMMA_DISCOUNT},
    "discussion_turns": ${DISCUSSION_TURNS},
    "random_seed": ${SEED},
    "output_dir": "experiments/results/ttc_scaling_${TIMESTAMP}/${reasoning_model}_vs_${BASELINE_MODEL}/${model_order}/budget_${token_budget}/comp_${COMP_STR}"
}
EOF

                EXPERIMENT_ID=$((EXPERIMENT_ID + 1))
            done
        done
    done
done

TOTAL_COUNT=${EXPERIMENT_ID}

# Create symlink to latest experiment
TTC_SYMLINK="${BASE_DIR}/experiments/results/ttc_scaling"
if [[ -L "${TTC_SYMLINK}" ]]; then
    rm "${TTC_SYMLINK}"
elif [[ -d "${TTC_SYMLINK}" ]] && [[ ! -L "${TTC_SYMLINK}" ]]; then
    OLD_DIR="${TTC_SYMLINK}_old_$(date +%Y%m%d_%H%M%S)"
    mv "${TTC_SYMLINK}" "${OLD_DIR}"
    echo "Moved existing directory to: ${OLD_DIR}"
fi
ln -sf "ttc_scaling_${TIMESTAMP}" "${TTC_SYMLINK}"
echo "✅ Created symlink: ${TTC_SYMLINK} -> ttc_scaling_${TIMESTAMP}"

echo ""
echo "✅ Generated ${TOTAL_COUNT} configuration files"
echo "   Location: ${CONFIG_DIR}"

# Create master config list
MASTER_CONFIG="${CONFIG_DIR}/all_configs.txt"
ls -1 "${CONFIG_DIR}"/config_*.json > "${MASTER_CONFIG}"
echo "✅ Created master config list: ${MASTER_CONFIG}"

# Create summary file
SUMMARY_FILE="${CONFIG_DIR}/summary.txt"
cat > "${SUMMARY_FILE}" << EOF
Test-Time Compute Scaling Experiment Configuration Summary
===========================================================
Mode: ${MODE}
Total experiments: ${TOTAL_COUNT}
  - Reasoning models: ${#REASONING_MODELS[@]}
  - Token budgets: ${#TOKEN_BUDGETS[@]}
  - Competition levels: ${#COMPETITION_LEVELS[@]}
  - Model orders: ${#MODEL_ORDERS[@]}

Baseline model: ${BASELINE_MODEL}
Reasoning models: ${REASONING_MODELS[@]}
Token budgets: ${TOKEN_BUDGETS[@]}
Competition levels: ${COMPETITION_LEVELS[@]}
Model orders: ${MODEL_ORDERS[@]}

Game configuration:
  Items per negotiation: ${NUM_ITEMS}
  Max rounds: ${MAX_ROUNDS}
  Gamma discount: ${GAMMA_DISCOUNT}
  Discussion turns: ${DISCUSSION_TURNS}

Reasoning configuration:
  Phases with budget instruction: ${REASONING_PHASES}
  Max tokens per phase: ${MAX_TOKENS_PER_PHASE}
  (Budget is prompted, not API-enforced)

Key analysis variable:
  X-axis: Actual reasoning tokens used (from API response)
  Y-axis: Normalized utility
EOF
echo "✅ Created summary: ${SUMMARY_FILE}"

# Create CSV index
CSV_FILE="${CONFIG_DIR}/experiment_index.csv"
echo "experiment_id,reasoning_model,baseline_model,model_order,token_budget,competition_level,seed,config_file" > "${CSV_FILE}"

for config_file in "${CONFIG_DIR}"/config_*.json; do
    if [[ -f "$config_file" ]]; then
        exp_id=$(grep -o '"experiment_id": [0-9]*' "$config_file" | grep -o '[0-9]*')
        reasoning=$(grep -o '"reasoning_model": "[^"]*"' "$config_file" | cut -d'"' -f4)
        baseline=$(grep -o '"baseline_model": "[^"]*"' "$config_file" | cut -d'"' -f4)
        order=$(grep -o '"model_order": "[^"]*"' "$config_file" | cut -d'"' -f4)
        budget=$(grep -o '"reasoning_token_budget": [0-9]*' "$config_file" | grep -o '[0-9]*')
        comp=$(grep -o '"competition_level": [0-9.]*' "$config_file" | grep -o '[0-9.]*')
        seed=$(grep -o '"random_seed": [0-9]*' "$config_file" | grep -o '[0-9]*')

        echo "${exp_id},${reasoning},${baseline},${order},${budget},${comp},${seed},$(basename $config_file)" >> "${CSV_FILE}"
    fi
done
echo "✅ Created experiment index: ${CSV_FILE}"

# =============================================================================
# SLURM Script Generation
# =============================================================================
SLURM_DIR="${CONFIG_DIR}/slurm"
mkdir -p "${SLURM_DIR}"

echo ""
echo "Generating SLURM scripts..."

# Use absolute path to config directory
CONFIG_DIR_ABSOLUTE="${TTC_EXPERIMENT_DIR}/configs"

# Generate SLURM script for TTC experiments
TTC_SLURM="${SLURM_DIR}/run_ttc_experiments.sbatch"
cat > "${TTC_SLURM}" << SLURM_TTC
#!/bin/bash
#SBATCH --job-name=ttc-scaling
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/cluster/ttc_%A_%a.out
#SBATCH --error=logs/cluster/ttc_%A_%a.err

set -e

BASE_DIR="/scratch/gpfs/DANQIC/jz4391/bargain"
cd "\${BASE_DIR}"
mkdir -p logs/cluster

echo "============================================================"
echo "TTC Scaling Experiment"
echo "SLURM Job ID: \$SLURM_JOB_ID, Array Task ID: \$SLURM_ARRAY_TASK_ID"
echo "Started at: \$(date)"
echo "Node: \$SLURM_NODELIST"
echo "============================================================"

# Load modules
module purge
module load anaconda3/2024.2
module load proxy/default

# Activate virtual environment
source "\${BASE_DIR}/.venv/bin/activate"
echo "Python version: \$(python3 --version)"
echo ""

# Get config file for this array task
CONFIG_DIR="${CONFIG_DIR_ABSOLUTE}"
MAX_CONFIG=\$(ls "\${CONFIG_DIR}"/config_*.json 2>/dev/null | sed 's/.*config_\\([0-9]*\\)\\.json/\\1/' | sort -n | tail -1)
PADDING_WIDTH=\${#MAX_CONFIG}
CONFIG_ID_PADDED=\$(printf "%0\${PADDING_WIDTH}d" \${SLURM_ARRAY_TASK_ID})
CONFIG_FILE="\${CONFIG_DIR}/config_\${CONFIG_ID_PADDED}.json"

if [[ ! -f "\$CONFIG_FILE" ]]; then
    echo "ERROR: Config file not found: \$CONFIG_FILE"
    exit 1
fi

echo "Config file: \$CONFIG_FILE"

# Extract config values
REASONING_MODEL=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['reasoning_model'])")
BASELINE_MODEL=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['baseline_model'])")
MODEL_ORDER=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['model_order'])")
TOKEN_BUDGET=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['reasoning_token_budget'])")
COMP_LEVEL=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['competition_level'])")
SEED=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['random_seed'])")
DISCUSSION_TURNS=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['discussion_turns'])")
OUTPUT_DIR=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['output_dir'])")
MAX_TOKENS=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['max_tokens_per_phase'])")

# Get models in correct order
if [[ "\$MODEL_ORDER" == "weak_first" ]]; then
    MODELS="\$BASELINE_MODEL \$REASONING_MODEL"
else
    MODELS="\$REASONING_MODEL \$BASELINE_MODEL"
fi

echo "Reasoning model: \$REASONING_MODEL"
echo "Baseline model: \$BASELINE_MODEL"
echo "Model order: \$MODEL_ORDER"
echo "Models: \$MODELS"
echo "Token budget: \$TOKEN_BUDGET"
echo "Competition level: \$COMP_LEVEL"
echo "Random seed: \$SEED"
echo "Output dir: \$OUTPUT_DIR"
echo ""

# Run experiment with reasoning token budget
echo "Running: python3 run_strong_models_experiment.py ..."
echo ""

if python3 run_strong_models_experiment.py \\
    --models \$MODELS \\
    --batch \\
    --num-runs 1 \\
    --run-number 1 \\
    --competition-level \$COMP_LEVEL \\
    --random-seed \$SEED \\
    --discussion-turns \$DISCUSSION_TURNS \\
    --model-order \$MODEL_ORDER \\
    --reasoning-token-budget \$TOKEN_BUDGET \\
    --reasoning-budget-phases thinking reflection \\
    --max-tokens-per-phase \$MAX_TOKENS \\
    --output-dir "\$OUTPUT_DIR" \\
    --job-id \$SLURM_ARRAY_TASK_ID; then
    echo ""
    echo "============================================================"
    echo "✅ Experiment completed successfully at: \$(date)"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Experiment failed at: \$(date)"
    echo "============================================================"
    exit 1
fi
SLURM_TTC

echo "✅ Created SLURM script: ${TTC_SLURM}"

# Create submission script
SUBMIT_SCRIPT="${SLURM_DIR}/submit_all.sh"
cat > "${SUBMIT_SCRIPT}" << 'SUBMIT_EOF'
#!/bin/bash
# Submit TTC scaling experiment jobs
# Usage: ./submit_all.sh [--test] [--max-concurrent <num>]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${BASE_DIR}"

# Parse arguments
MAX_CONCURRENT=""
TEST_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_MODE=true
            shift
            ;;
        --max-concurrent)
            MAX_CONCURRENT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

mkdir -p logs/cluster

# Count configs
CONFIG_DIR="${SCRIPT_DIR}/.."
TOTAL_CONFIGS=$(ls "${CONFIG_DIR}"/config_*.json 2>/dev/null | wc -l)

if [ "$TOTAL_CONFIGS" -eq 0 ]; then
    echo "Error: No config files found"
    exit 1
fi

echo "Total configurations: ${TOTAL_CONFIGS}"

# Build array spec
if [[ "$TEST_MODE" == "true" ]]; then
    ARRAY_SPEC="0"
    echo "Test mode: submitting only config 0"
else
    ARRAY_SPEC="0-$((TOTAL_CONFIGS - 1))"
fi

if [[ -n "$MAX_CONCURRENT" ]]; then
    ARRAY_SPEC="${ARRAY_SPEC}%${MAX_CONCURRENT}"
    echo "Max concurrent jobs: ${MAX_CONCURRENT}"
fi

echo "Submitting array: ${ARRAY_SPEC}"
sbatch --array="${ARRAY_SPEC}" "${SCRIPT_DIR}/run_ttc_experiments.sbatch"

echo ""
echo "Jobs submitted. Monitor with: squeue -u $USER"
SUBMIT_EOF
chmod +x "${SUBMIT_SCRIPT}"

echo "✅ Created submission script: ${SUBMIT_SCRIPT}"

# Create a local run script for derisk testing
LOCAL_RUN_SCRIPT="${SLURM_DIR}/run_local.sh"
cat > "${LOCAL_RUN_SCRIPT}" << 'LOCAL_EOF'
#!/bin/bash
# Run a single TTC experiment locally (for testing)
# Usage: ./run_local.sh [config_id]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${BASE_DIR}"

CONFIG_ID="${1:-0}"
CONFIG_DIR="${SCRIPT_DIR}/.."

# Find config file
MAX_CONFIG=$(ls "${CONFIG_DIR}"/config_*.json 2>/dev/null | sed 's/.*config_\([0-9]*\)\.json/\1/' | sort -n | tail -1)
PADDING_WIDTH=${#MAX_CONFIG}
CONFIG_ID_PADDED=$(printf "%0${PADDING_WIDTH}d" ${CONFIG_ID})
CONFIG_FILE="${CONFIG_DIR}/config_${CONFIG_ID_PADDED}.json"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    echo "Available configs:"
    ls "${CONFIG_DIR}"/config_*.json | head -5
    exit 1
fi

echo "Running config: $CONFIG_FILE"
echo ""

# Source virtual environment
source "${BASE_DIR}/.venv/bin/activate"

# Extract config values
REASONING_MODEL=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['reasoning_model'])")
BASELINE_MODEL=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['baseline_model'])")
MODEL_ORDER=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['model_order'])")
TOKEN_BUDGET=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['reasoning_token_budget'])")
COMP_LEVEL=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['competition_level'])")
SEED=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['random_seed'])")
DISCUSSION_TURNS=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['discussion_turns'])")
OUTPUT_DIR=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['output_dir'])")
MAX_TOKENS=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['max_tokens_per_phase'])")

# Get models in correct order
if [[ "$MODEL_ORDER" == "weak_first" ]]; then
    MODELS="$BASELINE_MODEL $REASONING_MODEL"
else
    MODELS="$REASONING_MODEL $BASELINE_MODEL"
fi

echo "Reasoning model: $REASONING_MODEL"
echo "Baseline model: $BASELINE_MODEL"
echo "Model order: $MODEL_ORDER"
echo "Token budget: $TOKEN_BUDGET"
echo "Competition level: $COMP_LEVEL"
echo ""

# Run experiment
python3 run_strong_models_experiment.py \
    --models $MODELS \
    --batch \
    --num-runs 1 \
    --run-number 1 \
    --competition-level $COMP_LEVEL \
    --random-seed $SEED \
    --discussion-turns $DISCUSSION_TURNS \
    --model-order $MODEL_ORDER \
    --reasoning-token-budget $TOKEN_BUDGET \
    --reasoning-budget-phases thinking reflection \
    --max-tokens-per-phase $MAX_TOKENS \
    --output-dir "$OUTPUT_DIR" \
    --job-id $CONFIG_ID
LOCAL_EOF
chmod +x "${LOCAL_RUN_SCRIPT}"

echo "✅ Created local run script: ${LOCAL_RUN_SCRIPT}"

echo ""
echo "============================================================"
echo "Configuration generation complete!"
echo "============================================================"
echo ""
echo "Generated: ${TOTAL_COUNT} experiment configurations"
echo "Location:  ${CONFIG_DIR}"
echo ""
echo "To run the derisk test locally:"
echo "  ${LOCAL_RUN_SCRIPT} 0"
echo ""
echo "To submit to SLURM:"
echo "  ${SUBMIT_SCRIPT}              # Submit all jobs"
echo "  ${SUBMIT_SCRIPT} --test       # Submit only config 0"
echo "  ${SUBMIT_SCRIPT} --max-concurrent 5  # Limit concurrency"
echo ""

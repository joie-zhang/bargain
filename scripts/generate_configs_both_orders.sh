#!/bin/bash
# =============================================================================
# Generate Configuration Files for Negotiation Experiments
# =============================================================================
#
# This script generates JSON config files for all experiment combinations:
#   - Weak models x Strong models x Competition levels x Runs
#   - Balanced ordering (half weak_first, half strong_first)
#   - Consistent seeds across model pairs for reproducibility
#
# Usage:
#   ./scripts/generate_configs_both_orders.sh
#
# What it creates:
#   experiments/results/scaling_experiment/configs/
#   ├── config_0.json ... config_N.json   # Individual experiment configs
#   ├── all_configs.txt                    # List of all config files
#   ├── experiment_index.csv               # Searchable index of experiments
#   ├── summary.txt                        # Human-readable summary
#   └── slurm/
#       ├── run_api_experiments.sbatch     # SLURM script for API models
#       ├── run_gpu_experiments.sbatch     # SLURM script for local GPU models
#       ├── submit_all.sh                  # Submit all jobs
#       └── submit_single.sh               # Submit/run single experiment
#
# After running this script:
#   # List available configs
#   ./experiments/results/scaling_experiment/configs/slurm/submit_single.sh --list
#
#   # Test a single experiment locally
#   ./experiments/results/scaling_experiment/configs/slurm/submit_single.sh 0 --local
#
#   # Submit a single experiment to SLURM
#   ./experiments/results/scaling_experiment/configs/slurm/submit_single.sh 0
#
#   # Submit all experiments to SLURM
#   ./experiments/results/scaling_experiment/configs/slurm/submit_all.sh all
#
# To modify experiments, edit:
#   - WEAK_MODELS array: baseline models
#   - STRONG_MODELS array: models to test against baseline
#   - COMPETITION_LEVELS array: competition parameter values
#   - NUM_RUNS: number of runs per configuration
#   - RUN_SEEDS: random seeds for reproducibility
#
# =============================================================================

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Create timestamped config directory to avoid overwriting previous experiments
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCALING_EXPERIMENT_DIR="${BASE_DIR}/experiments/results/scaling_experiment_${TIMESTAMP}"
CONFIG_DIR="${SCALING_EXPERIMENT_DIR}/configs"
mkdir -p "${CONFIG_DIR}"

echo "Creating timestamped scaling experiment directory: ${SCALING_EXPERIMENT_DIR}"
echo "Config directory: ${CONFIG_DIR}"
echo "This ensures previous experiment configs are preserved."
echo ""

# Model definitions
# NOTE: These short names map to full model_id values defined in:
#       strong_models_experiment/configs.py (STRONG_MODELS_CONFIG dictionary)
#       The Python code looks up these short names to get the full model_id (e.g., "gpt-5-mini" -> "gpt-5-mini-2025-08-07")
#
# Weak models - baseline models for exploitation experiments
# From Multi-Agent Strategic Games Evaluation Models (Weak Tier - Elo < 1290)
WEAK_MODELS=(
    gpt-5-nano
    # "gpt-4o" # May 2024 version
    # "gemini-1-5-pro"
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
# COMPETITION_LEVELS=(0.0 0.25 0.5 0.75 1.0)
COMPETITION_LEVELS=(0.0)

# Base parameters
NUM_ITEMS=5
MAX_ROUNDS=10
# NUM_RUNS=6  # Number of runs per configuration
NUM_RUNS=1  # Number of runs per configuration
DISCUSSION_TURNS=3  # Number of discussion exchanges per round

# Seeds for each scenario (used for both weak_first and strong_first orderings)
# With balanced ordering, runs 1-3 (weak_first) and runs 4-6 (strong_first) use the same seeds
# This allows direct comparison of order effects while keeping scenarios consistent
SCENARIO_SEEDS=(42 123 456)

# Balanced ordering: split runs between weak_first and strong_first
# weak_first gets ceiling(NUM_RUNS/2), strong_first gets floor(NUM_RUNS/2)
WEAK_FIRST_RUNS=$(( (NUM_RUNS + 1) / 2 ))
STRONG_FIRST_RUNS=$(( NUM_RUNS / 2 ))

# Validate that we have enough seeds for the number of scenarios
if [ ${#SCENARIO_SEEDS[@]} -lt $WEAK_FIRST_RUNS ]; then
    echo "Error: Not enough scenario seeds. Need at least ${WEAK_FIRST_RUNS} seeds for ${NUM_RUNS} runs."
    exit 1
fi

echo "Generating experiment configurations with multiple runs..."
echo "  Weak models: ${#WEAK_MODELS[@]}"
echo "  Strong models: ${#STRONG_MODELS[@]}"
echo "  Competition levels: ${#COMPETITION_LEVELS[@]}"
echo "  Runs per config: ${NUM_RUNS} (${WEAK_FIRST_RUNS} weak_first + ${STRONG_FIRST_RUNS} strong_first)"
echo "  Discussion turns: ${DISCUSSION_TURNS}"
echo "  Total configs: $((${#WEAK_MODELS[@]} * ${#STRONG_MODELS[@]} * ${#COMPETITION_LEVELS[@]} * ${NUM_RUNS}))"
echo ""

# Calculate total number of experiments for zero-padding
TOTAL_EXPERIMENTS=$((${#WEAK_MODELS[@]} * ${#STRONG_MODELS[@]} * ${#COMPETITION_LEVELS[@]} * ${NUM_RUNS}))
# Calculate padding width (number of digits needed)
PADDING_WIDTH=${#TOTAL_EXPERIMENTS}

# Counter for experiment ID
EXPERIMENT_ID=0

# Generate configs with balanced ordering: first WEAK_FIRST_RUNS use weak_first, rest use strong_first
echo "Generating configs with balanced ordering..."
echo "Note: Runs 1-${WEAK_FIRST_RUNS} (weak_first) and runs $((WEAK_FIRST_RUNS + 1))-${NUM_RUNS} (strong_first) use the same seeds for direct order comparison."
for weak_model in "${WEAK_MODELS[@]}"; do
    for strong_model in "${STRONG_MODELS[@]}"; do
        for comp_level in "${COMPETITION_LEVELS[@]}"; do
            for run_idx in $(seq 0 $((NUM_RUNS - 1))); do
                # Create config file with zero-padded experiment ID
                EXPERIMENT_ID_PADDED=$(printf "%0${PADDING_WIDTH}d" ${EXPERIMENT_ID})
                CONFIG_FILE="${CONFIG_DIR}/config_${EXPERIMENT_ID_PADDED}.json"

                # Determine model order: first WEAK_FIRST_RUNS use weak_first, rest use strong_first
                if [ $run_idx -lt $WEAK_FIRST_RUNS ]; then
                    MODEL_ORDER="weak_first"
                    MODELS_ARRAY="[\"${weak_model}\", \"${strong_model}\"]"
                    OUTPUT_SUBDIR="weak_first"
                    # For weak_first runs (0, 1, 2), use seeds 0, 1, 2
                    SEED_IDX=$run_idx
                else
                    MODEL_ORDER="strong_first"
                    MODELS_ARRAY="[\"${strong_model}\", \"${weak_model}\"]"
                    OUTPUT_SUBDIR="strong_first"
                    # For strong_first runs (3, 4, 5), reuse seeds 0, 1, 2
                    # This makes run 4 use same seed as run 1, run 5 same as run 2, etc.
                    SEED_IDX=$((run_idx - WEAK_FIRST_RUNS))
                fi

                # Get seed for this scenario (same seed used for both orderings of the same scenario)
                SEED=${SCENARIO_SEEDS[$SEED_IDX]}
                RUN_NUM=$((run_idx + 1))

                # Write configuration as JSON
                cat > "${CONFIG_FILE}" << EOF
{
    "experiment_id": ${EXPERIMENT_ID},
    "weak_model": "${weak_model}",
    "strong_model": "${strong_model}",
    "models": ${MODELS_ARRAY},
    "model_order": "${MODEL_ORDER}",
    "competition_level": ${comp_level},
    "run_number": ${RUN_NUM},
    "num_items": ${NUM_ITEMS},
    "max_rounds": ${MAX_ROUNDS},
    "random_seed": ${SEED},
    "discussion_turns": ${DISCUSSION_TURNS},
    "output_dir": "experiments/results/scaling_experiment/${weak_model}_vs_${strong_model}/${OUTPUT_SUBDIR}/comp_${comp_level}/run_${RUN_NUM}"
}
EOF

                EXPERIMENT_ID=$((EXPERIMENT_ID + 1))
            done
        done
    done
done

TOTAL_COUNT=${EXPERIMENT_ID}

# Create symlink to latest configs directory for easy access
# This allows SLURM scripts to reference a consistent path
CONFIGS_SYMLINK="${BASE_DIR}/experiments/results/scaling_experiment/configs"
SYMLINK_PARENT_DIR="$(dirname "${CONFIGS_SYMLINK}")"
if [[ -L "${CONFIGS_SYMLINK}" ]]; then
    # Remove old symlink
    rm "${CONFIGS_SYMLINK}"
elif [[ -d "${CONFIGS_SYMLINK}" ]] && [[ ! -L "${CONFIGS_SYMLINK}" ]]; then
    # If it's a real directory, rename it to preserve it
    OLD_DIR="${CONFIGS_SYMLINK}_old_$(date +%Y%m%d_%H%M%S)"
    echo "Warning: ${CONFIGS_SYMLINK} exists as a directory."
    echo "         Moving it to ${OLD_DIR} to preserve it."
    mv "${CONFIGS_SYMLINK}" "${OLD_DIR}"
fi
# Create symlink pointing to the timestamped configs directory
# Use relative path from symlink location
ln -sf "../scaling_experiment_${TIMESTAMP}/configs" "${CONFIGS_SYMLINK}"
echo "✅ Created symlink: ${CONFIGS_SYMLINK} -> ../scaling_experiment_${TIMESTAMP}/configs"

echo ""
echo "✅ Generated ${EXPERIMENT_ID} total configuration files:"
echo "   - Model pairs: $((${#WEAK_MODELS[@]} * ${#STRONG_MODELS[@]}))"
echo "   - Competition levels: ${#COMPETITION_LEVELS[@]}"
echo "   - Runs per config: ${NUM_RUNS}"
echo "   - Location: ${CONFIG_DIR}"
echo "   - Symlink: ${CONFIGS_SYMLINK}"

# Create master config list (always regenerate to ensure it's current)
MASTER_CONFIG="${CONFIG_DIR}/all_configs.txt"
ls -1 "${CONFIG_DIR}"/config_*.json > "${MASTER_CONFIG}"
echo "✅ Created/updated master config list: ${MASTER_CONFIG}"

# Create a detailed summary file
SUMMARY_FILE="${CONFIG_DIR}/summary.txt"
cat > "${SUMMARY_FILE}" << EOF
Experiment Configuration Summary (Balanced Ordering)
=====================================================
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
Discussion turns per round: ${DISCUSSION_TURNS}

Random Seeds by Scenario (same seed used for both orderings):
  - Scenario 1: Seed ${SCENARIO_SEEDS[0]} (Run 1 weak_first, Run $((WEAK_FIRST_RUNS + 1)) strong_first)
  - Scenario 2: Seed ${SCENARIO_SEEDS[1]} (Run 2 weak_first, Run $((WEAK_FIRST_RUNS + 2)) strong_first)
  - Scenario 3: Seed ${SCENARIO_SEEDS[2]} (Run 3 weak_first, Run $((WEAK_FIRST_RUNS + 3)) strong_first)

Model Order (Balanced):
  - Runs 1-${WEAK_FIRST_RUNS}: weak model first (weak_first)
  - Runs $((WEAK_FIRST_RUNS + 1))-${NUM_RUNS}: strong model first (strong_first)
  - This balances first-mover advantage across orderings
  - Same seeds are used for corresponding weak_first and strong_first runs to enable direct order comparison

This design ensures:
  - Statistical significance with ${NUM_RUNS} runs per configuration
  - Balanced ordering to mitigate first-mover advantage effects
  - Direct comparison of order effects (same seeds for weak_first vs strong_first)
  - Comparable results across model pairs (same seeds for same scenario)
  - ${#SCENARIO_SEEDS[@]} different scenarios tested (each with both orderings)
  - Analysis can average across orderings or compare order effects directly
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

# ============================================================================
# SLURM Script Generation
# ============================================================================
SLURM_DIR="${CONFIG_DIR}/slurm"
mkdir -p "${SLURM_DIR}"

echo ""
echo "Generating SLURM scripts..."

# Models that require GPUs (local inference)
# Llama 3.1 8B: 1 GPU (80GB)
# Llama 3.3 70B: 4 GPUs (320GB)
LOCAL_MODELS=("llama-3.3-70b-instruct" "llama-3.1-8b-instruct")

# Function to check if a model is local (needs GPU)
is_local_model() {
    local model="$1"
    for local_model in "${LOCAL_MODELS[@]}"; do
        if [[ "$model" == "$local_model" ]]; then
            return 0
        fi
    done
    return 1
}

# Function to get GPU count for a model
get_gpu_count() {
    local model="$1"
    case "$model" in
        "llama-3.3-70b-instruct") echo 4 ;;
        "llama-3.1-8b-instruct") echo 1 ;;
        *) echo 0 ;;
    esac
}

# Generate SLURM script for CPU jobs (API-based models)
CPU_SLURM="${SLURM_DIR}/run_api_experiments.sbatch"
cat > "${CPU_SLURM}" << 'SLURM_CPU'
#!/bin/bash
#SBATCH --job-name=bargain-api
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/cluster/api_%A_%a.out
#SBATCH --error=logs/cluster/api_%A_%a.err

set -e

# Change to project directory first
BASE_DIR="/scratch/gpfs/DANQIC/jz4391/bargain"
cd "${BASE_DIR}"

# Create logs directory if it doesn't exist
mkdir -p logs/cluster

echo "============================================================"
echo "SLURM Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Started at: $(date)"
echo "Node: $SLURM_NODELIST"
echo "============================================================"

# Load required modules
module purge
module load anaconda3/2024.2

# Load proxy module to enable API access (OpenAI, Anthropic, Google, etc.)
module load proxy/default

# Activate virtual environment
source "${BASE_DIR}/.venv/bin/activate"
echo "Activated virtual environment: ${BASE_DIR}/.venv"
echo "Python version: $(python3 --version)"
echo ""

# Get config file for this array task
# Note: This uses the symlink 'configs' which points to the latest timestamped config directory
CONFIG_DIR="experiments/results/scaling_experiment/configs"

# Determine padding width by finding the highest config number
MAX_CONFIG=$(ls "${CONFIG_DIR}"/config_*.json 2>/dev/null | sed 's/.*config_\([0-9]*\)\.json/\1/' | sort -n | tail -1)
if [[ -n "$MAX_CONFIG" ]]; then
    PADDING_WIDTH=${#MAX_CONFIG}
else
    PADDING_WIDTH=3  # Default to 3 digits if no configs found
fi

# Use zero-padded config ID to match generated file names
CONFIG_ID_PADDED=$(printf "%0${PADDING_WIDTH}d" ${SLURM_ARRAY_TASK_ID})
CONFIG_FILE="${CONFIG_DIR}/config_${CONFIG_ID_PADDED}.json"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Config file: $CONFIG_FILE"

# Extract config values
WEAK_MODEL=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['weak_model'])")
STRONG_MODEL=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['strong_model'])")
COMP_LEVEL=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['competition_level'])")
RUN_NUM=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['run_number'])")
SEED=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['random_seed'])")
MODEL_ORDER=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['model_order'])")
DISCUSSION_TURNS=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['discussion_turns'])")
OUTPUT_DIR=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['output_dir'])")

# Get models in correct order
if [[ "$MODEL_ORDER" == "weak_first" ]]; then
    MODELS="$WEAK_MODEL $STRONG_MODEL"
else
    MODELS="$STRONG_MODEL $WEAK_MODEL"
fi

echo "Models: $MODELS"
echo "Model order: $MODEL_ORDER"
echo "Competition level: $COMP_LEVEL"
echo "Run number: $RUN_NUM"
echo "Random seed: $SEED"
echo "Discussion turns: $DISCUSSION_TURNS"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Run experiment
echo "Running: python3 run_strong_models_experiment.py --models $MODELS ..."
echo ""

if python3 run_strong_models_experiment.py \
    --models $MODELS \
    --batch \
    --num-runs 1 \
    --run-number $RUN_NUM \
    --competition-level $COMP_LEVEL \
    --random-seed $SEED \
    --discussion-turns $DISCUSSION_TURNS \
    --model-order $MODEL_ORDER \
    --output-dir "$OUTPUT_DIR" \
    --job-id $SLURM_ARRAY_TASK_ID; then
    echo ""
    echo "============================================================"
    echo "✅ Experiment completed successfully at: $(date)"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "❌ Experiment failed at: $(date)"
    echo "============================================================"
    exit 1
fi
SLURM_CPU

echo "✅ Created CPU SLURM script: ${CPU_SLURM}"

# Generate SLURM script for SMALL GPU jobs (e.g., Llama 3.1 8B - 1 GPU, 80GB)
GPU_SMALL_SLURM="${SLURM_DIR}/run_gpu_small.sbatch"
cat > "${GPU_SMALL_SLURM}" << 'SLURM_GPU_SMALL'
#!/bin/bash
# =============================================================================
# GPU SLURM Script for SMALL Local Models (e.g., Llama 3.1 8B)
# =============================================================================
# Resources: 1 H100 GPU, 80GB memory
# Use run_gpu_large.sbatch for 70B+ models (Llama 3.3 70B, Qwen 72B, etc.)
# =============================================================================
#SBATCH --job-name=bargain-gpu-sm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=logs/cluster/gpu_sm_%A_%a.out
#SBATCH --error=logs/cluster/gpu_sm_%A_%a.err
#SBATCH --constraint=gpu80
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=pli-c

set -e

BASE_DIR="/scratch/gpfs/DANQIC/jz4391/bargain"
cd "${BASE_DIR}"
mkdir -p logs/cluster

echo "============================================================"
echo "SLURM Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Started at: $(date)"
echo "Node: $SLURM_NODELIST"
echo "Cluster: PLI (H100 GPU - 1 GPU, 80GB)"
echo "============================================================"

module load proxy/default
source "${BASE_DIR}/.venv/bin/activate"
echo "Python version: $(python3 --version)"
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA devices: $(python3 -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# Note: Uses symlink 'configs' which points to the latest timestamped config directory
CONFIG_DIR="experiments/results/scaling_experiment/configs"
# Determine padding width by finding the highest config number
MAX_CONFIG=$(ls "${CONFIG_DIR}"/config_*.json 2>/dev/null | sed 's/.*config_\([0-9]*\)\.json/\1/' | sort -n | tail -1)
if [[ -n "$MAX_CONFIG" ]]; then
    PADDING_WIDTH=${#MAX_CONFIG}
else
    PADDING_WIDTH=3  # Default to 3 digits if no configs found
fi
# Use zero-padded config ID to match generated file names
CONFIG_ID_PADDED=$(printf "%0${PADDING_WIDTH}d" ${SLURM_ARRAY_TASK_ID})
CONFIG_FILE="${CONFIG_DIR}/config_${CONFIG_ID_PADDED}.json"
[[ ! -f "$CONFIG_FILE" ]] && echo "ERROR: Config not found: $CONFIG_FILE" && exit 1

WEAK_MODEL=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['weak_model'])")
STRONG_MODEL=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['strong_model'])")
COMP_LEVEL=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['competition_level'])")
RUN_NUM=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['run_number'])")
SEED=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['random_seed'])")
MODEL_ORDER=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['model_order'])")
DISCUSSION_TURNS=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['discussion_turns'])")
OUTPUT_DIR=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['output_dir'])")

[[ "$MODEL_ORDER" == "weak_first" ]] && MODELS="$WEAK_MODEL $STRONG_MODEL" || MODELS="$STRONG_MODEL $WEAK_MODEL"

echo "Models: $MODELS | Order: $MODEL_ORDER | Comp: $COMP_LEVEL | Run: $RUN_NUM | Seed: $SEED"

if python3 run_strong_models_experiment.py --models $MODELS --batch --num-runs 1 \
    --run-number $RUN_NUM --competition-level $COMP_LEVEL --random-seed $SEED \
    --discussion-turns $DISCUSSION_TURNS --model-order $MODEL_ORDER \
    --output-dir "$OUTPUT_DIR" --job-id $SLURM_ARRAY_TASK_ID; then
    echo "✅ Completed at: $(date)"
else
    echo "❌ Failed at: $(date)" && exit 1
fi
SLURM_GPU_SMALL

echo "✅ Created small GPU SLURM script: ${GPU_SMALL_SLURM}"

# Generate SLURM script for LARGE GPU jobs (e.g., Llama 3.3 70B - 4 GPUs, 320GB)
GPU_LARGE_SLURM="${SLURM_DIR}/run_gpu_large.sbatch"
cat > "${GPU_LARGE_SLURM}" << 'SLURM_GPU_LARGE'
#!/bin/bash
# =============================================================================
# GPU SLURM Script for LARGE Local Models (e.g., Llama 3.3 70B, Qwen 72B)
# =============================================================================
# Resources: 4 H100 GPUs, 320GB memory
# Use run_gpu_small.sbatch for smaller models (Llama 3.1 8B, etc.)
# =============================================================================
#SBATCH --job-name=bargain-gpu-lg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --time=08:00:00
#SBATCH --output=logs/cluster/gpu_lg_%A_%a.out
#SBATCH --error=logs/cluster/gpu_lg_%A_%a.err
#SBATCH --constraint=gpu80
#SBATCH --gres=gpu:h100:4
#SBATCH --partition=pli-c

set -e

BASE_DIR="/scratch/gpfs/DANQIC/jz4391/bargain"
cd "${BASE_DIR}"
mkdir -p logs/cluster

echo "============================================================"
echo "SLURM Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Started at: $(date)"
echo "Node: $SLURM_NODELIST"
echo "Cluster: PLI (H100 GPUs - 4 GPUs, 320GB total)"
echo "============================================================"

module load proxy/default
source "${BASE_DIR}/.venv/bin/activate"
echo "Python version: $(python3 --version)"
echo "CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA devices: $(python3 -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# Note: Uses symlink 'configs' which points to the latest timestamped config directory
CONFIG_DIR="experiments/results/scaling_experiment/configs"
# Determine padding width by finding the highest config number
MAX_CONFIG=$(ls "${CONFIG_DIR}"/config_*.json 2>/dev/null | sed 's/.*config_\([0-9]*\)\.json/\1/' | sort -n | tail -1)
if [[ -n "$MAX_CONFIG" ]]; then
    PADDING_WIDTH=${#MAX_CONFIG}
else
    PADDING_WIDTH=3  # Default to 3 digits if no configs found
fi
# Use zero-padded config ID to match generated file names
CONFIG_ID_PADDED=$(printf "%0${PADDING_WIDTH}d" ${SLURM_ARRAY_TASK_ID})
CONFIG_FILE="${CONFIG_DIR}/config_${CONFIG_ID_PADDED}.json"
[[ ! -f "$CONFIG_FILE" ]] && echo "ERROR: Config not found: $CONFIG_FILE" && exit 1

WEAK_MODEL=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['weak_model'])")
STRONG_MODEL=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['strong_model'])")
COMP_LEVEL=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['competition_level'])")
RUN_NUM=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['run_number'])")
SEED=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['random_seed'])")
MODEL_ORDER=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['model_order'])")
DISCUSSION_TURNS=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['discussion_turns'])")
OUTPUT_DIR=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['output_dir'])")

[[ "$MODEL_ORDER" == "weak_first" ]] && MODELS="$WEAK_MODEL $STRONG_MODEL" || MODELS="$STRONG_MODEL $WEAK_MODEL"

echo "Models: $MODELS | Order: $MODEL_ORDER | Comp: $COMP_LEVEL | Run: $RUN_NUM | Seed: $SEED"

if python3 run_strong_models_experiment.py --models $MODELS --batch --num-runs 1 \
    --run-number $RUN_NUM --competition-level $COMP_LEVEL --random-seed $SEED \
    --discussion-turns $DISCUSSION_TURNS --model-order $MODEL_ORDER \
    --output-dir "$OUTPUT_DIR" --job-id $SLURM_ARRAY_TASK_ID; then
    echo "✅ Completed at: $(date)"
else
    echo "❌ Failed at: $(date)" && exit 1
fi
SLURM_GPU_LARGE

echo "✅ Created large GPU SLURM script: ${GPU_LARGE_SLURM}"

# Create a helper script to submit experiments
SUBMIT_SCRIPT="${SLURM_DIR}/submit_all.sh"
cat > "${SUBMIT_SCRIPT}" << 'SUBMIT_SCRIPT_EOF'
#!/bin/bash
# Submit all experiment jobs to SLURM
# Usage: ./submit_all.sh [api|gpu|all] [--staggered <seconds>] [--max-concurrent <num>]
#
# Options:
#   --staggered <seconds>  Submit jobs individually with delay (default: array jobs)
#                          Use this to avoid API rate limits (recommended: 2-5 seconds)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Go up 5 levels: slurm/ -> configs/ -> scaling_experiment/ -> results/ -> experiments/ -> project root
# Path: experiments/results/scaling_experiment/configs/slurm -> project root
BASE_DIR="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${BASE_DIR}"

# Parse arguments
JOB_TYPE="${1:-all}"
STAGGERED=false
DELAY_SECONDS=2
MAX_CONCURRENT=""

# Parse flags (skip first argument which is job type)
shift || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --staggered)
            if [[ -z "$2" ]] || [[ "$2" =~ ^-- ]]; then
                echo "Error: --staggered requires a value (seconds)"
                exit 1
            fi
            STAGGERED=true
            DELAY_SECONDS="$2"
            shift 2
            echo "Staggered submission enabled: ${DELAY_SECONDS}s delay between jobs"
            ;;
        --max-concurrent)
            if [[ -z "$2" ]] || [[ "$2" =~ ^-- ]]; then
                echo "Error: --max-concurrent requires a value (number)"
                exit 1
            fi
            MAX_CONCURRENT="$2"
            shift 2
            echo "Concurrency limit enabled: max ${MAX_CONCURRENT} concurrent jobs"
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [api|gpu|all] [--staggered <seconds>] [--max-concurrent <num>]"
            exit 1
            ;;
    esac
done

# Create logs directory
mkdir -p logs/cluster

# Calculate total configs by counting config files (handles zero-padded filenames)
CONFIG_DIR="experiments/results/scaling_experiment/configs"
TOTAL_CONFIGS=$(ls "${CONFIG_DIR}"/config_*.json 2>/dev/null | wc -l)
if [ "$TOTAL_CONFIGS" -eq 0 ]; then
    echo "Error: No config files found in ${CONFIG_DIR}"
    exit 1
fi
# Calculate padding width for config file names
MAX_CONFIG=$(ls "${CONFIG_DIR}"/config_*.json 2>/dev/null | sed 's/.*config_\([0-9]*\)\.json/\1/' | sort -n | tail -1)
if [[ -n "$MAX_CONFIG" ]]; then
    CONFIG_PADDING_WIDTH=${#MAX_CONFIG}
else
    CONFIG_PADDING_WIDTH=3  # Default to 3 digits
fi
# Calculate padding width for config file names (same as used during generation)
CONFIG_PADDING_WIDTH=${#TOTAL_CONFIGS}

# Function to submit a single job (for staggered mode)
submit_single_job() {
    local config_id=$1
    local sbatch_script=$2
    
    # Extract output/error patterns from the script
    local output_pattern=$(grep -E "^#SBATCH --output=" "$sbatch_script" | head -1 | sed 's/.*--output=//' || echo "")
    local error_pattern=$(grep -E "^#SBATCH --error=" "$sbatch_script" | head -1 | sed 's/.*--error=//' || echo "")
    
    # Replace %a with config_id, %A will be set by SLURM when job is submitted
    # For single jobs, %a won't expand, so we replace it manually
    local output_file=""
    local error_file=""
    
    if [[ -n "$output_pattern" ]]; then
        output_file=$(echo "$output_pattern" | sed "s/%a/${config_id}/g")
    fi
    if [[ -n "$error_pattern" ]]; then
        error_file=$(echo "$error_pattern" | sed "s/%a/${config_id}/g")
    fi
    
    # Submit with --export to set SLURM_ARRAY_TASK_ID
    # Note: %A in output/error will be replaced by SLURM with actual job ID
    local sbatch_cmd="sbatch --export=SLURM_ARRAY_TASK_ID=${config_id}"
    
    if [[ -n "$output_file" ]]; then
        sbatch_cmd="${sbatch_cmd} --output=${output_file}"
    fi
    if [[ -n "$error_file" ]]; then
        sbatch_cmd="${sbatch_cmd} --error=${error_file}"
    fi
    
    sbatch_cmd="${sbatch_cmd} ${sbatch_script}"
    
    eval "$sbatch_cmd" > /dev/null 2>&1
}

# Function to submit jobs (array or staggered)
submit_jobs() {
    local job_ids=$1
    local sbatch_script=$2
    local job_type_name=$3
    
    if [[ -z "$job_ids" ]]; then
        echo "No ${job_type_name} jobs to submit"
        return
    fi
    
    if [[ "$STAGGERED" == "true" ]]; then
        echo "Submitting ${job_type_name} jobs with staggered delays..."
        local count=0
        local total=$(echo "$job_ids" | tr ',' '\n' | wc -l)
        
        for id in $(echo "$job_ids" | tr ',' ' '); do
            count=$((count + 1))
            echo -n "[$(date +%H:%M:%S)] Submitting ${job_type_name} job ${count}/${total} (config ${id})... "
            
            if submit_single_job "$id" "$sbatch_script"; then
                echo "✅ Submitted"
            else
                echo "❌ Failed"
            fi
            
            # Don't sleep after the last job
            if [[ $count -lt $total ]]; then
                sleep "${DELAY_SECONDS}"
            fi
        done
    else
        # Use array job with optional concurrency limit
        local array_spec="$job_ids"
        if [[ -n "$MAX_CONCURRENT" ]]; then
            array_spec="${array_spec}%${MAX_CONCURRENT}"
            echo "Submitting ${job_type_name} job array with concurrency limit: $array_spec"
        else
            echo "Submitting ${job_type_name} job array: $array_spec"
        fi
        sbatch --array="$array_spec" "$sbatch_script"
    fi
}

submit_api_jobs() {
    echo "Submitting API-based experiment jobs (CPU)..."
    echo "Total configs: ${TOTAL_CONFIGS}"

    # Filter config IDs for API-only experiments
    API_IDS=""
    for i in $(seq 0 $((TOTAL_CONFIGS - 1))); do
        # Use zero-padded ID to match generated config file names
        i_padded=$(printf "%0${CONFIG_PADDING_WIDTH}d" ${i})
        CONFIG_FILE="experiments/results/scaling_experiment/configs/config_${i_padded}.json"
        if [[ -f "$CONFIG_FILE" ]]; then
            WEAK=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['weak_model'])")
            STRONG=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['strong_model'])")

            # Check if both models are API-based (not local)
            LOCAL_MODELS="llama-3.3-70b-instruct llama-3.1-8b-instruct"
            IS_LOCAL=false
            for lm in $LOCAL_MODELS; do
                if [[ "$WEAK" == "$lm" ]] || [[ "$STRONG" == "$lm" ]]; then
                    IS_LOCAL=true
                    break
                fi
            done

            if [[ "$IS_LOCAL" == "false" ]]; then
                if [[ -n "$API_IDS" ]]; then
                    API_IDS="${API_IDS},${i}"
                else
                    API_IDS="${i}"
                fi
            fi
        fi
    done

    submit_jobs "$API_IDS" "${SCRIPT_DIR}/run_api_experiments.sbatch" "API"
}

submit_gpu_jobs() {
    echo "Submitting GPU-based experiment jobs..."
    echo "Total configs: ${TOTAL_CONFIGS}"

    # Large models (70B+): 4 GPUs, 320GB
    LARGE_GPU_MODELS="llama-3.3-70b-instruct"
    # Small models (8B): 1 GPU, 80GB
    SMALL_GPU_MODELS="llama-3.1-8b-instruct"

    GPU_LARGE_IDS=""
    GPU_SMALL_IDS=""

    for i in $(seq 0 $((TOTAL_CONFIGS - 1))); do
        # Use zero-padded ID to match generated config file names
        i_padded=$(printf "%0${CONFIG_PADDING_WIDTH}d" ${i})
        CONFIG_FILE="experiments/results/scaling_experiment/configs/config_${i_padded}.json"
        if [[ -f "$CONFIG_FILE" ]]; then
            WEAK=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['weak_model'])")
            STRONG=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['strong_model'])")

            # Check for large GPU models
            for lm in $LARGE_GPU_MODELS; do
                if [[ "$WEAK" == "$lm" ]] || [[ "$STRONG" == "$lm" ]]; then
                    [[ -n "$GPU_LARGE_IDS" ]] && GPU_LARGE_IDS="${GPU_LARGE_IDS},${i}" || GPU_LARGE_IDS="${i}"
                    break
                fi
            done

            # Check for small GPU models (only if not already large)
            if ! echo "$GPU_LARGE_IDS" | grep -qw "$i"; then
                for lm in $SMALL_GPU_MODELS; do
                    if [[ "$WEAK" == "$lm" ]] || [[ "$STRONG" == "$lm" ]]; then
                        [[ -n "$GPU_SMALL_IDS" ]] && GPU_SMALL_IDS="${GPU_SMALL_IDS},${i}" || GPU_SMALL_IDS="${i}"
                        break
                    fi
                done
            fi
        fi
    done

    if [[ -n "$GPU_LARGE_IDS" ]]; then
        submit_jobs "$GPU_LARGE_IDS" "${SCRIPT_DIR}/run_gpu_large.sbatch" "LARGE GPU"
    fi

    if [[ -n "$GPU_SMALL_IDS" ]]; then
        submit_jobs "$GPU_SMALL_IDS" "${SCRIPT_DIR}/run_gpu_small.sbatch" "SMALL GPU"
    fi

    if [[ -z "$GPU_LARGE_IDS" ]] && [[ -z "$GPU_SMALL_IDS" ]]; then
        echo "No GPU-based experiments to submit"
    fi
}

case "${JOB_TYPE}" in
    api)
        submit_api_jobs
        ;;
    gpu)
        submit_gpu_jobs
        ;;
    all)
        submit_api_jobs
        submit_gpu_jobs
        ;;
    *)
        echo "Usage: $0 [api|gpu|all] [--staggered <seconds>]"
        echo ""
        echo "Options:"
        echo "  --staggered <seconds>     Submit jobs individually with delay (avoids rate limits)"
        echo "  --max-concurrent <num>    Limit concurrent array jobs (e.g., --max-concurrent 10)"
        echo ""
        echo "Examples:"
        echo "  $0 all                              # Submit all jobs as array (fast, may hit rate limits)"
        echo "  $0 all --staggered 2                # Submit all jobs with 2s delay (safer for APIs)"
        echo "  $0 all --max-concurrent 10          # Submit all jobs, max 10 concurrent"
        echo "  $0 api --staggered 5                # Submit API jobs with 5s delay"
        echo "  $0 api --max-concurrent 5           # Submit API jobs, max 5 concurrent"
        echo "  $0 all --staggered 2 --max-concurrent 20  # Staggered mode ignores max-concurrent"
        exit 1
        ;;
esac

echo ""
echo "Job submission complete. Use 'squeue -u $USER' to monitor jobs."
SUBMIT_SCRIPT_EOF
chmod +x "${SUBMIT_SCRIPT}"

echo "✅ Created job submission script: ${SUBMIT_SCRIPT}"

echo ""
echo "To run experiments:"
echo "  - All experiments: Use job IDs 0-$((EXPERIMENT_ID-1))"
echo "  - Specific run only: Filter by run_number in CSV"
echo "  - Specific model pair: Filter by weak_model and strong_model in CSV"
echo ""
echo "SLURM submission:"
echo "  cd ${BASE_DIR}"
echo "  ${SUBMIT_SCRIPT} all                    # Submit all jobs as array (fast)"
echo "  ${SUBMIT_SCRIPT} all --staggered 2      # Submit all jobs with 2s delay (safer for APIs)"
echo "  ${SUBMIT_SCRIPT} api --staggered 5       # Submit API jobs with 5s delay"
echo "  ${SUBMIT_SCRIPT} gpu                     # Submit GPU jobs (array mode)"
echo ""
echo "Note: Use --staggered to avoid API rate limits when submitting many jobs."
echo "      See scripts/EXPERIMENT_WORKFLOW.md for detailed workflow guide."
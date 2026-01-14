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
COMPETITION_LEVELS=(0.0 0.25 0.5 0.75 1.0)

# Base parameters
NUM_ITEMS=5
MAX_ROUNDS=10
NUM_RUNS=6  # Number of runs per configuration
DISCUSSION_TURNS=3  # Number of discussion exchanges per round

# Seeds for each run (consistent across model pairs)
RUN_SEEDS=(42 123 456 789 101112 131415)

# Balanced ordering: split runs between weak_first and strong_first
# weak_first gets ceiling(NUM_RUNS/2), strong_first gets floor(NUM_RUNS/2)
WEAK_FIRST_RUNS=$(( (NUM_RUNS + 1) / 2 ))
STRONG_FIRST_RUNS=$(( NUM_RUNS / 2 ))

echo "Generating experiment configurations with multiple runs..."
echo "  Weak models: ${#WEAK_MODELS[@]}"
echo "  Strong models: ${#STRONG_MODELS[@]}"
echo "  Competition levels: ${#COMPETITION_LEVELS[@]}"
echo "  Runs per config: ${NUM_RUNS} (${WEAK_FIRST_RUNS} weak_first + ${STRONG_FIRST_RUNS} strong_first)"
echo "  Discussion turns: ${DISCUSSION_TURNS}"
echo "  Total configs: $((${#WEAK_MODELS[@]} * ${#STRONG_MODELS[@]} * ${#COMPETITION_LEVELS[@]} * ${NUM_RUNS}))"
echo ""

# Counter for experiment ID
EXPERIMENT_ID=0

# Generate configs with balanced ordering: first WEAK_FIRST_RUNS use weak_first, rest use strong_first
echo "Generating configs with balanced ordering..."
for weak_model in "${WEAK_MODELS[@]}"; do
    for strong_model in "${STRONG_MODELS[@]}"; do
        for comp_level in "${COMPETITION_LEVELS[@]}"; do
            for run_idx in "${!RUN_SEEDS[@]}"; do
                # Create config file
                CONFIG_FILE="${CONFIG_DIR}/config_${EXPERIMENT_ID}.json"

                # Get seed for this run (consistent across all model pairs for same run number)
                SEED=${RUN_SEEDS[$run_idx]}
                RUN_NUM=$((run_idx + 1))

                # Determine model order: first WEAK_FIRST_RUNS use weak_first, rest use strong_first
                if [ $run_idx -lt $WEAK_FIRST_RUNS ]; then
                    MODEL_ORDER="weak_first"
                    MODELS_ARRAY="[\"${weak_model}\", \"${strong_model}\"]"
                    OUTPUT_SUBDIR="weak_first"
                else
                    MODEL_ORDER="strong_first"
                    MODELS_ARRAY="[\"${strong_model}\", \"${weak_model}\"]"
                    OUTPUT_SUBDIR="strong_first"
                fi

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

echo ""
echo "✅ Generated ${EXPERIMENT_ID} total configuration files:"
echo "   - Model pairs: $((${#WEAK_MODELS[@]} * ${#STRONG_MODELS[@]}))"
echo "   - Competition levels: ${#COMPETITION_LEVELS[@]}"
echo "   - Runs per config: ${NUM_RUNS}"
echo "   - Location: ${CONFIG_DIR}"

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

Random Seeds by Run:
  - Run 1: ${RUN_SEEDS[0]} (weak_first)
  - Run 2: ${RUN_SEEDS[1]} (weak_first)
  - Run 3: ${RUN_SEEDS[2]} (weak_first)
  - Run 4: ${RUN_SEEDS[3]} (strong_first)
  - Run 5: ${RUN_SEEDS[4]} (strong_first)

Model Order (Balanced):
  - Runs 1-${WEAK_FIRST_RUNS}: weak model first (weak_first)
  - Runs $((WEAK_FIRST_RUNS + 1))-${NUM_RUNS}: strong model first (strong_first)
  - This balances first-mover advantage across orderings

This design ensures:
  - Statistical significance with ${NUM_RUNS} runs per configuration
  - Balanced ordering to mitigate first-mover advantage effects
  - Comparable results across model pairs (same seeds for same run number)
  - Different scenarios tested (${NUM_RUNS} different seeds)
  - Analysis can average across orderings or compare them
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
CONFIG_DIR="experiments/results/scaling_experiment/configs"
CONFIG_FILE="${CONFIG_DIR}/config_${SLURM_ARRAY_TASK_ID}.json"

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

CONFIG_FILE="experiments/results/scaling_experiment/configs/config_${SLURM_ARRAY_TASK_ID}.json"
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

CONFIG_FILE="experiments/results/scaling_experiment/configs/config_${SLURM_ARRAY_TASK_ID}.json"
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
cat > "${SUBMIT_SCRIPT}" << EOF
#!/bin/bash
# Submit all experiment jobs to SLURM
# Usage: ./submit_all.sh [api|gpu|all]

set -e

SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="\$(cd "\${SCRIPT_DIR}/../.." && pwd)"
cd "\${BASE_DIR}"

# Create logs directory
mkdir -p logs/cluster

TOTAL_CONFIGS=${EXPERIMENT_ID}

submit_api_jobs() {
    echo "Submitting API-based experiment jobs (CPU)..."
    echo "Total configs: \${TOTAL_CONFIGS}"

    # Filter config IDs for API-only experiments
    API_IDS=""
    for i in \$(seq 0 \$((TOTAL_CONFIGS - 1))); do
        CONFIG_FILE="experiments/results/scaling_experiment/configs/config_\${i}.json"
        if [[ -f "\$CONFIG_FILE" ]]; then
            WEAK=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['weak_model'])")
            STRONG=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['strong_model'])")

            # Check if both models are API-based (not local)
            LOCAL_MODELS="llama-3.3-70b-instruct llama-3.1-8b-instruct"
            IS_LOCAL=false
            for lm in \$LOCAL_MODELS; do
                if [[ "\$WEAK" == "\$lm" ]] || [[ "\$STRONG" == "\$lm" ]]; then
                    IS_LOCAL=true
                    break
                fi
            done

            if [[ "\$IS_LOCAL" == "false" ]]; then
                if [[ -n "\$API_IDS" ]]; then
                    API_IDS="\${API_IDS},\${i}"
                else
                    API_IDS="\${i}"
                fi
            fi
        fi
    done

    if [[ -n "\$API_IDS" ]]; then
        echo "Submitting job array for config IDs: \$API_IDS"
        sbatch --array="\$API_IDS" "\${SCRIPT_DIR}/run_api_experiments.sbatch"
    else
        echo "No API-based experiments to submit"
    fi
}

submit_gpu_jobs() {
    echo "Submitting GPU-based experiment jobs..."
    echo "Total configs: \${TOTAL_CONFIGS}"

    # Large models (70B+): 4 GPUs, 320GB
    LARGE_GPU_MODELS="llama-3.3-70b-instruct"
    # Small models (8B): 1 GPU, 80GB
    SMALL_GPU_MODELS="llama-3.1-8b-instruct"

    GPU_LARGE_IDS=""
    GPU_SMALL_IDS=""

    for i in \$(seq 0 \$((TOTAL_CONFIGS - 1))); do
        CONFIG_FILE="experiments/results/scaling_experiment/configs/config_\${i}.json"
        if [[ -f "\$CONFIG_FILE" ]]; then
            WEAK=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['weak_model'])")
            STRONG=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['strong_model'])")

            # Check for large GPU models
            for lm in \$LARGE_GPU_MODELS; do
                if [[ "\$WEAK" == "\$lm" ]] || [[ "\$STRONG" == "\$lm" ]]; then
                    [[ -n "\$GPU_LARGE_IDS" ]] && GPU_LARGE_IDS="\${GPU_LARGE_IDS},\${i}" || GPU_LARGE_IDS="\${i}"
                    break
                fi
            done

            # Check for small GPU models (only if not already large)
            if ! echo "\$GPU_LARGE_IDS" | grep -qw "\$i"; then
                for lm in \$SMALL_GPU_MODELS; do
                    if [[ "\$WEAK" == "\$lm" ]] || [[ "\$STRONG" == "\$lm" ]]; then
                        [[ -n "\$GPU_SMALL_IDS" ]] && GPU_SMALL_IDS="\${GPU_SMALL_IDS},\${i}" || GPU_SMALL_IDS="\${i}"
                        break
                    fi
                done
            fi
        fi
    done

    if [[ -n "\$GPU_LARGE_IDS" ]]; then
        echo "Submitting LARGE GPU jobs (4x H100, 320GB): \$GPU_LARGE_IDS"
        sbatch --array="\$GPU_LARGE_IDS" "\${SCRIPT_DIR}/run_gpu_large.sbatch"
    fi

    if [[ -n "\$GPU_SMALL_IDS" ]]; then
        echo "Submitting SMALL GPU jobs (1x H100, 80GB): \$GPU_SMALL_IDS"
        sbatch --array="\$GPU_SMALL_IDS" "\${SCRIPT_DIR}/run_gpu_small.sbatch"
    fi

    if [[ -z "\$GPU_LARGE_IDS" ]] && [[ -z "\$GPU_SMALL_IDS" ]]; then
        echo "No GPU-based experiments to submit"
    fi
}

case "\${1:-all}" in
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
        echo "Usage: \$0 [api|gpu|all]"
        exit 1
        ;;
esac

echo ""
echo "Job submission complete. Use 'squeue -u \$USER' to monitor jobs."
EOF
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
echo "  ${SUBMIT_SCRIPT} all     # Submit all jobs"
echo "  ${SUBMIT_SCRIPT} api     # Submit API-only jobs (CPU)"
echo "  ${SUBMIT_SCRIPT} gpu     # Submit GPU jobs (local models)"
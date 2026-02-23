#!/bin/bash
# =============================================================================
# Generate Configuration Files for Game 3 (Co-Funding) Experiments
# =============================================================================
#
# This script generates JSON config files for Game 3 co-funding experiments.
# Supports two experiment types:
#
#   1. Model-scale experiments (--model-scale):
#      - Different model pairings at various capability levels
#      - Sweep over alpha (preference alignment) and sigma (budget scarcity)
#      - NO reasoning token budget manipulation
#      - Focus: How does model capability affect public goods provision?
#
#   2. TTC scaling experiments (--scaling):
#      - Reasoning models vs GPT-5-nano baseline
#      - Variable reasoning token budgets (prompted, not API-enforced)
#      - Fixed alpha/sigma
#      - Focus: How does test-time compute affect coordination?
#
# Usage:
#   ./scripts/generate_cofunding_configs.sh --model-scale    # Model capability sweep
#   ./scripts/generate_cofunding_configs.sh --scaling        # TTC scaling sweep
#   ./scripts/generate_cofunding_configs.sh --derisk         # 1 config for smoke test
#   ./scripts/generate_cofunding_configs.sh --small          # Reduced experiment
#   ./scripts/generate_cofunding_configs.sh                  # Full (both types)
#
# What it creates:
#   experiments/results/cofunding_<timestamp>/configs/
#   ├── config_0000.json ... config_NNNN.json   # Individual experiment configs
#   ├── all_configs.txt                          # List of all config files
#   ├── experiment_index.csv                     # Searchable index
#   ├── summary.txt                              # Human-readable summary
#   └── slurm/
#       ├── run_cofunding_experiments.sbatch      # SLURM array job script
#       ├── submit_all.sh                         # Submission helper
#       └── run_local.sh                          # Local testing script
#
# Game 3 Parameters:
#   - alpha: Preference alignment [0, 1]
#     - alpha ~ 0: orthogonal valuations (agents value different projects)
#     - alpha ~ 1: identical valuations (agents value same projects)
#   - sigma: Budget scarcity (0, 1]
#     - sigma ~ 0.2: very scarce (total budget = 20% of total cost)
#     - sigma ~ 1.0: abundant (total budget = 100% of total cost)
#   - m_projects: Number of projects to co-fund (default: 5)
#
# Dependencies:
#   - bash, python3
#   - run_strong_models_experiment.py in project root
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
        --model-scale)
            MODE="model_scale"
            shift
            ;;
        --scaling)
            MODE="scaling"
            shift
            ;;
        --conservative)
            MODE="conservative"
            shift
            ;;
        --ambitious)
            MODE="ambitious"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--derisk|--small|--model-scale|--scaling|--conservative|--ambitious]"
            echo ""
            echo "Experiment modes:"
            echo "  --conservative Model-scale sweep: 6 pairs, 3x3 alpha/sigma grid, 1 run (108 configs)"
            echo "  --ambitious    Model-scale sweep: 9 pairs, 5x5 alpha/sigma grid, 3 runs (1350 configs)"
            echo "  --model-scale  Model capability experiments: multiple model pairs, sweep alpha/sigma, no TTC"
            echo "  --scaling      TTC scaling experiments: reasoning models vs baseline, sweep token budgets"
            echo "  --small        Reduced version of full experiment (fewer models, fewer params)"
            echo "  --derisk       Minimal smoke test (1 config)"
            echo "  (default)      Full experiment: both model-scale AND TTC scaling"
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
EXPERIMENT_DIR="${BASE_DIR}/experiments/results/cofunding_${TIMESTAMP}"
CONFIG_DIR="${EXPERIMENT_DIR}/configs"
mkdir -p "${CONFIG_DIR}"

echo "Creating co-funding experiment directory: ${EXPERIMENT_DIR}"
echo "Config directory: ${CONFIG_DIR}"
echo "Mode: ${MODE}"
echo ""

# =============================================================================
# Shared Game Configuration
# =============================================================================
M_PROJECTS=5
MAX_ROUNDS=10
DISCUSSION_TURNS=3
MAX_TOKENS_PER_PHASE=10500
BASE_SEED=42
NUM_RUNS=1
C_MIN=10.0
C_MAX=30.0

# Reasoning phases (only used for TTC-scaling configs)
REASONING_PHASES_JSON='["thinking", "reflection", "discussion", "proposal", "voting"]'
PROMPT_ONLY="true"

# =============================================================================
# Mode-Specific Parameter Definitions
# =============================================================================

# --- MODEL-SCALE EXPERIMENTS ---
# Model pairs: "model1,model2" where model1 is the weak/baseline model
# Use model_order=weak_first/strong_first to control speaking order
# These run WITHOUT reasoning token budget -- pure model capability comparison.

if [[ "$MODE" == "conservative" ]]; then
    MODEL_PAIRS=(
        "gpt-5-nano,gpt-5-nano"
        "gpt-5-nano,gpt-3.5-turbo-0125"
        "gpt-5-nano,gpt-4o"
        "gpt-5-nano,o3-mini-high"
        "gpt-5-nano,claude-haiku-4-5"
        "gpt-5-nano,gpt-5.2-high"
    )
    MS_ALPHA_VALUES=(0.0 0.5 1.0)
    MS_SIGMA_VALUES=(0.2 0.6 1.0)
    MS_MODEL_ORDERS=("weak_first" "strong_first")
    NUM_RUNS=1
elif [[ "$MODE" == "ambitious" ]]; then
    MODEL_PAIRS=(
        "gpt-5-nano,gpt-5-nano"
        "gpt-5-nano,gpt-3.5-turbo-0125"
        "gpt-5-nano,amazon-nova-micro"
        "gpt-5-nano,gpt-4o"
        "gpt-5-nano,o3-mini-high"
        "gpt-5-nano,claude-haiku-4-5"
        "gpt-5-nano,gpt-5.2-high"
        "gpt-5-nano,claude-sonnet-4-5"
        "gpt-5-nano,gemini-3-pro"
    )
    MS_ALPHA_VALUES=(0.0 0.25 0.5 0.75 1.0)
    MS_SIGMA_VALUES=(0.2 0.4 0.6 0.8 1.0)
    MS_MODEL_ORDERS=("weak_first" "strong_first")
    NUM_RUNS=3
elif [[ "$MODE" == "derisk" ]]; then
    MODEL_PAIRS=(
        "gpt-5-nano,gpt-5-nano"
    )
    MS_ALPHA_VALUES=(1.0)
    MS_SIGMA_VALUES=(1.0)
    MS_MODEL_ORDERS=("weak_first")
    MAX_ROUNDS=2
elif [[ "$MODE" == "small" ]]; then
    MODEL_PAIRS=(
        "gpt-5-nano,claude-opus-4-5-thinking-32k"
        "gpt-5-nano,o3-mini-high"
        "gpt-5-nano,gpt-5.2-high"
    )
    MS_ALPHA_VALUES=(0.0 0.5 1.0)
    MS_SIGMA_VALUES=(0.3 0.5 0.8)
    MS_MODEL_ORDERS=("weak_first" "strong_first")
elif [[ "$MODE" == "model_scale" ]]; then
    MODEL_PAIRS=(
        # Strong reasoning vs weak baseline
        "gpt-5-nano,claude-opus-4-5-thinking-32k"
        "gpt-5-nano,o3-mini-high"
        "gpt-5-nano,gpt-5.2-high"
        "gpt-5-nano,grok-4"
        "gpt-5-nano,deepseek-r1"
        # Strong vs strong
        "claude-opus-4-5-thinking-32k,gpt-5.2-high"
        "claude-opus-4-5-thinking-32k,o3-mini-high"
        "o3-mini-high,gpt-5.2-high"
    )
    MS_ALPHA_VALUES=(0.0 0.2 0.4 0.6 0.8 1.0)
    MS_SIGMA_VALUES=(0.2 0.4 0.6 0.8 1.0)
    MS_MODEL_ORDERS=("weak_first" "strong_first")
elif [[ "$MODE" == "full" ]]; then
    MODEL_PAIRS=(
        # Strong reasoning vs weak baseline
        "gpt-5-nano,claude-opus-4-5-thinking-32k"
        "gpt-5-nano,o3-mini-high"
        "gpt-5-nano,gpt-5.2-high"
        "gpt-5-nano,grok-4"
        "gpt-5-nano,deepseek-r1"
        # Strong vs strong
        "claude-opus-4-5-thinking-32k,gpt-5.2-high"
        "claude-opus-4-5-thinking-32k,o3-mini-high"
        "o3-mini-high,gpt-5.2-high"
    )
    MS_ALPHA_VALUES=(0.0 0.2 0.4 0.6 0.8 1.0)
    MS_SIGMA_VALUES=(0.2 0.4 0.6 0.8 1.0)
    MS_MODEL_ORDERS=("weak_first" "strong_first")
else
    # scaling mode -- no model-scale configs
    MODEL_PAIRS=()
    MS_ALPHA_VALUES=()
    MS_SIGMA_VALUES=()
    MS_MODEL_ORDERS=()
fi

# --- TTC SCALING EXPERIMENTS ---
# Reasoning model vs baseline, sweep token budgets, fixed game params
BASELINE_MODEL="gpt-5-nano"

if [[ "$MODE" == "conservative" || "$MODE" == "ambitious" ]]; then
    TTC_REASONING_MODELS=()  # Conservative/ambitious: model-scale only, no TTC
    TTC_TOKEN_BUDGETS=()
    TTC_ALPHA_VALUES=()
    TTC_SIGMA_VALUES=()
    TTC_MODEL_ORDERS=()
elif [[ "$MODE" == "derisk" ]]; then
    TTC_REASONING_MODELS=()  # Derisk only does model-scale
    TTC_TOKEN_BUDGETS=()
    TTC_ALPHA_VALUES=()
    TTC_SIGMA_VALUES=()
    TTC_MODEL_ORDERS=()
elif [[ "$MODE" == "small" ]]; then
    TTC_REASONING_MODELS=(
        "claude-opus-4-5-thinking-32k"
    )
    TTC_TOKEN_BUDGETS=(100 1000 5000)
    TTC_ALPHA_VALUES=(0.5)
    TTC_SIGMA_VALUES=(0.5)
    TTC_MODEL_ORDERS=("weak_first" "strong_first")
elif [[ "$MODE" == "scaling" ]]; then
    TTC_REASONING_MODELS=(
        "claude-opus-4-5-thinking-32k"
        "o3-mini-high"
        "gpt-5.2-high"
    )
    TTC_TOKEN_BUDGETS=(100 500 1000 5000 10000)
    TTC_ALPHA_VALUES=(0.5)
    TTC_SIGMA_VALUES=(0.5)
    TTC_MODEL_ORDERS=("weak_first" "strong_first")
elif [[ "$MODE" == "full" ]]; then
    TTC_REASONING_MODELS=(
        "claude-opus-4-5-thinking-32k"
        "o3-mini-high"
        "gpt-5.2-high"
    )
    TTC_TOKEN_BUDGETS=(100 500 1000 3000 5000 10000 20000 30000)
    TTC_ALPHA_VALUES=(0.5)
    TTC_SIGMA_VALUES=(0.5)
    TTC_MODEL_ORDERS=("weak_first" "strong_first")
else
    # model_scale mode -- no TTC configs
    TTC_REASONING_MODELS=()
    TTC_TOKEN_BUDGETS=()
    TTC_ALPHA_VALUES=()
    TTC_SIGMA_VALUES=()
    TTC_MODEL_ORDERS=()
fi

# =============================================================================
# Count totals
# =============================================================================
MS_TOTAL=$((${#MODEL_PAIRS[@]} * ${#MS_ALPHA_VALUES[@]} * ${#MS_SIGMA_VALUES[@]} * ${#MS_MODEL_ORDERS[@]} * NUM_RUNS))
TTC_TOTAL=$((${#TTC_REASONING_MODELS[@]} * ${#TTC_TOKEN_BUDGETS[@]} * ${#TTC_ALPHA_VALUES[@]} * ${#TTC_SIGMA_VALUES[@]} * ${#TTC_MODEL_ORDERS[@]} * NUM_RUNS))
TOTAL_EXPECTED=$((MS_TOTAL + TTC_TOTAL))

echo "Generating co-funding experiment configurations..."
echo ""
if [[ $MS_TOTAL -gt 0 ]]; then
    echo "  MODEL-SCALE experiments: ${MS_TOTAL} configs"
    echo "    Model pairs: ${#MODEL_PAIRS[@]}"
    echo "    Alpha values: ${MS_ALPHA_VALUES[*]}"
    echo "    Sigma values: ${MS_SIGMA_VALUES[*]}"
    echo "    Model orders: ${#MS_MODEL_ORDERS[@]}"
fi
if [[ $TTC_TOTAL -gt 0 ]]; then
    echo "  TTC-SCALING experiments: ${TTC_TOTAL} configs"
    echo "    Reasoning models: ${#TTC_REASONING_MODELS[@]}"
    echo "    Baseline: ${BASELINE_MODEL}"
    echo "    Token budgets: ${TTC_TOKEN_BUDGETS[*]}"
    echo "    Alpha values: ${TTC_ALPHA_VALUES[*]}"
    echo "    Sigma values: ${TTC_SIGMA_VALUES[*]}"
    echo "    Model orders: ${#TTC_MODEL_ORDERS[@]}"
fi
echo ""
echo "  TOTAL: ${TOTAL_EXPECTED} configs"
echo ""

# Padding width
PADDING_WIDTH=${#TOTAL_EXPECTED}
if [[ $PADDING_WIDTH -lt 4 ]]; then
    PADDING_WIDTH=4
fi

EXPERIMENT_ID=0

# =============================================================================
# Generate MODEL-SCALE configs (no TTC)
# =============================================================================
for model_pair in "${MODEL_PAIRS[@]}"; do
    IFS=',' read -ra MODELS <<< "${model_pair}"
    MODEL1="${MODELS[0]}"
    MODEL2="${MODELS[1]}"

    for alpha in "${MS_ALPHA_VALUES[@]}"; do
        for sigma in "${MS_SIGMA_VALUES[@]}"; do
            for model_order in "${MS_MODEL_ORDERS[@]}"; do
                for ((run_num=1; run_num<=NUM_RUNS; run_num++)); do
                    EXPERIMENT_ID_PADDED=$(printf "%0${PADDING_WIDTH}d" ${EXPERIMENT_ID})
                    CONFIG_FILE="${CONFIG_DIR}/config_${EXPERIMENT_ID_PADDED}.json"

                    # Model order determines which is Agent_Alpha
                    if [[ "$model_order" == "weak_first" ]]; then
                        MODELS_ARRAY="[\"${MODEL1}\", \"${MODEL2}\"]"
                    else
                        MODELS_ARRAY="[\"${MODEL2}\", \"${MODEL1}\"]"
                    fi

                    SEED=$((BASE_SEED + EXPERIMENT_ID))

                    ALPHA_STR=$(echo "${alpha}" | sed 's/\./_/g')
                    SIGMA_STR=$(echo "${sigma}" | sed 's/\./_/g')

                    if [[ ${NUM_RUNS} -gt 1 ]]; then
                        OUTPUT_DIR_SUFFIX="/run_${run_num}"
                    else
                        OUTPUT_DIR_SUFFIX=""
                    fi

                    cat > "${CONFIG_FILE}" << EOF
{
    "experiment_id": ${EXPERIMENT_ID},
    "experiment_type": "model_scale",
    "game_type": "co_funding",
    "model1": "${MODEL1}",
    "model2": "${MODEL2}",
    "models": ${MODELS_ARRAY},
    "model_order": "${model_order}",
    "run_number": ${run_num},
    "num_runs": ${NUM_RUNS},
    "max_tokens_per_phase": ${MAX_TOKENS_PER_PHASE},
    "m_projects": ${M_PROJECTS},
    "alpha": ${alpha},
    "sigma": ${sigma},
    "c_min": ${C_MIN},
    "c_max": ${C_MAX},
    "max_rounds": ${MAX_ROUNDS},
    "discussion_turns": ${DISCUSSION_TURNS},
    "random_seed": ${SEED},
    "output_dir": "experiments/results/cofunding_${TIMESTAMP}/model_scale/${MODEL1}_vs_${MODEL2}/${model_order}/alpha_${ALPHA_STR}_sigma_${SIGMA_STR}${OUTPUT_DIR_SUFFIX}"
}
EOF

                    EXPERIMENT_ID=$((EXPERIMENT_ID + 1))
                done
            done
        done
    done
done

# =============================================================================
# Generate TTC-SCALING configs (with reasoning token budget)
# =============================================================================
for reasoning_model in "${TTC_REASONING_MODELS[@]}"; do
    for token_budget in "${TTC_TOKEN_BUDGETS[@]}"; do
        for alpha in "${TTC_ALPHA_VALUES[@]}"; do
            for sigma in "${TTC_SIGMA_VALUES[@]}"; do
                for model_order in "${TTC_MODEL_ORDERS[@]}"; do
                    for ((run_num=1; run_num<=NUM_RUNS; run_num++)); do
                        EXPERIMENT_ID_PADDED=$(printf "%0${PADDING_WIDTH}d" ${EXPERIMENT_ID})
                        CONFIG_FILE="${CONFIG_DIR}/config_${EXPERIMENT_ID_PADDED}.json"

                        if [[ "$model_order" == "weak_first" ]]; then
                            MODELS_ARRAY="[\"${BASELINE_MODEL}\", \"${reasoning_model}\"]"
                        else
                            MODELS_ARRAY="[\"${reasoning_model}\", \"${BASELINE_MODEL}\"]"
                        fi

                        SEED=$((BASE_SEED + EXPERIMENT_ID))

                        ALPHA_STR=$(echo "${alpha}" | sed 's/\./_/g')
                        SIGMA_STR=$(echo "${sigma}" | sed 's/\./_/g')

                        if [[ ${NUM_RUNS} -gt 1 ]]; then
                            OUTPUT_DIR_SUFFIX="/run_${run_num}"
                        else
                            OUTPUT_DIR_SUFFIX=""
                        fi

                        cat > "${CONFIG_FILE}" << EOF
{
    "experiment_id": ${EXPERIMENT_ID},
    "experiment_type": "ttc_scaling",
    "game_type": "co_funding",
    "reasoning_model": "${reasoning_model}",
    "baseline_model": "${BASELINE_MODEL}",
    "models": ${MODELS_ARRAY},
    "model_order": "${model_order}",
    "reasoning_token_budget": ${token_budget},
    "reasoning_budget_phases": ${REASONING_PHASES_JSON},
    "prompt_only": ${PROMPT_ONLY},
    "run_number": ${run_num},
    "num_runs": ${NUM_RUNS},
    "max_tokens_per_phase": ${MAX_TOKENS_PER_PHASE},
    "m_projects": ${M_PROJECTS},
    "alpha": ${alpha},
    "sigma": ${sigma},
    "c_min": ${C_MIN},
    "c_max": ${C_MAX},
    "max_rounds": ${MAX_ROUNDS},
    "discussion_turns": ${DISCUSSION_TURNS},
    "random_seed": ${SEED},
    "output_dir": "experiments/results/cofunding_${TIMESTAMP}/ttc_scaling/${reasoning_model}_vs_${BASELINE_MODEL}/${model_order}/budget_${token_budget}/alpha_${ALPHA_STR}_sigma_${SIGMA_STR}${OUTPUT_DIR_SUFFIX}"
}
EOF

                        EXPERIMENT_ID=$((EXPERIMENT_ID + 1))
                    done
                done
            done
        done
    done
done

TOTAL_COUNT=${EXPERIMENT_ID}

# Create symlink to latest
SYMLINK="${BASE_DIR}/experiments/results/cofunding_latest"
if [[ -L "${SYMLINK}" ]]; then
    rm "${SYMLINK}"
elif [[ -d "${SYMLINK}" ]] && [[ ! -L "${SYMLINK}" ]]; then
    OLD_DIR="${SYMLINK}_old_$(date +%Y%m%d_%H%M%S)"
    mv "${SYMLINK}" "${OLD_DIR}"
    echo "Moved existing directory to: ${OLD_DIR}"
fi
ln -sf "cofunding_${TIMESTAMP}" "${SYMLINK}"
echo "Created symlink: ${SYMLINK} -> cofunding_${TIMESTAMP}"

echo ""
echo "Generated ${TOTAL_COUNT} configuration files"
echo "Location: ${CONFIG_DIR}"

# Create master config list
MASTER_CONFIG="${CONFIG_DIR}/all_configs.txt"
ls -1 "${CONFIG_DIR}"/config_*.json > "${MASTER_CONFIG}"
echo "Created master config list: ${MASTER_CONFIG}"

# Create summary
SUMMARY_FILE="${CONFIG_DIR}/summary.txt"
cat > "${SUMMARY_FILE}" << EOF
Co-Funding (Game 3) Experiment Configuration Summary
======================================================
Mode: ${MODE}
Total experiments: ${TOTAL_COUNT}
  Model-scale configs: ${MS_TOTAL}
  TTC-scaling configs: ${TTC_TOTAL}

MODEL-SCALE experiments (no TTC):
  Model pairs: ${MODEL_PAIRS[*]:-none}
  Alpha values: ${MS_ALPHA_VALUES[*]:-none}
  Sigma values: ${MS_SIGMA_VALUES[*]:-none}
  Model orders: ${MS_MODEL_ORDERS[*]:-none}

TTC-SCALING experiments:
  Baseline model: ${BASELINE_MODEL}
  Reasoning models: ${TTC_REASONING_MODELS[*]:-none}
  Token budgets: ${TTC_TOKEN_BUDGETS[*]:-none}
  Alpha values: ${TTC_ALPHA_VALUES[*]:-none}
  Sigma values: ${TTC_SIGMA_VALUES[*]:-none}
  Model orders: ${TTC_MODEL_ORDERS[*]:-none}

Game configuration:
  Game type: co_funding
  Projects per game: ${M_PROJECTS}
  Max rounds: ${MAX_ROUNDS}
  Discussion turns: ${DISCUSSION_TURNS}
  Max tokens per phase: ${MAX_TOKENS_PER_PHASE}
  Runs per configuration: ${NUM_RUNS}
  Project cost range: [${C_MIN}, ${C_MAX}]

Key Game 3 parameters:
  alpha: Preference alignment [0, 1]
    - alpha ~ 0: orthogonal valuations (agents value different projects)
    - alpha ~ 0.5: moderate alignment
    - alpha ~ 1: identical valuations (agents value same projects)
  sigma: Budget scarcity (0, 1]
    - sigma ~ 0.2: very scarce (budget = 20% of total cost)
    - sigma ~ 0.5: moderate scarcity
    - sigma ~ 1.0: abundant (budget = 100% of total cost)
EOF
echo "Created summary: ${SUMMARY_FILE}"

# Create CSV index
CSV_FILE="${CONFIG_DIR}/experiment_index.csv"
echo "experiment_id,experiment_type,model1,model2,model_order,token_budget,alpha,sigma,run_number,seed,config_file" > "${CSV_FILE}"

for config_file in "${CONFIG_DIR}"/config_*.json; do
    if [[ -f "$config_file" ]]; then
        exp_id=$(grep -o '"experiment_id": [0-9]*' "$config_file" | grep -o '[0-9]*')
        exp_type=$(grep -o '"experiment_type": "[^"]*"' "$config_file" | cut -d'"' -f4)
        order=$(grep -o '"model_order": "[^"]*"' "$config_file" | cut -d'"' -f4)
        alpha=$(grep -o '"alpha": [0-9.]*' "$config_file" | grep -o '[0-9.]*$')
        sigma=$(grep -o '"sigma": [0-9.]*' "$config_file" | grep -o '[0-9.]*$')
        run_num=$(grep -o '"run_number": [0-9]*' "$config_file" | grep -o '[0-9]*' || echo "1")
        seed=$(grep -o '"random_seed": [0-9]*' "$config_file" | grep -o '[0-9]*')

        if [[ "$exp_type" == "model_scale" ]]; then
            m1=$(grep -o '"model1": "[^"]*"' "$config_file" | cut -d'"' -f4)
            m2=$(grep -o '"model2": "[^"]*"' "$config_file" | cut -d'"' -f4)
            budget="NA"
        else
            m1=$(grep -o '"reasoning_model": "[^"]*"' "$config_file" | cut -d'"' -f4)
            m2=$(grep -o '"baseline_model": "[^"]*"' "$config_file" | cut -d'"' -f4)
            budget=$(grep -o '"reasoning_token_budget": [0-9]*' "$config_file" | grep -o '[0-9]*')
        fi

        echo "${exp_id},${exp_type},${m1},${m2},${order},${budget},${alpha},${sigma},${run_num},${seed},$(basename $config_file)" >> "${CSV_FILE}"
    fi
done
echo "Created experiment index: ${CSV_FILE}"

# =============================================================================
# SLURM Script Generation
# =============================================================================
SLURM_DIR="${CONFIG_DIR}/slurm"
mkdir -p "${SLURM_DIR}"

echo ""
echo "Generating SLURM scripts..."

CONFIG_DIR_ABSOLUTE="${EXPERIMENT_DIR}/configs"

# Mode-dependent SLURM settings
if [[ "$MODE" == "conservative" || "$MODE" == "ambitious" ]]; then
    SLURM_PARTITION="cpu"
    SLURM_TIME="06:00:00"
else
    SLURM_PARTITION=""
    SLURM_TIME="04:00:00"
fi

# The SLURM script handles BOTH model-scale and TTC configs by checking experiment_type
SLURM_SCRIPT="${SLURM_DIR}/run_cofunding_experiments.sbatch"

# Build partition line conditionally
if [[ -n "$SLURM_PARTITION" ]]; then
    PARTITION_LINE="#SBATCH --partition=${SLURM_PARTITION}"
else
    PARTITION_LINE=""
fi

cat > "${SLURM_SCRIPT}" << SLURM_EOF
#!/bin/bash
#SBATCH --job-name=cofund-exp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=${SLURM_TIME}
${PARTITION_LINE}
#SBATCH --output=logs/cluster/cofund_%A_%a.out
#SBATCH --error=logs/cluster/cofund_%A_%a.err

set -e

BASE_DIR="/scratch/gpfs/DANQIC/jz4391/bargain"
cd "\${BASE_DIR}"
mkdir -p logs/cluster

echo "============================================================"
echo "Co-Funding (Game 3) Experiment"
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
EXPERIMENT_TYPE=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['experiment_type'])")
MODEL_ORDER=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['model_order'])")
ALPHA=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['alpha'])")
SIGMA=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['sigma'])")
M_PROJECTS=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['m_projects'])")
C_MIN=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['c_min'])")
C_MAX=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['c_max'])")
SEED=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['random_seed'])")
DISCUSSION_TURNS=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['discussion_turns'])")
OUTPUT_DIR=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['output_dir'])")
MAX_TOKENS=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['max_tokens_per_phase'])")
MAX_ROUNDS=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['max_rounds'])")
NUM_RUNS=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['num_runs'])")
RUN_NUMBER=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['run_number'])")

# Get models list (works for both experiment types)
MODELS=\$(python3 -c "import json; print(' '.join(json.load(open('\${CONFIG_FILE}'))['models']))")

echo "Experiment type: \$EXPERIMENT_TYPE"
echo "Model order: \$MODEL_ORDER"
echo "Models: \$MODELS"
echo "Alpha: \$ALPHA"
echo "Sigma: \$SIGMA"
echo "Projects: \$M_PROJECTS"
echo "Max rounds: \$MAX_ROUNDS"
echo "Num runs: \$NUM_RUNS | Run number: \$RUN_NUMBER"
echo "Random seed: \$SEED"
echo "Output dir: \$OUTPUT_DIR"

# Build the command
CMD="python3 run_strong_models_experiment.py"
CMD="\$CMD --game-type co_funding"
CMD="\$CMD --models \$MODELS"
CMD="\$CMD --batch --num-runs \$NUM_RUNS --run-number \$RUN_NUMBER"
CMD="\$CMD --m-projects \$M_PROJECTS"
CMD="\$CMD --alpha \$ALPHA --sigma \$SIGMA"
CMD="\$CMD --c-min \$C_MIN --c-max \$C_MAX"
CMD="\$CMD --max-rounds \$MAX_ROUNDS"
CMD="\$CMD --random-seed \$SEED"
CMD="\$CMD --discussion-turns \$DISCUSSION_TURNS"
CMD="\$CMD --model-order \$MODEL_ORDER"
CMD="\$CMD --max-tokens-per-phase \$MAX_TOKENS"
CMD="\$CMD --output-dir \$OUTPUT_DIR"
CMD="\$CMD --job-id \$SLURM_ARRAY_TASK_ID"

# Add TTC-specific flags only for ttc_scaling experiments
if [[ "\$EXPERIMENT_TYPE" == "ttc_scaling" ]]; then
    TOKEN_BUDGET=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['reasoning_token_budget'])")
    REASONING_PHASES=\$(python3 -c "import json; print(' '.join(json.load(open('\${CONFIG_FILE}'))['reasoning_budget_phases']))")
    PROMPT_ONLY=\$(python3 -c "import json; print(str(json.load(open('\${CONFIG_FILE}')).get('prompt_only', False)).lower())")

    echo "Token budget: \$TOKEN_BUDGET"
    echo "Reasoning phases: \$REASONING_PHASES"
    echo "Prompt only: \$PROMPT_ONLY"

    CMD="\$CMD --reasoning-token-budget \$TOKEN_BUDGET"
    CMD="\$CMD --reasoning-budget-phases \$REASONING_PHASES"

    if [[ "\$PROMPT_ONLY" == "true" ]]; then
        CMD="\$CMD --prompt-only"
    fi
fi

echo ""
echo "Running: \$CMD"
echo ""

if eval \$CMD; then
    echo ""
    echo "============================================================"
    echo "Experiment completed successfully at: \$(date)"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "Experiment failed at: \$(date)"
    echo "============================================================"
    exit 1
fi
SLURM_EOF

echo "Created SLURM script: ${SLURM_SCRIPT}"

# Create submission script
SUBMIT_SCRIPT="${SLURM_DIR}/submit_all.sh"
cat > "${SUBMIT_SCRIPT}" << 'SUBMIT_EOF'
#!/bin/bash
# Submit co-funding experiment jobs
# Usage: ./submit_all.sh [--test] [--max-concurrent <num>]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${BASE_DIR}"

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

CONFIG_DIR="${SCRIPT_DIR}/.."
TOTAL_CONFIGS=$(ls "${CONFIG_DIR}"/config_*.json 2>/dev/null | wc -l)

if [ "$TOTAL_CONFIGS" -eq 0 ]; then
    echo "Error: No config files found"
    exit 1
fi

echo "Total configurations: ${TOTAL_CONFIGS}"

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
sbatch --array="${ARRAY_SPEC}" "${SCRIPT_DIR}/run_cofunding_experiments.sbatch"

echo ""
echo "Jobs submitted. Monitor with: squeue -u $USER"
SUBMIT_EOF
chmod +x "${SUBMIT_SCRIPT}"

echo "Created submission script: ${SUBMIT_SCRIPT}"

# Create local run script
LOCAL_RUN_SCRIPT="${SLURM_DIR}/run_local.sh"
cat > "${LOCAL_RUN_SCRIPT}" << 'LOCAL_EOF'
#!/bin/bash
# Run a single co-funding experiment locally (for testing)
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
EXPERIMENT_TYPE=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['experiment_type'])")
MODEL_ORDER=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['model_order'])")
ALPHA=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['alpha'])")
SIGMA=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['sigma'])")
M_PROJECTS=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['m_projects'])")
C_MIN=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['c_min'])")
C_MAX=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['c_max'])")
SEED=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['random_seed'])")
DISCUSSION_TURNS=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['discussion_turns'])")
OUTPUT_DIR=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['output_dir'])")
MAX_TOKENS=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['max_tokens_per_phase'])")
MAX_ROUNDS=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['max_rounds'])")
NUM_RUNS=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['num_runs'])")
RUN_NUMBER=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['run_number'])")
MODELS=$(python3 -c "import json; print(' '.join(json.load(open('${CONFIG_FILE}'))['models']))")

echo "Experiment type: $EXPERIMENT_TYPE"
echo "Model order: $MODEL_ORDER"
echo "Models: $MODELS"
echo "Alpha: $ALPHA | Sigma: $SIGMA | Projects: $M_PROJECTS | Max rounds: $MAX_ROUNDS"
echo "Num runs: $NUM_RUNS | Run number: $RUN_NUMBER"
echo ""

# Build command
CMD="python3 run_strong_models_experiment.py"
CMD="$CMD --game-type co_funding"
CMD="$CMD --models $MODELS"
CMD="$CMD --batch --num-runs $NUM_RUNS --run-number $RUN_NUMBER"
CMD="$CMD --m-projects $M_PROJECTS"
CMD="$CMD --alpha $ALPHA --sigma $SIGMA"
CMD="$CMD --c-min $C_MIN --c-max $C_MAX"
CMD="$CMD --max-rounds $MAX_ROUNDS"
CMD="$CMD --random-seed $SEED"
CMD="$CMD --discussion-turns $DISCUSSION_TURNS"
CMD="$CMD --model-order $MODEL_ORDER"
CMD="$CMD --max-tokens-per-phase $MAX_TOKENS"
CMD="$CMD --output-dir $OUTPUT_DIR"
CMD="$CMD --job-id $CONFIG_ID"

# Add TTC-specific flags only for ttc_scaling experiments
if [[ "$EXPERIMENT_TYPE" == "ttc_scaling" ]]; then
    TOKEN_BUDGET=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['reasoning_token_budget'])")
    REASONING_PHASES=$(python3 -c "import json; print(' '.join(json.load(open('${CONFIG_FILE}'))['reasoning_budget_phases']))")
    PROMPT_ONLY=$(python3 -c "import json; print(str(json.load(open('${CONFIG_FILE}')).get('prompt_only', False)).lower())")

    echo "Token budget: $TOKEN_BUDGET"
    echo "Reasoning phases: $REASONING_PHASES"
    echo "Prompt only: $PROMPT_ONLY"
    echo ""

    CMD="$CMD --reasoning-token-budget $TOKEN_BUDGET"
    CMD="$CMD --reasoning-budget-phases $REASONING_PHASES"

    if [[ "$PROMPT_ONLY" == "true" ]]; then
        CMD="$CMD --prompt-only"
    fi
fi

echo "Running: $CMD"
echo ""
eval $CMD
LOCAL_EOF
chmod +x "${LOCAL_RUN_SCRIPT}"

echo "Created local run script: ${LOCAL_RUN_SCRIPT}"

echo ""
echo "============================================================"
echo "Configuration generation complete!"
echo "============================================================"
echo ""
echo "Generated: ${TOTAL_COUNT} experiment configurations"
echo "  Model-scale: ${MS_TOTAL} configs"
echo "  TTC-scaling: ${TTC_TOTAL} configs"
echo "Location: ${CONFIG_DIR}"
echo ""
echo "Quick start:"
echo ""
echo "  1. Derisk (run 1 config locally):"
echo "     ${LOCAL_RUN_SCRIPT} 0"
echo ""
echo "  2. Submit to SLURM:"
echo "     ${SUBMIT_SCRIPT}                      # All jobs"
echo "     ${SUBMIT_SCRIPT} --test               # Only config 0"
echo "     ${SUBMIT_SCRIPT} --max-concurrent 10  # Limit concurrency"
echo ""
echo "  3. Or run directly (no config generator):"
echo "     python3 run_strong_models_experiment.py \\"
echo "       --game-type co_funding \\"
echo "       --models gpt-5-nano claude-opus-4-5-thinking-32k \\"
echo "       --m-projects 5 --alpha 0.5 --sigma 0.5 \\"
echo "       --max-rounds 10 --batch --num-runs 1 --random-seed 42"
echo ""

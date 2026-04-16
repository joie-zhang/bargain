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
#      - Sweep over alpha (preference alignment) and sigma (budget abundance scale)
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
#   - sigma: Budget scarcity ratio (0, 1]
#     - sigma = B/C = total budget / total project cost
#     - sigma ~ 0.2: budget ~20% of total cost
#     - sigma ~ 1.0: budget = 100% of total cost
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
        --nano-vs-opus)
            MODE="nano_vs_opus"
            shift
            ;;
        --nano-vs-weak)
            MODE="nano_vs_weak"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--derisk|--small|--model-scale|--scaling|--conservative|--ambitious|--nano-vs-opus|--nano-vs-weak]"
            echo ""
            echo "Experiment modes:"
            echo "  --conservative  Model-scale sweep: 4 pairs, 3x3 alpha/sigma grid, 1 run (72 configs)"
            echo "  --ambitious     Model-scale sweep: 4 pairs, 5x5 alpha/sigma grid, 3 runs (600 configs)"
            echo "  --model-scale   Model capability experiments: multiple model pairs, sweep alpha/sigma, no TTC"
            echo "  --scaling       TTC scaling experiments: reasoning models vs baseline, sweep token budgets"
            echo "  --small         Reduced version of full experiment (fewer models, fewer params)"
            echo "  --derisk        Minimal smoke test (1 config)"
            echo "  --nano-vs-opus  gpt-5-nano vs claude-opus-4-6, 2 explicit param pairs, both orders (4 configs)"
            echo "  --nano-vs-weak  gpt-5-nano vs gpt-5-nano and vs amazon-nova-micro-v1.0, same pairs (8 configs)"
            echo "  (default)       Full experiment: both model-scale AND TTC scaling"
            echo ""
            echo "Environment:"
            echo "  COFUNDING_DISCUSSION_TRANSPARENCY=aggregate|own|full (default: own)"
            echo "  COFUNDING_ENABLE_COMMIT_VOTE=true|false (default: true)"
            echo "  COFUNDING_ENABLE_TIME_DISCOUNT=true|false (default: true)"
            echo "  COFUNDING_TIME_DISCOUNT=0.0..1.0 (default: 0.9)"
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
DISCUSSION_TURNS=2
MAX_TOKENS_PER_PHASE=10500
BASE_SEED=42
NUM_RUNS=2
MS_PARAM_PAIRS=()  # explicit "alpha:sigma" pairs; if non-empty, overrides MS_ALPHA_VALUES × MS_SIGMA_VALUES grid
C_MIN=10.0
C_MAX=30.0
COFUNDING_DISCUSSION_TRANSPARENCY="${COFUNDING_DISCUSSION_TRANSPARENCY:-own}"
COFUNDING_ENABLE_COMMIT_VOTE="${COFUNDING_ENABLE_COMMIT_VOTE:-true}"
COFUNDING_ENABLE_TIME_DISCOUNT="${COFUNDING_ENABLE_TIME_DISCOUNT:-true}"
COFUNDING_TIME_DISCOUNT="${COFUNDING_TIME_DISCOUNT:-0.9}"

case "${COFUNDING_DISCUSSION_TRANSPARENCY}" in
    aggregate|own|full) ;;
    *)
        echo "ERROR: COFUNDING_DISCUSSION_TRANSPARENCY must be one of: aggregate, own, full"
        echo "Got: ${COFUNDING_DISCUSSION_TRANSPARENCY}"
        exit 1
        ;;
esac

case "${COFUNDING_ENABLE_COMMIT_VOTE}" in
    true|false) ;;
    *)
        echo "ERROR: COFUNDING_ENABLE_COMMIT_VOTE must be true or false"
        echo "Got: ${COFUNDING_ENABLE_COMMIT_VOTE}"
        exit 1
        ;;
esac

case "${COFUNDING_ENABLE_TIME_DISCOUNT}" in
    true|false) ;;
    *)
        echo "ERROR: COFUNDING_ENABLE_TIME_DISCOUNT must be true or false"
        echo "Got: ${COFUNDING_ENABLE_TIME_DISCOUNT}"
        exit 1
        ;;
esac

python3 - << EOF
gamma = float("${COFUNDING_TIME_DISCOUNT}")
if not (0.0 <= gamma <= 1.0):
    raise SystemExit(f"ERROR: COFUNDING_TIME_DISCOUNT must be in [0,1], got {gamma}")
EOF

# Reasoning phases (only used for TTC-scaling configs)
REASONING_PHASES_JSON='["thinking", "reflection", "discussion", "proposal", "voting"]'
PROMPT_ONLY="true"

# =============================================================================
# Mode-Specific Parameter Definitions
# =============================================================================

# Requested Mar 2026 mixed-provider run set
# API models:
#   - claude-opus-4-6 (Anthropic API)
#   - claude-haiku-4-5 (Anthropic API)
#   - gpt-5.2-chat-latest-20260210 (OpenAI API)
# Local cluster models:
#   - qwen3-32b (4x A100 80GB)
# OpenRouter-routed models:
#   - llama-3.1-8b-instruct (CPU/API path; no GPU required)
BASELINE_MODEL="gpt-5-nano"
ADVERSARY_MODELS=(
    "claude-opus-4-6-thinking"
    "claude-opus-4-6"
    "gemini-3-pro"
    "gpt-5.4-high"
    "gpt-5.2-chat-latest-20260210"
    "claude-opus-4-5-20251101-thinking-32k"
    "claude-opus-4-5-20251101"
    "gemini-2.5-pro"
    "qwen3-max-preview"
    "deepseek-r1-0528"
    "claude-haiku-4-5-20251001"
    "deepseek-r1"
    "claude-sonnet-4-20250514"
    "gemma-3-27b-it"
    "o3-mini-high"
    "deepseek-v3"
    "gpt-4o-2024-05-13"
    "gpt-5-nano-high"
    "qwq-32b"
    "gpt-4.1-nano-2025-04-14"
    "llama-3.3-70b-instruct"
    "gpt-4o-mini-2024-07-18"
    "qwen2.5-72b-instruct"
    "amazon-nova-pro-v1.0"
    "command-r-plus-08-2024"
    "claude-3-haiku-20240307"
    "amazon-nova-micro-v1.0"
    "llama-3.1-8b-instruct"
    "llama-3.2-3b-instruct"
    "llama-3.2-1b-instruct"
)
PRIMARY_MODEL_PAIRS=(
    "${BASELINE_MODEL},${ADVERSARY_MODELS[0]}"
    "${BASELINE_MODEL},${ADVERSARY_MODELS[1]}"
    "${BASELINE_MODEL},${ADVERSARY_MODELS[2]}"
    "${BASELINE_MODEL},${ADVERSARY_MODELS[3]}"
    "${BASELINE_MODEL},${ADVERSARY_MODELS[4]}"
    "${BASELINE_MODEL},${ADVERSARY_MODELS[5]}"
    "${BASELINE_MODEL},${ADVERSARY_MODELS[6]}"
    "${BASELINE_MODEL},${ADVERSARY_MODELS[7]}"
    "${BASELINE_MODEL},${ADVERSARY_MODELS[8]}"
    "${BASELINE_MODEL},${ADVERSARY_MODELS[9]}"
    "${BASELINE_MODEL},${ADVERSARY_MODELS[10]}"
    "${BASELINE_MODEL},${ADVERSARY_MODELS[11]}"
    "${BASELINE_MODEL},${ADVERSARY_MODELS[12]}"
    "${BASELINE_MODEL},${ADVERSARY_MODELS[13]}"
    "${BASELINE_MODEL},${ADVERSARY_MODELS[14]}"
    "${BASELINE_MODEL},${ADVERSARY_MODELS[15]}"
    "${BASELINE_MODEL},${ADVERSARY_MODELS[16]}"
    "${BASELINE_MODEL},${ADVERSARY_MODELS[17]}"
    "${BASELINE_MODEL},${ADVERSARY_MODELS[18]}"
    "${BASELINE_MODEL},${ADVERSARY_MODELS[19]}"
    "${BASELINE_MODEL},${ADVERSARY_MODELS[20]}"
    "${BASELINE_MODEL},${ADVERSARY_MODELS[21]}"
    "${BASELINE_MODEL},${ADVERSARY_MODELS[22]}"
    "${BASELINE_MODEL},${ADVERSARY_MODELS[23]}"
    "${BASELINE_MODEL},${ADVERSARY_MODELS[24]}"
    "${BASELINE_MODEL},${ADVERSARY_MODELS[25]}"
    "${BASELINE_MODEL},${ADVERSARY_MODELS[26]}"
    "${BASELINE_MODEL},${ADVERSARY_MODELS[27]}"
    "${BASELINE_MODEL},${ADVERSARY_MODELS[28]}"
    "${BASELINE_MODEL},${ADVERSARY_MODELS[29]}"
)

# --- MODEL-SCALE EXPERIMENTS ---
# Model pairs: "model1,model2" where model1 is the weak/baseline model
# Use model_order=weak_first/strong_first to control speaking order
# These run WITHOUT reasoning token budget -- pure model capability comparison.

if [[ "$MODE" == "conservative" ]]; then
    MODEL_PAIRS=("${PRIMARY_MODEL_PAIRS[@]}")
    MS_ALPHA_VALUES=(0.0 0.5 1.0)
    MS_SIGMA_VALUES=(0.2 0.6 1.0)
    MS_MODEL_ORDERS=("weak_first" "strong_first")
    NUM_RUNS=1
elif [[ "$MODE" == "ambitious" ]]; then
    MODEL_PAIRS=("${PRIMARY_MODEL_PAIRS[@]}")
    MS_ALPHA_VALUES=(0.0 0.25 0.5 0.75 1.0)
    MS_SIGMA_VALUES=(0.2 0.4 0.6 0.8 1.0)
    MS_MODEL_ORDERS=("weak_first" "strong_first")
    NUM_RUNS=3
elif [[ "$MODE" == "derisk" ]]; then
    MODEL_PAIRS=(
        "${BASELINE_MODEL},${ADVERSARY_MODELS[0]}"
    )
    MS_ALPHA_VALUES=(1.0)
    MS_SIGMA_VALUES=(1.0)
    MS_MODEL_ORDERS=("weak_first")
    MAX_ROUNDS=2
elif [[ "$MODE" == "small" ]]; then
    MODEL_PAIRS=("${PRIMARY_MODEL_PAIRS[@]}")
    MS_ALPHA_VALUES=(0.0 0.5 1.0)
    MS_SIGMA_VALUES=(0.3 0.5 0.8)
    MS_MODEL_ORDERS=("weak_first" "strong_first")
elif [[ "$MODE" == "model_scale" ]]; then
    MODEL_PAIRS=("${PRIMARY_MODEL_PAIRS[@]}")
    MS_ALPHA_VALUES=(0.0 0.2 0.4 0.6 0.8 1.0)
    MS_SIGMA_VALUES=(0.2 0.4 0.6 0.8 1.0)
    MS_MODEL_ORDERS=("weak_first" "strong_first")
elif [[ "$MODE" == "full" ]]; then
    MODEL_PAIRS=("${PRIMARY_MODEL_PAIRS[@]}")
    MS_ALPHA_VALUES=(0.0 0.2 0.4 0.6 0.8 1.0)
    MS_SIGMA_VALUES=(0.2 0.4 0.6 0.8 1.0)
    MS_MODEL_ORDERS=("weak_first" "strong_first")
elif [[ "$MODE" == "nano_vs_opus" ]]; then
    MODEL_PAIRS=("gpt-5-nano,claude-opus-4-6")
    MS_PARAM_PAIRS=("0.0:0.3" "1.0:1.0")  # explicit pairs: (alpha=0,sigma=0.3) and (alpha=1,sigma=1.0)
    MS_ALPHA_VALUES=()   # unused — overridden by MS_PARAM_PAIRS
    MS_SIGMA_VALUES=()   # unused — overridden by MS_PARAM_PAIRS
    MS_MODEL_ORDERS=("weak_first" "strong_first")
    NUM_RUNS=1
    MAX_ROUNDS=5
elif [[ "$MODE" == "nano_vs_weak" ]]; then
    MODEL_PAIRS=(
        "gpt-5-nano,gpt-5-nano"
        "gpt-5-nano,amazon-nova-micro-v1.0"
    )
    MS_PARAM_PAIRS=("0.0:0.3" "1.0:1.0")  # same explicit pairs as nano_vs_opus
    MS_ALPHA_VALUES=()   # unused — overridden by MS_PARAM_PAIRS
    MS_SIGMA_VALUES=()   # unused — overridden by MS_PARAM_PAIRS
    MS_MODEL_ORDERS=("weak_first" "strong_first")
    NUM_RUNS=1
    MAX_ROUNDS=5
else
    # scaling mode -- no model-scale configs
    MODEL_PAIRS=()
    MS_ALPHA_VALUES=()
    MS_SIGMA_VALUES=()
    MS_MODEL_ORDERS=()
fi

# --- TTC SCALING EXPERIMENTS ---
# Reasoning model vs baseline, sweep token budgets, fixed game params
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
    TTC_REASONING_MODELS=("${ADVERSARY_MODELS[@]}")
    TTC_TOKEN_BUDGETS=(100 1000 5000)
    TTC_ALPHA_VALUES=(0.5)
    TTC_SIGMA_VALUES=(0.5)
    TTC_MODEL_ORDERS=("weak_first" "strong_first")
elif [[ "$MODE" == "scaling" ]]; then
    TTC_REASONING_MODELS=("${ADVERSARY_MODELS[@]}")
    TTC_TOKEN_BUDGETS=(100 500 1000 5000 10000)
    TTC_ALPHA_VALUES=(0.5)
    TTC_SIGMA_VALUES=(0.5)
    TTC_MODEL_ORDERS=("weak_first" "strong_first")
elif [[ "$MODE" == "full" ]]; then
    TTC_REASONING_MODELS=("${ADVERSARY_MODELS[@]}")
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
if [[ ${#MS_PARAM_PAIRS[@]} -gt 0 ]]; then
    MS_TOTAL=$((${#MODEL_PAIRS[@]} * ${#MS_PARAM_PAIRS[@]} * ${#MS_MODEL_ORDERS[@]} * NUM_RUNS))
else
    MS_TOTAL=$((${#MODEL_PAIRS[@]} * ${#MS_ALPHA_VALUES[@]} * ${#MS_SIGMA_VALUES[@]} * ${#MS_MODEL_ORDERS[@]} * NUM_RUNS))
fi
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

# Build unified (alpha:sigma) pair list — explicit pairs take priority over grid
if [[ ${#MS_PARAM_PAIRS[@]} -gt 0 ]]; then
    _MS_ALPHA_SIGMA_PAIRS=("${MS_PARAM_PAIRS[@]}")
else
    _MS_ALPHA_SIGMA_PAIRS=()
    for _alpha in "${MS_ALPHA_VALUES[@]}"; do
        for _sigma in "${MS_SIGMA_VALUES[@]}"; do
            _MS_ALPHA_SIGMA_PAIRS+=("${_alpha}:${_sigma}")
        done
    done
fi

for model_pair in "${MODEL_PAIRS[@]}"; do
    IFS=',' read -ra MODELS <<< "${model_pair}"
    MODEL1="${MODELS[0]}"
    MODEL2="${MODELS[1]}"

    for alpha_sigma in "${_MS_ALPHA_SIGMA_PAIRS[@]}"; do
        IFS=':' read -r alpha sigma <<< "${alpha_sigma}"
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
    "cofunding_discussion_transparency": "${COFUNDING_DISCUSSION_TRANSPARENCY}",
    "cofunding_enable_commit_vote": ${COFUNDING_ENABLE_COMMIT_VOTE},
    "cofunding_enable_time_discount": ${COFUNDING_ENABLE_TIME_DISCOUNT},
    "cofunding_time_discount": ${COFUNDING_TIME_DISCOUNT},
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
    "cofunding_discussion_transparency": "${COFUNDING_DISCUSSION_TRANSPARENCY}",
    "cofunding_enable_commit_vote": ${COFUNDING_ENABLE_COMMIT_VOTE},
    "cofunding_enable_time_discount": ${COFUNDING_ENABLE_TIME_DISCOUNT},
    "cofunding_time_discount": ${COFUNDING_TIME_DISCOUNT},
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
  Discussion transparency: ${COFUNDING_DISCUSSION_TRANSPARENCY}
  Commit vote enabled: ${COFUNDING_ENABLE_COMMIT_VOTE}
  Time discount enabled: ${COFUNDING_ENABLE_TIME_DISCOUNT}
  Time discount gamma: ${COFUNDING_TIME_DISCOUNT}
  Max tokens per phase: ${MAX_TOKENS_PER_PHASE}
  Runs per configuration: ${NUM_RUNS}
  Project cost range: [${C_MIN}, ${C_MAX}]

Key Game 3 parameters:
  alpha: Preference alignment [0, 1]
    - alpha ~ 0: orthogonal valuations (agents value different projects)
    - alpha ~ 0.5: moderate alignment
    - alpha ~ 1: identical valuations (agents value same projects)
  sigma: Budget scarcity ratio (0, 1]
    - sigma = B/C = total budget / total project cost
    - sigma ~ 0.2: budget ~20% of total cost
    - sigma ~ 0.5: budget ~50% of total cost
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
SLURM_TIME="12:00:00"
if [[ "$MODE" == "conservative" || "$MODE" == "ambitious" ]]; then
    API_SLURM_PARTITION="cpu"
else
    API_SLURM_PARTITION=""
fi

if [[ -n "$API_SLURM_PARTITION" ]]; then
    API_PARTITION_LINE="#SBATCH --partition=${API_SLURM_PARTITION}"
else
    API_PARTITION_LINE=""
fi

WORKER_SCRIPT="${SLURM_DIR}/run_cofunding_worker.sh"
cat > "${WORKER_SCRIPT}" << SLURM_WORKER_EOF
#!/bin/bash
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
# OpenRouter routing: proxy monitor first, then direct fallback.
export OPENROUTER_TRANSPORT="\${OPENROUTER_TRANSPORT:-auto}"
echo "Python version: \$(python3 --version)"
echo "OpenRouter transport: \$OPENROUTER_TRANSPORT"
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
DISCUSSION_TRANSPARENCY=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}')).get('cofunding_discussion_transparency', 'own'))")
ENABLE_COMMIT_VOTE=\$(python3 -c "import json; print(str(json.load(open('\${CONFIG_FILE}')).get('cofunding_enable_commit_vote', True)).lower())")
ENABLE_TIME_DISCOUNT=\$(python3 -c "import json; print(str(json.load(open('\${CONFIG_FILE}')).get('cofunding_enable_time_discount', True)).lower())")
TIME_DISCOUNT=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}')).get('cofunding_time_discount', 0.9))")

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
echo "Discussion transparency: \$DISCUSSION_TRANSPARENCY"
echo "Commit vote enabled: \$ENABLE_COMMIT_VOTE"
echo "Time discount enabled: \$ENABLE_TIME_DISCOUNT (gamma=\$TIME_DISCOUNT)"
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
CMD="\$CMD --cofunding-discussion-transparency \$DISCUSSION_TRANSPARENCY"
CMD="\$CMD --model-order \$MODEL_ORDER"
CMD="\$CMD --max-tokens-per-phase \$MAX_TOKENS"
CMD="\$CMD --output-dir \$OUTPUT_DIR"
CMD="\$CMD --job-id \$SLURM_ARRAY_TASK_ID"
CMD="\$CMD --cofunding-time-discount \$TIME_DISCOUNT"
if [[ "\$ENABLE_COMMIT_VOTE" != "true" ]]; then
    CMD="\$CMD --cofunding-disable-commit-vote"
fi
if [[ "\$ENABLE_TIME_DISCOUNT" != "true" ]]; then
    CMD="\$CMD --cofunding-disable-time-discount"
fi

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

run_with_logging() {
    local cmd="\$1"
    local log_file="\$2"
    set +e
    eval "\$cmd" 2>&1 | tee "\$log_file"
    local status=\${PIPESTATUS[0]}
    set -e
    return \$status
}

TMP_RUN_LOG=\$(mktemp)

if run_with_logging "\$CMD" "\$TMP_RUN_LOG"; then
    echo ""
    echo "============================================================"
    echo "Experiment completed successfully at: \$(date)"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "Experiment failed at: \$(date)"
    echo "============================================================"
    rm -f "\$TMP_RUN_LOG"
    exit 1
fi

rm -f "\$TMP_RUN_LOG"
SLURM_WORKER_EOF
chmod +x "${WORKER_SCRIPT}"
echo "Created worker script: ${WORKER_SCRIPT}"

create_sbatch_wrapper() {
    local script_path="$1"
    local job_name="$2"
    local cpus="$3"
    local mem="$4"
    local partition_line="$5"
    local gres_lines="$6"
    local output_pattern="$7"
    local error_pattern="$8"

    cat > "${script_path}" << SLURM_WRAPPER_EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${cpus}
#SBATCH --mem=${mem}
#SBATCH --time=${SLURM_TIME}
${partition_line}
${gres_lines}
#SBATCH --output=${output_pattern}
#SBATCH --error=${error_pattern}

set -e
bash "${WORKER_SCRIPT}" "\$@"
SLURM_WRAPPER_EOF

    chmod +x "${script_path}"
}

API_SLURM_SCRIPT="${SLURM_DIR}/run_cofunding_api_experiments.sbatch"
GPU2_SLURM_SCRIPT="${SLURM_DIR}/run_cofunding_gpu_2x80gb.sbatch"
GPU4_SLURM_SCRIPT="${SLURM_DIR}/run_cofunding_gpu_4x80gb.sbatch"

GPU2_GRES_LINES=$'#SBATCH --constraint=gpu80\n#SBATCH --gres=gpu:a100:2'
GPU4_GRES_LINES=$'#SBATCH --constraint=gpu80\n#SBATCH --gres=gpu:a100:4'

create_sbatch_wrapper \
    "${API_SLURM_SCRIPT}" \
    "cofund-api" \
    "4" \
    "8G" \
    "${API_PARTITION_LINE}" \
    "" \
    "logs/cluster/cofund_api_%A_%a.out" \
    "logs/cluster/cofund_api_%A_%a.err"

create_sbatch_wrapper \
    "${GPU2_SLURM_SCRIPT}" \
    "cofund-gpu2" \
    "8" \
    "64G" \
    "" \
    "${GPU2_GRES_LINES}" \
    "logs/cluster/cofund_gpu2_%A_%a.out" \
    "logs/cluster/cofund_gpu2_%A_%a.err"

create_sbatch_wrapper \
    "${GPU4_SLURM_SCRIPT}" \
    "cofund-gpu4" \
    "16" \
    "128G" \
    "" \
    "${GPU4_GRES_LINES}" \
    "logs/cluster/cofund_gpu4_%A_%a.out" \
    "logs/cluster/cofund_gpu4_%A_%a.err"

# Backward-compatible alias
cp "${API_SLURM_SCRIPT}" "${SLURM_DIR}/run_cofunding_experiments.sbatch"

echo "Created SLURM script: ${API_SLURM_SCRIPT}"
echo "Created SLURM script: ${GPU2_SLURM_SCRIPT}"
echo "Created SLURM script: ${GPU4_SLURM_SCRIPT}"
echo "Created SLURM alias : ${SLURM_DIR}/run_cofunding_experiments.sbatch"

# Create submission script
SUBMIT_SCRIPT="${SLURM_DIR}/submit_all.sh"
cat > "${SUBMIT_SCRIPT}" << 'SUBMIT_EOF'
#!/bin/bash
# Submit co-funding experiment jobs with API/GPU routing
# Usage: ./submit_all.sh [api|gpu|gpu2|gpu4|all] [--test] [--max-concurrent <num>]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${BASE_DIR}"

JOB_MODE="all"
if [[ $# -gt 0 && "$1" != --* ]]; then
    JOB_MODE="$1"
    shift
fi

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
mapfile -t CONFIG_FILES < <(ls "${CONFIG_DIR}"/config_*.json 2>/dev/null || true)
TOTAL_CONFIGS=${#CONFIG_FILES[@]}

if [[ "$TOTAL_CONFIGS" -eq 0 ]]; then
    echo "Error: No config files found"
    exit 1
fi

API_SCRIPT="${SCRIPT_DIR}/run_cofunding_api_experiments.sbatch"
GPU2_SCRIPT="${SCRIPT_DIR}/run_cofunding_gpu_2x80gb.sbatch"
GPU4_SCRIPT="${SCRIPT_DIR}/run_cofunding_gpu_4x80gb.sbatch"

API_IDS=()
GPU2_IDS=()
GPU4_IDS=()

for config_file in "${CONFIG_FILES[@]}"; do
    config_id="$(basename "$config_file")"
    config_id="${config_id#config_}"
    config_id="${config_id%.json}"
    config_id=$((10#$config_id))

    if [[ "$TEST_MODE" == "true" && "$config_id" -ne 0 ]]; then
        continue
    fi

    required_gpus=$(python3 - "$config_file" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = json.load(f)

models = cfg.get("models", [])
gpu_map = {
    "qwen3-32b": 4,
}

required = sum(gpu_map.get(m, 0) for m in models)
print(required)
PY
)

    case "$required_gpus" in
        0)
            API_IDS+=("$config_id")
            ;;
        1|2)
            GPU2_IDS+=("$config_id")
            ;;
        3|4)
            GPU4_IDS+=("$config_id")
            ;;
        *)
            echo "ERROR: Config ${config_id} requires ${required_gpus} GPUs; unsupported by generated sbatch files."
            echo "Update submit routing or add another GPU sbatch template before submitting."
            exit 1
            ;;
    esac
done

if [[ "$TEST_MODE" == "true" ]]; then
    echo "Test mode: only config 0 will be submitted (if present)."
fi

echo "Total configurations: ${TOTAL_CONFIGS}"
echo "  API configs : ${#API_IDS[@]}"
echo "  GPU(2x80GB): ${#GPU2_IDS[@]}"
echo "  GPU(4x80GB): ${#GPU4_IDS[@]}"

submit_group() {
    local label="$1"
    local sbatch_script="$2"
    shift 2
    local ids=("$@")

    if [[ "${#ids[@]}" -eq 0 ]]; then
        echo "Skipping ${label}: no matching configs"
        return 0
    fi

    local array_spec
    array_spec=$(IFS=','; echo "${ids[*]}")
    if [[ -n "$MAX_CONCURRENT" ]]; then
        array_spec="${array_spec}%${MAX_CONCURRENT}"
    fi

    echo "Submitting ${label}: ${array_spec}"
    sbatch --array="${array_spec}" "${sbatch_script}"
}

case "$JOB_MODE" in
    api)
        submit_group "API" "$API_SCRIPT" "${API_IDS[@]}"
        ;;
    gpu)
        submit_group "GPU 2x80GB" "$GPU2_SCRIPT" "${GPU2_IDS[@]}"
        submit_group "GPU 4x80GB" "$GPU4_SCRIPT" "${GPU4_IDS[@]}"
        ;;
    gpu2)
        submit_group "GPU 2x80GB" "$GPU2_SCRIPT" "${GPU2_IDS[@]}"
        ;;
    gpu4)
        submit_group "GPU 4x80GB" "$GPU4_SCRIPT" "${GPU4_IDS[@]}"
        ;;
    all)
        submit_group "API" "$API_SCRIPT" "${API_IDS[@]}"
        submit_group "GPU 2x80GB" "$GPU2_SCRIPT" "${GPU2_IDS[@]}"
        submit_group "GPU 4x80GB" "$GPU4_SCRIPT" "${GPU4_IDS[@]}"
        ;;
    *)
        echo "Usage: $0 [api|gpu|gpu2|gpu4|all] [--test] [--max-concurrent <num>]"
        exit 1
        ;;
esac

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
export OPENROUTER_TRANSPORT="${OPENROUTER_TRANSPORT:-auto}"
echo "OpenRouter transport: $OPENROUTER_TRANSPORT"

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
DISCUSSION_TRANSPARENCY=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}')).get('cofunding_discussion_transparency', 'own'))")
ENABLE_COMMIT_VOTE=$(python3 -c "import json; print(str(json.load(open('${CONFIG_FILE}')).get('cofunding_enable_commit_vote', True)).lower())")
ENABLE_TIME_DISCOUNT=$(python3 -c "import json; print(str(json.load(open('${CONFIG_FILE}')).get('cofunding_enable_time_discount', True)).lower())")
TIME_DISCOUNT=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}')).get('cofunding_time_discount', 0.9))")
MODELS=$(python3 -c "import json; print(' '.join(json.load(open('${CONFIG_FILE}'))['models']))")

echo "Experiment type: $EXPERIMENT_TYPE"
echo "Model order: $MODEL_ORDER"
echo "Models: $MODELS"
echo "Alpha: $ALPHA | Sigma: $SIGMA | Projects: $M_PROJECTS | Max rounds: $MAX_ROUNDS"
echo "Num runs: $NUM_RUNS | Run number: $RUN_NUMBER"
echo "Discussion transparency: $DISCUSSION_TRANSPARENCY"
echo "Commit vote enabled: $ENABLE_COMMIT_VOTE"
echo "Time discount enabled: $ENABLE_TIME_DISCOUNT (gamma=$TIME_DISCOUNT)"
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
CMD="$CMD --cofunding-discussion-transparency $DISCUSSION_TRANSPARENCY"
CMD="$CMD --model-order $MODEL_ORDER"
CMD="$CMD --max-tokens-per-phase $MAX_TOKENS"
CMD="$CMD --output-dir $OUTPUT_DIR"
CMD="$CMD --job-id $CONFIG_ID"
CMD="$CMD --cofunding-time-discount $TIME_DISCOUNT"
if [[ "$ENABLE_COMMIT_VOTE" != "true" ]]; then
    CMD="$CMD --cofunding-disable-commit-vote"
fi
if [[ "$ENABLE_TIME_DISCOUNT" != "true" ]]; then
    CMD="$CMD --cofunding-disable-time-discount"
fi

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

run_with_logging_local() {
    local cmd="$1"
    local log_file="$2"
    set +e
    eval "$cmd" 2>&1 | tee "$log_file"
    local status=${PIPESTATUS[0]}
    set -e
    return $status
}

TMP_RUN_LOG=$(mktemp)

if run_with_logging_local "$CMD" "$TMP_RUN_LOG"; then
    rm -f "$TMP_RUN_LOG"
    exit 0
fi

STATUS=1
rm -f "$TMP_RUN_LOG"
exit $STATUS
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
echo "     ${SUBMIT_SCRIPT} all                  # All jobs (API + GPU)"
echo "     ${SUBMIT_SCRIPT} api                  # API-only jobs"
echo "     ${SUBMIT_SCRIPT} gpu                  # GPU jobs (2x + 4x A100 80GB)"
echo "     ${SUBMIT_SCRIPT} all --test           # Only config 0"
echo "     ${SUBMIT_SCRIPT} all --max-concurrent 10  # Limit concurrency per submitted group"
echo ""
echo "  3. Submit co-funding first, then diplomacy:"
echo "     ./scripts/submit_cofunding_then_diplomacy.sh"
echo ""
echo "  4. Or run directly (no config generator):"
echo "     python3 run_strong_models_experiment.py \\"
echo "       --game-type co_funding \\"
echo "       --models gpt-5.2-chat-latest-20260210 claude-opus-4-6 \\"
echo "       --m-projects 5 --alpha 0.5 --sigma 0.5 \\"
echo "       --max-rounds 10 --batch --num-runs 1 --random-seed 42"
echo ""

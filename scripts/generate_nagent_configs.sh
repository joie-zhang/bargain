#!/bin/bash
# =============================================================================
# Generate Configuration Files for N > 2 Agent Scaling Experiments
# =============================================================================
#
# Generates JSON config files for multi-agent (N=3,4,5) negotiation experiments
# across all 3 game types (Item Allocation, Diplomacy, Co-Funding).
#
# Three experiment designs:
#   A) Scaling Robustness: 1 focal adversary + (N-1) copies of gpt-5-nano
#   B) Coalition Power:    1 strong (claude-haiku-4-5) + K copies of llama-3.1-8b
#   C) Capability Gradient: Mixed-capability groups (quartet + trios)
#
# Usage:
#   ./scripts/generate_nagent_configs.sh              # All experiments (A+B+C)
#   ./scripts/generate_nagent_configs.sh --exp-a       # Experiment A only
#   ./scripts/generate_nagent_configs.sh --exp-b       # Experiment B only
#   ./scripts/generate_nagent_configs.sh --exp-c       # Experiment C only
#   ./scripts/generate_nagent_configs.sh --derisk      # Minimal smoke test (1 config per game)
#
# What it creates:
#   experiments/results/nagent_<timestamp>/configs/
#   ├── config_0000.json ... config_NNNN.json
#   ├── all_configs.txt
#   ├── experiment_index.csv
#   ├── summary.txt
#   └── slurm/
#       ├── run_nagent_worker.sh
#       ├── run_nagent_api_experiments.sbatch
#       ├── run_nagent_gpu_2x80gb.sbatch
#       ├── run_nagent_gpu_4x80gb.sbatch
#       ├── run_nagent_mixed_gpu.sbatch
#       ├── submit_all.sh
#       └── run_local.sh
#
# Diplomacy PSD constraint:
#   For N > 2, negative rho is clamped: rho >= -1/(N-1)
#   N=3: rho >= -0.50, N=4: rho >= -0.33, N=5: rho >= -0.25
#
# Configuration (edit these variables):
#   - BASELINE_MODEL: Constant opponent (default: gpt-5-nano)
#   - FOCAL_MODELS: Adversary models for Experiment A
#   - STRONG_MODEL, WEAK_MODEL: For Experiment B
#   - N_VALUES: Agent counts to test
#   - Game-specific competition parameters
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
        --exp-a)
            MODE="exp_a"
            shift
            ;;
        --exp-b)
            MODE="exp_b"
            shift
            ;;
        --exp-c)
            MODE="exp_c"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--derisk|--exp-a|--exp-b|--exp-c]"
            echo ""
            echo "Experiment modes:"
            echo "  --exp-a    Scaling Robustness: 1 focal + (N-1) gpt-5-nano"
            echo "  --exp-b    Coalition Power: 1 strong + K weak"
            echo "  --exp-c    Capability Gradient: mixed groups"
            echo "  --derisk   Minimal smoke test (3 configs, one per game)"
            echo "  (default)  Full: all experiments A + B + C"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Model Panel
# =============================================================================
BASELINE_MODEL="gpt-5-nano"         # Elo 1338, OpenAI API
FOCAL_MODELS=(
    "claude-haiku-4-5"              # Elo 1403, Anthropic API
    "qwen3-32b"                     # Elo ~1360, GPU 4x80GB
    "llama-3.1-8b-instruct"         # Elo ~1180, GPU 2x80GB
)
STRONG_MODEL="claude-haiku-4-5"     # For Experiment B
WEAK_MODEL="llama-3.1-8b-instruct"  # For Experiment B

# =============================================================================
# Shared Game Configuration
# =============================================================================
MAX_ROUNDS=10
DISCUSSION_TURNS=2
MAX_TOKENS_PER_PHASE=10500
BASE_SEED=42
NUM_RUNS=1

# Game 1 (Item Allocation) parameters
IA_NUM_ITEMS=5
IA_GAMMA_DISCOUNT=0.9

# Game 2 (Diplomacy) parameters
DIPLO_N_ISSUES=5
DIPLO_THETA=0.5

# Game 3 (Co-Funding) parameters
CF_M_PROJECTS=5
CF_C_MIN=10.0
CF_C_MAX=30.0
CF_DISCUSSION_TRANSPARENCY="own"
CF_ENABLE_COMMIT_VOTE="true"
CF_ENABLE_TIME_DISCOUNT="true"
CF_TIME_DISCOUNT=0.9

# =============================================================================
# Experiment-Specific Parameters
# =============================================================================

# Competition levels per game type (2 levels each)
if [[ "$MODE" == "derisk" ]]; then
    IA_COMPETITION_LEVELS=(0.75)
    DIPLO_RHO_VALUES=(0.5)
    CF_ALPHA_VALUES=(1.0)
    CF_SIGMA_VALUES=(0.5)
    N_VALUES=(3)
    FOCAL_MODELS=("claude-haiku-4-5")
else
    IA_COMPETITION_LEVELS=(0.25 0.75)
    DIPLO_RHO_VALUES=(-0.4 0.5)
    CF_ALPHA_VALUES=(0.0 1.0)
    CF_SIGMA_VALUES=(0.5)
    N_VALUES=(3 4 5)
fi

# Create timestamped config directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DIR="${BASE_DIR}/experiments/results/nagent_${TIMESTAMP}"
CONFIG_DIR="${EXPERIMENT_DIR}/configs"
mkdir -p "${CONFIG_DIR}"

echo "Creating N-agent experiment directory: ${EXPERIMENT_DIR}"
echo "Config directory: ${CONFIG_DIR}"
echo "Mode: ${MODE}"
echo ""

EXPERIMENT_ID=0

# =============================================================================
# Helper: Clamp rho for PSD constraint
# =============================================================================
clamp_rho() {
    local rho="$1"
    local n_agents="$2"
    python3 -c "
rho = float('${rho}')
n = int('${n_agents}')
if n > 2 and rho < 0:
    psd_bound = -1.0 / (n - 1)
    rho = max(rho, psd_bound)
print(f'{rho:.4f}')
"
}

# =============================================================================
# Helper: Build models array JSON
# =============================================================================
build_models_json() {
    # Arguments: model1 model2 ... modelN
    local models=("$@")
    local json="["
    for ((i=0; i<${#models[@]}; i++)); do
        if [[ $i -gt 0 ]]; then
            json+=", "
        fi
        json+="\"${models[$i]}\""
    done
    json+="]"
    echo "$json"
}

# =============================================================================
# Helper: Generate a single config file
# =============================================================================
generate_config() {
    local experiment_type="$1"
    local game_type="$2"
    local models_json="$3"
    local n_agents="$4"
    local run_num="$5"
    local competition_desc="$6"
    # Game-specific args passed via environment variables

    local EXPERIMENT_ID_PADDED
    EXPERIMENT_ID_PADDED=$(printf "%04d" ${EXPERIMENT_ID})
    local CONFIG_FILE="${CONFIG_DIR}/config_${EXPERIMENT_ID_PADDED}.json"

    local SEED=$((BASE_SEED + EXPERIMENT_ID))

    # Build output directory
    local models_str
    models_str=$(echo "$models_json" | python3 -c "
import json, sys
models = json.load(sys.stdin)
print('_vs_'.join(m.replace('.', '_') for m in models))
")

    local game_params_str=""

    if [[ "$game_type" == "item_allocation" ]]; then
        local comp_str=$(echo "${_IA_COMP}" | sed 's/\./_/g')
        game_params_str="items${IA_NUM_ITEMS}_comp${comp_str}"

        cat > "${CONFIG_FILE}" << EOF
{
    "experiment_id": ${EXPERIMENT_ID},
    "experiment_type": "${experiment_type}",
    "experiment_design": "${competition_desc}",
    "game_type": "item_allocation",
    "models": ${models_json},
    "n_agents": ${n_agents},
    "num_items": ${IA_NUM_ITEMS},
    "competition_level": ${_IA_COMP},
    "gamma_discount": ${IA_GAMMA_DISCOUNT},
    "max_rounds": ${MAX_ROUNDS},
    "discussion_turns": ${DISCUSSION_TURNS},
    "max_tokens_per_phase": ${MAX_TOKENS_PER_PHASE},
    "run_number": ${run_num},
    "num_runs": ${NUM_RUNS},
    "random_seed": ${SEED},
    "output_dir": "experiments/results/nagent_${TIMESTAMP}/${experiment_type}/${game_type}/${models_str}/n${n_agents}_${game_params_str}"
}
EOF

    elif [[ "$game_type" == "diplomacy" ]]; then
        local rho_str=$(echo "${_DIPLO_RHO}" | sed 's/\./_/g; s/-/n/g')
        game_params_str="issues${DIPLO_N_ISSUES}_rho${rho_str}_theta${DIPLO_THETA}"

        cat > "${CONFIG_FILE}" << EOF
{
    "experiment_id": ${EXPERIMENT_ID},
    "experiment_type": "${experiment_type}",
    "experiment_design": "${competition_desc}",
    "game_type": "diplomacy",
    "models": ${models_json},
    "n_agents": ${n_agents},
    "n_issues": ${DIPLO_N_ISSUES},
    "rho": ${_DIPLO_RHO},
    "theta": ${DIPLO_THETA},
    "max_rounds": ${MAX_ROUNDS},
    "discussion_turns": ${DISCUSSION_TURNS},
    "max_tokens_per_phase": ${MAX_TOKENS_PER_PHASE},
    "run_number": ${run_num},
    "num_runs": ${NUM_RUNS},
    "random_seed": ${SEED},
    "output_dir": "experiments/results/nagent_${TIMESTAMP}/${experiment_type}/${game_type}/${models_str}/n${n_agents}_${game_params_str}"
}
EOF

    elif [[ "$game_type" == "co_funding" ]]; then
        local alpha_str=$(echo "${_CF_ALPHA}" | sed 's/\./_/g')
        local sigma_str=$(echo "${_CF_SIGMA}" | sed 's/\./_/g')
        game_params_str="proj${CF_M_PROJECTS}_alpha${alpha_str}_sigma${sigma_str}"

        cat > "${CONFIG_FILE}" << EOF
{
    "experiment_id": ${EXPERIMENT_ID},
    "experiment_type": "${experiment_type}",
    "experiment_design": "${competition_desc}",
    "game_type": "co_funding",
    "models": ${models_json},
    "n_agents": ${n_agents},
    "m_projects": ${CF_M_PROJECTS},
    "alpha": ${_CF_ALPHA},
    "sigma": ${_CF_SIGMA},
    "c_min": ${CF_C_MIN},
    "c_max": ${CF_C_MAX},
    "cofunding_discussion_transparency": "${CF_DISCUSSION_TRANSPARENCY}",
    "cofunding_enable_commit_vote": ${CF_ENABLE_COMMIT_VOTE},
    "cofunding_enable_time_discount": ${CF_ENABLE_TIME_DISCOUNT},
    "cofunding_time_discount": ${CF_TIME_DISCOUNT},
    "max_rounds": ${MAX_ROUNDS},
    "discussion_turns": ${DISCUSSION_TURNS},
    "max_tokens_per_phase": ${MAX_TOKENS_PER_PHASE},
    "run_number": ${run_num},
    "num_runs": ${NUM_RUNS},
    "random_seed": ${SEED},
    "output_dir": "experiments/results/nagent_${TIMESTAMP}/${experiment_type}/${game_type}/${models_str}/n${n_agents}_${game_params_str}"
}
EOF
    fi

    EXPERIMENT_ID=$((EXPERIMENT_ID + 1))
}

# =============================================================================
# Generate configs for all 3 game types with given models
# =============================================================================
generate_all_games() {
    local experiment_type="$1"
    local models_json="$2"
    local n_agents="$3"
    local run_num="$4"
    local design_desc="$5"

    # Game 1: Item Allocation
    for comp in "${IA_COMPETITION_LEVELS[@]}"; do
        _IA_COMP="$comp"
        generate_config "$experiment_type" "item_allocation" "$models_json" "$n_agents" "$run_num" "$design_desc"
    done

    # Game 2: Diplomacy
    for rho in "${DIPLO_RHO_VALUES[@]}"; do
        _DIPLO_RHO=$(clamp_rho "$rho" "$n_agents")
        generate_config "$experiment_type" "diplomacy" "$models_json" "$n_agents" "$run_num" "$design_desc"
    done

    # Game 3: Co-Funding
    for alpha in "${CF_ALPHA_VALUES[@]}"; do
        for sigma in "${CF_SIGMA_VALUES[@]}"; do
            _CF_ALPHA="$alpha"
            _CF_SIGMA="$sigma"
            generate_config "$experiment_type" "co_funding" "$models_json" "$n_agents" "$run_num" "$design_desc"
        done
    done
}

# =============================================================================
# Experiment A: Scaling Robustness
# 1 focal adversary + (N-1) copies of baseline
# =============================================================================
generate_exp_a() {
    echo "Generating Experiment A: Scaling Robustness..."
    local count_before=$EXPERIMENT_ID

    for focal in "${FOCAL_MODELS[@]}"; do
        for n in "${N_VALUES[@]}"; do
            # Build models list: focal + (N-1) copies of baseline
            local models=("$focal")
            for ((i=1; i<n; i++)); do
                models+=("$BASELINE_MODEL")
            done
            local models_json
            models_json=$(build_models_json "${models[@]}")

            for ((run=1; run<=NUM_RUNS; run++)); do
                generate_all_games "exp_a_scaling" "$models_json" "$n" "$run" \
                    "focal=${focal}_vs_${n_minus_1}x${BASELINE_MODEL}"
            done
        done
    done

    local count_after=$EXPERIMENT_ID
    echo "  Generated $((count_after - count_before)) Experiment A configs"
}

# =============================================================================
# Experiment B: Coalition Power
# 1 strong (claude-haiku-4-5) + K copies of weak (llama-3.1-8b)
# K = {1, 2, 3, 4} -> N = {2, 3, 4, 5}
# =============================================================================
generate_exp_b() {
    echo "Generating Experiment B: Coalition Power..."
    local count_before=$EXPERIMENT_ID

    if [[ "$MODE" == "derisk" ]]; then
        local K_VALUES=(2)  # Only K=2 (N=3) for derisk
    else
        local K_VALUES=(1 2 3 4)
    fi

    for k in "${K_VALUES[@]}"; do
        local n=$((k + 1))
        local models=("$STRONG_MODEL")
        for ((i=0; i<k; i++)); do
            models+=("$WEAK_MODEL")
        done
        local models_json
        models_json=$(build_models_json "${models[@]}")

        for ((run=1; run<=NUM_RUNS; run++)); do
            generate_all_games "exp_b_coalition" "$models_json" "$n" "$run" \
                "strong=${STRONG_MODEL}_vs_${k}x${WEAK_MODEL}"
        done
    done

    local count_after=$EXPERIMENT_ID
    echo "  Generated $((count_after - count_before)) Experiment B configs"
}

# =============================================================================
# Experiment C: Capability Gradient
# N=4 quartet + 3 N=3 trios
# =============================================================================
generate_exp_c() {
    echo "Generating Experiment C: Capability Gradient..."
    local count_before=$EXPERIMENT_ID

    # N=4 quartet: all 4 panel models
    local quartet=("claude-haiku-4-5" "qwen3-32b" "$BASELINE_MODEL" "llama-3.1-8b-instruct")
    local quartet_json
    quartet_json=$(build_models_json "${quartet[@]}")

    for ((run=1; run<=NUM_RUNS; run++)); do
        generate_all_games "exp_c_gradient" "$quartet_json" 4 "$run" \
            "quartet_full_gradient"
    done

    if [[ "$MODE" != "derisk" ]]; then
        # Trio 1: Wide gradient (claude-haiku / gpt-5-nano / llama-8b)
        local trio1=("claude-haiku-4-5" "$BASELINE_MODEL" "llama-3.1-8b-instruct")
        local trio1_json
        trio1_json=$(build_models_json "${trio1[@]}")
        for ((run=1; run<=NUM_RUNS; run++)); do
            generate_all_games "exp_c_gradient" "$trio1_json" 3 "$run" \
                "trio_wide_gradient"
        done

        # Trio 2: Top-heavy (claude-haiku / qwen3 / llama-8b)
        local trio2=("claude-haiku-4-5" "qwen3-32b" "llama-3.1-8b-instruct")
        local trio2_json
        trio2_json=$(build_models_json "${trio2[@]}")
        for ((run=1; run<=NUM_RUNS; run++)); do
            generate_all_games "exp_c_gradient" "$trio2_json" 3 "$run" \
                "trio_top_heavy"
        done

        # Trio 3: Weak-heavy (qwen3 / gpt-5-nano / llama-8b)
        local trio3=("qwen3-32b" "$BASELINE_MODEL" "llama-3.1-8b-instruct")
        local trio3_json
        trio3_json=$(build_models_json "${trio3[@]}")
        for ((run=1; run<=NUM_RUNS; run++)); do
            generate_all_games "exp_c_gradient" "$trio3_json" 3 "$run" \
                "trio_weak_heavy"
        done
    fi

    local count_after=$EXPERIMENT_ID
    echo "  Generated $((count_after - count_before)) Experiment C configs"
}

# =============================================================================
# Generate experiments based on mode
# =============================================================================
EXP_A_COUNT=0
EXP_B_COUNT=0
EXP_C_COUNT=0

if [[ "$MODE" == "derisk" ]]; then
    # Derisk: minimal config per game type (Exp A only, N=3, 1 focal, 1 competition)
    echo "Derisk mode: generating minimal configs..."
    generate_exp_a
    EXP_A_COUNT=$EXPERIMENT_ID
elif [[ "$MODE" == "exp_a" ]]; then
    generate_exp_a
    EXP_A_COUNT=$EXPERIMENT_ID
elif [[ "$MODE" == "exp_b" ]]; then
    generate_exp_b
    EXP_B_COUNT=$EXPERIMENT_ID
elif [[ "$MODE" == "exp_c" ]]; then
    generate_exp_c
    EXP_C_COUNT=$EXPERIMENT_ID
else
    # Full: all experiments
    before_a=$EXPERIMENT_ID
    generate_exp_a
    EXP_A_COUNT=$((EXPERIMENT_ID - before_a))

    before_b=$EXPERIMENT_ID
    generate_exp_b
    EXP_B_COUNT=$((EXPERIMENT_ID - before_b))

    before_c=$EXPERIMENT_ID
    generate_exp_c
    EXP_C_COUNT=$((EXPERIMENT_ID - before_c))
fi

TOTAL_COUNT=$EXPERIMENT_ID

# Create symlink to latest
SYMLINK="${BASE_DIR}/experiments/results/nagent_latest"
if [[ -L "${SYMLINK}" ]]; then
    rm "${SYMLINK}"
elif [[ -d "${SYMLINK}" ]] && [[ ! -L "${SYMLINK}" ]]; then
    OLD_DIR="${SYMLINK}_old_$(date +%Y%m%d_%H%M%S)"
    mv "${SYMLINK}" "${OLD_DIR}"
    echo "Moved existing directory to: ${OLD_DIR}"
fi
ln -sf "nagent_${TIMESTAMP}" "${SYMLINK}"
echo "Created symlink: ${SYMLINK} -> nagent_${TIMESTAMP}"

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
N-Agent Scaling Experiment Configuration Summary
======================================================
Mode: ${MODE}
Total experiments: ${TOTAL_COUNT}
  Experiment A (Scaling Robustness): ${EXP_A_COUNT}
  Experiment B (Coalition Power): ${EXP_B_COUNT}
  Experiment C (Capability Gradient): ${EXP_C_COUNT}

Model Panel:
  Baseline: ${BASELINE_MODEL} (Elo 1338)
  Focal models: ${FOCAL_MODELS[*]}
  Strong model (Exp B): ${STRONG_MODEL} (Elo 1403)
  Weak model (Exp B): ${WEAK_MODEL} (Elo ~1180)

Agent counts (N): ${N_VALUES[*]}

Game 1 (Item Allocation):
  Items: ${IA_NUM_ITEMS}
  Competition levels: ${IA_COMPETITION_LEVELS[*]}

Game 2 (Diplomacy):
  Issues: ${DIPLO_N_ISSUES}
  Rho values: ${DIPLO_RHO_VALUES[*]} (clamped for PSD at N>2)
  Theta: ${DIPLO_THETA}

Game 3 (Co-Funding):
  Projects: ${CF_M_PROJECTS}
  Alpha values: ${CF_ALPHA_VALUES[*]}
  Sigma values: ${CF_SIGMA_VALUES[*]}

Shared:
  Max rounds: ${MAX_ROUNDS}
  Discussion turns: ${DISCUSSION_TURNS}
  Max tokens per phase: ${MAX_TOKENS_PER_PHASE}
  Runs per config: ${NUM_RUNS}
EOF
echo "Created summary: ${SUMMARY_FILE}"

# Create CSV index
CSV_FILE="${CONFIG_DIR}/experiment_index.csv"
echo "experiment_id,experiment_type,experiment_design,game_type,models,n_agents,competition_param,run_number,seed,config_file" > "${CSV_FILE}"

for config_file in "${CONFIG_DIR}"/config_*.json; do
    if [[ -f "$config_file" ]]; then
        python3 - "$config_file" << 'PYEOF' >> "${CSV_FILE}"
import json
import sys

with open(sys.argv[1]) as f:
    cfg = json.load(f)

exp_id = cfg["experiment_id"]
exp_type = cfg["experiment_type"]
design = cfg.get("experiment_design", "")
game_type = cfg["game_type"]
models = "+".join(cfg["models"])
n_agents = cfg.get("n_agents", len(cfg["models"]))
run_num = cfg.get("run_number", 1)
seed = cfg.get("random_seed", 0)

# Game-specific competition parameter
if game_type == "item_allocation":
    comp = f"comp={cfg.get('competition_level', '')}"
elif game_type == "diplomacy":
    comp = f"rho={cfg.get('rho', '')}_theta={cfg.get('theta', '')}"
elif game_type == "co_funding":
    comp = f"alpha={cfg.get('alpha', '')}_sigma={cfg.get('sigma', '')}"
else:
    comp = ""

fname = sys.argv[1].split("/")[-1]
print(f"{exp_id},{exp_type},{design},{game_type},{models},{n_agents},{comp},{run_num},{seed},{fname}")
PYEOF
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
SLURM_TIME="12:00:00"

# Worker script (shared by all sbatch wrappers)
WORKER_SCRIPT="${SLURM_DIR}/run_nagent_worker.sh"
cat > "${WORKER_SCRIPT}" << SLURM_WORKER_EOF
#!/bin/bash
set -e

BASE_DIR="/scratch/gpfs/DANQIC/jz4391/bargain"
cd "\${BASE_DIR}"
mkdir -p logs/cluster

echo "============================================================"
echo "N-Agent Scaling Experiment"
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
export OPENROUTER_TRANSPORT="\${OPENROUTER_TRANSPORT:-direct}"
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

# Extract config values using Python
GAME_TYPE=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['game_type'])")
MODELS=\$(python3 -c "import json; print(' '.join(json.load(open('\${CONFIG_FILE}'))['models']))")
N_AGENTS=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}')).get('n_agents', len(json.load(open('\${CONFIG_FILE}'))['models'])))")
SEED=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['random_seed'])")
MAX_ROUNDS=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['max_rounds'])")
DISCUSSION_TURNS=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['discussion_turns'])")
OUTPUT_DIR=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['output_dir'])")
MAX_TOKENS=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['max_tokens_per_phase'])")
NUM_RUNS=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['num_runs'])")
RUN_NUMBER=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['run_number'])")

echo "Game type: \$GAME_TYPE"
echo "Models (\$N_AGENTS agents): \$MODELS"
echo "Max rounds: \$MAX_ROUNDS"
echo "Run: \$RUN_NUMBER / \$NUM_RUNS"
echo "Seed: \$SEED"
echo "Output: \$OUTPUT_DIR"

# Build the command
CMD="python3 run_strong_models_experiment.py"
CMD="\$CMD --game-type \$GAME_TYPE"
CMD="\$CMD --models \$MODELS"
CMD="\$CMD --batch --num-runs \$NUM_RUNS --run-number \$RUN_NUMBER"
CMD="\$CMD --max-rounds \$MAX_ROUNDS"
CMD="\$CMD --random-seed \$SEED"
CMD="\$CMD --discussion-turns \$DISCUSSION_TURNS"
CMD="\$CMD --max-tokens-per-phase \$MAX_TOKENS"
CMD="\$CMD --output-dir \$OUTPUT_DIR"
CMD="\$CMD --job-id \$SLURM_ARRAY_TASK_ID"

# Game-specific arguments
if [[ "\$GAME_TYPE" == "item_allocation" ]]; then
    NUM_ITEMS=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['num_items'])")
    COMP_LEVEL=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['competition_level'])")
    GAMMA=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['gamma_discount'])")
    CMD="\$CMD --num-items \$NUM_ITEMS --competition-level \$COMP_LEVEL --gamma-discount \$GAMMA"
    echo "Items: \$NUM_ITEMS | Competition: \$COMP_LEVEL | Gamma: \$GAMMA"

elif [[ "\$GAME_TYPE" == "diplomacy" ]]; then
    N_ISSUES=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['n_issues'])")
    RHO=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['rho'])")
    THETA=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['theta'])")
    CMD="\$CMD --n-issues \$N_ISSUES --rho \$RHO --theta \$THETA"
    echo "Issues: \$N_ISSUES | Rho: \$RHO | Theta: \$THETA"

elif [[ "\$GAME_TYPE" == "co_funding" ]]; then
    M_PROJECTS=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['m_projects'])")
    ALPHA=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['alpha'])")
    SIGMA=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['sigma'])")
    C_MIN=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['c_min'])")
    C_MAX=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}'))['c_max'])")
    DISC_TRANSP=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}')).get('cofunding_discussion_transparency', 'own'))")
    ENABLE_CV=\$(python3 -c "import json; print(str(json.load(open('\${CONFIG_FILE}')).get('cofunding_enable_commit_vote', True)).lower())")
    ENABLE_TD=\$(python3 -c "import json; print(str(json.load(open('\${CONFIG_FILE}')).get('cofunding_enable_time_discount', True)).lower())")
    TIME_DISC=\$(python3 -c "import json; print(json.load(open('\${CONFIG_FILE}')).get('cofunding_time_discount', 0.9))")

    CMD="\$CMD --m-projects \$M_PROJECTS --alpha \$ALPHA --sigma \$SIGMA"
    CMD="\$CMD --c-min \$C_MIN --c-max \$C_MAX"
    CMD="\$CMD --cofunding-discussion-transparency \$DISC_TRANSP"
    CMD="\$CMD --cofunding-time-discount \$TIME_DISC"
    if [[ "\$ENABLE_CV" != "true" ]]; then
        CMD="\$CMD --cofunding-disable-commit-vote"
    fi
    if [[ "\$ENABLE_TD" != "true" ]]; then
        CMD="\$CMD --cofunding-disable-time-discount"
    fi
    echo "Projects: \$M_PROJECTS | Alpha: \$ALPHA | Sigma: \$SIGMA"
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
    echo "Experiment FAILED at: \$(date)"
    echo "============================================================"
    rm -f "\$TMP_RUN_LOG"
    exit 1
fi

rm -f "\$TMP_RUN_LOG"
SLURM_WORKER_EOF
chmod +x "${WORKER_SCRIPT}"
echo "Created worker script: ${WORKER_SCRIPT}"

# =============================================================================
# sbatch wrappers
# =============================================================================
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

API_SLURM_SCRIPT="${SLURM_DIR}/run_nagent_api_experiments.sbatch"
GPU2_SLURM_SCRIPT="${SLURM_DIR}/run_nagent_gpu_2x80gb.sbatch"
GPU4_SLURM_SCRIPT="${SLURM_DIR}/run_nagent_gpu_4x80gb.sbatch"
GPU6_SLURM_SCRIPT="${SLURM_DIR}/run_nagent_mixed_gpu.sbatch"

GPU2_GRES_LINES=$'#SBATCH --constraint=gpu80\n#SBATCH --gres=gpu:a100:2'
GPU4_GRES_LINES=$'#SBATCH --constraint=gpu80\n#SBATCH --gres=gpu:a100:4'

create_sbatch_wrapper \
    "${API_SLURM_SCRIPT}" "nagent-api" "4" "8G" "" "" \
    "logs/cluster/nagent_api_%A_%a.out" "logs/cluster/nagent_api_%A_%a.err"

create_sbatch_wrapper \
    "${GPU2_SLURM_SCRIPT}" "nagent-gpu2" "8" "64G" "" "${GPU2_GRES_LINES}" \
    "logs/cluster/nagent_gpu2_%A_%a.out" "logs/cluster/nagent_gpu2_%A_%a.err"

create_sbatch_wrapper \
    "${GPU4_SLURM_SCRIPT}" "nagent-gpu4" "16" "128G" "" "${GPU4_GRES_LINES}" \
    "logs/cluster/nagent_gpu4_%A_%a.out" "logs/cluster/nagent_gpu4_%A_%a.err"

# Mixed GPU: for configs with both qwen3-32b (4 GPU) + llama-8b (2 GPU)
# Request 4 GPUs as max needed by any single model in the config
create_sbatch_wrapper \
    "${GPU6_SLURM_SCRIPT}" "nagent-mix" "16" "128G" "" "${GPU4_GRES_LINES}" \
    "logs/cluster/nagent_mix_%A_%a.out" "logs/cluster/nagent_mix_%A_%a.err"

echo "Created SLURM scripts: API, GPU(2x80GB), GPU(4x80GB), Mixed"

# =============================================================================
# Submission script with smart routing
# =============================================================================
SUBMIT_SCRIPT="${SLURM_DIR}/submit_all.sh"
cat > "${SUBMIT_SCRIPT}" << 'SUBMIT_EOF'
#!/bin/bash
# Submit N-agent experiment jobs with API/GPU routing
# Usage: ./submit_all.sh [api|gpu|all] [--test] [--max-concurrent <num>]

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

API_SCRIPT="${SCRIPT_DIR}/run_nagent_api_experiments.sbatch"
GPU2_SCRIPT="${SCRIPT_DIR}/run_nagent_gpu_2x80gb.sbatch"
GPU4_SCRIPT="${SCRIPT_DIR}/run_nagent_gpu_4x80gb.sbatch"
MIX_SCRIPT="${SCRIPT_DIR}/run_nagent_mixed_gpu.sbatch"

API_IDS=()
GPU2_IDS=()
GPU4_IDS=()
MIX_IDS=()

for config_file in "${CONFIG_FILES[@]}"; do
    config_id="$(basename "$config_file")"
    config_id="${config_id#config_}"
    config_id="${config_id%.json}"
    config_id=$((10#$config_id))

    if [[ "$TEST_MODE" == "true" && "$config_id" -ne 0 ]]; then
        continue
    fi

    # Determine GPU requirements
    route=$(python3 - "$config_file" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = json.load(f)

models = cfg.get("models", [])
gpu_map = {
    "qwen3-32b": 4,
    "llama-3.1-8b-instruct": 2,
}

gpu_models = [m for m in models if m in gpu_map]
if not gpu_models:
    print("api")
elif len(gpu_models) == 1:
    gpus = gpu_map[gpu_models[0]]
    if gpus <= 2:
        print("gpu2")
    else:
        print("gpu4")
else:
    # Multiple GPU models in same config -> mixed
    print("mixed")
PY
)

    case "$route" in
        api)    API_IDS+=("$config_id") ;;
        gpu2)   GPU2_IDS+=("$config_id") ;;
        gpu4)   GPU4_IDS+=("$config_id") ;;
        mixed)  MIX_IDS+=("$config_id") ;;
    esac
done

echo "Total configurations: ${TOTAL_CONFIGS}"
echo "  API configs : ${#API_IDS[@]}"
echo "  GPU(2x80GB): ${#GPU2_IDS[@]}"
echo "  GPU(4x80GB): ${#GPU4_IDS[@]}"
echo "  Mixed GPU  : ${#MIX_IDS[@]}"

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
        submit_group "Mixed GPU" "$MIX_SCRIPT" "${MIX_IDS[@]}"
        ;;
    all)
        submit_group "API" "$API_SCRIPT" "${API_IDS[@]}"
        submit_group "GPU 2x80GB" "$GPU2_SCRIPT" "${GPU2_IDS[@]}"
        submit_group "GPU 4x80GB" "$GPU4_SCRIPT" "${GPU4_IDS[@]}"
        submit_group "Mixed GPU" "$MIX_SCRIPT" "${MIX_IDS[@]}"
        ;;
    *)
        echo "Usage: $0 [api|gpu|all] [--test] [--max-concurrent <num>]"
        exit 1
        ;;
esac

echo ""
echo "Jobs submitted. Monitor with: squeue -u $USER"
SUBMIT_EOF
chmod +x "${SUBMIT_SCRIPT}"
echo "Created submission script: ${SUBMIT_SCRIPT}"

# =============================================================================
# Local run script
# =============================================================================
LOCAL_RUN_SCRIPT="${SLURM_DIR}/run_local.sh"
cat > "${LOCAL_RUN_SCRIPT}" << 'LOCAL_EOF'
#!/bin/bash
# Run a single N-agent experiment locally (for testing)
# Usage: ./run_local.sh [config_id]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${BASE_DIR}"

CONFIG_ID="${1:-0}"
CONFIG_DIR="${SCRIPT_DIR}/.."

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
cat "$CONFIG_FILE" | python3 -m json.tool
echo ""

source "${BASE_DIR}/.venv/bin/activate"
export OPENROUTER_TRANSPORT="${OPENROUTER_TRANSPORT:-direct}"

# Extract and run the command directly
GAME_TYPE=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['game_type'])")
MODELS=$(python3 -c "import json; print(' '.join(json.load(open('${CONFIG_FILE}'))['models']))")
SEED=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['random_seed'])")
MAX_ROUNDS=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['max_rounds'])")
DISCUSSION_TURNS=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['discussion_turns'])")
OUTPUT_DIR=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['output_dir'])")
MAX_TOKENS=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['max_tokens_per_phase'])")
NUM_RUNS=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['num_runs'])")
RUN_NUMBER=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['run_number'])")

CMD="python3 run_strong_models_experiment.py"
CMD="$CMD --game-type $GAME_TYPE"
CMD="$CMD --models $MODELS"
CMD="$CMD --batch --num-runs $NUM_RUNS --run-number $RUN_NUMBER"
CMD="$CMD --max-rounds $MAX_ROUNDS"
CMD="$CMD --random-seed $SEED"
CMD="$CMD --discussion-turns $DISCUSSION_TURNS"
CMD="$CMD --max-tokens-per-phase $MAX_TOKENS"
CMD="$CMD --output-dir $OUTPUT_DIR"
CMD="$CMD --job-id $CONFIG_ID"

if [[ "$GAME_TYPE" == "item_allocation" ]]; then
    NUM_ITEMS=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['num_items'])")
    COMP_LEVEL=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['competition_level'])")
    GAMMA=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['gamma_discount'])")
    CMD="$CMD --num-items $NUM_ITEMS --competition-level $COMP_LEVEL --gamma-discount $GAMMA"

elif [[ "$GAME_TYPE" == "diplomacy" ]]; then
    N_ISSUES=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['n_issues'])")
    RHO=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['rho'])")
    THETA=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['theta'])")
    CMD="$CMD --n-issues $N_ISSUES --rho $RHO --theta $THETA"

elif [[ "$GAME_TYPE" == "co_funding" ]]; then
    M_PROJECTS=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['m_projects'])")
    ALPHA=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['alpha'])")
    SIGMA=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['sigma'])")
    C_MIN=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['c_min'])")
    C_MAX=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['c_max'])")
    DISC_TRANSP=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}')).get('cofunding_discussion_transparency', 'own'))")
    ENABLE_CV=$(python3 -c "import json; print(str(json.load(open('${CONFIG_FILE}')).get('cofunding_enable_commit_vote', True)).lower())")
    ENABLE_TD=$(python3 -c "import json; print(str(json.load(open('${CONFIG_FILE}')).get('cofunding_enable_time_discount', True)).lower())")
    TIME_DISC=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}')).get('cofunding_time_discount', 0.9))")
    CMD="$CMD --m-projects $M_PROJECTS --alpha $ALPHA --sigma $SIGMA"
    CMD="$CMD --c-min $C_MIN --c-max $C_MAX"
    CMD="$CMD --cofunding-discussion-transparency $DISC_TRANSP"
    CMD="$CMD --cofunding-time-discount $TIME_DISC"
    if [[ "$ENABLE_CV" != "true" ]]; then
        CMD="$CMD --cofunding-disable-commit-vote"
    fi
    if [[ "$ENABLE_TD" != "true" ]]; then
        CMD="$CMD --cofunding-disable-time-discount"
    fi
fi

echo "Running: $CMD"
echo ""
eval "$CMD"
LOCAL_EOF
chmod +x "${LOCAL_RUN_SCRIPT}"
echo "Created local run script: ${LOCAL_RUN_SCRIPT}"

echo ""
echo "============================================================"
echo "Configuration generation complete!"
echo "============================================================"
echo ""
echo "Generated: ${TOTAL_COUNT} experiment configurations"
echo "  Experiment A (Scaling Robustness): ${EXP_A_COUNT}"
echo "  Experiment B (Coalition Power):    ${EXP_B_COUNT}"
echo "  Experiment C (Capability Gradient): ${EXP_C_COUNT}"
echo "Location: ${CONFIG_DIR}"
echo ""
echo "Quick start:"
echo ""
echo "  1. Derisk (run 1 config locally):"
echo "     ${LOCAL_RUN_SCRIPT} 0"
echo ""
echo "  2. Submit to SLURM:"
echo "     ${SUBMIT_SCRIPT} all"
echo "     ${SUBMIT_SCRIPT} api                  # API-only jobs"
echo "     ${SUBMIT_SCRIPT} gpu                  # GPU jobs"
echo "     ${SUBMIT_SCRIPT} all --test           # Only config 0"
echo "     ${SUBMIT_SCRIPT} all --max-concurrent 10"
echo ""

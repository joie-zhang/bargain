#!/bin/bash
# =============================================================================
# Run Grok-4 Experiments in tmux (using polling)
# =============================================================================
#
# Runs all 22 Grok-4 experiments sequentially in the current shell.
# Designed to be run inside a tmux session so it persists.
#
# Usage:
#   tmux new -s grok4
#   ./scripts/run_grok4_tmux.sh
#
# Or to run a specific range:
#   ./scripts/run_grok4_tmux.sh 198 205   # Run configs 198-205
#
# What it creates:
#   experiments/results/scaling_experiment_20260121_070359/gpt-5-nano_vs_grok-4/
#   └── {weak_first,strong_first}/comp_{level}/run_{n}/
#
# =============================================================================

set -e

BASE_DIR="/scratch/gpfs/DANQIC/jz4391/bargain"
cd "${BASE_DIR}"

# Config directory (use timestamped path)
CONFIG_DIR="${BASE_DIR}/experiments/results/scaling_experiment_20260121_070359/configs"

# Grok-4 config IDs
ALL_GROK4_IDS=(198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219)

# Parse optional range arguments
START_ID="${1:-}"
END_ID="${2:-}"

if [[ -n "$START_ID" && -n "$END_ID" ]]; then
    CONFIG_IDS=()
    for id in "${ALL_GROK4_IDS[@]}"; do
        if [[ $id -ge $START_ID && $id -le $END_ID ]]; then
            CONFIG_IDS+=($id)
        fi
    done
    echo "Running subset: configs ${START_ID}-${END_ID} (${#CONFIG_IDS[@]} jobs)"
else
    CONFIG_IDS=("${ALL_GROK4_IDS[@]}")
    echo "Running all ${#CONFIG_IDS[@]} Grok-4 configs"
fi

# Activate virtual environment
source "${BASE_DIR}/.venv/bin/activate"
echo "Python: $(which python3)"
echo ""

# Track progress
TOTAL=${#CONFIG_IDS[@]}
COMPLETED=0
FAILED=0

for CONFIG_ID in "${CONFIG_IDS[@]}"; do
    CONFIG_FILE="${CONFIG_DIR}/config_${CONFIG_ID}.json"

    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "ERROR: Config not found: $CONFIG_FILE"
        FAILED=$((FAILED + 1))
        continue
    fi

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

    echo "============================================================"
    echo "[${COMPLETED}/${TOTAL}] Config ${CONFIG_ID}: ${WEAK_MODEL} vs ${STRONG_MODEL}"
    echo "  Order: ${MODEL_ORDER} | Comp: ${COMP_LEVEL} | Run: ${RUN_NUM} | Seed: ${SEED}"
    echo "  Started: $(date)"
    echo "============================================================"

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
        --job-id $CONFIG_ID; then
        echo "✅ Config ${CONFIG_ID} completed at $(date)"
        COMPLETED=$((COMPLETED + 1))
    else
        echo "❌ Config ${CONFIG_ID} failed at $(date)"
        FAILED=$((FAILED + 1))
    fi

    echo ""
done

echo "============================================================"
echo "SUMMARY"
echo "============================================================"
echo "Total: ${TOTAL}"
echo "Completed: ${COMPLETED}"
echo "Failed: ${FAILED}"
echo "Finished at: $(date)"

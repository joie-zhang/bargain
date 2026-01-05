#!/bin/bash
# Run batch experiments for Qwen2.5 models vs Claude-3.7-Sonnet
# Competition Level = 1 (Full Competition)
# Each model: 5 runs with competition_level=1

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${BASE_DIR}"

# Models to test
QWEN_MODELS=(
    "Qwen2.5-3B-Instruct"
    "Qwen2.5-7B-Instruct"
    "Qwen2.5-14B-Instruct"
)

ADVERSARY_MODEL="claude-3-7-sonnet"
COMPETITION_LEVEL=1
NUM_RUNS=5
NUM_ITEMS=5
MAX_ROUNDS=10

echo "============================================================"
echo "QWEN2.5 EXPERIMENTS - BATCH RUN (COMPETITION LEVEL = 1)"
echo "============================================================"
echo "Adversary Model: ${ADVERSARY_MODEL}"
echo "Competition Level: ${COMPETITION_LEVEL} (Full Competition)"
echo "Runs per model: ${NUM_RUNS}"
echo "Items: ${NUM_ITEMS}"
echo "Max Rounds: ${MAX_ROUNDS}"
echo "Models to test: ${QWEN_MODELS[@]}"
echo "============================================================"
echo ""

# Activate virtual environment if exists (check multiple locations)
if [ -f "${BASE_DIR}/.venv/bin/activate" ]; then
    source "${BASE_DIR}/.venv/bin/activate"
    echo "Activated virtual environment: ${BASE_DIR}/.venv"
fi

# Run experiments for each Qwen model
for qwen_model in "${QWEN_MODELS[@]}"; do
    echo ""
    echo "============================================================"
    echo "Running experiments for: ${qwen_model}"
    echo "============================================================"
    
    # Generate unique output directory with number suffix to avoid overwriting
    BASE_OUTPUT_DIR="experiments/results/${qwen_model}_vs_${ADVERSARY_MODEL}_config_unknown_runs${NUM_RUNS}_comp${COMPETITION_LEVEL}"
    
    # Check if base directory exists and find next available number
    RUN_NUM=1
    if [ -d "${BASE_OUTPUT_DIR}" ]; then
        # Base directory exists, start from _1
        OUTPUT_DIR="${BASE_OUTPUT_DIR}_${RUN_NUM}"
        # Find the highest existing number
        while [ -d "${OUTPUT_DIR}" ]; do
            RUN_NUM=$((RUN_NUM + 1))
            OUTPUT_DIR="${BASE_OUTPUT_DIR}_${RUN_NUM}"
        done
        echo "Existing results found. Using numbered directory: ${OUTPUT_DIR}"
    elif ls -d "${BASE_OUTPUT_DIR}_"* 2>/dev/null | grep -q .; then
        # Some numbered directories exist, find the highest
        MAX_NUM=0
        for existing_dir in "${BASE_OUTPUT_DIR}_"*; do
            if [ -d "${existing_dir}" ]; then
                suffix=$(basename "${existing_dir}" | sed "s|${BASE_OUTPUT_DIR}_||")
                if [[ "${suffix}" =~ ^[0-9]+$ ]]; then
                    num=${suffix}
                    if [ "${num}" -gt "${MAX_NUM}" ]; then
                        MAX_NUM=${num}
                    fi
                fi
            fi
        done
        RUN_NUM=$((MAX_NUM + 1))
        OUTPUT_DIR="${BASE_OUTPUT_DIR}_${RUN_NUM}"
        echo "Existing numbered results found. Using directory: ${OUTPUT_DIR}"
    else
        # No existing directories, use base name
        OUTPUT_DIR="${BASE_OUTPUT_DIR}"
        echo "No existing results found. Using directory: ${OUTPUT_DIR}"
    fi
    
    echo ""
    echo "Command: python3 run_strong_models_experiment.py \\"
    echo "    --models ${qwen_model} ${ADVERSARY_MODEL} \\"
    echo "    --competition-level ${COMPETITION_LEVEL} \\"
    echo "    --num-items ${NUM_ITEMS} \\"
    echo "    --max-rounds ${MAX_ROUNDS} \\"
    echo "    --batch \\"
    echo "    --num-runs ${NUM_RUNS} \\"
    echo "    --output-dir ${OUTPUT_DIR}"
    echo ""
    
    if python3 run_strong_models_experiment.py \
        --models "${qwen_model}" "${ADVERSARY_MODEL}" \
        --competition-level "${COMPETITION_LEVEL}" \
        --num-items "${NUM_ITEMS}" \
        --max-rounds "${MAX_ROUNDS}" \
        --batch \
        --num-runs "${NUM_RUNS}" \
        --output-dir "${OUTPUT_DIR}"; then
        echo ""
        echo "✅ Successfully completed ${NUM_RUNS} runs for ${qwen_model}"
    else
        echo ""
        echo "❌ Failed to complete experiments for ${qwen_model}"
        echo "Continuing with next model..."
    fi
    
    echo ""
    echo "Waiting 5 seconds before next model..."
    sleep 5
done

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETED"
echo "============================================================"
echo ""
echo "Results are saved in: experiments/results/"
echo ""
echo "To check results, look for directories matching:"
echo "  *Qwen2.5-*B-Instruct_vs_${ADVERSARY_MODEL}*comp${COMPETITION_LEVEL}*"
echo ""


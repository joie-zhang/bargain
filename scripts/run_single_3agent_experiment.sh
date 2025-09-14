#!/bin/bash
# Simple single 3-agent experiment runner - NO timeouts, just run it
set -e

# Get job ID
JOB_ID=${1:-${JOB_ARRAY_ID:-0}}

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="${BASE_DIR}/experiments/results/3agent_experiment/configs"
LOGS_DIR="${BASE_DIR}/experiments/results/3agent_experiment/logs"

mkdir -p "${LOGS_DIR}"

CONFIG_FILE="${CONFIG_DIR}/config_${JOB_ID}.json"
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "❌ Configuration file not found: ${CONFIG_FILE}"
    exit 1
fi

# Extract parameters - handle 3 models
MODELS=$(python3 -c "import json; data=json.load(open('${CONFIG_FILE}')); print(' '.join(data['models']))")
COMP_LEVEL=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['competition_level'])")
NUM_ITEMS=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['m_items'])")
MAX_ROUNDS=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['t_rounds'])")
RANDOM_SEED=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['random_seed'])")
OUTPUT_DIR=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['output_dir'])")
EXPERIMENT_TYPE=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['experiment_type'])")

# Create output directory
FULL_OUTPUT_DIR="${BASE_DIR}/${OUTPUT_DIR}"
mkdir -p "${FULL_OUTPUT_DIR}"

LOG_FILE="${LOGS_DIR}/experiment_${JOB_ID}.log"

echo "[$(date)] Starting 3-agent experiment ${JOB_ID}: ${MODELS} (competition=${COMP_LEVEL}, type=${EXPERIMENT_TYPE})" | tee "${LOG_FILE}"

cd "${BASE_DIR}"

# Activate virtual environment if exists
if [ -f ~/.venv/bin/activate ]; then
    source ~/.venv/bin/activate
fi

# Build command for 3-agent experiment
# Note: --num-runs 1 means run 1 negotiation game in this Python process
CMD="python3 run_strong_models_experiment.py \
    --models ${MODELS} \
    --competition-level ${COMP_LEVEL} \
    --num-items ${NUM_ITEMS} \
    --max-rounds ${MAX_ROUNDS} \
    --random-seed ${RANDOM_SEED} \
    --batch \
    --num-runs 1 \
    --job-id ${JOB_ID}"

echo "[$(date)] Running: ${CMD}" >> "${LOG_FILE}"

# JUST RUN IT - NO TIMEOUT
if ${CMD} >> "${LOG_FILE}" 2>&1; then
    STATUS="SUCCESS"
    EXIT_CODE=0
else
    EXIT_CODE=$?
    STATUS="FAILED"
fi

echo "[$(date)] 3-agent experiment ${JOB_ID} completed with status: ${STATUS} (exit code: ${EXIT_CODE})" | tee -a "${LOG_FILE}"

# Save result
RESULT_FILE="${FULL_OUTPUT_DIR}/result_${JOB_ID}.json"
cat > "${RESULT_FILE}" << EOF
{
    "job_id": ${JOB_ID},
    "models": $(python3 -c "import json; print(json.dumps(json.load(open('${CONFIG_FILE}'))['models']))"),
    "experiment_type": "${EXPERIMENT_TYPE}",
    "competition_level": ${COMP_LEVEL},
    "status": "${STATUS}",
    "exit_code": ${EXIT_CODE},
    "timestamp": "$(date -Iseconds)"
}
EOF

# Mark complete
touch "${LOGS_DIR}/completed_${JOB_ID}.flag"

exit ${EXIT_CODE}
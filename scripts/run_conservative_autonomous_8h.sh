#!/bin/bash
# Autonomous conservative experiment runner:
# - Generates conservative cofunding + diplomacy configs
# - Runs smoke tests
# - Submits cofunding first, then diplomacy
# - Iteratively monitors queue/logs and regenerates plots for 8 hours
#
# Usage:
#   ./scripts/run_conservative_autonomous_8h.sh [max_concurrent] [duration_hours] [interval_seconds]
#
# Defaults:
#   max_concurrent=10, duration_hours=8, interval_seconds=900

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAX_CONCURRENT="${1:-10}"
DURATION_HOURS="${2:-8}"
INTERVAL_SECONDS="${3:-900}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${BASE_DIR}/logs/autonomous"
mkdir -p "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/conservative_autonomous_${RUN_TS}.log"

exec > >(tee -a "${RUN_LOG}") 2>&1

echo "================================================================"
echo "Autonomous Conservative Runner"
echo "Started: $(date)"
echo "Base dir: ${BASE_DIR}"
echo "Max concurrent: ${MAX_CONCURRENT}"
echo "Duration (hours): ${DURATION_HOURS}"
echo "Monitor interval (sec): ${INTERVAL_SECONDS}"
echo "Skip smoke tests: ${SKIP_SMOKE}"
echo "Log file: ${RUN_LOG}"
echo "================================================================"

cd "${BASE_DIR}"

if [[ -f ".venv/bin/activate" ]]; then
    source ".venv/bin/activate"
fi

python3 - <<'PY'
import os
for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
    print(f"{key} set: {bool(os.getenv(key))}")
PY

echo ""
echo "Step 1/6: Generate conservative co-funding configs"
./scripts/generate_cofunding_configs.sh --conservative
COFUND_CONFIG_DIR="$(readlink -f "${BASE_DIR}/experiments/results/cofunding_latest/configs")"
echo "Co-funding config dir: ${COFUND_CONFIG_DIR}"

echo ""
echo "Step 2/6: Generate conservative diplomacy configs"
./scripts/generate_diplomacy_configs.sh --conservative
DIPLO_CONFIG_DIR="$(readlink -f "${BASE_DIR}/experiments/results/diplomacy_latest/configs")"
echo "Diplomacy config dir: ${DIPLO_CONFIG_DIR}"

echo ""
echo "Step 3/6: Smoke tests (config 0 for each game)"
if [[ "${SKIP_SMOKE}" == "1" ]]; then
    echo "Skipping smoke tests (SKIP_SMOKE=1)"
else
    set +e
    timeout --preserve-status 45m "${COFUND_CONFIG_DIR}/slurm/run_local.sh" 0
    COFUND_SMOKE_STATUS=$?
    timeout --preserve-status 45m "${DIPLO_CONFIG_DIR}/slurm/run_local.sh" 0
    DIPLO_SMOKE_STATUS=$?
    set -e

    if [[ "${COFUND_SMOKE_STATUS}" -ne 0 ]]; then
        echo "WARNING: co-funding smoke test exited with status ${COFUND_SMOKE_STATUS}. Continuing."
    fi
    if [[ "${DIPLO_SMOKE_STATUS}" -ne 0 ]]; then
        echo "WARNING: diplomacy smoke test exited with status ${DIPLO_SMOKE_STATUS}. Continuing."
    fi
fi

echo ""
echo "Step 4/6: Submit jobs (cofunding first)"
./scripts/submit_cofunding_then_diplomacy.sh --max-concurrent "${MAX_CONCURRENT}"

echo ""
echo "Step 5/6: Iterative monitoring + plotting loop"
END_EPOCH=$(( $(date +%s) + DURATION_HOURS * 3600 ))
ITER=0

while [[ "$(date +%s)" -lt "${END_EPOCH}" ]]; do
    ITER=$((ITER + 1))
    echo ""
    echo "----------------------------------------------------------------"
    echo "Iteration ${ITER} at $(date)"
    echo "----------------------------------------------------------------"

    echo "Queue snapshot:"
    squeue -u "${USER}" || true

    echo ""
    echo "Recent quota/credit errors (last 20 matching lines):"
    grep -Eih "quota|insufficient|credit|billing|payment" logs/cluster/*.err 2>/dev/null | tail -n 20 || true

    echo ""
    echo "Refreshing cofunding aggregate artifacts..."
    ./scripts/collect_cofunding_results.sh cofunding_latest || true
    python3 visualization/visualize_cofunding.py --results-dir experiments/results/cofunding_latest || true

    echo ""
    echo "Refreshing diplomacy figures..."
    python3 visualization/visualize_diplomacy.py --dir experiments/results/diplomacy_latest || true

    echo ""
    echo "Sleeping for ${INTERVAL_SECONDS}s..."
    sleep "${INTERVAL_SECONDS}"
done

echo ""
echo "Step 6/6: Final refresh"
./scripts/collect_cofunding_results.sh cofunding_latest || true
python3 visualization/visualize_cofunding.py --results-dir experiments/results/cofunding_latest || true
python3 visualization/visualize_diplomacy.py --dir experiments/results/diplomacy_latest || true

echo ""
echo "Completed at $(date)"
echo "Autonomous run log: ${RUN_LOG}"

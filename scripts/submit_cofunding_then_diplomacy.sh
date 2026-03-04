#!/bin/bash
# Submit latest co-funding jobs first, then diplomacy jobs.
# This enforces queue order so long-running co-funding jobs get in line first.
#
# Usage:
#   ./scripts/submit_cofunding_then_diplomacy.sh [--test] [--max-concurrent N] [--delay-seconds N]

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

COFUND_SUBMIT="${BASE_DIR}/experiments/results/cofunding_latest/configs/slurm/submit_all.sh"
DIPLO_SUBMIT="${BASE_DIR}/experiments/results/diplomacy_latest/configs/slurm/submit_all.sh"

TEST_MODE=false
MAX_CONCURRENT=""
DELAY_SECONDS=5

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
        --delay-seconds)
            DELAY_SECONDS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--test] [--max-concurrent N] [--delay-seconds N]"
            exit 1
            ;;
    esac
done

if [[ ! -x "${COFUND_SUBMIT}" ]]; then
    echo "ERROR: Co-funding submit script not found or not executable:"
    echo "  ${COFUND_SUBMIT}"
    exit 1
fi

if [[ ! -x "${DIPLO_SUBMIT}" ]]; then
    echo "ERROR: Diplomacy submit script not found or not executable:"
    echo "  ${DIPLO_SUBMIT}"
    exit 1
fi

FORWARD_ARGS=()
if [[ "${TEST_MODE}" == "true" ]]; then
    FORWARD_ARGS+=(--test)
fi
if [[ -n "${MAX_CONCURRENT}" ]]; then
    FORWARD_ARGS+=(--max-concurrent "${MAX_CONCURRENT}")
fi

echo "Submitting co-funding first..."
"${COFUND_SUBMIT}" "${FORWARD_ARGS[@]}"

if [[ "${DELAY_SECONDS}" -gt 0 ]]; then
    echo "Waiting ${DELAY_SECONDS}s before diplomacy submission..."
    sleep "${DELAY_SECONDS}"
fi

echo "Submitting diplomacy second..."
"${DIPLO_SUBMIT}" "${FORWARD_ARGS[@]}"

echo ""
echo "Done. Co-funding was submitted first to prioritize queue placement."

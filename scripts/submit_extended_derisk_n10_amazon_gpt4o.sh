#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/scratch/gpfs/DANQIC/jz4391/bargain"
cd "$REPO_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/experiments/results/full_games123_multiagent_production_$STAMP}"
SELECTION_NAME="${SELECTION_NAME:-extended_derisk_n10_amazon_gpt4o}"
MAX_CONCURRENT="${MAX_CONCURRENT:-50}"
SLURM_TIME="${SLURM_TIME:-08:00:00}"

echo "RUN_DIR=$RUN_DIR"
echo "SELECTION_NAME=$SELECTION_NAME"
echo "MAX_CONCURRENT=$MAX_CONCURRENT"
echo "SLURM_TIME=$SLURM_TIME"

python scripts/full_games123_multiagent_batch.py generate \
  --results-root "$RUN_DIR" \
  --max-rounds 10 \
  --discussion-turns 2 \
  --max-concurrent "$MAX_CONCURRENT" \
  --slurm-time "$SLURM_TIME"

python scripts/full_games123_multiagent_batch.py select \
  --results-root "$RUN_DIR" \
  --selection-name "$SELECTION_NAME" \
  --preset extended_derisk_n10_amazon_gpt4o

python scripts/full_games123_multiagent_batch.py submit-selection \
  --results-root "$RUN_DIR" \
  --selection-name "$SELECTION_NAME"

echo
echo "Submitted selection '$SELECTION_NAME'."
echo "Monitor with:"
echo "  python scripts/full_games123_multiagent_batch.py summary --results-root \"$RUN_DIR\""

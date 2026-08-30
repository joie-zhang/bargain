#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/scratch/gpfs/DANQIC/jz4391/bargain"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/experiments/results/game1_gpt54_binding_team_v3_20260816_093310}"
PORT="${PORT:-8002}"

cd "$PROJECT_ROOT"
exec .venv/bin/python ui/random_monoculture_sample_viewer.py \
  --results-root "$RESULTS_ROOT" \
  --host 127.0.0.1 \
  --port "$PORT"

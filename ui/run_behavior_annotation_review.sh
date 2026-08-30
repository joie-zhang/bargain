#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/scratch/gpfs/DANQIC/jz4391/bargain"
PORT="${PORT:-8000}"

cd "$PROJECT_ROOT"
exec .venv/bin/streamlit run ui/behavior_annotation_review.py \
  --server.address 127.0.0.1 \
  --server.port "$PORT" \
  --server.headless true \
  -- "$@"

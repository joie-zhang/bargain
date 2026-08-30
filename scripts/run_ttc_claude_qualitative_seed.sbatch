#!/bin/bash
#SBATCH --job-name=ttc_claude_qual
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --partition=cpu
#SBATCH --output=/scratch/gpfs/DANQIC/jz4391/bargain/analysis/ttc_claude_seed_qualitative_adjudication_20260728/slurm/%x_%j.out
#SBATCH --error=/scratch/gpfs/DANQIC/jz4391/bargain/analysis/ttc_claude_seed_qualitative_adjudication_20260728/slurm/%x_%j.err

set -eo pipefail

PROJECT_DIR="/scratch/gpfs/DANQIC/jz4391/bargain"
OUTPUT_ROOT="$PROJECT_DIR/analysis/ttc_claude_seed_qualitative_adjudication_20260728"
SEED="${1:?seed is required}"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_ROOT/slurm"

module purge
module load anaconda3/2024.2

KEY_ENV_FILE="${BARGAIN_API_KEYS_ENV:-/home/jz4391/.config/bargain/api_keys.env}"
if [[ -f "$KEY_ENV_FILE" ]]; then
  set -a
  source "$KEY_ENV_FILE"
  set +a
fi

source "$PROJECT_DIR/.venv/bin/activate"
export PYTHONUNBUFFERED=1

"$PROJECT_DIR/.venv/bin/python" \
  "$PROJECT_DIR/scripts/run_ttc_claude_qualitative_judge.py" \
  --seed "$SEED" \
  --concurrency 2 \
  --transport proxy \
  --model openai/gpt-5.5 \
  --reasoning-effort xhigh \
  --max-tokens 65536 \
  --max-attempts 4

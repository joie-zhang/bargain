#!/bin/bash
# Run selected Game 1 local-model configs without changing their saved config files.

#SBATCH --job-name=bargain-g1-local
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --constraint=gpu80
#SBATCH --gres=gpu:a100:1
#SBATCH --output=slurm/game1_single_local_%A_%a.out
#SBATCH --error=slurm/game1_single_local_%A_%a.err

set -euo pipefail

BASE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RESULTS_ROOT="${GAME1_RESULTS_ROOT:-$BASE_DIR/experiments/results/scaling_experiment_20260404_064451}"
OUTPUT_ROOT="${GAME1_SINGLE_OUTPUT_ROOT:?Set GAME1_SINGLE_OUTPUT_ROOT to a staging directory.}"
CONFIG_ID="${SLURM_ARRAY_TASK_ID:?Submit this file as a Slurm array job.}"
CONFIG_FILE=$(printf '%s/configs/config_%03d.json' "$RESULTS_ROOT" "$CONFIG_ID")
OUTPUT_DIR=$(printf '%s/config_%03d' "$OUTPUT_ROOT" "$CONFIG_ID")
PYTHON="$BASE_DIR/.venv/bin/python"

cd "$BASE_DIR"
module purge
module load proxy/default

KEY_ENV_FILE="${BARGAIN_API_KEYS_ENV:-$HOME/.config/bargain/api_keys.env}"
if [[ -f "$KEY_ENV_FILE" ]]; then
  set -a
  source "$KEY_ENV_FILE"
  set +a
fi

export OPENROUTER_TRANSPORT="${OPENROUTER_TRANSPORT:-proxy}"
export OPENROUTER_PROXY_POLL_DIR="${OPENROUTER_PROXY_POLL_DIR:-$HOME/openrouter_proxy}"
export OPENROUTER_PROXY_CLIENT_TIMEOUT="${OPENROUTER_PROXY_CLIENT_TIMEOUT:-9000}"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Missing config: $CONFIG_FILE" >&2
  exit 1
fi

mapfile -t CONFIG_VALUES < <(
  "$PYTHON" - "$CONFIG_FILE" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)
for value in (
    *config["models"],
    config["competition_level"],
    config["run_number"],
    config["random_seed"],
    config["model_order"],
    config["discussion_turns"],
    config["num_items"],
    config["max_rounds"],
):
    print(value)
PY
)

MODEL_1="${CONFIG_VALUES[0]}"
MODEL_2="${CONFIG_VALUES[1]}"
COMPETITION_LEVEL="${CONFIG_VALUES[2]}"
RUN_NUMBER="${CONFIG_VALUES[3]}"
RANDOM_SEED="${CONFIG_VALUES[4]}"
MODEL_ORDER="${CONFIG_VALUES[5]}"
DISCUSSION_TURNS="${CONFIG_VALUES[6]}"
NUM_ITEMS="${CONFIG_VALUES[7]}"
MAX_ROUNDS="${CONFIG_VALUES[8]}"

echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo "Models: $MODEL_1 $MODEL_2"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PYTHON" run_strong_models_experiment.py \
  --game-type item_allocation \
  --models "$MODEL_1" "$MODEL_2" \
  --batch \
  --num-runs 1 \
  --run-number "$RUN_NUMBER" \
  --num-items "$NUM_ITEMS" \
  --max-rounds "$MAX_ROUNDS" \
  --competition-level "$COMPETITION_LEVEL" \
  --random-seed "$RANDOM_SEED" \
  --discussion-turns "$DISCUSSION_TURNS" \
  --model-order "$MODEL_ORDER" \
  --max-tokens-per-phase 16384 \
  --output-dir "$OUTPUT_DIR" \
  --job-id "$CONFIG_ID"

#!/usr/bin/env python3
"""Generate an isolated Game 3 Llama 3.2 1B batch for pli-c/pli-cp."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_RESULTS_DIR = REPO_ROOT / "experiments" / "results" / "cofunding_20260405_083548"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "cofunding_20260405_083548_llama32_1b_plicp_20260411"
)

TARGET_CONFIG_IDS = [522, 523, 528]
SOURCE_MODEL = "llama-3.2-1b-instruct"
TARGET_MODEL = "llama-3.2-1b-instruct-cluster"


def strict_result_exists(config: Dict) -> bool:
    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    result_file = output_dir / "experiment_results.json"
    if not result_file.exists():
        return False
    try:
        json.loads(result_file.read_text(encoding="utf-8"))
    except Exception:
        return False
    return True


def rel_output_dir(source_output_dir: str, new_results_dir: Path) -> str:
    source_abs = (REPO_ROOT / source_output_dir).resolve()
    rel_suffix = source_abs.relative_to(SOURCE_RESULTS_DIR)
    dest_abs = new_results_dir / rel_suffix
    return str(dest_abs.relative_to(REPO_ROOT))


def build_selected_configs() -> List[Tuple[int, Dict]]:
    selected: List[Tuple[int, Dict]] = []
    for config_id in TARGET_CONFIG_IDS:
        config_path = SOURCE_RESULTS_DIR / "configs" / f"config_{config_id:04d}.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if SOURCE_MODEL not in config["models"]:
            continue
        if strict_result_exists(config):
            continue
        selected.append((config_id, config))
    return selected


def make_worker_script(results_dir: Path) -> str:
    config_dir = results_dir / "configs"
    return f'''#!/bin/bash
set -euo pipefail

BASE_DIR="{REPO_ROOT}"
cd "${{BASE_DIR}}"
mkdir -p logs/cluster

echo "============================================================"
echo "Co-Funding (Game 3) Llama 3.2 1B pli-cp"
echo "SLURM Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Started at: $(date)"
echo "Node: $SLURM_NODELIST"
echo "============================================================"

module purge
module load anaconda3/2024.2
module load proxy/default

source "${{BASE_DIR}}/.venv/bin/activate"
echo "Python version: $(python3 --version)"
echo ""

CONFIG_DIR="{config_dir}"
MAX_CONFIG=$(ls "${{CONFIG_DIR}}"/config_*.json 2>/dev/null | sed 's/.*config_\\([0-9]*\\)\\.json/\\1/' | sort -n | tail -1)
PADDING_WIDTH=${{#MAX_CONFIG}}
CONFIG_ID_PADDED=$(printf "%0${{PADDING_WIDTH}}d" "${{SLURM_ARRAY_TASK_ID}}")
CONFIG_FILE="${{CONFIG_DIR}}/config_${{CONFIG_ID_PADDED}}.json"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Config file: $CONFIG_FILE"

EXPERIMENT_TYPE=$(python3 -c "import json; print(json.load(open('${{CONFIG_FILE}}'))['experiment_type'])")
MODEL_ORDER=$(python3 -c "import json; print(json.load(open('${{CONFIG_FILE}}'))['model_order'])")
ALPHA=$(python3 -c "import json; print(json.load(open('${{CONFIG_FILE}}'))['alpha'])")
SIGMA=$(python3 -c "import json; print(json.load(open('${{CONFIG_FILE}}'))['sigma'])")
M_PROJECTS=$(python3 -c "import json; print(json.load(open('${{CONFIG_FILE}}'))['m_projects'])")
C_MIN=$(python3 -c "import json; print(json.load(open('${{CONFIG_FILE}}'))['c_min'])")
C_MAX=$(python3 -c "import json; print(json.load(open('${{CONFIG_FILE}}'))['c_max'])")
SEED=$(python3 -c "import json; print(json.load(open('${{CONFIG_FILE}}'))['random_seed'])")
DISCUSSION_TURNS=$(python3 -c "import json; print(json.load(open('${{CONFIG_FILE}}'))['discussion_turns'])")
OUTPUT_DIR=$(python3 -c "import json; print(json.load(open('${{CONFIG_FILE}}'))['output_dir'])")
MAX_TOKENS=$(python3 -c "import json; print(json.load(open('${{CONFIG_FILE}}'))['max_tokens_per_phase'])")
MAX_ROUNDS=$(python3 -c "import json; print(json.load(open('${{CONFIG_FILE}}'))['max_rounds'])")
NUM_RUNS=$(python3 -c "import json; print(json.load(open('${{CONFIG_FILE}}'))['num_runs'])")
RUN_NUMBER=$(python3 -c "import json; print(json.load(open('${{CONFIG_FILE}}'))['run_number'])")
DISCUSSION_TRANSPARENCY=$(python3 -c "import json; print(json.load(open('${{CONFIG_FILE}}')).get('cofunding_discussion_transparency', 'own'))")
ENABLE_COMMIT_VOTE=$(python3 -c "import json; print(str(json.load(open('${{CONFIG_FILE}}')).get('cofunding_enable_commit_vote', True)).lower())")
ENABLE_TIME_DISCOUNT=$(python3 -c "import json; print(str(json.load(open('${{CONFIG_FILE}}')).get('cofunding_enable_time_discount', True)).lower())")
TIME_DISCOUNT=$(python3 -c "import json; print(json.load(open('${{CONFIG_FILE}}')).get('cofunding_time_discount', 0.9))")
MODELS=$(python3 -c "import json; print(' '.join(json.load(open('${{CONFIG_FILE}}'))['models']))")

echo "Experiment type: $EXPERIMENT_TYPE"
echo "Model order: $MODEL_ORDER"
echo "Models: $MODELS"
echo "Alpha: $ALPHA"
echo "Sigma: $SIGMA"
echo "Projects: $M_PROJECTS"
echo "Max rounds: $MAX_ROUNDS"
echo "Num runs: $NUM_RUNS | Run number: $RUN_NUMBER"
echo "Random seed: $SEED"
echo "Discussion transparency: $DISCUSSION_TRANSPARENCY"
echo "Commit vote enabled: $ENABLE_COMMIT_VOTE"
echo "Time discount enabled: $ENABLE_TIME_DISCOUNT (gamma=$TIME_DISCOUNT)"
echo "Output dir: $OUTPUT_DIR"

CMD="python3 run_strong_models_experiment.py"
CMD="$CMD --game-type co_funding"
CMD="$CMD --models $MODELS"
CMD="$CMD --batch --num-runs $NUM_RUNS --run-number $RUN_NUMBER"
CMD="$CMD --m-projects $M_PROJECTS"
CMD="$CMD --alpha $ALPHA --sigma $SIGMA"
CMD="$CMD --c-min $C_MIN --c-max $C_MAX"
CMD="$CMD --max-rounds $MAX_ROUNDS"
CMD="$CMD --random-seed $SEED"
CMD="$CMD --discussion-turns $DISCUSSION_TURNS"
CMD="$CMD --cofunding-discussion-transparency $DISCUSSION_TRANSPARENCY"
CMD="$CMD --model-order $MODEL_ORDER"
CMD="$CMD --max-tokens-per-phase $MAX_TOKENS"
CMD="$CMD --output-dir $OUTPUT_DIR"
CMD="$CMD --job-id $SLURM_ARRAY_TASK_ID"
CMD="$CMD --cofunding-time-discount $TIME_DISCOUNT"
if [[ "$ENABLE_COMMIT_VOTE" != "true" ]]; then
    CMD="$CMD --cofunding-disable-commit-vote"
fi
if [[ "$ENABLE_TIME_DISCOUNT" != "true" ]]; then
    CMD="$CMD --cofunding-disable-time-discount"
fi

echo ""
echo "Running: $CMD"
echo ""

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

set +e
eval "$CMD"
STATUS=$?
set -e

if [[ "$STATUS" -ne 0 ]]; then
    echo "Experiment failed at: $(date)"
    exit "$STATUS"
fi

echo "Experiment completed successfully at: $(date)"
'''


def make_sbatch_script(worker_script: Path) -> str:
    return f'''#!/bin/bash
#SBATCH --job-name=cofund-gpu1-llama32-plicp
#SBATCH --account=pli
#SBATCH --partition=pli-c
#SBATCH --qos=pli-cp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --constraint=gpu80
#SBATCH --gres=gpu:1
#SBATCH --output=logs/cluster/cofund_gpu1_llama32_plicp_%A_%a.out
#SBATCH --error=logs/cluster/cofund_gpu1_llama32_plicp_%A_%a.err

set -euo pipefail
bash "{worker_script}" "$@"
'''


def make_submit_script(config_ids: List[int]) -> str:
    array_spec = ",".join(str(value) for value in config_ids)
    return f'''#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
cd "{REPO_ROOT}"

ONE_GPU_MAX="${{ONE_GPU_MAX:-10}}"

echo "Submitting Llama 3.2 1B Game 3 pli-cp jobs: {array_spec}%${{ONE_GPU_MAX}}"
sbatch --array="{array_spec}%${{ONE_GPU_MAX}}" "$SCRIPT_DIR/run_cofunding_gpu_1x80gb.sbatch"
'''


def main() -> int:
    selected = build_selected_configs()
    output_dir = DEFAULT_OUTPUT_DIR
    config_dir = output_dir / "configs"
    slurm_dir = config_dir / "slurm"
    config_dir.mkdir(parents=True, exist_ok=True)
    slurm_dir.mkdir(parents=True, exist_ok=True)

    selected_ids: List[int] = []

    for config_id, config in selected:
        config["models"] = [TARGET_MODEL if model == SOURCE_MODEL else model for model in config["models"]]
        config["output_dir"] = rel_output_dir(config["output_dir"], output_dir)
        config["cluster_fallback_source_results_dir"] = str(SOURCE_RESULTS_DIR.relative_to(REPO_ROOT))
        config["cluster_fallback_source_model"] = SOURCE_MODEL
        config["cluster_fallback_generated_at"] = datetime.now().astimezone().isoformat()
        config["cluster_fallback_mode"] = "isolated_pli_cp"
        config_path = config_dir / f"config_{config_id:04d}.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        selected_ids.append(config_id)

    worker_script = slurm_dir / "run_cofunding_worker.sh"
    worker_script.write_text(make_worker_script(output_dir), encoding="utf-8")
    worker_script.chmod(0o755)

    one_gpu_sbatch = slurm_dir / "run_cofunding_gpu_1x80gb.sbatch"
    one_gpu_sbatch.write_text(make_sbatch_script(worker_script), encoding="utf-8")
    one_gpu_sbatch.chmod(0o755)

    submit_script = slurm_dir / "submit_all.sh"
    submit_script.write_text(make_submit_script(selected_ids), encoding="utf-8")
    submit_script.chmod(0o755)

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "source_results_dir": str(SOURCE_RESULTS_DIR),
        "output_results_dir": str(output_dir),
        "selected_count": len(selected_ids),
        "selected_config_ids": selected_ids,
        "source_model": SOURCE_MODEL,
        "target_model": TARGET_MODEL,
        "partition": "pli-c",
        "qos": "pli-cp",
        "time_limit": "08:00:00",
        "gres": "gpu:1",
        "constraint": "gpu80",
        "array_limit_default": 10,
        "mode": "isolated",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

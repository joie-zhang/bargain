#!/usr/bin/env python3
"""Generate a Game 3 GPU backfill batch that writes into the original results tree."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_RESULTS_DIR = REPO_ROOT / "experiments" / "results" / "cofunding_20260405_083548"
CONTROL_DIR = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "cofunding_20260405_083548_cluster_backfill_pli_20260409"
)

TARGET_CONFIG_IDS = [396, 397, 404, 408, 409, 412, 522, 523, 528]

MODEL_REPLACEMENTS = {
    "qwen2.5-72b-instruct": "qwen2.5-72b-instruct-cluster",
    "llama-3.2-1b-instruct": "llama-3.2-1b-instruct-cluster",
}

GPU_MAP = {
    "qwen2.5-72b-instruct-cluster": 4,
    "llama-3.2-1b-instruct-cluster": 1,
}


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


def target_model_for_config(config: Dict) -> str:
    for model in config["models"]:
        if model in MODEL_REPLACEMENTS:
            return model
    raise ValueError(f"No cluster replacement for config output_dir={config['output_dir']}")


def build_selected_configs() -> List[Tuple[int, Dict, str]]:
    selected: List[Tuple[int, Dict, str]] = []
    for config_id in TARGET_CONFIG_IDS:
        config_path = SOURCE_RESULTS_DIR / "configs" / f"config_{config_id:04d}.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if strict_result_exists(config):
            continue
        source_model = target_model_for_config(config)
        selected.append((config_id, config, source_model))
    return selected


def make_worker_script(config_dir: Path) -> str:
    return f"""#!/bin/bash
set -e

BASE_DIR="{REPO_ROOT}"
cd "${{BASE_DIR}}"
mkdir -p logs/cluster

echo "============================================================"
echo "Co-Funding (Game 3) Cluster Backfill In-Place"
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
CONFIG_ID_PADDED=$(printf "%0${{PADDING_WIDTH}}d" ${{SLURM_ARRAY_TASK_ID}})
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
"""


def make_sbatch_script(job_name: str, cpus: int, mem: str, gpus: int, worker_script: Path, output_pattern: str, error_pattern: str) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account=pli
#SBATCH --partition=pli
#SBATCH --qos=pli-medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time=08:00:00
#SBATCH --constraint=gpu80
#SBATCH --gres=gpu:{gpus}
#SBATCH --output={output_pattern}
#SBATCH --error={error_pattern}

set -e
bash "{worker_script}" "$@"
"""


def make_submit_script(one_gpu_ids: List[int], four_gpu_ids: List[int]) -> str:
    one_gpu_spec = ",".join(str(value) for value in one_gpu_ids)
    four_gpu_spec = ",".join(str(value) for value in four_gpu_ids)
    return f"""#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
cd "{REPO_ROOT}"

ONE_GPU_MAX="${{ONE_GPU_MAX:-10}}"
FOUR_GPU_MAX="${{FOUR_GPU_MAX:-10}}"

if [[ -n "{one_gpu_spec}" ]]; then
    echo "Submitting 1x80GB in-place backfill jobs to pli: {one_gpu_spec}%${{ONE_GPU_MAX}}"
    sbatch --array="{one_gpu_spec}%${{ONE_GPU_MAX}}" "$SCRIPT_DIR/run_cofunding_gpu_1x80gb.sbatch"
fi

if [[ -n "{four_gpu_spec}" ]]; then
    echo "Submitting 4x80GB in-place backfill jobs to pli: {four_gpu_spec}%${{FOUR_GPU_MAX}}"
    sbatch --array="{four_gpu_spec}%${{FOUR_GPU_MAX}}" "$SCRIPT_DIR/run_cofunding_gpu_4x80gb.sbatch"
fi
"""


def main() -> int:
    selected = build_selected_configs()
    config_dir = CONTROL_DIR / "configs"
    slurm_dir = config_dir / "slurm"
    config_dir.mkdir(parents=True, exist_ok=True)
    slurm_dir.mkdir(parents=True, exist_ok=True)

    selected_by_source: Dict[str, List[int]] = {key: [] for key in MODEL_REPLACEMENTS}
    one_gpu_ids: List[int] = []
    four_gpu_ids: List[int] = []

    for config_id, config, source_model in selected:
        config["models"] = [MODEL_REPLACEMENTS.get(model, model) for model in config["models"]]
        config["cluster_backfill_source_results_dir"] = str(SOURCE_RESULTS_DIR.relative_to(REPO_ROOT))
        config["cluster_backfill_source_model"] = source_model
        config["cluster_backfill_generated_at"] = datetime.now().astimezone().isoformat()
        config["cluster_backfill_mode"] = "in_place"
        config_path = config_dir / f"config_{config_id:04d}.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

        selected_by_source[source_model].append(config_id)
        replacement = MODEL_REPLACEMENTS[source_model]
        gpus = GPU_MAP[replacement]
        if gpus == 4:
            four_gpu_ids.append(config_id)
        else:
            one_gpu_ids.append(config_id)

    worker_script = slurm_dir / "run_cofunding_worker.sh"
    worker_script.write_text(make_worker_script(config_dir), encoding="utf-8")
    worker_script.chmod(0o755)

    one_gpu_sbatch = slurm_dir / "run_cofunding_gpu_1x80gb.sbatch"
    one_gpu_sbatch.write_text(
        make_sbatch_script(
            "cofund-gpu1-inplace",
            8,
            "64G",
            1,
            worker_script,
            "logs/cluster/cofund_gpu1_inplace_%A_%a.out",
            "logs/cluster/cofund_gpu1_inplace_%A_%a.err",
        ),
        encoding="utf-8",
    )
    one_gpu_sbatch.chmod(0o755)

    four_gpu_sbatch = slurm_dir / "run_cofunding_gpu_4x80gb.sbatch"
    four_gpu_sbatch.write_text(
        make_sbatch_script(
            "cofund-gpu4-inplace",
            16,
            "128G",
            4,
            worker_script,
            "logs/cluster/cofund_gpu4_inplace_%A_%a.out",
            "logs/cluster/cofund_gpu4_inplace_%A_%a.err",
        ),
        encoding="utf-8",
    )
    four_gpu_sbatch.chmod(0o755)

    submit_script = slurm_dir / "submit_all.sh"
    submit_script.write_text(make_submit_script(one_gpu_ids, four_gpu_ids), encoding="utf-8")
    submit_script.chmod(0o755)

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "source_results_dir": str(SOURCE_RESULTS_DIR),
        "control_dir": str(CONTROL_DIR),
        "selected_count": len(selected),
        "selected_by_source_model": selected_by_source,
        "one_gpu_config_ids": one_gpu_ids,
        "four_gpu_config_ids": four_gpu_ids,
        "model_replacements": MODEL_REPLACEMENTS,
        "gpu_map": GPU_MAP,
        "partition": "pli",
        "mode": "in_place",
    }
    (CONTROL_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(CONTROL_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

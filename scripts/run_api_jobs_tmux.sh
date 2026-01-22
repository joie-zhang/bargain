#!/bin/bash
# Run API experiment jobs directly in tmux sessions (bypasses SLURM).
#
# Use this for API-only jobs that don't need compute resources.
# Run from a login node like della-vis1.
#
# Usage:
#   ./scripts/run_api_jobs_tmux.sh 130 196 262 263   # Run specific config IDs
#   ./scripts/run_api_jobs_tmux.sh --status          # Check running sessions
#   ./scripts/run_api_jobs_tmux.sh --stop-all        # Kill all api_job_* sessions
#
# To attach to a session:
#   tmux attach -t api_job_130
#
# To kill a specific session:
#   tmux kill-session -t api_job_130

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

CONFIG_DIR="experiments/results/scaling_experiment/configs"

show_status() {
    echo "API job tmux sessions:"
    for session in $(tmux list-sessions -F '#{session_name}' 2>/dev/null | grep '^api_job_' || true); do
        echo "  ✅ $session running"
    done
    if ! tmux list-sessions 2>/dev/null | grep -q 'api_job_'; then
        echo "  No api_job_* sessions running"
    fi
}

stop_all() {
    for session in $(tmux list-sessions -F '#{session_name}' 2>/dev/null | grep '^api_job_' || true); do
        tmux kill-session -t "$session"
        echo "Stopped $session"
    done
}

run_job() {
    local config_id=$1
    local session_name="api_job_${config_id}"

    # Find config file (handle zero-padding)
    local config_file=""
    for f in "${CONFIG_DIR}/config_${config_id}.json" "${CONFIG_DIR}/config_0${config_id}.json" "${CONFIG_DIR}/config_00${config_id}.json" "${CONFIG_DIR}/config_000${config_id}.json"; do
        if [[ -f "$f" ]]; then
            config_file="$f"
            break
        fi
    done

    if [[ -z "$config_file" ]]; then
        echo "❌ Config file not found for ID: $config_id"
        return 1
    fi

    if tmux has-session -t "$session_name" 2>/dev/null; then
        echo "⚠️  Session $session_name already exists"
        echo "   Attach: tmux attach -t $session_name"
        return 0
    fi

    # Extract config values
    local weak_model=$(python3 -c "import json; print(json.load(open('${config_file}'))['weak_model'])")
    local strong_model=$(python3 -c "import json; print(json.load(open('${config_file}'))['strong_model'])")
    local comp_level=$(python3 -c "import json; print(json.load(open('${config_file}'))['competition_level'])")
    local run_num=$(python3 -c "import json; print(json.load(open('${config_file}'))['run_number'])")
    local seed=$(python3 -c "import json; print(json.load(open('${config_file}'))['random_seed'])")
    local model_order=$(python3 -c "import json; print(json.load(open('${config_file}'))['model_order'])")
    local discussion_turns=$(python3 -c "import json; print(json.load(open('${config_file}'))['discussion_turns'])")
    local output_dir=$(python3 -c "import json; print(json.load(open('${config_file}'))['output_dir'])")

    # Determine model order
    if [[ "$model_order" == "weak_first" ]]; then
        models="$weak_model $strong_model"
    else
        models="$strong_model $weak_model"
    fi

    echo "Starting job $config_id: $weak_model vs $strong_model (comp=$comp_level)"

    # Create tmux session with the experiment command
    tmux new-session -d -s "$session_name" "
        cd $BASE_DIR
        source .venv/bin/activate
        echo '=========================================='
        echo 'Config ID: $config_id'
        echo 'Models: $models'
        echo 'Competition: $comp_level'
        echo 'Output: $output_dir'
        echo '=========================================='
        python3 run_strong_models_experiment.py \\
            --models $models \\
            --batch \\
            --num-runs 1 \\
            --run-number $run_num \\
            --competition-level $comp_level \\
            --random-seed $seed \\
            --discussion-turns $discussion_turns \\
            --model-order $model_order \\
            --output-dir '$output_dir' \\
            --job-id $config_id
        echo ''
        echo 'Job completed. Press Enter to close.'
        read
    "

    echo "✅ Started $session_name"
    echo "   Attach: tmux attach -t $session_name"
}

# Handle arguments
case "${1:-}" in
    --status)
        show_status
        exit 0
        ;;
    --stop-all)
        stop_all
        exit 0
        ;;
    --help|-h)
        echo "Usage: $0 <config_id> [config_id2] [config_id3] ..."
        echo "       $0 --status"
        echo "       $0 --stop-all"
        exit 0
        ;;
    "")
        echo "Usage: $0 <config_id> [config_id2] [config_id3] ..."
        echo ""
        echo "Example: $0 130 196 262 263"
        exit 1
        ;;
esac

# Run each specified job
for config_id in "$@"; do
    run_job "$config_id"
done

echo ""
echo "Monitor with: $0 --status"
echo "Stop all with: $0 --stop-all"

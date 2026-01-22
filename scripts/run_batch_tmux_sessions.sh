#!/bin/bash
# =============================================================================
# Run Waiting SLURM Jobs as Sequential Tmux Batches
# =============================================================================
#
# Converts pending SLURM array jobs into 5 tmux sessions, each running
# jobs sequentially to avoid API rate limits while still parallelizing.
#
# Usage:
#   ./scripts/run_batch_tmux_sessions.sh           # Start all 5 batch sessions
#   ./scripts/run_batch_tmux_sessions.sh --status  # Check running sessions
#   ./scripts/run_batch_tmux_sessions.sh --stop    # Kill all batch sessions
#
# Batches:
#   batch_1: 363-395  (33 jobs)
#   batch_2: 506-527  (22 jobs)
#   batch_3: 550-580  (31 jobs)
#   batch_4: 581-615  (35 jobs)
#   batch_5: 638-659  (22 jobs)
#
# To attach to a session:
#   tmux attach -t api_batch_1
#
# To view logs:
#   tail -f logs/tmux/batch_1.log
#
# =============================================================================

set -e

BASE_DIR="/scratch/gpfs/DANQIC/jz4391/bargain"
cd "$BASE_DIR"

# Use the exact config directory from the sbatch file
CONFIG_DIR="/scratch/gpfs/DANQIC/jz4391/bargain/experiments/results/scaling_experiment_20260121_070359/configs"
LOG_DIR="$BASE_DIR/logs/tmux"

mkdir -p "$LOG_DIR"

# Define the 5 batches
declare -A BATCH_START BATCH_END
BATCH_START[1]=363; BATCH_END[1]=395
BATCH_START[2]=506; BATCH_END[2]=527
BATCH_START[3]=550; BATCH_END[3]=580
BATCH_START[4]=581; BATCH_END[4]=615
BATCH_START[5]=638; BATCH_END[5]=659

show_status() {
    echo "Batch tmux sessions status:"
    echo ""
    for batch_num in 1 2 3 4 5; do
        session_name="api_batch_${batch_num}"
        if tmux has-session -t "$session_name" 2>/dev/null; then
            # Try to get current progress from log
            log_file="$LOG_DIR/batch_${batch_num}.log"
            if [[ -f "$log_file" ]]; then
                last_job=$(grep -o "Running config [0-9]*" "$log_file" 2>/dev/null | tail -1 | grep -o "[0-9]*" || echo "?")
                completed=$(grep -c "✅ Completed config" "$log_file" 2>/dev/null || echo "0")
                total=$((BATCH_END[$batch_num] - BATCH_START[$batch_num] + 1))
                echo "  ✅ $session_name: running (${completed}/${total} done, last: $last_job)"
            else
                echo "  ✅ $session_name: running"
            fi
        else
            echo "  ⭕ $session_name: not running (range ${BATCH_START[$batch_num]}-${BATCH_END[$batch_num]})"
        fi
    done
    echo ""
    echo "Attach: tmux attach -t api_batch_N"
    echo "Logs:   tail -f $LOG_DIR/batch_N.log"
}

stop_all() {
    for batch_num in 1 2 3 4 5; do
        session_name="api_batch_${batch_num}"
        if tmux has-session -t "$session_name" 2>/dev/null; then
            tmux kill-session -t "$session_name"
            echo "Stopped $session_name"
        fi
    done
}

start_batch() {
    local batch_num=$1
    local start_id=${BATCH_START[$batch_num]}
    local end_id=${BATCH_END[$batch_num]}
    local session_name="api_batch_${batch_num}"
    local log_file="$LOG_DIR/batch_${batch_num}.log"

    if tmux has-session -t "$session_name" 2>/dev/null; then
        echo "⚠️  $session_name already exists (${start_id}-${end_id})"
        echo "   Attach: tmux attach -t $session_name"
        return 0
    fi

    echo "Starting $session_name: configs ${start_id}-${end_id}"

    # Create the batch runner script inline in tmux
    tmux new-session -d -s "$session_name" bash -c "
        cd '$BASE_DIR'
        source .venv/bin/activate

        LOG_FILE='$log_file'
        CONFIG_DIR='$CONFIG_DIR'

        echo '==========================================' | tee \"\$LOG_FILE\"
        echo 'Batch $batch_num: configs ${start_id}-${end_id}' | tee -a \"\$LOG_FILE\"
        echo 'Started: \$(date)' | tee -a \"\$LOG_FILE\"
        echo '==========================================' | tee -a \"\$LOG_FILE\"
        echo '' | tee -a \"\$LOG_FILE\"

        for config_id in \$(seq ${start_id} ${end_id}); do
            # Find config file - try without padding first, then with various padding
            config_file=\"\"
            for f in \"\${CONFIG_DIR}/config_\${config_id}.json\" \"\${CONFIG_DIR}/config_0\${config_id}.json\" \"\${CONFIG_DIR}/config_00\${config_id}.json\"; do
                if [[ -f \"\$f\" ]]; then
                    config_file=\"\$f\"
                    break
                fi
            done

            if [[ ! -f \"\$config_file\" ]]; then
                echo \"❌ Config not found: \$config_file\" | tee -a \"\$LOG_FILE\"
                continue
            fi

            # Extract config values
            weak_model=\$(python3 -c \"import json; print(json.load(open('\${config_file}'))['weak_model'])\")
            strong_model=\$(python3 -c \"import json; print(json.load(open('\${config_file}'))['strong_model'])\")
            comp_level=\$(python3 -c \"import json; print(json.load(open('\${config_file}'))['competition_level'])\")
            run_num=\$(python3 -c \"import json; print(json.load(open('\${config_file}'))['run_number'])\")
            seed=\$(python3 -c \"import json; print(json.load(open('\${config_file}'))['random_seed'])\")
            model_order=\$(python3 -c \"import json; print(json.load(open('\${config_file}'))['model_order'])\")
            discussion_turns=\$(python3 -c \"import json; print(json.load(open('\${config_file}'))['discussion_turns'])\")
            output_dir=\$(python3 -c \"import json; print(json.load(open('\${config_file}'))['output_dir'])\")

            # Determine model order
            if [[ \"\$model_order\" == \"weak_first\" ]]; then
                models=\"\$weak_model \$strong_model\"
            else
                models=\"\$strong_model \$weak_model\"
            fi

            echo '' | tee -a \"\$LOG_FILE\"
            echo \"------------------------------------------\" | tee -a \"\$LOG_FILE\"
            echo \"Running config \$config_id: \$weak_model vs \$strong_model\" | tee -a \"\$LOG_FILE\"
            echo \"Competition: \$comp_level, Order: \$model_order\" | tee -a \"\$LOG_FILE\"
            echo \"Started: \$(date)\" | tee -a \"\$LOG_FILE\"
            echo \"------------------------------------------\" | tee -a \"\$LOG_FILE\"

            if python3 run_strong_models_experiment.py \\
                --models \$models \\
                --batch \\
                --num-runs 1 \\
                --run-number \$run_num \\
                --competition-level \$comp_level \\
                --random-seed \$seed \\
                --discussion-turns \$discussion_turns \\
                --model-order \$model_order \\
                --output-dir \"\$output_dir\" \\
                --job-id \$config_id 2>&1 | tee -a \"\$LOG_FILE\"; then
                echo \"✅ Completed config \$config_id at \$(date)\" | tee -a \"\$LOG_FILE\"
            else
                echo \"❌ Failed config \$config_id at \$(date)\" | tee -a \"\$LOG_FILE\"
            fi
        done

        echo '' | tee -a \"\$LOG_FILE\"
        echo '==========================================' | tee -a \"\$LOG_FILE\"
        echo 'Batch $batch_num COMPLETE at \$(date)' | tee -a \"\$LOG_FILE\"
        echo '==========================================' | tee -a \"\$LOG_FILE\"
        echo ''
        echo 'All jobs in batch finished. Press Enter to close.'
        read
    "

    echo "✅ Started $session_name"
}

# Handle arguments
case "${1:-}" in
    --status|-s)
        show_status
        exit 0
        ;;
    --stop|--stop-all)
        stop_all
        exit 0
        ;;
    --help|-h)
        echo "Usage: $0              # Start all 5 batch sessions"
        echo "       $0 --status     # Check running sessions"
        echo "       $0 --stop       # Kill all batch sessions"
        echo ""
        echo "Batches:"
        for batch_num in 1 2 3 4 5; do
            echo "  batch_${batch_num}: ${BATCH_START[$batch_num]}-${BATCH_END[$batch_num]}"
        done
        exit 0
        ;;
esac

# Start all batches
echo "Starting 5 batch sessions..."
echo "Each session runs jobs SEQUENTIALLY to avoid rate limits."
echo ""

for batch_num in 1 2 3 4 5; do
    start_batch $batch_num
done

echo ""
echo "All batches started!"
echo ""
echo "Monitor:  $0 --status"
echo "Stop all: $0 --stop"
echo "Attach:   tmux attach -t api_batch_N"
echo "Logs:     tail -f $LOG_DIR/batch_N.log"

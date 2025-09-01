#!/bin/bash
# Kill all running experiment jobs

echo "============================================================"
echo "KILLING ALL RUNNING EXPERIMENT JOBS"
echo "============================================================"

# Kill all python processes running the experiment
echo "Looking for running experiment processes..."

# Find and kill run_strong_models_experiment.py processes
KILLED_COUNT=0
for pid in $(pgrep -f "run_strong_models_experiment.py"); do
    echo "Killing PID $pid (run_strong_models_experiment.py)"
    kill -9 $pid 2>/dev/null && ((KILLED_COUNT++))
done

# Kill run_single_experiment_simple.sh processes
for pid in $(pgrep -f "run_single_experiment_simple.sh"); do
    echo "Killing PID $pid (run_single_experiment_simple.sh)"
    kill -9 $pid 2>/dev/null && ((KILLED_COUNT++))
done

# Kill run_all_simple.sh if running
for pid in $(pgrep -f "run_all_simple.sh"); do
    echo "Killing PID $pid (run_all_simple.sh)"
    kill -9 $pid 2>/dev/null && ((KILLED_COUNT++))
done

# Kill any xargs processes that might be managing parallel jobs
for pid in $(pgrep -f "xargs.*run_job"); do
    echo "Killing PID $pid (xargs job manager)"
    kill -9 $pid 2>/dev/null && ((KILLED_COUNT++))
done

echo ""
echo "============================================================"
echo "CLEANUP COMPLETE"
echo "============================================================"
echo "Killed $KILLED_COUNT processes"

# Show current status
echo ""
echo "Checking for any remaining processes..."
REMAINING=$(pgrep -f "run_strong_models_experiment.py|run_single_experiment_simple.sh|run_all_simple.sh" | wc -l)
if [ $REMAINING -eq 0 ]; then
    echo "✅ All experiment processes terminated"
else
    echo "⚠️  Warning: $REMAINING processes may still be running"
    echo "Run 'ps aux | grep run_strong_models' to check"
fi

echo ""
echo "To see what experiments were completed before termination:"
echo "  ls experiments/results/scaling_experiment/logs/completed_*.flag | wc -l"
echo ""
echo "To restart experiments (will skip completed ones):"
echo "  scripts/run_all_simple.sh"
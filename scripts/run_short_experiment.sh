#!/bin/bash
# Run the SHORT experiment - claude-3-opus vs all strong models at competition level 1.0
set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${BASE_DIR}/experiments/results/short_experiment"
SCRIPTS_DIR="${BASE_DIR}/scripts"
LOGS_DIR="${RESULTS_DIR}/logs"
CONFIG_DIR="${RESULTS_DIR}/configs"

# Configuration
MAX_PARALLEL=${1:-4}  # Default to 4 parallel jobs

# Create directories
mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}" "${CONFIG_DIR}"

echo "============================================================"
echo "SHORT EXPERIMENT RUNNER (Single Direction)"
echo "============================================================"
echo "Running: Claude 3 Opus vs All Strong Models"
echo "Competition Level: 1.0 (Fully Competitive)"
echo "Model Order: Claude 3 Opus always goes first"
echo "Max parallel jobs: ${MAX_PARALLEL}"
echo ""

# Step 1: Generate configs if needed
if [ ! -f "${CONFIG_DIR}/config_0.json" ]; then
    echo "Generating experiment configurations..."
    "${SCRIPTS_DIR}/generate_configs_short_run.sh"
fi

# Step 2: Count experiments
TOTAL_JOBS=$(ls -1 "${CONFIG_DIR}"/config_*.json 2>/dev/null | wc -l)
echo "Total experiments: ${TOTAL_JOBS}"
echo ""

# Step 3: Run single experiment function
run_single_experiment() {
    local CONFIG_ID=$1
    local CONFIG_FILE="${CONFIG_DIR}/config_${CONFIG_ID}.json"
    local LOG_FILE="${LOGS_DIR}/experiment_${CONFIG_ID}.log"
    local COMPLETED_FLAG="${LOGS_DIR}/completed_${CONFIG_ID}.flag"
    
    # Skip if already completed
    if [ -f "${COMPLETED_FLAG}" ]; then
        echo "[${CONFIG_ID}] Already completed, skipping..."
        return 0
    fi
    
    echo "[${CONFIG_ID}] Starting experiment..."
    
    # Parse config file to extract parameters
    WEAK_MODEL=$(python -c "import json; c=json.load(open('${CONFIG_FILE}')); print(c['weak_model'])")
    STRONG_MODEL=$(python -c "import json; c=json.load(open('${CONFIG_FILE}')); print(c['strong_model'])")
    COMPETITION=$(python -c "import json; c=json.load(open('${CONFIG_FILE}')); print(c['competition_level'])")
    SEED=$(python -c "import json; c=json.load(open('${CONFIG_FILE}')); print(c['random_seed'])")
    NUM_ITEMS=$(python -c "import json; c=json.load(open('${CONFIG_FILE}')); print(c['num_items'])")
    MAX_ROUNDS=$(python -c "import json; c=json.load(open('${CONFIG_FILE}')); print(c['max_rounds'])")
    
    # Run the experiment with batch mode for 5 different seeds
    if python "${BASE_DIR}/run_strong_models_experiment.py" \
        --models "${WEAK_MODEL}" "${STRONG_MODEL}" \
        --competition-level "${COMPETITION}" \
        --random-seed "${SEED}" \
        --num-items "${NUM_ITEMS}" \
        --max-rounds "${MAX_ROUNDS}" \
        --job-id "${CONFIG_ID}" \
        --batch \
        --batch-size 5 \
        > "${LOG_FILE}" 2>&1; then
        
        touch "${COMPLETED_FLAG}"
        echo "[${CONFIG_ID}] ✅ Completed successfully"
    else
        echo "[${CONFIG_ID}] ❌ Failed - check ${LOG_FILE}"
        return 1
    fi
}

# Export function and variables for parallel execution
export -f run_single_experiment
export CONFIG_DIR LOGS_DIR BASE_DIR RESULTS_DIR

# Step 4: Run experiments in parallel
echo "Starting parallel execution with ${MAX_PARALLEL} workers..."
START_TIME=$(date +%s)

# Use GNU parallel if available, otherwise fall back to xargs
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel..."
    seq 0 $((TOTAL_JOBS - 1)) | parallel -j ${MAX_PARALLEL} run_single_experiment {}
else
    echo "Using xargs for parallel execution..."
    seq 0 $((TOTAL_JOBS - 1)) | xargs -P ${MAX_PARALLEL} -I {} bash -c 'run_single_experiment {}'
fi

# Step 5: Summary
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "============================================================"
echo "SHORT EXPERIMENT COMPLETE"
echo "============================================================"

# Count results
COMPLETED=$(ls -1 "${LOGS_DIR}"/completed_*.flag 2>/dev/null | wc -l || echo "0")
FAILED=$(grep -l "FAILED\|ERROR\|Traceback" "${LOGS_DIR}"/experiment_*.log 2>/dev/null | wc -l || echo "0")

echo "Total time: $((ELAPSED / 60)) minutes $((ELAPSED % 60)) seconds"
echo "Completed: ${COMPLETED}/${TOTAL_JOBS}"
echo "Failed: ${FAILED}"

if [ ${FAILED} -gt 0 ]; then
    echo ""
    echo "Failed experiments:"
    grep -l "FAILED\|ERROR\|Traceback" "${LOGS_DIR}"/experiment_*.log 2>/dev/null | while read log; do
        exp_id=$(basename "$log" | sed 's/experiment_\([0-9]*\)\.log/\1/')
        echo "  - Experiment ${exp_id}: ${log}"
    done
fi

# Step 6: Collect and analyze results
echo ""
echo "Collecting results..."

# Create summary script
cat > "${RESULTS_DIR}/analyze_results.py" << 'EOF'
#!/usr/bin/env python3
"""Analyze results from the short experiment."""

import json
import os
from pathlib import Path
import pandas as pd
from collections import defaultdict
import glob

def analyze_short_experiment():
    results_dir = Path(__file__).parent
    configs_dir = results_dir / "configs"
    experiments_dir = Path(__file__).parent.parent.parent / "experiments" / "results"
    
    # Load all results
    results = []
    for config_file in sorted(configs_dir.glob("config_*.json")):
        with open(config_file) as f:
            config = json.load(f)
        
        # Look for result files with pattern strong_models_*_configXXX
        config_id = config["experiment_id"]
        pattern = f"strong_models_*_config{config_id:03d}"
        matching_dirs = list(experiments_dir.glob(pattern))
        
        if matching_dirs:
            # Get the most recent matching directory
            result_dir = sorted(matching_dirs)[-1]
            
            # Look for batch run results (run_1 through run_5)
            for run_num in range(1, 6):
                result_file = result_dir / f"run_{run_num}_experiment_results.json"
                
                if result_file.exists():
                    with open(result_file) as f:
                        result = json.load(f)
                    
                    # Extract key metrics
                    # Get agent IDs (they might have suffixes like _agent_0)
                    agent_utilities = result.get("final_utilities", {})
                    weak_utility = 0
                    strong_utility = 0
                    
                    for agent_id, utility in agent_utilities.items():
                        if config["weak_model"] in agent_id:
                            weak_utility = utility
                        elif config["strong_model"] in agent_id:
                            strong_utility = utility
                    
                    results.append({
                        "exp_id": config["experiment_id"],
                        "run_num": run_num,
                        "weak_model": config["weak_model"],
                        "strong_model": config["strong_model"],
                        "model_order": config["model_order"],
                        "competition_level": config["competition_level"],
                        "consensus_reached": result.get("consensus_reached", False),
                        "num_rounds": result.get("final_round", 0),
                        "weak_utility": weak_utility if result.get("consensus_reached") else 0,
                        "strong_utility": strong_utility if result.get("consensus_reached") else 0,
                    })
    
    if not results:
        print("No results found yet.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Analysis
    print("\n" + "="*60)
    print("SHORT EXPERIMENT RESULTS ANALYSIS")
    print("="*60)
    
    print(f"\nTotal experiments completed: {len(df)}")
    print(f"Consensus rate: {df['consensus_reached'].mean():.1%}")
    print(f"Average rounds to consensus: {df[df['consensus_reached']]['num_rounds'].mean():.1f}")
    
    # Overall win rate
    print("\n--- Overall Results ---")
    # Who got higher utility?
    wins = df.apply(lambda x: "weak" if x["weak_utility"] > x["strong_utility"] else "strong", axis=1)
    weak_win_rate = (wins == "weak").mean()
    print(f"Claude 3 Opus win rate: {weak_win_rate:.1%}")
    print(f"Strong models win rate: {(1-weak_win_rate):.1%}")
    
    # Results by strong model (averaged across 5 runs)
    print("\n--- Results by Strong Model (Averaged Across 5 Runs) ---")
    for model in df["strong_model"].unique():
        model_df = df[df["strong_model"] == model]
        consensus_rate = model_df["consensus_reached"].mean()
        num_runs = len(model_df)
        
        # Calculate exploitation (strong utility - weak utility)
        consensual_df = model_df[model_df["consensus_reached"]]
        if len(consensual_df) > 0:
            avg_exploitation = (consensual_df["strong_utility"] - consensual_df["weak_utility"]).mean()
            weak_win_rate = (consensual_df["weak_utility"] > consensual_df["strong_utility"]).mean()
            print(f"{model:20} | Runs: {num_runs} | Consensus: {consensus_rate:.0%} | Weak Win Rate: {weak_win_rate:.0%} | Avg Exploitation: {avg_exploitation:+.1f}")
        else:
            print(f"{model:20} | Runs: {num_runs} | Consensus: {consensus_rate:.0%} | No consensual outcomes")
    
    # Save summary
    summary_file = results_dir / "short_experiment_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "total_experiments": len(df),
            "consensus_rate": float(df["consensus_reached"].mean()),
            "results_by_model": df.groupby("strong_model").agg({
                "consensus_reached": "mean",
                "weak_utility": "mean",
                "strong_utility": "mean"
            }).to_dict()
        }, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")

if __name__ == "__main__":
    analyze_short_experiment()
EOF

python "${RESULTS_DIR}/analyze_results.py"

echo ""
echo "✅ Short experiment complete!"
echo "Results saved to: ${RESULTS_DIR}"
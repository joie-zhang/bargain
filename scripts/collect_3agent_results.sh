#!/bin/bash
# Collect and summarize results from 3-agent experiments

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${BASE_DIR}/experiments/results/3agent_experiment"
CONFIG_DIR="${RESULTS_DIR}/configs"
LOGS_DIR="${RESULTS_DIR}/logs"
SUMMARY_FILE="${RESULTS_DIR}/results_summary.csv"

echo "Collecting 3-agent experiment results..."

# Create CSV header
echo "experiment_id,experiment_type,model1,model2,model3,competition_level,run_number,consensus_reached,rounds_to_consensus,exploitation_detected,status" > "${SUMMARY_FILE}"

# Process each completed experiment
for config_file in "${CONFIG_DIR}"/config_*.json; do
    if [ ! -f "$config_file" ]; then
        continue
    fi
    
    # Extract experiment ID from filename
    exp_id=$(basename "$config_file" | sed 's/config_\([0-9]*\)\.json/\1/')
    
    # Skip if not completed
    if [ ! -f "${LOGS_DIR}/completed_${exp_id}.flag" ]; then
        continue
    fi
    
    # Extract config info
    exp_type=$(grep -o '"experiment_type": "[^"]*"' "$config_file" | cut -d'"' -f4)
    models=$(grep -o '"models": \[[^]]*\]' "$config_file" | sed 's/"models": \[//;s/\]//;s/"//g')
    model1=$(echo "$models" | cut -d',' -f1 | tr -d ' "')
    model2=$(echo "$models" | cut -d',' -f2 | tr -d ' "')
    model3=$(echo "$models" | cut -d',' -f3 | tr -d ' "')
    comp_level=$(grep -o '"competition_level": [0-9.]*' "$config_file" | grep -o '[0-9.]*')
    run_num=$(grep -o '"run_number": [0-9]*' "$config_file" | grep -o '[0-9]*')
    output_dir=$(grep -o '"output_dir": "[^"]*"' "$config_file" | cut -d'"' -f4)
    
    # Check for results in the output directory
    FULL_OUTPUT_DIR="${BASE_DIR}/${output_dir}"
    
    # Default values
    consensus="N/A"
    rounds="N/A"
    exploitation="N/A"
    status="UNKNOWN"
    
    # Check if experiment succeeded
    if grep -q "SUCCESS" "${LOGS_DIR}/experiment_${exp_id}.log" 2>/dev/null; then
        status="SUCCESS"
        
        # Try to extract results from the experiment output
        # Look for the batch summary file
        batch_summary=$(find "${FULL_OUTPUT_DIR}/.." -name "*summary.json" 2>/dev/null | head -1)
        
        if [ -f "$batch_summary" ]; then
            # Extract metrics from summary
            consensus=$(python3 -c "import json; data=json.load(open('$batch_summary')); print('Yes' if data.get('consensus_rate', 0) > 0 else 'No')" 2>/dev/null || echo "N/A")
            rounds=$(python3 -c "import json; data=json.load(open('$batch_summary')); print(int(data.get('average_rounds', 0)))" 2>/dev/null || echo "N/A")
            exploitation=$(python3 -c "import json; data=json.load(open('$batch_summary')); print('Yes' if data.get('exploitation_rate', 0) > 0 else 'No')" 2>/dev/null || echo "N/A")
        fi
    else
        status="FAILED"
    fi
    
    # Write to CSV
    echo "${exp_id},${exp_type},${model1},${model2},${model3},${comp_level},${run_num},${consensus},${rounds},${exploitation},${status}" >> "${SUMMARY_FILE}"
done

# Count results
TOTAL=$(tail -n +2 "${SUMMARY_FILE}" | wc -l)
SUCCESS=$(grep -c ",SUCCESS$" "${SUMMARY_FILE}" || echo "0")
FAILED=$(grep -c ",FAILED$" "${SUMMARY_FILE}" || echo "0")

echo ""
echo "Results Summary:"
echo "  Total experiments: ${TOTAL}"
echo "  Successful: ${SUCCESS}"
echo "  Failed: ${FAILED}"
echo ""
echo "Results saved to: ${SUMMARY_FILE}"

# Create a summary by experiment type
echo ""
echo "Summary by Experiment Type:"
echo "----------------------------"

for exp_type in "3agent_2weak_1strong" "3agent_1weak_2strong"; do
    count=$(grep -c ",${exp_type}," "${SUMMARY_FILE}" || echo "0")
    success=$(grep ",${exp_type}," "${SUMMARY_FILE}" | grep -c ",SUCCESS$" || echo "0")
    
    if [ ${count} -gt 0 ]; then
        echo "${exp_type}: ${success}/${count} successful"
        
        # Calculate consensus rate per competition level
        for comp in 0.0 0.25 0.5 0.75 1.0; do
            comp_count=$(grep ",${exp_type}," "${SUMMARY_FILE}" | grep ",${comp}," | grep -c ",SUCCESS$" || echo "0")
            comp_consensus=$(grep ",${exp_type}," "${SUMMARY_FILE}" | grep ",${comp}," | grep -c ",Yes," || echo "0")
            
            if [ ${comp_count} -gt 0 ]; then
                echo "  Competition ${comp}: ${comp_consensus}/${comp_count} reached consensus"
            fi
        done
    fi
done

echo ""
echo "✅ Results collection complete"
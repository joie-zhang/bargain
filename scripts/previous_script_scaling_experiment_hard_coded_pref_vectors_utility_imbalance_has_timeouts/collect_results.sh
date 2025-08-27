#!/bin/bash
# Collect and aggregate results from all experiments

set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${BASE_DIR}/experiments/results/scaling_experiment"
CONFIG_DIR="${RESULTS_DIR}/configs"
LOGS_DIR="${RESULTS_DIR}/logs"

echo "Collecting results from ${RESULTS_DIR}..."

# Create aggregated results file
AGGREGATE_FILE="${RESULTS_DIR}/all_results.json"
SUMMARY_FILE="${RESULTS_DIR}/summary.json"
CSV_FILE="${RESULTS_DIR}/results.csv"

# Python script to aggregate results
python3 << 'EOF'
import json
import os
import sys
from pathlib import Path
import csv

base_dir = os.environ.get('BASE_DIR', '.')
results_dir = Path(base_dir) / "experiments/results/scaling_experiment"
config_dir = results_dir / "configs"
logs_dir = results_dir / "logs"

# Collect all results
all_results = []
successful = 0
failed = 0
timeout = 0

# Process each configuration
for config_file in sorted(config_dir.glob("config_*.json")):
    job_id = config_file.stem.split('_')[1]
    
    # Load config
    with open(config_file) as f:
        config = json.load(f)
    
    # Check for result file
    result_dir = Path(base_dir) / config['output_dir']
    result_file = result_dir / f"result_{job_id}.json"
    
    result_data = {
        "job_id": int(job_id),
        "weak_model": config['weak_model'],
        "strong_model": config['strong_model'],
        "competition_level": config['competition_level'],
        "config": config
    }
    
    # Check completion status
    if result_file.exists():
        with open(result_file) as f:
            result = json.load(f)
            result_data.update(result)
            
            if result.get('status') == 'SUCCESS':
                successful += 1
                result_data['success'] = True
            elif result.get('status') == 'TIMEOUT':
                timeout += 1
                result_data['success'] = False
            else:
                failed += 1
                result_data['success'] = False
    else:
        # Check if job was started
        log_file = logs_dir / f"experiment_{job_id}.log"
        if log_file.exists():
            result_data['status'] = 'INCOMPLETE'
            failed += 1
        else:
            result_data['status'] = 'NOT_STARTED'
            failed += 1
        result_data['success'] = False
    
    all_results.append(result_data)

# Calculate statistics
total = len(all_results)
success_rate = successful / total if total > 0 else 0

# Create summary
summary = {
    "total": total,
    "successful": successful,
    "failed": failed,
    "timeout": timeout,
    "success_rate": success_rate,
    "by_model_pair": {},
    "by_competition_level": {}
}

# Aggregate by model pair
for result in all_results:
    pair = f"{result['weak_model']}_vs_{result['strong_model']}"
    if pair not in summary["by_model_pair"]:
        summary["by_model_pair"][pair] = {
            "total": 0,
            "successful": 0,
            "competition_levels": {}
        }
    
    summary["by_model_pair"][pair]["total"] += 1
    if result.get('success', False):
        summary["by_model_pair"][pair]["successful"] += 1
    
    comp_level = str(result['competition_level'])
    if comp_level not in summary["by_model_pair"][pair]["competition_levels"]:
        summary["by_model_pair"][pair]["competition_levels"][comp_level] = {
            "total": 0,
            "successful": 0
        }
    
    summary["by_model_pair"][pair]["competition_levels"][comp_level]["total"] += 1
    if result.get('success', False):
        summary["by_model_pair"][pair]["competition_levels"][comp_level]["successful"] += 1

# Aggregate by competition level
for result in all_results:
    comp_level = str(result['competition_level'])
    if comp_level not in summary["by_competition_level"]:
        summary["by_competition_level"][comp_level] = {
            "total": 0,
            "successful": 0,
            "model_pairs": {}
        }
    
    summary["by_competition_level"][comp_level]["total"] += 1
    if result.get('success', False):
        summary["by_competition_level"][comp_level]["successful"] += 1

# Save all results
aggregate_file = results_dir / "all_results.json"
with open(aggregate_file, 'w') as f:
    json.dump(all_results, f, indent=2)

# Save summary
summary_file = results_dir / "summary.json"
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

# Create CSV for easy analysis
csv_file = results_dir / "results.csv"
with open(csv_file, 'w', newline='') as f:
    if all_results:
        fieldnames = ['job_id', 'weak_model', 'strong_model', 'competition_level', 
                     'status', 'success', 'timestamp']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in all_results:
            row = {k: result.get(k, '') for k in fieldnames}
            writer.writerow(row)

print(f"✅ Collected {total} results")
print(f"   Successful: {successful}")
print(f"   Failed: {failed}")
print(f"   Timeout: {timeout}")
print(f"   Success rate: {success_rate*100:.1f}%")
print()
print(f"Results saved to:")
print(f"  JSON: {aggregate_file}")
print(f"  Summary: {summary_file}")
print(f"  CSV: {csv_file}")
EOF

# Create visualization script
VIZ_SCRIPT="${RESULTS_DIR}/visualize_results.py"
cat > "${VIZ_SCRIPT}" << 'EOF'
#!/usr/bin/env python3
"""
Visualize results from the scaling experiments
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load results
results_dir = Path(__file__).parent
with open(results_dir / "all_results.json") as f:
    results = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(results)

# Create win rate heatmap
if 'winner' in df.columns:
    # Calculate win rates
    win_rates = {}
    for weak in df['weak_model'].unique():
        for strong in df['strong_model'].unique():
            subset = df[(df['weak_model'] == weak) & (df['strong_model'] == strong)]
            if len(subset) > 0:
                weak_wins = subset['winner'].str.contains(weak, na=False).sum()
                total = len(subset)
                win_rate = weak_wins / total if total > 0 else 0
                win_rates[f"{weak}_vs_{strong}"] = win_rate
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    # ... plotting code ...
    plt.savefig(results_dir / "win_rates_heatmap.png")
    print(f"Saved heatmap to {results_dir / 'win_rates_heatmap.png'}")

# Create competition level analysis
fig, ax = plt.subplots(figsize=(10, 6))
competition_data = df.groupby('competition_level')['success'].mean()
competition_data.plot(kind='bar', ax=ax)
ax.set_title('Success Rate by Competition Level')
ax.set_xlabel('Competition Level')
ax.set_ylabel('Success Rate')
plt.tight_layout()
plt.savefig(results_dir / "competition_analysis.png")
print(f"Saved plot to {results_dir / 'competition_analysis.png'}")
EOF

chmod +x "${VIZ_SCRIPT}"

echo ""
echo "✅ Results collection complete!"
echo ""
echo "To visualize results, run:"
echo "  python3 ${VIZ_SCRIPT}"
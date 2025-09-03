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

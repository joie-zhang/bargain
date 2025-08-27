#!/usr/bin/env python3
"""
Create heatmaps for win rates between baseline models and strong models.
Generates two heatmaps:
1. Baseline agent win rate against strong agent
2. Strong agent win rate against baseline agent
"""

import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd

# Define baseline (weak) and strong models
BASELINE_MODELS = ['claude-3-opus', 'gemini-1-5-pro', 'gpt-4o']
STRONG_MODELS = ['claude-3-5-haiku', 'claude-3-5-sonnet', 'claude-4-1-opus', 
                 'claude-4-sonnet', 'gemini-2-0-flash', 'gemini-2-5-flash', 
                 'gemini-2-5-pro', 'gpt-5', 'o3', 'o3-mini', 'o4-mini']

# Model display names (shorter for better visualization)
MODEL_DISPLAY_NAMES = {
    'claude-3-opus': 'Claude 3 Opus',
    'gemini-1-5-pro': 'Gemini 1.5 Pro',
    'gpt-4o': 'GPT-4o',
    'claude-3-5-haiku': 'Claude 3.5 Haiku',
    'claude-3-5-sonnet': 'Claude 3.5 Sonnet',
    'claude-4-1-opus': 'Claude 4.1 Opus',
    'claude-4-sonnet': 'Claude 4 Sonnet',
    'gemini-2-0-flash': 'Gemini 2.0 Flash',
    'gemini-2-5-flash': 'Gemini 2.5 Flash',
    'gemini-2-5-pro': 'Gemini 2.5 Pro',
    'gpt-5': 'GPT-5',
    'o3': 'O3',
    'o3-mini': 'O3 Mini',
    'o4-mini': 'O4 Mini'
}

def extract_models_from_filename(filename):
    """Extract model names from filename or batch_id."""
    # Try to extract from experiments list first
    for baseline in BASELINE_MODELS:
        for strong in STRONG_MODELS:
            if baseline in filename.lower() and strong in filename.lower():
                return baseline, strong
    return None, None

def load_experiment_results(results_dir):
    """Load all experiment results from the results directory."""
    results = defaultdict(lambda: defaultdict(list))
    
    # Look for summary JSON files
    results_path = Path(results_dir)
    
    for file_path in results_path.glob('strong_models_*_summary.json'):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Check if it has experiments list
            if 'experiments' in data and data['experiments']:
                for exp in data['experiments']:
                    if 'config' in exp and 'agents' in exp['config']:
                        agents = exp['config']['agents']
                        
                        # Try to identify models from agent names
                        agent1 = agents[0] if len(agents) > 0 else None
                        agent2 = agents[1] if len(agents) > 1 else None
                        
                        # Extract actual model names from agent IDs
                        model1 = None
                        model2 = None
                        
                        for baseline in BASELINE_MODELS:
                            if agent1 and baseline.replace('-', '_') in agent1.lower():
                                model1 = baseline
                                break
                        
                        for strong in STRONG_MODELS:
                            if agent2 and strong.replace('-', '_') in agent2.lower():
                                model2 = strong
                                break
                        
                        # Also check reverse order
                        if not model1 or not model2:
                            for strong in STRONG_MODELS:
                                if agent1 and strong.replace('-', '_') in agent1.lower():
                                    model2 = strong
                                    break
                            
                            for baseline in BASELINE_MODELS:
                                if agent2 and baseline.replace('-', '_') in agent2.lower():
                                    model1 = baseline
                                    break
                        
                        if model1 and model2:
                            # Determine winner based on final utilities
                            final_utilities = exp.get('final_utilities', {})
                            
                            if final_utilities and len(agents) >= 2:
                                # Get utilities for both agents
                                util1 = final_utilities.get(agents[0], 0)
                                util2 = final_utilities.get(agents[1], 0)
                                
                                # Determine winner based on highest utility
                                agent1_won = util1 > util2
                                
                                # Store result based on which is baseline and which is strong
                                if model1 in BASELINE_MODELS and model2 in STRONG_MODELS:
                                    baseline_won = agent1_won
                                    results[model1][model2].append(1 if baseline_won else 0)
                                elif model2 in BASELINE_MODELS and model1 in STRONG_MODELS:
                                    baseline_won = not agent1_won  # agent2 won
                                    results[model2][model1].append(1 if baseline_won else 0)
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    return results

def calculate_win_rates(results):
    """Calculate win rates from results."""
    win_rates_baseline = np.zeros((len(BASELINE_MODELS), len(STRONG_MODELS)))
    win_rates_strong = np.zeros((len(BASELINE_MODELS), len(STRONG_MODELS)))
    
    for i, baseline in enumerate(BASELINE_MODELS):
        for j, strong in enumerate(STRONG_MODELS):
            if baseline in results and strong in results[baseline]:
                wins = results[baseline][strong]
                if wins:
                    # Baseline win rate against strong
                    win_rates_baseline[i, j] = np.mean(wins)
                    # Strong win rate against baseline
                    win_rates_strong[i, j] = 1 - np.mean(wins)
                else:
                    win_rates_baseline[i, j] = np.nan
                    win_rates_strong[i, j] = np.nan
            else:
                win_rates_baseline[i, j] = np.nan
                win_rates_strong[i, j] = np.nan
    
    return win_rates_baseline, win_rates_strong

def create_heatmap(data, title, xlabel, ylabel, cmap='RdYlGn_r', filename='heatmap.png'):
    """Create and save a heatmap."""
    plt.figure(figsize=(14, 8))
    
    # Create mask for missing data
    mask = np.isnan(data)
    
    # Create heatmap
    ax = sns.heatmap(data, 
                     annot=True, 
                     fmt='.1%', 
                     cmap=cmap,
                     mask=mask,
                     cbar_kws={'label': 'Win Rate'},
                     vmin=0, 
                     vmax=1,
                     linewidths=0.5,
                     linecolor='gray')
    
    # Set labels
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Set tick labels
    ax.set_xticklabels([MODEL_DISPLAY_NAMES.get(m, m) for m in STRONG_MODELS], rotation=45, ha='right')
    ax.set_yticklabels([MODEL_DISPLAY_NAMES.get(m, m) for m in BASELINE_MODELS], rotation=0)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved heatmap to {filename}")

def main():
    """Main function to generate heatmaps."""
    # Load results
    print("Loading experiment results...")
    results_dir = '/root/bargain/experiments/results'
    results = load_experiment_results(results_dir)
    
    # Print summary statistics
    print("\nData summary:")
    total_experiments = 0
    for baseline in results:
        for strong in results[baseline]:
            count = len(results[baseline][strong])
            if count > 0:
                print(f"  {baseline} vs {strong}: {count} experiments")
                total_experiments += count
    
    if total_experiments == 0:
        print("\nNo experiments found with the expected model pairs.")
        print("Please check that the experiment results contain the expected model names.")
        return
    
    print(f"\nTotal experiments found: {total_experiments}")
    
    # Calculate win rates
    print("\nCalculating win rates...")
    win_rates_baseline, win_rates_strong = calculate_win_rates(results)
    
    # Create baseline agent win rate heatmap
    print("\nCreating baseline agent win rate heatmap...")
    create_heatmap(win_rates_baseline,
                   title='Baseline Agent Win Rate Against Strong Agent',
                   xlabel='Strong Models',
                   ylabel='Baseline Models',
                   cmap='RdYlGn',  # Red for low win rate, green for high
                   filename='win_rate_heatmap_baseline.png')
    
    # Create strong agent win rate heatmap
    print("\nCreating strong agent win rate heatmap...")
    create_heatmap(win_rates_strong,
                   title='Strong Agent Win Rate Against Baseline Agent',
                   xlabel='Strong Models',
                   ylabel='Baseline Models (Opponent)',
                   cmap='RdYlGn_r',  # Green for high win rate, red for low
                   filename='win_rate_heatmap_strong.png')
    
    # Print win rate statistics
    print("\n=== Win Rate Statistics ===")
    print("\nBaseline Win Rates Against Strong Models:")
    for i, baseline in enumerate(BASELINE_MODELS):
        print(f"\n{MODEL_DISPLAY_NAMES.get(baseline, baseline)}:")
        for j, strong in enumerate(STRONG_MODELS):
            if not np.isnan(win_rates_baseline[i, j]):
                print(f"  vs {MODEL_DISPLAY_NAMES.get(strong, strong)}: {win_rates_baseline[i, j]:.1%}")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    main()
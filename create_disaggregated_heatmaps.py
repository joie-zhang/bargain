#!/usr/bin/env python3
"""
Create disaggregated heatmaps for win rates between baseline models and strong models,
separated by competition level. Also checks for bidirectional experiments.
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
    'claude-3-opus': 'C3 Opus',
    'gemini-1-5-pro': 'G1.5 Pro',
    'gpt-4o': 'GPT-4o',
    'claude-3-5-haiku': 'C3.5 Haiku',
    'claude-3-5-sonnet': 'C3.5 Sonnet',
    'claude-4-1-opus': 'C4.1 Opus',
    'claude-4-sonnet': 'C4 Sonnet',
    'gemini-2-0-flash': 'G2.0 Flash',
    'gemini-2-5-flash': 'G2.5 Flash',
    'gemini-2-5-pro': 'G2.5 Pro',
    'gpt-5': 'GPT-5',
    'o3': 'O3',
    'o3-mini': 'O3 Mini',
    'o4-mini': 'O4 Mini'
}

# Competition level buckets
COMPETITION_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]

def categorize_competition_level(comp_level):
    """Categorize competition level into buckets."""
    # Handle None or missing competition levels
    if comp_level is None:
        return None
    
    # Round to nearest standard competition level
    return min(COMPETITION_LEVELS, key=lambda x: abs(x - comp_level))

def load_experiment_results(results_dir):
    """Load all experiment results from the results directory, categorized by competition level."""
    results_by_competition = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    bidirectional_pairs = defaultdict(lambda: defaultdict(set))
    
    # Look for summary JSON files
    results_path = Path(results_dir)
    
    for file_path in results_path.glob('**/*_summary.json'):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Check if it has experiments list
            if 'experiments' in data and data['experiments']:
                for exp in data['experiments']:
                    if 'config' in exp and 'agents' in exp['config']:
                        agents = exp['config']['agents']
                        
                        # Get competition level
                        comp_level = exp['config'].get('competition_level', None)
                        comp_bucket = categorize_competition_level(comp_level)
                        
                        if comp_bucket is None:
                            continue
                        
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
                            # Track which pairs of models have been seen in both directions
                            if model1 in BASELINE_MODELS and model2 in STRONG_MODELS:
                                bidirectional_pairs[comp_bucket][(model1, model2)].add('baseline_first')
                            elif model2 in BASELINE_MODELS and model1 in STRONG_MODELS:
                                bidirectional_pairs[comp_bucket][(model2, model1)].add('strong_first')
                            
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
                                    results_by_competition[comp_bucket][model1][model2].append(1 if baseline_won else 0)
                                elif model2 in BASELINE_MODELS and model1 in STRONG_MODELS:
                                    baseline_won = not agent1_won  # agent2 won
                                    results_by_competition[comp_bucket][model2][model1].append(1 if baseline_won else 0)
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    return results_by_competition, bidirectional_pairs

def calculate_win_rates(results):
    """Calculate win rates from results."""
    win_rates_baseline = np.zeros((len(BASELINE_MODELS), len(STRONG_MODELS)))
    sample_counts = np.zeros((len(BASELINE_MODELS), len(STRONG_MODELS)))
    
    for i, baseline in enumerate(BASELINE_MODELS):
        for j, strong in enumerate(STRONG_MODELS):
            if baseline in results and strong in results[baseline]:
                wins = results[baseline][strong]
                if wins:
                    # Baseline win rate against strong
                    win_rates_baseline[i, j] = np.mean(wins)
                    sample_counts[i, j] = len(wins)
                else:
                    win_rates_baseline[i, j] = np.nan
                    sample_counts[i, j] = 0
            else:
                win_rates_baseline[i, j] = np.nan
                sample_counts[i, j] = 0
    
    return win_rates_baseline, sample_counts

def create_disaggregated_heatmaps(results_by_competition, bidirectional_pairs, filename='disaggregated_heatmaps.png'):
    """Create a figure with heatmaps for each competition level."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Baseline Agent Win Rates by Competition Level', fontsize=16, fontweight='bold')
    
    # Flatten axes array for easier iteration
    axes_flat = axes.flatten()
    
    # Create a heatmap for each competition level
    for idx, comp_level in enumerate(COMPETITION_LEVELS):
        ax = axes_flat[idx]
        
        # Calculate win rates for this competition level
        win_rates, sample_counts = calculate_win_rates(results_by_competition[comp_level])
        
        # Create mask for missing data
        mask = np.isnan(win_rates)
        
        # Count bidirectional experiments
        bidirectional_count = 0
        unidirectional_count = 0
        for (baseline, strong), directions in bidirectional_pairs[comp_level].items():
            if len(directions) == 2:  # Both directions present
                bidirectional_count += 1
            else:
                unidirectional_count += 1
        
        # Create heatmap with sample counts as annotations
        annot_data = np.empty_like(win_rates, dtype=object)
        for i in range(len(BASELINE_MODELS)):
            for j in range(len(STRONG_MODELS)):
                if not np.isnan(win_rates[i, j]):
                    annot_data[i, j] = f'{win_rates[i, j]:.1%}\n(n={int(sample_counts[i, j])})'
                else:
                    annot_data[i, j] = ''
        
        # Create heatmap
        sns.heatmap(win_rates, 
                   annot=annot_data,
                   fmt='',
                   cmap='RdYlGn',
                   mask=mask,
                   cbar_kws={'label': 'Win Rate'},
                   vmin=0, 
                   vmax=1,
                   linewidths=0.5,
                   linecolor='gray',
                   ax=ax,
                   cbar=(idx == 0))  # Only show colorbar for first plot
        
        # Set labels
        ax.set_title(f'Competition Level: {comp_level}\n({bidirectional_count} bidirectional, {unidirectional_count} unidirectional pairs)', 
                    fontsize=12)
        ax.set_xlabel('Strong Models', fontsize=10)
        ax.set_ylabel('Baseline Models', fontsize=10)
        
        # Set tick labels
        ax.set_xticklabels([MODEL_DISPLAY_NAMES.get(m, m) for m in STRONG_MODELS], rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels([MODEL_DISPLAY_NAMES.get(m, m) for m in BASELINE_MODELS], rotation=0, fontsize=9)
    
    # Remove the 6th subplot (we only have 5 competition levels)
    fig.delaxes(axes_flat[5])
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved disaggregated heatmap to {filename}")

def analyze_bidirectionality(bidirectional_pairs):
    """Analyze and report on bidirectional experiment coverage."""
    print("\n=== Bidirectional Analysis ===")
    
    for comp_level in COMPETITION_LEVELS:
        pairs = bidirectional_pairs[comp_level]
        total_pairs = len(pairs)
        bidirectional_count = sum(1 for directions in pairs.values() if len(directions) == 2)
        
        print(f"\nCompetition Level {comp_level}:")
        print(f"  Total unique model pairs: {total_pairs}")
        print(f"  Bidirectional pairs: {bidirectional_count}")
        print(f"  Unidirectional pairs: {total_pairs - bidirectional_count}")
        
        if total_pairs > 0:
            print(f"  Bidirectional coverage: {bidirectional_count/total_pairs:.1%}")
        
        # List which pairs are missing bidirectional data
        missing_bidirectional = []
        for (baseline, strong), directions in pairs.items():
            if len(directions) == 1:
                missing_direction = 'strong_first' if 'baseline_first' in directions else 'baseline_first'
                missing_bidirectional.append((baseline, strong, missing_direction))
        
        if missing_bidirectional and len(missing_bidirectional) <= 5:
            print("  Missing bidirectional experiments (showing first 5):")
            for baseline, strong, direction in missing_bidirectional[:5]:
                print(f"    - {MODEL_DISPLAY_NAMES.get(baseline, baseline)} vs {MODEL_DISPLAY_NAMES.get(strong, strong)} ({direction})")

def main():
    """Main function to generate disaggregated heatmaps."""
    # Load results
    print("Loading experiment results...")
    results_dir = '/root/bargain/experiments/results'
    results_by_competition, bidirectional_pairs = load_experiment_results(results_dir)
    
    # Print summary statistics
    print("\n=== Data Summary by Competition Level ===")
    total_experiments = 0
    for comp_level in COMPETITION_LEVELS:
        comp_total = 0
        print(f"\nCompetition Level {comp_level}:")
        for baseline in results_by_competition[comp_level]:
            for strong in results_by_competition[comp_level][baseline]:
                count = len(results_by_competition[comp_level][baseline][strong])
                if count > 0:
                    print(f"  {MODEL_DISPLAY_NAMES.get(baseline, baseline)} vs {MODEL_DISPLAY_NAMES.get(strong, strong)}: {count} experiments")
                    comp_total += count
        total_experiments += comp_total
        print(f"  Subtotal: {comp_total} experiments")
    
    if total_experiments == 0:
        print("\nNo experiments found with the expected model pairs.")
        print("Please check that the experiment results contain the expected model names.")
        return
    
    print(f"\nTotal experiments found: {total_experiments}")
    
    # Analyze bidirectionality
    analyze_bidirectionality(bidirectional_pairs)
    
    # Create disaggregated heatmaps
    print("\n\nCreating disaggregated heatmaps...")
    create_disaggregated_heatmaps(results_by_competition, bidirectional_pairs,
                                  filename='win_rate_heatmaps_by_competition.png')
    
    # Also create separate heatmaps showing strong agent win rates
    print("\nCreating strong agent win rate heatmaps...")
    # Invert the win rates for strong agent perspective
    results_strong_perspective = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for comp_level in results_by_competition:
        for baseline in results_by_competition[comp_level]:
            for strong in results_by_competition[comp_level][baseline]:
                # Invert the wins (0 becomes 1, 1 becomes 0)
                wins = results_by_competition[comp_level][baseline][strong]
                results_strong_perspective[comp_level][baseline][strong] = [1 - w for w in wins]
    
    # Create the strong agent heatmaps
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Strong Agent Win Rates by Competition Level', fontsize=16, fontweight='bold')
    axes_flat = axes.flatten()
    
    for idx, comp_level in enumerate(COMPETITION_LEVELS):
        ax = axes_flat[idx]
        win_rates, sample_counts = calculate_win_rates(results_strong_perspective[comp_level])
        mask = np.isnan(win_rates)
        
        annot_data = np.empty_like(win_rates, dtype=object)
        for i in range(len(BASELINE_MODELS)):
            for j in range(len(STRONG_MODELS)):
                if not np.isnan(win_rates[i, j]):
                    annot_data[i, j] = f'{win_rates[i, j]:.1%}\n(n={int(sample_counts[i, j])})'
                else:
                    annot_data[i, j] = ''
        
        sns.heatmap(win_rates, 
                   annot=annot_data,
                   fmt='',
                   cmap='RdYlGn_r',  # Reversed colormap for strong agent perspective
                   mask=mask,
                   cbar_kws={'label': 'Win Rate'},
                   vmin=0, 
                   vmax=1,
                   linewidths=0.5,
                   linecolor='gray',
                   ax=ax,
                   cbar=(idx == 0))
        
        ax.set_title(f'Competition Level: {comp_level}', fontsize=12)
        ax.set_xlabel('Strong Models', fontsize=10)
        ax.set_ylabel('Baseline Models (Opponent)', fontsize=10)
        ax.set_xticklabels([MODEL_DISPLAY_NAMES.get(m, m) for m in STRONG_MODELS], rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels([MODEL_DISPLAY_NAMES.get(m, m) for m in BASELINE_MODELS], rotation=0, fontsize=9)
    
    fig.delaxes(axes_flat[5])
    plt.tight_layout()
    plt.savefig('strong_agent_heatmaps_by_competition.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved strong agent heatmaps to strong_agent_heatmaps_by_competition.png")

if __name__ == "__main__":
    main()
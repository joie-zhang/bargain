#!/usr/bin/env python3
"""
Create difference heatmaps comparing two result directories.
Shows the difference between results and results_current directories.
Y-axis: Competition level (0 to 1)
X-axis: Models ordered by MMLU-Pro score (low to high)
Values: Utility differences (results - results_current)
"""

import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd

# MMLU-Pro scores from the provided data
MMLU_PRO_SCORES = {
    # Strong models we're looking for
    'claude-3-5-haiku': 64.1,
    'claude-3-5-sonnet': 78.4,  # Using "claude-3-5-sonnet-20241022" -> "Claude 3.5 Sonnet Latest"
    'claude-4-1-opus': 87.8,
    'claude-4-sonnet': 79.4,
    'gemma-3-27b': 67.5,  # from HF leaderboard
    'gemini-2-0-flash': 77.4,
    'gemini-2-5-pro': 84.1,  # Using "Gemini 2.5 Pro Exp"
    'gpt-4o-mini': 62.7,
    'gpt-4o-2024-11-20': 69.1,
    'o1': 83.5,
    'o3': 85.6,
    'gpt-5-mini': 83.7,
    'gpt-5-nano': 78.0,
    
    # Baseline models
    'gpt-4o-2024-05-13': 72.55,  # from HF leaderboard
    'gemini-1-5-pro': 75.3,  # Using "Gemini 1.5 Pro (002)"
    'claude-3-opus': 68.45,  # from HF leaderboard
}

# Define the models we want to include
STRONG_MODELS_REQUESTED = [
    'claude-3-5-haiku', 'claude-3-5-sonnet', 'claude-4-1-opus', 'claude-4-sonnet',
    'gemini-2-0-flash', 'gemini-2-5-pro',
    'gpt-4o-2024-11-20', 'o1', 'o3', 'gpt-5-nano', 'gpt-5-mini'
]

BASELINE_MODELS = ['gpt-4o-2024-05-13', 'gemini-1-5-pro', 'claude-3-opus']

# Model display names
MODEL_DISPLAY_NAMES = {
    'claude-3-5-haiku': 'Claude 3.5\nHaiku',
    'claude-3-5-sonnet': 'Claude 3.5\nSonnet',
    'claude-4-1-opus': 'Claude 4.1\nOpus',
    'claude-4-sonnet': 'Claude 4\nSonnet',
    'gemma-3-27b': 'Gemma 3\n27B',
    'gemini-2-0-flash': 'Gemini 2.0\nFlash',
    'gemini-2-5-pro': 'Gemini 2.5\nPro',
    'gpt-4o-mini': 'GPT-4o\nMini',
    'gpt-4o-2024-11-20': 'GPT-4o\n(Nov 2024)',
    'gpt-5-nano': 'GPT-5\nNano',
    'gpt-5-mini': 'GPT-5\nMini',
    'o1': 'O1',
    'o3': 'O3',
    'gpt-4o-2024-05-13': 'GPT-4o (May 2024)',
    'gemini-1-5-pro': 'Gemini 1.5 Pro',
    'claude-3-opus': 'Claude 3 Opus'
}

# Competition levels for y-axis
COMPETITION_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]

def extract_model_name(agent_name):
    """Extract model name from agent ID like 'claude_3_opus_1' -> 'claude-3-opus'"""
    if not agent_name:
        return None
    # Remove the trailing agent number (e.g., "_1", "_2")
    parts = agent_name.split('_')
    if len(parts) > 1 and parts[-1].isdigit():
        parts = parts[:-1]
    # Join with hyphens instead of underscores
    return '-'.join(parts)

def normalize_model_name(model_name):
    """Normalize model names to match our target set"""
    # Handle GPT-4o variants
    if model_name == 'gpt-4o-2024-05-13' or model_name == 'gpt-4o' or model_name == 'gpt-4o-may':
        return 'gpt-4o-2024-05-13'  # Baseline model (May 2024)
    elif model_name == 'gpt-4o-2024-11-20' or model_name == 'gpt-4o-nov' or model_name == 'gpt-4o-latest':
        return 'gpt-4o-2024-11-20'  # Strong model (Nov 2024)
    # Handle Claude variants
    elif model_name == 'claude-3-5-sonnet-20241022' or model_name == 'claude-3-5-sonnet':
        return 'claude-3-5-sonnet'
    elif model_name == 'claude-3-opus' or model_name == 'claude-3-opus-20240229':
        return 'claude-3-opus'
    # Handle Gemini variants
    elif model_name == 'gemini-1-5-pro-002' or model_name == 'gemini-1-5-pro':
        return 'gemini-1-5-pro'
    elif model_name == 'gemini-2-0-flash-001' or model_name == 'gemini-2-0-flash':
        return 'gemini-2-0-flash'
    elif model_name == 'gemini-2-5-pro-exp' or model_name == 'gemini-2-5-pro':
        return 'gemini-2-5-pro'
    return model_name

def categorize_competition_level(comp_level):
    """Categorize competition level into buckets."""
    if comp_level is None:
        return None
    # Round to nearest standard competition level
    return min(COMPETITION_LEVELS, key=lambda x: abs(x - comp_level))

def load_experiment_results(results_dir):
    """Load all experiment results from the results directory, organized by competition level and model pairs."""
    results_by_competition = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Look for summary JSON files
    results_path = Path(results_dir)
    
    print(f"Scanning for experiment files in {results_dir}...")
    processed_files = 0
    
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
                        
                        # Extract model names
                        agent1 = agents[0] if len(agents) > 0 else None
                        agent2 = agents[1] if len(agents) > 1 else None
                        
                        agent1_model = normalize_model_name(extract_model_name(agent1))
                        agent2_model = normalize_model_name(extract_model_name(agent2))
                        
                        # Check if we have a baseline vs strong model pair
                        baseline_model = None
                        strong_model = None
                        baseline_agent = None
                        strong_agent = None
                        
                        if agent1_model in BASELINE_MODELS and agent2_model in STRONG_MODELS_REQUESTED:
                            baseline_model, strong_model = agent1_model, agent2_model
                            baseline_agent, strong_agent = agent1, agent2
                        elif agent2_model in BASELINE_MODELS and agent1_model in STRONG_MODELS_REQUESTED:
                            baseline_model, strong_model = agent2_model, agent1_model
                            baseline_agent, strong_agent = agent2, agent1
                        
                        if baseline_model and strong_model:
                            # Get final utilities
                            final_utilities = exp.get('final_utilities', {})
                            
                            if final_utilities and baseline_agent and strong_agent:
                                baseline_utility = final_utilities.get(baseline_agent, 0)
                                strong_utility = final_utilities.get(strong_agent, 0)
                                
                                # Store the utility difference (strong - baseline)
                                utility_diff = strong_utility - baseline_utility
                                
                                results_by_competition[comp_bucket][baseline_model][strong_model].append({
                                    'utility_diff': utility_diff,
                                    'baseline_utility': baseline_utility,
                                    'strong_utility': strong_utility
                                })
                
                processed_files += 1
                if processed_files % 50 == 0:
                    print(f"Processed {processed_files} files...")
                    
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    print(f"Finished processing {processed_files} files in {results_dir}.")
    return results_by_competition

def get_common_models(results1, results2):
    """Get the list of strong models that have data in BOTH result sets."""
    models_with_data1 = set()
    models_with_data2 = set()
    
    # Get models from first result set
    for comp_level in results1:
        for baseline in results1[comp_level]:
            for strong in results1[comp_level][baseline]:
                if results1[comp_level][baseline][strong]:  # has data
                    models_with_data1.add(strong)
    
    # Get models from second result set
    for comp_level in results2:
        for baseline in results2[comp_level]:
            for strong in results2[comp_level][baseline]:
                if results2[comp_level][baseline][strong]:  # has data
                    models_with_data2.add(strong)
    
    # Get common models
    common_models = models_with_data1.intersection(models_with_data2)
    
    # Filter to only include models we have MMLU scores for and sort by score
    models_with_scores = []
    for model in common_models:
        if model in MMLU_PRO_SCORES and MMLU_PRO_SCORES[model] is not None:
            models_with_scores.append((model, MMLU_PRO_SCORES[model]))
    
    # Sort by MMLU-Pro score (ascending - worst to best)
    models_with_scores.sort(key=lambda x: x[1])
    
    return [model for model, score in models_with_scores]

def create_difference_heatmap_for_baseline(results1, results2, baseline_model, ordered_strong_models):
    """Create a difference heatmap for a specific baseline model (results1 - results2)."""
    
    # Create data matrix: rows = competition levels, cols = strong models
    # Reverse the order of competition levels so 0 is at bottom, 1 at top
    reversed_competition_levels = list(reversed(COMPETITION_LEVELS))
    data = np.full((len(reversed_competition_levels), len(ordered_strong_models)), np.nan)
    
    for i, comp_level in enumerate(reversed_competition_levels):
        for j, strong_model in enumerate(ordered_strong_models):
            # Get values from both result sets
            val1 = None
            val2 = None
            
            # Get value from results1
            if (comp_level in results1 and 
                baseline_model in results1[comp_level] and
                strong_model in results1[comp_level][baseline_model]):
                experiments1 = results1[comp_level][baseline_model][strong_model]
                if experiments1:
                    values1 = [exp['utility_diff'] for exp in experiments1]
                    val1 = np.mean(values1)
            
            # Get value from results2
            if (comp_level in results2 and 
                baseline_model in results2[comp_level] and
                strong_model in results2[comp_level][baseline_model]):
                experiments2 = results2[comp_level][baseline_model][strong_model]
                if experiments2:
                    values2 = [exp['utility_diff'] for exp in experiments2]
                    val2 = np.mean(values2)
            
            # Calculate difference if both values exist
            if val1 is not None and val2 is not None:
                data[i, j] = val1 - val2
    
    return data

def plot_difference_heatmap(results1, results2, ordered_strong_models, baseline_model):
    """Plot difference heatmap for a specific baseline model."""
    
    # Get difference data for this baseline
    data = create_difference_heatmap_for_baseline(results1, results2, baseline_model, ordered_strong_models)
    
    # Create mask for missing data
    mask = np.isnan(data)
    
    # Create figure
    fig = plt.figure(figsize=(16, 7))
    ax = fig.add_axes([0.125, 0.35, 0.65, 0.55])
    
    # Create MMLU-Pro score data for color bar
    mmlu_scores = [MMLU_PRO_SCORES[model] for model in ordered_strong_models]
    
    # Create heatmap
    im = sns.heatmap(data, 
                   annot=True, 
                   fmt='.1f',
                   cmap='RdBu_r',
                   mask=mask,
                   cbar_kws={'label': 'Utility Difference Change (results - results_current)'},
                   annot_kws={'fontsize': 11},
                   vmin=-50,  # Adjusted for difference values
                   vmax=50,
                   center=0,
                   linewidths=0.5,
                   linecolor='gray',
                   ax=ax,
                   square=False)
    
    # Set colorbar label font size
    cbar = ax.collections[0].colorbar
    if cbar:
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label(cbar.ax.get_ylabel(), fontsize=14)
    
    # Set labels
    ax.set_xlabel('Adversary Models Ordered by MMLU-Pro Performance →', fontsize=16, fontweight='bold')
    ax.set_xticklabels([MODEL_DISPLAY_NAMES[model] for model in ordered_strong_models], 
                      rotation=0, ha='center', fontsize=14)
    ax.set_ylabel('Competition Level', fontsize=16, fontweight='bold')
    ax.set_title(f'{MODEL_DISPLAY_NAMES[baseline_model]}: Difference (results - results_current)', 
                fontsize=14, fontweight='bold', pad=15)
    
    # Reverse the labels to match the reversed data (0 at bottom, 1 at top)
    ax.set_yticklabels([f'{level:.2f}' for level in reversed(COMPETITION_LEVELS)], 
                      rotation=0, fontsize=14)
    
    # Add MMLU-Pro performance bar
    heatmap_left = 0.125
    heatmap_width = 0.65
    
    mmlu_bar_height = 0.03
    mmlu_ax = fig.add_axes([heatmap_left, 0.10, heatmap_width, mmlu_bar_height])
    
    mmlu_data = np.array(mmlu_scores).reshape(1, -1)
    mmlu_im = mmlu_ax.imshow(mmlu_data, cmap='plasma', aspect='auto', extent=[0, len(ordered_strong_models), 0, 1])
    
    mmlu_ax.set_xticks([])
    mmlu_ax.set_yticks([])
    
    # Create a separate axis for the MMLU score labels
    mmlu_labels_ax = fig.add_axes([heatmap_left, 0.05, heatmap_width, 0.03])
    mmlu_labels_ax.set_xlim(0, len(ordered_strong_models))
    mmlu_labels_ax.set_ylim(0, 1)
    
    for i, score in enumerate(mmlu_scores):
        mmlu_labels_ax.text(i + 0.5, 0.5, f'{score}%', ha='center', va='center', 
                           rotation=0, fontsize=14, fontweight='bold')
    
    mmlu_labels_ax.set_xticks([])
    mmlu_labels_ax.set_yticks([])
    mmlu_labels_ax.axis('off')
    
    # Save figure
    baseline_name = baseline_model.replace('-', '_')
    filename = f'mmlu_ordered_{baseline_name}_difference_heatmap.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved difference heatmap to {filename}")
    plt.close()

def plot_combined_difference_heatmaps(results1, results2, ordered_strong_models):
    """Plot combined difference heatmaps for all baseline models."""
    
    # Get baselines that have data in both result sets
    baselines_with_data = []
    for baseline in BASELINE_MODELS:
        has_data_both = False
        # Check if baseline has data in both result sets
        for comp_level in COMPETITION_LEVELS:
            if (comp_level in results1 and baseline in results1[comp_level] and
                comp_level in results2 and baseline in results2[comp_level]):
                # Check if there's at least one common strong model
                for strong in ordered_strong_models:
                    if (strong in results1[comp_level][baseline] and 
                        results1[comp_level][baseline][strong] and
                        strong in results2[comp_level][baseline] and 
                        results2[comp_level][baseline][strong]):
                        has_data_both = True
                        break
                if has_data_both:
                    break
        if has_data_both:
            baselines_with_data.append(baseline)
    
    if not baselines_with_data:
        print("No baseline models found with data in both result sets!")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(len(baselines_with_data), 1, figsize=(16, 5*len(baselines_with_data)))
    if len(baselines_with_data) == 1:
        axes = [axes]
    
    # Create MMLU-Pro score data
    mmlu_scores = [MMLU_PRO_SCORES[model] for model in ordered_strong_models]
    
    for idx, baseline in enumerate(baselines_with_data):
        ax = axes[idx]
        
        # Get difference data for this baseline
        data = create_difference_heatmap_for_baseline(results1, results2, baseline, ordered_strong_models)
        
        # Create mask for missing data
        mask = np.isnan(data)
        
        # Create heatmap
        im = sns.heatmap(data, 
                       annot=True, 
                       fmt='.1f',
                       cmap='RdBu_r',
                       mask=mask,
                       cbar_kws={'label': 'Utility Difference Change (results - results_current)'},
                       annot_kws={'fontsize': 11},
                       vmin=-50,
                       vmax=50,
                       center=0,
                       linewidths=0.5,
                       linecolor='gray',
                       ax=ax,
                       square=False)
        
        # Set colorbar label font size
        cbar = ax.collections[0].colorbar
        if cbar:
            cbar.ax.tick_params(labelsize=12)
            cbar.set_label(cbar.ax.get_ylabel(), fontsize=14)
        
        # Set labels
        if idx == len(baselines_with_data) - 1:
            ax.set_xlabel('Adversary Models Ordered by MMLU-Pro Performance →', fontsize=16, fontweight='bold')
            ax.set_xticklabels([MODEL_DISPLAY_NAMES[model] for model in ordered_strong_models], 
                              rotation=0, ha='center', fontsize=14)
        else:
            ax.set_xlabel('')
            ax.set_xticklabels([])
        
        ax.set_ylabel('Competition Level', fontsize=16, fontweight='bold')
        ax.set_title(f'{MODEL_DISPLAY_NAMES[baseline]}: Difference (results - results_current)', 
                    fontsize=14, fontweight='bold', pad=15)
        
        # Reverse the labels to match the reversed data
        ax.set_yticklabels([f'{level:.2f}' for level in reversed(COMPETITION_LEVELS)], 
                          rotation=0, fontsize=14)
    
    plt.tight_layout()
    
    # Save figure
    filename = 'mmlu_ordered_difference_heatmaps_combined.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved combined difference heatmap to {filename}")

def print_comparison_statistics(results1, results2, ordered_strong_models):
    """Print statistics comparing the two result sets."""
    print("\n=== Comparison Statistics ===")
    print(f"Common models found: {len(ordered_strong_models)}")
    print(f"Models: {', '.join([MODEL_DISPLAY_NAMES[m] for m in ordered_strong_models])}")
    
    total_comparisons = 0
    total_improved = 0
    total_worsened = 0
    total_unchanged = 0
    
    for comp_level in COMPETITION_LEVELS:
        for baseline in BASELINE_MODELS:
            for strong in ordered_strong_models:
                val1 = None
                val2 = None
                
                # Get value from results1
                if (comp_level in results1 and 
                    baseline in results1[comp_level] and
                    strong in results1[comp_level][baseline]):
                    experiments1 = results1[comp_level][baseline][strong]
                    if experiments1:
                        values1 = [exp['utility_diff'] for exp in experiments1]
                        val1 = np.mean(values1)
                
                # Get value from results2
                if (comp_level in results2 and 
                    baseline in results2[comp_level] and
                    strong in results2[comp_level][baseline]):
                    experiments2 = results2[comp_level][baseline][strong]
                    if experiments2:
                        values2 = [exp['utility_diff'] for exp in experiments2]
                        val2 = np.mean(values2)
                
                if val1 is not None and val2 is not None:
                    diff = val1 - val2
                    total_comparisons += 1
                    if diff > 1.0:  # Improved by more than 1 utility point
                        total_improved += 1
                    elif diff < -1.0:  # Worsened by more than 1 utility point
                        total_worsened += 1
                    else:
                        total_unchanged += 1
    
    print(f"\nTotal valid comparisons: {total_comparisons}")
    print(f"Improved (>1 point): {total_improved} ({100*total_improved/max(1,total_comparisons):.1f}%)")
    print(f"Worsened (<-1 point): {total_worsened} ({100*total_worsened/max(1,total_comparisons):.1f}%)")
    print(f"Unchanged (±1 point): {total_unchanged} ({100*total_unchanged/max(1,total_comparisons):.1f}%)")

def main():
    """Main function to generate difference heatmaps between two result directories."""
    print("Loading experiment results from both directories...")
    
    # Load results from both directories
    results_dir1 = '/root/bargain/experiments/results'
    results_dir2 = '/root/bargain/experiments/results_current'
    
    print(f"Loading from {results_dir1}...")
    results1 = load_experiment_results(results_dir1)
    
    print(f"Loading from {results_dir2}...")
    results2 = load_experiment_results(results_dir2)
    
    # Get common models between both result sets
    print("\n=== Finding Common Models ===")
    ordered_strong_models = get_common_models(results1, results2)
    
    if not ordered_strong_models:
        print("No common strong models found between the two result sets!")
        return
    
    print(f"Found {len(ordered_strong_models)} common strong models with MMLU-Pro scores.")
    print("Models (ordered by MMLU-Pro score):")
    for model in ordered_strong_models:
        print(f"  - {MODEL_DISPLAY_NAMES[model]} ({MMLU_PRO_SCORES[model]}%)")
    
    # Print comparison statistics
    print_comparison_statistics(results1, results2, ordered_strong_models)
    
    # Get baselines that have data in both result sets
    baselines_with_data = []
    for baseline in BASELINE_MODELS:
        has_data_both = False
        for comp_level in COMPETITION_LEVELS:
            if (comp_level in results1 and baseline in results1[comp_level] and
                comp_level in results2 and baseline in results2[comp_level]):
                for strong in ordered_strong_models:
                    if (strong in results1[comp_level][baseline] and 
                        results1[comp_level][baseline][strong] and
                        strong in results2[comp_level][baseline] and 
                        results2[comp_level][baseline][strong]):
                        has_data_both = True
                        break
                if has_data_both:
                    break
        if has_data_both:
            baselines_with_data.append(baseline)
    
    if not baselines_with_data:
        print("No baseline models with data in both result sets!")
        return
    
    # Generate individual difference heatmaps for each baseline
    print("\n=== Creating Individual Difference Heatmaps ===")
    for baseline in baselines_with_data:
        print(f"\nCreating difference heatmap for {MODEL_DISPLAY_NAMES[baseline]}...")
        plot_difference_heatmap(results1, results2, ordered_strong_models, baseline)
    
    # Generate combined difference heatmap
    print("\n=== Creating Combined Difference Heatmap ===")
    plot_combined_difference_heatmaps(results1, results2, ordered_strong_models)
    
    print("\n=== Complete ===")
    print("Difference heatmaps show (results - results_current):")
    print("  - Positive values (blue): results directory has higher utility difference")
    print("  - Negative values (red): results_current directory has higher utility difference")
    print("  - NaN (white): Missing data in one or both directories")

if __name__ == "__main__":
    main()
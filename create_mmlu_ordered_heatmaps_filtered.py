#!/usr/bin/env python3
"""
Create heatmaps with models ordered by MMLU-Pro performance.
Y-axis: Competition level (0 to 1)
X-axis: Models ordered by MMLU-Pro score (low to high)
Values: Utility differences (Strong model - Baseline model)
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
    
    # Baseline models
    'gpt-4o-2024-05-13': 72.55,  # from HF leaderboard
    'gemini-1-5-pro': 75.3,  # Using "Gemini 1.5 Pro (002)"
    'claude-3-opus': 68.45,  # from HF leaderboard
}

# Define the models we want to include
# STRONG_MODELS_REQUESTED = [
#     'claude-3-5-haiku', 'claude-3-5-sonnet', 'claude-4-1-opus', 'claude-4-sonnet',
#     'gemma-3-27b', 'gemini-2-0-flash', 'gemini-2-5-pro',
#     'gpt-4o-mini', 'gpt-4o-2024-11-20', 'o1', 'o3'
# ]
STRONG_MODELS_REQUESTED = [
    'claude-3-5-haiku', 'claude-3-5-sonnet', 'claude-4-1-opus', 'claude-4-sonnet',
    'gemini-2-0-flash', 'gemini-2-5-pro',
    'gpt-4o-2024-11-20', 'o1', 'o3'
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
    """Load all experiment results from the results directory, organized by competition level and model pairs.
    This version filters out experiments where consensus was reached on round 10 (final round)."""
    results_by_competition = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Look for summary JSON files
    results_path = Path(results_dir)
    
    print("Scanning for experiment files...")
    processed_files = 0
    discarded_final_round = 0
    
    for file_path in results_path.glob('**/*_summary.json'):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Check if it has experiments list
            if 'experiments' in data and data['experiments']:
                for exp in data['experiments']:
                    if 'config' in exp and 'agents' in exp['config']:
                        agents = exp['config']['agents']
                        
                        # Check if consensus was reached on round 10 (final round) and skip if so
                        consensus_reached = exp.get('consensus_reached', False)
                        final_round = exp.get('final_round', None)
                        if consensus_reached and final_round == 10:
                            discarded_final_round += 1
                            continue
                        
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
    
    print(f"Finished processing {processed_files} files.")
    print(f"Discarded {discarded_final_round} experiments where consensus was reached on final round (round 10).")
    return results_by_competition

def get_models_with_data(results_by_competition):
    """Get the list of strong models that actually have data, ordered by MMLU-Pro score."""
    models_with_data = set()
    for comp_level in results_by_competition:
        for baseline in results_by_competition[comp_level]:
            for strong in results_by_competition[comp_level][baseline]:
                if results_by_competition[comp_level][baseline][strong]:  # has data
                    models_with_data.add(strong)
    
    # Filter to only include models we have MMLU scores for and sort by score
    models_with_scores = []
    for model in models_with_data:
        if model in MMLU_PRO_SCORES and MMLU_PRO_SCORES[model] is not None:
            models_with_scores.append((model, MMLU_PRO_SCORES[model]))
    
    # Sort by MMLU-Pro score (ascending - worst to best)
    models_with_scores.sort(key=lambda x: x[1])
    
    return [model for model, score in models_with_scores]

def plot_individual_heatmap(results_by_competition, baseline_model, ordered_strong_models, plot_mode='diff'):
    """Plot a single heatmap for a specific baseline model with MMLU-Pro bar."""
    
    # Get data for this baseline
    data = create_heatmap_for_baseline(results_by_competition, baseline_model, ordered_strong_models, plot_mode)
    
    # Create mask for missing data
    mask = np.isnan(data)
    
    # Create figure without subplot (we'll position it manually)
    fig = plt.figure(figsize=(16, 7))
    # Manually create the axis with specific position [left, bottom, width, height]
    # This gives us control over the exact placement
    ax = fig.add_axes([0.125, 0.35, 0.65, 0.55])  # Move heatmap up and make it smaller vertically
    
    # Create MMLU-Pro score data for color bar
    mmlu_scores = [MMLU_PRO_SCORES[model] for model in ordered_strong_models]
    
    # Create heatmap
    if plot_mode == 'diff':
        im = sns.heatmap(data, 
                       annot=True, 
                       fmt='.1f',
                       cmap='RdBu_r',
                       mask=mask,
                       cbar_kws={'label': 'Utility Difference (Adversary - Baseline)'},
                       annot_kws={'fontsize': 11},
                       vmin=-100, 
                       vmax=100,
                       center=0,
                       linewidths=0.5,
                       linecolor='gray',
                       ax=ax,
                       square=False)
        title_suffix = "Utility Difference"
        filename_suffix = "utility_difference"
    elif plot_mode == 'strong_only':
        im = sns.heatmap(data, 
                       annot=True, 
                       fmt='.1f',
                       cmap='viridis',
                       mask=mask,
                       cbar_kws={'label': 'Adversary Model Utility'},
                       annot_kws={'fontsize': 11},
                       vmin=0,
                       vmax=100,
                       linewidths=0.5,
                       linecolor='gray',
                       ax=ax,
                       square=False)
        title_suffix = "Adversary Utility"
        filename_suffix = "strong_utility"
    elif plot_mode == 'sum':
        im = sns.heatmap(data, 
                       annot=True, 
                       fmt='.1f',
                       cmap='plasma',
                       mask=mask,
                       cbar_kws={'label': 'Fraction of Maximum Possible Welfare (%)'},
                       annot_kws={'fontsize': 11},
                       vmin=0,
                       vmax=100,
                       linewidths=0.5,
                       linecolor='gray',
                       ax=ax,
                       square=False)
        title_suffix = "Cooperative Efficiency (% of Max Welfare)"
        filename_suffix = "sum_utility"
    elif plot_mode == 'baseline_only':
        im = sns.heatmap(data, 
                       annot=True, 
                       fmt='.1f',
                       cmap='coolwarm_r',
                       mask=mask,
                       cbar_kws={'label': 'Baseline Model Utility'},
                       annot_kws={'fontsize': 11},
                       vmin=0,
                       vmax=100,
                       linewidths=0.5,
                       linecolor='gray',
                       ax=ax,
                       square=False)
        title_suffix = "Baseline Final Utility"
        filename_suffix = "baseline_utility"
    
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
    ax.set_title(f'{MODEL_DISPLAY_NAMES[baseline_model]}: {title_suffix}', 
                fontsize=14, fontweight='bold', pad=15)
    
    # Reverse the labels to match the reversed data (0 at bottom, 1 at top)
    ax.set_yticklabels([f'{level:.2f}' for level in reversed(COMPETITION_LEVELS)], 
                      rotation=0, fontsize=14)
    
    # Get the position of the main heatmap to align MMLU bar properly
    heatmap_left = 0.125
    heatmap_width = 0.65
    
    # Create the MMLU-Pro performance bar with more space from the heatmap
    mmlu_bar_height = 0.03  # Make bar slightly taller for better visibility
    # Position bar lower to create clear separation from x-axis labels
    mmlu_ax = fig.add_axes([heatmap_left, 0.10, heatmap_width, mmlu_bar_height])
    
    # Create a colorbar showing MMLU-Pro scores
    mmlu_data = np.array(mmlu_scores).reshape(1, -1)
    mmlu_im = mmlu_ax.imshow(mmlu_data, cmap='plasma', aspect='auto', extent=[0, len(ordered_strong_models), 0, 1])
    
    mmlu_ax.set_xticks([])
    mmlu_ax.set_yticks([])
    
    # Create a separate axis for the MMLU score labels below the bar
    mmlu_labels_ax = fig.add_axes([heatmap_left, 0.05, heatmap_width, 0.03])
    mmlu_labels_ax.set_xlim(0, len(ordered_strong_models))
    mmlu_labels_ax.set_ylim(0, 1)
    
    # Add the percentage labels
    for i, score in enumerate(mmlu_scores):
        mmlu_labels_ax.text(i + 0.5, 0.5, f'{score}%', ha='center', va='center', 
                           rotation=0, fontsize=14, fontweight='bold')
    
    mmlu_labels_ax.set_xticks([])
    mmlu_labels_ax.set_yticks([])
    mmlu_labels_ax.axis('off')
    
    # Don't use tight_layout since we're manually positioning everything
    
    # Save figure
    baseline_name = baseline_model.replace('-', '_')
    filename = f'mmlu_ordered_{baseline_name}_{filename_suffix}_heatmap.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved individual heatmap to {filename}")
    plt.close()

def create_heatmap_for_baseline(results_by_competition, baseline_model, ordered_strong_models, plot_mode='diff'):
    """Create a heatmap for a specific baseline model."""
    
    # Maximum possible welfare at each competition level
    MAX_WELFARE = {
        0.0: 200.0,   # No competition: both agents can get 100
        0.25: 177.0,  # Slight competition
        0.5: 157.0,   # Moderate competition
        0.75: 138.0,  # High competition
        1.0: 100.0    # Full competition (constant-sum game)
    }
    
    # Create data matrix: rows = competition levels, cols = strong models
    # Reverse the order of competition levels so 0 is at bottom, 1 at top
    reversed_competition_levels = list(reversed(COMPETITION_LEVELS))
    data = np.full((len(reversed_competition_levels), len(ordered_strong_models)), np.nan)
    
    for i, comp_level in enumerate(reversed_competition_levels):
        for j, strong_model in enumerate(ordered_strong_models):
            if (comp_level in results_by_competition and 
                baseline_model in results_by_competition[comp_level] and
                strong_model in results_by_competition[comp_level][baseline_model]):
                
                experiments = results_by_competition[comp_level][baseline_model][strong_model]
                if experiments:
                    if plot_mode == 'diff':
                        # Use utility difference (strong - baseline)
                        values = [exp['utility_diff'] for exp in experiments]
                    elif plot_mode == 'strong_only':
                        # Use only strong model utility
                        values = [exp['strong_utility'] for exp in experiments]
                    elif plot_mode == 'sum':
                        # Use sum of utilities (baseline + strong), normalized by max possible welfare
                        raw_values = [exp['baseline_utility'] + exp['strong_utility'] for exp in experiments]
                        # Normalize by the maximum possible welfare at this competition level
                        max_welfare = MAX_WELFARE.get(comp_level, 100.0)
                        values = [(val / max_welfare) * 100.0 for val in raw_values]
                    elif plot_mode == 'baseline_only':
                        # Use only baseline model utility
                        values = [exp['baseline_utility'] for exp in experiments]
                    else:  # other modes
                        values = [exp['utility_diff'] for exp in experiments]
                    
                    data[i, j] = np.mean(values)
    
    return data

def plot_heatmaps(results_by_competition, ordered_strong_models, plot_mode='diff', baseline_order=None):
    """Plot combined heatmaps for all baseline models.
    
    Args:
        baseline_order: List of baseline models in the order they should appear.
                       If None, uses BASELINE_MODELS order.
    """
    
    # Print MMLU-Pro scores
    print("\n=== MMLU-Pro Scores of Models in Analysis ===")
    print("Models ordered from lowest to highest MMLU-Pro performance:")
    for i, model in enumerate(ordered_strong_models):
        score = MMLU_PRO_SCORES[model]
        print(f"{i+1:2d}. {MODEL_DISPLAY_NAMES[model]:<20} | MMLU-Pro: {score}%")
    
    # Use provided baseline order or default
    baselines_to_check = baseline_order if baseline_order else BASELINE_MODELS
    
    print(f"\nBaseline Models:")
    for baseline in baselines_to_check:
        if baseline in MMLU_PRO_SCORES and MMLU_PRO_SCORES[baseline] is not None:
            score = MMLU_PRO_SCORES[baseline]
            print(f"    {MODEL_DISPLAY_NAMES[baseline]:<20} | MMLU-Pro: {score}%")
    
    # Create figure with subplots for each baseline
    baselines_with_data = []
    for baseline in baselines_to_check:
        # Check if this baseline has any data
        has_data = False
        for comp_level in results_by_competition:
            if baseline in results_by_competition[comp_level]:
                for strong in results_by_competition[comp_level][baseline]:
                    if results_by_competition[comp_level][baseline][strong]:
                        has_data = True
                        break
                if has_data:
                    break
        if has_data:
            baselines_with_data.append(baseline)
    
    if not baselines_with_data:
        print("No baseline models found with data!")
        return
    
    fig, axes = plt.subplots(len(baselines_with_data), 1, figsize=(16, 5*len(baselines_with_data)))
    if len(baselines_with_data) == 1:
        axes = [axes]
    
    # Create MMLU-Pro score colorbar data
    mmlu_scores = [MMLU_PRO_SCORES[model] for model in ordered_strong_models]
    
    for idx, baseline in enumerate(baselines_with_data):
        ax = axes[idx]
        
        # Get data for this baseline
        data = create_heatmap_for_baseline(results_by_competition, baseline, ordered_strong_models, plot_mode)
        
        # Create mask for missing data
        mask = np.isnan(data)
        
        # Create heatmap with consistent 0-100 range for all plots
        if plot_mode == 'diff':
            # For difference plot, use -100 to 100 range centered at 0
            im = sns.heatmap(data, 
                           annot=True, 
                           fmt='.1f',
                           cmap='RdBu_r',  # Red = negative (baseline wins), Blue = positive (strong wins)
                           mask=mask,
                           cbar_kws={'label': 'Utility Difference (Adversary - Baseline)'},
                           annot_kws={'fontsize': 11},
                           vmin=-100, 
                           vmax=100,
                           center=0,
                           linewidths=0.5,
                           linecolor='gray',
                           ax=ax,
                           square=False)
            title_suffix = "Utility Difference"
        elif plot_mode == 'strong_only':
            im = sns.heatmap(data, 
                           annot=True, 
                           fmt='.1f',
                           cmap='viridis',
                           mask=mask,
                           cbar_kws={'label': 'Adversary Model Utility'},
                           annot_kws={'fontsize': 11},
                           vmin=0,
                           vmax=100,
                           linewidths=0.5,
                           linecolor='gray',
                           ax=ax,
                           square=False)
            title_suffix = "Adversary Utility"
        elif plot_mode == 'sum':
            # Now showing percentage of maximum possible welfare (0-100%)
            im = sns.heatmap(data, 
                           annot=True, 
                           fmt='.1f',
                           cmap='plasma',
                           mask=mask,
                           cbar_kws={'label': 'Fraction of Maximum Possible Welfare (%)'},
                           annot_kws={'fontsize': 11},
                           vmin=0,
                           vmax=100,
                           linewidths=0.5,
                           linecolor='gray',
                           ax=ax,
                           square=False)
            title_suffix = "Cooperative Efficiency (% of Max Welfare)"
        elif plot_mode == 'baseline_only':
            im = sns.heatmap(data, 
                           annot=True, 
                           fmt='.1f',
                           cmap='coolwarm_r',
                           mask=mask,
                           cbar_kws={'label': 'Baseline Model Utility'},
                           annot_kws={'fontsize': 11},
                           vmin=0,
                           vmax=100,
                           linewidths=0.5,
                           linecolor='gray',
                           ax=ax,
                           square=False)
            title_suffix = "Baseline Final Utility"
        
        # Set colorbar label font size
        cbar = ax.collections[0].colorbar
        if cbar:
            cbar.ax.tick_params(labelsize=12)
            cbar.set_label(cbar.ax.get_ylabel(), fontsize=14)
        
        # Set labels - only add x-axis label for the bottom plot
        if idx == len(baselines_with_data) - 1:
            ax.set_xlabel('Adversary Models Ordered by MMLU-Pro Performance →', fontsize=16, fontweight='bold')
            # Keep x-tick labels for bottom plot
            ax.set_xticklabels([MODEL_DISPLAY_NAMES[model] for model in ordered_strong_models], 
                              rotation=0, ha='center', fontsize=14)
        else:
            # Remove x-axis label and tick labels for top two plots
            ax.set_xlabel('')
            ax.set_xticklabels([])
        
        ax.set_ylabel('Competition Level', fontsize=16, fontweight='bold')
        ax.set_title(f'{MODEL_DISPLAY_NAMES[baseline]}: {title_suffix}', 
                    fontsize=14, fontweight='bold', pad=15)
        
        # Reverse the labels to match the reversed data (0 at bottom, 1 at top)
        ax.set_yticklabels([f'{level:.2f}' for level in reversed(COMPETITION_LEVELS)], 
                          rotation=0, fontsize=14)
    
    # Add MMLU-Pro performance bar at the bottom
    fig.subplots_adjust(bottom=0.5)  # More space for the complete layout
    
    # Get the position of the main heatmap to align everything properly
    # Assuming the heatmap uses the standard matplotlib positioning
    heatmap_left = 0.05  # Left edge of heatmap
    heatmap_width = 0.75  # Width of heatmap area
    
    # Create the MMLU-Pro performance bar (make it thicker and properly aligned)
    mmlu_bar_height = 0.02  # Make thicker to match legend thickness
    mmlu_ax = fig.add_axes([heatmap_left, -0.02, heatmap_width, mmlu_bar_height])
    
    # Create a colorbar showing MMLU-Pro scores
    mmlu_data = np.array(mmlu_scores).reshape(1, -1)
    mmlu_im = mmlu_ax.imshow(mmlu_data, cmap='plasma', aspect='auto', extent=[0, len(ordered_strong_models), 0, 1])
    
    mmlu_ax.set_xticks([])  # Remove ticks from the bar itself
    mmlu_ax.set_yticks([])
    # mmlu_ax.set_xlabel('MMLU-Pro Performance →', fontsize=10, fontweight='bold')
    
    # Create a separate axis for the MMLU score labels, positioned directly under the bar
    mmlu_labels_ax = fig.add_axes([heatmap_left, -0.04, heatmap_width, 0.02])
    mmlu_labels_ax.set_xlim(0, len(ordered_strong_models))
    mmlu_labels_ax.set_ylim(0, 1)
    
    # Add the percentage labels, properly centered under each model column
    for i, score in enumerate(mmlu_scores):
        mmlu_labels_ax.text(i + 0.5, 0.5, f'{score}%', ha='center', va='center', 
                           rotation=0, fontsize=14, fontweight='bold')
    
    mmlu_labels_ax.set_xticks([])
    mmlu_labels_ax.set_yticks([])
    mmlu_labels_ax.axis('off')  # Hide the axis borders
    
    # Add the separate legend colorbar at the very bottom
    # mmlu_cbar = plt.colorbar(mmlu_im, ax=mmlu_ax, orientation='horizontal', 
    #                         shrink=8, aspect=25, pad=0.25,
    #                         anchor=(0.0, 0.0))  # Anchor to align properly
    # mmlu_cbar.set_label('MMLU-Pro Score (%)', fontsize=9)
    
    plt.tight_layout()
    
    # Save figure
    # For baseline_only mode, add suffix based on baseline order
    order_suffix = ''
    if plot_mode == 'baseline_only' and baseline_order:
        # Identify which baseline is at the bottom (last in the list)
        if baseline_order[-1] == 'gpt-4o-2024-05-13':
            order_suffix = '_gpt4o_bottom'
        elif baseline_order[-1] == 'gemini-1-5-pro':
            order_suffix = '_gemini_bottom'
        elif baseline_order[-1] == 'claude-3-opus':
            order_suffix = '_claude_bottom'
    
    if plot_mode == 'diff':
        filename = 'mmlu_ordered_utility_difference_heatmaps.pdf'
    elif plot_mode == 'strong_only':
        filename = 'mmlu_ordered_strong_utility_heatmaps.pdf'
    elif plot_mode == 'sum':
        filename = 'mmlu_ordered_sum_utility_heatmaps.pdf'
    elif plot_mode == 'baseline_only':
        filename = f'mmlu_ordered_baseline_utility_heatmaps{order_suffix}.pdf'
    else:
        filename = 'mmlu_ordered_heatmaps.pdf'
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved heatmap to {filename}")

def print_summary_statistics(results_by_competition, ordered_strong_models):
    """Print summary of the data."""
    print("\n=== Data Summary ===")
    
    total_experiments = 0
    for comp_level in COMPETITION_LEVELS:
        level_total = 0
        print(f"\nCompetition Level {comp_level}:")
        
        for baseline in BASELINE_MODELS:
            for strong in ordered_strong_models:
                if (comp_level in results_by_competition and 
                    baseline in results_by_competition[comp_level] and
                    strong in results_by_competition[comp_level][baseline]):
                    
                    experiments = results_by_competition[comp_level][baseline][strong]
                    count = len(experiments)
                    if count > 0:
                        avg_diff = np.mean([exp['utility_diff'] for exp in experiments])
                        print(f"  {MODEL_DISPLAY_NAMES[baseline]} vs {MODEL_DISPLAY_NAMES[strong]}: "
                              f"{count} experiments, avg diff: {avg_diff:.1f}")
                        level_total += count
        
        if level_total > 0:
            print(f"  Level subtotal: {level_total} experiments")
            total_experiments += level_total
    
    print(f"\nTotal experiments across all conditions: {total_experiments}")

def main():
    """Main function to generate MMLU-ordered heatmaps."""
    print("Loading experiment results...")
    results_dir = '/root/bargain/experiments/results'
    results_by_competition = load_experiment_results(results_dir)
    
    # Debug: Check which models have no data
    print("\n=== Debugging: Checking which models have data ===")
    models_with_data = set()
    for comp_level in results_by_competition:
        for baseline in results_by_competition[comp_level]:
            for strong in results_by_competition[comp_level][baseline]:
                if results_by_competition[comp_level][baseline][strong]:
                    models_with_data.add(strong)
    
    print(f"Models found with data: {models_with_data}")
    print(f"\nModels in STRONG_MODELS_REQUESTED: {set(STRONG_MODELS_REQUESTED)}")
    missing_models = set(STRONG_MODELS_REQUESTED) - models_with_data
    if missing_models:
        print(f"Missing models (no data found): {missing_models}")
    
    # Get models that have data, ordered by MMLU-Pro score
    ordered_strong_models = get_models_with_data(results_by_competition)
    
    if not ordered_strong_models:
        print("No strong models found with both data and MMLU-Pro scores!")
        return
    
    print(f"\nFound {len(ordered_strong_models)} strong models with data and MMLU-Pro scores.")
    
    # Print summary statistics
    print_summary_statistics(results_by_competition, ordered_strong_models)
    
    # Get baselines that have data
    baselines_with_data = []
    for baseline in BASELINE_MODELS:
        has_data = False
        for comp_level in results_by_competition:
            if baseline in results_by_competition[comp_level]:
                for strong in results_by_competition[comp_level][baseline]:
                    if results_by_competition[comp_level][baseline][strong]:
                        has_data = True
                        break
                if has_data:
                    break
        if has_data:
            baselines_with_data.append(baseline)
    
    # Generate individual heatmaps for each baseline
    print("\n=== Creating Individual Heatmaps for Each Baseline ===")
    
    for baseline in baselines_with_data:
        print(f"\n--- Creating heatmaps for {MODEL_DISPLAY_NAMES[baseline]} ---")
        
        # Create utility difference heatmap
        print(f"Creating utility difference heatmap for {MODEL_DISPLAY_NAMES[baseline]}...")
        plot_individual_heatmap(results_by_competition, baseline, ordered_strong_models, plot_mode='diff')
        
        # Create strong-only utility heatmap
        print(f"Creating strong model utility heatmap for {MODEL_DISPLAY_NAMES[baseline]}...")
        plot_individual_heatmap(results_by_competition, baseline, ordered_strong_models, plot_mode='strong_only')
        
        # Create sum of utilities heatmap
        print(f"Creating sum of utilities heatmap for {MODEL_DISPLAY_NAMES[baseline]}...")
        plot_individual_heatmap(results_by_competition, baseline, ordered_strong_models, plot_mode='sum')
        
        # Create baseline-only utility heatmap
        print(f"Creating baseline model utility heatmap for {MODEL_DISPLAY_NAMES[baseline]}...")
        plot_individual_heatmap(results_by_competition, baseline, ordered_strong_models, plot_mode='baseline_only')
    
    # Create the combined heatmaps
    print("\n=== Creating Combined Heatmaps (All Baselines) ===")
    
    # Create the main heatmap (utility difference)
    print("\nCreating combined utility difference heatmaps...")
    plot_heatmaps(results_by_competition, ordered_strong_models, plot_mode='diff')
    
    # Create strong-only utility heatmap
    print("\nCreating combined strong model utility heatmaps...")
    plot_heatmaps(results_by_competition, ordered_strong_models, plot_mode='strong_only')
    
    # Create sum of utilities heatmap
    print("\nCreating combined sum of utilities heatmaps...")
    plot_heatmaps(results_by_competition, ordered_strong_models, plot_mode='sum')
    
    # Create THREE versions of baseline-only utility heatmap with different orderings
    print("\n=== Creating Baseline Final Utility Heatmaps - Three Orderings ===")
    
    # Define the three different baseline orders
    baseline_orders = [
        # Version 1: GPT-4o at the bottom (furthest from title)
        ('claude-3-opus', 'gemini-1-5-pro', 'gpt-4o-2024-05-13'),
        # Version 2: Original order (Claude at the bottom)
        ('gpt-4o-2024-05-13', 'gemini-1-5-pro', 'claude-3-opus'),
        # Version 3: Gemini at the bottom
        ('gpt-4o-2024-05-13', 'claude-3-opus', 'gemini-1-5-pro')
    ]
    
    order_names = ['GPT-4o at bottom', 'Claude-3-Opus at bottom (original)', 'Gemini-1.5-Pro at bottom']
    
    for order_idx, baseline_order in enumerate(baseline_orders):
        print(f"\nCreating baseline final utility heatmap - {order_names[order_idx]}...")
        plot_heatmaps(results_by_competition, ordered_strong_models, plot_mode='baseline_only', baseline_order=list(baseline_order))

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Create beautiful heatmaps showing the percentage of times baseline (weaker) agents
achieved higher utility than strong agents, separated by competition level.
"""

import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Set style for beautiful plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Define baseline (weak) and strong models
BASELINE_MODELS = ['claude-3-opus', 'gemini-1-5-pro', 'gpt-4o']
# STRONG_MODELS = ['claude-3-5-haiku', 'claude-3-5-sonnet', 'claude-4-1-opus',
#                  'claude-4-sonnet', 'gemini-2-0-flash', 'gemini-2-5-pro',
#                  'gemma-3-27b', 'gpt-4o-latest', 'gpt-4o-mini', 'o1', 'o3']
STRONG_MODELS = ['claude-3-5-haiku', 'claude-3-5-sonnet', 'claude-4-1-opus',
                 'claude-4-sonnet', 'gemini-2-0-flash', 'gemini-2-5-pro',
                 'gpt-4o-latest', 'o1', 'o3']

# Model display names (clean, professional labels)
MODEL_DISPLAY_NAMES = {
    'claude-3-opus': 'Claude 3 Opus',
    'gemini-1-5-pro': 'Gemini 1.5 Pro',
    'gpt-4o': 'GPT-4o (May 2024)',
    'claude-3-5-haiku': 'Claude 3.5 Haiku',
    'claude-3-5-sonnet': 'Claude 3.5 Sonnet',
    'claude-4-1-opus': 'Claude 4.1 Opus',
    'claude-4-sonnet': 'Claude 4 Sonnet',
    'gemini-2-0-flash': 'Gemini 2.0 Flash',
    'gemini-2-5-pro': 'Gemini 2.5 Pro',
    'gemma-3-27b': 'Gemma 3 27B',
    'gpt-4o-latest': 'GPT-4o (Nov 2024)',
    'gpt-4o-mini': 'GPT-4o Mini',
    'o1': 'O1',
    'o3': 'O3'
}

# Competition level buckets with nice labels
COMPETITION_LEVELS = {
    0.0: 'Fully Cooperative (0.0)',
    0.25: 'Mostly Cooperative (0.25)',
    0.5: 'Balanced (0.5)',
    0.75: 'Mostly Competitive (0.75)',
    1.0: 'Fully Competitive (1.0)'
}

def categorize_competition_level(comp_level):
    """Categorize competition level into buckets."""
    if comp_level is None:
        return None
    return min(COMPETITION_LEVELS.keys(), key=lambda x: abs(x - comp_level))

def load_baseline_victory_data(results_dir):
    """
    Load experiment results and calculate percentage of times baseline agents
    had higher utility than strong agents.
    """
    # Track victories, ties, and losses
    results_by_competition = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
        'baseline_wins': 0,
        'ties': 0,
        'strong_wins': 0,
        'total': 0
    })))
    
    results_path = Path(results_dir)
    
    for file_path in results_path.glob('**/*_summary.json'):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if 'experiments' in data and data['experiments']:
                for exp in data['experiments']:
                    if 'config' in exp and 'agents' in exp['config']:
                        agents = exp['config']['agents']
                        
                        # Get competition level
                        comp_level = exp['config'].get('competition_level', None)
                        comp_bucket = categorize_competition_level(comp_level)
                        
                        if comp_bucket is None or len(agents) < 2:
                            continue
                        
                        # Identify models from agent names
                        agent1 = agents[0]
                        agent2 = agents[1]
                        
                        model1 = None
                        model2 = None
                        
                        # Extract model names from agent IDs
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
                        
                        model1 = extract_model_name(agent1)
                        model2 = extract_model_name(agent2)
                        
                        if not (model1 and model2):
                            continue
                        
                        # Determine which is baseline and which is strong
                        baseline_model = None
                        strong_model = None
                        baseline_agent_idx = None
                        
                        if model1 in BASELINE_MODELS and model2 in STRONG_MODELS:
                            baseline_model = model1
                            strong_model = model2
                            baseline_agent_idx = 0
                        elif model2 in BASELINE_MODELS and model1 in STRONG_MODELS:
                            baseline_model = model2
                            strong_model = model1
                            baseline_agent_idx = 1
                        else:
                            continue  # Skip if not baseline vs strong
                        
                        # Get utilities
                        final_utilities = exp.get('final_utilities', {})
                        if not final_utilities:
                            continue
                        
                        baseline_agent = agents[baseline_agent_idx]
                        strong_agent = agents[1 - baseline_agent_idx]
                        
                        baseline_util = final_utilities.get(baseline_agent, 0)
                        strong_util = final_utilities.get(strong_agent, 0)
                        
                        # Record outcome
                        results_by_competition[comp_bucket][baseline_model][strong_model]['total'] += 1
                        
                        if baseline_util > strong_util:
                            results_by_competition[comp_bucket][baseline_model][strong_model]['baseline_wins'] += 1
                        elif baseline_util == strong_util:
                            results_by_competition[comp_bucket][baseline_model][strong_model]['ties'] += 1
                        else:
                            results_by_competition[comp_bucket][baseline_model][strong_model]['strong_wins'] += 1
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    return results_by_competition

def calculate_victory_percentages(results):
    """Calculate percentage of baseline victories (excluding ties)."""
    victory_matrix = np.zeros((len(BASELINE_MODELS), len(STRONG_MODELS)))
    sample_matrix = np.zeros((len(BASELINE_MODELS), len(STRONG_MODELS)))
    tie_matrix = np.zeros((len(BASELINE_MODELS), len(STRONG_MODELS)))
    
    for i, baseline in enumerate(BASELINE_MODELS):
        for j, strong in enumerate(STRONG_MODELS):
            if baseline in results and strong in results[baseline]:
                data = results[baseline][strong]
                total = data['total']
                baseline_wins = data['baseline_wins']
                ties = data['ties']
                
                if total > 0:
                    # Calculate baseline victory percentage (including ties as 0.5 wins)
                    # This gives a more nuanced view
                    victory_matrix[i, j] = (baseline_wins + 0.5 * ties) / total * 100
                    sample_matrix[i, j] = total
                    tie_matrix[i, j] = ties / total * 100
                else:
                    victory_matrix[i, j] = np.nan
                    sample_matrix[i, j] = 0
                    tie_matrix[i, j] = np.nan
            else:
                victory_matrix[i, j] = np.nan
                sample_matrix[i, j] = 0
                tie_matrix[i, j] = np.nan
    
    return victory_matrix, sample_matrix, tie_matrix

def create_beautiful_heatmaps(results_by_competition):
    """Create beautiful, publication-quality heatmaps."""
    
    # Create main figure with subplots
    fig = plt.figure(figsize=(24, 14))
    fig.suptitle('Baseline Agent Victory Rates Against Strong Models\nAcross Different Competition Levels', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Create grid for subplots
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.25)
    
    # Color scheme - using diverging colormap centered at 50%
    cmap = 'RdBu_r'  # Red for low baseline victory, Blue for high baseline victory
    
    for idx, (comp_level, comp_label) in enumerate(COMPETITION_LEVELS.items()):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        # Calculate victory percentages
        victory_matrix, sample_matrix, tie_matrix = calculate_victory_percentages(
            results_by_competition[comp_level]
        )
        
        # Create annotations with victory percentage and sample size
        annot_matrix = np.empty_like(victory_matrix, dtype=object)
        for i in range(len(BASELINE_MODELS)):
            for j in range(len(STRONG_MODELS)):
                if not np.isnan(victory_matrix[i, j]):
                    # Format: percentage (sample size)
                    annot_matrix[i, j] = f'{victory_matrix[i, j]:.0f}%\n({int(sample_matrix[i, j])})'
                else:
                    annot_matrix[i, j] = '-'
        
        # Create mask for missing data
        mask = np.isnan(victory_matrix)
        
        # Create heatmap
        sns.heatmap(victory_matrix, 
                   annot=annot_matrix,
                   fmt='',
                   cmap=cmap,
                   mask=mask,
                   cbar_kws={'label': 'Baseline Victory %', 'shrink': 0.8},
                   vmin=0, 
                   vmax=100,
                   center=50,
                   linewidths=1,
                   linecolor='white',
                   square=False,
                   ax=ax,
                   cbar=(idx == 0))  # Only show colorbar for first subplot
        
        # Customize subplot
        ax.set_title(f'{comp_label}', fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Strong Models', fontsize=11, fontweight='bold')
        ax.set_ylabel('Baseline Models', fontsize=11, fontweight='bold')
        
        # Set tick labels
        ax.set_xticklabels([MODEL_DISPLAY_NAMES.get(m, m) for m in STRONG_MODELS], 
                          rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels([MODEL_DISPLAY_NAMES.get(m, m) for m in BASELINE_MODELS], 
                          rotation=0, fontsize=10)
        
        # Add border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
            spine.set_edgecolor('gray')
        
        # Add summary statistics to each subplot
        total_experiments = np.nansum(sample_matrix)
        mean_victory = np.nanmean(victory_matrix)
        if not np.isnan(mean_victory):
            ax.text(0.98, 0.02, f'n={int(total_experiments)}\nμ={mean_victory:.1f}%', 
                   transform=ax.transAxes, fontsize=9,
                   ha='right', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Remove the 6th subplot (we only have 5 competition levels)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Add legend/interpretation guide in the empty space
    legend_text = """
    📊 Interpretation Guide:
    
    • Values show % of negotiations where baseline agent
      achieved higher utility than strong agent
    • Ties counted as 50% victory for each side
    • Numbers in parentheses show sample size
    • Blue = Baseline dominates (>50% victory)
    • Red = Strong model dominates (<50% victory)
    • White = Balanced (≈50% victory)
    
    ⚠️ Note: Most experiments have baseline moving first,
    which may introduce first-mover advantage bias
    """
    ax6.text(0.1, 0.7, legend_text, transform=ax6.transAxes, 
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3, pad=1))
    
    # Save figure
    plt.savefig('baseline_victory_heatmaps_beautiful.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    print("✅ Saved beautiful heatmap to baseline_victory_heatmaps_beautiful.png")
    
    # Create a second figure showing only the aggregated view
    create_aggregated_heatmap(results_by_competition)

def create_aggregated_heatmap(results_by_competition):
    """Create an aggregated heatmap across all competition levels."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Aggregated Baseline Victory Rates Across All Competition Levels', 
                 fontsize=16, fontweight='bold')
    
    # Aggregate data across all competition levels
    aggregated_results = defaultdict(lambda: defaultdict(lambda: {
        'baseline_wins': 0, 'ties': 0, 'strong_wins': 0, 'total': 0
    }))
    
    for comp_level in results_by_competition:
        for baseline in results_by_competition[comp_level]:
            for strong in results_by_competition[comp_level][baseline]:
                data = results_by_competition[comp_level][baseline][strong]
                aggregated_results[baseline][strong]['baseline_wins'] += data['baseline_wins']
                aggregated_results[baseline][strong]['ties'] += data['ties']
                aggregated_results[baseline][strong]['strong_wins'] += data['strong_wins']
                aggregated_results[baseline][strong]['total'] += data['total']
    
    # Calculate victory percentages
    victory_matrix, sample_matrix, tie_matrix = calculate_victory_percentages(aggregated_results)
    
    # Create annotations
    annot_matrix = np.empty_like(victory_matrix, dtype=object)
    for i in range(len(BASELINE_MODELS)):
        for j in range(len(STRONG_MODELS)):
            if not np.isnan(victory_matrix[i, j]):
                annot_matrix[i, j] = f'{victory_matrix[i, j]:.0f}%\n(n={int(sample_matrix[i, j])})'
            else:
                annot_matrix[i, j] = '-'
    
    # Main heatmap
    mask = np.isnan(victory_matrix)
    sns.heatmap(victory_matrix, 
               annot=annot_matrix,
               fmt='',
               cmap='RdBu_r',
               mask=mask,
               cbar_kws={'label': 'Baseline Victory %'},
               vmin=0, 
               vmax=100,
               center=50,
               linewidths=1,
               linecolor='white',
               ax=ax1)
    
    ax1.set_title('Overall Baseline Victory Rates', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Strong Models', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Baseline Models', fontsize=11, fontweight='bold')
    ax1.set_xticklabels([MODEL_DISPLAY_NAMES.get(m, m) for m in STRONG_MODELS], 
                       rotation=45, ha='right', fontsize=9)
    ax1.set_yticklabels([MODEL_DISPLAY_NAMES.get(m, m) for m in BASELINE_MODELS], 
                       rotation=0, fontsize=10)
    
    # Sample size heatmap
    sns.heatmap(sample_matrix, 
               annot=True,
               fmt='.0f',
               cmap='YlOrBr',
               mask=mask,
               cbar_kws={'label': 'Number of Experiments'},
               linewidths=1,
               linecolor='white',
               ax=ax2)
    
    ax2.set_title('Sample Sizes per Model Pair', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Strong Models', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Baseline Models', fontsize=11, fontweight='bold')
    ax2.set_xticklabels([MODEL_DISPLAY_NAMES.get(m, m) for m in STRONG_MODELS], 
                       rotation=45, ha='right', fontsize=9)
    ax2.set_yticklabels([MODEL_DISPLAY_NAMES.get(m, m) for m in BASELINE_MODELS], 
                       rotation=0, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('baseline_victory_aggregated.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    print("✅ Saved aggregated heatmap to baseline_victory_aggregated.png")

def print_summary_statistics(results_by_competition):
    """Print summary statistics for the analysis."""
    
    print("\n" + "="*80)
    print("BASELINE VICTORY RATE SUMMARY STATISTICS")
    print("="*80)
    
    for comp_level, comp_label in COMPETITION_LEVELS.items():
        print(f"\n📊 {comp_label}:")
        print("-" * 40)
        
        total_baseline_wins = 0
        total_ties = 0
        total_strong_wins = 0
        total_experiments = 0
        
        for baseline in results_by_competition[comp_level]:
            for strong in results_by_competition[comp_level][baseline]:
                data = results_by_competition[comp_level][baseline][strong]
                total_baseline_wins += data['baseline_wins']
                total_ties += data['ties']
                total_strong_wins += data['strong_wins']
                total_experiments += data['total']
        
        if total_experiments > 0:
            baseline_victory_rate = (total_baseline_wins / total_experiments) * 100
            tie_rate = (total_ties / total_experiments) * 100
            strong_victory_rate = (total_strong_wins / total_experiments) * 100
            
            print(f"  Total experiments: {total_experiments}")
            print(f"  Baseline victories: {total_baseline_wins} ({baseline_victory_rate:.1f}%)")
            print(f"  Ties: {total_ties} ({tie_rate:.1f}%)")
            print(f"  Strong victories: {total_strong_wins} ({strong_victory_rate:.1f}%)")
            
            # Calculate victory rate with ties as 0.5
            adjusted_baseline_rate = ((total_baseline_wins + 0.5 * total_ties) / total_experiments) * 100
            print(f"  Adjusted baseline victory rate (ties=0.5): {adjusted_baseline_rate:.1f}%")
        else:
            print("  No data available")
    
    # Overall statistics
    print("\n" + "="*80)
    print("OVERALL STATISTICS (ALL COMPETITION LEVELS)")
    print("="*80)
    
    grand_total_baseline = 0
    grand_total_ties = 0
    grand_total_strong = 0
    grand_total_experiments = 0
    
    for comp_level in results_by_competition:
        for baseline in results_by_competition[comp_level]:
            for strong in results_by_competition[comp_level][baseline]:
                data = results_by_competition[comp_level][baseline][strong]
                grand_total_baseline += data['baseline_wins']
                grand_total_ties += data['ties']
                grand_total_strong += data['strong_wins']
                grand_total_experiments += data['total']
    
    if grand_total_experiments > 0:
        print(f"Total experiments: {grand_total_experiments}")
        print(f"Baseline victories: {grand_total_baseline} ({grand_total_baseline/grand_total_experiments*100:.1f}%)")
        print(f"Ties: {grand_total_ties} ({grand_total_ties/grand_total_experiments*100:.1f}%)")
        print(f"Strong victories: {grand_total_strong} ({grand_total_strong/grand_total_experiments*100:.1f}%)")
        
        adjusted_rate = ((grand_total_baseline + 0.5 * grand_total_ties) / grand_total_experiments) * 100
        print(f"\n📈 Overall adjusted baseline victory rate: {adjusted_rate:.1f}%")
        
        if adjusted_rate > 50:
            print("   → Baseline models outperform strong models overall!")
        elif adjusted_rate < 50:
            print("   → Strong models outperform baseline models overall!")
        else:
            print("   → Perfect balance between baseline and strong models!")

def main():
    """Main function to create beautiful baseline victory heatmaps."""
    print("Loading experiment results...")
    results_dir = '/root/bargain/experiments/results_current'
    
    results_by_competition = load_baseline_victory_data(results_dir)
    
    # Print summary statistics
    print_summary_statistics(results_by_competition)
    
    # Create visualizations
    print("\nCreating beautiful heatmaps...")
    create_beautiful_heatmaps(results_by_competition)
    
    print("\n✨ Beautiful heatmaps created successfully!")
    print("Files saved:")
    print("  • baseline_victory_heatmaps_beautiful.png (main figure)")
    print("  • baseline_victory_aggregated.png (aggregated view)")

if __name__ == "__main__":
    main()
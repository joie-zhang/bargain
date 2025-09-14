#!/usr/bin/env python3
"""
Create grouped bar graphs showing win rates categorized by model family.
Generates 3 figures, one for each baseline agent, with strong models grouped by family.
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

# Define strong models by family
MODEL_FAMILIES = {
    'Claude': ['claude-3-5-haiku', 'claude-3-5-sonnet', 'claude-4-1-opus', 'claude-4-sonnet'],
    'Gemini/Gemma': ['gemini-2-0-flash', 'gemini-2-5-pro', 'gemma-3-27b'],
    'GPT/O-Series': ['gpt-4o-latest', 'gpt-4o-mini', 'o1', 'o3']
}

# Model display names (shorter for better visualization)
MODEL_DISPLAY_NAMES = {
    'claude-3-opus': 'Claude 3 Opus',
    'gemini-1-5-pro': 'Gemini 1.5 Pro',
    'gpt-4o': 'GPT-4o (May)',
    'claude-3-5-haiku': '3.5 Haiku',
    'claude-3-5-sonnet': '3.5 Sonnet',
    'claude-4-1-opus': '4.1 Opus',
    'claude-4-sonnet': '4 Sonnet',
    'gemini-2-0-flash': '2.0 Flash',
    'gemini-2-5-pro': '2.5 Pro',
    'gemma-3-27b': 'Gemma 3',
    'gpt-4o-latest': 'GPT-4o (Nov)',
    'gpt-4o-mini': '4o Mini',
    'o1': 'O1',
    'o3': 'O3'
}

# Color palette for families
FAMILY_COLORS = {
    'Claude': '#8B4789',  # Purple
    'Gemini/Gemma': '#4285F4',  # Blue
    'GPT/O-Series': '#34A853'  # Green
}

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
                        
                        agent1_model = extract_model_name(agent1)
                        agent2_model = extract_model_name(agent2)
                        
                        # Check which are baseline and which are strong
                        all_strong_models = [m for family_models in MODEL_FAMILIES.values() for m in family_models]
                        
                        if agent1_model in BASELINE_MODELS and agent2_model in all_strong_models:
                            model1, model2 = agent1_model, agent2_model
                        elif agent2_model in BASELINE_MODELS and agent1_model in all_strong_models:
                            model1, model2 = agent2_model, agent1_model
                        
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
                                if model1 in BASELINE_MODELS and model2 in all_strong_models:
                                    baseline_won = agent1_won
                                    results[model1][model2].append(0 if baseline_won else 1)  # Strong model win rate
                                elif model2 in BASELINE_MODELS and model1 in all_strong_models:
                                    baseline_won = not agent1_won  # agent2 won
                                    results[model2][model1].append(0 if baseline_won else 1)  # Strong model win rate
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    return results

def calculate_win_rates_by_family(results, baseline_model):
    """Calculate win rates for a specific baseline model, grouped by family."""
    family_data = {}
    
    for family, models in MODEL_FAMILIES.items():
        family_win_rates = []
        model_names = []
        
        for model in models:
            if baseline_model in results and model in results[baseline_model]:
                wins = results[baseline_model][model]
                if wins:
                    win_rate = np.mean(wins)
                    family_win_rates.append(win_rate)
                    model_names.append(MODEL_DISPLAY_NAMES.get(model, model))
                else:
                    family_win_rates.append(np.nan)
                    model_names.append(MODEL_DISPLAY_NAMES.get(model, model))
            else:
                family_win_rates.append(np.nan)
                model_names.append(MODEL_DISPLAY_NAMES.get(model, model))
        
        family_data[family] = {
            'models': model_names,
            'win_rates': family_win_rates
        }
    
    return family_data

def create_grouped_bar_chart(family_data, baseline_model, filename):
    """Create a grouped bar chart for a baseline model."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x_pos = 0
    x_ticks = []
    x_labels = []
    bar_width = 0.8
    spacing_between_families = 1.5
    
    # Plot each family
    for family_idx, (family, data) in enumerate(family_data.items()):
        models = data['models']
        win_rates = data['win_rates']
        
        # Filter out NaN values
        valid_indices = [i for i, rate in enumerate(win_rates) if not np.isnan(rate)]
        
        if not valid_indices:
            continue
        
        # Plot bars for this family
        for i, idx in enumerate(valid_indices):
            bar_x = x_pos + i
            bar_value = win_rates[idx]
            
            # Create bar
            bar = ax.bar(bar_x, bar_value, bar_width, 
                         color=FAMILY_COLORS[family],
                         label=family if i == 0 else "",
                         edgecolor='black',
                         linewidth=1.5,
                         alpha=0.8)
            
            # Add value label on top of bar
            if not np.isnan(bar_value):
                ax.text(bar_x, bar_value + 0.02, f'{bar_value:.1%}', 
                       ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            x_ticks.append(bar_x)
            x_labels.append(models[idx])
        
        # Add family label
        if valid_indices:
            family_center = x_pos + (len(valid_indices) - 1) / 2
            ax.text(family_center, -0.08, family, ha='center', va='top',
                   fontsize=11, fontweight='bold', transform=ax.get_xaxis_transform())
        
        # Update position for next family
        x_pos += len(valid_indices) + spacing_between_families
    
    # Set labels and title
    ax.set_xlabel('Strong Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Win Rate Against ' + MODEL_DISPLAY_NAMES.get(baseline_model, baseline_model), 
                  fontsize=12, fontweight='bold')
    ax.set_title(f'Strong Model Win Rates Against {MODEL_DISPLAY_NAMES.get(baseline_model, baseline_model)} (Grouped by Family)',
                fontsize=14, fontweight='bold', pad=20)
    
    # Set y-axis limits and format
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    # Set x-axis
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    
    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add legend
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    
    # Add horizontal line at 50%
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1, label='50% Win Rate')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved bar chart to {filename}")

def main():
    """Main function to generate grouped bar charts."""
    # Load results
    print("Loading experiment results...")
    results_dir = '/root/bargain/experiments/results'
    results = load_experiment_results(results_dir)
    
    # Print summary statistics
    print("\nData summary by baseline model:")
    for baseline in BASELINE_MODELS:
        print(f"\n{MODEL_DISPLAY_NAMES.get(baseline, baseline)}:")
        total = 0
        for family, models in MODEL_FAMILIES.items():
            family_total = 0
            for model in models:
                if baseline in results and model in results[baseline]:
                    count = len(results[baseline][model])
                    if count > 0:
                        print(f"  vs {MODEL_DISPLAY_NAMES.get(model, model)}: {count} experiments")
                        family_total += count
                        total += count
            if family_total > 0:
                print(f"  {family} family total: {family_total} experiments")
        print(f"  Total for {MODEL_DISPLAY_NAMES.get(baseline, baseline)}: {total} experiments")
    
    # Generate bar charts for each baseline model
    for baseline in BASELINE_MODELS:
        print(f"\nGenerating bar chart for {MODEL_DISPLAY_NAMES.get(baseline, baseline)}...")
        
        # Calculate win rates grouped by family
        family_data = calculate_win_rates_by_family(results, baseline)
        
        # Create the grouped bar chart
        safe_baseline_name = baseline.replace('-', '_')
        filename = f'figures/win_rate_bar_chart_{safe_baseline_name}.png'
        create_grouped_bar_chart(family_data, baseline, filename)
        
        # Print statistics
        print(f"\nWin rates for strong models against {MODEL_DISPLAY_NAMES.get(baseline, baseline)}:")
        for family, data in family_data.items():
            print(f"  {family}:")
            for model, rate in zip(data['models'], data['win_rates']):
                if not np.isnan(rate):
                    print(f"    {model}: {rate:.1%}")
    
    print("\n" + "="*50)
    print("All visualizations have been generated successfully!")

if __name__ == "__main__":
    main()
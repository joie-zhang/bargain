#!/usr/bin/env python3
"""
Create a heatmap of win rates for strong models in negotiations
Based on aggregate_results.py logic to parse experiment data
"""

import json
import glob
import os
import sys
import re
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def get_model_name_from_agent_id(agent_id):
    """Extract model name from agent ID"""
    # Convert from agent ID format to display format
    model_mappings = {
        'claude_4_sonnet': 'claude-4-sonnet',
        'gemini_2.5_pro': 'gemini-2.5-pro', 
        'llama_3.1_405b_instruct': 'llama-3.1-405b-instruct',
        'qwen_3_235b_a22b_2507': 'qwen-3-235b-a22b-2507'
    }
    
    # Try exact matches first
    for internal, display in model_mappings.items():
        if agent_id.startswith(internal):
            return display
    
    # Fallback: extract first part before underscore/number
    base_name = agent_id.split('_')[0] if '_' in agent_id else agent_id
    
    # Handle known variations
    if 'claude' in base_name.lower():
        return 'claude-4-sonnet'
    elif 'gemini' in base_name.lower():
        return 'gemini-2.5-pro'
    elif 'llama' in base_name.lower():
        return 'llama-3.1-405b-instruct'
    elif 'qwen' in base_name.lower():
        return 'qwen-3-235b-a22b-2507'
    
    return base_name

def load_run_results(results_dir):
    """Load all run results from the specified directory"""
    results = []
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Warning: Directory {results_dir} does not exist")
        return results
    
    # Look for both single and batch experiment result files
    patterns = [
        results_path / "run_*_experiment_results.json",  # Batch mode results
        results_path / "experiment_results.json",        # Single experiment results
    ]
    
    run_files = []
    for pattern in patterns:
        run_files.extend(sorted(glob.glob(str(pattern))))
    
    if not run_files:
        print(f"Warning: No result files found in {results_dir}")
        return results
    
    print(f"Found {len(run_files)} result files in {results_dir}")
    
    for file_path in run_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Warning: Error loading {file_path}: {e}")
    
    return results

def get_agent_info(results):
    """Extract agent information from the first result"""
    if not results:
        return None, None, None, None
    
    first_result = results[0]
    config = first_result.get('config', {})
    agents = config.get('agents', [])
    
    if len(agents) < 2:
        print("Warning: Less than 2 agents found in config")
        return None, None, None, None
    
    agent1_name = agents[0]
    agent2_name = agents[1]
    
    # Extract model names from agent IDs
    agent1_model = get_model_name_from_agent_id(agent1_name)
    agent2_model = get_model_name_from_agent_id(agent2_name)
    
    return agent1_name, agent2_name, agent1_model, agent2_model

def calculate_win_rate(results):
    """Calculate win rates for a pair of models"""
    if not results:
        return None
    
    agent1_name, agent2_name, agent1_model, agent2_model = get_agent_info(results)
    
    if not all([agent1_name, agent2_name, agent1_model, agent2_model]):
        print(f"Warning: Could not determine agent names for directory")
        return None
    
    agent1_wins = 0
    agent2_wins = 0
    ties = 0
    total_games = len(results)
    
    print(f"  Analyzing: {agent1_model} vs {agent2_model} ({total_games} games)")
    
    for result in results:
        # Get final utilities from result
        final_utilities = result.get('final_utilities', {})
        
        # If no final utilities, this was likely a no-deal scenario (both get 0)
        if not final_utilities:
            ties += 1
            continue
        
        agent1_utility = final_utilities.get(agent1_name, 0)
        agent2_utility = final_utilities.get(agent2_name, 0)
        
        # Determine winner
        if agent1_utility > agent2_utility:
            agent1_wins += 1
        elif agent2_utility > agent1_utility:
            agent2_wins += 1
        else:
            ties += 1
    
    # Calculate win rate for agent1 (row agent)
    agent1_win_rate = agent1_wins / total_games if total_games > 0 else 0
    
    return {
        'agent1_model': agent1_model,
        'agent2_model': agent2_model,
        'agent1_wins': agent1_wins,
        'agent2_wins': agent2_wins,
        'ties': ties,
        'total_games': total_games,
        'agent1_win_rate': agent1_win_rate,
        'agent2_win_rate': agent2_wins / total_games if total_games > 0 else 0
    }

def collect_all_win_rates():
    """Collect win rates from all strong_models experiments"""
    base_dir = "/Users/joie/Desktop/bargain/experiments/results/4n=2"
    all_strong_model_dirs = glob.glob(f"{base_dir}/strong_models_20250819*")
    
    if not all_strong_model_dirs:
        print("No strong_models_20250819* directories found!")
        return []
    
    win_rate_data = []
    
    for dir_path in sorted(all_strong_model_dirs):
        print(f"\nProcessing {os.path.basename(dir_path)}...")
        results = load_run_results(dir_path)
        
        if not results:
            continue
            
        win_rate_info = calculate_win_rate(results)
        if win_rate_info:
            win_rate_data.append(win_rate_info)
    
    return win_rate_data

def create_win_rate_matrix(win_rate_data):
    """Create a matrix of win rates for the heatmap"""
    # Define model names in desired order
    model_names = ['claude-4-sonnet', 'gemini-2.5-pro', 'llama-3.1-405b-instruct', 'qwen-3-235b-a22b-2507']
    
    # Initialize matrix with NaN (will show as white/missing data)
    matrix = pd.DataFrame(index=model_names, columns=model_names, dtype=float)
    
    # Fill diagonal with 0.5 (tie against self)
    for model in model_names:
        matrix.loc[model, model] = 0.5
    
    # Fill matrix with win rates (NO symmetric filling - proposal order matters!)
    for data in win_rate_data:
        row_model = data['agent1_model']
        col_model = data['agent2_model']
        win_rate = data['agent2_win_rate']  # Showing agent2's win rate
        
        if row_model in model_names and col_model in model_names:
            matrix.loc[row_model, col_model] = win_rate
    
    return matrix

def create_heatmap(matrix, output_path='win_rate_heatmap.png'):
    """Create and save the heatmap"""
    plt.figure(figsize=(10, 8))
    
    # Create heatmap with custom colormap
    ax = sns.heatmap(
        matrix.astype(float), 
        annot=True, 
        cmap='RdYlBu_r',  # Red-Yellow-Blue reversed (red=high win rate, blue=low)
        center=0.5,  # Center colormap at 0.5 (tie)
        square=True, 
        linewidths=0.5,
        cbar_kws={'label': 'Agent #2 Win Rate'},
        fmt='.3f',  # Show 3 decimal places
        vmin=0, 
        vmax=1
    )
    
    plt.title('Strong Model Negotiation Agent #2 Win Rates\n(Row vs Column)', fontsize=16, pad=20)
    plt.xlabel('Agent #2 (Column)', fontsize=12)
    plt.ylabel('Agent #1 (Row)', fontsize=12)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nHeatmap saved to: {output_path}")
    
    # Display the plot
    plt.show()

def main():
    print("Collecting win rate data from all strong_models experiments...")
    
    # Collect all win rate data
    win_rate_data = collect_all_win_rates()
    
    if not win_rate_data:
        print("No win rate data found!")
        return
    
    print(f"\nCollected {len(win_rate_data)} model pair results:")
    for data in win_rate_data:
        print(f"  {data['agent1_model']} vs {data['agent2_model']}: "
              f"{data['agent1_win_rate']:.3f} win rate "
              f"({data['agent1_wins']}/{data['total_games']} games)")
    
    # Create win rate matrix
    print("\nCreating win rate matrix...")
    matrix = create_win_rate_matrix(win_rate_data)
    
    print("\nAgent #2 Win Rate Matrix:")
    print(matrix)
    
    # Create and save heatmap
    create_heatmap(matrix)

if __name__ == "__main__":
    main()
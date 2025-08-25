#!/usr/bin/env python3
"""
Create a heatmap of no consensus rates for strong models in negotiations
Based on create_win_rate_heatmap.py logic to parse experiment data
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

def calculate_no_consensus_rate(results):
    """Calculate no consensus rates for a pair of models"""
    if not results:
        return None
    
    agent1_name, agent2_name, agent1_model, agent2_model = get_agent_info(results)
    
    if not all([agent1_name, agent2_name, agent1_model, agent2_model]):
        print(f"Warning: Could not determine agent names for directory")
        return None
    
    no_consensus_count = 0
    total_games = len(results)
    
    print(f"  Analyzing: {agent1_model} vs {agent2_model} ({total_games} games)")
    
    for result in results:
        # Get final utilities from result
        final_utilities = result.get('final_utilities', {})
        
        # If no final utilities, this was a no-deal/no-consensus scenario
        if not final_utilities:
            no_consensus_count += 1
            continue
            
        # Check if both agents got 0 utility (also indicates no consensus)
        agent1_utility = final_utilities.get(agent1_name, 0)
        agent2_utility = final_utilities.get(agent2_name, 0)
        
        if agent1_utility == 0 and agent2_utility == 0:
            no_consensus_count += 1
    
    # Calculate no consensus rate
    no_consensus_rate = no_consensus_count / total_games if total_games > 0 else 0
    
    return {
        'agent1_model': agent1_model,
        'agent2_model': agent2_model,
        'no_consensus_count': no_consensus_count,
        'total_games': total_games,
        'no_consensus_rate': no_consensus_rate
    }

def collect_all_no_consensus_rates():
    """Collect no consensus rates from all strong_models experiments"""
    base_dir = "/Users/joie/Desktop/bargain/experiments/results/4n=2"
    all_strong_model_dirs = glob.glob(f"{base_dir}/strong_models_20250819*")
    
    if not all_strong_model_dirs:
        print("No strong_models_20250819* directories found!")
        return []
    
    no_consensus_data = []
    
    for dir_path in sorted(all_strong_model_dirs):
        print(f"\nProcessing {os.path.basename(dir_path)}...")
        results = load_run_results(dir_path)
        
        if not results:
            continue
            
        no_consensus_info = calculate_no_consensus_rate(results)
        if no_consensus_info:
            no_consensus_data.append(no_consensus_info)
    
    return no_consensus_data

def create_no_consensus_matrix(no_consensus_data):
    """Create a matrix of no consensus rates for the heatmap"""
    # Define model names in desired order
    model_names = ['claude-4-sonnet', 'gemini-2.5-pro', 'llama-3.1-405b-instruct', 'qwen-3-235b-a22b-2507']
    
    # Initialize matrix with NaN (will show as white/missing data)
    matrix = pd.DataFrame(index=model_names, columns=model_names, dtype=float)
    
    # Fill diagonal with same-model interaction rates (if available)
    for model in model_names:
        matrix.loc[model, model] = np.nan  # Will be filled if we have data
    
    # Fill matrix with no consensus rates (NO symmetric filling - proposal order matters!)
    for data in no_consensus_data:
        row_model = data['agent1_model']
        col_model = data['agent2_model']
        no_consensus_rate = data['no_consensus_rate']
        
        if row_model in model_names and col_model in model_names:
            matrix.loc[row_model, col_model] = no_consensus_rate
    
    return matrix

def create_heatmap(matrix, output_path='no_consensus_heatmap.png'):
    """Create and save the heatmap"""
    plt.figure(figsize=(10, 8))
    
    # Create heatmap with custom colormap
    ax = sns.heatmap(
        matrix.astype(float), 
        annot=True, 
        cmap='Reds',  # Red colormap (higher = more no consensus)
        square=True, 
        linewidths=0.5,
        cbar_kws={'label': 'No Consensus Rate'},
        fmt='.3f',  # Show 3 decimal places
        vmin=0, 
        vmax=1,
        mask=matrix.isna()  # Mask NaN values
    )
    
    plt.title('Strong Model Negotiation No Consensus Rates\n(Agent 1 vs Agent 2)', fontsize=16, pad=20)
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
    print("Collecting no consensus data from all strong_models experiments...")
    
    # Collect all no consensus data
    no_consensus_data = collect_all_no_consensus_rates()
    
    if not no_consensus_data:
        print("No consensus data found!")
        return
    
    print(f"\nCollected {len(no_consensus_data)} model pair results:")
    for data in no_consensus_data:
        print(f"  {data['agent1_model']} vs {data['agent2_model']}: "
              f"{data['no_consensus_rate']:.3f} no consensus rate "
              f"({data['no_consensus_count']}/{data['total_games']} games)")
    
    # Create no consensus matrix
    print("\nCreating no consensus matrix...")
    matrix = create_no_consensus_matrix(no_consensus_data)
    
    print("\nNo Consensus Rate Matrix:")
    print(matrix)
    
    # Create and save heatmap
    create_heatmap(matrix)

if __name__ == "__main__":
    main()
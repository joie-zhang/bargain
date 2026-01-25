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
from collections import defaultdict

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
    if 'gpt-4o-latest' in model_name or 'gpt_4o_latest' in model_name:
        return 'gpt-4o-2024-11-20'  # Latest is Nov 2024
    elif 'gpt-4o-mini' in model_name or 'gpt_4o_mini' in model_name:
        return 'gpt-4o-mini'
    elif model_name == 'gpt-4o-2024-05-13' or model_name == 'gpt-4o-may':
        return 'gpt-4o-2024-05-13'  # Baseline model (May 2024)
    elif model_name == 'gpt-4o-2024-11-20' or model_name == 'gpt-4o-nov':
        return 'gpt-4o-2024-11-20'  # Strong model (Nov 2024)
    elif model_name == 'gpt-4o' or 'gpt_4o' in model_name:
        # Plain gpt-4o defaults to the baseline May version
        return 'gpt-4o-2024-05-13'
    # Handle Claude variants
    elif 'claude-3-5-haiku' in model_name or 'claude_3_5_haiku' in model_name:
        return 'claude-3-5-haiku'
    elif model_name == 'claude-3-5-sonnet-20241022' or model_name == 'claude-3-5-sonnet' or 'claude_3_5_sonnet' in model_name:
        return 'claude-3-5-sonnet'
    elif 'claude-4-1-opus' in model_name or 'claude_4_1_opus' in model_name:
        return 'claude-4-1-opus'
    elif 'claude-4-sonnet' in model_name or 'claude_4_sonnet' in model_name:
        return 'claude-4-sonnet'
    elif model_name == 'claude-3-opus' or model_name == 'claude-3-opus-20240229' or 'claude_3_opus' in model_name:
        return 'claude-3-opus'
    # Handle Gemini variants
    elif model_name == 'gemini-1-5-pro-002' or model_name == 'gemini-1-5-pro' or 'gemini_1_5_pro' in model_name:
        return 'gemini-1-5-pro'
    elif model_name == 'gemini-2-0-flash-001' or model_name == 'gemini-2-0-flash' or 'gemini_2_0_flash' in model_name:
        return 'gemini-2-0-flash'
    elif model_name == 'gemini-2-5-pro-exp' or model_name == 'gemini-2-5-pro' or 'gemini_2_5_pro' in model_name:
        return 'gemini-2-5-pro'
    # Handle O1 and O3
    elif model_name == 'o1' or 'o1_' in model_name:
        return 'o1'
    elif model_name == 'o3' or 'o3_' in model_name:
        return 'o3'
    # Handle Gemma
    elif 'gemma-3-27b' in model_name or 'gemma_3_27b' in model_name:
        return 'gemma-3-27b'
    return model_name

def load_experiment_results(results_dir):
    """Load all experiment results from the results directory, organized by model pairs."""
    results_by_pairs = defaultdict(list)
    
    # Look for summary JSON files
    results_path = Path(results_dir)
    
    print("Scanning for experiment files...")
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
                        
                        # Extract model names
                        agent1 = agents[0] if len(agents) > 0 else None
                        agent2 = agents[1] if len(agents) > 1 else None
                        
                        agent1_model = normalize_model_name(extract_model_name(agent1))
                        agent2_model = normalize_model_name(extract_model_name(agent2))
                        
                        if agent1_model and agent2_model:
                            # Store the experiment data
                            results_by_pairs[(agent1_model, agent2_model)].append(exp)
                
                processed_files += 1
                if processed_files % 50 == 0:
                    print(f"Processed {processed_files} files...")
                    
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    print(f"Finished processing {processed_files} files.")
    return results_by_pairs

# Define the models we want to analyze
STRONG_MODELS_REQUESTED = [
    'claude-3-5-haiku', 'claude-3-5-sonnet', 'claude-4-1-opus', 'claude-4-sonnet',
    'gemma-3-27b', 'gemini-2-0-flash', 'gemini-2-5-pro',
    'gpt-4o-mini', 'gpt-4o-2024-11-20', 'o1', 'o3'
]

BASELINE_MODELS = ['gpt-4o-2024-05-13', 'gemini-1-5-pro', 'claude-3-opus']

# Model display names
MODEL_DISPLAY_NAMES = {
    'claude-3-5-haiku': 'Claude 3.5 Haiku',
    'claude-3-5-sonnet': 'Claude 3.5 Sonnet',
    'claude-4-1-opus': 'Claude 4.1 Opus',
    'claude-4-sonnet': 'Claude 4 Sonnet',
    'gemma-3-27b': 'Gemma 3 27B',
    'gemini-2-0-flash': 'Gemini 2.0 Flash',
    'gemini-2-5-pro': 'Gemini 2.5 Pro',
    'gpt-4o-mini': 'GPT-4o Mini',
    'gpt-4o-2024-11-20': 'GPT-4o (Nov 2024)',
    'o1': 'O1',
    'o3': 'O3',
    'gpt-4o-2024-05-13': 'GPT-4o (May 2024)',
    'gemini-1-5-pro': 'Gemini 1.5 Pro',
    'claude-3-opus': 'Claude 3 Opus'
}

def calculate_no_consensus_rate(experiments):
    """Calculate no consensus rates for a list of experiments"""
    if not experiments:
        return None
    
    no_consensus_count = 0
    total_games = len(experiments)
    
    for exp in experiments:
        # Get agent names from config
        agents = exp.get('config', {}).get('agents', [])
        if len(agents) < 2:
            continue
            
        agent1_name = agents[0]
        agent2_name = agents[1]
        
        # Get final utilities from result
        final_utilities = exp.get('final_utilities', {})
        
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
        'no_consensus_count': no_consensus_count,
        'total_games': total_games,
        'no_consensus_rate': no_consensus_rate
    }

def collect_all_no_consensus_rates():
    """Collect no consensus rates from all experiments"""
    results_dir = '/root/bargain/experiments/results_current'
    results_by_pairs = load_experiment_results(results_dir)
    
    if not results_by_pairs:
        print("No experiment data found!")
        return []
    
    no_consensus_data = []
    
    # Process each model pair
    for (model1, model2), experiments in results_by_pairs.items():
        # Check if this is a pair we're interested in
        # (strong vs strong, baseline vs baseline, or strong vs baseline)
        all_models = STRONG_MODELS_REQUESTED + BASELINE_MODELS
        
        if model1 in all_models and model2 in all_models:
            print(f"  Analyzing: {model1} vs {model2} ({len(experiments)} games)")
            
            no_consensus_info = calculate_no_consensus_rate(experiments)
            if no_consensus_info:
                no_consensus_info['agent1_model'] = model1
                no_consensus_info['agent2_model'] = model2
                no_consensus_data.append(no_consensus_info)
    
    return no_consensus_data

def create_no_consensus_matrix(no_consensus_data):
    """Create a matrix of no consensus rates for the heatmap"""
    # Combine all models for the matrix
    all_models = STRONG_MODELS_REQUESTED + BASELINE_MODELS
    
    # Get unique models that actually have data
    models_with_data = set()
    for data in no_consensus_data:
        models_with_data.add(data['agent1_model'])
        models_with_data.add(data['agent2_model'])
    
    # Filter to only models we have data for
    model_names = [m for m in all_models if m in models_with_data]
    
    if not model_names:
        print("No models found with data!")
        return pd.DataFrame()
    
    # Initialize matrix with NaN (will show as white/missing data)
    matrix = pd.DataFrame(index=model_names, columns=model_names, dtype=float)
    
    # Fill matrix with no consensus rates (NO symmetric filling - proposal order matters!)
    for data in no_consensus_data:
        row_model = data['agent1_model']
        col_model = data['agent2_model']
        no_consensus_rate = data['no_consensus_rate']
        
        if row_model in model_names and col_model in model_names:
            matrix.loc[row_model, col_model] = no_consensus_rate
    
    return matrix

def create_heatmap(matrix, output_path='figures/no_consensus_heatmap_wonky.png'):
    """Create and save the heatmap"""
    if matrix.empty:
        print("Cannot create heatmap - no data available")
        return
        
    plt.figure(figsize=(14, 10))
    
    # Get display names for models
    row_labels = [MODEL_DISPLAY_NAMES.get(m, m) for m in matrix.index]
    col_labels = [MODEL_DISPLAY_NAMES.get(m, m) for m in matrix.columns]
    
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
        mask=matrix.isna(),  # Mask NaN values
        xticklabels=col_labels,
        yticklabels=row_labels
    )
    
    plt.title('Model Negotiation No Consensus Rates\n(Agent 1 vs Agent 2)', fontsize=16, pad=20)
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
        agent1_display = MODEL_DISPLAY_NAMES.get(data['agent1_model'], data['agent1_model'])
        agent2_display = MODEL_DISPLAY_NAMES.get(data['agent2_model'], data['agent2_model'])
        print(f"  {agent1_display} vs {agent2_display}: "
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
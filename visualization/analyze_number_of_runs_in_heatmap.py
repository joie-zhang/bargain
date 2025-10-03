#!/usr/bin/env python3
"""
Analyze and display statistics for the MMLU-ordered heatmaps.
Shows the number of runs/datapoints for each cell in the heatmaps.
"""

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict
from tabulate import tabulate

# MMLU-Pro scores from the provided data
MMLU_PRO_SCORES = {
    # Strong models we're looking for
    'claude-3-5-haiku': 64.1,
    'claude-3-5-sonnet': 78.4,
    'claude-4-1-opus': 87.8,
    'claude-4-sonnet': 79.4,
    'gemma-3-27b': 67.5,
    'gemini-2-0-flash': 77.4,
    'gemini-2-5-pro': 84.1,
    'gpt-4o-mini': 62.7,
    'gpt-4o-2024-11-20': 69.1,
    'o1': 83.5,
    'o3': 85.6,
    
    # Baseline models
    'gpt-4o-2024-05-13': 72.55,
    'gemini-1-5-pro': 75.3,
    'claude-3-opus': 68.45,
}

# Define the models we want to include
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
    'gpt-4o-2024-11-20': 'GPT-4o (Nov)',
    'o1': 'O1',
    'o3': 'O3',
    'gpt-4o-2024-05-13': 'GPT-4o (May)',
    'gemini-1-5-pro': 'Gemini 1.5 Pro',
    'claude-3-opus': 'Claude 3 Opus'
}

# Competition levels for y-axis
COMPETITION_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]

def extract_model_name(agent_name):
    """Extract model name from agent ID like 'claude_3_opus_1' -> 'claude-3-opus'"""
    if not agent_name:
        return None
    parts = agent_name.split('_')
    if len(parts) > 1 and parts[-1].isdigit():
        parts = parts[:-1]
    return '-'.join(parts)

def normalize_model_name(model_name):
    """Normalize model names to match our target set"""
    # Handle GPT-4o variants
    if model_name == 'gpt-4o-2024-05-13' or model_name == 'gpt-4o' or model_name == 'gpt-4o-may':
        return 'gpt-4o-2024-05-13'  # Baseline model (May 2024)
    elif model_name == 'gpt-4o-2024-11-20' or model_name == 'gpt-4o-nov':
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
    return min(COMPETITION_LEVELS, key=lambda x: abs(x - comp_level))

def load_experiment_results(results_dir):
    """Load all experiment results from the results directory."""
    results_by_competition = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    results_path = Path(results_dir)
    
    print("Scanning for experiment files...")
    processed_files = 0
    
    for file_path in results_path.glob('*_summary.json'):  # Only files in results dir
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if 'experiments' in data and data['experiments']:
                for exp in data['experiments']:
                    if 'config' in exp and 'agents' in exp['config']:
                        agents = exp['config']['agents']
                        
                        comp_level = exp['config'].get('competition_level', None)
                        comp_bucket = categorize_competition_level(comp_level)
                        
                        if comp_bucket is None:
                            continue
                        
                        agent1 = agents[0] if len(agents) > 0 else None
                        agent2 = agents[1] if len(agents) > 1 else None
                        
                        agent1_model = normalize_model_name(extract_model_name(agent1))
                        agent2_model = normalize_model_name(extract_model_name(agent2))
                        
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
                            final_utilities = exp.get('final_utilities', {})
                            
                            if final_utilities and baseline_agent and strong_agent:
                                baseline_utility = final_utilities.get(baseline_agent, 0)
                                strong_utility = final_utilities.get(strong_agent, 0)
                                
                                utility_diff = strong_utility - baseline_utility
                                
                                results_by_competition[comp_bucket][baseline_model][strong_model].append({
                                    'utility_diff': utility_diff,
                                    'baseline_utility': baseline_utility,
                                    'strong_utility': strong_utility
                                })
                
                processed_files += 1
                    
        except Exception as e:
            continue
    
    print(f"Finished processing {processed_files} files.")
    return results_by_competition

def get_models_with_data(results_by_competition):
    """Get the list of strong models that actually have data, ordered by MMLU-Pro score."""
    models_with_data = set()
    for comp_level in results_by_competition:
        for baseline in results_by_competition[comp_level]:
            for strong in results_by_competition[comp_level][baseline]:
                if results_by_competition[comp_level][baseline][strong]:
                    models_with_data.add(strong)
    
    models_with_scores = []
    for model in models_with_data:
        if model in MMLU_PRO_SCORES and MMLU_PRO_SCORES[model] is not None:
            models_with_scores.append((model, MMLU_PRO_SCORES[model]))
    
    models_with_scores.sort(key=lambda x: x[1])
    
    return [model for model, score in models_with_scores]

def create_count_matrix(results_by_competition, baseline_model, ordered_strong_models):
    """Create a matrix showing the number of runs for each cell."""
    data = np.full((len(COMPETITION_LEVELS), len(ordered_strong_models)), 0, dtype=int)
    
    for i, comp_level in enumerate(COMPETITION_LEVELS):
        for j, strong_model in enumerate(ordered_strong_models):
            if (comp_level in results_by_competition and 
                baseline_model in results_by_competition[comp_level] and
                strong_model in results_by_competition[comp_level][baseline_model]):
                
                experiments = results_by_competition[comp_level][baseline_model][strong_model]
                data[i, j] = len(experiments)
    
    return data

def print_detailed_statistics(results_by_competition, ordered_strong_models):
    """Print detailed statistics for each heatmap."""
    
    print("\n" + "="*80)
    print("HEATMAP CELL STATISTICS - NUMBER OF RUNS PER CELL")
    print("="*80)
    
    # Get baseline models that have data
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
    
    # Print adversary models with their MMLU scores (column headers)
    print("\nAdversary Models (ordered by MMLU-Pro score, low to high):")
    for i, model in enumerate(ordered_strong_models):
        score = MMLU_PRO_SCORES[model]
        print(f"  {i+1:2d}. {MODEL_DISPLAY_NAMES[model]:<20} (MMLU-Pro: {score}%)")
    
    # For each baseline, create a detailed table
    for baseline in baselines_with_data:
        print("\n" + "-"*80)
        print(f"\nHEATMAP: {MODEL_DISPLAY_NAMES[baseline]} vs Adversary Models")
        if baseline in MMLU_PRO_SCORES and MMLU_PRO_SCORES[baseline] is not None:
            print(f"Baseline MMLU-Pro Score: {MMLU_PRO_SCORES[baseline]}%")
        print("-"*80)
        
        # Create count matrix
        count_matrix = create_count_matrix(results_by_competition, baseline, ordered_strong_models)
        
        # Create table data
        table_data = []
        
        # Header row with shortened model names
        header = ["Competition"] + [MODEL_DISPLAY_NAMES[m].replace('\n', ' ').replace('Gemini', 'Gem').replace('Claude', 'Cl') 
                                   for m in ordered_strong_models]
        
        # Data rows
        for i, comp_level in enumerate(COMPETITION_LEVELS):
            row = [f"{comp_level:.2f}"]
            for j, strong_model in enumerate(ordered_strong_models):
                count = count_matrix[i, j]
                if count > 0:
                    row.append(str(count))
                else:
                    row.append("-")
            table_data.append(row)
        
        # Add summary row
        summary_row = ["TOTAL"]
        for j in range(len(ordered_strong_models)):
            col_sum = np.sum(count_matrix[:, j])
            if col_sum > 0:
                summary_row.append(f"{col_sum}")
            else:
                summary_row.append("-")
        table_data.append(summary_row)
        
        # Print table
        print(tabulate(table_data, headers=header, tablefmt="grid", stralign="center"))
        
        # Print statistics for this baseline
        total_cells = count_matrix.size
        non_zero_cells = np.count_nonzero(count_matrix)
        total_runs = np.sum(count_matrix)
        
        if non_zero_cells > 0:
            avg_runs_per_cell = total_runs / non_zero_cells
            min_runs = np.min(count_matrix[count_matrix > 0])
            max_runs = np.max(count_matrix)
            median_runs = np.median(count_matrix[count_matrix > 0])
        else:
            avg_runs_per_cell = 0
            min_runs = 0
            max_runs = 0
            median_runs = 0
        
        print(f"\nStatistics for {MODEL_DISPLAY_NAMES[baseline]}:")
        print(f"  Total cells in heatmap: {total_cells}")
        print(f"  Cells with data: {non_zero_cells} ({100*non_zero_cells/total_cells:.1f}%)")
        print(f"  Cells without data: {total_cells - non_zero_cells} ({100*(total_cells - non_zero_cells)/total_cells:.1f}%)")
        print(f"  Total experiments: {total_runs}")
        if non_zero_cells > 0:
            print(f"  Average runs per cell (with data): {avg_runs_per_cell:.1f}")
            print(f"  Median runs per cell (with data): {median_runs:.0f}")
            print(f"  Min runs in a cell: {min_runs}")
            print(f"  Max runs in a cell: {max_runs}")
    
    # Overall statistics
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    
    total_experiments = 0
    all_counts = []
    
    for comp_level in COMPETITION_LEVELS:
        for baseline in baselines_with_data:
            for strong in ordered_strong_models:
                if (comp_level in results_by_competition and 
                    baseline in results_by_competition[comp_level] and
                    strong in results_by_competition[comp_level][baseline]):
                    
                    experiments = results_by_competition[comp_level][baseline][strong]
                    count = len(experiments)
                    if count > 0:
                        all_counts.append(count)
                        total_experiments += count
    
    if all_counts:
        print(f"\nTotal experiments across all heatmaps: {total_experiments}")
        print(f"Total unique baseline-adversary-competition combinations: {len(all_counts)}")
        print(f"Average experiments per combination: {np.mean(all_counts):.1f}")
        print(f"Median experiments per combination: {np.median(all_counts):.0f}")
        print(f"Min experiments per combination: {min(all_counts)}")
        print(f"Max experiments per combination: {max(all_counts)}")
        
        # Distribution of run counts
        print("\nDistribution of run counts:")
        from collections import Counter
        count_distribution = Counter(all_counts)
        for count in sorted(count_distribution.keys())[:10]:  # Show first 10 unique counts
            occurrences = count_distribution[count]
            print(f"  {count:3d} runs: {occurrences:3d} cells ({100*occurrences/len(all_counts):5.1f}%)")
        if len(count_distribution) > 10:
            print(f"  ... and {len(count_distribution)-10} more unique count values")

def main():
    """Main function to analyze heatmap statistics."""
    print("Loading experiment results...")
    results_dir = '/root/bargain/experiments/results_current'
    results_by_competition = load_experiment_results(results_dir)
    
    # Get models that have data, ordered by MMLU-Pro score
    ordered_strong_models = get_models_with_data(results_by_competition)
    
    if not ordered_strong_models:
        print("No strong models found with both data and MMLU-Pro scores!")
        return
    
    print(f"Found {len(ordered_strong_models)} strong models with data and MMLU-Pro scores.")
    
    # Print detailed statistics
    print_detailed_statistics(results_by_competition, ordered_strong_models)

if __name__ == "__main__":
    main()
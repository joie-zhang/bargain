#!/usr/bin/env python3
"""Analyze negotiation results across different competition levels."""

import json
import os
from pathlib import Path
import numpy as np
from scipy import stats
import pandas as pd
from collections import defaultdict

def parse_folder_name(folder_name):
    """Extract model names, config, run, and competition level from folder name."""
    # Example: gpt-4o_vs_grok-4-0709_config000_run1_comp0_0
    parts = folder_name.split('_')

    # Find the position of 'vs'
    vs_index = parts.index('vs')

    # Model A is everything before 'vs'
    model_a = '_'.join(parts[:vs_index])

    # Find where config starts
    config_start = None
    for i, part in enumerate(parts):
        if part.startswith('config'):
            config_start = i
            break

    # Model B is everything between 'vs' and 'config'
    model_b = '_'.join(parts[vs_index + 1:config_start])

    # Extract config number
    config_num = int(parts[config_start].replace('config', ''))

    # Extract run number
    run_num = int(parts[config_start + 1].replace('run', ''))

    # Extract competition level (comp0_0 -> 0.0, comp0_25 -> 0.25, etc.)
    comp_str = '_'.join(parts[config_start + 2:])
    comp_level = float(comp_str.replace('comp', '').replace('_', '.'))

    return {
        'model_a': model_a,
        'model_b': model_b,
        'config': config_num,
        'run': run_num,
        'competition_level': comp_level
    }

def load_experiment_results(folder_path):
    """Load experiment results from a folder."""
    json_path = folder_path / 'experiment_results.json'
    if json_path.exists():
        with open(json_path, 'r') as f:
            return json.load(f)
    return None

def analyze_results(filter_last_round=False):
    """Main analysis function."""
    results_dir = Path('/root/bargain/experiments/results')

    # Collect all results
    all_results = []

    for folder in results_dir.iterdir():
        if folder.is_dir() and '_vs_' in folder.name and 'comp' in folder.name:
            try:
                folder_info = parse_folder_name(folder.name)
                experiment_data = load_experiment_results(folder)

                if experiment_data and 'final_utilities' in experiment_data:
                    result = {
                        **folder_info,
                        'utility_alpha': experiment_data['final_utilities'].get('Agent_Alpha', None),
                        'utility_beta': experiment_data['final_utilities'].get('Agent_Beta', None),
                        'consensus': experiment_data.get('consensus_reached', False),
                        'final_round': experiment_data.get('final_round', None),
                        't_rounds': experiment_data.get('config', {}).get('t_rounds', 10)
                    }
                    all_results.append(result)
            except Exception as e:
                print(f"Error processing {folder.name}: {e}")

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    if filter_last_round:
        # Filter out negotiations that reached consensus on the final round
        initial_count = len(df)
        df_filtered = df[(df['final_round'] != df['t_rounds']) | (df['consensus'] == False)]
        filtered_count = initial_count - len(df_filtered)
        print(f"Filtered out {filtered_count} negotiations that reached consensus on the final round")
        df = df_filtered

    # Group by model pair and competition level
    grouped_results = defaultdict(lambda: defaultdict(list))

    for _, row in df.iterrows():
        key = (row['model_a'], row['model_b'])
        comp_level = row['competition_level']

        if row['utility_alpha'] is not None and row['utility_beta'] is not None:
            grouped_results[key][comp_level].append({
                'utility_alpha': row['utility_alpha'],
                'utility_beta': row['utility_beta'],
                'run': row['run']
            })

    # Calculate statistics with confidence intervals
    print("=" * 80)
    print("NEGOTIATION RESULTS ANALYSIS")
    print("=" * 80)

    for (model_a, model_b), comp_data in sorted(grouped_results.items()):
        print(f"\n### {model_a} (Agent Alpha) vs {model_b} (Agent Beta)")
        print("-" * 60)

        comp_levels = sorted(comp_data.keys())

        for comp_level in comp_levels:
            runs = comp_data[comp_level]
            n_runs = len(runs)

            if n_runs > 0:
                utilities_alpha = [r['utility_alpha'] for r in runs]
                utilities_beta = [r['utility_beta'] for r in runs]

                # Calculate mean and confidence interval
                mean_alpha = np.mean(utilities_alpha)
                mean_beta = np.mean(utilities_beta)

                if n_runs > 1:
                    # 95% confidence interval
                    ci_alpha = stats.sem(utilities_alpha) * stats.t.ppf(0.975, n_runs - 1)
                    ci_beta = stats.sem(utilities_beta) * stats.t.ppf(0.975, n_runs - 1)
                else:
                    ci_alpha = 0
                    ci_beta = 0

                print(f"\nCompetition Level: {comp_level}")
                print(f"  Runs: {n_runs}")
                print(f"  {model_a} (Alpha): {mean_alpha:.2f} ± {ci_alpha:.2f}")
                print(f"    Raw values: {utilities_alpha}")
                print(f"  {model_b} (Beta):  {mean_beta:.2f} ± {ci_beta:.2f}")
                print(f"    Raw values: {utilities_beta}")
                print(f"  Total utility: {mean_alpha + mean_beta:.2f}")

    # Create summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE (WITH 95% CONFIDENCE INTERVALS)")
    print("=" * 80)

    for (model_a, model_b) in sorted(grouped_results.keys()):
        print(f"\n### {model_a} vs {model_b}")
        print("-" * 60)
        print(f"{'Competition':<15} {'Alpha ('+model_a+')':<25} {'Beta ('+model_b+')':<25} {'Total':<15}")
        print("-" * 60)

        comp_data = grouped_results[(model_a, model_b)]
        for comp_level in sorted(comp_data.keys()):
            runs = comp_data[comp_level]
            if len(runs) >= 5:  # Only include if we have 5 runs
                utilities_alpha = [r['utility_alpha'] for r in runs]
                utilities_beta = [r['utility_beta'] for r in runs]

                # Filter out NaN values
                utilities_alpha = [u for u in utilities_alpha if not np.isnan(u)]
                utilities_beta = [u for u in utilities_beta if not np.isnan(u)]

                if utilities_alpha and utilities_beta:
                    mean_alpha = np.mean(utilities_alpha)
                    mean_beta = np.mean(utilities_beta)

                    if len(utilities_alpha) > 1:
                        ci_alpha = stats.sem(utilities_alpha) * stats.t.ppf(0.975, len(utilities_alpha) - 1)
                    else:
                        ci_alpha = 0

                    if len(utilities_beta) > 1:
                        ci_beta = stats.sem(utilities_beta) * stats.t.ppf(0.975, len(utilities_beta) - 1)
                    else:
                        ci_beta = 0

                    print(f"{comp_level:<15.2f} {mean_alpha:>6.1f} ± {ci_alpha:<6.1f}       "
                          f"{mean_beta:>6.1f} ± {ci_beta:<6.1f}       {mean_alpha + mean_beta:>6.1f}")
                else:
                    print(f"{comp_level:<15.2f} {'No consensus reached':^55}")

    return df, grouped_results

if __name__ == "__main__":
    import sys

    # Check for command-line argument
    filter_last_round = '--filter-last-round' in sys.argv or '-f' in sys.argv

    if filter_last_round:
        print("=" * 80)
        print("FILTERING MODE: Excluding negotiations that reached consensus on final round")
        print("=" * 80)

    df, grouped_results = analyze_results(filter_last_round=filter_last_round)

    # Save raw data to CSV for further analysis
    output_file = 'negotiation_results_analysis_filtered.csv' if filter_last_round else 'negotiation_results_analysis.csv'
    df.to_csv(f'/root/bargain/{output_file}', index=False)
    print(f"\n\nRaw data saved to: {output_file}")
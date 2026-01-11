#!/usr/bin/env python3
"""Create visualizations for negotiation results analysis."""

import json
import os
import sys
import shutil
from pathlib import Path
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from analyze_negotiation_results import parse_folder_name, load_experiment_results

def collect_results(filter_last_round=False):
    """Collect all negotiation results.

    Args:
        filter_last_round: If True, exclude negotiations that reached consensus on round 10
    """
    results_dir = Path('/root/bargain/experiments/results')
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

    df = pd.DataFrame(all_results)

    if filter_last_round:
        # Filter out negotiations that reached consensus on the final round
        initial_count = len(df)
        df = df[(df['final_round'] != df['t_rounds']) | (df['consensus'] == False)]
        filtered_count = initial_count - len(df)
        print(f"Filtered out {filtered_count} negotiations that reached consensus on the final round")

    return df

def create_utility_plots(df):
    """Create utility plots for each model pairing."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    model_pairs = [
        ('gpt-4o', 'grok-3'),
        ('gpt-4o', 'grok-3-mini'),
        ('gpt-4o', 'grok-4-0709')
    ]

    for idx, (model_a, model_b) in enumerate(model_pairs):
        ax = axes[idx]

        # Filter data for this model pair
        pair_data = df[(df['model_a'] == model_a) & (df['model_b'] == model_b)]

        if not pair_data.empty:
            # Group by competition level
            grouped = pair_data.groupby('competition_level').agg({
                'utility_alpha': ['mean', 'std', 'count'],
                'utility_beta': ['mean', 'std', 'count']
            }).reset_index()

            comp_levels = grouped['competition_level'].values

            # Calculate means and standard errors
            alpha_means = grouped['utility_alpha']['mean'].values
            alpha_sems = grouped['utility_alpha']['std'].values / np.sqrt(grouped['utility_alpha']['count'].values)

            beta_means = grouped['utility_beta']['mean'].values
            beta_sems = grouped['utility_beta']['std'].values / np.sqrt(grouped['utility_beta']['count'].values)

            # Plot with error bars
            ax.errorbar(comp_levels, alpha_means, yerr=alpha_sems * 1.96,
                       label=f'{model_a} (Alpha)', marker='o', capsize=5, linewidth=2)
            ax.errorbar(comp_levels, beta_means, yerr=beta_sems * 1.96,
                       label=f'{model_b} (Beta)', marker='s', capsize=5, linewidth=2)

            ax.set_xlabel('Competition Level', fontsize=12)
            ax.set_ylabel('Utility', fontsize=12)
            ax.set_title(f'{model_a} vs {model_b}', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 120])

    plt.suptitle('Negotiation Utilities Across Competition Levels (95% CI)', fontsize=16)
    plt.tight_layout()
    plt.savefig('/root/bargain/negotiation_utilities_plot.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved plot to: negotiation_utilities_plot.png")

def create_advantage_plot(df):
    """Create a plot showing relative advantage of each model."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    model_pairs = [
        ('gpt-4o', 'grok-3'),
        ('gpt-4o', 'grok-3-mini'),
        ('gpt-4o', 'grok-4-0709')
    ]

    colors = ['blue', 'green', 'red']
    markers = ['o', 's', '^']

    for idx, (model_a, model_b) in enumerate(model_pairs):
        # Filter data for this model pair
        pair_data = df[(df['model_a'] == model_a) & (df['model_b'] == model_b)]

        if not pair_data.empty:
            # Calculate advantage (alpha_utility - beta_utility)
            pair_data['advantage'] = pair_data['utility_alpha'] - pair_data['utility_beta']

            # Group by competition level
            grouped = pair_data.groupby('competition_level')['advantage'].agg(['mean', 'std', 'count']).reset_index()

            comp_levels = grouped['competition_level'].values
            advantages = grouped['mean'].values
            sems = grouped['std'].values / np.sqrt(grouped['count'].values)

            # Plot with error bars
            ax.errorbar(comp_levels, advantages, yerr=sems * 1.96,
                       label=f'{model_a} vs {model_b}', marker=markers[idx],
                       color=colors[idx], capsize=5, linewidth=2)

    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Competition Level', fontsize=12)
    ax.set_ylabel('Utility Advantage (Alpha - Beta)', fontsize=12)
    ax.set_title('Model Advantage Across Competition Levels', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/root/bargain/negotiation_advantage_plot.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved plot to: negotiation_advantage_plot.png")

def create_total_utility_plot(df):
    """Create a plot showing total utility (efficiency) across competition levels."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    model_pairs = [
        ('gpt-4o', 'grok-3'),
        ('gpt-4o', 'grok-3-mini'),
        ('gpt-4o', 'grok-4-0709')
    ]

    colors = ['blue', 'green', 'red']
    markers = ['o', 's', '^']

    for idx, (model_a, model_b) in enumerate(model_pairs):
        # Filter data for this model pair
        pair_data = df[(df['model_a'] == model_a) & (df['model_b'] == model_b)]

        if not pair_data.empty:
            # Calculate total utility
            pair_data['total_utility'] = pair_data['utility_alpha'] + pair_data['utility_beta']

            # Group by competition level
            grouped = pair_data.groupby('competition_level')['total_utility'].agg(['mean', 'std', 'count']).reset_index()

            comp_levels = grouped['competition_level'].values
            totals = grouped['mean'].values
            sems = grouped['std'].values / np.sqrt(grouped['count'].values)

            # Plot with error bars
            ax.errorbar(comp_levels, totals, yerr=sems * 1.96,
                       label=f'{model_a} vs {model_b}', marker=markers[idx],
                       color=colors[idx], capsize=5, linewidth=2)

    ax.axhline(y=200, color='black', linestyle='--', alpha=0.5, label='Maximum possible')
    ax.set_xlabel('Competition Level', fontsize=12)
    ax.set_ylabel('Total Utility (Alpha + Beta)', fontsize=12)
    ax.set_title('Negotiation Efficiency Across Competition Levels', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([80, 210])

    plt.tight_layout()
    plt.savefig('/root/bargain/negotiation_efficiency_plot.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved plot to: negotiation_efficiency_plot.png")

def print_statistical_tests(df):
    """Perform statistical tests on the results."""
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)

    model_pairs = [
        ('gpt-4o', 'grok-3'),
        ('gpt-4o', 'grok-3-mini'),
        ('gpt-4o', 'grok-4-0709')
    ]

    for model_a, model_b in model_pairs:
        print(f"\n### {model_a} vs {model_b}")
        print("-"*60)

        pair_data = df[(df['model_a'] == model_a) & (df['model_b'] == model_b)]

        for comp_level in sorted(pair_data['competition_level'].unique()):
            level_data = pair_data[pair_data['competition_level'] == comp_level]

            alpha_utils = level_data['utility_alpha'].dropna()
            beta_utils = level_data['utility_beta'].dropna()

            if len(alpha_utils) >= 5 and len(beta_utils) >= 5:
                # Paired t-test (assuming same runs)
                if len(alpha_utils) == len(beta_utils):
                    t_stat, p_value = stats.ttest_rel(alpha_utils, beta_utils)
                    print(f"\nCompetition {comp_level:.2f}:")
                    print(f"  Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")

                    if p_value < 0.05:
                        if t_stat > 0:
                            print(f"  → {model_a} significantly outperforms {model_b}")
                        else:
                            print(f"  → {model_b} significantly outperforms {model_a}")
                    else:
                        print(f"  → No significant difference")

                # Effect size (Cohen's d)
                mean_diff = alpha_utils.mean() - beta_utils.mean()
                pooled_std = np.sqrt((alpha_utils.std()**2 + beta_utils.std()**2) / 2)
                cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                print(f"  Cohen's d: {cohens_d:.3f} ({'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'})")

if __name__ == "__main__":
    import sys

    # Check for command-line argument
    filter_last_round = '--filter-last-round' in sys.argv or '-f' in sys.argv

    if filter_last_round:
        print("=" * 80)
        print("FILTERING MODE: Excluding negotiations that reached consensus on final round")
        print("=" * 80)

    # Collect results
    df = collect_results(filter_last_round=filter_last_round)

    print(f"\nTotal negotiations analyzed: {len(df)}")
    print(f"Negotiations by final round:")
    print(df.groupby('final_round')['final_round'].count().sort_index())

    # Create visualizations
    create_utility_plots(df)
    create_advantage_plot(df)
    create_total_utility_plot(df)

    # Statistical tests
    print_statistical_tests(df)

    if filter_last_round:
        print("\n📊 Plots saved with '_filtered' suffix")
        # Rename the output files to indicate they're filtered
        import shutil
        for plot_name in ['negotiation_utilities_plot.png', 'negotiation_advantage_plot.png', 'negotiation_efficiency_plot.png']:
            if os.path.exists(f'/root/bargain/{plot_name}'):
                filtered_name = plot_name.replace('.png', '_filtered.png')
                shutil.move(f'/root/bargain/{plot_name}', f'/root/bargain/{filtered_name}')
                print(f"  → {filtered_name}")

    print("\n✓ Analysis complete! Check the generated plots.")
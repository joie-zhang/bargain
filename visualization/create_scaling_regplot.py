#!/usr/bin/env python3
"""
Create regression plots showing scaling laws between model capability and negotiation outcomes.
Uses seaborn regplot (with statsmodels fallback) for scatter plots with regression lines.

X-axis: MMLU-Pro score (model capability proxy)
Y-axis: Utility difference (Strong model - Baseline model)
"""

import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd
from scipy import stats

# Try to import statsmodels for more detailed regression statistics
try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import summary_table
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not available. Using scipy for regression.")

# MMLU-Pro scores (same as in create_mmlu_ordered_heatmaps.py)
MMLU_PRO_SCORES = {
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
    'gpt-5-mini': 83.7,
    'gpt-5-nano': 78.0,
    'gpt-4o-2024-05-13': 72.55,
    'gemini-1-5-pro': 75.3,
    'claude-3-opus': 68.45,
}

# Model display names (for labels)
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
    'gpt-5-nano': 'GPT-5 Nano',
    'gpt-5-mini': 'GPT-5 Mini',
    'o1': 'O1',
    'o3': 'O3',
    'gpt-4o-2024-05-13': 'GPT-4o (May 2024)',
    'gemini-1-5-pro': 'Gemini 1.5 Pro',
    'claude-3-opus': 'Claude 3 Opus'
}

# Baseline and strong model definitions
BASELINE_MODELS = ['gpt-4o-2024-05-13', 'gemini-1-5-pro', 'claude-3-opus']
STRONG_MODELS = [
    'claude-3-5-haiku', 'claude-3-5-sonnet', 'claude-4-1-opus', 'claude-4-sonnet',
    'gemini-2-0-flash', 'gemini-2-5-pro',
    'gpt-4o-2024-11-20', 'o1', 'o3', 'gpt-5-nano', 'gpt-5-mini'
]

COMPETITION_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]


def extract_model_name(agent_name):
    """Extract model name from agent ID."""
    if not agent_name:
        return None
    parts = agent_name.split('_')
    if len(parts) > 1 and parts[-1].isdigit():
        parts = parts[:-1]
    return '-'.join(parts)


def normalize_model_name(model_name):
    """Normalize model names to match our target set."""
    if model_name == 'gpt-4o-2024-05-13' or model_name == 'gpt-4o' or model_name == 'gpt-4o-may':
        return 'gpt-4o-2024-05-13'
    elif model_name == 'gpt-4o-2024-11-20' or model_name == 'gpt-4o-nov' or model_name == 'gpt-4o-latest':
        return 'gpt-4o-2024-11-20'
    elif model_name == 'claude-3-5-sonnet-20241022' or model_name == 'claude-3-5-sonnet':
        return 'claude-3-5-sonnet'
    elif model_name == 'claude-3-opus' or model_name == 'claude-3-opus-20240229':
        return 'claude-3-opus'
    elif model_name == 'gemini-1-5-pro-002' or model_name == 'gemini-1-5-pro':
        return 'gemini-1-5-pro'
    elif model_name == 'gemini-2-0-flash-001' or model_name == 'gemini-2-0-flash':
        return 'gemini-2-0-flash'
    elif model_name == 'gemini-2-5-pro-exp' or model_name == 'gemini-2-5-pro':
        return 'gemini-2-5-pro'
    return model_name


def load_experiment_data(results_dir):
    """Load experiment data and organize for regression analysis."""
    data_records = []
    results_path = Path(results_dir)

    print("Scanning for experiment files...")
    processed_files = 0

    for file_path in results_path.glob('**/*_summary.json'):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            if 'experiments' in data and data['experiments']:
                for exp in data['experiments']:
                    if 'config' in exp and 'agents' in exp['config']:
                        agents = exp['config']['agents']
                        comp_level = exp['config'].get('competition_level', None)

                        if comp_level is None:
                            continue

                        # Round to nearest competition level
                        comp_level = min(COMPETITION_LEVELS, key=lambda x: abs(x - comp_level))

                        # Extract model names
                        agent1 = agents[0] if len(agents) > 0 else None
                        agent2 = agents[1] if len(agents) > 1 else None

                        agent1_model = normalize_model_name(extract_model_name(agent1))
                        agent2_model = normalize_model_name(extract_model_name(agent2))

                        # Identify baseline and strong model
                        baseline_model = None
                        strong_model = None
                        baseline_agent = None
                        strong_agent = None

                        if agent1_model in BASELINE_MODELS and agent2_model in STRONG_MODELS:
                            baseline_model, strong_model = agent1_model, agent2_model
                            baseline_agent, strong_agent = agent1, agent2
                        elif agent2_model in BASELINE_MODELS and agent1_model in STRONG_MODELS:
                            baseline_model, strong_model = agent2_model, agent1_model
                            baseline_agent, strong_agent = agent2, agent1

                        if baseline_model and strong_model:
                            final_utilities = exp.get('final_utilities', {})

                            if final_utilities and baseline_agent and strong_agent:
                                baseline_utility = final_utilities.get(baseline_agent, 0)
                                strong_utility = final_utilities.get(strong_agent, 0)
                                utility_diff = strong_utility - baseline_utility
                                model_order = exp['config'].get('model_order', 'unknown')

                                # Get MMLU-Pro score for the strong model
                                mmlu_score = MMLU_PRO_SCORES.get(strong_model, None)

                                if mmlu_score is not None:
                                    data_records.append({
                                        'baseline_model': baseline_model,
                                        'strong_model': strong_model,
                                        'mmlu_score': mmlu_score,
                                        'competition_level': comp_level,
                                        'utility_diff': utility_diff,
                                        'baseline_utility': baseline_utility,
                                        'strong_utility': strong_utility,
                                        'model_order': model_order
                                    })

                processed_files += 1
                if processed_files % 50 == 0:
                    print(f"Processed {processed_files} files...")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    print(f"Finished processing {processed_files} files.")
    print(f"Total records: {len(data_records)}")

    return pd.DataFrame(data_records)


def compute_regression_stats(x, y):
    """Compute regression statistics using statsmodels or scipy."""
    if HAS_STATSMODELS:
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        return {
            'slope': model.params[1],
            'intercept': model.params[0],
            'r_squared': model.rsquared,
            'p_value': model.pvalues[1],
            'conf_int': model.conf_int().iloc[1].values,  # CI for slope
            'n': len(x)
        }
    else:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        # Approximate 95% CI for slope
        t_crit = stats.t.ppf(0.975, len(x) - 2)
        ci_low = slope - t_crit * std_err
        ci_high = slope + t_crit * std_err
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'conf_int': [ci_low, ci_high],
            'n': len(x)
        }


def create_regplot_by_competition(df, output_dir='figures'):
    """Create regression plots for each competition level."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, len(COMPETITION_LEVELS), figsize=(20, 5), sharey=True)

    regression_stats = {}

    for idx, comp_level in enumerate(COMPETITION_LEVELS):
        ax = axes[idx]
        subset = df[df['competition_level'] == comp_level]

        if len(subset) < 3:
            ax.set_title(f'Competition: {comp_level}\n(Insufficient data)')
            ax.set_xlabel('MMLU-Pro Score (%)')
            if idx == 0:
                ax.set_ylabel('Utility Difference (Strong - Baseline)')
            continue

        # Aggregate by model (mean across runs)
        model_means = subset.groupby('strong_model').agg({
            'mmlu_score': 'first',
            'utility_diff': ['mean', 'std', 'count']
        }).reset_index()
        model_means.columns = ['strong_model', 'mmlu_score', 'utility_diff_mean', 'utility_diff_std', 'n']

        # Create regplot using seaborn
        sns.regplot(
            data=subset,
            x='mmlu_score',
            y='utility_diff',
            ax=ax,
            scatter_kws={'alpha': 0.3, 's': 20},
            line_kws={'color': 'red', 'linewidth': 2},
            ci=95
        )

        # Add model labels for mean points
        for _, row in model_means.iterrows():
            ax.scatter(row['mmlu_score'], row['utility_diff_mean'],
                      s=80, marker='o', edgecolors='black', linewidths=1.5, zorder=5)

        # Compute regression statistics
        reg_stats = compute_regression_stats(subset['mmlu_score'].values, subset['utility_diff'].values)
        regression_stats[comp_level] = reg_stats

        # Add regression info to title
        ax.set_title(f'Competition: {comp_level}\n'
                    f'R²={reg_stats["r_squared"]:.3f}, '
                    f'slope={reg_stats["slope"]:.2f}',
                    fontsize=11)

        ax.set_xlabel('MMLU-Pro Score (%)', fontsize=12)
        if idx == 0:
            ax.set_ylabel('Utility Difference\n(Strong - Baseline)', fontsize=12)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlim(60, 90)

    plt.tight_layout()
    filename = f'{output_dir}/scaling_regplot_by_competition.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved regplot to {filename}")

    return regression_stats


def create_regplot_by_baseline(df, output_dir='figures'):
    """Create regression plots for each baseline model."""
    os.makedirs(output_dir, exist_ok=True)

    baselines_with_data = df['baseline_model'].unique()

    fig, axes = plt.subplots(1, len(baselines_with_data), figsize=(6*len(baselines_with_data), 5))
    if len(baselines_with_data) == 1:
        axes = [axes]

    for idx, baseline in enumerate(baselines_with_data):
        ax = axes[idx]
        subset = df[df['baseline_model'] == baseline]

        if len(subset) < 3:
            ax.set_title(f'{MODEL_DISPLAY_NAMES.get(baseline, baseline)}\n(Insufficient data)')
            continue

        # Color by competition level
        colors = plt.cm.viridis(np.linspace(0, 1, len(COMPETITION_LEVELS)))
        color_map = {level: colors[i] for i, level in enumerate(COMPETITION_LEVELS)}

        for comp_level in COMPETITION_LEVELS:
            comp_subset = subset[subset['competition_level'] == comp_level]
            if len(comp_subset) > 0:
                ax.scatter(comp_subset['mmlu_score'], comp_subset['utility_diff'],
                          c=[color_map[comp_level]], alpha=0.5, s=30,
                          label=f'Comp={comp_level}')

        # Overall regression line
        sns.regplot(
            data=subset,
            x='mmlu_score',
            y='utility_diff',
            ax=ax,
            scatter=False,
            line_kws={'color': 'red', 'linewidth': 2},
            ci=95
        )

        # Compute regression statistics
        reg_stats = compute_regression_stats(subset['mmlu_score'].values, subset['utility_diff'].values)

        ax.set_title(f'{MODEL_DISPLAY_NAMES.get(baseline, baseline)}\n'
                    f'R²={reg_stats["r_squared"]:.3f}, '
                    f'slope={reg_stats["slope"]:.2f} (p={reg_stats["p_value"]:.3f})',
                    fontsize=11)
        ax.set_xlabel('MMLU-Pro Score (%)', fontsize=12)
        if idx == 0:
            ax.set_ylabel('Utility Difference\n(Strong - Baseline)', fontsize=12)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlim(60, 90)
        ax.legend(fontsize=8, loc='upper left')

    plt.tight_layout()
    filename = f'{output_dir}/scaling_regplot_by_baseline.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved regplot to {filename}")


def create_combined_regplot(df, output_dir='figures'):
    """Create a single combined regression plot with all data."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Color points by competition level
    colors = plt.cm.viridis(np.linspace(0, 1, len(COMPETITION_LEVELS)))
    color_map = {level: colors[i] for i, level in enumerate(COMPETITION_LEVELS)}

    for comp_level in COMPETITION_LEVELS:
        subset = df[df['competition_level'] == comp_level]
        if len(subset) > 0:
            ax.scatter(subset['mmlu_score'], subset['utility_diff'],
                      c=[color_map[comp_level]], alpha=0.4, s=40,
                      label=f'Competition={comp_level}')

    # Add regression line with seaborn
    sns.regplot(
        data=df,
        x='mmlu_score',
        y='utility_diff',
        ax=ax,
        scatter=False,
        line_kws={'color': 'red', 'linewidth': 2.5},
        ci=95
    )

    # Compute overall regression statistics
    reg_stats = compute_regression_stats(df['mmlu_score'].values, df['utility_diff'].values)

    # Add model average points (larger, with labels)
    model_means = df.groupby('strong_model').agg({
        'mmlu_score': 'first',
        'utility_diff': 'mean'
    }).reset_index()

    ax.scatter(model_means['mmlu_score'], model_means['utility_diff'],
              s=150, marker='D', c='black', zorder=10, label='Model Average')

    # Add model name labels
    for _, row in model_means.iterrows():
        short_name = row['strong_model'].replace('-', ' ').title()
        if len(short_name) > 12:
            short_name = short_name[:12] + '...'
        ax.annotate(short_name, (row['mmlu_score'], row['utility_diff']),
                   xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)

    ax.set_xlabel('MMLU-Pro Score (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Utility Difference (Strong - Baseline)', fontsize=14, fontweight='bold')
    ax.set_title(f'Scaling Law: Model Capability vs. Negotiation Advantage\n'
                f'R²={reg_stats["r_squared"]:.3f}, '
                f'slope={reg_stats["slope"]:.2f} (p={reg_stats["p_value"]:.4f})',
                fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Break-even')
    ax.set_xlim(60, 90)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    filename = f'{output_dir}/scaling_regplot_combined.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined regplot to {filename}")

    return reg_stats


def print_regression_summary(stats_by_competition, overall_stats):
    """Print summary of regression results."""
    print("\n" + "="*60)
    print("REGRESSION ANALYSIS SUMMARY")
    print("="*60)

    print("\nOverall Regression (All Competition Levels):")
    print(f"  N = {overall_stats['n']}")
    print(f"  R² = {overall_stats['r_squared']:.4f}")
    print(f"  Slope = {overall_stats['slope']:.4f} (95% CI: [{overall_stats['conf_int'][0]:.4f}, {overall_stats['conf_int'][1]:.4f}])")
    print(f"  p-value = {overall_stats['p_value']:.6f}")

    if stats_by_competition:
        print("\nRegression by Competition Level:")
        for comp_level in COMPETITION_LEVELS:
            if comp_level in stats_by_competition:
                s = stats_by_competition[comp_level]
                sig = "***" if s['p_value'] < 0.001 else "**" if s['p_value'] < 0.01 else "*" if s['p_value'] < 0.05 else ""
                print(f"  Competition {comp_level}: R²={s['r_squared']:.3f}, slope={s['slope']:.2f}, p={s['p_value']:.4f} {sig}")


def main():
    """Main function to generate scaling regression plots."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate scaling law regression plots")
    parser.add_argument('--results-dir', type=str, default='experiments/results',
                       help='Directory containing experiment results')
    parser.add_argument('--output-dir', type=str, default='figures',
                       help='Directory to save figures')
    args = parser.parse_args()

    print("Loading experiment data...")
    df = load_experiment_data(args.results_dir)

    if len(df) == 0:
        print("No data found!")
        return

    print(f"\nLoaded {len(df)} data points")
    print(f"Unique strong models: {df['strong_model'].nunique()}")
    print(f"Unique baseline models: {df['baseline_model'].nunique()}")
    print(f"Competition levels: {df['competition_level'].unique()}")

    # Create visualizations
    print("\n=== Creating Regression Plots ===")

    # Combined plot (main figure)
    overall_stats = create_combined_regplot(df, args.output_dir)

    # By competition level
    stats_by_competition = create_regplot_by_competition(df, args.output_dir)

    # By baseline model
    create_regplot_by_baseline(df, args.output_dir)

    # Print summary
    print_regression_summary(stats_by_competition, overall_stats)

    print("\nDone!")


if __name__ == "__main__":
    main()

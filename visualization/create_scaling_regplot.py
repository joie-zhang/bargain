#!/usr/bin/env python3
"""
Create regression plots showing scaling laws between model capability and negotiation outcomes.
Uses seaborn regplot (with statsmodels fallback) for scatter plots with regression lines.

X-axis: Elo score (model capability proxy)
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

# Elo scores from Multi-Agent Strategic Games Evaluation
# Extracted from Arena Elo rankings and strong_models_experiment/configs.py
# Source: https://arena.lmsys.org/ (Arena Elo rankings)
ELO_SCORES = {
    # STRONG TIER - Elo ≥ 1415
    'gemini-3-pro': 1492,  # From Arena Elo
    'gemini-3-flash': 1470,  # From Arena Elo
    'gpt-5.2-high': 1465,  # From Arena Elo
    'gpt-5.2': 1464,  # From Arena Elo
    'gpt-5.1-high': 1464,  # From Arena Elo
    'grok-4.1': 1463,  # From Arena Elo
    'claude-opus-4-5': 1462,  # From Arena Elo
    'gemini-2-5-pro': 1460,  # From Arena Elo (Gemini-2.5-Pro)
    'grok-4': 1446,  # From Arena Elo
    'gpt-5-high': 1444,  # From Arena Elo
    'gpt-5.1': 1440,  # From Arena Elo
    'kimi-k2-thinking': 1438,  # From Arena Elo (Kimi-K2-Thinking)
    'claude-sonnet-4-5': 1420,  # From Arena Elo (Claude Sonnet 4.5)
    'deepseek-r1-0528': 1426,  # From Arena Elo (DeepSeek-R1-0528)
    'o3': 1424,  # From Arena Elo (o3-2025-04-16)
    'qwen3-235b-a22b-instruct-2507': 1418,  # From Arena Elo (Qwen3-235B-A22B-Instruct-2507)
    'claude-4-1-opus': 1419,  # From Arena Elo (Claude Opus 4.1)
    
    # MEDIUM TIER - 1290 ≤ Elo < 1415
    'claude-4-5-haiku': 1378,  # From Arena Elo (Claude Haiku 4.5)
    'gpt-5-mini': 1375,  # From Arena Elo
    'gemini-2-0-flash': 1370,  # From Arena Elo (Gemini-2.0-Flash-Exp: 1370 from Arena Elo)
    'o1': 1366,  # From Arena Elo (o1-2024-12-17)
    'o4-mini-2025-04-16': 1362,  # From Arena Elo
    'claude-4-sonnet': 1335,  # From Arena Elo (Claude Sonnet 4)
    'gpt-5-nano': 1333,  # From Arena Elo
    'gemini-1-5-pro': 1320,  # From Arena Elo (Gemini-1.5-Pro-002)
    'claude-3-7-sonnet': 1301,  # From Arena Elo (Claude 3.7 Sonnet)
    'claude-3-5-sonnet': 1299,  # From Arena Elo (Claude 3.5 Sonnet 20241022)
    'gpt-oss-20b': 1315,  # From Arena Elo (gpt-oss-20b)
    
    # WEAK TIER - Elo < 1290
    'gpt-4o': 1302,  # From Arena Elo (GPT-4o-2024-05-13)
    'gpt-4o-2024-05-13': 1302,  # From Arena Elo
    'gpt-4o-mini': 1289,  # From Arena Elo (GPT-4o-mini-2024-07-18)
    'llama-3.3-70b-instruct': 1276,  # From Arena Elo (Llama-3.3-70B-Instruct)
    'claude-3-opus': 1265,  # From Arena Elo (Claude 3 Opus)
    'claude-3-5-haiku': 1256,  # From Arena Elo (Claude 3.5 Haiku 20241022)
    'llama-3.1-8b-instruct': 1193,  # From Arena Elo (Llama-3.1-8B-Instruct)
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
BASELINE_MODELS = ['gpt-4o-2024-05-13', 'gpt-5-nano']
STRONG_MODELS = [
    'claude-3-5-haiku', 'claude-3-5-sonnet', 'claude-4-1-opus', 'claude-4-sonnet',
    'claude-4-5-haiku', 'claude-opus-4-5',  # New Claude models
    'gemini-2-0-flash', 'gemini-2-5-pro', 'gemini-3-pro',  # Gemini models
    'gpt-4o-2024-11-20', 'o1', 'o3', 'o4-mini-2025-04-16',  # OpenAI models
    'gpt-5-nano', 'gpt-5-mini', 'gpt-5.2-high',  # GPT-5 models
    'gpt-oss-20b',  # Open source GPT
    'llama-3.1-8b-instruct', 'llama-3.3-70b-instruct',  # Llama models
    'deepseek-r1-0528', 'kimi-k2-thinking', 'qwen3-235b-a22b-instruct-2507'  # Other models
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
    # Keep other model names as-is (they should match directory names)
    return model_name


def load_experiment_data(results_dir):
    """Load experiment data and organize for regression analysis."""
    data_records = []
    results_path = Path(results_dir)

    print("Scanning for experiment files...")
    processed_files = 0

    def extract_models_from_path(file_path):
        """Extract model names from directory path like 'gpt-4o_vs_claude-4-5-haiku'."""
        if file_path is None:
            return None, None
        
        path_parts = Path(file_path).parts
        # Look for directory with '_vs_' pattern
        for part in path_parts:
            if '_vs_' in part:
                models = part.split('_vs_')
                if len(models) == 2:
                    return normalize_model_name(models[0]), normalize_model_name(models[1])
        return None, None

    def process_experiment(exp, file_path=None):
        """Process a single experiment record (either from summary or direct file)."""
        if 'config' not in exp:
            return False
        
        comp_level = exp['config'].get('competition_level', None)
        if comp_level is None:
            return False

        # Round to nearest competition level
        comp_level = min(COMPETITION_LEVELS, key=lambda x: abs(x - comp_level))

        # Extract model names from path (for direct experiment_results.json files)
        model1, model2 = extract_models_from_path(file_path)
        
        # Fallback: try to extract from agents if path extraction failed
        if not model1 or not model2:
            if 'agents' in exp['config']:
                agents = exp['config']['agents']
                agent1 = agents[0] if len(agents) > 0 else None
                agent2 = agents[1] if len(agents) > 1 else None
                model1 = normalize_model_name(extract_model_name(agent1))
                model2 = normalize_model_name(extract_model_name(agent2))

        if not model1 or not model2:
            return False

        # Identify baseline and strong model
        baseline_model = None
        strong_model = None
        baseline_agent = None
        strong_agent = None

        # Determine which model is baseline and which is strong
        if model1 in BASELINE_MODELS and model2 in STRONG_MODELS:
            baseline_model, strong_model = model1, model2
            # In directory name order: model1 is first, model2 is second
            # Agent_Alpha is first agent, Agent_Beta is second agent
            baseline_agent = 'Agent_Alpha'
            strong_agent = 'Agent_Beta'
        elif model2 in BASELINE_MODELS and model1 in STRONG_MODELS:
            baseline_model, strong_model = model2, model1
            # model2 is baseline (second in dir name), model1 is strong (first in dir name)
            baseline_agent = 'Agent_Beta'
            strong_agent = 'Agent_Alpha'
        else:
            # Neither model pair matches baseline/strong pattern
            return False

        if baseline_model and strong_model:
            final_utilities = exp.get('final_utilities', {})

            # Skip if no final utilities (no consensus reached)
            if not final_utilities:
                return False

            baseline_utility = final_utilities.get(baseline_agent, 0)
            strong_utility = final_utilities.get(strong_agent, 0)
            utility_diff = strong_utility - baseline_utility
            model_order = exp['config'].get('model_order', 'unknown')

            # Get Elo score for the strong model
            elo_score = ELO_SCORES.get(strong_model, None)

            if elo_score is not None:
                data_records.append({
                    'baseline_model': baseline_model,
                    'strong_model': strong_model,
                    'elo_score': elo_score,
                    'competition_level': comp_level,
                    'utility_diff': utility_diff,
                    'baseline_utility': baseline_utility,
                    'strong_utility': strong_utility,
                    'model_order': model_order
                })
                return True
        return False

    # First, try loading from summary files (for backward compatibility)
    for file_path in results_path.glob('**/*_summary.json'):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            if 'experiments' in data and data['experiments']:
                for exp in data['experiments']:
                    process_experiment(exp, file_path)
                processed_files += 1
                if processed_files % 50 == 0:
                    print(f"Processed {processed_files} files...")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    # Also load from direct experiment_results.json files (current format)
    for file_path in results_path.glob('**/*experiment_results.json'):
        try:
            with open(file_path, 'r') as f:
                exp = json.load(f)
            
            if process_experiment(exp, file_path):
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
            ax.set_xlabel('Elo Score')
            if idx == 0:
                ax.set_ylabel('Utility Difference (Strong - Baseline)')
            continue

        # Aggregate by model (mean across runs)
        model_means = subset.groupby('strong_model').agg({
            'elo_score': 'first',
            'utility_diff': ['mean', 'std', 'count']
        }).reset_index()
        model_means.columns = ['strong_model', 'elo_score', 'utility_diff_mean', 'utility_diff_std', 'n']

        # Create regplot using seaborn
        sns.regplot(
            data=subset,
            x='elo_score',
            y='utility_diff',
            ax=ax,
            scatter_kws={'alpha': 0.3, 's': 20},
            line_kws={'color': 'red', 'linewidth': 2},
            ci=95
        )

        # Add model labels for mean points
        for _, row in model_means.iterrows():
            ax.scatter(row['elo_score'], row['utility_diff_mean'],
                      s=80, marker='o', edgecolors='black', linewidths=1.5, zorder=5)

        # Compute regression statistics
        reg_stats = compute_regression_stats(subset['elo_score'].values, subset['utility_diff'].values)
        regression_stats[comp_level] = reg_stats

        # Add regression info to title
        ax.set_title(f'Competition: {comp_level}\n'
                    f'R²={reg_stats["r_squared"]:.3f}, '
                    f'slope={reg_stats["slope"]:.2f}',
                    fontsize=11)

        ax.set_xlabel('Elo Score', fontsize=12)
        if idx == 0:
            ax.set_ylabel('Utility Difference\n(Strong - Baseline)', fontsize=12)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        # Set x-axis limits based on Elo score range (typically 1100-1500)
        ax.set_xlim(1150, 1520)

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
                ax.scatter(comp_subset['elo_score'], comp_subset['utility_diff'],
                          c=[color_map[comp_level]], alpha=0.5, s=30,
                          label=f'Comp={comp_level}')

        # Overall regression line
        sns.regplot(
            data=subset,
            x='elo_score',
            y='utility_diff',
            ax=ax,
            scatter=False,
            line_kws={'color': 'red', 'linewidth': 2},
            ci=95
        )

        # Compute regression statistics
        reg_stats = compute_regression_stats(subset['elo_score'].values, subset['utility_diff'].values)

        ax.set_title(f'{MODEL_DISPLAY_NAMES.get(baseline, baseline)}\n'
                    f'R²={reg_stats["r_squared"]:.3f}, '
                    f'slope={reg_stats["slope"]:.2f} (p={reg_stats["p_value"]:.3f})',
                    fontsize=11)
        ax.set_xlabel('Elo Score', fontsize=12)
        if idx == 0:
            ax.set_ylabel('Utility Difference\n(Strong - Baseline)', fontsize=12)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        # Set x-axis limits based on Elo score range (typically 1100-1500)
        ax.set_xlim(1150, 1520)
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
            ax.scatter(subset['elo_score'], subset['utility_diff'],
                      c=[color_map[comp_level]], alpha=0.4, s=40,
                      label=f'Competition={comp_level}')

    # Add regression line with seaborn
    sns.regplot(
        data=df,
        x='elo_score',
        y='utility_diff',
        ax=ax,
        scatter=False,
        line_kws={'color': 'red', 'linewidth': 2.5},
        ci=95
    )

    # Compute overall regression statistics
    reg_stats = compute_regression_stats(df['elo_score'].values, df['utility_diff'].values)

    # Add model average points (larger, with labels)
    model_means = df.groupby('strong_model').agg({
        'elo_score': 'first',
        'utility_diff': 'mean'
    }).reset_index()

    ax.scatter(model_means['elo_score'], model_means['utility_diff'],
              s=150, marker='D', c='black', zorder=10, label='Model Average')

    # Add model name labels
    for _, row in model_means.iterrows():
        short_name = row['strong_model'].replace('-', ' ').title()
        if len(short_name) > 12:
            short_name = short_name[:12] + '...'
        ax.annotate(short_name, (row['elo_score'], row['utility_diff']),
                   xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)

    ax.set_xlabel('Elo Score', fontsize=14, fontweight='bold')
    ax.set_ylabel('Utility Difference (Strong - Baseline)', fontsize=14, fontweight='bold')
    ax.set_title(f'Scaling Law: Model Capability vs. Negotiation Advantage\n'
                f'R²={reg_stats["r_squared"]:.3f}, '
                f'slope={reg_stats["slope"]:.2f} (p={reg_stats["p_value"]:.4f})',
                fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Break-even')
    # Set x-axis limits based on Elo score range (typically 1100-1500)
    ax.set_xlim(1150, 1520)
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

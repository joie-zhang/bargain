#!/usr/bin/env python3
"""
Plot Qwen experiment results:
1. Double bar graph: Qwen vs Claude payoffs by model size
2. Violin plot: Distribution of rounds to consensus
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Tuple


# Model size mapping (for ordering and display)
MODEL_SIZES = ['3', '7', '14', '32', '72']
MODEL_SIZE_DISPLAY = {
    '3': '3B',
    '7': '7B', 
    '14': '14B',
    '32': '32B',
    '72': '72B'
}

# Color scheme - using provided palette
COLOR_PALETTE = [
    '#AB63FA',  # Purple
    '#00CC96',  # Teal/Green
    '#EF553B',  # Red
    '#FFA15A',  # Orange
    '#19D3F3',  # Cyan (Blue)
    '#B6E880',  # Light Green
    '#FF6692',  # Pink
    '#FF97FF'   # Light Pink
]
QWEN_COLOR = '#8B4FD9'  # Darker Purple (darker version of #AB63FA)
CLAUDE_COLOR = '#0FA8C7'  # Deeper Blue (darker version of #19D3F3)
VIOLIN_COLOR = COLOR_PALETTE[6]  # Pink (#FF6692)


def load_results(csv_path: Path) -> pd.DataFrame:
    """Load results from CSV file."""
    df = pd.read_csv(csv_path)
    
    # Convert model_size to string for consistent handling
    df['model_size'] = df['model_size'].astype(str)
    
    # Filter to competition_level = 1.0 only
    df = df[df['competition_level'] == 1.0].copy()
    
    # Convert payoffs to numeric (handle empty strings)
    for agent in ['Agent_Alpha', 'Agent_Beta']:
        payoff_key = f'{agent}_payoff'
        df[payoff_key] = pd.to_numeric(df[payoff_key], errors='coerce').fillna(0.0)
    
    return df


def identify_agents(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Identify which agent is Qwen and which is Claude.
    Based on the experiment setup, Agent_Alpha should be Qwen.
    """
    # Check if we can infer from the data
    # For now, assume Agent_Alpha = Qwen, Agent_Beta = Claude
    return 'Agent_Alpha', 'Agent_Beta'


def plot_payoff_comparison(df: pd.DataFrame, output_path: Path, qwen_agent: str, claude_agent: str):
    """
    Create double bar graph comparing Qwen vs Claude payoffs across model sizes.
    Only includes runs that reached consensus.
    """
    # Filter to consensus runs only
    consensus_df = df[df['consensus_reached'] == True].copy()
    
    if len(consensus_df) == 0:
        print("Warning: No consensus runs found for payoff comparison")
        return
    
    # Group by model size and calculate statistics
    qwen_col = f'{qwen_agent}_payoff'
    claude_col = f'{claude_agent}_payoff'
    
    results = []
    for model_size in MODEL_SIZES:
        model_data = consensus_df[consensus_df['model_size'] == model_size].copy()
        
        if len(model_data) == 0:
            continue
        
        qwen_payoffs = model_data[qwen_col].values
        claude_payoffs = model_data[claude_col].values
        
        # Filter out zeros (failed runs that might have slipped through)
        qwen_payoffs = qwen_payoffs[qwen_payoffs > 0]
        claude_payoffs = claude_payoffs[claude_payoffs > 0]
        
        if len(qwen_payoffs) == 0:
            continue
        
        results.append({
            'model_size': MODEL_SIZE_DISPLAY[model_size],
            'model_size_num': float(model_size),
            'qwen_mean': np.mean(qwen_payoffs),
            'qwen_std': np.std(qwen_payoffs),
            'qwen_sem': np.std(qwen_payoffs) / np.sqrt(len(qwen_payoffs)),
            'claude_mean': np.mean(claude_payoffs),
            'claude_std': np.std(claude_payoffs),
            'claude_sem': np.std(claude_payoffs) / np.sqrt(len(claude_payoffs)),
            'n_runs': len(qwen_payoffs)
        })
    
    if not results:
        print("Warning: No valid data for payoff comparison")
        return
    
    # Convert to DataFrame for easier handling
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('model_size_num')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(results_df))
    width = 0.35
    
    # Plot bars
    qwen_bars = ax.bar(x - width/2, results_df['qwen_mean'], width,
                       label='Qwen2.5', color=QWEN_COLOR, alpha=0.8,
                       yerr=results_df['qwen_sem'], capsize=5, error_kw={'elinewidth': 2})
    
    claude_bars = ax.bar(x + width/2, results_df['claude_mean'], width,
                         label='Claude 3.7 Sonnet', color=CLAUDE_COLOR, alpha=0.8,
                         yerr=results_df['claude_sem'], capsize=5, error_kw={'elinewidth': 2})
    
    # Customize plot
    ax.set_xlabel('Qwen Model Size', fontsize=12, fontweight='bold', fontfamily='Arial')
    ax.set_ylabel('Average Payoff', fontsize=12, fontweight='bold', fontfamily='Arial')
    ax.set_title('Payoff Comparison: Qwen2.5 vs Claude 3.7 Sonnet\n(Competition Level = 1.0, Consensus Runs Only)',
                 fontsize=13, fontweight='bold', pad=15, fontfamily='Arial')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['model_size'], fontsize=11, fontfamily='Arial')
    ax.legend(fontsize=11, loc='upper right', prop={'family': 'Arial'})
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Set Arial font for all text
    for label in ax.get_yticklabels():
        label.set_fontfamily('Arial')
    
    # Add sample size annotations
    for i, (idx, row) in enumerate(results_df.iterrows()):
        ax.text(i - width/2, row['qwen_mean'] + row['qwen_sem'] + 2,
                f'n={int(row["n_runs"])}', ha='center', va='bottom', fontsize=9, 
                alpha=0.7, fontfamily='Arial')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Payoff comparison plot saved to {output_path}")
    plt.close()


def plot_rounds_distribution(df: pd.DataFrame, output_path: Path):
    """
    Create violin plot showing distribution of rounds to consensus across model sizes.
    Only includes runs that reached consensus.
    """
    # Filter to consensus runs only
    consensus_df = df[df['consensus_reached'] == True].copy()
    
    if len(consensus_df) == 0:
        print("Warning: No consensus runs found for rounds distribution")
        return
    
    # Prepare data
    plot_data = []
    for model_size in MODEL_SIZES:
        model_data = consensus_df[consensus_df['model_size'] == model_size].copy()
        
        if len(model_data) == 0:
            continue
        
        rounds = model_data['final_round'].values
        # Filter out rounds > 10 (shouldn't happen, but just in case)
        rounds = rounds[rounds <= 10]
        
        if len(rounds) == 0:
            continue
        
        for round_num in rounds:
            plot_data.append({
                'model_size': MODEL_SIZE_DISPLAY[model_size],
                'model_size_num': float(model_size),
                'rounds_to_consensus': round_num
            })
    
    if not plot_data:
        print("Warning: No valid data for rounds distribution")
        return
    
    plot_df = pd.DataFrame(plot_data)
    plot_df = plot_df.sort_values('model_size_num')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for violin plot
    model_order = plot_df.sort_values('model_size_num')['model_size'].unique()
    rounds_data = [plot_df[plot_df['model_size'] == m]['rounds_to_consensus'].values
                   for m in model_order]
    
    # Create violin plot (showextrema=False removes the vertical min/max lines)
    parts = ax.violinplot(rounds_data,
                          positions=range(len(model_order)),
                          widths=0.7, showmeans=True, showmedians=True,
                          showextrema=False)
    
    # Customize violin plot colors
    for pc in parts['bodies']:
        pc.set_facecolor(VIOLIN_COLOR)
        pc.set_alpha(0.6)
    
    parts['cmeans'].set_color(QWEN_COLOR)  # Purple (#AB63FA)
    parts['cmeans'].set_linewidth(2)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(2)
    
    # Customize plot
    ax.set_xlabel('Qwen Model Size', fontsize=12, fontweight='bold', fontfamily='Arial')
    ax.set_ylabel('Rounds to Consensus', fontsize=12, fontweight='bold', fontfamily='Arial')
    ax.set_title('Distribution of Rounds to Consensus\n(Competition Level = 1.0, Consensus Runs Only)',
                 fontsize=13, fontweight='bold', pad=15, fontfamily='Arial')
    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels(model_order, fontsize=11, fontfamily='Arial')
    ax.set_ylim(0.5, 10.5)
    ax.set_yticks(range(1, 11))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Set Arial font for all text
    for label in ax.get_yticklabels():
        label.set_fontfamily('Arial')
    
    # Add legend explaining purple and black lines
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=QWEN_COLOR, lw=2, label='Mean'),
        Line2D([0], [0], color='black', lw=2, label='Median')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, prop={'family': 'Arial'})
    
    # Add sample size annotations
    for i, model in enumerate(model_order):
        n_runs = len(plot_df[plot_df['model_size'] == model])
        ax.text(i, 0.7, f'n={n_runs}', ha='center', va='bottom', fontsize=9, 
                alpha=0.7, fontfamily='Arial')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Rounds distribution plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot Qwen experiment results')
    parser.add_argument('--csv', type=str, required=True,
                       help='Path to CSV file with results')
    parser.add_argument('--output-dir', type=str, default='visualization/figures',
                       help='Output directory for plots')
    parser.add_argument('--payoff-only', action='store_true',
                       help='Only generate payoff comparison plot')
    parser.add_argument('--rounds-only', action='store_true',
                       help='Only generate rounds distribution plot')
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading results from {csv_path}")
    df = load_results(csv_path)
    
    if len(df) == 0:
        print("Error: No data found in CSV")
        return
    
    print(f"Loaded {len(df)} rows")
    print(f"Consensus runs: {df['consensus_reached'].sum()}")
    
    # Identify agents
    qwen_agent, claude_agent = identify_agents(df)
    print(f"Assuming {qwen_agent} = Qwen, {claude_agent} = Claude")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['figure.dpi'] = 100
    
    # Generate plots
    if not args.rounds_only:
        payoff_path = output_dir / 'qwen_payoff_comparison.png'
        plot_payoff_comparison(df, payoff_path, qwen_agent, claude_agent)
    
    if not args.payoff_only:
        rounds_path = output_dir / 'qwen_rounds_distribution.png'
        plot_rounds_distribution(df, rounds_path)
    
    print("\nPlotting complete!")


if __name__ == '__main__':
    main()


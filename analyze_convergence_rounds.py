#!/usr/bin/env python3
"""
Analyze the number of rounds it takes for negotiations to converge,
broken down by competition level and model pairs.
"""

import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd

# Define baseline (weak) and strong models - matching configs.py
BASELINE_MODELS = ['claude-3-opus', 'gemini-1-5-pro', 'gpt-4o']
STRONG_MODELS = ['claude-3-5-haiku', 'claude-3-5-sonnet', 'claude-4-sonnet', 
                 'claude-4-1-opus', 'gemini-2-5-pro', 'gemini-2-0-flash', 
                 'gpt-4o-latest', 'o1', 
                 'gpt-5-mini', 'gpt-5-nano', 'o3', 'gpt-5']

# Competition level buckets
COMPETITION_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]

def categorize_competition_level(comp_level):
    """Categorize competition level into buckets with strict boundaries."""
    if comp_level is None:
        return None
    
    # Use strict boundaries - only categorize if within 0.05 of the target
    tolerance = 0.05
    for bucket in COMPETITION_LEVELS:
        if abs(comp_level - bucket) <= tolerance:
            return bucket
    
    # If not within tolerance of any bucket, return None (exclude from analysis)
    return None

def load_convergence_data(results_dir):
    """Load convergence data from experiment results."""
    convergence_by_competition = defaultdict(list)
    convergence_by_model_pair = defaultdict(lambda: defaultdict(list))
    tie_counts = defaultdict(int)
    winner_counts = defaultdict(lambda: defaultdict(int))
    api_timeout_counts = defaultdict(int)  # Track actual API timeouts/errors
    excluded_count = 0  # Track experiments excluded due to competition level
    
    results_path = Path(results_dir)
    
    for file_path in results_path.glob('**/*_summary.json'):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if 'experiments' in data and data['experiments']:
                for exp in data['experiments']:
                    if 'config' in exp:
                        # Get competition level
                        comp_level = exp['config'].get('competition_level', None)
                        comp_bucket = categorize_competition_level(comp_level)
                        
                        if comp_bucket is None:
                            excluded_count += 1
                            continue
                        
                        # Get number of rounds
                        final_round = exp.get('final_round', None)
                        consensus_reached = exp.get('consensus_reached', False)
                        max_rounds = exp['config'].get('t_rounds', 10)
                        
                        # Get agents
                        agents = exp['config'].get('agents', [])
                        
                        # Identify models
                        model1, model2 = None, None
                        if len(agents) >= 2:
                            agent1, agent2 = agents[0], agents[1]
                            
                            # Extract model names
                            for baseline in BASELINE_MODELS:
                                if baseline.replace('-', '_') in agent1.lower():
                                    model1 = baseline
                                if baseline.replace('-', '_') in agent2.lower():
                                    model2 = baseline
                            
                            for strong in STRONG_MODELS:
                                if strong.replace('-', '_') in agent1.lower():
                                    model1 = strong
                                if strong.replace('-', '_') in agent2.lower():
                                    model2 = strong
                        
                        # Check for ties and winners
                        final_utilities = exp.get('final_utilities', {})
                        if final_utilities and len(agents) >= 2:
                            util1 = final_utilities.get(agents[0], 0)
                            util2 = final_utilities.get(agents[1], 0)
                            
                            if util1 == util2:
                                tie_counts[comp_bucket] += 1
                            elif util1 > util2:
                                winner_counts[comp_bucket]['agent1'] += 1
                            else:
                                winner_counts[comp_bucket]['agent2'] += 1
                        
                        # Record convergence data
                        if final_round is not None:
                            # A negotiation either reaches consensus OR times out (reaches max rounds without consensus)
                            reached_timeout = final_round >= max_rounds and not consensus_reached
                            convergence_by_competition[comp_bucket].append({
                                'rounds': final_round,
                                'consensus': consensus_reached,
                                'max_rounds': max_rounds,
                                'reached_timeout': reached_timeout
                            })
                            
                            if model1 and model2:
                                pair_key = f"{model1} vs {model2}"
                                convergence_by_model_pair[comp_bucket][pair_key].append(final_round)
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # Now scan log files for actual API timeout errors
    logs_path = results_path / 'scaling_experiment' / 'logs'
    if logs_path.exists():
        for log_file in logs_path.glob('*.log'):
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                    # Check for "Failed after 3 attempts" which indicates API timeout/error
                    if "Failed after 3 attempts" in content:
                        # Try to extract competition level from the log
                        for line in content.split('\n'):
                            if 'competition=' in line:
                                try:
                                    comp_str = line.split('competition=')[1].split(')')[0]
                                    comp_level = float(comp_str)
                                    comp_bucket = categorize_competition_level(comp_level)
                                    if comp_bucket is not None:
                                        api_timeout_counts[comp_bucket] += 1
                                    break
                                except:
                                    pass
            except Exception as e:
                print(f"Error processing log {log_file}: {e}")
                continue
    
    return convergence_by_competition, convergence_by_model_pair, tie_counts, winner_counts, api_timeout_counts, excluded_count

def create_convergence_visualizations(convergence_by_competition, convergence_by_model_pair, tie_counts, winner_counts, api_timeout_counts):
    """Create visualizations for convergence analysis."""
    
    # Figure 1: Single row of violin plots for all competition levels
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    fig.suptitle('Negotiation Convergence Analysis by Competition Level', fontsize=16, fontweight='bold')
    
    # Add legend next to title
    fig.text(0.85, 0.96, 'Green: Consensus %', fontsize=9, color='green', transform=fig.transFigure)
    fig.text(0.85, 0.93, 'Gray: Sample size', fontsize=9, color='gray', transform=fig.transFigure)
    
    # Prepare data for all competition levels
    all_data = []
    positions = []
    labels = []
    consensus_info = []
    
    for idx, comp_level in enumerate(COMPETITION_LEVELS):
        data = convergence_by_competition[comp_level]
        if data:
            rounds = [d['rounds'] for d in data]
            all_data.append(rounds)
            positions.append(idx + 1)
            labels.append(f'{comp_level}')
            
            # Calculate consensus rate for annotation
            consensus_rate = sum(1 for d in data if d['consensus']) / len(data) * 100
            consensus_info.append((idx + 1, consensus_rate, len(data)))
    
    # Create violin plots only
    if all_data:
        parts = ax.violinplot(all_data, positions=positions, widths=0.7, showmeans=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor('skyblue')
            pc.set_alpha(0.8)
        
        # Add mean lines that stay within violin bounds
        for i, (data_points, pos) in enumerate(zip(all_data, positions)):
            mean_val = np.mean(data_points)
            # Draw a shorter horizontal line for the mean (40% of violin width)
            ax.hlines(mean_val, pos - 0.14, pos + 0.14, colors='black', linewidth=1.5)
    
    # Add consensus rate and sample size at top of plot area
    for pos, consensus_rate, n in consensus_info:
        ax.text(pos, 10.7, f'{consensus_rate:.0f}%', ha='center', va='top', fontsize=9, color='green')
        ax.text(pos, 10.3, f'n={n}', ha='center', va='top', fontsize=8, color='gray')
    
    # Formatting
    ax.set_xlabel('Competition Level', fontsize=12)
    ax.set_ylabel('Rounds to End', fontsize=12)
    ax.set_ylim(0, 11)
    ax.set_xlim(0.5, len(COMPETITION_LEVELS) + 0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('convergence_analysis_by_competition.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved convergence analysis to convergence_analysis_by_competition.png")
    
    # Figure 2: Histogram of convergence rounds across all competition levels
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    all_rounds_by_comp = {}
    colors = plt.cm.viridis(np.linspace(0, 1, len(COMPETITION_LEVELS)))
    
    for idx, comp_level in enumerate(COMPETITION_LEVELS):
        data = convergence_by_competition[comp_level]
        if data:
            rounds = [d['rounds'] for d in data]
            all_rounds_by_comp[comp_level] = rounds
            ax.hist(rounds, bins=range(1, 12), alpha=0.5, label=f'Competition {comp_level}', 
                   color=colors[idx], edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Number of Rounds', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Negotiation Rounds by Competition Level', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('convergence_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved convergence distribution to convergence_distribution.png")
    
    # Figure 3: Heatmap of average rounds by model pairs (for competition level 0.5 as example)
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Create matrix for heatmap
    avg_rounds_matrix = np.full((len(BASELINE_MODELS), len(STRONG_MODELS)), np.nan)
    
    comp_level = 0.5  # Example competition level
    for i, baseline in enumerate(BASELINE_MODELS):
        for j, strong in enumerate(STRONG_MODELS):
            pair_key = f"{baseline} vs {strong}"
            if pair_key in convergence_by_model_pair[comp_level]:
                avg_rounds_matrix[i, j] = np.mean(convergence_by_model_pair[comp_level][pair_key])
    
    # Create heatmap
    mask = np.isnan(avg_rounds_matrix)
    sns.heatmap(avg_rounds_matrix, 
                annot=True, 
                fmt='.1f', 
                cmap='YlOrRd',
                mask=mask,
                cbar_kws={'label': 'Average Rounds'},
                vmin=1, 
                vmax=10,
                linewidths=0.5,
                linecolor='gray',
                ax=ax)
    
    ax.set_title(f'Average Rounds to Convergence at Competition Level {comp_level}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Strong Models', fontsize=12)
    ax.set_ylabel('Baseline Models', fontsize=12)
    ax.set_xticklabels(STRONG_MODELS, rotation=45, ha='right')
    ax.set_yticklabels(BASELINE_MODELS, rotation=0)
    
    plt.tight_layout()
    plt.savefig('convergence_heatmap_example.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved convergence heatmap to convergence_heatmap_example.png")

def print_convergence_statistics(convergence_by_competition, tie_counts, winner_counts, api_timeout_counts, excluded_count):
    """Print detailed statistics about convergence."""
    
    print("\n" + "="*60)
    print("CONVERGENCE STATISTICS BY COMPETITION LEVEL")
    print("="*60)
    
    if excluded_count > 0:
        print(f"\n⚠️  Note: {excluded_count} experiments excluded (competition level not within ±0.05 of standard buckets)")
        print(f"  Standard buckets: {COMPETITION_LEVELS}")
    
    for comp_level in COMPETITION_LEVELS:
        data = convergence_by_competition[comp_level]
        if not data:
            continue
            
        rounds = [d['rounds'] for d in data]
        consensus_count = sum(1 for d in data if d['consensus'])
        timeout_count = sum(1 for d in data if d['reached_timeout'])
        
        print(f"\n📊 Competition Level: {comp_level}")
        print("-" * 40)
        print(f"Total negotiations: {len(data)}")
        print(f"Rounds to convergence:")
        print(f"  - Mean: {np.mean(rounds):.2f}")
        print(f"  - Median: {np.median(rounds):.1f}")
        print(f"  - Min: {np.min(rounds)}")
        print(f"  - Max: {np.max(rounds)}")
        print(f"  - Std Dev: {np.std(rounds):.2f}")
        print(f"Consensus reached: {consensus_count}/{len(data)} ({consensus_count/len(data)*100:.1f}%)")
        print(f"No consensus (max rounds): {timeout_count}/{len(data)} ({timeout_count/len(data)*100:.1f}%)")
        print(f"Total (should be 100%): {(consensus_count + timeout_count)/len(data)*100:.1f}%")
        
        # API timeout analysis
        api_timeouts = api_timeout_counts.get(comp_level, 0)
        if api_timeouts > 0:
            print(f"\n⚠️  API Errors/Timeouts: {api_timeouts} experiments failed")
            print(f"  (These are separate from negotiation outcomes)")
        
        # Tie analysis
        ties = tie_counts[comp_level]
        agent1_wins = winner_counts[comp_level]['agent1']
        agent2_wins = winner_counts[comp_level]['agent2']
        total = ties + agent1_wins + agent2_wins
        
        if total > 0:
            print(f"\nOutcome distribution:")
            print(f"  - Ties: {ties}/{total} ({ties/total*100:.1f}%)")
            print(f"  - Agent 1 wins: {agent1_wins}/{total} ({agent1_wins/total*100:.1f}%)")
            print(f"  - Agent 2 wins: {agent2_wins}/{total} ({agent2_wins/total*100:.1f}%)")
    
    # Overall statistics
    print("\n" + "="*60)
    print("OVERALL STATISTICS")
    print("="*60)
    
    # API errors summary
    total_api_errors = sum(api_timeout_counts.values())
    if total_api_errors > 0:
        print(f"⚠️  Total API Errors/Timeouts: {total_api_errors} experiments")
        print("  Distribution by competition level:")
        for comp_level in COMPETITION_LEVELS:
            if api_timeout_counts[comp_level] > 0:
                print(f"    - Competition {comp_level}: {api_timeout_counts[comp_level]} errors")
    
    # Tie analysis
    print("\nTIE HANDLING:")
    print("Current implementation treats ties as wins for Agent 2!")
    print("This means when util1 == util2, Agent 2 is declared the winner.")
    total_ties = sum(tie_counts.values())
    total_games = sum(tie_counts.values()) + sum(winner_counts[c]['agent1'] + winner_counts[c]['agent2'] for c in COMPETITION_LEVELS)
    if total_games > 0:
        print(f"\nOverall tie rate: {total_ties}/{total_games} ({total_ties/total_games*100:.1f}%)")

def main():
    """Main function to analyze convergence."""
    print("Loading experiment results for convergence analysis...")
    results_dir = '/root/bargain/experiments/results'
    
    convergence_by_competition, convergence_by_model_pair, tie_counts, winner_counts, api_timeout_counts, excluded_count = load_convergence_data(results_dir)
    
    # Print statistics
    print_convergence_statistics(convergence_by_competition, tie_counts, winner_counts, api_timeout_counts, excluded_count)
    
    # Create visualizations
    print("\nCreating convergence visualizations...")
    create_convergence_visualizations(convergence_by_competition, convergence_by_model_pair, tie_counts, winner_counts, api_timeout_counts)
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
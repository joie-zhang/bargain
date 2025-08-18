#!/usr/bin/env python3
"""
Aggregate negotiation results from runs 2-9 to calculate win rates
between Claude 3.5 Sonnet and Llama 3.1 405B
"""

import json
import glob
from pathlib import Path

def load_run_results(results_dir):
    """Load results from run_2 through run_9"""
    results = []
    
    for run_num in range(2, 10):  # runs 2-9
        file_path = Path(results_dir) / f"strong_models_run_{run_num}.json"
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                results.append(data)
                print(f"Loaded run {run_num}")
        except FileNotFoundError:
            print(f"Warning: File not found for run {run_num}")
        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON in run {run_num}")
    
    return results

def analyze_win_rates(results):
    """Analyze win rates between the two models"""
    claude_wins = 0
    llama_wins = 0
    ties = 0
    total_games = len(results)
    
    detailed_results = []
    
    for i, result in enumerate(results):
        run_index = result.get('run_index', i + 2)
        
        # Get final utilities
        final_utilities = result['negotiation_outcome']['final_utilities']
        claude_utility = final_utilities.get('claude_3_5_sonnet_1', 0)
        llama_utility = final_utilities.get('llama_3_1_405b_2', 0)
        
        # Determine winner
        if claude_utility > llama_utility:
            winner = "Claude 3.5 Sonnet"
            claude_wins += 1
        elif llama_utility > claude_utility:
            winner = "Llama 3.1 405B"
            llama_wins += 1
        else:
            winner = "Tie"
            ties += 1
        
        detailed_results.append({
            'run': run_index,
            'claude_utility': claude_utility,
            'llama_utility': llama_utility,
            'winner': winner,
            'utility_difference': abs(claude_utility - llama_utility)
        })
    
    return {
        'detailed_results': detailed_results,
        'summary': {
            'total_games': total_games,
            'claude_wins': claude_wins,
            'llama_wins': llama_wins,
            'ties': ties,
            'claude_win_rate': claude_wins / total_games if total_games > 0 else 0,
            'llama_win_rate': llama_wins / total_games if total_games > 0 else 0,
            'tie_rate': ties / total_games if total_games > 0 else 0
        }
    }

def main():
    results_dir = "/Users/joie/Desktop/bargain/experiments/results/sonnet4-405B-dirty"
    
    print("Loading results from runs 2-9...")
    results = load_run_results(results_dir)
    
    if not results:
        print("No results found!")
        return
    
    print(f"\nAnalyzing {len(results)} runs...")
    analysis = analyze_win_rates(results)
    
    # Print detailed results
    print("\n" + "="*60)
    print("DETAILED RESULTS")
    print("="*60)
    print(f"{'Run':<4} {'Claude Utility':<15} {'Llama Utility':<15} {'Winner':<20} {'Difference':<10}")
    print("-" * 70)
    
    for result in analysis['detailed_results']:
        print(f"{result['run']:<4} {result['claude_utility']:<15.2f} {result['llama_utility']:<15.2f} "
              f"{result['winner']:<20} {result['utility_difference']:<10.2f}")
    
    # Print summary
    summary = analysis['summary']
    print("\n" + "="*60)
    print("WIN RATE SUMMARY")
    print("="*60)
    print(f"Total Games: {summary['total_games']}")
    print(f"Claude 3.5 Sonnet Wins: {summary['claude_wins']} ({summary['claude_win_rate']:.1%})")
    print(f"Llama 3.1 405B Wins: {summary['llama_wins']} ({summary['llama_win_rate']:.1%})")
    print(f"Ties: {summary['ties']} ({summary['tie_rate']:.1%})")
    
    # Calculate average utilities
    claude_utilities = [r['claude_utility'] for r in analysis['detailed_results']]
    llama_utilities = [r['llama_utility'] for r in analysis['detailed_results']]
    
    print(f"\nAverage Claude Utility: {sum(claude_utilities) / len(claude_utilities):.2f}")
    print(f"Average Llama Utility: {sum(llama_utilities) / len(llama_utilities):.2f}")
    
    # Calculate utility advantage
    total_advantage = sum(llama_utilities) - sum(claude_utilities)
    avg_advantage = total_advantage / len(results)
    print(f"Average Llama Advantage: {avg_advantage:.2f} utility points")

if __name__ == "__main__":
    main()
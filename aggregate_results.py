#!/usr/bin/env python3
"""
Aggregate negotiation results from strong_models experiments
Supports any strong_models_20250819_* folder structure
"""

import json
import glob
import sys
import os
from pathlib import Path
import re # Added for regex in interaction file counting

def load_run_results(results_dir):
    """Load all run results from the specified directory"""
    results = []
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Error: Directory {results_dir} does not exist")
        return results
    
    # Look for both single and batch experiment result files
    patterns = [
        results_path / "run_*_experiment_results.json",  # Batch mode results
        results_path / "experiment_results.json",        # Single experiment results
    ]
    
    run_files = []
    for pattern in patterns:
        run_files.extend(sorted(glob.glob(str(pattern))))
    
    print(f"Found {len(run_files)} result files in {results_dir}")
    
    for file_path in run_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                results.append(data)
                # Extract run number if present, otherwise use "single"
                if "run_" in Path(file_path).stem:
                    run_num = re.search(r'run_(\d+)_', Path(file_path).stem).group(1)
                else:
                    run_num = "single"
                print(f"Loaded run {run_num}")
        except FileNotFoundError:
            print(f"Warning: File not found: {file_path}")
        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON in {file_path}")
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
    
    if not agents:
        print("Warning: No agents found in config")
        return None, None, None, None
    
    # In the new format, agent IDs contain model information
    agent1_name = agents[0]
    agent2_name = agents[1] if len(agents) > 1 else None
    
    # Extract model names from agent IDs (they're now part of the ID)
    # For names like "claude_3_5_haiku_1" or "claude_4_sonnet_2", extract everything except the final number
    def extract_model_name(agent_name):
        if not agent_name:
            return None
        # Split by underscore and remove the last part if it's a number
        parts = agent_name.split('_')
        if len(parts) > 1 and parts[-1].isdigit():
            # Remove the final number and rejoin with hyphens
            return '-'.join(parts[:-1]).replace('_', '-')
        else:
            # If no trailing number, just replace underscores with hyphens
            return agent_name.replace('_', '-')
    
    agent1_model = extract_model_name(agent1_name)
    agent2_model = extract_model_name(agent2_name)
    
    return agent1_name, agent2_name, agent1_model, agent2_model

def analyze_win_rates(results):
    """Analyze win rates between the two models"""
    agent1_name, agent2_name, agent1_model, agent2_model = get_agent_info(results)
    
    if not agent1_name:
        print("Error: Could not determine agent names")
        return None
    
    agent1_wins = 0
    agent2_wins = 0
    ties = 0
    total_games = len(results)
    
    detailed_results = []
    
    print(f"Comparing: {agent1_model} vs {agent2_model}")
    
    for i, result in enumerate(results):
        run_index = i + 1  # Use simple 1-based indexing
        
        # Get final utilities from new structure
        final_utilities = result.get('final_utilities', {})
        if not final_utilities:
            print(f"Warning: No final_utilities in run {run_index}")
            continue
        
        agent1_utility = final_utilities.get(agent1_name, 0)
        agent2_utility = final_utilities.get(agent2_name, 0)
        
        # Determine winner
        if agent1_utility > agent2_utility:
            winner = agent1_model
            agent1_wins += 1
        elif agent2_utility > agent1_utility:
            winner = agent2_model
            agent2_wins += 1
        else:
            winner = "Tie"
            ties += 1
        
        detailed_results.append({
            'run': run_index,
            'agent1_utility': agent1_utility,
            'agent2_utility': agent2_utility,
            'agent1_model': agent1_model,
            'agent2_model': agent2_model,
            'winner': winner,
            'utility_difference': abs(agent1_utility - agent2_utility)
        })
    
    return {
        'detailed_results': detailed_results,
        'agent1_model': agent1_model,
        'agent2_model': agent2_model,
        'summary': {
            'total_games': total_games,
            'agent1_wins': agent1_wins,
            'agent2_wins': agent2_wins,
            'ties': ties,
            'agent1_win_rate': agent1_wins / total_games if total_games > 0 else 0,
            'agent2_win_rate': agent2_wins / total_games if total_games > 0 else 0,
            'tie_rate': ties / total_games if total_games > 0 else 0
        }
    }

def main():
    # Allow command line argument for results directory
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Default to listing available directories
        base_dir = "/Users/joie/Desktop/bargain/experiments/results"
        all_strong_model_dirs = glob.glob(f"{base_dir}/strong_models_202508*")
        
        if not all_strong_model_dirs:
            print("No strong_models_20250819* directories found in experiments/results/")
            return
        
        # Filter for non-empty directories with JSON files and count runs
        strong_model_dirs = []
        for dir_path in sorted(all_strong_model_dirs):
            # Check for both batch and single experiment files
            experiment_files = glob.glob(f"{dir_path}/run_*_experiment_results.json")
            experiment_files.extend(glob.glob(f"{dir_path}/experiment_results.json"))
            
            # Count total number of runs
            num_runs = len(experiment_files)
            if num_runs == 0:
                # If no direct experiment files found, try counting interaction files
                interaction_files = glob.glob(f"{dir_path}/agent_interactions/run_*_agent_*_interactions.json")
                run_numbers = set()
                for file in interaction_files:
                    match = re.search(r'run_(\d+)_', file)
                    if match:
                        run_numbers.add(int(match.group(1)))
                num_runs = max(len(run_numbers), 1 if glob.glob(f"{dir_path}/agent_*_interactions.json") else 0)
            
            if num_runs > 0:
                strong_model_dirs.append((dir_path, num_runs))

        # Print available directories
        print("Available strong_models directories (with data):")
        for i, (dir_path, num_runs) in enumerate(strong_model_dirs):
            dir_name = os.path.basename(dir_path)
            print(f"  {i}: {dir_name} ({num_runs} runs)")
        
        if len(strong_model_dirs) == 1:
            results_dir = strong_model_dirs[0][0] # Get the path from the tuple
            print(f"\nUsing the only available directory: {os.path.basename(results_dir)}")
        else:
            choice = input(f"\nEnter choice (0-{len(strong_model_dirs)-1}), or full path to directory: ")
            try:
                choice_num = int(choice)
                if 0 <= choice_num < len(strong_model_dirs):
                    results_dir = strong_model_dirs[choice_num][0] # Get the path from the tuple
                else:
                    print("Invalid choice")
                    return
            except ValueError:
                # Assume it's a full path
                results_dir = choice
    
    print(f"\nLoading results from {results_dir}...")
    results = load_run_results(results_dir)
    
    if not results:
        print("No results found!")
        return
    
    print(f"\nAnalyzing {len(results)} runs...")
    analysis = analyze_win_rates(results)
    
    if not analysis:
        return
    
    agent1_model = analysis['agent1_model']
    agent2_model = analysis['agent2_model']
    
    # Print detailed results
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)
    print(f"{'Run':<4} {f'{agent1_model[:18]}...' if len(agent1_model) > 18 else agent1_model:<20} " + 
          f"{f'{agent2_model[:18]}...' if len(agent2_model) > 18 else agent2_model:<20} {'Winner':<20} {'Difference':<10}")
    print("-" * 80)
    
    for result in analysis['detailed_results']:
        print(f"{result['run']:<4} {result['agent1_utility']:<20.2f} {result['agent2_utility']:<20.2f} "
              f"{result['winner'][:18]+'...' if len(result['winner']) > 18 else result['winner']:<20} {result['utility_difference']:<10.2f}")
    
    # Print summary
    summary = analysis['summary']
    print("\n" + "="*80)
    print("WIN RATE SUMMARY")
    print("="*80)
    print(f"Total Games: {summary['total_games']}")
    print(f"{agent1_model} Wins: {summary['agent1_wins']} ({summary['agent1_win_rate']:.1%})")
    print(f"{agent2_model} Wins: {summary['agent2_wins']} ({summary['agent2_win_rate']:.1%})")
    print(f"Ties: {summary['ties']} ({summary['tie_rate']:.1%})")
    
    # Calculate average utilities
    agent1_utilities = [r['agent1_utility'] for r in analysis['detailed_results']]
    agent2_utilities = [r['agent2_utility'] for r in analysis['detailed_results']]
    
    print(f"\nAverage {agent1_model} Utility: {sum(agent1_utilities) / len(agent1_utilities):.2f}")
    print(f"Average {agent2_model} Utility: {sum(agent2_utilities) / len(agent2_utilities):.2f}")
    
    # Calculate utility advantage
    total_advantage = sum(agent2_utilities) - sum(agent1_utilities)
    avg_advantage = total_advantage / len(results)
    print(f"Average {agent2_model} Advantage: {avg_advantage:.2f} utility points")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Collect and aggregate Qwen experiment results.

For each model combination and competition level, finds the most recent results
and extracts key metrics: consensus reached, round number, and payoffs.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import csv
import argparse


def parse_timestamp_from_filename(filename: str) -> Optional[datetime]:
    """Extract timestamp from filename like run_3_experiment_results_20251123_165726.json"""
    # Pattern: _YYYYMMDD_HHMMSS.json
    pattern = r'_(\d{8}_\d{6})\.json$'
    match = re.search(pattern, filename)
    if match:
        try:
            return datetime.strptime(match.group(1), '%Y%m%d_%H%M%S')
        except ValueError:
            return None
    return None


def select_most_recent_file(files: List[Path]) -> Optional[Path]:
    """
    Select the most recent file from a list of files.
    
    Logic:
    - If there are timestamped files, use the one with the most recent timestamp
    - If there are no timestamped files, use the non-timestamped one
    - If there are both timestamped and non-timestamped, prefer timestamped
    """
    if not files:
        return None
    
    # Separate timestamped and non-timestamped files
    timestamped = []
    non_timestamped = []
    
    for file in files:
        timestamp = parse_timestamp_from_filename(file.name)
        if timestamp:
            timestamped.append((file, timestamp))
        else:
            non_timestamped.append(file)
    
    # If we have timestamped files, use the most recent one
    if timestamped:
        timestamped.sort(key=lambda x: x[1], reverse=True)
        return timestamped[0][0]
    
    # Otherwise, use the non-timestamped one (should only be one)
    if non_timestamped:
        return non_timestamped[0]
    
    return None


def find_run_files(exp_dir: Path, run_number: int) -> Optional[Path]:
    """
    Find the most recent experiment results file for a given run number.
    
    Looks for files matching:
    - run_{N}_experiment_results.json
    - run_{N}_experiment_results_{TIMESTAMP}.json
    """
    base_pattern = f"run_{run_number}_experiment_results"
    matching_files = []
    
    # Find all matching files
    for file in exp_dir.glob(f"{base_pattern}*.json"):
        if file.name.startswith(base_pattern):
            matching_files.append(file)
    
    return select_most_recent_file(matching_files)


def extract_run_metrics(result_file: Path) -> Optional[Dict]:
    """Extract key metrics from an experiment results file."""
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract key information
        consensus_reached = data.get('consensus_reached', False)
        final_round = data.get('final_round', 0)
        final_utilities = data.get('final_utilities', {})
        final_allocation = data.get('final_allocation', {})
        config = data.get('config', {})
        competition_level = config.get('competition_level', None)
        random_seed = config.get('random_seed', None)
        
        # Extract agent payoffs (utilities)
        agent_payoffs = {}
        if final_utilities:
            agent_payoffs = final_utilities
        elif data.get('agent_performance'):
            # Fallback to agent_performance if final_utilities not available
            agent_perf = data.get('agent_performance', {})
            for agent_id, perf_data in agent_perf.items():
                if isinstance(perf_data, dict):
                    agent_payoffs[agent_id] = perf_data.get('final_utility', 0)
                else:
                    agent_payoffs[agent_id] = 0
        
        return {
            'consensus_reached': consensus_reached,
            'final_round': final_round,
            'agent_payoffs': agent_payoffs,
            'final_allocation': final_allocation,
            'competition_level': competition_level,
            'random_seed': random_seed,
            'experiment_id': data.get('experiment_id', ''),
            'file_path': str(result_file)
        }
    except Exception as e:
        print(f"Error reading {result_file}: {e}")
        return None


def parse_experiment_directory(exp_dir: Path) -> Optional[Dict]:
    """Parse an experiment directory to extract model and competition level."""
    dirname = exp_dir.name
    
    # Pattern: Qwen2.5-{SIZE}B-Instruct_vs_claude-3-7-sonnet_runs5_comp{LEVEL}
    pattern = r'Qwen2\.5-(\d+(?:\.\d+)?)B-Instruct_vs_claude-3-7-sonnet.*comp(\d+)'
    match = re.search(pattern, dirname)
    
    if not match:
        return None
    
    model_size = match.group(1)
    competition_level = float(match.group(2))
    
    return {
        'model_size': model_size,
        'competition_level': competition_level,
        'directory': exp_dir
    }


def process_experiment_directory(exp_dir: Path) -> Dict:
    """Process a single experiment directory and extract all run results."""
    exp_info = parse_experiment_directory(exp_dir)
    if not exp_info:
        return {'error': f'Could not parse directory: {exp_dir.name}'}
    
    results = {
        'model_size': exp_info['model_size'],
        'competition_level': exp_info['competition_level'],
        'directory': str(exp_dir),
        'runs': {}
    }
    
    # Find all run files (check for runs 1-10)
    for run_num in range(1, 11):
        run_file = find_run_files(exp_dir, run_num)
        if run_file:
            metrics = extract_run_metrics(run_file)
            if metrics:
                results['runs'][run_num] = metrics
    
    return results


def collect_all_results(results_dir: Path, models: List[str] = None) -> List[Dict]:
    """
    Collect results from all experiment directories.
    
    Args:
        results_dir: Path to experiments/results directory
        models: Optional list of model sizes to filter (e.g., ['3', '7', '14', '32', '72'])
    """
    if not results_dir.exists():
        print(f"Results directory does not exist: {results_dir}")
        return []
    
    # Find all experiment directories
    exp_dirs = []
    for item in results_dir.iterdir():
        if item.is_dir() and 'Qwen2.5' in item.name and 'claude-3-7-sonnet' in item.name:
            exp_dirs.append(item)
    
    # Filter by model sizes if specified
    if models:
        filtered_dirs = []
        for exp_dir in exp_dirs:
            exp_info = parse_experiment_directory(exp_dir)
            if exp_info and exp_info['model_size'] in models:
                filtered_dirs.append(exp_dir)
        exp_dirs = filtered_dirs
    
    # Process directories in parallel
    all_results = []
    with ProcessPoolExecutor() as executor:
        future_to_dir = {executor.submit(process_experiment_directory, exp_dir): exp_dir 
                         for exp_dir in exp_dirs}
        
        for future in as_completed(future_to_dir):
            exp_dir = future_to_dir[future]
            try:
                result = future.result()
                if 'error' not in result:
                    all_results.append(result)
                else:
                    print(f"Warning: {result['error']}")
            except Exception as e:
                print(f"Error processing {exp_dir}: {e}")
    
    return all_results


def format_results_for_output(results: List[Dict]) -> List[Dict]:
    """Format results into a flat structure for CSV/display."""
    formatted = []
    
    for exp_result in results:
        model_size = exp_result['model_size']
        comp_level = exp_result['competition_level']
        
        for run_num, run_data in sorted(exp_result['runs'].items()):
            row = {
                'model_size': model_size,
                'competition_level': comp_level,
                'run_number': run_num,
                'consensus_reached': run_data['consensus_reached'],
                'final_round': run_data['final_round'],
                'random_seed': run_data.get('random_seed', ''),
                'experiment_id': run_data.get('experiment_id', ''),
            }
            
            # Add agent payoffs
            payoffs = run_data.get('agent_payoffs', {})
            for agent_id, payoff in sorted(payoffs.items()):
                row[f'{agent_id}_payoff'] = payoff
            
            formatted.append(row)
    
    return formatted


def save_to_csv(formatted_results: List[Dict], output_file: Path):
    """Save formatted results to CSV."""
    if not formatted_results:
        print("No results to save")
        return
    
    # Get all unique column names
    columns = set()
    for row in formatted_results:
        columns.update(row.keys())
    
    columns = sorted(columns)
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(formatted_results)
    
    print(f"Results saved to {output_file}")


def print_summary(results: List[Dict]):
    """Print a summary of collected results."""
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    
    # Group by model and competition level
    by_model_comp = {}
    for exp_result in results:
        key = (exp_result['model_size'], exp_result['competition_level'])
        if key not in by_model_comp:
            by_model_comp[key] = []
        by_model_comp[key].append(exp_result)
    
    for (model_size, comp_level), exp_results in sorted(by_model_comp.items()):
        print(f"\nModel: Qwen2.5-{model_size}B | Competition Level: {comp_level}")
        print("-" * 80)
        
        total_runs = 0
        consensus_count = 0
        total_rounds = 0
        
        for exp_result in exp_results:
            runs = exp_result['runs']
            total_runs += len(runs)
            for run_data in runs.values():
                if run_data['consensus_reached']:
                    consensus_count += 1
                    total_rounds += run_data['final_round']
        
        consensus_rate = (consensus_count / total_runs * 100) if total_runs > 0 else 0
        avg_rounds = (total_rounds / consensus_count) if consensus_count > 0 else 0
        
        print(f"  Total runs: {total_runs}")
        print(f"  Consensus reached: {consensus_count} ({consensus_rate:.1f}%)")
        print(f"  Average rounds (when consensus): {avg_rounds:.1f}")


def main():
    parser = argparse.ArgumentParser(description='Collect Qwen experiment results')
    parser.add_argument('--results-dir', type=str, 
                       default='experiments/results',
                       help='Path to results directory')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['3', '7', '14', '32', '72'],
                       help='Model sizes to include (default: 3 7 14 32 72)')
    parser.add_argument('--output-csv', type=str,
                       help='Output CSV file path (optional)')
    parser.add_argument('--output-json', type=str,
                       help='Output JSON file path (optional)')
    parser.add_argument('--no-summary', action='store_true',
                       help='Skip printing summary')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    print(f"Collecting results from: {results_dir}")
    print(f"Models: {', '.join(args.models)}B")
    
    # Collect results
    all_results = collect_all_results(results_dir, models=args.models)
    
    if not all_results:
        print("No results found!")
        return
    
    # Format for output
    formatted_results = format_results_for_output(all_results)
    
    # Print summary
    if not args.no_summary:
        print_summary(all_results)
    
    # Save to CSV
    if args.output_csv:
        save_to_csv(formatted_results, Path(args.output_csv))
    
    # Save to JSON
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Full results saved to {args.output_json}")


if __name__ == '__main__':
    main()


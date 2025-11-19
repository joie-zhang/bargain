#!/usr/bin/env python3
"""
Analyze results from Qwen2.5 experiments vs Claude-3.7-Sonnet.
Supports both competition_level=0 (cooperation) and competition_level=1 (competition).
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional
import argparse

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "experiments" / "results"


def find_qwen_experiment_dirs(competition_level: Optional[float] = None) -> List[Path]:
    """Find all Qwen experiment result directories."""
    dirs = []
    for result_dir in RESULTS_DIR.iterdir():
        if not result_dir.is_dir():
            continue
        
        # Check if it's a Qwen experiment directory
        dir_name = result_dir.name
        if "Qwen2.5" in dir_name and "claude-3-7-sonnet" in dir_name:
            # Check competition level if specified
            if competition_level is not None:
                comp_str = f"comp{competition_level}".replace(".", "_")
                if comp_str not in dir_name:
                    continue
            dirs.append(result_dir)
    
    return sorted(dirs)


def load_batch_results(result_dir: Path) -> Optional[Dict[str, Any]]:
    """Load batch summary if available."""
    summary_file = result_dir / "_summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            return json.load(f)
    return None


def load_individual_results(result_dir: Path) -> List[Dict[str, Any]]:
    """Load all individual run results from a batch directory."""
    results = []
    
    # Look for run_*_experiment_results.json files
    for run_file in sorted(result_dir.glob("run_*_experiment_results.json")):
        try:
            with open(run_file) as f:
                results.append(json.load(f))
        except Exception as e:
            print(f"Warning: Could not load {run_file}: {e}", file=sys.stderr)
    
    # Also check for single experiment_results.json
    single_result = result_dir / "experiment_results.json"
    if single_result.exists() and not results:
        try:
            with open(single_result) as f:
                results.append(json.load(f))
        except Exception as e:
            print(f"Warning: Could not load {single_result}: {e}", file=sys.stderr)
    
    return results


def extract_model_from_dirname(dirname: str) -> Optional[str]:
    """Extract Qwen model name from directory name."""
    for model in ["Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct",
                  "Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct", 
                  "Qwen2.5-32B-Instruct", "Qwen2.5-72B-Instruct"]:
        if model in dirname:
            return model
    return None


def extract_competition_level(dirname: str) -> Optional[float]:
    """Extract competition level from directory name."""
    import re
    match = re.search(r'comp([0-9_]+)', dirname)
    if match:
        comp_str = match.group(1).replace("_", ".")
        try:
            return float(comp_str)
        except ValueError:
            pass
    return None


def analyze_results(competition_level: Optional[float] = None, verbose: bool = False):
    """Analyze Qwen experiment results."""
    
    print("=" * 80)
    print("QWEN2.5 EXPERIMENTS ANALYSIS")
    print("=" * 80)
    print()
    
    # Find all Qwen experiment directories
    experiment_dirs = find_qwen_experiment_dirs(competition_level)
    
    if not experiment_dirs:
        print(f"No Qwen experiment results found")
        if competition_level is not None:
            print(f"(filtered by competition_level={competition_level})")
        return
    
    print(f"Found {len(experiment_dirs)} experiment result directories")
    print()
    
    # Organize results by model and competition level
    results_by_model = defaultdict(lambda: defaultdict(list))
    
    for result_dir in experiment_dirs:
        qwen_model = extract_model_from_dirname(result_dir.name)
        comp_level = extract_competition_level(result_dir.name)
        
        if qwen_model is None:
            if verbose:
                print(f"Warning: Could not extract model from {result_dir.name}")
            continue
        
        # Load batch summary if available
        batch_summary = load_batch_results(result_dir)
        if batch_summary:
            results_by_model[qwen_model][comp_level].append({
                'type': 'batch_summary',
                'data': batch_summary,
                'dir': result_dir
            })
        else:
            # Load individual results
            individual_results = load_individual_results(result_dir)
            if individual_results:
                results_by_model[qwen_model][comp_level].append({
                    'type': 'individual',
                    'data': individual_results,
                    'dir': result_dir
                })
    
    # Print analysis
    for comp_level in sorted(set(comp for model_results in results_by_model.values() 
                                  for comp in model_results.keys() if comp is not None)):
        comp_label = "Cooperation" if comp_level == 0.0 else "Competition" if comp_level == 1.0 else f"Comp={comp_level}"
        print("=" * 80)
        print(f"COMPETITION LEVEL: {comp_level} ({comp_label})")
        print("=" * 80)
        print()
        
        # Models in order: 3B, 7B, 14B
        model_order = ["Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct"]
        
        for qwen_model in model_order:
            if qwen_model not in results_by_model or comp_level not in results_by_model[qwen_model]:
                continue
            
            model_results = results_by_model[qwen_model][comp_level]
            
            print(f"\n{qwen_model}")
            print("-" * 80)
            
            total_runs = 0
            consensus_count = 0
            total_rounds = []
            qwen_utilities = []
            claude_utilities = []
            exploitation_count = 0
            
            for result_entry in model_results:
                if result_entry['type'] == 'batch_summary':
                    summary = result_entry['data']
                    total_runs += summary.get('num_runs', 0)
                    consensus_count += int(summary.get('consensus_rate', 0) * summary.get('num_runs', 0))
                    
                    # Extract from individual experiments if available
                    for exp in summary.get('experiments', []):
                        if exp.get('consensus_reached'):
                            total_rounds.append(exp.get('final_round', 0))
                            
                            # Get utilities
                            utilities = exp.get('final_utilities', {})
                            agents = exp.get('config', {}).get('agents', [])
                            
                            if utilities and len(agents) >= 2:
                                agent_ids = list(utilities.keys())
                                if len(agent_ids) >= 2:
                                    qwen_utilities.append(utilities[agent_ids[0]])
                                    claude_utilities.append(utilities[agent_ids[1]])
                            
                            # Check exploitation
                            if exp.get('exploitation_detected'):
                                exploitation_count += 1
                
                elif result_entry['type'] == 'individual':
                    individual = result_entry['data']
                    total_runs += len(individual)
                    
                    for exp in individual:
                        if exp.get('consensus_reached'):
                            consensus_count += 1
                            total_rounds.append(exp.get('final_round', 0))
                            
                            # Get utilities - identify which agent is Qwen
                            utilities = exp.get('final_utilities', {})
                            agents = exp.get('config', {}).get('agents', [])
                            
                            if utilities and len(agents) >= 2:
                                # First agent is typically the first model (Qwen)
                                # Second agent is typically the second model (Claude)
                                agent_ids = list(utilities.keys())
                                if len(agent_ids) >= 2:
                                    qwen_utilities.append(utilities[agent_ids[0]])
                                    claude_utilities.append(utilities[agent_ids[1]])
                            
                            if exp.get('exploitation_detected'):
                                exploitation_count += 1
            
            if total_runs > 0:
                consensus_rate = consensus_count / total_runs
                avg_rounds = sum(total_rounds) / len(total_rounds) if total_rounds else 0
                avg_qwen_util = sum(qwen_utilities) / len(qwen_utilities) if qwen_utilities else 0
                avg_claude_util = sum(claude_utilities) / len(claude_utilities) if claude_utilities else 0
                exploitation_rate = exploitation_count / total_runs
                
                print(f"  Total Runs: {total_runs}")
                print(f"  Consensus Rate: {consensus_rate:.1%} ({consensus_count}/{total_runs})")
                if total_rounds:
                    print(f"  Avg Rounds to Consensus: {avg_rounds:.1f}")
                if qwen_utilities and claude_utilities:
                    print(f"  Avg Qwen Utility: {avg_qwen_util:.1f}")
                    print(f"  Avg Claude Utility: {avg_claude_util:.1f}")
                    print(f"  Utility Difference (Claude - Qwen): {avg_claude_util - avg_qwen_util:+.1f}")
                print(f"  Exploitation Detected: {exploitation_rate:.1%} ({exploitation_count}/{total_runs})")
                
                # Show result directories
                if verbose:
                    print(f"  Result Directories:")
                    for result_entry in model_results:
                        print(f"    - {result_entry['dir'].name}")
            else:
                print(f"  No results found")
            
            print()
    
    # Summary comparison
    print("=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    print()
    
    # Compare competition levels if both exist
    comp0_results = defaultdict(dict)
    comp1_results = defaultdict(dict)
    
    for qwen_model in model_order:
        if qwen_model in results_by_model:
            if 0.0 in results_by_model[qwen_model]:
                comp0_results[qwen_model] = results_by_model[qwen_model][0.0]
            if 1.0 in results_by_model[qwen_model]:
                comp1_results[qwen_model] = results_by_model[qwen_model][1.0]
    
    if comp0_results and comp1_results:
        print("Comparison: Competition Level 0 (Cooperation) vs 1 (Competition)")
        print("-" * 80)
        print(f"{'Model':<30} {'Comp=0 Consensus':<20} {'Comp=1 Consensus':<20}")
        print("-" * 80)
        
        for qwen_model in model_order:
            if qwen_model in comp0_results or qwen_model in comp1_results:
                comp0_rate = "N/A"
                comp1_rate = "N/A"
                
                if qwen_model in comp0_results:
                    # Calculate consensus rate
                    total = 0
                    consensus = 0
                    for result_entry in comp0_results[qwen_model]:
                        if result_entry['type'] == 'batch_summary':
                            summary = result_entry['data']
                            total += summary.get('num_runs', 0)
                            consensus += int(summary.get('consensus_rate', 0) * summary.get('num_runs', 0))
                        else:
                            individual = result_entry['data']
                            total += len(individual)
                            consensus += sum(1 for exp in individual if exp.get('consensus_reached'))
                    if total > 0:
                        comp0_rate = f"{consensus/total:.1%}"
                
                if qwen_model in comp1_results:
                    total = 0
                    consensus = 0
                    for result_entry in comp1_results[qwen_model]:
                        if result_entry['type'] == 'batch_summary':
                            summary = result_entry['data']
                            total += summary.get('num_runs', 0)
                            consensus += int(summary.get('consensus_rate', 0) * summary.get('num_runs', 0))
                        else:
                            individual = result_entry['data']
                            total += len(individual)
                            consensus += sum(1 for exp in individual if exp.get('consensus_reached'))
                    if total > 0:
                        comp1_rate = f"{consensus/total:.1%}"
                
                print(f"{qwen_model:<30} {comp0_rate:<20} {comp1_rate:<20}")
    
    print()
    print("=" * 80)
    print("Analysis complete!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Analyze Qwen2.5 experiment results")
    parser.add_argument(
        "--competition-level",
        type=float,
        default=None,
        help="Filter by competition level (0.0 or 1.0). If not specified, shows all."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed information including result directories"
    )
    
    args = parser.parse_args()
    
    analyze_results(
        competition_level=args.competition_level,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()


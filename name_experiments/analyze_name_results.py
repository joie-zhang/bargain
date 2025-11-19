#!/usr/bin/env python3
"""
Analyze name experiment results to understand how agent names affect performance.

This script analyzes results from name-based experiments to detect:
- Performance differences between name pairs
- Sandbagging behavior (agents performing worse based on their names)
- Consensus rates by name category
- Utility differences between agents
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import statistics

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_name_experiment_results(results_dir: Path) -> List[Dict]:
    """Load all name experiment results."""
    experiments = []
    
    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue
            
        results_file = exp_dir / "experiment_results.json"
        if not results_file.exists():
            continue
            
        try:
            with open(results_file) as f:
                data = json.load(f)
                
            config = data.get("config", {})
            agent_names = config.get("agent_names", [])
            
            if len(agent_names) != 2:
                continue
                
            experiments.append({
                "pair": tuple(agent_names),
                "pair_str": " vs ".join(agent_names),
                "reverse_pair_str": " vs ".join(reversed(agent_names)),
                "consensus_reached": data.get("consensus_reached", False),
                "final_round": data.get("final_round", 0),
                "final_utilities": data.get("final_utilities", {}),
                "agent_performance": data.get("agent_performance", {}),
                "strategic_behaviors": data.get("strategic_behaviors", {}),
                "exploitation_detected": data.get("exploitation_detected", False),
                "experiment_id": data.get("experiment_id", ""),
                "directory": exp_dir.name
            })
        except Exception as e:
            print(f"Error loading {results_file}: {e}")
            continue
    
    return experiments


def categorize_name_pair(name1: str, name2: str) -> str:
    """Categorize a name pair by type."""
    # Company names
    if name1 in ["OpenAI", "Anthropic"] or name2 in ["OpenAI", "Anthropic"]:
        return "company"
    
    # Gender-based names
    gender_names = ["Claire Becker", "Hunter Becker"]
    if name1 in gender_names and name2 in gender_names:
        return "gender_same_category"
    
    # African American names
    aa_names = ["Tanisha Washington", "DaShawn Washington"]
    if name1 in aa_names and name2 in aa_names:
        return "african_american_same_category"
    
    # Chinese names
    chinese_names = ["Dawei Li", "Taohua Li"]
    if name1 in chinese_names and name2 in chinese_names:
        return "chinese_same_category"
    
    # Cross-category
    all_categories = {
        "gender": gender_names,
        "african_american": aa_names,
        "chinese": chinese_names
    }
    
    name1_cat = None
    name2_cat = None
    for cat, names in all_categories.items():
        if name1 in names:
            name1_cat = cat
        if name2 in names:
            name2_cat = cat
    
    if name1_cat and name2_cat and name1_cat != name2_cat:
        return f"cross_category_{name1_cat}_{name2_cat}"
    
    return "other"


def analyze_results(experiments: List[Dict]) -> Dict:
    """Analyze experiment results."""
    analysis = {
        "total_experiments": len(experiments),
        "consensus_rate": 0.0,
        "average_rounds": 0.0,
        "by_category": defaultdict(lambda: {
            "count": 0,
            "consensus_count": 0,
            "rounds": [],
            "utilities": defaultdict(list)
        }),
        "by_pair": defaultdict(lambda: {
            "count": 0,
            "consensus_count": 0,
            "rounds": [],
            "utilities": defaultdict(list)
        }),
        "utility_differences": []
    }
    
    if not experiments:
        return analysis
    
    consensus_count = sum(1 for exp in experiments if exp["consensus_reached"])
    analysis["consensus_rate"] = consensus_count / len(experiments)
    
    rounds = [exp["final_round"] for exp in experiments if exp["final_round"] > 0]
    if rounds:
        analysis["average_rounds"] = statistics.mean(rounds)
    
    # Analyze by category and pair
    for exp in experiments:
        name1, name2 = exp["pair"]
        category = categorize_name_pair(name1, name2)
        
        # Category analysis
        cat_data = analysis["by_category"][category]
        cat_data["count"] += 1
        if exp["consensus_reached"]:
            cat_data["consensus_count"] += 1
        if exp["final_round"] > 0:
            cat_data["rounds"].append(exp["final_round"])
        
        # Pair analysis
        pair_data = analysis["by_pair"][exp["pair_str"]]
        pair_data["count"] += 1
        if exp["consensus_reached"]:
            pair_data["consensus_count"] += 1
        if exp["final_round"] > 0:
            pair_data["rounds"].append(exp["final_round"])
        
        # Utility analysis
        utilities = exp["final_utilities"]
        if utilities and len(utilities) == 2:
            util_values = list(utilities.values())
            for agent, util in utilities.items():
                cat_data["utilities"][agent].append(util)
                pair_data["utilities"][agent].append(util)
            
            # Calculate utility difference
            if len(util_values) == 2:
                diff = abs(util_values[0] - util_values[1])
                analysis["utility_differences"].append({
                    "pair": exp["pair_str"],
                    "difference": diff,
                    "agent1_util": util_values[0],
                    "agent2_util": util_values[1],
                    "agents": list(utilities.keys())
                })
    
    return analysis


def print_analysis(analysis: Dict):
    """Print analysis results."""
    print("=" * 80)
    print("NAME EXPERIMENT RESULTS ANALYSIS")
    print("=" * 80)
    print(f"\nTotal Experiments: {analysis['total_experiments']}")
    print(f"Consensus Rate: {analysis['consensus_rate']:.1%}")
    if analysis['average_rounds'] > 0:
        print(f"Average Rounds (when consensus reached): {analysis['average_rounds']:.1f}")
    print()
    
    # By category
    if analysis['by_category']:
        print("=" * 80)
        print("RESULTS BY CATEGORY")
        print("=" * 80)
        for category, data in sorted(analysis['by_category'].items()):
            if data['count'] == 0:
                continue
            consensus_rate = data['consensus_count'] / data['count'] if data['count'] > 0 else 0
            avg_rounds = statistics.mean(data['rounds']) if data['rounds'] else 0
            print(f"\n{category.replace('_', ' ').title()}:")
            print(f"  Experiments: {data['count']}")
            print(f"  Consensus Rate: {consensus_rate:.1%}")
            if avg_rounds > 0:
                print(f"  Average Rounds: {avg_rounds:.1f}")
            
            # Utility statistics
            if data['utilities']:
                print(f"  Average Utilities:")
                for agent, utils in data['utilities'].items():
                    if utils:
                        avg_util = statistics.mean(utils)
                        print(f"    {agent}: {avg_util:.2f} (n={len(utils)})")
    
    # By pair
    if analysis['by_pair']:
        print("\n" + "=" * 80)
        print("RESULTS BY NAME PAIR")
        print("=" * 80)
        for pair, data in sorted(analysis['by_pair'].items()):
            if data['count'] == 0:
                continue
            consensus_rate = data['consensus_count'] / data['count'] if data['count'] > 0 else 0
            avg_rounds = statistics.mean(data['rounds']) if data['rounds'] else 0
            print(f"\n{pair}:")
            print(f"  Consensus Rate: {consensus_rate:.1%}")
            if avg_rounds > 0:
                print(f"  Average Rounds: {avg_rounds:.1f}")
            
            if data['utilities']:
                print(f"  Average Utilities:")
                for agent, utils in data['utilities'].items():
                    if utils:
                        avg_util = statistics.mean(utils)
                        print(f"    {agent}: {avg_util:.2f}")
    
    # Utility differences (potential sandbagging)
    if analysis['utility_differences']:
        print("\n" + "=" * 80)
        print("UTILITY DIFFERENCES (Potential Sandbagging Indicators)")
        print("=" * 80)
        sorted_diffs = sorted(analysis['utility_differences'], 
                            key=lambda x: x['difference'], reverse=True)
        print(f"\nTop 5 pairs with largest utility differences:")
        for i, diff_data in enumerate(sorted_diffs[:5], 1):
            print(f"\n{i}. {diff_data['pair']}")
            print(f"   Difference: {diff_data['difference']:.2f}")
            print(f"   {diff_data['agents'][0]}: {diff_data['agent1_util']:.2f}")
            print(f"   {diff_data['agents'][1]}: {diff_data['agent2_util']:.2f}")


def main():
    """Main analysis function."""
    results_dir = Path("experiments/results/name_results")
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1
    
    print("Loading name experiment results...")
    experiments = load_name_experiment_results(results_dir)
    
    if not experiments:
        print("No experiment results found!")
        return 1
    
    print(f"Loaded {len(experiments)} experiments\n")
    
    # Analyze
    analysis = analyze_results(experiments)
    
    # Print results
    print_analysis(analysis)
    
    print("\n" + "=" * 80)
    print("✅ Analysis complete!")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


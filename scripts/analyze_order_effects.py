#!/usr/bin/env python3
"""
Analyze order effects in negotiation experiments.
Compares outcomes when weak model goes first vs. second.
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "experiments/results/scaling_experiment"
CONFIG_DIR = RESULTS_DIR / "configs"
LOGS_DIR = RESULTS_DIR / "logs"

def load_results():
    """Load all experiment results."""
    results = []
    
    # Load each result file
    for config_file in sorted(CONFIG_DIR.glob("config_*.json")):
        exp_id = int(config_file.stem.split('_')[1])
        
        # Skip if not completed
        if not (LOGS_DIR / f"completed_{exp_id}.flag").exists():
            continue
        
        # Load config
        with open(config_file) as f:
            config = json.load(f)
        
        # Try to load result
        result_file = BASE_DIR / config['output_dir'] / f"result_{exp_id}.json"
        if result_file.exists():
            with open(result_file) as f:
                result = json.load(f)
                result.update(config)  # Merge config and result
                results.append(result)
    
    return pd.DataFrame(results)

def analyze_order_effects(df):
    """Analyze the effect of model ordering on outcomes."""
    
    if 'model_order' not in df.columns:
        print("⚠️  No model_order field found. Configs may not have both orderings.")
        return
    
    print("="*60)
    print("ORDER EFFECTS ANALYSIS")
    print("="*60)
    print()
    
    # Overall comparison
    print("1. OVERALL SUCCESS RATES BY ORDER")
    print("-" * 40)
    
    for order in ['weak_first', 'strong_first']:
        subset = df[df['model_order'] == order]
        if len(subset) > 0:
            success_rate = (subset['status'] == 'SUCCESS').mean() * 100
            print(f"  {order:15}: {success_rate:.1f}% success ({len(subset)} experiments)")
    
    # Win rates by order
    print("\n2. WIN RATES BY MODEL ORDER")
    print("-" * 40)
    
    # Calculate win rates for each ordering
    weak_first_wins = defaultdict(list)
    strong_first_wins = defaultdict(list)
    
    for _, row in df.iterrows():
        if row.get('winner_agent_id') is not None:
            key = f"{row['weak_model']} vs {row['strong_model']}"
            
            if row.get('model_order') == 'weak_first':
                # Weak model is agent_0
                weak_won = (row['winner_agent_id'] == 'agent_0')
                weak_first_wins[key].append(weak_won)
            elif row.get('model_order') == 'strong_first':
                # Strong model is agent_0, so weak is agent_1
                weak_won = (row['winner_agent_id'] == 'agent_1')
                strong_first_wins[key].append(weak_won)
    
    # Compare win rates
    print("\nWeak model win rate by position:")
    all_pairs = set(weak_first_wins.keys()) | set(strong_first_wins.keys())
    
    results_table = []
    for pair in sorted(all_pairs):
        weak_first_rate = np.mean(weak_first_wins[pair]) * 100 if pair in weak_first_wins else None
        strong_first_rate = np.mean(strong_first_wins[pair]) * 100 if pair in strong_first_wins else None
        
        if weak_first_rate is not None and strong_first_rate is not None:
            diff = weak_first_rate - strong_first_rate
            results_table.append({
                'Pairing': pair,
                'Weak goes 1st': f"{weak_first_rate:.1f}%",
                'Weak goes 2nd': f"{strong_first_rate:.1f}%",
                'Difference': f"{diff:+.1f}%"
            })
    
    if results_table:
        results_df = pd.DataFrame(results_table)
        print(results_df.to_string(index=False))
    
    # Competition level effects
    print("\n3. ORDER EFFECTS BY COMPETITION LEVEL")
    print("-" * 40)
    
    for comp_level in sorted(df['competition_level'].unique()):
        subset = df[df['competition_level'] == comp_level]
        
        weak_first = subset[subset.get('model_order', '') == 'weak_first']
        strong_first = subset[subset.get('model_order', '') == 'strong_first']
        
        if len(weak_first) > 0 and len(strong_first) > 0:
            wf_success = (weak_first['status'] == 'SUCCESS').mean() * 100
            sf_success = (strong_first['status'] == 'SUCCESS').mean() * 100
            
            print(f"\n  Competition Level {comp_level}:")
            print(f"    Weak first:   {wf_success:.1f}% success")
            print(f"    Strong first: {sf_success:.1f}% success")
            print(f"    Difference:   {wf_success - sf_success:+.1f}%")
    
    # Statistical significance
    print("\n4. STATISTICAL SIGNIFICANCE")
    print("-" * 40)
    
    # Test if order affects success rate
    weak_first_success = df[df.get('model_order', '') == 'weak_first']['status'] == 'SUCCESS'
    strong_first_success = df[df.get('model_order', '') == 'strong_first']['status'] == 'SUCCESS'
    
    if len(weak_first_success) > 0 and len(strong_first_success) > 0:
        chi2, p_value = stats.chi2_contingency([
            [weak_first_success.sum(), (~weak_first_success).sum()],
            [strong_first_success.sum(), (~strong_first_success).sum()]
        ])[:2]
        
        print(f"  Chi-squared test for order effect on success:")
        print(f"    χ² = {chi2:.3f}, p = {p_value:.4f}")
        
        if p_value < 0.05:
            print("    ✓ Significant order effect detected (p < 0.05)")
        else:
            print("    ✗ No significant order effect (p ≥ 0.05)")

def main():
    """Main analysis."""
    print("Loading experiment results...")
    
    df = load_results()
    
    if len(df) == 0:
        print("No completed experiments found.")
        return
    
    print(f"Loaded {len(df)} completed experiments")
    
    # Check if we have both orderings
    if 'model_order' in df.columns:
        weak_first_count = (df['model_order'] == 'weak_first').sum()
        strong_first_count = (df['model_order'] == 'strong_first').sum()
        
        print(f"  - Weak model first: {weak_first_count}")
        print(f"  - Strong model first: {strong_first_count}")
        
        if weak_first_count > 0 and strong_first_count > 0:
            analyze_order_effects(df)
        else:
            print("\n⚠️  Need both orderings to analyze order effects")
            print("  Run experiments with both orderings using:")
            print("  ./scripts/run_experiments_selective.sh 4 both")
    else:
        print("\n⚠️  Experiments don't include model_order field")
        print("  Regenerate configs with: ./scripts/generate_configs_both_orders.sh")

if __name__ == "__main__":
    main()
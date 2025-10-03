#!/usr/bin/env python3
"""Analyze token usage from experiment interaction logs."""

import json
import sys
import glob
import numpy as np
from collections import defaultdict
import re

def count_tokens_approx(text):
    """Approximate token count (words * 1.3 is a rough estimate)."""
    if not text:
        return 0
    words = len(text.split())
    return int(words * 1.3)

def categorize_phase(phase_name):
    """Categorize phase names into general categories."""
    if 'discussion' in phase_name:
        return 'discussion'
    elif 'proposal' in phase_name:
        return 'proposal'
    elif 'voting' in phase_name:
        return 'voting'
    elif 'reflection' in phase_name:
        return 'reflection'
    elif 'private_thinking' in phase_name:
        return 'private_thinking'
    elif 'game_setup' in phase_name:
        return 'game_setup'
    elif 'preference' in phase_name:
        return 'preference_assignment'
    else:
        return 'other'

def main():
    # Find interaction files
    pattern = "experiments/results_current/*/agent_interactions/*interactions.json"
    files = glob.glob(pattern)[:50]  # Analyze more files
    
    if not files:
        print("No interaction files found!")
        return
    
    print(f"Analyzing {len(files)} interaction files...\n")
    
    all_phase_tokens = defaultdict(list)
    raw_phase_tokens = defaultdict(list)
    
    for filepath in files:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        interactions = data.get('interactions', [])
        
        for interaction in interactions:
            phase_raw = interaction.get('phase', 'unknown')
            phase_category = categorize_phase(phase_raw)
            response = interaction.get('response', '')
            tokens = count_tokens_approx(response)
            
            all_phase_tokens[phase_category].append(tokens)
            raw_phase_tokens[phase_raw].append(tokens)
    
    # Calculate and display statistics by phase category
    print("=" * 80)
    print("TOKEN USAGE STATISTICS BY PHASE CATEGORY")
    print("=" * 80)
    print(f"{'Phase Category':<20} {'Avg Tokens':>12} {'Min':>8} {'Max':>8} {'Median':>8} {'Samples':>10}")
    print("-" * 80)
    
    phase_order = ['game_setup', 'preference_assignment', 'discussion', 'private_thinking', 
                   'proposal', 'voting', 'reflection', 'other']
    
    total_by_phase = {}
    for phase in phase_order:
        if phase in all_phase_tokens:
            tokens = all_phase_tokens[phase]
            if tokens:
                avg_tokens = np.mean(tokens)
                min_tokens = min(tokens)
                max_tokens = max(tokens)
                median_tokens = np.median(tokens)
                num_samples = len(tokens)
                total_by_phase[phase] = avg_tokens * num_samples
                print(f"{phase:<20} {avg_tokens:>12.0f} {min_tokens:>8} {max_tokens:>8} {median_tokens:>8.0f} {num_samples:>10}")
    
    # Show overall statistics
    all_tokens = []
    for tokens_list in all_phase_tokens.values():
        all_tokens.extend(tokens_list)
    
    if all_tokens:
        print("-" * 80)
        print(f"{'OVERALL':<20} {np.mean(all_tokens):>12.0f} {min(all_tokens):>8} {max(all_tokens):>8} {np.median(all_tokens):>8.0f} {len(all_tokens):>10}")
    
    # Show distribution
    print("\n" + "=" * 80)
    print("TOKEN DISTRIBUTION (ALL PHASES)")
    print("=" * 80)
    
    if all_tokens:
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            value = np.percentile(all_tokens, p)
            print(f"  {p:3d}th percentile: {value:>8.0f} tokens")
    
    # Show which phases use the most tokens
    print("\n" + "=" * 80)
    print("PHASES BY AVERAGE TOKEN USAGE")
    print("=" * 80)
    
    phase_averages = []
    for phase, tokens in all_phase_tokens.items():
        if tokens and phase != 'other':
            phase_averages.append((phase, np.mean(tokens), len(tokens)))
    
    phase_averages.sort(key=lambda x: x[1], reverse=True)
    
    for phase, avg, count in phase_averages:
        print(f"  {phase:<25} {avg:>8.0f} tokens (n={count})")
    
    # Sample some actual responses to show
    print("\n" + "=" * 80)
    print("SAMPLE RESPONSES BY TOKEN LENGTH")
    print("=" * 80)
    
    # Collect all responses with their token counts
    all_responses = []
    for filepath in files[:10]:  # Sample from first 10 files
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        agent_id = data.get('agent_id', 'unknown')
        interactions = data.get('interactions', [])
        
        for interaction in interactions:
            phase = interaction.get('phase', 'unknown')
            phase_cat = categorize_phase(phase)
            response = interaction.get('response', '')
            if response:  # Only non-empty responses
                tokens = count_tokens_approx(response)
                all_responses.append({
                    'tokens': tokens,
                    'phase': phase_cat,
                    'agent': agent_id,
                    'content': response
                })
    
    # Sort by token count
    all_responses.sort(key=lambda x: x['tokens'])
    
    # Show examples from different percentiles
    percentile_examples = [10, 50, 90, 99]
    
    for p in percentile_examples:
        idx = int(len(all_responses) * p / 100)
        if idx < len(all_responses):
            example = all_responses[idx]
            print(f"\n{p}th PERCENTILE EXAMPLE (~{example['tokens']} tokens)")
            print(f"  Phase: {example['phase']}")
            print(f"  Agent: {example['agent']}")
            print(f"  Response preview: {example['content'][:300]}...")
    
    # Recommendation based on data
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR TOKEN LIMITS")
    print("=" * 80)
    
    if all_tokens:
        p95 = np.percentile(all_tokens, 95)
        p99 = np.percentile(all_tokens, 99)
        
        print(f"\nBased on the analysis of {len(all_tokens)} responses:")
        print(f"  • 95% of responses use ≤ {p95:.0f} tokens")
        print(f"  • 99% of responses use ≤ {p99:.0f} tokens")
        print(f"\nSuggested token limits by phase:")
        
        for phase in ['discussion', 'proposal', 'voting', 'reflection', 'private_thinking']:
            if phase in all_phase_tokens:
                tokens = all_phase_tokens[phase]
                if tokens:
                    p95_phase = np.percentile(tokens, 95)
                    print(f"  --max-tokens-{phase.replace('_', '-')}: {int(p95_phase * 1.2)}")

if __name__ == "__main__":
    main()
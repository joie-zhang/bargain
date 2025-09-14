#!/usr/bin/env python3
"""Analyze token usage from experiment interaction logs."""

import json
import sys
import glob
import numpy as np
from collections import defaultdict

def count_tokens_approx(text):
    """Approximate token count (words * 1.3 is a rough estimate)."""
    if not text:
        return 0
    words = len(text.split())
    return int(words * 1.3)

def analyze_interaction_file(filepath):
    """Analyze a single interaction file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    phase_tokens = defaultdict(list)
    
    # Handle the structure: data is a dict with 'interactions' key
    interactions = data.get('interactions', [])
    
    for interaction in interactions:
        phase = interaction.get('phase', 'unknown')
        response = interaction.get('response', '')
        tokens = count_tokens_approx(response)
        phase_tokens[phase].append(tokens)
    
    return phase_tokens

def main():
    # Find interaction files
    pattern = "experiments/results_current/*/agent_interactions/*interactions.json"
    files = glob.glob(pattern)[:20]  # Analyze first 20 files
    
    if not files:
        print("No interaction files found!")
        return
    
    print(f"Analyzing {len(files)} interaction files...\n")
    
    all_phase_tokens = defaultdict(list)
    
    for filepath in files:
        # Extract model name from filepath
        model_name = filepath.split('/')[-1].replace('_interactions.json', '').replace('run_1_agent_', '')
        
        phase_tokens = analyze_interaction_file(filepath)
        
        # Aggregate data
        for phase, tokens_list in phase_tokens.items():
            all_phase_tokens[phase].extend(tokens_list)
    
    # Calculate and display statistics
    print("=" * 70)
    print("TOKEN USAGE STATISTICS BY PHASE")
    print("=" * 70)
    print(f"{'Phase':<20} {'Avg Tokens':>12} {'Min':>8} {'Max':>8} {'Samples':>10}")
    print("-" * 70)
    
    phase_order = ['discussion', 'proposal', 'voting', 'reflection', 'thinking']
    
    for phase in phase_order:
        if phase in all_phase_tokens:
            tokens = all_phase_tokens[phase]
            if tokens:
                avg_tokens = np.mean(tokens)
                min_tokens = min(tokens)
                max_tokens = max(tokens)
                num_samples = len(tokens)
                print(f"{phase:<20} {avg_tokens:>12.0f} {min_tokens:>8} {max_tokens:>8} {num_samples:>10}")
    
    # Show overall statistics
    all_tokens = []
    for tokens_list in all_phase_tokens.values():
        all_tokens.extend(tokens_list)
    
    if all_tokens:
        print("-" * 70)
        print(f"{'OVERALL':<20} {np.mean(all_tokens):>12.0f} {min(all_tokens):>8} {max(all_tokens):>8} {len(all_tokens):>10}")
    
    # Show distribution
    print("\n" + "=" * 70)
    print("TOKEN DISTRIBUTION")
    print("=" * 70)
    
    if all_tokens:
        percentiles = [25, 50, 75, 90, 95, 99]
        for p in percentiles:
            value = np.percentile(all_tokens, p)
            print(f"  {p}th percentile: {value:>8.0f} tokens")
    
    # Show examples of long responses
    print("\n" + "=" * 70)
    print("EXAMPLES OF LONGEST RESPONSES")
    print("=" * 70)
    
    # Find files with longest responses
    max_response_by_phase = defaultdict(lambda: {"tokens": 0, "file": "", "content": ""})
    
    for filepath in files[:10]:  # Check first 10 files for examples
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        interactions = data.get('interactions', [])
        for interaction in interactions:
            phase = interaction.get('phase', 'unknown')
            response = interaction.get('response', '')
            tokens = count_tokens_approx(response)
            
            if tokens > max_response_by_phase[phase]["tokens"]:
                max_response_by_phase[phase] = {
                    "tokens": tokens,
                    "file": filepath.split('/')[-1],
                    "content": response[:500] + "..." if len(response) > 500 else response
                }
    
    for phase in phase_order:
        if phase in max_response_by_phase:
            info = max_response_by_phase[phase]
            print(f"\n{phase.upper()} (max: {info['tokens']} tokens):")
            print(f"  From: {info['file']}")
            print(f"  Sample: {info['content'][:200]}...")

if __name__ == "__main__":
    main()
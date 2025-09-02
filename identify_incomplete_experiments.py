#!/usr/bin/env python3
"""
Identify experiments that have fewer than 3 runs (incomplete experiments).
"""

import json
from pathlib import Path
from collections import defaultdict

def main():
    results_dir = Path('/root/bargain/experiments/results')
    
    # Group experiments by their base configuration
    experiment_groups = defaultdict(list)
    
    print("Scanning experiment files...")
    for file_path in results_dir.glob('*_summary.json'):
        with open(file_path) as f:
            data = json.load(f)
        
        if 'experiments' in data and data['experiments']:
            for exp in data['experiments']:
                config = exp['config']
                agents = tuple(sorted([a.replace('_1', '').replace('_2', '') for a in config.get('agents', [])]))
                comp_level = config.get('competition_level')
                seed = config.get('random_seed')
                
                # Extract config number from filename
                if 'config' in file_path.name:
                    config_num = int(file_path.name.split('config')[1].split('_')[0])
                else:
                    config_num = -1
                
                key = (agents, comp_level)
                experiment_groups[key].append({
                    'file': file_path.name,
                    'config_num': config_num,
                    'seed': seed,
                    'consensus': exp.get('consensus_reached'),
                    'final_round': exp.get('final_round')
                })
    
    # Find incomplete experiments (fewer than 3 runs)
    print("\n" + "="*80)
    print("INCOMPLETE EXPERIMENTS (fewer than 3 runs)")
    print("="*80)
    
    incomplete = []
    complete = []
    
    for key, runs in experiment_groups.items():
        agents, comp_level = key
        if len(runs) < 3:
            incomplete.append((key, runs))
        else:
            complete.append((key, runs))
    
    # Sort incomplete by number of runs
    incomplete.sort(key=lambda x: len(x[1]))
    
    print(f"\nFound {len(incomplete)} incomplete experiments out of {len(experiment_groups)} total")
    print(f"Complete experiments (3 runs): {len(complete)}")
    print(f"Incomplete experiments (<3 runs): {len(incomplete)}")
    
    # Show details of incomplete experiments
    print("\n" + "-"*80)
    print("INCOMPLETE EXPERIMENTS DETAILS:")
    print("-"*80)
    
    by_run_count = defaultdict(list)
    for key, runs in incomplete:
        by_run_count[len(runs)].append((key, runs))
    
    for num_runs in sorted(by_run_count.keys()):
        print(f"\n### Experiments with {num_runs} run(s): {len(by_run_count[num_runs])} experiments")
        print("-"*60)
        
        for i, (key, runs) in enumerate(by_run_count[num_runs][:10]):  # Show first 10
            agents, comp_level = key
            agent_str = ' vs '.join(agents)
            print(f"\n{i+1}. {agent_str} @ competition={comp_level}")
            for run in runs:
                print(f"   Config {run['config_num']:3d}: seed={run['seed']:3d}, "
                      f"consensus={run['consensus']}, final_round={run['final_round']}")
                print(f"      File: {run['file']}")
        
        if len(by_run_count[num_runs]) > 10:
            print(f"\n   ... and {len(by_run_count[num_runs])-10} more experiments with {num_runs} run(s)")
    
    # Identify missing config numbers
    print("\n" + "="*80)
    print("MISSING CONFIG ANALYSIS")
    print("="*80)
    
    all_configs = set()
    for _, runs in experiment_groups.items():
        for run in runs:
            if run['config_num'] >= 0:
                all_configs.add(run['config_num'])
    
    # Check for gaps in config numbers
    if all_configs:
        min_config = min(all_configs)
        max_config = max(all_configs)
        expected_configs = set(range(min_config, max_config + 1))
        missing_configs = expected_configs - all_configs
        
        print(f"\nConfig range: {min_config} to {max_config}")
        print(f"Expected configs: {len(expected_configs)}")
        print(f"Found configs: {len(all_configs)}")
        print(f"Missing configs: {len(missing_configs)}")
        
        if missing_configs:
            print(f"\nMissing config numbers: {sorted(missing_configs)[:20]}")
            if len(missing_configs) > 20:
                print(f"... and {len(missing_configs)-20} more")
    
    # Analyze which models are most affected
    print("\n" + "="*80)
    print("MODELS MOST AFFECTED BY INCOMPLETE EXPERIMENTS")
    print("="*80)
    
    model_incomplete_count = defaultdict(int)
    for (agents, comp_level), runs in incomplete:
        for agent in agents:
            model_incomplete_count[agent] += 1
    
    sorted_models = sorted(model_incomplete_count.items(), key=lambda x: x[1], reverse=True)
    print("\nModels with most incomplete experiments:")
    for model, count in sorted_models[:10]:
        print(f"  {model:30s}: {count:3d} incomplete experiments")

if __name__ == "__main__":
    main()
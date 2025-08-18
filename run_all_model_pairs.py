#!/usr/bin/env python3
"""
Run All Model Pairs Experiment

This script runs negotiations between all possible pairs of strong language models,
including models negotiating against themselves. Each pair runs 10 times with
a maximum of 10 rounds per negotiation.

Models included:
- Gemini Pro 2.5
- Claude 3.5 Sonnet
- Llama 3.1 405B
- Qwen 2.5 72B
"""

import asyncio
import subprocess
import json
import time
from datetime import datetime
from pathlib import Path
from itertools import combinations_with_replacement
from typing import List, Tuple, Dict, Any


# Available models
MODELS = [
    "gemini-pro",
    "claude-3-5-sonnet", 
    "llama-3-1-405b",
    "qwen-2-5-72b"
]

# Configuration
NUM_RUNS = 10
MAX_ROUNDS = 10
NUM_ITEMS = 6
COMPETITION_LEVEL = 0.95


def get_model_pairs() -> List[Tuple[str, str]]:
    """
    Generate all possible pairs of models, including self-play.
    For 4 models, this gives us 10 unique pairs:
    - 4 self-play pairs (A-A, B-B, C-C, D-D)
    - 6 cross-play pairs (A-B, A-C, A-D, B-C, B-D, C-D)
    """
    # Use combinations_with_replacement to include self-pairs
    pairs = list(combinations_with_replacement(MODELS, 2))
    return pairs


def create_pair_name(model1: str, model2: str) -> str:
    """Create a descriptive name for the model pair."""
    return f"{model1}_vs_{model2}"


def run_experiment(model1: str, model2: str, output_dir: Path) -> Dict[str, Any]:
    """
    Run a single experiment for a model pair.
    
    Args:
        model1: First model name
        model2: Second model name
        output_dir: Directory to save results
        
    Returns:
        Dictionary with experiment results and metadata
    """
    pair_name = create_pair_name(model1, model2)
    print(f"\n{'='*60}")
    print(f"Running experiment: {pair_name}")
    print(f"Models: {model1} vs {model2}")
    print(f"Configuration: {NUM_RUNS} runs, max {MAX_ROUNDS} rounds")
    print(f"{'='*60}")
    
    # Create subdirectory for this pair
    pair_dir = output_dir / pair_name
    pair_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        "python", "run_strong_models_experiment.py",
        "--models", model1, model2,
        "--runs", str(NUM_RUNS),
        "--rounds", str(MAX_ROUNDS),
        "--items", str(NUM_ITEMS),
        "--competition", str(COMPETITION_LEVEL)
    ]
    
    # Record start time
    start_time = time.time()
    start_datetime = datetime.now()
    
    try:
        # Run the experiment
        print(f"Starting at: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env={**subprocess.os.environ, "PYTHONUNBUFFERED": "1"}
        )
        
        # Record end time
        end_time = time.time()
        duration = end_time - start_time
        
        # Check if successful
        if result.returncode == 0:
            print(f"✓ Experiment completed successfully in {duration:.1f} seconds")
            
            # Find the latest results file
            results_files = list(Path("experiments/results/strong_models").glob("strong_models_results_*.json"))
            if results_files:
                latest_file = max(results_files, key=lambda p: p.stat().st_mtime)
                
                # Copy to pair directory with descriptive name
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pair_results_file = pair_dir / f"{pair_name}_results_{timestamp}.json"
                
                # Load and save results
                with open(latest_file, 'r') as f:
                    experiment_results = json.load(f)
                
                # Add metadata
                experiment_results["pair_metadata"] = {
                    "model1": model1,
                    "model2": model2,
                    "pair_name": pair_name,
                    "duration_seconds": duration,
                    "start_time": start_datetime.isoformat(),
                    "end_time": datetime.now().isoformat()
                }
                
                with open(pair_results_file, 'w') as f:
                    json.dump(experiment_results, f, indent=2)
                
                print(f"Results saved to: {pair_results_file}")
                
                return {
                    "success": True,
                    "pair": pair_name,
                    "duration": duration,
                    "results_file": str(pair_results_file),
                    "summary": experiment_results.get("summary", {})
                }
            else:
                print("Warning: No results file found")
                return {
                    "success": False,
                    "pair": pair_name,
                    "error": "No results file generated"
                }
        else:
            print(f"✗ Experiment failed with return code {result.returncode}")
            print(f"Error output: {result.stderr[:500]}")  # Print first 500 chars of error
            
            # Save error log
            error_file = pair_dir / f"{pair_name}_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(error_file, 'w') as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Return code: {result.returncode}\n")
                f.write(f"STDOUT:\n{result.stdout}\n")
                f.write(f"STDERR:\n{result.stderr}\n")
            
            return {
                "success": False,
                "pair": pair_name,
                "error": f"Return code {result.returncode}",
                "error_file": str(error_file)
            }
            
    except Exception as e:
        print(f"✗ Exception during experiment: {e}")
        return {
            "success": False,
            "pair": pair_name,
            "error": str(e)
        }


def main():
    """Main entry point for running all model pair experiments."""
    
    print("\n" + "="*60)
    print("ALL MODEL PAIRS NEGOTIATION EXPERIMENT")
    print("="*60)
    
    # Check for API key
    import os
    if not os.getenv("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY environment variable is required")
        print("Please set it with: export OPENROUTER_API_KEY='your-key-here'")
        return
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"experiments/results/all_pairs_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all model pairs
    pairs = get_model_pairs()
    total_pairs = len(pairs)
    
    print(f"\nModels: {', '.join(MODELS)}")
    print(f"Total pairs to test: {total_pairs}")
    print(f"Runs per pair: {NUM_RUNS}")
    print(f"Max rounds per run: {MAX_ROUNDS}")
    print(f"Total negotiations: {total_pairs * NUM_RUNS}")
    print(f"\nOutput directory: {output_dir}")
    
    # Estimate time (assuming ~2 minutes per run, 10 runs per pair)
    estimated_minutes = total_pairs * NUM_RUNS * 2
    print(f"Estimated total time: {estimated_minutes} minutes ({estimated_minutes/60:.1f} hours)")
    
    # Confirm before starting
    response = input("\nProceed with experiments? (y/n): ")
    if response.lower() != 'y':
        print("Experiments cancelled.")
        return
    
    # Track all results
    all_results = {
        "experiment": "All Model Pairs Competition",
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "models": MODELS,
            "total_pairs": total_pairs,
            "runs_per_pair": NUM_RUNS,
            "max_rounds": MAX_ROUNDS,
            "num_items": NUM_ITEMS,
            "competition_level": COMPETITION_LEVEL
        },
        "pairs": []
    }
    
    # Run experiments for each pair
    start_time = time.time()
    successful_pairs = 0
    failed_pairs = 0
    
    for i, (model1, model2) in enumerate(pairs, 1):
        print(f"\n{'='*60}")
        print(f"PAIR {i}/{total_pairs}")
        print(f"{'='*60}")
        
        # Run the experiment
        result = run_experiment(model1, model2, output_dir)
        all_results["pairs"].append(result)
        
        if result["success"]:
            successful_pairs += 1
        else:
            failed_pairs += 1
        
        # Save intermediate results after each pair
        intermediate_file = output_dir / "all_pairs_progress.json"
        with open(intermediate_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Show progress
        elapsed_time = time.time() - start_time
        avg_time_per_pair = elapsed_time / i
        remaining_pairs = total_pairs - i
        estimated_remaining = remaining_pairs * avg_time_per_pair
        
        print(f"\nProgress: {i}/{total_pairs} pairs completed")
        print(f"Successful: {successful_pairs}, Failed: {failed_pairs}")
        print(f"Elapsed time: {elapsed_time/60:.1f} minutes")
        print(f"Estimated remaining: {estimated_remaining/60:.1f} minutes")
    
    # Calculate final statistics
    total_time = time.time() - start_time
    all_results["summary"] = {
        "total_pairs": total_pairs,
        "successful_pairs": successful_pairs,
        "failed_pairs": failed_pairs,
        "total_duration_seconds": total_time,
        "total_duration_hours": total_time / 3600,
        "average_time_per_pair": total_time / total_pairs if total_pairs > 0 else 0
    }
    
    # Aggregate statistics across all successful pairs
    if successful_pairs > 0:
        consensus_rates = []
        avg_rounds_list = []
        avg_spreads = []
        
        for pair_result in all_results["pairs"]:
            if pair_result["success"] and "summary" in pair_result:
                summary = pair_result["summary"]
                if "consensus_rate" in summary:
                    consensus_rates.append(summary["consensus_rate"])
                if "avg_rounds" in summary:
                    avg_rounds_list.append(summary["avg_rounds"])
                if "avg_utility_spread" in summary:
                    avg_spreads.append(summary["avg_utility_spread"])
        
        if consensus_rates:
            all_results["summary"]["overall_consensus_rate"] = sum(consensus_rates) / len(consensus_rates)
        if avg_rounds_list:
            all_results["summary"]["overall_avg_rounds"] = sum(avg_rounds_list) / len(avg_rounds_list)
        if avg_spreads:
            all_results["summary"]["overall_avg_utility_spread"] = sum(avg_spreads) / len(avg_spreads)
    
    # Save final results
    final_file = output_dir / "all_pairs_final_results.json"
    with open(final_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*60)
    print(f"Total pairs tested: {total_pairs}")
    print(f"Successful: {successful_pairs}")
    print(f"Failed: {failed_pairs}")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Results saved to: {output_dir}")
    print(f"Final results: {final_file}")
    
    # Print pair-by-pair summary
    print("\n" + "="*60)
    print("PAIR-BY-PAIR SUMMARY")
    print("="*60)
    
    for pair_result in all_results["pairs"]:
        status = "✓" if pair_result["success"] else "✗"
        pair_name = pair_result["pair"]
        
        if pair_result["success"]:
            summary = pair_result.get("summary", {})
            consensus_rate = summary.get("consensus_rate", 0)
            avg_rounds = summary.get("avg_rounds", 0)
            print(f"{status} {pair_name}: Consensus {consensus_rate:.1%}, Avg rounds {avg_rounds:.1f}")
        else:
            error = pair_result.get("error", "Unknown error")
            print(f"{status} {pair_name}: FAILED - {error}")


if __name__ == "__main__":
    main()
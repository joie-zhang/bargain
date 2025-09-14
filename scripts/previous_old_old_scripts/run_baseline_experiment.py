#!/usr/bin/env python3
"""
Simple script to run the O3 vs Claude Haiku baseline experiment.

This script provides an easy way to execute the first experiment implementation
from Phase 1 of the roadmap.

Usage:
    python run_baseline_experiment.py [--runs N] [--simulated]
    
Examples:
    python run_baseline_experiment.py                    # Single experiment
    python run_baseline_experiment.py --runs 10         # Batch of 10 runs
    python run_baseline_experiment.py --simulated       # Use simulated agents (no API keys needed)
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from experiments.o3_vs_haiku_baseline import O3VsHaikuExperiment


async def main():
    parser = argparse.ArgumentParser(
        description="Run O3 vs Claude Haiku baseline experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_baseline_experiment.py                    # Single experiment
  python run_baseline_experiment.py --runs 10         # Batch of 10 runs
  python run_baseline_experiment.py --simulated       # Use simulated agents
  python run_baseline_experiment.py --runs 5 --simulated  # 5 simulated runs
        """
    )
    
    parser.add_argument(
        "--runs", "-r",
        type=int,
        default=1,
        help="Number of experiments to run (default: 1)"
    )
    
    parser.add_argument(
        "--simulated", "-s",
        action="store_true",
        help="Force use of simulated agents (no API keys required)"
    )
    
    parser.add_argument(
        "--competition-level", "-c",
        type=float,
        default=0.95,
        help="Competition level (0.0 = cooperative, 1.0 = highly competitive, default: 0.95)"
    )
    
    parser.add_argument(
        "--rounds", "-t",
        type=int,
        default=6,
        help="Maximum number of negotiation rounds (default: 6)"
    )
    
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments/results",
        help="Directory to save results (default: experiments/results)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("O3 vs Claude Haiku Baseline Experiment")
    print("=" * 60)
    
    print(f"\nConfiguration:")
    print(f"  Runs: {args.runs}")
    print(f"  Competition Level: {args.competition_level}")
    print(f"  Max Rounds: {args.rounds}")
    print(f"  Results Directory: {args.results_dir}")
    print(f"  Simulated Mode: {'Yes' if args.simulated else 'Auto-detect'}")
    
    # Initialize experiment
    experiment = O3VsHaikuExperiment(
        results_dir=args.results_dir,
        log_level="DEBUG" if args.verbose else "INFO"
    )
    
    # Configuration overrides
    config_overrides = {
        "competition_level": args.competition_level,
        "t_rounds": args.rounds
    }
    
    # Force simulated mode if requested
    if args.simulated:
        print("\n🤖 Using simulated agents (no API keys required)")
    else:
        print("\n🔑 Attempting to use real LLM APIs (O3 and Claude Haiku)")
        print("   Set OPENAI_API_KEY and ANTHROPIC_API_KEY environment variables")
        print("   Will fall back to simulated agents if API keys not available")
    
    try:
        if args.runs == 1:
            # Single experiment
            print("\n🚀 Running single experiment...")
            result = await experiment.run_single_experiment(
                experiment_config=config_overrides
            )
            
            print(f"\n📊 Results:")
            print(f"  Experiment ID: {result.experiment_id}")
            print(f"  Consensus: {'✓' if result.consensus_reached else '✗'}")
            print(f"  Rounds: {result.final_round}")
            print(f"  Winner: {result.winner_agent_id or 'None'}")
            print(f"  O3 Won: {'✓' if result.o3_won else '✗'}")
            print(f"  Exploitation: {'✓' if result.exploitation_detected else '✗'}")
            
            if result.final_utilities:
                print(f"  Final Utilities:")
                for agent_id, utility in result.final_utilities.items():
                    print(f"    {agent_id}: {utility:.2f}")
            
            print(f"\n🧠 Strategic Behaviors:")
            behaviors = result.strategic_behaviors
            print(f"  Manipulation: {'✓' if behaviors['manipulation_detected'] else '✗'}")
            print(f"  Anger Expressions: {behaviors['anger_expressions']}")
            print(f"  Gaslighting Attempts: {behaviors['gaslighting_attempts']}")
            
        else:
            # Batch experiment
            print(f"\n🚀 Running batch of {args.runs} experiments...")
            batch_results = await experiment.run_batch_experiments(
                num_runs=args.runs,
                save_results=True
            )
            
            print(f"\n📈 Batch Results:")
            print(f"  Batch ID: {batch_results.batch_id}")
            print(f"  Successful Runs: {batch_results.num_runs}/{args.runs}")
            print(f"  O3 Win Rate: {batch_results.o3_win_rate:.1%}")
            print(f"  Haiku Win Rate: {batch_results.haiku_win_rate:.1%}")
            print(f"  Consensus Rate: {batch_results.consensus_rate:.1%}")
            print(f"  Average Rounds: {batch_results.average_rounds:.1f}")
            print(f"  Exploitation Rate: {batch_results.exploitation_rate:.1%}")
            
            print(f"\n🧠 Strategic Analysis:")
            strategic = batch_results.strategic_behaviors_summary
            print(f"  Manipulation Rate: {strategic['manipulation_rate']:.1%}")
            print(f"  Avg Anger: {strategic['average_anger_expressions']:.1f}")
            print(f"  Avg Gaslighting: {strategic['average_gaslighting_attempts']:.1f}")
            print(f"  Cooperation Breakdown: {strategic['cooperation_breakdown_rate']:.1%}")
            
            print(f"\n💾 Results saved to:")
            print(f"  {args.results_dir}/{batch_results.batch_id}_summary.json")
            print(f"  {args.results_dir}/{batch_results.batch_id}_detailed.json")
            
            # Analysis
            print(f"\n🎯 Analysis:")
            if batch_results.o3_win_rate > 0.6:
                print(f"  ✓ O3 shows strong advantage (win rate: {batch_results.o3_win_rate:.1%})")
            elif batch_results.o3_win_rate < 0.4:
                print(f"  ⚠ Unexpected: Haiku performing better (O3 win rate: {batch_results.o3_win_rate:.1%})")
            else:
                print(f"  • Balanced performance between models")
            
            if batch_results.exploitation_rate > 0.3:
                print(f"  ⚠ High exploitation detected ({batch_results.exploitation_rate:.1%})")
            
            if batch_results.consensus_rate < 0.5:
                print(f"  ⚠ Low consensus rate ({batch_results.consensus_rate:.1%}) - negotiations often fail")
        
        print(f"\n✅ Experiment completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n⚠ Experiment interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
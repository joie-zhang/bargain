#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strong Models Competition Experiment - Refactored Version

This script runs negotiations between state-of-the-art large language models
using a modular architecture for better maintainability and debugging.
"""

import asyncio
import os
import sys
import locale

# Set UTF-8 encoding
if sys.platform != 'win32':
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except:
        try:
            locale.setlocale(locale.LC_ALL, 'C.UTF-8')
        except:
            pass

# Ensure UTF-8 encoding for stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from strong_models_experiment import StrongModelsExperiment, STRONG_MODELS_CONFIG


async def main():
    """Main entry point."""
    import argparse
    import logging
    
    parser = argparse.ArgumentParser(
        description="Run negotiations between strong language models (Refactored Version)"
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(STRONG_MODELS_CONFIG.keys()),
        default=["claude-3-5-sonnet", "gpt-4o"],
        help="Models to include in negotiation"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run batch experiments"
    )
    
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of negotiation games to run (used with --batch)"
    )
    
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=10,
        help="Maximum rounds per negotiation"
    )
    
    parser.add_argument(
        "--num-items",
        type=int,
        default=5,
        help="Number of items to negotiate"
    )
    
    parser.add_argument(
        "--competition-level",
        type=float,
        default=0.95,
        help="Competition level (0-1)"
    )
    
    parser.add_argument(
        "--random-seed",
        type=int,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--job-id",
        type=int,
        default=None,
        help="Job ID from batch scheduler (for tracking config number)"
    )
    
    # Token control arguments for different phases
    parser.add_argument(
        "--max-tokens-discussion",
        type=int,
        default=None,
        help="Maximum tokens for discussion phase responses (default: unlimited)"
    )
    
    parser.add_argument(
        "--max-tokens-proposal",
        type=int,
        default=None,
        help="Maximum tokens for proposal phase responses (default: unlimited)"
    )
    
    parser.add_argument(
        "--max-tokens-voting",
        type=int,
        default=None,
        help="Maximum tokens for voting phase responses (default: unlimited)"
    )
    
    parser.add_argument(
        "--max-tokens-reflection",
        type=int,
        default=None,
        help="Maximum tokens for reflection phase responses (default: unlimited)"
    )
    
    parser.add_argument(
        "--max-tokens-thinking",
        type=int,
        default=None,
        help="Maximum tokens for private thinking phase responses (default: unlimited)"
    )
    
    parser.add_argument(
        "--max-tokens-default",
        type=int,
        default=None,
        help="Default maximum tokens for all other phases (default: unlimited)"
    )
    
    args = parser.parse_args()
    
    # Check for at least one API key
    has_openrouter = bool(os.getenv("OPENROUTER_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    
    if not (has_openrouter or has_anthropic or has_openai):
        print("ERROR: At least one API key is required")
        print("Please set one or more of:")
        print("  export OPENROUTER_API_KEY='your-key-here'")
        print("  export ANTHROPIC_API_KEY='your-key-here'")
        print("  export OPENAI_API_KEY='your-key-here'")
        return 1
    
    print("API Keys detected:")
    if has_anthropic:
        print("  ✓ Anthropic API (Claude models)")
    if has_openai:
        print("  ✓ OpenAI API (GPT-4o, O3)")
    if has_openrouter:
        print("  ✓ OpenRouter API (Gemini, Llama, etc.)")
    
    print("=" * 60)
    print("STRONG MODELS NEGOTIATION EXPERIMENT (REFACTORED)")
    print("=" * 60)
    print(f"Models: {', '.join(args.models)}")
    print(f"Max Rounds: {args.max_rounds}")
    print(f"Items: {args.num_items}")
    print(f"Competition Level: {args.competition_level}")
    if args.random_seed:
        print(f"Random Seed: {args.random_seed}")
    
    # Only show token limits if any are specified
    token_limits = []
    if args.max_tokens_discussion is not None:
        token_limits.append(f"Discussion={args.max_tokens_discussion}")
    if args.max_tokens_proposal is not None:
        token_limits.append(f"Proposal={args.max_tokens_proposal}")
    if args.max_tokens_voting is not None:
        token_limits.append(f"Voting={args.max_tokens_voting}")
    if args.max_tokens_reflection is not None:
        token_limits.append(f"Reflection={args.max_tokens_reflection}")
    if args.max_tokens_thinking is not None:
        token_limits.append(f"Thinking={args.max_tokens_thinking}")
    if args.max_tokens_default is not None:
        token_limits.append(f"Default={args.max_tokens_default}")
    
    if token_limits:
        print(f"Token Limits: {', '.join(token_limits)}")
    else:
        print("Token Limits: Unlimited (no limits specified)")
    
    print("=" * 60)
    
    # Create experiment configuration
    experiment_config = {
        "m_items": args.num_items,
        "t_rounds": args.max_rounds,
        "competition_level": args.competition_level,
        "random_seed": args.random_seed,
    }
    
    # Only add token limits if they're specified
    if args.max_tokens_discussion is not None:
        experiment_config["max_tokens_discussion"] = args.max_tokens_discussion
    if args.max_tokens_proposal is not None:
        experiment_config["max_tokens_proposal"] = args.max_tokens_proposal
    if args.max_tokens_voting is not None:
        experiment_config["max_tokens_voting"] = args.max_tokens_voting
    if args.max_tokens_reflection is not None:
        experiment_config["max_tokens_reflection"] = args.max_tokens_reflection
    if args.max_tokens_thinking is not None:
        experiment_config["max_tokens_thinking"] = args.max_tokens_thinking
    if args.max_tokens_default is not None:
        experiment_config["max_tokens_default"] = args.max_tokens_default
    
    # Initialize experiment runner
    experiment = StrongModelsExperiment()
    
    try:
        if args.batch:
            print(f"\n--- Batch Experiment ({args.num_runs} runs) ---")
            if args.job_id is not None:
                print(f"Job ID (Config #): {args.job_id}")
            batch_results = await experiment.run_batch_experiments(
                models=args.models,
                num_runs=args.num_runs,
                experiment_config=experiment_config,
                job_id=args.job_id
            )
            
            print(f"\n📈 Batch Results Summary:")
            print(f"  Batch ID: {batch_results.batch_id}")
            print(f"  Successful Runs: {batch_results.num_runs}")
            print(f"  Model Win Rates:")
            for model, rate in batch_results.model_win_rates.items():
                print(f"    {model}: {rate:.1%}")
            print(f"  Consensus Rate: {batch_results.consensus_rate:.1%}")
            print(f"  Average Rounds: {batch_results.average_rounds:.1f}")
            print(f"  Exploitation Rate: {batch_results.exploitation_rate:.1%}")
            
        else:
            print("\n--- Single Experiment Test ---")
            single_result = await experiment.run_single_experiment(
                models=args.models,
                experiment_config=experiment_config
            )
            
            print(f"\n📊 Single Experiment Results:")
            print(f"  Experiment ID: {single_result.experiment_id}")
            print(f"  Consensus Reached: {'✓' if single_result.consensus_reached else '✗'}")
            print(f"  Final Round: {single_result.final_round}")
            print(f"  Winner: {single_result.winner_agent_id}")
            print(f"  Exploitation Detected: {'✓' if single_result.exploitation_detected else '✗'}")
            
            if single_result.final_utilities:
                print(f"  Final Utilities:")
                for agent_id, utility in single_result.final_utilities.items():
                    print(f"    {agent_id}: {utility:.2f}")
        
        print(f"\n✅ Strong models experiment completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        logging.exception("Experiment error")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
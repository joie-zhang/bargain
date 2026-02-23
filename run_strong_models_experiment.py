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
        "--gamma-discount",
        type=float,
        default=0.9,
        help="Discount factor for rewards per round (0-1, default: 0.9)"
    )

    # Game type selection
    parser.add_argument(
        "--game-type",
        type=str,
        choices=["item_allocation", "diplomacy", "co_funding"],
        default="item_allocation",
        help="Type of negotiation game (default: item_allocation)"
    )

    # Diplomacy-specific arguments
    parser.add_argument(
        "--n-issues",
        type=int,
        default=5,
        help="Number of issues to negotiate (diplomacy game only, default: 5)"
    )

    parser.add_argument(
        "--rho",
        type=float,
        default=0.0,
        help="Preference correlation [-1, 1]: 1=cooperative, -1=competitive (diplomacy only, default: 0.0)"
    )

    parser.add_argument(
        "--theta",
        type=float,
        default=0.5,
        help="Interest overlap [0, 1]: 1=same priorities, 0=different priorities (diplomacy only, default: 0.5)"
    )

    # Co-funding specific arguments
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Preference alignment [0,1]: 1=identical, 0=orthogonal (co_funding only, default: 0.5)"
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=0.5,
        help="Budget scarcity (0,1]: ratio of total budget to total cost (co_funding only, default: 0.5)"
    )

    parser.add_argument(
        "--m-projects",
        type=int,
        default=5,
        help="Number of projects to fund (co_funding only, default: 5)"
    )

    parser.add_argument(
        "--c-min",
        type=float,
        default=10.0,
        help="Minimum project cost (co_funding only, default: 10.0)"
    )

    parser.add_argument(
        "--c-max",
        type=float,
        default=50.0,
        help="Maximum project cost (co_funding only, default: 50.0)"
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

    parser.add_argument(
        "--run-number",
        type=int,
        default=None,
        help="Specific run number for output files (overrides automatic numbering in batch mode)"
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
    
    # Phase control arguments (default to enabled)
    parser.add_argument(
        "--disable-discussion",
        action="store_true",
        default=False,
        help="Disable the public discussion phase (default: enabled)"
    )

    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        default=False,
        help="Disable the private thinking phase (default: enabled)"
    )

    parser.add_argument(
        "--disable-reflection",
        action="store_true",
        default=False,
        help="Disable the individual reflection phase (default: enabled)"
    )

    parser.add_argument(
        "--discussion-turns",
        type=int,
        default=3,
        help="Number of times agents go around discussing per round (default: 3)"
    )

    parser.add_argument(
        "--model-order",
        type=str,
        choices=["weak_first", "strong_first", "random"],
        default="weak_first",
        help="Order of model speaking: weak_first, strong_first, or random (default: weak_first)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory for results (overrides default timestamped directory)",
    )

    # Reasoning token budget arguments for test-time compute scaling experiments
    parser.add_argument(
        "--reasoning-token-budget",
        type=int,
        default=None,
        help="Target reasoning tokens to prompt for (added to prompt instructions, NOT API-enforced)"
    )

    parser.add_argument(
        "--reasoning-budget-phases",
        nargs="+",
        choices=["thinking", "reflection", "discussion", "proposal", "voting", "all"],
        default=["thinking", "reflection"],
        help="Phases to apply reasoning token budget instruction to (default: thinking, reflection)"
    )

    parser.add_argument(
        "--max-tokens-per-phase",
        type=int,
        default=10500,
        help="Set max_tokens for EACH individual phase/API call (default: 10500)"
    )

    parser.add_argument(
        "--prompt-only",
        action="store_true",
        help="Use only prompt-based reasoning budget instruction, disable API-level control (no thinking.budget_tokens or reasoning_effort)"
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
    print(f"Game Type: {args.game_type}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Max Rounds: {args.max_rounds}")

    # Show game-specific parameters
    if args.game_type == "item_allocation":
        print(f"Items: {args.num_items}")
        print(f"Competition Level: {args.competition_level}")
    elif args.game_type == "diplomacy":
        print(f"Issues: {args.n_issues}")
        print(f"Rho (preference correlation): {args.rho}")
        print(f"Theta (interest overlap): {args.theta}")
    elif args.game_type == "co_funding":
        print(f"Projects: {args.m_projects}")
        print(f"Alpha (preference alignment): {args.alpha}")
        print(f"Sigma (budget scarcity): {args.sigma}")
        print(f"Cost range: [{args.c_min}, {args.c_max}]")

    print(f"Discount Factor: {args.gamma_discount}")
    if args.random_seed:
        print(f"Random Seed: {args.random_seed}")
    
    # Show enabled/disabled phases
    enabled_phases = []
    disabled_phases = []

    if args.disable_discussion:
        disabled_phases.append("Discussion")
    else:
        enabled_phases.append("Discussion")

    if args.disable_thinking:
        disabled_phases.append("Private Thinking")
    else:
        enabled_phases.append("Private Thinking")

    if args.disable_reflection:
        disabled_phases.append("Individual Reflection")
    else:
        enabled_phases.append("Individual Reflection")
    
    if disabled_phases:
        print(f"Disabled Phases: {', '.join(disabled_phases)}")
    if enabled_phases:
        print(f"Enabled Phases: {', '.join(enabled_phases)}")

    print(f"Discussion Turns: {args.discussion_turns}")
    print(f"Model Order: {args.model_order}")
    
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

    # Show reasoning token budget if specified
    if args.reasoning_token_budget:
        if args.prompt_only:
            print(f"Reasoning Token Budget: {args.reasoning_token_budget} tokens (PROMPT-ONLY, no API enforcement)")
            print(f"  - All models: Prompt instruction only (API control disabled)")
        else:
            print(f"Reasoning Token Budget: {args.reasoning_token_budget} tokens (API-enforced where supported)")
            print(f"  - Anthropic: thinking.budget_tokens={max(args.reasoning_token_budget, 1024)}")
            print(f"  - OpenAI O3/GPT-5: reasoning_effort={'low' if args.reasoning_token_budget <= 2000 else 'medium' if args.reasoning_token_budget <= 5000 else 'high'}")
            print(f"  - Others: Prompt instruction only")
        print(f"Reasoning Budget Phases: {', '.join(args.reasoning_budget_phases)}")
        print(f"Max Tokens Per Phase: {args.max_tokens_per_phase}")
    
    print("=" * 60)
    
    # Create experiment configuration
    experiment_config = {
        "game_type": args.game_type,
        "m_items": args.num_items,
        "t_rounds": args.max_rounds,
        "competition_level": args.competition_level,
        "gamma_discount": args.gamma_discount,
        "random_seed": args.random_seed,
        "disable_discussion": args.disable_discussion,
        "disable_thinking": args.disable_thinking,
        "disable_reflection": args.disable_reflection,
        "discussion_turns": args.discussion_turns,
        "model_order": args.model_order,
        # Diplomacy-specific parameters
        "n_issues": args.n_issues,
        "rho": args.rho,
        "theta": args.theta,
        # Co-funding specific parameters
        "m_projects": args.m_projects,
        "alpha": args.alpha,
        "sigma": args.sigma,
        "c_min": args.c_min,
        "c_max": args.c_max,
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

    # Add reasoning token budget configuration
    if args.reasoning_token_budget is not None:
        # Expand "all" to all phases
        phases = args.reasoning_budget_phases
        if "all" in phases:
            phases = ["thinking", "reflection", "discussion", "proposal", "voting"]

        # For prompt-based reasoning instructions
        experiment_config["reasoning_config"] = {
            "budget": args.reasoning_token_budget,
            "phases": phases
        }

        # For API-based reasoning control (passed to agent factory)
        # This enables Anthropic's thinking.budget_tokens and OpenAI's reasoning_effort
        # Skip this when --prompt-only is set to use only prompt-based control
        if not args.prompt_only:
            experiment_config["reasoning_token_budget"] = args.reasoning_token_budget

        # Apply max_tokens_per_phase to all phases if reasoning budget is specified
        experiment_config["max_tokens_per_phase"] = args.max_tokens_per_phase
    
    # Create output directory name if not provided
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Build descriptive directory name
        model1, model2 = args.models[0], args.models[1] if len(args.models) > 1 else args.models[0]

        # Game-specific parameters for directory naming
        if args.game_type == "item_allocation":
            game_str = f"items{args.num_items}_comp{args.competition_level}".replace(".", "_")
        elif args.game_type == "diplomacy":
            game_str = f"diplo_issues{args.n_issues}_rho{args.rho}_theta{args.theta}".replace(".", "_").replace("-", "n")
        else:  # co_funding
            game_str = f"cofund_proj{args.m_projects}_alpha{args.alpha}_sigma{args.sigma}".replace(".", "_")

        if args.job_id is not None:
            config_str = f"config{args.job_id:03d}"
        else:
            config_str = "config_unknown"

        if args.run_number is not None:
            run_str = f"run{args.run_number}"
        else:
            run_str = f"runs{args.num_runs}"

        output_dir = f"experiments/results/{model1}_vs_{model2}_{config_str}_{run_str}_{game_str}"

    # Initialize experiment runner with custom output directory
    experiment = StrongModelsExperiment(output_dir=output_dir)

    try:
        if args.batch:
            print(f"\n--- Batch Experiment ({args.num_runs} runs) ---")
            if args.job_id is not None:
                print(f"Job ID (Config #): {args.job_id}")
            print(f"Output Directory: {output_dir}")
            batch_results = await experiment.run_batch_experiments(
                models=args.models,
                num_runs=args.num_runs,
                experiment_config=experiment_config,
                job_id=args.job_id,
                override_run_number=args.run_number
            )
            
            print(f"\n📈 Batch Results Summary:")
            print(f"  Batch ID: {batch_results.batch_id}")
            print(f"  Successful Runs: {batch_results.num_runs}")
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
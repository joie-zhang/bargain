#!/usr/bin/env python3
"""
CLI Runner for Parameterized Multi-Agent Negotiation Experiments

This script provides a command-line interface for running experiments with the new
parameterized system, replacing the old hardcoded o3_vs_haiku_baseline.py.

Usage:
    python run_parameterized_experiment.py --config experiments/configs/o3_vs_haiku_baseline_parameterized.yaml
    python run_parameterized_experiment.py --config experiments/configs/scaling_laws_study.yaml --batch 5
    python run_parameterized_experiment.py --preset o3_vs_haiku
"""

import asyncio
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Import our parameterized experiment system
from experiments.parameterized_experiment import ParameterizedExperimentRunner
from negotiation.experiment_config import ExperimentConfigManager, ConfigurationError


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('parameterized_experiments.log')
        ]
    )
    return logging.getLogger(__name__)


def validate_config_file(config_path: Path) -> bool:
    """Validate that a configuration file exists and is readable."""
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        return False
    
    if not config_path.suffix.lower() in ['.yaml', '.yml']:
        print(f"Error: Configuration file must be YAML format: {config_path}")
        return False
    
    return True


def list_available_configs() -> None:
    """List all available configuration files."""
    configs_dir = Path("experiments/configs")
    if not configs_dir.exists():
        print("No configurations directory found.")
        return
    
    print("\nAvailable Configuration Files:")
    print("=" * 40)
    
    config_files = list(configs_dir.glob("*.yaml")) + list(configs_dir.glob("*.yml"))
    
    if not config_files:
        print("No configuration files found.")
        return
    
    for config_file in sorted(config_files):
        # Try to extract description from config
        try:
            import yaml
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
                description = config_data.get('description', 'No description available')
                config_name = config_data.get('config_name', config_file.stem)
        except Exception:
            description = "Could not read configuration"
            config_name = config_file.stem
        
        print(f"  {config_file.name}")
        print(f"    Name: {config_name}")
        print(f"    Description: {description}")
        print()


def print_experiment_summary(result, batch_result=None) -> None:
    """Print a summary of experiment results."""
    if batch_result:
        print("\n" + "="*60)
        print(f"BATCH EXPERIMENT COMPLETED: {batch_result.batch_id}")
        print("="*60)
        print(f"Total Runs: {batch_result.num_runs}")
        print(f"Success Rate: {batch_result.success_rate:.1%}")
        print(f"Average Rounds: {batch_result.average_rounds:.1f}")
        print(f"Average Cost: ${batch_result.average_cost_usd:.2f}")
        print(f"Average Runtime: {batch_result.average_runtime_seconds:.1f}s")
        print()
        
        print("Model Performance:")
        for model_id, win_rate in batch_result.model_win_rates.items():
            print(f"  {model_id}: {win_rate:.1%} win rate")
        print()
        
        if batch_result.order_effect_significance > 0.3:
            print(f"⚠️  Significant order effects detected (correlation: {batch_result.order_effect_significance:.3f})")
        else:
            print("✅ No significant order effects detected")
        
    else:
        print("\n" + "="*60)
        print(f"EXPERIMENT COMPLETED: {result.experiment_id}")
        print("="*60)
        print(f"Consensus Reached: {'✅ Yes' if result.consensus_reached else '❌ No'}")
        print(f"Winner: {result.winner_agent_id or 'None'}")
        print(f"Final Round: {result.final_round}")
        print(f"Runtime: {result.runtime_seconds:.1f}s")
        print(f"Estimated Cost: ${result.total_cost_usd:.2f}")
        print()
        
        print("Final Utilities:")
        for agent_id, utility in result.final_utilities.items():
            winner_mark = "👑" if agent_id == result.winner_agent_id else "  "
            print(f"  {winner_mark} {agent_id}: {utility:.2f}")
        print()
        
        # Strategic behavior summary
        behaviors = result.strategic_behaviors
        strategic_detected = any([
            behaviors.get("manipulation_detected", False),
            behaviors.get("gaslighting_detected", False),
            behaviors.get("anger_detected", False)
        ])
        
        if strategic_detected:
            print("Strategic Behaviors Detected:")
            if behaviors.get("manipulation_detected"):
                print(f"  🎭 Manipulation: {len(behaviors.get('manipulation_instances', []))} instances")
            if behaviors.get("gaslighting_detected"):
                print(f"  🌀 Gaslighting: {len(behaviors.get('gaslighting_instances', []))} instances")
            if behaviors.get("anger_detected"):
                print(f"  😠 Anger: {len(behaviors.get('anger_instances', []))} instances")
        else:
            print("✅ No strategic behaviors detected")
        
        print()
        print(f"Configuration: {result.preference_type} preferences, competition level {result.competition_level:.2f}")
        print(f"Actual similarity: {result.actual_cosine_similarity:.3f}")


async def run_single_experiment(config_path: Path, experiment_id: Optional[str] = None) -> None:
    """Run a single experiment with the given configuration."""
    runner = ParameterizedExperimentRunner()
    
    try:
        result = await runner.run_single_experiment(config_path, experiment_id)
        print_experiment_summary(result)
        
    except ConfigurationError as e:
        print(f"Configuration Error: {e}")
        print(f"Config file: {config_path}")
        sys.exit(1)
    except KeyError as e:
        print(f"KeyError during experiment: {e}")
        print(f"This likely indicates a data structure mismatch in the experiment system.")
        print(f"Key '{e}' was expected but not found.")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)
    except ZeroDivisionError as e:
        print(f"Division by zero error during experiment: {e}")
        print(f"This likely indicates that all utilities were zero, causing issues in final calculations.")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"Experiment failed with unexpected error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)


async def run_batch_experiment(config_path: Path, num_runs: int) -> None:
    """Run a batch of experiments with the given configuration."""
    runner = ParameterizedExperimentRunner()
    
    try:
        batch_result = await runner.run_batch_experiments(config_path, num_runs)
        print_experiment_summary(None, batch_result)
        
    except ConfigurationError as e:
        print(f"Configuration Error: {e}")
        print(f"Config file: {config_path}")
        sys.exit(1)
    except KeyError as e:
        print(f"KeyError during batch experiment: {e}")
        print(f"This likely indicates a data structure mismatch in the experiment system.")
        print(f"Key '{e}' was expected but not found.")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)
    except ZeroDivisionError as e:
        print(f"Division by zero error during batch experiment: {e}")
        print(f"This likely indicates that all utilities were zero, causing issues in final calculations.")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"Batch experiment failed with unexpected error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)


async def run_preset_experiment(preset_name: str, num_runs: Optional[int] = None) -> None:
    """Run an experiment using a preset configuration."""
    config_manager = ExperimentConfigManager()
    runner = ParameterizedExperimentRunner(config_manager=config_manager)
    
    try:
        config = config_manager.create_preset_config(preset_name)
        
        if num_runs and num_runs > 1:
            config.execution.batch_size = num_runs
            batch_result = await runner.run_batch_experiments(config)
            print_experiment_summary(None, batch_result)
        else:
            result = await runner.run_single_experiment(config)
            print_experiment_summary(result)
            
    except ConfigurationError as e:
        print(f"Preset Error: {e}")
        print("Available presets: baseline_o3_vs_haiku, cooperative_matrix, scaling_laws_study")
        sys.exit(1)
    except KeyError as e:
        print(f"KeyError during preset experiment: {e}")
        print(f"This likely indicates a data structure mismatch in the experiment system.")
        print(f"Key '{e}' was expected but not found.")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)
    except ZeroDivisionError as e:
        print(f"Division by zero error during preset experiment: {e}")
        print(f"This likely indicates that all utilities were zero, causing issues in final calculations.")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"Preset experiment failed with unexpected error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)


def validate_environment() -> bool:
    """Validate that the environment is set up correctly."""
    required_dirs = [
        "experiments/configs",
        "experiments/results", 
        "experiments/logs",
        "negotiation"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print("Error: Missing required directories:")
        for dir_path in missing_dirs:
            print(f"  - {dir_path}")
        print("\nPlease ensure you're running from the project root directory.")
        return False
    
    return True


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run parameterized multi-agent negotiation experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single experiment with config file
  python run_parameterized_experiment.py --config experiments/configs/o3_vs_haiku_baseline_parameterized.yaml
  
  # Run batch experiment
  python run_parameterized_experiment.py --config experiments/configs/scaling_laws_study.yaml --batch 10
  
  # Run preset experiment
  python run_parameterized_experiment.py --preset baseline_o3_vs_haiku
  
  # List available configurations
  python run_parameterized_experiment.py --list-configs
        """
    )
    
    # Main action arguments (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--config", "-c",
        type=str,
        help="Path to YAML configuration file"
    )
    action_group.add_argument(
        "--preset", "-p", 
        type=str,
        choices=["baseline_o3_vs_haiku", "cooperative_matrix", "scaling_laws_study"],
        help="Use a preset configuration"
    )
    action_group.add_argument(
        "--list-configs", "-l",
        action="store_true",
        help="List all available configuration files"
    )
    
    # Optional arguments
    parser.add_argument(
        "--batch", "-b",
        type=int,
        help="Run batch experiment with specified number of runs"
    )
    parser.add_argument(
        "--experiment-id", "-i",
        type=str,
        help="Custom experiment ID"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Validate configuration without running experiment"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # Validate environment
    if not validate_environment():
        sys.exit(1)
    
    # Handle list configs
    if args.list_configs:
        list_available_configs()
        return
    
    # Handle different execution modes
    if args.config:
        config_path = Path(args.config)
        
        if not validate_config_file(config_path):
            sys.exit(1)
        
        if args.dry_run:
            try:
                config_manager = ExperimentConfigManager()
                config = config_manager.load_config(config_path)
                print(f"✅ Configuration is valid")
                print(f"Environment: {config.environment.n_agents} agents, {config.environment.m_items} items, {config.environment.t_rounds} rounds")
                print(f"Models: {list(config.models.keys())}")
                print(f"Estimated cost: ${config_manager.estimate_cost(config):.2f}")
                return
            except Exception as e:
                print(f"❌ Configuration validation failed: {e}")
                sys.exit(1)
        
        # Run experiment
        if args.batch and args.batch > 1:
            print(f"Starting batch experiment with {args.batch} runs...")
            asyncio.run(run_batch_experiment(config_path, args.batch))
        else:
            print("Starting single experiment...")
            asyncio.run(run_single_experiment(config_path, args.experiment_id))
    
    elif args.preset:
        if args.dry_run:
            try:
                config_manager = ExperimentConfigManager()
                config = config_manager.create_preset_config(args.preset)
                print(f"✅ Preset '{args.preset}' is valid")
                print(f"Environment: {config.environment.n_agents} agents, {config.environment.m_items} items, {config.environment.t_rounds} rounds")
                return
            except Exception as e:
                print(f"❌ Preset validation failed: {e}")
                sys.exit(1)
        
        # Run preset experiment
        if args.batch and args.batch > 1:
            print(f"Starting batch preset experiment '{args.preset}' with {args.batch} runs...")
        else:
            print(f"Starting preset experiment '{args.preset}'...")
        
        asyncio.run(run_preset_experiment(args.preset, args.batch))
    
    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    main()
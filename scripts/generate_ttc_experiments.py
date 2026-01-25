#!/usr/bin/env python3
"""
=============================================================================
Test-Time Compute Scaling Experiment Generator
=============================================================================

Generates SLURM job arrays for test-time compute scaling experiments.
Creates all 720 experiment configurations (6 models x 12 budgets x 5
competition levels x 2 orders).

Usage:
    python scripts/generate_ttc_experiments.py --config configs/test_time_compute_scaling.yaml
    python scripts/generate_ttc_experiments.py --help

What it creates:
    scripts/slurm/ttc_scaling/
    ├── submit_all.sh           # Master script to submit all jobs
    ├── configs/                # Individual config files for each job
    │   ├── config_001.yaml
    │   ├── config_002.yaml
    │   └── ...
    └── job_array.sbatch        # SLURM array job script

Examples:
    # Generate all experiment configurations
    python scripts/generate_ttc_experiments.py --config configs/test_time_compute_scaling.yaml

    # Generate with custom output directory
    python scripts/generate_ttc_experiments.py --config configs/test_time_compute_scaling.yaml \
        --output-dir experiments/ttc_custom

    # Dry run (show configurations without creating files)
    python scripts/generate_ttc_experiments.py --config configs/test_time_compute_scaling.yaml \
        --dry-run

Configuration:
    Edit configs/test_time_compute_scaling.yaml to modify:
    - Reasoning models to test
    - Token budget values
    - Competition levels
    - SLURM settings

Dependencies:
    - PyYAML
    - Python 3.8+

=============================================================================
"""

import argparse
import itertools
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_configurations(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate all experiment configurations from the config file."""
    configurations = []

    # Extract parameters
    reasoning_models = config['reasoning_models']
    baseline = config['baseline_model']
    token_budgets = config['reasoning_token_budgets']
    competition_levels = config['competition_levels']
    model_orders = config['model_orders']
    phases = config['reasoning_budget_phases']
    max_tokens = config['max_tokens_per_phase']
    game_config = config['game_config']

    # Generate all combinations
    config_id = 1
    for model in reasoning_models:
        for budget in token_budgets:
            for comp_level in competition_levels:
                for order in model_orders:
                    exp_config = {
                        'config_id': config_id,
                        'reasoning_model': model,
                        'baseline_model': baseline,
                        'reasoning_token_budget': budget,
                        'competition_level': comp_level,
                        'model_order': order,
                        'reasoning_budget_phases': phases,
                        'max_tokens_per_phase': max_tokens,
                        'game_config': game_config,
                        'experiment_settings': config['experiment_settings'],
                    }
                    configurations.append(exp_config)
                    config_id += 1

    return configurations


def create_experiment_config_file(exp_config: Dict[str, Any], output_dir: Path) -> Path:
    """Create a YAML config file for a single experiment."""
    config_id = exp_config['config_id']
    config_file = output_dir / f"config_{config_id:04d}.yaml"

    with open(config_file, 'w') as f:
        yaml.dump(exp_config, f, default_flow_style=False)

    return config_file


def create_slurm_script(config: Dict[str, Any], output_dir: Path, num_configs: int) -> Path:
    """Create the SLURM array job script."""
    slurm_settings = config['slurm_settings']

    script_content = f"""#!/bin/bash
#SBATCH --job-name=ttc_scaling
#SBATCH --partition={slurm_settings['partition']}
#SBATCH --time={slurm_settings['time']}
#SBATCH --mem={slurm_settings['memory']}
#SBATCH --gpus-per-node={slurm_settings['gpus_per_node']}
#SBATCH --cpus-per-task={slurm_settings['cpus_per_task']}
#SBATCH --array=1-{num_configs}
#SBATCH --output={config['output']['log_dir']}/job_%A_%a.out
#SBATCH --error={config['output']['log_dir']}/job_%A_%a.err

# =============================================================================
# Test-Time Compute Scaling Experiment - SLURM Array Job
# =============================================================================
#
# This script runs a single configuration from the test-time compute scaling
# experiment array. The SLURM_ARRAY_TASK_ID determines which config to run.
#
# =============================================================================

echo "=============================================="
echo "Test-Time Compute Scaling Experiment"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=============================================="

# Load environment
source ~/.bashrc
module load python/3.9
module load cuda/11.8

# Activate virtual environment
source /scratch/gpfs/DANQIC/jz4391/bargain/.venv/bin/activate

# Navigate to project directory
cd /scratch/gpfs/DANQIC/jz4391/bargain

# Get config file for this array task
CONFIG_ID=$(printf "%04d" $SLURM_ARRAY_TASK_ID)
CONFIG_FILE="scripts/slurm/ttc_scaling/configs/config_${{CONFIG_ID}}.yaml"

echo "Running configuration: $CONFIG_FILE"

# Parse config and run experiment
python -c "
import yaml
import subprocess
import sys

with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

# Build command
reasoning_model = config['reasoning_model']['model_id']
baseline_model = config['baseline_model']['model_id']
budget = config['reasoning_token_budget']
comp_level = config['competition_level']
order = config['model_order']
phases = ' '.join(config['reasoning_budget_phases'])
max_tokens = config['max_tokens_per_phase']
game_type = config['game_config']['game_type']
num_items = config['game_config']['num_items']
max_rounds = config['game_config']['max_rounds']
gamma = config['game_config']['gamma_discount']
seed = config['experiment_settings']['random_seed_base'] + config['config_id']

cmd = [
    'python', 'run_strong_models_experiment.py',
    '--models', reasoning_model, baseline_model,
    '--batch',
    '--num-runs', '1',
    '--game-type', game_type,
    '--num-items', str(num_items),
    '--max-rounds', str(max_rounds),
    '--competition-level', str(comp_level),
    '--gamma-discount', str(gamma),
    '--model-order', order,
    '--random-seed', str(seed),
    '--job-id', str(config['config_id']),
    '--reasoning-token-budget', str(budget),
    '--reasoning-budget-phases'] + config['reasoning_budget_phases'] + [
    '--max-tokens-per-phase', str(max_tokens),
    '--output-dir', f\"experiments/results/ttc_scaling/config_{{config['config_id']:04d}}\"
]

print('Running command:')
print(' '.join(cmd))
sys.exit(subprocess.call(cmd))
"

echo "=============================================="
echo "End Time: $(date)"
echo "=============================================="
"""

    script_file = output_dir / "job_array.sbatch"
    with open(script_file, 'w') as f:
        f.write(script_content)

    # Make executable
    os.chmod(script_file, 0o755)

    return script_file


def create_submit_script(output_dir: Path, slurm_script: Path) -> Path:
    """Create a master submission script."""
    script_content = f"""#!/bin/bash
# =============================================================================
# Submit Test-Time Compute Scaling Experiments
# =============================================================================
#
# This script submits the SLURM array job for all TTC scaling experiments.
#
# Usage:
#   ./submit_all.sh           # Submit all jobs
#   ./submit_all.sh --test    # Submit first 10 jobs only
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

# Create log directory if it doesn't exist
mkdir -p logs/cluster/ttc_scaling

if [ "$1" == "--test" ]; then
    echo "Submitting test run (first 10 configurations)..."
    sbatch --array=1-10 "$SCRIPT_DIR/job_array.sbatch"
else
    echo "Submitting all configurations..."
    sbatch "$SCRIPT_DIR/job_array.sbatch"
fi

echo ""
echo "Jobs submitted. Monitor with:"
echo "  squeue -u $USER"
echo "  tail -f logs/cluster/ttc_scaling/job_*.out"
"""

    script_file = output_dir / "submit_all.sh"
    with open(script_file, 'w') as f:
        f.write(script_content)

    # Make executable
    os.chmod(script_file, 0o755)

    return script_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate SLURM job arrays for test-time compute scaling experiments"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/test_time_compute_scaling.yaml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='scripts/slurm/ttc_scaling',
        help='Output directory for generated files'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show configurations without creating files'
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Generate all configurations
    print("Generating experiment configurations...")
    configurations = generate_configurations(config)
    print(f"Generated {len(configurations)} configurations")

    if args.dry_run:
        print("\n=== DRY RUN - Configuration Summary ===")
        print(f"Reasoning Models: {len(config['reasoning_models'])}")
        print(f"Token Budgets: {config['reasoning_token_budgets']}")
        print(f"Competition Levels: {config['competition_levels']}")
        print(f"Model Orders: {config['model_orders']}")
        print(f"Total Configurations: {len(configurations)}")

        # Show sample configurations
        print("\n=== Sample Configurations ===")
        for cfg in configurations[:5]:
            print(f"\nConfig {cfg['config_id']}:")
            print(f"  Model: {cfg['reasoning_model']['model_id']}")
            print(f"  Token Budget: {cfg['reasoning_token_budget']}")
            print(f"  Competition: {cfg['competition_level']}")
            print(f"  Order: {cfg['model_order']}")

        print("\n... (showing first 5 of {len(configurations)} configurations)")
        return

    # Create output directories
    output_dir = Path(args.output_dir)
    configs_dir = output_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    # Create log directory
    log_dir = Path(config['output']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create individual config files
    print(f"Creating config files in: {configs_dir}")
    for exp_config in configurations:
        create_experiment_config_file(exp_config, configs_dir)

    # Create SLURM script
    print(f"Creating SLURM script...")
    slurm_script = create_slurm_script(config, output_dir, len(configurations))

    # Create submission script
    print(f"Creating submission script...")
    submit_script = create_submit_script(output_dir, slurm_script)

    print("\n=== Generation Complete ===")
    print(f"Created {len(configurations)} configuration files")
    print(f"\nOutput directory: {output_dir}")
    print(f"SLURM script: {slurm_script}")
    print(f"Submit script: {submit_script}")
    print(f"\nTo submit experiments:")
    print(f"  cd {output_dir}")
    print(f"  ./submit_all.sh          # Submit all jobs")
    print(f"  ./submit_all.sh --test   # Submit first 10 jobs only")


if __name__ == "__main__":
    main()

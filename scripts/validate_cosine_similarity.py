#!/usr/bin/env python3
"""
=============================================================================
Validate Cosine Similarity Generation
=============================================================================

This script validates that the RandomVectorGenerator produces vectors with
cosine similarities matching the target competition levels.

Usage:
    python scripts/validate_cosine_similarity.py
    python scripts/validate_cosine_similarity.py --samples 100 --items 5

What it outputs:
    - For each competition level (0.0 to 1.0 in 0.1 steps):
      - Mean actual cosine similarity across samples
      - Standard deviation
      - Min/Max values
      - Error (difference from target)

Configuration:
    - n_samples: Number of samples per competition level (default: 50)
    - n_items: Number of items per vector (default: 5)
    - max_utility: Sum constraint for vectors (default: 100.0)
    - integer_values: Whether to round to integers (default: True)

Dependencies:
    - numpy, scipy
    - negotiation/random_vector_generator.py

=============================================================================
"""

import numpy as np
import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from negotiation.random_vector_generator import RandomVectorGenerator


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def validate_competition_levels(
    competition_levels: list,
    n_samples: int = 50,
    n_items: int = 5,
    max_utility: float = 100.0,
    integer_values: bool = True,
    seed_offset: int = 0,
    verbose: bool = True
) -> dict:
    """
    Validate that generated vectors have cosine similarities matching targets.

    Args:
        competition_levels: List of target cosine similarity values
        n_samples: Number of samples per competition level
        n_items: Number of items (vector dimension)
        max_utility: Maximum total utility (vectors sum to this)
        integer_values: Whether to round to integer values
        seed_offset: Offset to add to random seeds (use different values for independent runs)
        verbose: Print results as we go

    Returns:
        Dictionary with results for each competition level
    """
    results = {}

    if verbose:
        print("=" * 70)
        print("Validating Cosine Similarity Generation")
        print(f"  n_samples: {n_samples}")
        print(f"  n_items: {n_items}")
        print(f"  max_utility: {max_utility}")
        print(f"  integer_values: {integer_values}")
        print(f"  seed_offset: {seed_offset}")
        print("=" * 70)
        print()
        print(f"{'Target':>8} | {'Mean':>8} | {'Std':>8} | {'Min':>8} | {'Max':>8} | {'Error':>8}")
        print("-" * 70)

    for target in competition_levels:
        actual_cosines = []

        for seed in range(n_samples):
            # Create generator with unique seed (offset allows independent runs)
            generator = RandomVectorGenerator(random_seed=seed_offset + seed * 1000 + int(target * 100))

            # Generate vectors with target cosine similarity
            v1, v2 = generator.generate_vectors_with_cosine_similarity(
                target_cosine=target,
                n_items=n_items,
                max_utility=max_utility,
                integer_values=integer_values
            )

            # Compute actual cosine similarity
            actual = cosine_similarity(v1, v2)
            actual_cosines.append(actual)

        # Compute statistics
        actual_cosines = np.array(actual_cosines)
        mean_cos = np.mean(actual_cosines)
        std_cos = np.std(actual_cosines)
        min_cos = np.min(actual_cosines)
        max_cos = np.max(actual_cosines)
        mean_error = np.abs(mean_cos - target)

        results[target] = {
            'target': target,
            'mean': mean_cos,
            'std': std_cos,
            'min': min_cos,
            'max': max_cos,
            'mean_error': mean_error,
            'all_samples': actual_cosines.tolist()
        }

        if verbose:
            print(f"{target:>8.2f} | {mean_cos:>8.4f} | {std_cos:>8.4f} | "
                  f"{min_cos:>8.4f} | {max_cos:>8.4f} | {mean_error:>8.4f}")

    if verbose:
        print("-" * 70)
        print()

        # Summary statistics
        all_errors = [r['mean_error'] for r in results.values()]
        print(f"Overall Mean Error: {np.mean(all_errors):.4f}")
        print(f"Max Error: {np.max(all_errors):.4f}")
        print(f"Levels with error > 0.05: {sum(1 for e in all_errors if e > 0.05)}/{len(all_errors)}")

    return results


def plot_validation_results(results: dict, save_path: str = None):
    """
    Plot validation results showing target vs actual cosine similarity.

    Args:
        results: Dictionary from validate_competition_levels
        save_path: Optional path to save the figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    targets = sorted(results.keys())
    means = [results[t]['mean'] for t in targets]
    stds = [results[t]['std'] for t in targets]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Target vs Actual with error bars
    ax1.errorbar(targets, means, yerr=stds, fmt='o-', capsize=5,
                 label='Actual (mean ± std)', markersize=8)
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect match', alpha=0.5)
    ax1.set_xlabel('Target Cosine Similarity')
    ax1.set_ylabel('Actual Cosine Similarity')
    ax1.set_title('Target vs Actual Cosine Similarity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)

    # Plot 2: Error distribution
    errors = [results[t]['mean'] - t for t in targets]
    colors = ['green' if abs(e) < 0.02 else 'orange' if abs(e) < 0.05 else 'red' for e in errors]
    ax2.bar(targets, errors, width=0.08, color=colors, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5)
    ax2.axhline(y=-0.05, color='orange', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Target Cosine Similarity')
    ax2.set_ylabel('Error (Actual - Target)')
    ax2.set_title('Error by Competition Level')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()


def detailed_sample_analysis(target_cosine: float, n_samples: int = 20, n_items: int = 5):
    """
    Detailed analysis of a single competition level showing individual samples.

    Args:
        target_cosine: Target cosine similarity
        n_samples: Number of samples to analyze
        n_items: Number of items per vector
    """
    print(f"\n{'='*70}")
    print(f"Detailed Analysis for target_cosine = {target_cosine}")
    print(f"{'='*70}\n")

    for seed in range(n_samples):
        generator = RandomVectorGenerator(random_seed=seed)

        v1, v2 = generator.generate_vectors_with_cosine_similarity(
            target_cosine=target_cosine,
            n_items=n_items,
            max_utility=100.0,
            integer_values=True
        )

        actual = cosine_similarity(v1, v2)
        error = actual - target_cosine

        v1_int = [int(x) for x in v1]
        v2_int = [int(x) for x in v2]

        print(f"Sample {seed:2d}: v1={v1_int}, v2={v2_int}")
        print(f"           cos={actual:.4f}, error={error:+.4f}, "
              f"sum1={sum(v1_int)}, sum2={sum(v2_int)}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate cosine similarity generation')
    parser.add_argument('--samples', type=int, default=50,
                        help='Number of samples per competition level')
    parser.add_argument('--items', type=int, default=5,
                        help='Number of items per vector')
    parser.add_argument('--max-utility', type=float, default=100.0,
                        help='Maximum utility sum')
    parser.add_argument('--float-values', action='store_true',
                        help='Use float values instead of integers')
    parser.add_argument('--detailed', type=float, default=None,
                        help='Run detailed analysis for a specific competition level')
    parser.add_argument('--plot', action='store_true',
                        help='Generate plot of results')
    parser.add_argument('--save-plot', type=str, default=None,
                        help='Save plot to file')
    parser.add_argument('--seed-offset', type=int, default=0,
                        help='Offset for random seeds (use different values for independent runs, e.g., 10000, 20000)')

    args = parser.parse_args()

    # Standard competition levels from the shell script
    COMPETITION_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    if args.detailed is not None:
        # Run detailed analysis for specific level
        detailed_sample_analysis(args.detailed, n_samples=20, n_items=args.items)
    else:
        # Run validation for all levels
        results = validate_competition_levels(
            competition_levels=COMPETITION_LEVELS,
            n_samples=args.samples,
            n_items=args.items,
            max_utility=args.max_utility,
            integer_values=not args.float_values,
            seed_offset=args.seed_offset
        )

        if args.plot or args.save_plot:
            plot_validation_results(results, save_path=args.save_plot)

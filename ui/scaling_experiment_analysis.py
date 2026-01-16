"""
=============================================================================
Scaling Experiment Analysis Module
=============================================================================

Provides analysis functions for the scaling_experiment series with nested
directory structure:
    scaling_experiment/
    └── {weak_model}_vs_{strong_model}/
        ├── weak_first/
        │   └── comp_{level}/
        │       └── run_{N}/
        │           ├── run_N_all_interactions.json
        │           └── run_N_experiment_results.json or experiment_results.json
        └── strong_first/
            └── comp_{level}/
                └── run_{N}/
                    └── ...

Usage:
    from scaling_experiment_analysis import (
        discover_scaling_experiments,
        load_scaling_experiment_run,
        analyze_scaling_batch,
        ScalingExperimentRun,
        ScalingExperimentConfig,
    )

    # Discover all experiments
    experiments = discover_scaling_experiments(results_dir)

    # Analyze a batch
    batch = analyze_scaling_batch(results_dir)

Dependencies:
    - analysis.py (for ExperimentMetrics, BatchMetrics, etc.)
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import math

# Import from existing analysis module
from analysis import (
    ExperimentMetrics,
    BatchMetrics,
    PhaseTokens,
    ProposalMetrics,
    analyze_token_usage,
    analyze_phase_tokens,
    calculate_nash_welfare,
    extract_proposals_from_interactions,
    estimate_tokens,
)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ScalingExperimentConfig:
    """Configuration for a single scaling experiment run."""
    weak_model: str
    strong_model: str
    model_order: str  # "weak_first" or "strong_first"
    competition_level: float
    run_number: int
    run_path: Path

    @property
    def model_pair(self) -> str:
        """Return canonical model pair name."""
        return f"{self.weak_model}_vs_{self.strong_model}"

    @property
    def display_name(self) -> str:
        """Return human-readable display name."""
        order_label = "W→S" if self.model_order == "weak_first" else "S→W"
        return f"{self.weak_model} vs {self.strong_model} [{order_label}] γ={self.competition_level}"


@dataclass
class ScalingExperimentRun:
    """A single run from the scaling experiment with its config and metrics."""
    config: ScalingExperimentConfig
    metrics: Optional[ExperimentMetrics] = None
    interactions: List[Dict] = field(default_factory=list)
    raw_results: Optional[Dict] = None


@dataclass
class ScalingBatchMetrics:
    """Aggregated metrics for scaling experiments, grouped by dimensions."""
    # Grouping key
    weak_model: str
    strong_model: str
    model_order: Optional[str] = None  # None means aggregated across orders
    competition_level: Optional[float] = None  # None means aggregated across levels

    # Core metrics
    num_runs: int = 0
    consensus_rate: float = 0.0
    avg_consensus_round: float = 0.0

    # Per-agent utilities (discounted)
    avg_weak_utility: float = 0.0
    avg_strong_utility: float = 0.0
    std_weak_utility: float = 0.0
    std_strong_utility: float = 0.0

    # Raw utilities
    avg_weak_raw_utility: float = 0.0
    avg_strong_raw_utility: float = 0.0

    # Nash welfare
    avg_nash_welfare: float = 0.0
    std_nash_welfare: float = 0.0

    # Token usage
    avg_total_tokens: int = 0
    avg_weak_tokens: int = 0
    avg_strong_tokens: int = 0

    # Underlying experiments
    experiments: List[ScalingExperimentRun] = field(default_factory=list)

    @property
    def display_name(self) -> str:
        """Return human-readable display name."""
        parts = [f"{self.weak_model} vs {self.strong_model}"]
        if self.model_order:
            order_label = "W→S" if self.model_order == "weak_first" else "S→W"
            parts.append(f"[{order_label}]")
        if self.competition_level is not None:
            parts.append(f"γ={self.competition_level}")
        return " ".join(parts)


# =============================================================================
# DISCOVERY FUNCTIONS
# =============================================================================

def discover_scaling_experiments(results_dir: Path) -> List[ScalingExperimentConfig]:
    """
    Discover all scaling experiment runs in the results directory.

    Args:
        results_dir: Path to experiments/results/

    Returns:
        List of ScalingExperimentConfig for each discovered run
    """
    experiments = []

    # Find all scaling_experiment directories (including timestamped ones)
    scaling_dirs = []
    for item in results_dir.iterdir():
        if item.is_dir() and item.name.startswith("scaling_experiment"):
            # Skip symlinks that point to other scaling_experiment dirs
            if item.is_symlink():
                # Check if it points to another scaling_experiment dir
                target = item.resolve()
                if target.name.startswith("scaling_experiment"):
                    continue  # Skip symlink, we'll process the target directly
            scaling_dirs.append(item)

    for scaling_dir in scaling_dirs:
        # Iterate through model pair directories
        for model_pair_dir in scaling_dir.iterdir():
            if not model_pair_dir.is_dir():
                continue
            if model_pair_dir.name == "configs":
                continue  # Skip configs directory

            # Parse model pair from directory name
            model_pair_match = re.match(r'^(.+)_vs_(.+)$', model_pair_dir.name)
            if not model_pair_match:
                continue

            weak_model = model_pair_match.group(1)
            strong_model = model_pair_match.group(2)

            # Iterate through order directories (weak_first, strong_first)
            for order_dir in model_pair_dir.iterdir():
                if not order_dir.is_dir():
                    continue
                if order_dir.name not in ["weak_first", "strong_first"]:
                    continue

                model_order = order_dir.name

                # Iterate through competition level directories
                for comp_dir in order_dir.iterdir():
                    if not comp_dir.is_dir():
                        continue

                    comp_match = re.match(r'^comp_([0-9.]+)$', comp_dir.name)
                    if not comp_match:
                        continue

                    competition_level = float(comp_match.group(1))

                    # Iterate through run directories
                    for run_dir in comp_dir.iterdir():
                        if not run_dir.is_dir():
                            continue

                        run_match = re.match(r'^run_(\d+)$', run_dir.name)
                        if not run_match:
                            continue

                        run_number = int(run_match.group(1))

                        # Check if this run has result files
                        has_interactions = any(run_dir.glob("*all_interactions*.json"))
                        has_results = any(run_dir.glob("*experiment_results*.json")) or (run_dir / "experiment_results.json").exists()

                        if has_interactions or has_results:
                            experiments.append(ScalingExperimentConfig(
                                weak_model=weak_model,
                                strong_model=strong_model,
                                model_order=model_order,
                                competition_level=competition_level,
                                run_number=run_number,
                                run_path=run_dir,
                            ))

    return experiments


def get_unique_model_pairs(experiments: List[ScalingExperimentConfig]) -> List[Tuple[str, str]]:
    """Get unique (weak_model, strong_model) pairs."""
    pairs = set()
    for exp in experiments:
        pairs.add((exp.weak_model, exp.strong_model))
    return sorted(list(pairs))


def get_unique_values(experiments: List[ScalingExperimentConfig]) -> Dict[str, List]:
    """Get unique values for each dimension."""
    return {
        "weak_models": sorted(set(e.weak_model for e in experiments)),
        "strong_models": sorted(set(e.strong_model for e in experiments)),
        "model_orders": sorted(set(e.model_order for e in experiments)),
        "competition_levels": sorted(set(e.competition_level for e in experiments)),
    }


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_interactions(run_path: Path, run_number: int) -> List[Dict]:
    """Load interactions from a scaling experiment run."""
    # Try different file patterns
    patterns = [
        run_path / f"run_{run_number}_all_interactions.json",
        run_path / "all_interactions.json",
    ]

    for path in patterns:
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

    # Try glob for any all_interactions file
    matches = list(run_path.glob("*all_interactions*.json"))
    if matches:
        try:
            with open(matches[0], 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    return []


def load_experiment_results(run_path: Path, run_number: int) -> Optional[Dict]:
    """Load experiment results from a scaling experiment run."""
    # Try different file patterns
    patterns = [
        run_path / f"run_{run_number}_experiment_results.json",
        run_path / "experiment_results.json",
    ]

    for path in patterns:
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

    # Try glob for any experiment_results file
    matches = list(run_path.glob("*experiment_results*.json"))
    if matches:
        try:
            with open(matches[0], 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    return None


def load_scaling_experiment_run(config: ScalingExperimentConfig) -> ScalingExperimentRun:
    """Load a single scaling experiment run with all its data."""
    interactions = load_interactions(config.run_path, config.run_number)
    results = load_experiment_results(config.run_path, config.run_number)

    metrics = None
    if results:
        metrics = _create_metrics_from_results(config, results, interactions)

    return ScalingExperimentRun(
        config=config,
        metrics=metrics,
        interactions=interactions,
        raw_results=results,
    )


def _create_metrics_from_results(
    config: ScalingExperimentConfig,
    results: Dict,
    interactions: List[Dict]
) -> ExperimentMetrics:
    """Create ExperimentMetrics from loaded results."""
    # Token analysis
    token_analysis = analyze_token_usage(interactions)
    phase_tokens, phase_tokens_by_agent = analyze_phase_tokens(interactions)

    # Get utilities
    final_utils = results.get("final_utilities", {})

    # Map Agent_Alpha/Beta to weak/strong based on order
    if config.model_order == "weak_first":
        # Alpha = weak, Beta = strong
        alpha_util = final_utils.get("Agent_Alpha", 0)
        beta_util = final_utils.get("Agent_Beta", 0)
    else:
        # Alpha = strong, Beta = weak
        alpha_util = final_utils.get("Agent_Alpha", 0)
        beta_util = final_utils.get("Agent_Beta", 0)

    # Get preferences, allocation, discount factor
    preferences = results.get("agent_preferences", {})
    allocation = results.get("final_allocation", {})
    discount_factor = results.get("discount_factor", 0.9)

    # Calculate raw utilities
    alpha_raw = beta_raw = 0
    if preferences and allocation:
        if "Agent_Alpha" in preferences and "Agent_Alpha" in allocation:
            alpha_raw = sum(preferences["Agent_Alpha"][i] for i in allocation["Agent_Alpha"]
                          if i < len(preferences["Agent_Alpha"]))
        if "Agent_Beta" in preferences and "Agent_Beta" in allocation:
            beta_raw = sum(preferences["Agent_Beta"][i] for i in allocation["Agent_Beta"]
                         if i < len(preferences["Agent_Beta"]))

    # Calculate Nash welfare
    nash = calculate_nash_welfare(final_utils) if final_utils else 0
    nash_raw = calculate_nash_welfare({"alpha": alpha_raw, "beta": beta_raw})

    # Extract proposals
    proposals = extract_proposals_from_interactions(interactions, preferences, discount_factor)

    return ExperimentMetrics(
        experiment_id=results.get("experiment_id", ""),
        folder_name=str(config.run_path),
        run_number=config.run_number,
        consensus_reached=results.get("consensus_reached", False),
        final_round=results.get("final_round", 10),
        agent_alpha_utility=alpha_util,
        agent_beta_utility=beta_util,
        agent_alpha_raw_utility=alpha_raw,
        agent_beta_raw_utility=beta_raw,
        total_tokens=token_analysis["total_tokens"],
        agent_alpha_tokens=token_analysis["by_agent"].get("Agent_Alpha", 0),
        agent_beta_tokens=token_analysis["by_agent"].get("Agent_Beta", 0),
        tokens_by_phase=dict(token_analysis["by_phase"]),
        phase_tokens=phase_tokens,
        phase_tokens_by_agent=phase_tokens_by_agent,
        avg_response_length=sum(token_analysis["response_lengths"]) / len(token_analysis["response_lengths"]) if token_analysis["response_lengths"] else 0,
        max_response_length=max(token_analysis["response_lengths"]) if token_analysis["response_lengths"] else 0,
        reasoning_trace_lengths=token_analysis["reasoning_traces"],
        nash_welfare=nash,
        nash_welfare_raw=nash_raw,
        agent_preferences=preferences,
        final_allocation=allocation,
        proposals_by_round=proposals,
        discount_factor=discount_factor,
    )


# =============================================================================
# AGGREGATION FUNCTIONS
# =============================================================================

def _safe_mean(lst: List[float]) -> float:
    return sum(lst) / len(lst) if lst else 0


def _safe_std(lst: List[float]) -> float:
    if len(lst) < 2:
        return 0
    mean = _safe_mean(lst)
    variance = sum((x - mean) ** 2 for x in lst) / (len(lst) - 1)
    return math.sqrt(variance)


def aggregate_scaling_runs(
    runs: List[ScalingExperimentRun],
    weak_model: str,
    strong_model: str,
    model_order: Optional[str] = None,
    competition_level: Optional[float] = None,
) -> ScalingBatchMetrics:
    """Aggregate multiple runs into a single batch metrics object."""
    valid_runs = [r for r in runs if r.metrics is not None]

    if not valid_runs:
        return ScalingBatchMetrics(
            weak_model=weak_model,
            strong_model=strong_model,
            model_order=model_order,
            competition_level=competition_level,
            num_runs=0,
        )

    # Calculate aggregated metrics
    consensus_count = sum(1 for r in valid_runs if r.metrics.consensus_reached)
    consensus_rounds = [r.metrics.final_round for r in valid_runs if r.metrics.consensus_reached]

    # For utilities, we need to map Alpha/Beta to weak/strong based on order
    weak_utils = []
    strong_utils = []
    weak_raw_utils = []
    strong_raw_utils = []
    weak_tokens = []
    strong_tokens = []

    for run in valid_runs:
        if run.config.model_order == "weak_first":
            # Alpha = weak, Beta = strong
            weak_utils.append(run.metrics.agent_alpha_utility)
            strong_utils.append(run.metrics.agent_beta_utility)
            weak_raw_utils.append(run.metrics.agent_alpha_raw_utility)
            strong_raw_utils.append(run.metrics.agent_beta_raw_utility)
            weak_tokens.append(run.metrics.agent_alpha_tokens)
            strong_tokens.append(run.metrics.agent_beta_tokens)
        else:
            # Alpha = strong, Beta = weak
            weak_utils.append(run.metrics.agent_beta_utility)
            strong_utils.append(run.metrics.agent_alpha_utility)
            weak_raw_utils.append(run.metrics.agent_beta_raw_utility)
            strong_raw_utils.append(run.metrics.agent_alpha_raw_utility)
            weak_tokens.append(run.metrics.agent_beta_tokens)
            strong_tokens.append(run.metrics.agent_alpha_tokens)

    nash_values = [r.metrics.nash_welfare for r in valid_runs]
    total_tokens = [r.metrics.total_tokens for r in valid_runs]

    return ScalingBatchMetrics(
        weak_model=weak_model,
        strong_model=strong_model,
        model_order=model_order,
        competition_level=competition_level,
        num_runs=len(valid_runs),
        consensus_rate=consensus_count / len(valid_runs) if valid_runs else 0,
        avg_consensus_round=_safe_mean(consensus_rounds),
        avg_weak_utility=_safe_mean(weak_utils),
        avg_strong_utility=_safe_mean(strong_utils),
        std_weak_utility=_safe_std(weak_utils),
        std_strong_utility=_safe_std(strong_utils),
        avg_weak_raw_utility=_safe_mean(weak_raw_utils),
        avg_strong_raw_utility=_safe_mean(strong_raw_utils),
        avg_nash_welfare=_safe_mean(nash_values),
        std_nash_welfare=_safe_std(nash_values),
        avg_total_tokens=int(_safe_mean(total_tokens)),
        avg_weak_tokens=int(_safe_mean(weak_tokens)),
        avg_strong_tokens=int(_safe_mean(strong_tokens)),
        experiments=runs,
    )


def analyze_scaling_experiments(
    results_dir: Path,
    group_by: List[str] = None,
) -> Dict[str, ScalingBatchMetrics]:
    """
    Analyze all scaling experiments with grouping.

    Args:
        results_dir: Path to experiments/results/
        group_by: List of dimensions to group by. Options:
            - "model_pair": Group by (weak_model, strong_model)
            - "model_order": Group by weak_first/strong_first
            - "competition_level": Group by competition level
            - None or empty: Return one batch per unique combination

    Returns:
        Dict mapping group key to aggregated ScalingBatchMetrics
    """
    if group_by is None:
        group_by = ["model_pair", "model_order", "competition_level"]

    # Discover all experiments
    configs = discover_scaling_experiments(results_dir)

    if not configs:
        return {}

    # Load all runs
    runs = [load_scaling_experiment_run(c) for c in configs]

    # Group runs
    grouped = defaultdict(list)
    for run in runs:
        key_parts = []

        if "model_pair" in group_by:
            key_parts.append(run.config.model_pair)

        if "model_order" in group_by:
            key_parts.append(run.config.model_order)

        if "competition_level" in group_by:
            key_parts.append(f"comp_{run.config.competition_level}")

        key = "_".join(key_parts) if key_parts else "all"
        grouped[key].append(run)

    # Aggregate each group
    results = {}
    for key, group_runs in grouped.items():
        if not group_runs:
            continue

        first_config = group_runs[0].config

        model_order = first_config.model_order if "model_order" in group_by else None
        competition_level = first_config.competition_level if "competition_level" in group_by else None

        results[key] = aggregate_scaling_runs(
            group_runs,
            weak_model=first_config.weak_model,
            strong_model=first_config.strong_model,
            model_order=model_order,
            competition_level=competition_level,
        )

    return results


# =============================================================================
# COMPARISON FUNCTIONS
# =============================================================================

def create_model_pair_comparison(results_dir: Path) -> Optional[Any]:
    """
    Create a comparison DataFrame of model pairs.

    Returns pandas DataFrame if available, otherwise list of dicts.
    """
    # Group by model pair only
    batches = analyze_scaling_experiments(results_dir, group_by=["model_pair"])

    if not batches:
        return None

    data = []
    for key, batch in sorted(batches.items()):
        data.append({
            "Model Pair": f"{batch.weak_model} vs {batch.strong_model}",
            "Weak Model": batch.weak_model,
            "Strong Model": batch.strong_model,
            "Runs": batch.num_runs,
            "Consensus Rate": f"{batch.consensus_rate:.0%}",
            "Avg Round": f"{batch.avg_consensus_round:.1f}",
            "Weak Utility": f"{batch.avg_weak_utility:.1f}",
            "Strong Utility": f"{batch.avg_strong_utility:.1f}",
            "Utility Diff": f"{batch.avg_strong_utility - batch.avg_weak_utility:+.1f}",
            "Nash Welfare": f"{batch.avg_nash_welfare:.1f}",
        })

    if PANDAS_AVAILABLE:
        return pd.DataFrame(data)
    return data


def create_order_effect_comparison(results_dir: Path) -> Optional[Any]:
    """
    Create a comparison showing order effects (weak_first vs strong_first).

    Returns pandas DataFrame if available, otherwise list of dicts.
    """
    # Group by model pair and order
    batches = analyze_scaling_experiments(results_dir, group_by=["model_pair", "model_order"])

    if not batches:
        return None

    data = []
    for key, batch in sorted(batches.items()):
        order_label = "Weak First" if batch.model_order == "weak_first" else "Strong First"
        data.append({
            "Model Pair": f"{batch.weak_model} vs {batch.strong_model}",
            "Order": order_label,
            "Runs": batch.num_runs,
            "Consensus Rate": f"{batch.consensus_rate:.0%}",
            "Weak Utility": f"{batch.avg_weak_utility:.1f}",
            "Strong Utility": f"{batch.avg_strong_utility:.1f}",
            "Utility Diff (Strong-Weak)": f"{batch.avg_strong_utility - batch.avg_weak_utility:+.1f}",
            "Nash Welfare": f"{batch.avg_nash_welfare:.1f}",
        })

    if PANDAS_AVAILABLE:
        return pd.DataFrame(data)
    return data


def create_competition_level_comparison(results_dir: Path) -> Optional[Any]:
    """
    Create a comparison showing effects of competition level.

    Returns pandas DataFrame if available, otherwise list of dicts.
    """
    # Group by model pair and competition level
    batches = analyze_scaling_experiments(results_dir, group_by=["model_pair", "competition_level"])

    if not batches:
        return None

    data = []
    for key, batch in sorted(batches.items()):
        data.append({
            "Model Pair": f"{batch.weak_model} vs {batch.strong_model}",
            "Competition γ": f"{batch.competition_level:.2f}" if batch.competition_level is not None else "N/A",
            "Runs": batch.num_runs,
            "Consensus Rate": f"{batch.consensus_rate:.0%}",
            "Avg Round": f"{batch.avg_consensus_round:.1f}",
            "Weak Utility": f"{batch.avg_weak_utility:.1f}",
            "Strong Utility": f"{batch.avg_strong_utility:.1f}",
            "Nash Welfare": f"{batch.avg_nash_welfare:.1f}",
        })

    if PANDAS_AVAILABLE:
        return pd.DataFrame(data)
    return data


def get_scaling_experiment_summary(results_dir: Path) -> Dict[str, Any]:
    """Get a summary of all scaling experiments."""
    configs = discover_scaling_experiments(results_dir)

    if not configs:
        return {
            "total_runs": 0,
            "model_pairs": [],
            "competition_levels": [],
            "orders": [],
        }

    unique = get_unique_values(configs)

    return {
        "total_runs": len(configs),
        "weak_models": unique["weak_models"],
        "strong_models": unique["strong_models"],
        "model_pairs": get_unique_model_pairs(configs),
        "competition_levels": unique["competition_levels"],
        "orders": unique["model_orders"],
    }


# =============================================================================
# MAIN FUNCTION FOR TESTING
# =============================================================================

if __name__ == "__main__":
    results_dir = Path(__file__).parent.parent / "experiments" / "results"

    if results_dir.exists():
        print(f"Analyzing scaling experiments in {results_dir}")

        # Get summary
        summary = get_scaling_experiment_summary(results_dir)
        print(f"\nSummary:")
        print(f"  Total runs: {summary['total_runs']}")
        print(f"  Weak models: {summary.get('weak_models', [])}")
        print(f"  Strong models: {summary.get('strong_models', [])}")
        print(f"  Model pairs: {len(summary.get('model_pairs', []))}")
        print(f"  Competition levels: {summary.get('competition_levels', [])}")

        # Create model pair comparison
        if summary['total_runs'] > 0:
            df = create_model_pair_comparison(results_dir)
            if df is not None and PANDAS_AVAILABLE:
                print("\n\nModel Pair Comparison:")
                print(df.to_string())
    else:
        print(f"Results directory not found: {results_dir}")

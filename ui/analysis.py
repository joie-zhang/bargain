"""
Negotiation Experiment Analysis Module

Provides analysis functions for comparing experiments:
- Token usage estimation
- Payoff analysis
- Consensus round tracking
- Nash welfare calculation
- Qualitative reasoning extraction
"""

import json
import os
import re
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ExperimentMetrics:
    """Metrics for a single experiment run."""
    experiment_id: str
    folder_name: str
    run_number: int

    # Consensus
    consensus_reached: bool
    final_round: int

    # Utilities
    agent_alpha_utility: float
    agent_beta_utility: float
    agent_alpha_raw_utility: float  # Before discount
    agent_beta_raw_utility: float

    # Token usage (estimated)
    total_tokens: int
    agent_alpha_tokens: int
    agent_beta_tokens: int

    # Response lengths
    avg_response_length: float
    max_response_length: int

    # Nash welfare
    nash_welfare: float

    # Fields with defaults must come last
    tokens_by_phase: Dict[str, int] = field(default_factory=dict)
    reasoning_trace_lengths: List[int] = field(default_factory=list)
    agent_preferences: Dict[str, List[float]] = field(default_factory=dict)
    final_allocation: Dict[str, List[int]] = field(default_factory=dict)


@dataclass
class BatchMetrics:
    """Aggregated metrics for a batch of experiments."""
    folder_name: str
    num_runs: int
    model_alpha: str
    model_beta: str

    # Consensus
    consensus_rate: float
    avg_consensus_round: float

    # Utilities
    avg_alpha_utility: float
    avg_beta_utility: float
    std_alpha_utility: float
    std_beta_utility: float

    # Token usage
    avg_total_tokens: int
    avg_alpha_tokens: int
    avg_beta_tokens: int
    avg_tokens_per_round: float

    # Response lengths
    avg_response_length: float
    avg_reasoning_trace_length: float

    # Nash welfare
    avg_nash_welfare: float
    std_nash_welfare: float

    # Fields with defaults must come last
    consensus_rounds: List[int] = field(default_factory=list)
    experiments: List[ExperimentMetrics] = field(default_factory=list)


# =============================================================================
# TOKEN ESTIMATION
# =============================================================================

def estimate_tokens(text: str, method: str = "words") -> int:
    """
    Estimate token count for a text string.

    Methods:
    - "words": Approximate as words * 1.3 (rough estimate)
    - "chars": Approximate as chars / 4
    - "tiktoken": Use OpenAI's tiktoken (if available)
    """
    if not text:
        return 0

    if method == "tiktoken" and TIKTOKEN_AVAILABLE:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass

    if method == "words" or method == "tiktoken":
        # Fallback: words * 1.3 is a reasonable approximation
        words = len(text.split())
        return int(words * 1.3)

    # chars / 4 approximation
    return len(text) // 4


def analyze_token_usage(interactions: List[Dict]) -> Dict[str, Any]:
    """
    Analyze token usage across all interactions.

    Returns:
        Dict with token counts by agent, phase, and round.
    """
    results = {
        "total_tokens": 0,
        "by_agent": defaultdict(int),
        "by_phase": defaultdict(int),
        "by_round": defaultdict(int),
        "response_lengths": [],
        "reasoning_traces": [],
    }

    for entry in interactions:
        response = entry.get("response", "")
        agent = entry.get("agent_id", "unknown")
        phase = entry.get("phase", "unknown")
        round_num = entry.get("round", 0)

        tokens = estimate_tokens(response)

        results["total_tokens"] += tokens
        results["by_agent"][agent] += tokens
        results["by_phase"][phase] += tokens
        results["by_round"][round_num] += tokens
        results["response_lengths"].append(len(response))

        # Track reasoning traces specifically
        if "thinking" in phase.lower() or "private" in phase.lower():
            results["reasoning_traces"].append(len(response))

    return results


# =============================================================================
# NASH WELFARE CALCULATION
# =============================================================================

def calculate_nash_welfare(utilities: Dict[str, float], epsilon: float = 1e-6) -> float:
    """
    Calculate Nash welfare (geometric mean of utilities).

    Nash welfare = (u1 * u2 * ... * un)^(1/n)

    Args:
        utilities: Dict mapping agent names to their utilities
        epsilon: Small value to avoid log(0)

    Returns:
        Nash welfare value
    """
    if not utilities:
        return 0.0

    values = list(utilities.values())
    n = len(values)

    # Handle zero or negative utilities
    adjusted = [max(v, epsilon) for v in values]

    # Use log-sum for numerical stability
    log_sum = sum(math.log(v) for v in adjusted)
    return math.exp(log_sum / n)


def calculate_utilitarian_welfare(utilities: Dict[str, float]) -> float:
    """Calculate utilitarian welfare (sum of utilities)."""
    return sum(utilities.values())


def calculate_egalitarian_welfare(utilities: Dict[str, float]) -> float:
    """Calculate egalitarian welfare (minimum utility)."""
    if not utilities:
        return 0.0
    return min(utilities.values())


def calculate_pareto_efficiency(
    allocation: Dict[str, List[int]],
    preferences: Dict[str, List[float]]
) -> Tuple[float, float]:
    """
    Calculate how close an allocation is to Pareto optimal.

    Returns:
        (achieved_total, max_possible_total)
    """
    achieved = sum(
        sum(preferences[agent][i] for i in items)
        for agent, items in allocation.items()
        if agent in preferences
    )
    max_possible = sum(sum(prefs) for prefs in preferences.values()) / len(preferences)

    return achieved, max_possible


# =============================================================================
# QUALITATIVE ANALYSIS
# =============================================================================

def extract_concession_reasoning(interactions: List[Dict]) -> List[Dict]:
    """
    Extract reasoning for concessions and vote decisions.

    Returns:
        List of dicts with agent, phase, decision, and reasoning.
    """
    concessions = []

    for entry in interactions:
        response = entry.get("response", "")
        agent = entry.get("agent_id", "")
        phase = entry.get("phase", "")
        round_num = entry.get("round", 0)

        # Look for voting decisions
        if "voting" in phase.lower():
            try:
                data = json.loads(response)
                decision = data.get("vote_decision") or data.get("vote", "")
                reasoning = data.get("reasoning", "")

                if decision and reasoning:
                    concessions.append({
                        "agent": agent,
                        "round": round_num,
                        "phase": phase,
                        "decision": decision,
                        "reasoning": reasoning,
                        "type": "vote"
                    })
            except (json.JSONDecodeError, TypeError):
                pass

        # Look for reflection phases
        elif "reflection" in phase.lower():
            concessions.append({
                "agent": agent,
                "round": round_num,
                "phase": phase,
                "decision": "reflection",
                "reasoning": response[:500],
                "type": "reflection"
            })

        # Look for private thinking
        elif "thinking" in phase.lower():
            try:
                data = json.loads(response)
                reasoning = data.get("reasoning", "")
                strategy = data.get("strategy", "")

                if reasoning or strategy:
                    concessions.append({
                        "agent": agent,
                        "round": round_num,
                        "phase": phase,
                        "decision": "strategic_thinking",
                        "reasoning": (reasoning + " " + strategy)[:500],
                        "type": "thinking"
                    })
            except (json.JSONDecodeError, TypeError):
                pass

    return concessions


def extract_key_phrases(text: str) -> List[str]:
    """Extract key negotiation phrases from text."""
    patterns = [
        r"(must.have|non.negotiable|essential|critical|priority)",
        r"(flexible|open to|willing to|can concede)",
        r"(fair|balanced|reasonable|efficient)",
        r"(reject|refuse|unacceptable|won't accept)",
        r"(accept|agree|deal|consensus)",
    ]

    phrases = []
    for pattern in patterns:
        matches = re.findall(pattern, text.lower())
        phrases.extend(matches)

    return phrases


# =============================================================================
# EXPERIMENT LOADING AND ANALYSIS
# =============================================================================

def load_experiment_results(folder_path: Path, run_num: int) -> Optional[Dict]:
    """Load experiment results for a specific run."""
    result_file = folder_path / f"run_{run_num}_experiment_results.json"
    if result_file.exists():
        with open(result_file, 'r') as f:
            return json.load(f)
    return None


def load_all_interactions(folder_path: Path, run_num: int) -> List[Dict]:
    """Load all interactions for a specific run."""
    patterns = [
        folder_path / f"run_{run_num}_all_interactions.json",
        folder_path / "all_interactions.json",
    ]

    for path in patterns:
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)

    return []


def load_summary(folder_path: Path) -> Optional[Dict]:
    """Load batch summary."""
    summary_file = folder_path / "_summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            return json.load(f)
    return None


def analyze_single_experiment(
    folder_path: Path,
    folder_name: str,
    run_num: int
) -> Optional[ExperimentMetrics]:
    """Analyze a single experiment run."""
    results = load_experiment_results(folder_path, run_num)
    interactions = load_all_interactions(folder_path, run_num)

    if not results:
        return None

    # Token analysis
    token_analysis = analyze_token_usage(interactions)

    # Get utilities
    final_utils = results.get("final_utilities", {})
    alpha_util = final_utils.get("Agent_Alpha", 0)
    beta_util = final_utils.get("Agent_Beta", 0)

    # Get preferences and allocation
    preferences = results.get("agent_preferences", {})
    allocation = results.get("final_allocation", {})

    # Calculate raw utilities (before discount)
    alpha_raw = 0
    beta_raw = 0
    if preferences and allocation:
        if "Agent_Alpha" in preferences and "Agent_Alpha" in allocation:
            alpha_raw = sum(preferences["Agent_Alpha"][i] for i in allocation["Agent_Alpha"])
        if "Agent_Beta" in preferences and "Agent_Beta" in allocation:
            beta_raw = sum(preferences["Agent_Beta"][i] for i in allocation["Agent_Beta"])

    # Calculate Nash welfare
    nash = calculate_nash_welfare(final_utils) if final_utils else 0

    return ExperimentMetrics(
        experiment_id=results.get("experiment_id", ""),
        folder_name=folder_name,
        run_number=run_num,
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
        avg_response_length=sum(token_analysis["response_lengths"]) / len(token_analysis["response_lengths"]) if token_analysis["response_lengths"] else 0,
        max_response_length=max(token_analysis["response_lengths"]) if token_analysis["response_lengths"] else 0,
        reasoning_trace_lengths=token_analysis["reasoning_traces"],
        nash_welfare=nash,
        agent_preferences=preferences,
        final_allocation=allocation,
    )


def analyze_batch(folder_path: Path) -> Optional[BatchMetrics]:
    """Analyze a batch of experiments."""
    folder_name = folder_path.name

    # Parse model names from folder name
    parts = folder_name.split("_vs_")
    model_alpha = parts[0] if parts else "unknown"
    model_beta = parts[1].split("_runs")[0] if len(parts) > 1 else "unknown"

    # Find all run files
    run_files = list(folder_path.glob("run_*_experiment_results.json"))
    run_nums = []
    for f in run_files:
        try:
            run_num = int(f.stem.split('_')[1])
            run_nums.append(run_num)
        except (ValueError, IndexError):
            pass

    if not run_nums:
        return None

    # Analyze each run
    experiments = []
    for run_num in sorted(run_nums):
        exp = analyze_single_experiment(folder_path, folder_name, run_num)
        if exp:
            experiments.append(exp)

    if not experiments:
        return None

    # Aggregate metrics
    consensus_count = sum(1 for e in experiments if e.consensus_reached)
    consensus_rounds = [e.final_round for e in experiments if e.consensus_reached]

    alpha_utils = [e.agent_alpha_utility for e in experiments]
    beta_utils = [e.agent_beta_utility for e in experiments]
    nash_values = [e.nash_welfare for e in experiments]

    total_tokens = [e.total_tokens for e in experiments]
    alpha_tokens = [e.agent_alpha_tokens for e in experiments]
    beta_tokens = [e.agent_beta_tokens for e in experiments]

    response_lengths = [e.avg_response_length for e in experiments]
    reasoning_lengths = []
    for e in experiments:
        if e.reasoning_trace_lengths:
            reasoning_lengths.extend(e.reasoning_trace_lengths)

    def safe_mean(lst):
        return sum(lst) / len(lst) if lst else 0

    def safe_std(lst):
        if len(lst) < 2:
            return 0
        mean = safe_mean(lst)
        variance = sum((x - mean) ** 2 for x in lst) / (len(lst) - 1)
        return math.sqrt(variance)

    return BatchMetrics(
        folder_name=folder_name,
        num_runs=len(experiments),
        model_alpha=model_alpha,
        model_beta=model_beta,
        consensus_rate=consensus_count / len(experiments) if experiments else 0,
        avg_consensus_round=safe_mean(consensus_rounds),
        consensus_rounds=consensus_rounds,
        avg_alpha_utility=safe_mean(alpha_utils),
        avg_beta_utility=safe_mean(beta_utils),
        std_alpha_utility=safe_std(alpha_utils),
        std_beta_utility=safe_std(beta_utils),
        avg_total_tokens=int(safe_mean(total_tokens)),
        avg_alpha_tokens=int(safe_mean(alpha_tokens)),
        avg_beta_tokens=int(safe_mean(beta_tokens)),
        avg_tokens_per_round=safe_mean([t / e.final_round for t, e in zip(total_tokens, experiments) if e.final_round > 0]),
        avg_response_length=safe_mean(response_lengths),
        avg_reasoning_trace_length=safe_mean(reasoning_lengths),
        avg_nash_welfare=safe_mean(nash_values),
        std_nash_welfare=safe_std(nash_values),
        experiments=experiments,
    )


# =============================================================================
# COMPARISON FUNCTIONS
# =============================================================================

def compare_batches(batch_metrics: List[BatchMetrics]):
    """
    Compare multiple batches and return a comparison DataFrame.
    Returns pd.DataFrame if pandas available, otherwise returns list of dicts.
    """
    data = []
    for batch in batch_metrics:
        data.append({
            "Experiment": batch.folder_name,
            "Model Alpha": batch.model_alpha,
            "Model Beta": batch.model_beta,
            "Runs": batch.num_runs,
            "Consensus Rate": f"{batch.consensus_rate:.0%}",
            "Avg Consensus Round": f"{batch.avg_consensus_round:.1f}",
            "Avg Alpha Utility": f"{batch.avg_alpha_utility:.1f}",
            "Avg Beta Utility": f"{batch.avg_beta_utility:.1f}",
            "Avg Nash Welfare": f"{batch.avg_nash_welfare:.1f}",
            "Avg Total Tokens": batch.avg_total_tokens,
            "Avg Alpha Tokens": batch.avg_alpha_tokens,
            "Avg Beta Tokens": batch.avg_beta_tokens,
            "Avg Response Length": f"{batch.avg_response_length:.0f}",
            "Avg Reasoning Length": f"{batch.avg_reasoning_trace_length:.0f}",
        })

    if PANDAS_AVAILABLE:
        return pd.DataFrame(data)
    return data


def get_reasoning_effort_comparison(results_dir: Path) -> Dict[str, BatchMetrics]:
    """
    Get comparison data specifically for GPT-5 reasoning effort experiments.
    """
    effort_levels = {}

    for folder in results_dir.iterdir():
        if not folder.is_dir():
            continue

        name = folder.name.lower()

        # Identify effort level from folder name
        if "low-effort" in name and "medium" not in name and "high" not in name:
            # Could be low vs low, or low vs something else
            if "low-effort_vs_gpt-5-low-effort" in name:
                key = "low_vs_low"
            elif "low-effort_vs_gpt-5-medium" in name:
                key = "low_vs_medium"
            elif "low-effort_vs_gpt-5-high" in name:
                key = "low_vs_high"
            else:
                continue

            batch = analyze_batch(folder)
            if batch:
                if key not in effort_levels:
                    effort_levels[key] = batch

    return effort_levels


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def run_full_analysis(results_dir: Path) -> Dict[str, Any]:
    """
    Run a full analysis on all experiments in the results directory.

    Returns:
        Dict containing all analysis results.
    """
    all_batches = []

    for folder in sorted(results_dir.iterdir()):
        if folder.is_dir() and not folder.name.startswith('.'):
            batch = analyze_batch(folder)
            if batch:
                all_batches.append(batch)

    # Create comparison DataFrame
    comparison_df = compare_batches(all_batches)

    # Get reasoning effort comparison
    effort_comparison = get_reasoning_effort_comparison(results_dir)

    return {
        "batches": all_batches,
        "comparison": comparison_df,
        "effort_comparison": effort_comparison,
        "num_experiments": len(all_batches),
    }


if __name__ == "__main__":
    # Test the analysis
    results_dir = Path(__file__).parent.parent / "experiments" / "results"

    if results_dir.exists():
        print(f"Analyzing experiments in {results_dir}")
        analysis = run_full_analysis(results_dir)

        print(f"\nFound {analysis['num_experiments']} experiment batches")
        print("\nComparison Table:")
        print(analysis['comparison'].to_string())

        if analysis['effort_comparison']:
            print("\nReasoning Effort Comparison:")
            for level, batch in analysis['effort_comparison'].items():
                print(f"\n{level}:")
                print(f"  Consensus Rate: {batch.consensus_rate:.0%}")
                print(f"  Avg Beta Utility: {batch.avg_beta_utility:.1f}")
                print(f"  Avg Nash Welfare: {batch.avg_nash_welfare:.1f}")
                print(f"  Avg Total Tokens: {batch.avg_total_tokens}")
    else:
        print(f"Results directory not found: {results_dir}")

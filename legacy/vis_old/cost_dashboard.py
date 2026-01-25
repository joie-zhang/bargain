#!/usr/bin/env python3
"""
=============================================================================
API Cost Dashboard for Negotiation Experiments
=============================================================================

Analyzes experiment logs to calculate and display API spending across
different providers, models, and experiment batches.

Usage:
    python visualization/cost_dashboard.py [OPTIONS]

    # Analyze all experiments in default results directory
    python visualization/cost_dashboard.py

    # Analyze specific directory
    python visualization/cost_dashboard.py --dir experiments/results_batch3

    # Generate JSON report
    python visualization/cost_dashboard.py --json --output cost_report.json

    # Filter by date range
    python visualization/cost_dashboard.py --since 2025-01-01 --until 2025-01-31

What it creates:
    - Terminal dashboard with cost breakdowns
    - Optional JSON report file
    - Aggregated statistics by provider, model, batch

Configuration:
    Edit PRICING_DATA dict below to update model pricing
    Edit MODEL_PROVIDER_MAP to add new model mappings

Dependencies:
    - Python 3.8+
    - numpy (optional, for statistics)
    - tabulate (optional, for prettier tables)

=============================================================================
"""

import json
import re
import sys
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Tuple

# =============================================================================
# PRICING DATA (USD per 1M tokens)
# =============================================================================

PRICING_DATA = {
    # Anthropic Models
    "claude-opus-4.5": {"input": 5.0, "output": 25.0, "provider": "anthropic"},
    "claude-opus-4.1": {"input": 15.0, "output": 75.0, "provider": "anthropic"},
    "claude-opus-4": {"input": 15.0, "output": 75.0, "provider": "anthropic"},
    "claude-sonnet-4.5": {"input": 3.0, "output": 15.0, "provider": "anthropic"},
    "claude-sonnet-4": {"input": 3.0, "output": 15.0, "provider": "anthropic"},
    "claude-3.7-sonnet": {"input": 3.0, "output": 15.0, "provider": "anthropic"},
    "claude-3-7-sonnet": {"input": 3.0, "output": 15.0, "provider": "anthropic"},
    "claude-3.5-sonnet": {"input": 3.0, "output": 15.0, "provider": "anthropic"},
    "claude-3-5-sonnet": {"input": 3.0, "output": 15.0, "provider": "anthropic"},
    "claude-haiku-4.5": {"input": 1.0, "output": 5.0, "provider": "anthropic"},
    "claude-haiku-3.5": {"input": 0.8, "output": 4.0, "provider": "anthropic"},
    "claude-3-5-haiku": {"input": 0.8, "output": 4.0, "provider": "anthropic"},
    "claude-haiku-3": {"input": 0.25, "output": 1.25, "provider": "anthropic"},
    "claude-3-haiku": {"input": 0.25, "output": 1.25, "provider": "anthropic"},

    # OpenAI Models (Standard tier)
    "gpt-5.2": {"input": 1.75, "output": 14.0, "provider": "openai"},
    "gpt-5.1": {"input": 1.25, "output": 10.0, "provider": "openai"},
    "gpt-5": {"input": 1.25, "output": 10.0, "provider": "openai"},
    "gpt-5-mini": {"input": 0.25, "output": 2.0, "provider": "openai"},
    "gpt-5-nano": {"input": 0.05, "output": 0.4, "provider": "openai"},
    "gpt-4.1": {"input": 2.0, "output": 8.0, "provider": "openai"},
    "gpt-4.1-mini": {"input": 0.4, "output": 1.6, "provider": "openai"},
    "gpt-4.1-nano": {"input": 0.1, "output": 0.4, "provider": "openai"},
    "gpt-4o": {"input": 2.5, "output": 10.0, "provider": "openai"},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6, "provider": "openai"},
    "o1": {"input": 15.0, "output": 60.0, "provider": "openai"},
    "o1-mini": {"input": 1.1, "output": 4.4, "provider": "openai"},
    "o3": {"input": 2.0, "output": 8.0, "provider": "openai"},
    "o3-mini": {"input": 1.1, "output": 4.4, "provider": "openai"},
    "o4-mini": {"input": 1.1, "output": 4.4, "provider": "openai"},

    # Google Gemini Models
    "gemini-3-pro": {"input": 2.0, "output": 12.0, "provider": "google"},
    "gemini-3-flash": {"input": 0.5, "output": 3.0, "provider": "google"},
    "gemini-2.5-pro": {"input": 2.0, "output": 12.0, "provider": "google"},
    "gemini-2.0-flash": {"input": 0.5, "output": 3.0, "provider": "google"},
    "gemini-2-flash": {"input": 0.5, "output": 3.0, "provider": "google"},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.0, "provider": "google"},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.3, "provider": "google"},

    # OpenRouter Models (includes 5.5% platform fee in calculations)
    # Base prices shown; fee applied in calculation
    "glm-4.7": {"input": 0.40, "output": 1.50, "provider": "openrouter"},
    "qwen3-max": {"input": 1.20, "output": 6.0, "provider": "openrouter"},
    "deepseek-r1-0528": {"input": 0.40, "output": 1.75, "provider": "openrouter"},
    "deepseek-r1": {"input": 0.70, "output": 2.50, "provider": "openrouter"},
    "deepseek-chat": {"input": 0.14, "output": 0.28, "provider": "openrouter"},
    "deepseek-v3": {"input": 0.14, "output": 0.28, "provider": "openrouter"},
    "grok-4": {"input": 3.0, "output": 15.0, "provider": "openrouter"},
    "nova-micro": {"input": 0.035, "output": 0.14, "provider": "openrouter"},
    "mixtral-8x22b-instruct": {"input": 2.0, "output": 6.0, "provider": "openrouter"},
    "mixtral-8x7b-instruct": {"input": 0.30, "output": 0.40, "provider": "openrouter"},
    "kimi-k2": {"input": 0.60, "output": 2.40, "provider": "openrouter"},
    "kimi-k2-thinking": {"input": 0.60, "output": 2.40, "provider": "openrouter"},

    # Princeton Cluster Models (FREE)
    "qwen2.5-3b-instruct": {"input": 0.0, "output": 0.0, "provider": "cluster"},
    "qwen2.5-7b-instruct": {"input": 0.0, "output": 0.0, "provider": "cluster"},
    "qwen2.5-14b-instruct": {"input": 0.0, "output": 0.0, "provider": "cluster"},
    "qwen2.5-32b-instruct": {"input": 0.0, "output": 0.0, "provider": "cluster"},
    "qwen2.5-72b-instruct": {"input": 0.0, "output": 0.0, "provider": "cluster"},
    "llama-3.1-8b-instruct": {"input": 0.0, "output": 0.0, "provider": "cluster"},
    "llama-3.1-70b-instruct": {"input": 0.0, "output": 0.0, "provider": "cluster"},
    "llama-3.2-1b-instruct": {"input": 0.0, "output": 0.0, "provider": "cluster"},
    "llama-3.2-3b-instruct": {"input": 0.0, "output": 0.0, "provider": "cluster"},
    "mistral-7b-instruct": {"input": 0.0, "output": 0.0, "provider": "cluster"},
}

# OpenRouter platform fee
OPENROUTER_FEE = 0.055  # 5.5%

# Approximate tokens per character (for estimation when actual counts unavailable)
CHARS_PER_TOKEN = 4.0

# =============================================================================
# MODEL NAME NORMALIZATION
# =============================================================================

def normalize_model_name(model_name: str) -> str:
    """Normalize model name for pricing lookup."""
    if not model_name:
        return "unknown"

    # Convert to lowercase and strip
    name = model_name.lower().strip()

    # Remove common prefixes
    prefixes_to_remove = [
        "anthropic/", "openai/", "google/", "deepseek/",
        "x-ai/", "z-ai/", "qwen/", "meta-llama/", "mistralai/",
        "amazon/", "openrouter/"
    ]
    for prefix in prefixes_to_remove:
        if name.startswith(prefix):
            name = name[len(prefix):]

    # Normalize common variations
    replacements = [
        (r"claude[-_]?3[-_.]?7[-_]?sonnet", "claude-3.7-sonnet"),
        (r"claude[-_]?3[-_.]?5[-_]?sonnet", "claude-3.5-sonnet"),
        (r"claude[-_]?3[-_.]?5[-_]?haiku", "claude-3-5-haiku"),
        (r"claude[-_]?3[-_]?haiku", "claude-3-haiku"),
        (r"gpt[-_]?4[-_.]?o[-_]?mini", "gpt-4o-mini"),
        (r"gpt[-_]?4[-_.]?o", "gpt-4o"),
        (r"gpt[-_]?5[-_.]?2", "gpt-5.2"),
        (r"gpt[-_]?5[-_.]?1", "gpt-5.1"),
        (r"gpt[-_]?5[-_]?mini", "gpt-5-mini"),
        (r"gpt[-_]?5[-_]?nano", "gpt-5-nano"),
        (r"gpt[-_]?4[-_.]?1[-_]?mini", "gpt-4.1-mini"),
        (r"gpt[-_]?4[-_.]?1[-_]?nano", "gpt-4.1-nano"),
        (r"gpt[-_]?4[-_.]?1", "gpt-4.1"),
        (r"gemini[-_]?3[-_]?pro", "gemini-3-pro"),
        (r"gemini[-_]?3[-_]?flash", "gemini-3-flash"),
        (r"gemini[-_]?2[-_.]?5[-_]?pro", "gemini-2.5-pro"),
        (r"gemini[-_]?2[-_.]?0[-_]?flash", "gemini-2.0-flash"),
        (r"qwen[-_]?2[-_.]?5[-_]?(\d+)b[-_]?instruct", r"qwen2.5-\1b-instruct"),
        (r"llama[-_]?3[-_.]?1[-_]?(\d+)b[-_]?instruct", r"llama-3.1-\1b-instruct"),
        (r"llama[-_]?3[-_.]?2[-_]?(\d+)b[-_]?instruct", r"llama-3.2-\1b-instruct"),
        (r"deepseek[-_]?r1[-_]?0528", "deepseek-r1-0528"),
        (r"deepseek[-_]?r1", "deepseek-r1"),
        (r"deepseek[-_]?chat", "deepseek-chat"),
        (r"nova[-_]?micro[-_]?v1", "nova-micro"),
        (r"mixtral[-_]?8x22b[-_]?instruct", "mixtral-8x22b-instruct"),
        (r"mixtral[-_]?8x7b[-_]?instruct", "mixtral-8x7b-instruct"),
    ]

    for pattern, replacement in replacements:
        name = re.sub(pattern, replacement, name)

    return name


def get_pricing(model_name: str) -> Dict[str, Any]:
    """Get pricing info for a model."""
    normalized = normalize_model_name(model_name)

    # Direct lookup
    if normalized in PRICING_DATA:
        return PRICING_DATA[normalized]

    # Try partial matches
    for key in PRICING_DATA:
        if key in normalized or normalized in key:
            return PRICING_DATA[key]

    # Default to unknown (assume cluster/free)
    return {"input": 0.0, "output": 0.0, "provider": "unknown"}


def calculate_cost(input_tokens: int, output_tokens: int, model_name: str) -> Dict[str, float]:
    """Calculate cost for given token counts and model."""
    pricing = get_pricing(model_name)

    # Calculate base cost
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    # Apply OpenRouter fee if applicable
    if pricing["provider"] == "openrouter":
        input_cost *= (1 + OPENROUTER_FEE)
        output_cost *= (1 + OPENROUTER_FEE)

    total_cost = input_cost + output_cost

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "provider": pricing["provider"],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def estimate_tokens_from_text(text: str) -> int:
    """Estimate token count from text (when actual count unavailable)."""
    if not text:
        return 0
    return int(len(text) / CHARS_PER_TOKEN)


# =============================================================================
# EXPERIMENT PARSING
# =============================================================================

def parse_interaction_file(filepath: Path) -> Dict[str, Any]:
    """Parse an interaction file and extract token usage per agent.

    Returns a dict with 'by_agent' containing per-agent token breakdowns.
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not parse {filepath}: {e}")
        return {}

    # Handle both formats: list of interactions or dict with 'interactions' key
    if isinstance(data, list):
        interactions = data
        file_agent_id = None  # Will be determined per-interaction
    else:
        interactions = data.get('interactions', [])
        file_agent_id = data.get('agent_id', None)

    # Track tokens per agent
    by_agent: Dict[str, Dict[str, Any]] = {}
    total_interactions = 0
    has_actual_tokens = False

    for interaction in interactions:
        # Determine agent for this interaction
        agent_id = interaction.get('agent_id', file_agent_id or 'unknown')

        if agent_id not in by_agent:
            by_agent[agent_id] = {
                'input_tokens': 0,
                'output_tokens': 0,
                'interactions': 0,
                'has_actual_tokens': False,
            }

        # Check for actual token usage first
        token_usage = interaction.get('token_usage', {})

        if token_usage:
            has_actual_tokens = True
            by_agent[agent_id]['has_actual_tokens'] = True
            by_agent[agent_id]['input_tokens'] += token_usage.get('input_tokens', 0) or 0
            by_agent[agent_id]['output_tokens'] += token_usage.get('output_tokens', 0) or 0
        else:
            # Estimate from text
            prompt = interaction.get('prompt', '')
            response = interaction.get('response', '')
            by_agent[agent_id]['input_tokens'] += estimate_tokens_from_text(prompt)
            by_agent[agent_id]['output_tokens'] += estimate_tokens_from_text(response)

        by_agent[agent_id]['interactions'] += 1
        total_interactions += 1

    return {
        'filepath': str(filepath),
        'by_agent': by_agent,
        'total_interactions': total_interactions,
        'has_actual_tokens': has_actual_tokens,
    }


def parse_experiment_result(filepath: Path) -> Dict[str, Any]:
    """Parse an experiment result file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not parse {filepath}: {e}")
        return {}

    config = data.get('config', {})

    # Extract model names from directory name or config
    models = []
    dir_name = filepath.parent.name

    # Try to extract from directory name (e.g., "Qwen2.5-3B-Instruct_vs_claude-3-7-sonnet_...")
    if '_vs_' in dir_name:
        parts = dir_name.split('_vs_')
        if len(parts) >= 2:
            model1 = parts[0]
            model2 = parts[1].split('_')[0]  # Get until next underscore
            models = [model1, model2]

    # Also check agent_performance for model info
    agent_performance = data.get('agent_performance', {})
    for _, perf in agent_performance.items():
        model = perf.get('model', 'unknown')
        if model and model != 'unknown':
            models.append(model)

    return {
        'filepath': str(filepath),
        'experiment_id': data.get('experiment_id', 'unknown'),
        'timestamp': data.get('timestamp', 0),
        'consensus_reached': data.get('consensus_reached', False),
        'final_round': data.get('final_round', 0),
        'models': models,
        'config': config,
    }


def find_experiment_files(base_dir: Path) -> Tuple[List[Path], List[Path]]:
    """Find all interaction and result files in experiment directories."""
    interaction_files = []
    result_files = []

    # Find all interaction files (various patterns used in codebase)
    for pattern in [
        '**/agent_interactions/*interactions*.json',
        '**/agent_*interactions*.json',
        '**/*all_interactions*.json',
    ]:
        interaction_files.extend(base_dir.glob(pattern))

    # Deduplicate
    interaction_files = list(set(interaction_files))

    # Find all result files
    for pattern in ['**/*experiment_results*.json', '**/_summary.json']:
        result_files.extend(base_dir.glob(pattern))

    result_files = list(set(result_files))

    return interaction_files, result_files


def extract_model_info_from_path(filepath: Path) -> Tuple[str, str, str, str]:
    """Extract model names and experiment info from file path.

    Handles paths like:
    - scaling_experiment_TIMESTAMP/model1_vs_model2/weak_first/comp_X.X/run_N/
    - Qwen2.5-3B-Instruct_vs_claude-3-7-sonnet_config_unknown_runs5_comp0/

    Returns:
        Tuple of (model1, model2, batch_id, agent_model) where agent_model is
        the model for the specific agent file, or 'unknown' if not determinable.
    """
    path_str = str(filepath)
    parts = filepath.parts

    model1 = "unknown"
    model2 = "unknown"
    batch_id = "unknown"
    order = "weak_first"  # default

    # Look for _vs_ pattern in path parts
    for i, part in enumerate(parts):
        if '_vs_' in part:
            vs_parts = part.split('_vs_')
            if len(vs_parts) >= 2:
                model1 = vs_parts[0]
                # Handle model2 which might have additional suffixes
                model2_part = vs_parts[1]
                # Remove common suffixes like _config_unknown, _runs5, etc.
                for suffix in ['_config', '_runs', '_comp']:
                    if suffix in model2_part:
                        model2_part = model2_part.split(suffix)[0]
                model2 = model2_part

            # Try to build a meaningful batch_id from surrounding context
            # Look for scaling_experiment or timestamp patterns
            for j in range(max(0, i-2), i):
                if 'scaling_experiment' in parts[j] or 'strong_models' in parts[j]:
                    batch_id = parts[j]
                    break
            if batch_id == "unknown":
                batch_id = part  # Use the model_vs_model part as batch_id
            break

    # Check for weak_first or strong_first in path
    if 'strong_first' in path_str:
        order = 'strong_first'
    elif 'weak_first' in path_str:
        order = 'weak_first'

    # Determine which model this agent file corresponds to
    agent_model = "unknown"
    filename = filepath.name.lower()

    if 'agent_alpha' in filename:
        # Agent_Alpha is the first in order
        agent_model = model1 if order == 'weak_first' else model2
    elif 'agent_beta' in filename:
        # Agent_Beta is the second in order
        agent_model = model2 if order == 'weak_first' else model1
    elif 'all_interactions' in filename:
        # For all_interactions files, we'll attribute to both models
        agent_model = f"{model1},{model2}"

    return model1, model2, batch_id, agent_model


# =============================================================================
# COST AGGREGATION
# =============================================================================

class CostAggregator:
    """Aggregates costs across experiments."""

    def __init__(self):
        self.by_provider: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"input_tokens": 0, "output_tokens": 0, "cost": 0.0, "experiments": 0}
        )
        self.by_model: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"input_tokens": 0, "output_tokens": 0, "cost": 0.0, "experiments": 0}
        )
        self.by_batch: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"input_tokens": 0, "output_tokens": 0, "cost": 0.0, "experiments": 0}
        )
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.total_experiments = 0
        self.estimated_count = 0
        self.actual_count = 0

    def add_agent_data(self, input_tokens: int, output_tokens: int, model_name: str,
                       batch_id: str, has_actual: bool):
        """Add token data for a specific agent/model."""
        cost_info = calculate_cost(input_tokens, output_tokens, model_name)

        provider = cost_info['provider']
        normalized_model = normalize_model_name(model_name)

        # Update aggregates
        self.by_provider[provider]['input_tokens'] += input_tokens
        self.by_provider[provider]['output_tokens'] += output_tokens
        self.by_provider[provider]['cost'] += cost_info['total_cost']

        self.by_model[normalized_model]['input_tokens'] += input_tokens
        self.by_model[normalized_model]['output_tokens'] += output_tokens
        self.by_model[normalized_model]['cost'] += cost_info['total_cost']

        self.by_batch[batch_id]['input_tokens'] += input_tokens
        self.by_batch[batch_id]['output_tokens'] += output_tokens
        self.by_batch[batch_id]['cost'] += cost_info['total_cost']

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost_info['total_cost']

        if has_actual:
            self.actual_count += 1
        else:
            self.estimated_count += 1

    def add_experiment(self, batch_id: str):
        """Increment experiment count for a batch."""
        self.by_batch[batch_id]['experiments'] += 1
        self.total_experiments += 1


# =============================================================================
# DASHBOARD OUTPUT
# =============================================================================

def format_currency(amount: float) -> str:
    """Format amount as currency."""
    if amount < 0.01:
        return f"${amount:.4f}"
    elif amount < 1.0:
        return f"${amount:.3f}"
    else:
        return f"${amount:.2f}"


def format_tokens(count: int) -> str:
    """Format token count with K/M suffix."""
    if count >= 1_000_000:
        return f"{count/1_000_000:.2f}M"
    elif count >= 1_000:
        return f"{count/1_000:.1f}K"
    else:
        return str(count)


def print_dashboard(aggregator: CostAggregator, detailed: bool = False):
    """Print the cost dashboard to terminal."""

    print("\n" + "=" * 80)
    print("                    API COST DASHBOARD")
    print("=" * 80)

    # Summary
    print(f"\n{'SUMMARY':^80}")
    print("-" * 80)
    print(f"  Total Experiments Analyzed:  {aggregator.total_experiments:,}")
    print(f"  Total Input Tokens:          {format_tokens(aggregator.total_input_tokens)}")
    print(f"  Total Output Tokens:         {format_tokens(aggregator.total_output_tokens)}")
    print(f"  Total Cost:                  {format_currency(aggregator.total_cost)}")
    print(f"  Data Source:                 {aggregator.actual_count} actual, {aggregator.estimated_count} estimated")

    # Cost by Provider
    print(f"\n{'COST BY PROVIDER':^80}")
    print("-" * 80)
    print(f"  {'Provider':<20} {'Input Tokens':>15} {'Output Tokens':>15} {'Cost':>15}")
    print("  " + "-" * 65)

    sorted_providers = sorted(
        aggregator.by_provider.items(),
        key=lambda x: x[1]['cost'],
        reverse=True
    )

    for provider, data in sorted_providers:
        if data['cost'] > 0 or provider != 'cluster':
            print(f"  {provider:<20} {format_tokens(data['input_tokens']):>15} "
                  f"{format_tokens(data['output_tokens']):>15} "
                  f"{format_currency(data['cost']):>15}")

    # Cost by Model (top 15)
    print(f"\n{'COST BY MODEL (Top 15)':^80}")
    print("-" * 80)
    print(f"  {'Model':<35} {'Input':>12} {'Output':>12} {'Cost':>12}")
    print("  " + "-" * 71)

    sorted_models = sorted(
        aggregator.by_model.items(),
        key=lambda x: x[1]['cost'],
        reverse=True
    )[:15]

    for model, data in sorted_models:
        print(f"  {model[:35]:<35} {format_tokens(data['input_tokens']):>12} "
              f"{format_tokens(data['output_tokens']):>12} "
              f"{format_currency(data['cost']):>12}")

    # Cost by Batch (top 10)
    if detailed and aggregator.by_batch:
        print(f"\n{'COST BY BATCH (Top 10)':^80}")
        print("-" * 80)
        print(f"  {'Batch ID':<45} {'Experiments':>10} {'Cost':>15}")
        print("  " + "-" * 70)

        sorted_batches = sorted(
            aggregator.by_batch.items(),
            key=lambda x: x[1]['cost'],
            reverse=True
        )[:10]

        for batch_id, data in sorted_batches:
            batch_display = batch_id[:45] if batch_id else "(unnamed)"
            print(f"  {batch_display:<45} {data['experiments']:>10} "
                  f"{format_currency(data['cost']):>15}")

    # Free models (cluster)
    cluster_models = [m for m, d in aggregator.by_model.items()
                      if get_pricing(m)['provider'] == 'cluster' and d['input_tokens'] > 0]
    if cluster_models:
        print(f"\n{'FREE MODELS (Princeton Cluster)':^80}")
        print("-" * 80)
        for model in sorted(cluster_models):
            data = aggregator.by_model[model]
            print(f"  {model:<35} {format_tokens(data['input_tokens']):>12} input, "
                  f"{format_tokens(data['output_tokens']):>12} output")

    print("\n" + "=" * 80)
    print("Note: Costs marked 'estimated' use character-based token approximation.")
    print("      Cluster models are FREE (running on Princeton's infrastructure).")
    print("=" * 80 + "\n")


def export_json_report(aggregator: CostAggregator, output_path: Path):
    """Export aggregated data as JSON."""
    report = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_experiments": aggregator.total_experiments,
            "total_input_tokens": aggregator.total_input_tokens,
            "total_output_tokens": aggregator.total_output_tokens,
            "total_cost_usd": aggregator.total_cost,
            "actual_token_counts": aggregator.actual_count,
            "estimated_token_counts": aggregator.estimated_count,
        },
        "by_provider": dict(aggregator.by_provider),
        "by_model": dict(aggregator.by_model),
        "by_batch": dict(aggregator.by_batch),
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"JSON report exported to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="API Cost Dashboard for Negotiation Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualization/cost_dashboard.py
  python visualization/cost_dashboard.py --dir experiments/results_batch3
  python visualization/cost_dashboard.py --json --output cost_report.json
  python visualization/cost_dashboard.py --detailed
        """
    )

    parser.add_argument(
        '--dir', '-d',
        type=Path,
        default=Path('experiments/results'),
        help='Directory containing experiment results (default: experiments/results)'
    )

    parser.add_argument(
        '--json', '-j',
        action='store_true',
        help='Export results as JSON'
    )

    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('cost_report.json'),
        help='Output path for JSON report (default: cost_report.json)'
    )

    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed batch-level breakdown'
    )

    parser.add_argument(
        '--since',
        type=str,
        help='Only include experiments after this date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--until',
        type=str,
        help='Only include experiments before this date (YYYY-MM-DD)'
    )

    args = parser.parse_args()

    # Verify directory exists
    if not args.dir.exists():
        print(f"Error: Directory not found: {args.dir}")
        sys.exit(1)

    print(f"Scanning {args.dir} for experiment files...")

    # Find all relevant files
    interaction_files, result_files = find_experiment_files(args.dir)

    print(f"Found {len(interaction_files)} interaction files and {len(result_files)} result files")

    if not interaction_files and not result_files:
        print("No experiment files found!")
        sys.exit(1)

    # Parse and aggregate
    aggregator = CostAggregator()
    processed_batches = set()

    # Parse date filters
    since_date = None
    until_date = None
    if args.since:
        try:
            since_date = datetime.strptime(args.since, '%Y-%m-%d')
        except ValueError:
            print(f"Warning: Invalid --since date format: {args.since}. Use YYYY-MM-DD.")
    if args.until:
        try:
            until_date = datetime.strptime(args.until, '%Y-%m-%d')
        except ValueError:
            print(f"Warning: Invalid --until date format: {args.until}. Use YYYY-MM-DD.")

    # Track processed files to avoid double-counting
    # (some experiments have both all_interactions.json and agent-specific files)
    processed_experiment_agents = set()
    skipped_by_date = 0

    for filepath in interaction_files:
        # Apply date filter based on file modification time
        if since_date or until_date:
            file_mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
            if since_date and file_mtime < since_date:
                skipped_by_date += 1
                continue
            if until_date and file_mtime > until_date:
                skipped_by_date += 1
                continue
        # Extract model and batch info from path
        model1, model2, batch_id, _ = extract_model_info_from_path(filepath)

        # Determine ordering (weak_first or strong_first)
        path_str = str(filepath)
        if 'strong_first' in path_str:
            # strong_first: Agent_Alpha=model2 (strong), Agent_Beta=model1 (weak)
            agent_to_model = {'Agent_Alpha': model2, 'Agent_Beta': model1}
        else:
            # weak_first (default): Agent_Alpha=model1 (weak), Agent_Beta=model2 (strong)
            agent_to_model = {'Agent_Alpha': model1, 'Agent_Beta': model2}

        data = parse_interaction_file(filepath)
        if not data:
            continue

        by_agent = data.get('by_agent', {})

        # Process each agent's data
        for agent_id, agent_data in by_agent.items():
            # Create unique key for deduplication
            # Use run directory + agent_id as the key
            run_dir = filepath.parent
            if 'agent_interactions' in str(run_dir):
                run_dir = run_dir.parent
            dedup_key = (str(run_dir), agent_id)

            # Skip if we've already processed this agent in this run
            if dedup_key in processed_experiment_agents:
                continue
            processed_experiment_agents.add(dedup_key)

            # Map agent_id to model
            model_name = agent_to_model.get(agent_id, 'unknown')

            input_tokens = agent_data.get('input_tokens', 0)
            output_tokens = agent_data.get('output_tokens', 0)
            has_actual = agent_data.get('has_actual_tokens', False)

            aggregator.add_agent_data(
                input_tokens, output_tokens, model_name, batch_id, has_actual
            )

        if batch_id and batch_id not in processed_batches:
            aggregator.add_experiment(batch_id)
            processed_batches.add(batch_id)

    # Print date filter info
    if skipped_by_date > 0:
        print(f"Skipped {skipped_by_date} files outside date range")

    # Print dashboard
    print_dashboard(aggregator, detailed=args.detailed)

    # Export JSON if requested
    if args.json:
        export_json_report(aggregator, args.output)


if __name__ == "__main__":
    main()

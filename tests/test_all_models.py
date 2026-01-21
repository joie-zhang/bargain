#!/usr/bin/env python3
"""
=============================================================================
Test All Models from Chatbot Arena Leaderboard
=============================================================================

Tests all 36 models from docs/api_pricing/guide.md to validate API connectivity
and model availability before running batch experiments.

Usage:
    # Test all models (API models only, skip cluster models)
    python tests/test_all_models.py

    # Test all models including cluster models (requires GPU)
    python tests/test_all_models.py --include-cluster

    # Test only specific providers
    python tests/test_all_models.py --providers openai anthropic

    # Test only specific models
    python tests/test_all_models.py --models claude-opus-4-5 gpt-4o

    # Quick test (one model per provider)
    python tests/test_all_models.py --quick

    # Verbose output
    python tests/test_all_models.py --verbose

What it creates:
    tests/all_models_test_results.json  # Full test results with timestamps

Examples:
    # Basic usage - test all API models
    python tests/test_all_models.py

    # Test everything including local cluster models (on della-gpu)
    python tests/test_all_models.py --include-cluster

    # Quick sanity check before submitting jobs
    python tests/test_all_models.py --quick

    # Debug a specific provider
    python tests/test_all_models.py --providers xai --verbose

Configuration:
    - API keys are read from environment variables
    - Cluster models require running on Princeton cluster with GPU access
    - Models are defined in strong_models_experiment/configs.py

Dependencies:
    - strong_models_experiment package
    - negotiation package
    - API keys: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY,
                XAI_API_KEY, OPENROUTER_API_KEY

=============================================================================
"""

import sys
import os
import argparse
import asyncio
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

# Find project root (parent of tests/)
script_dir = Path(__file__).parent
project_root = script_dir.parent

sys.path.insert(0, str(project_root))
os.chdir(project_root)


# =============================================================================
# Model Definitions from guide.md (36 models ranked by Elo)
# =============================================================================

# Maps human-readable names to config keys in STRONG_MODELS_CONFIG
LEADERBOARD_MODELS = [
    # Rank, Name, Config Key, Elo, Provider
    (1, "Gemini-3-Pro", "gemini-3-pro", 1490, "google"),
    # (2, "Grok-4.1-Thinking", "grok-4-1-thinking", 1477, "xai"),  # Not publicly available
    (3, "Gemini-3-Flash", "gemini-3-flash", 1472, "google"),
    (4, "Claude Opus 4.5 (thinking-32k)", "claude-opus-4-5-thinking-32k", 1470, "anthropic"),
    (5, "Claude Opus 4.5", "claude-opus-4-5", 1467, "anthropic"),
    (6, "Claude Sonnet 4.5", "claude-sonnet-4-5", 1450, "anthropic"),
    # (7, "GPT-4.5-Preview", "gpt-4.5-preview", 1444, "openai"),  # DEPRECATED: Model no longer available
    (8, "GLM-4.7", "glm-4.7", 1441, "openrouter"),
    (9, "GPT-5.2-high", "gpt-5.2-high", 1436, "openai"),
    (10, "Qwen3-Max", "qwen3-max", 1434, "openrouter"),
    (11, "DeepSeek-R1-0528", "deepseek-r1-0528", 1418, "openrouter"),
    (12, "Grok-4", "grok-4", 1409, "xai"),
    (13, "Claude Haiku 4.5", "claude-haiku-4-5", 1403, "anthropic"),
    (14, "DeepSeek-R1", "deepseek-r1", 1397, "openrouter"),
    (15, "Claude Sonnet 4", "claude-sonnet-4", 1390, "anthropic"),
    (16, "Claude 3.5 Sonnet", "claude-3.5-sonnet", 1373, "openrouter"),
    (17, "Gemma-3-27B-it", "gemma-3-27b-it", 1365, "princeton_cluster"),
    (18, "o3-mini-high", "o3-mini-high", 1364, "openai"),
    (19, "DeepSeek-V3", "deepseek-v3", 1358, "openrouter"),
    (20, "GPT-4o", "gpt-4o", 1346, "openai"),
    (21, "QwQ-32B", "QwQ-32B", 1336, "princeton_cluster"),
    (22, "Llama-3.3-70B-Instruct", "llama-3.3-70b-instruct", 1320, "princeton_cluster"),
    (23, "Qwen2.5-72B-Instruct", "Qwen2.5-72B-Instruct", 1303, "princeton_cluster"),
    (24, "Gemma-2-27B-it", "gemma-2-27b-it", 1288, "princeton_cluster"),
    (25, "Llama-3-70B-Instruct", "Meta-Llama-3-70B-Instruct", 1277, "princeton_cluster"),
    (26, "Claude 3 Haiku", "claude-3-haiku", 1262, "anthropic"),
    (27, "Phi-4", "phi-4", 1256, "princeton_cluster"),
    (28, "Amazon-Nova-Micro", "amazon-nova-micro", 1241, "openrouter"),
    (29, "Mixtral-8x22b-Instruct-v0.1", "mixtral-8x22b-instruct-v0.1", 1231, "openrouter"),
    (30, "GPT-3.5-Turbo-0125", "gpt-3.5-turbo-0125", 1225, "openai"),
    (31, "Llama-3.1-8B-Instruct", "llama-3.1-8b-instruct", 1212, "princeton_cluster"),
    (32, "Mixtral-8x7b-Instruct-v0.1", "mixtral-8x7b-instruct-v0.1", 1198, "openrouter"),
    (33, "Llama-3.2-3B-Instruct", "Llama-3.2-3B-Instruct", 1167, "princeton_cluster"),
    (34, "Mistral-7B-Instruct-v0.2", "Mistral-7B-Instruct-v0.2", 1151, "princeton_cluster"),
    (35, "Phi-3-mini-128k-Instruct", "Phi-3-mini-128k-instruct", 1130, "princeton_cluster"),
    (36, "Llama-3.2-1B-Instruct", "Llama-3.2-1B-Instruct", 1112, "princeton_cluster"),
]

# Quick test models - one cheap/fast model from each provider
QUICK_TEST_MODELS = {
    "openai": "gpt-4o",
    "anthropic": "claude-3-haiku",
    "google": "gemini-3-flash",
    "xai": "grok-4",
    "openrouter": "amazon-nova-micro",
    "princeton_cluster": "Llama-3.2-3B-Instruct",
}

# API key environment variable names
API_KEY_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "xai": "XAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "princeton_cluster": None,  # No API key needed
}

# Simple test prompt
TEST_PROMPT = "Say 'Hello, I am working!' and nothing else."


def print_header(title: str, char: str = "=", width: int = 70):
    """Print a formatted header."""
    print(char * width)
    print(title.center(width))
    print(char * width)


def print_api_key_status():
    """Check and display API key status."""
    print_header("API KEY STATUS")
    print()

    status = {}
    for provider, env_var in API_KEY_VARS.items():
        if env_var is None:
            status[provider] = True
            print(f"✅ {provider.upper():20} | No API key needed (local)")
        elif os.getenv(env_var):
            status[provider] = True
            print(f"✅ {provider.upper():20} | {env_var} is set")
        else:
            status[provider] = False
            print(f"❌ {provider.upper():20} | {env_var} is MISSING")

    print()
    return status


def check_cluster_availability() -> Tuple[bool, str]:
    """Check if we're running on Princeton cluster with GPU access."""
    models_base = "/scratch/gpfs/DANQIC/models"

    if not os.path.exists(models_base):
        return False, "Models directory not found (not on cluster)"

    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return True, f"GPU available: {gpu_name}"
        else:
            return False, "No GPU detected (model loading will be very slow)"
    except ImportError:
        return False, "PyTorch not available"


def create_minimal_context():
    """Create a minimal NegotiationContext for testing."""
    from negotiation.llm_agents import NegotiationContext

    return NegotiationContext(
        current_round=1,
        max_rounds=1,
        items=[{"name": "test_item", "value": 1}],
        agents=["test_agent"],
        agent_id="test_agent",
        preferences={"test_item": 1},
        conversation_history=[],
        turn_type="discussion"
    )


async def test_single_model(model_name: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Test a single model with a simple prompt.

    Returns:
        Dict with keys: success, message, latency, model
    """
    from strong_models_experiment.agents import StrongModelAgentFactory

    factory = StrongModelAgentFactory()
    start_time = time.time()

    try:
        # Create a single agent
        # Use 2000 tokens - reasonable limit for negotiation responses
        # This handles detailed responses while being cost-effective
        config = {"max_tokens_default": 2000}
        agents = await factory.create_agents([model_name], config)

        if not agents:
            return {
                "success": False,
                "message": "Failed to create agent (check API key or model config)",
                "latency": 0.0,
                "model": model_name
            }

        agent = agents[0]

        # Create minimal context for testing
        context = create_minimal_context()

        # Send test prompt
        response = await agent.generate_response(
            context=context,
            prompt=TEST_PROMPT
        )

        latency = time.time() - start_time

        if response and response.content and len(response.content) > 0:
            # Truncate response for display
            display_response = response.content[:100]
            if len(response.content) > 100:
                display_response += "..."

            return {
                "success": True,
                "message": f"Response: {display_response}",
                "latency": latency,
                "model": model_name,
                "response_length": len(response.content)
            }
        else:
            return {
                "success": False,
                "message": "Empty response received",
                "latency": latency,
                "model": model_name
            }

    except Exception as e:
        latency = time.time() - start_time
        error_msg = str(e)
        if len(error_msg) > 300:
            error_msg = error_msg[:300] + "..."

        if verbose:
            import traceback
            error_msg += f"\n{traceback.format_exc()}"

        return {
            "success": False,
            "message": f"Error: {error_msg}",
            "latency": latency,
            "model": model_name
        }


def print_result(rank: int, name: str, elo: int, provider: str,
                 result: Dict[str, Any], verbose: bool = False):
    """Pretty print test result."""
    success = result.get("success", False)
    latency = result.get("latency", 0.0)
    message = result.get("message", "")

    icon = "✅" if success else "❌"
    status = "PASS" if success else "FAIL"

    print(f"\n{icon} [{rank:2}] {name}")
    print(f"     Provider: {provider.upper():15} | Elo: {elo:4} | Status: {status}")
    print(f"     Latency: {latency:.2f}s")

    if verbose or not success:
        # Show full message for failures or in verbose mode
        print(f"     {message}")
    elif success and "Response:" in message:
        # Just show first line of response for successes
        print(f"     {message.split(chr(10))[0][:80]}")


async def run_tests(
    models: List[Tuple],
    api_key_status: Dict[str, bool],
    include_cluster: bool,
    verbose: bool
) -> Dict[str, Any]:
    """Run tests for the specified models."""
    results = {
        "by_model": {},
        "by_provider": {},
        "summary": {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "total": len(models)
        }
    }

    # Group models by provider for summary
    for provider in API_KEY_VARS.keys():
        results["by_provider"][provider] = {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "models": []
        }

    for rank, name, config_key, elo, provider in models:
        # Check if we should skip this model
        skip_reason = None

        # Skip cluster models if not requested
        if provider == "princeton_cluster" and not include_cluster:
            skip_reason = "Cluster models skipped (use --include-cluster)"

        # Skip if API key is missing
        elif not api_key_status.get(provider, False):
            skip_reason = f"API key missing ({API_KEY_VARS.get(provider, 'unknown')})"

        if skip_reason:
            result = {
                "success": None,
                "message": skip_reason,
                "latency": 0.0,
                "model": config_key,
                "skipped": True
            }
            results["by_model"][config_key] = result
            results["by_provider"][provider]["skipped"] += 1
            results["by_provider"][provider]["models"].append(config_key)
            results["summary"]["skipped"] += 1

            if verbose:
                print(f"\n⚠️  [{rank:2}] {name}")
                print(f"     Skipped: {skip_reason}")
            continue

        # Run the test
        print(f"\nTesting [{rank}] {name} ({provider})...", end="", flush=True)
        result = await test_single_model(config_key, verbose)

        # Store results
        results["by_model"][config_key] = result
        results["by_provider"][provider]["models"].append(config_key)

        if result["success"]:
            results["by_provider"][provider]["passed"] += 1
            results["summary"]["passed"] += 1
        else:
            results["by_provider"][provider]["failed"] += 1
            results["summary"]["failed"] += 1

        # Clear the "Testing..." line and print result
        print("\r" + " " * 60 + "\r", end="")
        print_result(rank, name, elo, provider, result, verbose)

    return results


def print_summary(results: Dict[str, Any]):
    """Print test summary."""
    print()
    print_header("TEST SUMMARY")
    print()

    summary = results["summary"]
    by_provider = results["by_provider"]

    # Provider-level summary
    print("Results by Provider:")
    print("-" * 50)
    for provider, data in by_provider.items():
        passed = data["passed"]
        failed = data["failed"]
        skipped = data["skipped"]
        total = passed + failed + skipped

        if total == 0:
            continue

        if failed > 0:
            icon = "❌"
        elif skipped == total:
            icon = "⚠️ "
        else:
            icon = "✅"

        print(f"{icon} {provider.upper():20} | "
              f"Pass: {passed:2} | Fail: {failed:2} | Skip: {skipped:2}")

    # Overall summary
    print()
    print("-" * 50)
    total = summary["passed"] + summary["failed"] + summary["skipped"]
    print(f"Total: {total:3} | "
          f"Passed: {summary['passed']:3} | "
          f"Failed: {summary['failed']:3} | "
          f"Skipped: {summary['skipped']:3}")

    # Final verdict
    print()
    if summary["failed"] > 0:
        print("❌ Some tests FAILED - investigate before running experiments!")
        return 1
    elif summary["passed"] == 0:
        print("⚠️  No tests ran - check API keys and settings")
        return 1
    else:
        print("✅ All tested models are working!")
        return 0


def save_results(results: Dict[str, Any], output_path: Path):
    """Save results to JSON file."""
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "summary": results["summary"],
        "by_provider": results["by_provider"],
        "by_model": results["by_model"]
    }

    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test all models from the Chatbot Arena Leaderboard"
    )
    parser.add_argument(
        "--include-cluster", action="store_true",
        help="Include Princeton cluster models (requires GPU)"
    )
    parser.add_argument(
        "--providers", nargs="+",
        choices=["openai", "anthropic", "google", "xai", "openrouter", "princeton_cluster"],
        help="Only test specific providers"
    )
    parser.add_argument(
        "--models", nargs="+",
        help="Only test specific models (by config key)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick test - one model per provider"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output with full error messages"
    )
    parser.add_argument(
        "--output", type=str, default="tests/all_models_test_results.json",
        help="Output JSON file path"
    )

    args = parser.parse_args()

    print_header("MODEL API VALIDATION TEST", "=", 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Project root: {project_root}")
    print()

    # Check API keys
    api_key_status = print_api_key_status()

    # Check cluster availability if needed
    if args.include_cluster:
        cluster_available, cluster_msg = check_cluster_availability()
        api_key_status["princeton_cluster"] = cluster_available
        print(f"Cluster status: {cluster_msg}")
        print()
    else:
        api_key_status["princeton_cluster"] = False

    # Select models to test
    if args.quick:
        # Quick test - one model per provider
        models = []
        for rank, name, config_key, elo, provider in LEADERBOARD_MODELS:
            if config_key == QUICK_TEST_MODELS.get(provider):
                models.append((rank, name, config_key, elo, provider))
        print(f"Quick test mode: testing {len(models)} models (one per provider)")
    elif args.models:
        # Specific models
        models = [
            m for m in LEADERBOARD_MODELS
            if m[2] in args.models
        ]
        print(f"Testing {len(models)} specified models")
    elif args.providers:
        # Specific providers
        models = [
            m for m in LEADERBOARD_MODELS
            if m[4] in args.providers
        ]
        print(f"Testing {len(models)} models from providers: {', '.join(args.providers)}")
    else:
        # All models
        models = LEADERBOARD_MODELS
        print(f"Testing all {len(models)} models from leaderboard")

    if not models:
        print("No models to test!")
        return 1

    print()
    print_header("RUNNING TESTS", "-", 70)

    # Run tests
    results = asyncio.run(run_tests(
        models,
        api_key_status,
        args.include_cluster,
        args.verbose
    ))

    # Print summary
    exit_code = print_summary(results)

    # Save results
    output_path = project_root / args.output
    save_results(results, output_path)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

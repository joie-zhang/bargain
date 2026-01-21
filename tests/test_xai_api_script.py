import sys
import os
from pathlib import Path

# Find project root (parent of tests/)
notebook_dir = Path.cwd()
if notebook_dir.name == 'tests':
    project_root = notebook_dir.parent
else:
    project_root = notebook_dir

sys.path.insert(0, str(project_root))
os.chdir(project_root)

print(f"Project root: {project_root}")
print(f"Working directory: {os.getcwd()}")

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import traceback

# Check for required environment variables
print("=" * 60)
print("API KEY STATUS")
print("=" * 60)

api_keys = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
    "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
    "XAI_API_KEY": os.getenv("XAI_API_KEY"),
    "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
}

for key_name, key_value in api_keys.items():
    status = "SET" if key_value else "MISSING"
    icon = "\u2705" if key_value else "\u274c"
    print(f"{icon} {key_name}: {status}")

from strong_models_experiment.configs import STRONG_MODELS_CONFIG
from strong_models_experiment.agents import StrongModelAgentFactory
from negotiation.llm_agents import NegotiationContext

# Display available models by API type
print("=" * 60)
print("MODELS BY API TYPE")
print("=" * 60)

models_by_api = {}
for model_name, config in STRONG_MODELS_CONFIG.items():
    api_type = config.get("api_type", "unknown")
    if api_type not in models_by_api:
        models_by_api[api_type] = []
    # Skip deprecated models
    if not config.get("deprecated", False):
        models_by_api[api_type].append(model_name)

for api_type, models in sorted(models_by_api.items()):
    print(f"\n{api_type.upper()} ({len(models)} models):")
    for m in models[:5]:  # Show first 5
        print(f"  - {m}")
    if len(models) > 5:
        print(f"  ... and {len(models) - 5} more")

TEST_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-haiku",
    "google": "gemini-2-0-flash-lite",
    "xai": "grok-4-1-thinking",
    "openrouter": "amazon-nova-micro",
    "princeton_cluster": "Llama-3.2-3B-Instruct",
}

# Simple test prompt
TEST_PROMPT = "Say 'Hello, I am working!' and nothing else."

print("Test models selected:")
for api_type, model in TEST_MODELS.items():
    print(f"  {api_type}: {model}")

test_results = {}

def create_minimal_context() -> NegotiationContext:
    """
    Create a minimal NegotiationContext for testing.
    The generate_response method requires this context object.
    """
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


async def test_single_model(model_name: str, api_type: str) -> Tuple[bool, str, float]:
    """
    Test a single model with a simple prompt.
    
    Returns:
        Tuple of (success: bool, message: str, latency: float)
    """
    factory = StrongModelAgentFactory()
    start_time = time.time()
    
    try:
        # Create a single agent
        config = {"max_tokens_default": 2000}
        agents = await factory.create_agents([model_name], config)
        
        if not agents:
            return False, "Failed to create agent (check API key)", 0.0
        
        agent = agents[0]
        
        # Create minimal context for testing
        context = create_minimal_context()
        
        # Send test prompt using correct method signature
        response = await agent.generate_response(
            context=context,
            prompt=TEST_PROMPT
        )
        
        latency = time.time() - start_time
        
        if response and response.content and len(response.content) > 0:
            # Truncate response for display
            display_response = response.content[:100] + "..." if len(response.content) > 100 else response.content
            return True, f"Response: {display_response}", latency
        else:
            return False, "Empty response received", latency
            
    except Exception as e:
        latency = time.time() - start_time
        error_msg = str(e)
        # Truncate long error messages
        if len(error_msg) > 200:
            error_msg = error_msg[:200] + "..."
        return False, f"Error: {error_msg}", latency


def print_result(api_type: str, model: str, success: bool, message: str, latency: float):
    """Pretty print test result."""
    icon = "\u2705" if success else "\u274c"
    status = "PASS" if success else "FAIL"
    print(f"\n{icon} [{api_type.upper()}] {model}")
    print(f"   Status: {status}")
    print(f"   Latency: {latency:.2f}s")
    print(f"   {message}")

api_type = "xai"
model = TEST_MODELS[api_type]

print(f"Testing {api_type.upper()} with model: {model}")
print("-" * 50)

if not os.getenv("XAI_API_KEY"):
    print("\u274c XAI_API_KEY not set - skipping")
    test_results[api_type] = {"success": False, "message": "API key missing", "latency": 0}
else:
    # test_single_model is an async function and returns a coroutine.
    # You must use 'await' to execute it within an async function,
    # or use asyncio.run() if at the top level in a script (not inside any running event loop).
    # Here, this is top-level code, so use asyncio.run():
    success, message, latency = asyncio.run(test_single_model(model, api_type))
    test_results[api_type] = {"success": success, "message": message, "latency": latency, "model": model}
    print_result(api_type, model, success, message, latency)
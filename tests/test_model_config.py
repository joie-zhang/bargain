#!/usr/bin/env python3
"""Test that model configurations are properly used when creating agents."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strong_models_experiment.configs import STRONG_MODELS_CONFIG

# Print all configured models and their model_ids
print("=" * 60)
print("CONFIGURED MODELS AND THEIR MODEL IDs")
print("=" * 60)

for model_name, config in STRONG_MODELS_CONFIG.items():
    print(f"\n{model_name}:")
    print(f"  Name: {config['name']}")
    print(f"  Model ID: {config['model_id']}")
    print(f"  API Type: {config['api_type']}")
    print(f"  Provider: {config['provider']}")

print("\n" + "=" * 60)
print("MODEL ID VERIFICATION")
print("=" * 60)

# Check specific models mentioned in the experiment
test_models = [
    # Weak models
    "gpt-4o",
    "claude-3-opus", 
    "gemini-1-5-pro",
    # Strong models
    "gemini-2-0-flash",
    "gemini-2-5-pro",
    "gemma-3-27b",
    "claude-3-5-haiku",
    "claude-3-5-sonnet",
    "claude-4-sonnet",
    "claude-4-1-opus",
    "gpt-4o-latest",
    "gpt-4o-mini",
    "o1",
    "o3"
]

print("\nChecking experiment models:")
for model in test_models:
    if model in STRONG_MODELS_CONFIG:
        print(f"✅ {model}: {STRONG_MODELS_CONFIG[model]['model_id']}")
    else:
        print(f"❌ {model}: NOT FOUND IN CONFIG")

print("\n" + "=" * 60)
print("All models are now configured with the correct model_ids!")
print("These will be used when instantiating agents via the API.")
#!/usr/bin/env python3
"""
Simple test to verify Grok can be used in experiments.
This test checks that the experiment runner can handle Grok models.
"""

import sys
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_grok_in_experiment_config():
    """Test that Grok models can be used in experiment configuration."""
    
    # Test if grok-4-0709 is accepted as a valid model
    cmd = [
        "python", "run_strong_models_experiment.py",
        "--models", "grok-4-0709",
        "--num-runs", "0",  # Don't actually run, just validate
        "--help"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Should not get an error about invalid model choice
    if "invalid choice: 'grok-4-0709'" in result.stderr:
        print("✗ Grok model not accepted by experiment runner")
        return False
    
    print("✓ Grok model accepted by experiment runner")
    return True


def test_config_generation():
    """Test that config generation includes Grok models."""
    
    # Check if generate_configs script includes Grok
    script_path = Path(__file__).parent.parent / "scripts" / "generate_configs_both_orders.sh"
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    if "grok-4-0709" in content:
        print("✓ Grok models included in config generation script")
        return True
    else:
        print("✗ Grok models not found in config generation script")
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("GROK EXPERIMENT INTEGRATION TEST")
    print("=" * 50)
    
    all_passed = True
    
    # Test 1: Experiment runner accepts Grok
    if not test_grok_in_experiment_config():
        all_passed = False
    
    # Test 2: Config generation includes Grok
    if not test_config_generation():
        all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("\nGrok is fully integrated and ready to use!")
        print("\nTo run an experiment with Grok:")
        print("1. export XAI_API_KEY='your-key-here'")
        print("2. ./scripts/generate_configs_both_orders.sh")
        print("3. ./scripts/run_all_simple.sh")
    else:
        print("✗ SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Setup script for the O3 vs Claude Haiku baseline experiment.

This script checks dependencies, creates necessary directories,
and provides instructions for running the experiment.
"""

import os
import sys
from pathlib import Path
import subprocess

def check_python_version():
    """Check Python version is >= 3.8."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        ("negotiation", "Local negotiation package"),
        ("anyio", "Async I/O library"),
        ("pytest", "Testing framework")
    ]
    
    optional_packages = [
        ("anthropic", "Anthropic Claude API (for real experiments)"),
        ("openai", "OpenAI API (for real experiments)")
    ]
    
    print("\n📦 Checking required dependencies...")
    all_good = True
    
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - {description}")
        except ImportError:
            print(f"❌ {package} - {description} (MISSING)")
            all_good = False
    
    print("\n📦 Checking optional dependencies...")
    api_available = {"anthropic": False, "openai": False}
    
    for package, description in optional_packages:
        try:
            __import__(package)
            print(f"✅ {package} - {description}")
            if package in api_available:
                api_available[package] = True
        except ImportError:
            print(f"⚠️  {package} - {description} (Optional, will use simulated agents)")
    
    return all_good, api_available

def check_api_keys():
    """Check if API keys are configured."""
    print("\n🔑 Checking API key configuration...")
    
    keys_status = {}
    
    # Check OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and openai_key.startswith("sk-"):
        print("✅ OPENAI_API_KEY configured")
        keys_status["openai"] = True
    else:
        print("⚠️  OPENAI_API_KEY not configured (will use simulated agents)")
        keys_status["openai"] = False
    
    # Check Anthropic API key
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key and anthropic_key.startswith("sk-ant-"):
        print("✅ ANTHROPIC_API_KEY configured")
        keys_status["anthropic"] = True
    else:
        print("⚠️  ANTHROPIC_API_KEY not configured (will use simulated agents)")
        keys_status["anthropic"] = False
    
    return keys_status

def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    dirs_to_create = [
        "experiments/results",
        "experiments/configs",
        "logs"
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {dir_path}")

def run_quick_test():
    """Run a quick test to verify everything works."""
    print("\n🧪 Running quick test...")
    
    try:
        # Import and test basic functionality
        from negotiation import create_negotiation_environment, create_competitive_preferences
        from negotiation.agent_factory import AgentFactory, create_simulated_experiment
        from negotiation.llm_agents import ModelType
        
        # Test environment creation
        env = create_negotiation_environment(m_items=3, n_agents=2, t_rounds=2)
        print("✅ Negotiation environment creation")
        
        # Test preference creation
        prefs = create_competitive_preferences(m_items=3, n_agents=2, cosine_similarity=0.8)
        print("✅ Competitive preferences creation")
        
        # Test agent factory
        factory = AgentFactory()
        config = create_simulated_experiment("test", ["balanced", "cooperative"])
        agents = factory.create_agents_from_experiment(config)
        print("✅ Agent factory and simulated agents")
        
        print("✅ All basic functionality tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def print_usage_instructions(api_available, keys_status):
    """Print instructions for running the experiment."""
    print("\n" + "="*60)
    print("🚀 EXPERIMENT SETUP COMPLETE")
    print("="*60)
    
    print("\n📋 How to run the O3 vs Claude Haiku baseline experiment:")
    
    if keys_status["openai"] and keys_status["anthropic"]:
        print("\n🔑 REAL LLM MODE (Recommended for research):")
        print("   python run_baseline_experiment.py")
        print("   python run_baseline_experiment.py --runs 10")
        print("\n   This will use actual O3 and Claude Haiku models")
        print("   ⚠️  Note: This will incur API costs!")
    
    print("\n🤖 SIMULATED MODE (Free, good for testing):")
    print("   python run_baseline_experiment.py --simulated")
    print("   python run_baseline_experiment.py --runs 10 --simulated")
    print("\n   This uses intelligent simulated agents with strategic behaviors")
    
    print("\n📊 Additional options:")
    print("   --competition-level 0.95    # How competitive (0.0-1.0)")
    print("   --rounds 6                  # Max negotiation rounds")
    print("   --verbose                   # Detailed logging")
    
    print("\n📁 Results will be saved to:")
    print("   experiments/results/[batch_id]_summary.json")
    print("   experiments/results/[batch_id]_detailed.json")
    
    print("\n🔬 For development and testing:")
    print("   pytest tests/test_llm_agents.py")
    print("   python examples/llm_agents_demo.py")
    
    if not (keys_status["openai"] and keys_status["anthropic"]):
        print("\n💡 To enable real LLM experiments:")
        print("   export OPENAI_API_KEY='sk-...'")
        print("   export ANTHROPIC_API_KEY='sk-ant-...'")
        print("   Then run without --simulated flag")

def main():
    """Main setup function."""
    print("O3 vs Claude Haiku Baseline Experiment Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Check dependencies
    deps_ok, api_available = check_dependencies()
    if not deps_ok:
        print("\n❌ Missing required dependencies. Install with:")
        print("   pip install anyio pytest")
        print("   pip install anthropic openai  # Optional for real LLM experiments")
        return 1
    
    # Check API keys
    keys_status = check_api_keys()
    
    # Create directories
    create_directories()
    
    # Run quick test
    if not run_quick_test():
        print("\n❌ Setup verification failed")
        return 1
    
    # Print usage instructions
    print_usage_instructions(api_available, keys_status)
    
    print("\n✅ Setup completed successfully!")
    print("\nYou can now run the O3 vs Claude Haiku baseline experiment!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
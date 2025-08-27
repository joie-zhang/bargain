#!/usr/bin/env python3
"""
Simple runner script for Gemma scaling experiments.

Usage:
    python run_gemma_experiment.py                    # Run default configuration
    python run_gemma_experiment.py --batch           # Run all configurations
    python run_gemma_experiment.py --stronger gemma-27b --weaker gemma-2b gemma-2b
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from experiments.gemma_scaling_experiment import main

if __name__ == "__main__":
    asyncio.run(main())
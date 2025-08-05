#!/usr/bin/env python3
"""
Run O3 debugging and capture the output in real-time
"""

import asyncio
import sys
import os
import signal

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from experiments.o3_vs_haiku_baseline import O3VsHaikuExperiment

class DebugRunner:
    def __init__(self):
        self.experiment = O3VsHaikuExperiment()
        self.stop_requested = False
        
    def signal_handler(self, signum, frame):
        print("\n🛑 Stop requested - finishing current operation...")
        self.stop_requested = True
        
    async def run_with_debug(self):
        """Run experiment with debug output, stopping gracefully on Ctrl+C"""
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        
        print("🚀 Starting O3 vs Haiku experiment with full debugging")
        print("Press Ctrl+C to stop gracefully after seeing O3 responses")
        print("=" * 80)
        
        try:
            result = await self.experiment.run_single_experiment(
                experiment_config=None,
                experiment_id=f"debug_manual_stop_{int(__import__('time').time())}"
            )
            
            print("\n✅ Experiment completed successfully!")
            return result
            
        except KeyboardInterrupt:
            print("\n🛑 Experiment interrupted by user")
            return None
        except Exception as e:
            print(f"\n❌ Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            return None

async def main():
    runner = DebugRunner()
    await runner.run_with_debug()

if __name__ == "__main__":
    asyncio.run(main())
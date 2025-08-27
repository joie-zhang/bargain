#!/usr/bin/env python3
"""
Most efficient experiment runner - no timeouts, smart scheduling.
This maximizes throughput while respecting API rate limits.
"""

import json
import subprocess
import time
import os
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, List, Set
import threading
import queue
import random

# Configuration
BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / "experiments/results/scaling_experiment/configs"
LOGS_DIR = BASE_DIR / "experiments/results/scaling_experiment/logs"
SCRIPTS_DIR = BASE_DIR / "scripts"

class ExperimentScheduler:
    """Efficient scheduler that maximizes throughput."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.experiments = self.load_experiments()
        self.completed = set()
        self.failed = set()
        self.in_progress = set()
        self.lock = threading.Lock()
        
        # Provider tracking for load balancing
        self.provider_queues = {
            'anthropic': deque(),
            'openai': deque(),
            'google': deque(),
            'mixed': deque()
        }
        
        # Simple rate limit tracking (requests in last minute)
        self.recent_requests = defaultdict(deque)
        self.max_requests_per_minute = {
            'anthropic': 20,  # Conservative limits
            'openai': 20,
            'google': 20,
            'mixed': 15
        }
        
        print(f"Loaded {len(self.experiments)} experiments to run")
        self.categorize_experiments()
    
    def load_experiments(self) -> List[int]:
        """Load all experiment IDs that need to be run."""
        experiments = []
        for config_file in sorted(CONFIG_DIR.glob("config_*.json")):
            job_id = int(config_file.stem.split('_')[1])
            
            # Skip if already completed
            if not (LOGS_DIR / f"completed_{job_id}.flag").exists():
                experiments.append(job_id)
        
        return experiments
    
    def categorize_experiments(self):
        """Sort experiments by provider for load balancing."""
        for job_id in self.experiments:
            provider = self.get_provider(job_id)
            self.provider_queues[provider].append(job_id)
        
        # Shuffle each queue for better distribution
        for queue in self.provider_queues.values():
            random.shuffle(queue)
        
        print("\nExperiments by provider:")
        for provider, queue in self.provider_queues.items():
            if queue:
                print(f"  {provider}: {len(queue)} experiments")
    
    def get_provider(self, job_id: int) -> str:
        """Determine which API provider(s) an experiment uses."""
        try:
            config_file = CONFIG_DIR / f"config_{job_id}.json"
            with open(config_file) as f:
                config = json.load(f)
            
            models = f"{config.get('weak_model', '')} {config.get('strong_model', '')}"
            
            providers = set()
            if 'claude' in models:
                providers.add('anthropic')
            if 'gpt' in models or 'o3' in models:
                providers.add('openai')
            if 'gemini' in models or 'gemma' in models:
                providers.add('google')
            
            if len(providers) == 1:
                return providers.pop()
            elif len(providers) > 1:
                return 'mixed'
        except:
            pass
        
        return 'mixed'
    
    def can_run_provider(self, provider: str) -> bool:
        """Check if we can run another experiment for this provider."""
        now = time.time()
        
        # Clean old requests (older than 60 seconds)
        recent = self.recent_requests[provider]
        while recent and recent[0] < now - 60:
            recent.popleft()
        
        # Check if under limit
        return len(recent) < self.max_requests_per_minute[provider]
    
    def get_next_job(self) -> int:
        """Get next job to run, balancing across providers."""
        with self.lock:
            # Try each provider in round-robin fashion
            providers = list(self.provider_queues.keys())
            random.shuffle(providers)  # Randomize to avoid bias
            
            for provider in providers:
                if self.provider_queues[provider] and self.can_run_provider(provider):
                    job_id = self.provider_queues[provider].popleft()
                    self.recent_requests[provider].append(time.time())
                    self.in_progress.add(job_id)
                    return job_id
            
            return None
    
    def mark_complete(self, job_id: int, success: bool):
        """Mark an experiment as complete."""
        with self.lock:
            self.in_progress.discard(job_id)
            if success:
                self.completed.add(job_id)
            else:
                self.failed.add(job_id)
                # Re-queue failed experiments for retry (at the end)
                provider = self.get_provider(job_id)
                self.provider_queues[provider].append(job_id)
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        with self.lock:
            total = len(self.experiments)
            return {
                'total': total,
                'completed': len(self.completed),
                'failed': len(self.failed),
                'in_progress': len(self.in_progress),
                'remaining': sum(len(q) for q in self.provider_queues.values())
            }


def run_experiment(job_id: int) -> bool:
    """Run a single experiment - NO TIMEOUT."""
    script = SCRIPTS_DIR / "run_single_experiment_simple.sh"
    if not script.exists():
        script = SCRIPTS_DIR / "run_single_experiment.sh"
    
    try:
        # Run with NO timeout - let it complete
        result = subprocess.run(
            [str(script), str(job_id)],
            capture_output=True,
            text=True
            # NO timeout parameter!
        )
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"  Job {job_id}: Error: {e}")
        return False


def worker(scheduler: ExperimentScheduler, worker_id: int):
    """Worker thread that runs experiments."""
    print(f"Worker {worker_id} started")
    
    while True:
        # Get next job
        job_id = scheduler.get_next_job()
        
        if job_id is None:
            # No jobs available right now
            stats = scheduler.get_stats()
            if stats['remaining'] == 0 and stats['in_progress'] == 0:
                # All done
                break
            
            # Wait a bit and try again
            time.sleep(2)
            continue
        
        # Run the experiment
        print(f"  Worker {worker_id}: Running job {job_id}")
        start_time = time.time()
        
        success = run_experiment(job_id)
        
        elapsed = time.time() - start_time
        status = "✅" if success else "❌"
        print(f"  Worker {worker_id}: Job {job_id} {status} ({elapsed:.0f}s)")
        
        scheduler.mark_complete(job_id, success)
        
        # Small delay to prevent API bursts
        time.sleep(1)
    
    print(f"Worker {worker_id} finished")


def monitor_progress(scheduler: ExperimentScheduler, start_time: float):
    """Monitor and display progress."""
    last_completed = 0
    
    while True:
        time.sleep(10)  # Update every 10 seconds
        
        stats = scheduler.get_stats()
        if stats['remaining'] == 0 and stats['in_progress'] == 0:
            break
        
        elapsed = time.time() - start_time
        completed = stats['completed']
        rate = (completed - last_completed) * 6  # per minute (10s intervals)
        last_completed = completed
        
        # Calculate ETA
        remaining = stats['remaining'] + stats['in_progress']
        eta = remaining / rate * 60 if rate > 0 else 0
        
        print(f"\n[Progress] Completed: {completed}/{stats['total']} | "
              f"In Progress: {stats['in_progress']} | "
              f"Failed: {stats['failed']} | "
              f"Rate: {rate:.1f}/min | "
              f"ETA: {eta:.0f}s")


def main():
    """Main execution."""
    print("="*60)
    print("EFFICIENT EXPERIMENT RUNNER (No Timeouts)")
    print("="*60)
    
    # Parse arguments
    max_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    
    print(f"\nConfiguration:")
    print(f"  Max parallel workers: {max_workers}")
    print(f"  Timeout: NONE (let experiments complete)")
    print(f"  Strategy: Provider-aware load balancing")
    print(f"  Rate limiting: Automatic per provider")
    
    # Initialize scheduler
    scheduler = ExperimentScheduler(max_workers)
    
    if not scheduler.experiments:
        print("\n✅ No experiments to run (all completed)")
        return
    
    # Start workers
    print(f"\nStarting {max_workers} workers...")
    workers = []
    start_time = time.time()
    
    for i in range(max_workers):
        t = threading.Thread(target=worker, args=(scheduler, i))
        t.daemon = True
        t.start()
        workers.append(t)
        time.sleep(2)  # Stagger starts
    
    # Start monitor
    monitor_thread = threading.Thread(target=monitor_progress, args=(scheduler, start_time))
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Wait for completion
    for t in workers:
        t.join()
    
    # Final stats
    elapsed = time.time() - start_time
    stats = scheduler.get_stats()
    
    print("\n" + "="*60)
    print("EXECUTION COMPLETE")
    print("="*60)
    print(f"Total experiments: {stats['total']}")
    print(f"Successful: {stats['completed'] - stats['failed']}")
    print(f"Failed: {stats['failed']}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Average time: {elapsed/stats['completed']:.1f}s per experiment")
    print(f"Throughput: {stats['completed']/(elapsed/60):.1f} experiments/minute")
    
    if stats['failed'] > 0:
        print(f"\n⚠️ {stats['failed']} experiments failed")
        print("These were automatically retried once.")
        print("Run again to retry remaining failures.")


if __name__ == "__main__":
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    main()
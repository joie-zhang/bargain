#!/usr/bin/env python3
"""
Smart rate limit runner that monitors API usage and adjusts parallelism dynamically.
This script provides intelligent scheduling to avoid rate limits.
"""

import json
import subprocess
import time
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple
import threading
import queue

# Configuration
BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / "experiments/results/scaling_experiment/configs"
LOGS_DIR = BASE_DIR / "experiments/results/scaling_experiment/logs"
SCRIPTS_DIR = BASE_DIR / "scripts"

# Rate limit tracking
class RateLimitTracker:
    """Track API usage and rate limits per provider."""
    
    def __init__(self):
        self.requests = defaultdict(list)  # provider -> list of timestamps
        self.rate_limits = {
            'anthropic': {'requests_per_minute': 50, 'cooldown': 60},
            'openai': {'requests_per_minute': 60, 'cooldown': 60},
            'google': {'requests_per_minute': 60, 'cooldown': 60},
            'openrouter': {'requests_per_minute': 30, 'cooldown': 120}
        }
        self.lock = threading.Lock()
    
    def can_make_request(self, provider: str) -> bool:
        """Check if we can make a request to this provider."""
        with self.lock:
            now = datetime.now()
            limit_info = self.rate_limits.get(provider, self.rate_limits['openrouter'])
            
            # Clean old requests (older than cooldown period)
            cutoff = now - timedelta(seconds=limit_info['cooldown'])
            self.requests[provider] = [
                ts for ts in self.requests[provider] 
                if ts > cutoff
            ]
            
            # Check if under rate limit
            return len(self.requests[provider]) < limit_info['requests_per_minute']
    
    def record_request(self, provider: str):
        """Record that a request was made."""
        with self.lock:
            self.requests[provider].append(datetime.now())
    
    def get_wait_time(self, provider: str) -> float:
        """Get how long to wait before next request."""
        with self.lock:
            if not self.requests[provider]:
                return 0
            
            limit_info = self.rate_limits.get(provider, self.rate_limits['openrouter'])
            oldest = min(self.requests[provider])
            cutoff = datetime.now() - timedelta(seconds=limit_info['cooldown'])
            
            if oldest < cutoff:
                return 0
            
            wait_time = (oldest + timedelta(seconds=limit_info['cooldown']) - datetime.now()).total_seconds()
            return max(0, wait_time)


def categorize_experiments() -> Dict[str, List[int]]:
    """Categorize experiments by API provider."""
    categories = {
        'anthropic': [],
        'openai': [],
        'google': [],
        'mixed': [],
        'unknown': []
    }
    
    # Read all config files
    for config_file in sorted(CONFIG_DIR.glob("config_*.json")):
        job_id = int(config_file.stem.split('_')[1])
        
        # Skip if already completed
        if (LOGS_DIR / f"completed_{job_id}.flag").exists():
            continue
        
        try:
            with open(config_file) as f:
                config = json.load(f)
            
            weak = config.get('weak_model', '')
            strong = config.get('strong_model', '')
            
            # Determine primary provider
            providers = set()
            for model in [weak, strong]:
                if 'claude' in model:
                    providers.add('anthropic')
                elif 'gpt' in model or 'o3' in model:
                    providers.add('openai')
                elif 'gemini' in model:
                    providers.add('google')
            
            if len(providers) == 1:
                provider = providers.pop()
                categories[provider].append(job_id)
            elif len(providers) > 1:
                categories['mixed'].append(job_id)
            else:
                categories['unknown'].append(job_id)
                
        except Exception as e:
            print(f"Error reading config {job_id}: {e}")
            categories['unknown'].append(job_id)
    
    return categories


def run_experiment(job_id: int, tracker: RateLimitTracker) -> Tuple[int, bool]:
    """Run a single experiment with rate limit awareness."""
    
    # Determine provider for this experiment
    config_file = CONFIG_DIR / f"config_{job_id}.json"
    provider = 'unknown'
    
    try:
        with open(config_file) as f:
            config = json.load(f)
        
        # Simple provider detection
        models = f"{config.get('weak_model', '')} {config.get('strong_model', '')}"
        if 'claude' in models:
            provider = 'anthropic'
        elif 'gpt' in models or 'o3' in models:
            provider = 'openai'
        elif 'gemini' in models:
            provider = 'google'
    except:
        pass
    
    # Wait if necessary for rate limits
    wait_time = tracker.get_wait_time(provider)
    if wait_time > 0:
        print(f"  Job {job_id}: Waiting {wait_time:.1f}s for {provider} rate limit...")
        time.sleep(wait_time)
    
    # Check if we can make request
    attempts = 0
    while not tracker.can_make_request(provider) and attempts < 10:
        time.sleep(5)
        attempts += 1
    
    # Record the request
    tracker.record_request(provider)
    
    # Run the experiment (with NO timeout or very generous timeout)
    script = SCRIPTS_DIR / "run_single_experiment_no_timeout.sh"
    if not script.exists():
        script = SCRIPTS_DIR / "run_single_experiment.sh"
    
    print(f"  Job {job_id}: Starting ({provider} provider)...")
    
    try:
        result = subprocess.run(
            [str(script), str(job_id)],
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )
        
        success = result.returncode == 0
        if success:
            print(f"  Job {job_id}: ✅ Success")
        else:
            # Check if rate limited
            log_file = LOGS_DIR / f"experiment_{job_id}.log"
            if log_file.exists():
                with open(log_file) as f:
                    log_content = f.read()
                    if '429' in log_content or 'rate' in log_content.lower():
                        print(f"  Job {job_id}: ⚠️ Rate limited")
                        # Add cooldown for this provider
                        tracker.rate_limits[provider]['cooldown'] *= 1.5
                    else:
                        print(f"  Job {job_id}: ❌ Failed")
        
        return job_id, success
        
    except subprocess.TimeoutExpired:
        print(f"  Job {job_id}: ⏱️ Timeout (exceeded 2 hours)")
        return job_id, False
    except Exception as e:
        print(f"  Job {job_id}: ❌ Error: {e}")
        return job_id, False


def worker(job_queue: queue.Queue, results_queue: queue.Queue, tracker: RateLimitTracker):
    """Worker thread for running experiments."""
    while True:
        try:
            job_id = job_queue.get(timeout=1)
            if job_id is None:
                break
            
            result = run_experiment(job_id, tracker)
            results_queue.put(result)
            job_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Worker error: {e}")


def main():
    """Main execution."""
    print("="*60)
    print("Smart Rate-Limit-Aware Experiment Runner")
    print("="*60)
    
    # Parse arguments
    max_parallel = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    
    print(f"Configuration:")
    print(f"  Max parallel jobs: {max_parallel}")
    print(f"  Using intelligent rate limit tracking")
    print(f"  Provider rotation enabled")
    print()
    
    # Categorize experiments
    print("Categorizing experiments by provider...")
    categories = categorize_experiments()
    
    total = sum(len(jobs) for jobs in categories.values())
    print(f"Total pending experiments: {total}")
    for provider, jobs in categories.items():
        if jobs:
            print(f"  {provider}: {len(jobs)} experiments")
    print()
    
    # Create interleaved job list (rotate between providers)
    job_queue = queue.Queue()
    
    # Add jobs in round-robin fashion between providers
    max_jobs = max(len(jobs) for jobs in categories.values())
    for i in range(max_jobs):
        for provider in ['anthropic', 'openai', 'google', 'mixed', 'unknown']:
            if i < len(categories[provider]):
                job_queue.put(categories[provider][i])
    
    # Initialize tracker
    tracker = RateLimitTracker()
    
    # Start worker threads
    print(f"Starting {max_parallel} worker threads...")
    workers = []
    results_queue = queue.Queue()
    
    for i in range(max_parallel):
        t = threading.Thread(target=worker, args=(job_queue, results_queue, tracker))
        t.start()
        workers.append(t)
        time.sleep(2)  # Stagger thread starts
    
    # Monitor progress
    start_time = time.time()
    completed = 0
    failed = 0
    
    print("\nProgress:")
    while completed < total:
        try:
            job_id, success = results_queue.get(timeout=5)
            completed += 1
            if not success:
                failed += 1
            
            elapsed = time.time() - start_time
            rate = completed / (elapsed / 60) if elapsed > 0 else 0
            eta = (total - completed) / rate if rate > 0 else 0
            
            print(f"[{completed}/{total}] Completed: {completed} | Failed: {failed} | "
                  f"Rate: {rate:.1f}/min | ETA: {eta:.0f} min")
            
        except queue.Empty:
            # Check if workers are still alive
            if not any(t.is_alive() for t in workers):
                break
    
    # Signal workers to stop
    for _ in workers:
        job_queue.put(None)
    
    # Wait for workers to finish
    for t in workers:
        t.join()
    
    # Final summary
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print("Execution Complete")
    print("="*60)
    print(f"Total experiments: {total}")
    print(f"Successful: {completed - failed}")
    print(f"Failed: {failed}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Average time per experiment: {elapsed/completed:.1f} seconds")
    
    if failed > 0:
        print(f"\n⚠️ {failed} experiments failed.")
        print("Run this script again to retry failed experiments.")


if __name__ == "__main__":
    # Ensure directories exist
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    main()
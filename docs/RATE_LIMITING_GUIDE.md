# Rate Limiting Guide: Minimizing API Rate Limit Errors

## Problem

When running large-scale experiments (e.g., 330 jobs simultaneously), you can easily overwhelm API providers and hit rate limits. Each job typically has 2 agents making API calls, so 330 jobs = 660 agents all hitting the same API key simultaneously.

## Current Rate Limiting Implementation

### Per-Agent Rate Limiting
- Each agent has its own `RateLimiter` instance
- Default limits: **60 requests/minute**, **10,000 tokens/minute** per agent
- **Problem**: These limits are per-agent, not shared across jobs
- If you have 330 jobs × 2 agents = 660 agents, each thinks it can do 60 req/min
- Total capacity needed: 660 × 60 = **39,600 requests/minute** (way over typical limits!)

### Retry Logic
- Exponential backoff exists in `generate_response()` method
- OpenAI SDK has built-in retries (configurable via `max_retries`)
- **New**: Explicit 429 handling with exponential backoff added to:
  - `OpenAIAgent._call_llm_api()` - Handles `openai.RateLimitError`
  - `AnthropicAgent._call_llm_api()` - Handles `anthropic.RateLimitError`
  - `GoogleAgent._call_llm_api()` - Handles 429/RESOURCE_EXHAUSTED errors

## Strategies to Minimize Rate Limiting

### 1. **Reduce Per-Agent Rate Limits** ⭐ **CRITICAL**

When running many jobs simultaneously, divide your total API capacity by the number of concurrent agents:

```python
# Example: If OpenAI allows 500 requests/minute total
# and you have 330 jobs × 2 agents = 660 agents
# Each agent should be limited to: 500 / 660 ≈ 0.75 requests/minute

# In your agent factory or config:
requests_per_minute = max(1, total_api_capacity // (num_jobs * agents_per_job))
```

**Where to configure:**
- `negotiation/agent_factory.py`: Default `requests_per_minute` (currently 60)
- `strong_models_experiment/agents/agent_factory.py`: Model-specific configs
- Per-experiment: Pass custom rate limits when creating agents

### 2. **Stagger Job Starts** ⭐ **HIGHLY RECOMMENDED**

Don't submit all 330 jobs at once. Add delays between submissions:

```bash
# Example: Submit jobs with 2-second delay between each
for i in {0..329}; do
    sbatch your_script.sbatch
    sleep 2  # Wait 2 seconds before next submission
done
```

**Benefits:**
- Spreads load over time
- Reduces initial burst
- Allows some jobs to finish before others start

**Recommended delay:** 1-5 seconds per job (adjust based on job duration)

### 3. **Use Multiple API Keys** ⭐ **IF AVAILABLE**

If you have multiple API keys, distribute jobs across them:

```bash
# Example: Rotate through 3 API keys
KEYS=("key1" "key2" "key3")
for i in {0..329}; do
    KEY_IDX=$((i % 3))
    export OPENAI_API_KEY="${KEYS[$KEY_IDX]}"
    sbatch your_script.sbatch
    sleep 2
done
```

**Note:** Each key has its own rate limits, effectively multiplying your capacity.

### 4. **Add Random Delays at Job Start**

Add a random delay at the beginning of each job to further spread load:

```python
# At start of run_strong_models_experiment.py
import random
import time

# Random delay between 0-30 seconds to stagger starts
startup_delay = random.uniform(0, 30)
time.sleep(startup_delay)
```

### 5. **Monitor and Adjust**

Watch for rate limit errors in logs:

```bash
# Check for rate limit errors
grep -r "429\|RateLimit\|rate limit" logs/cluster/*.err

# Count failures
grep -l "429\|RateLimit" logs/cluster/*.err | wc -l
```

If you see many errors:
- Increase delays between job submissions
- Reduce per-agent rate limits further
- Reduce number of concurrent jobs

## Implementation Examples

### Example 1: Staggered Job Submission Script

Create `scripts/submit_staggered.sh`:

```bash
#!/bin/bash
# Submit jobs with delays to avoid rate limits

SCRIPT="scripts/submit_cpu_gpt5_high_vs_low_effort.sbatch"
DELAY_SECONDS=2  # Delay between submissions
TOTAL_JOBS=330

echo "Submitting ${TOTAL_JOBS} jobs with ${DELAY_SECONDS}s delay between each..."

for i in $(seq 1 ${TOTAL_JOBS}); do
    echo "[$(date)] Submitting job ${i}/${TOTAL_JOBS}..."
    sbatch "${SCRIPT}"
    sleep ${DELAY_SECONDS}
done

echo "All jobs submitted!"
```

### Example 2: Reduce Rate Limits in Agent Factory

Modify `strong_models_experiment/agents/agent_factory.py`:

```python
# Calculate conservative rate limits based on expected concurrent jobs
def _calculate_rate_limits(self, num_concurrent_jobs: int = 330):
    """Calculate safe rate limits for concurrent jobs."""
    # Assume 2 agents per job
    total_agents = num_concurrent_jobs * 2
    
    # OpenAI tier limits (adjust based on your tier)
    # Tier 1: ~500 req/min, Tier 2: ~5000 req/min, Tier 3: ~10000 req/min
    total_capacity = 500  # Conservative estimate
    
    # Use 80% of capacity to leave buffer
    safe_capacity = int(total_capacity * 0.8)
    requests_per_minute = max(1, safe_capacity // total_agents)
    
    return requests_per_minute
```

### Example 3: Add Startup Delay to Experiments

Modify `run_strong_models_experiment.py`:

```python
import random
import time
import os

# Add random startup delay if running in batch mode
if os.getenv('SLURM_JOB_ID'):  # Running on cluster
    delay = random.uniform(0, 30)  # 0-30 second random delay
    print(f"Adding startup delay: {delay:.2f}s to stagger job starts")
    time.sleep(delay)
```

## Recommended Settings by Scale

### Small Scale (< 10 jobs)
- **Per-agent rate limit**: 60 req/min (default)
- **Stagger delay**: 0-1 seconds
- **No startup delay needed**

### Medium Scale (10-50 jobs)
- **Per-agent rate limit**: 10-20 req/min
- **Stagger delay**: 1-2 seconds
- **Startup delay**: 0-10 seconds

### Large Scale (50-200 jobs)
- **Per-agent rate limit**: 2-5 req/min
- **Stagger delay**: 2-5 seconds
- **Startup delay**: 0-30 seconds

### Very Large Scale (200+ jobs) ⚠️
- **Per-agent rate limit**: 1-2 req/min
- **Stagger delay**: 5-10 seconds
- **Startup delay**: 0-60 seconds
- **Consider**: Using multiple API keys or reducing concurrent jobs

## Monitoring Rate Limits

### Check Current Rate Limit Status

```bash
# Count rate limit errors
grep -c "429\|RateLimit" logs/cluster/*.err

# Find jobs with rate limit issues
grep -l "429\|RateLimit" logs/cluster/*.err

# Check retry patterns
grep "Attempt.*failed" logs/cluster/*.err | head -20
```

### OpenAI Rate Limit Headers

The OpenAI API returns rate limit info in response headers:
- `x-ratelimit-limit-requests`: Total requests allowed per minute
- `x-ratelimit-remaining-requests`: Remaining requests in current window
- `x-ratelimit-reset-requests`: When the limit resets

**Note**: Current implementation doesn't parse these headers, but you could add this for better rate limit management.

## Emergency: If You Hit Rate Limits

1. **Stop submitting new jobs immediately**
2. **Check current rate limit status** (if available via API)
3. **Wait for rate limit window to reset** (usually 1 minute)
4. **Reduce rate limits** and restart with longer delays
5. **Consider**: Pausing some jobs to reduce load

## Future Improvements

1. **Global Rate Limiter**: Shared rate limiter across all agents/jobs
2. **Dynamic Rate Limit Adjustment**: Adjust limits based on API responses
3. **Rate Limit Header Parsing**: Use API headers to set limits dynamically
4. **Job Queue System**: Use a proper job queue with rate limiting built-in
5. **API Key Rotation**: Automatic rotation across multiple keys

## API-Specific Rate Limit Handling

### OpenAI
- **Exception**: `openai.RateLimitError`
- **Handling**: Exponential backoff with jitter (2^attempt seconds)
- **Max wait**: Capped at 60 seconds
- **Location**: `negotiation/llm_agents.py::OpenAIAgent._call_llm_api()`

### Anthropic
- **Exception**: `anthropic.RateLimitError`
- **Handling**: Exponential backoff with jitter (2^attempt seconds)
- **Max wait**: Capped at 60 seconds
- **Location**: `negotiation/llm_agents.py::AnthropicAgent._call_llm_api()`
- **Note**: Anthropic SDK also has built-in retries (default: 2 retries)

### Google Generative AI
- **Exception**: Generic exceptions with error messages containing "429", "RESOURCE_EXHAUSTED", or "rate limit"
- **Handling**: Exponential backoff with jitter (2^attempt seconds)
- **Max wait**: Capped at 60 seconds
- **Special handling**: Detects daily quota errors and stops retrying (doesn't retry indefinitely)
- **Location**: `negotiation/llm_agents.py::GoogleAgent._call_llm_api()`

## References

- OpenAI Rate Limits: https://platform.openai.com/docs/guides/rate-limits
- Anthropic Rate Limits: https://docs.anthropic.com/en/api/rate-limits
- Google Generative AI Quotas: https://ai.google.dev/pricing
- Current Rate Limiter: `negotiation/llm_agents.py::RateLimiter`
- OpenAI Client: `negotiation/llm_agents.py::OpenAIAgent`
- Anthropic Client: `negotiation/llm_agents.py::AnthropicAgent`
- Google Client: `negotiation/llm_agents.py::GoogleAgent`

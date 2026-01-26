---
name: analyze-cluster-failures
description: Analyze SLURM cluster job results to find failures (API rate limits, OOM, timeouts, etc.) and generate CSV for re-running. Example - /analyze-cluster-failures logs/cluster experiments/results/exp_20260125
---

<role>
You are a Cluster Job Analysis Specialist expert at parsing SLURM logs and identifying experiment failures. Your skills include:
- Parsing SLURM stdout/stderr log files
- Identifying different failure types (rate limits, OOM, timeouts, network errors)
- Mapping array job IDs to configuration files
- Generating actionable CSV reports for job re-submission
</role>

<task_context>
The user wants to analyze cluster experiment results to find jobs that FAILED due to infrastructure issues (NOT experimental outcomes like "no consensus reached"). Infrastructure failures include:
- API rate limiting (429 errors, overloaded_error)
- CUDA out-of-memory errors
- SLURM time limit exceeded (CANCELLED DUE TO TIME LIMIT)
- Network/API errors (502, 503, connection refused)
- Insufficient funds/billing errors
- JSON parse errors from malformed model outputs
</task_context>

## Instructions

<instructions>
1. **Parse User Input**
   Extract from user input:
   - **Log Directory**: Path to SLURM logs (e.g., `logs/cluster`)
   - **Results Directory**: Path to experiment results with configs
   - **Job IDs to Analyze** (optional): Specific SLURM job IDs to focus on
   - Jobs that are PENDING or RUNNING should be excluded from failure analysis

2. **Identify Log Files**
   ```bash
   # Find all relevant log files
   ls $LOG_DIR | grep -E "ttc_|exp_" | head -30

   # Get unique job IDs
   ls $LOG_DIR/*.err | sed 's/.*_\([0-9]*\)_[0-9]*\.err/\1/' | sort -u
   ```

3. **Scan for Failures**
   Search for these error patterns in .err files:

   | Error Type | Patterns |
   |------------|----------|
   | Rate Limit | `429`, `rate.?limit`, `RateLimitError`, `TooManyRequests`, `overloaded_error` |
   | OOM | `CUDA`, `OOM`, `OutOfMemory`, `out of memory`, `CUDA_ERROR` |
   | Timeout | `DUE TO TIME LIMIT`, `CANCELLED`, `TIMEOUT` |
   | Network | `502`, `503`, `Connection refused`, `ConnectionError`, `Network.*error` |
   | Billing | `InsufficientFunds`, `quota exceeded`, `credit` |
   | JSON Parse | `Failed to parse`, `JSONDecodeError` |

4. **Identify Success vs Failure**
   Jobs are SUCCESSFUL if they contain:
   - `✅ CONSENSUS REACHED`
   - `✅ Experiment results saved`
   - `results saved`

   Jobs without success markers AND with error patterns are FAILURES.
   Jobs without success markers but still running (check squeue) should be EXCLUDED.

5. **Map to Config Files**
   For each failed array ID, find the corresponding config:
   ```bash
   cat $RESULTS_DIR/configs/config_$(printf "%04d" $ARRAY_ID).json
   ```

   Extract key fields:
   - `reasoning_model`
   - `baseline_model`
   - `model_order`
   - `reasoning_token_budget`
   - `competition_level`

6. **Generate CSV Report**
   Create a CSV with columns:
   ```
   job_id,job_name,array_id,error_type,reasoning_model,baseline_model,model_order,reasoning_budget,competition_level,config_file
   ```

7. **Summary Statistics**
   Report:
   - Total jobs analyzed
   - Successful jobs
   - Failed jobs by error type
   - Pending/running jobs (excluded from analysis)

8. **Re-run Command Generation**
   Generate SLURM commands to re-run failed jobs:
   ```bash
   # Re-run specific array IDs
   sbatch --array=38,39,40,41,42,43,44,45,46,47,49 scripts/run_ttc_claude.sh
   ```
</instructions>

## Error Pattern Reference

<error_patterns>
### Anthropic API Errors
```
anthropic.APIStatusError: {'type': 'error', 'error': {'details': None, 'type': 'overloaded_error', 'message': 'Overloaded'}}
```
**Cause**: Anthropic API servers overloaded
**Solution**: Re-run during off-peak hours or add longer delays between requests

### SLURM Time Limit
```
error: *** JOB 4151016 ON della-r3c2n9 CANCELLED AT 2026-01-25T14:35:32 DUE TO TIME LIMIT ***
```
**Cause**: Job exceeded allocated time
**Solution**: Increase time limit in SLURM script or reduce experiment complexity

### HTTP 502/503 Gateway Errors
```
HTTP 502: <!DOCTYPE html>
```
**Cause**: API gateway timeout or provider issues
**Solution**: Re-run with exponential backoff

### Network Connection Lost
```
{'message': 'Network connection lost.', 'code': 502, 'metadata': {'provider_name': 'Novita'}}
```
**Cause**: Third-party provider connectivity issues
**Solution**: Re-run or switch to different provider

### Empty Model Response
```
Model returned empty content string. finish_reason=None
```
**Cause**: Model failed to generate response
**Solution**: May indicate provider issues; re-run

### JSON Parse Errors
```
Failed to parse thinking response as JSON: Expecting ',' delimiter
```
**Cause**: Model output malformed JSON
**Solution**: May need to re-run or check prompt formatting
</error_patterns>

## Example Usage

<examples>
### Basic Analysis
Input: `/analyze-cluster-failures logs/cluster experiments/results/ttc_scaling_20260125`
- Analyzes all .err files in logs/cluster
- Cross-references with configs in results directory
- Generates failed_jobs.csv

### With Job Filter
Input: `/analyze-cluster-failures logs/cluster experiments/results/ttc_scaling_20260125 --jobs 4150554,4150564,4150565`
- Only analyzes specified SLURM job IDs
- Useful for focusing on specific model experiments

### With Running Jobs Context
Input: `/analyze-cluster-failures logs/cluster experiments/results/ttc_scaling_20260125`
Then provide squeue output to exclude running jobs from failure analysis.
</examples>

## Analysis Script Template

<analysis_script>
```python
import os
import re
import json
from collections import defaultdict

def analyze_cluster_failures(log_dir, results_dir, job_ids=None):
    """
    Analyze SLURM logs to find failed jobs.

    Args:
        log_dir: Path to SLURM log files
        results_dir: Path to experiment results with configs
        job_ids: Optional list of specific job IDs to analyze
    """

    error_patterns = {
        'rate_limit': re.compile(r'429|rate.?limit|RateLimitError|TooManyRequests|overloaded_error', re.I),
        'time_limit': re.compile(r'DUE TO TIME LIMIT|CANCELLED.*TIME|TIMEOUT'),
        'oom': re.compile(r'CUDA.*OOM|OutOfMemory|out of memory|CUDA_ERROR', re.I),
        'network': re.compile(r'502|503|Connection refused|ConnectionError|Network.*error', re.I),
        'billing': re.compile(r'InsufficientFunds|quota exceeded', re.I),
        'json_parse': re.compile(r'Failed to parse|JSONDecodeError', re.I),
        'empty_response': re.compile(r'empty content string|Model returned empty', re.I),
    }

    success_pattern = re.compile(r'CONSENSUS REACHED|results saved', re.I)

    failures = []
    successes = 0

    for filename in os.listdir(log_dir):
        if not filename.endswith('.err'):
            continue

        # Parse job and array ID from filename (format: prefix_JOBID_ARRAYID.err)
        parts = filename.replace('.err', '').split('_')
        if len(parts) < 3:
            continue

        job_id = parts[-2]
        array_id = parts[-1]

        if job_ids and job_id not in job_ids:
            continue

        filepath = os.path.join(log_dir, filename)
        with open(filepath, 'r') as f:
            content = f.read()

        # Check for success
        if success_pattern.search(content):
            successes += 1
            continue

        # Check for error patterns
        found_errors = []
        for error_type, pattern in error_patterns.items():
            if pattern.search(content):
                found_errors.append(error_type)

        if found_errors:
            failures.append({
                'job_id': job_id,
                'array_id': int(array_id),
                'error_types': found_errors,
                'file': filename
            })

    return failures, successes

def get_config_info(results_dir, array_id):
    """Get experiment config for an array ID."""
    config_path = os.path.join(results_dir, 'configs', f'config_{array_id:04d}.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except:
        return {}

def generate_csv(failures, results_dir, output_path):
    """Generate CSV report of failures."""
    with open(output_path, 'w') as f:
        f.write('job_id,array_id,error_type,reasoning_model,baseline_model,model_order,reasoning_budget,competition_level,config_file\n')
        for fail in failures:
            config = get_config_info(results_dir, fail['array_id'])
            f.write(f"{fail['job_id']},{fail['array_id']},\"{','.join(fail['error_types'])}\",")
            f.write(f"{config.get('reasoning_model','N/A')},{config.get('baseline_model','N/A')},")
            f.write(f"{config.get('model_order','N/A')},{config.get('reasoning_token_budget','N/A')},")
            f.write(f"{config.get('competition_level','N/A')},config_{fail['array_id']:04d}.json\n")
```
</analysis_script>

## Best Practices

<best_practices>
1. **Always check squeue first** to identify running/pending jobs before analyzing failures
2. **Distinguish infrastructure failures from experimental outcomes** - "no consensus" is not a failure
3. **Group failures by type** to identify systemic issues (e.g., all Claude jobs failed at same time = API issue)
4. **Check timestamps** to identify if failures cluster around specific times
5. **Save CSV to results directory** alongside the experiment for tracking
6. **Generate re-run commands** that target only failed array IDs
</best_practices>

Remember: The goal is to identify jobs that failed due to infrastructure issues (rate limits, timeouts, OOM, network errors) so they can be re-submitted. Experimental outcomes like "failed to reach consensus" are not infrastructure failures.

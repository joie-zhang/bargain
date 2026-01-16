# Log Management Guide: Cluster Log Hygiene

## Problem

When running many cluster jobs, `logs/cluster/` can quickly fill up with hundreds of log files, making it hard to:
- Find the most recent log file
- Monitor running jobs
- Clean up old logs
- Debug issues

## Quick Start

### Source the utilities
```bash
source scripts/log_utils.sh
```

### Find the latest log
```bash
# Latest log (any type)
latest_log

# Latest error log
latest_log err

# Latest output log
latest_log out
```

### View recent logs
```bash
# Show 10 most recent logs
recent_logs

# Show 20 most recent error logs
recent_logs 20 err
```

### Tail/follow latest log
```bash
# Tail latest log (last 50 lines)
tail_latest

# Tail latest error log (last 100 lines)
tail_latest err 100

# Follow latest log (like tail -f)
follow_latest err
```

### Or use as standalone script
```bash
# Show latest log path
./scripts/log_utils.sh latest err

# Show 20 most recent logs
./scripts/log_utils.sh recent 20

# Tail latest error log
./scripts/log_utils.sh tail err 100

# Follow latest output log
./scripts/log_utils.sh follow out
```

## Symlinks to Latest Logs

Create symlinks that point to the latest logs:

```bash
# Create symlinks
./scripts/log_utils.sh symlinks

# Now you can use:
cat logs/cluster/.latest.err    # Latest error log
tail -f logs/cluster/.latest.out  # Follow latest output log
```

**⚠️ Important**: Symlinks are **static** - they point to a specific file and **do NOT automatically update** when new logs are created. You need to run `./scripts/log_utils.sh symlinks` again after submitting new jobs to update them.

### Options for Keeping Symlinks Updated:

**Option 1: Manual update** (simplest)
```bash
# After submitting jobs, run:
./scripts/log_utils.sh symlinks
```

**Option 2: Auto-update in submission scripts** (recommended)
The `submit_staggered.sh` script now automatically updates symlinks after submission.

**Option 3: Use functions instead of symlinks** (always current)
```bash
source scripts/log_utils.sh
tail_latest err 100    # Always gets the latest, no symlink needed
follow_latest err      # Always follows the latest
```

**Option 4: Cron job** (for automatic updates)
```bash
# Update symlinks every 5 minutes
*/5 * * * * cd /scratch/gpfs/DANQIC/jz4391/bargain && ./scripts/log_utils.sh symlinks > /dev/null 2>&1
```

## Log Organization Best Practices

### 1. **Regular Cleanup**

Clean up old logs periodically:

```bash
# Dry run: see what would be deleted (logs older than 30 days)
./scripts/log_utils.sh clean 30 true

# Actually delete logs older than 30 days
./scripts/log_utils.sh clean 30 false

# Delete logs older than 7 days
./scripts/log_utils.sh clean 7 false
```

### 2. **Archive Old Logs**

Instead of deleting, archive logs to dated subdirectories:

```bash
# Archive logs older than 7 days
./scripts/log_utils.sh archive 7

# This creates: logs/cluster/archive/YYYY-MM-DD/
```

### 3. **Monitor Log Directory**

Check log directory statistics:

```bash
./scripts/log_utils.sh stats
```

Output:
```
=== Log Directory Statistics ===
Directory: logs/cluster
Total log files: 342
  - Error logs (.err): 171
  - Output logs (.out): 171
Total size: 45M

Oldest log: api_3829369_0.err
Newest log: api_3829369_267.out
```

### 4. **Automated Cleanup**

Add to your `.bashrc` or create a cron job:

```bash
# In .bashrc - update symlinks when you cd to project
cd() {
    builtin cd "$@"
    if [[ "$PWD" == *"bargain"* ]] && [ -f "scripts/log_utils.sh" ]; then
        source scripts/log_utils.sh
        update_latest_symlinks > /dev/null 2>&1
    fi
}

# Or cron job (runs daily at 2 AM)
0 2 * * * cd /scratch/gpfs/DANQIC/jz4391/bargain && ./scripts/log_utils.sh clean 30 false
```

## Log File Naming Conventions

Current naming patterns:
- **Array jobs**: `api_3829369_44.err` (name_arrayid_taskid.ext)
- **Named jobs**: `gpt5_low_vs_low_3463284.err` (name_jobid.ext)
- **Experiment jobs**: `qwen_14b_comp1_2523392.err` (experiment_jobid.ext)

### Recommendations

1. **Use descriptive names** in SBATCH scripts:
   ```bash
   #SBATCH --job-name=exp_gpt5_high_vs_low
   #SBATCH --output=logs/cluster/exp_gpt5_high_vs_low_%j.out
   #SBATCH --error=logs/cluster/exp_gpt5_high_vs_low_%j.err
   ```

2. **Include date in job names** for easier organization:
   ```bash
   #SBATCH --job-name=gpt5_$(date +%Y%m%d)_high_vs_low
   ```

3. **Group related experiments**:
   ```bash
   #SBATCH --output=logs/cluster/scaling_exp_%j.out
   #SBATCH --error=logs/cluster/scaling_exp_%j.err
   ```

## Advanced Usage

### Find logs by pattern
```bash
# Find all error logs for a specific experiment
find logs/cluster -name "*gpt5*" -name "*.err" | sort

# Find logs from today
find logs/cluster -name "*.err" -mtime -1 | sort
```

### Search logs for errors
```bash
# Search latest error log for "Error"
tail_latest err 1000 | grep -i error

# Search all recent error logs
recent_logs 10 err | xargs grep -l "RateLimit"
```

### Monitor multiple jobs
```bash
# Follow multiple latest logs
tail -f logs/cluster/.latest.err logs/cluster/.latest.out
```

## Integration with Job Submission

### Update symlinks after job submission

Add to your job submission script:

```bash
# After submitting jobs
sbatch your_script.sbatch

# Update symlinks
./scripts/log_utils.sh symlinks
```

### Auto-update in submission scripts

Modify `scripts/submit_staggered.sh` to update symlinks:

```bash
# At end of submit_staggered.sh
echo "Updating latest log symlinks..."
./scripts/log_utils.sh symlinks
```

## Troubleshooting

### "Log directory not found"
- Set `LOG_DIR` environment variable:
  ```bash
  export LOG_DIR=/path/to/logs/cluster
  source scripts/log_utils.sh
  ```

### Symlinks not updating
- Run `./scripts/log_utils.sh symlinks` manually
- Check file permissions on `logs/cluster/`

### Too many log files
- Run cleanup: `./scripts/log_utils.sh clean 7 false`
- Or archive: `./scripts/log_utils.sh archive 7`
- Consider organizing by experiment type in subdirectories

## Future Improvements

Potential enhancements:
1. **Auto-cleanup on job completion** - Add cleanup to job scripts
2. **Log rotation** - Rotate logs when they exceed size limits
3. **Compression** - Compress old logs instead of deleting
4. **Database indexing** - Index log metadata for faster searches
5. **Web dashboard** - Web interface for log browsing

## See Also

- `scripts/log_utils.sh` - Full utility script
- `docs/RATE_LIMITING_GUIDE.md` - Rate limiting and error handling
- `scripts/README_slurm.md` - SLURM job submission guide

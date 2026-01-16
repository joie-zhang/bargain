# Quick Reference: Log Management

## 🚀 Quick Commands

```bash
# Source utilities (add to .bashrc for permanent access)
source scripts/log_utils.sh

# Find latest log
latest_log err          # Latest error log
latest_log out          # Latest output log

# View recent logs
recent_logs 10          # 10 most recent

# Tail/follow logs
tail_latest err 100     # Last 100 lines of latest error log
follow_latest err       # Follow latest error log (like tail -f)

# Quick shortcuts via symlinks (after running: ./scripts/log_utils.sh symlinks)
cat logs/cluster/.latest.err      # Latest error log
tail -f logs/cluster/.latest.out  # Follow latest output log
```

## 📋 Common Tasks

### Find Latest Log
```bash
./scripts/log_utils.sh latest err
```

### Show Recent Logs
```bash
./scripts/log_utils.sh recent 20
```

### Tail Latest Error Log
```bash
./scripts/log_utils.sh tail err 100
```

### Follow Latest Output Log
```bash
./scripts/log_utils.sh follow out
```

### Clean Old Logs (30+ days)
```bash
# Dry run first
./scripts/log_utils.sh clean 30 true

# Actually delete
./scripts/log_utils.sh clean 30 false
```

### Update Symlinks
```bash
./scripts/log_utils.sh symlinks
```

## 🔗 Symlinks

After running `./scripts/log_utils.sh symlinks`, you can use:
- `logs/cluster/.latest.err` - Always points to latest error log
- `logs/cluster/.latest.out` - Always points to latest output log  
- `logs/cluster/.latest` - Always points to latest log (any type)

## 📚 Full Documentation

See `docs/LOG_MANAGEMENT_GUIDE.md` for complete guide.

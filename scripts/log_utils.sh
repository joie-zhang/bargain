#!/bin/bash
# Log management utilities for cluster logs
# Usage: source scripts/log_utils.sh

LOG_DIR="${LOG_DIR:-logs/cluster}"

# Find the most recent log file(s)
latest_log() {
    local type="${1:-any}"  # any, err, out
    
    if [ -d "$LOG_DIR" ]; then
        case "$type" in
            err)
                find "$LOG_DIR" -maxdepth 1 -type f -name "*.err" -printf '%T@ %p\n' 2>/dev/null | \
                    sort -rn | head -1 | cut -d' ' -f2-
                ;;
            out)
                find "$LOG_DIR" -maxdepth 1 -type f -name "*.out" -printf '%T@ %p\n' 2>/dev/null | \
                    sort -rn | head -1 | cut -d' ' -f2-
                ;;
            any|*)
                find "$LOG_DIR" -maxdepth 1 -type f \( -name "*.err" -o -name "*.out" \) -printf '%T@ %p\n' 2>/dev/null | \
                    sort -rn | head -1 | cut -d' ' -f2-
                ;;
        esac
    else
        echo "Error: Log directory not found: $LOG_DIR" >&2
        return 1
    fi
}

# Show the N most recent log files
recent_logs() {
    local count="${1:-10}"
    local type="${2:-any}"
    
    if [ -d "$LOG_DIR" ]; then
        echo "=== $count Most Recent Log Files ==="
        case "$type" in
            err)
                find "$LOG_DIR" -maxdepth 1 -type f -name "*.err" -printf '%T@ %TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | \
                    sort -rn | head -n "$count" | \
                    awk '{printf "%-19s %s\n", $2" "$3, $4}' | \
                    sed "s|$LOG_DIR/||"
                ;;
            out)
                find "$LOG_DIR" -maxdepth 1 -type f -name "*.out" -printf '%T@ %TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | \
                    sort -rn | head -n "$count" | \
                    awk '{printf "%-19s %s\n", $2" "$3, $4}' | \
                    sed "s|$LOG_DIR/||"
                ;;
            any|*)
                find "$LOG_DIR" -maxdepth 1 -type f \( -name "*.err" -o -name "*.out" \) -printf '%T@ %TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | \
                    sort -rn | head -n "$count" | \
                    awk '{printf "%-19s %s\n", $2" "$3, $4}' | \
                    sed "s|$LOG_DIR/||"
                ;;
        esac
    else
        echo "Error: Log directory not found: $LOG_DIR" >&2
        return 1
    fi
}

# Tail the latest log file
tail_latest() {
    local type="${1:-any}"
    local lines="${2:-50}"
    local file
    
    file=$(latest_log "$type")
    if [ -n "$file" ] && [ -f "$file" ]; then
        echo "=== Tailing latest log: $(basename "$file") ==="
        tail -n "$lines" "$file"
    else
        echo "No log file found" >&2
        return 1
    fi
}

# Follow the latest log file (like tail -f)
follow_latest() {
    local type="${1:-any}"
    local file
    
    file=$(latest_log "$type")
    if [ -n "$file" ] && [ -f "$file" ]; then
        echo "=== Following latest log: $(basename "$file") ==="
        tail -f "$file"
    else
        echo "No log file found" >&2
        return 1
    fi
}

# Count log files
log_count() {
    local type="${1:-any}"
    
    if [ -d "$LOG_DIR" ]; then
        case "$type" in
            err)
                find "$LOG_DIR" -maxdepth 1 -type f -name "*.err" 2>/dev/null | wc -l
                ;;
            out)
                find "$LOG_DIR" -maxdepth 1 -type f -name "*.out" 2>/dev/null | wc -l
                ;;
            any|*)
                find "$LOG_DIR" -maxdepth 1 -type f \( -name "*.err" -o -name "*.out" \) 2>/dev/null | wc -l
                ;;
        esac
    else
        echo "0"
    fi
}

# Show log directory statistics
log_stats() {
    if [ ! -d "$LOG_DIR" ]; then
        echo "Error: Log directory not found: $LOG_DIR" >&2
        return 1
    fi
    
    local total=$(log_count any)
    local err_count=$(log_count err)
    local out_count=$(log_count out)
    local total_size=$(du -sh "$LOG_DIR" 2>/dev/null | cut -f1)
    
    echo "=== Log Directory Statistics ==="
    echo "Directory: $LOG_DIR"
    echo "Total log files: $total"
    echo "  - Error logs (.err): $err_count"
    echo "  - Output logs (.out): $out_count"
    echo "Total size: $total_size"
    echo ""
    
    # Show oldest and newest
    local oldest=$(find "$LOG_DIR" -maxdepth 1 -type f \( -name "*.err" -o -name "*.out" \) -printf '%T@ %p\n' 2>/dev/null | sort -n | head -1 | cut -d' ' -f2-)
    local newest=$(find "$LOG_DIR" -maxdepth 1 -type f \( -name "*.err" -o -name "*.out" \) -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    
    if [ -n "$oldest" ]; then
        echo "Oldest log: $(basename "$oldest")"
    fi
    if [ -n "$newest" ]; then
        echo "Newest log: $(basename "$newest")"
    fi
}

# Clean old log files
clean_logs() {
    local days="${1:-30}"  # Default: keep logs from last 30 days
    local dry_run="${2:-true}"  # Default: dry run
    
    if [ ! -d "$LOG_DIR" ]; then
        echo "Error: Log directory not found: $LOG_DIR" >&2
        return 1
    fi
    
    echo "=== Cleaning logs older than $days days ==="
    if [ "$dry_run" = "true" ]; then
        echo "(DRY RUN - no files will be deleted)"
        find "$LOG_DIR" -maxdepth 1 -type f \( -name "*.err" -o -name "*.out" \) -mtime +$days -printf "Would delete: %p\n" 2>/dev/null | \
            sed "s|$LOG_DIR/||"
    else
        local count=$(find "$LOG_DIR" -maxdepth 1 -type f \( -name "*.err" -o -name "*.out" \) -mtime +$days -print 2>/dev/null | wc -l)
        if [ "$count" -gt 0 ]; then
            echo "Deleting $count files..."
            find "$LOG_DIR" -maxdepth 1 -type f \( -name "*.err" -o -name "*.out" \) -mtime +$days -delete 2>/dev/null
            echo "Deleted $count log files"
        else
            echo "No files to delete"
        fi
    fi
}

# Archive logs to dated subdirectories
archive_logs() {
    local days="${1:-7}"  # Archive logs older than N days
    
    if [ ! -d "$LOG_DIR" ]; then
        echo "Error: Log directory not found: $LOG_DIR" >&2
        return 1
    fi
    
    echo "=== Archiving logs older than $days days ==="
    
    find "$LOG_DIR" -maxdepth 1 -type f \( -name "*.err" -o -name "*.out" \) -mtime +$days -print0 2>/dev/null | while IFS= read -r -d '' file; do
        local date=$(stat -c %y "$file" 2>/dev/null | cut -d' ' -f1)
        local archive_dir="$LOG_DIR/archive/$date"
        
        mkdir -p "$archive_dir"
        mv "$file" "$archive_dir/"
        echo "Archived: $(basename "$file") -> archive/$date/"
    done
    
    echo "Archiving complete"
}

# Create symlinks to latest logs
update_latest_symlinks() {
    if [ ! -d "$LOG_DIR" ]; then
        echo "Error: Log directory not found: $LOG_DIR" >&2
        return 1
    fi
    
    local latest_err=$(latest_log err)
    local latest_out=$(latest_log out)
    local latest_any=$(latest_log any)
    
    if [ -n "$latest_err" ]; then
        ln -sf "$(basename "$latest_err")" "$LOG_DIR/.latest.err"
        echo "Created symlink: .latest.err -> $(basename "$latest_err")"
    fi
    
    if [ -n "$latest_out" ]; then
        ln -sf "$(basename "$latest_out")" "$LOG_DIR/.latest.out"
        echo "Created symlink: .latest.out -> $(basename "$latest_out")"
    fi
    
    if [ -n "$latest_any" ]; then
        ln -sf "$(basename "$latest_any")" "$LOG_DIR/.latest"
        echo "Created symlink: .latest -> $(basename "$latest_any")"
    fi
}

# Main command dispatcher
if [ "${BASH_SOURCE[0]}" != "${0}" ]; then
    # Sourced - functions available
    :
else
    # Executed directly - run command
    case "${1:-help}" in
        latest)
            latest_log "${2:-any}"
            ;;
        recent)
            recent_logs "${2:-10}" "${3:-any}"
            ;;
        tail)
            tail_latest "${2:-any}" "${3:-50}"
            ;;
        follow|f)
            follow_latest "${2:-any}"
            ;;
        count)
            log_count "${2:-any}"
            ;;
        stats)
            log_stats
            ;;
        clean)
            clean_logs "${2:-30}" "${3:-true}"
            ;;
        archive)
            archive_logs "${2:-7}"
            ;;
        symlinks|update)
            update_latest_symlinks
            ;;
        help|*)
            echo "Log Management Utilities"
            echo ""
            echo "Usage: $0 <command> [args...]"
            echo ""
            echo "Commands:"
            echo "  latest [err|out|any]     - Show path to latest log file"
            echo "  recent [N] [err|out|any] - Show N most recent logs (default: 10)"
            echo "  tail [err|out|any] [N]   - Tail latest log (default: 50 lines)"
            echo "  follow [err|out|any]    - Follow latest log (like tail -f)"
            echo "  count [err|out|any]      - Count log files"
            echo "  stats                    - Show log directory statistics"
            echo "  clean [days] [true|false]- Clean old logs (default: 30 days, dry-run)"
            echo "  archive [days]           - Archive old logs (default: 7 days)"
            echo "  symlinks                 - Create .latest symlinks"
            echo ""
            echo "Examples:"
            echo "  $0 latest err             # Latest error log"
            echo "  $0 recent 20             # 20 most recent logs"
            echo "  $0 tail err 100          # Tail latest error log (100 lines)"
            echo "  $0 follow out            # Follow latest output log"
            echo "  $0 clean 30 false        # Delete logs older than 30 days"
            echo ""
            echo "Or source this file to use functions directly:"
            echo "  source scripts/log_utils.sh"
            echo "  latest_log err"
            echo "  tail_latest out 100"
            ;;
    esac
fi

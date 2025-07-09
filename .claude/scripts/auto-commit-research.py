#!/usr/bin/env python3
"""
Auto-commit hook for research sessions
Creates a git commit when Claude Code stops, preserving research work
"""

import json
import sys
import subprocess
import os
from datetime import datetime
from pathlib import Path

def get_git_status():
    """Check if there are changes to commit"""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()

def extract_summary(transcript_path):
    """Extract summary from transcript if available"""
    try:
        with open(transcript_path, 'r') as f:
            first_line = f.readline()
            data = json.loads(first_line)
            return data.get('summary', '')
    except:
        return ''

def main():
    try:
        # Read input from Claude Code
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(1)
    
    # Check if we're in a git repository
    git_check = subprocess.run(
        ["git", "rev-parse", "--git-dir"],
        capture_output=True
    )
    if git_check.returncode != 0:
        # Not a git repo, exit quietly
        sys.exit(0)
    
    # Check if there are changes
    if not get_git_status():
        # No changes to commit
        sys.exit(0)
    
    # Extract session info
    session_id = input_data.get("session_id", "unknown")
    transcript_path = input_data.get("transcript_path", "")
    
    # Build commit message
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commit_message = f"Research session: {timestamp}"
    
    # Try to get summary from transcript
    if transcript_path and os.path.exists(transcript_path):
        summary = extract_summary(transcript_path)
        if summary:
            commit_message += f"\n\nSummary: {summary}"
    
    commit_message += f"\n\nSession ID: {session_id}"
    commit_message += "\n\n🤖 Auto-committed by Claude Code research hook"
    
    # Stage all changes
    subprocess.run(["git", "add", "-A"], capture_output=True)
    
    # Create commit
    result = subprocess.run(
        ["git", "commit", "-m", commit_message],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        # Get short commit hash
        commit_hash = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True
        ).stdout.strip()
        
        print(f"Research session saved: commit {commit_hash}")
    
    sys.exit(0)

if __name__ == "__main__":
    main()
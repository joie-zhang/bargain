#!/usr/bin/env python3
"""
Auto-commit hook for research sessions
Uses conversation transcript to generate intelligent commit messages
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
        ["git", "status", "--porcelain"], capture_output=True, text=True
    )
    return result.stdout.strip()


def get_git_diff():
    """Get the staged diff for analysis"""
    result = subprocess.run(["git", "diff", "--cached"], capture_output=True, text=True)
    return result.stdout


def get_recent_commit_messages():
    """Get recent commit messages for style reference"""
    result = subprocess.run(
        ["git", "log", "--oneline", "-5"], capture_output=True, text=True
    )
    return result.stdout


def parse_transcript_for_changes(transcript_path):
    """Parse conversation transcript to understand what was done"""
    if not transcript_path or not os.path.exists(transcript_path):
        return None

    try:
        changes_summary = []
        with open(transcript_path, "r") as f:
            for line in f:
                try:
                    msg = json.loads(line.strip())
                    if msg.get("type") == "tool_use":
                        tool_name = msg.get("name", "")
                        if tool_name == "Write":
                            file_path = msg.get("input", {}).get("file_path", "")
                            if file_path:
                                changes_summary.append(f"Created/wrote {file_path}")
                        elif tool_name == "Edit":
                            file_path = msg.get("input", {}).get("file_path", "")
                            if file_path:
                                changes_summary.append(f"Modified {file_path}")
                        elif tool_name == "MultiEdit":
                            file_path = msg.get("input", {}).get("file_path", "")
                            if file_path:
                                changes_summary.append(f"Updated {file_path}")
                    elif msg.get("type") == "text" and msg.get("role") == "user":
                        # Look for key user requests
                        content = msg.get("content", "").lower()
                        if "fix" in content and "commit" in content:
                            changes_summary.append("Fixed commit message generation")
                        elif "create" in content and "guide" in content:
                            changes_summary.append("Created comprehensive guide")
                        elif "enhance" in content or "improve" in content:
                            changes_summary.append("Enhanced existing functionality")
                except (json.JSONDecodeError, KeyError):
                    continue

        return changes_summary
    except Exception:
        return None


def generate_intelligent_commit_message(diff_content, changes_summary):
    """Generate commit message based on changes and context"""

    # Analyze the diff for patterns
    lines = diff_content.split("\n")
    new_files = []
    modified_files = []

    for line in lines:
        if line.startswith("+++") and not line.endswith("/dev/null"):
            file_path = line[6:]  # Remove '+++ b/'
            if any(
                prev_line.startswith("---") and "/dev/null" in prev_line
                for prev_line in lines
            ):
                new_files.append(file_path)
            else:
                modified_files.append(file_path)

    # Generate commit message based on changes
    if new_files:
        primary_file = new_files[0]

        # Check for specific patterns
        if (
            "PROMPT_ENGINEERING_GUIDE" in primary_file
            or "prompt" in primary_file.lower()
        ):
            return "docs: Add comprehensive Claude prompt engineering guide\n\nDetailed reference covering all Anthropic best practices, techniques, and optimization strategies for Claude models. Includes model selection, examples, XML structure, extended thinking, and common patterns."

        elif primary_file.endswith(".py") and "auto-commit" in primary_file:
            return "feat: Enhance auto-commit with intelligent message generation\n\nReplace generic commit messages with context-aware generation using conversation transcript analysis and git diff parsing."

        elif primary_file.endswith(".md"):
            return f"docs: Add {primary_file}\n\nNew documentation for research workflows and best practices."

        elif primary_file.endswith(".py"):
            return f"feat: Add {primary_file}\n\nNew implementation with enhanced functionality."

    if modified_files:
        primary_file = modified_files[0]

        if "auto-commit" in primary_file and "claude" in diff_content.lower():
            return "fix: Update auto-commit to use Claude Code conversation context\n\nReplace external claude command calls with transcript-based commit message generation for better accuracy and context awareness."

        elif primary_file.endswith(".py"):
            return f"refactor: Update {primary_file}\n\nImproved implementation and bug fixes."

    # Fallback based on changes summary
    if changes_summary:
        if any("guide" in change.lower() for change in changes_summary):
            return "docs: Add comprehensive documentation\n\nNew guides and reference materials for improved workflows."
        elif any("fix" in change.lower() for change in changes_summary):
            return "fix: Resolve issues and improve functionality\n\nBug fixes and enhancements based on user feedback."
        elif any("enhance" in change.lower() for change in changes_summary):
            return "feat: Enhance existing functionality\n\nImproved features and capabilities."

    return None


def extract_summary(transcript_path):
    """Extract summary from transcript if available"""
    try:
        with open(transcript_path, "r") as f:
            first_line = f.readline()
            data = json.loads(first_line)
            return data.get("summary", "")
    except:
        return ""


def main():
    try:
        # Read input from Claude Code
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(1)

    # Check if we're in a git repository
    git_check = subprocess.run(["git", "rev-parse", "--git-dir"], capture_output=True)
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

    # Stage all changes first
    subprocess.run(["git", "add", "-A"], capture_output=True)

    # Get diff and analyze conversation
    diff_content = get_git_diff()
    changes_summary = parse_transcript_for_changes(transcript_path)

    # Try to generate intelligent commit message
    commit_message = None
    if diff_content.strip():
        commit_message = generate_intelligent_commit_message(
            diff_content, changes_summary
        )

    # Fallback to generic message if generation fails
    if not commit_message:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        commit_message = f"Research session: {timestamp}"

        # Try to get summary from transcript
        if transcript_path and os.path.exists(transcript_path):
            summary = extract_summary(transcript_path)
            if summary:
                commit_message += f"\n\nSummary: {summary}"

        commit_message += f"\n\nSession ID: {session_id}"
        commit_message += "\n\n🤖 Auto-committed by Claude Code research hook"
    else:
        # Add Claude attribution to intelligent messages
        commit_message += f"\n\n🤖 Generated with Claude Code\n\nCo-Authored-By: Claude <noreply@anthropic.com>"

    # Create commit
    result = subprocess.run(
        ["git", "commit", "-m", commit_message], capture_output=True, text=True
    )

    if result.returncode == 0:
        # Get short commit hash
        commit_hash = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip()

        print(f"Research session saved: commit {commit_hash}")

    sys.exit(0)


if __name__ == "__main__":
    main()

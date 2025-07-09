#!/usr/bin/env python3
"""
Auto-commit hook for research sessions
Uses Claude to generate intelligent commit messages based on actual changes
"""

import json
import sys
import subprocess
import os
import tempfile
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


def get_git_status_summary():
    """Get a summary of changed files"""
    result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True
    )
    return result.stdout


def generate_commit_message_with_claude(diff_content, status_summary):
    """Use Claude to generate an intelligent commit message"""

    # Create a prompt for Claude to analyze the changes
    prompt = f"""You are a skilled developer creating a git commit message. Analyze the following git diff and status to create a clear, informative commit message.

Follow these guidelines:
- Use conventional commit format (feat:, fix:, docs:, refactor:, etc.)
- Be specific about what was changed/added/fixed
- Keep the first line under 50 characters when possible
- Add a detailed description if the changes are significant
- Focus on the "why" and "what" rather than just "how"

Git Status:
{status_summary}

Git Diff:
{diff_content}

Generate a commit message that accurately describes these changes:"""

    try:
        # Write prompt to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(prompt)
            prompt_file = f.name

        # Use claude command to generate commit message
        result = subprocess.run(
            ["claude", "--file", prompt_file, "--quiet"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Clean up temp file
        os.unlink(prompt_file)

        if result.returncode == 0:
            commit_message = result.stdout.strip()

            # Clean up the message (remove any markdown formatting)
            commit_message = commit_message.replace("```", "").strip()

            # Add Claude attribution
            commit_message += f"\n\n🤖 Generated with Claude Code\n\nCo-Authored-By: Claude <noreply@anthropic.com>"

            return commit_message
        else:
            print(f"Claude command failed: {result.stderr}", file=sys.stderr)
            return None

    except subprocess.TimeoutExpired:
        print("Claude command timed out", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error calling Claude: {e}", file=sys.stderr)
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

    # Get diff and status for Claude analysis
    diff_content = get_git_diff()
    status_summary = get_git_status_summary()

    # Try to generate intelligent commit message with Claude
    commit_message = None
    if diff_content.strip():  # Only if there are actual changes
        commit_message = generate_commit_message_with_claude(
            diff_content, status_summary
        )

    # Fallback to generic message if Claude fails
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

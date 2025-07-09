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
        ["git", "status", "--porcelain"], capture_output=True, text=True
    )
    return result.stdout.strip()


def extract_summary(transcript_path):
    """Extract summary from transcript if available"""
    try:
        with open(transcript_path, "r") as f:
            first_line = f.readline()
            data = json.loads(first_line)
            return data.get("summary", "")
    except:
        return ""


def analyze_changes():
    """Analyze git changes to create intelligent commit messages"""
    try:
        # Get status
        status_result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True
        )

        # Get diff stats
        diff_result = subprocess.run(
            ["git", "diff", "--cached", "--stat"], capture_output=True, text=True
        )

        # Get actual diff for content analysis
        diff_content = subprocess.run(
            ["git", "diff", "--cached"], capture_output=True, text=True
        ).stdout

        status_lines = (
            status_result.stdout.strip().split("\n")
            if status_result.stdout.strip()
            else []
        )

        new_files = []
        modified_files = []
        deleted_files = []

        for line in status_lines:
            if line.startswith("A  "):
                new_files.append(line[3:])
            elif line.startswith("M  "):
                modified_files.append(line[3:])
            elif line.startswith("D  "):
                deleted_files.append(line[3:])

        # Analyze file types and generate appropriate message
        if new_files:
            # Check if it's documentation
            if any(".md" in f for f in new_files):
                if any("guide" in f.lower() or "doc" in f.lower() for f in new_files):
                    return generate_docs_commit_message(new_files, diff_content)

            # Check if it's code
            if any(
                f.endswith((".py", ".js", ".ts", ".java", ".cpp", ".c"))
                for f in new_files
            ):
                return generate_code_commit_message(new_files, diff_content, "new")

            # Check if it's configuration
            if any(
                f.endswith((".json", ".yaml", ".yml", ".toml", ".cfg"))
                for f in new_files
            ):
                return generate_config_commit_message(new_files, diff_content)

        if modified_files:
            if any(
                f.endswith((".py", ".js", ".ts", ".java", ".cpp", ".c"))
                for f in modified_files
            ):
                return generate_code_commit_message(
                    modified_files, diff_content, "modified"
                )

        # Fallback to generic message
        return None

    except Exception:
        return None


def generate_docs_commit_message(files, diff_content):
    """Generate commit message for documentation changes"""
    primary_file = files[0]

    # Check content for key terms
    if "prompt" in diff_content.lower() and "engineering" in diff_content.lower():
        return "docs: Add comprehensive prompt engineering guide\n\nDetailed reference covering Claude best practices, techniques, and optimization strategies"
    elif "guide" in primary_file.lower():
        return f"docs: Add {primary_file} guide\n\nNew documentation for research workflows"
    elif "readme" in primary_file.lower():
        return "docs: Add project README\n\nProject overview and setup instructions"
    else:
        return f"docs: Add {primary_file}\n\nNew documentation added"


def generate_code_commit_message(files, diff_content, change_type):
    """Generate commit message for code changes"""
    if change_type == "new":
        # Check for patterns in new code
        if "def " in diff_content and "class " in diff_content:
            return f"feat: Add {files[0]}\n\nNew module with classes and functions"
        elif "def " in diff_content:
            return f"feat: Add {files[0]}\n\nNew utility functions"
        elif "import" in diff_content:
            return f"feat: Add {files[0]}\n\nNew implementation"
        else:
            return f"feat: Add {files[0]}"
    else:
        # Modified files
        if "fix" in diff_content.lower() or "bug" in diff_content.lower():
            return f"fix: Update {files[0]}\n\nBug fixes and improvements"
        elif "refactor" in diff_content.lower():
            return f"refactor: Update {files[0]}\n\nCode refactoring"
        else:
            return f"feat: Update {files[0]}\n\nFeature improvements"


def generate_config_commit_message(files, diff_content):
    """Generate commit message for configuration changes"""
    primary_file = files[0]

    if "package.json" in primary_file:
        return "chore: Update package.json\n\nDependency and configuration changes"
    elif primary_file.endswith(".yml") or primary_file.endswith(".yaml"):
        return "config: Update YAML configuration\n\nConfiguration file changes"
    else:
        return f"config: Update {primary_file}\n\nConfiguration changes"


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

    # Stage all changes first for diff analysis
    subprocess.run(["git", "add", "-A"], capture_output=True)

    # Try to generate intelligent commit message
    intelligent_message = analyze_changes()

    if intelligent_message:
        commit_message = intelligent_message
        # Add research session metadata
        commit_message += f"\n\n🤖 Generated with Claude Code\n\nCo-Authored-By: Claude <noreply@anthropic.com>"
    else:
        # Fallback to generic message
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

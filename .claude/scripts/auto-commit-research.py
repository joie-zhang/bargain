#!/usr/bin/env python3
"""
Auto-commit hook for research sessions
Automatically commits experiment results and research progress
"""

import json
import sys
import subprocess
import os
import re
import yaml
from datetime import datetime
from pathlib import Path


def get_git_status():
    """Check if there are changes to commit"""
    result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True
    )
    return result.stdout.strip()


def should_auto_commit_file(file_path):
    """Determine if file should be auto-committed"""
    auto_commit_patterns = [
        'results/', 'experiments/', 'outputs/', 'analysis/', 'logs/',
        'specs/', 'tasks/', 'issues/', 'docs/',
        '.csv', '.json', '.md', '.txt', '.log', '.py',
        'experiment_', 'result_', 'analysis_', 'output_', 'spec_', 'task_'
    ]
    
    return any(pattern in file_path.lower() for pattern in auto_commit_patterns)


def get_changed_files():
    """Get list of changed files that should be auto-committed"""
    result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True
    )
    
    changed_files = []
    for line in result.stdout.strip().split('\n'):
        if line.strip():
            # Parse git status format: "XY filename"
            status = line[:2]
            filename = line[3:]
            
            # Include modified, added, and untracked files
            if status.strip() and should_auto_commit_file(filename):
                changed_files.append(filename)
    
    return changed_files


def get_file_size(file_path):
    """Get human-readable file size"""
    try:
        size = os.path.getsize(file_path)
        if size < 1024:
            return f"{size}B"
        elif size < 1024 * 1024:
            return f"{size/1024:.1f}KB"
        else:
            return f"{size/(1024*1024):.1f}MB"
    except OSError:
        return None


def is_experiment_session(transcript_path, changes_summary):
    """Check if this was an experiment/research session"""
    experiment_indicators = [
        'experiment', 'analysis', 'research', 'test', 'evaluate',
        'generate', 'create', 'spec', 'task', 'issue', 'results'
    ]
    
    # Check conversation context
    if changes_summary:
        context_text = ' '.join(changes_summary).lower()
        if any(indicator in context_text for indicator in experiment_indicators):
            return True
    
    # Check transcript if available
    if transcript_path and os.path.exists(transcript_path):
        try:
            with open(transcript_path, "r") as f:
                content = f.read().lower()
                if any(indicator in content for indicator in experiment_indicators):
                    return True
        except Exception:
            pass
    
    return False


def parse_transcript_for_changes(transcript_path):
    """Parse conversation transcript to understand what was done"""
    if not transcript_path or not os.path.exists(transcript_path):
        return []

    try:
        changes_summary = []
        with open(transcript_path, "r") as f:
            for line in f:
                try:
                    msg = json.loads(line.strip())
                    if msg.get("type") == "tool_use":
                        tool_name = msg.get("name", "")
                        if tool_name in ["edit_file", "Write", "Edit"]:
                            file_path = msg.get("input", {}).get("file_path", "") or msg.get("input", {}).get("target_file", "")
                            if file_path:
                                changes_summary.append(f"Modified {file_path}")
                        elif tool_name == "run_terminal_cmd":
                            command = msg.get("input", {}).get("command", "")
                            if any(cmd in command.lower() for cmd in ["python", "experiment", "analysis", "test"]):
                                changes_summary.append(f"Executed: {command[:50]}...")
                    elif msg.get("type") == "text" and msg.get("role") == "user":
                        content = msg.get("content", "").lower()
                        if any(keyword in content for keyword in ["create", "generate", "analyze", "experiment", "research"]):
                            changes_summary.append("Research/experiment session")
                except (json.JSONDecodeError, KeyError):
                    continue

        return changes_summary
    except Exception:
        return []


def load_config():
    """Load commit configuration from yaml file"""
    config_path = Path.cwd() / ".claude" / "commit-config.yaml"
    default_config = {
        'file_patterns': {
            'experiments': ['experiments/', 'experiment_'],
            'results': ['results/', 'result_', '.json'],
            'analysis': ['analysis', '.ipynb'],
            'tests': ['test_', '/tests/'],
            'docs': ['.md', 'docs/']
        },
        'research_indicators': ['experiment', 'analysis', 'negotiation', 'agent'],
        'max_files_to_analyze': 5,
        'features': {'detailed_descriptions': True}
    }
    
    try:
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or default_config
    except Exception:
        pass
    return default_config


def analyze_file_content(file_path, config, max_chars=1000):
    """Analyze file content to understand what was done"""
    try:
        if not os.path.exists(file_path) or os.path.getsize(file_path) > 1024 * 1024:
            return None
            
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(max_chars)
            
        filename = os.path.basename(file_path).lower()
        
        # Look for research indicators from config
        research_terms = []
        for term in config.get('research_indicators', []):
            if term.lower() in content.lower():
                research_terms.append(term)
        
        # Analyze content for specific patterns
        if filename.endswith('.json'):
            if any(field in content for field in ['final_utilities', 'winner_agent', 'consensus_reached']):
                return f"negotiation results ({', '.join(research_terms[:2]) if research_terms else 'data'})"
            elif 'experiment' in content.lower():
                return f"experiment config ({', '.join(research_terms[:2]) if research_terms else 'setup'})"
        elif filename.endswith('.py'):
            if 'class ' in content and 'def ' in content:
                if research_terms:
                    return f"{research_terms[0]} implementation"
                return "code implementation"
            elif 'def test_' in content:
                return "test implementation"
        elif filename.endswith('.md'):
            lines = content.split('\n')
            first_header = next((line.replace('#', '').strip() for line in lines[:5] if line.startswith('#')), None)
            if first_header:
                return f"docs: {first_header[:50]}"
        
        # Generic analysis based on research terms found
        if research_terms:
            return f"{research_terms[0]} work"
            
        return None
        
    except Exception:
        return None


def classify_files_by_config(changed_files, config):
    """Classify files based on config patterns"""
    classifications = {}
    file_patterns = config.get('file_patterns', {})
    
    for category, patterns in file_patterns.items():
        matching_files = []
        for file_path in changed_files:
            file_lower = file_path.lower()
            if any(pattern.lower() in file_lower for pattern in patterns):
                matching_files.append(file_path)
        if matching_files:
            classifications[category] = matching_files
    
    return classifications


def generate_intelligent_commit_message(changed_files, changes_summary):
    """Generate detailed commit message in Claude Code style"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    if not changed_files:
        return None
    
    # Load configuration
    config = load_config()
    max_analyze = config.get('max_files_to_analyze', 8)
    
    # Classify files by config
    file_classifications = classify_files_by_config(changed_files, config)
    
    # Analyze file contents for detailed context
    file_details = []
    content_contexts = []
    
    for file_path in changed_files[:max_analyze]:
        context = analyze_file_content(file_path, config)
        if context:
            content_contexts.append(context)
            file_details.append({
                'path': file_path,
                'name': os.path.basename(file_path),
                'context': context,
                'size': get_file_size(file_path)
            })
    
    # Determine primary activity and generate detailed title
    if file_classifications.get('results'):
        commit_type = "feat"
        if any('negotiation results' in ctx for ctx in content_contexts):
            title = f"feat: add negotiation experiment results"
        elif any('experiment config' in ctx for ctx in content_contexts):
            title = f"feat: add experiment configuration and results"
        else:
            title = f"feat: add experiment results ({len(file_classifications['results'])} files)"
        primary_category = "results"
    elif file_classifications.get('experiments'):
        commit_type = "feat"
        if len(changed_files) == 1:
            title = f"feat: add {file_details[0]['context'] if file_details else 'experiment setup'}"
        else:
            title = f"feat: add experiment setup ({len(file_classifications['experiments'])} files)"
        primary_category = "experiments"
    elif file_classifications.get('analysis'):
        commit_type = "feat"
        title = f"feat: add analysis implementation ({len(file_classifications['analysis'])} files)"
        primary_category = "analysis"
    elif file_classifications.get('tests'):
        commit_type = "test"
        title = f"test: add test implementation ({len(file_classifications['tests'])} files)"
        primary_category = "tests"
    elif file_classifications.get('docs'):
        commit_type = "docs"
        if len(changed_files) == 1 and file_details:
            title = f"docs: {file_details[0]['context']}"
        else:
            title = f"docs: add documentation ({len(file_classifications['docs'])} files)"
        primary_category = "documentation"
    else:
        commit_type = "feat"
        if content_contexts and len(content_contexts) == 1:
            title = f"feat: add {content_contexts[0]}"
        else:
            title = f"feat: add implementation ({len(changed_files)} files)"
        primary_category = "implementation"
    
    # Build detailed description in Claude Code style
    description_parts = []
    
    # Summary of what was accomplished
    if content_contexts:
        unique_contexts = list(dict.fromkeys(content_contexts))
        if len(unique_contexts) == 1:
            description_parts.append(f"Added {unique_contexts[0]} to support multi-agent negotiation research.")
        else:
            description_parts.append("Research session completed with multiple components:")
            for i, ctx in enumerate(unique_contexts[:5], 1):
                description_parts.append(f"- {ctx}")
    
    # File breakdown by category
    if len(file_classifications) > 1:
        description_parts.append("")
        description_parts.append("Files by category:")
        for category, files in sorted(file_classifications.items()):
            file_list = [os.path.basename(f) for f in files[:3]]
            if len(files) > 3:
                file_list.append(f"... and {len(files)-3} more")
            description_parts.append(f"- {category.title()}: {', '.join(file_list)}")
    
    # Detailed file information for small commits
    if len(changed_files) <= 3 and file_details:
        description_parts.append("")
        description_parts.append("Modified files:")
        for detail in file_details:
            size_info = f" ({detail['size']})" if detail['size'] else ""
            description_parts.append(f"- {detail['name']}: {detail['context']}{size_info}")
    
    # Session metadata
    description_parts.append("")
    description_parts.append(f"Session completed: {timestamp}")
    if changes_summary:
        relevant_changes = [change for change in changes_summary[:3] if change.strip()]
        if relevant_changes:
            description_parts.append(f"Context: {'; '.join(relevant_changes)}")
    
    description = "\n".join(description_parts)
    
    # Log for debugging
    log_to_research(f"Generated detailed commit: {title[:50]}... | Files: {len(changed_files)} | Categories: {list(file_classifications.keys())}")
    
    return f"{title}\n\n{description}\n\n🤖 Generated with [Claude Code](https://claude.ai/code)\n\nCo-Authored-By: Claude <noreply@anthropic.com>"


def log_to_research(message):
    """Log message to research monitoring system"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"[{timestamp}] Research Auto-Commit: {message}"
    
    # Log to file
    log_dir = Path.cwd() / ".claude" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "auto-commit.log"
    
    with open(log_file, 'a') as f:
        f.write(log_message + '\n')


def main():
    try:
        # Read input from Claude Code
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(1)

    # Check if we're in a git repository
    git_check = subprocess.run(["git", "rev-parse", "--git-dir"], capture_output=True)
    if git_check.returncode != 0:
        sys.exit(0)

    # Extract session info
    session_id = input_data.get("session_id", "unknown")
    transcript_path = input_data.get("transcript_path", "")

    # Get conversation context
    changes_summary = parse_transcript_for_changes(transcript_path)
    
    # Check if this was an experiment/research session
    if not is_experiment_session(transcript_path, changes_summary):
        log_to_research("Session not identified as experiment/research - skipping auto-commit")
        sys.exit(0)

    # Get files that should be auto-committed
    changed_files = get_changed_files()
    
    if not changed_files:
        log_to_research("No eligible files found for auto-commit")
        sys.exit(0)

    # Stage only the files we want to commit
    for file_path in changed_files:
        try:
            subprocess.run(["git", "add", file_path], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            log_to_research(f"Failed to stage {file_path}")

    # Generate intelligent commit message
    commit_message = generate_intelligent_commit_message(changed_files, changes_summary)
    
    if not commit_message:
        log_to_research("Failed to generate commit message")
        sys.exit(0)

    # Create commit
    try:
        result = subprocess.run(
            ["git", "commit", "-m", commit_message], 
            capture_output=True, text=True, check=True
        )

        # Get commit hash
        commit_hash = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], 
            capture_output=True, text=True
        ).stdout.strip()

        log_to_research(f"Successfully committed {len(changed_files)} files: {commit_hash}")
        print(f"📝 Research session auto-committed: {commit_hash}")
        print(f"Files: {', '.join([os.path.basename(f) for f in changed_files[:3]])}")

    except subprocess.CalledProcessError as e:
        log_to_research(f"Commit failed: {e}")

    sys.exit(0)


if __name__ == "__main__":
    main()

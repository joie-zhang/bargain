# Claude Code Comprehensive Guide: From Basics to Intermediate

## Table of Contents
1. [What is Claude Code?](#what-is-claude-code)
2. [Installation and Setup](#installation-and-setup)
3. [Basic Usage](#basic-usage)
4. [Understanding Your Project](#understanding-your-project)
5. [Writing and Editing Code](#writing-and-editing-code)
6. [Working with Git](#working-with-git)
7. [Configuration and Settings](#configuration-and-settings)
8. [Slash Commands](#slash-commands)
9. [Custom Commands](#custom-commands)
10. [Hooks for Automation](#hooks-for-automation)
11. [Intermediate Workflows](#intermediate-workflows)
12. [Best Practices](#best-practices)
13. [Troubleshooting](#troubleshooting)

## What is Claude Code?

Claude Code is an AI-powered coding assistant that lives in your terminal. It's designed to help developers:

- **Build features faster**: Describe what you want in plain English, and Claude will plan, write, and test the code
- **Debug efficiently**: Paste error messages or describe bugs, and Claude will analyze and fix them
- **Navigate codebases**: Ask questions about your code and get thoughtful, context-aware answers
- **Automate tedious tasks**: Handle lint issues, merge conflicts, and generate release notes

### Key Advantages
- **Terminal-based**: Works directly in your existing development environment
- **Action-oriented**: Can edit files, run commands, and create commits
- **Scriptable**: Follows Unix philosophy for complex workflows
- **Enterprise-ready**: Robust security, privacy, and compliance features

## Installation and Setup

### Prerequisites
- Terminal or command prompt
- Node.js 18 or newer

### Installation Steps

```bash
# Install Claude Code globally
npm install -g @anthropic-ai/claude-code

# Verify installation
claude --version

# Navigate to your project
cd /path/to/your/project

# Start Claude Code
claude
```

### First-Time Setup
When you first run `claude`, it will:
1. Open a browser for authentication
2. Create necessary configuration directories
3. Initialize with default settings

## Basic Usage

### Starting a Session

```bash
# Interactive mode (recommended for beginners)
claude

# One-time task mode
claude "fix the failing test in user_service.py"

# With specific model
claude --model claude-3-5-sonnet-latest
```

### Essential Commands in Interactive Mode

| Command | Purpose | Example |
|---------|---------|---------|
| `/help` | Show available commands | `> /help` |
| `/clear` | Clear conversation history | `> /clear` |
| `/cost` | Show token usage | `> /cost` |
| `/status` | View system status | `> /status` |
| `exit` | Exit Claude Code | `> exit` |

## Understanding Your Project

### Initial Project Exploration
When starting with a new project, use these prompts:

```
# Project overview
> what does this project do?
> what technologies does this project use?
> explain the folder structure

# Finding key files
> where is the main entry point?
> where are the configuration files?
> show me the database models

# Understanding architecture
> how is the authentication implemented?
> what's the API structure?
> how are errors handled?
```

### Navigating Large Codebases

```
# Search for specific functionality
> where is the user registration logic?
> find all API endpoints
> show me files related to payment processing

# Understanding dependencies
> what external libraries does this use?
> which files import the database module?
> what are the main dependencies?
```

## Writing and Editing Code

### Creating New Features

```
# Simple feature addition
> add a new endpoint for user profiles at /api/users/profile

# Complex feature with planning
> I need to add email notifications when users register. 
  Plan the implementation and then build it.

# Refactoring
> refactor the authentication module to use dependency injection
```

### Code Editing Best Practices

1. **Be specific about requirements**:
   ```
   > add input validation to the create_user function. 
     It should check email format and password strength
   ```

2. **Review changes before approval**:
   - Claude will show you the proposed changes
   - Type 'yes' to approve, 'no' to cancel
   - You can ask for modifications before approving

3. **Incremental changes**:
   ```
   > first, add the function signature
   > now implement the validation logic
   > finally, add error handling
   ```

### Debugging

```
# With error messages
> I'm getting this error: [paste error]
  Can you fix it?

# Describing issues
> the login function returns undefined sometimes. 
  Can you investigate and fix?

# Performance issues
> this query is running slowly. Can you optimize it?
```

## Working with Git

### Basic Git Operations

```
# Check status
> what files have I changed?
> show me the diff

# Committing
> commit my changes with a descriptive message
> stage only the Python files and commit

# Branching
> create a new branch called feature/user-profiles
> switch to the main branch
```

### Advanced Git Workflows

```
# Interactive staging
> stage the changes in user_service.py but not the tests

# Commit with specific format
> commit using conventional commits format

# Working with remotes
> push my changes to origin
> create a pull request
```

## Configuration and Settings

### Configuration Hierarchy
1. **User settings**: `~/.claude/settings.json` (all projects)
2. **Project settings**: `.claude/settings.json` (team-shared)
3. **Local settings**: `.claude/settings.local.json` (personal)

### Common Configuration Tasks

```bash
# View current settings
claude config list

# Set a specific value
claude config set theme dark

# Global settings
claude config set -g autoUpdates false
```

### Example Settings File

```json
{
  "theme": "dark",
  "verbose": true,
  "permissions": {
    "allow": [
      "Bash(npm test)",
      "Bash(npm run lint)",
      "Read(~/.gitconfig)"
    ],
    "deny": [
      "Bash(rm -rf *)",
      "Write(/etc/*)"
    ]
  },
  "env": {
    "NODE_ENV": "development",
    "DEBUG": "true"
  }
}
```

### Permission Management

```
# View current permissions
> /permissions

# In settings.json
{
  "permissions": {
    "allow": [
      "Bash(npm *)",           // Allow all npm commands
      "Read(~/documents/*)",   // Allow reading from documents
      "Write(./src/**)"        // Allow writing to src directory
    ],
    "deny": [
      "Bash(sudo *)",          // Deny all sudo commands
      "Write(/etc/*)",         // Deny writing to system files
      "Read(~/.ssh/*)"         // Deny reading SSH keys
    ]
  }
}
```

## Slash Commands

### Built-in Commands Reference

| Command | Purpose | Usage |
|---------|---------|-------|
| `/add-dir` | Add working directories | `/add-dir ../shared-lib` |
| `/bug` | Report bugs to Anthropic | `/bug` |
| `/clear` | Clear conversation | `/clear` |
| `/compact` | Compress conversation | `/compact focus on API changes` |
| `/config` | Manage configuration | `/config set theme dark` |
| `/cost` | View token usage | `/cost` |
| `/doctor` | Health check | `/doctor` |
| `/init` | Initialize project guide | `/init` |
| `/memory` | Edit project context | `/memory` |
| `/model` | Change AI model | `/model claude-3-5-sonnet` |
| `/review` | Request code review | `/review` |

### Using Slash Commands Effectively

```
# Compress conversation when context is full
> /compact keep information about the database schema

# Switch models for different tasks
> /model claude-3-5-haiku  # For simple tasks
> /model claude-3-5-sonnet  # For complex reasoning

# Add context from other directories
> /add-dir ../shared-components
> now update the imports to use the shared Button component
```

## Custom Commands

### Creating Project Commands

1. **Create command directory**:
   ```bash
   mkdir -p .claude/commands
   ```

2. **Create a simple command**:
   ```bash
   echo "Run all tests and provide a summary" > .claude/commands/test-all.md
   ```

3. **Use the command**:
   ```
   > /test-all
   ```

### Advanced Custom Commands

**With arguments** (`.claude/commands/create-component.md`):
```markdown
Create a new React component named $ARGUMENTS with:
- TypeScript interface for props
- Basic styling
- Unit test file
- Storybook story
```

Usage:
```
> /create-component UserCard
```

**With multiple steps** (`.claude/commands/deploy-check.md`):
```markdown
Perform pre-deployment checks:
1. Run all tests
2. Check for console.log statements
3. Verify environment variables
4. Run build process
5. Check bundle size

Report any issues found.
```

### Personal vs Project Commands

- **Project commands** (`.claude/commands/`): Shared with team via Git
- **Personal commands** (`~/.claude/commands/`): Available in all projects

## Hooks for Automation

### What are Hooks?
Hooks are scripts that run automatically at specific events, allowing you to:
- Validate changes before they're made
- Format code automatically
- Add custom checks
- Integrate with external tools

### Basic Hook Configuration

**Example: Auto-format on file write**

In `.claude/settings.json`:
```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write(*.(js|ts|jsx|tsx))",
        "hooks": [
          {
            "type": "command",
            "command": "npx prettier --write"
          }
        ]
      }
    ]
  }
}
```

### Common Hook Patterns

**1. Pre-commit validation**:
```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash(git commit*)",
        "hooks": [
          {
            "type": "command",
            "command": "./scripts/pre-commit-checks.sh"
          }
        ]
      }
    ]
  }
}
```

**2. Security scanning**:
```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write(*.py)",
        "hooks": [
          {
            "type": "command",
            "command": "bandit -f json"
          }
        ]
      }
    ]
  }
}
```

**3. Test runner**:
```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "npm test -- --changed"
          }
        ]
      }
    ]
  }
}
```

### Hook Best Practices

1. **Use absolute paths** for scripts
2. **Handle errors gracefully** - don't block on non-critical failures
3. **Keep hooks fast** - they run synchronously
4. **Test thoroughly** before enabling
5. **Document** hook behavior for your team

## Intermediate Workflows

### 1. Test-Driven Development (TDD)

```
# Start with tests
> create a test for a function that validates email addresses

# Implement to pass tests
> now implement the email validation function to pass these tests

# Refactor
> refactor the validation function to be more efficient
```

### 2. API Development Workflow

```
# Design first
> create an OpenAPI spec for a user management API

# Generate boilerplate
> generate Express routes from the OpenAPI spec

# Add business logic
> implement the user creation endpoint with validation

# Add tests
> write integration tests for all endpoints
```

### 3. Debugging Production Issues

```
# Analyze logs
> here's a production error log [paste]. What's causing this?

# Reproduce locally
> create a script to reproduce this issue locally

# Fix and test
> fix the issue and add a test to prevent regression
```

### 4. Code Review Workflow

```
# Before PR
> /review

# Address feedback
> the reviewer said the function is too complex. 
  Can you split it into smaller functions?

# Document changes
> add JSDoc comments to all public functions
```

### 5. Performance Optimization

```
# Profile first
> analyze this function for performance bottlenecks

# Optimize
> optimize the database queries in this module

# Measure
> add performance benchmarks for the critical paths
```

## Best Practices

### 1. Clear Communication
- Be specific about what you want
- Provide context and constraints
- Share error messages completely

### 2. Incremental Development
- Break large tasks into smaller steps
- Review and test each change
- Commit frequently

### 3. Safety First
- Review all changes before approving
- Use permission settings to prevent accidents
- Keep sensitive data out of prompts

### 4. Efficient Context Usage
- Use `/compact` when context is filling up
- Clear conversation with `/clear` when switching tasks
- Use specific file references with `@filename`

### 5. Team Collaboration
- Document custom commands
- Share useful settings via `.claude/settings.json`
- Use conventional commit messages

## Troubleshooting

### Common Issues and Solutions

**1. "Context window full" errors**:
```
> /compact focus on the current feature
# or
> /clear
```

**2. Permission denied errors**:
```bash
# Check current permissions
> /permissions

# Update settings to allow specific operations
claude config add permissions.allow "Bash(npm *)"
```

**3. Claude can't find files**:
```
# Add additional directories
> /add-dir ../other-project

# Or use full paths
> read the file at /full/path/to/file.js
```

**4. Slow responses**:
```
# Switch to a faster model for simple tasks
> /model claude-3-5-haiku

# Or be more specific in requests
> just fix the syntax error in line 42
```

**5. Installation issues**:
```bash
# Run diagnostics
claude doctor

# Check Node.js version
node --version  # Should be 18+

# Reinstall if needed
npm uninstall -g @anthropic-ai/claude-code
npm install -g @anthropic-ai/claude-code
```

### Getting Help

- Use `/help` for command reference
- Run `claude doctor` for system diagnostics
- Check `~/.claude/logs/` for detailed logs
- Report bugs with `/bug` command

## Next Steps

1. **Experiment with custom commands** for your workflow
2. **Set up hooks** for automation
3. **Configure permissions** for your security needs
4. **Create project-specific documentation** with `/init`
5. **Explore advanced features** like MCP servers and integrations

Remember: Claude Code is most effective when you communicate clearly and incrementally. Start with simple tasks and gradually work up to more complex workflows as you become comfortable with the tool.
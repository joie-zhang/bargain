# Command Context Loader

This file ensures Claude Code has access to all custom commands for autonomous execution.

## Core Template Commands

### Setup & Management
- @.claude/commands/setup.md - Interactive setup wizard for new projects
- @.claude/commands/crud-claude-commands.md - Create custom commands for your research
- @.claude/commands/page.md - Session state preservation
- @.claude/commands/plan-with-context.md - Smart implementation planning

### Example Patterns
- @.claude/commands/parallel-analysis-example.md - Multi-agent analysis pattern

## Research Documentation

### Core Guides
- @README.md - Repository setup and supported workflows
- @docs/reproduction.md - Experiment reproduction commands
- @docs/operations.md - Provider, Slurm, and log operations
- @docs/guides/VISUALIZATION_GUIDE.md - Paper figure workflows
- @docs/reproducibility/script_retention_manifest.md - Script retention decisions
- @docs/reproducibility/paper_figure_manifest.md - Figure manifest guide

## Script Guides
- @scripts/paper_figures/README.md - Paper figure renderers
- @scripts/retained_analysis/README.md - Temporary qualitative and exploratory analyses

## Loading Instructions

Claude Code should:
1. Use the current reproduction guide before it runs an experiment.
2. Use the manifests to identify paper scripts and inputs.
3. Use the retained-analysis README only for temporary analysis work.
4. Do not infer a workflow from an old result directory.

## Context Management

When context usage reaches:
- 50%: Start planning for eventual paging
- 70%: Actively suggest /page command
- 80%: Automatically execute /page with descriptive name
- 90%: Force page immediately to prevent loss

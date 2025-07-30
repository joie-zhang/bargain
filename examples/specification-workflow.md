# Specification-Driven Development Workflow

This example demonstrates how to use the specification-driven development tools in this template.

## Overview

The workflow enforces a discipline of:
1. **Define** what you want clearly (specification)
2. **Validate** that it's well-defined (validator)
3. **Test** that it works as specified (generated tests)
4. **Track** progress systematically (task management)

## Step-by-Step Example

### 1. Create a Specification

Use the `/spec-driven` command in Claude Code:

```
> /spec-driven create a feature for tracking experiment results in ML research
```

Or manually create a specification using the template:

```bash
cp docs/templates/specification-template.md docs/specs/SPEC-2024-01-15-experiment-tracking.md
```

### 2. Validate the Specification

```bash
./scripts/run-spec-tests.sh validate docs/specs/SPEC-2024-01-15-experiment-tracking.md
```

This will check for:
- Missing required sections
- Vague language in requirements
- Untestable requirements
- Missing test cases

### 3. Generate Tests

```bash
./scripts/run-spec-tests.sh generate docs/specs/SPEC-2024-01-15-experiment-tracking.md
```

This creates a test file with:
- Test cases for each requirement
- Tests for each acceptance criterion
- Edge case tests
- Integration test templates

### 4. Create Implementation Tasks

```bash
./utils/task_manager.py create "Implement experiment tracking" \
  --why "Researchers need to track and compare ML experiments systematically" \
  --what "Store experiment metadata" "Query results by date" "Export to CSV" \
  --how "Design database schema" "Implement storage API" "Add query interface" \
  --priority high
```

### 5. Work Through Implementation

With Claude Code:
```
> Let's implement the experiment tracking feature according to SPEC-2024-01-15
```

Claude will:
- Read the specification
- Understand the requirements
- Implement according to the plan
- Run tests to validate

### 6. Track Progress

View task board:
```bash
./utils/task_manager.py board
```

Update task status:
```bash
./utils/task_manager.py update TASK-2024-01-15-001 in-progress
```

### 7. Run Full Test Suite

```bash
./scripts/run-spec-tests.sh all
```

This will:
1. Validate all specifications
2. Generate/update tests
3. Run all tests
4. Report results

## Benefits of This Approach

1. **Clear Requirements**: No ambiguity about what needs to be built
2. **Testable Outcomes**: Success is measurable, not subjective
3. **AI-Friendly**: Claude can understand and follow specifications precisely
4. **Experimentation Freedom**: Within the specification bounds, you can try different approaches
5. **Quality Assurance**: Tests ensure the specification is met

## Example Specification

Here's a simplified example of what a good specification looks like:

```markdown
## What (Requirements & Acceptance Criteria)

### Functional Requirements
1. **[REQ-001]** The system MUST store experiment metadata including model name, hyperparameters, and timestamp
2. **[REQ-002]** The system MUST calculate and store performance metrics for each experiment
3. **[REQ-003]** The system SHOULD provide a query interface to filter experiments by date range

### Acceptance Criteria
- [ ] **AC-1**: Given an experiment with metrics, when queried by ID, then return all associated data
- [ ] **AC-2**: Given a date range, when querying experiments, then return only experiments within that range
- [ ] **AC-3**: Given invalid experiment data, when attempting to store, then return a validation error
```

## Tips for Success

1. **Be Specific**: "Fast response time" ❌ vs "Response within 200ms" ✅
2. **Make it Testable**: Every requirement should be verifiable
3. **Start Small**: Create specifications for individual features, not entire systems
4. **Iterate**: Specifications can evolve as you learn
5. **Use with AI**: Let Claude help create and validate specifications

## Integration with Multi-Agent Workflows

Combine with other commands for comprehensive analysis:

```
# Get multiple perspectives on your specification
> /multi-mind review the experiment tracking specification for completeness

# Search for similar implementations
> /search-all "experiment tracking implementation"

# Deep dive into specific functions
> /analyze-function "def store_experiment_results(data):"
```

## Troubleshooting

### "Specification validation failed"
- Check for missing sections
- Look for vague language
- Ensure all requirements are testable

### "Tests failing"
- Verify implementation matches specification
- Check edge cases
- Ensure test data is realistic

### "Too many requirements"
- Break into smaller specifications
- Create parent-child task relationships
- Focus on MVP first

Remember: The specification is your contract. Make it clear, make it testable, and success becomes inevitable.
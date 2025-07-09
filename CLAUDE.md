# AI Safety Research Assistant Configuration

<role>
You are an expert AI Safety Research Assistant specialized in conducting rigorous technical research, implementing complex systems, and maintaining high standards of code quality and documentation. Your role encompasses:
- Technical implementation with a focus on correctness and safety
- Research methodology and experimental design
- Documentation and knowledge management
- Multi-agent coordination for complex analyses
- Systematic problem-solving with verifiable results
</role>

<project_context>
This is a template repository for AI safety research projects, designed for seamless integration with Claude Code and optimized for managing complex research workflows.

**Project Type**: AI Safety Research Template
**Primary Focus**: Research infrastructure, experimental frameworks, and safety analysis tools
**Key Stakeholders**: AI safety researchers, alignment engineers, and technical contributors
</project_context>

## Core Principles & Success Criteria

<principles>
1. **Clarity Above All**: Think of me as a brilliant but context-limited assistant who needs explicit, unambiguous instructions
2. **Context is King**: Every request should include:
   - Purpose and intended use of the output
   - Target audience and their technical level
   - Success criteria and definition of "done"
   - Any constraints or requirements
3. **Specification-Driven Development**: Create testable requirements before implementation
4. **Parallel Execution**: Leverage multiple Claude instances for independent tasks
5. **Systematic Documentation**: Capture insights immediately in appropriate locations
6. **Iterative Refinement**: Build understanding incrementally with continuous validation
7. **Test-First Approach**: Generate tests from specifications to ensure correctness
</principles>

<success_criteria>
For research tasks:
- Accuracy: Results must be reproducible with <0.1% variance
- Documentation: Every finding must include methodology and limitations
- Verification: Critical results require multi-agent validation
- Traceability: Full audit trail from hypothesis to conclusion

For implementation tasks:
- Code Quality: 95%+ test coverage for critical paths
- Performance: Response time <200ms for user-facing operations
- Security: Zero exposed credentials or sensitive data
- Maintainability: Clear architecture with <20 cyclomatic complexity
</success_criteria>

## Directory Structure & Organization

<directory_structure>
```
.
├── issues/          # Requirements and issue tracking
├── docs/           # Documentation and planning files
│   ├── guides/     # User guides and how-tos
│   ├── reference/  # Technical reference docs
│   └── templates/  # Document templates
├── scripts/        # Automation and utility scripts
├── utils/          # Reusable code utilities
├── tests/          # Test files and validation scripts
├── examples/       # Example implementations and templates
├── .claude/        # Claude Code configuration and commands
│   ├── commands/   # Custom slash commands
│   └── hooks/      # Pre/post processing hooks
└── logs/           # Experiment logs and results
```

**IMPORTANT**: When creating markdown files or documentation, always place them in the appropriate `docs/` subdirectory rather than the project root. Use:
- `docs/guides/` for tutorials and how-to documentation
- `docs/reference/` for API references and technical specifications
- `docs/templates/` for reusable document templates
- `docs/` root only for high-level planning documents
</directory_structure>

## Workflow Guidelines

<research_workflow>
1. **Requirement Formalization**
   - Create unambiguous requirements in `issues/`
   - Include testable acceptance criteria
   - Specify edge cases and failure modes

2. **Planning Phase**
   - Document approach in `docs/plan_<issue_no>.md`
   - Identify potential risks and mitigations
   - Define checkpoints and validation methods

3. **Implementation**
   - Work incrementally with continuous testing
   - Maintain clean git history with atomic commits
   - Document decisions and trade-offs inline

4. **Documentation**
   - Update findings in real-time
   - Include both successes and failures
   - Cross-reference related work

5. **Validation**
   - Run comprehensive test suites
   - Perform multi-agent verification for critical components
   - Document performance characteristics
</research_workflow>

## Code Quality Standards

<code_quality>
### Implementation Guidelines
1. **No Mock Data**: Never create mock data or placeholder functions unless explicitly requested
   - Always implement real functionality
   - Use actual data from files or generate realistic test data
   - Avoid TODO placeholders in production code

2. **Concise and Correct**: Optimize for correctness over verbosity
   - Write clean, focused code that does one thing well
   - Avoid over-engineering or unnecessary abstractions
   - Remove commented-out code and excessive documentation
   - Keep functions under 50 lines when possible

3. **File Organization**: Maintain strict file organization
   - Place all markdown documentation in `docs/` subdirectories
   - Never dump files in the project root
   - Use appropriate subdirectories for different file types
   - Follow the established directory structure

4. **Testing First**: Write tests before or alongside implementation
   - Generate tests from specifications
   - Include edge cases and error conditions
   - Aim for high coverage on critical paths
   - Use property-based testing where appropriate

5. **Performance Aware**: Consider performance implications
   - Profile before optimizing
   - Document performance characteristics
   - Use appropriate data structures
   - Minimize context usage in Claude commands
</code_quality>

## Claude Code Integration

<custom_commands>
### 🚀 Getting Started - Run /setup First!

**New to this template?** Start with:
```
/setup
```
This will guide you through integrating your research codebase and creating custom commands for your specific workflow.

### 📦 Core Template Commands

#### `/setup` - Interactive Setup Wizard
- **Purpose**: Guide you through integrating your research codebase with AI agent workflows
- **When to use**: First time using this template or adding a new codebase
- **What it does**: 
  - Helps merge template with your existing code
  - Guides context gathering (papers, documentation)
  - Assists in creating custom commands for your research
  - Sets up verification infrastructure
- **Command file**: @.claude/commands/setup.md

#### `/crud-claude-commands` - Create Your Own Commands
- **Purpose**: Dynamically create, update, or delete custom slash commands
- **When to use**: 
  - You find yourself repeating the same requests
  - Need to automate a specific workflow
  - Want to create domain-specific commands
- **Example**: `/crud-claude-commands create run-ablation-study`
- **Command file**: @.claude/commands/crud-claude-commands.md

#### `/page` - Save Your Work
- **Purpose**: Save complete session state before context fills up
- **When to use**:
  - Context usage exceeds 70%
  - Switching between major tasks
  - End of work session
- **Auto-trigger**: At 80% context usage
- **Example**: `/page "experiment-2024-01-15-results"`
- **Command file**: @.claude/commands/page.md

#### `/plan-with-context` - Smart Planning
- **Purpose**: Create implementation plans while managing context efficiently
- **When to use**:
  - Breaking down complex tasks
  - Limited context but big goals
  - Need phased approach
- **Command file**: @.claude/commands/plan-with-context.md

#### `/plan-auto-context-enhanced` - Natural Language Planning
- **Purpose**: Create implementation plans by simply describing what you want in natural language
- **When to use**:
  - You want to describe your goal without specifying files
  - Need automatic discovery of relevant code
  - Want intelligent context selection
- **Example**: `/plan-auto-context-enhanced I want to add rate limiting to our API`
- **Command file**: @.claude/commands/plan-auto-context-enhanced.md
- **Note**: Also integrates with repomix and Gemini CLI when available

#### `/parallel-analysis-example` - Multi-Agent Pattern Example
- **Purpose**: EXAMPLE showing how to use multiple agents in parallel
- **When to use**: As a template for creating your own multi-agent commands
- **Adapt for**:
  - Multi-model ensemble analysis
  - Parallel literature review
  - Distributed experiment validation
- **Command file**: @.claude/commands/parallel-analysis-example.md

#### `/integrate-external-codebase` - Integrate External Repositories
- **Purpose**: Integrate external codebases (GitHub repos or local) for AI-assisted analysis
- **When to use**:
  - Adding a dependency or framework to analyze
  - Integrating research code from collaborators
  - Studying existing implementations
- **Example**: `/integrate-external-codebase https://github.com/user/repo`
- **Command file**: @.claude/commands/integrate-external-codebase.md

#### `/clean-and-organize` - Repository Maintenance
- **Purpose**: Clean temporary files and organize misplaced files
- **When to use**:
  - Repository has accumulated temp files
  - Markdown files cluttering the root directory
  - After extensive development sessions
- **What it does**:
  - Removes: *.tmp, *.temp, __pycache__, *.pyc, etc.
  - Organizes: Moves stray markdown → ai_docs/temp_markdowns/
  - Preserves: .env, .claude/logs/, and other important files
- **Command file**: @.claude/commands/clean-and-organize.md

### 🛠️ Creating Your Research-Specific Commands

After running `/setup`, you'll want to create commands specific to your research domain. Here are patterns to consider:

1. **Experiment Automation**
   ```
   /crud-claude-commands create run-experiment
   ```
   Design it to handle your specific:
   - Configuration management
   - Result validation
   - Metric tracking

2. **Analysis Workflows**
   ```
   /crud-claude-commands create analyze-results
   ```
   Customize for your:
   - Statistical tests
   - Visualization needs
   - Comparison methods

3. **Debugging Helpers**
   ```
   /crud-claude-commands create debug-training
   ```
   Include your common:
   - Error patterns
   - Diagnostic steps
   - Quick fixes

4. **Literature Integration**
   ```
   /crud-claude-commands create paper-to-code
   ```
   For implementing papers:
   - Extract key algorithms
   - Generate test cases
   - Verify correctness
</custom_commands>

<autonomous_command_usage>
### Guiding Principles for Your Custom Commands

When creating commands for your research, consider these patterns:

#### Command Triggers
Design your commands to activate on specific patterns:
- Research-specific terminology
- Repeated workflows
- Common analysis needs
- Debugging scenarios

#### Example Patterns to Implement

1. **Experiment Management**
   ```python
   # When user mentions "run experiment" or "test hypothesis"
   if "experiment" in message or "hypothesis" in message:
       use_command("/run-experiment", config=extract_config())
   ```

2. **Results Analysis**
   ```python
   # When results are generated
   if "results ready" or "analyze output":
       use_command("/analyze-results", metrics=domain_specific_metrics)
   ```

3. **Context Preservation**
   ```python
   # Always monitor context usage
   if context_usage > 0.7:
       suggest_command("/page", reason="preserve progress")
   elif context_usage > 0.8:
       auto_command("/page", name=generate_checkpoint_name())
   ```

#### Building Reliable Automation

1. **Start with Manual Commands**: Run commands manually first to understand patterns
2. **Identify Repetition**: Notice when you use the same command sequence
3. **Create Composite Commands**: Combine common sequences into single commands
4. **Add Verification**: Always include checks to ensure commands worked
5. **Document Failures**: Track when commands fail and why

#### Integration with External Tools
Your custom commands can integrate with:
- Experiment tracking (Weights & Biases, MLflow)
- Compute clusters (SLURM, Ray)
- Version control (Git workflows)
- Data pipelines (DVC, Pachyderm)
- Analysis tools (Jupyter, Pandas)
</autonomous_command_usage>

<autonomous_scripts>
### Autonomous Research Scripts

All slash commands now have programmatic versions in `scripts/commands/` that can be executed autonomously:

#### Codebase Integration
```bash
# Integrate external codebase
python scripts/integrate_codebase.py https://github.com/org/repo.git
```

#### Automatic Context Planning
```bash
# Create implementation plan with automatic context selection
python scripts/commands/plan_auto_context.py "add caching layer to improve API performance"

# With external tools (repomix + Gemini)
python scripts/commands/plan_auto_context.py "implement websocket notifications" --output plan.md
```

#### Orchestrated Workflows
```bash
# Run comprehensive analysis workflow
python scripts/commands/orchestrate_research.py \
    --workflow comprehensive_analysis \
    --codebase ./project \
    --task "Analyze security architecture"
```

#### Multi-Model Coordination
The orchestration script supports:
- Parallel execution of multiple tools
- Custom workflow definitions
- Automatic report generation
- Integration with Gemini, O3, and other models

See `scripts/README.md` for complete documentation.
</autonomous_scripts>

<thinking_guidelines>
## Extended Thinking Usage

For complex problems, I will use extended thinking when you include trigger phrases:
- "think step by step" - Basic reasoning
- "think deeply" - More thorough analysis
- "think harder" - Complex problem solving
- "ultrathink" - Maximum reasoning depth

Best used for:
- Mathematical proofs and derivations
- Architecture decisions with multiple trade-offs
- Debugging subtle issues
- Research hypothesis formation
- Safety analysis and failure mode identification
</thinking_guidelines>

## Best Practices for Maximum Efficiency

<prompting_best_practices>
### Be Explicit and Context-Rich
Instead of: "Implement caching"
Use: "Implement an LRU cache for model embeddings with 10GB capacity, <100ms lookup time, and thread-safe operations. This will be used in production serving 1M+ requests/day."

### Provide Success Examples
<example>
Task: "Create a configuration system"
Good: "Create a configuration system like Hydra that supports:
- Hierarchical configs with overrides
- Type validation using Pydantic
- Environment variable interpolation
- Config versioning for reproducibility
Output should follow this structure:
```yaml
model:
  type: transformer
  params:
    layers: 12
    hidden_dim: 768
```
"
</example>

### Leverage Parallel Operations
"For maximum efficiency, analyze these three papers simultaneously using separate agents:
1. Paper A: Focus on methodology
2. Paper B: Extract experimental results
3. Paper C: Identify limitations
Synthesize findings focusing on common patterns."

### Control Output Format
<formatting>
For structured outputs:
- Use XML tags: "Provide analysis in <findings>, <methodology>, and <limitations> tags"
- Request specific formats: "Output as JSON with schema: {metric: string, value: float, confidence: float}"
- Prefill patterns: "Begin your response with '## Technical Analysis\n### Key Findings:'"
</formatting>
</prompting_best_practices>

## Integration with External Tools

<tool_integration>
### Gemini CLI for Large-Scale Analysis
Use when Claude's context would be exceeded:
```bash
# Analyze entire codebases
gemini -p "@src/ Explain the architecture and identify coupling points"

# Compare multiple papers
gemini -p "@paper1.pdf @paper2.pdf @paper3.pdf Synthesize approaches to adversarial robustness"

# Find patterns across many files
gemini -p "@**/*.py List all custom loss functions with their mathematical formulations"
```

### Git Integration
- Commits: Use conventional commit format (feat:, fix:, docs:, etc.)
- Branches: `feature/<issue-no>-<description>` or `experiment/<name>`
- PRs: Include test results and performance impact

### Testing Frameworks
- Always verify framework first: `grep -r "test_" . | head -5`
- Prefer pytest for Python, jest for JavaScript
- Include property-based tests for critical functions
</tool_integration>

## Session Management Best Practices

<session_management>
1. **Track Progress**: Maintain `todo.md` with checkboxes for complex tasks
2. **Checkpoint Regularly**: `/page` when switching contexts or at major milestones
3. **Resume Gracefully**: `claude --resume <checkpoint>` preserves full context
4. **Clean Up**: Remove temporary files and close unused resources

### Memory Limits and Optimization
- Monitor context usage with `/compact` when >70% full
- Summarize intermediate results rather than keeping full outputs
- Use external storage for large datasets or logs
</session_management>

## Security and Safety Guidelines

<security_guidelines>
1. **Never Expose**: API keys, credentials, or personal information
2. **Validate Inputs**: Sanitize all external data before processing
3. **Audit Outputs**: Review generated code for potential vulnerabilities
4. **Version Control**: Never commit sensitive data, use .gitignore
5. **Access Control**: Implement principle of least privilege
</security_guidelines>

## Common Patterns and Solutions

<common_patterns>
### For Research Algorithms
"Implement the algorithm ensuring:
- Numerical stability and error propagation analysis
- Comprehensive unit tests with edge cases
- Performance benchmarks on relevant datasets
- Memory and computational complexity profiling
- Full reproducibility (seeds, environments, dependencies)
- Systematic ablation study support
Document theoretical foundations, empirical validation, and limitations."

### For Experimental Analysis
"Implement the analysis pipeline with:
- Statistical significance testing
- Multiple hypothesis correction
- Confidence interval calculation
- Effect size estimation
- Robustness checks across conditions
- Visualization of key relationships
Include assumption validation and sensitivity analysis."

### For Research Infrastructure
"Build the experimental framework with:
- Parameter sweep and grid search capabilities
- Distributed computation support
- Checkpoint and recovery mechanisms
- Comprehensive logging and monitoring
- Experiment tracking and versioning
- Automated report generation
Ensure reproducibility across different environments and scales."
</common_patterns>

## Notes on Hooks and Automation

<automation>
- Mock data validation: Blocked by default, docs/ exempt
- Git auto-commit: Triggers when Claude stops, preserves work
- Python formatting: Black runs automatically post-edit
- Activity logging: All actions logged to ~/.claude/research-activity.log
- Test execution: Automatic on file changes matching test_*.py
</automation>

## Important File References

<key_files>
- Project overview: @README.md
- Custom commands: @.claude/commands/
- Setup script: @scripts/setup.sh
- Integration script: @scripts/integrate_codebase.py

### Available Utilities (@utils/)
- **context_analyzer.py**: Analyze and optimize context usage
- **session_manager.py**: Manage Claude sessions and checkpoints
- **spec_validator.py**: Validate experiment specifications
- **task_manager.py**: Task tracking and management
- **todo_manager.py**: Todo list management utilities
- **verification_framework.py**: Build verification loops for experiments
- **verification_metrics.py**: Common metrics for result validation

### Key Directories
- **ai_docs/**: Store AI-optimized documentation and summaries
- **specs/**: Experiment and feature specifications
  - Contains `EXAMPLE_RESEARCH_SPEC.md` - Template for creating detailed specs
- **experiments/**: Experiment configs, results, and logs
- **external_codebases/**: Integration point for external code
- **.claude/**: Claude-specific configuration and hooks
- **scripts/**: Automation and command implementations
- **docs/**: Template documentation and guides
- **examples/**: Example workflows and patterns

### Important Templates
- **Specification Template**: @specs/EXAMPLE_RESEARCH_SPEC.md
  - Use this as starting point for any new research task
  - Copy and customize for your specific needs
  - Specs should be the source of truth for implementation
</key_files>

Remember: Every conversation should create lasting value. Build knowledge bases that compound over time. Small, well-designed tools compose into powerful workflows.
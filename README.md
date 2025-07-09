# AI Research Assistant Template

A minimal template for transforming your research codebase into an AI-agent-friendly environment. This template helps you integrate Claude Code and other AI agents into your research workflow.

## 🚀 Quick Start

1. **Clone this template**
   ```bash
   git clone [this-repo] ai-research-template
   cd ai-research-template
   ```

2. **Merge with your research codebase**
   ```bash
   cd [your-research-project]
   git remote add template [path-to-template]
   git fetch template
   git merge template/main --allow-unrelated-histories
   ```

3. **Run the setup wizard**
   ```bash
   claude
   > /setup
   ```

The `/setup` command will guide you through:
- Integrating your existing code
- Creating comprehensive documentation
- Building custom commands for your workflow
- Setting up verification infrastructure

## 📁 What's Included

```
.claude/
├── commands/          # Slash commands (start with /setup)
├── hooks/            # Safety hooks and automation
└── scripts/          # Integration utilities

ai_docs/              # Your research documentation (empty)
specs/                # Experiment specifications
│   └── EXAMPLE_RESEARCH_SPEC.md  # Template for creating detailed specs
experiments/          # Experiment tracking (empty)
scripts/              # Automation scripts
docs/                 # Template documentation
utils/               # Python utilities for research workflows
external_codebases/  # For integrating external code
```

## 🎯 Core Philosophy

This template provides the **minimal infrastructure** to:
1. Help AI agents understand your research context
2. Create custom automation for your specific workflow
3. Build verification loops so AI can check its work
4. Manage long research sessions efficiently

## 🛠️ Key Commands

- `/setup` - Interactive setup wizard (START HERE)
- `/crud-claude-commands` - Create your own custom commands
- `/page` - Save session state before context fills
- `/plan-with-context` - Smart planning for complex tasks
- `/parallel-analysis-example` - Example of multi-agent patterns

## 📚 Creating Your Own Commands

After running `/setup`, create domain-specific commands:

```bash
# Create a command for your experiment workflow
/crud-claude-commands create run-experiment

# Create analysis commands
/crud-claude-commands create analyze-results

# Create debugging helpers
/crud-claude-commands create debug-training
```

## 🔄 Workflow Overview

1. **Context Gathering** (Most Important!)
   - 15-20 minute speech-to-text brain dump
   - Add papers and documentation to `ai_docs/`
   - Create comprehensive CLAUDE.md

2. **Specification Writing**
   - Copy `specs/EXAMPLE_RESEARCH_SPEC.md` as starting point
   - Fill out problem statement, approach, and success criteria
   - Use AI to help draft technical sections
   - Specs become your source of truth

3. **Custom Command Creation**
   - Identify repetitive tasks
   - Create commands for your workflow
   - Add verification steps
   - Link commands to specs

4. **Experiment Automation**
   - Implement based on spec
   - Build with verification loops
   - Test against success criteria
   - Iterate until reliable

5. **Scale Your Research**
   - Use multiple models (Gemini for large context, O3 for reasoning)
   - Run parallel experiments
   - Maintain rigorous verification

## 📖 Key Features

- **Specification-Driven Development**: Use the example spec template to plan before coding
- **Custom Command System**: Build reusable workflows specific to your research
- **Context Management**: Efficient handling of long research sessions
- **Verification Loops**: Ensure AI agents can check their own work
- **Multi-Model Support**: Integrate Gemini, O3, and other models as needed

## 🤝 Contributing

This template is designed to be minimal and extensible. Feel free to:
- Add domain-specific examples
- Share successful command patterns
- Improve the setup workflow

## 📝 License

MIT - Use freely for your research!
# AI Agent Research Workshop: Transform Your Research with Claude Code

## 🎯 Workshop Overview

**Duration**: 3 hours  
**Goal**: Transform your research codebase into an AI-enhanced environment where Claude Code and other AI agents can reliably assist with experiments, implementation, and verification.

By the end of this workshop, you will have:
- ✅ Integrated your research codebase with AI agent workflows
- ✅ Created comprehensive context documentation for AI understanding
- ✅ Built custom commands for your repetitive research tasks
- ✅ Implemented verification loops for reliable AI assistance
- ✅ Completed at least one automated research task

---

## 📋 Pre-Workshop Setup (Do Before Workshop)

### Required Tools
- [ ] **Claude Code** installed with subscription
- [ ] **Git** command line tools
- [ ] **Python 3.9+** installed
- [ ] **Speech-to-Text Tool** - Install ONE:
  - [Whisper Flow](https://whisperflow.app/) (Recommended - All platforms)
  - [SuperWhisper](https://superwhisper.com/) (Mac)
  - [MacWhisper](https://goodsnooze.gumroad.com/l/macwhisper) (Mac)

### Clone the Template
```bash
# Clone the AI research template
git clone https://github.com/[template-repo] ai-research-workshop
cd ai-research-workshop

# Verify Claude Code works
claude --version
```

---

## 🚀 Part 0: Quick Start with Setup Wizard (15 min)

We'll begin with the interactive setup wizard that customizes everything for your research:

```bash
# Start Claude Code
claude

# Run the setup wizard
/setup
```

The `/setup` wizard will guide you through:
1. **Codebase Assessment** - Understanding your research domain
2. **Integration Options** - Merging template with your code
3. **Directory Creation** - Setting up AI-friendly structure
4. **Context Gathering** - Recording your research knowledge
5. **CLAUDE.md Creation** - Custom instructions for your project
6. **First Specification** - Planning your first task
7. **Custom Commands** - Building research-specific tools

### 💡 Setup Tips
- Be specific about your research domain and challenges
- Have your existing codebase path/GitHub URL ready
- Prepare to talk for 15-20 minutes about your research
- Think of 2-3 repetitive tasks you want to automate

---

## 📝 Part 1: Deep Context Gathering (45 min)

### Step 1.1: Research Brain Dump (20 min)
The setup wizard will prompt you to record a comprehensive brain dump. Use your speech-to-text tool to answer:

#### 🎤 What to Cover in Your Recording
1. **Research Overview**
   - Your specific research question and hypothesis
   - Why this matters and potential impact
   - What success looks like concretely

2. **Technical Foundation**
   - Existing codebases/papers you build on
   - Key frameworks and libraries
   - Previous experiments and results
   - What failed and why (crucial!)

3. **Hidden Knowledge**
   - Assumptions you're making
   - Known pitfalls in your area
   - Debugging tricks you've learned
   - "If you see X, it usually means Y"

4. **Immediate Goals**
   - Minimal experiment to validate hypothesis
   - Key metrics you care about
   - Comparison baselines

Save as: `ai_docs/context/research_overview_transcript.txt`

### Step 1.2: Gather Supporting Materials (15 min)
Collect and organize in the template structure:
```
ai_docs/
├── papers/          # Your key papers (PDFs)
├── summaries/       # AI-generated summaries
├── context/         # Your brain dumps and notes
└── cheatsheets/     # Quick references for packages

specs/
├── experiments/     # Experiment specifications
├── features/        # Feature implementations
└── analysis/        # Analysis plans
```

### Step 1.3: Process with AI (10 min)
Let Claude help structure your context:
```bash
# Have Claude create a research plan from your brain dump
> Read ai_docs/context/research_overview_transcript.txt
> Create a structured research plan with:
> - Clear hypothesis and success metrics
> - Minimal viable experiment design
> - Required tools and dependencies
> - Verification strategies
> Save as specs/experiments/initial_research_plan.md
```

---

## 📋 Part 2: Specification-Driven Development (30 min)

### Step 2.1: Understanding the Spec Template (10 min)
This template uses specification-driven development. Examine the example:

```bash
# Read the example specification
> Show me specs/EXAMPLE_RESEARCH_SPEC.md
> Explain how each section helps with implementation
```

### Step 2.2: Create Your First Spec (20 min)
Copy and customize for your research:

```bash
# Copy the template
cp specs/EXAMPLE_RESEARCH_SPEC.md specs/experiments/my_first_task.md

# Have Claude help fill it out
> Based on my research context, help me complete specs/experiments/my_first_task.md
> Focus on:
> - Specific, measurable success criteria
> - Clear implementation phases
> - Self-validation strategies
> - Risk mitigation
```

Key sections to complete:
- **Problem Statement**: What exactly are you trying to solve?
- **Technical Approach**: How will you solve it?
- **Success Criteria**: How do you know it worked?
- **Verification Methods**: How to check correctness automatically
- **Implementation Phases**: Break into manageable chunks

---

## 🛠 Part 3: Building Custom Commands (30 min)

### Step 3.1: Identify Automation Opportunities (10 min)
Think about your repetitive tasks:
- Running experiments with different parameters
- Processing and visualizing results
- Comparing against baselines
- Generating reports or plots
- Debugging common issues

### Step 3.2: Create Your First Command (20 min)
Use the CRUD command system:

```bash
# Create a command for your most common task
/crud-claude-commands create run-my-experiment

# Claude will guide you through:
# 1. Defining the command's role
# 2. Setting up the workflow
# 3. Adding verification steps
# 4. Creating examples
```

Example custom command creation:
```bash
> I need a command that:
> - Sets up my experiment environment
> - Runs with different hyperparameters
> - Validates results aren't NaN
> - Compares with baseline
> - Saves results with timestamps
```

Your new command will be saved in `.claude/commands/run-my-experiment.md`

### Test Your Command
```bash
# Try your new command
/run-my-experiment

# Debug if needed
> The command failed at step X. Let's fix it by...
```

---

## 🔬 Part 4: Implementation with Verification (45 min)

### Step 4.1: Set Up Verification Infrastructure (15 min)

Use the template's verification framework:

```bash
# Explore available verification tools
> Show me what's in utils/verification_framework.py
> Create a verification script for my experiment using this framework

# Set up continuous verification
> Create scripts/watch_my_experiment.sh that:
> - Monitors for file changes
> - Runs my verification checks
> - Shows clear pass/fail status
> - Logs any issues
```

### Step 4.2: Implement with Verification Loops (20 min)

Follow the spec you created:

```bash
# Start implementation
> Implement Phase 1 from specs/experiments/my_first_task.md
> After each function:
> 1. Add verification using utils/verification_framework.py
> 2. Ensure no mock data (check with grep)
> 3. Add logging for debugging
> 4. Run the verification script

# Use the verification utilities
> Show me how to use verification_metrics.py for my metrics
> Add sanity checks using the framework
```

### Step 4.3: Test and Iterate (10 min)

```bash
# Run comprehensive tests
python scripts/verify_experiment.py

# If issues arise, debug systematically
> Something failed. Let's debug step by step:
> 1. Check the logs
> 2. Verify inputs
> 3. Test with minimal data
> 4. Compare with expected behavior
```

---

## 🤝 Part 5: Multi-Agent Analysis (30 min)

### Step 5.1: Understanding Parallel Analysis (10 min)

The template includes powerful multi-agent patterns:

```bash
# Examine the example
> Show me the /parallel-analysis-example command
> Explain how to adapt this for my research
```

### Step 5.2: Create Domain-Specific Analysis (20 min)

Adapt the pattern for your needs:

```bash
# Create your own multi-agent command
/crud-claude-commands create multi-verify-results

# Design specialist agents for your domain
> Create a multi-agent verification that uses:
> - Agent 1: Statistical validity checker
> - Agent 2: Comparison with literature
> - Agent 3: Visual inspection and plots
> - Agent 4: Edge case analysis
> Synthesize their findings
```

### Run Multi-Agent Analysis
```bash
# Test your multi-agent command
/multi-verify-results

# Or use the Python script directly
python scripts/parallel_analysis_example.py --task "Verify my results"
```

---

## 💾 Part 6: Session Management (15 min)

### Step 6.1: Save Your Work (5 min)
Before context fills up:

```bash
# Save session state
/page workshop-checkpoint-1

# This creates:
# - Full conversation history
# - Key code snippets
# - Current state snapshot
# - Resume instructions
```

### Step 6.2: Document and Commit (10 min)

```bash
# Update your research plan
> Update specs/experiments/initial_research_plan.md with:
> - What we accomplished today
> - Refined hypothesis based on results
> - Next experiments to run
> - New automation opportunities identified

# Generate comprehensive commit message
> Create a detailed commit message covering:
> - New commands created
> - Verification infrastructure added
> - Specifications written
> - Results obtained

# Commit everything
git add -A
git commit
git push origin ai-enhanced-research
```

---

## 🚀 Advanced Features to Explore

### After the Workshop

1. **Orchestrated Workflows**
   ```bash
   # Run complex multi-step research workflows
   python scripts/orchestrate_research.py \
     --workflow analyze_and_iterate \
     --task "Optimize my model architecture"
   ```

2. **External Codebase Integration**
   ```bash
   # Integrate another research codebase
   python scripts/integrate_codebase.py https://github.com/other/research
   ```

3. **Session Analysis**
   ```bash
   # Analyze your Claude sessions for patterns
   python utils/session_manager.py analyze --last 7d
   ```

4. **Context Optimization**
   ```bash
   # Optimize context usage
   python utils/context_analyzer.py suggest --target efficiency
   ```

---

## 📚 Command Quick Reference

### Essential Commands
```bash
/setup                    # Initial setup wizard
/crud-claude-commands     # Create/manage custom commands
/page                     # Save session state
/plan-with-context       # Smart planning for complex tasks
/parallel-analysis-example # Multi-agent analysis template
/compact                 # Check context usage

# Thinking modes
think step by step       # Basic reasoning
think deeply            # Thorough analysis
think harder            # Complex problems
ultrathink             # Maximum reasoning depth
```

### Your Custom Commands
```bash
/run-my-experiment      # Your experiment runner
/multi-verify-results   # Your verification command
# ... other commands you create
```

### Useful Scripts
```bash
scripts/verify_experiment.py      # Verification runner
scripts/integrate_codebase.py    # Add external code
scripts/orchestrate_research.py  # Complex workflows
utils/verification_framework.py  # Verification tools
utils/task_manager.py           # Task tracking
```

---

## 🎯 Success Checklist

By the end of the workshop:
- [ ] Completed `/setup` wizard customization
- [ ] Created comprehensive research context in `ai_docs/`
- [ ] Written at least one detailed specification
- [ ] Built 2-3 custom commands for your workflow
- [ ] Implemented one task with verification
- [ ] Tested multi-agent analysis pattern
- [ ] Saved session with `/page`
- [ ] Committed everything to git

---

## 🆘 Troubleshooting

### "Claude doesn't understand my domain"
- Add more context to CLAUDE.md
- Create domain-specific glossary in `ai_docs/`
- Use speech-to-text to explain concepts

### "Verification keeps failing"
- Start with simpler checks
- Use `utils/verification_framework.py` helpers
- Add more logging to understand failures
- Test with tiny datasets first

### "Running out of context"
- Use `/page` to save state regularly
- `/compact` to check usage
- Break large tasks into phases
- Use `/plan-with-context` for smart loading

### "Custom command not working"
- Check command syntax in `.claude/commands/`
- Verify all steps are clear
- Test each step individually
- Add error handling

---

## 📈 Next Steps

1. **Expand Your Commands**: Create commands for all repetitive tasks
2. **Build Verification Suite**: Comprehensive checks for your domain
3. **Integrate More Tools**: Connect other analysis tools
4. **Share With Team**: Your setup benefits everyone
5. **Join Community**: Share successes and get help

### Resources
- Template Documentation: Check `README.md`
- Command Examples: See `.claude/commands/`
- Utility Functions: Explore `utils/`
- Community: #ai-research-assistants

Remember: **You're the conductor of an AI orchestra.** The goal is to amplify your research capabilities while maintaining rigor and correctness!
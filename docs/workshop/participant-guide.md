# AI Agent Research Workshop: Transform Your Research with Claude Code

## 🎯 Workshop Overview

**Duration**: 3-4 hours  
**Goal**: Master AI-assisted research by progressively building from basic Claude Code usage to implementing new features in your codebase with AI agents.

By the end of this workshop, you will have:
- ✅ Mastered Claude Code basics and advanced features
- ✅ Created comprehensive AI-ready documentation 
- ✅ Automated a repetitive research task reliably
- ✅ Implemented new functionality with AI assistance
- ✅ Built custom tools and workflows for your research

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

### Prepare Your Research
- [ ] Have an existing research codebase ready (local or GitHub)
- [ ] Think of 2-3 repetitive tasks you do often
- [ ] Gather key papers (3-5 PDFs) related to your work
- [ ] Have a feature idea you'd like to implement

### Clone the Template
```bash
# Clone the AI research template
git clone https://github.com/[template-repo] ai-research-workshop
cd ai-research-workshop
```

---

## 🚀 Part 1: Get Comfortable with Claude Code (20 min)

### Step 1.1: Basic Commands and Interface (10 min)

Start Claude Code and explore:
```bash
# Start Claude Code
claude

# Try basic interactions
> What files are in this directory?
> Create a simple Python script that prints "Hello Research"
> Run the script
```

### Key Commands to Try:
```bash
# Context management
/compact              # Check how much context you're using
/clear               # Start fresh if needed

# Thinking modes
think                # Basic reasoning
think hard           # More thorough analysis
think harder         # Complex problems
ultrathink          # Maximum reasoning (slow but thorough)

# Keyboard shortcuts
Shift+Tab           # Toggle auto-accept mode
```

### Step 1.2: Explore the Template (10 min)

```bash
# See what commands are available
> Show me all available slash commands in .claude/commands/

# Try the help command
> What does the /setup command do?

# Explore utilities
> What verification tools are in utils/?
```

**Key Learning**: Claude Code is your research assistant. You provide context and instructions, it helps execute.

---

## 📚 Part 2: Gather Context for Your Codebase (30 min)

### Step 2.1: Run Setup Wizard (10 min)

```bash
# Start the interactive setup
/setup
```

The wizard will guide you through:
- Assessing your research domain
- Integrating your codebase
- Creating initial structure

### Step 2.2: Organize Research Materials (10 min)

```bash
# Create proper structure
mkdir -p ai_docs/{papers,summaries,context,cheatsheets}

# Add your papers
cp ~/Downloads/*.pdf ai_docs/papers/

# If you have existing code
cp -r ~/my-research-code/* .
# OR use the integration script
python scripts/integrate_codebase.py https://github.com/myuser/myresearch
```

### Step 2.3: Process Papers with AI (10 min)

For each key paper:
```bash
# Use Gemini for large PDFs (if available)
gemini -p "@ai_docs/papers/key_paper.pdf Summarize focusing on methods and implementation details"

# Or use Claude
> Read ai_docs/papers/[paper].pdf and create a summary focusing on:
> - Key algorithms and methods
> - Implementation details
> - Metrics and baselines used
> - How this relates to my research
> Save as ai_docs/summaries/[paper]_summary.md
```

---

## 🎤 Part 3: Speech-to-Text Project Outline (20 min)

### Step 3.1: Review Specification Template (5 min)

```bash
# Examine the template
> Show me specs/EXAMPLE_RESEARCH_SPEC.md
> Explain each section and why it's important
```

### Step 3.2: Record Comprehensive Brain Dump (15 min)

Use the specification template sections as prompts. **Record yourself answering each section**:

1. **Problem Statement**
   - "The specific problem I'm solving is..."
   - "This matters because..."
   - "Success looks like..."

2. **Technical Approach**
   - "My approach works by..."
   - "The key innovation is..."
   - "I'm building on [previous work] by..."

3. **Current Implementation**
   - "The codebase currently does..."
   - "The main files and functions are..."
   - "Common issues I face are..."

4. **Experimental Setup**
   - "My typical workflow is..."
   - "I validate results by..."
   - "Key metrics I track are..."

5. **Known Challenges**
   - "Things that often go wrong..."
   - "Debugging usually involves..."
   - "Assumptions that might not hold..."

Save as: `ai_docs/context/project_outline_transcript.txt`

---

## 📝 Part 4: Create Your Ideal CLAUDE.md (20 min)

### Step 4.1: Generate CLAUDE.md from Context (10 min)

```bash
# Have Claude synthesize everything
> Read:
> - ai_docs/context/project_outline_transcript.txt  
> - ai_docs/summaries/*.md
> - The existing codebase structure
> 
> Create a comprehensive CLAUDE.md that includes:
> 1. Research Context (from transcript)
> 2. Technical Details (from code analysis)
> 3. Common Workflows (from my description)
> 4. Verification Protocols (domain-specific)
> 5. Model Orchestration Rules
> 6. Failed Attempts Log
> 
> Make it specific to my research domain
```

### Step 4.2: Refine and Personalize (10 min)

```bash
# Review and enhance
> The CLAUDE.md looks good but add:
> - My specific debugging strategies
> - Common commands I run
> - Specific pitfalls in my domain
> - Preferred coding patterns from my codebase
```

Your CLAUDE.md should now be a comprehensive guide for any AI agent working on your research.

---

## 🔧 Part 5: Automate an Existing Task (45 min)

### Step 5.1: Choose Your Task (5 min)

Pick a **medium-difficulty, repetitive task** you already do, such as:
- Running experiments with different hyperparameters
- Processing datasets with specific transforms
- Generating comparison plots
- Running evaluation metrics on results

### Step 5.2: Create Detailed Specification (15 min)

```bash
# Copy the template
cp specs/EXAMPLE_RESEARCH_SPEC.md specs/experiments/automate_[task].md

# Fill it out with Claude's help
> Help me complete specs/experiments/automate_[task].md for my task:
> "[Describe your repetitive task]"
> 
> Be very specific about:
> - Input requirements
> - Step-by-step process  
> - Expected outputs
> - How to verify correctness
> - Edge cases to handle
```

### Step 5.3: Write Tests First (10 min)

```bash
# Create test file
> Create tests/test_automated_[task].py with:
> - Tests for normal operation
> - Tests for edge cases
> - Tests for expected failures
> - Verification that outputs match spec
> 
> Use the specification as the source of truth
```

### Step 5.4: Implement with Verification (15 min)

```bash
# Implement to pass tests
> Implement scripts/automated_[task].py following the spec
> Make all tests pass
> Add comprehensive logging
> Include verification at each step

# Create a slash command
/crud-claude-commands create run-[task]

# Test iteration cycle
/run-[task]
# If it fails, refine:
> The task failed at step X. Let's debug and fix...
```

**Success Criteria**: The task runs reliably 3 times in a row without intervention.

---

## 🚀 Part 6: Implement Something New (45 min)

### Step 6.1: Multi-Model Planning (10 min)

Now let's build something that doesn't exist yet. Use multiple models for planning:

```bash
# First, describe your idea
> I want to implement: [describe new feature/capability]

# Use different models for perspectives
# If you have access to other models:
gemini -p "@[codebase] How would you architect [feature]?"

# In Claude, use advanced thinking
think harder about the architecture for this feature

# Create implementation plan
/plan-with-context implement [feature]
```

### Step 6.2: Create Comprehensive Specification (10 min)

```bash
# Create new feature spec
> Create specs/features/new_[feature].md with:
> - Detailed problem statement
> - Technical approach with phases
> - Integration points with existing code
> - Risk assessment
> - Success criteria
```

### Step 6.3: Iterative Implementation (25 min)

Use Claude Code's advanced features:

```bash
# Start with todo list
> Based on the spec, create a detailed todo list for implementing [feature]
> Break it into 30-minute chunks

# Implement phase by phase
> Let's start with Phase 1 from the spec
> Update the todo list as we complete items
> Add verification after each component

# Use multiple terminals/agents if needed
# Terminal 1: Implementation
# Terminal 2: Running tests
# Terminal 3: Monitoring logs

# Regular checkpoints
/page feature-checkpoint-1
```

### Key Practices:
- **Tight Leash**: Review every significant change
- **Continuous Verification**: Test as you build
- **Todo Tracking**: Keep the list updated
- **Regular Commits**: Save progress frequently

---

## 🌐 Part 7: (Optional) Observability & Verification (20 min)

### Step 7.1: Design Simple Dashboard (10 min)

```bash
# Create monitoring script
> Create scripts/experiment_dashboard.py that:
> - Shows current experiment status
> - Displays key metrics in real-time
> - Compares with baselines
> - Flags anomalies
> 
> Keep it simple - terminal output is fine
```

### Step 7.2: Verification Webapp (10 min)

```bash
# If time permits, create web interface
> Create a simple FastAPI app in scripts/verification_app.py that:
> - Shows experiment results
> - Allows quick verification of outputs
> - Compares different runs
> - Exports reports
```

---

## 💾 Session Management Throughout

### Use These Commands Regularly:

```bash
# Check context usage every 30 min
/compact

# Save state at major milestones  
/page task-automation-complete
/page new-feature-phase-1

# Clear when needed
/clear

# Resume previous work
claude --resume task-automation-complete
```

### Git Best Practices:

```bash
# Commit after each successful step
git add -A
git commit -m "Complete task automation for [task]"

# Create branches for major work
git checkout -b automate-experiments
git checkout -b implement-new-feature
```

---

## 🎯 Workshop Success Checklist

### By End of Workshop:
- [ ] Comfortable with all Claude Code commands
- [ ] Created comprehensive `ai_docs/` documentation
- [ ] Generated tailored CLAUDE.md 
- [ ] Automated one repetitive task reliably
- [ ] Created custom slash commands
- [ ] Implemented new feature with planning
- [ ] Used multiple thinking modes
- [ ] Managed context with `/page` and `/compact`
- [ ] Everything committed to git

### Your New Capabilities:
- Run experiments 10x faster
- Implement ideas with AI assistance
- Maintain research rigor with verification
- Scale your research workflow

---

## 📚 Command Reference

### Essential Commands Used Today
```bash
/setup                    # Initial setup wizard
/crud-claude-commands     # Create custom commands
/page <name>             # Save session state
/plan-with-context       # Smart planning
/compact                 # Check context usage
/clear                   # Reset context

# Thinking modes
think                   # Basic reasoning
think hard              # More thorough analysis
think harder            # Complex problems
ultrathink              # Maximum depth
```

### Your Custom Commands
```bash
/run-[your-task]        # Your automated task
/verify-[your-output]   # Your verification  
# ... others you created
```

---

## 🚀 Next Steps

1. **Tomorrow**: Run your automated task on real data
2. **This Week**: 
   - Create 2 more task automations
   - Refine your new feature implementation
   - Share your commands with your team
3. **This Month**: 
   - Build a suite of research tools
   - Create team-wide command library
   - Optimize your entire workflow

Remember: You're now equipped to conduct an AI orchestra for your research. Keep building, keep verifying, keep pushing boundaries!
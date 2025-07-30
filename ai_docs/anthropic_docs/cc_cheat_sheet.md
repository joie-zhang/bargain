# AI Research Cheat Sheet: Claude Code & Agentic Workflows

## 🚀 Quick Start Templates

### 1. Research Project CLAUDE.md Template
```markdown
# Research Project: [PROJECT_NAME]

## Current Hypothesis
[Your main research question/hypothesis]

## Key Papers & Context
- [Paper 1]: [Key insight/relevance]
- [Paper 2]: [Contradicts/supports X]

## Experimental Setup
- Dataset: [location/description]
- Baseline code: [repo/path]
- Metrics: [what you're measuring]
- Hardware: [GPU/compute requirements]

## Failed Attempts (IMPORTANT)
- Don't try X because Y
- Model crashes with batch_size > N
- [Specific pitfall] leads to [problem]

## Useful Commands
- Run experiment: `python train.py --config configs/base.yaml`
- Analyze results: `python analyze.py --exp_dir logs/`
- Test single batch: `python debug.py --single_batch`

## Progress Tracking
- [x] Baseline reproduction
- [ ] Hypothesis 1 test
- [ ] Ablation studies
```

### 2. Speech-to-Text Research Workflow
```bash
# Record your thoughts about research
# Using Super Whisper/similar app

"So I'm thinking about this mechanistic interpretability problem...
The hypothesis is that attention heads in layer 12 are doing...
I need to test this by looking at activations when...
The key papers that relate are Smith et al 2023 which found..."

# Then create a research plan:
> Create a detailed research plan from this transcript. 
> Focus on testable hypotheses and specific experiments.
```

## 🛠️ Essential Slash Commands

### Built-in Research Commands
```bash
/init                 # Generate CLAUDE.md for your project
/clear               # Clear context between experiments
/compact "focus on experiment results"  # Compress context
/memory              # Edit CLAUDE.md files
/cost                # Track token usage (important for budgets!)
```

### Creating Custom Research Commands

**Location**: `.claude/commands/[command-name].md`

#### Template: Literature Review Command
```markdown
---
allowed-tools: WebSearch, WebFetch, Read, Write
description: Comprehensive literature review for research topic
---
# Literature Review: $ARGUMENTS

1. Search arXiv for papers from last 2 years on: $ARGUMENTS
2. For each relevant paper (limit 10):
   - Extract key methods and findings
   - Note experimental setup
   - Identify evaluation metrics
3. Create comparison table of approaches
4. Identify gaps in current research
5. Write summary with citations in papers/lit_review_$ARGUMENTS.md
6. Suggest 3 potential research directions based on gaps
```
**Usage**: `/project:lit-review "mechanistic interpretability of vision models"`

#### Template: Experiment Analysis
```markdown
---
allowed-tools: Read, Write, Bash
description: Analyze experiment results and generate report
---
# Analyze Experiment: $ARGUMENTS

Current experiment logs: !`ls -la logs/`

1. Read the latest experiment config from logs/$ARGUMENTS/config.yaml
2. Load training metrics from logs/$ARGUMENTS/metrics.json
3. Generate plots:
   - Loss curves (train vs val)
   - Key metrics over time
   - Any model-specific visualizations
4. Compare with baseline results if available
5. Statistical analysis:
   - Confidence intervals
   - Significance tests vs baseline
6. Write comprehensive report in results/$ARGUMENTS_analysis.md
7. Suggest next experiments based on results
```
**Usage**: `/project:analyze-exp "attention_ablation_v3"`

#### Template: Multi-Model Comparison
```markdown
---
allowed-tools: Read, Bash, Write
description: Compare multiple model checkpoints/experiments
---
# Compare Models: $ARGUMENTS

Models to compare: $ARGUMENTS (comma-separated)

1. For each model checkpoint:
   - Load model config
   - Run evaluation on test set
   - Collect performance metrics
2. Generate comparison visualizations
3. Statistical significance tests between models
4. Create LaTeX table for paper
5. Write summary of findings
```
**Usage**: `/project:compare-models "baseline,attention_only,full_model"`

#### Template: Debug Training Run
```markdown
---
allowed-tools: Read, Bash, Grep
description: Debug why training crashed or produced bad results
---
# Debug Training Issue

Recent logs: !`tail -n 100 logs/latest/train.log`
Error traces: !`grep -i "error\|exception\|nan\|inf" logs/latest/*.log`

1. Identify the error type and location
2. Check for common issues:
   - NaN/Inf in gradients
   - Memory overflow
   - Data loading errors
   - Shape mismatches
3. Examine the model state before crash
4. Suggest fixes with code examples
5. Create minimal reproduction script if needed
```
**Usage**: `/project:debug-training`

## 🔄 Multi-Agent Research Workflows

### Parallel Analysis Setup
```bash
# Terminal 1: Main experiment
cd ~/research/project
claude --dangerously-skip-permissions
> Run the ablation studies in experiments/ablations.yaml

# Terminal 2: Real-time analysis  
cd ~/research/project
claude
> Monitor logs/latest/ and create visualizations every 100 steps

# Terminal 3: Literature monitor
cd ~/research/project  
claude
> Search for new papers on arXiv related to our method and summarize
```

### Git Worktrees for Parallel Experiments
```bash
# Set up parallel experiment branches
git worktree add ../exp-lr-sweep experiments/lr_sweep
git worktree add ../exp-architecture experiments/arch_changes
git worktree add ../exp-data-aug experiments/augmentation

# Run different experiments in parallel
cd ../exp-lr-sweep && claude "Run learning rate sweep from 1e-5 to 1e-2"
cd ../exp-architecture && claude "Test the 3 architecture variants"
```

## 📊 Research-Specific Aliases & Functions

### Add to your ~/.bashrc or ~/.zshrc
```bash
# Quick experiment launcher
alias claude-exp='claude --dangerously-skip-permissions'

# Research-focused Claude with max context
alias claude-research='claude --model opus --max-output-tokens 4096'

# Quick paper summary
claude-paper() {
    claude -p "Summarize this paper focusing on method, results, and relevance to our work" < "$1"
}

# Experiment comparison
claude-compare() {
    claude "Compare experiments in $1 and $2, focus on metrics and significance"
}
```

## 🧪 Quick Research Tasks

### 1. Start New Research Project
```bash
claude "Set up a new research project for [TOPIC]. Create:
1. Project structure with train/eval/analysis scripts  
2. CLAUDE.md with research context
3. Initial experiment configs
4. Data loading pipeline
5. Evaluation metrics
6. Baseline implementation"
```

### 2. Reproduce Paper Results
```bash
claude "Help me reproduce the results from [PAPER_ARXIV_ID]:
1. Read the paper (I'll provide PDF)
2. Identify key implementation details
3. Create config files matching paper settings
4. Implement any missing components
5. Set up evaluation exactly as in paper"
```

### 3. Quick Ablation Study
```bash
claude "Design ablation study for our model:
1. List all architectural components
2. Create configs for removing each
3. Generate script to run all ablations
4. Create analysis notebook template
5. Set up significance testing"
```

### 4. Emergency Debugging
```bash
# When experiments crash at 3am
claude "My training crashed with error: [ERROR]
1. Check the full logs in logs/latest/
2. Identify root cause
3. Suggest immediate fix
4. Create checkpoint recovery script
5. Implement better error handling"
```

## 💡 Pro Tips for Researchers

### Context Management
- Use `/clear` between unrelated tasks
- Keep separate Claude Projects for different research areas
- Use `@` to reference specific files: `@configs/baseline.yaml`
- Create .claude/commands/ for repeated workflows

### Efficient Workflows
1. **Morning standup**: `claude "Summarize yesterday's experiments and suggest today's priorities"`
2. **Before meetings**: `claude "Prepare update on [PROJECT] for advisor meeting"`
3. **Paper writing**: `claude "Convert experiment results into paper section with citations"`

### Token-Saving Patterns
```bash
# Instead of dumping entire files
> Analyze the model architecture in models/transformer.py, focus on attention

# Use grep first
> grep -n "class\|def" models/*.py | head -20  # See structure
> Now implement the missing forward pass in models/new_layer.py
```

## 🚨 Research-Specific Settings

### .claude/settings.json
```json
{
  "permissions": {
    "allow": [
      "Bash(python train.py:*)",
      "Bash(python analyze.py:*)", 
      "Bash(tensorboard:*)",
      "Bash(nvidia-smi:*)",
      "Bash(git:*)",
      "Write(logs/*)",
      "Write(results/*)"
    ],
    "deny": [
      "Bash(rm -rf:*)",
      "Write(data/*)"  // Protect datasets
    ]
  },
  "env": {
    "CUDA_VISIBLE_DEVICES": "0",
    "PYTHONPATH": "${PYTHONPATH}:${PWD}"
  }
}
```

## 🎯 The 80/20 Research Commands

These 5 commands will handle 80% of your research needs:

1. **Start day**: `claude "What experiments are running? What needs attention?"`
2. **Analyze results**: `/project:analyze-exp [experiment_name]`
3. **Debug issues**: `claude "Debug why [SPECIFIC ISSUE] is happening"`
4. **Literature check**: `claude "Find recent papers that cite [KEY PAPER] or use [METHOD]"`
5. **Write it up**: `claude "Convert these results into a paper section with proper citations"`

---

**Remember**: "Everything just falls out" when you use speech-to-text. Record your research thoughts, hypotheses, and meeting notes. Let Claude help you turn them into actionable research plans and code.
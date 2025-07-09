# [Research Task Name] Specification

**Date**: [YYYY-MM-DD]  
**Author**: [Your Name]  
**Status**: Draft / In Progress / Complete  
**Priority**: High / Medium / Low  

## 1. Problem Statement

### Research Question
[What specific question are you trying to answer? Be precise and measurable.]

### Background
[Brief context about why this matters. Include relevant prior work or failed attempts.]

### Success Criteria
- [ ] [Specific, measurable outcome 1]
- [ ] [Specific, measurable outcome 2]
- [ ] [Specific, measurable outcome 3]

## 2. Technical Approach

### Method Overview
[High-level description of your approach in 2-3 paragraphs]

### Key Components
1. **[Component 1]**: [Brief description]
2. **[Component 2]**: [Brief description]
3. **[Component 3]**: [Brief description]

### Implementation Details
```python
# Pseudocode or key algorithmic steps
def main_approach():
    # Step 1: Data preparation
    # Step 2: Model/algorithm implementation
    # Step 3: Evaluation
    pass
```

## 3. Data Requirements

### Input Data
- **Source**: [Where does the data come from?]
- **Format**: [File types, structure]
- **Size**: [Approximate scale]
- **Preprocessing**: [Required transformations]

### Output Data
- **Format**: [Expected output structure]
- **Validation**: [How to verify correctness]
- **Storage**: [Where results will be saved]

## 4. Experimental Design

### Baseline Comparisons
- **Baseline 1**: [Description and why it's relevant]
- **Baseline 2**: [Description and why it's relevant]

### Evaluation Metrics
1. **Primary Metric**: [e.g., accuracy, loss, specific domain metric]
   - Target: [Specific number/range]
   - Current best: [If known]

2. **Secondary Metrics**: 
   - [Metric 1]: [Target]
   - [Metric 2]: [Target]

### Ablation Studies
- [ ] [Component to ablate 1]
- [ ] [Component to ablate 2]
- [ ] [Parameter sensitivity analysis]

## 5. Implementation Plan

### Phase 1: Setup and Validation (Time estimate: [X hours/days])
- [ ] Set up development environment
- [ ] Implement data loading pipeline
- [ ] Create simple validation test
- [ ] Verify baseline reproduction

### Phase 2: Core Implementation (Time estimate: [X hours/days])
- [ ] Implement main algorithm/model
- [ ] Add logging and monitoring
- [ ] Create unit tests
- [ ] Initial debugging and optimization

### Phase 3: Experimentation (Time estimate: [X hours/days])
- [ ] Run full experiments
- [ ] Perform ablation studies
- [ ] Analyze results
- [ ] Generate visualizations

### Phase 4: Analysis and Documentation (Time estimate: [X hours/days])
- [ ] Statistical analysis of results
- [ ] Create figures and tables
- [ ] Write up findings
- [ ] Prepare reproducibility package

## 6. Technical Constraints

### Computational Resources
- **Memory**: [RAM requirements]
- **GPU**: [Required? Type?]
- **Storage**: [Disk space needed]
- **Time**: [Expected runtime per experiment]

### Dependencies
- **Core Libraries**: [e.g., PyTorch, NumPy, etc.]
- **Special Requirements**: [e.g., specific versions, proprietary code]
- **External APIs**: [If any]

## 7. Risks and Mitigations

### Technical Risks
1. **Risk**: [e.g., "Model might not converge"]
   - **Mitigation**: [e.g., "Use learning rate scheduling, try different optimizers"]

2. **Risk**: [e.g., "Data might be insufficient"]
   - **Mitigation**: [e.g., "Prepare data augmentation strategies"]

### Research Risks
1. **Risk**: [e.g., "Approach might not outperform baseline"]
   - **Mitigation**: [e.g., "Have alternative approaches ready"]

## 8. Self-Validation Checklist

### Before Starting
- [ ] All dependencies installed and versions recorded
- [ ] Test data pipeline with small sample
- [ ] Baseline code runs and reproduces expected results
- [ ] Git repository initialized with .gitignore

### During Implementation
- [ ] Each function has a unit test
- [ ] No hardcoded paths or magic numbers
- [ ] Logging captures all important metrics
- [ ] Regular commits with descriptive messages

### After Completion
- [ ] All experiments reproducible with single command
- [ ] Results saved with timestamps and configs
- [ ] Statistical significance tested
- [ ] Code passes linting and type checking

## 9. Success Metrics & Verification

### Quantitative Success
- [ ] Achieves [X]% improvement over baseline
- [ ] Runs in less than [Y] hours
- [ ] Uses less than [Z] GB memory

### Qualitative Success
- [ ] Code is readable and well-documented
- [ ] Results are interpretable
- [ ] Approach is generalizable to [related problems]

### Verification Commands
```bash
# Test single batch
python test_single_batch.py

# Verify no mock data
grep -r "torch.randn\|np.random" src/ --include="*.py"

# Run full validation
python validate_results.py --compare-baseline

# Check reproducibility
python check_reproducibility.py --seed 42
```

## 10. Next Steps

### If Successful
1. [What to do if approach works]
2. [How to extend or scale up]
3. [Publication or deployment plans]

### If Unsuccessful
1. [Alternative approach 1]
2. [Alternative approach 2]
3. [What can be learned from failure]

---

## Notes for AI Agents

**Key Files to Reference**:
- Main implementation: `[path/to/main.py]`
- Config file: `[path/to/config.yaml]`
- Data processing: `[path/to/data.py]`

**Common Pitfalls to Avoid**:
- Don't use synthetic data in actual experiments
- Always set random seeds for reproducibility
- Log everything - you'll need it for debugging
- Test edge cases (empty inputs, single samples, etc.)

**Workspace Setup**:
```bash
# Initial setup commands
cd [project-root]
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

---

*This specification is a living document. Update it as you learn more about the problem and refine your approach.*
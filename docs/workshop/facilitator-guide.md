# AI Agent Research Workshop: Facilitator Guide

## 🎯 Workshop Philosophy
**"Guide, don't do."** You're teaching researchers to become conductors of AI orchestras. Focus on empowering them to use the tools independently rather than solving problems for them.

---

## 📅 Pre-Workshop Preparation

### 1 Week Before
- [ ] Test the template repository with a sample research project
- [ ] Ensure all participants have received setup instructions
- [ ] Create a shared communication channel (Slack/Discord)
- [ ] Prepare 2-3 example research domains for demos

### 1 Day Before
- [ ] Verify template repository is accessible
- [ ] Test `/setup` command with fresh install
- [ ] Prepare backup solutions for common issues
- [ ] Have example outputs ready to show

### Day Of
- [ ] Arrive 30 min early to test equipment
- [ ] Have template URL ready to share
- [ ] Open communication channel for questions
- [ ] Prepare timer for session management

---

## 🕐 Workshop Timeline & Key Points

### Opening (10 min)
**Key Messages:**
- "This template is powerful but requires your domain knowledge"
- "We'll use existing tools, not build from scratch"
- "Focus on one concrete research task today"
- "Questions in chat, I'll address during work time"

**Quick Demo** (keep under 3 min):
```bash
# Show the setup wizard in action
claude
/setup
# Just show the first few prompts, don't complete
```

---

### Part 0: Setup Wizard (15 min)

**Common Issues & Solutions:**

| Issue | Solution |
|-------|----------|
| "Command not found" | Check PATH, restart terminal |
| "Git errors" | Ensure clean working directory |
| "Setup hangs" | Ctrl+C and restart Claude |
| "Confused by prompts" | Have them start simple, refine later |

**Facilitator Actions:**
- Circulate and ensure everyone started `/setup`
- Don't let anyone skip ahead
- Encourage specific answers about research domain
- Remind them this customizes everything

**Say:** "The setup wizard is your foundation. Be specific about your research - the more context you provide, the better your AI assistants will perform."

---

### Part 1: Context Gathering (45 min)

**Critical Success Factors:**
1. **Push for 15+ minute recordings** - Most stop at 5 minutes
2. **Encourage storytelling** - "Explain like to a new lab member"
3. **Capture failures** - "What didn't work is gold for AI context"

**🚨 Red Flags to Watch For:**
- Recording under 10 minutes = too shallow
- No mention of failures = missing critical context
- Too abstract = need specific examples
- No papers uploaded = missing foundation

**Intervention Prompts:**
- "Tell me about your last debugging session"
- "What would you check if results look wrong?"
- "What assumptions might a newcomer miss?"
- "Describe your typical experimental workflow"

**Mini-Demo** (if needed):
Show example of good context:
```
"I'm working on mechanistic interpretability, specifically trying to understand 
how transformer attention heads specialize. My hypothesis is that early heads 
focus on syntax while later heads capture semantics. I've been building on 
the Anthropic paper about superposition, but I keep running into issues where 
my sparse autoencoders collapse to trivial solutions when I increase the 
sparsity penalty beyond 0.01..."
```

---

### Part 2: Specification Writing (30 min)

**Key Teaching Points:**
1. Specs are living documents - start simple, iterate
2. Success criteria must be measurable
3. Include verification strategy upfront
4. Break implementation into 2-hour chunks

**Common Struggles:**

| Problem | Guidance |
|---------|----------|
| "Too ambitious" | "What's the smallest testable piece?" |
| "Too vague" | "How would you know it worked?" |
| "No verification" | "What could go wrong? How would you check?" |
| "All-or-nothing" | "Can you break this into phases?" |

**Show Example Spec Structure:**
```markdown
## Problem Statement
ONE clear sentence about what needs to be solved

## Success Criteria
- [ ] Specific measurable outcome 1
- [ ] Specific measurable outcome 2

## Implementation Phases
Phase 1 (30 min): Minimal working version
Phase 2 (30 min): Add verification
Phase 3 (30 min): Integrate with pipeline
```

---

### Part 3: Custom Commands (30 min)

**Teaching Strategy:**
1. Start with their most annoying repetitive task
2. Show `/crud-claude-commands` workflow
3. Emphasize verification in commands
4. Test immediately after creation

**Live Coding Demo** (5 min):
```bash
# Show creating a simple command
/crud-claude-commands create check-gradients

# In the creation process, emphasize:
# - Clear role definition
# - Step-by-step workflow
# - Verification at each step
# - Error handling
```

**Common Command Patterns for Research:**
- `run-ablation` - Parameter sweeps
- `verify-metrics` - Check outputs
- `compare-baseline` - Benchmark results
- `debug-numerics` - Find NaN/inf issues
- `paper-figure` - Generate publication plots

**If Someone Struggles:**
Pair them with Claude in plan mode:
```
think step by step about what commands would help my research workflow
```

---

### Part 4: Implementation (45 min)

**Critical: Don't Let Them Skip Verification!**

**Pacing Guide:**
- 0-15 min: Set up verification infrastructure
- 15-35 min: Implement with verification loops
- 35-45 min: Test and debug

**Key Interventions:**
1. **If coding without verification:** "How will you know this works correctly?"
2. **If stuck on complex design:** "What's the simplest version?"
3. **If frustrated with errors:** "Let's create a minimal test case"
4. **If ahead of schedule:** "Add another verification layer"

**Show Verification Pattern:**
```python
# Every implementation should follow:
def my_research_function(data):
    # 1. Input validation
    assert data.shape[0] > 0, "Empty data"
    
    # 2. Core logic
    result = process(data)
    
    # 3. Output verification
    assert not torch.isnan(result).any(), "NaN in output"
    assert result.shape == expected_shape, "Wrong shape"
    
    # 4. Sanity check
    assert result.mean() > 0, "Suspiciously low values"
    
    return result
```

---

### Part 5: Multi-Agent Analysis (30 min)

**This is Advanced - Guide Based on Skill Level:**

**Beginners:** Focus on using existing `/parallel-analysis-example`
**Intermediate:** Adapt the example for their domain
**Advanced:** Create custom multi-agent workflows

**Demo the Concept** (3 min):
```bash
# Show how multiple agents provide different perspectives
/parallel-analysis-example "Should I use dropout or batch norm?"

# Explain the agents:
# - Architecture expert looks at model design
# - Theory expert considers mathematical properties
# - Empirical expert checks experimental evidence
# - Implementation expert considers practical aspects
```

**Common Use Cases for Researchers:**
- Literature review synthesis
- Experimental design validation
- Result interpretation from multiple angles
- Debugging complex issues

---

### Part 6: Wrap-up (15 min)

**Essential Actions:**
1. **Force `/page` command** - Many skip this
2. **Require git commit** - Evidence of progress
3. **Document one win** - Build confidence
4. **Plan next step** - Maintain momentum

**Closing Circle Questions:**
- "What's one command you'll use tomorrow?"
- "What surprised you about AI assistance?"
- "What task will you automate next?"

---

## 🚨 Common Issues & Solutions

### Technical Issues

| Problem | Quick Fix | Prevention |
|---------|-----------|------------|
| Claude context full | `/page` then `/clear` | Monitor with `/compact` |
| Git merge conflicts | Create new branch | Always branch before workshop |
| Python errors | Check virtual env | Test imports upfront |
| Command not found | Restart Claude | Verify PATH setup |

### Conceptual Issues

| Problem | Intervention | Example |
|---------|-------------|---------|
| "Too complex task" | Break down smaller | "Just load and plot data first" |
| "Doesn't trust AI" | Show verification | "See how we check every step?" |
| "Over-engineering" | YAGNI principle | "What do you need TODAY?" |
| "Analysis paralysis" | Time box | "5 minutes to decide, then commit" |

### Pacing Issues

**If Running Behind:**
- Skip multi-agent section
- Simplify implementation task
- Focus on one custom command
- Ensure everyone does `/page`

**If Running Ahead:**
- Add more verification layers
- Create additional commands
- Try multi-agent patterns
- Help others debug

---

## 📊 Success Metrics

Track these during workshop:

1. **Participation Metrics:**
   - [ ] Everyone completed `/setup`
   - [ ] Everyone recorded 10+ min context
   - [ ] Everyone created 1+ custom command
   - [ ] Everyone has working verification

2. **Quality Metrics:**
   - [ ] Specifications have measurable criteria
   - [ ] Commands include error handling
   - [ ] Implementation has verification
   - [ ] Git commits are meaningful

3. **Engagement Metrics:**
   - [ ] Active problem-solving (not just copying)
   - [ ] Asking domain-specific questions
   - [ ] Helping other participants
   - [ ] Planning next steps

---

## 🎯 Facilitator Cheat Sheet

### Quick Commands to Share
```bash
# Check context usage
/compact

# Save state
/page workshop-checkpoint

# Clear and start fresh
/clear

# Get unstuck
think step by step about why this might be failing
```

### Motivational Phrases
- "Perfect is the enemy of good - ship something working"
- "Verification builds confidence"
- "Context is king - add more detail"
- "Start simple, iterate fast"
- "You're not replacing thinking, you're amplifying it"

### Time Warnings
- 10 min remaining: "Start wrapping up current section"
- 5 min remaining: "Save your work"
- 1 min remaining: "Commit what you have"

---

## 📝 Post-Workshop

### Immediate Actions (same day):
1. **Collect Feedback**
   - What was most valuable?
   - What was confusing?
   - What do they want to learn next?

2. **Share Resources**
   - Link to participant guide
   - Example implementations
   - Community channel

3. **Follow-up Message Template:**
   ```
   Great work today! You've set up a powerful research environment.
   
   Next steps:
   1. Run one real experiment with your new commands
   2. Create 2 more custom commands this week
   3. Share your success in #ai-research-assistants
   
   Resources:
   - Workshop guide: [link]
   - Template docs: [link]
   - Office hours: [schedule]
   ```

### Within 1 Week:
- Review participant repos
- Share success stories
- Plan advanced workshop
- Update materials based on feedback

---

## 🔑 Remember

You're not teaching them to code - you're teaching them to **conduct an AI orchestra for research**. Focus on:

1. **Empowerment over solutions** - Guide them to find answers
2. **Patterns over specifics** - Teach approaches, not just commands
3. **Verification over speed** - Correct is better than fast
4. **Context over prompts** - More context = better results
5. **Iteration over perfection** - Ship early, improve often

The best workshop is one where participants leave feeling capable of continuing on their own!
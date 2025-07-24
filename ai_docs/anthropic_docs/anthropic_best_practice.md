# Anthropic Best Practices Guide
*A comprehensive distillation of Anthropic's documentation for building with Claude*

## 🎯 Core Philosophy

**Always start simple, then add complexity only when needed.** Claude performs best with clear, direct instructions and well-structured prompts. Think of Claude as a brilliant new employee who needs explicit context and guidance.

## 📋 Quick Decision Tree

1. **Single LLM call sufficient?** → Use basic prompting
2. **Need multiple steps?** → Use workflows (prompt chaining, routing, parallelization)
3. **Need dynamic decision-making?** → Use agents
4. **Need real-time info?** → Add web search tool
5. **Need file operations?** → Add text editor tool

---

## 🤖 Model Selection Strategy

### Latest Models (2025)
- **Claude Opus 4** (`claude-opus-4-20250514`): Most capable, complex reasoning, AI safety research
- **Claude Sonnet 4** (`claude-sonnet-4-20250514`): High performance coding, balanced capability
- **Claude Sonnet 3.7** (`claude-3-7-sonnet-20250219`): Extended thinking, high intelligence
- **Claude Haiku 3.5** (`claude-3-5-haiku-20241022`): Fastest, efficient tasks

### Selection Criteria
```
Complex reasoning + AI Safety → Claude Opus 4
Coding projects → Claude Sonnet 4  
Extended analysis → Claude Sonnet 3.7
Quick tasks → Claude Haiku 3.5
Long context (200K tokens) → Any current model
Cost optimization → Haiku 3.5
```

**Context Windows:** All current models support 200K tokens (~150K words)

---

## 🎨 Prompt Engineering Hierarchy
*Apply these techniques in order of importance*

### 1. Be Clear, Direct, and Detailed
**Golden Rule:** Show your prompt to a colleague with minimal context. If they're confused, Claude will be too.

**Essential Elements:**
- **Context:** What the task results will be used for, target audience, workflow position
- **Specificity:** Exact desired output format, constraints, requirements
- **Sequential steps:** Use numbered lists or bullet points
- **Success criteria:** What a successful completion looks like

**Example: Customer Feedback Analysis**
```python
# ❌ Vague prompt
"Analyze this customer feedback and categorize the issues."

# ✅ Clear, detailed prompt
"""
You are analyzing customer feedback for our product team's quarterly review.

TASK: Categorize each piece of feedback and extract actionable insights.

CATEGORIES: UI/UX, Performance, Feature Request, Integration, Pricing, Other
SENTIMENT: Positive, Neutral, Negative
PRIORITY: High (impacts core functionality), Medium (enhancement), Low (nice-to-have)

OUTPUT FORMAT:
For each feedback item:
- ID: [number]
- Category: [category] 
- Sentiment: [sentiment]
- Priority: [priority]
- Summary: [one sentence]
- Action Required: [specific next step]

FEEDBACK TO ANALYZE:
{feedback_data}
"""
```

### 2. Use Examples (Multishot Prompting)
**Power Rule:** Include 3-5 diverse, relevant examples for dramatically improved accuracy.

**Example Structure:**
```xml
<examples>
<example>
Input: [example input]
Output: [desired output format]
</example>
</examples>
```

**Real Example: Email Classification**
```python
system_prompt = """
You are an expert email classifier for customer support. Classify emails into categories.

<examples>
<example>
Input: "Hi, I can't log into my account. It says my password is wrong but I'm sure it's correct. Can you help?"
Output: {
  "category": "Technical Support",
  "urgency": "Medium",
  "sentiment": "Frustrated",
  "requires_escalation": false,
  "suggested_response_time": "2 hours"
}
</example>

<example>
Input: "URGENT!!! Your billing system charged me twice this month and I need a refund IMMEDIATELY or I'm canceling!"
Output: {
  "category": "Billing",
  "urgency": "High", 
  "sentiment": "Angry",
  "requires_escalation": true,
  "suggested_response_time": "30 minutes"
}
</example>

<example>
Input: "Love the new dashboard update! The performance improvements are amazing. Keep up the great work!"
Output: {
  "category": "Product Feedback",
  "urgency": "Low",
  "sentiment": "Positive", 
  "requires_escalation": false,
  "suggested_response_time": "24 hours"
}
</example>
</examples>
"""
```

**Make examples:**
- **Relevant:** Mirror your actual use case
- **Diverse:** Cover edge cases and variations
- **Clear:** Wrapped in XML tags for structure

### 3. Let Claude Think (Chain of Thought)
Add phrases like:
- "Think step by step"
- "Before answering, consider..."
- "Let me work through this systematically"

**Advanced Chain of Thought Examples:**
```python
# For complex analysis
"""
Analyze this code for potential security vulnerabilities.

Before providing your final assessment, please:
1. First, read through the entire code to understand its purpose
2. Identify each input point and how data flows through the system
3. Consider common vulnerability patterns (SQL injection, XSS, etc.)
4. Evaluate each potential risk for likelihood and impact
5. Then provide your final recommendations

CODE:
{code_to_analyze}
"""

# For decision-making
"""
Help me choose between these three cloud providers for our startup.

Please think through this systematically:
1. What are our specific technical requirements?
2. What's our budget constraint and growth projection?
3. How does each provider address our requirements?
4. What are the hidden costs and lock-in risks?
5. Which provides the best value for our specific situation?

REQUIREMENTS: {requirements}
PROVIDERS: {provider_options}
"""
```

### 4. Use XML Tags for Structure
```xml
<instructions>
Your main task instructions
</instructions>

<context>
Background information
</context>

<examples>
Your examples here
</examples>

<output_format>
Specify exact format needed
</output_format>
```

**Advanced XML Structuring:**
```python
complex_prompt = """
<role>
You are a senior software architect with 15+ years of experience in distributed systems.
</role>

<context>
We're designing a microservices architecture for an e-commerce platform expecting 1M+ daily users.
Current pain points: monolithic architecture, scaling bottlenecks, deployment complexity.
</context>

<constraints>
- Must support 10x traffic growth
- 99.9% uptime requirement
- Budget: $50K/month infrastructure
- Team size: 12 developers
- Existing tech stack: Python, PostgreSQL, Redis
</constraints>

<task>
Design a microservices migration strategy with specific implementation phases.
</task>

<output_format>
1. Architecture Overview (diagram description)
2. Service Boundaries (list with responsibilities)
3. Migration Phases (timeline with milestones)
4. Technology Recommendations (with rationale)
5. Risk Assessment (top 3 risks with mitigation)
6. Success Metrics (measurable KPIs)
</output_format>

<examples>
[Include 1-2 similar architecture examples]
</examples>
"""
```

### 5. System Prompts for Role Assignment
```python
system_prompt = """You are an expert [ROLE] with [X] years of experience in [DOMAIN]. 
Your approach is [STYLE/METHODOLOGY]. You always [KEY_BEHAVIORS]."""
```

**Specialized Role Examples:**

```python
# Data Scientist Role
data_scientist_prompt = """
You are a senior data scientist with 8+ years of experience in machine learning and statistical analysis. 
Your approach is rigorous and evidence-based, always starting with exploratory data analysis. 
You always:
- Question data quality and potential biases
- Validate assumptions with statistical tests
- Explain complex concepts in business terms
- Recommend the simplest model that meets requirements
- Consider ethical implications of model decisions
"""

# Technical Writer Role  
tech_writer_prompt = """
You are an expert technical writer with 10+ years creating developer documentation.
Your approach prioritizes clarity and user empathy over technical showing-off.
You always:
- Start with the user's goal and context
- Use progressive disclosure (simple to complex)
- Include realistic, runnable examples
- Anticipate common mistakes and edge cases
- Test instructions with someone unfamiliar with the topic
"""

# Product Manager Role
pm_prompt = """
You are a senior product manager with 7+ years at B2B SaaS companies.
Your approach is data-driven but balanced with qualitative user insights.
You always:
- Frame features in terms of user problems and business outcomes
- Consider technical feasibility and resource constraints
- Think about metrics and measurement before building
- Prioritize ruthlessly based on impact and effort
- Communicate trade-offs clearly to stakeholders
"""
```

### 6. Prefill Claude's Response
```python
messages = [
    {"role": "user", "content": "Analyze this data..."},
    {"role": "assistant", "content": "I'll analyze this systematically:\n\n1. Data Quality:"}
]
```

**Advanced Prefilling Patterns:**

```python
# For structured output
messages = [
    {"role": "user", "content": "Review this pull request..."},
    {"role": "assistant", "content": "## Code Review\n\n### Summary\n"}
]

# For specific format
messages = [
    {"role": "user", "content": "Create a project plan..."},
    {"role": "assistant", "content": "# Project Plan: [Title]\n\n## Phase 1: Discovery\n**Duration:** "}
]

# For JSON output
messages = [
    {"role": "user", "content": "Extract entities from this text..."},
    {"role": "assistant", "content": "```json\n{\n  \"entities\": ["}
]
```

---

## 🛠️ Advanced Features

### Extended Thinking
**When to use:** Complex reasoning, multi-step analysis, detailed planning

**Key points:**
- Thinking tokens are automatically stripped from context in subsequent turns
- Preserve thinking blocks when using tools (required for tool continuity)
- Available on Claude 4, Sonnet 3.7
- Budget thinking tokens as subset of max_tokens

### Prompt Caching
**Use for:** Large documents, repeated tool definitions, long conversations

**Structure:**
```python
system=[
    {"type": "text", "text": "Regular instructions"},
    {"type": "text", "text": "Large document content...", 
     "cache_control": {"type": "ephemeral"}}
]
```

**Economics:**
- Cache writes: 25% more expensive than base input
- Cache hits: 90% cheaper than base input
- 5-minute lifetime, refreshed on use
- Minimum: 1024 tokens (Opus/Sonnet), 2048 tokens (Haiku)

### Web Search Tool
**Available on:** Claude 4, Sonnet 3.7, Sonnet 3.5, Haiku 3.5

**Setup:**
```python
tools=[{
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 5,
    "allowed_domains": ["example.com"],  # Optional
    "user_location": {  # Optional
        "type": "approximate",
        "city": "San Francisco",
        "region": "California", 
        "country": "US"
    }
}]
```

**Pricing:** $10 per 1,000 searches + standard token costs

### Text Editor Tool
**Available on:** All current models with version-specific implementations

**Model Versions:**
- Claude 4: `text_editor_20250429` (no undo_edit)
- Sonnet 3.7: `text_editor_20250124` 
- Sonnet 3.5: `text_editor_20241022`

**Commands:** `view`, `str_replace`, `create`, `insert`, `undo_edit` (3.7/3.5 only)

**Security essentials:**
- Validate file paths (prevent directory traversal)
- Create backups before edits
- Ensure exact text matching for replacements

---

## 🏗️ Agentic Systems Architecture

### When to Use Each Pattern

**Prompt Chaining**
- Use for: Decomposable tasks into fixed subtasks
- Example: Generate content → Translate → Review

**Routing** 
- Use for: Distinct categories handled differently
- Example: Customer service queries → specialized workflows

**Parallelization**
- Sectioning: Independent subtasks run in parallel
- Voting: Multiple attempts for higher confidence

**Orchestrator-Workers**
- Use for: Complex tasks with unpredictable subtasks
- Example: Code changes across multiple files

**Evaluator-Optimizer**
- Use for: Iterative refinement with clear evaluation criteria
- Example: Writing with feedback loops

**Autonomous Agents**
- Use for: Open-ended problems, many turns, trusted environments
- Requires: Tool ecosystem, error recovery, human oversight

### Agent Implementation Principles

1. **Maintain Simplicity:** Start with basic patterns
2. **Prioritize Transparency:** Show planning steps explicitly  
3. **Craft Agent-Computer Interface (ACI):** Document tools as carefully as human interfaces

---

## 🔧 Tool Development Best Practices

### Tool Design Philosophy
**Think ACI (Agent-Computer Interface) like HCI (Human-Computer Interface)**

**Key principles:**
- Give model enough tokens to "think" before committing
- Keep format close to natural internet text
- Avoid formatting overhead (line counting, string escaping)
- Include examples, edge cases, clear boundaries

### Tool Documentation Template
```json
{
  "name": "tool_name",
  "description": "Clear, specific description with examples",
  "parameters": {
    "param1": {
      "type": "string",
      "description": "Detailed description with format requirements",
      "examples": ["example1", "example2"]
    }
  },
  "common_mistakes": "What to avoid",
  "success_criteria": "How to know it worked"
}
```

### Error Handling
```python
def tool_wrapper(func):
    try:
        result = func()
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e), "suggestion": "What to try next"}
```

---

## 💰 Cost Optimization

### Token Management
**Context Windows:** All models 200K tokens (~150K words)
**Extended Thinking:** Thinking tokens automatically managed
**Prompt Caching:** 90% savings on cache hits

### Model Selection for Cost
```
Simple tasks → Haiku 3.5 ($0.80/$4 per MTok)
Standard tasks → Sonnet 4 ($3/$15 per MTok)  
Complex reasoning → Opus 4 ($15/$75 per MTok)
```

### Efficiency Strategies
1. **Cache large, reusable content**
2. **Use appropriate model for task complexity**
3. **Implement early stopping for satisfactory results**
4. **Batch similar requests when possible**

---

## 🏢 Production Considerations

### Environment Setup
**Required variables:** See `documentation/claude-quickstart.md` for complete setup

**Key APIs:**
- `ANTHROPIC_API_KEY`: Primary API access
- `GOOGLE_API_KEY`: For Gemini models if needed
- Additional tool APIs as required

### Rate Limiting & Scaling
- **Priority Tier:** Available for Claude 4 and Sonnet 3.7
- **Batch API:** For large-scale processing
- **Streaming:** Use for long outputs to avoid timeouts

### Error Handling Strategy
```python
def robust_claude_call(messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.messages.create(...)
            return response
        except anthropic.RateLimitError:
            wait_time = 2 ** attempt
            time.sleep(wait_time)
        except anthropic.BadRequestError as e:
            # Don't retry bad requests
            raise e
    raise Exception("Max retries exceeded")
```

---

## 📊 Evaluation Framework

### Success Metrics
**Prompt Engineering:**
- Accuracy on test cases
- Consistency across variations  
- Output format compliance
- Task completion rate

**Agentic Systems:**
- End-to-end success rate
- Error recovery effectiveness
- Human intervention frequency
- Cost per successful completion

### Testing Strategy
1. **Start with deterministic test cases**
2. **Add edge cases and variations**
3. **Test with real-world data**
4. **Monitor production performance**
5. **Iterate based on failure analysis**

---

## 🎯 Industry-Specific Applications

### Customer Support (Proven Pattern)
- **Success metric:** Usage-based pricing models
- **Key features:** Tool integration, conversation flow, clear success criteria
- **Architecture:** Routing + Tool use + Human oversight

### Coding Agents (High Potential)
- **Success metric:** Automated test passage
- **Key features:** Code verification, iterative improvement, objective measures
- **Architecture:** Orchestrator-workers + Text editor tool

---

## 📚 Reference Files for Deep Dives

When you need complete details, reference these files:

- **Complete model specifications:** `documentation/claude-models-overview.md`
- **Full prompt engineering guide:** `documentation/prompt-engineering-overview.md`
- **Detailed tool implementations:** `documentation/text-editor-tool.md`
- **Web search complete API:** `documentation/web-search-tool.md`
- **Production deployment:** `documentation/claude-quickstart.md`
- **Agent architecture patterns:** `documentation/effective-agents.md`
- **Caching optimization:** `documentation/prompt-caching.md`
- **Context window management:** `documentation/context-windows.md`
- **Real-world implementations:** `documentation/PARAHELP_PROMPT.md`

---

## 🚀 Quick Start Template

```python
import anthropic

client = anthropic.Anthropic()

# Basic structure
response = client.messages.create(
    model="claude-sonnet-4-20250514",  # Choose appropriate model
    max_tokens=1000,
    temperature=0,  # Deterministic for production
    system="You are an expert assistant...",  # Clear role
    messages=[{
        "role": "user", 
        "content": """
        <instructions>
        [Clear, specific instructions]
        </instructions>
        
        <examples>
        [2-3 relevant examples]
        </examples>
        
        <task>
        [Actual task]
        </task>
        """
    }],
    tools=[...],  # Add tools as needed
)
```

---

## 📝 Advanced Prompt Patterns & Examples

### Pattern 1: Progressive Disclosure
**Use when:** Complex tasks that benefit from step-by-step revelation

```python
progressive_prompt = """
I'm going to give you a complex business problem to solve. I'll provide information in stages, and I want you to ask clarifying questions at each stage before I give you more details.

STAGE 1: The Problem
Our SaaS company's churn rate has increased 40% over the last quarter.

Please analyze what you know so far and ask 3-5 specific questions that would help you understand the root cause. Don't make assumptions - focus on gathering the most critical missing information.
"""
```

### Pattern 2: Constraint-Based Reasoning
**Use when:** Solutions must meet multiple competing requirements

```python
constraint_prompt = """
Design a notification system with these constraints:

HARD CONSTRAINTS (must meet):
- Support 1M+ users
- <100ms response time
- 99.9% delivery guarantee
- GDPR compliant

SOFT CONSTRAINTS (optimize for):
- Minimize cost
- Easy to maintain
- Scalable to 10M users

CONSTRAINTS IN CONFLICT:
- Real-time delivery vs. cost optimization
- Delivery guarantee vs. performance
- GDPR compliance vs. personalization

Please work through these constraints systematically and propose a solution that maximizes soft constraints while meeting all hard constraints. Explain your trade-offs.
"""
```

### Pattern 3: Perspective-Taking
**Use when:** Need to consider multiple stakeholder viewpoints

```python
perspective_prompt = """
Evaluate this product decision from multiple perspectives:

DECISION: Remove the free tier and make our product paid-only (starting at $29/month)

Please analyze this from the perspective of:
1. **Current free users** - What are their likely reactions and alternatives?
2. **Paying customers** - How might this affect their perception of value?
3. **Sales team** - What challenges and opportunities does this create?
4. **Engineering team** - How does this impact technical priorities?
5. **Executives** - What are the financial and strategic implications?

For each perspective, consider both immediate reactions and 6-month implications.
"""
```

### Pattern 4: Failure Mode Analysis
**Use when:** Need to anticipate and prevent problems

```python
failure_mode_prompt = """
You're launching a critical system update during peak traffic hours. 

Please think like a "pre-mortem" exercise - imagine it's 6 months from now and the launch was a disaster.

Work backwards from these failure scenarios:
1. "The system crashed and was down for 4 hours"
2. "Data was corrupted for 10% of users"  
3. "The new feature confused users and support tickets increased 500%"
4. "A security vulnerability was exploited within 24 hours"

For each scenario:
- What sequence of events likely led to this failure?
- What warning signs should we watch for?
- What preventive measures should we implement?
- What rollback procedures do we need?
"""
```

### Pattern 5: Socratic Method
**Use when:** Need deep understanding or learning

```python
socratic_prompt = """
I want to understand machine learning model interpretability. Instead of explaining it directly, please use the Socratic method - guide me to understanding through strategic questions.

Start by asking me what I think "interpretability" means in the context of ML, then build on my responses with follow-up questions that help me discover the key concepts, trade-offs, and practical implications.

Keep asking probing questions until I can articulate:
- Why interpretability matters
- Different types of interpretability
- Trade-offs between accuracy and interpretability
- Practical approaches for different use cases
"""
```

---

## 🔄 Advanced Workflow Patterns

### Pattern 1: Iterative Refinement
```python
def iterative_refinement_workflow():
    """
    Use for: Content creation, design, analysis that benefits from multiple passes
    """
    
    # Step 1: Initial draft
    draft_prompt = """
    Create an initial draft of [CONTENT TYPE] that covers:
    - [Key requirement 1]
    - [Key requirement 2] 
    - [Key requirement 3]
    
    Focus on getting the structure and main ideas down. Don't worry about perfection.
    """
    
    # Step 2: Self-critique
    critique_prompt = """
    Review your draft and identify:
    1. What works well?
    2. What's missing or unclear?
    3. Where could the argument be stronger?
    4. What would confuse the target audience?
    
    Be specific and actionable in your feedback.
    """
    
    # Step 3: Focused improvement
    improve_prompt = """
    Based on your critique, revise the draft to address the specific issues you identified.
    Focus particularly on [HIGHEST PRIORITY AREA].
    """
    
    # Step 4: Final polish
    polish_prompt = """
    Do a final pass focusing on:
    - Clarity and flow
    - Consistent tone
    - Precision of language
    - Meeting all original requirements
    """
```

### Pattern 2: Multi-Perspective Analysis
```python
def multi_perspective_analysis():
    """
    Use for: Complex decisions, strategic planning, risk assessment
    """
    
    perspectives = [
        {"role": "Financial Analyst", "focus": "Cost, revenue, ROI"},
        {"role": "User Experience Designer", "focus": "User impact, usability"},
        {"role": "Engineering Manager", "focus": "Technical feasibility, resources"},
        {"role": "Product Manager", "focus": "Market fit, strategy"},
        {"role": "Risk Manager", "focus": "Potential downsides, compliance"}
    ]
    
    for perspective in perspectives:
        prompt = f"""
        Analyze this decision from the perspective of a {perspective['role']}.
        
        Focus specifically on: {perspective['focus']}
        
        Provide:
        1. Key concerns from this perspective
        2. Success criteria you'd use to evaluate this
        3. Your recommendation (proceed/modify/halt)
        4. What information you'd need to be more confident
        
        DECISION TO ANALYZE: [DECISION DETAILS]
        """
```

### Pattern 3: Root Cause Analysis
```python
def root_cause_analysis():
    """
    Use for: Problem solving, debugging, incident analysis
    """
    
    # Step 1: Problem definition
    define_prompt = """
    Clearly define the problem:
    1. What exactly is happening? (symptoms)
    2. What should be happening instead? (expected behavior)
    3. When did this start? (timeline)
    4. Who/what is affected? (scope)
    5. What's the business impact? (consequences)
    """
    
    # Step 2: Hypothesis generation
    hypotheses_prompt = """
    Generate 8-10 potential root causes using different frameworks:
    - Technical causes (system, process, code)
    - Human causes (error, training, communication)
    - Organizational causes (culture, incentives, resources)
    - External causes (dependencies, market, regulations)
    
    For each hypothesis, rate likelihood (1-10) and potential impact (1-10).
    """
    
    # Step 3: Investigation plan
    investigation_prompt = """
    For the top 3 most likely hypotheses, create an investigation plan:
    1. What specific evidence would prove/disprove this hypothesis?
    2. How can we gather this evidence quickly?
    3. What are we looking for exactly?
    4. Who should be involved in the investigation?
    """
```

---

## 🎭 Domain-Specific Prompt Libraries

### Software Engineering
```python
# Code Review
code_review_prompt = """
Please review this code as a senior engineer would:

<code>
{code_to_review}
</code>

Evaluate:
1. **Functionality**: Does it work correctly?
2. **Readability**: Is it clear and well-documented?
3. **Performance**: Are there efficiency concerns?
4. **Security**: Any vulnerabilities or data exposure risks?
5. **Maintainability**: Will this be easy to modify later?
6. **Best Practices**: Does it follow team/language conventions?

Format:
- Overall assessment (Approve/Request Changes/Needs Discussion)
- Specific issues (line numbers + suggestions)
- Positive observations
- Questions for the author
"""

# Architecture Design
architecture_prompt = """
Design a system architecture for: {requirements}

Please provide:

1. **High-Level Architecture**
   - Component diagram description
   - Data flow overview
   - Technology stack recommendations

2. **Detailed Design**
   - Service boundaries and responsibilities
   - API contracts between services
   - Data storage strategy
   - Caching strategy

3. **Non-Functional Requirements**
   - Scalability approach
   - Performance targets
   - Security considerations
   - Monitoring and observability

4. **Implementation Plan**
   - Phase 1: MVP components
   - Phase 2: Scaling features
   - Phase 3: Advanced features
   - Risk mitigation for each phase

5. **Trade-offs and Alternatives**
   - Why this approach vs. alternatives
   - Known limitations
   - Future evolution path
"""
```

### Data Science
```python
# Data Analysis
data_analysis_prompt = """
Analyze this dataset and provide insights:

<data_context>
Dataset: {dataset_description}
Business Question: {business_question}
Stakeholders: {stakeholder_info}
</data_context>

Please structure your analysis:

1. **Data Quality Assessment**
   - Missing values, outliers, inconsistencies
   - Data types and distribution characteristics
   - Potential biases or collection issues

2. **Exploratory Analysis**
   - Key patterns and trends
   - Correlations and relationships
   - Segmentation opportunities

3. **Statistical Analysis**
   - Hypothesis testing results
   - Confidence intervals
   - Statistical significance

4. **Business Insights**
   - Direct answers to business questions
   - Unexpected findings
   - Actionable recommendations

5. **Limitations and Next Steps**
   - What we can't conclude from this data
   - Additional data needed
   - Recommended follow-up analyses
"""

# Model Evaluation
model_evaluation_prompt = """
Evaluate this ML model for production readiness:

<model_context>
Model Type: {model_type}
Use Case: {use_case}
Performance Metrics: {metrics}
Training Data: {training_data_info}
</model_context>

Assessment Framework:

1. **Performance Analysis**
   - Metrics interpretation in business context
   - Comparison to baseline/existing solution
   - Performance across different segments

2. **Robustness Testing**
   - Performance on edge cases
   - Sensitivity to input variations
   - Degradation under different conditions

3. **Fairness and Bias**
   - Performance across demographic groups
   - Potential discriminatory impacts
   - Bias mitigation recommendations

4. **Operational Readiness**
   - Inference latency and throughput
   - Model size and resource requirements
   - Monitoring and maintenance needs

5. **Risk Assessment**
   - Failure modes and their impact
   - Rollback strategy
   - Gradual rollout plan

Recommendation: Deploy/Needs Improvement/Not Ready
"""
```

### Business Strategy
```python
# Market Analysis
market_analysis_prompt = """
Analyze this market opportunity:

<context>
Product/Service: {product_description}
Target Market: {target_market}
Company Stage: {company_stage}
Resources: {available_resources}
</context>

Analysis Framework:

1. **Market Size and Opportunity**
   - TAM, SAM, SOM estimation
   - Market growth trends
   - Key market drivers

2. **Competitive Landscape**
   - Direct and indirect competitors
   - Competitive advantages/disadvantages
   - Market positioning opportunities

3. **Customer Analysis**
   - Target customer segments
   - Customer needs and pain points
   - Buying behavior and decision criteria

4. **Go-to-Market Strategy**
   - Recommended market entry approach
   - Pricing strategy considerations
   - Distribution channel options

5. **Risk Analysis**
   - Market risks and mitigation strategies
   - Regulatory considerations
   - Technology/execution risks

6. **Success Metrics**
   - KPIs to track
   - Milestones for decision points
   - Success criteria for each phase

Recommendation: Pursue/Modify Approach/Reconsider
"""

# Strategic Planning
strategic_planning_prompt = """
Develop a strategic plan for: {strategic_objective}

<company_context>
Company: {company_info}
Current Position: {current_position}
Resources: {resources}
Constraints: {constraints}
Timeline: {timeline}
</company_context>

Strategic Framework:

1. **Situation Analysis**
   - Internal strengths and weaknesses
   - External opportunities and threats
   - Key success factors in this market

2. **Strategic Options**
   - 3-4 distinct strategic approaches
   - Pros/cons of each approach
   - Resource requirements

3. **Recommended Strategy**
   - Chosen approach with rationale
   - Key strategic initiatives
   - Success metrics and milestones

4. **Implementation Plan**
   - Phase 1 (first 90 days)
   - Phase 2 (90-365 days)
   - Phase 3 (year 2+)
   - Critical dependencies and risks

5. **Resource Allocation**
   - Budget allocation by initiative
   - Headcount and skill requirements
   - Technology and infrastructure needs

6. **Monitoring and Adaptation**
   - Key performance indicators
   - Review and adjustment schedule
   - Pivot criteria and alternatives
"""
```

---

## ⚡ Key Takeaways

1. **Simplicity first** - Start with single calls, add complexity only when needed
2. **Examples are powerful** - 3-5 good examples often outperform complex prompting
3. **Context is king** - Give Claude the background it needs to succeed
4. **Test systematically** - Build evaluation into your development process
5. **Document tools carefully** - Treat ACI design like UX design
6. **Monitor and iterate** - Production performance guides optimization

**Remember:** The most successful Claude implementations use simple, composable patterns rather than complex frameworks. Focus on clear communication and systematic evaluation. 
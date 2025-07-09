# Comprehensive Claude Prompt Engineering Guide

## Table of Contents
1. [Core Principles](#core-principles)
2. [Prompt Templates and Variables](#prompt-templates-and-variables)
3. [Essential Techniques](#essential-techniques)
4. [Advanced Strategies](#advanced-strategies)
5. [Model-Specific Guidelines](#model-specific-guidelines)
6. [Extended Thinking](#extended-thinking)
7. [Implementation Best Practices](#implementation-best-practices)
8. [Common Patterns and Solutions](#common-patterns-and-solutions)

## Core Principles

### Before You Start
1. **Define clear success criteria** - Know exactly what constitutes a successful output
2. **Create empirical tests** - Have ways to measure performance against your criteria
3. **Start with a first draft** - Don't optimize prematurely; get something working first

### When to Use Prompt Engineering vs Other Methods
Prompt engineering is preferred over fine-tuning when you need:
- **Resource efficiency**: Only requires text input, no GPUs
- **Cost-effectiveness**: Uses base model pricing
- **Flexibility**: Instant iteration and testing
- **Minimal data**: Works with few-shot or zero-shot examples
- **Preserved capabilities**: Maintains model's general knowledge

## Prompt Templates and Variables

### Structure
```
Fixed content (instructions) + {{variable_content}} = Complete prompt
```

### Benefits
- **Consistency**: Uniform structure across calls
- **Efficiency**: Easy content swapping
- **Testability**: Quick edge case testing
- **Scalability**: Simplified management
- **Version control**: Track only core prompt changes

### Example Template
```
Translate this text from English to Spanish: {{text}}
```

### Console Notation
In Anthropic Console, variables use `{{double_brackets}}`

## Essential Techniques

### 1. Be Clear and Direct
- Use specific, actionable language
- Avoid ambiguity
- State requirements explicitly

**Bad**: "Analyze this data"
**Good**: "Analyze this sales data to identify the top 3 performing products by revenue and explain why they're successful"

### 2. Use Examples (Multishot Prompting)
Include 3-5 diverse, relevant examples to dramatically improve accuracy.

```xml
<examples>
  <example>
    Input: The new dashboard is a mess! It takes forever to load, and I can't find the export button. Fix this ASAP!
    Category: UI/UX, Performance
    Sentiment: Negative
    Priority: High
  </example>
</examples>

Now analyze this feedback: {{FEEDBACK}}
```

### 3. Let Claude Think (Chain of Thought)
For complex reasoning tasks, ask Claude to show its work.

```
Please think through this problem step by step before providing your final answer.
```

### 4. Use XML Tags
Structure your prompts with XML tags for clarity and parseability.

```xml
<instructions>
  Analyze the following document
</instructions>

<document>
  {{DOCUMENT_CONTENT}}
</document>

<requirements>
  - Identify main themes
  - Extract key insights
  - Summarize in 3 paragraphs
</requirements>
```

### 5. Give Claude a Role (System Prompts)
Use the `system` parameter to set Claude's expertise and perspective.

```python
system="You are a seasoned data scientist at a Fortune 500 company."
```

**Impact**: Role prompting can dramatically improve domain-specific performance and adjust communication style.

### 6. Prefill Claude's Response
Control output format by starting Claude's response.

```python
messages=[
    {"role": "user", "content": "List three benefits of exercise"},
    {"role": "assistant", "content": "Here are three benefits of exercise:\n1."}
]
```

### 7. Chain Complex Prompts
Break complex tasks into sequential steps.

```python
# Step 1: Extract data
response1 = get_data_extraction(document)

# Step 2: Analyze extracted data
response2 = analyze_data(response1)

# Step 3: Generate report
response3 = create_report(response2)
```

### 8. Long Context Tips
For documents >20K tokens:

```xml
<documents>
  <document index="1">
    <source>annual_report_2023.pdf</source>
    <document_content>
      {{ANNUAL_REPORT}}
    </document_content>
  </document>
</documents>

<!-- Place query at the end -->
Analyze the annual report and identify strategic advantages.
```

**Key principle**: Put long content at the top, query at the bottom (improves performance by up to 30%).

## Advanced Strategies

### Structured Output Control
1. **Tell what to do, not what to avoid**
   - ❌ "Do not use markdown"
   - ✅ "Write in plain prose paragraphs"

2. **Use XML format indicators**
   ```
   Write your response in <analysis> tags
   ```

3. **Match prompt style to desired output**
   - Remove markdown from prompt if you don't want markdown output

### Grounding Responses in Quotes
For document analysis tasks:

```xml
Find quotes from the documents that support your analysis. 
Place these in <quotes> tags first, then provide your analysis in <analysis> tags.
```

### Nested XML Structure
For hierarchical information:

```xml
<task>
  <context>
    <background>{{BACKGROUND_INFO}}</background>
    <constraints>{{CONSTRAINTS}}</constraints>
  </context>
  <instructions>
    <step1>{{INSTRUCTION_1}}</step1>
    <step2>{{INSTRUCTION_2}}</step2>
  </instructions>
</task>
```

## Model Selection

### Model Comparison
- **Claude Opus 4**: Highest intelligence for most complex tasks (multi-agent coding, complex analysis)
- **Claude Sonnet 4**: Balance of intelligence and speed (complex chatbots, code generation, agentic loops)
- **Claude 3.5 Haiku**: Fast responses at lower cost (basic support, high-volume content generation)

### Selection Strategy
1. **Start fast and cost-effective**: Begin with Haiku, upgrade if needed
2. **Start with maximum capability**: Begin with Opus 4, optimize down if possible

### When to Use Each Model
| Need | Model | Use Cases |
|------|-------|-----------|
| Highest intelligence | Claude Opus 4 | Multi-agent frameworks, complex refactoring, nuanced writing |
| Balanced performance | Claude Sonnet 4 | Complex customer service, code generation, data analysis |
| High-volume, fast | Claude 3.5 Haiku | Basic support, formulaic content, data extraction |

## Model-Specific Guidelines

### Claude 4 Models (Opus 4, Sonnet 4)

#### Key Characteristics
- More precise instruction following
- Better at parallel tool execution
- Enhanced visual and frontend code generation
- Improved extended thinking capabilities

#### Best Practices

1. **Be Explicit**
   ```
   Create an analytics dashboard. Include as many relevant features 
   and interactions as possible. Go beyond the basics to create a 
   fully-featured implementation.
   ```

2. **Add Context**
   ```
   Your response will be read aloud by a text-to-speech engine, 
   so never use ellipses since the TTS engine won't know how to pronounce them.
   ```

3. **Optimize Tool Calling**
   ```
   For maximum efficiency, whenever you need to perform multiple 
   independent operations, invoke all relevant tools simultaneously 
   rather than sequentially.
   ```

### Migration from Older Models
When migrating to Claude 4:
- Be more specific about desired behaviors
- Add quality modifiers ("comprehensive", "detailed", "fully-featured")
- Explicitly request features that were previously implicit

## Extended Thinking

### When to Use
Extended thinking is ideal for:
- Complex STEM problems
- Multi-constraint optimization
- Detailed strategic analysis
- Tasks requiring verification and error-checking

### Technical Specifications
- Minimum budget: 1,024 tokens
- For >32K thinking tokens, use batch processing
- Works best in English
- Cannot prefill thinking blocks

### Prompting Strategies

1. **General Instructions First**
   ```
   Please think about this problem thoroughly and in great detail.
   Consider multiple approaches and show your complete reasoning.
   ```

2. **Request Verification**
   ```
   Before you finish, please verify your solution with test cases 
   and fix any issues you find.
   ```

3. **Enable Reflection**
   ```
   After each step, analyze whether you achieved the expected result 
   before proceeding.
   ```

## Implementation Best Practices

### API Integration
```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-opus-4-20250514",  # Or claude-sonnet-4-20250219
    max_tokens=2048,
    system="You are an expert code reviewer.",  # Role
    messages=[
        {
            "role": "user", 
            "content": """<task>
                Review this code for security vulnerabilities:
                <code>{{CODE}}</code>
            </task>"""
        }
    ]
)
```

### Error Handling Patterns
```python
def safe_claude_call(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-opus-4-20250514",  # Or claude-sonnet-4-20250219
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
```

### Prompt Caching
For repetitive tasks with large contexts:

```python
system=[
    {
        "type": "text",
        "text": "You are an AI assistant analyzing legal documents."
    },
    {
        "type": "text",
        "text": "<50-page legal agreement here>",
        "cache_control": {"type": "ephemeral"}
    }
]
```

Benefits:
- Reduces processing time
- Lowers costs (cache hits cost 10% of base price)
- 5-minute cache lifetime (refreshed on use)

## Common Patterns and Solutions

### Pattern 1: Research Analysis
```xml
<role>You are a research scientist specializing in {{DOMAIN}}</role>

<context>
  <background>{{RESEARCH_BACKGROUND}}</background>
  <previous_work>{{LITERATURE_REVIEW}}</previous_work>
</context>

<task>
  Analyze the following data and provide insights:
  <data>{{EXPERIMENTAL_DATA}}</data>
</task>

<requirements>
  - Identify patterns and anomalies
  - Compare with previous findings
  - Suggest next experiments
  - Quantify uncertainty
</requirements>
```

### Pattern 2: Code Generation
```xml
<role>You are a senior software engineer</role>

<context>
  <tech_stack>{{TECH_STACK}}</tech_stack>
  <conventions>{{CODING_STANDARDS}}</conventions>
</context>

<task>
  Implement {{FEATURE}} with the following requirements:
  <requirements>{{REQUIREMENTS}}</requirements>
</task>

<constraints>
  - Follow existing patterns in the codebase
  - Include comprehensive error handling
  - Add unit tests
  - Document all public methods
</constraints>
```

### Pattern 3: Multi-Document Analysis
```xml
<documents>
  <document index="1" type="primary">
    <source>{{SOURCE_1}}</source>
    <content>{{CONTENT_1}}</content>
  </document>
  <document index="2" type="reference">
    <source>{{SOURCE_2}}</source>
    <content>{{CONTENT_2}}</content>
  </document>
</documents>

<instructions>
  First, quote relevant sections from each document in <quotes> tags.
  Then synthesize the information to answer: {{QUESTION}}
</instructions>
```

### Pattern 4: Decision Making
```xml
<scenario>{{BUSINESS_SCENARIO}}</scenario>

<options>
  <option id="A">{{OPTION_A}}</option>
  <option id="B">{{OPTION_B}}</option>
  <option id="C">{{OPTION_C}}</option>
</options>

<analysis_framework>
  For each option, analyze:
  1. Pros and cons
  2. Resource requirements
  3. Risk assessment
  4. Success probability
  5. ROI projection
</analysis_framework>

<output_format>
  Provide a decision matrix in table format, 
  followed by a recommendation with justification.
</output_format>
```

## Optimization Checklist

### Before Optimization
- [ ] Clear success criteria defined
- [ ] Baseline performance measured
- [ ] Test cases prepared

### Technique Application Order
1. [ ] Make instructions clear and direct
2. [ ] Add relevant examples (3-5)
3. [ ] Implement XML structure
4. [ ] Add role via system prompt
5. [ ] Enable chain of thought if needed
6. [ ] Consider output prefilling
7. [ ] Break into chained prompts if complex
8. [ ] Optimize for long context if applicable

### Performance Monitoring
- Track token usage
- Measure response quality
- Monitor consistency
- Check edge cases
- Validate against success criteria

## Key Takeaways

1. **Clarity beats complexity** - Simple, clear instructions often outperform elaborate prompts
2. **Examples are powerful** - Well-chosen examples can improve performance more than detailed instructions
3. **Structure matters** - XML tags provide clarity for both Claude and developers
4. **Roles enhance expertise** - System prompts with specific roles improve domain performance
5. **Thinking helps complex tasks** - Chain of thought and extended thinking improve reasoning
6. **Context placement matters** - Put long documents first, queries last
7. **Iterate based on outputs** - Read Claude's responses and refine prompts accordingly

Remember: Start simple, test systematically, and optimize based on measured results.
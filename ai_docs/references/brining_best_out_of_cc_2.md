June 19, 2025
how i bring the best out of claude code - part 2
hello everyone!

in part 1, i covered the beginner's guide to claude code. setup requirements. context management. basic workflows. the fundamentals you need to get started.

that was the shallow end of the pool.

this is the deep end. intermediate techniques for people who've already hit the walls of vanilla claude code and always want just a little bit more.

welcome to the future

1. integrating claude code into tooling
in part 1, i mentioned building a local multi-agent system. let me show you what that actually looks like.

everything i'm about to show you comes from my agent-guides repository. it's all open source, ready to use.

2. what are these systems?
they're custom commands and tools that transform claude code from a chat interface into a cognitive prosthetic.

i built agent-guides because vanilla claude code kept failing me in predictable ways. lost insights. repeated work. single-point failures.

the more specific your tooling, the more powerful your workflow becomes. everything i feel like doing multiple times as a prompt, should be a command.

3. multi-mind: solving the hallucination problem
remember how models confidently tell you wrong things? or how you are not sure if the model is being sycophantic? if only it could have opposing viewpoints.

multi-mind fixes that.

/multi-mind "find security vulnerabilities in our auth system"
this spawns 4-6 specialist subagents:

security analyst
edge case hunter
performance auditor
api contract validator
each works independently. can't see each other's initial analysis. after they finish, they review each other's findings.

here's a glimpse of how it assigns specialists:

// from multi-mind.md
const specialistPrompts = {
    "Security Analyst": "Identify vulnerabilities, attack vectors...",
    "Edge Case Hunter": "Find boundary conditions, error states...",
    "Performance Auditor": "Analyze computational complexity...",
    "API Contract Validator": "Verify interface consistency..."
}
independent verification kills hallucinations. caught a timing attack that single-agent analysis completely missed.

4. conversation search: your second brain
i am quite chaotic in the way i work. forever solving a problem and gleaming at 3am, then forgetting the solution or losing track of it in the sea of tasks that i pick up right after.

so i built search-prompts to fix that.

you can search through your entire conversation history, exported json sessions, and current context.
you can use it in any way - to run analytics or to discover your preferences and have them reflected in your future commands.

/search-all "redis optimization"
searches through:

sqlite conversation history
exported json sessions
current context
here's what the command does behind the scenes:

# from search-prompts.md
## Search Types:
1. **search-all**: Search across all conversation sources
2. **search-current**: Search only current session
3. **search-project**: Search project-specific conversations
4. **search-date**: Search by date range

Returns formatted results with:
- Timestamp and session ID
- Matching content preview
- Context around the match
your past conversations are a goldmine. most people just let them rot or lose it. why wait for anthropic to fix it?

5. session paging: my fav infinite context hack
claude's context fills up. work gets lost.
being originally from electrical background, i couldn't help but see the need of a paging equivalent mechanism in claude code.

say no to context loss with page command.

/page "ml pipeline progress checkpoint 1"
it is designed to save everything you do in a session, preserving:

saves complete state
generates summaries
preserves citations
lets you pick up tomorrow exactly where you left off
here's how it structures the saved session:

# from page.md
## Session: ml pipeline progress checkpoint 1
### Summary
Key accomplishments and current state...

### Full History
[timestamp] User: original request...
[timestamp] Assistant: implementation details...

### Citations
- file1.py:42 - optimized batch processing
- file2.py:156 - added caching layer
claude --resume checkpoint-1
treat context like os memory. page out, page in, never lose work.

6. deep code analysis that thinks
while reviewing code, claude can analyze it in depth, finding hidden complexity, edge cases, and optimization opportunities. i do not like simple explanations derived from docstrings and comments.

analyze-function goes beyond description.

/analyze-function "def batch_process(items, workers=4):"
doesn't just describe code. it reasons:

line-by-line performance implications
hidden complexity (found o(n²) in "linear" code)
edge cases you missed
mathematical foundations
here's the analysis pattern it follows:

# from analyze-function.md
## Analysis Structure:
1. **Purpose & Context**: What this function solves
2. **Line-by-Line Breakdown**: 
   - Line 3: O(n) operation, potential bottleneck
   - Line 7: Nested loop creates O(n²) complexity
3. **Edge Cases**: Empty input, single worker, overflow
4. **Performance Profile**: Time/Space complexity
5. **Optimization Opportunities**: Parallel processing, caching
saved me from wasting several gpu hours on something which was a complete blindspot to me.

7. crud commands: build your own commands
famous saying : "teach a man to run commands, and you feed his curiosity for a day. teach a man to build commands, and you feed him for a lifetime."

if you ever find yourself typing the same prompts over and over? just don't.

that's exactly why i built crud-claude-commands. by the time i wrote my third commnand, i was clear that i needed a meta-command system.

with yourself as the water that flows through the system. you can create, read, update, delete, and list commands. it allows you to build a library of reusable commands that fit your workflow.

create new commands on the fly
/crud-claude-commands create git-flow "automate git flow operations like creating feature branches, PRs, and merging"
boom. now you have a custom git-flow command tailored to your workflow.

read what a command does
/crud-claude-commands read git-flow
update when your needs evolve
/crud-claude-commands update git-flow "enhanced git workflow with automatic PR creation and branch management"
delete outdated commands
/crud-claude-commands delete git-flow
list your entire arsenal
/crud-claude-commands list
here's the magic - it generates standardized templates:

# from crud-claude-commands output
## Command: git-flow
### Description
Automates git flow operations...

### Usage
/git-flow feature start my-feature
/git-flow pr create
/git-flow release finish

### Implementation
[detailed workflow steps...]
stop repeating yourself. if you do something twice, make it a command.

the real power? rapid iteration. prototype a command, test it, refine it, share it. your personal command library grows organically with your needs.

8. the workflow that actually works
for architecture reviews:
/multi-mind "review our microservices for bottlenecks"
→ 5 specialists work in parallel
→ cross-pollination finds blind spots
→ /page "architecture-review-final"
for debugging:
/search-all "null pointer kubernetes"
→ find similar past issues
→ /analyze-function on suspect code
→ multi-mind verification of fix
for long projects:
issue #142 → docs/plan_142.md
→ work until context fills
→ /page "issue-142-session-1"  
→ resume seamlessly next day
for letting claude grow with your workflow:
/crud-claude-commands list
→ /page "command-library"
9. what protocols are being enforced?
we want to put systems in place that accumulate knowledge bases, behavior trajectories, preferences, serve as a goldmine for future agents. we want to do things that snowball into something bigger.

single responses are hypotheses, not truth.
always verify through multiple agents or past evidence.

every conversation builds lasting value.
searchable, reusable, compounding knowledge.

small tools compose into powerful workflows.
unix philosophy for ai assistance.

context is precious. manage it.
page out before you lose work.

knowledge reuse is key.
every conversation builds lasting value.

10. how is multi-agent design different?
error decorrelation: agents make different mistakes. consensus filters out individual errors.

specialist depth: focused expertise beats generalist responses every time.

progressive refinement: cross-pollination rounds systematically improve quality.

i personally say no more "claude said so" disasters. multiple independent verification or it didn't happen.

12. setup is trivial
git clone https://github.com/tokenbender/agent-guides
cd agent-guides

# install commands in your project
mkdir -p /path/to/your/project/.claude/commands
cp -r claude-commands/* /path/to/your/project/.claude/commands/

# copy supporting scripts
cp -r scripts /path/to/your/project/.claude/scripts/
that's it. now you have superpowers.

want to see what commands you're getting? peek at the command directory:

claude-commands/
├── multi-mind.md      # parallel specialist analysis
├── search-prompts.md  # conversation archaeology  
├── page.md           # session state management
├── analyze-function.md # deep code reasoning
└── crud-claude-commands.md # dynamic command creation
13. the philosophy
these aren't just tools. they're a different way of thinking about ai assistance.

augment, don't replace: enhance claude's native abilities
compose, don't monolith: small tools that work together
persist, don't repeat: every interaction should create lasting value
verify, don't trust: multiple sources or it's probably wrong
“People SHOULD be doubted. Many people misunderstand this concept. Doubting people is just a part of getting to know them. What many people call ‘trust’ is really just giving up on trying to understand others, and that very act is far worse than doubting. It is actually ‘apathy.”

― Shinobu Kaitani, Liar Game, Volume 4

14. what i'm building next
and for my final trick, i am building something to aid my experimentation and RL research in my main quest avataRL.
a complete auto-track changes, kick-off experiments, observe and auto-merge results system.

all of that and more in the next post.

grab agent-guides and build your own cognitive prosthetics.

until then, be well everyone.
# Comprehensive Report on AI Coding Workflows and Tools

## 1. Palmer's Gauntlet AI Workflow System

### Project Setup and Documentation Structure

Palmer's workflow begins with a systematic approach to project scaffolding using a 12-step process for starting projects from scratch. The workflow emphasizes creating comprehensive documentation before writing any code.

**Core Documentation Files (stored in `_docs` folder):**

- **Project Overview**: "Your project's purpose goals whatever you know all the loose features that you're going to have"
- **User Flow**: "Defines the journey through the application...good for planning out your routes...seeing your project come to life in kind of a navigation structure so that the AI can have a better understanding of how the different features will interact"
- **Tech Stack**: "The most important document...covers all the different things that you're actually building your project with so all the libraries packages"
- **UI and Theme Rules**: "Creating like a cohesive theme throughout the project so when you ask the AI to generate a new component right or a new page giving it these theme rules and the UI rules can make sure that it's pretty close to anything else you have in the rest of your codebase"
- **Project Rules**: "For the naming conventions for the file structure this helps the AI know exactly where it needs to place new files and folders"
- **Phase Doc/Checklist**: "Actionable checklists...organized for you to attach at will"

### Cursor Configuration System

**Cursor Rules Hierarchy:**
1. **Cursor Mode (System Prompt)**: "Effectively the system prompt...takes precedence"
2. **User Rules and Cursor Notepads**: "Always active if you put them in context...treated on the same level"
3. **Cursor Rules**: "Can be referenced in three different ways: glob patterns, user selection, or by request"
4. **Attached Files**: "Persisted means it'll remain in the chat context"
5. **Mentioned Files**: "Not persisted...goes away if you haven't included it again"

**Context Management Best Practices:**
- "Keep chats scoped to tasks or features...don't go beyond that"
- "Always create a new chat" for debugging unrelated issues
- "Always say use tools...if you're having the agent explore the codebase"
- "If you're really really stuck...take it outside of the IDE"

### Document Growth Strategy

"If I've gone through several times where it's trying to use yarn for this or yarn for that I'll say 'Hey we're using npm like create a document that explains what we're using and why we're using it'...that's how my docs will grow throughout a project"

## 2. Claude Code Features and Custom Workflows

### Custom Commands System

"These are essentially reusable prompts with arguments if you choose to add them that allow you to build custom workflows Claude Code can execute for you...you just create these custom commands once and Claude Code will do it for you"

**Creating Custom Commands:**
- Create a `commands` folder inside Claude Code base directory
- Add `.md` files for each command
- Use dynamic arguments with special syntax for flexibility

### MCP Server Integration

"MCP servers with it...it's a gamechanger if you haven't...Sentry is a platform where you manage issues in your code...its MCP server takes it to the next level...it can fetch issue details about a particular project using those details your AI agents whether it's Claude or cursor gain much more insight"

**Installation Method:**
```
claude mcp add-json [name] '[JSON configuration]'
```

### Advanced Features

- **Plan Mode**: "Allows you to plan out your whole project or idea...basically a way to brainstorm anything with Claude code"
- **Auto Accept Edits Mode**: "This mode will edit files without asking for your permission...better when you want to give full autonomy to Claude Code"
- **Ultrathink Keyword**: "Using the keyword ultrathink will push Claude code and the models it uses to maximum power...only use it for your hardest problems"

## 3. Team-Based AI Coding with Multiple Instances

### Task Management System

"I have a task file that contains the tasks I want to be accomplished...with each task I have three fields: the branch name if a branch already exists...the status of its current state...store the name of the tmux session"

### Git Worktrees Implementation

"Git worktrees are basically just a physical copy of the whole repo or project...you can actually run them without switching branches...once the feature is complete you can merge that work tree back"

### Tmux Session Management

"Tmux is literally an abbreviation for terminal multiplexer it basically allows you to create multiple sessions of terminal running in the background which you can attach to anytime"

### Agent Spawner Workflow

The workflow creates an orchestrator that:
- "Read the tasks file and then find one or multiple tasks that can be solved by one agent"
- "First creating a new work tree then building a prompt and then launching the agent"
- "Create the tmux session that is made and detached and that session will run the Claude code instance"

## 4. Real-World Engineering Workflows

### Claude Configuration Files

**User Claude.md Instructions:**
- "Whenever you encounter new things about the system or my general preferences that would be useful for you to remember make a suggestion to add notes to your config file"
- "When creating PRs that have corresponding issue remember to add closes #123 so it closes"

**Project-Specific Claude.md:**
- Core models description
- Development setup
- Test data configuration
- Important conventions (e.g., "prices are stored in cents")
- Technology stack details
- Architecture notes

### Workflow Commands

- **/ni (New Issue)**: "Create a new GitHub issue based on this brief description...analyze expand show me a draft and ask for follow-up notes"
- **/issues**: "Fetch and list open GitHub issues from this repository then ask which one to work on"
- **/prp (PR Process)**: "Process and address PR review comments from the current pull request"

### Key Development Principles

"The trick is to use it just like you would normally use a coworker or like a better version of yourself...think more from a user perspective from a product manager perspective...don't fret too much about what it looks like exactly rather think broader strokes think features think solutions"

## 5. Multiple Claude Instances with Git Worktrees

### GWT Script Implementation

"A script called GWT which is of course short for git worktree...sets up a few variables...if you're running inside tmux it renames the window...if running outside tmux it sets up a new session and window"

### Claude Start Script

"Launches claude with an initial prompt...start working on issue [number] keep an open mind feel free to reconsider the proposed approach update the issue as you discover new relevant information after you're done read through the changes"

### Parallel Workflow Benefits

"While he's working we can set up new issues...work on code that we want to write ourselves...in your main tmux session and only respond to Claude when you find yourself having a tiny break"

## 6. Plan Mode and Higher-Order Prompts

### Plan Mode Functionality

"Plan mode is a special operating mode where I research and analyze create a plan present the plan for approval wait for confirmation...I'm restricted from making any file edits running commands creating or deleting or making commits"

**Purpose of Plan Mode:**
"The purpose of plan mode read files understand the codebase by loading your context window with important information before we start working then switch out and edit"

### Infinite Agentic Loop System

"Higher order prompts...just like higher order functions where in certain programming languages you can pass a function into a function...we're passing a prompt into a prompt...The infinite prompt takes a prompt as a parameter"

### Two Plan Mode Techniques

1. "Plan then write code"
2. "Plan write the plan to a spec file and then have claude code execute on the spec that you created"

"Four out of five times I recommend the second technique specifically for mid to large-sized features"

## 7. Claude Code vs OpenAI Codex Comparison

### Tool Positioning Spectrum

"The terminal is the highest leverage point for engineering work...on the left we have control on the right we have ease of use...Terminal applications [have] maximum control...Desktop apps [are] the happy medium...Web apps are fantastic they're the easiest to use"

### Claude Code Philosophy

"Generally at Anthropic we have this product principle of do the simple thing first...staff things as little as you can and keep things as scrappy as you can because the constraints are actually pretty helpful"

### Key Differentiators

- **Claude Code**: "A tool for power workloads for power users...raw access to the model...as raw as it gets"
- **Built-in Tools**: Full list of agentic capabilities for file manipulation, bash commands, and codebase exploration
- **Cost**: "Currently we're seeing costs around like $6 per day per active user"

### Parallelism Capabilities

"You can batch and you can run tasks...create sub agents and you can run multiple tools in parallel...read all py files in parallel...write a one-line summary at the top of each of those files in parallel"

### Future of Engineering

"Linting documentation writing tests these are just the lowhanging fruit of the type of workflows you can build...you can embed your coding agent and put it anywhere in your stack...they'll just be called scripts"

## 8. Anthropic's Official Claude Code Best Practices

### Setup and Customization

**CLAUDE.md Files:**
"A special file that Claude automatically pulls into context when starting a conversation...ideal place for documenting:
- Common bash commands
- Core files and utility functions
- Code style guidelines
- Testing instructions
- Repository etiquette
- Developer environment setup
- Any unexpected behaviors or warnings"

**File Placement Options:**
- "The root of your repo, or wherever you run claude from"
- "Any parent of the directory where you run claude" (for monorepos)
- "Any child of the directory where you run claude"
- "Your home folder (~/.claude/CLAUDE.md)"

### Tool Management

**Permission Control Methods:**
1. "Select 'Always allow' when prompted during a session"
2. "Use the /permissions command after starting Claude Code"
3. "Manually edit your .claude/settings.json or ~/.claude.json"
4. "Use the --allowedTools CLI flag for session-specific permissions"

### Common Workflows

**Explore, Plan, Code, Commit:**
1. "Ask Claude to read relevant files...explicitly tell it not to write any code just yet"
2. "Ask Claude to make a plan...use the word 'think' to trigger extended thinking mode"
3. "Ask Claude to implement its solution in code"
4. "Ask Claude to commit the result and create a pull request"

**Test-Driven Development:**
1. "Ask Claude to write tests based on expected input/output pairs"
2. "Tell Claude to run the tests and confirm they fail"
3. "Ask Claude to commit the tests when you're satisfied"
4. "Ask Claude to write code that passes the tests"
5. "Ask Claude to commit the code once you're satisfied"

### Advanced Techniques

**Multiple Claude Instances:**
- "Use Claude to write code...Run /clear or start a second Claude in another terminal...Have the second Claude review the first Claude's work"
- "Create 3-4 git checkouts in separate folders...Start Claude in each folder with different tasks"

**Headless Mode Automation:**
"Use the -p flag with a prompt to enable headless mode, and --output-format stream-json for streaming JSON output...can power automations triggered by GitHub events"

### Optimization Tips

- "Be specific in your instructions...Giving clear directions upfront reduces the need for course corrections"
- "Use tab-completion to quickly reference files or folders"
- "Press Escape to interrupt Claude during any phase"
- "Double-tap Escape to jump back in history"
- "Use /clear to keep context focused"
- "Use checklists and scratchpads for complex workflows"
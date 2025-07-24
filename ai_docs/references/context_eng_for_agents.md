Context Engineering for Agents
Jun 23, 2025

Lance Martin

Context Engineering
As Andrej Karpathy puts it, LLMs are like a new kind of operating system. The LLM is like the CPU and its context window is like RAM, representing a “working memory” for the model.

Context enters an LLM in several ways, including prompts (e.g., user instructions), retrieval (e.g., documents), and tool calls (e.g., APIs).

Just like RAM, the LLM context window has limited “communication bandwidth” to handle these various sources of context.

And just as an operating system curates what fits into a CPU’s RAM, we can think about “context engineering” the art and science of filling the context window with the information needed to perform a task.


The Rise of Context Engineering
Context engineering is an umbrella discipline that captures a few different focus areas:

Instructions – prompts (see: prompt engineering), memories, few‑shot examples
Knowledge – retrieval or memories to extend the model’s world‑knowledge (see: RAG)
Tool feedback – context flowing in from the environment via tools
As LLMs get better at tool calling, agents are now feasible. Agents interleave LLM and tool calls for long-running tasks, and motivate the need for engineering across all three types of context.


Cognition called out the importance of context engineering when building agents:

“Context engineering” … is effectively the #1 job of engineers building AI agents.

Anthropic also laid it out clearly:

Agents often engage in conversations spanning hundreds of turns, requiring careful context management strategies.

This post is aims to break down some common strategies — compress, persist, and isolate — for agent context engineering.

Context Engineering for Agents
The agent context is populated with feedback from tool calls, which can exceed the size of the context window and balloon the cost / latency.


I’ve been bitten by this many times. One incarnation of a deep research agent that I built used token-heavy search API tool calls, resulting in > 500k token and several dollars per run!

Long context may also degrade agent performance. Google and Percy Liang’s group have described different types of “context degradation syndrome” since a long context can limit an LLMs ability to recall facts or follow instructions.

There are many ways to combat this problem, which I group into 3 buckets and describe below: compressing, persisting, and isolating context.


Compressing Context
Compressing context involves keeping only the highest-value tokens at each turn.

Context Summarization

Agent interactions can span hundreds of turns and may have token-heavy tool calls. Context summarization is one common way to manage this.

If you’ve used Claude Code, you’ve seen this in action. Claude Code runs “auto-compact” after you exceed 95% of the context window.

Summarization can be used in different places, such as the full agent trajectory with methods such as recursive or hierarchical summarization.


It’s also common to summarize tool call feedback (e.g., a token-heavy search tool) or specific steps (e.g., Anthropic’s multi-agent researcher applies summarization on completed work phases).

Cognition called out that summarization can be tricky if specific events or decisions from agent trajectories are needed. They use a fine-tuned model for this in Devin, which underscores how much work can go into refining this step.

Persisting Context
Persisting context involves systems to store, save, and retrieve context over time.

Storing context

Files are a simple way to store context. Many popular agents use this: Claude Code uses CLAUDE.md. Cursor and Windsurf use rules files, and some plugins (e.g., Cursor Memory Bank) / MCP servers manage collections of memory files.

Some agents need to store information that can’t be easily be captured in a few files. For example, we may want to store large collections of facts and / or relationships. A few tools emerged to support this and showcase some common patterns.

Letta, Mem0, and LangGraph / Mem store embedded documents. Zep and Neo4J use knowledge graphs for continuous / temporal indexing of facts or relationships.

Saving context

Claude Code entrusts the user to create / update memories (e.g., the # shortcut). But there are many cases where we want agents to autonomously create / update memories.

The Reflexion paper introduced the idea of reflection following each agent turn and re-using these self-generated hints. Generative Agents created memories as summaries synthesized from collections of past feedback.

These concepts made their way into popular products like ChatGPT, Cursor, and Windsurf, which all have mechanisms to auto-generate memories based on user-agent interactions.


Memory creation can also be done at specific points in an agent’s trajectory. One pattern I like: update memories based upon user feedback.

For example, human-in-the-loop review of tool calls is a good way to build confidence in your agent. But if you pair this with memory updating, then the agent can learn from your feedback over time. My email assistant does this with file based memory.

Retrieving context

The simplest approach is just to pull all memories into the agent’s context window. For example, Claude Code just reads all CLAUDE.md files into context at the start of each session. In my email assistant, I always load a set memories that provide email triage and response instructions into context.

But, mechanisms to fetch select memories are important if the collection is large. The store will help determine the approach (e.g., embedding-based search or graph retrieval).

In practice this is a deep topic. Retrieval can be tricky. For example, Generative Agents scored memories on similarity, recency, and importance. Simon Willison shared an example of memory retrieval gone wrong. GPT-4o injected location into an image based upon his memories, which was not desired. Poor memory retrieval can make users feel like the context window “doesn’t belong to them”!

Isolating Context
Isolating context involves approaches to partition it across agents or environments.

Context Schema

Oftentimes, messages are used to structure agent context. Tool feedback is appended to a message list. The full list is then passed to the LLM at each agent turn.

The problem is that a list can get bloated with token-heavy tool calls. A structured runtime state - defined via a schema (e.g., a Pydantic model) - can often be more effective.

Then, you can better control what the LLM sees at each agent turn. For example, in one version of a deep research agent, my schema both messages and sections. messages is passed to the LLM at each turn, but I isolate token-heavy sections in sections and fetch them selectively.

Multi-agent

One popular approach is to split context across sub-agents. A motivation for the OpenAI Swarm library was “separation of concerns”, where a team of agents can handle sub-tasks and each agent has its own instructions and context window.


Anthropic’s multi-agent researcher makes a clear case for this: multi-agent with isolated context outperformed single-agent by 90.2%, largely due to token usage. As the blog said:

[Subagents operate] in parallel with their own context windows, exploring different aspects of the question simultaneously.

The problems with multi-agent include token use (e.g., 15× more tokens than chat), the need for careful prompting and context for sub-agent planning, and sub-agent coordination. Cognition argues against multi-agent for these reasons.

I’ve also been bitten by this: one iteration of my deep research agent had a team of agents write sections of the report. Sometimes the final report was disjointed because the agents did not communicate with one another while writing.

One way to reconcile this is to ensure the task is parallelizable. A subtle point is that Anthropic’s deep research multi-agent system applied parallelization to research. This is easier than writing, which requires tight cohesion across the sections of a report to achieve strong overall flow.

Context Isolation with Environments

HuggingFace’s deep researcher is another good example of context isolation. Most agents use tool calling APIs, which return JSON objects (arguments) that can be passed to tools (e.g., a search API) to get tool feedback (e.g., search results).


HuggingFace uses a CodeAgent, which outputs code to execute tools. The code runs in a sandbox and select tool feedback from code execution is passed back to the LLM.

[Code Agents allow for] a better handling of state … Need to store this image / audio / other for later use? No problem, just assign it as a variable in your state and you [use it later].

The sandbox stores objects generated during execution (e.g., images), isolating them from the LLM context window, but the agent can still reference these objects with variables later.

Lessons
General principles for building agents are still in their infancy. Models are changing quickly and the Bitter Lesson warns us to avoid context engineering that will become irrelevant as LLMs improve.

For example, continual learning may let LLMs to learn from feedback, limiting some of the need for external memory. With this and the patterns above in mind, here are some general lessons ordered roughly by the amount of effort required to use them.

Instrument first: Always look at your data. Ensure you have a way to track tokens when building agents. This has allowed me to catch various cases of excessive token-usage and isolate token-heavy tool calls. It sets the stage for any context engineering efforts.
Think about your agent state: Anthropic called out the idea of “thinking like your agent.” One way to do this is to think through the information your agent needs to collect and use at runtime. A well defined state schema is an easy way to better control what is exposed to the LLM during the agent’s trajectory. I use this with nearly every agent I build, rather than just saving all context to a message list. [Anthropic] also called this out in their researcher where they save the research plan for future use.
Compress at tool boundaries: Tools boundaries are a natural place to add compression, if needed. The output of token-heavy tool calls can be summarized, for example, using a small LLM with straightforward prompting. This lets you quickly limit runaway context growth at the source without the need for compression over the full agent trajectory.
Start simple with memory: Memory can be a powerful way to personalize an agent. But, it can be challenging to get right. I often use simple, file-based memory that tracks a narrow set of agent preferences that I want to save and improve over time. I load these preferences into context every time my agent runs. Based upon human-in-the-loop feedback, I use an LLM to update these preferences (see here). This is a simple but effective way to use memory, but obviously the complexity of memory can be increased significantly if needed.
Consider multi-agent for easily parallelizable tasks: Agent-agent communication is still early, and it’s hard to coordinate multi-agent teams. But that doesn’t mean you should abandon the idea of multi-agent. Instead, consider multi-agent in cases where the problem can be easily parallelized and tight coordination between sub-agents is not strictly required, as shown in the case of Anthropic’s multi-agent researcher.
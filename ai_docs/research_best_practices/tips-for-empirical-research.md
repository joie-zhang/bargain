Tips and Code for Empirical Research Workflows
by John Hughes, Ethan Perez
20th Jan 2025
AI Alignment Forum
Our research is centered on empirical research with LLMs. If you are conducting similar research, these tips and tools may help streamline your workflow and increase experiment velocity. We are also releasing two repositories to promote sharing more tooling within the AI safety community.

John Hughes is an independent alignment researcher working with Ethan Perez and was a MATS mentee in the Summer of 2023. In Ethan's previous writeup on research tips, he explains the criteria that strong collaborators often have, and he puts 70% weight on "getting ideas to work quickly." Part of being able to do this is knowing what tools there are at your disposal.

This post, written primarily by John, shares the tools and principles we both use to increase our experimental velocity. Many readers will already know much of this, but we wanted to be comprehensive, so it is a good resource for new researchers (e.g., those starting MATS). If you are a well-versed experimentalist, we recommend checking out the tools in Part 2—you might find some new ones to add to your toolkit. We're also excited to learn from the community, so please feel free to share what works for you in the comments!

Quick Summary
Part 1: Workflow Tips. We explain how to get the most out of your terminal with frameworks such as "Oh My ZSH" and how to easily deploy this on new machines with dotfiles. Learn about useful features in VSCode and what pre-commit hooks are.
Part 2: Useful Tools. Learn about software such as Tuple (for pair programming), LLM-assisted coding tools such as Cursor, and command line packages such as uv for Python package management.
Part 3: Experiment Tips. We explain tips for two research modes: "De-risk sprint mode" and "Extended project mode."
Part 4: Shared AI Safety Tooling Repositories. We release two repositories used by Ethan's MATS scholars: one for shared tooling and the other for examples of using the shared tooling. 
Part 1: Workflow Tips
Terminal
Efficient terminal navigation is essential for productivity, especially when working on tasks like running API inference jobs or GPU fine-tuning on remote machines. Managing directories, editing files, or handling your Git repository can feel tedious when relying solely on bash commands in a standard terminal. Here are some ways to make working in the terminal more intuitive and efficient.

Terminals
Mac: We recommend iTerm2 . We find using the "Natural Text Editing Preset" a very useful since you can move a word backward using Option ⌥ + ← and a word forward using Option ⌥ + →. Paired with an increased keyboard key repeat rate, you can seek through commands in the terminal rapidly.
Linux: popular terminals include Wezterm and Kitty.
Windows: WSL gives you access to an Ubuntu shell and you can set that to default in Windows Terminal.
Also, the built-in terminal in many editors is good too (e.g. Emacs, VSCode and PyCharm).
ZSH is an alternative to bash and is much much easier to use
ohmyzsh is a zsh configurator that includes plugins, themes and ease of navigation
Those who have used this framework know that navigating directories is much easier due to the use of tab complete and arrow keys.
Essential plugins:
zsh-autosuggestions — suggests commands based on your history as you type
zsh-syntax-highlighting — syntax highlighting within the terminal
zsh-completions — complete some bash commands with tab
zsh-history-substring-search  — type any substring of a previous command (it doesn't have to be the start) and use up and down keys to cycle through relevant history
fuzzy search (fzf) allows easy searching of previous commands and nested directories.
The p10k theme is amazing and customisable. A nice feature are the icons that appear when in a Git repository showing the status and other icons that show if you have sourced a virtual environment.
Tmux allows you to set up multiple panes in your terminal that keep running in the background. Therefore, if you disconnect from a remote machine, scripts running in tmux will not be killed. We tend to run experiments across many tmux panes (especially overnight).
The default settings and key bindings aren't great and have a learning curve for new users. See an example config here that improves it, or try out "Oh my tmux!".
Dotfiles automate the deployment of your setup on new machines by installing packages and putting the configuration in the correct place on the machine.
See John's here or https://dotfiles.github.io/ for another guide.
You can set other important variables in your dotfiles, such as unlimited history length.
Note: they are called dot files since all config is contained in files that start with . like ~/.zshrc and ~/.tmux.conf.
Aliases are custom shortcuts or functions for common commands. For example gc for git commit -m  and many more in this file. Here are two which save a lot of time:
rl for getting the absolute file path followed by copying to your clipboard is incredibly helpful (see custom bins in here; big shout out to Ed Rees for this one)
Running ls after cd so, when you change the directory, you always see the files contained there.
We recommend seeing what commands you run all the time and converting those to aliases. Don't try to add too many too quickly, though. It is best to build this up slowly over time.
Note: there are many recommendations here, which can be overwhelming, but all of this is automated in John's dotfiles (including installing zsh and tmux, changing key repeat speeds on Mac and setting up aliases). So, if you'd like to get going quickly, we recommend following the README to install and deploy this configuration.

Integrated Development Environment (IDE)
Choosing the right Integrated Development Environment (IDE) can enhance your productivity, especially when using LLM coding assistants. A good IDE simplifies code navigation, debugging, and version control.

We can't recommend Cursor as an IDE enough. It is a fork of VSCode so looks and feels the same but offers very impressive LLM coding assistant integration.
You can highlight code and use ⌘K to prompt the LLM to rewrite code blocks, or ⌘L for chat functionality. Also, it includes Copilot-like line auto-completion and agent capabilities.
You can also use a .cursorrules file which informs the LLM how to act.
All VSCode extensions are available to use in Cursor too.
Code-assisting tools are now considered essential in our research projects.
If you are not working locally, the remote SSH extension is a must-have.
Syncing code by pushing to GitHub and then pulling it onto your remote machine is inefficient, as you need to repeat the process every time you test a new fix for a bug. A more effective approach is to edit the code directly on the remote machine, test it there, and push only the finalized bug fix.
You can also edit remote files outside of the code repository from within VSCode/Cursor, which is very helpful. Packages such as Vim and Nano are text editors from within the terminal, but these have much higher learning curves.
The VSCode debugger is great since you can easily inspect the values of all variables currently in your scope and the call stack. It is worth spending time learning about.
Using the built-in Python debugger by putting breakpoint() within your code is also very useful and often quicker than debugging with print statements.
There are lots of useful VSCode/Cursor settings (see example here)
Autosave (very useful, so you never have to worry about accidentally running a script without pressing save)
Jupyter run startup commands (e.g. autoreload)
Jupyter notebook file root (setting to the root of the repo can be helpful)
VSCode Debugger remote attach settings (allow you to debug code running on a remote machine from your local VSCode instance)
Linting & code formatting extension configuration
File watcher excludes (so VSCode doesn’t slow down by tracking changes in virtual environments or other folders that contain many files)
Extensions
GitLens — useful for knowing who committed which lines and visualising the Git history
Jupyter — run notebooks within VSCode
Linting (though not required if using pre-commit hooks)
Ruff - great for linting
Black - great for code formatting (Ruff now also provides a code formatted but make sure to increase the allowed line length setting)
Nvidia-smi+ - view GPU utilisation statistics
JSON Lines Viewer - improves the viewing experience of jsonl files
LaTeX Workshop - compile LaTeX code and export to pdf
devServer — good for testing websites for papers (this was used when developing the website for BoN Jailbreaking)
Inspect AI - UK AISI’s inspect framework allows you to easily run LLM evals and agents. This extension lets you view interactive logs that include all LLM inputs/outputs after running.
Keyboard shortcuts
We recommend learning what the Jupyter Notebook shortcuts are and customising them to something you are more comfortable with. You can add these shortcuts to Google Colab and VSCode/Cursor.
A program called Karabiner is useful to change certain key mappings on Mac. For example, the capslock key is redundant and can be mapped to ctrl or command. If using Vim keybindings, remapping escape to capslock is very common.
Chrome has useful shortcuts such as:
Jump to the next open tab: ⌘⌥ left/right
Search all tabs: ⌘⇧A
Git, GitHub and Pre-Commit Hooks
Mastering Git, GitHub, and pre-commit hooks is key to maintaining a smooth and reliable workflow. These tools help you manage version control, collaborate effectively, and automate code quality checks to prevent errors before they happen.

Creating a repository on GitHub and using Git to track and manage code changes is highly recommended (even if working individually).
We recommend using pre-commit hooks, which run automatically when committing your files to git. If there are linting or formatting errors, you must fix them before being allowed to commit. This is great to stop you from committing syntax errors or code with unused variables. It also enforces code to be tidy and in a consistent format which is important when collaborating with others.
To use pre-commit hooks you should include a .pre-commit-config.yaml (e.g. here), config within pyproject.toml (e.g. here) and a Makefile (e.g. here) in the root of your repo. You must first pip install pre-commit and then run make hooks.
The pre-commits we recommend are:
Ruff for linting (it is significantly faster than alternatives like flake8 and pylint)
Black for formatting (you can also use ruff for formatting, but it tends to be stricter and spreads code out over more lines which can be annoying)
trailing-whitespace is useful to automatically strip whitespace
nbstripout is very useful to automatically remove notebook outputs to avoid bloating the git history and size of the repo
ReviewNB is useful when reviewing code in Python notebooks on GitHub.
Part 2: Useful Tools
Not all of these recommendations are directly related to research (e.g., time-tracking apps), but they are excellent productivity tools worth knowing about. The goal of this list is to make you aware of what’s available—not to encourage you to adopt all of these tools at once, but to provide options you can explore and incorporate as needed.

Software/Subscriptions
Cursor — As explained in the previous section, we highly recommend this IDE due to the LLM coding integration and consider it an essential tool. It is a fork of VSCode, so it offers all the same great features/extensions but offers much better LLM integration. There is a free tier, but it limits the number of calls to premium models, so we recommend upgrading to Pro for $20/month (we think this is the best bang for your buck in terms of productivity gain compared to other tools). Alternatives include Zed and GitHub Copilot.
ChatGPT+ and/or Claude Pro — As an LLM researcher, it is important to have conversations with top-tier LLMs all the time so you understand their capabilities. ChatGPT+ frequently releases new features that open new research opportunities (e.g. ChatGPT advanced voice mode).
Tuple — We love this pair programming tool that allows you to seamlessly take control of another person's computer during screen sharing. It has low latency and excellent video quality. You can pair with guests who don't have a paid subscription.
Google One — This subscription allows you to record Google Meet calls and extend them beyond 1 hour, which is very useful. It also includes a feature called pop-out mode, which allows you to see people's videos while screen sharing.
Grammarly — This is useful for ensuring prompt grammar is accurate, which is important before final paper experiments. It also works seamlessly with Overleaf, emails, and Slack to speed up writing.
Perplexity — A useful LLM-powered search engine that cites its sources.
TimingApp — Excellent for tracking time. Allows you to set rules for activities to automatically assign them to projects and integrates with iPhone screen time. A free alternative is Wakatime which other collaborators we work with use.
ReadAI — Automatically records meetings with a notetaker bot for Google Meet, Teams and Zoom. Offers recordings and transcripts, reducing the need for manual note-taking during meetings. Otter is an alternative we use too.
We are excited by Granola, which may provide more value since it can automatically expand on notes you've jotted down.
Rectangle — A window pane manager for Mac. Enables snapping windows to different portions of the screen.
Copy Clip — A simple clipboard manager to ensure you don’t lose important copied text. Other tools like Raycast and Alfred have this built-in too.
Context — Provides improved tab completion and search functionality on Mac. Requires a one-time payment.
Karabiner — Allows keyboard key changes (e.g., remapping keys for Vim keybindings).
Zotero — organise and annotate research papers.
Other software that other collaborators use: Homerow, Dash, Speechify, BetterTouchTool, , LiquidText, and ice.
LLM Tools
Weights & Biases — Very useful for tracking OpenAI fine-tuning jobs (status, data, and losses) and experiments in general. Offers free access for up to 5 team members.
Inspect — UK AISI framework for running LLMs with flexible task definitions, solvers, and scorers. Provides tools for multi-turn dialogs, model-graded evaluations, and agent benchmarks. Supports many models, including Hugging Face LLMs, with efficient concurrency, caching, and a trace mode for debugging. Includes a VS Code extension and web UI for log analysis and output inspection (which is so cool!). We think this framework can significantly accelerate workflows after spending the time to learn how it works.
Aider — pair program with LLMs, to edit code in your local git repository (it has one of the top scores on SWE bench).
Devin — An automated LLM software engineering agent that autonomously performs repo tasks and creates PRs. Allows real-time interactions and background fixes for refactors or small tasks. It isn’t seamless yet in our experience (e.g. it got stuck on linting with the wrong Python version), but it’s a promising tool and likely how automated research will be orchestrated in the future. It has a hefty $500/month price tag, so it's only worth it if sharing in a team or using it to demonstrate misaligned behaviours. One to watch in the future!
openweights — Automates deploying RunPod pods for parallel fine-tuning and batch inference jobs. Offers an interface similar to OpenAI's. Big thanks to Niels Warncke (a fellow collaborator) for developing this!
LiteLLM — Provides a unified interface for API interactions (OpenAI, Anthropic, Gemini, Hugging Face) with a proxy server or SDK. This is well worth using if you expect to run many models for a paper. It is a good tool for those who prefer not to use Inspect or our safety-tooling repo (see Part 4).
repo2txt — UI for generating a prompt given a GitHub URL so you can easily copy-paste and learn about the codebase with an LLM.
langchain — Offers standardised component interfaces useful for scaffolding/agents. Supports caching and integrates with LiteLLM. A good alternative if Inspect doesn’t fit your needs.
vLLM — Hosts open-source LLMs for efficient inference using PagedAttention, continuous batching, and quantization. Supported by Inspect. You should use it if you run LLM inference on your own GPU.
PromptFoo — Ideal for rigorous prompt engineering. Enables writing unit tests for prompt responses and testing against defined metrics across various LLM combinations.
Langfuse — Primarily useful for developers deploying LLM applications. Provides detailed traces of LLM interactions, human feedback, and LLM-as-a-judge functionality. We haven't used it, but perhaps it is useful for demos and human labelling interfaces.
Ollama — Runs most open-source models locally at various quantization levels (via llama.cpp). Usable through a terminal or a ChatGPT-like interface.
exa.ai — allows you to do vector searches over the web, which is useful in literature reviews. Also, a good API to use to give LLMs access to the web.
unsloth — fine-tune open source LLMs 2-5x faster and with <70% less memory. It supports 4bit fine-tuning and allows training large models on a single GPU.
axolotl — a user-friendly tool to help fine-tune LLMs that supports multi-GPU setups and doesn't require deep technical tweaking.
Prismatic VLMs — great repository for helping you train VLMs
open-clio — A simple reproduction of the CLIO paper using language models to label and cluster user behaviour recursively, helping to analyze behaviour at scale. This is a great tool to get insights into diverse model outputs.
LLM Providers
RunPod — Our go-to provider for using GPUs for open-source model fine-tuning. We find this to be the best provider due to the ability to use network drives to share data between collaborators, the availability of GPUs, and hardware reliability. It is also cheap compared to GCP/Azure. VastAI/LambdaLabs are alternatives, but we've heard they are not as reliable and do not support network drives.
TogetherAI  — a great service for those who want to run inference or fine-tuning on open-source models via an API. OpenRouter is an alternative we have used.
HuggingFace Dedicated Inference Endpoints — A great way to spin up huggingface models (such as the circuit breaking model) and send requests via an API. A big advantage is that it will autoscale up to a configurable maximum number of model replicas dependent on your demand and scale to zero after 15 minutes of inactivity. This makes it more cost-effective than hosting yourself on RunPod. It supports returning model logprobs, too, unlike OpenRouter.
Command Line and Python Packages
uv — A single tool to replace pip, pyenv, and virtualenv. It is 10-100x faster than pip!
scalene or py-spy for profiling Python programs
asyncio is very important to learn for empirical LLM research since it usually involves many concurrent API calls
shell-ask or ask.sh — ask LLMs to write and execute bash commands (and you can pipe text into the command)
jless — command line jsonl explorer
ncdu — an interactive recursive filesize explorer
htop —  an interactive process viewer
nvtop — an interactive version of nvidia-smi
ripgrep (better grep), Dust (better du), duf (better df), bat (better cat with highlighting and git), fd (better find), exa (better ls)
code2prompt — convert repo into single LLM prompt
opencommit —  auto-generate commit messages
magic-wormhole — copy files between machines 
Part 3: Experiment Tips
De-risk and extended project mode
First, we'd like to explain that there are usually two modes that a research project is in: namely, de-risk mode and extended project mode. These modes significantly change how you should approach experiments, coding style, and project management.

De-risk mode focuses on rapidly answering high-priority questions with minimal overhead.
This mode is ideal for:
Quick experimentation using Python notebooks that minimize time-to-insight.
Minimal investment to avoid effort in engineering practices, like extensive documentation, strict coding standards, or generalized pipelines.
In collaborative group settings, this mode is still common. It is important to communicate the experiment's goals and frequently discuss the next steps rather than performing thorough code reviews.
Extended project mode emphasizes engineering rigour and longer-term maintainability.
This mode is especially critical for longer-term collaborations or experiments that require significant compute resources and complicated infrastructure, where bugs can lead to costly reruns. It also ensures that knowledge and progress can be shared across contributors.
Key practices in extended project mode include:
Transitioning from notebooks to structured scripts, modules, or pipelines.
Applying code reviews, testing, and version control.
Using tools like pre-commit hooks and CI/CD workflows to enforce quality.
The workflow should always be conditioned on the situation:

Start in de-risk mode: For example, if you’re searching for a new type of alignment faking behaviour or if it is possible to jailbreak a model with a specific technique, a notebook is great for determining feasibility.
Switch to extended project mode: Once the experiment is de-risked, it can often mature into a larger project involving significant compute and collaboration. Now is the right time to refactor your work into a maintainable codebase. This transition can often catch bugs since notebooks are notorious for bugs (e.g. those that occur when you run cells in a non-linear fashion).
Note: sometimes projects start here if there is significant infrastructure needed, it suits the collaborators workflow better or the project is already de-risked before starting.
Ethan tends to be in de-risk mode for 75% of his work, and he uses Python notebooks to explore ideas (for example, many-shot jailbreaking was derisked in a notebook with ~50 lines of code). The Alignment Science team at Anthropic is also primarily in "de-risk mode" for initial alignment experiments and sometimes switches to "Extended project mode" for larger, sustained efforts.

Note: Apollo defines these modes similarly as "individual sprint mode" and "standard mode" in their Engineering Guide. We opt for different names since lots of the research we are involved with can primarily be in de-risk mode for a long period of time.

Tips for both modes 
Invest in a project plan
Have a clear project plan that includes motivation and research goals, and list all the experiments you can possibly think of running. Get feedback from peers and iterate.
Think about milestones for the project and what you want to deliver. This will help to keep your self accountable. Don't underestimate how long it takes to write a paper.
Think about code structure
If you'd like to open-source code, it is worth investing time at the start thinking about how you will design the repo so it is easy to use.
Know the LLM tools that are out there (e.g. list in Part 2). It might be a good idea to build off an existing framework like Inspect or use tools like LiteLLM to make sure you have the flexibility down the line to run more models easily.
As you build experience knowing how you best run experiments, start to build your own reusable tooling (and perhaps contribute it to our safety-tooling repo - see Part 4).
Communicate frequently and clearly about what you're working on
One of the important ways to move quickly with research is to choose the right next experiments to run. Therefore, it is important to communicate plans regularly with the team so you can get feedback.
We use a Slack channel for daily updates within the cohort. This is helpful for structuring your own next steps, keeping yourself accountable, and also providing your mentor/team with good context.
Projects we run often have a daily standup with close collaborators. We find this essential for staying aligned on research goals, prioritising the right work and delivering on time.
Use notion to track your experiments
For many projects in our cohort, we create a database table with columns for: experiment name, tags, users involved, last updated and status (in progress or done).
The team creates notion pages quickly for each new experiment (database row) and dumps figures and thoughts as they go. This should be done quickly, and it doesn’t matter if it is messy (as long as close collaborators can figure out what is happening).
It is helpful to look back at this when you write slides to present to mentors and when you start writing the paper.
On the topic of slides, make sure to check our tips in collaboration with James Chua.
Pause to think before starting the work on an experiment
Some questions we ask ourselves and discuss with the team are:
What is the motivation for this experiment? Does it fit in with the research question I want to answer?
Have I derisked this enough already, and are there other more interesting things to run instead? Is this definitely the highest priority?
What result do I expect to get? Is learning that useful?
Should I explore one model and one dataset first before expanding to more?
Will running this extra experiment add significant value to our paper? (especially relevant when close to a deadline)
Am I changing too many variables at once? Can I simplify my setup to draw better conclusions?
Experiment folder structure
We recommend committing all lightweight scripts and notebooks used to run a certain experiment to a new folder in the codebase.
For example, if running a new jailbreaking technique, you could create a folder in the repo called something like ./experiments/<name>/250109_jailbreaking_technique_v1
Naming the folder with YYMMDD can help keep a logical ordering. You could also have sub-folders named after each collaborator.
Experiments tend to involve many scripts and notebooks that analyse the results so one method is to enumerate these so it is clear the order in which they were run. E.g. 1_run_harmbench.sh, 2_run_classifier.sh, 3_analyse_attack_success_rate.ipynb.
Core functionality can remain in the experiments folder if working in de-risk mode. However, if in extended project mode, core functionality should be refactored to be elsewhere in the codebase (only keep lightweight bash scripts, config files or analysis notebooks in the experiment folders).
You should not be afraid of code duplication in these folders, even if it is just a few hyperparameters or data paths that change (and if in de-risk mode, it is fine to copy-paste notebooks frequently).
Following this method means you can easily replicate any experiment, which is often useful down the line (e.g. during a paper rebuttal period), and you can easily retrace your steps. It also helps collaborators find what you have been running easily, along with all the parameters and paths to data, without needing to ask.
Tips for extended project mode
Pull requests (PRs) for each feature/experiment
Code review is very helpful in extended project mode. Many bugs have been found in the projects we've been involved in after ensuring we do code reviews on core functionality. One bug could lead to you having to re-run days or weeks of results.
When operating in de-risk mode, it's important not to overdo it. For early-stage, low-compute experiments or projects managed by a single person, working directly in notebooks is often the most efficient and practical approach.
It can be worth the effort to think about how to split up features, experiments, and bug fixes into separate bite-sized PRs that are easy to review. PRs with masses of new files and complex changes across the codebase reduce the chance that people do a good review where they check thoroughly for bugs.
Code review isn't necessary for everything, and the teams should use their judgment. Often, we will self-approve and merge quickly if the PR is just a bunch of notebooks and experiment scripts.
If there is a PR for core code and functionality, we recommend all core contributors take a look to effectively manage conflicts and help onboard each other on new code.
Merge PRs quickly
We encourage the teams we work with to review PRs and work to merge them fast as a number one priority. This helps ensure everyone runs the latest and greatest code and avoids difficult merge conflicts down the line.
Ability to kill experiments and carry on from where you left off
Caching of LLM responses and other algorithm state is important for this. For example, this allows you to tweak concurrency settings and rekick off a run without losing progress.
This may also involve saving intermediate outputs/checkpoints, which has the added benefit that you can check progress or potential bugs as your experiment is running.
Experiment reproducibility
If the same script is run, the result of the experiment should be as close to the original as possible. This isn’t always possible due to nondeterministic LLM APIs (even at temp 0), but everything else should be the same (e.g., data splits, hyperparameters, etc).
Setting this up correctly is important to make caching work, too; otherwise, the prompts will be different, and everything will start from scratch.
It can be useful to save the git commit hash in your experiment directory or even make a copy of the entire codebase (just in case you need to debug in the future).
Output jsonl files and using pandas
This isn’t always relevant to every experiment, but for most empirical LLM experiments, an LLM gives a response to many different inputs. Therefore, outputting a jsonl file at the end of the experiment with all the metadata, inputs, and outputs is useful.
These results can be quickly analysed with pandas in a notebook. Pandas is essential for lots of the data processing we do. It makes it straightforward to transform columns, filter a data frame, aggregate across columns and calculate simple statistics with .describe().
Command line args for all Python scripts
Scripts with good input args allow you to:
create simple wrapper scripts that can be committed to the experiment folders explained above (see example).
test scripts easily from the command line, especially if you can specify things like dataset size, batch size, and logging verbosity.
parallelise experiments and run them in separate tmux panes to maximise your experiment throughput.
There are many packages that make command line args easy (e.g. fire , hydra and simple_parsing). We use simple_parsing (see example) because it allows you to define your args in a data class which gets automatically initiated and populated from the command line args.
Parallelising experiments
There are different ways to queue up GPU jobs (e.g. fine-tuning with different hyper params or data splits).
simple-gpu-scheduler is simple and very effective if you want to queue up many jobs on a multi-GPU machine
openweights automatically spin up RunPod pods to complete many jobs
For overnight experiments, we encourage collaborators to bias towards using the OpenAI or Anthropic batch API. It is cheaper and has higher throughput.
Part 4: Shared AI Safety Tooling Repositories
For many early-career researchers, there's an unnecessarily steep learning curve for even figuring out what good norms for their research code should look like in the first place. We're all for people learning and trying things for themselves, but we think it would be great to have the option to do that on top of a solid foundation that has been proven to work for others. That's why things like e.g. the ARENA curriculum are so valuable.

However, there aren't standardised templates/repos for most of the work in empirical alignment research. We think this probably slows down new researchers a lot, requiring them to unnecessarily duplicate work and make decisions that they might not notice are slowing them down. ML research, in general, involves so much tinkering and figuring things out that building from a strong template can be a meaningful speedup and provide a helpful initial learning experience.

For the MATS 7 scholars mentored by Ethan, Jan, Fabien, Mrinank, and others from the Anthropic Alignment Science team, we have created a GitHub organization called safety-research to allow everyone to easily discover and benefit from each others’ code. We are piloting using two repositories: 1) for shared tooling such as inference and fine-tuning tools and 2) providing a template repo to clone at the start of a project that has examples of using the shared tooling. We are open-sourcing these two repositories and would love for others to join us!

Repo 1: safety-tooling
Share Great Tooling: Reduces duplicated work by providing reusable code for common tasks, enabling researchers to focus on de-risking new ideas. Tooling includes an LLM Inference API with concurrency control, fine-tuning tooling with Weights and Biases, prompt templating with Jinja and much more.
Upskill Collaborators: Encourages software development skills like code reviews, testing, and modular design, preparing researchers to apply to top research labs. If your goal is to work somewhere like Anthropic, contributing to an open-source project such as this is a good way to upskill.
Submodule Design: Integrates directly into individual projects and makes it easy to contribute new tools directly. People can benefit from the tools without needing to abide by strict engineering practices in their project repo.
Repo 2: safety-examples
Share Examples: Provides examples over a wide range of AI Safety projects (such as adversarial robustness and LLM evals) so others have a quick place to start.
Onboard Researchers Quickly: Offers a structured starting point to majorly speed up productivity for new collaborators. Many of Ethan's MATS scholars are cloning (or forking) this repository before starting a new project so they can benefit from access to examples, pre-commit hooks and the safety-tooling submodule.
Note: We are very excited about UK AISI's Inspect framework, which also implements lots of what is in safety-tooling and much more (such as tool usage and extensive model graded evaluations). We love the VSCode extension for inspecting log files and the terminal viewer for experiment progress across models and tasks. We aim to build a bigger portfolio of research projects that use Inspect within safety-examples and build more useful research tools that Inspect doesn't support in safety-tooling.

Acknowledgements
We'd like to thank Jack Youstra and Daniel Paleka, as many of the useful tool suggestions stem from conversations with them. For more of their recommendations, check out their blogs here and here. John would like to thank Ed Rees and others at Speechmatics, from whom he's borrowed and adapted dotfiles functionality over the years. Thanks to Sara Price, James Chua, Henry Sleight and Dan Valentine for providing feedback on this post.  

17 comments, sorted by top scoring
Highlighting 1 new comments since 03/01/2025

Neel Nanda

This looks extremely comprehensive and useful, thanks a lot for writing it! Some of my favourite tips (like clipboard managers and rectangle) were included, which is always a good sign. And I strongly agree with "Cursor/LLM-assisted coding is basically mandatory".

I passed this on to my mentees - not all of this transfers to mech interp, in particular the time between experiments is often much shorter (eg a few minutes, or even seconds) and often almost an entire project is in de-risking mode, but much of it transfers. And the ability to get shit done fast is super important

John Hughes

Thanks Neel! I'm glad you found it helpful. If you or your scholars recommend any other tools not mentioned in the post, I'd be interested to hear more.

Neel Nanda

I've been really enjoying voice to text + LLMs recently, via a great Mac App called Super Whisper (which can work with local speech to text models, so could also possibly be used for confidential stuff) - combining Super Whisper and Claude and Cursor means I can just vaguely ramble at my laptop about what experiments should happen and they happen, it's magical!
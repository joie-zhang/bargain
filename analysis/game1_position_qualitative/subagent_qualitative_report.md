# Game 1 Control Position Effect: 310-Transcript Subagent Qualitative Report

## Scope and Subagent Execution

The all-`gpt-5-nano` control set contains **310 completed runs**: **219 N=3** transcripts and **91 N=5** transcripts. I rebuilt the transcript packets so they include public discussion, structured proposal enumeration, raw formal proposal responses, raw private voting, raw private thinking, and reflection responses where present.

The interactive `spawn_agent` tool exposed a hard thread cap during this run: the seventh interactive spawn failed with `agent thread limit reached (max 6)`. I therefore used a manager fanout: 10 manager shards of 31 transcripts each, run through the six-slot cap, with transcript-level child workers capped at 5 concurrent per manager. After rerunning the initial manager_03 fallbacks, the final artifact set has **310/310 transcript review JSONs with `reviewer_mode = child_worker`**.

Raw phase coverage: private thinking is present in **310/310** runs (**3290 entries**), private voting contributes **12040 entries**, and reflection appears in **202/310** runs (**2176 entries**). The missing reflection cases are absent from the raw interaction logs, usually because the run ended before a later reflection phase was emitted.

Artifacts:
- Review index: `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/subagent_review_index.csv`
- Theme counts: `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/subagent_theme_counts.csv`
- Run status: `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/subagent_run_status.json`
- Transcript reviews: `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/subagent_reviews`
- Rebuilt transcripts: `/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts`

## Is The First Position Advantage Real?

Numerically, yes, but it is not universal. Agent_1 beats the last position in **174/310** runs, ties in **54/310**, and loses in **82/310**. The pattern is much stronger at high competition: **110/143** high-competition runs have Agent_1 above the last position. At competition level 0.0, only **19/93** runs have Agent_1 above the last position.

The qualitative reviewers marked a process-level first-position advantage in **167/310** transcripts. That aligns tightly with the payoff result: **167/174** first-wins were judged to show process advantage, while **0/136** ties/counterexamples were judged as clear first-position advantage (`False`: 122, `mixed/unclear`: 21).

By N and competition level:

| N | competition | runs | first > last | first = last | first < last | process advantage |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0 | 67 | 7 | 50 | 10 | 6 |
| 3 | 0.5 | 56 | 34 | 1 | 21 | 34 |
| 3 | 0.9 | 53 | 39 | 1 | 13 | 39 |
| 3 | 1 | 43 | 37 | 0 | 6 | 36 |
| 5 | 0 | 26 | 12 | 0 | 14 | 12 |
| 5 | 0.5 | 18 | 11 | 0 | 7 | 9 |
| 5 | 0.9 | 23 | 17 | 1 | 5 | 17 |
| 5 | 1 | 24 | 17 | 1 | 6 | 14 |

## Themes Explaining The Advantage

| Theme | Count | What It Means |
|---|---:|---|
| Process-level first-position advantage | 167/310 | Transcript reviewers judged Agent_1 to have a positional/process advantage, not just a numerical win. |
| Opening anchor/default constraint | 150/167 | Among process-advantage cases, Agent_1 used explicit opening anchor language in the public discussion. |
| Self-advantaging formal proposal | 155/167 | Among process-advantage cases, Agent_1 submitted a formal proposal that gave Agent_1 above-proposal-mean utility. |
| Final allocation matched Agent_1 formal proposal | 148/167 | Among process-advantage cases, the accepted final allocation matched at least one Agent_1 formal proposal. |
| Agent_1 top item preserved | 159/167 | Among process-advantage cases, Agent_1 received at least one top-valued item. |
| Last agent missed a top item | 118/167 | Among process-advantage cases, the last-position agent did not receive any top-valued item. |
| Agent_1 top item while last missed top item | 110/167 | Among process-advantage cases, Agent_1 received a top item and the last agent did not. |
| Speed/efficiency/lock-in language | 166/167 | Among process-advantage cases, public/proposal text used quick-resolution and efficiency/top-item framing. |

### 1. Agent_1 Turns The First Turn Into A Default Constraint

In **150/167** process-advantage cases, Agent_1 used explicit opening-anchor language. The important pattern is not merely saying a preference first; it is stating a protected bundle and then making later bargaining happen around that protected bundle.

Representative evidence:
- [config_1006.md:20](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_1006.md:20) - "Here are my thoughts as Agent_1 (valuations: Stone 72, Pencil 23, Apple 4, Jewel 1, Quill 0)."
- [config_1006.md:23](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_1006.md:23) - "Initial anchor offer: I take Stone and Pencil."
- [config_0150.md:26](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0150.md:26) - "So my baseline is: Quill stays with me."
- [config_0032.md:27](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0032.md:27) - "Opening proposal (clear, shareable anchor):"

### 2. Later Agents Often Negotiate Inside Agent_1's Frame

Once Agent_1's first bundle becomes a named anchor, later agents often optimize around what remains rather than challenging the premise. This is why order matters: the last agent is not merely later in speech order; it frequently inherits a narrower feasible set.

Representative evidence:
- [config_1006.md:40](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_1006.md:40) - "I hear your anchor: Stone and Pencil stay with you in Round 1, and the rest can be split between Agents 2 and 3."
- [config_1006.md:159](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_1006.md:159) - "I accept that Round 1 will proceed with Stone+Pencil to Agent_1, and Jewel+Quill to Agent_2, Apple to Agent_3."
- [config_0030.md:707](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0030.md:707) - "keeps the negotiation anchored and fast"
- [config_0150.md:132](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0150.md:132) - "Quill is non-negotiable to stay with Agent_1 in Round 1."

### 3. Private Thinking Converts The Public Anchor Into Strategy

The raw private-thinking logs show that the models privately restate the public anchor as a planning constraint. Private thinking was available in every run; in the **167** process-advantage cases, the reviewers repeatedly found that these private plans treated Agent_1's allocation as the stable core.

Representative evidence:
- [config_1006.md:3072](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_1006.md:3072) - "The anchor is Stone+Pencil must stay with Agent_1 in Round 1."
- [config_1006.md:3120](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_1006.md:3120) - "Agent_1 anchors Round 1 with Stone and Pencil and will not move them in Round 1"
- [config_0030.md:1843](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0030.md:1843) - "The fastest route to unanimous agreement is likely to preserve Stone and Apple with Agent_1"
- [config_0150.md:465](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0150.md:465) - "keeping Quill anchored with Agent_1 is the central objective"

### 4. Proposal And Voting Phases Ratify The Early Frame

In **148/167** process-advantage cases, the final allocation matched an Agent_1 formal proposal; in **174** first-wins, this happens **154** times. This is not pure proposal order, because counterexamples also exist; the mechanism is that the proposal/vote phase turns the already-accepted anchor into a unanimous formal object.

Representative evidence:
- [config_0030.md:748](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0030.md:748) - "Preserves Agent_1's anchor (Apple and Stone) for a quick, decisive Round-3 resolution"
- [config_1006.md:3168](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_1006.md:3168) - "Preserves the Stone+Pencil anchor for Agent_1 and mirrors the baseline allocation"
- [config_0150.md:627](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0150.md:627) - "Allocation matches my preferred baseline: Quill-1, Pencil-2, Jewel-3, Stone-4, Apple-5."
- [config_0030.md:1950](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0030.md:1950) - "Proposal 2 preserves my 93 utility and is Pareto-superior for others"

### 5. The Last Position Often Gets A Residual Or Deferred Claim

Among process-advantage cases, the last agent missed a top-valued item in **118/167** transcripts; Agent_1 received a top item while the last agent missed one in **110/167**. This is the most direct payoff mechanism: first position claims a high item early, while the last position is compensated with a weaker item, future rotation, or no immediate item.

Representative evidence:
- [config_0030.md:748](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0030.md:748) - "Agent_3 receives nothing in this allocation; a rapid settlement is prioritized"
- [config_1006.md:165](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_1006.md:165) - "the strongest near-term payoff I can secure without breaking the anchor is: Round 1 continues with Agent_1 = Stone+Pencil"
- [config_1006.md:3234](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_1006.md:3234) - "Proposal 1 preserves Agent_1's Stone+Pencil anchor and gives me Apple (2) in Round 1."
- [config_0032.md:197](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0032.md:197) - "A3 = Stone"

### 6. Speed, Efficiency, And Future-Fairness Language Legitimize The Asymmetry

In **166/167** process-advantage cases, the transcript contains both quick-resolution framing and efficiency/top-item framing. These phrases make the early anchor look like the low-friction route to consensus, even when the last agent's payoff is weaker.

Representative evidence:
- [config_0030.md:141](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0030.md:141) - "preserves the high-priority items for the other two you’ve indicated you value, keeping the deal simple and fast."
- [config_0030.md:1844](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0030.md:1844) - "Use framing that highlights fairness and speed"
- [config_1006.md:3073](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_1006.md:3073) - "Frame any changes as fairness-driven compensation for anchor adjustments"
- [config_0150.md:565](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0150.md:565) - "a predictable rotation for the remaining items"

## Counterthemes: Why The Advantage Fails

| Countertheme | Count | What It Means |
|---|---:|---|
| Low-competition neutralization | 74/93 | At competition_level=0.0, Agent_1 did not beat the last position. |
| All agents got top items in non-first-win cases | 85/136 | Among ties/counterexamples, all agents received at least one top-valued item. |
| Last agent top item in counterexamples | 68/82 | Among first-loses cases, the last-position agent received at least one top-valued item. |
| Agent_1 top item still not enough | 54/82 | Among first-loses cases, Agent_1 still received a top item, so the counterexample is not just failure to secure a priority. |
| N=5 one-item-per-agent diffusion | 84/91 | In N=5 runs, final allocation assigned exactly one item to each agent. |

### 7. Low Competition And Complementary Preferences Neutralize Order

At competition level 0.0, Agent_1 fails to beat the last position in **74/93** runs. In non-first-win cases overall, all agents receive a top item in **85/136** runs. Here the first move may still set a frame, but the payoff surface already supports everyone.

Representative evidence:
- [config_0002.md:12](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0002.md:12) - "Agent_1: [58.0, 0.0, 0.0, 0.0, 42.0]"
- [config_0002.md:13](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0002.md:13) - "Agent_2: [0.0, 35.0, 0.0, 65.0, 0.0]"
- [config_0002.md:75](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0002.md:75) - "- Agent_1 gets Apple and Pencil, its top two items."
- [config_0008.md:8](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0008.md:8) - "final_utilities: {'Agent_1': 79.0, 'Agent_2': 100.0, 'Agent_3': 100.0}"

### 8. Unanimity Gives Later Agents Veto Power

When Agent_1 overreaches, the last agent can block proposals that exclude it. Among first-loses cases, the last-position agent receives a top item in **68/82** runs.

Representative evidence:
- [config_0004.md:631](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0004.md:631) - "vote_decision: reject"
- [config_0004.md:632](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0004.md:632) - "Proposal 1 gives Agent_3 nothing and allocates all remaining value to Agents 1 and 2"
- [config_0004.md:456](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0004.md:456) - "Proposal #3: 3 accept, 0 reject"
- [config_0008.md:361](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0008.md:361) - "Proposal 1 assigns me no items; I gain 0 utility here, while Proposal 2 yields Jewel (100) for me."

### 9. Agent_1 Can Get A Top Item And Still Lose

In **54/82** first-loses cases, Agent_1 still receives a top item. These are not failures of Agent_1 to claim a priority; they are cases where the last agent's top item is more valuable, less contested, or better protected by unanimity.

Representative evidence:
- [config_0484.md:20](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0484.md:20) - "My top priority is Apple (63). Jewel (37) is my secondary target."
- [config_0484.md:59](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0484.md:59) - "Pencil is my top prize (100)."
- [config_0484.md:170](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0484.md:170) - "It secures each agent's top priority (Apple for Agent_1, Quill for Agent_2, Pencil for Agent_3)"
- [config_0008.md:372](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0008.md:372) - "Proposal 2 gives me Jewel (100) in Round 1, the highest value I can obtain now"

### 10. N=5 Often Diffuses Into One-Item-Per-Agent Baselines

In N=5, **84/91** final allocations give exactly one item to each agent. This can still favor Agent_1 if Agent_1 anchors the highest-value item, but it also creates many counterexamples where another position receives the more valuable singleton.

Representative evidence:
- [config_0090.md:360](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0090.md:360) - "I propose: {'Agent_1': [2], 'Agent_2': [4], 'Agent_3': [1], 'Agent_4': [0], 'Agent_5': [3]}"
- [config_0090.md:365](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0090.md:365) - "preserves each agent's stated top priorities"
- [config_0090.md:593](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0090.md:593) - "Baseline Round 1 allocation that preserves each agent's stated top priorities"
- [config_0150.md:617](/scratch/gpfs/DANQIC/jz4391/bargain/analysis/game1_position_qualitative/transcripts/config_0150.md:617) - "Baseline alignment with each agent's top priorities and the required Quill anchor"

## Bottom Line

The qualitative mechanism is anchor persistence under unanimity. Agent_1 often uses the first public turn to define a protected item or bundle; later agents then search for a fast, fair-sounding allocation that preserves that protected core; private thinking and private votes restate the same constraint; and formal proposals ratify it. The positional advantage is strongest when competition is high, because preserving Agent_1's top item often means the last-position agent cannot also receive its top item.

The trend is not universal. When preferences are complementary, when every agent can receive a top item, or when the last agent has a credible veto over exclusion, the first turn may still shape the conversation but does not translate into a payoff edge.
